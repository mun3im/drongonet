"""
evaluate_mybad.py — score a DCASE-trained model on MyBAD (Malaysian) clips.

This is the generalization test that matters for the original goal: the models are
trained on UK/US/European recordings (freefield1010, warblr, BirdVox) and MyBAD is
Malaysian. Nothing from MyBAD is used for training, calibration, or threshold
selection here.

Two geometry notes:
  * MyBAD clips are 3s / 184 frames. drongonet-micro is natively 184 frames, so that
    model transfers exactly.
  * A SparrowNet trained in whole-clip mode saw 622 frames. Its weights still apply at
    184 frames (fully convolutional), but `padding="same"` distributes padding based on
    input length, so the trunk is not numerically identical to training geometry
    (~3e-3 on logits, and no integer shift aligns the two). A SparrowNet trained with
    --mode crop is 184 frames natively and avoids the question; both are reported so
    the effect is visible rather than assumed.

INT8 is re-converted at MyBAD's geometry, calibrated on DCASE training data only.

Usage:
    python evaluate_mybad.py results/F1_sparrow_all_s42
    python evaluate_mybad.py results/F*_*                    # several at once
"""

import os
import json
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from config import N_FRAMES, N_MELS, FULL_N_FRAMES
from models import build_sparrownet, build_drongonet_micro
from benchmark_archs import BUILDERS as BENCHMARK_BUILDERS
from dataset import normalize_np
from train import threshold_sweep, convert_int8, representative_dataset_fn, IndexedMel
from build_cache import load_cache
import mybad

try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


def rebuild_model(summary, n_frames):
    """Rebuild the trained architecture at MyBAD's frame count and load its weights."""
    arch = summary["arch"]
    n_mels = summary.get("n_mels", N_MELS)
    shape = (n_frames, n_mels, 1)
    width = tuple(summary.get("width") or (8, 16))

    if arch == "sparrownet":
        model = build_sparrownet(input_shape=shape,
                                 n_time_stages=summary["stages"],
                                 pool=summary.get("pool", "max"),
                                 head_kernel_t=summary.get("head_kernel", 1),
                                 width=width,
                                 dropout=summary.get("dropout", 0.1),
                                 batch_norm=summary.get("batch_norm", True))
    elif arch == "drongonet_micro":
        model = build_drongonet_micro(input_shape=shape,
                                      dropout=summary.get("dropout", 0.1),
                                      batch_norm=summary.get("batch_norm", False))
    else:
        model = BENCHMARK_BUILDERS[arch](input_shape=shape)

    weights = summary.get("weights")
    if not weights or not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found for {summary['tag']}: {weights}")
    model.load_weights(weights)
    return model


def load_mybad_inputs():
    """Load MyBAD and apply only the transform its scaling actually calls for."""
    info, labeled = mybad.inspect()
    kind, note = mybad.describe_scaling(info)
    print(f"MyBAD: {info['n_labeled']} clips ({info['n_positive']} positive), "
          f"shape {info['shapes']}, dtype {info['dtype']}")
    print(f"  values: min={info['min']:.4g} max={info['max']:.4g} mean={info['mean']:.4g}"
          f"  per-clip[0,1]={info['per_clip_min_is_0']}/{info['per_clip_max_is_1']}")
    print(f"  scaling: {kind} — {note}")
    if kind == "unknown":
        raise RuntimeError("MyBAD scaling not recognized; refusing to report an AUC "
                           "that would be meaningless")

    x, y = mybad.load_all(labeled=labeled)
    if kind == "linear_minmax":
        # Shipped arrays are min-max'd LINEAR amplitude. Squaring to power before
        # power_to_db assumes they are magnitude; treating them as power instead
        # (variant "power") is the other reading, and the record does not document
        # which. Both are evaluated so the choice's effect is visible.
        variants = {"magnitude": normalize_np(x ** 2), "power": normalize_np(x)}
    elif kind == "raw_power":
        variants = {"raw_power": normalize_np(x)}
    elif kind == "log_db":
        lo = x.min(axis=(1, 2), keepdims=True)
        hi = x.max(axis=(1, 2), keepdims=True)
        variants = {"log_db": np.where(hi > lo, (x - lo) / np.maximum(hi - lo, 1e-12),
                                       0.0).astype(np.float32)}
    else:
        variants = {"as_shipped": x}
    return variants, y, kind, info


def evaluate(results_dir, x, y, scaling, variant="default", n_calib=500, batch=256):
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)

    n_frames = x.shape[1]
    model = rebuild_model(summary, n_frames)

    probs = model.predict(x[..., np.newaxis], batch_size=batch, verbose=0)[:, 1]
    auc = float(roc_auc_score(y, probs))

    # INT8 at MyBAD geometry, calibrated on DCASE training data only
    mel, lab, _ = load_cache(summary["train_corpora"][0], mmap=True,
                             n_mels=summary.get("n_mels", N_MELS))
    repr_fn = representative_dataset_fn(mel, lab, "crop", n=n_calib, seed=summary["seed"])
    tflite_path = os.path.join(results_dir,
                               f"{summary['arch']}_mybad184_{variant}_int8.tflite")
    size = convert_int8(model, repr_fn, tflite_path)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], [batch, *x.shape[1:], 1])
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    isc, izp = inp["quantization"]
    osc, ozp = out["quantization"]

    q_probs = np.zeros(len(y), dtype=np.float32)
    for s in range(0, len(y), batch):
        chunk = x[s:s + batch][..., np.newaxis]
        q = np.round(chunk / isc + izp).clip(-128, 127).astype(np.int8)
        if len(q) < batch:
            q = np.concatenate([q, np.zeros((batch - len(q),) + q.shape[1:], np.int8)])
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[:len(chunk)]
        q_probs[s:s + len(chunk)] = (raw[:, 1].astype(np.float32) - ozp) * osc
    q_auc = float(roc_auc_score(y, q_probs))

    rows, chosen = threshold_sweep(y, q_probs)
    result = {
        "tag": summary["tag"], "arch": summary["arch"], "mode": summary["mode"],
        "trained_on": summary["train_corpora"], "seed": summary["seed"],
        "params": summary["params"],
        "train_input_frames": FULL_N_FRAMES if summary["mode"] == "full" else N_FRAMES,
        "mybad_input_frames": int(n_frames),
        "geometry_matched": (summary["mode"] == "crop"),
        "mybad_scaling": scaling, "mybad_variant": variant,
        "n_mybad": int(len(y)), "n_mybad_positive": int(y.sum()),
        "float32_mybad_auc": auc,
        "int8_mybad_auc": q_auc,
        "int8_size_kb": round(size / 1024, 2),
        "dcase_val_auc": summary["float32"]["val_auc"],
        "threshold_sweep_int8_mybad": rows, "chosen_tau": chosen,
    }
    with open(os.path.join(results_dir, f"mybad_summary_{variant}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", nargs="+")
    args = ap.parse_args()

    variants, y, scaling, info = load_mybad_inputs()
    print()

    for vname, x in variants.items():
        print(f"--- MyBAD interpreted as {vname} "
              f"(mean={x.mean():.3f}, median={np.median(x):.3f}) ---")
        rows = []
        for d in args.results_dirs:
            if not os.path.exists(os.path.join(d, "summary.json")):
                continue
            r = evaluate(d, x, y, scaling, variant=vname)
            rows.append(r)
            geo = "matched" if r["geometry_matched"] else f"{r['train_input_frames']}->184"
            print(f"{r['tag']:40s} p={r['params']:5d} geo={geo:9s} "
                  f"DCASEval={r['dcase_val_auc']:.4f}  "
                  f"MyBAD f32={r['float32_mybad_auc']:.4f} int8={r['int8_mybad_auc']:.4f}")
        if rows:
            print(f"  ({len(rows)} models on {rows[0]['n_mybad']} clips, "
                  f"{rows[0]['n_mybad_positive']} positive)")
        print()
