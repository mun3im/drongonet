"""
finalize.py — produce the flashable artifact for a trained MyBAD model.

Rebuilds the trained weights into the static-shape graph (GlobalMaxPooling2D instead
of Reshape(-1,C) + 1D pooling), re-converts to INT8 calibrated on the model's own
MyBAD training split, verifies the op set is TFLite-Micro-safe, and emits the C header.

The static-graph rebuild is required, not cosmetic: the Reshape(-1,C) form converted
to SHAPE / STRIDED_SLICE / PACK plus four dynamic-shape tensors, and TFLite Micro
allocates statically. It also happens to save ~1.1 KB.

Usage:
    python deploy/finalize.py results/G1_sparrow_db_fft512_s42
"""

import os
import sys
import json
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from config import N_FRAMES, N_MELS, SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES
from models import build_sparrownet, build_drongonet_micro
from dataset import normalize_np, log1p_minmax_np
from train import convert_int8, threshold_sweep
import mybad_cache
from export_c_array import to_header

# Ops TFLite Micro must have registered. Anything outside this set means the sketch's
# MicroMutableOpResolver needs updating, or the graph is not micro-friendly.
TFLM_SAFE = {"MUL", "CONV_2D", "DEPTHWISE_CONV_2D", "REDUCE_MAX", "MEAN",
             "SOFTMAX", "FULLY_CONNECTED", "MAX_POOL_2D", "AVERAGE_POOL_2D",
             "QUANTIZE", "DEQUANTIZE", "DELEGATE"}
DYNAMIC_SHAPE_OPS = {"SHAPE", "STRIDED_SLICE", "PACK", "RESHAPE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out-dir", default="deploy")
    ap.add_argument("--var-name", default="sparrownet_model_data")
    args = ap.parse_args()

    with open(os.path.join(args.results_dir, "summary.json")) as f:
        s = json.load(f)
    print(f"=== finalizing {s['tag']} ===")
    print(f"  arch={s['arch']} frontend={s['frontend']} n_fft={s['n_fft']} seed={s['seed']}")

    if s["arch"] == "sparrownet":
        model = build_sparrownet(input_shape=(N_FRAMES, N_MELS, 1),
                                 n_time_stages=s["stages"], width=tuple(s["width"]),
                                 batch_norm=True)
    else:
        model = build_drongonet_micro(input_shape=(N_FRAMES, N_MELS, 1))
    model.load_weights(s["weights"])
    print(f"  weights loaded: {model.count_params()} params")

    mel, labels, groups, _ = mybad_cache.load(n_mels=N_MELS, n_fft=s["n_fft"])
    tr, va, te = mybad_cache.grouped_split(groups, labels, seed=s["seed"])
    norm = normalize_np if s["frontend"] == "db" else log1p_minmax_np

    def rep():
        rng = np.random.RandomState(s["seed"])
        for i in rng.choice(tr, size=500, replace=False):
            yield [norm(np.asarray(mel[int(i)]))[None, ..., None].astype(np.float32)]

    os.makedirs(args.out_dir, exist_ok=True)
    tflite_path = os.path.join(args.out_dir, f"{s['arch']}_deploy_int8.tflite")
    size = convert_int8(model, rep, tflite_path)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    ops = collections.Counter(o["op_name"] for o in interp._get_ops_details())
    td = interp.get_tensor_details()
    dyn = [t for t in td if len(t["shape"]) == 0 or any(int(v) <= 0 for v in t["shape"])]

    print(f"  ops: {dict(ops)}")
    unsafe = set(ops) - TFLM_SAFE
    dynamic = set(ops) & DYNAMIC_SHAPE_OPS
    if unsafe or dynamic or dyn:
        print(f"  WARNING unsafe={unsafe} dynamic_ops={dynamic} dyn_tensors={len(dyn)}")
    else:
        print(f"  TFLM-safe: no dynamic-shape ops, {len(dyn)} dynamic tensors")

    verdict = ("OK" if size <= SIZE_SOFT_LIMIT_BYTES else
               "OVER_SOFT" if size <= SIZE_HARD_LIMIT_BYTES else "OVER_HARD")
    print(f"  size {size/1024:.2f} KB -> {verdict}")

    # verify the rebuilt graph still performs, on the held-out MyBAD sources
    X = norm(np.array([mel[i] for i in te]))[..., None]
    Y = labels[te]
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    B = 256
    interp.resize_tensor_input(inp["index"], [B, N_FRAMES, N_MELS, 1])
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    isc, izp = inp["quantization"]; osc, ozp = out["quantization"]
    probs = np.zeros(len(Y), np.float32)
    for i in range(0, len(Y), B):
        c = X[i:i + B]
        q = np.round(c / isc + izp).clip(-128, 127).astype(np.int8)
        if len(q) < B:
            q = np.concatenate([q, np.zeros((B - len(q),) + q.shape[1:], np.int8)])
        interp.set_tensor(inp["index"], q); interp.invoke()
        probs[i:i + len(c)] = (interp.get_tensor(out["index"])[:len(c), 1].astype(
            np.float32) - ozp) * osc
    auc = float(roc_auc_score(Y, probs))
    print(f"  MyBAD test AUC (held-out sources, static graph): {auc:.4f} "
          f"(was {s['mybad_test_auc_int8']:.4f})")

    rows, chosen = threshold_sweep(Y, probs)
    header, meta, _ = to_header(tflite_path, args.var_name)
    hpath = os.path.join(args.out_dir, "sparrownet-micro.h")
    with open(hpath, "w") as f:
        f.write(header)
    print(f"  wrote {tflite_path} and {hpath}")

    info = {"source_run": s["tag"], "arch": s["arch"], "frontend": s["frontend"],
            "n_fft": s["n_fft"], "n_mels": N_MELS, "n_frames": N_FRAMES,
            "params": int(model.count_params()), "size_bytes": size,
            "size_kb": round(size/1024, 2), "verdict": verdict,
            "ops": dict(ops), "tflm_safe": not (unsafe or dynamic or dyn),
            "quantization": meta, "mybad_test_auc_int8": auc,
            "threshold_sweep": rows, "chosen_tau": chosen}
    with open(os.path.join(args.out_dir, "deploy_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  operating points: "
          + ", ".join(f"tau={r['tau']:.2f} R={r['recall']:.3f} P={r['precision']:.3f}"
                      for r in rows if r["tau"] in (0.30, 0.40, 0.50)))


if __name__ == "__main__":
    main()
