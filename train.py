"""
train.py — train + evaluate one SparrowNet (or drongonet-micro baseline) arm.

Protocol (DCASE cross-corpus): train on two corpora with a stratified in-domain
validation split, then test on the entirely held-out third corpus. In-domain val AUC
and cross-corpus test AUC are both reported — the gap between them is the whole point,
since the three corpora have very different positive rates (25% / 50% / 76%) and
recording conditions.

Reports, per arm: float32 val/test AUC, INT8 val/test AUC, params, .tflite size with a
verdict against the 10 KB soft / 16 KB hard budget, and a threshold sweep.

Usage:
    python train.py --arch sparrownet --held-out birdvox --seed 42
    python train.py --arch drongonet_micro --held-out birdvox --seed 42
"""

import os
import json
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from config import (
    DCASE_CORPORA, RESULTS_BASE, N_FRAMES, N_MELS, FULL_N_FRAMES,
    SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES,
)
from build_cache import load_cache
from dataset import make_dataset, stratified_split, ConcatMel, normalize_tf, normalize_np
from models import build_sparrownet, build_drongonet_micro, focal_loss
from benchmark_archs import BUILDERS as BENCHMARK_BUILDERS

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


# ---------------------------------------------------------------- data

def build_splits_single_corpus(corpus, seed, n_mels=N_MELS):
    """Diagnostic split: train/val/test all inside ONE corpus (70/15/15).

    Answers a question the cross-corpus protocol cannot: if cross-corpus AUC sits at
    chance, is the target corpus unlearnable with these features (a pipeline bug), or
    is it merely unreachable from the other two (genuine domain shift)? A high
    in-domain AUC here means the features and labels are sound."""
    mel, labels, _ = load_cache(corpus, mmap=True, n_mels=n_mels)
    rest_idx, test_idx = stratified_split(labels, val_fraction=0.15, seed=seed)
    rest_labels = labels[rest_idx]
    sub_tr, sub_va = stratified_split(rest_labels, val_fraction=0.176, seed=seed)

    tr_idx = rest_idx[sub_tr]
    va_idx = rest_idx[sub_va]
    return {
        "train_corpora": [corpus],
        "train_mel": mel, "train_labels": labels,
        "tr_idx": tr_idx, "va_idx": va_idx,
        "test_mel": IndexedMel(mel, test_idx),
        "test_labels": labels[test_idx],
    }


def build_splits(held_out, seed, n_mels=N_MELS):
    train_corpora = [c for c in DCASE_CORPORA if c != held_out]

    mels, labels = [], []
    for name in train_corpora:
        m, lab, _ = load_cache(name, mmap=True, n_mels=n_mels)
        mels.append(m)
        labels.append(lab)
    train_mel = ConcatMel(mels)
    train_labels = np.concatenate(labels)

    test_mel, test_labels, _ = load_cache(held_out, mmap=True, n_mels=n_mels)

    tr_idx, va_idx = stratified_split(train_labels, val_fraction=0.15, seed=seed)
    return {
        "train_corpora": train_corpora,
        "train_mel": train_mel, "train_labels": train_labels,
        "tr_idx": tr_idx, "va_idx": va_idx,
        "test_mel": test_mel, "test_labels": test_labels,
    }


class IndexedMel:
    """View of a ConcatMel/memmap restricted to a subset of rows."""

    def __init__(self, base, idx):
        self.base, self.idx = base, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


# ---------------------------------------------------------------- eval

def predict_keras(model, mel, labels, mode, batch_size=128, n_mels=N_MELS):
    ds = make_dataset(mel, labels, mode=mode, training=False, batch_size=batch_size,
                      augment=False, n_mels=n_mels)
    probs = model.predict(ds, verbose=0)
    return probs[:, 1]


def threshold_sweep(y_true, scores, targets=(0.95, 0.98)):
    rows = []
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        rows.append({"tau": t, "precision": float(p), "recall": float(r), "f1": float(f1)})

    chosen = {}
    for target in targets:
        ok = [r for r in rows if r["recall"] >= target]
        # highest tau that still meets the recall target => best precision (drongonet rule)
        chosen[f"tau_at_recall_{target}"] = max(ok, key=lambda r: r["tau"]) if ok else None
    return rows, chosen


# ---------------------------------------------------------------- tflite

def representative_dataset_fn(mel, labels, mode, n=500, seed=0):
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(labels), size=min(n, len(labels)), replace=False)
    n_frames = FULL_N_FRAMES if mode == "full" else N_FRAMES

    def gen():
        for i in pick:
            power = np.asarray(mel[int(i)], dtype=np.float32)
            if mode == "crop":
                start = max((power.shape[0] - n_frames) // 2, 0)
                power = power[start:start + n_frames]
            x = normalize_tf(tf.convert_to_tensor(power)).numpy()
            yield [x[np.newaxis, ..., np.newaxis].astype(np.float32)]
    return gen


def convert_int8(model, repr_fn, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = repr_fn
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(out_path, "wb") as f:
        f.write(blob)
    return len(blob)


def eval_tflite(tflite_path, mel, labels, mode, limit=None, batch=64, n_mels=N_MELS):
    """Batched INT8 inference. The interpreter is resized to `batch` so one invoke
    covers many clips; normalization is vectorized in numpy (see normalize_np)."""
    n_frames = FULL_N_FRAMES if mode == "full" else N_FRAMES
    n = len(labels) if limit is None else min(limit, len(labels))

    interp = tf.lite.Interpreter(model_path=tflite_path)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], [batch, n_frames, n_mels, 1])
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        idx = range(start, min(start + batch, n))
        chunk = np.stack([np.asarray(mel[i], dtype=np.float32) for i in idx])
        if mode == "crop":
            off = max((chunk.shape[1] - n_frames) // 2, 0)
            chunk = chunk[:, off:off + n_frames]

        x = normalize_np(chunk)[..., np.newaxis]
        q = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)

        if len(q) < batch:  # final short chunk: pad up to the allocated batch size
            pad = np.zeros((batch - len(q),) + q.shape[1:], dtype=np.int8)
            q = np.concatenate([q, pad])

        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out["index"])[:len(idx)]
        scores[start:start + len(idx)] = (raw[:, 1].astype(np.float32) - out_zp) * out_scale

    return scores, labels[:n]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sparrownet",
                    choices=["sparrownet", "drongonet_micro"] + list(BENCHMARK_BUILDERS))
    ap.add_argument("--n-mels", type=int, default=N_MELS,
                    help="must match a built cache; 80 for the bulbul/sparrow baselines")
    ap.add_argument("--held-out", default="birdvox", choices=list(DCASE_CORPORA))
    ap.add_argument("--mode", default=None, choices=["full", "crop"],
                    help="default: full for sparrownet, crop for drongonet_micro")
    ap.add_argument("--stages", type=int, default=6, help="sparrownet time stages (RF)")
    ap.add_argument("--pool", default="max", choices=["max", "avg"])
    ap.add_argument("--head-kernel", type=int, default=1)
    ap.add_argument("--width", default="8,16")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--qat", action="store_true",
                    help="quantization-aware training (drongonet lesson: PTQ is the "
                         "top recall killer)")
    ap.add_argument("--single-corpus", default=None, choices=list(DCASE_CORPORA),
                    help="diagnostic: train AND test inside one corpus (no cross-corpus)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tflite-limit", type=int, default=None,
                    help="cap clips used for INT8 eval (speed); default all")
    args = ap.parse_args()

    # benchmark archs are whole-clip models like sparrownet; only drongonet_micro
    # defaults to the 3s crop (its native input geometry)
    mode = args.mode or ("crop" if args.arch == "drongonet_micro" else "full")
    n_mels = args.n_mels
    width = tuple(int(v) for v in args.width.split(","))
    tf.keras.utils.set_random_seed(args.seed)

    tag = args.tag or (
        f"{args.arch}_{mode}" + (f"_m{n_mels}" if n_mels != N_MELS else "")
        + f"_ho-{args.held_out}"
        + (f"_st{args.stages}_{args.pool}" if args.arch == "sparrownet" else "")
        + (f"_hk{args.head_kernel}" if args.head_kernel > 1 else "")
        + ("_noaug" if args.no_augment else "")
        + ("_qat" if args.qat else "")
        + f"_s{args.seed}"
    )
    out_dir = os.path.join(RESULTS_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {tag} ===", flush=True)

    if args.single_corpus:
        S = build_splits_single_corpus(args.single_corpus, args.seed, n_mels=n_mels)
    else:
        S = build_splits(args.held_out, args.seed, n_mels=n_mels)
    n_frames = FULL_N_FRAMES if mode == "full" else N_FRAMES
    target = args.single_corpus or args.held_out
    print(f"train corpora {S['train_corpora']} -> "
          + (f"in-domain test '{target}'" if args.single_corpus
             else f"held out '{target}'"))
    print(f"train {len(S['tr_idx'])} / val {len(S['va_idx'])} / test {len(S['test_labels'])}"
          f"  input ({n_frames},{n_mels},1)")

    tr_mel = IndexedMel(S["train_mel"], S["tr_idx"])
    tr_lab = S["train_labels"][S["tr_idx"]]
    va_mel = IndexedMel(S["train_mel"], S["va_idx"])
    va_lab = S["train_labels"][S["va_idx"]]

    train_ds = make_dataset(tr_mel, tr_lab, mode=mode, training=True,
                            batch_size=args.batch_size, augment=not args.no_augment,
                            seed=args.seed, n_mels=n_mels)
    val_ds = make_dataset(va_mel, va_lab, mode=mode, training=False,
                          batch_size=args.batch_size, augment=False, n_mels=n_mels)

    if args.arch == "sparrownet":
        model = build_sparrownet(input_shape=(n_frames, n_mels, 1),
                                 n_time_stages=args.stages, pool=args.pool,
                                 head_kernel_t=args.head_kernel, width=width,
                                 dropout=args.dropout)
        rf = model.receptive_field_frames
    elif args.arch == "drongonet_micro":
        model = build_drongonet_micro(input_shape=(n_frames, n_mels, 1),
                                      dropout=args.dropout)
        rf = None
    else:
        model = BENCHMARK_BUILDERS[args.arch](input_shape=(n_frames, n_mels, 1))
        rf = None

    base_params = int(model.count_params())
    model_name = model.name
    if args.qat:
        from qat import apply_qat, count_quantized_layers
        model = apply_qat(model)
        print(f"QAT: {count_quantized_layers(model)} quant-wrapped layers")

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=focal_loss(2.0, 0.5), metrics=[tf.keras.metrics.AUC(name="auc")])
    print(f"{model_name}: {base_params} params"
          + (f", RF={rf} frames ({rf*256/16000:.2f}s)" if rf else ""))

    ckpt = os.path.join(out_dir, "best.weights.h5")
    callbacks = [
        # val_loss, not val_accuracy (drongonet lesson: stopping on accuracy traps the
        # model in local minima), paired with ReduceLROnPlateau.
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                             min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                           save_best_only=True, save_weights_only=True),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=callbacks, verbose=2)
    train_min = (time.time() - t0) / 60
    print(f"trained {len(hist.history['loss'])} epochs in {train_min:.1f} min")

    # ---- float32 evaluation
    va_scores = predict_keras(model, va_mel, va_lab, mode, n_mels=n_mels)
    te_scores = predict_keras(model, S["test_mel"], S["test_labels"], mode, n_mels=n_mels)
    val_auc = float(roc_auc_score(va_lab, va_scores))
    test_auc = float(roc_auc_score(S["test_labels"], te_scores))
    kind = "in-domain" if args.single_corpus else "cross-corpus"
    print(f"float32:  val AUC {val_auc:.4f} | {kind}({target}) AUC {test_auc:.4f}")

    # ---- INT8 (post-training quantization, calibrated on the training split)
    tflite_path = os.path.join(out_dir, f"{args.arch}_int8.tflite")
    repr_fn = representative_dataset_fn(tr_mel, tr_lab, mode, n=500, seed=args.seed)
    size_bytes = convert_int8(model, repr_fn, tflite_path)

    within_soft = size_bytes <= SIZE_SOFT_LIMIT_BYTES
    within_hard = size_bytes <= SIZE_HARD_LIMIT_BYTES
    verdict = "OK" if within_soft else ("OVER_SOFT" if within_hard else "OVER_HARD")
    print(f"INT8 size: {size_bytes/1024:.2f} KB -> {verdict} "
          f"(soft {SIZE_SOFT_LIMIT_BYTES/1024:.0f} KB / hard {SIZE_HARD_LIMIT_BYTES/1024:.0f} KB)")

    q_va_scores, q_va_lab = eval_tflite(tflite_path, va_mel, va_lab, mode,
                                        limit=args.tflite_limit, n_mels=n_mels)
    q_te_scores, q_te_lab = eval_tflite(tflite_path, S["test_mel"], S["test_labels"],
                                        mode, limit=args.tflite_limit, n_mels=n_mels)
    q_val_auc = float(roc_auc_score(q_va_lab, q_va_scores))
    q_test_auc = float(roc_auc_score(q_te_lab, q_te_scores))
    print(f"INT8:     val AUC {q_val_auc:.4f} | {kind}({target}) AUC {q_test_auc:.4f}"
          f"  (delta {q_test_auc-test_auc:+.4f})")

    # Threshold sweep on the INT8 model (drongonet lesson: thresholds shift 0.01-0.05
    # under quantization, so sweeping on float32 and reusing tau is wrong).
    sweep_rows, chosen = threshold_sweep(q_te_lab, q_te_scores)

    result = {
        "tag": tag, "arch": args.arch, "mode": mode, "n_mels": n_mels,
        "held_out": args.held_out, "single_corpus": args.single_corpus,
        "train_corpora": S["train_corpora"], "seed": args.seed,
        "params": base_params,
        "qat": args.qat,
        "receptive_field_frames": rf,
        "receptive_field_seconds": (rf * 256 / 16000) if rf else None,
        "stages": args.stages if args.arch == "sparrownet" else None,
        "pool": args.pool if args.arch == "sparrownet" else None,
        "head_kernel": args.head_kernel if args.arch == "sparrownet" else None,
        "augment": not args.no_augment,
        "epochs_run": len(hist.history["loss"]), "train_minutes": round(train_min, 2),
        "float32": {"val_auc": val_auc, "test_auc": test_auc},
        "int8": {"val_auc": q_val_auc, "test_auc": q_test_auc,
                 "size_bytes": size_bytes, "size_kb": round(size_bytes / 1024, 2),
                 "within_soft_limit": within_soft, "within_hard_limit": within_hard,
                 "verdict": verdict},
        "threshold_sweep_int8_crosscorpus": sweep_rows,
        "chosen_tau": chosen,
        "n_train": int(len(S["tr_idx"])), "n_val": int(len(S["va_idx"])),
        "n_test": int(len(S["test_labels"])),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_dir}/summary.json")
    return result


if __name__ == "__main__":
    main()
