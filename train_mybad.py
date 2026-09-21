"""
train_mybad.py — train on MyBAD (Malaysian) with source-grouped splits.

The objective is acceptable accuracy for Malaysian bird detection, so MyBAD is the
training set and the primary test set; DCASE is only a secondary cross-corpus check.
This inverts everything earlier in the project, where models trained on DCASE and
MyBAD was the external test.

Splits are by SOURCE RECORDING, never by clip: MyBAD's 56,000 clips come from 37,136
recordings (~1.9 segments each for positives), and a clip-level split would put
segments of one recording on both sides -- measured at 48.7% of test clips leaking.

The DCASE cross-check is reported but is NOT clean for any corpus: MyBAD's negatives
absorb the entire negative half of all three (freefield1010, warblr, BirdVox -- 19,643
clips, 35.1% of MyBAD), so a MyBAD-trained model has seen part of every DCASE corpus.
The numbers are kept for continuity with earlier phases and flagged, not trusted.

Usage:
    python train_mybad.py --frontend db    --n-fft 1024 --seed 42
    python train_mybad.py --frontend log1p --n-fft 512  --seed 42   # MyBAD's own recipe
"""

import os
import json
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from config import (
    N_FRAMES, N_MELS, RESULTS_BASE, DCASE_CORPORA,
    SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES,
)
from models import build_sparrownet, build_drongonet_micro, focal_loss
from dataset import make_dataset, FRONTENDS_TF
from train import threshold_sweep, convert_int8, eval_tflite, IndexedMel
import mybad_cache
from build_cache import load_cache

try:
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


def representative_from(mel, n=500, seed=0, frontend="db"):
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(mel), size=min(n, len(mel)), replace=False)
    norm = FRONTENDS_TF[frontend]

    def gen():
        for i in pick:
            p = np.asarray(mel[int(i)], dtype=np.float32)
            x = norm(tf.convert_to_tensor(p)).numpy()
            yield [x[np.newaxis, ..., np.newaxis].astype(np.float32)]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sparrownet",
                    choices=["sparrownet", "drongonet_micro"])
    ap.add_argument("--frontend", default="db", choices=["db", "log1p"])
    ap.add_argument("--n-fft", type=int, default=1024, choices=[512, 1024])
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--width", default="8,8")
    ap.add_argument("--pool", default="max", choices=["max", "avg"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-bn", action="store_true")
    ap.add_argument("--bn", action="store_true", help="drongonet_micro control arm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--pos-source-frac", type=float, default=1.0,
                    help="keep only this fraction of positive TRAINING source "
                         "recordings (learning curve; test set stays fixed)")
    ap.add_argument("--segments-per-source", type=int, default=0,
                    help="cap training clips per source recording (0 = no cap). "
                         "MyBAD has ~2 clips per source, numbered _1/_2 by clip index. "
                         "Measured: clip _1 averages 2.2 dB hotter than _2, consistent "
                         "with the extractor emitting its picks strongest-first.")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    width = tuple(int(v) for v in args.width.split(","))
    tf.keras.utils.set_random_seed(args.seed)

    tag = args.tag or (f"MB_{args.arch}_{args.frontend}_fft{args.n_fft}"
                       + (f"_st{args.stages}_w{args.width.replace(',', '-')}"
                          if args.arch == "sparrownet" else "")
                       + f"_s{args.seed}")
    out_dir = os.path.join(RESULTS_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {tag} ===", flush=True)

    mel, labels, groups, origins = mybad_cache.load(n_mels=N_MELS, n_fft=args.n_fft)
    tr_idx, va_idx, te_idx = mybad_cache.grouped_split(groups, labels, seed=args.seed)

    # the split is the load-bearing claim of this script; assert it, don't trust it
    gtr, gva, gte = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (gtr & gva) and not (gtr & gte) and not (gva & gte), \
        "source-recording leakage between splits"
    print(f"MyBAD: train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} clips "
          f"from {len(gtr)}/{len(gva)}/{len(gte)} sources (no overlap)")
    print(f"frontend={args.frontend} n_fft={args.n_fft} input ({N_FRAMES},{N_MELS},1)")

    # Learning-curve / segment-cap subsetting. Applied to TRAINING ONLY -- val and
    # test keep every clip, so every point on the curve is scored on the same data.
    if args.pos_source_frac < 1.0 or args.segments_per_source:
        keep = []
        by_src = {}
        for i in tr_idx:
            by_src.setdefault(groups[i], []).append(i)
        pos_srcs = sorted(s for s, ix in by_src.items() if labels[ix[0]] == 1)
        neg_srcs = sorted(s for s, ix in by_src.items() if labels[ix[0]] == 0)
        rng = np.random.RandomState(args.seed)
        if args.pos_source_frac < 1.0:
            n = int(round(len(pos_srcs) * args.pos_source_frac))
            pos_srcs = list(rng.permutation(pos_srcs)[:n])
        for s in pos_srcs + neg_srcs:
            ix = sorted(by_src[s])
            keep.extend(ix[:args.segments_per_source] if args.segments_per_source else ix)
        tr_idx = np.array(sorted(keep))
        rng.shuffle(tr_idx)
        npos = int(labels[tr_idx].sum())
        print(f"subset: {len(tr_idx)} training clips ({npos} pos / {len(tr_idx)-npos} neg) "
              f"from {len(pos_srcs)} positive sources "
              f"(frac={args.pos_source_frac}, seg_cap={args.segments_per_source or 'none'})")

    tr_mel, tr_lab = IndexedMel(mel, tr_idx), labels[tr_idx]
    va_mel, va_lab = IndexedMel(mel, va_idx), labels[va_idx]
    te_mel, te_lab = IndexedMel(mel, te_idx), labels[te_idx]

    kw = dict(mode="crop", n_frames=N_FRAMES, n_mels=N_MELS, frontend=args.frontend)
    train_ds = make_dataset(tr_mel, tr_lab, training=True, batch_size=args.batch_size,
                            augment=not args.no_augment, seed=args.seed, **kw)
    val_ds = make_dataset(va_mel, va_lab, training=False, batch_size=args.batch_size,
                          augment=False, **kw)

    if args.arch == "sparrownet":
        model = build_sparrownet(input_shape=(N_FRAMES, N_MELS, 1),
                                 n_time_stages=args.stages, pool=args.pool, width=width,
                                 dropout=args.dropout, batch_norm=not args.no_bn)
        rf = model.receptive_field_frames
    else:
        model = build_drongonet_micro(input_shape=(N_FRAMES, N_MELS, 1),
                                      dropout=args.dropout, batch_norm=args.bn)
        rf = None

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=focal_loss(2.0, 0.5), metrics=[tf.keras.metrics.AUC(name="auc")])
    print(f"{model.name}: {model.count_params()} params"
          + (f", RF={rf}f ({rf*256/16000:.2f}s)" if rf else ""))

    ckpt = os.path.join(out_dir, "best.weights.h5")
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4,
                                             min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                           save_best_only=True, save_weights_only=True),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=cbs, verbose=2)
    mins = (time.time() - t0) / 60

    def score(m, lab):
        ds = make_dataset(m, lab, training=False, batch_size=128, augment=False, **kw)
        return model.predict(ds, verbose=0)[:, 1]

    val_auc = float(roc_auc_score(va_lab, score(va_mel, va_lab)))
    test_auc = float(roc_auc_score(te_lab, score(te_mel, te_lab)))
    print(f"float32:  MyBAD val {val_auc:.4f} | MyBAD test (held-out sources) {test_auc:.4f}")

    # INT8, calibrated on the MyBAD training split only
    tflite_path = os.path.join(out_dir, f"{args.arch}_int8.tflite")
    size = convert_int8(model, representative_from(tr_mel, 500, args.seed,
                                                   args.frontend), tflite_path)
    verdict = ("OK" if size <= SIZE_SOFT_LIMIT_BYTES else
               "OVER_SOFT" if size <= SIZE_HARD_LIMIT_BYTES else "OVER_HARD")
    print(f"INT8 size: {size/1024:.2f} KB -> {verdict}")

    q_scores, q_lab = eval_tflite(tflite_path, te_mel, te_lab, "crop", n_mels=N_MELS)
    q_test_auc = float(roc_auc_score(q_lab, q_scores))
    print(f"INT8:     MyBAD test {q_test_auc:.4f} (delta {q_test_auc-test_auc:+.4f})")

    # secondary: DCASE cross-corpus, EXCLUDING freefield1010 (it is inside MyBAD)
    dcase = {}
    for corpus in DCASE_CORPORA:
        # Every DCASE corpus's negatives are inside MyBAD, so none is a clean
        # cross-corpus test. Score them anyway for continuity, but record the taint.
        contaminated = corpus in mybad_cache.DCASE_DERIVED
        dm, dl, _ = load_cache(corpus, mmap=True)
        ds = make_dataset(dm, dl, mode="crop", n_frames=N_FRAMES, n_mels=N_MELS,
                          frontend=args.frontend, training=False, batch_size=128,
                          augment=False)
        p = model.predict(ds, verbose=0)[:, 1]
        dcase[corpus] = {"auc": float(roc_auc_score(dl, p)), "n": int(len(dl)),
                         "contaminated": contaminated,
                         "note": ("this corpus's negatives are inside MyBAD's training "
                                  "pool" if contaminated else "")}
        print(f"DCASE cross-corpus {corpus}: AUC {dcase[corpus]['auc']:.4f}"
              + ("  [CONTAMINATED: its negatives are in MyBAD]" if contaminated else ""))

    rows, chosen = threshold_sweep(q_lab, q_scores)
    result = {
        "tag": tag, "arch": args.arch, "trained_on": "mybad",
        "frontend": args.frontend, "n_fft": args.n_fft, "seed": args.seed,
        "params": int(model.count_params()), "stages": args.stages if rf else None,
        "width": list(width), "receptive_field_frames": rf,
        "epochs_run": len(hist.history["loss"]), "train_minutes": round(mins, 2),
        "pos_source_frac": args.pos_source_frac,
        "segments_per_source": args.segments_per_source,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
        "n_sources_train": len(gtr), "n_sources_test": len(gte),
        "split": "grouped_by_source_recording",
        "mybad_val_auc": val_auc, "mybad_test_auc": test_auc,
        "mybad_test_auc_int8": q_test_auc,
        "int8": {"size_bytes": size, "size_kb": round(size/1024, 2), "verdict": verdict},
        "dcase_crosscorpus": dcase,
        "threshold_sweep_int8_mybad_test": rows, "chosen_tau": chosen,
        "weights": ckpt,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
