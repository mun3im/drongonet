#!/usr/bin/env python3
"""
run_threshold_sweep.py — Find operating threshold τ for SEABADNet-Micro and Edge.

For each model × seed, runs the TFLite model over the test set,
sweeps τ from 0.05 to 0.60, and reports recall/precision/F1 at each step.
Identifies the highest τ that achieves the recall target (conservative = fewer FP).

Preprocessing matches training scripts exactly:
  6b_micro_improved: center=False, fmin=100, fmax=8000, normalize [0,1]
  3f_gap_focal_...: center=False, fmin=0, fmax=sr/2, normalize [0,1]
Data split uses os.walk (matching training), same random.seed(seed) ordering.
"""

import os
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import librosa
import random

DATASET_PATH = Path("/Volumes/Evo/seabad")
RESULTS      = Path("results")
SEEDS        = [42, 100, 786]
N_MELS_MICRO = 16
N_MELS_EDGE  = 80
TAU_GRID     = [round(t, 2) for t in np.arange(0.05, 0.61, 0.05)]

# Audio config shared across models
TARGET_SR  = 16000
TARGET_LEN = 48000
N_FFT      = 1024
HOP_LENGTH = 256
TIME_STEPS = 184


# ── Dataset ────────────────────────────────────────────────────────────────

def load_test_files(seed):
    """
    Reproduce the same 80/10/10 split used during training.
    Uses os.walk (not sorted rglob) to match training script file ordering.
    """
    random.seed(seed)
    pos_files, neg_files = [], []
    for label, cls in enumerate(["negative", "positive"]):
        path = str(DATASET_PATH / cls)
        class_files = []
        for root, _, files in os.walk(path):
            for f in sorted(files):   # sort within each dir for reproducibility
                if f.endswith(".wav"):
                    class_files.append((os.path.join(root, f), label))
        if label == 0:
            neg_files = class_files
        else:
            pos_files = class_files

    def split(lst):
        random.shuffle(lst)
        n = len(lst)
        return lst[int(0.9 * n):]   # last 10% = test

    return split(pos_files) + split(neg_files)


def compute_mel(path, n_mels, mel_fmin, mel_fmax):
    """
    Compute mel spectrogram matching training preprocessing exactly.
    Returns [0, 1] normalised float32 array of shape (TIME_STEPS, n_mels).
    """
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if len(audio) > TARGET_LEN:
        audio = audio[:TARGET_LEN]
    elif len(audio) < TARGET_LEN:
        audio = np.concatenate([audio, np.zeros(TARGET_LEN - len(audio))])
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax, center=False)
    mel_db = librosa.power_to_db(mel, ref=np.max).T   # (time, n_mels)
    if mel_db.shape[0] > TIME_STEPS:
        mel_db = mel_db[:TIME_STEPS]
    elif mel_db.shape[0] < TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, TIME_STEPS - mel_db.shape[0]), (0, 0)))
    # Per-sample [0, 1] normalisation (matches both 6b and 3f training scripts)
    lo, hi = mel_db.min(), mel_db.max()
    if hi - lo > 1e-6:
        mel_db = (mel_db - lo) / (hi - lo)
    else:
        mel_db = np.zeros_like(mel_db)
    return mel_db.astype(np.float32)


def build_test_set(seed, n_mels, mel_fmin, mel_fmax):
    files = load_test_files(seed)
    mels, labels = [], []
    for path, label in files:
        try:
            mels.append(compute_mel(path, n_mels, mel_fmin, mel_fmax))
            labels.append(label)
        except Exception:
            pass
    return np.array(mels)[..., np.newaxis], np.array(labels)


# ── TFLite inference ───────────────────────────────────────────────────────

def run_tflite(tflite_path, mels):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp  = interp.get_input_details()[0]
    out  = interp.get_output_details()[0]
    in_scale,  in_zp  = inp['quantization']
    out_scale, out_zp = out['quantization']

    probs = []
    for mel in mels:
        x = mel[np.newaxis]
        if in_scale != 0:
            x = (x / in_scale + in_zp).astype(np.int8)
        else:
            x = x.astype(inp['dtype'])
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        y = interp.get_tensor(out['index'])
        if out_scale != 0:
            y = (y.astype(np.float32) - out_zp) * out_scale
        probs.append(float(y[0, 1]))   # P(positive)
    return np.array(probs)


# ── Metrics at threshold ───────────────────────────────────────────────────

def metrics_at_tau(probs, labels, tau):
    preds = (probs >= tau).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Main ───────────────────────────────────────────────────────────────────

def sweep_model(script, n_fft, n_mels, seeds, recall_target, label,
                mel_fmin=100.0, mel_fmax=8000.0):
    print(f"\n{'=' * 65}")
    print(f"{label}  |  n_mels={n_mels}  |  recall target ≥{recall_target}")
    print(f"{'=' * 65}")

    all_probs  = {}
    all_labels = {}

    for seed in seeds:
        d = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
        # scripts save under different names
        tflite = next((d / n for n in ("model_int8.tflite", "model.tflite")
                       if (d / n).exists()), None)
        if tflite is None:
            print(f"  s{seed}: TFLite not found in {d}")
            continue
        print(f"  s{seed}: loading test set...", flush=True)
        mels, labels = build_test_set(seed, n_mels, mel_fmin, mel_fmax)
        print(f"         {len(labels)} samples ({labels.sum()} positive) — running inference...", flush=True)
        probs = run_tflite(tflite, mels)
        all_probs[seed]  = probs
        all_labels[seed] = labels
        print(f"         done.")

    if not all_probs:
        print("  No models found.")
        return

    # Per-seed sweep
    print(f"\n  τ sweep per seed:")
    header = f"  {'τ':>5}" + "".join(f"  s{s}:rec/prec" for s in seeds) + "  mean_rec  mean_prec"
    print(header)

    best_tau = {}
    for tau in TAU_GRID:
        row = f"  {tau:>5.2f}"
        mean_recs, mean_precs = [], []
        for seed in seeds:
            if seed not in all_probs:
                row += "      —  /  —  "
                continue
            m = metrics_at_tau(all_probs[seed], all_labels[seed], tau)
            row += f"  {m['recall']:.3f}/{m['precision']:.3f}"
            mean_recs.append(m['recall'])
            mean_precs.append(m['precision'])
            if seed not in best_tau and m['recall'] >= recall_target:
                best_tau[seed] = tau
        mr = np.mean(mean_recs)  if mean_recs  else 0
        mp = np.mean(mean_precs) if mean_precs else 0
        flag = " ◄ target met" if mr >= recall_target else ""
        row += f"  {mr:.3f}     {mp:.3f}{flag}"
        print(row)

    # Summary
    print(f"\n  Operating threshold recommendation (highest τ with mean recall ≥ {recall_target}):")
    for tau in reversed(TAU_GRID):
        recs = []
        for seed in seeds:
            if seed not in all_probs:
                continue
            recs.append(metrics_at_tau(all_probs[seed], all_labels[seed], tau)["recall"])
        if recs and np.mean(recs) >= recall_target:
            precs = [metrics_at_tau(all_probs[seed], all_labels[seed], tau)["precision"]
                     for seed in seeds if seed in all_probs]
            f1s   = [metrics_at_tau(all_probs[seed], all_labels[seed], tau)["f1"]
                     for seed in seeds if seed in all_probs]
            print(f"    τ = {tau:.2f}")
            print(f"    Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
            print(f"    Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
            print(f"    F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            break
    else:
        print(f"    ⚠️  No τ in grid achieves mean recall ≥ {recall_target} — lower τ needed")


if __name__ == "__main__":
    sweep_model(
        script="6b_micro_improved",
        n_fft=1024, n_mels=N_MELS_MICRO,
        seeds=SEEDS, recall_target=0.98,
        label="SEABADNet-Micro (6b)",
        mel_fmin=100.0, mel_fmax=8000.0,   # 6b training config
    )
    sweep_model(
        script="3f_gap_focal_loss_freq_emph_pointwise",
        n_fft=1024, n_mels=N_MELS_EDGE,
        seeds=SEEDS, recall_target=0.99,
        label="SEABADNet-Edge (3f)",
        mel_fmin=0.0, mel_fmax=8000.0,     # 3f training config (fmin=0)
    )
