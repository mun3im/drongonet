#!/usr/bin/env python3
"""
_zeroshot_common.py — shared harness for the TinyChirp -> SEABAD zero-shot evals (0f-0j).

Each 0f-0j script is a thin wrapper that supplies (a) the Corn-Bunting-trained INT8
model and (b) the matching input transform (mel for CNN-Mel / SqueezeNet-Mel, raw
waveform for CNN-Time / Transformer / SqueezeNet-Time). All five share one SEABAD
test split (seed=42) so their AUCs are directly comparable.

Deterministic: single eval, no seeds. Run each wrapper from the repo root, e.g.
  conda run -n tf215_gpu python pre-ablation/0f_tinychirp_cnnmel_zeroshot.py
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

SEABAD = '/Volumes/Evo/SEABAD'
SR, NFFT, HOP, NMELS, FRAMES = 16000, 1024, 256, 80, 184
TARGET_LEN = 16000 * 3        # 3 s @ 16 kHz = 48000 samples
SEED = 42


def seabad_test_files(seed=SEED):
    """Last 10% (test split) of each SEABAD class, shuffled deterministically by `seed`.
    Returns a list of (filepath, label) with label 0=negative, 1=positive."""
    import random
    random.seed(seed)
    pos, neg = [], []
    for label, cls in enumerate(['negative', 'positive']):   # 0=negative, 1=positive
        files = []
        for root, _, fnames in os.walk(os.path.join(SEABAD, cls)):
            files += [os.path.join(root, f) for f in fnames if f.endswith('.wav')]
        (neg if cls == 'negative' else pos).extend((f, label) for f in files)
    random.shuffle(pos)
    random.shuffle(neg)
    return pos[int(len(pos) * 0.9):] + neg[int(len(neg) * 0.9):]


def mel_input(path):
    """Log-mel, TinyChirp config (fmin=0, fmax=sr/2, center=False), min-max [0,1], pad to FRAMES.
    Returns model-ready tensor (1, 184, 80, 1)."""
    import librosa
    y, _ = librosa.load(path, sr=SR)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMELS,
        fmin=0.0, fmax=SR / 2.0, center=False)
    m = librosa.power_to_db(m, ref=np.max).T
    if m.shape[0] > FRAMES:
        m = m[:FRAMES]
    elif m.shape[0] < FRAMES:
        m = np.pad(m, ((0, FRAMES - m.shape[0]), (0, 0)), mode='constant', constant_values=0)
    m = (m - m.min()) / (m.max() - m.min() + 1e-08)
    return m.astype(np.float32)[np.newaxis, ..., np.newaxis]


def waveform_input(path):
    """Raw waveform @16kHz, pad/truncate to TARGET_LEN, peak-normalise.
    Returns model-ready tensor (1, 48000, 1)."""
    import librosa
    y, _ = librosa.load(path, sr=SR)
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.concatenate([y, np.zeros(TARGET_LEN - len(y), dtype=y.dtype)])
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32).reshape(1, TARGET_LEN, 1)


def run_zeroshot(tflite, input_fn, tag, out, note):
    """Run an INT8 TFLite model zero-shot on the SEABAD test split; save + return AUC."""
    import tensorflow as tf
    os.makedirs(out, exist_ok=True)
    files = seabad_test_files()

    interp = tf.lite.Interpreter(model_path=tflite)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp   = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    probs, labels = [], []
    for i, (path, label) in enumerate(files):
        if i % 1000 == 0:
            print(f'  {i}/{len(files)}')
        x = input_fn(path)
        q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], q)
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])
        probs.append((raw[0, 1].astype(np.float32) - out_zp) * out_scale)
        labels.append(label)

    auc = float(roc_auc_score(labels, probs))
    json.dump(
        {'tag': tag, 'n': len(files), 'n_pos': int(sum(labels)), 'auc': auc, 'note': note},
        open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print(f'\n=== {tag} (n={len(files)}) ===\n  AUC = {auc:.4f}\n  saved -> {out}/summary.json')
    return auc


def main_cli(model_dir_prefix, out_prefix, input_fn, tag_base, note, platform='linux'):
    """CLI entry for 0f-0j. Resolves the per-seed Corn-Bunting model from --seed and runs
    the zero-shot eval. The SEABAD test split stays fixed (seed=42) so only the *model*
    training seed varies, isolating seed-sensitivity. --model overrides the resolved path."""
    import argparse
    ap = argparse.ArgumentParser(description=f'{tag_base} (zero-shot on SEABAD)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Corn-Bunting model training seed (42/100/786)')
    ap.add_argument('--platform', default=platform, help='platform suffix in the model dir name')
    ap.add_argument('--model', default=None,
                    help='explicit model_int8.tflite path (overrides --seed resolution)')
    args = ap.parse_args()
    model = args.model or \
        f'results4arxiv/{model_dir_prefix}_r{args.seed}_{args.platform}/model_int8.tflite'
    return run_zeroshot(model, input_fn,
                        f'{tag_base}_r{args.seed}',
                        f'results4arxiv/{out_prefix}_r{args.seed}',
                        note)
