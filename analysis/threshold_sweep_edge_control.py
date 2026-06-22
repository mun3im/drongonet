#!/usr/bin/env python3
"""
threshold_sweep_edge_control.py
================================
Re-runs the SEABADNet-Edge threshold sweep for all three seeds with XNNPACK
disabled, saving results to a separate control directory.

Purpose: diagnose why locked-τ values differ between the Mac run (s42=0.50,
s100=0.40, s786=0.30) and the Linux/XNNPACK run (s42=0.60, s100=0.55,
s786=0.45). If disabling XNNPACK recovers the Mac τ values, the divergence
is caused by XNNPACK INT8 kernel rounding differences.

Usage:
    conda run -n tf215_gpu python threshold_sweep_edge_control.py \
        --results-dir results4arxiv \
        --cache-dir /Volumes/Evo/cache4arxiv_fft1024_m80 \
        --out-dir   results4arxiv_edge_control
"""

import os
# Must be set before TensorFlow is imported so the XNNPACK delegate is never loaded.
os.environ['TFLITE_DISABLE_XNNPACK_DELEGATE'] = '1'

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SEEDS         = [42, 100, 786]
DIRNAME_TMPL  = '6c_edge_final_fft1024_m80_s{seed}'
TFLITE_NAME   = 'model_int8.tflite'
TARGET_RECALL = 0.99
THRESHOLDS    = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Mac-run reference from CLAUDE.md (for comparison printout)
MAC_REFERENCE = {
    42:  dict(tau=0.50, recall=0.9900, precision=0.9892, f1=0.9896, fpr=0.0108),
    100: dict(tau=0.40, recall=0.9900, precision=0.9837, f1=0.9868, fpr=0.0164),
    786: dict(tau=0.30, recall=0.9900, precision=0.9864, f1=0.9882, fpr=0.0136),
}


def load_test_mels(cache_dir: Path):
    cache_file = cache_dir / 'test' / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Test cache not found: {cache_file}")
    data = np.load(cache_file)
    mels, labels = data['mels'], data['labels']
    logger.info(f"Loaded {len(mels)} test samples from {cache_file}")
    return mels, labels


def run_tflite_inference(tflite_path: Path, mels: np.ndarray):
    import tensorflow as tf
    # XNNPACK already disabled via env var; double-confirm in log.
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_d  = interpreter.get_input_details()[0]
    out_d  = interpreter.get_output_details()[0]
    in_sc, in_zp   = inp_d['quantization']
    out_sc, out_zp = out_d['quantization']
    logger.info(f"Input  dtype={inp_d['dtype'].__name__}  scale={in_sc:.6f}  zp={in_zp}")
    logger.info(f"Output dtype={out_d['dtype'].__name__}  scale={out_sc:.6f}  zp={out_zp}")

    mels_4d = mels[..., np.newaxis]
    probs, times = [], []
    for sample in mels_4d:
        inp = sample[np.newaxis]
        inp_q = np.clip(np.round(inp / in_sc + in_zp), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp_d['index'], inp_q)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - t0)
        out_q = interpreter.get_tensor(out_d['index'])
        p1 = (out_q[0, 1].astype(np.float32) - out_zp) * out_sc
        probs.append(p1)

    return np.array(probs), float(np.mean(times)) * 1000


def sweep_thresholds(probs, labels):
    from sklearn.metrics import confusion_matrix
    rows = []
    for tau in THRESHOLDS:
        preds = (probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append(dict(tau=tau, recall=recall, precision=precision,
                         f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn))
    return rows


def find_locked_tau(rows):
    candidates = [r for r in rows if r['recall'] >= TARGET_RECALL]
    return max(candidates, key=lambda r: r['tau']) if candidates else rows[0]


def write_sweep_table(rows, locked, out_path: Path, auc: float):
    lines = [
        'SEABADNet-Edge — Threshold Sweep (XNNPACK disabled)',
        f'Target recall : ≥{TARGET_RECALL}',
        f'AUC           : {auc:.4f}',
        '',
        f'{"τ":>8}  {"Recall":>8}  {"Precision":>10}  {"F1":>8}  {"FPR":>8}  '
        f'{"TP":>6}  {"FP":>6}  {"FN":>6}  {"TN":>6}',
        '-' * 72,
    ]
    for r in rows:
        marker = ' <-- LOCKED' if r['tau'] == locked['tau'] else ''
        lines.append(
            f"  {r['tau']:.2f}    {r['recall']:.4f}      {r['precision']:.4f}   "
            f"{r['f1']:.4f}   {r['fpr']:.4f}    {r['tp']:4d}    {r['fp']:4d}    "
            f"{r['fn']:4d}    {r['tn']:4d}{marker}"
        )
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results4arxiv')
    parser.add_argument('--cache-dir',   default='/Volumes/Evo/cache4arxiv_fft1024_m80')
    parser.add_argument('--out-dir',     default='results4arxiv_edge_control')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir   = Path(args.cache_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mels, true_labels = load_test_mels(cache_dir)

    control_locked = {}
    for seed in SEEDS:
        seed_dir   = results_dir / DIRNAME_TMPL.format(seed=seed)
        tflite_path = seed_dir / TFLITE_NAME
        if not tflite_path.exists():
            logger.error(f"Missing: {tflite_path}")
            continue

        size_kb = tflite_path.stat().st_size / 1024
        logger.info(f"\n=== seed {seed} ({size_kb:.2f} KB) ===")

        probs, avg_ms = run_tflite_inference(tflite_path, mels)
        auc = float(roc_auc_score(true_labels, probs))
        logger.info(f"AUC: {auc:.4f}  latency: {avg_ms:.3f} ms")

        rows   = sweep_thresholds(probs, true_labels)
        locked = find_locked_tau(rows)
        control_locked[seed] = locked

        seed_out = out_dir / f's{seed}'
        seed_out.mkdir(exist_ok=True)
        write_sweep_table(rows, locked, seed_out / 'threshold_sweep.txt', auc)
        logger.info(f"Locked τ={locked['tau']:.2f}  recall={locked['recall']:.4f}  "
                    f"precision={locked['precision']:.4f}  f1={locked['f1']:.4f}")
        logger.info(f"Saved → {seed_out}/threshold_sweep.txt")

    # ── comparison table ─────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('COMPARISON: Mac reference vs Linux XNNPACK vs Linux no-XNNPACK (control)')
    print('='*70)
    print(f"{'Seed':>6}  {'Platform':>20}  {'τ':>5}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")
    print('-'*70)
    for seed in SEEDS:
        mac = MAC_REFERENCE[seed]
        ctrl = control_locked.get(seed)
        print(f"  s{seed}  {'Mac (original)':>20}  {mac['tau']:.2f}  "
              f"{mac['recall']:.4f}  {mac['precision']:.4f}      {mac['f1']:.4f}")
        if ctrl:
            print(f"  s{seed}  {'Linux no-XNNPACK':>20}  {ctrl['tau']:.2f}  "
                  f"{ctrl['recall']:.4f}  {ctrl['precision']:.4f}      {ctrl['f1']:.4f}")
        # Linux-XNNPACK values from the run we already did
        xnnpack = {42: (0.60, 0.9900, 0.9841, 0.9870),
                   100: (0.55, 0.9916, 0.9880, 0.9898),
                   786: (0.45, 0.9900, 0.9872, 0.9886)}[seed]
        print(f"  s{seed}  {'Linux XNNPACK':>20}  {xnnpack[0]:.2f}  "
              f"{xnnpack[1]:.4f}  {xnnpack[2]:.4f}      {xnnpack[3]:.4f}")
        print()

    comp_path = out_dir / 'comparison.txt'
    lines = ['Mac vs Linux XNNPACK vs Linux no-XNNPACK\n',
             f"{'Seed':>6}  {'Platform':>20}  {'τ':>5}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}\n"]
    for seed in SEEDS:
        mac  = MAC_REFERENCE[seed]
        ctrl = control_locked.get(seed)
        xnnpack = {42: (0.60, 0.9900, 0.9841, 0.9870),
                   100: (0.55, 0.9916, 0.9880, 0.9898),
                   786: (0.45, 0.9900, 0.9872, 0.9886)}[seed]
        lines.append(f"s{seed} Mac          τ={mac['tau']:.2f}  recall={mac['recall']:.4f}  "
                     f"prec={mac['precision']:.4f}  f1={mac['f1']:.4f}\n")
        if ctrl:
            lines.append(f"s{seed} no-XNNPACK   τ={ctrl['tau']:.2f}  recall={ctrl['recall']:.4f}  "
                         f"prec={ctrl['precision']:.4f}  f1={ctrl['f1']:.4f}\n")
        lines.append(f"s{seed} XNNPACK      τ={xnnpack[0]:.2f}  recall={xnnpack[1]:.4f}  "
                     f"prec={xnnpack[2]:.4f}  f1={xnnpack[3]:.4f}\n\n")
    comp_path.write_text(''.join(lines))
    print(f"Comparison saved → {comp_path}")


if __name__ == '__main__':
    main()
