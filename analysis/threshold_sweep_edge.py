#!/usr/bin/env python3
"""
threshold_sweep_edge.py: Threshold sweep for DrongoNet-Edge

Loads the saved INT8 TFLite model from 6b_edge_final_fft1024_m80_s42,
runs inference on the test set, sweeps thresholds, and locks the lowest
τ that achieves ≥0.99 recall.

Output:
  results/6b_edge_final_fft1024_m80_s42/threshold_sweep.txt  — full table
  results/6b_edge_final_fft1024_m80_s42/threshold_locked.txt — locked τ + metrics
  results/6b_edge_final_fft1024_m80_s42/threshold_sweep_pr.png

Usage:
  python threshold_sweep_edge.py \
      --results-dir results/6b_edge_final_fft1024_m80_s42 \
      --cache-dir   /Volumes/Evo/cache_seabad_m80
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_RECALL = 0.99
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_mels(cache_dir: Path):
    cache_file = cache_dir / 'test' / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Test cache not found: {cache_file}")
    data = np.load(cache_file)
    mels = data['mels'].astype(np.float32)
    labels = data['labels'].astype(np.int32)
    logger.info(f"Loaded {len(mels)} test samples from {cache_file}")
    return mels, labels


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def run_tflite_inference(tflite_path: Path, mels: np.ndarray):
    """Returns float probabilities for the positive class."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale,  input_zero_point  = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    logger.info(f"Input  dtype={input_details['dtype']}  scale={input_scale}  zp={input_zero_point}")
    logger.info(f"Output dtype={output_details['dtype']} scale={output_scale} zp={output_zero_point}")

    mels_4d = mels[..., np.newaxis]  # (N, H, W, 1)
    probabilities = []
    inference_times = []

    for i, sample in enumerate(mels_4d):
        inp = sample[np.newaxis]  # (1, H, W, 1)

        if input_scale != 0.0:
            inp = np.round(inp / input_scale + input_zero_point).astype(input_details['dtype'])
        else:
            inp = inp.astype(input_details['dtype'])

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details['index'])
        inference_times.append((time.perf_counter() - t0) * 1000)

        if output_scale != 0.0:
            out = (raw.astype(np.float32) - output_zero_point) * output_scale
        else:
            out = raw.astype(np.float32)

        probabilities.append(float(out[0, 1]))

    probabilities = np.array(probabilities)
    avg_ms = float(np.mean(inference_times))
    logger.info(f"Inference complete — avg {avg_ms:.3f} ms/sample over {len(mels)} samples")
    return probabilities, avg_ms


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(probabilities: np.ndarray, true_labels: np.ndarray):
    rows = []
    for tau in THRESHOLDS:
        preds = (probabilities >= tau).astype(int)
        tp = int(np.sum((preds == 1) & (true_labels == 1)))
        fp = int(np.sum((preds == 1) & (true_labels == 0)))
        fn = int(np.sum((preds == 0) & (true_labels == 1)))
        tn = int(np.sum((preds == 0) & (true_labels == 0)))
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append(dict(tau=tau, recall=recall, precision=precision,
                         f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn))
    return rows


def find_locked_tau(rows: list):
    """Highest τ where recall ≥ TARGET_RECALL (maximises precision while meeting target)."""
    candidates = [r for r in rows if r['recall'] >= TARGET_RECALL]
    if not candidates:
        # fall back to highest recall row
        return max(rows, key=lambda r: r['recall'])
    return max(candidates, key=lambda r: r['tau'])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sweep_table(rows: list, locked: dict, out_path: Path, auc: float):
    lines = [
        "DrongoNet-Edge — Threshold Sweep",
        f"Target recall : ≥{TARGET_RECALL}",
        f"AUC           : {auc:.4f}",
        "",
        f"{'τ':>6}  {'Recall':>8}  {'Precision':>10}  {'F1':>7}  {'FPR':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}",
        "-" * 72,
    ]
    for r in rows:
        marker = " <-- LOCKED" if r['tau'] == locked['tau'] else ""
        lines.append(
            f"{r['tau']:>6.2f}  {r['recall']:>8.4f}  {r['precision']:>10.4f}  "
            f"{r['f1']:>7.4f}  {r['fpr']:>7.4f}  {r['tp']:>6}  {r['fp']:>6}  "
            f"{r['fn']:>6}  {r['tn']:>6}{marker}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Sweep table → {out_path}")


def write_locked(locked: dict, out_path: Path, auc: float, avg_ms: float, size_kb: float):
    lines = [
        "DrongoNet-Edge — Locked Threshold",
        "=" * 40,
        f"tau       = {locked['tau']:.2f}",
        f"recall    = {locked['recall']:.4f}",
        f"precision = {locked['precision']:.4f}",
        f"f1        = {locked['f1']:.4f}",
        f"fpr       = {locked['fpr']:.4f}",
        f"tp        = {locked['tp']}",
        f"fp        = {locked['fp']}",
        f"fn        = {locked['fn']}",
        f"tn        = {locked['tn']}",
        "",
        f"auc       = {auc:.4f}",
        f"latency   = {avg_ms:.3f} ms/sample (INT8 TFLite)",
        f"size_kb   = {size_kb:.2f}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Locked config → {out_path}")


def plot_pr_curve(probabilities: np.ndarray, true_labels: np.ndarray,
                  locked: dict, out_path: Path):
    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(
        true_labels, probabilities, pos_label=1
    )
    plt.figure(figsize=(7, 5))
    plt.plot(recall_curve, precision_curve, lw=1.5, label='PR curve')
    plt.axvline(x=locked['recall'], color='red', linestyle='--', lw=1,
                label=f"τ={locked['tau']:.2f}  recall={locked['recall']:.4f}  prec={locked['precision']:.4f}")
    plt.axvline(x=TARGET_RECALL, color='orange', linestyle=':', lw=1,
                label=f'Target recall ≥{TARGET_RECALL}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DrongoNet-Edge — Precision-Recall curve (INT8 TFLite)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"PR curve → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Threshold sweep for DrongoNet-Edge')
    parser.add_argument('--results-dir', type=str,
                        default='results/6c_edge_final_fft1024_m80_s42',
                        help='Path to the 6b_edge_final result directory')
    parser.add_argument('--cache-dir', type=str,
                        default='/Volumes/Evo/cache_seabad_m80',
                        help='Path to the mel cache (must contain test/mels.npz)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir   = Path(args.cache_dir)
    tflite_path = results_dir / 'model_int8.tflite'

    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    size_kb = tflite_path.stat().st_size / 1024
    logger.info(f"Model: {tflite_path}  ({size_kb:.2f} KB)")

    mels, true_labels = load_test_mels(cache_dir)
    probabilities, avg_ms = run_tflite_inference(tflite_path, mels)

    auc = roc_auc_score(true_labels, probabilities)
    logger.info(f"AUC: {auc:.4f}")

    rows  = sweep_thresholds(probabilities, true_labels)
    locked = find_locked_tau(rows)

    logger.info(f"Locked τ={locked['tau']:.2f}  recall={locked['recall']:.4f}  "
                f"precision={locked['precision']:.4f}  f1={locked['f1']:.4f}")

    if locked['recall'] < TARGET_RECALL:
        logger.warning(f"No threshold achieves recall ≥ {TARGET_RECALL}. "
                       f"Best recall = {locked['recall']:.4f} at τ={locked['tau']:.2f}")

    write_sweep_table(rows, locked, results_dir / 'threshold_sweep.txt', auc)
    write_locked(locked, results_dir / 'threshold_locked.txt', auc, avg_ms, size_kb)
    plot_pr_curve(probabilities, true_labels, locked, results_dir / 'threshold_sweep_pr.png')

    print()
    print("=" * 50)
    print("DrongoNet-Edge — locked values")
    print("=" * 50)
    print(f"  τ         = {locked['tau']:.2f}")
    print(f"  recall    = {locked['recall']:.4f}")
    print(f"  precision = {locked['precision']:.4f}")
    print(f"  f1        = {locked['f1']:.4f}")
    print(f"  AUC       = {auc:.4f}")
    print(f"  size      = {size_kb:.2f} KB INT8")
    print(f"  latency   = {avg_ms:.3f} ms/sample")
    print("=" * 50)


if __name__ == '__main__':
    main()
