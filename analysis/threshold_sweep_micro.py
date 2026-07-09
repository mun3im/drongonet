#!/usr/bin/env python3
"""
threshold_sweep_micro.py: Threshold sweep for DrongoNet-Micro

Loads the saved INT8 TFLite model from each of the three validation seeds
(42, 100, 786) of 6b_micro_improved_fft1024_m16, runs inference on the
test set, sweeps thresholds, and locks the lowest τ that achieves ≥0.98
recall on the mean across seeds.

Output (written into each seed's results dir):
  threshold_sweep.txt     — per-seed full table
  threshold_sweep_pr.png  — per-seed PR curve

Output (written into --out-dir, default: results/micro_threshold_sweep):
  threshold_sweep_combined.txt  — all-seed table + mean ± std per τ
  threshold_locked.txt          — locked τ + mean ± std metrics
  threshold_sweep_pr_combined.png

Usage:
  python threshold_sweep_micro.py \
      --results-base results \
      --cache-dir    /Volumes/Evo/cache_seabad_m16 \
      --out-dir      results/micro_threshold_sweep
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_RECALL = 0.98
SEEDS = [42, 100, 786]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
RESULTS_DIRNAME = "6b_micro_final_fft1024_m16_s{seed}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_mels(cache_dir: Path):
    cache_file = cache_dir / 'test' / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Test cache not found: {cache_file}")
    data = np.load(cache_file)
    mels   = data['mels'].astype(np.float32)
    labels = data['labels'].astype(np.int32)
    logger.info(f"Loaded {len(mels)} test samples from {cache_file}")
    return mels, labels


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def run_tflite_inference(tflite_path: Path, mels: np.ndarray):
    """Returns (probabilities, avg_inference_ms)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale,  input_zero_point  = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    logger.info(f"Input  dtype={input_details['dtype']}  scale={input_scale}  zp={input_zero_point}")
    logger.info(f"Output dtype={output_details['dtype']} scale={output_scale} zp={output_zero_point}")

    mels_4d = mels[..., np.newaxis]
    probabilities   = []
    inference_times = []

    for sample in mels_4d:
        inp = sample[np.newaxis]

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
    logger.info(f"Inference done — avg {avg_ms:.3f} ms/sample")
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


def find_locked_tau(mean_rows: list):
    """Highest τ where mean recall ≥ TARGET_RECALL (maximises precision while meeting target)."""
    candidates = [r for r in mean_rows if r['recall'] >= TARGET_RECALL]
    if not candidates:
        return max(mean_rows, key=lambda r: r['recall'])
    return max(candidates, key=lambda r: r['tau'])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_per_seed_table(rows: list, out_path: Path, auc: float, seed: int):
    lines = [
        f"DrongoNet-Micro — Threshold Sweep  (seed={seed})",
        f"Target recall : ≥{TARGET_RECALL}",
        f"AUC           : {auc:.4f}",
        "",
        f"{'τ':>6}  {'Recall':>8}  {'Precision':>10}  {'F1':>7}  {'FPR':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}",
        "-" * 72,
    ]
    for r in rows:
        lines.append(
            f"{r['tau']:>6.2f}  {r['recall']:>8.4f}  {r['precision']:>10.4f}  "
            f"{r['f1']:>7.4f}  {r['fpr']:>7.4f}  {r['tp']:>6}  {r['fp']:>6}  "
            f"{r['fn']:>6}  {r['tn']:>6}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Per-seed table → {out_path}")


def write_combined_table(all_rows: dict, mean_rows: list, std_rows: list,
                         locked: dict, out_path: Path, aucs: list):
    lines = [
        "DrongoNet-Micro — Combined Threshold Sweep (seeds 42 / 100 / 786)",
        f"Target recall : ≥{TARGET_RECALL}",
        f"AUC           : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        "",
    ]
    # per-seed columns
    header = f"{'τ':>6}"
    for s in SEEDS:
        header += f"  {'rec_s'+str(s):>10}  {'prec_s'+str(s):>11}"
    header += f"  {'mean_rec':>10}  {'std_rec':>9}  {'mean_prec':>10}  {'std_prec':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, tau in enumerate(THRESHOLDS):
        row = f"{tau:>6.2f}"
        for s in SEEDS:
            r = all_rows[s][i]
            row += f"  {r['recall']:>10.4f}  {r['precision']:>11.4f}"
        mr = mean_rows[i]
        sr = std_rows[i]
        marker = " <-- LOCKED" if tau == locked['tau'] else ""
        row += (f"  {mr['recall']:>10.4f}  {sr['recall']:>9.4f}"
                f"  {mr['precision']:>10.4f}  {sr['precision']:>9.4f}{marker}")
        lines.append(row)

    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Combined table → {out_path}")


def write_locked(locked: dict, locked_std: dict, out_path: Path,
                 aucs: list, avg_ms_list: list, size_kb: float):
    lines = [
        "DrongoNet-Micro — Locked Threshold",
        "=" * 40,
        f"tau           = {locked['tau']:.2f}",
        f"recall        = {locked['recall']:.4f} ± {locked_std['recall']:.4f}",
        f"precision     = {locked['precision']:.4f} ± {locked_std['precision']:.4f}",
        f"f1            = {locked['f1']:.4f} ± {locked_std['f1']:.4f}",
        f"fpr           = {locked['fpr']:.4f} ± {locked_std['fpr']:.4f}",
        "",
        f"auc           = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        f"latency       = {np.mean(avg_ms_list):.3f} ms/sample (INT8 TFLite, mean over seeds)",
        f"size_kb       = {size_kb:.2f}",
        f"seeds         = {SEEDS}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Locked config → {out_path}")


def plot_per_seed_pr(probabilities: np.ndarray, true_labels: np.ndarray,
                     locked_tau: float, seed: int, out_path: Path):
    prec_curve, rec_curve, _ = precision_recall_curve(true_labels, probabilities, pos_label=1)
    # find point on curve closest to locked tau
    preds = (probabilities >= locked_tau).astype(int)
    tp = int(np.sum((preds == 1) & (true_labels == 1)))
    fp = int(np.sum((preds == 1) & (true_labels == 0)))
    fn = int(np.sum((preds == 0) & (true_labels == 1)))
    rec_pt  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    plt.figure(figsize=(7, 5))
    plt.plot(rec_curve, prec_curve, lw=1.5, label='PR curve')
    plt.scatter([rec_pt], [prec_pt], color='red', zorder=5,
                label=f"τ={locked_tau:.2f}  recall={rec_pt:.4f}  prec={prec_pt:.4f}")
    plt.axvline(x=TARGET_RECALL, color='orange', linestyle=':', lw=1,
                label=f'Target recall ≥{TARGET_RECALL}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'DrongoNet-Micro — PR curve (INT8, seed={seed})')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Per-seed PR curve → {out_path}")


def plot_combined_pr(all_probs: dict, true_labels: np.ndarray,
                     locked: dict, out_path: Path):
    plt.figure(figsize=(7, 5))
    colors = ['steelblue', 'seagreen', 'mediumpurple']
    for (seed, probs), color in zip(all_probs.items(), colors):
        prec_c, rec_c, _ = precision_recall_curve(true_labels, probs, pos_label=1)
        plt.plot(rec_c, prec_c, lw=1, color=color, alpha=0.7, label=f'seed={seed}')

    plt.axvline(x=locked['recall'], color='red', linestyle='--', lw=1.2,
                label=f"τ={locked['tau']:.2f}  mean recall={locked['recall']:.4f}")
    plt.axvline(x=TARGET_RECALL, color='orange', linestyle=':', lw=1,
                label=f'Target recall ≥{TARGET_RECALL}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DrongoNet-Micro — PR curves across seeds (INT8 TFLite)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Combined PR curve → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Threshold sweep for DrongoNet-Micro (3 seeds)')
    parser.add_argument('--results-base', type=str, default='results',
                        help='Parent directory containing 6b_micro_improved_fft1024_m16_s* dirs')
    parser.add_argument('--cache-dir', type=str, default='/Volumes/Evo/cache_seabad_m16',
                        help='Mel cache dir (must contain test/mels.npz)')
    parser.add_argument('--out-dir', type=str, default='results/micro_threshold_sweep',
                        help='Directory for combined outputs')
    args = parser.parse_args()

    results_base = Path(args.results_base)
    cache_dir    = Path(args.cache_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mels, true_labels = load_test_mels(cache_dir)

    all_rows     = {}   # seed -> list of row dicts
    all_probs    = {}   # seed -> probabilities array
    aucs         = []
    avg_ms_list  = []
    size_kb      = None

    for seed in SEEDS:
        dirname    = RESULTS_DIRNAME.format(seed=seed)
        result_dir = results_base / dirname
        tflite_path = result_dir / 'model.tflite'

        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

        if size_kb is None:
            size_kb = tflite_path.stat().st_size / 1024

        logger.info(f"--- Seed {seed} ---")
        probs, avg_ms = run_tflite_inference(tflite_path, mels)
        auc = roc_auc_score(true_labels, probs)
        logger.info(f"AUC: {auc:.4f}")

        rows = sweep_thresholds(probs, true_labels)
        all_rows[seed]  = rows
        all_probs[seed] = probs
        aucs.append(auc)
        avg_ms_list.append(avg_ms)

        write_per_seed_table(rows, result_dir / 'threshold_sweep.txt', auc, seed)

    # compute mean and std rows across seeds
    mean_rows = []
    std_rows  = []
    for i in range(len(THRESHOLDS)):
        mean_rows.append({
            k: float(np.mean([all_rows[s][i][k] for s in SEEDS]))
            for k in ('tau', 'recall', 'precision', 'f1', 'fpr')
        })
        std_rows.append({
            k: float(np.std([all_rows[s][i][k] for s in SEEDS]))
            for k in ('tau', 'recall', 'precision', 'f1', 'fpr')
        })

    locked     = find_locked_tau(mean_rows)
    locked_idx = list(THRESHOLDS).index(locked['tau']) if locked['tau'] in THRESHOLDS else \
                 int(np.argmin(np.abs(THRESHOLDS - locked['tau'])))
    locked_std = std_rows[locked_idx]

    if locked['recall'] < TARGET_RECALL:
        logger.warning(f"No threshold achieves mean recall ≥ {TARGET_RECALL}. "
                       f"Best mean recall = {locked['recall']:.4f} at τ={locked['tau']:.2f}")

    write_combined_table(all_rows, mean_rows, std_rows, locked,
                         out_dir / 'threshold_sweep_combined.txt', aucs)
    write_locked(locked, locked_std, out_dir / 'threshold_locked.txt',
                 aucs, avg_ms_list, size_kb)

    # per-seed PR plots with locked point marked
    for seed in SEEDS:
        result_dir = results_base / RESULTS_DIRNAME.format(seed=seed)
        plot_per_seed_pr(all_probs[seed], true_labels, locked['tau'],
                         seed, result_dir / 'threshold_sweep_pr.png')

    plot_combined_pr(all_probs, true_labels, locked, out_dir / 'threshold_sweep_pr_combined.png')

    print()
    print("=" * 55)
    print("DrongoNet-Micro — locked values (mean ± std, 3 seeds)")
    print("=" * 55)
    print(f"  τ         = {locked['tau']:.2f}")
    print(f"  recall    = {locked['recall']:.4f} ± {locked_std['recall']:.4f}")
    print(f"  precision = {locked['precision']:.4f} ± {locked_std['precision']:.4f}")
    print(f"  f1        = {locked['f1']:.4f} ± {locked_std['f1']:.4f}")
    print(f"  AUC       = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  size      = {size_kb:.2f} KB INT8")
    print(f"  latency   = {np.mean(avg_ms_list):.3f} ms/sample")
    print("=" * 55)


if __name__ == '__main__':
    main()
