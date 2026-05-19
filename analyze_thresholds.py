#!/usr/bin/env python3
"""
Analyze thresholds for saved TFLite predictions.
Finds optimal threshold for target recall (default 0.98).
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, classification_report


def find_threshold_for_target_recall(y_true, y_probs, target_recall=0.98):
    """Find threshold that achieves target recall with best precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Find all thresholds where recall >= target
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        max_recall = recalls.max()
        print(f"  WARNING: Cannot achieve {target_recall:.2%} recall. Max recall: {max_recall:.4f}")
        # Return threshold for max recall
        best_idx = np.argmax(recalls[:-1])
        return thresholds[best_idx], precisions[best_idx], recalls[best_idx], False

    # Among valid thresholds, pick the one with highest precision
    best_valid_idx = valid_idx[np.argmax(precisions[valid_idx])]

    threshold = thresholds[best_valid_idx]
    precision = precisions[best_valid_idx]
    recall = recalls[best_valid_idx]

    return threshold, precision, recall, True


def analyze_predictions(predictions_path: Path, target_recall: float = 0.98):
    """Analyze a single predictions file."""
    if not predictions_path.exists():
        print(f"  File not found: {predictions_path}")
        return None

    data = np.load(predictions_path)
    probabilities = data['probabilities']
    true_labels = data['true_labels']

    # Find threshold for target recall
    threshold, precision, recall, achieved = find_threshold_for_target_recall(
        true_labels, probabilities, target_recall
    )

    # Calculate metrics at this threshold
    predictions_at_threshold = (probabilities >= threshold).astype(int)

    # Calculate false positive rate
    negatives = true_labels == 0
    false_positives = predictions_at_threshold[negatives].sum()
    fpr = false_positives / negatives.sum()

    # Calculate metrics at default 0.5 threshold
    predictions_default = (probabilities >= 0.5).astype(int)
    default_recall = (predictions_default[true_labels == 1].sum()) / (true_labels == 1).sum()
    default_precision = (predictions_default[true_labels == 1].sum()) / max(predictions_default.sum(), 1)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'achieved': achieved,
        'default_recall': default_recall,
        'default_precision': default_precision,
        'n_samples': len(true_labels),
        'n_positive': (true_labels == 1).sum(),
        'n_negative': (true_labels == 0).sum()
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze thresholds for TFLite predictions')
    parser.add_argument('--target-recall', type=float, default=0.98,
                        help='Target recall (default: 0.98)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Results directory (default: results)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['1a_baseline2d', '4a_baseline_gap', '7a_gap_focal_loss'],
                        help='Model prefixes to analyze')
    parser.add_argument('--config', type=str, default='fft512_m16_s42',
                        help='Configuration suffix (default: fft512_m16_s42)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    target_recall = args.target_recall

    print("=" * 70)
    print(f"THRESHOLD ANALYSIS FOR {target_recall:.0%} RECALL")
    print("=" * 70)
    print()

    results = []

    for model_prefix in args.models:
        model_dir = results_dir / f"{model_prefix}_{args.config}"
        predictions_path = model_dir / 'tflite_predictions.npz'

        print(f"Analyzing: {model_prefix}_{args.config}")
        result = analyze_predictions(predictions_path, target_recall)

        if result:
            result['model'] = model_prefix
            results.append(result)

            status = "OK" if result['achieved'] else "FAILED"
            print(f"  Status: {status}")
            print(f"  Optimal threshold: {result['threshold']:.6f}")
            print(f"  Recall at threshold: {result['recall']:.4f} ({result['recall']:.2%})")
            print(f"  Precision at threshold: {result['precision']:.4f} ({result['precision']:.2%})")
            print(f"  False Positive Rate: {result['fpr']:.4f} ({result['fpr']:.2%})")
            print(f"  Default (0.5) recall: {result['default_recall']:.4f}")
            print(f"  Default (0.5) precision: {result['default_precision']:.4f}")
        print()

    # Summary table
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Model':<25} {'Threshold':<12} {'Recall':<10} {'Precision':<10} {'FPR':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['model']:<25} {r['threshold']:.6f}     {r['recall']:.4f}     {r['precision']:.4f}     {r['fpr']:.4f}")

        print()
        print("=" * 70)
        print("DEPLOYMENT RECOMMENDATION")
        print("=" * 70)

        # Find best model (achieved target with highest precision)
        achieved = [r for r in results if r['achieved']]
        if achieved:
            best = max(achieved, key=lambda x: x['precision'])
            print(f"\nBest model for {target_recall:.0%} recall: {best['model']}")
            print(f"  Use threshold: {best['threshold']:.6f}")
            print(f"  Expected recall: {best['recall']:.2%}")
            print(f"  Expected precision: {best['precision']:.2%}")
            print(f"  False positive rate: {best['fpr']:.2%}")
        else:
            print(f"\nNo model achieved {target_recall:.0%} recall.")
            best = max(results, key=lambda x: x['recall'])
            print(f"Best available: {best['model']} with {best['recall']:.2%} recall")


if __name__ == '__main__':
    main()
