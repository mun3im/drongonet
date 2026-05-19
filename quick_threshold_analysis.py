#!/usr/bin/env python3
"""Quick threshold analysis for gatekeeper candidates."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

def find_threshold_for_recall(y_true, y_probs, target_recall=0.98):
    """Find threshold that achieves target recall with best precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        return None, None, recalls.max(), False

    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], True

def run_inference(model_path, test_mels):
    """Run TFLite inference and return probabilities."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zp = input_details['quantization']
    output_scale, output_zp = output_details['quantization']

    probs = []
    for i in tqdm(range(len(test_mels)), desc="Inference", leave=False):
        sample = test_mels[i:i+1].astype(np.float32)

        if input_scale != 0:
            sample = np.round(sample / input_scale + input_zp).astype(input_details['dtype'])

        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])

        if output_scale != 0:
            output = (output.astype(np.float32) - output_zp) * output_scale

        probs.append(output[0, 1])

    return np.array(probs)

def main():
    # Load test data
    cache_path = Path("/tmp/cache_mybad_m16_fft512/test/mels.npz")
    print(f"Loading test data from {cache_path}...")
    data = np.load(cache_path)
    test_mels = data['mels'][..., np.newaxis]
    test_labels = data['labels']
    print(f"  Samples: {len(test_mels)}, Positive: {test_labels.sum()}")

    # Models to analyze
    models = [
        ("7e_tuned_m16", "results/7e_strided_focal_tuned_fft512_m16_s42_9747/model_int8.tflite"),
        ("7f_no1x1_m16", "results/7f_strided_focal_no1x1_fft512_m16_s42_9635/model_int8.tflite"),
        ("7a_focal_m16", "results/7a_gap_focal_loss_fft512_m16_s42_9746/model_int8.tflite"),
    ]

    results = []

    for name, model_path in models:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"\n{name}: Model not found")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")

        probs = run_inference(model_path, test_mels)

        # Find threshold for 98% recall
        threshold, precision, recall, achieved = find_threshold_for_recall(
            test_labels, probs, target_recall=0.98
        )

        # Calculate FPR at this threshold
        if threshold is not None:
            preds = (probs >= threshold).astype(int)
            negatives = test_labels == 0
            fpr = preds[negatives].sum() / negatives.sum()
        else:
            fpr = None

        # Default threshold metrics
        default_preds = (probs >= 0.5).astype(int)
        default_recall = default_preds[test_labels == 1].sum() / (test_labels == 1).sum()
        default_precision = default_preds[test_labels == 1].sum() / max(default_preds.sum(), 1)

        print(f"\nDefault (0.5) threshold:")
        print(f"  Recall: {default_recall:.2%}")
        print(f"  Precision: {default_precision:.2%}")

        if achieved:
            print(f"\nFor 98% recall:")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Recall: {recall:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  FPR: {fpr:.2%}")
        else:
            print(f"\nCannot achieve 98% recall (max: {recall:.2%})")

        results.append({
            'name': name,
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'fpr': fpr,
            'achieved': achieved,
            'default_recall': default_recall
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Threshold for 98% Recall")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Threshold':<12} {'Recall':<10} {'Precision':<10} {'FPR':<10}")
    print("-"*70)

    for r in results:
        if r['achieved']:
            print(f"{r['name']:<20} {r['threshold']:.4f}       {r['recall']:.2%}     {r['precision']:.2%}      {r['fpr']:.2%}")
        else:
            print(f"{r['name']:<20} {'N/A':<12} {r['recall']:.2%}     {'N/A':<10} {'N/A':<10}")

if __name__ == "__main__":
    main()
