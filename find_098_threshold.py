#!/usr/bin/env python3
"""
Find optimal threshold for 0.98 recall on gatekeeper models.
Analyzes both 2D (mel-spectrogram) and 1D (raw waveform) models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report
import argparse

def load_tflite_model(model_path):
    """Load TFLite model and create interpreter"""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, input_data):
    """Run inference on TFLite model, return probabilities"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape and type
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    all_probs = []

    for i in range(len(input_data)):
        # Prepare single sample
        sample = input_data[i:i+1].astype(np.float32)

        # Handle quantized input if needed
        if input_dtype == np.int8:
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            sample = (sample / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle quantized output
        if output_details[0]['dtype'] == np.int8:
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        # Get probability for positive class (index 1)
        if output.shape[-1] == 2:
            prob = output[0, 1]
        else:
            prob = output[0, 0]

        all_probs.append(prob)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(input_data)} samples...")

    return np.array(all_probs)

def find_threshold_for_target_recall(y_true, y_probs, target_recall=0.98):
    """Find threshold that achieves target recall with best precision"""
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

def analyze_model(model_name, model_path, test_data, test_labels, target_recall=0.98):
    """Analyze a single model and find optimal threshold"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    # Load model
    print(f"Loading TFLite model from {model_path}...")
    interpreter = load_tflite_model(model_path)

    # Get model info
    input_details = interpreter.get_input_details()
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")

    # Run inference
    print(f"Running inference on {len(test_data)} test samples...")
    probs = run_tflite_inference(interpreter, test_data)

    # Find threshold for target recall
    print(f"\nFinding threshold for {target_recall:.0%} recall...")
    threshold, precision, recall, achieved = find_threshold_for_target_recall(
        test_labels, probs, target_recall
    )

    # Calculate metrics at this threshold
    predictions = (probs >= threshold).astype(int)

    # Calculate false positive rate
    negatives = test_labels == 0
    false_positives = predictions[negatives].sum()
    fpr = false_positives / negatives.sum()

    # Print results
    print(f"\n--- Results for {target_recall:.0%} Recall Target ---")
    print(f"Achieved target: {'YES' if achieved else 'NO'}")
    print(f"Optimal threshold: {threshold:.6f}")
    print(f"Recall at threshold: {recall:.4f} ({recall:.2%})")
    print(f"Precision at threshold: {precision:.4f} ({precision:.2%})")
    print(f"False Positive Rate: {fpr:.4f} ({fpr:.2%})")

    # Show what happens at default 0.5 threshold
    default_preds = (probs >= 0.5).astype(int)
    default_recall = (default_preds[test_labels == 1].sum()) / (test_labels == 1).sum()
    default_precision = (default_preds[test_labels == 1].sum()) / max(default_preds.sum(), 1)

    print(f"\n--- Comparison with Default (0.5) Threshold ---")
    print(f"Default recall: {default_recall:.4f}")
    print(f"Default precision: {default_precision:.4f}")

    return {
        'model_name': model_name,
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'fpr': fpr,
        'achieved': achieved,
        'default_recall': default_recall,
        'default_precision': default_precision
    }

def main():
    parser = argparse.ArgumentParser(description='Find threshold for target recall')
    parser.add_argument('--target-recall', type=float, default=0.98,
                        help='Target recall (default: 0.98)')
    args = parser.parse_args()

    target_recall = args.target_recall

    print(f"="*60)
    print(f"GATEKEEPER THRESHOLD ANALYSIS")
    print(f"Target Recall: {target_recall:.0%}")
    print(f"="*60)

    results = []

    # ===== Model 1: 7a_gap_focal_loss_fft1024_m16 (2D) =====
    model_2d_path = Path("results/7a_gap_focal_loss_fft1024_m16_s42/model_int8.tflite")
    cache_2d_path = Path("/Volumes/Evo/cache_mybad_m16_fft1024/test/mels.npz")

    if model_2d_path.exists() and cache_2d_path.exists():
        print("\n\nLoading 2D test data (mel spectrograms)...")
        data = np.load(cache_2d_path)
        test_mels = data['mels']
        test_labels_2d = data['labels']

        # Add channel dimension if needed
        if len(test_mels.shape) == 3:
            test_mels = test_mels[..., np.newaxis]

        print(f"Test data shape: {test_mels.shape}")
        print(f"Labels shape: {test_labels_2d.shape}")
        print(f"Positive samples: {test_labels_2d.sum()}, Negative: {(1-test_labels_2d).sum()}")

        result = analyze_model(
            "7a_gap_focal_loss_fft1024_m16",
            model_2d_path,
            test_mels,
            test_labels_2d,
            target_recall
        )
        results.append(result)
    else:
        print(f"\nSkipping 2D model - files not found:")
        print(f"  Model: {model_2d_path.exists()}")
        print(f"  Cache: {cache_2d_path.exists()}")

    # ===== Model 2: 5c_gatekeeper_deeper_RE_PW (1D) =====
    model_1d_path = Path("results/5c_gatekeeper_deeper_RE_PW_r42_linux/model_int8.tflite")
    cache_1d_path = Path("/Volumes/Evo/cache_mybad_waveforms/testing/waveforms.npz")

    if model_1d_path.exists() and cache_1d_path.exists():
        print("\n\nLoading 1D test data (raw waveforms)...")
        data = np.load(cache_1d_path)
        test_waveforms = data['waveforms']
        test_labels_1d = data['labels']

        # Add channel dimension if needed
        if len(test_waveforms.shape) == 2:
            test_waveforms = test_waveforms[..., np.newaxis]

        print(f"Test data shape: {test_waveforms.shape}")
        print(f"Labels shape: {test_labels_1d.shape}")
        print(f"Positive samples: {test_labels_1d.sum()}, Negative: {(1-test_labels_1d).sum()}")

        result = analyze_model(
            "5c_gatekeeper_deeper_RE_PW (1D)",
            model_1d_path,
            test_waveforms,
            test_labels_1d,
            target_recall
        )
        results.append(result)
    else:
        print(f"\nSkipping 1D model - files not found:")
        print(f"  Model: {model_1d_path.exists()}")
        print(f"  Cache: {cache_1d_path.exists()}")

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY: OPTIMAL THRESHOLDS FOR 0.98 RECALL")
    print("="*70)
    print(f"{'Model':<40} {'Threshold':<12} {'Recall':<10} {'Precision':<10} {'FPR':<10}")
    print("-"*70)

    for r in results:
        status = "OK" if r['achieved'] else "FAIL"
        print(f"{r['model_name']:<40} {r['threshold']:.6f}     {r['recall']:.4f}     {r['precision']:.4f}     {r['fpr']:.4f}")

    print("\n" + "="*70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*70)

    for r in results:
        print(f"\n{r['model_name']}:")
        if r['achieved']:
            print(f"  Use threshold: {r['threshold']:.6f}")
            print(f"  Expected recall: {r['recall']:.2%}")
            print(f"  Expected precision: {r['precision']:.2%}")
            print(f"  False positive rate: {r['fpr']:.2%}")
            print(f"  -> For every 100 non-bird sounds, {r['fpr']*100:.1f} will be false alarms")
        else:
            print(f"  WARNING: Cannot achieve {target_recall:.0%} recall")
            print(f"  Best achievable recall: {r['recall']:.2%}")
            print(f"  Consider using a different model or retraining")

if __name__ == "__main__":
    main()
