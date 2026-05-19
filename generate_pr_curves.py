#!/usr/bin/env python3
"""
Generate Precision-Recall Curves with Threshold Annotations
For XiaoChirp Gatekeeper and Beast models
Shows thresholds required for 98% and 99% recall
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc
import pickle
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

FIGURES_DIR = Path('figures_publication')
FIGURES_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS = {
    'gatekeeper': {
        'name': 'Gatekeeper (7d fft512 m16)',
        'model_path': 'results_linux/7d_gap_focal_loss_freq_emph_pointwise_fft512_m16_s42_9861/model_int8.tflite',
        'cache_dir': '/Volumes/Evo/cache_mybad_m16_fft512',
        'n_mels': 16,
        'n_fft': 512,
        'color': '#2ca02c',
        'marker': '^',
    },
    'beast': {
        'name': 'Beast (7d fft1024 m80)',
        'model_path': 'results_linux/7d_gap_focal_loss_freq_emph_pointwise_fft1024_m80_s42_9962/model_int8.tflite',
        'cache_dir': '/Volumes/Evo/cache_mybad_m80_fft1024',
        'n_mels': 80,
        'n_fft': 1024,
        'color': '#9467bd',
        'marker': '*',
    }
}

def load_test_data(cache_dir):
    """Load test mel spectrograms and labels from cache"""
    cache_path = Path(cache_dir)
    test_npz = cache_path / 'test' / 'mels.npz'

    if not test_npz.exists():
        raise FileNotFoundError(f"Test data not found: {test_npz}")

    data = np.load(test_npz)
    X_test = data['mels']  # Shape: (N, time, mels)
    y_test = data['labels']  # Shape: (N,)

    print(f"  Loaded {len(X_test)} test samples ({sum(y_test)} positive, {len(y_test)-sum(y_test)} negative)")
    print(f"  Input shape: {X_test.shape}")
    return X_test, y_test

def run_tflite_inference(model_path, X_test):
    """Run TFLite int8 inference"""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    print(f"  Model input shape: {input_shape}, dtype: {input_dtype}")

    # Get quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    print(f"  Input quant: scale={input_scale}, zero_point={input_zero_point}")
    print(f"  Output quant: scale={output_scale}, zero_point={output_zero_point}")

    predictions = []

    for sample in tqdm(X_test, desc="  Running inference"):
        # Input shape from cache is (time, mels) = (184, n_mels)
        # Model expects (batch, time, mels, channel) = (1, 184, n_mels, 1)
        if len(sample.shape) == 2:
            sample = np.expand_dims(sample, axis=-1)  # Add channel: (184, n_mels, 1)
        sample = np.expand_dims(sample, axis=0)  # Add batch: (1, 184, n_mels, 1)

        # Normalize to [0, 1] range if needed, then quantize
        sample = sample.astype(np.float32)

        # Quantize input for int8 model
        if input_dtype == np.int8:
            sample = np.clip(sample / input_scale + input_zero_point, -128, 127).astype(np.int8)
        else:
            sample = sample.astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output
        if output_details[0]['dtype'] == np.int8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(output[0, 0])

    predictions = np.array(predictions)

    # Check if predictions are inverted (high for negatives, low for positives)
    # If mean prediction for class 0 > mean prediction for class 1, invert
    # This handles the case where model outputs P(negative) instead of P(positive)
    return predictions

def find_threshold_for_recall(y_true, y_scores, target_recall):
    """Find the threshold that achieves the target recall"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Find threshold closest to target recall (from high recall side)
    valid_idx = recall[:-1] >= target_recall
    if not any(valid_idx):
        return None, None

    # Get the highest threshold that still meets target recall
    idx = np.where(valid_idx)[0][-1]
    threshold = thresholds[idx]
    actual_precision = precision[idx]
    actual_recall = recall[idx]

    return threshold, actual_precision, actual_recall

def generate_pr_curves():
    """Generate precision-recall curves for both models"""
    print("\n" + "="*70)
    print("GENERATING PRECISION-RECALL CURVES")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    results = {}

    for idx, (model_key, config) in enumerate(MODELS.items()):
        print(f"\nProcessing {config['name']}...")

        # Load test data
        X_test, y_test = load_test_data(config['cache_dir'])

        # Run inference
        y_scores = run_tflite_inference(config['model_path'], X_test)

        # Check if predictions are inverted (model outputs P(negative) instead of P(positive))
        # If mean prediction for negatives > mean prediction for positives, invert
        mean_neg = y_scores[y_test == 0].mean()
        mean_pos = y_scores[y_test == 1].mean()
        print(f"  Mean pred for negatives: {mean_neg:.4f}, for positives: {mean_pos:.4f}")
        if mean_neg > mean_pos:
            print("  Inverting predictions (model outputs P(negative))")
            y_scores = 1 - y_scores

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        # Find thresholds for target recalls
        thresh_98, prec_98, rec_98 = find_threshold_for_recall(y_test, y_scores, 0.98)
        thresh_99, prec_99, rec_99 = find_threshold_for_recall(y_test, y_scores, 0.99)

        results[model_key] = {
            'pr_auc': pr_auc,
            'thresh_98': thresh_98, 'prec_98': prec_98, 'rec_98': rec_98,
            'thresh_99': thresh_99, 'prec_99': prec_99, 'rec_99': rec_99,
        }

        # Plot PR curve
        ax = axes[idx]
        ax.plot(recall, precision, color=config['color'], linewidth=2.5,
                label=f'PR Curve (AUC={pr_auc:.4f})')

        # Mark 98% recall point
        if thresh_98 is not None:
            ax.scatter([rec_98], [prec_98], c='red', s=150, marker='o', zorder=10,
                      edgecolors='black', linewidth=1.5)
            ax.annotate(f'98% Recall\nThresh={thresh_98:.3f}\nPrec={prec_98:.3f}',
                       (rec_98, prec_98), xytext=(-80, -60), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red'))

        # Mark 99% recall point
        if thresh_99 is not None:
            ax.scatter([rec_99], [prec_99], c='blue', s=150, marker='s', zorder=10,
                      edgecolors='black', linewidth=1.5)
            ax.annotate(f'99% Recall\nThresh={thresh_99:.3f}\nPrec={prec_99:.3f}',
                       (rec_99, prec_99), xytext=(-80, 30), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', edgecolor='blue'))

        # Reference lines
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Precision')
        ax.axvline(x=0.98, color='red', linestyle=':', alpha=0.5, label='98% Recall')
        ax.axvline(x=0.99, color='blue', linestyle=':', alpha=0.5, label='99% Recall')

        ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontweight='bold')
        ax.set_title(f'{config["name"]}\nPR-AUC = {pr_auc:.4f}', fontweight='bold')
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(0.5, 1.02)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('XiaoChirp: Precision-Recall Curves with Threshold Selection',
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pr_curves_threshold.png')
    plt.savefig(FIGURES_DIR / 'pr_curves_threshold.pdf')
    print(f"\nSaved: {FIGURES_DIR}/pr_curves_threshold.png/.pdf")

    # Print summary
    print("\n" + "="*70)
    print("THRESHOLD SELECTION SUMMARY")
    print("="*70)

    for model_key, res in results.items():
        config = MODELS[model_key]
        print(f"\n{config['name']}:")
        print(f"  PR-AUC: {res['pr_auc']:.4f}")
        if res['thresh_98']:
            print(f"  For 98% Recall: Threshold={res['thresh_98']:.4f}, Precision={res['prec_98']:.4f}")
        if res['thresh_99']:
            print(f"  For 99% Recall: Threshold={res['thresh_99']:.4f}, Precision={res['prec_99']:.4f}")

    return results

def generate_combined_pr_curve():
    """Generate combined PR curve comparing both models"""
    print("\n" + "="*70)
    print("GENERATING COMBINED PR CURVE")
    print("="*70)

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_key, config in MODELS.items():
        print(f"\nProcessing {config['name']}...")

        # Load test data
        X_test, y_test = load_test_data(config['cache_dir'])

        # Run inference
        y_scores = run_tflite_inference(config['model_path'], X_test)

        # Check if predictions are inverted (model outputs P(negative) instead of P(positive))
        # If mean prediction for negatives > mean prediction for positives, invert
        mean_neg = y_scores[y_test == 0].mean()
        mean_pos = y_scores[y_test == 1].mean()
        print(f"  Mean pred for negatives: {mean_neg:.4f}, for positives: {mean_pos:.4f}")
        if mean_neg > mean_pos:
            print("  Inverting predictions (model outputs P(negative))")
            y_scores = 1 - y_scores

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)

        # Plot
        ax.plot(recall, precision, color=config['color'], linewidth=2.5,
                marker=config['marker'], markevery=50, markersize=8,
                label=f'{config["name"]} (AUC={pr_auc:.4f})')

        # Find and mark key thresholds
        thresh_98, prec_98, rec_98 = find_threshold_for_recall(y_test, y_scores, 0.98)
        thresh_99, prec_99, rec_99 = find_threshold_for_recall(y_test, y_scores, 0.99)

        if thresh_98:
            ax.scatter([rec_98], [prec_98], c=config['color'], s=200, marker='o',
                      zorder=10, edgecolors='black', linewidth=2)
        if thresh_99:
            ax.scatter([rec_99], [prec_99], c=config['color'], s=200, marker='s',
                      zorder=10, edgecolors='black', linewidth=2)

    # Reference lines
    ax.axvline(x=0.98, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='98% Recall')
    ax.axvline(x=0.99, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label='99% Recall')

    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold', fontsize=13)
    ax.set_title('XiaoChirp: Precision-Recall Curve Comparison\nGatekeeper vs Beast Models',
                fontweight='bold', fontsize=14)
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.7, 1.01)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation for markers
    ax.text(0.98, 0.72, 'Circle: 98% Recall\nSquare: 99% Recall',
           fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pr_curves_combined.png')
    plt.savefig(FIGURES_DIR / 'pr_curves_combined.pdf')
    print(f"\nSaved: {FIGURES_DIR}/pr_curves_combined.png/.pdf")

def main():
    results = generate_pr_curves()
    generate_combined_pr_curve()

    print("\n" + "="*70)
    print("PR CURVES GENERATED SUCCESSFULLY")
    print("="*70)

if __name__ == '__main__':
    main()
