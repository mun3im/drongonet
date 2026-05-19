#!/usr/bin/env python3
"""
Regenerate Figures 3 & 4 for Gatekeeper model only
Uses TFLite model to generate predictions, then creates publication figures
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from scipy import stats
import tensorflow as tf
from tqdm import tqdm

# Configure
GATEKEEPER_MODEL = "results_linux/7d_gap_focal_loss_freq_emph_pointwise_fft512_m80_s42_9949"
OUTPUT_DIR = "/Users/mun3im/Dropbox/Paper Bioacoustics"
CACHE_DIR = "/Volumes/Evo/cache_mybad2_m80"

def load_test_data(cache_dir: str):
    """Load cached test mel spectrograms"""
    cache_file = Path(cache_dir) / 'test' / 'mels.npz'

    if not cache_file.exists():
        print(f"Error: Cache file not found: {cache_file}")
        sys.exit(1)

    print(f"Loading test data from {cache_file}")
    data = np.load(cache_file)
    mel_specs = data['mels']
    labels = data['labels']

    # Add channel dimension
    mel_specs = mel_specs[..., np.newaxis]

    print(f"Loaded {len(mel_specs)} test samples")
    print(f"Shape: {mel_specs.shape}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Negative samples: {np.sum(labels == 0)}")

    return mel_specs, labels

def evaluate_tflite_model(tflite_path: Path, test_data, test_labels):
    """Evaluate TFLite model and return predictions"""
    print(f"\nEvaluating TFLite model: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")

    predictions = []
    probabilities = []

    for i in tqdm(range(len(test_data)), desc="TFLite inference"):
        input_data = test_data[i:i+1]

        # Quantize input if needed
        if input_scale != 0.0:
            input_quantized = np.round(input_data / input_scale + input_zero_point).astype(input_details['dtype'])
        else:
            input_quantized = input_data.astype(input_details['dtype'])

        # Run inference
        interpreter.set_tensor(input_details['index'], input_quantized)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])

        # Dequantize output if needed
        if output_scale != 0.0:
            output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output_data.astype(np.float32)

        # Extract probabilities
        prob_positive = float(output_float[0, 1])
        pred = int(np.argmax(output_float, axis=1)[0])

        predictions.append(pred)
        probabilities.append(prob_positive)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # Compute metrics
    auc = roc_auc_score(test_labels, probabilities)
    acc = np.mean(predictions == test_labels)

    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC: {auc:.4f}")

    return predictions, probabilities

def find_optimal_threshold(true_labels, probabilities, target_recall=0.98):
    """Find threshold that achieves target recall"""
    thresholds = np.linspace(0, 1, 1000)

    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        tp = np.sum((preds == 1) & (true_labels == 1))
        fn = np.sum((preds == 0) & (true_labels == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if recall >= target_recall:
            return thresh

    return 0.5

def generate_figure3(true_labels, probabilities, output_path):
    """Generate Figure 3: Probability distributions"""
    print("\nGenerating Figure 3: Probability distributions...")

    # Separate by class
    probs_negative = probabilities[true_labels == 0]
    probs_positive = probabilities[true_labels == 1]

    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(true_labels, probabilities, target_recall=0.98)

    # Compute statistics
    stats_neg = {
        'mean': np.mean(probs_negative),
        'median': np.median(probs_negative),
        'std': np.std(probs_negative),
        'q25': np.percentile(probs_negative, 25),
        'q75': np.percentile(probs_negative, 75)
    }

    stats_pos = {
        'mean': np.mean(probs_positive),
        'median': np.median(probs_positive),
        'std': np.std(probs_positive),
        'q25': np.percentile(probs_positive, 25),
        'q75': np.percentile(probs_positive, 75)
    }

    separation = stats_pos['mean'] - stats_neg['mean']

    print(f"  True Negatives: mean={stats_neg['mean']:.3f}, median={stats_neg['median']:.3f}")
    print(f"  True Positives: mean={stats_pos['mean']:.3f}, median={stats_pos['median']:.3f}")
    print(f"  Separation: {separation:.3f}")
    print(f"  Optimal threshold (98% recall): {optimal_thresh:.3f}")

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Histograms
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 50)
    ax1.hist(probs_negative, bins=bins, alpha=0.6, color='purple', label='True Negative', density=True)
    ax1.hist(probs_positive, bins=bins, alpha=0.6, color='steelblue', label='True Positive', density=True)
    ax1.axvline(optimal_thresh, color='orange', linestyle='--', linewidth=2, label=f'τ* = {optimal_thresh:.3f}')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('(A) Probability Distribution by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Box and violin plots
    ax2 = fig.add_subplot(gs[0, 1])
    data_to_plot = [probs_negative, probs_positive]
    positions = [1, 2]
    bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.4, patch_artist=True,
                      medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('purple')
    bp['boxes'][1].set_facecolor('steelblue')

    violin_parts = ax2.violinplot(data_to_plot, positions=positions, widths=0.6, showmeans=True)
    for pc in violin_parts['bodies']:
        pc.set_alpha(0.3)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['True Negative', 'True Positive'])
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('(B) Distribution Shape Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: Cumulative distributions
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(probs_negative, bins=100, cumulative=True, density=True, histtype='step',
             color='purple', linewidth=2, label='True Negative')
    ax3.hist(probs_positive, bins=100, cumulative=True, density=True, histtype='step',
             color='steelblue', linewidth=2, label='True Positive')
    ax3.axvline(optimal_thresh, color='orange', linestyle='--', linewidth=2, label=f'τ* = {optimal_thresh:.3f}')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('(C) Cumulative Distribution Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    table_data = [
        ['Metric', 'True Negative', 'True Positive'],
        ['Mean', f"{stats_neg['mean']:.3f}", f"{stats_pos['mean']:.3f}"],
        ['Median', f"{stats_neg['median']:.3f}", f"{stats_pos['median']:.3f}"],
        ['Std Dev', f"{stats_neg['std']:.3f}", f"{stats_pos['std']:.3f}"],
        ['Q25', f"{stats_neg['q25']:.3f}", f"{stats_pos['q25']:.3f}"],
        ['Q75', f"{stats_neg['q75']:.3f}", f"{stats_pos['q75']:.3f}"],
        ['', '', ''],
        ['Separation', f"{separation:.3f}", ''],
        ['Optimal τ*', f"{optimal_thresh:.3f}", f'(98% recall)']
    ]

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('(D) Summary Statistics', pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")

    return optimal_thresh, stats_neg, stats_pos, separation

def generate_figure4(true_labels, probabilities, optimal_thresh, output_path):
    """Generate Figure 4: Confusion matrices at multiple thresholds"""
    print("\nGenerating Figure 4: Confusion matrices...")

    # Test thresholds
    thresholds = [0.1, 0.2, optimal_thresh, 0.4, 0.5, 0.7]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, thresh in enumerate(thresholds):
        predictions = (probabilities >= thresh).astype(int)
        cm = confusion_matrix(true_labels, predictions)

        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ax = axes[idx]

        # Choose colormap
        if abs(thresh - optimal_thresh) < 0.001:
            cmap = 'Oranges'
            border_color = 'orange'
            border_width = 4
            title_suffix = ' (Optimal)'
        else:
            cmap = 'Blues'
            border_color = 'black'
            border_width = 1
            title_suffix = ''

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])

        ax.set_title(f'τ = {thresh:.3f}{title_suffix}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # Add border for optimal threshold
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

        # Add metrics as text
        metrics_text = f'Recall: {recall:.1%}\nPrecision: {precision:.1%}\nSpec: {specificity:.1%}\nF₁: {f1:.3f}'
        ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
                verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        print(f"  τ={thresh:.3f}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")

def main():
    print("="*60)
    print("Regenerating Figures 3 & 4 for Gatekeeper Model")
    print("="*60)

    # Load test data
    test_data, test_labels = load_test_data(CACHE_DIR)

    # Load and evaluate TFLite model
    tflite_path = Path(GATEKEEPER_MODEL) / "model_int8.tflite"

    if not tflite_path.exists():
        print(f"Error: TFLite model not found: {tflite_path}")
        sys.exit(1)

    predictions, probabilities = evaluate_tflite_model(tflite_path, test_data, test_labels)

    # Save predictions
    predictions_file = Path(GATEKEEPER_MODEL) / "tflite_predictions.npz"
    np.savez_compressed(
        predictions_file,
        probabilities=probabilities,
        predictions=predictions,
        true_labels=test_labels
    )
    print(f"\nSaved predictions to: {predictions_file}")

    # Generate figures
    output_fig3 = Path(OUTPUT_DIR) / "fig3_probability_distributions.pdf"
    output_fig4 = Path(OUTPUT_DIR) / "fig4_confusion_matrices.pdf"

    optimal_thresh, stats_neg, stats_pos, separation = generate_figure3(
        test_labels, probabilities, output_fig3
    )

    generate_figure4(test_labels, probabilities, optimal_thresh, output_fig4)

    print("\n" + "="*60)
    print("✓ Figures regenerated successfully!")
    print("="*60)
    print(f"\nFigure 3: {output_fig3}")
    print(f"Figure 4: {output_fig4}")
    print(f"\nKey Statistics:")
    print(f"  True Negatives: mean={stats_neg['mean']:.3f}, median={stats_neg['median']:.3f}")
    print(f"  True Positives: mean={stats_pos['mean']:.3f}, median={stats_pos['median']:.3f}")
    print(f"  Separation: {separation:.3f}")
    print(f"  Optimal threshold: {optimal_thresh:.3f}")

if __name__ == '__main__':
    main()
