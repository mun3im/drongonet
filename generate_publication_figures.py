#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for XiaoChirp Paper
Includes: PR curves, ROC curves, Pareto frontier, architecture comparisons
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

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
    'axes.axisbelow': True,
})

# Create output directory
FIGURES_DIR = Path('figures_publication')
FIGURES_DIR.mkdir(exist_ok=True)

def load_sweep_results():
    """Load sweep results from CSV"""
    df = pd.read_csv('sweep_results_all_models.csv')
    return df

def load_tflite_model_and_predict(model_path, test_data):
    """Load TFLite model and run inference"""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    predictions = []
    for sample in test_data:
        # Reshape and convert dtype as needed
        if len(sample.shape) == 2:
            sample = np.expand_dims(sample, axis=(0, -1))
        elif len(sample.shape) == 3:
            sample = np.expand_dims(sample, axis=0)

        sample = sample.astype(input_dtype)

        # Handle int8 quantization
        if input_dtype == np.int8:
            scale = input_details[0]['quantization'][0]
            zero_point = input_details[0]['quantization'][1]
            sample = (sample / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output if needed
        if output_details[0]['dtype'] == np.int8:
            scale = output_details[0]['quantization'][0]
            zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - zero_point) * scale

        predictions.append(output[0])

    return np.array(predictions)

def figure_pareto_frontier():
    """Figure 1: Pareto Frontier - AUC vs M4 Total Latency"""
    print("\nGenerating Figure: Pareto Frontier...")

    df = load_sweep_results()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Define model groups with colors and markers
    model_styles = {
        '1a': {'color': '#1f77b4', 'marker': 'o', 'label': '1a Baseline (Flatten+Dense)', 'size': 80},
        '4a': {'color': '#ff7f0e', 'marker': 's', 'label': '4a Baseline GAP', 'size': 80},
        '7d': {'color': '#2ca02c', 'marker': '^', 'label': '7d GAP+Focal+FreqEmph', 'size': 100},
        '7e': {'color': '#d62728', 'marker': 'D', 'label': '7e Strided (Efficient)', 'size': 100},
    }

    # Plot each model group
    for model_id, style in model_styles.items():
        model_df = df[df['model'] == model_id]
        ax.scatter(model_df['m4_total_ms'], model_df['tflite_auc'] * 100,
                  c=style['color'], marker=style['marker'], s=style['size'],
                  label=style['label'], alpha=0.8, edgecolors='black', linewidth=0.5)

    # Highlight key models (Gatekeeper and Beast)
    # Gatekeeper: 7d fft512 m16
    gatekeeper = df[(df['model'] == '7d') & (df['n_fft'] == 512) & (df['n_mels'] == 16)]
    if len(gatekeeper) > 0:
        ax.scatter(gatekeeper['m4_total_ms'].values[0], gatekeeper['tflite_auc'].values[0] * 100,
                  c='gold', marker='*', s=400, edgecolors='black', linewidth=2, zorder=10,
                  label='Gatekeeper (7d m16)')
        ax.annotate('Gatekeeper\n(70ms, 98.6%)',
                   (gatekeeper['m4_total_ms'].values[0], gatekeeper['tflite_auc'].values[0] * 100),
                   xytext=(15, -15), textcoords='offset points', fontsize=10, fontweight='bold')

    # Beast: 7d fft1024 m80
    beast = df[(df['model'] == '7d') & (df['n_fft'] == 1024) & (df['n_mels'] == 80)]
    if len(beast) > 0:
        ax.scatter(beast['m4_total_ms'].values[0], beast['tflite_auc'].values[0] * 100,
                  c='purple', marker='*', s=400, edgecolors='black', linewidth=2, zorder=10,
                  label='Beast (7d m80)')
        ax.annotate('Beast\n(206ms, 99.6%)',
                   (beast['m4_total_ms'].values[0], beast['tflite_auc'].values[0] * 100),
                   xytext=(15, -10), textcoords='offset points', fontsize=10, fontweight='bold')

    # Reference lines
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.5, label='98% AUC threshold')

    ax.set_xlabel('Cortex-M4 @ 60MHz Total Latency (ms)', fontweight='bold')
    ax.set_ylabel('TFLite int8 AUC (%)', fontweight='bold')
    ax.set_title('XiaoChirp: Accuracy vs Latency Tradeoff\n(Mel Preprocessing + CNN Inference on Cortex-M4)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_xlim(40, 220)
    ax.set_ylim(83, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pareto_frontier.png')
    plt.savefig(FIGURES_DIR / 'pareto_frontier.pdf')
    print(f"  Saved: {FIGURES_DIR}/pareto_frontier.png/.pdf")
    plt.close()

def figure_nmels_sweep():
    """Figure 2: n_mels Impact on Performance"""
    print("\nGenerating Figure: n_mels Sweep Impact...")

    df = load_sweep_results()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color scheme for n_fft
    colors_fft = {'512': '#1f77b4', '1024': '#d62728'}

    models_to_plot = ['1a', '7d', '7e']
    model_labels = {'1a': 'Baseline', '7d': 'GAP+Focal', '7e': 'Strided'}
    markers = {'1a': 'o', '7d': '^', '7e': 'D'}

    # Subplot 1: AUC vs n_mels
    for model_id in models_to_plot:
        for n_fft in [512, 1024]:
            subset = df[(df['model'] == model_id) & (df['n_fft'] == n_fft)].sort_values('n_mels')
            linestyle = '-' if n_fft == 512 else '--'
            label = f'{model_labels[model_id]} (fft{n_fft})'
            axes[0].plot(subset['n_mels'], subset['tflite_auc'] * 100,
                        marker=markers[model_id], linestyle=linestyle, linewidth=2,
                        markersize=8, label=label, alpha=0.8)

    axes[0].axhline(y=98, color='red', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('n_mels (Mel Frequency Bins)', fontweight='bold')
    axes[0].set_ylabel('TFLite int8 AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Accuracy vs Frequency Resolution', fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=8, ncol=2)
    axes[0].set_ylim(92, 100)

    # Subplot 2: M4 Latency vs n_mels
    for model_id in models_to_plot:
        for n_fft in [512, 1024]:
            subset = df[(df['model'] == model_id) & (df['n_fft'] == n_fft)].sort_values('n_mels')
            linestyle = '-' if n_fft == 512 else '--'
            label = f'{model_labels[model_id]} (fft{n_fft})'
            axes[1].plot(subset['n_mels'], subset['m4_total_ms'],
                        marker=markers[model_id], linestyle=linestyle, linewidth=2,
                        markersize=8, label=label, alpha=0.8)

    axes[1].set_xlabel('n_mels (Mel Frequency Bins)', fontweight='bold')
    axes[1].set_ylabel('M4 Total Latency (ms)', fontweight='bold')
    axes[1].set_title('(b) Latency vs Frequency Resolution', fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=8, ncol=2)

    # Subplot 3: Model Size vs n_mels
    for model_id in models_to_plot:
        for n_fft in [512, 1024]:
            subset = df[(df['model'] == model_id) & (df['n_fft'] == n_fft)].sort_values('n_mels')
            linestyle = '-' if n_fft == 512 else '--'
            label = f'{model_labels[model_id]} (fft{n_fft})'
            axes[2].plot(subset['n_mels'], subset['model_size_kb'],
                        marker=markers[model_id], linestyle=linestyle, linewidth=2,
                        markersize=8, label=label, alpha=0.8)

    axes[2].set_xlabel('n_mels (Mel Frequency Bins)', fontweight='bold')
    axes[2].set_ylabel('Model Size (KB)', fontweight='bold')
    axes[2].set_title('(c) Size vs Frequency Resolution', fontweight='bold')
    axes[2].legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nmels_sweep.png')
    plt.savefig(FIGURES_DIR / 'nmels_sweep.pdf')
    print(f"  Saved: {FIGURES_DIR}/nmels_sweep.png/.pdf")
    plt.close()

def figure_architecture_comparison():
    """Figure 3: Architecture Comparison Bar Chart"""
    print("\nGenerating Figure: Architecture Comparison...")

    df = load_sweep_results()

    # Select representative configs (fft512, m32 for fair comparison)
    configs = [
        ('1a', 512, 32, 'Baseline\n(Flatten+Dense)'),
        ('4a', 512, 32, 'Baseline GAP'),
        ('7d', 512, 32, 'GAP+Focal\n+FreqEmph'),
        ('7e', 512, 32, 'Strided\nEfficient'),
    ]

    data = []
    for model_id, n_fft, n_mels, label in configs:
        row = df[(df['model'] == model_id) & (df['n_fft'] == n_fft) & (df['n_mels'] == n_mels)]
        if len(row) > 0:
            data.append({
                'label': label,
                'auc': row['tflite_auc'].values[0] * 100,
                'latency': row['m4_total_ms'].values[0],
                'size': row['model_size_kb'].values[0],
                'macs': row['total_macs'].values[0] / 1000  # KMACs
            })

    comp_df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # AUC
    bars1 = axes[0].bar(comp_df['label'], comp_df['auc'], color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=98, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('TFLite int8 AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Accuracy', fontweight='bold')
    axes[0].set_ylim(90, 100)
    for bar, val in zip(bars1, comp_df['auc']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Latency
    bars2 = axes[1].bar(comp_df['label'], comp_df['latency'], color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('M4 @ 60MHz Latency (ms)', fontweight='bold')
    axes[1].set_title('(b) Latency', fontweight='bold')
    for bar, val in zip(bars2, comp_df['latency']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    # Size
    bars3 = axes[2].bar(comp_df['label'], comp_df['size'], color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Model Size (KB)', fontweight='bold')
    axes[2].set_title('(c) Model Size', fontweight='bold')
    for bar, val in zip(bars3, comp_df['size']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    # MACs
    bars4 = axes[3].bar(comp_df['label'], comp_df['macs'], color=colors, alpha=0.8, edgecolor='black')
    axes[3].set_ylabel('Total MACs (K)', fontweight='bold')
    axes[3].set_title('(d) Computational Cost', fontweight='bold')
    for bar, val in zip(bars4, comp_df['macs']):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:.0f}K', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'architecture_comparison.png')
    plt.savefig(FIGURES_DIR / 'architecture_comparison.pdf')
    print(f"  Saved: {FIGURES_DIR}/architecture_comparison.png/.pdf")
    plt.close()

def figure_mel_latency_breakdown():
    """Figure 4: Latency Breakdown (Mel vs CNN)"""
    print("\nGenerating Figure: Latency Breakdown...")

    df = load_sweep_results()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Select 7d model for demonstration
    model_7d_512 = df[(df['model'] == '7d') & (df['n_fft'] == 512)].sort_values('n_mels')
    model_7d_1024 = df[(df['model'] == '7d') & (df['n_fft'] == 1024)].sort_values('n_mels')

    # Stacked bar chart - FFT 512
    x = np.arange(len(model_7d_512))
    width = 0.35

    axes[0].bar(x, model_7d_512['m4_mel_ms'], width, label='Mel Preprocessing', color='#1f77b4', alpha=0.8)
    axes[0].bar(x, model_7d_512['m4_cnn_ms'], width, bottom=model_7d_512['m4_mel_ms'],
               label='CNN Inference', color='#ff7f0e', alpha=0.8)

    axes[0].set_xlabel('n_mels', fontweight='bold')
    axes[0].set_ylabel('Latency (ms)', fontweight='bold')
    axes[0].set_title('(a) Latency Breakdown (n_fft=512)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_7d_512['n_mels'].values)
    axes[0].legend(loc='upper left')

    # Add total labels
    for i, (mel, cnn) in enumerate(zip(model_7d_512['m4_mel_ms'], model_7d_512['m4_cnn_ms'])):
        axes[0].text(i, mel + cnn + 2, f'{mel+cnn:.0f}ms', ha='center', fontsize=9, fontweight='bold')

    # Stacked bar chart - FFT 1024
    axes[1].bar(x, model_7d_1024['m4_mel_ms'], width, label='Mel Preprocessing', color='#1f77b4', alpha=0.8)
    axes[1].bar(x, model_7d_1024['m4_cnn_ms'], width, bottom=model_7d_1024['m4_mel_ms'],
               label='CNN Inference', color='#ff7f0e', alpha=0.8)

    axes[1].set_xlabel('n_mels', fontweight='bold')
    axes[1].set_ylabel('Latency (ms)', fontweight='bold')
    axes[1].set_title('(b) Latency Breakdown (n_fft=1024)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_7d_1024['n_mels'].values)
    axes[1].legend(loc='upper left')

    # Add total labels
    for i, (mel, cnn) in enumerate(zip(model_7d_1024['m4_mel_ms'], model_7d_1024['m4_cnn_ms'])):
        axes[1].text(i, mel + cnn + 2, f'{mel+cnn:.0f}ms', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'latency_breakdown.png')
    plt.savefig(FIGURES_DIR / 'latency_breakdown.pdf')
    print(f"  Saved: {FIGURES_DIR}/latency_breakdown.png/.pdf")
    plt.close()

def figure_quantization_robustness():
    """Figure 5: Quantization Robustness (Float vs TFLite AUC)"""
    print("\nGenerating Figure: Quantization Robustness...")

    df = load_sweep_results()
    df['auc_degradation'] = (df['float_auc'] - df['tflite_auc']) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: Float vs TFLite AUC
    colors = {'1a': '#1f77b4', '4a': '#ff7f0e', '7d': '#2ca02c', '7e': '#d62728'}
    for model_id in ['1a', '4a', '7d', '7e']:
        subset = df[df['model'] == model_id]
        axes[0].scatter(subset['float_auc'] * 100, subset['tflite_auc'] * 100,
                       c=colors[model_id], s=60, alpha=0.7, label=model_id, edgecolors='black', linewidth=0.5)

    # Perfect line
    axes[0].plot([92, 100], [92, 100], 'k--', alpha=0.3, label='No degradation')
    axes[0].set_xlabel('Float32 AUC (%)', fontweight='bold')
    axes[0].set_ylabel('TFLite int8 AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Quantization Impact on AUC', fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim(92, 100)
    axes[0].set_ylim(92, 100)

    # Histogram of degradation
    axes[1].hist(df['auc_degradation'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('AUC Degradation (Float - TFLite) %', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('(b) Distribution of Quantization Degradation', fontweight='bold')

    # Add statistics
    mean_deg = df['auc_degradation'].mean()
    max_deg = df['auc_degradation'].max()
    axes[1].text(0.95, 0.95, f'Mean: {mean_deg:.3f}%\nMax: {max_deg:.3f}%',
                transform=axes[1].transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'quantization_robustness.png')
    plt.savefig(FIGURES_DIR / 'quantization_robustness.pdf')
    print(f"  Saved: {FIGURES_DIR}/quantization_robustness.png/.pdf")
    plt.close()

def figure_mel_sparsity():
    """Figure 6: Mel Filterbank Sparsity Analysis"""
    print("\nGenerating Figure: Mel Filterbank Sparsity...")

    df = load_sweep_results()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get unique n_mels and n_fft combinations
    df_unique = df.drop_duplicates(subset=['n_fft', 'n_mels'])[['n_fft', 'n_mels', 'mel_nnz', 'mel_sparsity']]

    for n_fft in [512, 1024]:
        subset = df_unique[df_unique['n_fft'] == n_fft].sort_values('n_mels')
        color = '#1f77b4' if n_fft == 512 else '#d62728'
        marker = 'o' if n_fft == 512 else 's'

        axes[0].plot(subset['n_mels'], subset['mel_nnz'],
                    marker=marker, linestyle='-', linewidth=2, markersize=10,
                    color=color, label=f'n_fft={n_fft}', alpha=0.8)

        axes[1].plot(subset['n_mels'], subset['mel_sparsity'] * 100,
                    marker=marker, linestyle='-', linewidth=2, markersize=10,
                    color=color, label=f'n_fft={n_fft}', alpha=0.8)

    axes[0].set_xlabel('n_mels', fontweight='bold')
    axes[0].set_ylabel('Non-Zero Elements (NNZ)', fontweight='bold')
    axes[0].set_title('(a) Mel Filterbank Non-Zero Elements', fontweight='bold')
    axes[0].legend()

    axes[1].set_xlabel('n_mels', fontweight='bold')
    axes[1].set_ylabel('Sparsity (%)', fontweight='bold')
    axes[1].set_title('(b) Mel Filterbank Sparsity', fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(85, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mel_sparsity.png')
    plt.savefig(FIGURES_DIR / 'mel_sparsity.pdf')
    print(f"  Saved: {FIGURES_DIR}/mel_sparsity.png/.pdf")
    plt.close()

def figure_final_models_summary():
    """Figure 7: Final Models Summary Table (as figure)"""
    print("\nGenerating Figure: Final Models Summary...")

    df = load_sweep_results()

    # Get Gatekeeper and Beast configs
    gatekeeper = df[(df['model'] == '7d') & (df['n_fft'] == 512) & (df['n_mels'] == 16)].iloc[0]
    beast = df[(df['model'] == '7d') & (df['n_fft'] == 1024) & (df['n_mels'] == 80)].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Create table data
    columns = ['Property', 'Gatekeeper', 'Beast']
    table_data = [
        ['Configuration', 'fft512, m16', 'fft1024, m80'],
        ['TFLite AUC', f"{gatekeeper['tflite_auc']*100:.2f}%", f"{beast['tflite_auc']*100:.2f}%"],
        ['TFLite Accuracy', f"{gatekeeper['tflite_acc']*100:.2f}%", f"{beast['tflite_acc']*100:.2f}%"],
        ['Model Size', f"{gatekeeper['model_size_kb']:.2f} KB", f"{beast['model_size_kb']:.2f} KB"],
        ['Total MACs', f"{gatekeeper['total_macs']/1000:.0f}K", f"{beast['total_macs']/1000:.0f}K"],
        ['M4 Mel Latency', f"{gatekeeper['m4_mel_ms']:.1f} ms", f"{beast['m4_mel_ms']:.1f} ms"],
        ['M4 CNN Latency', f"{gatekeeper['m4_cnn_ms']:.1f} ms", f"{beast['m4_cnn_ms']:.1f} ms"],
        ['M4 Total Latency', f"{gatekeeper['m4_total_ms']:.1f} ms", f"{beast['m4_total_ms']:.1f} ms"],
        ['Input Shape', '16 x 184', '80 x 184'],
        ['Use Case', 'MCU Gatekeeper', 'SBC/Cloud Beast'],
    ]

    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                    cellLoc='center', colColours=['#f0f0f0']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[(0, j)].set_text_props(fontweight='bold')
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style first column
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_text_props(fontweight='bold')
        table[(i, 0)].set_facecolor('#f0f0f0')

    ax.set_title('XiaoChirp Final Model Configurations', fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'final_models_summary.png')
    plt.savefig(FIGURES_DIR / 'final_models_summary.pdf')
    print(f"  Saved: {FIGURES_DIR}/final_models_summary.png/.pdf")
    plt.close()

def generate_roc_comparison():
    """Generate combined ROC curves from existing individual ROC images"""
    print("\nGenerating Figure: ROC Curve Comparison (from existing results)...")

    # Read ROC data from existing classification reports if available
    # For now, we'll create a summary figure pointing to the individual ROC curves

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot reference diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')

    # Since we don't have raw predictions, we'll annotate with key results
    df = load_sweep_results()

    # Annotate key models
    models_info = [
        ('1a', 512, 80, '1a Baseline m80'),
        ('4a', 512, 48, '4a GAP m48'),
        ('7d', 512, 16, '7d Gatekeeper'),
        ('7d', 1024, 80, '7d Beast'),
        ('7e', 512, 80, '7e Strided m80'),
    ]

    text_lines = []
    for model_id, n_fft, n_mels, label in models_info:
        row = df[(df['model'] == model_id) & (df['n_fft'] == n_fft) & (df['n_mels'] == n_mels)]
        if len(row) > 0:
            auc_val = row['tflite_auc'].values[0]
            text_lines.append(f"{label}: AUC = {auc_val:.4f}")

    ax.text(0.6, 0.3, '\n'.join(text_lines), fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           transform=ax.transAxes)

    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve Summary\n(See individual results for full curves)', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')

    # Add note about individual ROC curves
    ax.text(0.5, 0.02, 'Note: Individual ROC curves available in results/*_fft*_m*_s42/tflite_roc_curve.png',
           ha='center', fontsize=8, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_summary.png')
    plt.savefig(FIGURES_DIR / 'roc_summary.pdf')
    print(f"  Saved: {FIGURES_DIR}/roc_summary.png/.pdf")
    plt.close()

def main():
    """Generate all publication figures"""
    print("\n" + "="*70)
    print("XIAOCHIRP PAPER - PUBLICATION FIGURE GENERATION")
    print("="*70)

    # Check if sweep results exist
    if not Path('sweep_results_all_models.csv').exists():
        print("ERROR: sweep_results_all_models.csv not found!")
        print("Run analyze_sweep_results.py first to generate the results CSV.")
        sys.exit(1)

    # Generate all figures
    figure_pareto_frontier()
    figure_nmels_sweep()
    figure_architecture_comparison()
    figure_mel_latency_breakdown()
    figure_quantization_robustness()
    figure_mel_sparsity()
    figure_final_models_summary()
    generate_roc_comparison()

    print("\n" + "="*70)
    print("All publication figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}/")
    print("="*70)
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  - {f.name}")
    print("\nPDF versions also available for LaTeX inclusion.")
    print("="*70)

if __name__ == '__main__':
    main()
