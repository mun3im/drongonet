#!/usr/bin/env python3
"""
Generate all figures for MyBAD paper (Bioacoustics journal)
Based on PAPER_PLANNING.md specifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_results():
    """Load experimental results"""
    df = pd.read_csv('resource_usage_analysis.csv')
    df = df.rename(columns={
        'model': 'experiment',
        'inference_ms': 'inference_time_ms'
    })
    return df

def figure1_pareto_frontier():
    """Figure 1: Pareto Frontier - Latency vs Accuracy Tradeoff"""
    print("\nGenerating Figure 1: Pareto Frontier (Latency vs AUC)...")

    df = load_results()

    # Filter models meeting >98% threshold
    df_good = df[df['tflite_auc'] >= 0.98].copy()

    # Highlight the 4 MyBAD variants
    mybad_models = {
        '5_filters_m48_s42': 'MyBAD-Accurate',
        '9a_depthwise_drop01_m48_s42': 'MyBAD-Balanced',
        '1_baseline_m32_s42': 'MyBAD-Fast',
        '10_depthwise_f6_m16_s42': 'MyBAD-Tiny'
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all models
    ax.scatter(df_good['inference_time_ms'], df_good['tflite_auc'] * 100,
               alpha=0.4, s=50, c='gray', label='Other models')

    # Highlight MyBAD variants
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, (model, name) in enumerate(mybad_models.items()):
        if model in df_good['experiment'].values:
            row = df_good[df_good['experiment'] == model].iloc[0]
            ax.scatter(row['inference_time_ms'], row['tflite_auc'] * 100,
                      s=200, marker='*', c=colors[i], edgecolors='black',
                      linewidth=1.5, label=name, zorder=10)
            # Add label
            ax.annotate(name.replace('MyBAD-', ''),
                       (row['inference_time_ms'], row['tflite_auc'] * 100),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')

    ax.set_xlabel('Inference Latency (ms)', fontweight='bold')
    ax.set_ylabel('AUC (%)', fontweight='bold')
    ax.set_title('Pareto Frontier: Latency-Accuracy Tradeoff\n(Models with AUC ≥ 98%)',
                 fontweight='bold')
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.3, label='98% threshold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figures/figure1_pareto_frontier.png', bbox_inches='tight')
    plt.savefig('figures/figure1_pareto_frontier.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure1_pareto_frontier.png/.pdf")
    plt.close()

def figure2_nmels_impact():
    """Figure 2: n_mels Impact on Performance"""
    print("\nGenerating Figure 2: n_mels Impact...")

    df = load_results()

    # Filter baseline model n_mels sweep
    baseline_sweep = df[df['model_name'] == '1_baseline'].copy()
    baseline_sweep = baseline_sweep.sort_values('n_mels')

    # Also show depthwise model sweep
    depthwise_sweep = df[df['model_name'] == '10_depthwise_f6'].copy()
    depthwise_sweep = depthwise_sweep.sort_values('n_mels')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Subplot 1: AUC vs n_mels
    axes[0].plot(baseline_sweep['n_mels'], baseline_sweep['tflite_auc'] * 100,
                'o-', linewidth=2, markersize=8, label='Conv2D Baseline', color='#3498db')
    axes[0].plot(depthwise_sweep['n_mels'], depthwise_sweep['tflite_auc'] * 100,
                's--', linewidth=2, markersize=8, label='Depthwise (6 filters)', color='#e74c3c')
    axes[0].axhline(y=98, color='red', linestyle=':', alpha=0.5, label='98% threshold')
    axes[0].set_xlabel('n_mels (Mel Bins)', fontweight='bold')
    axes[0].set_ylabel('AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Accuracy vs Frequency Resolution', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Latency vs n_mels
    axes[1].plot(baseline_sweep['n_mels'], baseline_sweep['inference_time_ms'],
                'o-', linewidth=2, markersize=8, label='Conv2D Baseline', color='#3498db')
    axes[1].plot(depthwise_sweep['n_mels'], depthwise_sweep['inference_time_ms'],
                's--', linewidth=2, markersize=8, label='Depthwise (6 filters)', color='#e74c3c')
    axes[1].set_xlabel('n_mels (Mel Bins)', fontweight='bold')
    axes[1].set_ylabel('Latency (ms)', fontweight='bold')
    axes[1].set_title('(b) Latency vs Frequency Resolution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Subplot 3: Model Size vs n_mels
    axes[2].plot(baseline_sweep['n_mels'], baseline_sweep['model_size_kb'],
                'o-', linewidth=2, markersize=8, label='Conv2D Baseline', color='#3498db')
    axes[2].plot(depthwise_sweep['n_mels'], depthwise_sweep['model_size_kb'],
                's--', linewidth=2, markersize=8, label='Depthwise (6 filters)', color='#e74c3c')
    axes[2].set_xlabel('n_mels (Mel Bins)', fontweight='bold')
    axes[2].set_ylabel('Model Size (KB)', fontweight='bold')
    axes[2].set_title('(c) Size vs Frequency Resolution', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('figures/figure2_nmels_impact.png', bbox_inches='tight')
    plt.savefig('figures/figure2_nmels_impact.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure2_nmels_impact.png/.pdf")
    plt.close()

def figure3_dropout_comparison():
    """Figure 3: Dropout Regularization Comparison"""
    print("\nGenerating Figure 3: Dropout Comparison...")

    df = load_results()

    # Conv2D dropout models
    conv2d_dropout = df[df['experiment'].isin([
        '3a_dropout01_m48_s42',
        '3b_dropout02_m48_s42',
        '3_dropout_m48_s42',
        '3d_dropout04_m48_s42'
    ])].copy()
    conv2d_dropout['dropout_rate'] = [0.1, 0.2, 0.3, 0.4]
    conv2d_dropout = conv2d_dropout.sort_values('dropout_rate')

    # Depthwise dropout models
    depthwise_dropout = df[df['experiment'].isin([
        '9a_depthwise_drop01_m48_s42',
        '9b_depthwise_drop02_m48_s42',
        '9c_depthwise_drop03_m48_s42',
        '9d_depthwise_drop04_m48_s42'
    ])].copy()
    depthwise_dropout['dropout_rate'] = [0.1, 0.2, 0.3, 0.4]
    depthwise_dropout = depthwise_dropout.sort_values('dropout_rate')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: AUC vs Dropout
    axes[0].plot(conv2d_dropout['dropout_rate'], conv2d_dropout['tflite_auc'] * 100,
                'o-', linewidth=2, markersize=10, label='Conv2D', color='#3498db')
    axes[0].plot(depthwise_dropout['dropout_rate'], depthwise_dropout['tflite_auc'] * 100,
                's--', linewidth=2, markersize=10, label='Depthwise Separable', color='#e74c3c')
    axes[0].axhline(y=98, color='red', linestyle=':', alpha=0.5, label='98% threshold')
    axes[0].set_xlabel('Dropout Rate', fontweight='bold')
    axes[0].set_ylabel('AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Dropout Impact on Accuracy', fontweight='bold')
    axes[0].set_xticks([0.1, 0.2, 0.3, 0.4])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Overfitting (Float - TFLite gap)
    conv2d_overfit = (conv2d_dropout['float_auc'] - conv2d_dropout['tflite_auc']) * 100
    depthwise_overfit = (depthwise_dropout['float_auc'] - depthwise_dropout['tflite_auc']) * 100

    x = np.arange(len(conv2d_dropout))
    width = 0.35

    axes[1].bar(x - width/2, conv2d_overfit, width, label='Conv2D', color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, depthwise_overfit, width, label='Depthwise Separable',
                color='#e74c3c', alpha=0.8)
    axes[1].set_xlabel('Dropout Rate', fontweight='bold')
    axes[1].set_ylabel('Quantization Degradation (%)', fontweight='bold')
    axes[1].set_title('(b) Dropout Impact on Quantization Robustness', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['0.1', '0.2', '0.3', '0.4'])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('figures/figure3_dropout_comparison.png', bbox_inches='tight')
    plt.savefig('figures/figure3_dropout_comparison.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure3_dropout_comparison.png/.pdf")
    plt.close()

def figure4_architecture_comparison():
    """Figure 4: Architecture Ablation Study"""
    print("\nGenerating Figure 4: Architecture Comparison...")

    df = load_results()

    # Select core architecture variants at n_mels=48
    models = [
        ('1_baseline_m48_s42', 'Baseline'),
        ('2_depthwise_m48_s42', 'Depthwise'),
        ('3_dropout_m48_s42', '+Dropout'),
        ('4_batchnorm_m48_s42', '+BatchNorm'),
        ('5_dense_m48_s42', '+Dense32'),
        ('5_filters_m48_s42', '+Filters'),
        ('8_hybrid_m48_s42', 'Hybrid')
    ]

    arch_data = []
    for model_id, name in models:
        if model_id in df['experiment'].values:
            row = df[df['experiment'] == model_id].iloc[0]
            arch_data.append({
                'name': name,
                'auc': row['tflite_auc'] * 100,
                'latency': row['inference_time_ms'],
                'size': row['model_size_kb']
            })

    arch_df = pd.DataFrame(arch_data)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(arch_df)))

    # Subplot 1: AUC comparison
    bars1 = axes[0].barh(arch_df['name'], arch_df['auc'], color=colors, alpha=0.8)
    axes[0].axvline(x=98, color='red', linestyle='--', alpha=0.5, label='98% threshold')
    axes[0].set_xlabel('AUC (%)', fontweight='bold')
    axes[0].set_title('(a) Accuracy by Architecture', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].legend()

    # Subplot 2: Latency comparison
    bars2 = axes[1].barh(arch_df['name'], arch_df['latency'], color=colors, alpha=0.8)
    axes[1].set_xlabel('Latency (ms)', fontweight='bold')
    axes[1].set_title('(b) Latency by Architecture', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Subplot 3: Model size comparison
    bars3 = axes[2].barh(arch_df['name'], arch_df['size'], color=colors, alpha=0.8)
    axes[2].set_xlabel('Model Size (KB)', fontweight='bold')
    axes[2].set_title('(c) Size by Architecture', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('figures/figure4_architecture_comparison.png', bbox_inches='tight')
    plt.savefig('figures/figure4_architecture_comparison.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure4_architecture_comparison.png/.pdf")
    plt.close()

def figure5_quantization_robustness():
    """Figure 5: Quantization Robustness Analysis"""
    print("\nGenerating Figure 5: Quantization Robustness...")

    df = load_results()

    # Calculate quantization degradation for all models
    df['quant_deg'] = (df['float_auc'] - df['tflite_auc']) * 100

    # Group by architecture type
    df_sorted = df.sort_values('quant_deg')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Degradation scatter plot
    mybad_models = ['5_filters_m48_s42', '9a_depthwise_drop01_m48_s42',
                    '1_baseline_m32_s42', '10_depthwise_f6_m16_s42']

    for idx, row in df.iterrows():
        if row['experiment'] in mybad_models:
            axes[0].scatter(row['tflite_auc'] * 100, row['quant_deg'],
                          s=200, marker='*', c='red', edgecolors='black',
                          linewidth=1.5, zorder=10)
        else:
            axes[0].scatter(row['tflite_auc'] * 100, row['quant_deg'],
                          s=50, alpha=0.6, c='gray')

    axes[0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5,
                   label='0.1% threshold')
    axes[0].set_xlabel('TFLite int8 AUC (%)', fontweight='bold')
    axes[0].set_ylabel('Quantization Degradation (%)', fontweight='bold')
    axes[0].set_title('(a) Quantization Impact Across All Models', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Top 15 most robust models
    top_robust = df_sorted.head(15)

    colors_robust = ['red' if exp in mybad_models else 'steelblue'
                     for exp in top_robust['experiment']]

    axes[1].barh(range(len(top_robust)), top_robust['quant_deg'],
                color=colors_robust, alpha=0.8)
    axes[1].set_yticks(range(len(top_robust)))
    axes[1].set_yticklabels([exp.replace('_m48_s42', '').replace('_s42', '')
                             for exp in top_robust['experiment']], fontsize=8)
    axes[1].set_xlabel('Quantization Degradation (%)', fontweight='bold')
    axes[1].set_title('(b) Top 15 Most Robust Models', fontweight='bold')
    axes[1].axvline(x=0.1, color='orange', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/figure5_quantization_robustness.png', bbox_inches='tight')
    plt.savefig('figures/figure5_quantization_robustness.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure5_quantization_robustness.png/.pdf")
    plt.close()

def figure6_efficiency_metrics():
    """Figure 6: Efficiency Metrics (AUC per resource unit)"""
    print("\nGenerating Figure 6: Efficiency Metrics...")

    df = load_results()

    mybad_models = {
        '5_filters_m48_s42': 'Accurate',
        '9a_depthwise_drop01_m48_s42': 'Balanced',
        '1_baseline_m32_s42': 'Fast',
        '10_depthwise_f6_m16_s42': 'Tiny'
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: AUC per ms
    colors = ['red' if exp in mybad_models else 'gray' for exp in df['experiment']]
    alphas = [0.9 if exp in mybad_models else 0.4 for exp in df['experiment']]
    sizes = [150 if exp in mybad_models else 30 for exp in df['experiment']]

    axes[0].scatter(df['inference_time_ms'], df['auc_per_ms'],
                   c=colors, alpha=alphas, s=sizes, edgecolors='black', linewidth=0.5)

    # Label MyBAD models
    for model, name in mybad_models.items():
        if model in df['experiment'].values:
            row = df[df['experiment'] == model].iloc[0]
            axes[0].annotate(name, (row['inference_time_ms'], row['auc_per_ms']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')

    axes[0].set_xlabel('Latency (ms)', fontweight='bold')
    axes[0].set_ylabel('Efficiency (AUC / ms)', fontweight='bold')
    axes[0].set_title('(a) Speed Efficiency', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: AUC per KB
    axes[1].scatter(df['model_size_kb'], df['auc_per_kb'],
                   c=colors, alpha=alphas, s=sizes, edgecolors='black', linewidth=0.5)

    # Label MyBAD models
    for model, name in mybad_models.items():
        if model in df['experiment'].values:
            row = df[df['experiment'] == model].iloc[0]
            axes[1].annotate(name, (row['model_size_kb'], row['auc_per_kb']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')

    axes[1].set_xlabel('Model Size (KB)', fontweight='bold')
    axes[1].set_ylabel('Efficiency (AUC / KB)', fontweight='bold')
    axes[1].set_title('(b) Size Efficiency', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/figure6_efficiency_metrics.png', bbox_inches='tight')
    plt.savefig('figures/figure6_efficiency_metrics.pdf', bbox_inches='tight')
    print("  ✓ Saved: figures/figure6_efficiency_metrics.png/.pdf")
    plt.close()

def main():
    """Generate all paper figures"""
    print("\n" + "="*80)
    print("MYBAD PAPER - FIGURE GENERATION")
    print("Generating all figures for Bioacoustics journal submission")
    print("="*80)

    # Create figures directory
    Path('figures').mkdir(exist_ok=True)

    # Generate all figures
    figure1_pareto_frontier()
    figure2_nmels_impact()
    figure3_dropout_comparison()
    figure4_architecture_comparison()
    figure5_quantization_robustness()
    figure6_efficiency_metrics()

    print("\n" + "="*80)
    print("✓ All figures generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - figures/figure1_pareto_frontier.png/.pdf")
    print("  - figures/figure2_nmels_impact.png/.pdf")
    print("  - figures/figure3_dropout_comparison.png/.pdf")
    print("  - figures/figure4_architecture_comparison.png/.pdf")
    print("  - figures/figure5_quantization_robustness.png/.pdf")
    print("  - figures/figure6_efficiency_metrics.png/.pdf")
    print("\nNext steps:")
    print("  1. Review all figures for clarity")
    print("  2. Adjust colors/labels if needed")
    print("  3. Include figures in paper manuscript")
    print("="*80)

if __name__ == '__main__':
    main()
