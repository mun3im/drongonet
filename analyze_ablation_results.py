#!/usr/bin/env python3
"""
Comprehensive analysis script for ablation study results
Analyzes all 75 experiments (5 models × 5 n_mels × 3 seeds)
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
MODELS = ['1a_baseline', '3a_dropout', '3b_filters', '3c_dense', '3d_batchnorm', '3e_hybrid', '3f_depthwise']
MODEL_NAMES = {
    '1a_baseline': 'Baseline',
    '3a_dropout': 'Dropout',
    '3b_filters': 'More Filters',
    '3c_dense': 'Bigger Dense',
    '3d_batchnorm': 'BatchNorm',
    '3e_hybrid': 'Hybrid (BN+Drop)',
    '3f_depthwise': 'Depthwise Sep'
}
N_MELS = [16, 32, 48, 64, 80]
SEEDS = [42, 100, 786]
BASELINE_BEST = 0.9695  # n_mels=48 baseline

def extract_results():
    """Extract results from all experiment result directories"""
    results = []

    for model in MODELS:
        for n_mels in N_MELS:
            for seed in SEEDS:
                # Result directory pattern - updated to include model name
                model_prefix = {
                    '1a_baseline': 'baseline',
                    '3a_dropout': 'dropout',
                    '3b_filters': 'filters',
                    '3c_dense': 'dense',
                    '3d_batchnorm': 'batchnorm',
                    '3e_hybrid': 'hybrid',
                    '3f_depthwise': 'depthwise'
                }[model]
                result_dir = Path(f'results_{model_prefix}_m{n_mels}_s{seed}')
                summary_file = result_dir / 'results_summary.txt'

                if not summary_file.exists():
                    print(f"⚠ Missing: {model}, n_mels={n_mels}, seed={seed}")
                    continue

                # Read summary
                content = summary_file.read_text()

                # Extract metrics
                float_auc_match = re.search(r'Float32 Model AUC:\s+([\d.]+)', content)
                tflite_auc_match = re.search(r'AUC:\s+([\d.]+)', content.split('TFLite int8 Model:')[1])
                size_match = re.search(r'Model Size:\s+([\d.]+)\s+KB', content)
                time_match = re.search(r'Avg Inference Time:\s+([\d.]+)ms', content)

                if all([float_auc_match, tflite_auc_match, size_match, time_match]):
                    results.append({
                        'model': model,
                        'model_name': MODEL_NAMES[model],
                        'n_mels': n_mels,
                        'seed': seed,
                        'float32_auc': float(float_auc_match.group(1)),
                        'tflite_auc': float(tflite_auc_match.group(1)),
                        'model_size_kb': float(size_match.group(1)),
                        'inference_time_ms': float(time_match.group(1))
                    })

    return pd.DataFrame(results)

def analyze_by_model(df):
    """Analyze performance grouped by model"""
    print("\n" + "="*80)
    print("ANALYSIS BY MODEL")
    print("="*80)

    # Group by model and compute statistics
    model_stats = df.groupby('model_name')['float32_auc'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).round(4)
    model_stats['range'] = (model_stats['max'] - model_stats['min']).round(4)

    # Add comparison to baseline
    model_stats['vs_baseline'] = (model_stats['mean'] - BASELINE_BEST).round(4)
    model_stats['improvement'] = ((model_stats['mean'] - BASELINE_BEST) / BASELINE_BEST * 100).round(2)

    # Sort by mean performance
    model_stats = model_stats.sort_values('mean', ascending=False)

    print(model_stats.to_string())

    return model_stats

def analyze_by_n_mels(df):
    """Analyze performance grouped by n_mels"""
    print("\n" + "="*80)
    print("ANALYSIS BY N_MELS")
    print("="*80)

    # Group by model and n_mels
    pivot = df.pivot_table(
        values='float32_auc',
        index='n_mels',
        columns='model_name',
        aggfunc='mean'
    ).round(4)

    print(pivot.to_string())

    return pivot

def find_best_configs(df):
    """Find best configurations"""
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)

    top10 = df.nlargest(10, 'float32_auc')[
        ['model_name', 'n_mels', 'seed', 'float32_auc', 'tflite_auc', 'model_size_kb', 'inference_time_ms']
    ]
    print(top10.to_string(index=False))

    print("\n" + "="*80)
    print("BOTTOM 10 CONFIGURATIONS")
    print("="*80)

    bottom10 = df.nsmallest(10, 'float32_auc')[
        ['model_name', 'n_mels', 'seed', 'float32_auc', 'tflite_auc', 'model_size_kb', 'inference_time_ms']
    ]
    print(bottom10.to_string(index=False))

    return top10, bottom10

def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (16, 12)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Ablation Study Results - Comprehensive Analysis',
                 fontsize=16, fontweight='bold')

    # Plot 1: Performance by model (boxplot)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('model_name')
    sns.boxplot(data=df_sorted, x='model_name', y='float32_auc', ax=ax1)
    ax1.axhline(y=BASELINE_BEST, color='r', linestyle='--', label=f'Baseline ({BASELINE_BEST:.4f})')
    ax1.set_xlabel('Model Variant', fontsize=11)
    ax1.set_ylabel('Float32 AUC', fontsize=11)
    ax1.set_title('Performance Distribution by Model', fontsize=12)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance by n_mels
    ax2 = axes[0, 1]
    for model in MODEL_NAMES.values():
        model_data = df[df['model_name'] == model].groupby('n_mels')['float32_auc'].mean()
        ax2.plot(model_data.index, model_data.values, marker='o', label=model, linewidth=2)
    ax2.axhline(y=BASELINE_BEST, color='r', linestyle='--', label=f'Baseline ({BASELINE_BEST:.4f})')
    ax2.set_xlabel('Number of Mel Bins', fontsize=11)
    ax2.set_ylabel('Mean Float32 AUC', fontsize=11)
    ax2.set_title('Performance vs. Mel Bins by Model', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Seed variance by model
    ax3 = axes[1, 0]
    variance = df.groupby('model_name')['float32_auc'].std().sort_values()
    variance.plot(kind='barh', ax=ax3, color='skyblue')
    ax3.set_xlabel('Standard Deviation', fontsize=11)
    ax3.set_ylabel('Model Variant', fontsize=11)
    ax3.set_title('Seed Stability (Lower is Better)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Model size vs performance
    ax4 = axes[1, 1]
    for model in MODEL_NAMES.values():
        model_data = df[df['model_name'] == model]
        avg_size = model_data['model_size_kb'].mean()
        avg_auc = model_data['float32_auc'].mean()
        ax4.scatter(avg_size, avg_auc, s=200, label=model, alpha=0.7)
    ax4.axhline(y=BASELINE_BEST, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Model Size (KB)', fontsize=11)
    ax4.set_ylabel('Mean Float32 AUC', fontsize=11)
    ax4.set_title('Size-Performance Trade-off', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Heatmap of performance
    ax5 = axes[2, 0]
    pivot_data = df.pivot_table(
        values='float32_auc',
        index='model_name',
        columns='n_mels',
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn',
                center=BASELINE_BEST, ax=ax5, cbar_kws={'label': 'AUC'})
    ax5.set_xlabel('N_mels', fontsize=11)
    ax5.set_ylabel('Model Variant', fontsize=11)
    ax5.set_title('Performance Heatmap', fontsize=12)

    # Plot 6: Improvement over baseline
    ax6 = axes[2, 1]
    improvement = df.groupby('model_name')['float32_auc'].mean() - BASELINE_BEST
    improvement = improvement.sort_values()
    colors = ['red' if x < 0 else 'green' for x in improvement]
    improvement.plot(kind='barh', ax=ax6, color=colors)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('AUC Improvement vs Baseline', fontsize=11)
    ax6.set_ylabel('Model Variant', fontsize=11)
    ax6.set_title('Model Improvements', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('ablation_study_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ablation_study_analysis.png")

def main():
    print("="*80)
    print("ABLATION STUDY - COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract results
    print("\nExtracting results from experiment directories...")
    df = extract_results()

    if df.empty:
        print("❌ No results found! Check that experiments have completed.")
        return

    print(f"\n✓ Found {len(df)} completed experiments")
    print(f"  Expected: {len(MODELS) * len(N_MELS) * len(SEEDS)} experiments")

    # Save raw results
    df.to_csv('ablation_study_results.csv', index=False)
    print("✓ Saved: ablation_study_results.csv")

    # Analyses
    model_stats = analyze_by_model(df)
    n_mels_analysis = analyze_by_n_mels(df)
    top10, bottom10 = find_best_configs(df)

    # Visualizations
    create_visualizations(df)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    best_model = model_stats.index[0]
    best_auc = model_stats['mean'].iloc[0]
    improvement = model_stats['improvement'].iloc[0]

    print(f"🏆 Best Model: {best_model}")
    print(f"   Mean AUC: {best_auc:.4f}")
    print(f"   Improvement vs Baseline: {improvement:+.2f}%")
    print(f"   Stability (std): {model_stats['std'].iloc[0]:.4f}")
    print(f"\n📊 Total experiments analyzed: {len(df)}")
    print(f"📁 Results saved to: ablation_study_results.csv")
    print(f"📈 Visualizations saved to: ablation_study_analysis.png")
    print("="*80)

if __name__ == '__main__':
    main()
