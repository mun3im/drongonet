#!/usr/bin/env python3
"""
Comprehensive analysis of ablation studies 1-10 with Pareto frontier
Analyzes results from n_fft=1024, n_mels=48, seed=42
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Experiment configuration
EXPERIMENTS = [
    ('1_baseline', 'Baseline'),
    ('2_depthwise', 'Depthwise Sep'),
    ('3_batchnorm', 'BatchNorm'),
    ('4_dense', 'Dense x2'),
    ('5_filters', 'Filters x1.5'),
    ('6_best_accuracy', 'Best (DW+BN+Drop)'),
    ('7_hybrid', 'Hybrid'),
    ('8a_dropout01', 'Dropout 0.1'),
    ('8b_dropout02', 'Dropout 0.2'),
    ('8c_dropout03', 'Dropout 0.3'),
    ('8d_dropout04', 'Dropout 0.4'),
    ('9a_depthwise_drop01', 'DW+Drop 0.1'),
    ('9b_depthwise_drop02', 'DW+Drop 0.2'),
    ('9c_depthwise_drop03', 'DW+Drop 0.3'),
    ('9d_depthwise_drop04', 'DW+Drop 0.4'),
    ('10_depthwise_f6', 'DW Filters x1.5'),
]

N_FFT = 1024
N_MELS = 48
SEED = 42

def extract_results():
    """Extract all metrics from experiment result directories"""
    results = []

    for exp_id, exp_name in EXPERIMENTS:
        result_dir = Path(f'results/{exp_id}_fft{N_FFT}_m{N_MELS}_s{SEED}')
        summary_file = result_dir / 'results_summary.txt'

        if not summary_file.exists():
            print(f"⚠️  Missing: {exp_id}")
            continue

        # Read summary
        content = summary_file.read_text()

        # Extract metrics using robust patterns
        patterns = {
            'float_auc': r'Float32 Model AUC:\s*([\d.]+)',
            'tflite_acc': r'(?:TFLite int8 )?Accuracy:\s*([\d.]+)',
            'tflite_auc': r'TFLite.*?AUC:\s*([\d.]+)',
            'model_size_kb': r'Model Size:\s*([\d.]+)\s*KB',
            'inference_ms': r'Avg Inference Time:\s*([\d.]+)ms',
            'auc_degradation': r'AUC Degradation:\s*([\d.]+)',
        }

        metrics = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                metrics[key] = float(match.group(1))
            else:
                print(f"⚠️  Missing {key} in {exp_id}")
                metrics[key] = None

        # Calculate parameter count estimate from model size
        # INT8 model: ~1 byte per parameter + overhead
        params_estimate = int(metrics['model_size_kb'] * 1000) if metrics['model_size_kb'] else None

        results.append({
            'experiment': exp_id,
            'name': exp_name,
            'float_auc': metrics['float_auc'],
            'tflite_accuracy': metrics['tflite_acc'],
            'tflite_auc': metrics['tflite_auc'],
            'model_size_kb': metrics['model_size_kb'],
            'inference_ms': metrics['inference_ms'],
            'auc_degradation': metrics['auc_degradation'],
            'params_estimate': params_estimate,
        })

    return pd.DataFrame(results)

def identify_pareto_frontier(df, objectives=['tflite_accuracy', 'model_size_kb', 'inference_ms']):
    """
    Identify Pareto frontier points
    - Maximize: tflite_accuracy
    - Minimize: model_size_kb, inference_ms
    """
    pareto_points = []

    for i, row in df.iterrows():
        is_pareto = True
        for j, other in df.iterrows():
            if i == j:
                continue

            # Check if 'other' dominates 'row'
            better_acc = other['tflite_accuracy'] >= row['tflite_accuracy']
            better_size = other['model_size_kb'] <= row['model_size_kb']
            better_time = other['inference_ms'] <= row['inference_ms']

            # At least one strictly better
            strictly_better = (
                (other['tflite_accuracy'] > row['tflite_accuracy']) or
                (other['model_size_kb'] < row['model_size_kb']) or
                (other['inference_ms'] < row['inference_ms'])
            )

            if better_acc and better_size and better_time and strictly_better:
                is_pareto = False
                break

        if is_pareto:
            pareto_points.append(i)

    return pareto_points

def find_best_models(df):
    """Identify top models for different use cases"""
    best = {}

    # 1. Highest Accuracy
    best['accurate'] = df.loc[df['tflite_accuracy'].idxmax()]

    # 2. Smallest Model
    best['tiny'] = df.loc[df['model_size_kb'].idxmin()]

    # 3. Fastest Inference
    best['fast'] = df.loc[df['inference_ms'].idxmin()]

    # 4. Balanced (normalize and find best composite score)
    df_norm = df.copy()
    df_norm['acc_norm'] = (df['tflite_accuracy'] - df['tflite_accuracy'].min()) / (df['tflite_accuracy'].max() - df['tflite_accuracy'].min())
    df_norm['size_norm'] = 1 - (df['model_size_kb'] - df['model_size_kb'].min()) / (df['model_size_kb'].max() - df['model_size_kb'].min())
    df_norm['time_norm'] = 1 - (df['inference_ms'] - df['inference_ms'].min()) / (df['inference_ms'].max() - df['inference_ms'].min())

    # Equal weighting
    df_norm['composite'] = (df_norm['acc_norm'] + df_norm['size_norm'] + df_norm['time_norm']) / 3
    best['balanced'] = df.loc[df_norm['composite'].idxmax()]

    return best

def print_summary_table(df):
    """Print formatted summary table"""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS SUMMARY (n_fft=1024, n_mels=48, seed=42)")
    print("="*100)

    # Sort by accuracy descending
    df_sorted = df.sort_values('tflite_accuracy', ascending=False)

    print(f"{'Experiment':<25} {'Accuracy':>8} {'AUC':>8} {'Size (KB)':>10} {'Time (ms)':>10} {'Params':>10}")
    print("-"*100)

    for _, row in df_sorted.iterrows():
        print(f"{row['name']:<25} {row['tflite_accuracy']:>8.4f} {row['tflite_auc']:>8.4f} "
              f"{row['model_size_kb']:>10.2f} {row['inference_ms']:>10.2f} {row['params_estimate']:>10,}")

    print("="*100)

def print_pareto_analysis(df, pareto_indices):
    """Print Pareto frontier analysis"""
    print("\n" + "="*100)
    print("PARETO FRONTIER ANALYSIS")
    print("="*100)
    print(f"\nPareto-optimal models (non-dominated solutions): {len(pareto_indices)}")
    print("\nModels on Pareto frontier:")
    print("-"*100)

    for idx in sorted(pareto_indices):
        row = df.iloc[idx]
        print(f"  • {row['name']:<25} - Acc: {row['tflite_accuracy']:.4f}, "
              f"Size: {row['model_size_kb']:.2f} KB, Time: {row['inference_ms']:.2f} ms")

    print("\n" + "="*100)

def print_best_models(best):
    """Print best models for each use case"""
    print("\n" + "="*100)
    print("TOP MODELS BY USE CASE")
    print("="*100)

    for category, row in best.items():
        print(f"\n🏆 {category.upper()}:")
        print(f"   Model: {row['name']}")
        print(f"   Accuracy: {row['tflite_accuracy']:.4f} ({row['tflite_accuracy']*100:.2f}%)")
        print(f"   Model Size: {row['model_size_kb']:.2f} KB")
        print(f"   Inference: {row['inference_ms']:.2f} ms")
        print(f"   Parameters: ~{row['params_estimate']:,}")

    print("\n" + "="*100)

def create_pareto_visualizations(df, pareto_indices):
    """Create comprehensive Pareto frontier visualizations"""
    print("\n" + "="*100)
    print("CREATING VISUALIZATIONS")
    print("="*100)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(18, 12))

    # Define colors for Pareto vs non-Pareto points
    colors = ['red' if i in pareto_indices else 'lightblue' for i in range(len(df))]
    sizes = [200 if i in pareto_indices else 100 for i in range(len(df))]

    # Plot 1: Accuracy vs Model Size
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(df['model_size_kb'], df['tflite_accuracy'],
                          c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidths=1)

    # Add labels for Pareto points
    for idx in pareto_indices:
        row = df.iloc[idx]
        ax1.annotate(row['experiment'],
                    (row['model_size_kb'], row['tflite_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Model Size (KB)', fontsize=11)
    ax1.set_ylabel('TFLite Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs Model Size', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy vs Inference Time
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(df['inference_ms'], df['tflite_accuracy'],
                          c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidths=1)

    for idx in pareto_indices:
        row = df.iloc[idx]
        ax2.annotate(row['experiment'],
                    (row['inference_ms'], row['tflite_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Inference Time (ms)', fontsize=11)
    ax2.set_ylabel('TFLite Accuracy', fontsize=11)
    ax2.set_title('Accuracy vs Inference Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Model Size vs Inference Time (colored by accuracy)
    ax3 = plt.subplot(2, 3, 3)
    scatter3 = ax3.scatter(df['model_size_kb'], df['inference_ms'],
                          c=df['tflite_accuracy'], cmap='RdYlGn', s=sizes,
                          alpha=0.7, edgecolors='black', linewidths=1)

    for idx in pareto_indices:
        row = df.iloc[idx]
        ax3.annotate(row['experiment'],
                    (row['model_size_kb'], row['inference_ms']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax3.set_xlabel('Model Size (KB)', fontsize=11)
    ax3.set_ylabel('Inference Time (ms)', fontsize=11)
    ax3.set_title('Size vs Time (colored by Accuracy)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter3, ax=ax3)
    cbar.set_label('Accuracy', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Bar chart - Accuracy comparison
    ax4 = plt.subplot(2, 3, 4)
    df_sorted = df.sort_values('tflite_accuracy', ascending=True)
    bar_colors = ['red' if i in pareto_indices else 'lightblue'
                  for i in df_sorted.index]
    ax4.barh(range(len(df_sorted)), df_sorted['tflite_accuracy'], color=bar_colors)
    ax4.set_yticks(range(len(df_sorted)))
    ax4.set_yticklabels(df_sorted['experiment'], fontsize=8)
    ax4.set_xlabel('TFLite Accuracy', fontsize=11)
    ax4.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Plot 5: Bar chart - Model Size comparison
    ax5 = plt.subplot(2, 3, 5)
    df_sorted_size = df.sort_values('model_size_kb', ascending=True)
    bar_colors_size = ['red' if i in pareto_indices else 'lightblue'
                       for i in df_sorted_size.index]
    ax5.barh(range(len(df_sorted_size)), df_sorted_size['model_size_kb'],
            color=bar_colors_size)
    ax5.set_yticks(range(len(df_sorted_size)))
    ax5.set_yticklabels(df_sorted_size['experiment'], fontsize=8)
    ax5.set_xlabel('Model Size (KB)', fontsize=11)
    ax5.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    # Plot 6: Bar chart - Inference Time comparison
    ax6 = plt.subplot(2, 3, 6)
    df_sorted_time = df.sort_values('inference_ms', ascending=True)
    bar_colors_time = ['red' if i in pareto_indices else 'lightblue'
                       for i in df_sorted_time.index]
    ax6.barh(range(len(df_sorted_time)), df_sorted_time['inference_ms'],
            color=bar_colors_time)
    ax6.set_yticks(range(len(df_sorted_time)))
    ax6.set_yticklabels(df_sorted_time['experiment'], fontsize=8)
    ax6.set_xlabel('Inference Time (ms)', fontsize=11)
    ax6.set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Ablation Study Analysis - Pareto Frontier & Trade-offs',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = 'ablation_pareto_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")

    return fig

def save_results(df, pareto_indices, best):
    """Save results to CSV"""
    # Add Pareto frontier flag
    df['pareto_frontier'] = df.index.isin(pareto_indices)

    # Add use case flags
    df['best_accurate'] = df['experiment'] == best['accurate']['experiment']
    df['best_tiny'] = df['experiment'] == best['tiny']['experiment']
    df['best_fast'] = df['experiment'] == best['fast']['experiment']
    df['best_balanced'] = df['experiment'] == best['balanced']['experiment']

    filename = 'ablation_complete_results.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

def main():
    print("="*100)
    print("ABLATION STUDY ANALYSIS - COMPREHENSIVE REPORT WITH PARETO FRONTIER")
    print("="*100)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: n_fft={N_FFT}, n_mels={N_MELS}, seed={SEED}")

    # Extract results
    print("\n" + "="*100)
    print("EXTRACTING RESULTS")
    print("="*100)
    df = extract_results()

    if df.empty:
        print("❌ No results found!")
        return

    print(f"✓ Found {len(df)}/{len(EXPERIMENTS)} experiments")

    # Summary table
    print_summary_table(df)

    # Identify Pareto frontier
    print("\n" + "="*100)
    print("COMPUTING PARETO FRONTIER")
    print("="*100)
    pareto_indices = identify_pareto_frontier(df)
    print(f"✓ Identified {len(pareto_indices)} Pareto-optimal models")
    print_pareto_analysis(df, pareto_indices)

    # Find best models
    best = find_best_models(df)
    print_best_models(best)

    # Create visualizations
    create_pareto_visualizations(df, pareto_indices)

    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    save_results(df, pareto_indices, best)

    # Final summary
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"📊 Total experiments: {len(df)}")
    print(f"🎯 Pareto-optimal models: {len(pareto_indices)}")
    print(f"📁 Results saved to: ablation_complete_results.csv")
    print(f"📈 Visualizations saved to: ablation_pareto_analysis.png")
    print("="*100)

if __name__ == '__main__':
    main()
