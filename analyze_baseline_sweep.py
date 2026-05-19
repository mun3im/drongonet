#!/usr/bin/env python3
"""
Analyze complete baseline sweep: n_mels × n_fft
Extract all metrics and generate comparison tables
"""

import os
import re
from pathlib import Path
from typing import Dict, Any
import pandas as pd

def extract_metrics(result_dir: Path) -> Dict[str, Any]:
    """Extract all metrics from a result directory"""
    metrics = {
        'n_fft': None,
        'n_mels': None,
        'seed': None,
        'tflite_acc': None,
        'tflite_auc': None,
        'float_auc': None,
        'model_size_kb': None,
        'inference_ms': None,
        'training_time': None,
        'precision': None,
        'recall': None,
        'f1': None,
    }

    # Parse directory name: 1_baseline_fft<nfft>_m<nmels>_s<seed>
    dir_name = result_dir.name
    match = re.match(r'1_baseline_fft(\d+)_m(\d+)_s(\d+)', dir_name)
    if match:
        metrics['n_fft'] = int(match.group(1))
        metrics['n_mels'] = int(match.group(2))
        metrics['seed'] = int(match.group(3))

    # Read results_summary.txt
    summary_file = result_dir / 'results_summary.txt'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            content = f.read()

            # Extract metrics using regex
            patterns = {
                'tflite_acc': r'Accuracy:\s*([\d.]+)',
                'float_auc': r'Float32 Model AUC:\s*([\d.]+)',
                'model_size_kb': r'Model Size:\s*([\d.]+)\s*KB',
                'inference_ms': r'Avg Inference Time:\s*([\d.]+)ms',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[key] = float(match.group(1))

            # Special handling for TFLite AUC (appears after "TFLite int8 Model:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'TFLite int8 Model:' in line:
                    # Look for AUC in next few lines
                    for j in range(i+1, min(i+5, len(lines))):
                        match = re.search(r'AUC:\s*([\d.]+)', lines[j])
                        if match:
                            metrics['tflite_auc'] = float(match.group(1))
                            break

    # Read tflite_classification_report.txt for precision/recall/f1
    report_file = result_dir / 'tflite_classification_report.txt'
    if report_file.exists():
        with open(report_file, 'r') as f:
            content = f.read()

            # Extract macro avg metrics
            match = re.search(r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
            if match:
                metrics['precision'] = float(match.group(1))
                metrics['recall'] = float(match.group(2))
                metrics['f1'] = float(match.group(3))

    # Read elapsed.txt for training time
    elapsed_file = result_dir / 'elapsed.txt'
    if elapsed_file.exists():
        with open(elapsed_file, 'r') as f:
            content = f.read().strip()
            # Parse format like "3m 45s" or "120s"
            match = re.match(r'(\d+)m\s*(\d+)s', content)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                metrics['training_time'] = minutes * 60 + seconds
            else:
                match = re.match(r'(\d+)s', content)
                if match:
                    metrics['training_time'] = int(match.group(1))

    return metrics

def main():
    results_dir = Path('results')

    # Find all baseline result directories
    baseline_dirs = sorted(results_dir.glob('1_baseline_fft*_m*_s42'))

    if not baseline_dirs:
        print("❌ No baseline results found!")
        return

    print("="*100)
    print(f"BASELINE SWEEP ANALYSIS: n_mels × n_fft")
    print("="*100)
    print(f"Found {len(baseline_dirs)} result directories\n")

    # Extract metrics from all directories
    all_metrics = []
    for result_dir in baseline_dirs:
        metrics = extract_metrics(result_dir)
        if metrics['n_fft'] is not None:
            all_metrics.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(['n_fft', 'n_mels'])

    # Print full table
    print("\n" + "="*100)
    print("COMPLETE RESULTS TABLE")
    print("="*100)
    print(df.to_string(index=False))

    # Group by n_mels and compare n_fft
    print("\n\n" + "="*100)
    print("n_fft COMPARISON (512 vs 1024)")
    print("="*100)
    print(f"{'n_mels':<10} {'Metric':<20} {'fft=512':<15} {'fft=1024':<15} {'Diff':<15} {'Winner':<10}")
    print("="*100)

    for nmels in sorted(df['n_mels'].unique()):
        df_nmels = df[df['n_mels'] == nmels]

        if len(df_nmels) != 2:
            continue

        row_512 = df_nmels[df_nmels['n_fft'] == 512].iloc[0]
        row_1024 = df_nmels[df_nmels['n_fft'] == 1024].iloc[0]

        print(f"\nn_mels={nmels}")
        print("-"*100)

        # Compare key metrics
        comparisons = [
            ('TFLite Accuracy', 'tflite_acc', '%', True),  # Higher is better
            ('TFLite AUC', 'tflite_auc', '', True),
            ('Model Size (KB)', 'model_size_kb', 'KB', False),  # Lower is better
            ('Inference Time', 'inference_ms', 'ms', False),
            ('Training Time', 'training_time', 's', False),
            ('F1 Score', 'f1', '', True),
        ]

        for metric_name, metric_key, unit, higher_is_better in comparisons:
            val_512 = row_512[metric_key]
            val_1024 = row_1024[metric_key]

            if val_512 is None or val_1024 is None:
                continue

            diff = val_1024 - val_512
            diff_pct = (diff / val_512) * 100 if val_512 != 0 else 0

            if higher_is_better:
                winner = 'fft=1024' if val_1024 > val_512 else 'fft=512'
                symbol = '↑' if val_1024 > val_512 else '↓'
            else:
                winner = 'fft=512' if val_512 < val_1024 else 'fft=1024'
                symbol = '↓' if val_1024 < val_512 else '↑'

            # Format values
            if metric_key == 'tflite_acc':
                val_512_str = f"{val_512*100:.2f}%"
                val_1024_str = f"{val_1024*100:.2f}%"
                diff_str = f"{diff_pct:+.2f}% {symbol}"
            elif metric_key in ['tflite_auc', 'f1']:
                val_512_str = f"{val_512:.4f}"
                val_1024_str = f"{val_1024:.4f}"
                diff_str = f"{diff:+.4f} {symbol}"
            elif metric_key == 'training_time':
                val_512_str = f"{val_512//60}m {val_512%60}s"
                val_1024_str = f"{val_1024//60}m {val_1024%60}s"
                diff_str = f"{diff:+.0f}s {symbol}"
            else:
                val_512_str = f"{val_512:.2f}{unit}"
                val_1024_str = f"{val_1024:.2f}{unit}"
                diff_str = f"{diff:+.2f}{unit} {symbol}"

            print(f"  {metric_name:<20} {val_512_str:<15} {val_1024_str:<15} {diff_str:<15} {winner:<10}")

    # Find optimal configurations
    print("\n\n" + "="*100)
    print("OPTIMAL CONFIGURATIONS")
    print("="*100)

    # Filter out rows with missing accuracy
    df_valid = df.dropna(subset=['tflite_acc'])

    if len(df_valid) == 0:
        print("\n⚠️  No valid results found (all accuracy values are missing)")
        return

    # Best accuracy
    best_acc = df_valid.loc[df_valid['tflite_acc'].idxmax()]
    print(f"\n🏆 Best Accuracy: n_fft={int(best_acc['n_fft'])}, n_mels={int(best_acc['n_mels'])}")
    print(f"   Accuracy: {best_acc['tflite_acc']*100:.2f}%")
    print(f"   AUC: {best_acc['tflite_auc']:.4f}")
    print(f"   Model Size: {best_acc['model_size_kb']:.2f} KB")
    print(f"   Inference: {best_acc['inference_ms']:.2f} ms")

    # Best AUC
    df_valid_auc = df_valid.dropna(subset=['tflite_auc'])
    if len(df_valid_auc) > 0:
        best_auc = df_valid_auc.loc[df_valid_auc['tflite_auc'].idxmax()]
        print(f"\n⭐ Best AUC: n_fft={int(best_auc['n_fft'])}, n_mels={int(best_auc['n_mels'])}")
        print(f"   AUC: {best_auc['tflite_auc']:.4f}")
        print(f"   Accuracy: {best_auc['tflite_acc']*100:.2f}%")

    # Best efficiency (smallest model with >85% accuracy)
    df_efficient = df_valid[df_valid['tflite_acc'] > 0.85]
    if len(df_efficient) > 0:
        best_efficient = df_efficient.loc[df_efficient['model_size_kb'].idxmin()]
        print(f"\n⚡ Best Efficiency (>85% acc, smallest model):")
        print(f"   n_fft={int(best_efficient['n_fft'])}, n_mels={int(best_efficient['n_mels'])}")
        print(f"   Accuracy: {best_efficient['tflite_acc']*100:.2f}%")
        print(f"   Model Size: {best_efficient['model_size_kb']:.2f} KB")
        print(f"   Inference: {best_efficient['inference_ms']:.2f} ms")

    # Fastest inference with >85% accuracy
    if len(df_efficient) > 0:
        fastest = df_efficient.loc[df_efficient['inference_ms'].idxmin()]
        print(f"\n🚀 Fastest Inference (>85% acc):")
        print(f"   n_fft={int(fastest['n_fft'])}, n_mels={int(fastest['n_mels'])}")
        print(f"   Inference: {fastest['inference_ms']:.2f} ms")
        print(f"   Accuracy: {fastest['tflite_acc']*100:.2f}%")
        print(f"   Model Size: {fastest['model_size_kb']:.2f} KB")

    # Summary statistics
    print("\n\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    # Group by n_fft
    for nfft in sorted(df['n_fft'].unique()):
        df_nfft = df[df['n_fft'] == nfft]
        print(f"\nn_fft={nfft}:")
        print(f"  Accuracy range: {df_nfft['tflite_acc'].min()*100:.2f}% - {df_nfft['tflite_acc'].max()*100:.2f}%")
        print(f"  AUC range: {df_nfft['tflite_auc'].min():.4f} - {df_nfft['tflite_auc'].max():.4f}")
        print(f"  Model size range: {df_nfft['model_size_kb'].min():.2f} - {df_nfft['model_size_kb'].max():.2f} KB")
        print(f"  Inference range: {df_nfft['inference_ms'].min():.2f} - {df_nfft['inference_ms'].max():.2f} ms")
        print(f"  Best n_mels: {df_nfft.loc[df_nfft['tflite_acc'].idxmax(), 'n_mels']:.0f}")

    # Save results to CSV
    csv_file = 'baseline_sweep_complete_analysis.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved to: {csv_file}")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()
