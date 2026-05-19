#!/usr/bin/env python3
"""Extract results from baseline sweep experiments"""

import re
from pathlib import Path
import pandas as pd

results = []

for result_dir in sorted(Path('.').glob('results_1d_mybad2_cnnmel_m*_r*_darwin')):
    summary_file = result_dir / 'results_summary.txt'

    if not summary_file.exists():
        continue

    # Extract n_mels and seed from directory name
    match = re.search(r'm(\d+)_r(\d+)', result_dir.name)
    if not match:
        continue

    n_mels = int(match.group(1))
    seed = int(match.group(2))

    # Read summary file
    content = summary_file.read_text()

    # Extract metrics
    float_auc_match = re.search(r'Float32 Model AUC:\s+([\d.]+)', content)
    tflite_auc_match = re.search(r'AUC:\s+([\d.]+)', content.split('TFLite int8 Model:')[1])
    size_match = re.search(r'Model Size:\s+([\d.]+)\s+KB', content)
    time_match = re.search(r'Avg Inference Time:\s+([\d.]+)ms', content)

    if all([float_auc_match, tflite_auc_match, size_match, time_match]):
        results.append({
            'n_mels': n_mels,
            'seed': seed,
            'float32_auc': float(float_auc_match.group(1)),
            'tflite_auc': float(tflite_auc_match.group(1)),
            'model_size_kb': float(size_match.group(1)),
            'inference_time_ms': float(time_match.group(1))
        })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values(['n_mels', 'seed'])

# Print results table
print("=" * 100)
print("MyBADv2 Baseline Sweep - Complete Results")
print("=" * 100)
print(df.to_string(index=False))
print()

# Compute statistics by n_mels
print("=" * 100)
print("Statistics by n_mels (Float32 AUC)")
print("=" * 100)
stats = df.groupby('n_mels')['float32_auc'].agg(['mean', 'std', 'min', 'max'])
stats['range'] = stats['max'] - stats['min']
stats = stats.round(4)
print(stats.to_string())
print()

# Overall statistics
print("=" * 100)
print("Overall Statistics (Float32 AUC)")
print("=" * 100)
print(f"Mean: {df['float32_auc'].mean():.4f}")
print(f"Std Dev: {df['float32_auc'].std():.4f}")
print(f"Min: {df['float32_auc'].min():.4f}")
print(f"Max: {df['float32_auc'].max():.4f}")
print(f"Range: {df['float32_auc'].max() - df['float32_auc'].min():.4f}")
print()

# Best performing configuration
best_idx = df['float32_auc'].idxmax()
best = df.loc[best_idx]
print("=" * 100)
print("Best Performance")
print("=" * 100)
print(f"n_mels={best['n_mels']:.0f}, seed={best['seed']:.0f}: {best['float32_auc']:.4f} AUC")
print()

# Save CSV
csv_path = 'baseline_sweep_results.csv'
df.to_csv(csv_path, index=False)
print(f"Saved results to {csv_path}")
