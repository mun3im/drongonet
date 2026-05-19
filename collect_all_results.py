#!/usr/bin/env python3
"""
Collect and analyze all experimental results
"""
import os
import re
from pathlib import Path
import pandas as pd

def parse_results_summary(file_path):
    """Parse a results_summary.txt file"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    float_auc = re.search(r'Float32 Model AUC:\s+([\d.]+)', content)
    tflite_auc = re.search(r'TFLite int8 Model:.*?AUC:\s+([\d.]+)', content, re.DOTALL)
    accuracy = re.search(r'Accuracy:\s+([\d.]+)', content)
    inference_time = re.search(r'Avg Inference Time:\s+([\d.]+)ms', content)
    model_size = re.search(r'Model Size:\s+([\d.]+)\s+KB', content)
    degradation = re.search(r'AUC Degradation:\s+([\d.]+)\s+\(([\d.]+)%\)', content)
    training_time = re.search(r'Training:\s+([\dhms.]+)', content)

    return {
        'float_auc': float(float_auc.group(1)) if float_auc else None,
        'tflite_auc': float(tflite_auc.group(1)) if tflite_auc else None,
        'accuracy': float(accuracy.group(1)) if accuracy else None,
        'inference_time_ms': float(inference_time.group(1)) if inference_time else None,
        'model_size_kb': float(model_size.group(1)) if model_size else None,
        'degradation_abs': float(degradation.group(1)) if degradation else None,
        'degradation_pct': float(degradation.group(2)) if degradation else None,
        'training_time': training_time.group(1) if training_time else None,
    }

def parse_experiment_name(dir_name):
    """Parse experiment directory name to extract model and parameters"""
    # Format: {model_name}_m{n_mels}_s{seed}
    match = re.match(r'(.+)_m(\d+)_s(\d+)', dir_name)
    if match:
        return {
            'model_name': match.group(1),
            'n_mels': int(match.group(2)),
            'seed': int(match.group(3)),
        }
    return None

def collect_all_results(results_dir='results'):
    """Collect all results from the results directory"""
    results = []

    for dir_name in sorted(os.listdir(results_dir)):
        dir_path = Path(results_dir) / dir_name
        summary_file = dir_path / 'results_summary.txt'

        if not summary_file.exists():
            continue

        # Parse experiment name
        exp_info = parse_experiment_name(dir_name)
        if not exp_info:
            continue

        # Parse results
        metrics = parse_results_summary(summary_file)

        # Combine info
        result = {**exp_info, **metrics, 'experiment': dir_name}
        results.append(result)

    return pd.DataFrame(results)

def categorize_model(model_name):
    """Categorize models into groups"""
    if model_name.startswith('1_baseline'):
        return 'Phase 1: Baseline'
    elif model_name.startswith('7_best'):
        return 'Phase 2: Best Model'
    elif model_name in ['2_depthwise', '3_dropout', '4_batchnorm', '5_dense', '6_filters', '8_hybrid']:
        return 'Phase 2: Core Variants'
    elif model_name.startswith('9'):
        return 'Phase 2: Dropout Sweep'
    elif model_name in ['10_depthwise_f6', '11_depthwise_bn_f6', '12_depthwise_f5']:
        return 'Phase 3A: Power Efficiency'
    else:
        return 'Other'

if __name__ == '__main__':
    # Collect all results
    df = collect_all_results()

    # Add category
    df['category'] = df['model_name'].apply(categorize_model)

    # Sort by AUC descending
    df_sorted = df.sort_values('float_auc', ascending=False)

    # Save to CSV
    df_sorted.to_csv('all_results_comparison.csv', index=False)
    print(f"Collected {len(df)} experiments")
    print(f"Saved to all_results_comparison.csv")

    # Print top 10
    print("\n=== TOP 10 MODELS BY AUC ===")
    top10 = df_sorted.head(10)[['experiment', 'float_auc', 'tflite_auc', 'model_size_kb', 'inference_time_ms']]
    print(top10.to_string(index=False))

    # Print by category
    print("\n=== BEST IN EACH CATEGORY ===")
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].sort_values('float_auc', ascending=False)
        if len(cat_df) > 0:
            best = cat_df.iloc[0]
            print(f"\n{category}:")
            print(f"  {best['experiment']} - AUC: {best['float_auc']:.4f}")
