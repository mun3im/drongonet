#!/usr/bin/env python3
"""
Analyze resource usage metrics for MCU deployment
Focus: Latency, Model Size, MACs, RAM
"""
import os
import re
import pandas as pd
from pathlib import Path

def extract_model_params(summary_file):
    """Extract parameter count from model summary"""
    model_summary = summary_file.parent / 'model_summary.txt'
    if not model_summary.exists():
        return None

    with open(model_summary, 'r') as f:
        content = f.read()
        match = re.search(r'Total params:\s+([\d,]+)', content)
        if match:
            return int(match.group(1).replace(',', ''))
    return None

def calculate_macs(model_summary_file):
    """
    Estimate MACs from model architecture
    For CNN: MACs ≈ (kernel_h × kernel_w × in_channels × out_channels × output_h × output_w)
    """
    if not model_summary_file.exists():
        return None

    with open(model_summary_file, 'r') as f:
        content = f.read()

    # Simple heuristic: Extract conv layers and sum up operations
    # This is an approximation
    conv_layers = re.findall(r'conv2d.*?\(None, (\d+), (\d+), (\d+)\).*?(\d+)', content)

    total_macs = 0
    # This is a rough estimate - proper calculation needs layer-by-layer analysis
    # For now, return None and rely on model size/latency as proxies
    return None

def analyze_resources(results_dir='results'):
    """Analyze resource usage for all models"""
    results = []

    for dir_name in sorted(os.listdir(results_dir)):
        dir_path = Path(results_dir) / dir_name
        summary_file = dir_path / 'results_summary.txt'

        if not summary_file.exists():
            continue

        with open(summary_file, 'r') as f:
            content = f.read()

        # Extract metrics
        float_auc = re.search(r'Float32 Model AUC:\s+([\d.]+)', content)
        tflite_auc = re.search(r'TFLite int8 Model:.*?AUC:\s+([\d.]+)', content, re.DOTALL)
        model_size = re.search(r'Model Size:\s+([\d.]+)\s+KB', content)
        inference_time = re.search(r'Avg Inference Time:\s+([\d.]+)ms', content)
        training_time = re.search(r'Training:\s+([\dhms.]+)', content)

        # Extract model name and parameters
        match = re.match(r'(.+)_m(\d+)_s(\d+)', dir_name)
        if not match:
            continue

        model_name = match.group(1)
        n_mels = int(match.group(2))

        # Get parameter count
        params = extract_model_params(summary_file)

        result = {
            'model': dir_name,
            'model_name': model_name,
            'n_mels': n_mels,
            'float_auc': float(float_auc.group(1)) if float_auc else None,
            'tflite_auc': float(tflite_auc.group(1)) if tflite_auc else None,
            'model_size_kb': float(model_size.group(1)) if model_size else None,
            'inference_ms': float(inference_time.group(1)) if inference_time else None,
            'parameters': params,
            'training_time': training_time.group(1) if training_time else None,
        }

        # Calculate efficiency metrics
        if result['tflite_auc'] and result['inference_ms']:
            result['auc_per_ms'] = result['tflite_auc'] / result['inference_ms']

        if result['tflite_auc'] and result['model_size_kb']:
            result['auc_per_kb'] = result['tflite_auc'] / result['model_size_kb']

        results.append(result)

    return pd.DataFrame(results)

if __name__ == '__main__':
    df = analyze_resources()

    # Filter for >98% AUC
    df_good = df[df['tflite_auc'] >= 0.98].copy()

    print("=" * 80)
    print("RESOURCE USAGE ANALYSIS - MCU DEPLOYMENT FOCUS")
    print("=" * 80)
    print(f"\nTotal models: {len(df)}")
    print(f"Models with >98% AUC: {len(df_good)}")

    # Sort by latency (fastest first)
    df_latency = df_good.sort_values('inference_ms')

    print("\n=== FASTEST MODELS (>98% AUC) ===")
    print(df_latency[['model', 'tflite_auc', 'inference_ms', 'model_size_kb', 'parameters']].head(10).to_string(index=False))

    # Sort by model size (smallest first)
    df_size = df_good.sort_values('model_size_kb')

    print("\n=== SMALLEST MODELS (>98% AUC) ===")
    print(df_size[['model', 'tflite_auc', 'model_size_kb', 'inference_ms', 'parameters']].head(10).to_string(index=False))

    # Best efficiency (AUC per ms)
    df_efficient = df_good.sort_values('auc_per_ms', ascending=False)

    print("\n=== MOST EFFICIENT (AUC/ms) ===")
    print(df_efficient[['model', 'tflite_auc', 'inference_ms', 'auc_per_ms']].head(10).to_string(index=False))

    # Specific analysis for depthwise models
    df_depthwise = df[df['model_name'].str.contains('depthwise', case=False)].copy()

    if len(df_depthwise) > 0:
        print("\n=== DEPTHWISE MODELS COMPARISON ===")
        df_dw_sorted = df_depthwise.sort_values('inference_ms')
        print(df_dw_sorted[['model', 'n_mels', 'tflite_auc', 'inference_ms', 'model_size_kb']].to_string(index=False))

    # Save detailed analysis
    df.to_csv('resource_usage_analysis.csv', index=False)
    print("\n✓ Detailed analysis saved to: resource_usage_analysis.csv")

    # Recommendations
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATIONS (Priority: Latency > Accuracy)")
    print("=" * 80)

    if len(df_good) > 0:
        fastest = df_latency.iloc[0]
        smallest = df_size.iloc[0]
        best_efficiency = df_efficient.iloc[0]

        print(f"\n🚀 FASTEST (Lowest Latency):")
        print(f"   Model: {fastest['model']}")
        print(f"   AUC: {fastest['tflite_auc']:.4f} | Latency: {fastest['inference_ms']:.2f}ms | Size: {fastest['model_size_kb']:.2f} KB")

        print(f"\n📦 SMALLEST (Lowest Flash):")
        print(f"   Model: {smallest['model']}")
        print(f"   AUC: {smallest['tflite_auc']:.4f} | Size: {smallest['model_size_kb']:.2f} KB | Latency: {smallest['inference_ms']:.2f}ms")

        print(f"\n⚡ BEST EFFICIENCY (AUC/ms):")
        print(f"   Model: {best_efficiency['model']}")
        print(f"   AUC: {best_efficiency['tflite_auc']:.4f} | Efficiency: {best_efficiency['auc_per_ms']:.4f} AUC/ms")
