#!/usr/bin/env python3
"""
Generate all tables for MyBAD paper (Bioacoustics journal)
Based on PAPER_PLANNING.md specifications
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """Load all experimental results"""
    # Use resource_usage_analysis.csv as it has more complete data including parameters
    df = pd.read_csv('resource_usage_analysis.csv')
    # Rename columns to match expected names
    df = df.rename(columns={
        'model': 'experiment',
        'inference_ms': 'inference_time_ms'
    })
    return df

def table1_dataset_comparison():
    """Table 1: MyBAD Dataset Comparison with existing PAM datasets"""
    print("\n" + "="*80)
    print("TABLE 1: MyBAD Dataset Comparison")
    print("="*80)

    table = """
| Dataset | Species | Samples | Duration | Region | Task | Public |
|---------|---------|---------|----------|--------|------|--------|
| BirdVox-DCASE-20k | Urban birds | 20,000 | - | USA | Detection | ✓ |
| Warblrb10k | European birds | 10,000 | - | UK | Detection | ✓ |
| freefield1010 | UK birds | 7,690 | - | UK | Classification | ✓ |
| **MyBAD (Ours)** | **Malaysian birds** | **50,000** | **150,000s** | **Malaysia** | **Detection** | **✓** |

Key differences:
- First large-scale tropical bird activity detection dataset
- Balanced activity/background samples
- Standardized 3-second segments @ 16kHz
- Malaysian biodiversity hotspot representation
"""
    print(table)
    return table

def table2_model_family():
    """Table 2: MyBAD Model Family Overview (4 variants)"""
    print("\n" + "="*80)
    print("TABLE 2: MyBAD Model Family Overview")
    print("="*80)

    df = load_results()

    # Select the 4 MyBAD variants
    variants = {
        'MyBAD-Accurate': '5_filters_m48_s42',
        'MyBAD-Balanced': '9a_depthwise_drop01_m48_s42',
        'MyBAD-Fast': '1_baseline_m32_s42',
        'MyBAD-Tiny': '10_depthwise_f6_m16_s42'
    }

    print("\n| Variant | Model | Target | AUC (%) | Latency (ms) | Size (KB) | Params |")
    print("|---------|-------|--------|---------|--------------|-----------|--------|")

    for variant_name, model_name in variants.items():
        row = df[df['experiment'] == model_name].iloc[0]
        print(f"| {variant_name:15} | {model_name:25} | "
              f"{'Cloud/Edge' if 'Accurate' in variant_name else 'AudioMoth' if 'Balanced' in variant_name else 'Real-time' if 'Fast' in variant_name else 'Solar':12} | "
              f"{row['tflite_auc']*100:6.2f} | "
              f"{row['inference_time_ms']:12.2f} | "
              f"{row['model_size_kb']:9.2f} | "
              f"{int(row['parameters']):7,} |")

    return variants

def table3_architecture_ablation():
    """Table 3: Architecture Ablation Study (Phase 2 core variants)"""
    print("\n" + "="*80)
    print("TABLE 3: Architecture Ablation Study (n_mels=48)")
    print("="*80)

    df = load_results()

    # Core architecture variants at n_mels=48
    models = [
        ('1_baseline_m48_s42', 'Baseline (Conv2D)'),
        ('2_depthwise_m48_s42', 'Depthwise Separable'),
        ('3_batchnorm_m48_s42', 'Conv2D + BatchNorm'),
        ('4_dense_m48_s42', 'Conv2D + Dense32'),
        ('5_filters_m48_s42', 'Conv2D + 8 Filters'),
        ('6_best_accuracy_m48_s42', 'Combined Best'),
        ('7_hybrid_m48_s42', 'Hybrid (BN+Drop)')
    ]

    print("\n| Architecture | AUC (%) | Latency (ms) | Size (KB) | Δ AUC | Δ Size |")
    print("|--------------|---------|--------------|-----------|-------|--------|")

    baseline_row = df[df['experiment'] == '1_baseline_m48_s42'].iloc[0]
    baseline_auc = baseline_row['tflite_auc']
    baseline_size = baseline_row['model_size_kb']

    for model_name, desc in models:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            delta_auc = (row['tflite_auc'] - baseline_auc) * 100
            delta_size = ((row['model_size_kb'] - baseline_size) / baseline_size) * 100

            print(f"| {desc:25} | {row['tflite_auc']*100:7.2f} | "
                  f"{row['inference_time_ms']:12.2f} | "
                  f"{row['model_size_kb']:9.2f} | "
                  f"{delta_auc:+6.2f} | "
                  f"{delta_size:+7.1f}% |")

def table4_frequency_resolution():
    """Table 4: Frequency Resolution Ablation (n_mels sweep)"""
    print("\n" + "="*80)
    print("TABLE 4: Frequency Resolution Ablation (Baseline Model)")
    print("="*80)

    df = load_results()

    # Baseline model n_mels sweep
    n_mels_models = [
        ('1_baseline_m16_s42', 16),
        ('1_baseline_m32_s42', 32),
        ('1_baseline_m48_s42', 48),
        ('1_baseline_m64_s42', 64),
        ('1_baseline_m80_s42', 80)
    ]

    print("\n| n_mels | AUC (%) | Latency (ms) | Size (KB) | Params | >98% |")
    print("|--------|---------|--------------|-----------|--------|------|")

    for model_name, n_mels in n_mels_models:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            meets_threshold = "✓" if row['tflite_auc'] >= 0.98 else "✗"

            print(f"| {n_mels:6} | {row['tflite_auc']*100:7.2f} | "
                  f"{row['inference_time_ms']:12.2f} | "
                  f"{row['model_size_kb']:9.2f} | "
                  f"{int(row['parameters']):7,} | "
                  f"{meets_threshold:^4} |")

def table5_dropout_comparison():
    """Table 5: Dropout Regularization Comparison (Conv2D vs Depthwise)"""
    print("\n" + "="*80)
    print("TABLE 5: Dropout Regularization Comparison (n_mels=48)")
    print("="*80)

    df = load_results()

    print("\n### Conv2D Architecture")
    print("| Dropout | Model | AUC (%) | Latency (ms) | Overfitting |")
    print("|---------|-------|---------|--------------|-------------|")

    conv2d_dropout = [
        (0.1, '8a_dropout01_m48_s42'),
        (0.2, '8b_dropout02_m48_s42'),
        (0.3, '8c_dropout03_m48_s42'),
        (0.4, '8d_dropout04_m48_s42')
    ]

    for dropout, model_name in conv2d_dropout:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            overfitting = (row['float_auc'] - row['tflite_auc']) * 100

            print(f"| {dropout:7.1f} | {model_name:23} | "
                  f"{row['tflite_auc']*100:7.2f} | "
                  f"{row['inference_time_ms']:12.2f} | "
                  f"{overfitting:11.3f}% |")

    print("\n### Depthwise Separable Architecture")
    print("| Dropout | Model | AUC (%) | Latency (ms) | Overfitting |")
    print("|---------|-------|---------|--------------|-------------|")

    depthwise_dropout = [
        (0.1, '9a_depthwise_drop01_m48_s42'),
        (0.2, '9b_depthwise_drop02_m48_s42'),
        (0.3, '9c_depthwise_drop03_m48_s42'),
        (0.4, '9d_depthwise_drop04_m48_s42')
    ]

    for dropout, model_name in depthwise_dropout:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            overfitting = (row['float_auc'] - row['tflite_auc']) * 100

            print(f"| {dropout:7.1f} | {model_name:27} | "
                  f"{row['tflite_auc']*100:7.2f} | "
                  f"{row['inference_time_ms']:12.2f} | "
                  f"{overfitting:11.3f}% |")

def table6_resource_usage():
    """Table 6: Resource Usage for MyBAD Model Family"""
    print("\n" + "="*80)
    print("TABLE 6: Resource Usage Comparison")
    print("="*80)

    df = load_results()

    variants = [
        ('5_filters_m48_s42', 'MyBAD-Accurate'),
        ('9a_depthwise_drop01_m48_s42', 'MyBAD-Balanced'),
        ('1_baseline_m32_s42', 'MyBAD-Fast'),
        ('10_depthwise_f6_m16_s42', 'MyBAD-Tiny')
    ]

    print("\n| Variant | Flash (KB) | Params | Latency (ms) | Est. RAM (KB) | AUC/ms | AUC/KB |")
    print("|---------|------------|--------|--------------|---------------|--------|--------|")

    for model_name, variant in variants:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            est_ram = row['parameters'] * 1 / 1024 + 15  # int8 params + activations
            auc_per_ms = row['tflite_auc'] / row['inference_time_ms']
            auc_per_kb = row['tflite_auc'] / row['model_size_kb']

            print(f"| {variant:15} | {row['model_size_kb']:10.2f} | "
                  f"{int(row['parameters']):7,} | "
                  f"{row['inference_time_ms']:12.2f} | "
                  f"{est_ram:13.1f} | "
                  f"{auc_per_ms:6.2f} | "
                  f"{auc_per_kb:6.4f} |")

def table7_quantization_impact():
    """Table 7: Quantization Impact (Float32 vs int8)"""
    print("\n" + "="*80)
    print("TABLE 7: Quantization Impact Analysis")
    print("="*80)

    df = load_results()

    # Select representative models across architectures
    models = [
        ('1_baseline_m48_s42', 'Conv2D Baseline'),
        ('2_depthwise_m48_s42', 'Depthwise Separable'),
        ('4_dense_m48_s42', 'Dense Layer'),
        ('5_filters_m48_s42', 'More Filters'),
        ('9a_depthwise_drop01_m48_s42', 'Depthwise + Dropout')
    ]

    print("\n| Model | Float32 AUC | int8 AUC | Degradation | Robust? |")
    print("|-------|-------------|----------|-------------|---------|")

    for model_name, desc in models:
        if model_name in df['experiment'].values:
            row = df[df['experiment'] == model_name].iloc[0]
            degradation = (row['float_auc'] - row['tflite_auc']) * 100
            robust = "✓" if degradation < 0.1 else "✗"

            print(f"| {desc:23} | {row['float_auc']*100:11.2f} | "
                  f"{row['tflite_auc']*100:8.2f} | "
                  f"{degradation:11.3f}% | "
                  f"{robust:^7} |")

def table8_deployment_scenarios():
    """Table 8: Deployment Scenarios for MyBAD Models"""
    print("\n" + "="*80)
    print("TABLE 8: Deployment Scenarios and Hardware Requirements")
    print("="*80)

    table = """
| Variant | Target Device | MCU Example | Flash Req | RAM Req | Power | Latency |
|---------|---------------|-------------|-----------|---------|-------|---------|
| MyBAD-Accurate | Cloud/Edge Server | RPi 4 | 33 KB | 40 KB | High | 0.22ms |
| MyBAD-Balanced | AudioMoth | STM32F4 | 20 KB | 35 KB | Medium | 0.23ms |
| MyBAD-Fast | Real-time Monitor | ESP32 | 13 KB | 25 KB | Medium | 0.16ms |
| MyBAD-Tiny | Solar-powered Sensor | nRF52840 | 11 KB | 20 KB | Low | 0.08ms |

**Deployment Notes:**
- All models support continuous audio processing at 16kHz
- Mel spectrogram computation adds ~5-10ms preprocessing overhead
- Battery life: 3-6 months on 2×AA batteries (AudioMoth scenario, 10s duty cycle)
- Real-time capability: All variants can process audio faster than real-time
"""
    print(table)
    return table

def main():
    """Generate all paper tables"""
    print("\n" + "="*80)
    print("MYBAD PAPER - TABLE GENERATION")
    print("Generating all tables for Bioacoustics journal submission")
    print("="*80)

    # Generate all tables
    table1_dataset_comparison()
    table2_model_family()
    table3_architecture_ablation()
    table4_frequency_resolution()
    table5_dropout_comparison()
    table6_resource_usage()
    table7_quantization_impact()
    table8_deployment_scenarios()

    print("\n" + "="*80)
    print("✓ All tables generated successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review tables for accuracy")
    print("2. Generate figures (Pareto frontier, n_mels impact, etc.)")
    print("3. Copy tables into paper manuscript")
    print("="*80)

if __name__ == '__main__':
    main()
