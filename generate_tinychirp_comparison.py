#!/usr/bin/env python3
"""
Generate TinyChirp comparison table for MyBAD paper
Objective: Show latency improvements with MyBAD-Fast and MyBAD-Tiny
"""

import pandas as pd

def generate_tinychirp_comparison():
    """
    Table: Latency Comparison with TinyChirp Baseline

    Note: TinyChirp values need to be filled in from literature
    """

    print("\n" + "="*80)
    print("TABLE: Latency Comparison with TinyChirp Baseline")
    print("="*80)

    # Load MyBAD results
    df = pd.read_csv('resource_usage_analysis.csv')

    # TinyChirp baseline values (TO BE FILLED FROM LITERATURE)
    # IMPORTANT: TinyChirp uses FIXED n_mels=80 (no ablation)
    # MyBAD's contribution: systematic n_mels exploration for stage-1 detection
    tinychirp_baseline = {
        'model': 'TinyChirp CNN-Mel',
        'n_mels': 80,  # FIXED - no ablation in original work
        'auc': None,  # Fill from literature
        'latency_ms': None,  # Fill from literature - this is the key comparison
        'size_kb': None,  # Fill from literature
        'params': None,
        'platform': 'STM32F4',
        'reference': '[TinyChirp citation]'
    }

    # MyBAD variants for comparison
    # IMPORTANT: Include n_mels=80 for apples-to-apples comparison with TinyChirp
    mybad_models = [
        ('1_baseline_m80_s42', 'MyBAD @ n_mels=80'),
        ('1_baseline_m48_s42', 'MyBAD @ n_mels=48'),
        ('1_baseline_m32_s42', 'MyBAD-Fast (n_mels=32)'),
        ('10_depthwise_f6_m16_s42', 'MyBAD-Tiny (n_mels=16)'),
        ('9a_depthwise_drop01_m48_s42', 'MyBAD-Balanced (n_mels=48)')
    ]

    print("\n| Model | n_mels | AUC (%) | Latency (ms) | Size (KB) | Params | Platform | Speedup |")
    print("|-------|--------|---------|--------------|-----------|--------|----------|---------|")

    # TinyChirp baseline (values to be filled)
    print(f"| TinyChirp CNN-Mel | {tinychirp_baseline['n_mels']} | "
          f"{'[TBD]':>7} | {'[TBD]':>12} | {'[TBD]':>9} | {'[TBD]':>7} | "
          f"{tinychirp_baseline['platform']:8} | 1.00× |")

    print("|-------|--------|---------|--------------|-----------|--------|----------|---------|")

    # MyBAD variants
    for model_id, name in mybad_models:
        if model_id in df['model'].values:
            row = df[df['model'] == model_id].iloc[0]

            # Speedup calculation (will be computed once TinyChirp latency is filled)
            speedup_str = '[TBD]'  # = TinyChirp_latency / MyBAD_latency

            print(f"| {name:26} | {int(row['n_mels']):6} | "
                  f"{row['tflite_auc']*100:7.2f} | "
                  f"{row['inference_ms']:12.2f} | "
                  f"{row['model_size_kb']:9.2f} | "
                  f"{int(row['parameters']):7,} | "
                  f"{'GTX1080':8} | {speedup_str:>7} |")

    print("\n**Notes:**")
    print("- [TBD] values require TinyChirp paper citation and measurements")
    print("- MyBAD latency measured on GTX 1080 Ti GPU (Linux)")
    print("- Speedup = TinyChirp_latency / MyBAD_latency")
    print("- **KEY INSIGHT**: TinyChirp uses FIXED n_mels=80 (no ablation)")
    print("- **MyBAD CONTRIBUTION**: Systematic n_mels exploration for stage-1 detection")
    print("- **FINDING**: n_mels=32-48 sufficient for activity detection (vs 80 for species ID)")
    print("- Latency reduction comes from: (1) lower n_mels, (2) architecture optimization")
    print("- Same test platform needed for fair comparison")

    print("\n" + "="*80)
    print("ACTION ITEMS:")
    print("="*80)
    print("1. [ ] Find TinyChirp paper and extract:")
    print("       - Exact latency measurement (ms)")
    print("       - Model size (KB)")
    print("       - Accuracy/AUC metric")
    print("       - Test platform specifications")
    print("2. [ ] Verify our models measured on same/comparable platform")
    print("3. [ ] Calculate speedup ratios")
    print("4. [ ] Add proper citation to TinyChirp in references")
    print("="*80)

    # Generate comparison visualization data
    print("\n" + "="*80)
    print("EXPECTED LATENCY IMPROVEMENTS (Hypothetical)")
    print("="*80)
    print("\nAssuming TinyChirp @ n_mels=80 has similar latency to our n_mels=80 baseline:")
    print("(TinyChirp latency ≈ MyBAD @ n_mels=80 = 0.42ms)")
    print("\n| Model | n_mels | Latency (ms) | vs TinyChirp | Improvement |")
    print("|-------|--------|--------------|--------------|-------------|")

    # Use our n_mels=80 result as proxy for TinyChirp
    baseline_80_row = df[df['model'] == '1_baseline_m80_s42'].iloc[0]
    hypothetical_tinychirp = baseline_80_row['inference_ms']  # 0.42ms

    print(f"| TinyChirp (n_mels=80) | 80 | {hypothetical_tinychirp:12.2f} | 1.00× | baseline |")
    print("|-------|--------|--------------|--------------|-------------|")

    for model_id, name in mybad_models:
        if model_id in df['model'].values:
            row = df[df['model'] == model_id].iloc[0]
            speedup = hypothetical_tinychirp / row['inference_ms']
            improvement = (1 - row['inference_ms']/hypothetical_tinychirp) * 100

            print(f"| {name:26} | {int(row['n_mels']):6} | "
                  f"{row['inference_ms']:12.2f} | "
                  f"{speedup:12.2f}× | {improvement:10.1f}% |")

    print("\n**Key Finding:**")
    print(f"- Reducing n_mels from 80→32: {baseline_80_row['inference_ms']/df[df['model']=='1_baseline_m32_s42'].iloc[0]['inference_ms']:.1f}× speedup "
          f"({(1 - df[df['model']=='1_baseline_m32_s42'].iloc[0]['inference_ms']/baseline_80_row['inference_ms'])*100:.0f}% faster)")
    print(f"- Reducing n_mels from 80→16: {baseline_80_row['inference_ms']/df[df['model']=='10_depthwise_f6_m16_s42'].iloc[0]['inference_ms']:.1f}× speedup "
          f"({(1 - df[df['model']=='10_depthwise_f6_m16_s42'].iloc[0]['inference_ms']/baseline_80_row['inference_ms'])*100:.0f}% faster)")
    print("- Minimal accuracy loss: 98.55% (n_mels=80) → 98.32% (n_mels=32) = -0.23%")

    print("\n⚠️  Assumes TinyChirp latency ≈ our n_mels=80 baseline (same architecture)")
    print("    Verify with actual TinyChirp measurements from literature.")

if __name__ == '__main__':
    generate_tinychirp_comparison()
