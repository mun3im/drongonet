#!/usr/bin/env python3
"""
Comprehensive sweep results analysis for XiaoChirp models.
Collects results from all 4 architectures × 10 configs (n_mels × n_fft).
Includes estimated Cortex M4 @ 60MHz latency with sparse mel preprocessing.
"""

import os
import re
import csv
import math
import numpy as np
from pathlib import Path

# ─── Audio / Mel parameters ──────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION_S = 3.0
HOP_LENGTH = 256
N_FRAMES = 184  # cropped/padded to 184

# ─── Cortex M4 @ 60 MHz parameters ──────────────────────────────────────
M4_FREQ_HZ = 60_000_000

# CMSIS-NN int8 convolution: ~2 int8 MACs/cycle via SIMD (SMLAD)
# With overhead (loop, memory), effective ~1.2 MACs/cycle for conv
M4_CONV_MACS_PER_CYCLE = 1.2

# Dense/FC: similar to conv, ~1.5 MACs/cycle (better data reuse)
M4_FC_MACS_PER_CYCLE = 1.5

# MaxPool/GAP: ~1 op/cycle (just comparisons/additions)
M4_POOL_OPS_PER_CYCLE = 1.0

# FFT: ARM CMSIS-DSP radix-4/2 real FFT
# Benchmark: 512-pt real FFT ~0.14ms @ 80MHz → ~11200 cycles
# 1024-pt real FFT ~0.30ms @ 80MHz → ~24000 cycles
# Scale to 60MHz
FFT_CYCLES = {512: 14933, 1024: 32000}  # cycles at 60MHz

# Log-mel: fixed-point log approximation ~30 cycles per bin
LOG_CYCLES_PER_BIN = 30

# Power spectrum: |X[k]|^2 = re^2 + im^2, ~3 cycles per bin
POWER_SPECTRUM_CYCLES_PER_BIN = 3


def compute_mel_filterbank_nnz(n_fft, n_mels, sr=16000, fmin=0, fmax=None):
    """Compute number of nonzero entries in mel filterbank (sparse matrix)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freq = n_fft // 2 + 1

    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    total_nnz = 0
    for i in range(n_mels):
        f_start = bin_points[i]
        f_center = bin_points[i + 1]
        f_end = bin_points[i + 2]
        # Rising slope: f_start+1 to f_center (inclusive)
        # Falling slope: f_center to f_end-1 (inclusive)
        nnz = max(0, f_end - f_start - 1)  # total nonzero bins for this filter
        total_nnz += nnz

    return total_nnz


def estimate_mel_preprocessing_cycles(n_fft, n_mels):
    """Estimate total cycles for mel spectrogram on Cortex M4."""
    n_freq = n_fft // 2 + 1
    nnz = compute_mel_filterbank_nnz(n_fft, n_mels)

    per_frame_cycles = (
        FFT_CYCLES[n_fft]                          # Real FFT
        + n_freq * POWER_SPECTRUM_CYCLES_PER_BIN    # Power spectrum
        + nnz * 2                                   # Sparse mel multiply-accumulate (~2 cycles/nnz with index lookup)
        + n_mels * LOG_CYCLES_PER_BIN               # Log-mel (fixed-point approx)
    )
    total_cycles = per_frame_cycles * N_FRAMES
    return total_cycles


def compute_conv2d_macs(in_h, in_w, in_c, out_c, kh, kw, stride=1, padding='same'):
    """Compute MACs for a Conv2D layer."""
    if padding == 'same':
        out_h = math.ceil(in_h / stride)
        out_w = math.ceil(in_w / stride)
    else:  # valid
        out_h = math.ceil((in_h - kh + 1) / stride)
        out_w = math.ceil((in_w - kw + 1) / stride)
    macs = out_h * out_w * out_c * kh * kw * in_c
    return macs, out_h, out_w


def compute_maxpool_ops(in_h, in_w, in_c, pool_h=2, pool_w=2, stride_h=2, stride_w=2):
    """Compute operations for MaxPooling."""
    out_h = in_h // stride_h
    out_w = in_w // stride_w
    ops = out_h * out_w * in_c * pool_h * pool_w  # comparisons
    return ops, out_h, out_w


def compute_gap_ops(in_h, in_w, in_c):
    """Compute operations for GlobalAveragePooling2D."""
    return in_h * in_w * in_c  # additions + 1 division per channel


def compute_dense_macs(in_features, out_features):
    """Compute MACs for Dense layer."""
    return in_features * out_features


def estimate_cnn_cycles(model_name, n_mels, n_fft):
    """Estimate CNN inference cycles on Cortex M4 for each architecture."""
    # Input shape: (184, n_mels, 1) — but n_fft only affects mel freq resolution
    # Actually n_fft determines the number of frequency bins before mel,
    # but after mel filterbank, we always have n_mels. Time frames = 184.
    H, W, C = 184, n_mels, 1
    total_conv_macs = 0
    total_pool_ops = 0
    total_fc_macs = 0

    if model_name == '1a':
        # Conv2D(4, 3x3, valid) → ReLU → MaxPool(2,2) → Conv2D(4, 3x3, valid) → ReLU → MaxPool(2,2) → Flatten → Dense(8) → Dense(2)
        macs, H, W = compute_conv2d_macs(H, W, C, 4, 3, 3, padding='valid')
        total_conv_macs += macs; C = 4
        ops, H, W = compute_maxpool_ops(H, W, C)
        total_pool_ops += ops
        macs, H, W = compute_conv2d_macs(H, W, C, 4, 3, 3, padding='valid')
        total_conv_macs += macs; C = 4
        ops, H, W = compute_maxpool_ops(H, W, C)
        total_pool_ops += ops
        flat_size = H * W * C
        total_fc_macs += compute_dense_macs(flat_size, 8)
        total_fc_macs += compute_dense_macs(8, 2)

    elif model_name == '4a':
        # Conv2D(4, 3x3, valid) → ReLU → MaxPool(2,2) → Conv2D(4, 3x3, valid) → ReLU → MaxPool(2,2) → GAP → Dense(2)
        macs, H, W = compute_conv2d_macs(H, W, C, 4, 3, 3, padding='valid')
        total_conv_macs += macs; C = 4
        ops, H, W = compute_maxpool_ops(H, W, C)
        total_pool_ops += ops
        macs, H, W = compute_conv2d_macs(H, W, C, 4, 3, 3, padding='valid')
        total_conv_macs += macs; C = 4
        ops, H, W = compute_maxpool_ops(H, W, C)
        total_pool_ops += ops
        total_pool_ops += compute_gap_ops(H, W, C)
        total_fc_macs += compute_dense_macs(C, 2)

    elif model_name == '7d':
        # FreqEmphasis(trainable, n_mels weights) → Conv2D(8, 3x3, same) → MaxPool(2,2)
        # → Conv2D(16, 3x3, same) → Conv2D(16, 1x1) → GAP → Dense(2)
        # FreqEmphasis: element-wise multiply along freq axis = H * W * C ops
        total_pool_ops += H * W * C  # frequency emphasis (element-wise multiply)
        macs, H, W = compute_conv2d_macs(H, W, C, 8, 3, 3, padding='same')
        total_conv_macs += macs; C = 8
        ops, H, W = compute_maxpool_ops(H, W, C)
        total_pool_ops += ops
        macs, H, W = compute_conv2d_macs(H, W, C, 16, 3, 3, padding='same')
        total_conv_macs += macs; C = 16
        macs, H, W = compute_conv2d_macs(H, W, C, 16, 1, 1, padding='same')
        total_conv_macs += macs; C = 16
        total_pool_ops += compute_gap_ops(H, W, C)
        total_fc_macs += compute_dense_macs(C, 2)

    elif model_name == '7e':
        # FreqEmphasis → Conv2D(8, 3x3, stride=2, same) → Conv2D(16, 3x3, same) → Conv2D(16, 1x1) → GAP → Dense(2)
        total_pool_ops += H * W * C  # frequency emphasis
        macs, H, W = compute_conv2d_macs(H, W, C, 8, 3, 3, stride=2, padding='same')
        total_conv_macs += macs; C = 8
        macs, H, W = compute_conv2d_macs(H, W, C, 16, 3, 3, padding='same')
        total_conv_macs += macs; C = 16
        macs, H, W = compute_conv2d_macs(H, W, C, 16, 1, 1, padding='same')
        total_conv_macs += macs; C = 16
        total_pool_ops += compute_gap_ops(H, W, C)
        total_fc_macs += compute_dense_macs(C, 2)

    # Convert MACs to cycles
    conv_cycles = total_conv_macs / M4_CONV_MACS_PER_CYCLE
    fc_cycles = total_fc_macs / M4_FC_MACS_PER_CYCLE
    pool_cycles = total_pool_ops / M4_POOL_OPS_PER_CYCLE

    total_cycles = conv_cycles + fc_cycles + pool_cycles
    return total_cycles, total_conv_macs, total_fc_macs, total_pool_ops


def parse_results_summary(filepath):
    """Parse a results_summary.txt file."""
    data = {}
    with open(filepath) as f:
        text = f.read()

    # Float AUC: handle both "Float32 Model AUC: X" and "Float32 Model:\n  AUC: X"
    m = re.search(r'Float32.*?AUC:\s*([\d.]+)', text, re.DOTALL)
    if m:
        data['float_auc'] = float(m.group(1))

    # TFLite accuracy
    m = re.search(r'Accuracy:\s*([\d.]+)', text)
    if m:
        data['tflite_acc'] = float(m.group(1))

    # TFLite AUC: look for AUC after "TFLite" or "int8" (case-insensitive, multiline)
    m = re.search(r'TFLite.*?AUC:\s*([\d.]+)', text, re.DOTALL | re.IGNORECASE)
    if m:
        # Make sure we got the TFLite AUC, not the float AUC
        # The TFLite section comes after the Float32 section
        tflite_section = re.search(r'TFLite.*', text, re.DOTALL | re.IGNORECASE)
        if tflite_section:
            section_text = tflite_section.group(0)
            m2 = re.search(r'AUC:\s*([\d.]+)', section_text)
            if m2:
                data['tflite_auc'] = float(m2.group(1))

    m = re.search(r'Inference Time:\s*([\d.]+)\s*ms', text)
    if m:
        data['inference_ms'] = float(m.group(1))

    m = re.search(r'Model Size:\s*([\d.]+)\s*KB', text)
    if m:
        data['model_size_kb'] = float(m.group(1))

    m = re.search(r'Total Params:\s*([\d,]+)', text)
    if m:
        data['params'] = int(m.group(1).replace(',', ''))

    return data


def main():
    base_dir = Path('/home/muneim/Dropbox/Conda/XiaoChirp')
    n_mels_list = [16, 32, 48, 64, 80]
    n_fft_list = [512, 1024]

    models = {
        '1a': {
            'name': '1a_baseline2d',
            'label': '1a Baseline 2D (Flatten+Dense)',
            'dir_pattern': 'results/1a_baseline2d_fft{n_fft}_m{n_mels}_s42',
        },
        '4a': {
            'name': '4a_baseline_gap',
            'label': '4a Baseline GAP',
            'dir_pattern': 'results/4a_baseline_gap_fft{n_fft}_m{n_mels}_s42',
        },
        '7d': {
            'name': '7d_gap_focal_freq_emph_pw',
            'label': '7d GAP+Focal+FreqEmph+PW',
            'dir_pattern': None,  # special handling - has AUC suffix
        },
        '7e': {
            'name': '7e_strided_focal_tuned',
            'label': '7e Strided Focal Tuned',
            'dir_pattern': 'results/7e_strided_focal_tuned_fft{n_fft}_m{n_mels}_s42',
        },
    }

    all_results = []

    for model_key, model_info in models.items():
        for n_fft in n_fft_list:
            for n_mels in n_mels_list:
                # Find result directory
                if model_key == '7d':
                    # 7d dirs have AUC suffix - find by glob
                    pattern = f'7d_gap_focal_loss_freq_emph_pointwise_fft{n_fft}_m{n_mels}_s42_*'
                    matches = list((base_dir / 'results_linux').glob(pattern))
                    if not matches:
                        print(f"WARNING: No results for 7d fft{n_fft} m{n_mels}")
                        continue
                    result_dir = matches[0]
                else:
                    result_dir = base_dir / model_info['dir_pattern'].format(n_fft=n_fft, n_mels=n_mels)

                summary_file = result_dir / 'results_summary.txt'
                if not summary_file.exists():
                    print(f"WARNING: Missing {summary_file}")
                    continue

                data = parse_results_summary(summary_file)

                # Get param count from model_summary.txt if not in results
                if 'params' not in data:
                    model_file = result_dir / 'model_summary.txt'
                    if model_file.exists():
                        with open(model_file) as f:
                            txt = f.read()
                        m = re.search(r'Total params:\s*([\d,]+)', txt)
                        if m:
                            data['params'] = int(m.group(1).replace(',', ''))

                # Estimate Cortex M4 latency
                mel_cycles = estimate_mel_preprocessing_cycles(n_fft, n_mels)
                cnn_cycles, conv_macs, fc_macs, pool_ops = estimate_cnn_cycles(model_key, n_mels, n_fft)

                mel_ms = (mel_cycles / M4_FREQ_HZ) * 1000
                cnn_ms = (cnn_cycles / M4_FREQ_HZ) * 1000
                total_m4_ms = mel_ms + cnn_ms

                # Mel filterbank sparsity info
                mel_nnz = compute_mel_filterbank_nnz(n_fft, n_mels)
                mel_dense = n_mels * (n_fft // 2 + 1)
                sparsity = 1.0 - (mel_nnz / mel_dense)

                row = {
                    'model': model_key,
                    'model_label': model_info['label'],
                    'n_fft': n_fft,
                    'n_mels': n_mels,
                    'float_auc': data.get('float_auc', 0),
                    'tflite_auc': data.get('tflite_auc', 0),
                    'tflite_acc': data.get('tflite_acc', 0),
                    'model_size_kb': data.get('model_size_kb', 0),
                    'params': data.get('params', 0),
                    'host_inference_ms': data.get('inference_ms', 0),
                    'conv_macs': conv_macs,
                    'fc_macs': fc_macs,
                    'total_macs': conv_macs + fc_macs,
                    'mel_nnz': mel_nnz,
                    'mel_sparsity': sparsity,
                    'm4_mel_ms': mel_ms,
                    'm4_cnn_ms': cnn_ms,
                    'm4_total_ms': total_m4_ms,
                }
                all_results.append(row)

    # Sort by model, n_fft, n_mels
    all_results.sort(key=lambda r: (r['model'], r['n_fft'], r['n_mels']))

    # ─── Print summary tables ────────────────────────────────────────
    print("=" * 140)
    print("XIAOCHIRP SWEEP RESULTS — All Models × n_fft × n_mels")
    print(f"Cortex M4 @ {M4_FREQ_HZ/1e6:.0f} MHz estimated latency with sparse mel preprocessing")
    print("=" * 140)

    for model_key in ['1a', '4a', '7d', '7e']:
        model_rows = [r for r in all_results if r['model'] == model_key]
        if not model_rows:
            continue

        label = model_rows[0]['model_label']
        print(f"\n{'─' * 140}")
        print(f"  {label}")
        print(f"{'─' * 140}")
        print(f"{'n_fft':>6} {'n_mels':>6} │ {'Float':>7} {'TFLite':>7} {'TFLite':>7} │ {'Model':>6} {'Params':>7} │ {'Host':>6} │ {'M4 Mel':>7} {'M4 CNN':>7} {'M4 Tot':>7} │ {'Conv':>9} {'Mel':>7}")
        print(f"{'':>6} {'':>6} │ {'AUC':>7} {'AUC':>7} {'Acc':>7} │ {'KB':>6} {'':>7} │ {'ms':>6} │ {'ms':>7} {'ms':>7} {'ms':>7} │ {'MACs':>9} {'NNZ':>7}")
        print(f"{'─' * 6}─{'─' * 6}─┼─{'─' * 7}─{'─' * 7}─{'─' * 7}─┼─{'─' * 6}─{'─' * 7}─┼─{'─' * 6}─┼─{'─' * 7}─{'─' * 7}─{'─' * 7}─┼─{'─' * 9}─{'─' * 7}")

        for r in model_rows:
            print(f"{r['n_fft']:>6} {r['n_mels']:>6} │ "
                  f"{r['float_auc']:>7.4f} {r['tflite_auc']:>7.4f} {r['tflite_acc']:>7.4f} │ "
                  f"{r['model_size_kb']:>6.2f} {r['params']:>7,} │ "
                  f"{r['host_inference_ms']:>6.2f} │ "
                  f"{r['m4_mel_ms']:>7.2f} {r['m4_cnn_ms']:>7.2f} {r['m4_total_ms']:>7.2f} │ "
                  f"{r['total_macs']:>9,} {r['mel_nnz']:>7,}")

    # ─── Cross-model comparison at each config ───────────────────────
    print(f"\n\n{'=' * 140}")
    print("CROSS-MODEL COMPARISON (sorted by TFLite AUC within each config)")
    print(f"{'=' * 140}")

    for n_fft in n_fft_list:
        for n_mels in n_mels_list:
            config_rows = [r for r in all_results if r['n_fft'] == n_fft and r['n_mels'] == n_mels]
            config_rows.sort(key=lambda r: -r['tflite_auc'])
            print(f"\n  n_fft={n_fft}, n_mels={n_mels}:")
            print(f"  {'Model':.<40} {'TFLite AUC':>10} {'Acc':>7} {'Size KB':>8} {'Host ms':>8} {'M4 Tot ms':>10}")
            for r in config_rows:
                print(f"  {r['model_label']:.<40} {r['tflite_auc']:>10.4f} {r['tflite_acc']:>7.4f} {r['model_size_kb']:>8.2f} {r['host_inference_ms']:>8.2f} {r['m4_total_ms']:>10.2f}")

    # ─── Pareto-optimal models ───────────────────────────────────────
    print(f"\n\n{'=' * 140}")
    print("PARETO FRONTIER: TFLite AUC vs M4 Total Latency")
    print(f"{'=' * 140}")

    # Find Pareto-optimal: no other model has both higher AUC AND lower latency
    pareto = []
    for r in all_results:
        dominated = False
        for other in all_results:
            if other is r:
                continue
            if other['tflite_auc'] >= r['tflite_auc'] and other['m4_total_ms'] <= r['m4_total_ms']:
                if other['tflite_auc'] > r['tflite_auc'] or other['m4_total_ms'] < r['m4_total_ms']:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda r: r['m4_total_ms'])
    print(f"\n{'Model':.<40} {'n_fft':>5} {'n_mels':>6} {'TFLite AUC':>11} {'TFLite Acc':>11} {'Size KB':>8} {'M4 Total ms':>12}")
    for r in pareto:
        print(f"{r['model_label']:.<40} {r['n_fft']:>5} {r['n_mels']:>6} {r['tflite_auc']:>11.4f} {r['tflite_acc']:>11.4f} {r['model_size_kb']:>8.2f} {r['m4_total_ms']:>12.2f}")

    # ─── Best model per latency budget ───────────────────────────────
    print(f"\n\n{'=' * 140}")
    print("BEST MODEL PER LATENCY BUDGET (M4 @ 60MHz)")
    print(f"{'=' * 140}")
    budgets = [50, 100, 150, 200, 300, 500]
    for budget in budgets:
        candidates = [r for r in all_results if r['m4_total_ms'] <= budget]
        if candidates:
            best = max(candidates, key=lambda r: r['tflite_auc'])
            print(f"  ≤{budget:>4}ms: {best['model_label']:<40} fft{best['n_fft']} m{best['n_mels']:<2}  AUC={best['tflite_auc']:.4f}  Acc={best['tflite_acc']:.4f}  Size={best['model_size_kb']:.2f}KB  Latency={best['m4_total_ms']:.1f}ms")
        else:
            print(f"  ≤{budget:>4}ms: No model fits this budget")

    # ─── Save CSV ────────────────────────────────────────────────────
    csv_path = base_dir / 'sweep_results_all_models.csv'
    fieldnames = ['model', 'model_label', 'n_fft', 'n_mels',
                  'float_auc', 'tflite_auc', 'tflite_acc',
                  'model_size_kb', 'params',
                  'host_inference_ms',
                  'm4_mel_ms', 'm4_cnn_ms', 'm4_total_ms',
                  'conv_macs', 'fc_macs', 'total_macs', 'mel_nnz', 'mel_sparsity']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n\nCSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
