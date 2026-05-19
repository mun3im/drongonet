#!/usr/bin/env python3
"""
Benchmark mel spectrogram computation latency
Compares n_fft=512 vs n_fft=1024 for n_mels=64
"""

import numpy as np
import librosa
import time
from dataclasses import dataclass

@dataclass
class Config:
    target_sr: int = 16000
    target_length: int = 16000 * 3  # 3 seconds
    n_mels: int = 64
    n_fft: int = 1024  # Will vary
    hop_length: int = 256

def compute_mel_spectrogram(waveform: np.ndarray, config: Config) -> np.ndarray:
    """Compute mel spectrogram from waveform."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.target_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=0.0,
        fmax=config.target_sr / 2.0,
        center=False
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Transpose to (time, freq) format
    mel_spec_db = mel_spec_db.T

    # Ensure exactly 184 time steps
    if mel_spec_db.shape[0] > 184:
        mel_spec_db = mel_spec_db[:184, :]
    elif mel_spec_db.shape[0] < 184:
        pad_width = ((0, 184 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=0)

    # Normalize to [0, 1] range
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_db

def benchmark_mel_computation(n_fft: int, n_mels: int = 64, num_trials: int = 100, warmup: int = 10):
    """Benchmark mel spectrogram computation."""
    config = Config(n_mels=n_mels, n_fft=n_fft)

    # Generate synthetic audio (3 seconds @ 16kHz)
    waveform = np.random.randn(config.target_length).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = compute_mel_spectrogram(waveform, config)

    # Benchmark
    latencies = []
    for _ in range(num_trials):
        start = time.perf_counter()
        mel_spec = compute_mel_spectrogram(waveform, config)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to milliseconds

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'output_shape': mel_spec.shape
    }

def main():
    print("="*60)
    print("Mel Spectrogram Computation Latency Benchmark")
    print("="*60)
    print()

    configs = [
        (512, 64),
        (1024, 64),
    ]

    results = {}

    for n_fft, n_mels in configs:
        print(f"Benchmarking n_fft={n_fft}, n_mels={n_mels}...")
        results[(n_fft, n_mels)] = benchmark_mel_computation(n_fft, n_mels, num_trials=100, warmup=10)
        print(f"  ✓ Complete")

    print()
    print("="*60)
    print("Results Summary")
    print("="*60)
    print()

    # Detailed results
    for (n_fft, n_mels), stats in results.items():
        print(f"Configuration: n_fft={n_fft}, n_mels={n_mels}")
        print(f"  Output Shape: {stats['output_shape']}")
        print(f"  Mean Latency: {stats['mean_ms']:.3f} ms")
        print(f"  Std Dev:      {stats['std_ms']:.3f} ms")
        print(f"  Min Latency:  {stats['min_ms']:.3f} ms")
        print(f"  Max Latency:  {stats['max_ms']:.3f} ms")
        print(f"  Median:       {stats['median_ms']:.3f} ms")
        print(f"  95th %ile:    {stats['p95_ms']:.3f} ms")
        print(f"  99th %ile:    {stats['p99_ms']:.3f} ms")
        print()

    # Comparison
    print("="*60)
    print("Comparison: n_fft=512 vs n_fft=1024 (n_mels=64)")
    print("="*60)
    print()

    stats_512 = results[(512, 64)]
    stats_1024 = results[(1024, 64)]

    speedup = stats_1024['mean_ms'] / stats_512['mean_ms']
    difference_ms = stats_1024['mean_ms'] - stats_512['mean_ms']
    difference_pct = ((stats_1024['mean_ms'] - stats_512['mean_ms']) / stats_512['mean_ms']) * 100

    print(f"n_fft=512  Mean Latency: {stats_512['mean_ms']:.3f} ms")
    print(f"n_fft=1024 Mean Latency: {stats_1024['mean_ms']:.3f} ms")
    print()
    print(f"Difference: +{difference_ms:.3f} ms ({difference_pct:+.1f}%)")
    print(f"Speedup (512 vs 1024): {speedup:.2f}x")
    print()

    # Per-file analysis (preprocessing context)
    print("="*60)
    print("Preprocessing Context (40k samples)")
    print("="*60)
    print()

    num_samples = 40000
    total_512_s = (stats_512['mean_ms'] / 1000) * num_samples
    total_1024_s = (stats_1024['mean_ms'] / 1000) * num_samples

    print(f"Total preprocessing time estimate (40k samples):")
    print(f"  n_fft=512:  {total_512_s/60:.1f} minutes ({total_512_s:.0f} seconds)")
    print(f"  n_fft=1024: {total_1024_s/60:.1f} minutes ({total_1024_s:.0f} seconds)")
    print(f"  Difference: {(total_1024_s - total_512_s)/60:.1f} minutes")
    print()

    # Inference context (real-time)
    print("="*60)
    print("Inference Context (Real-time Audio)")
    print("="*60)
    print()

    audio_duration_s = 3.0
    rtf_512 = stats_512['mean_ms'] / (audio_duration_s * 1000)
    rtf_1024 = stats_1024['mean_ms'] / (audio_duration_s * 1000)

    print(f"Real-time Factor (RTF) for 3-second audio:")
    print(f"  n_fft=512:  {rtf_512:.4f} (faster is better)")
    print(f"  n_fft=1024: {rtf_1024:.4f}")
    print()
    print(f"Inference budget (3s audio):")
    print(f"  n_fft=512:  {stats_512['mean_ms']:.2f} ms / 3000 ms = {rtf_512*100:.2f}%")
    print(f"  n_fft=1024: {stats_1024['mean_ms']:.2f} ms / 3000 ms = {rtf_1024*100:.2f}%")
    print()

    # Recommendation
    print("="*60)
    print("Recommendation")
    print("="*60)
    print()

    if difference_ms < 1.0:
        print("✅ Latency difference is NEGLIGIBLE (<1ms)")
        print("   → Use n_fft=1024 for better accuracy (no real speed cost)")
    elif difference_pct < 20:
        print("✅ Latency difference is SMALL (<20%)")
        print("   → Use n_fft=1024 for better accuracy (minor speed cost)")
    else:
        print("⚠️  Latency difference is SIGNIFICANT (≥20%)")
        print("   → Consider trade-off: accuracy vs speed")

    print()

if __name__ == '__main__':
    main()
