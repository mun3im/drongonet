#!/usr/bin/env python3
"""
train_micro.py — SEABADNet-Micro
Primary model (6.26 KB INT8, 919 params, AUC 0.9743 ± 0.0011).
Targets ARM Cortex-M4 (AudioMoth, STM32F4). Meets ≥0.98 recall at τ=0.35.

Locked configuration (do not change):
  n_mels=16, n_fft=1024, dropout=0.1, focal loss, GAP
  Architecture: FrequencyEmphasis → Conv(6) → MaxPool → Conv(12) → Conv(12,1×1) → GAP → Dropout → Dense

Usage:
    python deploy/train_micro.py \\
        --dataset-path /path/to/seabad \\
        --cache-dir    /path/to/cache_fft1024_m16

Optional:
    --random_seed  INT   (default 42)

Results land in results/seabadnet_micro_s{seed}/ (set by the underlying script).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'develop'))

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SEABADNet-Micro (locked config, 6.26 KB INT8, ≥0.98 recall)"
    )
    parser.add_argument('--dataset-path', required=True,
                        help='Path to the SEABAD dataset root')
    parser.add_argument('--cache-dir', required=True,
                        help='Path for mel spectrogram cache (keyed to fft1024, m16)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.random_seed

    # Inject locked parameters as argv for the underlying script
    sys.argv = [
        '6b_micro_final.py',
        '--dataset-path', args.dataset_path,
        '--cache-dir',    args.cache_dir,
        '--n_mels',       '16',
        '--n_fft',        '1024',
        '--random_seed',  str(seed),
    ]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '6b_micro_final',
        Path(__file__).parent.parent / 'develop' / '6b_micro_final.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == '__main__':
    main()
