#!/usr/bin/env python3
"""
train_nano.py — DrongoNet-Nano
Smallest variant (5.09 KB INT8, 763 params). No recall target; use when
flash budget is the hard constraint.

Locked configuration (do not change):
  n_mels=16, n_fft=512, dropout=0.1, focal loss, GAP
  Architecture: FrequencyEmphasis → Conv(6) → MaxPool → Conv(12) → GAP → Dropout → Dense

Usage:
    python deploy/train_nano.py \\
        --dataset-path /path/to/seabad \\
        --cache-dir    /path/to/cache_fft512_m16

Optional:
    --random_seed  INT   (default 42)

Results land in results/drongonet_nano_s{seed}/ (set by the underlying script).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'develop'))

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DrongoNet-Nano (locked config, 5.41 KB INT8)"
    )
    parser.add_argument('--dataset-path', required=True,
                        help='Path to the SEABAD dataset root')
    parser.add_argument('--cache-dir', required=True,
                        help='Path for mel spectrogram cache (keyed to fft512, m16)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.random_seed

    # Inject locked parameters as argv for the underlying script
    sys.argv = [
        '6a_nano_final.py',
        '--dataset-path', args.dataset_path,
        '--cache-dir',    args.cache_dir,
        '--n_mels',       '16',
        '--n_fft',        '512',
        '--random_seed',  str(seed),
    ]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '6a_nano_final',
        Path(__file__).parent.parent / 'develop' / '6a_nano_final.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == '__main__':
    main()
