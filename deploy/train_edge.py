#!/usr/bin/env python3
"""
train_edge.py — DrongoNet-Edge
Reference model for SBC deployment (33.06 KB INT8, 25,890 params, AUC 0.9990 ± 0.0002).
Targets Raspberry Pi, Portenta X8. Meets ≥0.99 recall at a single τ=0.50,
uniform across all three seeds on Linux x86-64 INT8.

Locked configuration (do not change):
  n_mels=80, n_fft=1024, focal loss, GAP, BatchNorm
  Architecture: Conv(16)+BN → Conv(32)+BN → Conv(64)+BN → GAP → Dense(8)

Usage:
    python deploy/train_edge.py \\
        --dataset-path /path/to/seabad \\
        --cache-dir    /path/to/cache_fft1024_m80

Optional:
    --random_seed  INT   (default 42)

Results land in results/drongonet_edge_s{seed}/ (set by the underlying script).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'develop'))

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DrongoNet-Edge (locked config, 33.06 KB INT8, ≥0.99 recall)"
    )
    parser.add_argument('--dataset-path', required=True,
                        help='Path to the SEABAD dataset root')
    parser.add_argument('--cache-dir', required=True,
                        help='Path for mel spectrogram cache (keyed to fft1024, m80)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.random_seed

    # Inject locked parameters as argv for the underlying script
    sys.argv = [
        '6c_edge_final.py',
        '--dataset-path', args.dataset_path,
        '--cache-dir',    args.cache_dir,
        '--n_mels',       '80',
        '--n_fft',        '1024',
        '--random_seed',  str(seed),
    ]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '6c_edge_final',
        Path(__file__).parent.parent / 'develop' / '6c_edge_final.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == '__main__':
    main()
