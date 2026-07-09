#!/usr/bin/env python3
"""
generate_fig6_part1_extract.py — Part 1 (LINUX ONLY): run TFLite inference and cache the
raw probabilities/labels needed for Fig 6 to a local .npz file.

This machine has the GPU, the /Volumes/Evo mel caches, and the trained TFLite models, but
NOT real Arial/Helvetica fonts (DejaVu fallback only) — so the figure itself is rendered on
Mac in part 2, which has Arial but not the mel caches.

Output:
  analysis/fig6_data.npz  — micro_probs, micro_labels, edge_probs, edge_labels, taus
  (copy/sync this single file to the Mac; it's small, no caches or models needed there)

Usage:
  python generate_fig6_part1_extract.py \
      --micro-cache /Volumes/Evo/cache4arxiv_fft1024_m16 \
      --edge-cache  /Volumes/Evo/cache4arxiv_fft1024_m80 \
      --results-dir results4arxiv \
      [--out analysis/fig6_data.npz]
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MICRO_TAU   = 0.35   # 5-seed mean-basis operating threshold
EDGE_TAU    = 0.425  # 5-seed mean-basis: single threshold clearing >=0.99 mean recall
MICRO_SEEDS = [42, 100, 786, 7, 1234]
EDGE_SEED   = 42

DRONGONET_DIR = Path(__file__).parent.parent
RESULTS_DIR   = DRONGONET_DIR / 'results4arxiv'
OUT_DEFAULT   = Path(__file__).parent / 'fig6_data.npz'


def run_tflite_inference(tflite_path: Path, mels: np.ndarray):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale,  in_zp  = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    mels_4d = mels[..., np.newaxis]
    probs, times = [], []
    for sample in mels_4d:
        x = sample[np.newaxis]
        if in_scale != 0.0:
            x = np.round(x / in_scale + in_zp).astype(inp_d['dtype'])
        else:
            x = x.astype(inp_d['dtype'])
        t0 = time.perf_counter()
        interp.set_tensor(inp_d['index'], x)
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])
        times.append((time.perf_counter() - t0) * 1000)
        if out_scale != 0.0:
            out = (raw.astype(np.float32) - out_zp) * out_scale
        else:
            out = raw.astype(np.float32)
        probs.append(float(out[0, 1]))

    return np.array(probs), float(np.mean(times))


def load_mels(cache_dir: Path):
    f = cache_dir / 'test' / 'mels.npz'
    if not f.exists():
        raise FileNotFoundError(f"Mel cache not found: {f}")
    d = np.load(f)
    log.info(f"Loaded {len(d['mels'])} test samples from {f}")
    return d['mels'].astype(np.float32), d['labels'].astype(np.int32)


def extract(micro_cache: Path, edge_cache: Path, results_dir: Path):
    micro_mels, micro_labels = load_mels(micro_cache)
    edge_mels,  edge_labels  = load_mels(edge_cache)

    micro_probs_per_seed = []
    for seed in MICRO_SEEDS:
        path = results_dir / f'6b_micro_final_fft1024_m16_s{seed}' / 'model_int8.tflite'
        log.info(f"Running Micro seed {seed}: {path}")
        probs, ms = run_tflite_inference(path, micro_mels)
        micro_probs_per_seed.append(probs)
        log.info(f"  seed {seed}: avg={ms:.3f}ms")

    edge_path = results_dir / f'6c_edge_final_fft1024_m80_s{EDGE_SEED}' / 'model_int8.tflite'
    log.info(f"Running Edge seed {EDGE_SEED}: {edge_path}")
    edge_probs, edge_ms = run_tflite_inference(edge_path, edge_mels)
    log.info(f"  Edge: avg={edge_ms:.3f}ms")

    return {
        'micro_probs':  np.mean(micro_probs_per_seed, axis=0),
        'micro_labels': micro_labels,
        'micro_tau':    MICRO_TAU,
        'edge_probs':   edge_probs,
        'edge_labels':  edge_labels,
        'edge_tau':     EDGE_TAU,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--micro-cache', required=True,
                   help='Cache dir for Micro mel spectrograms (n_mels=16)')
    p.add_argument('--edge-cache', required=True,
                   help='Cache dir for Edge mel spectrograms (n_mels=80)')
    p.add_argument('--results-dir', default=str(RESULTS_DIR),
                   help='Parent dir containing 6b_micro_final_*/model.tflite etc.')
    p.add_argument('--out', default=str(OUT_DEFAULT),
                   help='Output .npz path (copy this file to the Mac for part 2)')
    return p.parse_args()


def main():
    args = parse_args()
    data = extract(
        micro_cache=Path(args.micro_cache),
        edge_cache=Path(args.edge_cache),
        results_dir=Path(args.results_dir),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    log.info(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")
    print(f"\nDone. Copy this file to the Mac, then run:\n"
          f"  python generate_fig6_part2_plot.py --data {out.name}")


if __name__ == '__main__':
    main()
