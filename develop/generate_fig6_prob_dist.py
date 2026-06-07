#!/usr/bin/env python3
"""
generate_fig6_prob_dist.py — Generate Fig 6: Predicted probability distributions.

Requires TFLite runtime and mel caches (Linux/GPU machine with Evo drive).

Models:
  Micro: ../Conda/seabadnet/results/6b_micro_final_fft1024_m16_s{42,100,786}/model.tflite
  Edge:  ../Conda/seabadnet/results/6c_edge_final_fft1024_m80_s42/model_int8.tflite

Caches:
  Micro: <micro-cache>/test/mels.npz   (n_mels=16)
  Edge:  <edge-cache>/test/mels.npz    (n_mels=80)

Output:
  images/fig6_probability_distributions.pdf

Usage:
  python generate_fig6_prob_dist.py \
      --micro-cache /data/cache_seabad_m16 \
      --edge-cache  /data/cache_seabad_m80 \
      [--results-dir ../Conda/seabadnet/results] \
      [--out-dir images/] \
      [--fmt pdf]
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MICRO_TAU   = 0.35
EDGE_TAU    = 0.40   # mean across seeds (0.50/0.40/0.30)
MICRO_SEEDS = [42, 100, 786]
EDGE_SEED   = 42

SEABADNET_DIR = Path(__file__).parent.parent / 'Conda' / 'seabadnet'
RESULTS_DIR   = SEABADNET_DIR / 'results'
IMAGES_DIR    = Path(__file__).parent / 'images'

FIGURE_STYLE = {
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       9,
    'axes.labelsize':  10,
    'axes.titlesize':  10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth':  0.8,
    'grid.linewidth':  0.4,
    'lines.linewidth': 1.4,
}

COLORS = {
    'micro': '#2E86AB',
    'edge':  '#A23B72',
    'tau':   '#D62828',
    'gray':  '#6C757D',
}


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


def load_predictions(micro_cache: Path, edge_cache: Path, results_dir: Path):
    micro_mels, micro_labels = load_mels(micro_cache)
    edge_mels,  edge_labels  = load_mels(edge_cache)

    micro_probs_per_seed = []
    for seed in MICRO_SEEDS:
        path = results_dir / f'6b_micro_final_fft1024_m16_s{seed}' / 'model.tflite'
        log.info(f"Running Micro seed {seed}: {path}")
        probs, ms = run_tflite_inference(path, micro_mels)
        micro_probs_per_seed.append(probs)
        log.info(f"  seed {seed}: avg={ms:.3f}ms")

    edge_path = results_dir / f'6c_edge_final_fft1024_m80_s{EDGE_SEED}' / 'model_int8.tflite'
    log.info(f"Running Edge seed {EDGE_SEED}: {edge_path}")
    edge_probs, edge_ms = run_tflite_inference(edge_path, edge_mels)
    log.info(f"  Edge: avg={edge_ms:.3f}ms")

    return {
        'micro': {
            'mean_probs': np.mean(micro_probs_per_seed, axis=0),
            'labels':     micro_labels,
            'tau':        MICRO_TAU,
            'label':      'SEABADNet-Micro',
            'color':      COLORS['micro'],
        },
        'edge': {
            'mean_probs': edge_probs,
            'labels':     edge_labels,
            'tau':        EDGE_TAU,
            'label':      'SEABADNet-Edge',
            'color':      COLORS['edge'],
        },
    }


def generate_fig6(data: dict, out: Path, dpi: int):
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

        for (vname, vd), ax, panel in zip(data.items(), axes, ('A', 'B')):
            probs = vd['mean_probs']
            labs  = vd['labels']
            color = vd['color']
            lbl   = vd['label']
            tau   = vd['tau']

            neg  = probs[labs == 0]
            pos  = probs[labs == 1]
            bins = np.linspace(0, 1, 60)

            ax.hist(neg, bins=bins, alpha=0.6, color=COLORS['gray'],
                    density=True, label='Negative (true)')
            ax.hist(pos, bins=bins, alpha=0.6, color=color,
                    density=True, label='Positive (true)')
            ax.axvline(tau, color=COLORS['tau'], lw=1.2, ls='--',
                       label=f'τ={tau} (op.)')
            ax.set_xlabel('P(bird present)')
            ax.set_ylabel('Density')
            ax.set_title(f'{panel}  {lbl}', loc='left', fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25, ls=':', axis='y')

        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    log.info(f"Saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--micro-cache', required=True,
                   help='Cache dir for Micro mel spectrograms (n_mels=16)')
    p.add_argument('--edge-cache', required=True,
                   help='Cache dir for Edge mel spectrograms (n_mels=80)')
    p.add_argument('--results-dir', default=str(RESULTS_DIR),
                   help='Parent dir containing 6b_micro_final_*/model.tflite etc.')
    p.add_argument('--out-dir', default=str(IMAGES_DIR))
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--fmt', default='pdf', choices=['pdf', 'png', 'svg'])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir     = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading TFLite predictions...")
    data = load_predictions(
        micro_cache=Path(args.micro_cache),
        edge_cache=Path(args.edge_cache),
        results_dir=results_dir,
    )

    out = out_dir / f'fig6_probability_distributions.{args.fmt}'
    generate_fig6(data, out, args.dpi)
    print(f"\nDone. Written to: {out}")


if __name__ == '__main__':
    main()
