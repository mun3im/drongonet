#!/usr/bin/env python3
"""
generate_fig6_part2_plot.py — Part 2 (MAC ONLY): render Fig 6 from the cached probabilities
produced by part 1 (generate_fig6_part1_extract.py on Linux).

This machine has real Arial/Helvetica (matching the house style already used in Fig 1-5) but
no /Volumes/Evo mel caches or GPU. Requires no TensorFlow, no mel caches — just numpy/matplotlib
and the .npz file copied over from Linux.

Output:
  images/fig6_probability_distributions.pdf

Usage:
  python generate_fig6_part2_plot.py --data fig6_data.npz [--out-dir images/] [--fmt pdf]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).parent / 'images'

FIGURE_STYLE = {
    'font.family':     'sans-serif',
    # Real Arial/Helvetica first; metric-compatible FOSS equivalents next
    # (Liberation Sans = Arial metrics, Nimbus Sans = Helvetica, Arimo = Arial);
    # DejaVu only as a last resort so a missing font is never silent.
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans',
                        'Arimo', 'DejaVu Sans'],
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

# House style enforcement: Arial/Helvetica metric family, no font larger than 12 pt.
MAX_FONT_PT = 12
assert all(v <= MAX_FONT_PT for k, v in FIGURE_STYLE.items()
           if k.endswith('size')), \
    f"Figure font sizes must be <= {MAX_FONT_PT} pt: {FIGURE_STYLE}"


def assert_arial_metric_font():
    """Hard-fail if no Arial/Helvetica-metric font is available — this script's whole purpose
    is to produce the Arial-accurate version of Fig 6, matching Fig 1-5."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [n for n in FIGURE_STYLE['font.sans-serif'] if n != 'DejaVu Sans']
    if not any(n in available for n in preferred):
        raise RuntimeError(
            "No Arial/Helvetica-metric font found (Arial, Helvetica, Liberation Sans, "
            "Nimbus Sans, Arimo). This script is meant to run where Arial is installed "
            "(e.g. macOS) to match Fig 1-5's font. Found fonts: "
            f"{sorted(available)[:20]}...")


assert_arial_metric_font()

COLORS = {
    'micro': '#2E86AB',
    'edge':  '#A23B72',
    'tau':   '#D62828',
    'gray':  '#6C757D',
}


def load_data(npz_path: Path):
    d = np.load(npz_path)
    return {
        'micro': {
            'mean_probs': d['micro_probs'],
            'labels':     d['micro_labels'],
            'tau':        float(d['micro_tau']),
            'label':      'DrongoNet-Micro',
            'color':      COLORS['micro'],
        },
        'edge': {
            'mean_probs': d['edge_probs'],
            'labels':     d['edge_labels'],
            'tau':        float(d['edge_tau']),
            'label':      'DrongoNet-Edge',
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
    p.add_argument('--data', required=True,
                   help='Path to fig6_data.npz produced by part 1 on Linux')
    p.add_argument('--out-dir', default=str(IMAGES_DIR))
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--fmt', default='pdf', choices=['pdf', 'png', 'svg'])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.data))
    out = out_dir / f'fig6_probability_distributions.{args.fmt}'
    generate_fig6(data, out, args.dpi)
    print(f"\nDone. Written to: {out}")


if __name__ == '__main__':
    main()
