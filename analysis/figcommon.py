#!/usr/bin/env python3
"""
figcommon.py — shared style, data loading, and model registry for the per-figure scripts.

This is the SINGLE SOURCE OF TRUTH for seeds and operating thresholds. Each figure has its
own runnable script (make_fig3_threshold_analysis.py, make_fig4_pareto.py, make_fig5_roc_pr.py,
make_fig7_confusion.py) that imports from here and writes exactly one PDF, so you can tweak and
regenerate a single figure without touching the others. generate_figures_from_sweep.py remains
as a convenience runner that calls all of them.

fig6 (probability distributions) is separate: generate_fig6_part1_extract.py (TFLite → npz) then
generate_fig6_part2_plot.py (npz → fig6 PDF).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ---- SINGLE SOURCE OF TRUTH: seeds + operating thresholds (5-seed mean-basis) ----
SEEDS = [42, 100, 786, 7, 1234]
TAU_OP = {'nano': 0.37, 'micro': 0.35, 'edge': 0.425}

RESULTS_DIR = Path(__file__).parent.parent / 'results4arxiv'
IMAGES_DIR = Path(__file__).parent / 'images'

# model name -> (result-dir prefix, n_fft, n_mels)
SPEC = {
    'nano':  ('6a_nano_final',  512,  16),
    'micro': ('6b_micro_final', 1024, 16),
    'edge':  ('6c_edge_final',  1024, 80),
}

FIGURE_STYLE = {
    'font.family':     'sans-serif',
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

MAX_FONT_PT = 12
assert all(v <= MAX_FONT_PT for k, v in FIGURE_STYLE.items() if k.endswith('size')), \
    f"Figure font sizes must be <= {MAX_FONT_PT} pt: {FIGURE_STYLE}"

COLORS = {
    'nano':  '#1B9E77',
    'micro': '#2E86AB',
    'edge':  '#A23B72',
    'tau':   '#D62828',
    'gray':  '#6C757D',
    'base':  '#E67E22',
}


def assert_arial_metric_font():
    """Warn loudly if no Arial/Helvetica-metric font is available (would fall back to DejaVu)."""
    import warnings
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [n for n in FIGURE_STYLE['font.sans-serif'] if n != 'DejaVu Sans']
    if not any(n in available for n in preferred):
        warnings.warn(
            "No Arial/Helvetica-metric font found (Arial, Helvetica, Liberation Sans, "
            "Nimbus Sans, Arimo). Figures would render in DejaVu Sans. Install e.g. "
            "fonts-liberation / fonts-croscore.", RuntimeWarning)


def parse_sweep(path: Path) -> dict:
    """Parse threshold_sweep.txt -> dict of arrays keyed by column name."""
    lines = path.read_text().splitlines()
    header_idx = next(i for i, l in enumerate(lines)
                      if 'τ' in l or 'tau' in l.lower() and 'Recall' in l)
    data_lines = [l.strip() for l in lines[header_idx + 2:] if l.strip() and not l.startswith('-')]
    rows = []
    for line in data_lines:
        line = re.sub(r'<.*', '', line).strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) >= 8:
            rows.append([float(v) for v in vals[:8]])
    arr = np.array(rows)
    auc = None
    for l in lines:
        m = re.search(r'AUC\s*[:\s]+([0-9.]+)', l)
        if m:
            auc = float(m.group(1))
            break
    return {
        'tau':       arr[:, 0],
        'recall':    arr[:, 1],
        'precision': arr[:, 2],
        'f1':        arr[:, 3],
        'fpr':       arr[:, 4],
        'tp':        arr[:, 5],
        'fp':        arr[:, 6],
        'fn':        arr[:, 7],
        'tn':        arr[:, 8] if arr.shape[1] > 8 else None,
        'auc':       auc,
        'specificity': 1.0 - arr[:, 4],
        'f2':        5 * arr[:, 2] * arr[:, 1] / (4 * arr[:, 2] + arr[:, 1] + 1e-9),
    }


def load_model_sweeps(name: str, seeds: list, fft: int, nmels: int, results_dir: Path = None) -> list:
    base = results_dir if results_dir is not None else RESULTS_DIR
    sweeps = []
    for seed in seeds:
        p = base / f'{name}_fft{fft}_m{nmels}_s{seed}' / 'threshold_sweep.txt'
        if not p.exists():
            print(f"  WARNING: {p} not found")
            continue
        sweeps.append(parse_sweep(p))
    return sweeps


def stack_sweeps(sweeps: list, key: str) -> tuple:
    arr = np.stack([s[key] for s in sweeps], axis=0)
    return arr.mean(0), arr.std(0)


def load_models(results_dir: Path = None) -> dict:
    """Ordered {nano, micro, edge} registry with sweeps + seeds + tau_op + color + label."""
    models = {}
    for name in ('nano', 'micro', 'edge'):
        pre, fft, nm = SPEC[name]
        sw = load_model_sweeps(pre, SEEDS, fft, nm, results_dir)
        if not sw:
            print(f"  WARNING: no sweep data for {name}")
            continue
        models[name] = {
            'sweeps': sw, 'seeds': SEEDS, 'tau_op': TAU_OP[name],
            'color': COLORS[name], 'label': f'SEABADNet-{name.capitalize()}',
        }
    return models


def common_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--results-dir', default=None,
                   help='dir containing 6{a,b,c}_*_s{seed}/threshold_sweep.txt (default: ../results4arxiv)')
    p.add_argument('--out-dir', default=str(IMAGES_DIR), help='output directory for the figure')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--fmt', default='pdf', choices=['pdf', 'png', 'svg'])
    return p


def resolve_out(args) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
