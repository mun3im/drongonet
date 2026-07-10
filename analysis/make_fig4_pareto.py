#!/usr/bin/env python3
"""make_fig4_pareto.py — generates ONLY fig4_pareto.pdf (size vs SEABAD AUC).

Self-contained data (DrongoNet 5-seed + baselines). Usage:
  python analysis/make_fig4_pareto.py [--out-dir images/] [--fmt pdf]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import figcommon as fc

# DrongoNet family (5-seed mean±std, full INT8) — seeds 42/100/786/7/1234
DRONGONET = [
    {'name': 'DrongoNet-Nano\n(512-FFT)',  'size': 5.09,  'auc': 0.9727, 'auc_std': 0.0016, 'color': fc.COLORS['nano'],  'marker': 'o'},
    {'name': 'DrongoNet-Micro\n(1024-FFT)', 'size': 6.26,  'auc': 0.9810, 'auc_std': 0.0016, 'color': fc.COLORS['micro'], 'marker': 'o'},
    {'name': 'DrongoNet-Edge\n(1024-FFT)', 'size': 33.06, 'auc': 0.9991, 'auc_std': 0.0002, 'color': fc.COLORS['edge'],  'marker': 'o'},
]
# TinyChirp CNN-Mel retrained from scratch on SEABAD (3-seed mean, paper Table)
BASELINE = [
    {'name': 'TinyChirp\n(Baseline)', 'size': 28.61, 'auc': 0.9815, 'auc_std': 0.0008, 'color': fc.COLORS['base'], 'marker': 's'},
]
# Standard CNNs fine-tuned on SEABAD (224x224 mel, 3-seed mean±std)
STANDARD = [
    {'name': 'MobileNetV3-S',   'size': 1122,  'auc': 0.9985, 'auc_std': 0.0002, 'color': fc.COLORS['gray'], 'marker': '^'},
    {'name': 'EfficientNet-B0', 'size': 4416,  'auc': 0.9991, 'auc_std': 0.0004, 'color': fc.COLORS['gray'], 'marker': 'D'},
    {'name': 'ResNet-50',       'size': 24153, 'auc': 0.9992, 'auc_std': 0.0003, 'color': fc.COLORS['gray'], 'marker': 'v'},
    {'name': 'VGG-16',          'size': 14881, 'auc': 0.9995, 'auc_std': 0.0001, 'color': fc.COLORS['gray'], 'marker': 's'},
]


def plot(out: Path, dpi: int):
    with plt.rc_context(fc.FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        std_offsets = [(0, -12), (0, 8), (0, -12), (0, 8)]
        for pt, off in zip(STANDARD, std_offsets):
            ax.scatter(pt['size'], pt['auc'], s=70, color=pt['color'], marker=pt['marker'],
                       zorder=3, alpha=0.7, edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'], fmt='none',
                            color=pt['color'], capsize=2, lw=0.7, alpha=0.5, zorder=2)
            ax.annotate(pt['name'], (pt['size'], pt['auc']), textcoords='offset points',
                        xytext=off, fontsize=6.5, ha='center', color=pt['color'])

        for pt in BASELINE:
            ax.scatter(pt['size'], pt['auc'], s=90, color=pt['color'], marker=pt['marker'],
                       zorder=4, edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'], fmt='none',
                            color=pt['color'], capsize=3, lw=0.8, zorder=3)
            ax.annotate(pt['name'], (pt['size'], pt['auc']), textcoords='offset points',
                        xytext=(10, -5), fontsize=6.5, ha='left', color=pt['color'])

        sb_sizes = [pt['size'] for pt in DRONGONET]
        sb_aucs = [pt['auc'] for pt in DRONGONET]
        ax.plot(sb_sizes, sb_aucs, color=fc.COLORS['micro'], lw=1.0, ls='--', alpha=0.5, zorder=2)
        for pt in DRONGONET:
            ax.scatter(pt['size'], pt['auc'], s=100, color=pt['color'], marker=pt['marker'],
                       zorder=5, edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'], fmt='none',
                            color=pt['color'], capsize=3, lw=1.0, zorder=4)
        offsets = [(8,-8), (6, -16), (8, -8)]
        for pt, off in zip(DRONGONET, offsets):
            ax.annotate(pt['name'], (pt['size'], pt['auc']), textcoords='offset points',
                        xytext=off, fontsize=7, ha='left', color=pt['color'], fontweight='bold')

        ax.set_xscale('log')
        ax.set_xlabel('INT8 Model Size (kB, log scale)')
        ax.set_ylabel('AUC (mean ± std, 5 seeds)')
        ax.set_title('Size–Accuracy Trade-off: DrongoNet Family vs Reference Models', fontweight='bold')
        ax.set_xlim(3, 40000)
        ax.set_ylim(0.966, 1.0025)
        ax.grid(True, alpha=0.25, ls=':', which='both')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))

        for xv, lbl in [(8, '8 kB'), (64, '64 kB'), (256, '256 kB')]:
            ax.axvline(xv, color='#aaaaaa', lw=0.7, ls=':')
            ax.text(xv, 0.9665, lbl, ha='center', fontsize=6.5, color='#888888')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='DrongoNet family (INT8, 184-frame mel)',
                   markerfacecolor=fc.COLORS['micro'], markersize=8),
            Line2D([0], [0], marker='s', color='w', label='TinyChirp baseline (INT8)',
                   markerfacecolor=fc.COLORS['base'], markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Standard CNN (fine-tuned on SEABAD, 224x224 mel)',
                   markerfacecolor=fc.COLORS['gray'], markersize=8),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


def main():
    fc.assert_arial_metric_font()
    args = fc.common_parser('Generate fig4_pareto').parse_args()
    out_dir = fc.resolve_out(args)
    plot(out_dir / f'fig4_pareto.{args.fmt}', args.dpi)


if __name__ == '__main__':
    main()
