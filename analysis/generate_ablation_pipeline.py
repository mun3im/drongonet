#!/usr/bin/env python3
"""
generate_ablation_pipeline.py — Ablation pipeline figure for SEABADNet paper

Horizontal flowchart, left-to-right: TinyChirp -> Ph1 -> ... -> SEABADNet-Micro
Edge branch drops DOWN from Ph.3 node.
Wide figure to fill \textwidth in the paper.

Output: images/fig_ablation_pipeline.pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

OUT = Path(__file__).parent / 'images' / 'fig_ablation_pipeline.pdf'

# ---------------------------------------------------------------------------
# Font — Arial (sans-serif)
# ---------------------------------------------------------------------------
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'pdf.fonttype': 42,   # embed TrueType — preserves full Unicode glyph coverage
    'ps.fonttype':  42,
}

C_BASELINE = '#DDEEFF'
C_PHASE    = '#EEF4DD'
C_MICRO    = '#D5E8D4'
C_EDGE     = '#DAE8FC'
C_BORDER   = '#555555'
C_ARROW    = '#444444'
C_EDGE_ARR = '#2255AA'
C_REJ      = '#AA2222'

# ---------------------------------------------------------------------------
# Layout — data coordinates, x increases rightward, y increases upward.
# Main chain: left→right along Y_MAIN.
# Edge box: below Ph.3 node at Y_EDGE.
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 13.0, 3.8   # wide, short — fills \textwidth

# 7 main-chain nodes evenly spaced across x
N      = 7
X_L    = 1.2    # leftmost node centre
X_R    = 19.0   # rightmost node centre
XS     = [X_L + i * (X_R - X_L) / (N - 1) for i in range(N)]
# XS[0]=TinyChirp, XS[3]=Ph.3 (branch point), XS[6]=SEABADNet-Micro

Y_MAIN = 2.8    # y of main chain
Y_EDGE = 0.9    # y of Edge box (below main chain)

BH  = 0.52   # box half-height, main chain
BW  = 1.22   # box half-width, main chain (fits 7 boxes in X_L..X_R)
# Gap between adjacent main boxes: XS[1]-XS[0] - 2*BW
# spacing = (X_R-X_L)/(N-1) = ~2.97; gap = 2.97 - 2*1.22 = 0.53  ✓

BWE = 1.8    # half-width, Edge box
BHE = 0.78   # half-height, Edge box

TOTAL_W = 20.5
TOTAL_H = 4.5

MAIN_CHAIN = [
    # (bold text,                         subtext,                  color)
    ('TinyChirp-CNNMel',                 'Starting point',          C_BASELINE),
    ('+ n_mels = 16',                    'Freq. resolution',        C_PHASE),
    ('+ Global Avg. Pooling',            'Param. reduction',        C_PHASE),
    ('+ Focal loss\n+ Freq. emphasis',   'Loss & preprocessing',    C_PHASE),
    ('+ Dropout = 0.1',                  'Regularisation',          C_PHASE),
    ('+ 6 filters\n+ 1×1 conv',          'Capacity tuning',         C_PHASE),
    ('SEABADNet-Micro',
     '919 params · 6.56 kB\nτ=0.35 · recall ≥98.1%',      C_MICRO),
]

EDGE_X = XS[3]   # same x as Ph.3


def arrow_h(ax, x_from, x_to, y, color=C_ARROW):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.1, mutation_scale=11))


def arrow_v(ax, x, y_from, y_to, color=C_ARROW):
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.1, mutation_scale=11))


def draw_box(ax, xc, yc, hw, hh, line1, line2, color,
             fs_bold=7.5, fs_sub=6.2):
    ax.add_patch(FancyBboxPatch(
        (xc - hw, yc - hh), 2*hw, 2*hh,
        boxstyle='round,pad=0.04',
        linewidth=0.7, edgecolor=C_BORDER, facecolor=color, zorder=2))
    if line2:
        ax.text(xc, yc + hh * 0.28, line1,
                ha='center', va='center', fontsize=fs_bold,
                fontweight='bold', linespacing=1.3, zorder=3)
        ax.text(xc, yc - hh * 0.32, line2,
                ha='center', va='center', fontsize=fs_sub,
                color='#444444', linespacing=1.3, zorder=3)
    else:
        ax.text(xc, yc, line1,
                ha='center', va='center', fontsize=fs_bold,
                fontweight='bold', linespacing=1.3, zorder=3)


def main():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        ax.set_xlim(0, TOTAL_W)
        ax.set_ylim(0, TOTAL_H)
        ax.axis('off')

        # --- Main chain boxes and arrows ---
        for i, (xc, (t1, t2, col)) in enumerate(zip(XS, MAIN_CHAIN)):
            draw_box(ax, xc, Y_MAIN, BW, BH, t1, t2, col)
            if i > 0:
                arrow_h(ax, XS[i-1] + BW + 0.06, xc - BW - 0.06, Y_MAIN)

        # --- Phase labels above each box ---
        phase_labels = ['Ph.0', 'Ph.1', 'Ph.2', 'Ph.3', 'Ph.4', 'Ph.5', '']
        for xc, lbl in zip(XS, phase_labels):
            if lbl:
                ax.text(xc, Y_MAIN + BH + 0.14, lbl,
                        ha='center', va='bottom', fontsize=5.8, color='#999999')

        # --- Edge box (below Ph.3) ---
        ax.add_patch(FancyBboxPatch(
            (EDGE_X - BWE, Y_EDGE - BHE), 2*BWE, 2*BHE,
            boxstyle='round,pad=0.04',
            linewidth=0.7, edgecolor=C_BORDER, facecolor=C_EDGE, zorder=2))
        ax.text(EDGE_X, Y_EDGE + BHE * 0.48, 'SEABADNet-Edge',
                ha='center', va='center', fontsize=7.5,
                fontweight='bold', zorder=3)
        ax.text(EDGE_X, Y_EDGE + BHE * 0.05,
                '25,890 params · 32.82 kB',
                ha='center', va='center', fontsize=6.2,
                color='#444444', zorder=3)
        ax.text(EDGE_X, Y_EDGE - BHE * 0.33,
                'τ=0.50 · recall ≥99.0%',
                ha='center', va='center', fontsize=6.2,
                color='#444444', zorder=3)
        ax.text(EDGE_X, Y_EDGE - BHE * 0.70,
                'Scale up: n_mels=80, filters 8→16→64',
                ha='center', va='center', fontsize=5.8,
                color='#555555', style='italic', zorder=3)

        # --- Branch arrow: Ph.3 box bottom -> Edge box top ---
        arrow_v(ax, EDGE_X,
                Y_MAIN - BH - 0.06,
                Y_EDGE + BHE + 0.06,
                color=C_EDGE_ARR)
        ax.text(EDGE_X + 0.12, (Y_MAIN - BH + Y_EDGE + BHE) / 2,
                'scale up\n(Edge branch)',
                ha='left', va='center', fontsize=5.8,
                color=C_EDGE_ARR, style='italic')

        # --- Rejected note: below Ph.3 branch arrow, left side ---
        ax.text(EDGE_X - BWE - 0.5, Y_EDGE + BHE * 0.1,
                'Depthwise sep.\nrejected\n(hurts at this scale)',
                ha='right', va='center', fontsize=5.8,
                color=C_REJ, style='italic')
        ax.plot([EDGE_X - BWE - 0.45, EDGE_X - BWE - 0.05],
                [Y_EDGE, Y_EDGE],
                color=C_REJ, lw=0.6, ls='dashed')

        # --- Legend ---
        handles = [
            mpatches.Patch(facecolor=C_BASELINE, edgecolor=C_BORDER,
                           label='Starting point'),
            mpatches.Patch(facecolor=C_PHASE,    edgecolor=C_BORDER,
                           label='Ablation step'),
            mpatches.Patch(facecolor=C_MICRO,    edgecolor=C_BORDER,
                           label='SEABADNet-Micro (primary)'),
            mpatches.Patch(facecolor=C_EDGE,     edgecolor=C_BORDER,
                           label='SEABADNet-Edge (reference)'),
        ]
        ax.legend(handles=handles, loc='lower right',
                  bbox_to_anchor=(1.0, 0.0), ncol=1,
                  fontsize=6, framealpha=0.95, edgecolor=C_BORDER)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
