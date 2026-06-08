#!/usr/bin/env python3
"""
generate_figures_from_sweep.py — Generate all publication figures from threshold_sweep.txt files.

Does NOT require mel caches or TFLite runtime — reads pre-saved sweep data.
Generates (filenames match PDF figure numbers):
  fig3_pareto.pdf              — Pareto front: AUC vs model size (Nano/Micro/Edge + baselines)
  fig4_threshold_analysis.pdf  — Recall/Precision/F1/Specificity vs tau + per-seed bars
  fig5_roc_pr_curves.pdf       — ROC (FPR/TPR) and PR curves from sweep data
  fig6_probability_distributions.pdf — NOTE: requires TFLite run, skipped here
  fig7_confusion_matrices.pdf  — Confusion matrices at τ/2, τ, 2τ for each model

Usage:
  python generate_figures_from_sweep.py [--out-dir images/] [--fmt pdf]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = Path(__file__).parent.parent / 'results4arxiv'
IMAGES_DIR  = Path(__file__).parent / 'images'

# Overridden by --results-dir and --out-dir at runtime
_RESULTS_DIR_OVERRIDE = None

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
    'nano':  '#1B9E77',
    'micro': '#2E86AB',
    'edge':  '#A23B72',
    'tau':   '#D62828',
    'gray':  '#6C757D',
    'base':  '#E67E22',
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_sweep(path: Path) -> dict:
    """Parse threshold_sweep.txt → dict of arrays keyed by column name."""
    lines = path.read_text().splitlines()
    # Find header line
    header_idx = next(i for i, l in enumerate(lines) if 'τ' in l or 'tau' in l.lower() and 'Recall' in l)
    data_lines = [l.strip() for l in lines[header_idx+2:] if l.strip() and not l.startswith('-')]
    rows = []
    for line in data_lines:
        # strip trailing comments like "<-- LOCKED"
        line = re.sub(r'<.*', '', line).strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) >= 8:
            rows.append([float(v) for v in vals[:8]])
    arr = np.array(rows)
    # Read AUC from header
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


def load_model_sweeps(name: str, seeds: list, fft: int, nmels: int,
                      results_dir: Path = None) -> list:
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
    """Stack per-seed arrays, return (mean, std) across seeds."""
    arr = np.stack([s[key] for s in sweeps], axis=0)
    return arr.mean(0), arr.std(0)


# ---------------------------------------------------------------------------
# FIG 1 — Threshold analysis
# ---------------------------------------------------------------------------

def fig1_threshold_analysis(models: dict, out: Path, dpi: int):
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(7.0, 7.5))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)
        ax_rp  = fig.add_subplot(gs[0, :])
        ax_f   = fig.add_subplot(gs[1, 0])
        ax_sp  = fig.add_subplot(gs[1, 1])
        ax_bar = fig.add_subplot(gs[2, :])

        for mname, md in models.items():
            sweeps   = md['sweeps']
            color    = md['color']
            lbl      = md['label']
            tau_op   = md['tau_op']
            tau_ops  = md.get('tau_ops', None)  # per-seed dict if present
            taus     = sweeps[0]['tau']

            rec_m,  rec_s  = stack_sweeps(sweeps, 'recall')
            prec_m, prec_s = stack_sweeps(sweeps, 'precision')
            f1_m,   f1_s   = stack_sweeps(sweeps, 'f1')
            f2_m,   f2_s   = stack_sweeps(sweeps, 'f2')
            sp_m,   sp_s   = stack_sweeps(sweeps, 'specificity')

            ax_rp.plot(taus, rec_m,  color=color, ls='-',  label=f'{lbl} recall')
            ax_rp.fill_between(taus, rec_m - rec_s, rec_m + rec_s, color=color, alpha=0.12)
            ax_rp.plot(taus, prec_m, color=color, ls='--', label=f'{lbl} prec.')
            ax_rp.fill_between(taus, prec_m - prec_s, prec_m + prec_s, color=color, alpha=0.08)

            ax_f.plot(taus, f1_m, color=color, ls='-',  label=f'{lbl} F1')
            ax_f.fill_between(taus, f1_m - f1_s, f1_m + f1_s, color=color, alpha=0.12)
            ax_f.plot(taus, f2_m, color=color, ls=':',  label=f'{lbl} F2')

            ax_sp.plot(taus, sp_m, color=color, label=lbl)
            ax_sp.fill_between(taus, sp_m - sp_s, sp_m + sp_s, color=color, alpha=0.12)

            for ax in (ax_rp, ax_f, ax_sp):
                if tau_ops is not None:
                    # Per-seed τ: draw range as shaded band + individual thin lines
                    tau_vals = sorted(tau_ops.values())
                    ax.axvspan(tau_vals[0], tau_vals[-1], color=color, alpha=0.07, lw=0)
                    for tv in tau_vals:
                        ax.axvline(tv, color=color, lw=0.6, ls=':', alpha=0.55)
                else:
                    ax.axvline(tau_op, color=color, lw=0.9, ls=':', alpha=0.7)

        # Recall targets
        ax_rp.axhline(0.98, color=COLORS['micro'], lw=0.6, ls=':', alpha=0.4)
        ax_rp.axhline(0.99, color=COLORS['edge'],  lw=0.6, ls=':', alpha=0.4)

        ax_rp.set_xlabel('Decision threshold τ'); ax_rp.set_ylabel('Score')
        ax_rp.set_title('A  Recall (solid) and Precision (dashed) vs threshold', loc='left', fontweight='bold')
        ax_rp.set_xlim(0, 1); ax_rp.set_ylim(0, 1.02)
        ax_rp.grid(True, alpha=0.25, ls=':')
        ax_rp.legend(fontsize=6.5, ncol=3)

        ax_f.set_xlabel('τ'); ax_f.set_ylabel('Score')
        ax_f.set_title('B  F1 (solid) and F2 (:) vs threshold', loc='left', fontweight='bold')
        ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1.02)
        ax_f.grid(True, alpha=0.25, ls=':')
        ax_f.legend(fontsize=6.5)

        ax_sp.set_xlabel('τ'); ax_sp.set_ylabel('Specificity')
        ax_sp.set_title('C  Specificity vs threshold', loc='left', fontweight='bold')
        ax_sp.set_xlim(0, 1); ax_sp.set_ylim(0, 1.02)
        ax_sp.grid(True, alpha=0.25, ls=':')
        ax_sp.legend(fontsize=6.5)

        # Panel D — per-seed recall bars at operating τ
        all_labels  = []
        all_recalls = []
        all_colors  = []
        for mname, md in models.items():
            tau_ops = md.get('tau_ops', None)
            for sw, seed in zip(md['sweeps'], md['seeds']):
                t = tau_ops[seed] if tau_ops else md['tau_op']
                idx = np.argmin(np.abs(sw['tau'] - t))
                all_recalls.append(sw['recall'][idx])
                all_labels.append(f"{md['label']}\nτ={t}")
                all_colors.append(md['color'])

        xs     = np.arange(len(all_recalls))
        ax_bar.bar(xs, all_recalls, color=all_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax_bar.axhline(0.98, color=COLORS['micro'], lw=0.8, ls='--', alpha=0.7, label='0.98 target (Nano/Micro)')
        ax_bar.axhline(0.99, color=COLORS['edge'],  lw=0.8, ls='--', alpha=0.7, label='0.99 target (Edge)')

        # x-tick labels: seed numbers per model
        xtick_labels = []
        for mname, md in models.items():
            for seed in md['seeds']:
                xtick_labels.append(f's{seed}')
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(xtick_labels, fontsize=7)

        # Add model group labels
        pos = 0
        for mname, md in models.items():
            n = len(md['seeds'])
            ax_bar.annotate(md['label'], xy=(pos + n/2 - 0.5, 0.952),
                            fontsize=7, ha='center', color=md['color'])
            pos += n

        ax_bar.set_ylabel('Recall')
        ax_bar.set_title('D  Per-seed recall at operating threshold', loc='left', fontweight='bold')
        ax_bar.set_ylim(0.95, 1.007)
        ax_bar.legend(fontsize=7, ncol=2)
        ax_bar.grid(True, alpha=0.25, ls=':', axis='y')

        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIG 2 — ROC and PR curves (from sweep data)
# ---------------------------------------------------------------------------

def fig2_roc_pr(models: dict, out: Path, dpi: int):
    """Approximate ROC/PR from threshold sweep (FPR, recall, precision at each τ)."""
    with plt.rc_context(FIGURE_STYLE):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.0, 3.5))

        for mname, md in models.items():
            sweeps = md['sweeps']
            color  = md['color']
            lbl    = md['label']
            tau_op = md['tau_op']

            # Mean curves across seeds
            fpr_m,  _  = stack_sweeps(sweeps, 'fpr')
            rec_m,  _  = stack_sweeps(sweeps, 'recall')
            prec_m, _  = stack_sweeps(sweeps, 'precision')
            auc_vals   = [s['auc'] for s in sweeps]
            auc_mean   = np.mean(auc_vals)
            auc_std    = np.std(auc_vals)

            # ROC: sort by fpr ascending (as τ decreases)
            sort_idx = np.argsort(fpr_m)
            fpr_s  = np.concatenate([[0], fpr_m[sort_idx], [1]])
            tpr_s  = np.concatenate([[0], rec_m[sort_idx], [1]])
            ax_roc.plot(fpr_s, tpr_s, color=color,
                        label=f'{lbl}\n(AUC={auc_mean:.4f}±{auc_std:.4f})')

            # Mark operating point(s) on mean curve
            tau_arr  = sweeps[0]['tau']
            tau_ops  = md.get('tau_ops', None)
            sort_idx2 = np.argsort(rec_m)
            rec_ps   = np.concatenate([[0], rec_m[sort_idx2], [1]])
            prec_ps  = np.concatenate([[1], prec_m[sort_idx2], [0]])
            ax_pr.plot(rec_ps, prec_ps, color=color, label=lbl)

            if tau_ops is not None:
                # Show each per-seed operating point on the mean curve
                op_label_done = False
                for seed_i, (seed, t) in enumerate(sorted(tau_ops.items())):
                    idx_op = np.argmin(np.abs(tau_arr - t))
                    lbl_op = f'  τ={list(tau_ops.values())[0]}–{list(tau_ops.values())[-1]}' if not op_label_done else None
                    ax_roc.scatter(fpr_m[idx_op], rec_m[idx_op], s=40, color=color,
                                   zorder=5, edgecolors='white', linewidths=0.8,
                                   marker=['o', 's', '^'][seed_i],
                                   label=lbl_op)
                    ax_pr.scatter(rec_m[idx_op], prec_m[idx_op], s=40, color=color,
                                  zorder=5, edgecolors='white', linewidths=0.8,
                                  marker=['o', 's', '^'][seed_i])
                    op_label_done = True
            else:
                idx_op = np.argmin(np.abs(tau_arr - tau_op))
                ax_roc.scatter(fpr_m[idx_op], rec_m[idx_op], s=55, color=color,
                               zorder=5, edgecolors='white', linewidths=0.8,
                               label=f'  τ={tau_op}')
                ax_pr.scatter(rec_m[idx_op], prec_m[idx_op], s=55, color=color,
                              zorder=5, edgecolors='white', linewidths=0.8)

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('A  ROC Curve', loc='left', fontweight='bold')
        ax_roc.legend(fontsize=6.5)
        ax_roc.set_xlim(-0.01, 1.01); ax_roc.set_ylim(-0.01, 1.01)
        ax_roc.set_aspect('equal')
        ax_roc.grid(True, alpha=0.25, ls=':')

        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_title('B  Precision-Recall Curve', loc='left', fontweight='bold')
        ax_pr.legend(fontsize=6.5)
        ax_pr.set_xlim(-0.01, 1.01); ax_pr.set_ylim(-0.01, 1.01)
        ax_pr.grid(True, alpha=0.25, ls=':')

        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIG 4 — Confusion matrices
# ---------------------------------------------------------------------------

def fig4_confusion_matrices(models: dict, out: Path, dpi: int):
    var_list = list(models.items())
    nrow, ncol = len(var_list), 3

    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.5 * nrow))

        for row, (mname, md) in enumerate(var_list):
            # For per-seed τ models, use the seed-42 (highest) τ as the reference column
            tau_ops = md.get('tau_ops', None)
            tau_op  = tau_ops[42] if tau_ops else md['tau_op']
            sweeps  = md['sweeps']
            lbl     = md['label']

            # Use mean of all seeds
            tau_arr = sweeps[0]['tau']
            tp_m, _ = stack_sweeps(sweeps, 'tp')
            fp_m, _ = stack_sweeps(sweeps, 'fp')
            fn_m, _ = stack_sweeps(sweeps, 'fn')
            # TN = total_neg - FP; need total
            total_pos = tp_m[0] + fn_m[0]  # at tau=0.05 approx all pos predicted
            # Better: use TP+FN at lowest tau (most permissive)
            total_pos_true = sweeps[0]['tp'][0] + sweeps[0]['fn'][0]
            total_neg_true = sweeps[0]['fp'][0] + (sweeps[0]['tn'][0] if sweeps[0]['tn'] is not None else 0)
            total = total_pos_true + total_neg_true

            taus_show = [tau_op * 0.5, tau_op, min(tau_op * 2, 0.90)]
            taus_show = [round(t, 2) for t in taus_show]

            for col, tau_show in enumerate(taus_show):
                ax  = axes[row, col] if nrow > 1 else axes[col]
                idx = np.argmin(np.abs(tau_arr - tau_show))

                # Mean counts
                tp_val = tp_m[idx]
                fp_val = fp_m[idx]
                fn_val = fn_m[idx]
                tn_val = total - tp_val - fp_val - fn_val

                cm = np.array([[tn_val, fp_val], [fn_val, tp_val]])
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_n = cm / (row_sums + 1e-9)

                highlight = (col == 1)
                cmap = 'Oranges' if highlight else 'Blues'
                ax.imshow(cm_n, vmin=0, vmax=1, cmap=cmap, aspect='equal')

                labels = [['TN', 'FP'], ['FN', 'TP']]
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i,
                                f'{labels[i][j]}\n{cm[i,j]:,.0f}\n({cm_n[i,j]:.1%})',
                                ha='center', va='center', fontsize=7,
                                color='white' if cm_n[i,j] > 0.55 else 'black')

                title = f'tau={tau_arr[idx]:.2f}{"  *" if highlight else ""}'
                ax.set_title(title, fontsize=8,
                             fontweight='bold' if highlight else 'normal')
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(['Neg', 'Pos'], fontsize=7)
                ax.set_yticklabels(['Neg', 'Pos'], fontsize=7)
                if col == 0:
                    ax.set_ylabel(lbl, fontsize=8)
                if row == nrow - 1:
                    ax.set_xlabel('Predicted', fontsize=7)

        fig.suptitle('Confusion matrices (mean over seeds); * = operating threshold',
                     fontsize=8, y=1.01)
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# FIG Pareto — Size vs AUC
# ---------------------------------------------------------------------------

def fig_pareto(out: Path, dpi: int):
    """Pareto plot: model size (kB) vs AUC. Shows SEABADNet family vs reference points."""

    # SEABADNet family (3-seed mean±std, INT8)
    seabadnet = [
        {'name': 'SEABADNet-Nano\n(512-FFT)',  'size': 5.41,  'auc': 0.9715, 'auc_std': 0.0013, 'color': COLORS['nano'],  'marker': 'o'},
        {'name': 'SEABADNet-Micro\n(1024-FFT)', 'size': 6.56,  'auc': 0.9743, 'auc_std': 0.0011, 'color': COLORS['micro'], 'marker': 'o'},
        {'name': 'SEABADNet-Edge\n(1024-FFT)', 'size': 33.06, 'auc': 0.9992, 'auc_std': 0.0002, 'color': COLORS['edge'],  'marker': 'o'},
    ]

    # Baseline TinyChirp (3-seed mean, from paper Table 1)
    baseline = [
        {'name': 'TinyChirp\n(Baseline)',   'size': 7.30,   'auc': 0.9706, 'auc_std': 0.0011, 'color': COLORS['base'], 'marker': 's'},
    ]

    # Standard CNNs fine-tuned on SEABAD (224x224 mel input, 3-seed mean±std from validation/results)
    standard = [
        {'name': 'MobileNetV3-S',  'size': 1122,  'auc': 0.9985, 'auc_std': 0.0002, 'color': COLORS['gray'], 'marker': '^'},
        {'name': 'EfficientNet-B0', 'size': 4416,  'auc': 0.9991, 'auc_std': 0.0004, 'color': COLORS['gray'], 'marker': 'D'},
        {'name': 'ResNet-50',       'size': 24153, 'auc': 0.9992, 'auc_std': 0.0003, 'color': COLORS['gray'], 'marker': 'v'},
        {'name': 'VGG-16',          'size': 14881, 'auc': 0.9995, 'auc_std': 0.0001, 'color': COLORS['gray'], 'marker': 's'},
    ]

    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        # Plot standard CNNs first (background)
        std_offsets = [(0, 6), (5, -10), (0, 6), (-5, 6)]
        std_haligns = ['center', 'left', 'center', 'right']
        for pt, off, ha in zip(standard, std_offsets, std_haligns):
            ax.scatter(pt['size'], pt['auc'], s=70, color=pt['color'],
                       marker=pt['marker'], zorder=3, alpha=0.7,
                       edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'],
                            fmt='none', color=pt['color'], capsize=2, lw=0.7, alpha=0.5, zorder=2)
            ax.annotate(pt['name'], (pt['size'], pt['auc']),
                        textcoords='offset points', xytext=off,
                        fontsize=6.5, ha=ha, color=pt['color'])

        # Plot baseline
        for pt in baseline:
            ax.scatter(pt['size'], pt['auc'],
                       s=90, color=pt['color'], marker=pt['marker'],
                       zorder=4, edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'],
                            fmt='none', color=pt['color'], capsize=3, lw=0.8, zorder=3)
            ax.annotate(pt['name'], (pt['size'], pt['auc']),
                        textcoords='offset points', xytext=(5, -10),
                        fontsize=6.5, ha='left', color=pt['color'])

        # Plot SEABADNet family with Pareto front
        sb_sizes = [pt['size'] for pt in seabadnet]
        sb_aucs  = [pt['auc']  for pt in seabadnet]
        ax.plot(sb_sizes, sb_aucs, color=COLORS['micro'],
                lw=1.0, ls='--', alpha=0.5, zorder=2)

        for pt in seabadnet:
            ax.scatter(pt['size'], pt['auc'],
                       s=100, color=pt['color'], marker=pt['marker'],
                       zorder=5, edgecolors='white', linewidths=0.8)
            if pt['auc_std']:
                ax.errorbar(pt['size'], pt['auc'], yerr=pt['auc_std'],
                            fmt='none', color=pt['color'], capsize=3, lw=1.0, zorder=4)

        # Offset labels for SEABADNet
        offsets = [(-5, 8), (5, 8), (0, 8)]
        haligns = ['right', 'left', 'center']
        for pt, off, ha in zip(seabadnet, offsets, haligns):
            ax.annotate(pt['name'], (pt['size'], pt['auc']),
                        textcoords='offset points', xytext=off,
                        fontsize=7, ha=ha, color=pt['color'], fontweight='bold')

        ax.set_xscale('log')
        ax.set_xlabel('INT8 Model Size (kB, log scale)')
        ax.set_ylabel('AUC (mean ± std, 3 seeds)')
        ax.set_title('Size–Accuracy Trade-off: SEABADNet Family vs Reference Models',
                     fontweight='bold')
        ax.set_xlim(3, 40000)
        ax.set_ylim(0.966, 1.0025)
        ax.grid(True, alpha=0.25, ls=':', which='both')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))

        # MCU flash budget lines
        ax.axvline(8,   color='#aaaaaa', lw=0.7, ls=':')
        ax.axvline(64,  color='#aaaaaa', lw=0.7, ls=':')
        ax.axvline(256, color='#aaaaaa', lw=0.7, ls=':')
        ax.text(8,   0.9665, '8 kB',   ha='center', fontsize=6.5, color='#888888')
        ax.text(64,  0.9665, '64 kB',  ha='center', fontsize=6.5, color='#888888')
        ax.text(256, 0.9665, '256 kB', ha='center', fontsize=6.5, color='#888888')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='SEABADNet family (INT8, 184-frame mel)',
                   markerfacecolor=COLORS['micro'], markersize=8),
            Line2D([0], [0], marker='s', color='w', label='TinyChirp baseline (INT8)',
                   markerfacecolor=COLORS['base'], markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Standard CNN (fine-tuned on SEABAD, 224x224 mel)',
                   markerfacecolor=COLORS['gray'], markersize=8),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=None,
                   help='Path to seabadnet/results dir (default: ../Conda/seabadnet/results)')
    p.add_argument('--out-dir', default=str(IMAGES_DIR),
                   help='Output directory for figures (default: images/)')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--fmt', default='pdf', choices=['pdf', 'png', 'svg'])
    return p.parse_args()


def main():
    import matplotlib.ticker
    args = parse_args()
    out_dir     = Path(args.out_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_dir:
        print(f"Results dir: {results_dir}")
    else:
        print(f"Results dir: {RESULTS_DIR} (default)")

    # Load sweep data
    print("Loading sweep data...")
    nano_sweeps  = load_model_sweeps('6a_nano_final',  [42, 100, 786], fft=512, nmels=16, results_dir=results_dir)
    micro_sweeps = load_model_sweeps('6b_micro_final', [42, 100, 786], fft=1024, nmels=16, results_dir=results_dir)
    edge_sweeps  = load_model_sweeps('6c_edge_final',  [42, 100, 786], fft=1024, nmels=80, results_dir=results_dir)

    print(f"  Nano:  {len(nano_sweeps)} seeds loaded")
    print(f"  Micro: {len(micro_sweeps)} seeds loaded")
    print(f"  Edge:  {len(edge_sweeps)} seeds loaded")

    # Check Nano sweeps availability
    if len(nano_sweeps) == 0:
        print("  NOTE: Nano sweep files not found in results/ — run threshold sweep first")
        print("        Figures will be generated for Micro and Edge only")

    # Model registry — skip models with no data
    models = {}
    if len(micro_sweeps) > 0:
        models['micro'] = {
            'sweeps': micro_sweeps, 'seeds': [42, 100, 786],
            'tau_op': 0.35, 'color': COLORS['micro'],
            'label': 'SEABADNet-Micro',
        }
    if len(edge_sweeps) > 0:
        models['edge'] = {
            'sweeps': edge_sweeps, 'seeds': [42, 100, 786],
            'tau_op': 0.55,           # approx mean τ for single-line annotations
            'tau_ops': {42: 0.60, 100: 0.55, 786: 0.45},  # per-seed locked τ (Linux x86-64)
            'color': COLORS['edge'],
            'label': 'SEABADNet-Edge',
        }
    if len(nano_sweeps) > 0:
        # Nano τ TBD — use 0.35 as placeholder same as Micro
        models['nano'] = {
            'sweeps': nano_sweeps, 'seeds': [42, 100, 786],
            'tau_op': 0.35, 'color': COLORS['nano'],
            'label': 'SEABADNet-Nano',
        }
        # Insert nano first (smallest model)
        models = {'nano': models.pop('nano'), **models}

    if not models:
        print("ERROR: No sweep data found. Check RESULTS_DIR.")
        return

    ext = args.fmt
    print("\nGenerating figures...")
    fig1_threshold_analysis(models, out_dir / f'fig4_threshold_analysis.{ext}', args.dpi)
    fig2_roc_pr(            models, out_dir / f'fig5_roc_pr_curves.{ext}',      args.dpi)
    fig4_confusion_matrices(models, out_dir / f'fig7_confusion_matrices.{ext}', args.dpi)
    fig_pareto(                     out_dir / f'fig3_pareto.{ext}',             args.dpi)
    print("\nDone.")
    print(f"Figures written to: {out_dir}")
    print("\nNote: fig3_probability_distributions.pdf requires TFLite inference + mel caches.")
    print("      Run generate_seabadnet_figures.py on Linux/GPU machine for fig3.")


if __name__ == '__main__':
    import matplotlib.ticker
    main()
