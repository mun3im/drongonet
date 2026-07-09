#!/usr/bin/env python3
"""make_fig3_threshold_analysis.py — generates ONLY fig3_threshold_analysis.pdf.

Usage:
  python analysis/make_fig3_threshold_analysis.py [--out-dir images/] [--fmt pdf]
Seeds and operating thresholds come from figcommon.py (single source of truth).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
import figcommon as fc


def plot(models: dict, out: Path, dpi: int):
    with plt.rc_context(fc.FIGURE_STYLE):
        fig = plt.figure(figsize=(7.0, 7.5))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)
        ax_rp = fig.add_subplot(gs[0, :])
        ax_f = fig.add_subplot(gs[1, 0])
        ax_sp = fig.add_subplot(gs[1, 1])
        ax_bar = fig.add_subplot(gs[2, :])

        for mname, md in models.items():
            sweeps = md['sweeps']
            color = md['color']
            lbl = md['label']
            tau_op = md['tau_op']
            tau_ops = md.get('tau_ops', None)
            taus = sweeps[0]['tau']

            rec_m, rec_s = fc.stack_sweeps(sweeps, 'recall')
            prec_m, prec_s = fc.stack_sweeps(sweeps, 'precision')
            f1_m, f1_s = fc.stack_sweeps(sweeps, 'f1')
            f2_m, f2_s = fc.stack_sweeps(sweeps, 'f2')
            sp_m, sp_s = fc.stack_sweeps(sweeps, 'specificity')

            ax_rp.plot(taus, rec_m, color=color, ls='-', label=f'{lbl} recall')
            ax_rp.fill_between(taus, rec_m - rec_s, rec_m + rec_s, color=color, alpha=0.12)
            ax_rp.plot(taus, prec_m, color=color, ls='--', label=f'{lbl} prec.')
            ax_rp.fill_between(taus, prec_m - prec_s, prec_m + prec_s, color=color, alpha=0.08)

            ax_f.plot(taus, f1_m, color=color, ls='-', label=f'{lbl} F1')
            ax_f.fill_between(taus, f1_m - f1_s, f1_m + f1_s, color=color, alpha=0.12)
            ax_f.plot(taus, f2_m, color=color, ls=':', label=f'{lbl} F2')

            ax_sp.plot(taus, sp_m, color=color, label=lbl)
            ax_sp.fill_between(taus, sp_m - sp_s, sp_m + sp_s, color=color, alpha=0.12)

            for ax in (ax_rp, ax_f, ax_sp):
                if tau_ops is not None:
                    tau_vals = sorted(tau_ops.values())
                    ax.axvspan(tau_vals[0], tau_vals[-1], color=color, alpha=0.07, lw=0)
                    for tv in tau_vals:
                        ax.axvline(tv, color=color, lw=0.6, ls=':', alpha=0.55)
                else:
                    ax.axvline(tau_op, color=color, lw=0.9, ls=':', alpha=0.7)

        ax_rp.axhline(0.98, color=fc.COLORS['micro'], lw=0.6, ls=':', alpha=0.4)
        ax_rp.axhline(0.99, color=fc.COLORS['edge'], lw=0.6, ls=':', alpha=0.4)

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

        all_recalls, all_colors = [], []
        for mname, md in models.items():
            tau_ops = md.get('tau_ops', None)
            for sw, seed in zip(md['sweeps'], md['seeds']):
                t = tau_ops[seed] if tau_ops else md['tau_op']
                idx = np.argmin(np.abs(sw['tau'] - t))
                all_recalls.append(sw['recall'][idx])
                all_colors.append(md['color'])

        xs = np.arange(len(all_recalls))
        ax_bar.bar(xs, all_recalls, color=all_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax_bar.axhline(0.98, color=fc.COLORS['micro'], lw=0.8, ls='--', alpha=0.7, label='0.98 target (Nano/Micro)')
        ax_bar.axhline(0.99, color=fc.COLORS['edge'], lw=0.8, ls='--', alpha=0.7, label='0.99 target (Edge)')

        xtick_labels = []
        for mname, md in models.items():
            for seed in md['seeds']:
                xtick_labels.append(f's{seed}')
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(xtick_labels, fontsize=7)

        global_max_recall = 0
        for mname, md in models.items():
            for i, seed in enumerate(md['seeds']):
                tau_ops = md.get('tau_ops', None)
                t = tau_ops[seed] if tau_ops else md['tau_op']
                idx = np.argmin(np.abs(md['sweeps'][i]['tau'] - t))
                global_max_recall = max(global_max_recall, md['sweeps'][i]['recall'][idx])
        label_y = global_max_recall + 0.002

        pos = 0
        for mname, md in models.items():
            n = len(md['seeds'])
            ax_bar.annotate(md['label'], xy=(pos + n / 2 - 0.5, label_y),
                            fontsize=7, ha='center', va='bottom', color='black', fontweight='bold')
            pos += n

        ax_bar.set_ylabel('Recall')
        ax_bar.set_title('D  Per-seed recall at operating threshold', loc='left', fontweight='bold')
        ax_bar.set_ylim(0.95, 1.005)
        ax_bar.legend(fontsize=7, ncol=2, loc='lower right')
        ax_bar.grid(True, alpha=0.25, ls=':', axis='y')

        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


def main():
    fc.assert_arial_metric_font()
    args = fc.common_parser('Generate fig3_threshold_analysis').parse_args()
    models = fc.load_models(Path(args.results_dir) if args.results_dir else None)
    if not models:
        print("ERROR: no sweep data found."); return
    out_dir = fc.resolve_out(args)
    plot(models, out_dir / f'fig3_threshold_analysis.{args.fmt}', args.dpi)


if __name__ == '__main__':
    main()
