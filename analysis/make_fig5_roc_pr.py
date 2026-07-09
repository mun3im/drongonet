#!/usr/bin/env python3
"""make_fig5_roc_pr.py — generates ONLY fig5_roc_pr_curves.pdf.

Approximate ROC/PR from the threshold sweep. Usage:
  python analysis/make_fig5_roc_pr.py [--out-dir images/] [--fmt pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figcommon as fc


def plot(models: dict, out: Path, dpi: int):
    with plt.rc_context(fc.FIGURE_STYLE):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.0, 3.5))

        for mname, md in models.items():
            sweeps = md['sweeps']
            color = md['color']
            lbl = md['label']
            tau_op = md['tau_op']

            fpr_m, _ = fc.stack_sweeps(sweeps, 'fpr')
            rec_m, _ = fc.stack_sweeps(sweeps, 'recall')
            prec_m, _ = fc.stack_sweeps(sweeps, 'precision')
            auc_vals = [s['auc'] for s in sweeps]
            auc_mean = np.mean(auc_vals)
            auc_std = np.std(auc_vals)

            sort_idx = np.argsort(fpr_m)
            fpr_s = np.concatenate([[0], fpr_m[sort_idx], [1]])
            tpr_s = np.concatenate([[0], rec_m[sort_idx], [1]])
            ax_roc.plot(fpr_s, tpr_s, color=color,
                        label=f'{lbl}\n(AUC={auc_mean:.4f}±{auc_std:.4f})')

            tau_arr = sweeps[0]['tau']
            tau_ops = md.get('tau_ops', None)
            sort_idx2 = np.argsort(rec_m)
            rec_ps = np.concatenate([[0], rec_m[sort_idx2], [1]])
            prec_ps = np.concatenate([[1], prec_m[sort_idx2], [0]])
            ax_pr.plot(rec_ps, prec_ps, color=color, label=lbl)

            if tau_ops is not None:
                op_label_done = False
                for seed_i, (seed, t) in enumerate(sorted(tau_ops.items())):
                    idx_op = np.argmin(np.abs(tau_arr - t))
                    lbl_op = f'  τ={list(tau_ops.values())[0]}–{list(tau_ops.values())[-1]}' if not op_label_done else None
                    ax_roc.scatter(fpr_m[idx_op], rec_m[idx_op], s=40, color=color, zorder=5,
                                   edgecolors='white', linewidths=0.8, marker=['o', 's', '^'][seed_i], label=lbl_op)
                    ax_pr.scatter(rec_m[idx_op], prec_m[idx_op], s=40, color=color, zorder=5,
                                  edgecolors='white', linewidths=0.8, marker=['o', 's', '^'][seed_i])
                    op_label_done = True
            else:
                idx_op = np.argmin(np.abs(tau_arr - tau_op))
                ax_roc.scatter(fpr_m[idx_op], rec_m[idx_op], s=55, color=color, zorder=5,
                               edgecolors='white', linewidths=0.8, label=f'  τ={tau_op}')
                ax_pr.scatter(rec_m[idx_op], prec_m[idx_op], s=55, color=color, zorder=5,
                              edgecolors='white', linewidths=0.8)

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


def main():
    fc.assert_arial_metric_font()
    args = fc.common_parser('Generate fig5_roc_pr_curves').parse_args()
    models = fc.load_models(Path(args.results_dir) if args.results_dir else None)
    if not models:
        print("ERROR: no sweep data found."); return
    out_dir = fc.resolve_out(args)
    plot(models, out_dir / f'fig5_roc_pr_curves.{args.fmt}', args.dpi)


if __name__ == '__main__':
    main()
