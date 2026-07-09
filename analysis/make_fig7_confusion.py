#!/usr/bin/env python3
"""make_fig7_confusion.py — generates ONLY fig7_confusion_matrices.pdf.

NOTE: fig7 is not currently included in the paper; kept for completeness. Usage:
  python analysis/make_fig7_confusion.py [--out-dir images/] [--fmt pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figcommon as fc


def plot(models: dict, out: Path, dpi: int):
    var_list = list(models.items())
    nrow, ncol = len(var_list), 3

    with plt.rc_context(fc.FIGURE_STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.5 * nrow))

        for row, (mname, md) in enumerate(var_list):
            tau_ops = md.get('tau_ops', None)
            tau_op = tau_ops[42] if tau_ops else md['tau_op']
            sweeps = md['sweeps']
            lbl = md['label']

            tau_arr = sweeps[0]['tau']
            tp_m, _ = fc.stack_sweeps(sweeps, 'tp')
            fp_m, _ = fc.stack_sweeps(sweeps, 'fp')
            fn_m, _ = fc.stack_sweeps(sweeps, 'fn')
            total_pos_true = sweeps[0]['tp'][0] + sweeps[0]['fn'][0]
            total_neg_true = sweeps[0]['fp'][0] + (sweeps[0]['tn'][0] if sweeps[0]['tn'] is not None else 0)
            total = total_pos_true + total_neg_true

            taus_show = [round(t, 3) for t in (tau_op * 0.5, tau_op, min(tau_op * 2, 0.90))]

            for col, tau_show in enumerate(taus_show):
                ax = axes[row, col] if nrow > 1 else axes[col]
                idx = np.argmin(np.abs(tau_arr - tau_show))
                tp_val, fp_val, fn_val = tp_m[idx], fp_m[idx], fn_m[idx]
                tn_val = total - tp_val - fp_val - fn_val

                cm = np.array([[tn_val, fp_val], [fn_val, tp_val]])
                cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

                highlight = (col == 1)
                ax.imshow(cm_n, vmin=0, vmax=1, cmap='Oranges' if highlight else 'Blues', aspect='equal')
                labels = [['TN', 'FP'], ['FN', 'TP']]
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f'{labels[i][j]}\n{cm[i,j]:,.0f}\n({cm_n[i,j]:.1%})',
                                ha='center', va='center', fontsize=7,
                                color='white' if cm_n[i, j] > 0.55 else 'black')
                ax.set_title(f'tau={tau_arr[idx]:.3f}{"  *" if highlight else ""}', fontsize=8,
                             fontweight='bold' if highlight else 'normal')
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(['Neg', 'Pos'], fontsize=7)
                ax.set_yticklabels(['Neg', 'Pos'], fontsize=7)
                if col == 0:
                    ax.set_ylabel(lbl, fontsize=8)
                if row == nrow - 1:
                    ax.set_xlabel('Predicted', fontsize=7)

        fig.suptitle('Confusion matrices (mean over seeds); * = operating threshold', fontsize=8, y=1.01)
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved {out}")


def main():
    fc.assert_arial_metric_font()
    args = fc.common_parser('Generate fig7_confusion_matrices').parse_args()
    models = fc.load_models(Path(args.results_dir) if args.results_dir else None)
    if not models:
        print("ERROR: no sweep data found."); return
    out_dir = fc.resolve_out(args)
    plot(models, out_dir / f'fig7_confusion_matrices.{args.fmt}', args.dpi)


if __name__ == '__main__':
    main()
