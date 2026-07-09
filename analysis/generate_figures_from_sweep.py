#!/usr/bin/env python3
"""
generate_figures_from_sweep.py — convenience runner: regenerate ALL sweep-based figures at once.

The figures now live in one script each so you can tweak/regenerate a single one:
    make_fig3_threshold_analysis.py -> fig3_threshold_analysis.pdf
    make_fig4_pareto.py             -> fig4_pareto.pdf
    make_fig5_roc_pr.py             -> fig5_roc_pr_curves.pdf
    make_fig7_confusion.py          -> fig7_confusion_matrices.pdf   (not used in the paper)
Shared style + seeds + operating thresholds live in figcommon.py (single source of truth).
fig6 is its own pipeline (generate_fig6_part1_extract.py -> generate_fig6_part2_plot.py).

This runner just forwards the same CLI args to each per-figure main(). Usage:
    python analysis/generate_figures_from_sweep.py [--out-dir images/] [--fmt pdf]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fig4_pareto
import make_fig3_threshold_analysis
import make_fig5_roc_pr
import make_fig7_confusion


def main():
    # each main() parses the shared sys.argv independently
    for mod in (make_fig4_pareto, make_fig3_threshold_analysis,
                make_fig5_roc_pr, make_fig7_confusion):
        mod.main()


if __name__ == '__main__':
    main()
