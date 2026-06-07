#!/usr/bin/env python3
"""
compile_paper_tables.py
Reads all per-seed threshold sweep tables and outputs ready-to-paste LaTeX rows
and narrative values for zabidi2026seabadnet_model.tex.
"""
import numpy as np
from pathlib import Path

RESULTS = Path("results")

MICRO_SEEDS = [42, 100, 786]
EDGE_SEEDS  = [42, 100, 786]

MICRO_DIR = "6b_micro_final_fft1024_m16_s{seed}"
EDGE_DIR  = "6c_edge_final_fft1024_m80_s{seed}"

MICRO_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]
EDGE_THRESHOLDS  = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]
MICRO_OP_TAU = 0.35
EDGE_OP_TAU  = 0.50


def parse_sweep(path: Path):
    """Return dict tau -> {recall, precision, f1, fpr, tp, fp, fn, tn}."""
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9:
            try:
                tau  = float(parts[0])
                rec  = float(parts[1])
                prec = float(parts[2])
                f1   = float(parts[3])
                fpr  = float(parts[4])
                tp   = int(parts[5])
                fp   = int(parts[6])
                fn   = int(parts[7])
                tn   = int(parts[8])
                rows[tau] = dict(recall=rec, precision=prec, f1=f1, fpr=fpr,
                                 tp=tp, fp=fp, fn=fn, tn=tn)
            except ValueError:
                continue
    return rows


def spec(tp, fp, fn, tn):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def f2(prec, rec):
    d = 4 * prec + rec
    return 5 * prec * rec / d if d > 0 else 0.0


def collect(model_dir_tpl, seeds, thresholds):
    per_seed = {}
    for s in seeds:
        p = RESULTS / model_dir_tpl.format(seed=s) / "threshold_sweep.txt"
        per_seed[s] = parse_sweep(p)
    return per_seed


def stats_at(per_seed, tau):
    vals = {k: [] for k in ('recall','precision','specificity','f1','f2_score')}
    for rows in per_seed.values():
        if tau not in rows:
            continue
        r = rows[tau]
        vals['recall'].append(r['recall'])
        vals['precision'].append(r['precision'])
        vals['specificity'].append(spec(r['tp'], r['fp'], r['fn'], r['tn']))
        vals['f1'].append(r['f1'])
        vals['f2_score'].append(f2(r['precision'], r['recall']))
    return {k: (np.mean(v), np.std(v, ddof=0)) for k, v in vals.items()}


def latex_row(tau, stats, op_tau, bold=False):
    def fmt(mean, std): return f"${mean:.3f}\\pm{std:.3f}$"
    is_op = abs(tau - op_tau) < 1e-9
    tau_str = f"\\textbf{{{tau:.2f}}}" if is_op else f"{tau:.2f}"
    rec  = fmt(*stats['recall'])
    prec = fmt(*stats['precision'])
    spc  = fmt(*stats['specificity'])
    f1   = fmt(*stats['f1'])
    f2s  = fmt(*stats['f2_score'])
    row  = f"\t\t{tau_str} & {rec} & {prec} & {spc} & {f1} & {f2s} \\\\"
    if is_op:
        row = row.replace("$", "$\\mathbf{", 1)
        # simpler: just prefix each cell with \mathbf
        cells = [tau_str, rec, prec, spc, f1, f2s]
        bold_cells = [f"$\\mathbf{{{tau:.2f}}}$" if i==0
                      else c.replace("$", "$\\mathbf{", 1).replace("\\pm", "}\\pm{").rstrip("$")+"}}$"
                      for i, c in enumerate(cells)]
        # rebuild cleanly
        def bfmt(mean, std): return f"$\\mathbf{{{mean:.3f}\\pm{std:.3f}}}$"
        row = (f"\t\t\\textbf{{{tau:.2f}}} & {bfmt(*stats['recall'])} & "
               f"{bfmt(*stats['precision'])} & {bfmt(*stats['specificity'])} & "
               f"{bfmt(*stats['f1'])} & {bfmt(*stats['f2_score'])} \\\\")
    return row


def main():
    micro = collect(MICRO_DIR, MICRO_SEEDS, MICRO_THRESHOLDS)
    edge  = collect(EDGE_DIR,  EDGE_SEEDS,  EDGE_THRESHOLDS)

    print("=" * 70)
    print("MICRO — tab:perf_table rows")
    print("=" * 70)
    for tau in MICRO_THRESHOLDS:
        s = stats_at(micro, tau)
        print(latex_row(tau, s, MICRO_OP_TAU))

    print()
    print("=" * 70)
    print("EDGE — tab:perf_table rows")
    print("=" * 70)
    for tau in EDGE_THRESHOLDS:
        s = stats_at(edge, tau)
        print(latex_row(tau, s, EDGE_OP_TAU))

    print()
    print("=" * 70)
    print("tab:threshold_comparison row values")
    print("=" * 70)
    ms = stats_at(micro, MICRO_OP_TAU)
    es = stats_at(edge,  EDGE_OP_TAU)
    def fmt2(mean,std): return f"{mean:.3f}±{std:.3f}"
    print(f"Micro τ={MICRO_OP_TAU}  recall={fmt2(*ms['recall'])}  prec={fmt2(*ms['precision'])}"
          f"  spec={fmt2(*ms['specificity'])}  F1={fmt2(*ms['f1'])}  F2={fmt2(*ms['f2_score'])}")
    print(f"Edge  τ={EDGE_OP_TAU}  recall={fmt2(*es['recall'])}  prec={fmt2(*es['precision'])}"
          f"  spec={fmt2(*es['specificity'])}  F1={fmt2(*es['f1'])}  F2={fmt2(*es['f2_score'])}")

    print()
    print("=" * 70)
    print("Per-seed values at operating tau (narrative text)")
    print("=" * 70)
    print(f"\nMicro @ τ={MICRO_OP_TAU}:")
    for s in MICRO_SEEDS:
        r = micro[s].get(MICRO_OP_TAU, {})
        if r:
            print(f"  s{s}: recall={r['recall']:.4f}  prec={r['precision']:.4f}"
                  f"  spec={spec(r['tp'],r['fp'],r['fn'],r['tn']):.4f}")
    print(f"\nEdge @ τ={EDGE_OP_TAU}:")
    for s in EDGE_SEEDS:
        r = edge[s].get(EDGE_OP_TAU, {})
        if r:
            print(f"  s{s}: recall={r['recall']:.4f}  prec={r['precision']:.4f}"
                  f"  spec={spec(r['tp'],r['fp'],r['fn'],r['tn']):.4f}")

    print()
    print("=" * 70)
    print("AUC values (from threshold_locked.txt / results_summary.txt)")
    print("=" * 70)
    micro_aucs, edge_aucs = [], []
    for s in MICRO_SEEDS:
        txt = (RESULTS / MICRO_DIR.format(seed=s) / "results_summary.txt").read_text()
        for line in txt.splitlines():
            if line.startswith("Float32 Model AUC:"):
                micro_aucs.append(float(line.split()[-1]))
    for s in EDGE_SEEDS:
        txt = (RESULTS / EDGE_DIR.format(seed=s) / "results_summary.txt").read_text()
        for line in txt.splitlines():
            if line.startswith("Float32 Model AUC:"):
                edge_aucs.append(float(line.split()[-1]))
    print(f"Micro AUC per seed: {micro_aucs}  → {np.mean(micro_aucs):.4f} ± {np.std(micro_aucs,ddof=0):.4f}")
    print(f"Edge  AUC per seed: {edge_aucs}   → {np.mean(edge_aucs):.4f} ± {np.std(edge_aucs,ddof=0):.4f}")

    print()
    print("=" * 70)
    print("Minimum recall across seeds at operating tau")
    print("=" * 70)
    micro_recs = [micro[s][MICRO_OP_TAU]['recall'] for s in MICRO_SEEDS if MICRO_OP_TAU in micro[s]]
    edge_recs  = [edge[s][EDGE_OP_TAU]['recall']   for s in EDGE_SEEDS  if EDGE_OP_TAU  in edge[s]]
    print(f"Micro min recall @ τ={MICRO_OP_TAU}: {min(micro_recs):.4f}  (target ≥0.98)")
    print(f"Edge  min recall @ τ={EDGE_OP_TAU}:  {min(edge_recs):.4f}  (target ≥0.99)")


if __name__ == "__main__":
    main()
