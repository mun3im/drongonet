#!/usr/bin/env python3
"""
compile_paper_tables.py
Reads all per-seed threshold sweep tables and outputs ready-to-paste LaTeX rows
and narrative values for zabidi2026seabadnet_model.tex.
"""
import numpy as np
from pathlib import Path

RESULTS = Path("results4arxiv")

MICRO_SEEDS = [42, 100, 786]
EDGE_SEEDS  = [42, 100, 786]

MICRO_DIR = "6b_micro_final_fft1024_m16_s{seed}"
EDGE_DIR  = "6c_edge_final_fft1024_m80_s{seed}"

MICRO_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]
# Post-timeshift-fix retrain: locked thresholds re-derived 2026-06-23.
# Edge now locks to a single tau=0.50 across all seeds (was per-seed 0.45/0.55/0.60).
EDGE_THRESHOLDS  = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MICRO_OP_TAU = 0.30
# Edge operating tau is now uniform across seeds; use None to suppress bold formatting
EDGE_OP_TAU  = 0.50
EDGE_OP_TAUS = {42: 0.50, 100: 0.50, 786: 0.50}  # uniform post-fix operating threshold


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


def float32_auc(txt):
    """Return float32 AUC, handling both summary formats.

    Edge: a single 'Float32 Model AUC: 0.9990' line.
    Micro: a 'Float32 Model:' header followed by an indented 'AUC: 0.9734' line.
    """
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Float32 Model AUC:"):
            return float(s.split()[-1])
        if s == "Float32 Model:":
            for nxt in lines[i + 1:]:
                if nxt.strip().startswith("AUC:"):
                    return float(nxt.split()[-1])
    return float("nan")


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


def latex_row_micro(tau, stats, op_tau):
    def fmt(mean, std): return f"${mean:.3f}\\pm{std:.3f}$"
    def bfmt(mean, std): return f"$\\mathbf{{{mean:.3f}\\pm{std:.3f}}}$"
    is_op = abs(tau - op_tau) < 1e-9
    if is_op:
        return (f"\t\t\\textbf{{{tau:.2f}}} & {bfmt(*stats['recall'])} & "
                f"{bfmt(*stats['precision'])} & {bfmt(*stats['specificity'])} & "
                f"{bfmt(*stats['f1'])} & {bfmt(*stats['f2_score'])} \\\\")
    tau_str = f"{tau:.2f}"
    return (f"\t\t{tau_str} & {fmt(*stats['recall'])} & {fmt(*stats['precision'])} & "
            f"{fmt(*stats['specificity'])} & {fmt(*stats['f1'])} & {fmt(*stats['f2_score'])} \\\\")


def latex_row_edge(tau, stats, op_taus):
    """op_taus: dict {seed: tau} — marks per-seed operating thresholds with †."""
    def fmt(mean, std): return f"${mean:.3f}\\pm{std:.3f}$"
    is_op = tau in op_taus.values()
    tau_str = f"${tau:.2f}^\\dagger$" if is_op else f"{tau:.2f}"
    return (f"\t\t{tau_str} & {fmt(*stats['recall'])} & {fmt(*stats['precision'])} & "
            f"{fmt(*stats['specificity'])} & {fmt(*stats['f1'])} & {fmt(*stats['f2_score'])} \\\\")


def main():
    micro = collect(MICRO_DIR, MICRO_SEEDS, MICRO_THRESHOLDS)
    edge  = collect(EDGE_DIR,  EDGE_SEEDS,  EDGE_THRESHOLDS)

    print("=" * 70)
    print("MICRO — tab:perf_table rows")
    print("=" * 70)
    for tau in MICRO_THRESHOLDS:
        s = stats_at(micro, tau)
        print(latex_row_micro(tau, s, MICRO_OP_TAU))

    print()
    print("=" * 70)
    print("EDGE — tab:perf_table rows († = per-seed operating threshold)")
    print("=" * 70)
    for tau in EDGE_THRESHOLDS:
        s = stats_at(edge, tau)
        print(latex_row_edge(tau, s, EDGE_OP_TAUS))

    print()
    print("=" * 70)
    print("tab:threshold_comparison row values")
    print("=" * 70)
    ms = stats_at(micro, MICRO_OP_TAU)
    def fmt2(mean,std): return f"{mean:.3f}±{std:.3f}"
    print(f"Micro τ={MICRO_OP_TAU}  recall={fmt2(*ms['recall'])}  prec={fmt2(*ms['precision'])}"
          f"  spec={fmt2(*ms['specificity'])}  F1={fmt2(*ms['f1'])}  F2={fmt2(*ms['f2_score'])}")
    print("Edge: per-seed operating thresholds (see EDGE_OP_TAUS)")
    for s, tau in sorted(EDGE_OP_TAUS.items()):
        r = edge[s].get(tau, {})
        if r:
            sp = spec(r['tp'], r['fp'], r['fn'], r['tn'])
            f2v = f2(r['precision'], r['recall'])
            print(f"  s{s} τ={tau}  recall={r['recall']:.3f}  prec={r['precision']:.3f}"
                  f"  spec={sp:.3f}  F1={r['f1']:.3f}  F2={f2v:.3f}")

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
    print("\nEdge @ per-seed operating τ:")
    for s, tau in sorted(EDGE_OP_TAUS.items()):
        r = edge[s].get(tau, {})
        if r:
            print(f"  s{s} (τ={tau}): recall={r['recall']:.4f}  prec={r['precision']:.4f}"
                  f"  spec={spec(r['tp'],r['fp'],r['fn'],r['tn']):.4f}")

    print()
    print("=" * 70)
    print("AUC values (from threshold_locked.txt / results_summary.txt)")
    print("=" * 70)
    micro_aucs, edge_aucs = [], []
    for s in MICRO_SEEDS:
        txt = (RESULTS / MICRO_DIR.format(seed=s) / "results_summary.txt").read_text()
        micro_aucs.append(float32_auc(txt))
    for s in EDGE_SEEDS:
        txt = (RESULTS / EDGE_DIR.format(seed=s) / "results_summary.txt").read_text()
        edge_aucs.append(float32_auc(txt))
    print(f"Micro AUC per seed: {micro_aucs}  → {np.mean(micro_aucs):.4f} ± {np.std(micro_aucs,ddof=0):.4f}")
    print(f"Edge  AUC per seed: {edge_aucs}   → {np.mean(edge_aucs):.4f} ± {np.std(edge_aucs,ddof=0):.4f}")

    print()
    print("=" * 70)
    print("Minimum recall across seeds at operating tau")
    print("=" * 70)
    micro_recs = [micro[s][MICRO_OP_TAU]['recall'] for s in MICRO_SEEDS if MICRO_OP_TAU in micro[s]]
    edge_recs  = [edge[s][tau]['recall'] for s, tau in EDGE_OP_TAUS.items() if tau in edge[s]]
    print(f"Micro min recall @ τ={MICRO_OP_TAU}: {min(micro_recs):.4f}  (target ≥0.98)")
    print(f"Edge  min recall @ per-seed τ {EDGE_OP_TAUS}: {min(edge_recs):.4f}  (target ≥0.99)")


if __name__ == "__main__":
    main()
