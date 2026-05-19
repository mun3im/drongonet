#!/usr/bin/env python3
"""analyze_final_results.py — Multi-seed publication analysis for 6b and 3f."""
import re
import numpy as np
from pathlib import Path

RESULTS = Path("results")
N_MELS  = [16, 32, 48, 64, 80]
SEEDS   = [42, 100, 786]
S3F     = "3f_gap_focal_loss_freq_emph_pointwise"


def get(script, n_fft, n_mels, seed=42):
    d = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
    p = d / "results_summary.txt"
    if not p.exists():
        return {}
    r = {}
    for line in p.read_text().splitlines():
        if m := re.search(r"(?<!\S)AUC[:\s]+([0-9.]+)", line, re.I):
            if "auc" not in r:
                r["auc"] = float(m.group(1))
        if m := re.search(r"(?:Model Size|Size)[:\s]+([0-9.]+)\s*KB", line, re.I):
            r["size"] = float(m.group(1))
        if m := re.search(r"(?:Avg Inference|Inference)[^:]*:[:\s]+([0-9.]+)\s*ms", line, re.I):
            r["lat"] = float(m.group(1))
    return r


def get_recall(script, n_fft, n_mels, seed=42):
    p = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}" / "float_classification_report.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if "Positive" in line or "positive" in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    return float(parts[2])
                except ValueError:
                    pass
    return None


def auc_table(script, label):
    print("=" * 68)
    print(f"{label} — AUC × n_mels × seed")
    print("=" * 68)
    print(f"  {'':4}  {'s42':>7}  {'s100':>7}  {'s786':>7}  {'mean±std':>14}  {'size':>7}  {'lat':>6}")
    rows = {}
    for m in N_MELS:
        aucs = [get(script, 1024, m, s).get("auc") for s in SEEDS]
        av   = [a for a in aucs if a is not None]
        r42  = get(script, 1024, m, 42)
        mean = np.mean(av) if av else 0.0
        std  = np.std(av)  if len(av) > 1 else 0.0
        rows[m] = (mean, std, r42)
        vals = "  ".join(f"{a:.4f}" if a is not None else "   —  " for a in aucs)
        print(f"  m{m:2d}: {vals}  {mean:.4f}±{std:.4f}  "
              f"{r42.get('size', '—'):>5} KB  {r42.get('lat', '—'):>4} ms")
    return rows


def recall_section():
    print()
    print("=" * 68)
    print("Recall @ τ=0.5 (default threshold) — Positive class")
    print("=" * 68)
    for label, script, tgt in [
        ("6b Micro", "6b_micro_improved", 0.98),
        ("3f Edge",  S3F,                 0.99),
    ]:
        print(f"  {label}  (target ≥{tgt}):")
        for m in N_MELS:
            rc   = [get_recall(script, 1024, m, s) for s in SEEDS]
            rv   = [x for x in rc if x is not None]
            vals = "  ".join(f"{x:.4f}" if x is not None else "—" for x in rc)
            mean = np.mean(rv) if rv else 0.0
            ok   = "✅" if mean >= tgt else "⚠️ "
            print(f"    m{m:2d}: {vals}  mean={mean:.4f} {ok}")


def verdict(micro_rows, edge_rows):
    print()
    print("=" * 68)
    print("PUBLICATION VERDICT")
    print("=" * 68)
    micro_best_m = max(N_MELS, key=lambda m: micro_rows[m][0])
    edge_best_m  = max(N_MELS, key=lambda m: edge_rows[m][0])

    for label, script, best_m, rows, tgt_size, tgt_lat, tgt_rc in [
        ("SEABADNet-Micro (6b)", "6b_micro_improved", micro_best_m, micro_rows, 8.0,  1.0, 0.98),
        ("SEABADNet-Edge  (3f)", S3F,                  edge_best_m,  edge_rows,  35.0, 2.0, 0.99),
    ]:
        mean, std, r42 = rows[best_m]
        rcs  = [get_recall(script, 1024, best_m, s) or 0.0 for s in SEEDS]
        size = r42.get("size", 99)
        lat  = r42.get("lat",  99)
        rc_mean = np.mean(rcs)
        print(f"\n  {label}  →  n_mels={best_m}")
        print(f"    AUC:     {mean:.4f} ± {std:.4f}")
        rc_ok = "✅" if rc_mean >= tgt_rc else "⚠️  needs threshold tuning"
        print(f"    Recall:  {rc_mean:.4f} ± {np.std(rcs):.4f}  (target ≥{tgt_rc}) {rc_ok}")
        print(f"    Size:    {size} KB  (target ≤{tgt_size} KB) {'✅' if size <= tgt_size else '❌'}")
        print(f"    Latency: {lat} ms  (target <{tgt_lat} ms) {'✅' if lat < tgt_lat else '❌'}")


if __name__ == "__main__":
    micro_rows = auc_table("6b_micro_improved", "6b_micro_improved")
    print()
    edge_rows  = auc_table(S3F, S3F)
    recall_section()
    verdict(micro_rows, edge_rows)
