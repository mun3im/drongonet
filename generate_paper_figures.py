#!/usr/bin/env python3
"""
generate_paper_figures.py — Paper figures for SEABADNet (Bioacoustics journal)

Reads directly from results/{script}_fft{n_fft}_m{n_mels}_s{seed}/results_summary.txt.
All naming uses current SEABADNet / SEABAD conventions.

Models:
  SEABADNet-Micro : 6b_micro_improved, n_mels=16, n_fft=1024, τ=0.35
  SEABADNet-Edge  : 3f_gap_focal_loss_freq_emph_pointwise, n_mels=80, n_fft=1024, τ=0.35

Figures produced:
  figure1_ablation_chain      — AUC step progression toward SEABADNet-Micro
  figure2_nmels_sweep         — AUC / size / latency vs n_mels (baseline, Edge, Micro)
  figure3_dropout_sweep       — Dropout sweep: Conv2D track vs Depthwise track
  figure4_efficiency_scatter  — AUC vs size scatter, all experiments, Micro & Edge highlighted
  figure5_multiseed_validation— Multi-seed AUC stability for final models (3 seeds × n_mels)
  figure6_quant_robustness    — Float32 vs TFLite INT8 AUC for final models
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize": 9,
})

RESULTS       = Path("results")
FIGURES       = Path("figures")
FIGURES.mkdir(exist_ok=True)

SEEDS         = [42, 100, 786]
N_MELS        = [16, 32, 48, 64, 80]
MICRO_SCRIPT  = "6b_micro_improved"
EDGE_SCRIPT   = "3f_gap_focal_loss_freq_emph_pointwise"

TAU           = 0.35   # operating threshold for both models
MICRO_NMELS   = 16
EDGE_NMELS    = 80

C_MICRO  = "#e74c3c"   # red  — SEABADNet-Micro
C_EDGE   = "#3498db"   # blue — SEABADNet-Edge
C_OTHER  = "#95a5a6"   # grey — other experiments


# ── Data helpers ─────────────────────────────────────────────────────────────

def get(script, n_fft, n_mels, seed=42):
    """Parse results_summary.txt; returns dict with keys auc, tflite_auc, size, lat, params."""
    p = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}" / "results_summary.txt"
    if not p.exists():
        return {}
    r = {}
    all_aucs = []
    for line in p.read_text().splitlines():
        if m := re.search(r"(?<![\w.])AUC[:\s]+([0-9.]+)", line, re.I):
            if "degrad" not in line.lower():
                all_aucs.append(float(m.group(1)))
        if m := re.search(r"(?:Model Size|Size)[:\s]+([0-9.]+)\s*KB", line, re.I):
            r["size"] = float(m.group(1))
        if m := re.search(r"(?:Avg Inference|Inference)[^:]*:[:\s]+([0-9.]+)\s*ms", line, re.I):
            r["lat"] = float(m.group(1))
        if m := re.search(r"Total Params[^:]*:[:\s]+([0-9,]+)", line, re.I):
            r["params"] = int(m.group(1).replace(",", ""))
    if all_aucs:
        r["auc"] = all_aucs[0]
        if len(all_aucs) >= 2:
            r["tflite_auc"] = all_aucs[1]
    return r


def multi_seed(script, n_fft, n_mels, seeds=SEEDS, metric="auc"):
    vals = [get(script, n_fft, n_mels, s).get(metric) for s in seeds]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = FIGURES / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: figures/{stem}.png/.pdf")
    plt.close(fig)


# ── Figure 1: Micro Ablation Chain ──────────────────────────────────────────

def figure1_ablation_chain():
    """
    Step chart showing AUC at each phase of the Micro ablation chain,
    all at n_mels=16, seed=42 (ablation protocol), except the final
    SEABADNet-Micro which uses mean ± std over 3 seeds.
    """
    print("\nFigure 1: Ablation chain...")

    steps = [
        # (label, script, n_fft, n_mels, seed, use_multiseed)
        ("TinyChirp\nCNN-Mel",         "1a_baseline2d",                       1024, 16, 42, False),
        ("+ GAP",                       "2a_baseline_gap",                     1024, 16, 42, False),
        ("+ Depthwise\nsep. conv",      "3a_depthwise",                        1024, 16, 42, False),
        ("+ Focal\nloss",               "3c_gap_focal_loss",                   1024, 16, 42, False),
        ("+ Freq.\nemphasis",           "3d_gap_freq_emphasis",                1024, 16, 42, False),
        ("+ Dropout\n=0.1",             "4e_depthwise_drop01",                 1024, 16, 42, False),
        ("+ 6 filters",                 "5a_depthwise_f6",                     1024, 16, 42, False),
        ("SEABADNet\n-Micro",           MICRO_SCRIPT,                          1024, 16, 42, True),
    ]

    aucs, errs, labels = [], [], []
    for label, script, n_fft, n_mels, seed, multiseed in steps:
        if multiseed:
            mean, std = multi_seed(script, n_fft, n_mels)
            aucs.append(mean)
            errs.append(std if std is not None else 0.0)
        else:
            r = get(script, n_fft, n_mels, seed)
            aucs.append(r.get("auc"))
            errs.append(0.0)
        labels.append(label)

    # Replace None with NaN for plotting
    aucs = [a if a is not None else float("nan") for a in aucs]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, [a * 100 for a in aucs], color=C_OTHER, alpha=0.75,
                  edgecolor="white", linewidth=0.8)

    # Final bar (SEABADNet-Micro) gets its own colour
    bars[-1].set_facecolor(C_MICRO)
    bars[-1].set_alpha(0.9)

    # Error bar on the final (multi-seed) entry
    if errs[-1] > 0:
        ax.errorbar(x[-1], aucs[-1] * 100, yerr=errs[-1] * 100,
                    fmt="none", color="black", capsize=4, linewidth=1.5)

    # Value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        if not np.isnan(auc):
            ax.text(bar.get_x() + bar.get_width() / 2, auc * 100 + 0.1,
                    f"{auc * 100:.2f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if i == len(aucs) - 1 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC (%)", fontweight="bold")
    ax.set_title("SEABADNet-Micro Ablation Chain (n_mels=16, seed=42)", fontweight="bold")
    ax.set_ylim(92, 100)
    ax.axhline(y=97.36, color=C_MICRO, linestyle="--", alpha=0.4, linewidth=1,
               label="SEABADNet-Micro (97.36%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    save(fig, "figure1_ablation_chain")


# ── Figure 2: n_mels Sweep ───────────────────────────────────────────────────

def figure2_nmels_sweep():
    """
    Three subplots: AUC / size / latency vs n_mels.
    Lines: 1a_baseline2d (starting arch), 3f (Edge arch), 6b_micro (all at m16 only).
    """
    print("\nFigure 2: n_mels sweep...")

    scripts = [
        ("1a_baseline2d",  1024, "Conv2D baseline (1a)",   "o-",  "#2ecc71"),
        (EDGE_SCRIPT,      1024, "SEABADNet-Edge arch (3f)","s--", C_EDGE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for script, n_fft, label, ls, color in scripts:
        aucs, sizes, lats = [], [], []
        for m in N_MELS:
            r = get(script, n_fft, m)
            aucs.append(r.get("tflite_auc", r.get("auc")) if r else None)
            sizes.append(r.get("size") if r else None)
            lats.append(r.get("lat") if r else None)

        x = N_MELS
        aucs  = [a * 100 if a is not None else None for a in aucs]

        axes[0].plot(x, aucs,  ls, linewidth=2, markersize=7, label=label, color=color)
        axes[1].plot(x, sizes, ls, linewidth=2, markersize=7, label=label, color=color)
        axes[2].plot(x, lats,  ls, linewidth=2, markersize=7, label=label, color=color)

    # SEABADNet-Micro operating point (m16, mean over 3 seeds)
    micro_mean, micro_std = multi_seed(MICRO_SCRIPT, 1024, MICRO_NMELS, metric="tflite_auc")
    micro_r = get(MICRO_SCRIPT, 1024, MICRO_NMELS)
    if micro_mean:
        axes[0].errorbar(MICRO_NMELS, micro_mean * 100, yerr=micro_std * 100,
                         fmt="*", color=C_MICRO, markersize=12, capsize=4, linewidth=1.5,
                         label=f"SEABADNet-Micro (m{MICRO_NMELS})")
    if micro_r.get("size"):
        axes[1].scatter([MICRO_NMELS], [micro_r["size"]], s=120, marker="*",
                        color=C_MICRO, zorder=5, label=f"SEABADNet-Micro (m{MICRO_NMELS})")
    if micro_r.get("lat"):
        axes[2].scatter([MICRO_NMELS], [micro_r["lat"]], s=120, marker="*",
                        color=C_MICRO, zorder=5, label=f"SEABADNet-Micro (m{MICRO_NMELS})")

    # Edge operating point (m80, mean over 3 seeds)
    edge_mean, edge_std = multi_seed(EDGE_SCRIPT, 1024, EDGE_NMELS, metric="tflite_auc")
    if edge_mean:
        axes[0].errorbar(EDGE_NMELS, edge_mean * 100, yerr=edge_std * 100,
                         fmt="D", color="#1a5276", markersize=8, capsize=4, linewidth=1.5,
                         label=f"SEABADNet-Edge (m{EDGE_NMELS})")

    for i, (ylabel, title) in enumerate([
        ("TFLite INT8 AUC (%)",  "(a) Accuracy vs Frequency Resolution"),
        ("Model Size (KB)",       "(b) Model Size vs Frequency Resolution"),
        ("Inference Latency (ms)","(c) Latency vs Frequency Resolution"),
    ]):
        axes[i].set_xlabel("n_mels (Mel Bins)", fontweight="bold")
        axes[i].set_ylabel(ylabel, fontweight="bold")
        axes[i].set_title(title, fontweight="bold")
        axes[i].set_xticks(N_MELS)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    save(fig, "figure2_nmels_sweep")


# ── Figure 3: Dropout Sweep ──────────────────────────────────────────────────

def figure3_dropout_sweep():
    """
    Two subplots: AUC vs dropout rate.
    (a) Conv2D track (4a–4d) at best n_mels (64 from Phase 1 gate).
    (b) Depthwise track (4e–4h) at n_mels=16 (Micro branch).
    """
    print("\nFigure 3: Dropout sweep...")

    dropout_rates = [0.1, 0.2, 0.3, 0.4]

    conv2d_scripts = [
        ("4a_dropout01", 1024, 64),
        ("4b_dropout02", 1024, 64),
        ("4c_dropout03", 1024, 64),
        ("4d_dropout04", 1024, 64),
    ]
    dw_scripts = [
        ("4e_depthwise_drop01", 1024, 16),
        ("4f_depthwise_drop02", 1024, 16),
        ("4g_depthwise_drop03", 1024, 16),
        ("4h_depthwise_drop04", 1024, 16),
    ]

    def fetch_aucs(script_list):
        aucs, sizes = [], []
        for script, n_fft, n_mels in script_list:
            r = get(script, n_fft, n_mels)
            aucs.append(r.get("tflite_auc", r.get("auc")))
            sizes.append(r.get("size"))
        return aucs, sizes

    conv_aucs, conv_sizes = fetch_aucs(conv2d_scripts)
    dw_aucs,   dw_sizes   = fetch_aucs(dw_scripts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def plot_track(ax, aucs, rates, color, label, best_idx):
        vals = [a * 100 if a is not None else None for a in aucs]
        ax.plot(rates, vals, "o-", linewidth=2, markersize=8, color=color, label=label)
        if best_idx is not None and vals[best_idx] is not None:
            ax.scatter([rates[best_idx]], [vals[best_idx]], s=150, marker="*",
                       color=color, edgecolors="black", linewidth=1, zorder=5)

    # Find best (highest AUC) index
    def best(aucs):
        valid = [(a, i) for i, a in enumerate(aucs) if a is not None]
        return max(valid)[1] if valid else None

    plot_track(axes[0], conv_aucs, dropout_rates, C_EDGE,
               "Conv2D (n_mels=64)", best(conv_aucs))
    plot_track(axes[1], dw_aucs, dropout_rates, C_MICRO,
               "Depthwise sep. (n_mels=16)", best(dw_aucs))

    for ax, title, lock_label in [
        (axes[0], "(a) Edge Track — Conv2D dropout sweep",   "Gate 4A: lock dropout for Edge"),
        (axes[1], "(b) Micro Track — Depthwise dropout sweep","Gate 4B: lock dropout for Micro"),
    ]:
        ax.set_xlabel("Dropout Rate", fontweight="bold")
        ax.set_ylabel("TFLite INT8 AUC (%)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(dropout_rates)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "figure3_dropout_sweep")


# ── Figure 4: Efficiency Scatter ─────────────────────────────────────────────

def figure4_efficiency_scatter():
    """
    AUC vs Model Size scatter across all experiments (seed=42),
    with SEABADNet-Micro and SEABADNet-Edge highlighted.
    Size budget lines overlaid.
    """
    print("\nFigure 4: Efficiency scatter...")

    all_experiments = [
        # (script, n_fft, n_mels)
        *[("1a_baseline2d",                            1024, m) for m in N_MELS],
        *[("2a_baseline_gap",                          1024, m) for m in N_MELS],
        *[("2b_baseline_gap_learned",                  1024, m) for m in N_MELS],
        *[("2c_baseline_gap_1x1",                      1024, m) for m in N_MELS],
        *[("3a_depthwise",                             1024, m) for m in N_MELS],
        *[("3b_filters8",                              1024, m) for m in N_MELS],
        *[("3c_gap_focal_loss",                        1024, m) for m in N_MELS],
        *[("3d_gap_freq_emphasis",                     1024, m) for m in N_MELS],
        *[("3e_gap_freq_emph_ds",                      1024, m) for m in N_MELS],
        *[(EDGE_SCRIPT,                                1024, m) for m in N_MELS],
        *[("4a_dropout01",                             1024, m) for m in N_MELS],
        *[("4e_depthwise_drop01",                      1024, m) for m in N_MELS],
        *[("5a_depthwise_f6",                          1024, m) for m in N_MELS],
        *[("6a_micro_final",                           1024, m) for m in N_MELS],
        *[("6b_edge_final",                            1024, m) for m in N_MELS],
        *[(MICRO_SCRIPT,                               1024, m) for m in N_MELS],
    ]

    aucs, sizes, colors, alphas = [], [], [], []
    micro_pts, edge_pts = [], []

    for script, n_fft, n_mels in all_experiments:
        r = get(script, n_fft, n_mels)
        auc  = r.get("tflite_auc", r.get("auc"))
        size = r.get("size")
        if auc is None or size is None:
            continue
        is_micro = (script == MICRO_SCRIPT and n_mels == MICRO_NMELS)
        is_edge  = (script == EDGE_SCRIPT  and n_mels == EDGE_NMELS)
        aucs.append(auc * 100)
        sizes.append(size)
        if is_micro:
            micro_pts.append((size, auc * 100))
        elif is_edge:
            edge_pts.append((size, auc * 100))
        colors.append(C_MICRO if is_micro else C_EDGE if is_edge else C_OTHER)
        alphas.append(0.9 if (is_micro or is_edge) else 0.35)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Other models
    other_x = [s for s, c in zip(sizes, colors) if c == C_OTHER]
    other_y = [a for a, c in zip(aucs, colors) if c == C_OTHER]
    ax.scatter(other_x, other_y, s=25, color=C_OTHER, alpha=0.35, label="Other experiments")

    # Highlighted models
    if micro_pts:
        ax.scatter([p[0] for p in micro_pts], [p[1] for p in micro_pts],
                   s=180, marker="*", color=C_MICRO, edgecolors="black",
                   linewidth=1, zorder=10, label="SEABADNet-Micro (m16)")
    if edge_pts:
        ax.scatter([p[0] for p in edge_pts], [p[1] for p in edge_pts],
                   s=180, marker="D", color=C_EDGE, edgecolors="black",
                   linewidth=1, zorder=10, label="SEABADNet-Edge (m80)")

    # Budget lines
    ax.axvline(x=8,  color=C_MICRO, linestyle="--", alpha=0.5, linewidth=1.2, label="Micro budget (8 KB)")
    ax.axvline(x=35, color=C_EDGE,  linestyle="--", alpha=0.5, linewidth=1.2, label="Edge budget (35 KB)")

    ax.set_xlabel("Model Size (KB, TFLite INT8)", fontweight="bold")
    ax.set_ylabel("TFLite INT8 AUC (%)", fontweight="bold")
    ax.set_title("Accuracy vs Size: All Experiments (seed=42)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save(fig, "figure4_efficiency_scatter")


# ── Figure 5: Multi-seed Validation ──────────────────────────────────────────

def figure5_multiseed_validation():
    """
    Two subplots showing AUC mean ± std over 3 seeds across all n_mels values,
    for SEABADNet-Micro (6b_micro_improved) and SEABADNet-Edge (3f).
    Highlights the operating n_mels for each model.
    """
    print("\nFigure 5: Multi-seed validation...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, script, label, color, op_mels in [
        (axes[0], MICRO_SCRIPT, "SEABADNet-Micro (6b_micro_improved)", C_MICRO, MICRO_NMELS),
        (axes[1], EDGE_SCRIPT,  "SEABADNet-Edge (3f)",                  C_EDGE,  EDGE_NMELS),
    ]:
        means, stds = [], []
        for m in N_MELS:
            mn, sd = multi_seed(script, 1024, m)
            means.append(mn * 100 if mn else None)
            stds.append(sd * 100 if sd else 0.0)

        x = np.arange(len(N_MELS))
        vals  = [m if m is not None else float("nan") for m in means]
        errs  = [e for e in stds]

        ax.bar(x, vals, color=color, alpha=0.65, edgecolor="white", linewidth=0.8)
        ax.errorbar(x, vals, yerr=errs, fmt="none", color="black",
                    capsize=4, linewidth=1.2)

        # Highlight operating n_mels
        op_idx = N_MELS.index(op_mels)
        ax.bar(op_idx, vals[op_idx], color=color, alpha=0.95,
               edgecolor="black", linewidth=1.5)

        # Seed-level scatter
        for i, m in enumerate(N_MELS):
            for s in SEEDS:
                r = get(script, 1024, m, s)
                a = r.get("auc")
                if a:
                    ax.scatter(i, a * 100, s=20, color="black", alpha=0.6, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"m{m}" for m in N_MELS])
        ax.set_xlabel("n_mels", fontweight="bold")
        ax.set_ylabel("AUC (%) — mean ± std, seeds 42/100/786", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(95, 101)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate operating point
        if vals[op_idx]:
            ax.annotate(f"Operating\n(m{op_mels})\n{vals[op_idx]:.2f}%",
                        xy=(op_idx, vals[op_idx]),
                        xytext=(op_idx + 0.4, vals[op_idx] - 1.2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                        fontsize=8, ha="center")

    plt.tight_layout()
    save(fig, "figure5_multiseed_validation")


# ── Figure 6: Quantization Robustness ────────────────────────────────────────

def figure6_quant_robustness():
    """
    Float32 vs TFLite INT8 AUC for both final models across all n_mels,
    showing quantization degradation is consistently < 0.1%.
    """
    print("\nFigure 6: Quantization robustness...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, script, label, color, op_mels in [
        (axes[0], MICRO_SCRIPT,
         "SEABADNet-Micro: Float32 vs INT8 AUC (seed=42)", C_MICRO, MICRO_NMELS),
        (axes[1], EDGE_SCRIPT,
         "SEABADNet-Edge: Float32 vs INT8 AUC (seed=42)",  C_EDGE,  EDGE_NMELS),
    ]:
        float_aucs, tflite_aucs, degs = [], [], []
        for m in N_MELS:
            r = get(script, 1024, m)
            fa = r.get("auc")
            ta = r.get("tflite_auc")
            float_aucs.append(fa * 100 if fa else None)
            tflite_aucs.append(ta * 100 if ta else None)
            if fa and ta:
                degs.append((fa - ta) * 100)
            else:
                degs.append(None)

        x = np.arange(len(N_MELS))
        w = 0.35

        fa_plot = [v if v is not None else 0.0 for v in float_aucs]
        ta_plot = [v if v is not None else 0.0 for v in tflite_aucs]

        bars_f = ax.bar(x - w / 2, fa_plot, w, label="Float32", color=color,
                        alpha=0.5, edgecolor="white")
        bars_t = ax.bar(x + w / 2, ta_plot, w, label="TFLite INT8", color=color,
                        alpha=0.9, edgecolor="white")

        # Degradation annotations
        for i, deg in enumerate(degs):
            if deg is not None and fa_plot[i] > 0:
                sign = "+" if deg > 0 else ""
                ax.text(x[i], max(fa_plot[i], ta_plot[i]) + 0.05,
                        f"{sign}{deg:.3f}%",
                        ha="center", va="bottom", fontsize=7.5, color="black")

        # Mark operating n_mels
        op_idx = N_MELS.index(op_mels)
        ax.axvline(x=op_idx, color="black", linestyle=":", alpha=0.5, linewidth=1.2)
        ax.text(op_idx + 0.05, ax.get_ylim()[0] + 0.2, f"m{op_mels}\n(operating)",
                fontsize=7.5, va="bottom", color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([f"m{m}" for m in N_MELS])
        ax.set_xlabel("n_mels", fontweight="bold")
        ax.set_ylabel("AUC (%, seed=42)", fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ymin = min(v for v in fa_plot + ta_plot if v > 0) - 0.5
        ax.set_ylim(ymin, 101)

    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.97, wspace=0.3)
    save(fig, "figure6_quant_robustness")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SEABADNet — Paper Figure Generation")
    print(f"  SEABADNet-Micro : {MICRO_SCRIPT}, m{MICRO_NMELS}, τ={TAU}")
    print(f"  SEABADNet-Edge  : {EDGE_SCRIPT}, m{EDGE_NMELS}, τ={TAU}")
    print(f"  Seeds           : {SEEDS}")
    print("=" * 70)

    figure1_ablation_chain()
    figure2_nmels_sweep()
    figure3_dropout_sweep()
    figure4_efficiency_scatter()
    figure5_multiseed_validation()
    figure6_quant_robustness()

    print("\n" + "=" * 70)
    print("Done. Files written to figures/")
    print("  figure1_ablation_chain.png/.pdf")
    print("  figure2_nmels_sweep.png/.pdf")
    print("  figure3_dropout_sweep.png/.pdf")
    print("  figure4_efficiency_scatter.png/.pdf")
    print("  figure5_multiseed_validation.png/.pdf")
    print("  figure6_quant_robustness.png/.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
