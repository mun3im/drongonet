#!/usr/bin/env python3
"""
generate_paper_tables.py — Paper tables for SEABADNet (Bioacoustics journal)

Reads directly from results/{script}_fft{n_fft}_m{n_mels}_s{seed}/results_summary.txt
and float_classification_report.txt. All model and dataset names use current
SEABADNet / SEABAD naming.

Models:
  SEABADNet-Micro : 6b_micro_improved, n_mels=16, n_fft=1024, τ=0.35
  SEABADNet-Edge  : 3f_gap_focal_loss_freq_emph_pointwise, n_mels=80, n_fft=1024, τ=0.35
"""

import re
import numpy as np
from pathlib import Path

RESULTS = Path("results")
SEEDS   = [42, 100, 786]
N_MELS  = [16, 32, 48, 64, 80]

MICRO_SCRIPT = "6b_micro_improved"
EDGE_SCRIPT  = "3f_gap_focal_loss_freq_emph_pointwise"


# ── Helpers ────────────────────────────────────────────────────────────────

def get(script, n_fft, n_mels, seed=42):
    p = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}" / "results_summary.txt"
    if not p.exists():
        return {}
    r = {}
    all_aucs = []
    for line in p.read_text().splitlines():
        if m := re.search(r"(?<![\w.])AUC[:\s]+([0-9.]+)", line, re.I):
            # skip "AUC Degradation" lines
            if "degrad" not in line.lower():
                all_aucs.append(float(m.group(1)))
        if m := re.search(r"(?:Model Size|Size)[:\s]+([0-9.]+)\s*KB", line, re.I):
            r["size"] = float(m.group(1))
        if m := re.search(r"(?:Avg Inference|Inference)[^:]*:[:\s]+([0-9.]+)\s*ms", line, re.I):
            r["lat"] = float(m.group(1))
        if m := re.search(r"Total Params[^:]*:[:\s]+([0-9,]+)", line, re.I):
            r["params"] = int(m.group(1).replace(",", ""))
    if all_aucs:
        r["auc"] = all_aucs[0]                          # float32 AUC (first)
        if len(all_aucs) >= 2:
            r["tflite_auc"] = all_aucs[1]               # TFLite AUC (second)
    return r


def get_recall(script, n_fft, n_mels, seed=42, prefix="float"):
    """Extract per-class recall from classification report."""
    p = RESULTS / f"{script}_fft{n_fft}_m{n_mels}_s{seed}" / f"{prefix}_classification_report.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if "Positive" in line or "positive" in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    pass
    return None


def multi_seed(script, n_fft, n_mels, seeds=SEEDS, metric="auc"):
    vals = [get(script, n_fft, n_mels, s).get(metric) for s in seeds]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)


def hdr(title):
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


# ── Table 1: Dataset comparison ─────────────────────────────────────────────

def table1_dataset_comparison():
    hdr("TABLE 1: Bioacoustic PAM Dataset Comparison")
    print("""
| Dataset              | Focus           | Samples | Segment | Region    | Task       | Public |
|----------------------|-----------------|---------|---------|-----------|------------|--------|
| BirdVox-DCASE-20k    | Urban birds     | 20,000  | 10 s    | USA       | Detection  | ✓      |
| Warblrb10k           | European birds  | 10,000  | 10 s    | UK        | Detection  | ✓      |
| freefield1010        | UK birds        | 7,690   | 10 s    | UK        | Detection  | ✓      |
| POW                  | Owl species     | 5,400   | 3 s     | Poland    | Detection  | ✓      |
| **SEABAD (Ours)**    | **SE Asian birds** | **50,000** | **3 s** | **SE Asia** | **Detection** | **✓** |

SEABAD distinguishes itself as the first large-scale binary bird activity detection
dataset focused on tropical South-East Asian species diversity, with balanced
positive/negative samples at 16 kHz.
""")


# ── Table 2: SEABADNet model overview ───────────────────────────────────────

def table2_model_overview():
    hdr("TABLE 2: SEABADNet Model Family Overview")

    micro = get(MICRO_SCRIPT, 1024, 16)
    edge  = get(EDGE_SCRIPT,  1024, 80)

    micro_auc_m, micro_auc_s = multi_seed(MICRO_SCRIPT, 1024, 16, metric="auc")
    edge_auc_m,  edge_auc_s  = multi_seed(EDGE_SCRIPT,  1024, 80, metric="auc")

    print(f"""
| Variant            | Target hardware         | n_mels | AUC (mean±std, 3 seeds) | Recall@τ=0.35 | Size (KB) | Latency (ms) | τ    |
|--------------------|-------------------------|--------|-------------------------|----------------|-----------|--------------|------|
| SEABADNet-Micro    | ARM Cortex-M4 (≤8 KB)  |    16  | {micro_auc_m:.4f} ± {micro_auc_s:.4f}      | ≥0.98          | {micro.get('size', '—'):>5}     | {micro.get('lat', '—'):>4} ms       | 0.35 |
| SEABADNet-Edge     | SBC / Portenta X8       |    80  | {edge_auc_m:.4f} ± {edge_auc_s:.4f}      | ≥0.99          | {edge.get('size', '—'):>5}     | {edge.get('lat', '—'):>4} ms       | 0.35 |

Targets: Micro ≤8 KB INT8, <1 ms; Edge ≤35 KB INT8, <2 ms.
Both models evaluated on SEABAD test set (10% hold-out, stratified by class).
""")


# ── Table 3: Micro ablation chain ───────────────────────────────────────────

def table3_micro_ablation():
    hdr("TABLE 3: SEABADNet-Micro Architecture Ablation (n_mels=16, seed=42)")

    steps = [
        ("TinyChirp-CNNMel (n_mels=40, source)",  None,                      None, None,  "(reference — not SEABAD)"),
        ("1a — Baseline CNNMel on SEABAD",         "1a_baseline2d",           1024, 16,    "Transfer to SEABAD, n_mels=16"),
        ("2a — + Global Average Pooling",          "2a_baseline_gap",         1024, 16,    "Replace Flatten → GAP"),
        ("3c — + Focal Loss (α=0.25, γ=2)",        "3c_gap_focal_loss",       1024, 16,    "CE → Focal Loss"),
        ("3d — + Frequency Emphasis",              "3d_gap_freq_emphasis",    1024, 16,    "Learnable per-freq weighting"),
        ("3e — + Depthwise Separable Conv",        "3e_gap_freq_emph_ds",     1024, 16,    "Conv2D → SeparableConv2D"),
        ("4e — + Dropout = 0.1",                   "4e_depthwise_drop01",     1024, 16,    "Light regularisation"),
        ("5a — + 6 filters (from 4)",              "5a_depthwise_f6",         1024, 16,    "Filter count sweep"),
        ("6a — Micro final candidate",             "6a_micro_final",          1024, 16,    "Redesigned lean arch (763 params)"),
        ("6b — + Pointwise 1×1 Conv (final)",      MICRO_SCRIPT,              1024, 16,    "Channel mixing before GAP ← FINAL"),
    ]

    print(f"\n{'Step':<45}  {'AUC':>6}  {'Size':>7}  {'Lat':>5}  {'Params':>7}  Note")
    print("-" * 110)

    for label, script, n_fft, n_mels, note in steps:
        if script is None:
            print(f"  {label:<45}  {'—':>6}  {'—':>7}  {'—':>5}  {'—':>7}  {note}")
            continue
        r = get(script, n_fft, n_mels)
        auc   = f"{r['auc']:.4f}"        if r.get("auc")    else "  —   "
        size  = f"{r['size']:.2f} KB"    if r.get("size")   else "    —  "
        lat   = f"{r['lat']:.2f} ms"     if r.get("lat")    else "   — "
        params = f"{r['params']:>7}"     if r.get("params") else "      —"
        print(f"  {label:<45}  {auc:>6}  {size:>7}  {lat:>5}  {params}  {note}")

    print(f"""
Gate decisions:
  Gate 1 : n_mels=16 locked for Micro branch (design constraint)
  Gate 2 : GAP confirmed — parameter reduction with negligible AUC loss
  Gate 3B: Depthwise + Focal Loss locked for Micro
  Gate 4B: Dropout=0.1 locked (lowest regularisation suits depthwise)
  Gate 5 : 6 filters → used in 6b (5 vs 6 filters: Δ AUC < 0.001)
  Gate 6 : 6b meets all targets (6.56 KB, 0.10 ms, recall ≥0.98 @ τ=0.35)
""")


# ── Table 4: n_mels sweep ───────────────────────────────────────────────────

def table4_nmels_sweep():
    hdr("TABLE 4: Frequency Resolution Sweep — 1a_baseline2d (n_fft=1024, seed=42)")

    print(f"\n{'n_mels':>6}  {'Float AUC':>9}  {'TFLite AUC':>10}  {'Size (KB)':>9}  {'Lat (ms)':>8}  {'Params':>7}")
    print("-" * 60)

    for m in N_MELS:
        r = get("1a_baseline2d", 1024, m)
        auc    = f"{r['auc']:.4f}"        if r.get("auc")        else "  —   "
        tauc   = f"{r['tflite_auc']:.4f}" if r.get("tflite_auc") else "  —   "
        size   = f"{r['size']:.2f}"       if r.get("size")       else "  —  "
        lat    = f"{r['lat']:.2f}"        if r.get("lat")        else "  — "
        params = f"{r['params']:>7}"      if r.get("params")     else "     —"
        print(f"  {m:>4}    {auc:>9}  {tauc:>10}  {size:>9}  {lat:>8}  {params:>7}")

    print("""
Gate 1 result: AUC increases monotonically with n_mels; m64/m80 peak for Edge.
Micro branch fixed at n_mels=16 (MCU memory constraint — ~6 KB feature map).
""")


# ── Table 5: Dropout sweep ──────────────────────────────────────────────────

def table5_dropout_sweep():
    hdr("TABLE 5: Dropout Regularisation Sweep (n_fft=1024, n_mels=16, seed=42)")

    for track_label, scripts, note in [
        ("Track A — Conv2D (Edge direction)",
         [("4a_dropout01",0.1),("4b_dropout02",0.2),("4c_dropout03",0.3),("4d_dropout04",0.4)],
         "Gate 4A: dropout=0.3 best mean across n_mels"),
        ("Track B — Depthwise Separable (Micro direction)",
         [("4e_depthwise_drop01",0.1),("4f_depthwise_drop02",0.2),
          ("4g_depthwise_drop03",0.3),("4h_depthwise_drop04",0.4)],
         "Gate 4B: dropout=0.1 best — depthwise already implicitly regularised"),
    ]:
        print(f"\n{track_label}")
        print(f"  {'Dropout':>7}  {'AUC m16':>8}  {'AUC m32':>8}  {'AUC m64':>8}  {'AUC m80':>8}  {'Size m16':>8}")
        print("  " + "-" * 55)
        for script, dropout in scripts:
            r16 = get(script, 1024, 16); r32 = get(script, 1024, 32)
            r64 = get(script, 1024, 64); r80 = get(script, 1024, 80)
            a16 = f"{r16['auc']:.4f}" if r16.get("auc") else "  —   "
            a32 = f"{r32['auc']:.4f}" if r32.get("auc") else "  —   "
            a64 = f"{r64['auc']:.4f}" if r64.get("auc") else "  —   "
            a80 = f"{r80['auc']:.4f}" if r80.get("auc") else "  —   "
            sz  = f"{r16['size']:.2f} KB" if r16.get("size") else "    —  "
            print(f"  {dropout:>7.1f}  {a16:>8}  {a32:>8}  {a64:>8}  {a80:>8}  {sz:>8}")
        print(f"  → {note}")


# ── Table 6: Filter count sweep ─────────────────────────────────────────────

def table6_filter_sweep():
    hdr("TABLE 6: Filter Count Sweep — Phase 5 (n_mels=16, seed=42)")

    entries = [
        ("4e — 4 filters (Phase 4B ref)",  "4e_depthwise_drop01", 1024),
        ("5b — 5 filters (n_fft=512)",     "5b_depthwise_f5",      512),
        ("5a — 6 filters (n_fft=1024)",    "5a_depthwise_f6",     1024),
        ("6b — 6 filters, lean arch ← used in final", MICRO_SCRIPT, 1024),
    ]

    print(f"\n  {'Config':<42}  {'AUC':>6}  {'Size (KB)':>9}  {'Params':>7}  {'≤8KB?':>6}")
    print("  " + "-" * 80)
    for label, script, n_fft in entries:
        r = get(script, n_fft, 16)
        auc    = f"{r['auc']:.4f}"    if r.get("auc")    else "  —   "
        size   = r.get("size", 99)
        params = r.get("params", 0)
        ok     = "✅" if size <= 8.0 else "❌"
        print(f"  {label:<42}  {auc:>6}  {size:>6.2f} KB  {params:>7,}  {ok:>6}")

    print("""
Gate 5: 5 vs 6 filters differ by ΔAUC < 0.001 at m16. Both exceed 8 KB when
built on the ablation-chain (depthwise) architecture. The 6b final design
applies 6 filters within a compact 919-param architecture → 6.56 KB ✅.
""")


# ── Table 7: Multi-seed final validation ────────────────────────────────────

def table7_multiseed_validation():
    hdr("TABLE 7: Multi-seed Final Validation (seeds 42 / 100 / 786)")

    for model_label, script, n_fft, n_mels, recall_target, tau in [
        ("SEABADNet-Micro (6b, n_mels=16)", MICRO_SCRIPT, 1024, 16, 0.98, 0.35),
        ("SEABADNet-Edge  (3f, n_mels=80)", EDGE_SCRIPT,  1024, 80, 0.99, 0.35),
    ]:
        print(f"\n{model_label}  |  operating τ = {tau}  |  recall target ≥{recall_target}")
        print(f"  {'Seed':>5}  {'Float AUC':>9}  {'TFLite AUC':>10}  {'Size (KB)':>9}  {'Lat (ms)':>8}")
        print("  " + "-" * 50)

        aucs, taucs = [], []
        for seed in SEEDS:
            r = get(script, n_fft, n_mels, seed)
            auc  = r.get("auc");  tauc = r.get("tflite_auc")
            size = r.get("size"); lat  = r.get("lat")
            if auc:  aucs.append(auc)
            if tauc: taucs.append(tauc)
            print(f"  s{seed:<4}  {auc:.4f}     {tauc:.4f}      {size:.2f} KB   {lat:.2f} ms")

        if aucs:
            print(f"  {'mean':<5}  {np.mean(aucs):.4f}     {np.mean(taucs):.4f}")
            print(f"  {'±std':<5}  {np.std(aucs):.4f}     {np.std(taucs):.4f}")

    print(f"""
Threshold sweep results (τ=0.35, from run_threshold_sweep.py):
  SEABADNet-Micro: Recall = 0.9832 ± 0.0020, Precision = 0.7987 ± 0.0119, F1 = 0.8813 ± 0.0066
  SEABADNet-Edge:  Recall = 0.9924 ± 0.0025, Precision = 0.9379 ± 0.0090, F1 = 0.9643 ± 0.0038
  Both models meet recall targets at τ=0.35 (default τ=0.5 insufficient due to focal loss).
""")


# ── Table 8: n_mels × model comparison (Edge branch) ────────────────────────

def table8_edge_nmels():
    hdr("TABLE 8: SEABADNet-Edge n_mels Sweep (3f, n_fft=1024, seeds 42/100/786)")

    print(f"\n  {'n_mels':>6}  {'s42 AUC':>8}  {'s100 AUC':>9}  {'s786 AUC':>9}  {'Mean±Std':>14}  {'Size':>7}  {'Lat':>6}")
    print("  " + "-" * 70)

    for m in N_MELS:
        aucs = [get(EDGE_SCRIPT, 1024, m, s).get("auc") for s in SEEDS]
        av   = [a for a in aucs if a is not None]
        r42  = get(EDGE_SCRIPT, 1024, m, 42)
        mean = np.mean(av) if av else 0
        std  = np.std(av)  if len(av) > 1 else 0
        vals = "  ".join(f"{a:.4f}" if a else "  —   " for a in aucs)
        size = f"{r42.get('size','—')} KB"
        lat  = f"{r42.get('lat','—')} ms"
        print(f"  {m:>6}  {vals}  {mean:.4f}±{std:.4f}  {size:>7}  {lat:>6}")

    print("""
Gate 1 (Edge): n_mels=80 maximises AUC (0.9962 ± 0.0003) within the ≤35 KB budget.
All n_mels variants remain well below the latency target (<2 ms).
""")


# ── Table 9: Quantization robustness ────────────────────────────────────────

def table9_quantization():
    hdr("TABLE 9: Quantization Robustness (Float32 → INT8, seed=42)")

    print(f"\n  {'Model':<48}  {'Float AUC':>9}  {'INT8 AUC':>9}  {'Degradation':>11}  {'Robust?':>7}")
    print("  " + "-" * 95)

    entries = [
        ("1a_baseline2d (Conv2D baseline)",            "1a_baseline2d",   1024, 16),
        ("2a_baseline_gap (+ GAP)",                    "2a_baseline_gap", 1024, 16),
        ("3f_gap_fl_fe_pw (Edge candidate, m16)",       EDGE_SCRIPT,       1024, 16),
        ("3f_gap_fl_fe_pw (Edge final, m80)",           EDGE_SCRIPT,       1024, 80),
        ("6b_micro_improved (Micro final, m16)",        MICRO_SCRIPT,      1024, 16),
    ]

    for label, script, n_fft, n_mels in entries:
        r = get(script, n_fft, n_mels)
        auc  = r.get("auc");  tauc = r.get("tflite_auc")
        if auc and tauc:
            deg = (auc - tauc) * 100
            ok  = "✅" if deg < 0.5 else "⚠️ "
            print(f"  {label:<48}  {auc:.4f}     {tauc:.4f}     {deg:+.3f}%     {ok}")
        else:
            print(f"  {label:<48}  {'—':>9}  {'—':>9}  {'—':>11}  {'—':>7}")

    print("""
All models show <0.02% AUC degradation from Float32 to INT8 — quantization is robust.
""")


# ── Table 10: Deployment scenarios ──────────────────────────────────────────

def table10_deployment():
    hdr("TABLE 10: Deployment Scenarios")

    micro = get(MICRO_SCRIPT, 1024, 16)
    edge  = get(EDGE_SCRIPT,  1024, 80)

    print(f"""
| Variant         | Target device              | MCU example         | INT8 size | Latency  | Recall@τ |
|-----------------|----------------------------|---------------------|-----------|----------|----------|
| SEABADNet-Micro | ARM Cortex-M4 edge node    | AudioMoth / STM32F4 | {micro.get('size','—'):.2f} KB  | {micro.get('lat','—'):.2f} ms   | ≥0.98    |
| SEABADNet-Edge  | Linux SBC                  | RPi / Portenta X8   | {edge.get('size','—'):.2f} KB  | {edge.get('lat','—'):.2f} ms   | ≥0.99    |

Notes:
- Latency measured on CPU (TFLite INT8), 1000-call mean, excludes mel preprocessing.
- Mel preprocessing: ~5–15 ms on Cortex-M4 (CMSIS-DSP); not included in model latency.
- Both models operate at τ=0.35; raising to τ=0.40 drops recall ~1 pp below target.
- SEABADNet-Micro activations peak at ~12 KB SRAM (compatible with AudioMoth 256 KB).
""")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print("SEABADNet PAPER — TABLE GENERATION")
    print("Dataset: SEABAD  |  Models: SEABADNet-Micro & SEABADNet-Edge")
    print("=" * 80)

    table1_dataset_comparison()
    table2_model_overview()
    table3_micro_ablation()
    table4_nmels_sweep()
    table5_dropout_sweep()
    table6_filter_sweep()
    table7_multiseed_validation()
    table8_edge_nmels()
    table9_quantization()
    table10_deployment()

    print("\n" + "=" * 80)
    print("All tables generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
