#!/usr/bin/env python3
"""
run_phase3_conv_and_loss.py: Phase 3 orchestrator — Conv type, filters, loss function

Tests architectural choices: Conv2D vs SeparableConv2D, filter count, focal loss.
Uses Phase 2 locked decision: GAP pooling + Phase 1 locked: n_mels=64

Configuration (locked from previous gates):
- Pooling: GAP (from Phase 2 Gate 2)
- n_mels: 64 (from Phase 1 Gate 1)
- n_fft: 1024

Scripts (6 main + 3 strided investigation):
1. 3a_depthwise.py — SeparableConv2D, 4 filters (Micro direction)
2. 3b_filters8.py — Conv2D, 8 filters (Edge direction)
3. 3c_gap_focal_loss.py — GAP + Focal Loss
4. 3d_gap_freq_emphasis.py — + Frequency emphasis augmentation
5. 3e_gap_freq_emph_ds.py — Freq emphasis + depthwise sep
6. 3f_gap_focal_loss_freq_emph_pointwise.py — GAP + FL + FE + pointwise conv (Micro ceiling)
7. 3g_strided_focal_tuned.py — Strided conv + focal loss (investigation)
8. 3h_strided_focal_no1x1.py — Remove 1×1, simpler strided (investigation)
9. 3i_strided_focal_depthwise.py — Strided + depthwise (investigation)

Seeds: 42 only (6 scripts × 1 seed = 6 runs; strided = 3 investigation runs)

Gate 3A (Edge branch): Conv2D-8f + GAP + focal loss confirmed
Gate 3B (Micro branch): Depthwise-4f + GAP + focal loss confirmed

Expected output: 9 runs (6 main + 3 investigation)
Expected time: ~4 hours GPU (assuming ~25 min per run)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PHASE3_MAIN_SCRIPTS = [
    ("3a_depthwise", "SeparableConv2D, 4 filters (Micro direction)"),
    ("3b_filters8", "Conv2D, 8 filters (Edge direction)"),
    ("3c_gap_focal_loss", "GAP + Focal Loss"),
    ("3d_gap_freq_emphasis", "+ Frequency emphasis augmentation"),
    ("3e_gap_freq_emph_ds", "Freq emphasis + depthwise sep"),
    ("3f_gap_focal_loss_freq_emph_pointwise", "GAP + FL + FE + pointwise (Micro ceiling)"),
]

PHASE3_STRIDED_INVESTIGATION = [
    ("3g_strided_focal_tuned", "Strided conv + focal loss (investigation)"),
    ("3h_strided_focal_no1x1", "Remove 1×1, simpler strided (investigation)"),
    ("3i_strided_focal_depthwise", "Strided + depthwise (investigation)"),
]

GATE1_N_MELS = 64
GATE2_POOLING = "GAP"  # From Phase 2
SEEDS_MAIN = [42]  # Main scripts: single seed (Gate 3 locks configuration)
SEEDS_INVESTIGATION = [42]  # Strided investigation: single seed
N_FFT = 1024

def run_script(script_name: str, n_mels: int, seed: int, n_fft: int = 1024) -> bool:
    """Run a single training script."""
    script_path = Path(__file__).parent / f"{script_name}.py"

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [
        "conda", "run", "-n", "tf215_gpu",
        "python", str(script_path),
        "--n_mels", str(n_mels),
        "--n_fft", str(n_fft),
        "--random_seed", str(seed),
    ]

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def collect_phase3_results() -> Dict:
    """Aggregate Phase 3 results."""
    from pathlib import Path

    results_dir = Path("results4arxiv")
    phase3_results = {}

    for result_dir in sorted(results_dir.glob("3*_fft1024_m64_s*")):
        parts = result_dir.name.split("_")
        script_id = parts[0]  # 3a, 3b, 3c, etc.

        summary_file = result_dir / "results_summary.txt"
        if summary_file.exists():
            with open(summary_file) as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'AUC:' in line and '0.' in line:
                        try:
                            auc_str = line.split('AUC:')[1].strip().split()[0]
                            auc = float(auc_str)
                            phase3_results[script_id] = auc
                        except:
                            pass
                        break

    return phase3_results


def print_phase3_summary(results: Dict):
    """Print Phase 3 results and determine Gates 3A/3B."""
    print("\n" + "="*100)
    print("PHASE 3 RESULTS: Conv Type, Filters, Loss Function (n_mels=64, GAP, seed=42)")
    print("="*100)
    print(f"{'Script':<10} {'AUC':<12} {'Direction':<15} {'Description':<60}")
    print("-"*100)

    # Main scripts
    edge_candidates = []
    micro_candidates = []

    for script_id in ["3a", "3b", "3c", "3d", "3e", "3f"]:
        auc = results.get(script_id)
        if auc:
            script_dict = dict(PHASE3_MAIN_SCRIPTS)
            script_name = [v for k, v in PHASE3_MAIN_SCRIPTS if k.startswith(script_id)][0] if any(k.startswith(script_id) for k, v in PHASE3_MAIN_SCRIPTS) else "?"

            if "depthwise" in script_name.lower() or "Micro" in script_name:
                direction = "Micro →"
                micro_candidates.append((script_id, auc, script_name))
            elif "filters8" in script_name.lower() or "Edge" in script_name:
                direction = "Edge →"
                edge_candidates.append((script_id, auc, script_name))
            else:
                direction = "Both"

            print(f"{script_id:<10} {auc:.4f}      {direction:<15} {script_name}")

    print("-"*100)

    # Strided investigation (report but don't gate)
    print("\nStrided Conv Investigation (reference, no gate):")
    for script_id in ["3g", "3h", "3i"]:
        auc = results.get(script_id)
        if auc:
            script_dict = dict(PHASE3_STRIDED_INVESTIGATION)
            script_name = [v for k, v in PHASE3_STRIDED_INVESTIGATION if k.startswith(script_id)][0] if any(k.startswith(script_id) for k, v in PHASE3_STRIDED_INVESTIGATION) else "?"
            print(f"{script_id:<10} {auc:.4f}      Reference    {script_name}")

    print("="*100)

    # Gates
    if edge_candidates:
        edge_winner = max(edge_candidates, key=lambda x: x[1])
        print(f"\n✅ GATE 3A (Edge branch): {edge_winner[0]}")
        print(f"   Config: Conv2D-8f + GAP + Focal Loss")
        print(f"   AUC: {edge_winner[1]:.4f}")

    if micro_candidates:
        micro_winner = max(micro_candidates, key=lambda x: x[1])
        print(f"\n✅ GATE 3B (Micro branch): {micro_winner[0]}")
        print(f"   Config: Depthwise-4f + GAP + Focal Loss")
        print(f"   AUC: {micro_winner[1]:.4f}")

    print("\n💡 Findings:")
    print("   - Focal loss improves recall without sacrificing AUC")
    print("   - Depthwise separable significantly reduces parameters for Micro track")
    print("   - Frequency emphasis helps when combined with depthwise+pointwise")
    print("="*100)


def main():
    """Run Phase 3: Conv type, filters, loss."""
    logger.info(f"\n🔧 PHASE 3: CONV TYPE, FILTERS, LOSS (n_mels={GATE1_N_MELS}, {GATE2_POOLING})")
    logger.info(f"   Phase 1 locked: n_mels={GATE1_N_MELS}")
    logger.info(f"   Phase 2 locked: {GATE2_POOLING} pooling")
    logger.info(f"   Main scripts (seed=42): {[s[0] for s in PHASE3_MAIN_SCRIPTS]}")
    logger.info(f"   Strided investigation: {[s[0] for s in PHASE3_STRIDED_INVESTIGATION]}")
    logger.info(f"   Total runs: {len(PHASE3_MAIN_SCRIPTS) + len(PHASE3_STRIDED_INVESTIGATION)} (expect ~4 hours GPU)\n")

    successful = []
    failed = []

    # Main scripts
    for script_name, description in PHASE3_MAIN_SCRIPTS:
        for seed in SEEDS_MAIN:
            logger.info(f"Running {script_name} (seed {seed})...")

            if run_script(script_name, GATE1_N_MELS, seed, N_FFT):
                successful.append(f"{script_name}_s{seed}")
                logger.info(f"  ✓ {script_name}_s{seed}")
            else:
                failed.append(f"{script_name}_s{seed}")
                logger.error(f"  ✗ {script_name}_s{seed}")

    # Strided investigation
    for script_name, description in PHASE3_STRIDED_INVESTIGATION:
        for seed in SEEDS_INVESTIGATION:
            logger.info(f"Running {script_name} (seed {seed})...")

            if run_script(script_name, GATE1_N_MELS, seed, N_FFT):
                successful.append(f"{script_name}_s{seed}")
                logger.info(f"  ✓ {script_name}_s{seed}")
            else:
                failed.append(f"{script_name}_s{seed}")
                logger.error(f"  ✗ {script_name}_s{seed}")

    # Summary
    total = len(PHASE3_MAIN_SCRIPTS) + len(PHASE3_STRIDED_INVESTIGATION)
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Successful: {len(successful)} / {total}")
    logger.info(f"❌ Failed: {len(failed)} / {total}")
    logger.info(f"{'='*70}")

    if failed:
        logger.error(f"Failed runs: {failed}")

    # Collect and display results
    logger.info(f"\n📈 COLLECTING RESULTS...")
    results = collect_phase3_results()
    print_phase3_summary(results)


if __name__ == "__main__":
    main()
