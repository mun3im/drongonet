#!/usr/bin/env python3
"""
run_phase1_sweep.py: Phase 1 — n_mels sweep on winning Phase 0 model (CNN-Mel).

Sweeps n_mels across 16, 32, 48, 64, 80 with seeds 42/100/786.
Tests: What frequency resolution does SEABAD need?

Output: results_phase1_baseline2d_seabad/{n_mels}_s{seed}/
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict

def run_n_mels_sweep(script_name: str, dataset_path: str, n_mels_values: List[int], seeds: List[int]) -> Dict[str, bool]:
    """
    Run Phase 1 n_mels sweep.

    Args:
        script_name: "1a_baseline2d"
        dataset_path: path to SEABAD dataset
        n_mels_values: [16, 32, 48, 64, 80]
        seeds: [42, 100, 786]

    Returns:
        Dict mapping "m{n_mels}_s{seed}" -> success bool
    """
    script_path = Path(__file__).parent / f"{script_name}.py"
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return {}

    results = {}
    total_runs = len(n_mels_values) * len(seeds)
    current = 0

    for n_mels in n_mels_values:
        for seed in seeds:
            current += 1
            cmd = [
                "conda", "run", "-n", "tf215_gpu",
                "python", str(script_path),
                "--dataset-path", dataset_path,
                "--n_mels", str(n_mels),
                "--random_seed", str(seed)
            ]

            run_id = f"m{n_mels}_s{seed}"
            print(f"\n{'='*70}")
            print(f"[{current}/{total_runs}] Running: {script_name} (n_mels={n_mels}, seed={seed})")
            print(f"{'='*70}")

            try:
                result = subprocess.run(cmd, cwd=Path(__file__).parent)
                results[run_id] = result.returncode == 0
                status = "✅" if results[run_id] else "❌"
                print(f"{status} {run_id}")
            except Exception as e:
                print(f"❌ Error: {e}")
                results[run_id] = False

    return results

def collect_phase1_results(results_dir: str = "results_phase1_baseline2d_seabad") -> None:
    """Aggregate Phase 1 n_mels sweep results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("\n" + "="*90)
    print("PHASE 1: N_MELS SWEEP ON SEABAD (CNN-Mel)")
    print("="*90)

    n_mels_values = [16, 32, 48, 64, 80]
    seeds = [42, 100, 786]

    # Collect AUC values
    results_table = {}
    for n_mels in n_mels_values:
        results_table[n_mels] = {"s42": None, "s100": None, "s786": None}

    for n_mels in n_mels_values:
        for seed in seeds:
            dir_name = f"1a_baseline2d_fft1024_m{n_mels}_s{seed}"
            summary_file = results_path / dir_name / "results_summary.txt"

            if summary_file.exists():
                with open(summary_file) as f:
                    for line in f:
                        if "auc=" in line.lower():
                            try:
                                auc_val = float(line.split("=")[1].strip())
                                results_table[n_mels][f"s{seed}"] = f"{auc_val:.4f}"
                            except:
                                pass

    # Print table
    print(f"{'n_mels':<10} {'Seed 42':<15} {'Seed 100':<15} {'Seed 786':<15} {'Mean':<15}")
    print("-" * 90)

    for n_mels in n_mels_values:
        s42 = results_table[n_mels]["s42"]
        s100 = results_table[n_mels]["s100"]
        s786 = results_table[n_mels]["s786"]

        # Compute mean if all exist
        mean_str = "—"
        if all([s42, s100, s786]):
            try:
                mean_val = (float(s42) + float(s100) + float(s786)) / 3
                mean_str = f"{mean_val:.4f}"
            except:
                pass

        print(f"{n_mels:<10} {str(s42):<15} {str(s100):<15} {str(s786):<15} {mean_str:<15}")

    print("-" * 90)
    print("\n💡 Gate 1 decision: Pick n_mels that maximizes validation AUC.")
    print("   Expected: n_mels=64 (from prior work), but confirm with results above.")

def main():
    """Run Phase 1 n_mels sweep."""
    from config import DATASET_PATH

    dataset_path = DATASET_PATH
    n_mels_values = [16, 32, 48, 64, 80]
    seeds = [42, 100, 786]

    print(f"\n🔧 PHASE 1: N_MELS SWEEP ON SEABAD (CNN-Mel)")
    print(f"   Dataset: {dataset_path}")
    print(f"   n_mels: {n_mels_values}")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {len(n_mels_values) * len(seeds)}\n")

    # Run sweep
    results = run_n_mels_sweep("1a_baseline2d", dataset_path, n_mels_values, seeds)

    # Summary
    successful = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n{'='*70}")
    print(f"✅ Successful: {successful} / {total}")
    if successful < total:
        print(f"❌ Failed: {total - successful} / {total}")
    print(f"{'='*70}")

    # Collect and display results
    print(f"\n📈 COLLECTING RESULTS...")
    collect_phase1_results("results_phase1_baseline2d_seabad")

    print(f"\n💾 Results saved to: results_phase1_baseline2d_seabad/")
    print(f"   Next: Use n_mels from highest AUC for Phase 2 (GAP variants).")

if __name__ == "__main__":
    main()
