#!/usr/bin/env python3
"""
retrain_phase0_on_seabad.py: Retrain all Phase 0 TinyChirp models on SEABAD dataset.

Retrains 5 architectures (CNN-Mel, CNN-Time, Transformer, SqueezeNet-Time, SqueezeNet-Mel)
on SEABAD to establish baseline performance and justify CNN-Mel as the primary model.

Metrics collected per model:
- Accuracy, AUC, Precision, Recall, F1
- Model size (float32, INT8)
- Inference latency (CPU, INT8)

Output: results_phase0_seabad_retrained/{model_name}_s{seed}/
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def run_model_retrain(script_name: str, dataset_path: str, seed: int = 42, force_reprocess: bool = False) -> bool:
    """
    Run a single Phase 0 model retraining on SEABAD.

    Args:
        script_name: e.g., "0a_tinychirp_cnnmel"
        dataset_path: path to SEABAD dataset
        seed: random seed
        force_reprocess: force reprocessing of mels

    Returns:
        True if successful, False otherwise
    """
    # 0a-0e baselines now live in the sibling pre-ablation/ folder (was develop/).
    script_path = Path(__file__).parent.parent / "pre-ablation" / f"{script_name}.py"
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    # Use conda run to ensure proper environment
    cmd = [
        "conda", "run", "-n", "tf215_gpu",
        "python", str(script_path),
        "--dataset-path", dataset_path,
        "--random_seed", str(seed)
    ]

    if force_reprocess:
        cmd.append("--force-reprocess")

    print(f"\n{'='*70}")
    print(f"Running: {script_name} (seed={seed})")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def collect_results(results_dir: str = "results_phase0_seabad_retrained") -> List[Dict]:
    """
    Aggregate results from all Phase 0 models trained on SEABAD.

    Looks for results_summary.txt files in output directories.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    models = {
        "0a": "CNN-Mel",
        "0b": "CNN-Time",
        "0c": "Transformer",
        "0d": "SqueezeNet-Time",
        "0e": "SqueezeNet-Mel"
    }

    results = []

    for code, model_name in models.items():
        # Find result directories for this model
        pattern = f"{code}_*_seabad*"
        dirs = list(results_path.glob(pattern))

        if not dirs:
            print(f"⚠️  No results found for {model_name} ({code})")
            continue

        for result_dir in dirs:
            summary_file = result_dir / "results_summary.txt"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = {}
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            summary[k.strip()] = v.strip()

                summary["model"] = model_name
                summary["code"] = code
                summary["result_dir"] = str(result_dir)
                results.append(summary)

    if len(results) > 0:
        print("\n" + "="*90)
        print("PHASE 0 BASELINE COMPARISON (SEABAD)")
        print("="*90)

        # Print header
        headers = ["model", "auc", "accuracy", "size_kb_int8", "latency_ms"]
        print(f"{'Model':<25} {'AUC':<12} {'Accuracy':<12} {'Size(KB)':<12} {'Latency(ms)':<12}")
        print("-" * 90)

        # Print rows
        for r in results:
            model = r.get("model", "?")
            auc = r.get("auc", "?")
            accuracy = r.get("accuracy", "?")
            size = r.get("size_kb_int8", "?")
            latency = r.get("latency_ms", "?")
            print(f"{model:<25} {auc:<12} {accuracy:<12} {size:<12} {latency:<12}")

        print("="*90)

        # Gate analysis
        print("\n📊 GATE ANALYSIS:")
        print("-" * 90)

        size_pass = []
        acc_pass = []
        both_pass = []

        for r in results:
            try:
                size = float(r.get("size_kb_int8", 0))
                auc = float(r.get("auc", 0))

                if size < 40:
                    size_pass.append(r["model"])
                if auc > 0.95:
                    acc_pass.append(r["model"])
                if size < 40 and auc > 0.95:
                    both_pass.append((r["model"], auc, size))
            except:
                pass

        print(f"✅ Size gate (< 40 KB INT8): {len(size_pass)} / {len(results)} models")
        if size_pass:
            print(f"   {', '.join(size_pass)}")

        print(f"✅ Accuracy gate (> 95% AUC): {len(acc_pass)} / {len(results)} models")
        if acc_pass:
            print(f"   {', '.join(acc_pass)}")

        print(f"✅ Both gates: {len(both_pass)} / {len(results)} models")
        if both_pass:
            best = max(both_pass, key=lambda x: x[1])
            print(f"   🏆 Best: {best[0]} (AUC={best[1]}, Size={best[2]} KB)")

        print("-" * 90)

    return results

def main():
    """Retrain all Phase 0 models on SEABAD."""
    from config import DATASET_PATH

    dataset_path = DATASET_PATH  # Should be /Volumes/Evo/SEABAD
    seeds = [42, 100, 786]

    print(f"\n🔧 PHASE 0 BASELINE RETRAINING ON SEABAD")
    print(f"   Dataset: {dataset_path}")
    print(f"   Seeds: {seeds}")

    # Scripts to retrain (in order)
    scripts = [
        "0a_tinychirp_cnnmel",
        "0b_tinychirp_cnntime",
        "0c_tinychirp_transformer",
        "0d_tinychirp_squeezenettime",
        "0e_tinychirp_squeezenetmel"
    ]

    # Modify each script's dataset_path and output_dir to point to SEABAD
    print(f"\n⚙️  Note: Each script will be invoked with --dataset-path {dataset_path}")
    print(f"   Scripts will auto-detect SEABAD vs TinyChirp and adjust output directories.\n")

    # Run each model with all seeds
    successful = []
    failed = []

    for script in scripts:
        for seed in seeds:
            success = run_model_retrain(script, dataset_path, seed=seed)
            if success:
                successful.append(f"{script}_s{seed}")
            else:
                failed.append(f"{script}_s{seed}")

    # Collect and display results
    print(f"\n\n📈 COLLECTING RESULTS...")
    results_list = collect_results("results_phase0_seabad_retrained")

    # Summary
    total_runs = len(scripts) * len(seeds)
    print(f"\n✅ Successful: {len(successful)} / {total_runs}")
    if successful:
        for s in successful[:10]:  # Show first 10
            print(f"   ✓ {s}")
        if len(successful) > 10:
            print(f"   ... and {len(successful) - 10} more")

    if failed:
        print(f"❌ Failed: {len(failed)} / {total_runs}")
        for f in failed[:10]:  # Show first 10
            print(f"   ✗ {f}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")

    print(f"\n💾 Results saved to: results_phase0_seabad_retrained/")
    print(f"   View collected metrics above for gate analysis.")

if __name__ == "__main__":
    main()
