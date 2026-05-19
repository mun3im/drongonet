#!/usr/bin/env python3
"""
run_final_validation.py — Final multi-seed validation sweep.

Runs after the 6b seed=42 sweep completes:
  - 6b_micro_improved  × n_mels [16,32,48,64,80] × seeds [42,100,786]
  - 3f_gap_focal_loss_freq_emph_pointwise (Edge) × same mels × same seeds

Skips any run where results_summary.txt already exists.
Logs to run_final_validation.log.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PYTHON      = "/home/muneim/miniconda3/envs/tf215_gpu/bin/python"
DATASET     = "/Volumes/Evo/seabad"
SCRIPT_DIR  = Path("/home/muneim/Dropbox/Conda/SEABADNet")
RESULTS_DIR = SCRIPT_DIR / "results"
LOG_FILE    = SCRIPT_DIR / "run_final_validation.log"

N_MELS  = [16, 32, 48, 64, 80]
SEEDS   = [42, 100, 786]

RUNS = [
    # (script_name, n_fft)
    ("6b_micro_improved",                        1024),
    ("3f_gap_focal_loss_freq_emph_pointwise",    1024),
]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_done(script: str, n_fft: int, n_mels: int, seed: int) -> bool:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
    return (d / "results_summary.txt").exists()


def wait_for_6b_seed42():
    """Block until all 6b seed=42 runs have results_summary.txt."""
    pending = [
        RESULTS_DIR / f"6b_micro_improved_fft1024_m{m}_s42"
        for m in N_MELS
    ]
    while True:
        remaining = [d for d in pending if not (d / "results_summary.txt").exists()]
        if not remaining:
            log("6b seed=42 sweep complete — starting final validation.")
            return
        names = [d.name for d in remaining]
        log(f"Waiting for 6b seed=42: {len(remaining)} pending: {names}")
        time.sleep(120)  # check every 2 minutes


def run_experiment(script: str, n_fft: int, n_mels: int, seed: int) -> bool:
    out_dir = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
    script_path = SCRIPT_DIR / f"{script}.py"
    if not script_path.exists():
        log(f"  SKIP (script not found): {script_path}")
        return False
    cmd = [
        PYTHON, str(script_path),
        "--n_mels",      str(n_mels),
        "--n_fft",       str(n_fft),
        "--random_seed", str(seed),
        "--dataset-path", DATASET,
    ]
    log(f"  RUN: {script} m{n_mels} s{seed}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), timeout=3600)
        elapsed = (time.time() - t0) / 60
        if result.returncode == 0:
            log(f"  OK:  {script} m{n_mels} s{seed} ({elapsed:.1f} min)")
            return True
        else:
            log(f"  FAIL (rc={result.returncode}): {script} m{n_mels} s{seed} ({elapsed:.1f} min)")
            return False
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: {script} m{n_mels} s{seed}")
        return False
    except Exception as e:
        log(f"  ERROR: {script} m{n_mels} s{seed} — {e}")
        return False


def main():
    log("=" * 60)
    log("Final validation sweep — 6b_micro_improved + 3f (Edge)")
    log(f"n_mels: {N_MELS}  |  seeds: {SEEDS}")
    log("=" * 60)

    # Build full work list
    work = [
        (script, n_fft, m, s)
        for script, n_fft in RUNS
        for m in N_MELS
        for s in SEEDS
    ]

    done_now  = sum(1 for sc, nf, m, s in work if is_done(sc, nf, m, s))
    todo_now  = len(work) - done_now
    log(f"Total: {len(work)}  already done: {done_now}  to run: {todo_now}")

    # Wait for ongoing 6b seed=42 sweep before proceeding
    wait_for_6b_seed42()

    failures = []
    total = len(work)
    for i, (script, n_fft, n_mels, seed) in enumerate(work, 1):
        tag = f"[{i}/{total}]"
        if is_done(script, n_fft, n_mels, seed):
            log(f"{tag} SKIP (done): {script} m{n_mels} s{seed}")
            continue
        log(f"{tag} Starting: {script} m{n_mels} s{seed}")
        ok = run_experiment(script, n_fft, n_mels, seed)
        if not ok:
            failures.append(f"{script}_fft{n_fft}_m{n_mels}_s{seed}")

    log("")
    log("=" * 60)
    log(f"Final validation complete. Failures ({len(failures)}): {failures}")
    log("=" * 60)


if __name__ == "__main__":
    main()
