#!/usr/bin/env python3
"""
run_full_sweep.py — Run all ablation phases 2-6 for n_mels ∈ {16,32,48,64,80}.
Skips runs where results_summary.txt already exists.
Logs to run_full_sweep.log.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PYTHON = "/home/muneim/miniconda3/envs/tf215_gpu/bin/python"
DATASET_PATH = "/Volumes/Evo/seabad"
SEED = 42
N_MELS = [16, 32, 48, 64, 80]
RESULTS_DIR = Path("/home/muneim/Dropbox/Conda/SEABADNet/results")
SCRIPT_DIR = Path("/home/muneim/Dropbox/Conda/SEABADNet")
LOG_FILE = SCRIPT_DIR / "run_full_sweep.log"

# Each entry: (script_name, n_fft)
# Phase 2 — GAP variants
PHASE2 = [
    ("2a_baseline_gap",         1024),
    ("2b_baseline_gap_learned", 1024),
    ("2c_baseline_gap_1x1",     1024),
]

# Phase 3 — Conv type, filter count, loss
PHASE3 = [
    ("3a_depthwise",                           1024),
    ("3b_filters8",                            1024),
    ("3c_gap_focal_loss",                      1024),
    ("3d_gap_freq_emphasis",                   1024),
    ("3e_gap_freq_emph_ds",                    1024),
    ("3f_gap_focal_loss_freq_emph_pointwise",  1024),
    ("3g_strided_focal_tuned",                 1024),
    ("3h_strided_focal_no1x1",                 1024),
    ("3i_strided_focal_depthwise",             1024),
]

# Phase 4A — Edge dropout sweep (Conv2D track)
PHASE4A = [
    ("4a_dropout01", 1024),
    ("4b_dropout02", 1024),
    ("4c_dropout03", 1024),
    ("4d_dropout04", 1024),
]

# Phase 4B — Micro dropout sweep (Depthwise track)
PHASE4B = [
    ("4e_depthwise_drop01", 1024),
    ("4f_depthwise_drop02", 1024),
    ("4g_depthwise_drop03", 1024),
    ("4h_depthwise_drop04", 1024),
]

# Phase 5 — Micro filter count sweep
PHASE5 = [
    ("5a_depthwise_f6", 1024),
    ("5b_depthwise_f5", 512),   # 5b uses n_fft=512 by default
]

# Phase 6 — Final candidates
PHASE6 = [
    ("6a_micro_final", 1024),
    ("6b_edge_final",  1024),
]

ALL_SCRIPTS = PHASE2 + PHASE3 + PHASE4A + PHASE4B + PHASE5 + PHASE6


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_done(script_name: str, n_fft: int, n_mels: int) -> bool:
    out_dir = RESULTS_DIR / f"{script_name}_fft{n_fft}_m{n_mels}_s{SEED}"
    return (out_dir / "results_summary.txt").exists()


def run_experiment(script_name: str, n_fft: int, n_mels: int) -> bool:
    """Run one experiment. Returns True on success."""
    out_dir = RESULTS_DIR / f"{script_name}_fft{n_fft}_m{n_mels}_s{SEED}"
    script = SCRIPT_DIR / f"{script_name}.py"

    if not script.exists():
        log(f"  SKIP (script not found): {script}")
        return False

    cmd = [
        PYTHON, str(script),
        "--n_mels",      str(n_mels),
        "--n_fft",       str(n_fft),
        "--random_seed", str(SEED),
        "--dataset-path", DATASET_PATH,
    ]
    log(f"  RUN: {script_name} n_mels={n_mels} n_fft={n_fft}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            timeout=3600,  # 1 hour max per run
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            log(f"  OK:  {script_name} m{n_mels} ({elapsed/60:.1f} min)")
            return True
        else:
            log(f"  FAIL (rc={result.returncode}): {script_name} m{n_mels} ({elapsed/60:.1f} min)")
            return False
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: {script_name} m{n_mels}")
        return False
    except Exception as e:
        log(f"  ERROR: {script_name} m{n_mels} — {e}")
        return False


def main():
    log("=" * 60)
    log("Full n_mels sweep — phases 2-6")
    log(f"n_mels: {N_MELS}  |  seed: {SEED}  |  dataset: {DATASET_PATH}")
    log("=" * 60)

    # Build work list
    work = []
    for script_name, n_fft in ALL_SCRIPTS:
        for n_mels in N_MELS:
            work.append((script_name, n_fft, n_mels))

    # Report plan
    todo = [(s, f, m) for s, f, m in work if not is_done(s, f, m)]
    done = [(s, f, m) for s, f, m in work if is_done(s, f, m)]
    log(f"Already done: {len(done)}  |  To run: {len(todo)}")
    log("")

    failures = []
    for i, (script_name, n_fft, n_mels) in enumerate(work):
        if is_done(script_name, n_fft, n_mels):
            log(f"[{i+1}/{len(work)}] SKIP (done): {script_name} m{n_mels}")
            continue
        log(f"[{i+1}/{len(work)}] Starting: {script_name} m{n_mels}")
        ok = run_experiment(script_name, n_fft, n_mels)
        if not ok:
            failures.append(f"{script_name}_fft{n_fft}_m{n_mels}")

    log("")
    log("=" * 60)
    log(f"Sweep complete. Failures ({len(failures)}): {failures}")
    log("=" * 60)


if __name__ == "__main__":
    main()
