#!/bin/bash
# Run all 1*.py and 2*.py scripts sequentially
# Logs per-script to /tmp/seabadnet_logs/
# Summary written to /tmp/seabadnet_logs/summary.txt

PYTHON=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
WORKDIR=/home/muneim/Dropbox/Conda/SEABADNet
DATASET=/Volumes/Evo/seabad
LOGDIR=/tmp/seabadnet_logs
mkdir -p "$LOGDIR"
cd "$WORKDIR"

SUMMARY="$LOGDIR/summary.txt"
echo "SEABADNet run started: $(date)" > "$SUMMARY"
echo "" >> "$SUMMARY"

run_script() {
    local label="$1"
    local logfile="$LOGDIR/${label}.log"
    shift
    echo "=== [$label] starting $(date) ===" | tee -a "$SUMMARY"
    "$PYTHON" "$@" 2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$label] SUCCESS" | tee -a "$SUMMARY"
    else
        echo "[$label] FAILED (exit $exit_code) — see $logfile" | tee -a "$SUMMARY"
    fi
    echo "" >> "$SUMMARY"
    return $exit_code
}

# Phase 1 — mel sweep
for MELS in 16 32 48 64 80; do
    run_script "1a_m${MELS}" 1a_baseline2d.py --dataset-path "$DATASET" --n_mels "$MELS" --random_seed 42
done

# Phase 1 — waveform scripts
run_script "1b_transformer"    1b_transformer.py    --dataset-path "$DATASET" --random_seed 42
run_script "1c_cnntime"        1c_cnntime.py        --dataset-path "$DATASET" --random_seed 42
run_script "1d_cnntime_gap"    1d_cnntime_gap.py    --dataset-path "$DATASET" --random_seed 42
run_script "1e_cnntime_enh"    1e_cnntime_enhanced.py --dataset-path "$DATASET" --random_seed 42
run_script "1f_cnntime_deeper" 1f_cnntime_deeper.py --dataset-path "$DATASET" --random_seed 42

# Phase 2 — GAP variants
run_script "2a_gap"      2a_baseline_gap.py          --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "2b_gap_lrn"  2b_baseline_gap_learned.py  --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "2c_gap_1x1"  2c_baseline_gap_1x1.py      --dataset-path "$DATASET" --n_mels 64 --random_seed 42

echo "=== All runs complete: $(date) ===" | tee -a "$SUMMARY"
