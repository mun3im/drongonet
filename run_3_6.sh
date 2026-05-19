#!/bin/bash
# Run all 3*.py through 6*.py scripts sequentially
# Phase 3: conv type + filter count + loss function  (n_mels=64)
# Phase 4A: dropout sweep Edge track                 (n_mels=64)
# Phase 4B: dropout sweep Micro track                (n_mels=16)
# Phase 5: Micro filter count sweep                  (n_mels=16)
# Phase 6: final candidates                          (n_mels=16 for Micro, 64 for Edge)

PYTHON=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
WORKDIR=/home/muneim/Dropbox/Conda/SEABADNet
DATASET=/Volumes/Evo/seabad
LOGDIR=/tmp/seabadnet_logs
mkdir -p "$LOGDIR"
cd "$WORKDIR"

SUMMARY="$LOGDIR/summary.txt"
echo "" >> "$SUMMARY"
echo "=== Phase 3-6 run started: $(date) ===" >> "$SUMMARY"

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
}

# ── Phase 3: conv type + filter count + loss (n_mels=64) ──────────────────────
run_script "3a_depthwise"      3a_depthwise.py      --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3b_filters8"       3b_filters8.py       --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3c_focal_loss"     3c_gap_focal_loss.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3d_freq_emph"      3d_gap_freq_emphasis.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3e_freq_emph_ds"   3e_gap_freq_emph_ds.py  --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3f_fl_fe_pw"       3f_gap_focal_loss_freq_emph_pointwise.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3g_strided"        3g_strided_focal_tuned.py    --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3h_strided_no1x1"  3h_strided_focal_no1x1.py   --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "3i_strided_dw"     3i_strided_focal_depthwise.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42

# ── Phase 4A: dropout sweep — Edge track (Conv2D, 8 filters, n_mels=64) ───────
run_script "4a_dropout01" 4a_dropout01.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "4b_dropout02" 4b_dropout02.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "4c_dropout03" 4c_dropout03.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42
run_script "4d_dropout04" 4d_dropout04.py --dataset-path "$DATASET" --n_mels 64 --random_seed 42

# ── Phase 4B: dropout sweep — Micro track (Depthwise, 4 filters, n_mels=16) ──
run_script "4e_dw_drop01" 4e_depthwise_drop01.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42
run_script "4f_dw_drop02" 4f_depthwise_drop02.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42
run_script "4g_dw_drop03" 4g_depthwise_drop03.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42
run_script "4h_dw_drop04" 4h_depthwise_drop04.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42

# ── Phase 5: Micro filter count sweep (n_mels=16) ────────────────────────────
run_script "5a_dw_f6" 5a_depthwise_f6.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42
run_script "5b_dw_f5" 5b_depthwise_f5.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42

# ── Phase 6: final candidates ─────────────────────────────────────────────────
run_script "6a_micro_final" 6a_micro_final.py --dataset-path "$DATASET" --n_mels 16 --random_seed 42
run_script "6b_edge_final"  6b_edge_final.py  --dataset-path "$DATASET" --n_mels 64 --random_seed 42

echo "" >> "$SUMMARY"
echo "=== ALL RUNS COMPLETE: $(date) ===" | tee -a "$SUMMARY"
