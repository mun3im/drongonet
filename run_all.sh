#!/bin/bash
# Run ALL ablation experiments across ALL n_mels values (16, 32, 48, 64, 80).
#
# Structure:
#   Phase 1   — n_mels sweep baseline (1a × 5 mels)
#   Phase 1b  — waveform reference experiments (1b–1f, run once, no n_mels)
#   Phase 2   — GAP pooling variants (2a–2c × 5 mels)
#   Phase 3   — Conv type / filter count / loss function (3a–3i × 5 mels)
#   Phase 4   — Dropout sweep, Conv2D track (4a–4d × 5 mels)
#               Dropout sweep, Depthwise track (4e–4h × 5 mels)
#   Phase 5   — Micro filter count sweep (5a–5b × 5 mels)
#   Phase 6   — Final candidates (6a–6b × 5 mels)
#   Alt       — Alternative compact 96-timestep designs (alt_* × 5 mels)
#   Cap       — Capacity reference points (cap_* × 5 mels)
#   Rej       — Confirmed-rejected configs, run for Table 3 (rej_* × 5 mels)
#
# Total: ~180 runs. Logs per-run to LOGDIR; rolling summary appended to
# LOGDIR/summary.txt. Each run is independent — failures are logged and skipped.

PYTHON=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
WORKDIR=/home/muneim/Dropbox/Conda/SEABADNet
DATASET=/Volumes/Evo/seabad
LOGDIR=/tmp/seabadnet_logs_all
SEED=42
ALL_MELS=(16 32 48 64 80)

mkdir -p "$LOGDIR"
cd "$WORKDIR"

SUMMARY="$LOGDIR/summary.txt"
echo "=== Full all-mels run started: $(date) ===" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one script with a given set of extra args.
# Usage: run_script LABEL SCRIPT [extra args...]
# ─────────────────────────────────────────────────────────────────────────────
run_script() {
    local label="$1"
    local script="$2"
    shift 2
    local logfile="$LOGDIR/${label}.log"
    echo "=== [$label] starting $(date) ===" | tee -a "$SUMMARY"
    "$PYTHON" "$script" "$@" 2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$label] SUCCESS" | tee -a "$SUMMARY"
    else
        echo "[$label] FAILED (exit $exit_code) — see $logfile" | tee -a "$SUMMARY"
    fi
    echo "" >> "$SUMMARY"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — CNN-Mel baseline + n_mels sweep
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 1: CNN-Mel baseline (n_mels sweep) ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "1a_m${M}" 1a_baseline2d.py \
        --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b — waveform / 1D reference experiments (run once, no n_mels arg)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 1b: Waveform reference experiments (once each) ===" | tee -a "$SUMMARY"
run_script "1b_transformer"    1b_transformer.py    --dataset-path "$DATASET" --random_seed "$SEED"
run_script "1c_cnntime"        1c_cnntime.py        --dataset-path "$DATASET" --random_seed "$SEED"
run_script "1d_cnntime_gap"    1d_cnntime_gap.py    --dataset-path "$DATASET" --random_seed "$SEED"
run_script "1e_cnntime_enh"    1e_cnntime_enhanced.py --dataset-path "$DATASET" --random_seed "$SEED"
run_script "1f_cnntime_deeper" 1f_cnntime_deeper.py --dataset-path "$DATASET" --random_seed "$SEED"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — GAP pooling variants
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 2: GAP variants ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "2a_gap_m${M}"     2a_baseline_gap.py         --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "2b_gapl_m${M}"    2b_baseline_gap_learned.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "2c_gap1x1_m${M}"  2c_baseline_gap_1x1.py     --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Conv type, filter count, loss function
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 3: Conv type / filter count / loss ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "3a_dw_m${M}"          3a_depthwise.py                              --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3b_f8_m${M}"          3b_filters8.py                               --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3c_focal_m${M}"       3c_gap_focal_loss.py                         --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3d_freqemph_m${M}"    3d_gap_freq_emphasis.py                      --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3e_freqds_m${M}"      3e_gap_freq_emph_ds.py                       --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3f_flfepw_m${M}"      3f_gap_focal_loss_freq_emph_pointwise.py     --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3g_strided_m${M}"     3g_strided_focal_tuned.py                    --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3h_stridedno1x1_m${M}" 3h_strided_focal_no1x1.py                  --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "3i_strideddw_m${M}"   3i_strided_focal_depthwise.py               --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4A — Dropout sweep, Conv2D track
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 4A: Dropout sweep — Conv2D track ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "4a_drop01_m${M}" 4a_dropout01.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4b_drop02_m${M}" 4b_dropout02.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4c_drop03_m${M}" 4c_dropout03.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4d_drop04_m${M}" 4d_dropout04.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4B — Dropout sweep, Depthwise track
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 4B: Dropout sweep — Depthwise track ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "4e_dwdrop01_m${M}" 4e_depthwise_drop01.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4f_dwdrop02_m${M}" 4f_depthwise_drop02.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4g_dwdrop03_m${M}" 4g_depthwise_drop03.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "4h_dwdrop04_m${M}" 4h_depthwise_drop04.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Micro filter count sweep
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 5: Micro filter count sweep ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "5a_dwf6_m${M}" 5a_depthwise_f6.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "5b_dwf5_m${M}" 5b_depthwise_f5.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Final candidates
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 6: Final candidates ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "6a_micro_m${M}" 6a_micro_final.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "6b_edge_m${M}"  6b_edge_final.py  --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Alt — Alternative compact 96-timestep architecture
# ─────────────────────────────────────────────────────────────────────────────
echo "=== ALT: Alternative compact designs ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "alt_tiny_m${M}"     alt_tiny.py         --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "alt_tinygap_m${M}"  alt_tiny_gap.py     --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "alt_tinygagg_m${M}" alt_tiny_gap_agg.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Cap — Capacity reference points
# ─────────────────────────────────────────────────────────────────────────────
echo "=== CAP: Capacity reference points ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "cap_high_m${M}" cap_high_accuracy.py --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "cap_low_m${M}"  cap_low_power.py     --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

# ─────────────────────────────────────────────────────────────────────────────
# Rej — Confirmed-rejected configs (run once for Tables 3 / appendix)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== REJ: Confirmed-rejected configurations ===" | tee -a "$SUMMARY"
for M in "${ALL_MELS[@]}"; do
    run_script "rej_se_m${M}"        rej_accurate_se.py       --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "rej_bn_m${M}"        rej_batchnorm.py         --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "rej_deep1x1_m${M}"   rej_deeper_1x1_gap.py    --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "rej_dense32_m${M}"   rej_dense32.py           --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
    run_script "rej_dw_bn_f6_m${M}"  rej_depthwise_bn_f6.py   --dataset-path "$DATASET" --n_mels "$M" --random_seed "$SEED"
done

echo "" >> "$SUMMARY"
echo "=== ALL RUNS COMPLETE: $(date) ===" | tee -a "$SUMMARY"
