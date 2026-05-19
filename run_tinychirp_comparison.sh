#!/usr/bin/env bash
# ============================================================
# run_tinychirp_comparison.sh
#
# Two experiments for a fair cross-dataset comparison of the
# TinyChirp CNN-Mel architecture:
#
#   Exp A: TinyChirp CNN-Mel on its original Corn Bunting
#          dataset (TinyChirp paper conditions, n_mels=80)
#
#   Exp B: TinyChirp CNN-Mel on SEABAD, same mel parameters
#          (n_mels=80, n_fft=1024, hop=256) — the fair
#          transfer baseline for the paper comparison table
#
# Both use seed=42, identical mel config, identical architecture.
# Results land in results/tc_cnnmel_tinychirp_s42/ and
#                   results/tc_cnnmel_seabad_m80_s42/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TINYCHIRP_DATA="/Volumes/Evo/TinyChirp"
SEABAD_DATA="/Volumes/Evo/seabad"
TINYCHIRP_CACHE="/Volumes/Evo/cache_tinychirp_mels"
SEABAD_CACHE_M80="/Volumes/Evo/cache_seabad_m80"
SEED=42

# ── sanity checks ────────────────────────────────────────────
if [ ! -d "$TINYCHIRP_DATA/training" ]; then
    echo "ERROR: TinyChirp dataset not found at $TINYCHIRP_DATA" >&2
    exit 1
fi
if [ ! -d "$SEABAD_DATA/positive" ]; then
    echo "ERROR: SEABAD dataset not found at $SEABAD_DATA" >&2
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Experiment A: TinyChirp CNN-Mel on TinyChirp dataset ────
log "=== Exp A: TinyChirp CNN-Mel → TinyChirp dataset (n_mels=80) ==="
log "Output: results/tc_cnnmel_tinychirp_s42/"

# 0a hardcodes n_mels=80 and n_fft=1024 (TinyChirp paper spec).
# Override only the paths and seed via CLI.
python 0a_tinychirp_cnnmel.py \
    --dataset-path  "$TINYCHIRP_DATA" \
    --random_seed   "$SEED"

# Rename output to a stable, unambiguous directory name
if [ -d "results/0a_tinychirp_cnnmel_r${SEED}_darwin" ]; then
    mv "results/0a_tinychirp_cnnmel_r${SEED}_darwin" \
       "results/tc_cnnmel_tinychirp_s${SEED}"
elif [ -d "results/0a_tinychirp_cnnmel_r${SEED}_linux" ]; then
    mv "results/0a_tinychirp_cnnmel_r${SEED}_linux" \
       "results/tc_cnnmel_tinychirp_s${SEED}"
fi

log "Exp A complete."

# ── Experiment B: TinyChirp CNN-Mel on SEABAD (same mel params) ─
log ""
log "=== Exp B: TinyChirp CNN-Mel → SEABAD (n_mels=80, n_fft=1024) ==="
log "Output: results/tc_cnnmel_seabad_m80_s42/"

# 1a supports --n_mels and --n_fft; pass identical values to 0a
# so the mel spectrograms are computed identically.
python 1a_baseline2d.py \
    --dataset-path  "$SEABAD_DATA" \
    --cache-dir     "$SEABAD_CACHE_M80" \
    --n_mels        80 \
    --n_fft         1024 \
    --random_seed   "$SEED"

# Rename to stable name
SEABAD_OUT=$(ls -d results/1a_baseline2d_fft1024_m80_s${SEED} 2>/dev/null \
             || ls -d results/*1a*m80*s${SEED}* 2>/dev/null \
             || echo "")
if [ -n "$SEABAD_OUT" ] && [ -d "$SEABAD_OUT" ]; then
    mv "$SEABAD_OUT" "results/tc_cnnmel_seabad_m80_s${SEED}"
fi

log "Exp B complete."

# ── Summary ─────────────────────────────────────────────────
log ""
log "=== Results ==="
for d in "results/tc_cnnmel_tinychirp_s${SEED}" \
         "results/tc_cnnmel_seabad_m80_s${SEED}"; do
    if [ -f "$d/results_summary.txt" ]; then
        log "--- $d ---"
        grep -E "AUC|Accuracy|Inference|Model Size" "$d/results_summary.txt" || true
    fi
done

log "Done. Compare the two results_summary.txt files for the paper table."
