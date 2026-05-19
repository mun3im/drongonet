# Ablation Studies - Updated for V3 Dataset (n_mels=64)

**Date:** 2026-01-09
**Status:** ✅ READY TO RUN
**Configuration:** n_fft=1024, **n_mels=64**, seed=42

---

## What Changed

### Dataset V3 Optimal Hyperparameters
- **OLD (V2):** n_fft=1024, **n_mels=48** → 94.30% baseline
- **NEW (V3):** n_fft=1024, **n_mels=64** → 94.27% baseline

**Reason for change:** V3 dataset with improved long-tail flattening benefits from higher frequency resolution (64 vs 48 mel bins).

### Scripts Updated

All 15 ablation scripts have been updated:
```
✓ 2_depthwise.py
✓ 3_batchnorm.py
✓ 4_dense.py
✓ 5_filters.py
✓ 6_best_accuracy.py
✓ 7_hybrid.py
✓ 8a_dropout01.py
✓ 8b_dropout02.py
✓ 8c_dropout03.py
✓ 8d_dropout04.py
✓ 9a_depthwise_drop01.py
✓ 9b_depthwise_drop02.py
✓ 9c_depthwise_drop03.py
✓ 9d_depthwise_drop04.py
✓ 10_depthwise_f6.py
```

**Changes per script:**
- `n_mels: 48 → 64` (TrainingConfig default)
- `input_shape: (184, 48, 1) → (184, 64, 1)` (model architecture)
- `parse_args default: 48 → 64`
- `help text updated`

### Results Backed Up

Old m48-based results preserved:
```
results_v2_m48_20260109_143138/
├── 2_depthwise_fft1024_m48_s42/
├── 3_batchnorm_fft1024_m48_s42/
└── ... (15 experiments)
```

---

## Expected Performance Changes

Based on V3 baseline sweep results, we expect m64 to perform **better** than m48:

| Metric | m48 (old) | m64 (new) | Change |
|--------|-----------|-----------|--------|
| **Baseline Accuracy** | 94.30% | 94.27% | -0.03% |
| **Mean (across n_mels)** | 93.19% | 93.78% | **+0.59%** |

**Predicted ablation results:**
- Most experiments should match or slightly exceed old m48 results
- Higher n_mels captures more frequency information
- Better for species with distinctive high-frequency calls

---

## Running Experiments

### Option 1: Test Single Experiment (Recommended First)

Test the best-performing ablation from V2:

```bash
# Test 3_batchnorm (was best ablation on V2)
python3 3_batchnorm.py --n_mels 64 --n_fft 1024 --random_seed 42 --use_cache

# Check result
cat results/3_batchnorm_fft1024_m64_s42/results_summary.txt | grep "Accuracy"
```

**Expected:** ~94-95% accuracy (should match or beat m48 version: 93.77%)

### Option 2: Run Full Ablation Sweep

Run all 15 experiments:

```bash
# Runs all experiments 2-10 sequentially
./rerun_ablations_2_10.sh
```

**Runtime:** ~15-25 hours (15 experiments × ~1.5 hours each)

**What it does:**
- Runs experiments sequentially with --use_cache
- Validates accuracy after each run (min 85%)
- Backs up failed experiments
- Logs everything to `rerun_logs/`

### Option 3: Run Top 3 Ablations Only

Run only the most promising experiments:

```bash
# Run top performers from V2 dataset
python3 3_batchnorm.py --n_mels 64 --n_fft 1024 --use_cache
python3 8c_dropout03.py --n_mels 64 --n_fft 1024 --use_cache
python3 2_depthwise.py --n_mels 64 --n_fft 1024 --use_cache
```

**Runtime:** ~4-6 hours
**Rationale:** These were the best 3 from V2, worth testing on V3 first

---

## Monitoring Progress

### Check if cache exists
```bash
# Should exist from baseline sweep
ls -lh /Volumes/Evo/cache_mybad_m64/
```

If cache doesn't exist, first experiment will create it (add ~20 min).

### Watch live progress
```bash
tail -f rerun_logs/rerun_*.log
```

### Check completed experiments
```bash
ls -d results/*_fft1024_m64_s42 | wc -l
echo "Expected: 15"
```

### Quick accuracy check
```bash
for dir in results/*_fft1024_m64_s42; do
  if [ -f "$dir/results_summary.txt" ]; then
    exp=$(basename $dir | cut -d'_' -f1-2)
    acc=$(grep -oP 'Accuracy:\s*\K[\d.]+' "$dir/results_summary.txt" | head -1)
    echo "$exp: $acc"
  fi
done | sort
```

---

## Expected Results

### V2 (m48) Results for Comparison

From previous ablation runs:

| Experiment | Accuracy (m48) | Rank |
|------------|----------------|------|
| 3_batchnorm | 93.77% | 2 |
| 5_filters | 93.70% | 3 |
| 4_dense | 93.50% | 4 |
| 8c_dropout03 | 93.43% | 5 |
| 8a_dropout01 | 93.33% | 6 |
| 7_hybrid | 93.03% | 7 |
| 8b_dropout02 | 92.93% | 8 |
| 2_depthwise | 92.73% | 9 |
| ... | ... | ... |

**Baseline (m48):** 94.30%

### V3 (m64) Predictions

Based on m64 baseline = 94.27%:

| Experiment | Predicted (m64) | Reasoning |
|------------|-----------------|-----------|
| 3_batchnorm | ~94.0-94.5% | BN + higher resolution |
| 5_filters | ~93.8-94.2% | More capacity for m64 |
| 4_dense | ~93.6-94.0% | Better feature extraction |
| 8c_dropout03 | ~93.5-94.0% | Regularization still helps |
| 2_depthwise | ~93.0-93.5% | Efficiency trade-off unchanged |

**Goal:** At least one ablation beats baseline (>94.27%)

---

## Success Criteria

✅ **Success if:**
- All 15 experiments complete without errors
- All accuracies > 85% (no collapse)
- At least one ablation matches or beats baseline (≥94.27%)
- Results consistent with m64 benefits higher frequency resolution

⚠️ **Warning if:**
- Accuracies drop significantly below m48 versions
- Multiple experiments < 90% (suggests dataset or config issue)

❌ **Failure if:**
- Multiple experiments < 85% (accuracy collapse)
- All results worse than m48 versions (suggests m64 was wrong choice)

---

## Analysis After Completion

### 1. Generate Pareto frontier
```bash
python3 analyze_all_ablations.py
```

### 2. Compare m48 vs m64
```bash
# Compare best from each config
m48_best=$(grep "Accuracy:" results_v2_m48_*/3_batchnorm_fft1024_m48_s42/results_summary.txt | grep -oP '\d+\.\d+')
m64_best=$(grep "Accuracy:" results/3_batchnorm_fft1024_m64_s42/results_summary.txt | grep -oP '\d+\.\d+')

echo "Best m48: $m48_best%"
echo "Best m64: $m64_best%"
```

### 3. Identify top 4 models
- Smallest (memory constrained)
- Fastest (latency constrained)
- Most accurate (accuracy critical)
- Balanced (general purpose)

---

## Quick Start Commands

### Test first (recommended):
```bash
python3 3_batchnorm.py --n_mels 64 --n_fft 1024 --use_cache
```

### Run full sweep:
```bash
./rerun_ablations_2_10.sh
```

### Monitor:
```bash
tail -f rerun_logs/rerun_*.log
```

---

## Files Modified

### Scripts:
- `2_depthwise.py` through `10_depthwise_f6.py` (15 files)
- `rerun_ablations_2_10.sh` (updated to m64)
- `1_baseline.py` (already updated for V3)

### Utilities:
- `update_ablations_to_m64.py` (update script)
- `analyze_all_ablations.py` (analysis script - unchanged)

### Documentation:
- `ABLATIONS_V3_M64_README.md` (this file)
- `BASELINE_SWEEP_V3_RESULTS.md` (V3 baseline analysis)

---

## Cache Information

**Cache location:** `/Volumes/Evo/cache_mybad_m64/`

**Created by:** Baseline sweep experiment 1_baseline_fft1024_m64_s42

**Size:** ~1-2 GB (20k samples × 3 splits)

**Reuse:** All ablation experiments will use this cache with `--use_cache` flag

If cache is corrupted or missing:
```bash
# Regenerate from any experiment
python3 2_depthwise.py --n_mels 64 --n_fft 1024 --force-reprocess
```

---

## Troubleshooting

### Cache not found
```bash
# Check if cache exists
ls /Volumes/Evo/cache_mybad_m64/

# If missing, create it
python3 1_baseline.py --n_mels 64 --n_fft 1024 --force-reprocess
```

### GPU OOM errors
Ablation experiments use similar memory to baseline. If OOM occurs:
- Close other GPU processes
- Reduce batch size in script (currently 32)
- Scripts already use `TF_FORCE_GPU_ALLOW_GROWTH`

### Accuracy collapse
If multiple experiments < 85%:
1. Check dataset path is correct
2. Verify cache was built with correct dataset
3. Check for data corruption
4. Review training logs for anomalies

---

**Ready to run!** Start with a test experiment or launch the full sweep.

---

**Generated:** 2026-01-09 14:35
**Status:** All scripts updated, ready for V3 dataset experiments
