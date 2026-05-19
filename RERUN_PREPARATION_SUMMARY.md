# Ablation Studies Rerun Preparation - Complete

**Date**: 2026-01-08
**Status**: ✅ READY FOR EXECUTION
**Target**: Scripts 2-10 (15 experiments)

---

## Executive Summary

All ablation scripts (2-10) have been **updated and validated** for rerun with optimal hyperparameters from the baseline sweep. The training pipeline includes **safeguards against accuracy collapse** and automatic validation.

---

## 🎯 Configuration Updates

### Optimal Hyperparameters (from baseline sweep)

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|---------|
| **n_fft** | Mixed (512/1024) | **1024** | Best accuracy (94.30% vs 93.93%) |
| **n_mels** | Mixed (48/80) | **48** | Optimal balance (94.30%) |
| **seed** | 42 | 42 | Unchanged |

**Baseline Reference**: 1_baseline_fft1024_m48_s42
- Accuracy: 94.30%
- AUC: 0.9860
- Model Size: 18.28 KB
- Inference: 0.19 ms

---

## ✅ Updated Scripts

All 15 scripts have been updated:

```
2_depthwise.py          ✓ Updated
3_batchnorm.py          ✓ Updated
4_dense.py              ✓ Updated
5_filters.py            ✓ Updated
6_best_accuracy.py      ✓ Updated
7_hybrid.py             ✓ Updated
8a_dropout01.py         ✓ Updated
8b_dropout02.py         ✓ Updated
8c_dropout03.py         ✓ Updated
8d_dropout04.py         ✓ Updated
9a_depthwise_drop01.py  ✓ Updated
9b_depthwise_drop02.py  ✓ Updated
9c_depthwise_drop03.py  ✓ Updated
9d_depthwise_drop04.py  ✓ Updated
10_depthwise_f6.py      ✓ Updated
```

### Changes Made Per Script

1. **n_mels**: Default changed to 48 in `TrainingConfig`
2. **n_fft**: Default changed to 1024 in `TrainingConfig`
3. **input_shape**: Updated from `(184, 80, 1)` to `(184, 48, 1)`
4. **parse_args defaults**: Updated where applicable

---

## 🛡️ Training Pipeline Safeguards

### 1. Loss Function ✅
```python
loss='categorical_crossentropy'  # Correct for balanced 2-class
```
- **Status**: All 15 scripts verified
- **Why**: Balanced dataset (50/50 split), no class weights needed

### 2. Optimizer ✅
```python
optimizer=get_optimizer(learning_rate, weight_decay=0.01)
```
- **Platform-aware**:
  - Linux: AdamW (better regularization)
  - macOS (Apple Silicon): legacy.Adam (performance)
  - Other: Adam (standard)
- **Status**: All 15 scripts use platform-aware optimizer

### 3. Class Weights ✅
```python
class_weights = None
trainer = ModelTrainer(model, config, class_weights=None)
```
- **Status**: All 15 scripts set to None
- **Why**: Balanced dataset doesn't need class weighting

### 4. Learning Rate Schedule ✅
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)
```
- **Status**: All 15 scripts have LR reduction
- **Strategy**: Reduce LR by 50% after 5 epochs without improvement

### 5. Early Stopping ✅
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,  # 3x LR patience
    restore_best_weights=True
)
```
- **Status**: All 15 scripts have early stopping
- **Strategy**: Stop after 15 epochs (3× LR patience) without improvement
- **Safety**: Restores best weights (prevents overfitting)

### 6. Accuracy Collapse Detection 🆕
```bash
MIN_ACC_THRESHOLD=0.85  # 90% of baseline (0.9430)
```
- **Rerun script validates** each experiment
- **Automatic backup** of failed results
- **Logs warning** if accuracy < 85%

---

## 📊 Expected Results

Based on baseline (94.30% accuracy), architectural modifications should:

| Architecture | Expected Impact | Target Accuracy |
|--------------|-----------------|-----------------|
| 2_depthwise | Parameter reduction | ~93-94% |
| 3_batchnorm | Training stability | ~94-95% |
| 4_dense | More capacity | ~94-95% |
| 5_filters | More features | ~94-95% |
| 6_best_accuracy | Combined improvements | ~95-96% 🎯 |
| 7_hybrid | Alternative combination | ~94-95% |
| 8a-d_dropout | Regularization sweep | ~93-95% |
| 9a-d_depthwise_drop | Efficient + regularized | ~93-95% |
| 10_depthwise_f6 | More filters | ~94-95% |

**Failure Criteria**: Accuracy < 85% (indicates model collapse or bug)

---

## 🚀 Execution Plan

### Step 1: Final Review (MANUAL)
```bash
# Review changes made
git diff 2_depthwise.py 3_batchnorm.py 4_dense.py

# Spot check a script
head -100 6_best_accuracy.py | grep -E "(n_mels|n_fft|input_shape)"
```

### Step 2: Run Rerun Script
```bash
./rerun_ablations_2_10.sh
```

**What it does**:
1. Runs all 15 experiments sequentially
2. Uses cached mel spectrograms (--use_cache)
3. Validates accuracy after each run
4. Backs up failed experiments
5. Logs everything to `rerun_logs/`

**Runtime**: ~25-40 hours (15 experiments × ~1.5-2.5 hrs each)

### Step 3: Monitor Progress
```bash
# Check logs
tail -f rerun_logs/rerun_*.log

# Check results
watch -n 60 'ls -ltr results/ | tail -20'

# Check latest accuracy
find results/ -name "results_summary.txt" -mmin -60 -exec grep -H "Accuracy" {} \;
```

### Step 4: Analyze Results (After Completion)
```bash
# Generate comparison table
python3 analyze_baseline_sweep.py

# Create visualization
python3 generate_pareto_analysis.py
```

---

## 📁 File Manifest

### Scripts Created
1. **`update_ablation_configs.py`** - Batch update hyperparameters
2. **`rerun_ablations_2_10.sh`** - Execution script with safeguards
3. **`analyze_baseline_sweep.py`** - Results analysis (already exists)

### Documentation
1. **`BASELINE_SWEEP_DECISION.md`** - Baseline analysis & recommendations
2. **`RERUN_PREPARATION_SUMMARY.md`** - This file
3. **`REVIEW_FINDINGS.md`** - Initial project review

### Logs
- `rerun_logs/` - Per-experiment logs
- `rerun_logs/rerun_YYYYMMDD_HHMMSS.log` - Master log

---

## ⚠️ Known Issues & Mitigations

### Issue 1: Old Results in results/ Folder
**Problem**: Some old results exist with inconsistent configs
**Mitigation**: Rerun script checks and backs up old results before overwriting

### Issue 2: Cache Mismatch
**Problem**: Cached mels might be for wrong n_mels
**Mitigation**: Scripts read from n_mels-specific cache dirs
**Cache locations**:
```
/Volumes/Evo/cache_mybad_m16/
/Volumes/Evo/cache_mybad_m32/
/Volumes/Evo/cache_mybad_m48/  ← Used for rerun
/Volumes/Evo/cache_mybad_m64/
/Volumes/Evo/cache_mybad_m80/
```

### Issue 3: GPU Memory
**Problem**: Long-running experiments may accumulate GPU memory
**Mitigation**: Each script sets `TF_FORCE_GPU_ALLOW_GROWTH=true`

### Issue 4: Disk Space
**Problem**: 15 experiments × ~200 KB = ~3 MB (negligible)
**Mitigation**: None needed

---

## 🔍 Verification Checklist

Before starting rerun, verify:

- [x] All 15 scripts updated to n_fft=1024, n_mels=48
- [x] Training pipeline consistent (loss, optimizer, callbacks)
- [x] Class weights set to None (balanced dataset)
- [x] Rerun script has accuracy collapse detection
- [x] Cache directories exist for n_mels=48
- [x] Execution script is executable (`chmod +x`)
- [ ] **User confirms ready to start** ⬅️ YOU ARE HERE

---

## 📋 Post-Rerun Checklist

After completion:

- [ ] Verify all 15 experiments succeeded
- [ ] Check no accuracy collapse (all > 85%)
- [ ] Generate unified results table
- [ ] Create Pareto frontier plot
- [ ] Identify top 4 models (tiny/fast/balanced/accurate)
- [ ] Document findings in paper

---

## 🎯 Success Criteria

Rerun is successful if:

1. ✅ All 15 experiments complete without errors
2. ✅ All accuracies > 85% (no collapse)
3. ✅ At least one architecture beats baseline (>94.30%)
4. ✅ Results consistent with expected ranges
5. ✅ Pareto frontier identifies clear winners

---

## 💡 Tips for Execution

### Run in Background
```bash
# Use screen or tmux for long runs
screen -S ablation_rerun
./rerun_ablations_2_10.sh
# Ctrl+A, D to detach

# Reattach later
screen -r ablation_rerun
```

### Parallel Execution (Advanced)
```bash
# Run 2-3 experiments in parallel if you have multiple GPUs
# Edit rerun script to use CUDA_VISIBLE_DEVICES

# GPU 0: experiments 2,3,4,5,6
# GPU 1: experiments 7,8a,8b,8c,8d
# GPU 2: experiments 9a,9b,9c,9d,10
```

### Quick Test (Before Full Run)
```bash
# Test one experiment first
python3 2_depthwise.py --n_mels 48 --n_fft 1024 --use_cache

# Check result
cat results/2_depthwise_fft1024_m48_s42/results_summary.txt
```

---

## 📊 Comparison with Old Results

### Before Rerun (Old Config)
- Mixed n_fft values (mostly 512)
- Mixed n_mels values (48/80)
- Some used old dataset (50k samples)
- Inconsistent results

### After Rerun (New Config)
- Uniform n_fft=1024
- Uniform n_mels=48
- New dataset (40k samples, better quality)
- **Expected: +5-10% improvement** across the board

---

## 🔗 References

- **Baseline Analysis**: `BASELINE_SWEEP_DECISION.md`
- **Baseline Results**: `baseline_sweep_complete_analysis.csv`
- **Best Configuration**: n_fft=1024, n_mels=48
- **Baseline Accuracy**: 94.30%
- **Dataset**: MyBAD v2 (40k samples, improved distribution)

---

## ✅ Ready to Execute

**All systems go!** 🚀

Run when ready:
```bash
./rerun_ablations_2_10.sh
```

Expected completion: ~25-40 hours
Monitor: `tail -f rerun_logs/rerun_*.log`

---

**Generated**: 2026-01-08
**Author**: Claude (Automated Preparation)
**Status**: Ready for user confirmation
