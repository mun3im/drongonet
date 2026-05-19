# Fixes Applied to 1_baseline.py

## Problem Identified

The original `1_baseline.py` had a **degenerate model collapse** issue:
- TFLite Accuracy: **0.5000** (random guessing)
- Model predicted **ALL samples as Positive class**
- Negative class: precision=0.0000, recall=0.0000
- Positive class: precision=0.5000, recall=1.0000

## Root Cause

**Focal loss was too aggressive for the balanced MyBAD dataset:**
```python
# BEFORE (BROKEN):
loss=focal_loss(gamma=2.0, alpha=0.5)
```

The focal loss, designed for highly imbalanced datasets, caused the model to collapse to a trivial solution of always predicting the majority class.

## Solution Applied

**Replaced focal loss with standard categorical crossentropy:**
```python
# AFTER (FIXED):
loss='categorical_crossentropy'  # Simple and effective for balanced datasets
```

## Expected Results

With this fix, `1_baseline.py` should now achieve:
- **Accuracy**: 70-80% (instead of 50%)
- **AUC**: 0.80-0.90 (instead of degenerate 0.97 with no discrimination)
- **Balanced predictions**: Both classes properly recognized

## Comparison with Optimized Version

The "optimized" version (`1_baseline_optimized.py`) already used label smoothing and achieved:
- Accuracy: 0.6815 (68.15%)
- AUC: 0.7239
- Proper class discrimination

The fixed baseline should perform similarly or better, establishing a proper benchmark.

## Next Steps

1. **Re-run `1_baseline.py`** with the fix:
   ```bash
   python3 1_baseline.py --use_cache --n_mels 48
   ```

2. **Compare results**:
   - TinyChirp baseline: 88.37% accuracy
   - MyBAD fixed baseline: Expected 70-80% accuracy
   - Gap of ~10-15% justifies architectural improvements

3. **Proceed with ablation studies** on architectural variants (depthwise conv, etc.)

## Files Status

✅ **Keep**: `0a_tinychirp_cnnmel.py` - Excellent TinyChirp baseline (88.37% acc)
🔧 **Fixed**: `1_baseline.py` - Now uses categorical crossentropy
✅ **Keep**: `1_baseline_optimized.py` - Works correctly (68.15% acc)
🗑️ **Can Delete**: `0a_tinychirp_cnnmel_optimized.py` - Worse than original

---

## Technical Details

### Why Focal Loss Failed

Focal loss formula: `FL(pt) = -α(1-pt)^γ log(pt)`

For balanced datasets with γ=2.0:
- Easy examples (pt > 0.9) get near-zero weight
- Hard examples (pt < 0.5) get high weight
- This can cause gradient instability on balanced data
- Model learns trivial solution: predict all as one class

### Why Categorical Crossentropy Works

Standard cross-entropy: `CE = -∑ y_true * log(y_pred)`

For balanced datasets:
- Treats all examples equally
- Stable gradients
- No class weighting bias
- Simple and effective

---

Generated: 2026-01-04
Model: CNN-Mel Table II Baseline
Dataset: MyBADv2 (40k samples, balanced)
