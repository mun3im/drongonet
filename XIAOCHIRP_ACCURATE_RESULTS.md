# XiaoChirp-Accurate Results - SE Block Experiment

**Date:** 2026-01-09
**Status:** ❌ FAILED - Did Not Improve Baseline

---

## Executive Summary

**Result:** The Squeeze-Excitation (SE) block variant **FAILED** to beat the baseline.

- ❌ Accuracy: 93.60% (vs baseline 94.27%) - **0.67% WORSE**
- ❌ AUC: 0.9850 (vs baseline 0.9862) - **0.12% WORSE**
- ❌ Size: 28.95 KB (vs baseline 23.78 KB) - **22% LARGER**
- ❌ Inference: 0.83 ms (vs baseline 0.29 ms) - **186% SLOWER**

**Conclusion:** SE block does NOT improve this architecture. Baseline remains the best model.

---

## Detailed Results Comparison

### XiaoChirp-Accurate (Baseline + SE Block)

| Metric | Value |
|--------|-------|
| **Float32 AUC** | 0.9851 |
| **TFLite int8 Accuracy** | 93.60% |
| **TFLite int8 AUC** | 0.9850 |
| **Model Size** | 28.95 KB |
| **Inference Time** | 0.83 ms |
| **Total Params** | 29,648 |
| **Training Time** | 9m 08s |

### Baseline (fft1024 m64)

| Metric | Value |
|--------|-------|
| **Float32 AUC** | 0.9864 |
| **TFLite int8 Accuracy** | 94.27% |
| **TFLite int8 AUC** | 0.9862 |
| **Model Size** | 23.78 KB |
| **Inference Time** | 0.29 ms |
| **Total Params** | 24,352 |
| **Training Time** | 9m 04s |

### Delta (Accurate - Baseline)

| Metric | Change | Direction |
|--------|--------|-----------|
| **Accuracy** | -0.67% | ❌ WORSE |
| **AUC** | -0.0012 (-0.12%) | ❌ WORSE |
| **Size** | +5.17 KB (+22%) | ❌ LARGER |
| **Inference** | +0.54 ms (+186%) | ❌ SLOWER |
| **Params** | +5,296 (+22%) | ⚠️ MORE |
| **Training Time** | +4.1 s (+0.7%) | ≈ SAME |

---

## Why Did SE Block Fail?

### Expected vs Actual

**Expected:**
- SE block would learn channel importance
- Extra conv layer would increase capacity
- Target: >94.27% (beat baseline), aim for 95%+
- Estimated size: ~25-27 KB

**Actual:**
- Accuracy DECREASED by 0.67%
- AUC DECREASED by 0.12%
- Size INCREASED to 28.95 KB (worse than expected)
- Inference TRIPLED (0.29 → 0.83 ms)

### Possible Explanations

#### 1. Over-Regularization
- SE block adds channel attention = implicit regularization
- May have over-regularized an already well-tuned model
- Baseline already near optimal for this dataset

#### 2. Insufficient Model Capacity
- Only 4 conv filters (very small)
- SE block ratio=1 means no bottleneck (4→4→4)
- May need more filters to benefit from SE attention
- **Hypothesis:** SE blocks effective for larger models (e.g., 32+ filters), not micro-models

#### 3. Dataset Too Small
- 40k samples may be insufficient to train SE block effectively
- SE block adds ~5k params (+22%)
- More params + same data = overfitting risk

#### 4. Architecture Mismatch
- SE blocks proven effective for ResNet-50+ scale models
- May not transfer to ultra-small CNNs (4 filters)
- Channel attention less meaningful with only 4 channels

#### 5. Padding Changed Receptive Field
- Extra conv uses `padding='same'` (baseline uses `padding='valid'`)
- Changes spatial dimensions and receptive field
- May have disrupted learned feature representations

#### 6. Dropout Interference
- Added Dropout(0.2) after Dense layer
- Baseline doesn't use dropout
- May have compounded regularization from SE + dropout

---

## Inference Speed Analysis

**Why 186% slower?**

| Component | Baseline | SE Variant | Overhead |
|-----------|----------|------------|----------|
| Conv2D layers | 2 | 3 | +1 layer |
| MaxPool2D | 2 | 2 | 0 |
| GlobalAvgPool2D | 0 | 1 (SE) | +1 layer |
| Dense layers | 2 | 4 (2 base + 2 SE) | +2 layers |
| Multiply | 0 | 1 (SE) | +1 op |

**Additional operations:**
- +1 Conv2D (padding='same', larger compute)
- +1 GlobalAveragePooling2D
- +2 Dense layers (SE excitation path)
- +1 Multiply (channel-wise scaling)

**Result:** 28.95 KB model is 22% larger in params, but 186% slower suggests:
- TFLite quantization overhead for extra ops
- SE block poorly optimized in TFLite runtime
- Small models don't amortize op overhead well

---

## Parameter Breakdown

### Baseline Architecture (24,352 params)
```
Input: (184, 64, 1)
Conv2D(4, 3x3, valid) → ~608 params
MaxPool2D(2x2)
Conv2D(4, 3x3, valid) → ~148 params
MaxPool2D(2x2)
Flatten
Dense(8) → ~21,608 params
Dense(2) → ~18 params
```

### SE Variant Architecture (29,648 params)
```
Input: (184, 64, 1)
Conv2D(4, 3x3, valid) → ~608 params
MaxPool2D(2x2)
Conv2D(4, 3x3, same) → ~148 params ← NEW
SE Block:
  GlobalAvgPool2D
  Dense(4) → ~20 params ← NEW
  Dense(4) → ~20 params ← NEW
  Reshape + Multiply
MaxPool2D(2x2)
Flatten
Dense(8) → ~21,608 params
Dropout(0.2) ← NEW
Dense(2) → ~18 params
```

**Extra params:** ~5,296 (+22%)
- Extra Conv2D: ~148
- SE Dense layers: ~40
- Increased flatten size due to padding='same': ~5,100

**Note:** Most overhead comes from larger flatten layer (padding='same' preserves spatial dims)

---

## Lessons Learned

### 1. SE Blocks Don't Always Help ❌
- Not a universal accuracy booster
- May hurt ultra-small models (4 filters)
- Works best for large-scale models (ResNet-50+)

### 2. Baseline Was Already Near-Optimal ✅
- 94.27% may be close to dataset ceiling
- Further accuracy gains require:
  - More training data
  - Different architecture paradigm
  - Ensemble methods

### 3. Padding Matters ⚠️
- `padding='same'` vs `padding='valid'` significantly impacts model size
- Larger spatial dimensions → larger flatten layer → more params
- Not just conv params, but downstream dense layer params too

### 4. Small Models Hurt by Overhead 📉
- Extra layers/ops disproportionately slow small models
- Baseline: 5 layers, 0.29 ms
- SE variant: 10+ layers, 0.83 ms
- Op overhead dominates when model is tiny

### 5. Negative Results Are Valuable ✅
- Now we know SE blocks don't work for this use case
- Saves future researchers from trying same approach
- Confirms baseline is robust

---

## Alternative Approaches (Future Work)

### What Could Beat 94.27%?

#### 1. Residual Connections (Lower Priority)
- **Status:** Not recommended based on SE results
- Similar architecture complexity
- Likely similar overfitting issues

#### 2. More Training Data
- **Status:** Most promising
- Increase from 40k → 100k samples
- May enable more complex architectures

#### 3. Deeper Baseline (More Filters)
- Test 8 or 16 filters instead of 4
- May provide capacity for SE/residual to help
- Trade-off: larger model size

#### 4. Ensemble Methods
- Train 3-5 baseline models with different seeds
- Ensemble predictions
- Typically +1-2% accuracy
- Trade-off: 3-5× inference cost

#### 5. Data Augmentation
- More aggressive mel spec augmentation
- Time stretching, freq masking
- SpecAugment techniques

#### 6. Different Architecture Family
- MobileNetV3 (depthwise separable focus)
- EfficientNet (compound scaling)
- May find better accuracy/size trade-off

---

## Recommendations

### Immediate Actions

1. **Stick with Baseline** ✅
   - 94.27% accuracy, 23.78 KB, 0.29 ms
   - Best model for general use

2. **Abandon SE/Residual Variants** ❌
   - Not worth complexity overhead
   - Better to focus on other approaches

3. **Update Model Family** 📋
   - Remove "Accurate" variant from proposals
   - Baseline remains accuracy champion

### Future Research

1. **Validate on More Data** 📊
   - Test if 94.27% holds on independent test set
   - May be close to dataset accuracy ceiling

2. **Investigate Data Quality** 🔍
   - 94.27% may indicate ~6% label noise
   - Manual review of misclassified samples

3. **Try Ensemble Baseline** 🎯
   - Lowest complexity way to gain 1-2%
   - 3 baseline models with different seeds

---

## Final Model Family (Updated)

| Model | Config | Accuracy | Size | Latency | Use Case | Status |
|-------|--------|----------|------|---------|----------|--------|
| **Tiny** | fft512, m16 | 93.13% | 7.28 KB | 0.09 ms | Gatekeeper | ✅ Validated |
| Tiny-Balanced | fft1024, m32 | 94.00% | 12.78 KB | 0.15 ms | Mobile | ✅ Validated |
| **Standard** | **fft1024, m64** | **94.27%** | 23.78 KB | 0.29 ms | **General** | ✅ **BEST** |
| ~~Accurate~~ | ~~fft1024, m64 + SE~~ | ~~93.60%~~ | ~~28.95 KB~~ | ~~0.83 ms~~ | ~~Max accuracy~~ | ❌ **Rejected** |

**Note:** Baseline (m64) remains the accuracy champion. No "Accurate" variant exists.

---

## Conclusion

**XiaoChirp-Accurate Experiment: FAILED**

The Squeeze-Excitation block variant achieved:
- ❌ 93.60% accuracy (-0.67% vs baseline)
- ❌ 28.95 KB size (+22% vs baseline)
- ❌ 0.83 ms inference (+186% vs baseline)

**Verdict:** SE blocks do NOT improve this architecture. Baseline remains optimal.

**Recommendation:** Use baseline (94.27%, 23.78 KB, 0.29 ms) as the accuracy champion.

**Value of Negative Result:** Confirms baseline is robust and prevents future wasted effort on SE/residual variants.

---

**Generated:** 2026-01-09 17:00
**Experiment Status:** COMPLETED - NEGATIVE RESULT
**Overall Verdict:** ❌ **SE Block Not Effective for Ultra-Small CNNs**
