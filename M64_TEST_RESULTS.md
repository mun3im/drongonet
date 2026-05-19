# n_mels=64 Test Results - 3_batchnorm

**Date:** 2026-01-09
**Status:** ⚠️ UNEXPECTED RESULTS
**Experiment:** 3_batchnorm with n_mels=64 (V3 dataset)

---

## Results Summary

### 3_BatchNorm Performance: m48 vs m64

| Metric | m48 (V2 dataset) | m64 (V3 dataset) | Change |
|--------|------------------|------------------|--------|
| **Accuracy** | **93.77%** | **93.55%** | **-0.22%** ⬇️ |
| **AUC** | 0.9844 | 0.9834 | -0.10% |
| **Size** | 18.61 KB | 24.11 KB | **+5.50 KB** ⬆️ |
| **Inference** | 0.33 ms | 0.34 ms | +0.01 ms |

**Finding:** BatchNorm performed WORSE with m64 despite baseline improving.

---

## Comparison to Baseline

### Same Dataset (V3), Different n_mels

| Model | n_mels | Accuracy | Size (KB) | Δ vs Baseline |
|-------|--------|----------|-----------|---------------|
| **Baseline** | 48 | 93.63% | 18.28 | reference |
| **Baseline** | **64** | **94.27%** | 23.78 | **+0.64%** ✅ |
| BatchNorm | 48 | 93.77% | 18.61 | +0.14% |
| BatchNorm | 64 | 93.55% | 24.11 | **-0.08%** ❌ |

**Key Insight:** Baseline benefits from m64 (+0.64%), but BatchNorm does NOT (-0.08%).

---

## Analysis

### Why m64 Hurts BatchNorm?

**Hypothesis 1: Over-regularization**
- m64 → 33% more parameters (18.61 KB → 24.11 KB)
- BatchNorm already provides regularization
- Combined effect: over-regularized model, reduced capacity

**Hypothesis 2: Frequency Resolution Mismatch**
- BatchNorm may normalize away high-frequency features
- m64 captures more high-freq information
- BN's normalization reduces this advantage

**Hypothesis 3: Architecture-Specific Optimal n_mels**
- Not all architectures have same optimal hyperparameters
- Baseline benefits from m64
- BatchNorm optimized for m48

**Hypothesis 4: Dataset Interaction**
- V3's improved long-tail flattening changes feature distribution
- BatchNorm's normalization conflicts with new distribution
- Baseline adapts better

---

## Implications for Full Ablation Sweep

### Critical Question: Should we revert to m48?

**Evidence suggesting YES (stay with m48):**
1. ✅ BatchNorm performs better at m48 (93.77% vs 93.55%)
2. ✅ Smaller models (18.61 KB vs 24.11 KB)
3. ✅ Proven results from V2 dataset
4. ✅ Less risky - known performance

**Evidence suggesting NO (continue with m64):**
1. ✅ Baseline improved significantly with m64 (94.27% vs 93.63%)
2. ✅ V3 dataset designed for m64 (optimal from sweep)
3. ✅ Only tested 1/15 ablations so far
4. ✅ May find ablations that DO benefit from m64

---

## Recommendations

### Option A: Architecture-Specific n_mels (Recommended)

Run ablations with **BEST n_mels per architecture**, not uniform m64:

| Architecture | Optimal n_mels | Reasoning |
|--------------|----------------|-----------|
| Baseline | 64 | Proven optimal (94.27%) |
| BatchNorm | **48** | Better performance (93.77%) |
| Dropout 0.3 | 64 | Likely benefits (minimal architecture change) |
| Depthwise | ? | Test both 48 and 64 |
| Dense | ? | Test both 48 and 64 |
| Filters | ? | Test both 48 and 64 |

**Pros:**
- ✅ Best performance per architecture
- ✅ Scientifically rigorous (optimize each separately)
- ✅ May discover architecture-specific sweet spots

**Cons:**
- ⚠️ More complex (different configs per architecture)
- ⚠️ Harder to compare apples-to-apples
- ⚠️ 2x experiments (test both m48 and m64 for unknowns)

---

### Option B: Uniform m64, Accept Trade-offs (Simpler)

Continue with m64 for all ablations:

**Pros:**
- ✅ Consistent comparison
- ✅ Dataset optimized for m64
- ✅ Simpler experimental design
- ✅ Some ablations may still beat baseline

**Cons:**
- ⚠️ BatchNorm underperforms
- ⚠️ May miss best configurations
- ⚠️ Suboptimal for some architectures

---

### Option C: Hybrid Approach (Balanced)

1. Run remaining ablations with m64 (as planned)
2. For promising candidates, test m48 variant
3. Report best result per architecture

**Pros:**
- ✅ Complete m64 sweep (consistent)
- ✅ Identify m48 sweet spots for key architectures
- ✅ Best of both worlds

**Cons:**
- ⚠️ More experiments (but only for promising ones)

---

## Decision Framework

### If Goal = "Find absolute best model"
→ **Option A** (architecture-specific n_mels)
- Optimize each architecture independently
- Accept complexity for peak performance

### If Goal = "Compare architectures fairly"
→ **Option B** (uniform m64)
- All experiments use same hyperparameters
- Dataset V3 optimized for m64
- Accept that some architectures underperform

### If Goal = "Best models + fair comparison"
→ **Option C** (hybrid)
- Complete m64 sweep first
- Re-run top 3-5 with m48 if underperforming
- Report both for completeness

---

## My Recommendation: Option C (Hybrid)

**Action Plan:**

1. **Continue m64 sweep** (14 experiments remaining)
   - Get complete picture of m64 performance
   - Consistent with V3 dataset optimization

2. **Identify underperformers**
   - Any ablation < baseline m48 (93.63%)
   - Any ablation < its own m48 version

3. **Re-test underperformers with m48** (if any)
   - BatchNorm already confirmed (93.77% @ m48 > 93.55% @ m64)
   - Test 2-3 others if they underperform

4. **Final model set**
   - Baseline: m64 (94.27%)
   - BatchNorm: **m48** (93.77%) - use old result
   - Others: best of m48 or m64

**Benefits:**
- ✅ Complete m64 dataset characterization
- ✅ Recover known good m48 performance for some
- ✅ Identify architecture-specific preferences
- ✅ Final model family has best of each architecture

**Cost:**
- Extra 2-3 experiments if needed (4-6 hours)

---

## Next Steps

### Immediate:
1. **Continue** m64 ablation sweep (already started)
2. **Flag** BatchNorm to use m48 result (93.77%)
3. **Monitor** other ablations for underperformance

### After sweep completes:
1. Compare all m64 results to baseline
2. Identify if any match BatchNorm pattern (worse than m48)
3. Re-run those specific ones with m48
4. Generate final Pareto frontier with best config per model

---

## Updated Model Family (Preliminary)

| Model | Config | Accuracy | Size | Status |
|-------|--------|----------|------|--------|
| Tiny-Ultra | m16 | 92.80% | 7.28 KB | ✅ Final |
| Tiny-Balanced | m32 | 94.00% | 12.78 KB | ✅ Final |
| **Baseline-Standard** | **m64** | **94.27%** | 23.78 KB | ✅ Final |
| **BatchNorm** | **m48** | **93.77%** | 18.61 KB | ✅ Use m48 |
| Dropout 0.3 | m64 | ? | ? | ⏳ Testing |
| ... | ... | ... | ... | ... |

---

## Conclusion

**n_mels=64 is NOT universally better.**

- ✅ Baseline benefits (+0.64%)
- ❌ BatchNorm suffers (-0.22%)
- ❓ Others TBD

**Recommendation:** Complete m64 sweep, fall back to m48 for architectures that underperform.

---

**Generated:** 2026-01-09 14:50
**Status:** 1/15 ablations tested (m64), mixed results
