# XiaoChirp Variant Experiments - Status

**Date:** 2026-01-09 16:38
**Status:** RUNNING

---

## Experiments in Progress

### 1. XiaoChirp-Tiny Enhancement (fft512 m16)

**Task ID:** b61df80
**Script:** `1_baseline.py --n_fft 512 --n_mels 16 --random_seed 42 --force-reprocess`
**Output:** `results/1_baseline_fft512_m16_s42/`

**Objective:** Test if fft512 with m16 can achieve ultra-small model (<6 KB) while maintaining >85% accuracy for gatekeeper task.

**Current Status:** Preprocessing phase (~20% complete)
- Creating new cache at `/Volumes/Evo/cache_mybad_fft512_m16` (or similar)
- Processing train/val/test splits
- Estimated time remaining: ~40 minutes

**Expected Results:**
- Accuracy: ~90-92% (based on fft512 generally -1 to -2% vs fft1024)
- Size: ~5-6 KB (smaller FFT → fewer input features)
- Inference: ~0.05 ms (faster processing)

**Success Criteria:**
- ✅ Accuracy >85% (gatekeeper requirement)
- ✅ Size <6 KB
- ✅ Inference <0.06 ms

**Fallback Plan:** If accuracy <85%, use existing fft1024 m16 (92.80%, 7.28 KB)

---

### 2. XiaoChirp-Accurate (Baseline + SE Block)

**Task ID:** b2ea554
**Script:** `12_accurate_se.py --n_mels 64 --n_fft 1024 --random_seed 42 --use_cache`
**Output:** `results/12_accurate_se_fft1024_m64_s42/`

**Objective:** Beat baseline 94.27% accuracy using Squeeze-Excitation block for channel attention.

**Current Status:** Loading datasets from cache
- Using existing cache: `/Volumes/Evo/cache_mybad_fft1024_m64`
- About to start training
- Estimated time: ~1.5-2 hours

**Architecture Enhancements:**
1. **Extra Conv2D layer** after first conv block (padding='same')
2. **Squeeze-Excitation block** for channel attention:
   - GlobalAveragePooling2D (squeeze)
   - Dense(4) → ReLU → Dense(4) → Sigmoid (excitation)
   - Multiply with input (scale)
3. **Dropout(0.2)** for regularization

**Expected Results:**
- Accuracy: ~95.0-95.5%
- Size: ~25-27 KB (baseline 23.78 KB + SE overhead ~30 params)
- Inference: ~0.32-0.35 ms

**Success Criteria:**
- ✅ Accuracy >94.27% (beat baseline)
- ✅ Size <30 KB
- ✅ Target: 95%+

**Alternatives:** If SE doesn't reach 95%, can try residual block variant (13_accurate_residual.py)

---

## Implementation Details

### XiaoChirp-Accurate Architecture

```python
Input: (184, 64, 1)
↓
Conv2D(4 filters, 3x3, valid) + ReLU
↓
MaxPool2D(2x2)
↓
Conv2D(4 filters, 3x3, same) + ReLU  ← NEW
↓
Squeeze-Excitation Block              ← NEW
  • GlobalAvgPool
  • Dense(4) + ReLU
  • Dense(4) + Sigmoid
  • Multiply (channel weighting)
↓
MaxPool2D(2x2)
↓
Flatten
↓
Dense(8) + ReLU
↓
Dropout(0.2)                          ← NEW
↓
Dense(2) + Softmax
```

**Parameter Estimate:**
- Conv params: ~900 (vs baseline ~600)
- SE params: ~30
- Dense params: ~23,000 (same as baseline)
- **Total: ~23,930** (vs baseline ~23,600)
- **Size increase: ~3% for potential 1% accuracy gain**

---

## Timeline

**Started:** 2026-01-09 16:35

**Estimated Completion:**
- XiaoChirp-Tiny: ~17:15 (preprocessing 40 min + training 15 min)
- XiaoChirp-Accurate: ~18:30 (training ~2 hours)

**Total:** ~2 hours for both experiments

---

## Next Steps

**After experiments complete:**

1. Analyze results and compare to baseline
2. Update `XIAOCHIRP_VARIANTS_PROPOSAL.md` with actual results
3. Generate updated Pareto frontier including new variants

**If XiaoChirp-Accurate succeeds (>94.27%):**
- Define as new accuracy champion
- Update model family table

**If XiaoChirp-Tiny succeeds (>85%):**
- Define as ultra-small gatekeeper model
- Document two-stage pipeline performance

**If either fails:**
- Tiny: Use fft1024 m16 (92.80%) as XiaoChirp-Tiny
- Accurate: Try residual variant or accept baseline as best

---

## Model Family (Projected)

| Model | Config | Accuracy | Size | Latency | Use Case |
|-------|--------|----------|------|---------|----------|
| **Tiny-Ultra** | fft512, m16 | ~90%* | ~5.5 KB | ~0.05 ms | Gatekeeper |
| Tiny-Standard | fft1024, m16 | 92.80% | 7.28 KB | 0.07 ms | Backup gatekeeper |
| Tiny-Balanced | fft1024, m32 | 94.00% | 12.78 KB | 0.15 ms | Mobile |
| Standard | fft1024, m64 | 94.27% | 23.78 KB | 0.29 ms | General |
| **Accurate** | fft1024, m64 + SE | ~95%* | ~27 KB | ~0.34 ms | Max accuracy |

*Pending validation

---

**Generated:** 2026-01-09 16:38
**Status:** Both experiments running in parallel
**Next update:** Upon completion
