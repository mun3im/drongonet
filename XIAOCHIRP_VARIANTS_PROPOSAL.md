# XiaoChirp Model Variants - Enhanced Proposal

**Date:** 2026-01-09
**Status:** Design Phase

---

## Model Family Strategy

| Model | Target | Config | Expected Perf | Status |
|-------|--------|--------|---------------|--------|
| **Tiny** | Gatekeeper | fft512, m16 | >85%, <6 KB | 🆕 Proposed |
| Standard | General | fft1024, m64 | 94.27% | ✅ Validated |
| **Accurate** | Max Performance | fft1024, m64 + SE/Residual | >95% | 🆕 Proposed |

---

## 1. XiaoChirp-Tiny Enhancement

### Current: fft1024, m16
- **Accuracy:** 92.80%
- **Size:** 7.28 KB
- **Inference:** 0.07 ms

### Proposed: fft512, m16
**Rationale:**
- Gatekeeper only needs >85% accuracy
- Lower FFT window = smaller model + faster
- m16 already has low frequency resolution, fft512 sufficient

**Expected Performance:**
- **Accuracy:** ~90-92% (based on fft512 generally -1 to -2% vs fft1024)
- **Size:** ~5-6 KB (smaller FFT → fewer input features)
- **Inference:** ~0.05 ms (faster processing)

**Benefits:**
- ✅ Ultra-small: <6 KB (smallest possible)
- ✅ Ultra-fast: <0.06 ms
- ✅ Still sufficient for gatekeeper (>85%)

**Risk:** Accuracy may drop below 90% - needs validation

**Action Required:**
```bash
# Test fft512 m16
python3 1_baseline.py --n_fft 512 --n_mels 16 --random_seed 42 --force-reprocess

# Expected runtime: ~1 hour
```

---

## 2. XiaoChirp-Accurate Design

### Goal: Beat 94.27% baseline

### Architecture Options

#### Option A: Baseline + Squeeze-Excitation (SE) Block

**Architecture:**
```python
Input: (184, 64, 1)
↓
Conv2D(4 filters) + ReLU
↓
MaxPool2D
↓
Conv2D(4 filters) + ReLU     ← NEW: Additional conv layer
↓
MaxPool2D
↓
Squeeze-Excitation Block      ← NEW: Channel attention
  • GlobalAvgPool
  • Dense(4 units) + ReLU
  • Dense(4 units) + Sigmoid
  • Multiply (channel weighting)
↓
Flatten
↓
Dense(8 units) + ReLU
↓
Dropout(0.2)
↓
Dense(2 units) + Softmax
```

**Squeeze-Excitation Benefits:**
- ✅ Learns channel importance (which filters matter most)
- ✅ Minimal parameter increase (~30 params for SE block)
- ✅ Proven effective in CNNs
- ✅ Helps model focus on discriminative features

**Estimated Performance:**
- **Accuracy:** ~94.5-95.5%
- **Size:** ~25-27 KB (baseline 23.78 + SE overhead)
- **Inference:** ~0.32-0.35 ms

---

#### Option B: Baseline + Residual Connection

**Architecture:**
```python
Input: (184, 64, 1)
↓
Conv2D(4 filters) + ReLU
↓
MaxPool2D
↓
┌─────────────────────────────┐ ← Residual connection
│ Conv2D(4 filters) + ReLU    │ ← NEW: Additional conv
│ Conv2D(4 filters) + ReLU    │
└─────────────────────────────┘
  Add (residual + input)
↓
MaxPool2D
↓
Flatten
↓
Dense(8 units) + ReLU
↓
Dropout(0.2)
↓
Dense(2 units) + Softmax
```

**Residual Benefits:**
- ✅ Easier gradient flow (deeper network training)
- ✅ Identity mapping preserves features
- ✅ Proven to improve accuracy
- ✅ Enables deeper architectures

**Estimated Performance:**
- **Accuracy:** ~94.5-95.0%
- **Size:** ~26-28 KB (extra conv layer)
- **Inference:** ~0.33-0.36 ms

---

#### Option C: Hybrid (SE + Residual)

**Architecture:**
```python
Input: (184, 64, 1)
↓
Conv2D(4 filters) + ReLU
↓
MaxPool2D
↓
┌─────────────────────────────┐ ← Residual block
│ Conv2D(4 filters) + ReLU    │ ← NEW: Additional conv
│ SE Block (channel attention)│ ← NEW: SE within residual
│ Conv2D(4 filters) + ReLU    │
└─────────────────────────────┘
  Add (residual + input)
↓
MaxPool2D
↓
Flatten
↓
Dense(8 units) + ReLU
↓
Dropout(0.2)
↓
Dense(2 units) + Softmax
```

**Hybrid Benefits:**
- ✅ Best of both: residual learning + channel attention
- ✅ Strongest expected performance
- ✅ State-of-art CNN design pattern

**Estimated Performance:**
- **Accuracy:** ~95.0-95.5% (highest potential)
- **Size:** ~28-30 KB
- **Inference:** ~0.35-0.38 ms

**Risk:** May overfit on small dataset

---

### Recommended: Option A (SE Block)

**Why SE over Residual:**
1. **Simpler:** One SE block vs two conv layers + skip connection
2. **Smaller:** ~2-3 KB less than residual
3. **Proven:** SE blocks very effective for small models
4. **Lower risk:** Less parameters = less overfitting risk

**SE Block Implementation:**
```python
def squeeze_excitation_block(x, ratio=1):
    """
    Squeeze-Excitation block for channel attention.

    Args:
        x: Input tensor (batch, height, width, channels)
        ratio: Reduction ratio for bottleneck (default: 1, no reduction for 4 filters)
    """
    channels = x.shape[-1]

    # Squeeze: Global average pooling
    se = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Excitation: FC -> ReLU -> FC -> Sigmoid
    se = tf.keras.layers.Dense(max(channels // ratio, 1), activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)

    # Scale: Multiply input by channel weights
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([x, se])
```

---

## Implementation Plan

### Phase 1: Validate XiaoChirp-Tiny (fft512, m16)

**Script:** Use existing `1_baseline.py`

```bash
# Test fft512 m16 for Tiny variant
python3 1_baseline.py --n_fft 512 --n_mels 16 --random_seed 42 --force-reprocess

# Runtime: ~1 hour
```

**Success criteria:**
- ✅ Accuracy >85% (gatekeeper requirement)
- ✅ Size <6 KB
- ✅ Inference <0.06 ms

**If successful:** Use fft512 m16 as XiaoChirp-Tiny
**If fails (<85%):** Stick with fft1024 m16 (92.80%)

---

### Phase 2: Create XiaoChirp-Accurate

**Script:** Create `12_accurate_se.py`

**Base it on:** `1_baseline.py` or `3_batchnorm.py`

**Changes:**
1. Add extra Conv2D layer after first conv
2. Add Squeeze-Excitation block
3. Use n_mels=64, n_fft=1024 (V3 optimal)

**Code structure:**
```python
def build_model_se(input_shape=(184, 64, 1)):
    """XiaoChirp-Accurate with Squeeze-Excitation"""
    inputs = tf.keras.Input(shape=input_shape)

    # First conv block
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Second conv block (NEW)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)

    # Squeeze-Excitation (NEW)
    x = squeeze_excitation_block(x, ratio=1)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='XiaoChirp_Accurate_SE')
    return model
```

**Training:**
```bash
python3 12_accurate_se.py --n_mels 64 --n_fft 1024 --random_seed 42 --use_cache

# Runtime: ~1.5 hours
```

**Success criteria:**
- ✅ Accuracy >94.27% (beat baseline)
- ✅ Size <30 KB
- ✅ Target: 95%+

---

## Expected Model Family (After Validation)

| Model | Config | Accuracy | Size | Latency | Use Case |
|-------|--------|----------|------|---------|----------|
| **Tiny** | fft512, m16 | ~90%* | ~5.5 KB | ~0.05 ms | Gatekeeper |
| Tiny-Balanced | fft1024, m32 | 94.00% | 12.78 KB | 0.15 ms | Mobile |
| Standard | fft1024, m64 | 94.27% | 23.78 KB | 0.29 ms | General |
| **Accurate** | fft1024, m64 + SE | ~95%* | ~27 KB | ~0.34 ms | Max accuracy |

*To be validated

---

## Architecture Comparison

### Parameter Count Estimates

| Model | Conv Params | Dense Params | SE Params | Total | Size (KB) |
|-------|-------------|--------------|-----------|-------|-----------|
| Baseline m16 | ~150 | ~7,000 | 0 | ~7,150 | 7.28 |
| Baseline m64 | ~600 | ~23,000 | 0 | ~23,600 | 23.78 |
| **Accurate (m64 + SE)** | ~900 | ~23,000 | ~30 | ~23,930 | ~24.5 |

**SE block adds only ~30 params but significant accuracy gain!**

---

## Next Steps

### Immediate Actions:

1. **Test XiaoChirp-Tiny (fft512 m16):**
   ```bash
   python3 1_baseline.py --n_fft 512 --n_mels 16 --random_seed 42 --force-reprocess
   ```
   **Priority:** HIGH
   **Time:** 1 hour

2. **Create XiaoChirp-Accurate script:**
   - Copy `1_baseline.py` → `12_accurate_se.py`
   - Add SE block implementation
   - Add extra conv layer
   - Update model name
   **Priority:** HIGH
   **Time:** 30 mins development

3. **Train XiaoChirp-Accurate:**
   ```bash
   python3 12_accurate_se.py --n_mels 64 --n_fft 1024 --random_seed 42 --use_cache
   ```
   **Priority:** MEDIUM
   **Time:** 1.5 hours

---

## Alternative: Residual Variant (Optional)

If SE doesn't reach 95%, try residual:

**Script:** `13_accurate_residual.py`

```python
def residual_block(x, filters):
    """Simple residual block"""
    # Store input for skip connection
    skip = x

    # Residual path
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add skip connection
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x
```

---

## Risk Analysis

### XiaoChirp-Tiny (fft512 m16)

**Risks:**
- ⚠️ May drop below 85% (fft512 generally worse than fft1024)
- ⚠️ May not converge (too little frequency information)

**Mitigation:**
- If <85%: Stick with fft1024 m16 (92.80%)
- Quick test (1 hour), low cost

### XiaoChirp-Accurate (SE/Residual)

**Risks:**
- ⚠️ May overfit (more parameters, same dataset size)
- ⚠️ May not beat baseline (diminishing returns)

**Mitigation:**
- Use dropout (already in baseline)
- SE adds minimal params (~30)
- Can try different SE ratios if needed
- If fails: Try residual or accept baseline as best

---

## Summary

**Proposed Enhancements:**

1. **XiaoChirp-Tiny:** Use fft512, m16 for ultra-small gatekeeper
   - Expected: ~90%, ~5.5 KB, ~0.05 ms
   - **Action:** Test with existing baseline script

2. **XiaoChirp-Accurate:** Add SE block + extra conv to m64 baseline
   - Expected: ~95%, ~27 KB, ~0.34 ms
   - **Action:** Create new script with SE implementation

**Ready to proceed?**
1. Test fft512 m16 first (1 hour, validates Tiny)
2. Create and train Accurate variant (2 hours total)

---

**Generated:** 2026-01-09 15:00
**Status:** Awaiting approval to proceed
