# Review: 6_high_accuracy.py and 7_low_power.py

**Date:** 2026-01-09
**Reviewer:** Claude
**Status:** ISSUES FOUND - Corrections Needed

---

## Executive Summary

**6_high_accuracy.py:** ✅ Mostly correct, ⚠️ Minor issues
**7_low_power.py:** ❌ Critical issues found - requires fixes

### Critical Issues
1. ❌ **7_low_power.py:** Wrong dataset path (`mybad2` instead of `mybad`)
2. ❌ **7_low_power.py:** Wrong cache path (not updated to use m16)
3. ⚠️ **7_low_power.py:** Missing AdamW optimizer for Linux
4. ⚠️ **7_low_power.py:** Output directory name doesn't match purpose
5. ⚠️ **Both:** Architecture may not match intended design goals

---

## 1. 6_high_accuracy.py Review

### Purpose
High accuracy model targeting maximum performance on m64 dataset

### Architecture Analysis

**Current Implementation (Lines 143-164):**
```python
def build_cnn_mel_high_acc(input_shape=(184, 64, 1), num_classes=2):
    # Block 1: Conv2D(16) + MaxPool
    # Block 2: Conv2D(32) + MaxPool
    # Block 3: Conv2D(64) + MaxPool
    # GlobalAveragePooling2D
    # Dense(num_classes) + Softmax
```

**Parameters Estimate:**
- Conv layers: 16 + 32 + 64 filters = ~50k params
- Dense layer: 64 → 2 = 130 params
- **Total: ~50-60k params**
- **Expected size:** ~200-250 KB (int8 TFLite)

### Issues Found

#### ⚠️ Issue 1: Inconsistent Documentation (Line 3-11)
```python
"""
Model 7: BEST ACCURACY - Lucky Number 7!
Enhanced architecture with optimized hyperparameters for maximum performance
- Depthwise Separable Convolutions (parameter efficiency)  ← NOT IMPLEMENTED
- Increased filter capacity (6 filters for better feature extraction)  ← SAYS 6, USES 16/32/64
- BatchNormalization (training stability)  ← NOT IMPLEMENTED
- Spatial Dropout (spatial regularization)  ← NOT IMPLEMENTED
- Larger dense layer (16 units with L2 regularization)  ← NOT IMPLEMENTED
- Optimized dropout rate (0.25)  ← NOT IMPLEMENTED
```

**Problem:** Documentation describes features that aren't in the code

**Fix:** Update docstring to match actual implementation:
```python
"""
6_high_accuracy.py: High Accuracy Model
Enhanced CNN with increased filter capacity for maximum performance
- 3 Conv blocks with increasing filters (16→32→64)
- GlobalAveragePooling2D (replaces flatten + dense)
- Optimized for n_mels=64
- Target: >94.27% accuracy
Compatible with both macOS (Metal) and Linux (CUDA)
"""
```

#### ⚠️ Issue 2: No Regularization
**Current:** No dropout, no batch normalization
**Problem:** Large model (64 filters) with no regularization may overfit

**Recommendation:** Add dropout before final dense layer:
```python
# Add after GlobalAveragePooling2D
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
```

#### ⚠️ Issue 3: Large Model Size
**Expected:** ~200-250 KB (much larger than baseline 23.78 KB)
**Problem:** May not fit "high accuracy" goal if it's too large

**Impact:** Consider this is for max accuracy, size is acceptable

#### ✅ Issue 4: Configuration Paths (Lines 86, 88, 790-791)
```python
# Lines 86-88 (defaults - WRONG but overridden)
dataset_path: str = '/Volumes/Evo/mybad'  # ✅ Correct
cache_dir: str = '/Volumes/Evo/cache_mybad_mels'  # ⚠️ Wrong, but overridden

# Lines 790-791 (actual usage - CORRECT)
config.cache_dir = f'/Volumes/Evo/cache_mybad_m{config.n_mels}'  # ✅ Correct
config.output_dir = f'results/6_best_accuracy_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'  # ✅ Correct
```

**Verdict:** Paths are correct in main(), defaults don't matter

#### ✅ Hyperparameters
- ✅ n_mels: 64 (optimal from baseline sweep)
- ✅ n_fft: 1024 (correct)
- ✅ batch_size: 32 (good)
- ✅ learning_rate: 0.001 (standard)
- ✅ early_stopping_patience: 15 (appropriate)
- ✅ Optimizer: AdamW on Linux ✅

### Verdict: 6_high_accuracy.py

**Status:** ⚠️ **USABLE with Minor Issues**

**Strengths:**
- ✅ Correct dataset and cache paths (in main())
- ✅ Proper optimizer selection (AdamW on Linux)
- ✅ Good hyperparameters (m64, learning rate, patience)
- ✅ Large capacity architecture (16→32→64 filters)

**Weaknesses:**
- ⚠️ Misleading documentation (describes features not implemented)
- ⚠️ No regularization (may overfit)
- ⚠️ Very large model (~200 KB vs baseline 24 KB)

**Recommended Actions:**
1. **Update docstring** to match actual implementation
2. **Add dropout (0.2)** before final dense layer
3. **Run experiment** to see if it beats baseline (94.27%)
4. **Expect:** May NOT beat baseline due to overfitting risk

---

## 2. 7_low_power.py Review

### Purpose
Low power model for ultra-efficient inference (targeting MCU/edge devices)

### Architecture Analysis

**Current Implementation (Lines 128-140):**
```python
def build_cnn_mel_low_power(input_shape=(184, 16, 1), num_classes=2):
    # Block 1: Conv2D(8) + MaxPool
    # Block 2: Conv2D(16) + GlobalAvgPool
    # Dense(num_classes) + Softmax
```

**Parameters Estimate:**
- Conv(8): ~80 params
- Conv(16): ~1200 params
- Dense: 16 → 2 = 34 params
- **Total: ~1,300 params**
- **Expected size:** ~5-6 KB (int8 TFLite)

**Design Intent:** Ultra-lightweight for m16 (similar to XiaoChirp-Tiny)

### Issues Found

#### ❌ CRITICAL Issue 1: Wrong Dataset Path (Lines 76, 89)
```python
# Line 76
dataset_path: str = '/Volumes/Evo/mybad2'  # ❌ WRONG - should be '/Volumes/Evo/mybad'

# Line 89
parser.add_argument('--dataset-path', type=str, default='/Volumes/Evo/mybad2',  # ❌ WRONG
```

**Problem:** Points to old dataset (mybad2 doesn't exist or is outdated)
**Expected:** Should use `/Volumes/Evo/mybad` (V3 dataset with 40k samples)

**Fix Required:** Change both occurrences to `/Volumes/Evo/mybad`

#### ❌ CRITICAL Issue 2: Wrong Cache Path (Lines 78, 766)
```python
# Line 78
cache_dir: str = '/Volumes/Evo/cache_mybad2_mels'  # ❌ WRONG

# Line 766 (overridden in main)
config.cache_dir = f'/Volumes/Evo/cache_mybad_m{config.n_mels}'  # ✅ Correct format but...
```

**Problem:** Default points to wrong cache, but main() override is correct format
**Note:** Main() override is correct, so this works if using `--use_cache`

**Fix Recommended:** Update default to match: `'/Volumes/Evo/cache_mybad_mels'`

#### ❌ CRITICAL Issue 3: Missing AdamW Optimizer (Lines 103-125)
```python
def get_optimizer(learning_rate: float):  # ❌ Missing weight_decay parameter
    # Only supports legacy Adam and standard Adam
    # Does NOT support AdamW for Linux
```

**Problem:** Linux performance will be worse without AdamW regularization
**Expected:** Should match baseline's get_optimizer() function

**Fix Required:** Replace with correct optimizer function:
```python
def get_optimizer(learning_rate: float, weight_decay: float = 0.01):
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info(f"Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif system == 'Linux':
        logger.info(f"Detected Linux - using AdamW optimizer with weight_decay={weight_decay}")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        logger.info(f"Detected {system} {machine} - using standard Adam optimizer")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
```

#### ⚠️ Issue 4: Output Directory Name (Line 767)
```python
config.output_dir = f'results/7_hybrid_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'
```

**Problem:** Says "7_hybrid" but this is "7_low_power"
**Fix:** Change to:
```python
config.output_dir = f'results/7_low_power_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'
```

#### ⚠️ Issue 5: Hardcoded Input Shape (Line 128)
```python
def build_cnn_mel_low_power(input_shape=(184, 16, 1), num_classes=2):
```

**Problem:** Hardcoded to 16 mel bins
**Impact:** Low - main() overrides with `input_shape=(184, config.n_mels, 1)`
**Recommendation:** Change default to match expected use:
```python
def build_cnn_mel_low_power(input_shape=(184, 16, 1), num_classes=2):  # Keep as 16 since this is for m16
```

Actually this is fine - it's meant for m16 specifically.

#### ⚠️ Issue 6: Inconsistent Documentation (Lines 3-7)
```python
"""
Option E: Combine BatchNorm + Dropout for maximum improvement  ← WRONG
Ablation Study Model: 3a_e_hybrid  ← WRONG
Based on 1a_mybad2_cnnmel.py with specific architectural modification  ← WRONG
```

**Problem:** Documentation is from a different script
**Fix:** Update docstring:
```python
"""
7_low_power.py: Ultra-Lightweight Low Power Model
Minimal CNN for edge devices and MCUs
- 2 Conv blocks (8→16 filters)
- GlobalAveragePooling2D
- Optimized for n_mels=16
- Target: <10 KB, <0.1 ms inference, >90% accuracy
Compatible with both macOS (Metal) and Linux (CUDA)
"""
```

### Verdict: 7_low_power.py

**Status:** ❌ **NOT READY - Requires Fixes**

**Critical Issues:**
- ❌ Wrong dataset path (mybad2 → mybad)
- ❌ Missing AdamW optimizer for Linux
- ❌ Wrong output directory name (hybrid → low_power)

**Minor Issues:**
- ⚠️ Misleading documentation
- ⚠️ Wrong default cache path (but overridden correctly)

**Recommended Actions:**
1. **FIX dataset path** to `/Volumes/Evo/mybad` (lines 76, 89)
2. **FIX optimizer** to include AdamW support (lines 103-125)
3. **FIX output_dir** to `7_low_power` (line 767)
4. **UPDATE docstring** to describe actual purpose
5. **VERIFY** target is m16 (should use `--n_mels 16`)

---

## Hyperparameter Recommendations

### 6_high_accuracy.py (for m64)

**Current Settings:** ✅ Mostly Optimal
```python
n_mels: 64  ✅ Optimal from baseline sweep
n_fft: 1024  ✅ Correct
batch_size: 32  ✅ Good
learning_rate: 0.001  ✅ Standard
epochs: 100  ✅ With early stopping
early_stopping_patience: 15  ✅ Appropriate
optimizer: AdamW (Linux)  ✅ Best for regularization
```

**Recommended Changes:**
```python
# Add to architecture:
x = tf.keras.layers.Dropout(0.2)(x)  # After GlobalAvgPool, before Dense

# Consider reducing filter sizes if model is too large:
# Option 1: 8→16→32 instead of 16→32→64
# Option 2: Add batch normalization
```

### 7_low_power.py (for m16)

**Current Settings:** ⚠️ Some Issues
```python
n_mels: 64  ❌ Should be 16 for low power!
n_fft: 1024  ⚠️ Consider 512 (based on Tiny results)
batch_size: 32  ✅ Good
learning_rate: 0.001  ✅ Standard
epochs: 100  ✅ With early stopping
early_stopping_patience: 15  ✅ Appropriate
optimizer: Adam  ❌ Should be AdamW on Linux
```

**Recommended Settings:**
```python
n_mels: 16  # Use --n_mels 16 when running
n_fft: 512  # Use --n_fft 512 (based on fft512 > fft1024 for m16)
optimizer: AdamW (Linux)  # Fix get_optimizer() function
```

**Command to run:**
```bash
python3 7_low_power.py --n_mels 16 --n_fft 512 --use_cache
```

---

## Architecture Suitability Analysis

### 6_high_accuracy.py: Is it right for "high accuracy"?

**Architecture:** 3 conv blocks (16→32→64), GlobalAvgPool, Dense(2)

**Pros:**
- ✅ Large capacity (64 filters in final conv)
- ✅ Progressive feature extraction (16→32→64)
- ✅ GlobalAvgPool reduces overfitting vs Flatten

**Cons:**
- ❌ No regularization (dropout, batchnorm)
- ❌ Very large model (~200 KB vs baseline 24 KB)
- ❌ May overfit on 40k samples

**Comparison to Baseline:**
- Baseline: 4 filters, 2 conv blocks, 24 KB, 94.27%
- This: 64 filters, 3 conv blocks, ~200 KB, ??? %

**Expected Result:** ⚠️ **Likely WORSE than baseline** due to overfitting

**Recommendation:** Add dropout (0.3-0.4) or batch normalization

### 7_low_power.py: Is it right for "low power"?

**Architecture:** 2 conv blocks (8→16), GlobalAvgPool, Dense(2)

**Pros:**
- ✅ Very lightweight (8, 16 filters)
- ✅ GlobalAvgPool (efficient)
- ✅ Minimal params (~1,300)
- ✅ Expected size: ~5-6 KB

**Cons:**
- ⚠️ May lack capacity for good accuracy
- ⚠️ Only 2 conv layers (vs baseline's 2)

**Comparison to XiaoChirp-Tiny:**
- XiaoChirp-Tiny: 4 filters, 2 conv blocks, 7.28 KB, 93.13% (fft512 m16)
- This: 8+16 filters, 2 conv blocks, ~5-6 KB, ??? %

**Expected Result:** ⚠️ **Similar or slightly better** than Tiny (more filters)

**Recommendation:** Use fft512, m16 for optimal results

---

## Comparison Table

| Aspect | 6_high_accuracy.py | 7_low_power.py |
|--------|-------------------|----------------|
| **Purpose** | Max accuracy | Min size/latency |
| **Target n_mels** | 64 ✅ | 16 ⚠️ (config says 64) |
| **Target n_fft** | 1024 ✅ | 512 recommended |
| **Filters** | 16→32→64 | 8→16 ✅ |
| **Expected Size** | ~200 KB ⚠️ | ~5-6 KB ✅ |
| **Expected Accuracy** | ~92-94% ⚠️ | ~91-93% ✅ |
| **Dataset Path** | ✅ Correct | ❌ WRONG (mybad2) |
| **Cache Path** | ✅ Correct | ⚠️ Wrong default, OK in main() |
| **Optimizer** | ✅ AdamW | ❌ Missing AdamW |
| **Output Dir** | ✅ Correct | ❌ WRONG (says hybrid) |
| **Documentation** | ❌ Misleading | ❌ Misleading |
| **Regularization** | ❌ None | ✅ None needed (small model) |
| **Status** | ⚠️ Usable | ❌ Needs fixes |

---

## Required Fixes

### 6_high_accuracy.py - Optional Improvements

1. **Update docstring** (Lines 2-12):
```python
"""
6_high_accuracy.py: High Accuracy Model
Enhanced CNN with increased filter capacity
- 3 Conv blocks with progressive filters (16→32→64)
- GlobalAveragePooling2D
- Optimized for n_mels=64, n_fft=1024
- Target: >94.27% accuracy
Compatible with both macOS (Metal) and Linux (CUDA)
"""
```

2. **Add dropout** (Line 159, before Dense):
```python
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # → (64,)
x = tf.keras.layers.Dropout(0.2)(x)  # ADD THIS LINE
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
```

### 7_low_power.py - REQUIRED Fixes

1. **Fix dataset path** (Lines 76, 89):
```python
# Line 76
dataset_path: str = '/Volumes/Evo/mybad'  # CHANGE FROM mybad2

# Line 89
parser.add_argument('--dataset-path', type=str, default='/Volumes/Evo/mybad',  # CHANGE FROM mybad2
```

2. **Fix optimizer** (Lines 103-125) - Replace entire function:
```python
def get_optimizer(learning_rate: float, weight_decay: float = 0.01):
    """
    Get appropriate optimizer based on platform.
    Uses legacy optimizer on M1/M2/M3/M4 Macs to avoid performance issues.
    Uses AdamW with weight decay on Linux for better regularization.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for AdamW optimizer (default: 0.01)

    Returns:
        TensorFlow optimizer instance
    """
    system = platform.system()
    machine = platform.machine()

    # Check if running on Apple Silicon (arm64)
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info(f"Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif system == 'Linux':
        logger.info(f"Detected Linux - using AdamW optimizer with weight_decay={weight_decay}")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        logger.info(f"Detected {system} {machine} - using standard Adam optimizer")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
```

3. **Fix output directory** (Line 767):
```python
config.output_dir = f'results/7_low_power_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'  # CHANGE hybrid→low_power
```

4. **Update docstring** (Lines 2-7):
```python
"""
7_low_power.py: Ultra-Lightweight Low Power Model
Minimal CNN for edge devices and MCUs
- 2 Conv blocks (8→16 filters)
- GlobalAveragePooling2D
- Optimized for n_mels=16, n_fft=512
- Target: <10 KB, <0.1 ms inference, >90% accuracy
Compatible with both macOS (Metal) and Linux (CUDA)
"""
```

5. **Fix default cache** (Line 78) - Optional:
```python
cache_dir: str = '/Volumes/Evo/cache_mybad_mels'  # CHANGE FROM cache_mybad2_mels
```

---

## Recommended Run Commands

### 6_high_accuracy.py (after adding dropout)
```bash
# Use m64 cache
python3 6_high_accuracy.py --n_mels 64 --n_fft 1024 --use_cache --random_seed 42
```

**Expected:**
- Size: ~200 KB
- Accuracy: ~92-94% (may NOT beat baseline due to overfitting)
- Inference: ~0.5-1.0 ms (larger model)

### 7_low_power.py (after fixes)
```bash
# Use m16 with fft512 for best results
python3 7_low_power.py --n_mels 16 --n_fft 512 --use_cache --random_seed 42
```

**Expected:**
- Size: ~5-6 KB
- Accuracy: ~91-93% (competitive with XiaoChirp-Tiny)
- Inference: <0.1 ms

---

## Conclusion

### 6_high_accuracy.py
**Status:** ⚠️ **Usable with Caveats**
- Can run as-is, but add dropout for better results
- May NOT beat baseline (94.27%) due to overfitting risk
- Worth testing to see if larger capacity helps

### 7_low_power.py
**Status:** ❌ **MUST FIX before running**
- Critical: Wrong dataset path (mybad2 → mybad)
- Critical: Missing AdamW optimizer
- Critical: Wrong output directory name
- Will fail or produce incorrect results without fixes

**Priority:** Fix 7_low_power.py first, then optionally improve 6_high_accuracy.py

---

**Generated:** 2026-01-09 17:30
**Recommendation:** Apply fixes before running experiments
