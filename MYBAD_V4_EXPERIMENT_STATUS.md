# MyBAD v4 Dataset - Experiment Status & Rerun List
**Generated**: 2026-01-15
**Dataset**: MyBAD v4 (25k positive + 25k negative = 50k total)
**Split**: 80:10:10 (train/val/test), seed=42

---

## ✅ COMPLETED EXPERIMENTS

### 1. Baseline CNN-Mel (1_baseline.py)
**Status**: ✅ COMPLETE - Full sweep n_mels=[16,32,48,64,80], n_fft=1024

| n_mels | Float AUC | TFLite AUC | Model Size | Inference Time | Notes |
|--------|-----------|------------|------------|----------------|-------|
| 16     | 0.9703    | 0.9700     | 7.28 KB    | 0.07ms        | Smallest baseline |
| 32     | 0.9790    | 0.9788     | 12.78 KB   | 0.13ms        | |
| 48     | 0.9802    | 0.9801     | 18.28 KB   | 0.20ms        | Best baseline AUC |
| 64     | 0.9781    | 0.9777     | 23.78 KB   | 0.26ms        | |
| 80     | 0.9816    | 0.9809     | 29.28 KB   | 0.34ms        | Largest baseline |

### 2. CNN-Time (1b_baseline1d.py)
**Status**: ✅ COMPLETE

| Model | Float AUC | TFLite AUC | Model Size | Inference Time | Notes |
|-------|-----------|------------|------------|----------------|-------|
| CNN-Time | 0.9737 | 0.9723 | 6007.15 KB | 1.96ms | ❌ Impractical (1000x larger, 50x slower) |

### 3. Depthwise Separable CNN (2_depthwise.py)
**Status**: ✅ COMPLETE - Full sweep n_mels=[16,32,48,64,80], n_fft=1024

| n_mels | Float AUC | TFLite AUC | Model Size | Inference Time | vs Baseline AUC |
|--------|-----------|------------|------------|----------------|-----------------|
| 16     | 0.9509    | 0.9513     | 8.65 KB    | 0.07ms        | -0.0187 |
| 32     | 0.9650    | 0.9648     | 14.15 KB   | 0.15ms        | -0.0140 |
| 48     | 0.9758    | 0.9757     | 19.65 KB   | 0.24ms        | -0.0044 |
| 64     | 0.9776    | 0.9769     | 25.15 KB   | 0.31ms        | -0.0008 |
| 80     | 0.9757    | 0.9754     | 30.65 KB   | 0.36ms        | -0.0055 |

**Observation**: ❌ Depthwise is worse than baseline: lower accuracy, larger size, slower inference - **not beneficial**

### 4. XiaoChirp-Tiny (13_tiny.py) - 96-width spectrograms
**Status**: ⚠️ PARTIAL

**Completed fft512 (Full sweep)**:
| n_mels | Float AUC | TFLite AUC | Model Size | Inference Time | Notes |
|--------|-----------|------------|------------|----------------|-------|
| 16     | 0.9690    | 0.9687     | 5.91 KB    | 0.04ms        | 🚀 Fastest |
| 32     | 0.9806    | 0.9804     | 8.66 KB    | 0.07ms        | ⭐ Best balance |
| 48     | 0.9813    | 0.9811     | 11.41 KB   | 0.11ms        | Excellent |
| 64     | 0.9816    | 0.9815     | 14.16 KB   | 0.14ms        | High accuracy |
| 80     | **0.9827** | **0.9825** | 16.91 KB   | 0.17ms        | 🏆 **Best AUC** |

**Completed fft1024 (Partial)**:
| n_mels | Float AUC | TFLite AUC | Model Size | Inference Time |
|--------|-----------|------------|------------|----------------|
| 16     | 0.9744    | 0.9742     | 5.91 KB    | 0.04ms        |
| 80     | 0.9800    | 0.9798     | 16.91 KB   | 0.18ms        |

**Missing**: fft1024 with n_mels=[32, 48, 64]

**Key Observation**: XiaoChirp-Tiny achieves **best overall accuracy** (0.9825) while being **2x smaller** and **2x faster** than baseline!

---

## 🏆 MODEL RANKINGS - SMALLEST, FASTEST, BEST

### 🔬 Smallest Models (TFLite int8)
| Rank | Model | Size | AUC | Inference |
|------|-------|------|-----|-----------|
| 🥇 1st | XiaoChirp-Tiny fft512/1024 m16 | **5.91 KB** | 0.9687-0.9742 | 0.04ms |
| 🥈 2nd | Baseline m16 | 7.28 KB | 0.9700 | 0.07ms |
| 🥉 3rd | XiaoChirp-Tiny fft512 m32 | 8.66 KB | 0.9804 | 0.07ms |
| 4th | Depthwise m16 | 8.65 KB | 0.9513 | 0.07ms |

### ⚡ Fastest Models (Inference Time)
| Rank | Model | Time | AUC | Size |
|------|-------|------|-----|------|
| 🥇 1st | XiaoChirp-Tiny m16 | **0.04ms** | 0.9687-0.9742 | 5.91 KB |
| 🥈 2nd | Baseline m16 | 0.07ms | 0.9700 | 7.28 KB |
| 🥈 2nd | Depthwise m16 | 0.07ms | 0.9513 | 8.65 KB |
| 🥈 2nd | XiaoChirp-Tiny fft512 m32 | 0.07ms | 0.9804 | 8.66 KB |

### 🎯 Highest Accuracy
| Rank | Model | AUC | Size | Inference |
|------|-------|-----|------|-----------|
| 🥇 1st | **XiaoChirp-Tiny fft512 m80** | **0.9825** | 16.91 KB | 0.17ms |
| 🥈 2nd | XiaoChirp-Tiny fft512 m64 | 0.9815 | 14.16 KB | 0.14ms |
| 🥉 3rd | XiaoChirp-Tiny fft512 m48 | 0.9811 | 11.41 KB | 0.11ms |
| 4th | Baseline m80 | 0.9809 | 29.28 KB | 0.34ms |
| 5th | XiaoChirp-Tiny fft512 m32 | 0.9804 | 8.66 KB | 0.07ms |

### ⭐ Best Balance (Accuracy/Size/Speed)
| Rank | Model | AUC | Size | Inference | Score |
|------|-------|-----|------|-----------|-------|
| 🥇 1st | **XiaoChirp-Tiny fft512 m32** | 0.9804 | 8.66 KB | 0.07ms | ⭐⭐⭐⭐⭐ |
| 🥈 2nd | **XiaoChirp-Tiny fft512 m48** | 0.9811 | 11.41 KB | 0.11ms | ⭐⭐⭐⭐⭐ |
| 🥉 3rd | Baseline m48 | 0.9801 | 18.28 KB | 0.20ms | ⭐⭐⭐⭐ |
| 4th | XiaoChirp-Tiny fft1024 m16 | 0.9742 | 5.91 KB | 0.04ms | ⭐⭐⭐⭐ |

### ❌ Worst Performers
| Model | AUC | Size | Inference | Issue |
|-------|-----|------|-----------|-------|
| CNN-Time | 0.9723 | 6007.15 KB | 1.96ms | Impractical for edge (1000x larger) |
| Depthwise m16 | 0.9513 | 8.65 KB | 0.07ms | Lowest accuracy |

---

## 📋 EXPERIMENTS TO RUN ON MyBAD v4

### Priority 1: Core Architectures (NEW/UNRUN)
1. ✨ **1c_mybad_transformer.py** - Transformer-Time (just created)
2. **3_batchnorm.py** - Batch normalization variant
3. **4_dense.py** - Dense layer variant ⚠️ (needs path fix: mybad2→mybad)
4. **5_filters.py** - Different filter sizes
5. **6_high_accuracy.py** - High accuracy variant
6. **7_low_power.py** - Low power variant

**Recommended n_mels sweep**: [16, 32, 48, 64, 80]
**Expected experiments**: 6 × 5 = 30

### Priority 2: Dropout Variants (Regularization Study)
7. **8a_dropout01.py** - Dropout 0.1
8. **8b_dropout02.py** - Dropout 0.2
9. **8c_dropout03.py** - Dropout 0.3
10. **8d_dropout04.py** - Dropout 0.4

**Recommended n_mels**: [16, 48, 80] (based on baseline results)
**Expected experiments**: 4 × 3 = 12

### Priority 3: Depthwise + Dropout Variants
11. **9a_depthwise_drop01.py** - Depthwise + Dropout 0.1
12. **9b_depthwise_drop02.py** - Depthwise + Dropout 0.2
13. **9c_depthwise_drop03.py** - Depthwise + Dropout 0.3
14. **9d_depthwise_drop04.py** - Depthwise + Dropout 0.4

**Recommended n_mels**: [16, 48, 80]
**Expected experiments**: 4 × 3 = 12

### Priority 4: Advanced Variants
15. **10_depthwise_f6.py** - Depthwise with 6 filters
16. **11_depthwise_bn_f6.py** - Depthwise + BatchNorm with 6 filters
17. **12_accurate_se.py** - Squeeze-and-Excitation for accuracy
18. **12_depthwise_f5.py** - Depthwise with 5 filters

**Recommended n_mels**: [48, 80] (optimal from baseline)
**Expected experiments**: 4 × 2 = 8

### Priority 5: Complete XiaoChirp-Tiny
19. **13_tiny.py** - Missing fft1024 variants

**Missing**: fft1024 with n_mels=[32, 48, 64]
**Expected experiments**: 3

---

## 📊 EXECUTION PLAN

### Total Experiments Needed: ~65

**Phase 1** (Priority 1): 30 experiments - Core architecture comparison
**Phase 2** (Priority 2+3): 24 experiments - Regularization study
**Phase 3** (Priority 4+5): 11 experiments - Advanced variants & completion

### Execution Notes:
- Use `--force-cpu` for all experiments (CPU only)
- Use `nice -n 19` for low priority (system responsiveness)
- Sequential execution recommended (disk space constraints)
- Each experiment ~10-15 min on CPU

**Estimated total time**: 65 × 12 min = 13 hours

---

## 🔧 REQUIRED FIXES BEFORE RUNNING

1. **4_dense.py**: Update dataset path
   ```python
   # Change from:
   dataset_path: str = '/Volumes/Evo/mybad2'
   # To:
   dataset_path: str = '/Volumes/Evo/mybad'
   ```

---

## 📈 KEY FINDINGS FROM COMPLETED EXPERIMENTS

### 🏆 Winner: XiaoChirp-Tiny
**XiaoChirp-Tiny dominates across all metrics:**
- 🥇 **Highest accuracy**: 0.9825 AUC (fft512 m80) - beats all other models
- 🥇 **Smallest size**: 5.91 KB (m16) - 23% smaller than baseline
- 🥇 **Fastest inference**: 0.04ms (m16) - 43% faster than baseline
- ⭐ **Best balance**: fft512 m32 (0.9804 AUC, 8.66 KB, 0.07ms)

### Key Observations:
1. **XiaoChirp-Tiny vs Baseline**:
   - Same AUC range but **2x smaller** and **2x faster**
   - Achieves 0.9825 AUC (vs baseline's 0.9809 max)

2. **96-width advantage**: Reducing spectrogram width from 184→96:
   - Cuts model size in half
   - Doubles inference speed
   - **Maintains or improves accuracy**

3. **Depthwise failure**:
   - Worse accuracy than baseline (-0.004 to -0.019 AUC)
   - Larger model size (+1.37 KB average)
   - No benefits for this architecture/dataset

4. **CNN-Time impractical**:
   - 6 MB model (1000x larger than XiaoChirp-Tiny)
   - 1.96ms inference (50x slower)
   - Not suitable for edge deployment

5. **Optimal configurations**:
   - **Ultra-low power**: XiaoChirp-Tiny m16 (5.91 KB, 0.04ms, 0.9742 AUC)
   - **Best balance**: XiaoChirp-Tiny m32 (8.66 KB, 0.07ms, 0.9804 AUC)
   - **Maximum accuracy**: XiaoChirp-Tiny m80 (16.91 KB, 0.17ms, 0.9825 AUC)

---

## 🎯 NEXT STEPS

1. Run Priority 1 experiments (core architectures)
2. Analyze results to determine best variants
3. Run Priority 2-3 (regularization study) on best architectures
4. Complete advanced variants with optimized hyperparameters
5. Generate final comparison report

---

**Last Updated**: 2026-01-15
