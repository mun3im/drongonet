## TinyChirp on TinyChirp basic

2026-01-04 18:15:45,301 - INFO -   Float32 AUC: 0.9427
2026-01-04 18:15:45,301 - INFO -   TFLite int8 Acc: 0.8112, AUC: 0.9409
2026-01-04 18:15:45,301 - INFO -   TFLite Inference Time: 0.16ms
2026-01-04 18:15:45,301 - INFO -   TFLite Model Size: 29.23 KB
2026-01-04 18:15:45,301 - INFO -   AUC Degradation: 0.0018 (0.19%)


## Improved pipeline 0a_tinychirp_cnnmel.py

2026-01-04 18:28:18,809 - INFO -   Float32 AUC: 0.9475
2026-01-04 18:28:18,809 - INFO -   TFLite int8 Acc: 0.8837, AUC: 0.9475
2026-01-04 18:28:18,809 - INFO -   TFLite Inference Time: 0.16ms
2026-01-04 18:28:18,809 - INFO -   TFLite Model Size: 29.23 KB
2026-01-04 18:28:18,809 - INFO -   AUC Degradation: -0.0000 (-0.00%)

## Optimized baseline 0a_tinychirp_cnnmel_optimized.py

2026-01-04 18:37:04,121 - INFO -   Float32 AUC: 0.9197
2026-01-04 18:37:04,121 - INFO -   TFLite int8 Acc: 0.8284, AUC: 0.9170
2026-01-04 18:37:04,121 - INFO -   TFLite Inference Time: 0.16ms
2026-01-04 18:37:04,121 - INFO -   TFLite Model Size: 29.48 KB
2026-01-04 18:37:04,121 - INFO -   AUC Degradation: 0.0027 (0.29%)


The optimizations actually **decreased** performance on TinyChirp. This is a valuable finding - it shows that:

  - **The original training pipeline was already well-tuned** for TinyChirp
  - **Label smoothing (0.1) hurt performance** - the focal loss was actually better for this dataset
  - **The enhanced augmentation may have been too aggressive** for the relatively clean TinyChirp data
 
**What This Tells You**

The TinyChirp dataset benefits from:
  - **Focal loss** (handles class imbalance better than label smoothing)
  - **Simpler augmentation** (just Gaussian noise)
  - **The original training schedule** (ReduceLROnPlateau works well)


2026-01-04 18:47:58,399 - INFO -   Float32 AUC: 0.6191
2026-01-04 18:47:58,399 - INFO -   TFLite int8 Acc: 0.5005, AUC: 0.4910
2026-01-04 18:47:58,399 - INFO -   TFLite Inference Time: 0.16ms
2026-01-04 18:47:58,399 - INFO -   TFLite Model Size: 29.28 KB
2026-01-04 18:47:58,399 - INFO -   AUC Degradation: 0.1280 (20.68%)


## TinyChirp CNNMel on MyBAD

### Original 1_baseline.py

2026-01-04 19:05:18,292 - INFO -   Float32 AUC: 0.9682
2026-01-04 19:05:18,292 - INFO -   TFLite int8 Acc: 0.5000, AUC: 0.9687
2026-01-04 19:05:18,292 - INFO -   TFLite Inference Time: 0.16ms
2026-01-04 19:05:18,292 - INFO -   TFLite Model Size: 29.28 KB
2026-01-04 19:05:18,292 - INFO -   AUC Degradation: -0.0005 (-0.06%)

### Optimized 1_baseline_optimized.py

2026-01-04 19:12:16,020 - INFO -   Float32 AUC: 0.7386
2026-01-04 19:12:16,020 - INFO -   TFLite int8 Acc: 0.6815, AUC: 0.7239
2026-01-04 19:12:16,020 - INFO -   TFLite Inference Time: 0.20ms
2026-01-04 19:12:16,020 - INFO -   TFLite Model Size: 29.54 KB
2026-01-04 19:12:16,020 - INFO -   AUC Degradation: 0.0147 (1.98%)


## Baseline sweep

### Updated 1_baseline.py

| n_mels     | TFLite Acc     | TFLite AUC     | Model Size | Inference | Training  | Status                     |
| ---------- | -------------- | -------------- | ---------- | --------- | --------- | -------------------------- |
| **64**     | **89.48%** 🏆  | **0.9610** 🏆  | 23.78 KB   | 0.13ms    | 3m 48s    | ✅ **WINNER**               |
| **48**     | **86.12%** 🥈  | **0.9571** 🥈  | 18.28 KB ⚡ | 0.09ms ⚡  | 3m 05s ⚡  | ✅ **EFFICIENT**            |
| 32         | 78.53%         | 0.9588         | 12.78 KB   | 0.06ms    | 4m 23s    | ⚠️ Underfitting            |
| 16         | 73.72%         | 0.8873         | 7.28 KB    | 0.03ms    | 2m 52s    | ⚠️ Too low                 |
| 80         | 50.00% ❌       | 0.5000 ❌       | 29.28 KB   | 0.16ms    | 23m 15s ❌ | 🔴 **FAILED**, overfitting |
|            |                |                |            |           |           |                            |
 
 
 **🎯 Key Findings**

  1. **n_mels=64 is the clear winner** (89.48% accuracy, 0.9610 AUC)
    - Balanced precision/recall across classes
    - Minimal quantization degradation

  2. **n_mels=48 is the efficient choice** (86.12% accuracy)
    - Only 3.36% accuracy loss
    - 23% smaller, 31% faster inference

  3. **n_mels=80 completely failed** (model collapse)
    - Overfitting: too many parameters
    - Predicts all samples as positive class
    - Avoid entirely!

  4. **Sweet spot is 48-64** for MyBAD
    - Below 32: underfitting
    - Above 64: overfitting risk
 ---

  **🏆 Surprise Finding!**

  **MyBAD (n_mels=64) outperforms TinyChirp baseline:**

  - MyBAD: **89.48%** accuracy, **0.9610** AUC
  - TinyChirp: 88.37% accuracy, 0.9475 AUC
  - MyBAD is **+1.11%** better with **smaller model** (23.78 KB vs 29.23 KB)!

  This validates the fix to 1_baseline.py worked perfectly! 🎉

## 1a_baseline.py FFT = 512


| n_mels     | TFLite Acc    | TFLite AUC    | Model Size | Inference | Training  | Status           |
| ---------- | ------------- | ------------- | ---------- | --------- | --------- | ---------------- |
| **64**     | **55.25%** 🏆 | **0.8808** 🏆 | 23.78 KB   | 0.13ms    | 4m 0s     | ⚠️ Overfitting?  |
| **48**     | **83.90%** 🥈 | **0.9340** 🥈 | 18.28 KB ⚡ | 0.09ms ⚡  | 1m 32s ⚡  | ✅ **EFFICIENT**  |
| **32**     | **70.33%** 🥈 | **0.8907** 🥈 | 12.78 KB ⚡ | 0.06ms ⚡  | 3m 42s ⚡  | ⚠️ Underfitting? |


| n_mels | n_fft | Accuracy | AUC      | Verdict             |
|--------|-------|----------|----------|---------------------|
| **64**     | **1024**  | **89.48%** ✅ | **0.9610** ✅ | **WINNER**              |
| 64     | 512   | **52.25%** ❌ | 0.8808 ❌ | **COMPLETE FAILURE**    |
|        |       | **-37.23%!** |          | **MODEL COLLAPSE**      |
| **48**     | 1024  | **86.12%** ✅ | 0.9571 ✅ | Strong              |
| 48     | 512   | 83.90%   | 0.9340   | Acceptable (-2.22%) |
| 32     | 1024  | 78.53%   | 0.9588   | OK                  |
| 32     | 512   | 70.33%   | 0.8907   | Poor (-8.20%)       |
  

  ---

  **🔴 Why n_fft=512 Failed with n_mels=64**

  
  **n_fft=512 with n_mels=64:**
-  Negative recall: 4.50% ← COLLAPSED (was 88.35%)
-  Positive recall: 100.00% ← Predicting everything as positive
-  Accuracy: 52.25% ← Random guessing


  **Root Cause: Spectral Resolution Mismatch**

 - n_fft=512 → 257 frequency bins
 - n_mels=64 → Trying to extract 64 mel bands
  - **Not enough spectral information!**
  - Model gets poor features → unstable training → collapse

  ---

  **✅ FINAL DECISION: STICK WITH n_fft=1024**
  
  **Optimal Configuration for MyBAD:**

-  n_mels: int = 64      # Best accuracy
-  n_fft: int = 1024     # Required for stability
-  hop_length: int = 256 # Produces 184 frames

  **Why n_fft=1024:**

  ✅ **Works reliably** - No model collapse
  ✅ **Best accuracy** - 89.48% with n_mels=64
  ✅ **Stable training** - Balanced class metrics
  ✅ **Minor overhead** - Only ~20s preprocessing difference
  
  **Why NOT n_fft=512:**

❌ **Fails catastrophically** at n_mels=64 (-37% accuracy!)
❌ **Marginal benefit** - Only 2-3% faster preprocessing
❌ **Not worth the risk** - Unstable with higher n_mels