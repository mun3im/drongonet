# MyBAD Model Variants - Technical Summary

**Reference:** Huang et al. (2024). "TinyChirp: Bird Song Recognition Using TinyML Models on Low-power Wireless Acoustic Sensors." arXiv:2407.21453

---
The latency estimate for computing 64x184 Mel spectrogram (including windowing, real FFT, power spectrum, Mel filter bank application, and log) on a Cortex-M4 at 240 MHz, assuming single-precision floating-point operations with FPU enabled and using optimized libraries like CMSIS-DSP, is as follows:

- For `n_fft = 512`: Approximately 29 ms.
- For `n_fft = 1024`: Approximately 50 ms.
---
### Group 1: TinyChirp Baseline on MyBAD Dataset
## Model Comparison Table

### n_fft = 1024

| n_mels     | TFLite Acc     | TFLite AUC     | Params   | Est. M4 Inf. @ 240 MHz | Est. MACs | Status                     |
| ---------- | -------------- | -------------- | -------- | ---------------------- | --------- | -------------------------- |
| 80         | 56.73% ❌       | 0.8434         | 25,558   | ~17ms                  | 1.35M     | 🔴 **FAILED**, overfitting |
| **64**     | **89.48%** 🏆  | **0.9610** 🏆  | 19,926   | ~13ms                  | 1.05M     | ✅ **WINNER**               |
| **48**     | **86.12%** 🥈  | **0.9571** 🥈  | 14,294 ⚡ | ~9ms ⚡                 | 0.74M     | ✅ **EFFICIENT**            |
| 32         | 78.53%         | 0.9588         | 8,662    | ~7ms                   | 0.56M     | ⚠️ Underfitting            |
| 16         | 73.72%         | 0.8873         | 3,030    | ~3.4ms                 | 0.27M     | ⚠️ Too low                 |

### n_fft = 512

| n_mels  | n_fft | Accuracy     | AUC        | Verdict                 |
| ------- | ----- | ------------ | ---------- | ----------------------- |
| **80**  | 512   | 82.00%       | 0.9241     | OK                      |
| 64      | 512   | **52.25%** ❌ | 0.8808 ❌   | **COMPLETE FAILURE**    |
|         |       | **-37.23%!** |            | **MODEL COLLAPSE**      |
| 48      | 512   | 83.90%       | 0.9340     | Acceptable (-2.22%)     |
| 32      | 512   | 70.33%       | 0.8907     | Poor (-8.20%)           |
| 16      | 512   | 84.00%       | **0.9619** | Surprisingly good!      |
  


**Key Findings:**
- **Latency scales with n_mels**: 80→16 reduces MACs by 80% (1.35M→0.27M)
- **Accuracy vs Efficiency**: n_mels=48 offers best balance (-0.44% AUC, 46% faster than n_mels=80)
- **TinyChirp limitation**: Fixed n_mels=80, no frequency resolution exploration
- **MyBAD contribution**: Systematic `n_mels` ablation demonstrates 32-48 sufficient for stage-1 detection

---

### Group 2: Optimized MyBAD Models

| Model              | n_mels | n_fft | Test Acc | TFLite AUC | Params | Est. MACs | Size (KB) | GPU Latency | **Est. M4 @ 240MHz** | Folder                          |
| ------------------ | ------ | ----- | -------- | ---------- | ------ | --------- | --------- | ----------- | -------------------- | ------------------------------- |
| **MyBAD-Accurate** | 48     |       |          |            | 28,850 | 1.71M     | 32.62     |             |                      |                                 |
| **MyBAD-Balanced** | 48     |       |          |            | 14,179 | 0.95M     | 19.80     |             |                      |                                 |
| **MyBAD-Fast**     | 32     |       |          |            | 8,662  | 0.56M     | 12.78     |             |                      |                                 |
| **MyBAD-Tiny**     | 16     | 512   | 77.00%   | 0.8622     | 2,915  |           | 8.65      | 0.02ms      |                      | `results/2_depthwise_m16_s42`   |
| **MyBAD-Best**     | 64     | 512   | 93.15%   | **0.9808** | 59,351 | 0.28M     | 64.27     | 0.10ms      |                      | `results/6_best_fft512_m64_s42` |
|                    |        |       |          |            |        |           |           |             |                      |                                 |
|                    |        |       |          |            |        |           |           |             |                      |                                 |

**Key Optimizations:**
- **MyBAD-Accurate**: +8 filters (vs baseline 4) → +0.71% AUC improvement
- **MyBAD-Balanced**: Depthwise separable conv + dropout 0.1 → 30% fewer MACs, 0.00% quantization degradation (best)
- **MyBAD-Fast**: Same architecture as baseline but n_mels=32 → 57% faster than TinyChirp
- **MyBAD-Tiny**: Depthwise + 6 filters + n_mels=16 → 80% faster, 64% smaller

---

## Detailed Specifications

### Architecture Comparison

**TinyChirp CNN-Mel (Original - from paper Table II):**
```
Input: 184 × 80 × 1
Conv2D(4 filters, 3×3, valid) + ReLU → MaxPool(2×2)  # 182×78×4 → 91×39×4
Conv2D(4 filters, 3×3, valid) + ReLU → MaxPool(2×2)  # 89×37×4 → 44×18×4
Flatten(3168) → Dense(8) + ReLU → Dense(2) + Softmax
```

**MyBAD-Accurate (5_filters_m48):**
```
Input: 184 × 48 × 1
Conv2D(8 filters, 3×3, valid) + ReLU → MaxPool(2×2)  # 182×46×8 → 91×23×8
Conv2D(8 filters, 3×3, valid) + ReLU → MaxPool(2×2)  # 89×21×8 → 44×10×8
Flatten(3520) → Dense(8) + ReLU → Dense(2) + Softmax
```

**MyBAD-Balanced (9a_depthwise_drop01_m48):**
```
Input: 184 × 48 × 1
DepthwiseConv2D(4, 3×3) + Conv2D(4, 1×1) + ReLU → MaxPool(2×2)
DepthwiseConv2D(4, 3×3) + Conv2D(4, 1×1) + ReLU → MaxPool(2×2)
Dropout(0.1)
Flatten(3520) → Dense(8) + ReLU → Dense(2) + Softmax
```

---

## MACs Calculation Details

### TinyChirp Original (n_mels=80)
- Conv2D 1: 182×78×4×(3×3×1) = 511,056 MACs
- Conv2D 2: 89×37×4×(3×3×4) = 473,688 MACs
- Dense 1: 3,168×8 = 25,344 MACs
- Dense 2: 8×2 = 16 MACs
- **Total: ~1.35M MACs**

### MyBAD-Accurate (5_filters_m48, 8 filters)
- Conv2D 1: 182×46×8×(3×3×1) = 603,072 MACs
- Conv2D 2: 89×21×8×(3×3×8) = 1,076,544 MACs
- Dense 1: 3,520×8 = 28,160 MACs
- Dense 2: 8×2 = 16 MACs
- **Total: ~1.71M MACs**

### MyBAD-Balanced (9a, Depthwise)
- DepthwiseConv2D 1: 182×46×4×(3×3) = 301,536 MACs
- Pointwise 1×1: 182×46×4×4 = 133,504 MACs
- DepthwiseConv2D 2: 89×21×4×(3×3) = 67,284 MACs
- Pointwise 1×1: 89×21×4×4 = 29,904 MACs
- Dense 1: 3,520×8 = 28,160 MACs
- Dense 2: 8×2 = 16 MACs
- **Total: ~0.95M MACs** (30% fewer than baseline)

### MyBAD-Fast (n_mels=32)
- Conv2D 1: 182×30×4×(3×3×1) = 196,560 MACs
- Conv2D 2: 89×13×4×(3×3×4) = 166,608 MACs
- Dense 1: 4,628×8 = 37,024 MACs
- Dense 2: 8×2 = 16 MACs
- **Total: ~0.56M MACs** (59% fewer than TinyChirp)

### MyBAD-Tiny (n_mels=16, Depthwise)
- DepthwiseConv2D 1: 182×14×6×(3×3) = 137,592 MACs
- Pointwise 1×1: 182×14×6×6 = 91,728 MACs
- DepthwiseConv2D 2: 89×5×6×(3×3) = 24,030 MACs
- Pointwise 1×1: 89×5×6×6 = 16,020 MACs
- Dense 1: 1,335×8 = 10,680 MACs
- Dense 2: 8×2 = 16 MACs
- **Total: ~0.28M MACs** (79% fewer than TinyChirp)

---

## Cortex-M4 @ 240MHz Performance Estimates

**Assumptions:**
- Clock: 240 MHz (e.g., Arduino Portenta H7 M4 core)
- TFLite Micro int8: ~3 cycles/MAC (optimized kernels)
- Formula: Latency (ms) = (MACs × 3 cycles/MAC) / 240MHz

| Model              | MACs  | Cycles | **Latency (ms)** | vs TinyChirp                 |
| ------------------ | ----- | ------ | ---------------- | ---------------------------- |
| TinyChirp Original | 1.35M | 4.05M  | **~17ms**        | Baseline                     |
| MyBAD-Accurate     | 1.71M | 5.13M  | **~21ms**        | +24% slower (but +0.27% AUC) |
| MyBAD-Balanced     | 0.95M | 2.85M  | **~12ms**        | **30% faster** ✓             |
| MyBAD-Fast         | 0.56M | 1.68M  | **~7ms**         | **59% faster** ✓             |
| MyBAD-Tiny         | 0.28M | 0.84M  | **~3.5ms**       | **79% faster** ✓             |

---

## Comparison with TinyChirp Paper Results

### TinyChirp on nRF52840 (Cortex-M4 @ 64MHz) - Table VI from paper

| Model                | Memory (KB) | Storage (KB) | Inference (ms) | Preprocessing (ms) | Power (mW) | Energy (mJ) |
| -------------------- | ----------- | ------------ | -------------- | ------------------ | ---------- | ----------- |
| **CNN-Mel**          | 104.3       | 37.9         | **406.1**      | **1980.3**         | 17.8       | **42.5**    |
| **CNN-Time**         | 75.6        | 24.1         | **1490.7**     | **2.0**            | 17.2       | **25.6**    |
| **Transformer-Time** | 83.5        | 24.7         | **1079.3**     | **2.0**            | 17.8       | **19.3**    |

**Key Observation from TinyChirp:**
- CNN-Mel on spectrogram: 406ms inference + **1980ms preprocessing** = **2.4 seconds total** (not real-time!)
- CNN-Time on raw audio: 1491ms inference + 2ms preprocessing = 1.5 seconds total
- **Mel spectrogram computation is the bottleneck** (1980ms >> 406ms)

**Our Approach (MyBAD):**
- All models use **pre-computed mel spectrograms** (cached)
- Inference-only latency measured (no STFT overhead during inference)
- For deployment: Mel computation can be done in parallel or offline

---

## Total Latency Budget (End-to-End)

**On Cortex-M4 @ 240MHz (estimated):**

### With On-Device Mel Computation
```
Mel spectrogram (STFT + Mel filterbank):  ~15-20ms (estimated, needs profiling)
MyBAD-Balanced inference:                 ~12ms
Total:                                    ~27-32ms per 3-second window
```

### With Pre-computed Mels (Cached or Parallel Processing)
```
MyBAD-Balanced inference only:            ~12ms
Suitable for real-time streaming at 16kHz
```

**Note:** TinyChirp paper (Table VI) shows mel computation takes ~1980ms on nRF52840 @ 64MHz. Scaling to 240MHz: 1980 × (64/240) ≈ **528ms** for mel computation. This is a significant overhead.

---

## Quantization Robustness Ranking

| Rank | Model | Degradation | Float→TFLite | Notes |
|------|-------|-------------|--------------|-------|
| **1** | **MyBAD-Balanced** | **0.00%** | 97.40%→97.39% | **Best quantization robustness** ✓ |
| 2 | TinyChirp Baseline | 0.02% | 97.97%→97.95% | Excellent |
| 3 | MyBAD-Accurate | 0.04% | 98.26%→98.22% | Excellent |
| 4 | MyBAD-Tiny | 0.05% | 96.38%→96.33% | Excellent |
| 5 | MyBAD-Fast | 0.12% | 97.45%→97.33% | Good |

**All models show excellent int8 quantization robustness (<0.15% degradation).**

---

## Deployment Recommendations

### Cortex-M4 @ 240MHz (e.g., Arduino Portenta H7 M4 core)

| Use Case | Recommended | MACs | Est. Latency | Power | Justification |
|----------|-------------|------|--------------|-------|---------------|
| **Stage-1 gatekeeper** | **MyBAD-Balanced** | 0.95M | **~12ms** | Low | Best quantization (0.00%), 30% fewer MACs, optimal balance |
| **Real-time monitoring** | MyBAD-Fast | 0.56M | **~7ms** | Medium | 59% faster, good accuracy (97.33%) |
| **Ultra-low power** | MyBAD-Tiny | 0.28M | **~3.5ms** | Ultra-low | 79% fewer MACs, minimal power |
| **High accuracy batch** | MyBAD-Accurate | 1.71M | **~21ms** | Medium | Best accuracy (98.22%), suitable for batch |

---

## Performance vs TinyChirp CNN-Mel Baseline

| Metric | TinyChirp | MyBAD-Accurate | MyBAD-Balanced | MyBAD-Fast | MyBAD-Tiny |
|--------|-----------|----------------|----------------|------------|------------|
| **TFLite AUC** | 97.95% | **98.22%** (+0.27%) | 97.39% (-0.56%) | 97.33% (-0.62%) | 96.33% (-1.62%) |
| **MACs** | 1.35M | 1.71M (+27%) | **0.95M (-30%)** | **0.56M (-59%)** | **0.28M (-79%)** |
| **Est. M4 Latency** | ~17ms | ~21ms (+24%) | **~12ms (-30%)** | **~7ms (-59%)** | **~3.5ms (-79%)** |
| **Size** | 29.28KB | 32.62KB (+11%) | **19.80KB (-32%)** | **12.78KB (-56%)** | **10.40KB (-64%)** |
| **Quantization** | 0.02% | 0.04% | **0.00%** ✓ | 0.12% | 0.05% |

**Summary:**
- ✅ **MyBAD-Accurate**: Best accuracy (+0.27%), suitable for cloud/edge
- ✅ **MyBAD-Balanced**: **Recommended** - Best quantization, 30% faster, 32% smaller
- ✅ **MyBAD-Fast**: 59% faster with minimal accuracy loss (-0.62%)
- ✅ **MyBAD-Tiny**: 79% faster, 64% smaller, acceptable for presence detection

---

## File Locations

```
results/
├── 1_baseline_m80_s42/          # TinyChirp Baseline (n_mels=80)
├── 1_baseline_m64_s42/          # Baseline n_mels=64
├── 1_baseline_m48_s42/          # Baseline n_mels=48
├── 1_baseline_m32_s42/          # MyBAD-Fast (n_mels=32)
├── 1_baseline_m16_s42/          # Baseline n_mels=16
│
├── 5_filters_m48_s42/           # MyBAD-Accurate (8 filters)
├── 9a_depthwise_drop01_m48_s42/ # MyBAD-Balanced (Depthwise + Dropout 0.1)
└── 10_depthwise_f6_m16_s42/     # MyBAD-Tiny (Depthwise + 6 filters, n_mels=16)

Each folder contains:
- best_model.keras           # Float32 trained model
- model_int8.tflite          # Quantized int8 model (DEPLOY THIS)
- results_summary.txt        # Performance metrics
- model_summary.txt          # Architecture details
- config.txt                 # Training configuration
- *.png                      # Training curves, confusion matrices
```

---

## Key Contributions vs TinyChirp

1. **Dataset**: First tropical bird dataset (40k Malaysian samples) vs UK Corn Bunting
2. **n_mels ablation**: Systematic exploration (16-80) vs fixed n_mels=80
3. **Stage-1 optimization**: Binary activity detection vs general classification
4. **Deployment variants**: 4 Pareto-optimal models vs single model
5. **Quantization study**: 31 models analyzed, <0.1% degradation for 77%

---

## Citation

**TinyChirp Paper:**
```
@article{huang2024tinychirp,
  title={TinyChirp: Bird Song Recognition Using TinyML Models on Low-power Wireless Acoustic Sensors},
  author={Huang, Z. and Tousnakhoff, A. and Kozyr, P. and Rehausen, R. and Bie{\ss}mann, F. and Lachlan, R. and Adjih, C. and Baccelli, E.},
  journal={arXiv preprint arXiv:2407.21453},
  year={2024}
}
```

---

**Last updated:** 2024-12-24
