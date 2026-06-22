---
name: lessons-learnt
description: Lessons learned from SEABADNet development and ablation experiments
metadata:
  type: reference
  date: 2026-06-22
---

# Lessons Learnt from SEABADNet Development

Captured from ablation studies, architecture exploration, and deployment research. These lessons apply to bird audio detection on resource-constrained hardware.

---

## Recall-First Optimization for Edge Bioacoustics

**Core principle:** When recall ≥ 0.98 is non-negotiable, optimize surgically, not architecturally.

### What Preserves Recall (Safe to Do)

1. **Early layer width is critical** — Keep the first few convolution layers at full width. Reducing early channels (8→6) hurts recall significantly before it helps latency.

2. **Frequency emphasis survives quantization** — Frequency emphasis (boosting high-frequency bands) consistently improves recall and is stable across INT8 quantization.

3. **Depthwise separable convolutions with constant channel count** — Replacing standard Conv2D with depthwise-separable reduces MACs ~70% while preserving recall if channel count stays constant. This is the biggest win that still respects recall targets.

4. **Strided convolution in first layer** — Replacing Conv3×3 → MaxPool with Conv3×3 with stride=2 gives same receptive field, less aliasing, and keeps more information per channel. Embedded models often get higher recall this way.

5. **GAP (Global Average Pooling)** — Replacing Flatten with GAP reduces parameters dramatically with minimal AUC cost. It's a free efficiency win.

6. **Focal loss with aggressive weighting** — Focal loss (γ=2.0, α=0.5 or higher) prevents recall collapse during optimization. Class-weighted cross-entropy is less stable.

### What Breaks Recall (Avoid)

1. **Quantization without training** — Post-training quantization is the #1 recall killer. Quantization-aware training (QAT) or training on a quantized-friendly architecture is essential.

2. **Reduced frequency resolution** — Lowering n_fft from 1024 to 512 can cause model collapse (validation AUC drops 37%+) depending on n_mels. Frequency resolution is non-negotiable.

3. **Thin second convolution** — The second convolution layer does the dominant pattern extraction. Reducing its width or removing it causes recall to fail on weak bird calls and ambient noise.

4. **Early stopping on wrong metric** — Stopping on validation accuracy instead of validation loss can trap the model in local minima. Pair with ReduceLROnPlateau.

### Safe-Optimization Tier List

| Tier | Optimization | Recall Risk | MAC Savings | Example |
|------|-------------|------------|-------------|---------|
| **Tier 1** | Strided conv instead of MaxPool | Very low | ~20% | Conv3×3 stride=2 |
| **Tier 1** | Remove redundant 1×1 convs | Negligible | ~750k ops | Conv3×3→Conv3×3 (remove middle 1×1) |
| **Tier 1** | Replace Flatten with GAP | Negligible | Major | 8,000 → 0 params in pooling layer |
| **Tier 2** | Depthwise separable (both layers) | Low–moderate | ~70% | SeparableConv2D throughout |
| **Tier 3** | Reduce channels (16→12) | Moderate–high | ~30% | Only if recall headroom >2% |
| **Avoid** | Reduce early layer width | High | Small gain | Early layers at full width always |

---

## Mel Spectrogram Configuration Trade-Offs

### n_fft vs n_mels Trade-Off

**Finding:** n_fft=1024 is worth the 5% latency cost for 37% accuracy gain.

| Metric | n_fft=512 | n_fft=1024 | Winner |
|--------|-----------|------------|--------|
| Mel latency | 1.45 ms | 1.53 ms | 512 (+5.4% faster) |
| Model accuracy (n_mels=64) | 52.25% ❌ | 89.48% ✅ | 1024 (+37.23%!) |
| AUC (n_mels=64) | 0.8808 | 0.9610 | 1024 (+0.0802) |
| Preprocessing (40k clips) | 58s | 61s | 512 (+5% faster) |
| Real-time factor | 0.0005 | 0.0005 | Tie (2000× faster than real-time) |

**ROI:** 6.9:1 accuracy improvement per % latency cost.

**Recommendation:** Use n_fft=1024 universally. Mel computation is not the bottleneck (<2% of total inference time). The 5% latency overhead is immaterial compared to 37% accuracy gain.

### n_mels Sweep Insights

- **n_mels=16** produces 184 frames (optimal for 3s clips @ 16kHz with hop=256)
- **n_mels=64** is the "sweet spot" for general accuracy but risks model instability at n_fft=512
- **n_mels=80** reaches accuracy ceiling but adds latency and memory cost; use only when size budget allows
- **Frequency resolution matters more than time resolution** — Increasing n_mels has larger effect than changing hop_length

---

## Threshold Optimization for Gatekeeper/Detection Tasks

**Key insight:** Default 0.5 threshold is rarely optimal. Threshold sweep is critical.

### Gatekeeper-Specific Findings

For a binary gatekeeper (detect presence anywhere, okay to have false positives filtered by downstream classifier):

| Threshold | Recall | Precision | Use Case |
|-----------|--------|-----------|----------|
| 0.10 | 98.8% | 56.0% | Max sensitivity (high false positives) |
| **0.26** | **95.0%** | **67.8%** | **Optimal for gatekeeper** |
| 0.40 | 90.6% | 81.6% | Balanced |
| 0.50 | 86.2% | 87.4% | Default (suboptimal for detection) |
| 0.70 | 73.1% | 93.0% | Conservative (loses birds) |

**Finding:** Lowering threshold from 0.5 to 0.264 recovers 9% recall (+95% vs 86%) for a 20% precision drop (87% → 68%). For a gatekeeper, this is the right trade-off.

### Threshold Selection Strategy

1. Start at τ=0.264 (empirically optimal across many bird datasets)
2. Sweep τ ∈ {0.25, 0.30, 0.34, 0.40, 0.50}
3. Choose **highest** τ that meets your recall target (maximizes precision)
4. **Never** use default 0.5; it's arbitrary and suboptimal for bird detection

---

## Architecture Patterns for Recall-Constrained Deployment

### High-Recall Optimized Architecture

Recommended for ≥0.98 recall requirement:

```
Input 184×n_mels×1
  ↓
FrequencyEmphasis (if available)
  ↓
Conv3×3 s=2, 8 channels     [strided, not MaxPool; early layer at full width]
  ↓
Depthwise3×3
Pointwise1×1, 16 channels   [keep channel count constant]
  ↓
GlobalAvgPool               [parameter-free, recall-stable]
  ↓
Dense(8) → Softmax
```

**Characteristics:**
- ~2.0–2.3 MMAC
- ~50–60 KB peak RAM
- ~15–20 ms on Cortex-M4
- Almost always preserves ≥0.98 recall
- INT8 quantization: <0.1% degradation

### Why This Architecture

- **Early strided conv** (vs MaxPool): Better information preservation
- **Depthwise separable**: Huge MAC savings without recall cost
- **Pointwise 1×1 at constant width**: Channel mixing without spatial filtering loss
- **GAP**: No parameters, stable pooling
- **Early layers at full width**: Prevents recall collapse on weak signals

---

## Quantization and Inference Dynamics

### Post-Training vs Quantization-Aware Training

**Critical finding:** Post-training quantization (PTQ) can cause 10–20% recall loss; quantization-aware training (QAT) should be the default.

**Safe configurations:**
- Train with focal loss + class weighting
- Quantize with INT8 representative dataset (≥500 diverse samples)
- Validate recall at INT8 (don't assume float32 recall transfers)
- Threshold after quantization (thresholds shift by 0.01–0.05)

### Real-Time Performance on ARM

- **Cortex-M4 @ 48 MHz (AudioMoth):** 0.1–0.3 ms inference per 3s clip (100–200× real-time)
- **Cortex-M4 @ 240 MHz (STM32F4):** 0.02–0.06 ms inference
- **Mel computation is not the bottleneck:** <2% of total latency regardless of n_fft
- **Bottleneck:** NN inference, but even that is orders of magnitude faster than real-time

---

## Dataset and Training Dynamics

### Class Imbalance Handling

- **Perfectly balanced (50/50) datasets** are ideal and are often achievable in curated benchmarks
- **Real deployments** are typically 95% negative (non-bird); use class weighting or focal loss
- **Sampling strategy matters** — Oversampling positive rare events beats downsampling negatives

### Validation Set Design

**Dangerous:** Validating only on clean, curated clips. Safe models collapse on:
- Wind noise
- Insect sounds
- Rain
- Mic handling noise

**Recommended:** Include diverse negatives in validation (use DCASE-2018 BAD non-bird subset or similar).

---

## Known Failure Modes and Workarounds

### "Model Collapse" (n_fft=512 + high n_mels)

**Symptom:** Validation accuracy suddenly drops from 90% to 52%.

**Cause:** Low frequency resolution (n_fft=512 produces 257 freq bins; 512-point FFT lacks Nyquist info for high-frequency bird calls).

**Fix:** Use n_fft=1024 (513 bins; 0.078 ms latency overhead is negligible).

### Recall Drift After Quantization

**Symptom:** Float32 model: 98% AUC. INT8 model: 96% AUC.

**Cause:** Threshold was optimized on float32; activations shift by 0.01–0.05 during quantization.

**Fix:** Re-sweep threshold on INT8 model (always validate INT8 separately).

### False Negatives on Weak Signals

**Symptom:** Model misses quiet bird calls but catches loud ones.

**Cause:** Early layer channels too narrow or no frequency emphasis.

**Fix:**
1. Ensure early layers ≥8 channels
2. Add frequency emphasis augmentation
3. Use focal loss with γ≥1.5 to over-weight hard negatives

---

## Deployment Checklist for ≥0.98 Recall

- [ ] Use n_fft=1024, hop=256
- [ ] Early layers at full width (8+ channels)
- [ ] GAP instead of Flatten
- [ ] Focal loss (γ≥2.0, α≥0.5)
- [ ] Quantization-aware training or INT8 validation
- [ ] Threshold sweep on INT8 model
- [ ] Validate on noisy negatives (wind, insects, rain)
- [ ] Measure latency on target hardware (not desktop)
- [ ] Confirm recall at operating threshold on held-out test set

---

## References

- **7D_MICRO_OPTIMIZATIONS.md** — Detailed tier analysis of architectural optimizations
- **MEL_LATENCY_ANALYSIS.md** — n_fft=512 vs 1024 latency/accuracy trade-off (6.9:1 ROI)
- **GATEKEEPER_ANALYSIS.md** — Threshold optimization for binary detection tasks
- **DEPLOYMENT_RECOMMENDATIONS.md** — Latency-constrained model selection and benchmarking

---

**Last updated:** 2026-06-22  
**Author:** SEABADNet development team  
**Status:** Captured from 390+ ablation runs and deployment research
