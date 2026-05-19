# Model Rankings (Updated 2026-01-26)

## Executive Summary

### Two-Family Cascade Architecture

| Stage | Family | Task | Best Model | Size | Latency |
|-------|--------|------|------------|------|---------|
| **Gatekeeper** | XiaoChirp | Bird? (binary) | 7e_strided_focal_m16 | 7 KB | 0.05ms |
| **Classifier** | MynaNet | Which species? | DS-CNN+SE+Att | 388 KB | ~15ms |

### XiaoChirp Role (This Repo)
- **Purpose**: Fast binary filter — reject non-bird audio before expensive classification
- **Best Gatekeeper**: `7e_strided_focal_tuned_fft512_m16`
  - Threshold: 0.34 for 98% recall
  - Precision: 81.81% at 98% recall
  - Size: 7.03 KB, Inference: 0.05ms
- **Dataset Ceiling**: 98.86% accuracy (14a_deeper_gap_fft1024_m80, 33 KB)

### MynaNet Role (~/Dropbox/Conda/mynanet)
- **Purpose**: Multi-class species classification
- **Current**: DS-CNN+SE+Att @ 94.11% accuracy, 388 KB
- **Target**: 95-95.5% via channel widening + residuals

### System Efficiency
```
1000 audio segments (50% birds) → Gatekeeper → 599 pass → Classifier
                                              (40% compute savings)
```

---

## Dataset Asymptote: The Beasts (14* Deeper Models)

| Model | Accuracy | AUC | Recall | Precision | Size | Inference |
|-------|----------|-----|--------|-----------|------|-----------|
| **14a_deeper_gap_fft1024_m80** | **98.86%** | 0.9985 | **99.08%** | 98.65% | 32.82 KB | 1.44ms |
| 14_deeper_gap_fft512_m80 | 98.70% | **0.9990** | 98.72% | 98.68% | 32.82 KB | 2.18ms |
| 14b_deeper_1x1_gap_fft1024_m80 | 98.82% | 0.9983 | 99.12% | 98.53% | 33.38 KB | 3.08ms |
| 14_deeper_gap_fft512_m64 | 97.90% | 0.9974 | - | - | 32.82 KB | 1.70ms |
| 14a_deeper_gap_fft1024_m64 | 97.70% | 0.9968 | - | - | 32.82 KB | 1.64ms |

**Performance Ceiling**: ~99% accuracy, 0.999 AUC is the asymptote for MyBAD dataset.

---

## Ablation Study Results (fft512_m16)

| Model | Accuracy | AUC | Size | Inference | Change |
|-------|----------|-----|------|-----------|--------|
| 1a_baseline2d | 91.34% | 0.9682 | 7.28 KB | 0.06ms | Baseline |
| 4a_baseline_gap | 88.56% | 0.9503 | 4.09 KB | 0.06ms | -2.78% (GAP hurts) |
| 7a_gap_focal_loss | 93.00% | 0.9746 | 5.23 KB | 0.08ms | +4.44% (Focal recovers) |

**Key Insight**: GAP alone reduces capacity too much (-2.78%), but focal loss more than compensates (+4.44% net gain over baseline).

---

## Top Models by Accuracy

| Rank | Model | Accuracy | AUC | Size | Inference |
|------|-------|----------|-----|------|-----------|
| 1 | 7e_strided_focal_tuned_fft512_m80 | **96.96%** | 0.9947 | 7.09 KB | 0.23ms |
| 2 | 7e_strided_focal_tuned_fft1024_m80 | 96.56% | 0.9938 | 7.09 KB | 0.22ms |
| 3 | 7f_strided_focal_no1x1_fft512_m80 | 95.42% | 0.9881 | 5.68 KB | 0.19ms |
| 4 | 7e_strided_focal_tuned_fft512_m64 | 94.26% | 0.9856 | 7.08 KB | 0.18ms |
| 5 | 7g_strided_focal_depthwise_fft512_m80 | 93.76% | 0.9832 | 5.87 KB | 0.15ms |

---

## Top Models by AUC

| Rank | Model | AUC | Accuracy | Size | Inference |
|------|-------|-----|----------|------|-----------|
| 1 | 7e_strided_focal_tuned_fft512_m80 | **0.9947** | 96.96% | 7.09 KB | 0.23ms |
| 2 | 7e_strided_focal_tuned_fft1024_m80 | 0.9938 | 96.56% | 7.09 KB | 0.22ms |
| 3 | 7f_strided_focal_no1x1_fft512_m80 | 0.9881 | 95.42% | 5.68 KB | 0.19ms |
| 4 | 7e_strided_focal_tuned_fft512_m64 | 0.9856 | 94.26% | 7.08 KB | 0.18ms |
| 5 | 7g_strided_focal_depthwise_fft512_m80 | 0.9832 | 93.76% | 5.87 KB | 0.15ms |

---

## Fastest Models (Gatekeeper Priority)

| Rank | Model | Inference | Size | Accuracy | AUC |
|------|-------|-----------|------|----------|-----|
| 1 | 7f_strided_focal_no1x1_fft512_m16 | **0.04ms** | 5.62 KB | 90.70% | 0.9635 |
| 2 | 7f_strided_focal_no1x1_fft1024_m16 | 0.04ms | 5.62 KB | 91.18% | 0.9660 |
| 3 | 7g_strided_focal_depthwise_fft512_m16 | 0.04ms | 5.80 KB | 88.98% | 0.9523 |
| 4 | 7g_strided_focal_depthwise_fft1024_m16 | 0.04ms | 5.80 KB | 88.86% | 0.9580 |
| 5 | 7e_strided_focal_tuned_fft512_m16 | 0.05ms | 7.03 KB | 92.34% | 0.9747 |
| 6 | 7e_strided_focal_tuned_fft1024_m16 | 0.05ms | 7.03 KB | 91.32% | 0.9672 |

---

## Smallest Models

| Rank | Model | Size | Accuracy | AUC | Inference |
|------|-------|------|----------|-----|-----------|
| 1 | 4a_baseline_gap_fft512_m16 | **4.09 KB** | 88.56% | 0.9503 | 0.06ms |
| 2 | 7a_gap_focal_loss_fft512_m16 | 5.23 KB | 93.00% | 0.9746 | 0.08ms |
| 3 | 7f_strided_focal_no1x1_fft512_m16 | 5.62 KB | 90.70% | 0.9635 | 0.04ms |
| 4 | 7f_strided_focal_no1x1_fft1024_m16 | 5.62 KB | 91.18% | 0.9660 | 0.04ms |
| 5 | 7g_strided_focal_depthwise_fft512_m16 | 5.80 KB | 88.98% | 0.9523 | 0.04ms |

---

## Model Variant Comparison (m80 config)

| Variant | Accuracy | AUC | Size | Inference | Notes |
|---------|----------|-----|------|-----------|-------|
| 7e_tuned_fft512 | **96.96%** | 0.9947 | 7.09 KB | 0.23ms | Best overall |
| 7f_no1x1_fft512 | 95.42% | 0.9881 | 5.68 KB | 0.19ms | Smaller, fast |
| 7g_depthwise_fft512 | 93.76% | 0.9832 | 5.87 KB | 0.15ms | Fastest m80 |

---

## Recommendations

### For Highest Accuracy
**7e_strided_focal_tuned_fft512_m80**
- 96.96% accuracy, 0.9947 AUC
- 7.09 KB, 0.23ms inference

### For Gatekeeper (Speed + Size Priority)
**7e_strided_focal_tuned_fft512_m16**
- 92.34% accuracy, 0.9747 AUC
- 7.03 KB, 0.05ms inference
- Best accuracy among fastest models

### For Ultra-Compact Gatekeeper
**7f_strided_focal_no1x1_fft512_m16**
- 90.70% accuracy, 0.9635 AUC
- 5.62 KB, 0.04ms inference
- Smallest + fastest

### For Balanced Performance
**7f_strided_focal_no1x1_fft512_m80**
- 95.42% accuracy, 0.9881 AUC
- 5.68 KB, 0.19ms inference
- Good accuracy with small size

---

## Architecture Evolution

```
1a_baseline2d (Flatten+Dense, BCE)
    │
    ▼ (-2.78%)
4a_baseline_gap (GAP, BCE) ── capacity loss
    │
    ▼ (+4.44%)
7a_gap_focal_loss (GAP, Focal) ── focal loss recovers
    │
    ├──▶ 7e_strided_focal_tuned ── best accuracy (96.96%)
    ├──▶ 7f_strided_focal_no1x1 ── balanced (95.42%, 5.68KB)
    └──▶ 7g_strided_focal_depthwise ── fastest (0.04ms)
```

---

---

## Threshold Analysis for 98% Recall (Gatekeeper Mode)

All fast m16 models can achieve 98% recall by lowering the threshold:

| Model | Threshold | Recall | Precision | FPR | Speed |
|-------|-----------|--------|-----------|-----|-------|
| **7e_tuned_m16** | 0.3398 | 98.04% | **81.81%** | 21.80% | 0.05ms |
| 7a_focal_m16 | 0.3555 | 98.00% | 81.69% | 21.96% | 0.08ms |
| 7f_no1x1_m16 | 0.3242 | 98.00% | 75.80% | 31.28% | **0.04ms** |

### Interpretation

- **7e_tuned_m16**: Best precision at 98% recall (81.81%), very fast (0.05ms)
- **7f_no1x1_m16**: Fastest (0.04ms) but 6% lower precision, 10% higher FPR
- **7a_focal_m16**: Same precision as 7e but slower

### Gatekeeper FPR Impact

At 98% recall with threshold adjustment:
- **7e/7a**: 21.8% FPR → For every 100 non-bird sounds, ~22 false alarms pass to classifier
- **7f**: 31.3% FPR → For every 100 non-bird sounds, ~31 false alarms pass to classifier

The classifier (7e_m80 at 97% accuracy) will filter most false alarms.

---

## Final Recommendations

### Best Gatekeeper
**7e_strided_focal_tuned_fft512_m16**
```python
MODEL = "7e_strided_focal_tuned_fft512_m16_s42_9747/model_int8.tflite"
THRESHOLD = 0.3398  # For 98% recall
# Size: 7.03 KB, Speed: 0.05ms
# Recall: 98.04%, Precision: 81.81%
```

### Fastest Gatekeeper (if 6% precision loss acceptable)
**7f_strided_focal_no1x1_fft512_m16**
```python
MODEL = "7f_strided_focal_no1x1_fft512_m16_s42_9635/model_int8.tflite"
THRESHOLD = 0.3242  # For 98% recall
# Size: 5.62 KB, Speed: 0.04ms
# Recall: 98.00%, Precision: 75.80%
```

### Best Classifier (2nd Stage) - Compact
**7e_strided_focal_tuned_fft512_m80**
```python
MODEL = "7e_strided_focal_tuned_fft512_m80_s42_9946/model_int8.tflite"
THRESHOLD = 0.5  # Default
# Size: 7.09 KB, Speed: 0.23ms
# Accuracy: 96.96%, AUC: 0.9947
```

### Ultimate Classifier (The Beast)
**14a_deeper_gap_fft1024_m80**
```python
MODEL = "14a_deeper_gap_fft1024_m80_s42_9985/model_int8.tflite"
THRESHOLD = 0.5  # Default
# Size: 32.82 KB, Speed: 1.44ms
# Accuracy: 98.86%, Recall: 99.08%, AUC: 0.9985
```

---

## Cascaded System: XiaoChirp → MynaNet

The full pipeline uses two model families:
1. **XiaoChirp** (this repo): Binary gatekeeper (bird vs no-bird)
2. **MynaNet** (~/Dropbox/Conda/mynanet): Multi-class species classifier

### MynaNet Classifier Status

| Model | INT8 Accuracy | Size | Notes |
|-------|---------------|------|-------|
| DS-CNN+SE+Att | 94.11% | 388 KB | Current best compact |
| MobileNetV3 | 95.33% | 1996 KB | Accuracy target |
| DS-CNN (target) | 95-95.5% | <512 KB | After optimizations |

*See: `~/Dropbox/Conda/mynanet/| Path to 95%+ Accuracy for DS-CNN+SE+Att.md`*

### Full Cascade Performance

```
XiaoChirp Gatekeeper (7e_m16) → MynaNet Classifier (DS-CNN+SE+Att)
         98% recall                    94.11% accuracy
         7.03 KB                       388 KB
         0.05ms                        ~10-20ms (estimated)
```

| Stage | Model | Metric | Size | Latency |
|-------|-------|--------|------|---------|
| Gatekeeper | 7e_strided_focal_m16 | 98% recall | 7 KB | 0.05ms |
| Classifier | DS-CNN+SE+Att | 94.11% acc | 388 KB | ~15ms |
| **Total** | - | - | **395 KB** | **~15ms** |

### System Efficiency

For 1000 audio segments (50% contain birds):
- 500 bird + 500 non-bird inputs
- Gatekeeper passes: 490 birds + 109 false alarms = 599 total
- Classifier processes: 599 (not all 1000)
- **Compute savings**: 40% fewer classifier invocations

### XiaoChirp Role Summary

XiaoChirp's job is to **filter quickly**, not classify:
- Reject obvious non-bird audio (wind, silence, rain, traffic)
- Pass anything that might be a bird to MynaNet
- Priority: High recall (98%+) over precision
- Threshold: 0.34 (lowered from 0.5 for 98% recall)

---

## XiaoChirp-Only Cascade (Alternative)

For deployments without MynaNet, XiaoChirp beasts can serve as classifiers:

| System | Recall | Size | Latency | Use Case |
|--------|--------|------|---------|----------|
| 7e_m16 → 7e_m80 | 95.06% | 14 KB | 0.28ms | Ultra-constrained MCU |
| 7e_m16 → 14a_m80 | 97.02% | 40 KB | 1.49ms | Binary detection only |
