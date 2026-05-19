# Baseline Sweep — Consolidated Reference

**Dataset:** MyBAD (50k — 40k train / 5k val / 5k test)
**Fixed:** n_fft=1024, hop_length=256, seed=42

---

## Overview

| | Baseline sweep | 1a — extended mels |
|---|---|---|
| Script | `1_baseline.py` | `1a_baseline2d.py` |
| n_fft | 1024 | 512, 1024 |
| n_mels | 16, 32, 48, 64, 80 | 16, 32, 48, 64, 80, 96, 112, 128 |
| Status | ✅ Complete | ✅ Complete |
| Best | 94.27% (m64) | 94.14% (fft1024, m128) |

---

## Results

### Baseline sweep (`1_baseline.py`, fft1024)

| n_mels | Accuracy | AUC    | Size (KB) | Inference (ms) |
|--------|----------|--------|-----------|----------------|
| 16     | 92.80%   | 0.9765 | 7.28      | 0.07           |
| 32     | 94.00%   | 0.9842 | 12.78     | 0.15           |
| 48     | 93.63%   | 0.9846 | 18.28     | 0.22           |
| **64** | **94.27%** 🏆 | 0.9862 | 23.78 | 0.29         |
| 80     | 94.20%   | **0.9868** | 29.28 | 0.37        |

### 1a — Extended mels sweep (`1a_baseline2d.py`)

#### n_fft = 512

| n_mels | Accuracy | AUC    | Size (KB) | Inference (ms) | F1     |
|--------|----------|--------|-----------|----------------|--------|
| 16     | 91.46%   | 0.9686 | 7.28      | 0.14           | 0.9146 |
| 32     | 93.18%   | 0.9764 | 12.78     | 0.34           | 0.9318 |
| 48     | 93.00%   | 0.9774 | 18.28     | 0.40           | 0.9300 |
| 64     | 93.78%   | 0.9806 | 23.78     | 0.72           | 0.9378 |
| **80** | **93.98%** | **0.9835** | 29.28 | 0.70        | **0.9398** |
| 96     | 93.50%   | 0.9818 | 34.78     | 0.90           | 0.9350 |
| 112    | 93.68%   | 0.9817 | 40.30     | 1.69           | 0.9368 |
| 128    | 93.92%   | 0.9835 | 45.80     | 1.40           | 0.9392 |

#### n_fft = 1024

| n_mels | Accuracy | AUC    | Size (KB) | Inference (ms) | F1     |
|--------|----------|--------|-----------|----------------|--------|
| 16     | 91.24%   | 0.9688 | 7.28      | 0.10           | 0.9124 |
| 32     | 92.34%   | 0.9746 | 12.78     | 0.22           | 0.9234 |
| 48     | 92.68%   | 0.9764 | 18.28     | 0.20           | 0.9268 |
| 64     | 92.80%   | 0.9769 | 23.78     | 0.27           | 0.9280 |
| 80     | 93.06%   | 0.9796 | 29.28     | 0.33           | 0.9306 |
| 96     | 92.98%   | 0.9772 | 34.78     | 0.41           | 0.9298 |
| 112    | 93.70%   | 0.9834 | 40.30     | 0.48           | 0.9370 |
| **128** | **94.14%** | **0.9847** | 45.80 | 0.55        | **0.9414** |

---

## Analysis

### n_fft Decision

n_fft=512 collapses at n_mels=64 (accuracy drops to ~52%, model predicts all-positive).
n_fft=1024 is stable across all n_mels tested. **Use n_fft=1024 throughout.**

### Accuracy vs n_mels

```
baseline fft=1024: 16→92.80  32→94.00  48→93.63  64→94.27★  80→94.20
1a       fft=512:  16→91.46  32→93.18  48→93.00  64→93.78   80→93.98★  96→93.50  112→93.68  128→93.92
1a       fft=1024: 16→91.24  32→92.34  48→92.68  64→92.80   80→93.06   96→92.98  112→93.70  128→94.14★
```

Baseline sweep peaks at m64. Extended sweep (1a) shows diminishing returns beyond m80 for fft=512 and slow continued gains to m128 for fft=1024.

---

## Decisions

- **Ablation standard config:** n_fft=1024, n_mels=64
- **Edge config:** n_fft=1024, n_mels=16 (7.28 KB, 0.07 ms, AUC 0.977)
- **1a finding:** gains above m80 are marginal (<0.5%) and cost disproportionate latency; m128 not used in final models

---

*Data: 2026-01-09 (baseline) · 2026-02-24 (1a sweep)*
