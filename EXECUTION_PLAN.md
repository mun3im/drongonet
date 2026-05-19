# MyBAD Optimization - Execution Plan

**Date:** 2025-12-16
**Project:** MyBADv2 Bird Activity Detection - CNN-Mel Optimization
**Goal:** Achieve >97% AUC with optimal n_mels and architecture

---

## Phase 1: Baseline n_mels Sweep

**Objective:** Identify optimal mel spectrogram frequency resolution

### Command
```bash
./run_baseline_sweep.sh
```

### Experiments (5 total)
- Model: `1_baseline.py`
- n_mels values: 16, 32, 48, 64, 80
- Random seed: 42
- Dataset: MyBADv2 (50k samples, 80/10/10 split)

### Expected Results
Based on previous findings:
- **n_mels=48**: Best performance (~96.95% AUC, 0.63% StdDev) ⭐
- n_mels=32: Good balance (~93.39% AUC)
- n_mels=80: Unstable (~79.20% AUC, 20.63% StdDev) ⚠️

### Output Location
```
results/1_baseline_m{16,32,48,64,80}_s42/
```

---

## Phase 2: Full Model Sweep with Optimal n_mels

**Objective:** Compare all architectural variants using best n_mels

### Command
```bash
./run_all_models.sh [best_n_mels]
# Example: ./run_all_models.sh 48
```

### Models (12 total)

#### Core Variations (Models 1-6)
| # | Model | Description | Key Feature |
|---|-------|-------------|-------------|
| 1 | baseline | Pure CNN-Mel | No modifications |
| 2 | depthwise | Depthwise Separable Conv | 75% fewer parameters |
| 3 | dropout | Standard + Dropout(0.3) | Regularization |
| 4 | batchnorm | Standard + BatchNorm | Training stability |
| 5 | dense | Larger dense layer (32 units) | More capacity |
| 6 | filters | More filters (8 vs 4) | Better feature extraction |

#### Advanced Models (7-8)
| # | Model | Description | Key Features |
|---|-------|-------------|--------------|
| **7** | **best_accuracy** | **Lucky #7 - Enhanced** ⭐ | **6 filters, Depthwise, BatchNorm, SpatialDropout2D, Dense(16) + L2, Dropout(0.25), He init, AdamW on Linux** |
| 8 | hybrid | BatchNorm + Dropout(0.3) | Combined stability + regularization |

#### Depthwise + Dropout Sweep (9a-9d)
| # | Model | Description | Dropout Rate |
|---|-------|-------------|--------------|
| 9a | depthwise_drop01 | Depthwise + light regularization | 0.1 |
| 9b | depthwise_drop02 | Depthwise + moderate regularization | 0.2 |
| 9c | depthwise_drop03 | Depthwise + strong regularization | 0.3 |
| 9d | depthwise_drop04 | Depthwise + very strong regularization | 0.4 |

### Output Location
```
results/{model_number}_{model_name}_m{n_mels}_s42/
```

---

## Model 7: Best Accuracy Architecture

**Why Model 7 is Expected to Win:**

### Architecture Enhancements
```python
# Block 1
SeparableConv2D(6, 3x3) + He Init  # ← More filters than baseline (4)
→ BatchNormalization()              # ← Training stability
→ ReLU()
→ SpatialDropout2D(0.1)            # ← Spatial regularization
→ MaxPool(2x2)

# Block 2
SeparableConv2D(6, 3x3) + He Init
→ BatchNormalization()
→ ReLU()
→ SpatialDropout2D(0.1)
→ MaxPool(2x2)

# Classifier
Flatten()
→ Dense(16) + L2(0.001) + He Init  # ← Larger + regularized (baseline: 8)
→ ReLU()
→ Dropout(0.25)                     # ← Optimized rate
→ Dense(2, softmax)
```

### Platform-Specific Optimization
- **macOS (Apple Silicon)**: `legacy.Adam` (performance workaround)
- **Linux**: `AdamW(weight_decay=0.01)` (better generalization)
- **Other**: Standard `Adam`

### Comparison vs Baseline
| Feature | Baseline | Model 7 | Improvement |
|---------|----------|---------|-------------|
| Conv Type | Conv2D | SeparableConv2D | 75% fewer params |
| Filters | 4 | 6 | +50% capacity |
| BatchNorm | ❌ | ✅ | Stable training |
| Spatial Dropout | ❌ | ✅ (0.1) | Spatial regularization |
| Dense Size | 8 | 16 | +100% representation |
| L2 Reg | ❌ | ✅ (0.001) | Weight decay |
| Dropout | ❌ | ✅ (0.25) | Regularization |
| Initialization | Default | He Normal | Better gradients |

**Expected:** >97% AUC, improved stability, better TFLite quantization

---

## Execution Timeline

### Estimated Duration
- **Phase 1 (Baseline sweep)**: ~25-50 minutes (5 experiments × 5-10 min each)
- **Phase 2 (Full sweep)**: ~2-4 hours (12 experiments × 10-20 min each)
- **Total**: ~2.5-5 hours

### Progress Monitoring
```bash
# Watch master log in real-time
tail -f ablation_logs/baseline_sweep_*.log

# Watch all models sweep
tail -f ablation_logs/all_models_m48_*.log
```

---

## Results Structure

### Per-Experiment Output
Each `results/{model}_{params}/` directory contains:
- `results_summary.txt` - AUC, accuracy, model size, inference time
- `model_summary.txt` - Keras architecture details
- `config.txt` - Full training configuration
- `training_history.png` - Loss/accuracy curves
- `float_confusion_matrix.png` - Float32 model confusion matrix
- `tflite_confusion_matrix.png` - TFLite int8 confusion matrix
- `float_roc_curve.png` - Float32 ROC curve
- `tflite_roc_curve.png` - TFLite ROC curve
- `float_classification_report.txt` - Precision/recall/F1
- `tflite_classification_report.txt` - TFLite metrics
- `model_int8.tflite` - Quantized model for deployment
- `elapsed.txt` - Execution time

### Logs
- `ablation_logs/baseline_sweep_*.log` - Phase 1 master log
- `ablation_logs/all_models_m48_*.log` - Phase 2 master log
- `ablation_logs/{model}_m{n_mels}_s42.log` - Individual experiment logs

---

## Success Criteria

### Phase 1 Completion
- ✅ 5/5 baseline experiments successful
- ✅ Best n_mels identified (expected: 48)
- ✅ AUC summary generated

### Phase 2 Completion
- ✅ 12/12 model experiments successful
- ✅ Model 7 achieves highest AUC
- ✅ Model 7 AUC > 97%
- ✅ TFLite quantization degradation < 1%

### Quality Gates
- No training failures or crashes
- All models converge (early stopping triggered)
- TFLite models successfully quantized
- Inference time < 1ms on all models

---

## Next Steps After Completion

1. **Review results_summary.txt** for all models
2. **Compare AUC scores** across architectures
3. **Analyze Model 7 performance** vs baseline
4. **Generate comparison plots** (optional)
5. **Update RESULTS.md** with final findings
6. **Select best model** for deployment (expected: Model 7)

---

## Dataset Information

- **Name**: MyBADv2
- **Task**: Baby cry detection (binary classification)
- **Size**: 50,000 samples (25k positive + 25k negative)
- **Split**: 80% train / 10% val / 10% test
- **Audio**: 16kHz, 3 seconds (48,000 samples)
- **Features**: Mel spectrograms (184 time steps, n_mels frequency bins)
- **Augmentation**: Gaussian noise on training set

---

## Cache Information

Mel spectrograms are pre-computed and cached for efficiency:
- Location: `/Volumes/Evo/cache_mybad_m{n_mels}/`
- Format: Compressed NumPy arrays (.npz)
- Splits: train/, val/, test/
- Reuse: `--use_cache` flag skips preprocessing

**Note:** First run requires preprocessing (~10-15 min), subsequent runs use cache.
