# MyBAD v4 - GPU Priority Experiments
**Updated**: 2026-01-15
**Environment**: tf215_gpu (GPU enabled)
**Execution**: Sequential with nice -n 10

---

## 🎯 UPDATED EVALUATION METRICS

### Primary Metrics (for ranking):
1. **TFLite int8 Accuracy** - Primary performance metric
2. **TFLite int8 F1-score** - Balanced classification quality
3. **Model Parameters (#params)** - Model size for int8 inference
4. **Inference Time** - Speed on target hardware

### Secondary Metrics (for analysis only):
- **Float32 AUC** - Only used to assess quantization quality
- **AUC Degradation** - Float → int8 degradation analysis

**Note**: Float models are NOT part of final ranking - we only deploy int8 models!

---

## 🚀 PRIORITY QUEUE (GPU Execution)

### Priority 1: Complete XiaoChirp-Tiny fft1024 ⭐
**Status**: QUEUED
**Expected time**: ~30 minutes (3 experiments × 10 min)

| Experiment | n_fft | n_mels | Status |
|------------|-------|--------|--------|
| 13_tiny.py | 1024  | 32     | ⏳ Queued |
| 13_tiny.py | 1024  | 48     | ⏳ Queued |
| 13_tiny.py | 1024  | 64     | ⏳ Queued |

**Rationale**: Complete the fft1024 sweep to compare against fft512 (already complete)

### Priority 2: Transformer-Time with Multiple Seeds ⭐
**Status**: QUEUED
**Expected time**: ~45 minutes (3 experiments × 15 min)

| Experiment | Seed | Status |
|------------|------|--------|
| 1c_mybad_transformer.py | 42  | ⏳ Queued |
| 1c_mybad_transformer.py | 100 | ⏳ Queued |
| 1c_mybad_transformer.py | 786 | ⏳ Queued |

**Rationale**: Assess transformer stability across different random initializations

---

## 📊 EXPECTED RESULTS FORMAT

For each experiment, we report:

```
Model: [name]
Seed: [value]

TFLite int8 Performance:
  Accuracy:     [value]
  F1-score:     [value] (macro avg)
  AUC:          [value]
  Model Size:   [#params] parameters
  Inference:    [time] ms

Quantization Quality:
  Float32 AUC:  [value]
  AUC Loss:     [degradation] ([percent]%)
```

---

## 🎯 NEXT PRIORITIES (After Priority 1-2)

### Priority 3: Core Architecture Sweep (30 experiments)
Run with GPU, selected n_mels based on Priority 1-2 insights:
- 3_batchnorm.py
- 4_dense.py (needs path fix)
- 5_filters.py
- 6_high_accuracy.py
- 7_low_power.py

### Priority 4: Regularization Study (24 experiments)
Dropout variants with selected n_mels:
- 8a-8d_dropout*.py (4 scripts)
- 9a-9d_depthwise_drop*.py (4 scripts)

### Priority 5: Advanced Variants (8 experiments)
- 10_depthwise_f6.py
- 11_depthwise_bn_f6.py
- 12_accurate_se.py
- 12_depthwise_f5.py

---

## 🔧 EXECUTION COMMANDS

### Start Priority Queue:
```bash
chmod +x run_priority_experiments.sh
./run_priority_experiments.sh
```

### Monitor Progress:
```bash
# Watch logs in real-time
tail -f results/13_tiny_fft1024_m32_s42/training.log

# Check running processes
ps aux | grep python | grep -E "(13_tiny|1c_mybad)"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Manual Run (if needed):
```bash
# XiaoChirp-Tiny
conda run -n tf215_gpu python 13_tiny.py --n_mels 32 --n_fft 1024

# Transformer
conda run -n tf215_gpu python 1c_mybad_transformer.py --random_seed 42
```

---

## 📈 ANALYSIS AFTER PRIORITY 1-2

After completing Priority 1-2, we will:
1. Compare XiaoChirp-Tiny fft512 vs fft1024
2. Assess transformer performance and seed stability
3. Update rankings with new int8 accuracy + F1-score focus
4. Determine optimal n_mels for remaining experiments
5. Generate comprehensive comparison report

---

**Total Priority Queue Time**: ~75 minutes (1.25 hours)
**Start Time**: [To be filled]
**Expected Completion**: [To be filled]
