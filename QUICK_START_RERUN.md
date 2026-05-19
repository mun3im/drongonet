# Quick Start: Ablation Studies Rerun

## 🚀 One-Command Execution

```bash
./rerun_ablations_2_10.sh
```

That's it! The script handles everything.

---

## 📊 What Gets Run

15 experiments with **n_fft=1024, n_mels=48**:

```
2_depthwise          → Depthwise separable convolutions
3_batchnorm          → BatchNormalization added
4_dense              → Larger dense layer (16 units)
5_filters            → More filters (6 vs 4)
6_best_accuracy      → Combined: depthwise + BN + dropout
7_hybrid             → Alternative combination
8a-8d_dropout        → Dropout rates: 0.1, 0.2, 0.3, 0.4
9a-9d_depthwise_drop → Depthwise + dropout combinations
10_depthwise_f6      → 6 filters with depthwise
```

**Baseline to beat**: 94.30% accuracy (1_baseline_fft1024_m48_s42)

---

## ⏱️ Timeline

- **Per experiment**: ~1.5-2.5 hours
- **Total runtime**: ~25-40 hours
- **Runs**: Sequential (one at a time)

---

## 👀 Monitor Progress

### Real-time log
```bash
tail -f rerun_logs/rerun_*.log
```

### Check results
```bash
watch -n 60 'ls -ltr results/ | tail -20'
```

### Latest accuracy
```bash
find results/ -name "results_summary.txt" -mmin -60 -exec grep -H "Accuracy" {} \;
```

---

## ✅ Success Indicators

Script will report:
- ✓ Validation passed (accuracy > 85%)
- Current accuracy vs baseline
- Progress: X successful, Y failed

---

## 🛑 What if Something Fails?

The script automatically:
1. **Backs up** failed results
2. **Logs** detailed errors
3. **Continues** to next experiment
4. **Reports** summary at end

Check `rerun_logs/` for details.

---

## 🔧 Advanced Options

### Test one experiment first
```bash
python3 2_depthwise.py --n_mels 48 --n_fft 1024 --use_cache
```

### Run in background (recommended)
```bash
screen -S rerun
./rerun_ablations_2_10.sh
# Press Ctrl+A, then D to detach

# Reattach later:
screen -r rerun
```

### Skip existing results
The script automatically skips valid existing results.

To force rerun all:
```bash
# Backup old results first!
mv results results_backup_$(date +%Y%m%d)
./rerun_ablations_2_10.sh
```

---

## 📈 After Completion

1. **Check summary**
   ```bash
   tail -100 rerun_logs/rerun_*.log
   ```

2. **Analyze results**
   ```bash
   python3 analyze_baseline_sweep.py
   ```

3. **Generate paper figures**
   ```bash
   python3 generate_pareto_analysis.py
   ```

---

## 🆘 Troubleshooting

### "Cache not found"
Create cache or run without --use_cache (slower):
```bash
python3 2_depthwise.py --n_mels 48 --n_fft 1024 --force-reprocess
```

### "CUDA out of memory"
Scripts handle this automatically with:
```python
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

### "Accuracy collapse"
Script validates and reports. Check:
- Model architecture (debug print statements)
- Input shape (should be 184×48×1)
- Training logs

---

## 📝 Files Generated

```
results/
  2_depthwise_fft1024_m48_s42/
  3_batchnorm_fft1024_m48_s42/
  ...
  10_depthwise_f6_fft1024_m48_s42/

rerun_logs/
  rerun_20260108_*.log
  2_depthwise_20260108_*.log
  ...
```

---

## 💾 Disk Space

~3 MB total (15 experiments × ~200 KB each)

---

## ⏸️ Pause/Resume

To stop:
```bash
Ctrl+C  (will finish current experiment)
```

To resume:
```bash
./rerun_ablations_2_10.sh
(skips completed experiments automatically)
```

---

Ready? Start here:

```bash
./rerun_ablations_2_10.sh
```

Good luck! 🍀
