# Baseline Sweep V3 - Running

**Started:** 2026-01-09 12:04:43
**Dataset:** V3 (20k samples, improved long-tail flattening)
**Status:** 🔄 IN PROGRESS

---

## Configuration

- **Experiments:** 10 (5 n_mels × 2 n_fft)
- **n_mels values:** 16, 32, 48, 64, 80
- **n_fft values:** 512, 1024
- **Seed:** 42
- **Expected runtime:** 12-20 hours

---

## What Changed

### Dataset V3 vs V2:
- **V2 (old):** 40k samples, undersampled from 39k
- **V3 (new):** 20k samples, improved long-tail flattening algorithm
- **Goal:** Better species balance, improved generalization

### Cache Status:
- ✅ Old caches deleted (1.2G + 1.9G freed)
- 🔄 New caches being created during first run
- ✅ Subsequent experiments will use `--use_cache` for speed

### Results Backed Up:
- **Backup location:** `results_backup_v2_40k_20260109_111408/`
- **Experiments backed up:** 20 baseline results from V2 dataset

---

## Monitoring Progress

### Check latest log:
```bash
tail -f baseline_sweep_logs/sweep_v3_*.log
```

### Check current experiment:
```bash
ls -lt baseline_sweep_logs/*.log | head -2
```

### Check completed results:
```bash
ls -d results/1_baseline_fft*_s42 | wc -l
echo "Expected: 10"
```

### Quick accuracy check:
```bash
for dir in results/1_baseline_fft*_s42; do
  if [ -f "$dir/results_summary.txt" ]; then
    config=$(basename $dir | cut -d'_' -f3-5)
    acc=$(grep -oP 'Accuracy:\s*\K[\d.]+' "$dir/results_summary.txt" | head -1)
    echo "$config: $acc"
  fi
done | sort
```

---

## Expected Timeline

| Experiment | n_fft | n_mels | Est. Time | Status |
|------------|-------|--------|-----------|--------|
| 1 | 512 | 16 | ~1.5 hrs | 🔄 Running (preprocessing + training) |
| 2 | 512 | 32 | ~1.2 hrs | ⏳ Queued (will use cache) |
| 3 | 512 | 48 | ~1.2 hrs | ⏳ Queued |
| 4 | 512 | 64 | ~1.5 hrs | ⏳ Queued |
| 5 | 512 | 80 | ~1.8 hrs | ⏳ Queued |
| 6 | 1024 | 16 | ~1.2 hrs | ⏳ Queued |
| 7 | 1024 | 32 | ~1.2 hrs | ⏳ Queued |
| 8 | 1024 | 48 | ~1.2 hrs | ⏳ Queued |
| 9 | 1024 | 64 | ~1.5 hrs | ⏳ Queued |
| 10 | 1024 | 80 | ~1.8 hrs | ⏳ Queued |

**Total:** ~12-15 hours

---

## What to Check When Complete

### 1. Validate all completed successfully:
```bash
./run_baseline_sweep_v3.sh  # Will show summary at end
```

### 2. Compare with old dataset (V2):
```bash
# Old best (V2): fft1024_m48_s42 = 94.30%
old_acc=$(grep "Accuracy:" results_backup_v2_40k_*/1_baseline_fft1024_m48_s42/results_summary.txt | grep -oP '\d+\.\d+')
new_acc=$(grep "Accuracy:" results/1_baseline_fft1024_m48_s42/results_summary.txt | grep -oP '\d+\.\d+')

echo "Old (40k samples): $old_acc%"
echo "New (20k improved): $new_acc%"
```

### 3. Run analysis:
```bash
python3 analyze_baseline_sweep.py
```

### 4. Generate comparison report:
```bash
# Will create:
# - baseline_sweep_complete_analysis.csv
# - baseline_sweep_analysis.png
```

---

## Expected Outcomes

### Scenario A: Accuracy Improves (Likely)
- **Old best:** 94.30% (fft1024, m48)
- **Expected new:** 95-96%
- **Action:** Celebrate! Dataset improvement worked
- **Next:** Optionally rerun top 3 ablations with new dataset

### Scenario B: Accuracy Similar (Possible)
- **Old best:** 94.30%
- **Expected new:** 93.5-94.5%
- **Action:** Hyperparameters still valid
- **Next:** Use current best config for ablations

### Scenario C: Accuracy Drops (Investigate)
- **Old best:** 94.30%
- **New:** <92%
- **Action:** Check dataset quality, species distribution
- **Next:** Debug long-tail flattening algorithm

---

## Files Being Generated

### Logs:
- `baseline_sweep_logs/sweep_v3_*.log` - Master log
- `baseline_sweep_logs/1_baseline_fft*_*.log` - Per-experiment logs

### Results:
- `results/1_baseline_fft512_m16_s42/` through `results/1_baseline_fft1024_m80_s42/`

### Cache:
- `/Volumes/Evo/cache_mybad_m16/` through `/Volumes/Evo/cache_mybad_m80/`

---

## If Something Goes Wrong

### Check for errors:
```bash
grep -i "error\|failed\|exception" baseline_sweep_logs/sweep_v3_*.log
```

### Check specific experiment:
```bash
tail -100 baseline_sweep_logs/1_baseline_fft512_m16_*.log
```

### Resume if interrupted:
```bash
# Script automatically skips completed experiments
./run_baseline_sweep_v3.sh
```

---

## Next Steps After Completion

1. ✅ Analyze results: `python3 analyze_baseline_sweep.py`
2. ✅ Compare with V2 dataset results
3. ✅ Identify new optimal hyperparameters
4. ✅ Decide whether to rerun ablations 2-10
5. ✅ Document findings in paper

---

**Status:** Running in background (task ID: b50f919)
**Last updated:** 2026-01-09 12:05
