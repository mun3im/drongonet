# MyBAD Complete Pipeline Guide

## Quick Start

After cleaning up your dataset, simply run:

```bash
./run_full_pipeline.sh
```

This single command will:
1. ✅ Delete all old cache directories
2. ✅ Backup existing results with timestamp
3. ✅ Run all 30 experiments with fresh cache generation
4. ✅ Collect all results into CSV
5. ✅ Analyze resource usage
6. ✅ Generate all 8 paper tables
7. ✅ Generate all 6 paper figures

## What Gets Created

### Cache Directories (can be deleted after)
- `/Volumes/Evo/cache_mybad_m16` (~500MB)
- `/Volumes/Evo/cache_mybad_m32` (~1GB)
- `/Volumes/Evo/cache_mybad_m48` (~1.5GB)
- `/Volumes/Evo/cache_mybad_m64` (~2GB)
- `/Volumes/Evo/cache_mybad_m80` (~2.5GB)

### Results
- `results/` - 30 experiment directories with all outputs
- `all_results_comparison.csv` - Complete results table
- `resource_usage_analysis.csv` - Deployment analysis

### Paper Outputs
- `paper_tables_output.txt` - All 8 tables formatted for paper
- `figures/` - 6 figures (PNG + PDF):
  - figure1_pareto_frontier
  - figure2_nmels_impact
  - figure3_dropout_comparison
  - figure4_architecture_comparison
  - figure5_quantization_robustness
  - figure6_efficiency_metrics

### Backups
- `results_backup_YYYYMMDD_HHMMSS/` - Old results (timestamped)
- `rerun_logs_backup_YYYYMMDD_HHMMSS/` - Old logs (timestamped)

## Expected Runtime

- **Cache generation** (first time): ~30-40 minutes total
  - First experiment per n_mels creates cache (~5-10 min each)
  - Subsequent experiments reuse cache
- **All 30 experiments**: ~5-7 hours total
- **Analysis & figures**: ~2-3 minutes

**Total: 6-8 hours** (run overnight)

## Monitoring Progress

### Watch live output
```bash
tail -f rerun_logs/1_baseline_m48_s42.log
```

### Check GPU usage
```bash
watch -n 5 nvidia-smi
```

### Count completed experiments
```bash
ls -1d results/*/ | wc -l
```

### Check which experiment is running
```bash
ps aux | grep python | grep -E "baseline|depthwise|dropout"
```

## If Something Fails

### View failed experiment log
```bash
tail -100 rerun_logs/FAILED_EXPERIMENT_NAME.log
```

### Re-run single experiment manually
```bash
python 1_baseline.py \
  --dataset-path /Volumes/Evo/mybad \
  --n_mels 48 \
  --random_seed 42 \
  --use_cache
```

### Continue from where it failed
The script will continue through all experiments even if some fail. Check the final summary for which ones failed, then re-run those individually.

## Cleaning Up

### Delete cache to save space (after pipeline completes)
```bash
rm -rf /Volumes/Evo/cache_mybad_m*
```

### Delete old backups
```bash
rm -rf results_backup_*
rm -rf rerun_logs_backup_*
```

### Keep only final results
```bash
# Keep these:
# - results/
# - all_results_comparison.csv
# - resource_usage_analysis.csv
# - paper_tables_output.txt
# - figures/

# Can delete:
# - rerun_logs/
# - All backups
# - All cache directories
```

## Configuration

Edit `run_full_pipeline.sh` to change:

```bash
DATASET_PATH="/Volumes/Evo/mybad"  # Your dataset location
SEED=42                             # Random seed
CACHE_BASE="/Volumes/Evo"          # Where to store cache
```

## Important Notes

1. **Dataset must be ready** before running - the script expects:
   - `/Volumes/Evo/mybad/positive/` (20,000 WAV files)
   - `/Volumes/Evo/mybad/negative/` (20,000 WAV files)

2. **GPU recommended** - CPU training will take 10x longer

3. **Disk space needed**:
   - Cache: ~8GB
   - Results: ~2GB
   - Total: ~10GB free space required

4. **The script uses `set -e`** - it will stop if critical steps fail (like result collection or table generation), but will continue through individual experiment failures.

## Verification Checklist

After pipeline completes, verify:

- [ ] 30 directories in `results/`
- [ ] `all_results_comparison.csv` has 30 rows
- [ ] All figures exist in `figures/` (12 files: 6 PNG + 6 PDF)
- [ ] `paper_tables_output.txt` contains 8 tables
- [ ] Check config.txt in any result dir shows: `Hop Length: 256`
- [ ] Top model achieves >98% AUC (check all_results_comparison.csv)

## Quick Commands

```bash
# Run full pipeline
./run_full_pipeline.sh

# Check top 10 models by AUC
head -15 all_results_comparison.csv | column -t -s,

# View specific table
grep -A 20 "TABLE 2" paper_tables_output.txt

# List all figures
ls -lh figures/

# Check hop_length in results
grep "Hop Length" results/1_baseline_m48_s42/config.txt
```
