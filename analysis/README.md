# Post-Ablation Analysis & Results Compilation

This directory contains scripts for post-training analysis, threshold optimization, and figure generation.

## Workflow

After completing all ablation runs in `develop/`, use these scripts in order:

1. **Threshold Sweeps** — Find optimal detection thresholds for each model variant
2. **Table Compilation** — Aggregate results into LaTeX-ready rows
3. **Figure Generation** — Create publication-quality plots and visualizations

## Scripts

### Threshold Sweeps

Find the optimal decision threshold τ that maximizes precision while meeting the recall target.

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `threshold_sweep_micro.py` | Micro model threshold sweep (τ ∈ {0.25–0.50}) | `results4arxiv/6b_micro_final_fft1024_m16_s{42,100,786}/` | `results/micro_threshold_sweep/threshold_locked.txt` |
| `threshold_sweep_edge.py` | Edge model per-seed threshold sweep | `results4arxiv/6c_edge_final_fft1024_m80_s{42,100,786}/` | `results/6c_edge_final_fft1024_m80_s*/threshold_locked.txt` |
| `threshold_sweep_nano.py` | Nano model threshold sweep (AUC-only, no recall target) | `results4arxiv/6a_nano_final_fft1024_m16_s{42,100,786}/` | `results/nano_threshold_sweep/threshold_locked.txt` |

**Key Design:**
- Sweeps select the **highest** τ that meets the recall target (maximizes precision)
- Reports recall, precision, F1, FPR, TP/FP/FN/TN at each threshold
- Outputs `threshold_locked.txt` with operating point

### Results Compilation

Convert raw threshold sweep results into publication-ready tables.

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `compile_paper_tables.py` | Aggregate per-seed threshold data into LaTeX rows | `results/*/threshold_sweep.txt` | stdout (copy to paper) |
| `extract_paper_numbers.py` | Extract key metrics for paper text | `results4arxiv/` | Key numbers (AUC mean±std, size KB, latency) |

**Usage:**
```bash
# Generate LaTeX-ready ablation table rows
python analysis/compile_paper_tables.py

# Extract numeric values for paper text
python analysis/extract_paper_numbers.py
```

### Figure Generation

Create plots from training results and threshold sweeps.

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `generate_figures_from_sweep.py` | Plot recall/precision/F1 vs threshold τ | `results/*/threshold_sweep.txt` | `figures_publication/` (PR curves, threshold plots) |
| `generate_fig6_part1_extract.py` | Fig 6 data extraction (LINUX: has GPU + `/Volumes/Evo` caches, DejaVu fonts only) | TFLite models + mel caches | `analysis/fig6_data.npz` |
| `generate_fig6_part2_plot.py` | Fig 6 rendering (MAC: has real Arial matching Fig 1-5, no caches needed) | `fig6_data.npz` (copied from Linux) | `analysis/images/fig6_probability_distributions.pdf` |
| `generate_int8_models.py` | Regenerate all quantised INT8 models | Training results | `results_int8/` (full INT8 models for all seeds) |

Fig 6 is split into two parts: this Linux machine has the GPU and mel caches but only
DejaVu-fallback fonts, while the Mac has real Arial but no GPU/caches. Part 1 runs here and
writes a small `.npz`; copy that file to the Mac and run part 2 there to get the Arial-accurate render.

**Usage:**
```bash
# Generate all figures after threshold sweeps complete
conda run -n tf215_gpu python analysis/generate_figures_from_sweep.py

# Fig 6, part 1 (Linux): extract probabilities to .npz
conda run -n tf215_gpu python analysis/generate_fig6_part1_extract.py \
    --micro-cache /Volumes/Evo/cache4arxiv_fft1024_m16 \
    --edge-cache  /Volumes/Evo/cache4arxiv_fft1024_m80 \
    --results-dir results4arxiv
# then copy analysis/fig6_data.npz to the Mac and run there:
#   python analysis/generate_fig6_part2_plot.py --data fig6_data.npz

conda run -n tf215_gpu python analysis/generate_int8_models.py
```

## End-to-End Example

```bash
# Step 1: Run all ablation scripts in develop/
conda run -n tf215_gpu bash develop/run_ablation_full.sh

# Step 2: Threshold sweeps (separate, after training complete)
conda run -n tf215_gpu python analysis/threshold_sweep_micro.py
conda run -n tf215_gpu python analysis/threshold_sweep_edge.py
conda run -n tf215_gpu python analysis/threshold_sweep_nano.py

# Step 3: Compile results into paper tables
python analysis/compile_paper_tables.py > paper_tables_output.txt

# Step 4: Extract numeric values for paper text
python analysis/extract_paper_numbers.py

# Step 5: Generate publication figures
conda run -n tf215_gpu python analysis/generate_figures_from_sweep.py
conda run -n tf215_gpu python analysis/generate_fig6_part1_extract.py \
    --micro-cache /Volumes/Evo/cache4arxiv_fft1024_m16 \
    --edge-cache  /Volumes/Evo/cache4arxiv_fft1024_m80 \
    --results-dir results4arxiv
# copy analysis/fig6_data.npz to the Mac, then there: python analysis/generate_fig6_part2_plot.py --data fig6_data.npz
conda run -n tf215_gpu python analysis/generate_int8_models.py
```

## Output Directories

- `results/` — Per-run threshold sweep results (copied from `results4arxiv/`)
- `results_int8/` — Full INT8 quantised models for all seeds
- `figures_publication/` — Publication-quality figures (PR curves, ROC, distributions)

## Notes

- Threshold sweeps read from `results4arxiv/` (the authoritative training output directory)
- All scripts assume `tf215_gpu` conda environment with TensorFlow 2.15 and TFLite
- Figure generation requires `matplotlib` with Arial/Helvetica fonts (see [[plots-use-arial-helvetica]] in project memory)
- `compile_paper_tables.py` uses `find_locked_tau()` to select the highest τ meeting recall target (not minimum)
