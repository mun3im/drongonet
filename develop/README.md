# develop/

Training and analysis scripts for the SEABADNet ablation chain.

Each numbered script answers one design question. The winning config carries forward to the next phase; nothing else does. All ablation runs use seed=42 on a Mac Mini. Multi-seed validation (seeds 42/100/786) is reserved for the two final candidates (`6b`, `6c`).

## Training a model

Three wrapper scripts expose the locked final configurations with minimal arguments:

```bash
# SEABADNet-Nano  (5.41 KB, no recall target)
python deploy/train_nano.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft512_m16

# SEABADNet-Micro  (6.56 KB, ≥0.98 recall @ τ=0.35)  ← primary model
python deploy/train_micro.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m16

# SEABADNet-Edge  (33 KB, ≥0.99 recall @ τ=0.50)
python deploy/train_edge.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m80
```

All hyperparameters are locked. The only optional argument is `--random_seed` (default: 42).

Pre-trained INT8 TFLite models (seed 42) are available in `deploy/` — no training required
if you just want to run inference.

## Running an ablation script directly

Ablation scripts accept the full parameter set:

```bash
python develop/6b_micro_final.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m16 \
    --n_mels       16 \
    --n_fft        1024 \
    --random_seed  42
```

Output lands in `results/{script_name}_fft{n_fft}_m{n_mels}_s{seed}/`:

```
config.txt              hyperparameters + git hash
results_summary.txt     AUC, accuracy, size KB, latency ms  (key=value)
threshold_sweep.txt     tau | recall | precision | f1 | fpr | tp | fp | fn | tn
threshold_locked.txt    locked τ + metrics
confusion_matrix.png
roc_curve.png
pr_curve.png
classification_report.txt
model.tflite            INT8 quantised
```

## Ablation scripts

### Phase 0 — TinyChirp baselines

Run TinyChirp architectures as published on SEABAD to establish the starting point.

| Script | Model |
|---|---|
| `0a_tinychirp_cnnmel.py` | CNN-Mel — primary baseline |
| `0b_tinychirp_cnntime.py` | CNN-Time |
| `0c_tinychirp_transformer.py` | Transformer |
| `0d_tinychirp_squeezenettime.py` | SqueezeNet-Time |
| `0e_tinychirp_squeezenetmel.py` | SqueezeNet-Mel |

### Phase 1 — Frequency resolution (n_mels sweep)

**Question:** What mel resolution does SEABAD need?

| Script | Purpose |
|---|---|
| `1a_baseline2d.py` | CNN-Mel on SEABAD, n_mels ∈ {16, 32, 48, 64, 80} |
| `1b_transformer.py` | Transformer on raw waveform (reference) |
| `1c_cnntime.py` | CNN-Time on SEABAD (reference) |
| `1d_cnntime_gap.py` | CNN-Time + GAP |
| `1e_cnntime_enhanced.py` | CNN-Time with extended metrics |
| `1f_cnntime_deeper.py` | CNN-Time with extra conv layer |

### Phase 2 — Pooling: Flatten vs GAP

**Question:** Does Global Average Pooling reduce parameters without sacrificing AUC?

| Script | Key change |
|---|---|
| `2a_baseline_gap.py` | Replace Flatten with GAP |
| `2b_baseline_gap_learned.py` | GAP + learned pooling weights |
| `2c_baseline_gap_1x1.py` | GAP + 1×1 bottleneck |

### Phase 3 — Conv type, filter count, loss function

**Question:** Standard Conv2D vs depthwise? How many filters? Does focal loss help?

| Script | Key change |
|---|---|
| `3a_depthwise.py` | SeparableConv2D, 4 filters |
| `3b_filters8.py` | Conv2D, 8 filters |
| `3c_gap_focal_loss.py` | GAP + focal loss (α=0.25, γ=2) |
| `3d_gap_freq_emphasis.py` | + frequency emphasis augmentation |
| `3e_gap_freq_emph_ds.py` | Frequency emphasis + depthwise separable |
| `3f_gap_focal_loss_freq_emph_pointwise.py` | GAP + focal loss + freq emphasis + pointwise conv |
| `3g_strided_focal_tuned.py` | Strided conv + focal loss |
| `3h_strided_focal_no1x1.py` | Strided, no 1×1 conv |
| `3i_strided_focal_depthwise.py` | Strided + depthwise separable |

### Phase 4 — Dropout sweep

**Question:** What dropout rate is optimal per architecture?

Track A — Edge (Conv2D, 8 filters): `4a_dropout01.py` – `4d_dropout04.py` (dropout 0.1–0.4)

Track B — Micro (depthwise sep, n_mels=16): `4e_depthwise_drop01.py` – `4h_depthwise_drop04.py` (dropout 0.1–0.4)

### Phase 5 — Micro filter count

**Question:** For Micro, is 4 filters the right count?

| Script | Filters | n_fft |
|---|---|---|
| `5a_depthwise_f5.py` | 5 | 512 |
| `5b_depthwise_f6.py` | 6 | 1024 |

Note: `5a` uses `n_fft=512`; all other scripts use `n_fft=1024`. These produce separate cache directories.

### Phase 6 — Final candidates

| Script | Model | Params | Size (INT8) | Target recall |
|---|---|---|---|---|
| `6a_nano_final.py` | SEABADNet-Nano | 763 | 5.41 KB | — |
| `6b_micro_final.py` | **SEABADNet-Micro** | 919 | 6.56 KB | ≥0.98 |
| `6c_edge_final.py` | **SEABADNet-Edge** | 25,890 | 33.06 KB | ≥0.99 |

## Analysis and threshold scripts

| Script | Purpose |
|---|---|
| `compile_paper_tables.py` | Aggregate per-seed sweep data into LaTeX-ready rows |
| `threshold_sweep_micro.py` | Sweep τ for SEABADNet-Micro, lock threshold |
| `threshold_sweep_edge.py` | Sweep τ for SEABADNet-Edge, lock threshold |
| `threshold_sweep_nano.py` | Sweep τ for SEABADNet-Nano |
| `threshold_sweep_edge_control.py` | Edge threshold sweep variant |
| `generate_int8_models.py` | Batch INT8 quantisation |
| `generate_figures_from_sweep.py` | Figures from sweep outputs |
| `generate_fig6_part1_extract.py` | Figure 6 data extraction (Linux: GPU + mel caches) |
| `generate_fig6_part2_plot.py` | Figure 6 rendering (Mac: Arial fonts, no caches needed) |
| `extract_paper_numbers.py` | Extract key numbers from results dirs |
| `ablation_freq_emphasis.py` | Frequency emphasis ablation |
| `_patch_ablation.py` | One-time patch: inject config imports into scripts |
| `config.py` | Shared path and hyperparameter constants |
