# SEABADNet

Lean TinyML CNN for binary bird activity detection, targeting embedded hardware (ARM Cortex-M4 and single-board computers). Derived from the [TinyChirp](https://arxiv.org/abs/2408.01976) CNN-Mel architecture and trained on the **SEABAD** (South-East Asian Bird Activity Detection) dataset.

## Models

| Variant | Target hardware | Size | **Target recall** | τ | AUC |
|---|---|---|---|---|---|
| **SEABADNet-Micro** | ARM Cortex-M4 (AudioMoth, STM32F4) | ≤8 KB INT8 | **≥0.98** | swept | 0.9741 |
| **SEABADNet-Edge** | SBC (Raspberry Pi, Portenta X8) | ≤35 KB INT8 | **≥0.99** | swept | 0.9994 |

Recall is the primary deployment metric — missed bird calls are false negatives, and the application tolerates some false positives to minimise misses. AUC is reported for comparison. Micro (`6b_micro_improved`) uses SeparableConv2D + pointwise conv (6 filters, n_mels=16, n_fft=1024, 919 params, 6.56 KB INT8). Edge (`6c_edge_final`) uses standard Conv2D 3-block 16→32→64 filters (n_mels=80, n_fft=1024, ~25k params, 32.82 KB INT8). Both use GAP and focal loss.

## Dataset

**SEABAD** — binary classification (bird activity present / absent), 16 kHz, 3-second clips, 80/10/10 split.

Mel caches are keyed by `(n_mels, n_fft, hop_length)` and stored separately from the raw audio:

```
Mac Mini:  /Volumes/Evo/cache_seabad_m{n_mels}/
Linux GPU: /data/cache_seabad_m{n_mels}/
```

## Quickstart

All training scripts share the same CLI:

```bash
python 6b_micro_improved.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_seabad_m16 \
    --n_mels       16 \
    --n_fft        1024 \
    --random_seed  42
```

Results land in `results/{script_name}_fft{n_fft}_m{n_mels}_s{seed}/` and include float32 + INT8 TFLite evaluation, confusion matrix, ROC/PR curves, and a parseable `results_summary.txt`.

## Repository structure

### Phase 0 — TinyChirp baselines (run as-published on SEABAD)

| Script | Model |
|---|---|
| `0a_tinychirp_cnnmel.py` | CNN-Mel — primary baseline |
| `0b_tinychirp_cnntime.py` | CNN-Time |
| `0c_tinychirp_transformer.py` | Transformer |
| `0d_tinychirp_squeezenettime.py` | SqueezeNet-Time |
| `0e_tinychirp_squeezenetmel.py` | SqueezeNet-Mel |

### Phase 1 — n_mels sweep (frequency resolution)

| Script | Purpose |
|---|---|
| `1a_baseline2d.py` | CNN-Mel on SEABAD, sweep n_mels ∈ {16,32,48,64,80} |
| `1b_transformer.py` | Transformer on raw waveform (reference) |
| `1c_cnntime.py` | CNN-Time on SEABAD (reference) |
| `1d_cnntime_gap.py` | CNN-Time + GAP |
| `1e_cnntime_enhanced.py` | CNN-Time with extended metrics |
| `1f_cnntime_deeper.py` | CNN-Time with extra conv layer |

### Phase 2 — Pooling: Flatten vs GAP

| Script | Key change |
|---|---|
| `2a_baseline_gap.py` | Replace Flatten with Global Average Pooling |
| `2b_baseline_gap_learned.py` | GAP + learned pooling weights |
| `2c_baseline_gap_1x1.py` | GAP + 1×1 bottleneck conv |

### Phase 3 — Conv type, filter count, loss function

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

**Track A — Edge (Conv2D, 8 filters):** `4a_dropout01.py` through `4d_dropout04.py` (dropout 0.1–0.4)

**Track B — Micro (depthwise sep branch, n_mels=16):** `4e_depthwise_drop01.py` through `4h_depthwise_drop04.py` (dropout 0.1–0.4) — this branch was abandoned; depthwise sep proved counter-productive at this scale (see Phase 6)

### Phase 5 — Micro filter count

| Script | Filters |
|---|---|
| `5a_depthwise_f6.py` | 6 |
| `5b_depthwise_f5.py` | 5 |

### Phase 6 — Final candidates

| Script | Model | Conv type | Filters | n_mels | Size (INT8) | Target recall |
|---|---|---|---|---|---|---|
| `6a_micro_final.py` | SEABADNet-Micro (v1, superseded) | SeparableConv2D | 6 | 16 | 5.41 KB | — |
| `6b_micro_improved.py` | **SEABADNet-Micro (final)** | SeparableConv2D + pointwise | 6 | 16 | 6.56 KB | ≥0.98 |
| `6c_edge_final.py` | **SEABADNet-Edge (final)** | Conv2D 3-block 16→32→64 | 64 | 80 | 32.82 KB | ≥0.99 |


## Requirements

- Python 3.10+
- TensorFlow 2.15 (GPU recommended for final validation runs)
- librosa, numpy, scikit-learn, matplotlib

## Reproducibility

Every training script writes a `config.txt` (all hyperparameters + git hash) and a parseable `results_summary.txt` to its output directory. Mel caches are generated once per `(n_mels, n_fft, hop_length)` triple and reused across all runs that share that configuration.

## Citation

> M. Zabidi, "SEABADNet: Lightweight CNNs for Tropical Bird Audio Detection on Edge Devices," *manuscript in preparation*, 2026.

Based on: Huang et al., "TinyChirp: Bird Song Recognition Using TinyML," 2024.
