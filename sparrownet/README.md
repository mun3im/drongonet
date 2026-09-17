# SparrowNet

A bird-call detector CNN for Cortex-M4, trained on the DCASE 2018 Bird Audio Detection
corpora. Combines **drongonet-micro's size/quantization budget** with the
**local-receptive-field + global-max-pool topology** of the `sparrow` submission in
Grill & Schlüter 2017 (`~/Dropbox/References/grill2017two.pdf`, Table II).

## Size budget

| | Limit |
|---|---|
| Soft target | **10 KB** INT8 `.tflite` |
| Hard ceiling | **16 KB** INT8 `.tflite` |

Anything over the hard ceiling is rejected regardless of AUC. `deploy/check_size.py`
enforces this; call it in every training script after conversion.

## Why this architecture

- **From the paper (`sparrow`, Table II):** fully-convolutional with a *short* receptive
  field (103 frames / 1.5s) applied across the clip, then **global max** over local
  predictions. Bird calls are brief local events, and the paper's Fig. 4 shows global-max
  substantially beats global-mean. The paper's 309,843 params are irrelevant to us — the
  *topology* is what transfers, not the widths.
- **From drongonet-micro:** the M4-fittable budget — INT8, FrequencyEmphasis gate,
  depthwise-separable convs, focal loss, n_fft=1024 / n_mels=16.
- **From drongonet's `LESSONS_LEARNT.md` (390+ ablations):** early convs stay full-width;
  strided conv beats MaxPool; GAP is free; focal loss (γ=2, α≥0.5) prevents recall
  collapse; n_fft=1024 is non-negotiable (512 caused a 37% AUC collapse); PTQ without QAT
  is the top recall killer.

The key departure from drongonet-micro: replace global-average-pool-over-the-whole-clip
with **per-frame local logits then global max**, which should help precisely where
drongonet was weakest — short, faint calls inside a mostly-silent clip.

```
Input (T, 16, 1)   T variable: 622 frames for a 10s clip, 184 for a 3s crop
  -> FrequencyEmphasis         learned per-mel-bin gate (~free, recall-positive)
  -> Conv3x3 stride=2, 8ch     full-width early layer
  -> DepthwiseConv3x3 + Pointwise1x1, 12-16ch
  -> Conv1x1 -> per-frame local logit map
  -> GlobalMaxPool over time   sparrow's finding: max >> mean for sparse events
  -> Dense(2) softmax, focal loss (gamma=2, alpha=0.5)
```

Because the net is fully convolutional up to the pooling stage, it consumes whole 10s
clips without cropping — matching the weak-label ("at least one event") structure of the
DCASE annotations, where a fixed 3s crop injects label noise.

## Layout

```
config.py            paths, frontend params, size limits
data.py              DCASE loader: corpora, cross-corpus splits, mel frontend
develop/             architecture + training scripts
benchmark/           baselines and comparison runs (drongonet-micro, bulbul, sparrow)
deploy/              INT8 conversion, size checks, Cortex-M4 artifacts
results/             per-run output dirs
cache/               precomputed mel cache
```

## Data

`/Volumes/Evo/datasets` — three corpora, all 10s mono wavs at 44.1kHz, one clip-level
weak label each (`itemid,datasetid,hasbird`):

| Corpus | Clips | Positive |
|---|---:|---:|
| birdvox (BirdVox-DCASE-20k) | 20,000 | 50.1% |
| freefield1010 | 7,690 | 25.2% |
| warblr | 8,000 | 75.6% |

The positive rates differ sharply across corpora — this is the domain shift the paper
discusses, and it means cross-corpus arms need explicit class handling rather than
assuming the training balance carries over.

Frontend (identical to drongonet-micro, so numbers stay comparable): resample to 16kHz,
`n_fft=1024`, `hop=256`, `n_mels=16`, `fmin=100`, `fmax=8000`, `center=False`, dB with
`ref=np.max`, then per-clip min-max to [0,1]. 10s -> 622 frames; 3s crop -> 184 frames.

```bash
python3 data.py    # sanity check: counts, class balance, shapes
```

## Plan / status

- [x] Repo scaffold, config, DCASE data loader (verified against all three corpora)
- [ ] Mel cache builder (write to `cache/`, **never** to `/tmp` or root — see drongonet's
      `PICKUP.md`: a full root partition caused repeated crashes there)
- [ ] Baseline: drongonet-micro **unmodified** on DCASE — isolates "dataset vs architecture"
      before changing anything
- [ ] SparrowNet float32 (`develop/`), focal loss + cyclic time-shift augmentation
      (the paper's most important augmentation for both of its models)
- [ ] Ablate local receptive field: 103 frames (paper's 1.5s) vs 184 (drongonet's 3s)
- [ ] INT8 **QAT** (not PTQ), verify recall floor post-quantization, re-sweep tau on the
      INT8 model
- [ ] Cross-corpus eval: in-domain CV AUC *and* held-out-corpus AUC for all three folds
- [ ] Benchmark table vs drongonet-micro / drongonet-edge / bulbul / sparrow at matched
      protocol (drongonet has a faithful `bulbul_arch.py` to pattern a `sparrow_arch.py` on)
- [ ] Cortex-M4 (AudioMoth) size + latency bench
