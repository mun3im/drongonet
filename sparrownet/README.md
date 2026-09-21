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

## Running things

```bash
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python   # TF 2.15.0 + tfmot 0.8.0, GPU

$PY build_cache.py                      # 16-mel cache (1.35 GB, ~8 min)
$PY build_cache.py --n-mels 80          # 80-mel cache for the paper baselines (7.1 GB)
$PY train.py --arch sparrownet --held-out birdvox --seed 42
$PY aggregate.py --group                # results table, grouped across seeds
$PY deploy/footprint.py results/*/*.tflite
$PY deploy/export_c_array.py results/<tag>/sparrownet_int8.tflite -o deploy/sparrownet-micro.h
```

Every module self-checks in its `__main__`: `data.py` (corpus counts/shapes),
`dataset.py` (TF and numpy normalizers vs librosa), `models.py` (param counts and
receptive fields), `benchmark_archs.py` (params vs published), `qat.py` (QAT clone).

## Plan / status

- [x] Repo scaffold, config, DCASE data loader (verified against all three corpora)
- [x] Mel cache builder — 35,690 clips, 0 failures. Caches **raw power mel** so one
      cache serves both the 3s-crop and whole-clip paths; a cache-derived crop matches
      direct computation to 1.3e-07. Lives in `~/.cache/sparrownet`, off Dropbox and
      off the root partition (drongonet's `PICKUP.md`: an 11 GB cache on `/tmp` filled
      root and crashed runs)
- [x] Baseline: drongonet-micro **unmodified** on DCASE — port reproduces its 919
      params and its documented INT8 quantization scales exactly
- [x] SparrowNet float32, focal loss + cyclic time-shift augmentation
- [x] INT8 quantization + threshold sweep on the INT8 model; QAT implemented
      (`qat.py`, selective annotation around the custom FrequencyEmphasis layer)
- [~] Ablate local receptive field — in progress (phase C); phase A's answer was
      invalid past stages=5, see **Batch norm** below
- [ ] Cross-corpus eval: all three held-out folds x 3 seeds
- [ ] Benchmark table vs drongonet-micro / drongonet-edge / bulbul / sparrow
- [x] Cortex-M4 footprint tooling (`deploy/footprint.py`, `deploy/export_c_array.py`)
- [ ] Latency **measured on hardware** — everything so far is an analytic estimate

## Findings so far

**Batch norm was worth ~0.13 cross-corpus AUC, and its absence caused a wrong
conclusion.** The honest sequence, because the mistake is instructive:

1. Held out birdvox (trained on freefield1010+warblr), no BN: drongonet-micro scored
   in-domain val 0.795 / cross-corpus 0.491; SparrowNet 0.887 / 0.502. Both at chance
   cross-corpus despite a 9-point in-domain gap.
2. Those scores were *not* degenerate — full range, precision pinned at 0.501, exactly
   birdvox's base rate — and they matched drongonet's own measured micro cross-corpus
   AUC (0.459 +- 0.034). That agreement made "irreducible domain shift" look
   well-evidenced, and it was recorded here as such.
3. It was wrong. Adding BN took the same SparrowNet config to **val 0.901 /
   cross-corpus 0.629**. A third of the apparent "domain shift" was our own missing
   normalization — something the paper explicitly specifies for this architecture
   ("In sparrow, we also apply batch normalization to all layers").

The lesson: reproducing someone else's number is not evidence that the number is a
floor. Both projects shared the same omission, so agreement confirmed nothing.

BN is also what keeps deep stacks alive at all — without it every stages>=6 SparrowNet
collapsed to a constant output (AUC exactly 0.5000), which silently invalidated phase
A's receptive-field sweep past stages=5. TFLite folds BN into the preceding conv, so
it is close to free at inference.

A cross-corpus gap remains (0.90 in-domain vs 0.63 held-out), so domain shift is real —
but its size is now an open question rather than a settled one.

**Depthwise strides must be equal in both dimensions.** `(2,1)` is rejected outright on
CPU and is a portability risk on TFLite Micro. Once frequency collapses to 1, stride 2
with `same` padding is a no-op on that axis, so `(2,2)` throughout costs nothing.

**Size is dominated by per-layer overhead, not weights.** SparrowNet at 1,219 params
converts to 10.91 KB while 919-param drongonet-micro converts to 6.00 KB: the
difference is FlatBuffer per-layer cost (quantization metadata, op descriptors). The
lever for the 10 KB target is layer count and width, not parameter count. Adding BN
pushed st4 to 12.41 KB — BN itself folds away, but conv->BN->ReLU stops the ReLU
fusing into the conv, so each block costs an extra op. Still inside the 16 KB hard
ceiling; whether anything clears 10 KB is what the width sweep is for.

**A latency figure in drongonet's docs looks wrong.** `LESSONS_LEARNT.md` claims
0.1-0.3 ms per 3s clip on a 48 MHz Cortex-M4. drongonet-micro is 741,912 MACs, which
at 48 MHz cannot complete faster than ~15 ms even at 1 MAC/cycle; 0.2 ms would need
~200 MACs/cycle. The claim appears ~100x optimistic and is worth rechecking before it
informs any power budget.
