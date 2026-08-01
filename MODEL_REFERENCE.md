# Model Reference

Everything needed to consume, re-train, or re-verify a DrongoNet model without reading
`develop/*.py`. Facts below were cross-checked against the actual shipped `.tflite` binaries
(FlatBuffer-parsed, not just docs) as of this writing — see [Verification method](#verification-method).
Where the code and its own comments disagree, that's called out explicitly rather than papered over.

## The three variants at a glance

| | Nano | Micro | Edge |
|---|---|---|---|
| Training script | `develop/6a_nano_final.py` | `develop/6b_micro_final.py` | `develop/6c_edge_final.py` |
| Wrapper (use this) | `deploy/train_nano.py` | `deploy/train_micro.py` | `deploy/train_edge.py` |
| Shipped model | `deploy/drongonet_nano_int8.tflite` | `deploy/drongonet_micro_int8.tflite` | `deploy/drongonet_edge_int8.tflite` |
| n_mels | 16 | 16 | 80 |
| n_fft | 1024 | 1024 | 1024 |
| Input tensor shape | `[1, 184, 16, 1]` | `[1, 184, 16, 1]` | `[1, 184, 80, 1]` |
| Params | 763 | 919 | 25,890 |
| INT8 size | 5.09 KB (5208 bytes) | 6.23 KB (6384 bytes) | 33.06 KB (33856 bytes) |
| Loss | focal (γ=2.0, α=0.5) | focal (γ=2.0, α=0.5) | categorical cross-entropy |
| Operating τ | not published (see below) | 0.30 (≥0.987 recall) | 0.50 (≥0.99 recall) |
| Target hardware | Cortex-M4 (AudioMoth) | Cortex-M4 (AudioMoth) | SBC (Raspberry Pi, Portenta X8) |

All three take the same 16 kHz, 3-second (48,000-sample) input window and emit the same
2-class INT8 softmax output shape `[1, 2]` — only the mel resolution and network depth change.

## Preprocessing pipeline (identical across all three variants)

```
raw audio (any sample rate)
  → resample to 16000 Hz
  → pad/truncate to 48000 samples (3.0s), zero-pad if short
  → librosa.feature.melspectrogram(sr=16000, n_fft=1024, hop_length=256,
                                     n_mels={16 or 80}, fmin=100.0, fmax=8000.0,
                                     center=False)
  → librosa.power_to_db(mel_spec, ref=np.max)      # dB, relative to this clip's own peak
  → transpose to (time, freq) = (184, n_mels)
  → per-clip min-max normalize to [0, 1]:  (x - x.min()) / (x.max() - x.min())
  → add channel dim: (184, n_mels, 1)
  → quantize to INT8 using the model's own input scale/zero_point
```

Frame count derivation: `1 + floor((48000 - 1024) / 256) = 184`. This is a **consequence** of
n_fft=1024/hop=256/window=48000, not an independently chosen number — if you ever change
n_fft or hop_length, 184 changes too. (`compute_mel_spectrogram()` in the training scripts
force-pads/truncates to a hardcoded `184` regardless of what the FFT math actually produces —
see [Known gotchas](#known-gotchas) for why this matters.)

**dB normalization is per-clip, not global.** `ref=np.max` and the min-max step both operate on
each 3-second clip's own dynamic range — there is no dataset-wide normalization constant to
replicate. A loud clip and a quiet clip with the same *shape* of spectrogram will normalize to
similar [0,1] values; only the clip's own energy distribution matters, not its absolute level.

## Quantization

All models are full INT8 (`OpsSet.TFLITE_BUILTINS_INT8`, both input and output tensors INT8).
Verified directly from the FlatBuffer for all three shipped models:

| | Input scale | Input zero_point | Output scale | Output zero_point |
|---|---|---|---|---|
| Nano/Micro/Edge (all) | 0.0039215689 (=1/255) | -128 | 0.00390625 (=1/256) | -128 |

Quantize: `q = round(x / scale) + zero_point`, clamped to `[-128, 127]`. Dequantize:
`x = (q - zero_point) * scale`.

Output is `[no_bird, bird]` — index **1** is the bird-positive probability (confirmed both from
`SEABADDataset.__init__`'s `enumerate(['negative', 'positive'])` label assignment and from
`_get_predictions`/`evaluate_tflite`'s use of `outputs[:, 1]` throughout the training scripts).

Representative dataset for calibration: 500 samples from the validation split
(`--repr_samples`, default 500).

## Architectures

All three follow the pattern: (optional FrequencyEmphasis) → conv stack → GlobalAveragePooling2D
→ (optional Dropout) → Dense(2, softmax). Input is `(184, n_mels, 1)`.

**FrequencyEmphasis** (Nano, Micro only — not Edge): a learnable per-mel-bin gate. One weight
per mel bin plus a scalar `scale`, combined as `sigmoid(freq_weights * scale)` and multiplied
elementwise into the spectrogram before the first conv. Adds `n_mels + 1` parameters. Purpose:
let training learn which frequency bands matter for bird calls, rather than hand-picking a band
(this replaces what a Goertzel-style fixed-band filter would have hardcoded).

**Nano** (`build_cnn_mel_low_power_optimized`, 763 params):
```
Input (184, 16, 1)
  → FrequencyEmphasis(16 bins)
  → Conv2D(6, 3x3, relu, l2=1e-4) → MaxPool(2x2)
  → Conv2D(12, 3x3, relu, l2=1e-4)
  → GlobalAveragePooling2D
  → Dropout(0.1)
  → Dense(2, softmax)
```

**Micro** (`build_cnn_mel_low_power_optimized`, 919 params) — identical to Nano plus one extra
1×1 pointwise conv before pooling (channel-mixing, added in ablation step 3f):
```
Input (184, 16, 1)
  → FrequencyEmphasis(16 bins)
  → Conv2D(6, 3x3, relu, l2=1e-4) → MaxPool(2x2)
  → Conv2D(12, 3x3, relu, l2=1e-4)
  → Conv2D(12, 1x1, relu)              # <- the only structural difference from Nano
  → GlobalAveragePooling2D
  → Dropout(0.1)
  → Dense(2, softmax)
```

**Edge** (`build_deeper_gap`, 25,890 params) — architecturally distinct, no FrequencyEmphasis,
3 BatchNorm'd conv blocks, larger channel counts, a hidden Dense layer:
```
Input (184, 80, 1)
  → Conv2D(16, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  → Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
  → Conv2D(64, 3x3) → BatchNorm → ReLU        # no pooling; GAP handles it
  → GlobalAveragePooling2D
  → Dense(32, relu)
  → Dropout(0.2)
  → Dense(2, softmax)
```

## Training procedure

- **Dataset:** SEABAD, `positive`/`negative` folders recursively globbed for `.wav`. Split
  80/10/10 (train/val/test) **per class**, shuffled with `--random_seed` (default 42), so class
  balance is preserved across splits.
- **Caching:** mel spectrograms are precomputed once per `(n_fft, n_mels)` combination and cached
  to `--cache-dir` as `mels.npz` per split — re-running training reuses the cache unless
  `--force-reprocess` is passed. Caches are **not portable between n_fft/n_mels choices** —
  wrong cache reuse won't error, it'll just silently train on stale features (only
  `cache_info.pkl`'s presence is checked, not its parameters).
- **Augmentation** (train split only): additive Gaussian noise (σ=0.02, clipped to [0,1]) and a
  50%-probability random time-shift (±10 frames, `tf.roll` along the time axis).
- **Loss:** focal loss (γ=2.0, α=0.5) for Nano/Micro; plain categorical cross-entropy for Edge.
  Focal loss down-weights easy examples so training doesn't collapse toward the majority class
  even though SEABAD itself is balanced 50/50.
- **Optimizer:** legacy Adam on Apple Silicon (`tf.keras.optimizers.legacy.Adam` — works around a
  Metal performance issue), AdamW (`weight_decay=1e-4`) elsewhere. LR=0.001, `ReduceLROnPlateau`
  on `val_auc` (patience 5, factor 0.5, min 1e-5), `EarlyStopping` on `val_auc` (patience 15,
  restores best weights). Up to 100 epochs, early stopping almost always triggers first.
- **TFLite conversion:** tries default INT8 quantization first (representative-dataset
  calibrated), falls back to float16, falls back to no quantization, in that order — the shipped
  `deploy/*.tflite` files all used the first (default INT8) path successfully.

## Threshold (τ) selection

τ is *not* baked into the model — it's a downstream decision threshold applied to the
dequantized `P(bird)` output, chosen separately per variant by sweeping the test set
(`analysis/threshold_sweep_{nano,micro,edge}.py`, `analysis/tune_thresholds.py`).

- **Micro:** τ=0.30 achieves ≥0.987 recall (locked, published in `deploy/README.md`).
- **Edge:** τ=0.50 achieves ≥0.99 recall (locked, published in `deploy/README.md`).
- **Nano:** no published locked τ. `deploy/README.md`'s table leaves Nano's τ/recall columns
  blank ("—"). If you need an operating threshold for Nano, run
  `analysis/threshold_sweep_nano.py` yourself against your own recall target — don't assume
  0.30 or 0.50 transfers, thresholds are not comparable across variants with different
  architectures/capacities.
- General guidance from `LESSONS_LEARNT.md`: the default 0.5 threshold is "rarely optimal" for
  bird-detection gatekeeper tasks; lower thresholds trade precision for recall, and for a
  gatekeeper (false positives filtered downstream) that trade is usually worth it. Re-sweep after
  any requantization — INT8 shifts activations by roughly 0.01–0.05 relative to float32.

## Generalization (cross-dataset, no fine-tuning)

From `BENCHMARK_RESULTS.md` (full detail and per-seed numbers there; headline only here) —
in-domain AUC on SEABAD's own test split is not the whole story, so these numbers matter if
you're evaluating whether a model will hold up on audio unlike its training set:

| Variant | Params | TinyChirp Corn Bunting (3s, same domain family) | DCASE-2018 in-domain (bulbul-matched) | DCASE-2018 cross-corpus (zero adaptation) |
|---|---:|---|---|---|
| Nano | 763 | 0.966 ± 0.008 | 0.821 ± 0.002 | 0.446 ± 0.014 (≈ chance) |
| Micro | 919 | 0.976 ± 0.009 | 0.847 ± 0.011 | 0.488 ± 0.018 (≈ chance) |
| Edge | 25,890 | 0.9997 ± 0.0004 | 0.938 ± 0.010 | 0.646 ± 0.021 |

Cross-corpus (train on ff1010bird+warblrb10k, test on held-out BirdVox, no adaptation) collapses
toward chance for Nano/Micro — this is a real domain-shift limit, not a bug, and it's the same
gap the reference paper (bulbul, Grill & Schlüter 2017) documents for its own architecture.
**Practical implication:** these models were trained on SEABAD (South-East Asian bird
vocalizations); expect degraded performance on very different acoustic environments (different
species mix, different background noise profile) without re-training or fine-tuning on
in-domain data.

## Known gotchas

These are places where the repo's own code/comments/docs disagree with each other or with the
actual shipped artifact. Verified by parsing the `.tflite` FlatBuffers directly — treat the
`.tflite` files as ground truth over any prose (including this file, if the models are ever
regenerated without updating it).

1. **`deploy/train_nano.py` hardcodes `--n_fft 512`, but the shipped `drongonet_nano_int8.tflite`
   was trained with n_fft=1024.** Verified: the shipped model's input tensor is `[1, 184, 16, 1]`;
   `1 + floor((48000-1024)/256) = 184` matches, `1 + floor((48000-512)/256) = 186` does not. The
   training script's `compute_mel_spectrogram()` silently truncates/pads to a hardcoded 184
   regardless of what the FFT actually produces, so running `train_nano.py` as currently written
   would **not** reproduce the shipped model — it would train on n_fft=512 features truncated to
   the wrong frame count. `LESSONS_LEARNT.md` explicitly documents n_fft=512 as a known failure
   mode ("model collapse... validation AUC drops 37%+"; deployment checklist says "Use n_fft=1024,
   hop=256" as a hard requirement). **If retraining Nano, override with `--n_fft 1024`
   explicitly, or fix the wrapper's hardcoded value before running it.**
   The same 512-vs-1024 mixup also appears in `analysis/threshold_sweep_nano.py`
   (`RESULTS_DIRNAME = "6a_nano_final_fft512_m16_s{seed}"` despite its own docstring/CLI help
   saying `fft1024`) — treat any `fft512` path/filename reference to Nano in this repo as legacy
   naming debt, not a real second configuration.
2. **`deploy/train_edge.py`'s docstring says "Dense(8)" and "focal loss"; the actual
   `build_deeper_gap()` code uses `Dense(32, activation='relu')` and
   `loss='categorical_crossentropy'`.** The code (and the table in this document) is correct;
   the docstring is stale.
3. **Mel cache directories are keyed by `(n_fft, n_mels)` in their path name only by convention**
   (e.g. `cache_fft1024_m16/`) — nothing checks that a cache actually matches the config that
   reads it beyond checking `cache_info.pkl` exists. Pointing `--cache-dir` at a cache built for
   a different `n_fft`/`n_mels` will train silently on mismatched features.
4. **`deploy/README.md` and the top-level `README.md` both list Nano's cache directory as
   `cache_fft512_m16`** — same stale value as gotcha #1, propagated into the docs. Use
   `cache_fft1024_m16` for Nano.

## Verification method

Facts in this document that could silently drift (input/output tensor shapes, INT8 quantization
scale/zero_point, file sizes) were checked directly against the shipped `.tflite` FlatBuffers
rather than trusted from docstrings, using the lightweight `tflite` PyPI package (schema
bindings only, no TensorFlow runtime needed):

```python
import tflite
from tflite.Model import Model

with open("deploy/drongonet_nano_int8.tflite", "rb") as f:
    model = Model.GetRootAsModel(f.read(), 0)
subgraph = model.Subgraphs(0)
# subgraph.Tensors(subgraph.Inputs(0)) / .Outputs(0) give shape, dtype, quantization
```

If you regenerate any `.tflite` file, re-run this check before trusting this document's
Quantization/shape tables again — they describe the files that existed at the time this was
written, not a guarantee about future retraining runs.

## Practical inference recipe (any variant)

This is the loop every consumer (Raspberry Pi Python, TFLite Micro C++, etc.) implements; see
`deploy/infer_edge_rpi.py` for a working Python reference and `deploy/convert_xxd.sh` for
embedding into C firmware.

```python
interpreter = tflite.Interpreter(model_path="drongonet_{variant}_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
scale, zero_point = input_details['quantization']

mel = compute_mel_spectrogram(waveform, ...)          # (184, n_mels), float32, normalized [0,1]
mel_q = np.round(mel / scale + zero_point).astype(np.int8)
interpreter.set_tensor(input_details['index'], mel_q[np.newaxis, ..., np.newaxis])
interpreter.invoke()
out_q = interpreter.get_tensor(output_details['index'])
out_scale, out_zero_point = output_details['quantization']
p_bird = (out_q[0, 1].astype(np.float32) - out_zero_point) * out_scale
```
