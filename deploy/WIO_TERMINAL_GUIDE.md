# Running SparrowNet on a Wio Terminal

A handoff guide. Assumes no prior knowledge of this project — just Arduino experience.

**Short version:** use the **Arduino IDE** path. Edge Impulse cannot run this model
as-is, for a reason explained in [Part 2](#part-2--edge-impulse) that is worth reading
before you try.

---

## What SparrowNet is

A bird / no-bird detector small enough for a microcontroller. It answers one question
about a 3-second audio clip: *was there a bird in it?* It does not identify species.

| | |
|---|---|
| Model file | `deploy/n_fft_1024/sparrownet_deploy_int8.tflite` (8.57 KB, INT8) — start here |
| C header | `deploy/n_fft_1024/sparrownet-micro.h` (ready to `#include`) |
| Faster variant | `deploy/sparrownet-micro.h` (n_fft=512, needs new mel tables — see 1.5d) |
| Parameters | 603 |
| Input | `[1, 184, 16, 1]` INT8 — a 184-frame × 16-mel-band spectrogram of 3 s of audio |
| Output | `[1, 2]` INT8 — index **1** is the bird probability |
| Accuracy | **AUC 0.970** (n_fft=1024) / **0.971** (n_fft=512) on 8,387 held-out MyBAD clips |
| Compute | 85,008 MACs; ~11 ms inference estimated on a Wio Terminal @120 MHz |

Trained on **MyBAD v0.8.0** (Malaysian bird audio, 56,000 clips). Evaluated on clips
from 5,571 source recordings that were never in training.

### Two honest limits, please read

1. **0.971 AUC is an upper bound, not a field expectation.** MyBAD's positive clips are
   ~97.5% Xeno-canto recordings, which are close-microphone recordings of a targeted
   bird. Real outdoor audio has the bird far away, plus wind, rain, insects and traffic.
   Expect noticeably worse in the field. This is not a defect in the model; it is what
   the available training data supports.
2. **A Wio Terminal is not a low-power logger.** 120 MHz with an LCD on draws far too
   much for a multi-week deployment on batteries. It is excellent for a demo, a
   workbench test, or a short attended field trial. For unattended long deployments the
   right hardware is an AudioMoth.

---

## Part 1 — Arduino IDE

### 1.1 Hardware

- Seeed Wio Terminal (chip: ATSAMD51P19A, Cortex-M4F @120 MHz, 192 KB SRAM)
- USB-C cable
- Nothing else — the Wio Terminal has a built-in microphone

### 1.2 Board support

In Arduino IDE: **File → Preferences → Additional Board Manager URLs**, add

```
https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
```

Then **Tools → Board → Boards Manager**, search `Seeed SAMD`, install
**Seeed SAMD Boards**. Select **Tools → Board → Seeed SAMD Boards → Seeeduino Wio
Terminal**.

> If that URL has moved, get the current one from Seeed's wiki page for the Wio
> Terminal — the URL is the only part of this guide likely to go stale.

### 1.3 Libraries

Install via **Tools → Manage Libraries** (or copy from
`~/Dropbox/Conda/argus/libraries/`, where known-good copies already live):

| Library | Why | Version known to work |
|---|---|---|
| `Chirale_TensorFlowLite` | TensorFlow Lite Micro — provides `<TensorFlowLite.h>` | 2.0.0 |
| `Seeed_Arduino_Mic` | microphone capture via DMA+ADC, Wio Terminal only | 1.0 |
| `TFT_eSPI` | the LCD, for showing status | Seeed's fork |
| `Arduino_CMSIS-DSP` | `arm_math.h`, the fast FFT | — |

`Chirale_TensorFlowLite` is the one that matters. Do **not** substitute the older
`Arduino_TensorFlowLite`; the existing sketches in this project are built against
Chirale and the `MicroMutableOpResolver` API differs between them.

### 1.4 Start from the existing benchmark sketch

Do not write a sketch from scratch. Copy this one:

```
~/Dropbox/Conda/argus/wt_drongonet_micro_bench/
```

It already does the hard parts: a CMSIS-DSP real-FFT mel spectrogram (4.1× faster than
the naive version), TFLite Micro setup, a 32 KB tensor arena, LCD status, and
per-stage timing. Rename the folder to `wt_sparrownet_bench` and make the four changes
below.

### 1.5 The four changes

**(a) Swap the model.** Copy `deploy/n_fft_1024/sparrownet-micro.h` into the sketch
folder (see (d) for why this one rather than `deploy/sparrownet-micro.h`), then:

```cpp
// #include "drongonet-micro.h"
#include "sparrownet-micro.h"
```

The array is named `sparrownet_model_data`, with length `sparrownet_model_data_len`.
Update the `tflite::GetModel(...)` call to match.

**(b) Fix the op resolver — the sketch will fail without this.** The existing resolver
registers the ops drongonet needs, which are not the ops SparrowNet needs. SparrowNet
uses exactly: `MUL ×1`, `CONV_2D ×4`, `DEPTHWISE_CONV_2D ×3`, `REDUCE_MAX ×1`,
`SOFTMAX ×1`. So:

```cpp
static tflite::MicroMutableOpResolver<5> resolver;   // was <6>
resolver.AddMul();                 // the learned frequency-emphasis gate
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();     // NEW — drongonet did not use depthwise convs
resolver.AddReduceMax();           // NEW — the global-max pooling head
resolver.AddSoftmax();
// AddMaxPool2D / AddMean / AddFullyConnected are no longer needed
```

If `AllocateTensors()` fails or `Invoke()` returns an error, a missing op is the first
thing to check — TFLM reports it on the serial console.

**(c) Add the missing 80 dB floor.** The sketch's normalisation computes
`10*log10(power)` then rescales to [0,1]. Training additionally clamps everything more
than 80 dB below the clip's peak, and without that clamp a clip with a very wide
dynamic range normalises differently on the device than it did in training. After
computing the mel frame values and before the min-max rescale:

```cpp
// match training: power_to_db(ref=max, top_db=80)
float peak;  uint32_t pkIdx;
arm_max_f32(melOut, INPUT_SIZE, &peak, &pkIdx);
const float floorDb = peak - 80.0f;
for (int i = 0; i < INPUT_SIZE; i++) {
  if (melOut[i] < floorDb) melOut[i] = floorDb;
}
// ... then the existing arm_min_f32 / arm_offset_f32 / arm_scale_f32 min-max rescale
```

This project has already lost time twice to preprocessing mismatches that produced
plausible-looking but wrong results (see `argus/MEL_NORMALISATION_FINDING.md`). Getting
this exactly right matters more than it looks.

**(d) Choose your FFT size.** Both versions are already built for you, same 8.57 KB,
same op set — pick one:

| Use this | Model | Mel tables | AUC | Why |
|---|---|---|---|---|
| **Start here** | `deploy/n_fft_1024/sparrownet-micro.h` | existing `mel_tables.h`, unchanged | 0.9702 | Works with the sketch as-is. Nothing to regenerate. |
| Later, for speed | `deploy/sparrownet-micro.h` | must regenerate for `n_fft=512` | 0.9714 | Roughly halves the mel computation, which is the real bottleneck (see below) |

The accuracy difference (0.0012 AUC) is smaller than the seed-to-seed spread, so it is
not a real consideration — this is purely a speed-versus-effort choice. **Start with
`n_fft_1024/`**, confirm the whole chain works end to end, and only then regenerate the
tables for 512 if you want the mel stage faster.

### 1.6 Flash and check the timing

Build and upload. The sketch prints per-stage timings to serial at 115200 baud. For
reference, measured on this exact hardware with the *previous* model
(`argus/WIO_TERMINAL_DRONGONET_LATENCY.md`, N=100, reproduced over three flashes):

| | mel | inference | total |
|---|---|---|---|
| drongonet-micro (n_fft=1024) | 141 ms *(measured)* | 97 ms *(measured)* | **239 ms** |
| SparrowNet, n_fft=1024 — what you flash first | ~141 ms *(same mel code)* | ~11 ms *(est.)* | **~152 ms** *(est.)* |
| SparrowNet, n_fft=512 — after regenerating tables | ~65–70 ms *(est.)* | ~11 ms *(est.)* | **~80 ms** *(est.)* |

SparrowNet needs 8.7× fewer multiply-accumulates, so inference should drop sharply.
**Note the mel spectrogram dominates, not the neural network** — so if you want it
faster, optimise the FFT, not the model. The SparrowNet figures above are estimates;
your serial output will be the first real measurement, and it is worth reporting back.

Arena use should be comfortably under the 32 KB already allocated (drongonet-micro used
23.6 KB and SparrowNet's activations are smaller). The sketch prints actual usage.

### 1.7 Add live microphone capture

The benchmark sketch feeds **synthetic audio** (`fillSyntheticAudio`) because it only
measures speed. For anything real you need the microphone. `Seeed_Arduino_Mic` does
this with DMA and its `mic_serial_recording` example already captures exactly 3 s at
16 kHz — the precise input this model wants:

```cpp
#include <mic.h>
mic_config_t mic_config{
  .channel_cnt   = 1,
  .sampling_rate = 16000,   // must be 16000
  .buf_size      = 320,
  .debug_pin     = 1
};
DMA_ADC_Class Mic(&mic_config);
// Mic.set_callback(cb); Mic.begin();  — cb receives uint16_t* chunks
```

Two things to get right:

- **Don't buffer all 3 seconds if you can avoid it.** 48,000 samples × 2 bytes = 96 KB,
  half the Wio Terminal's RAM. Better: compute one mel frame every 256 samples (16 ms)
  as audio arrives and keep a rolling 184 × 16 buffer — about 12 KB.
- **Do not apply a high-pass filter.** The `mic_serial_recording` example uses
  `FilterBuHp`. Training had no such filter, so adding one changes the features the
  model sees. Leave the audio unfiltered.

### 1.8 Turn the output into a decision

Output index **1** is the bird score. To convert the raw INT8 output:

```cpp
float p_bird = (out_int8[1] - (-128)) * 0.00390625f;   // zero_point, scale
```

Then compare against a threshold. Pick from the measured trade-off — these are real
numbers from the held-out test set, not guesses:

| threshold | recall (birds found) | precision (alarms that are real) |
|---|---|---|
| 0.25 | 99.0% | 70.2% |
| **0.30** | **98.4%** | **77.3%** |
| 0.35 | 97.7% | 81.9% |
| **0.40** | **96.6%** | **86.8%** |
| 0.50 | 93.4% | 91.5% |
| 0.60 | 87.1% | 95.1% |

Use **0.30** if missing a bird is worse than a false alarm (typical for a survey
trigger). Use **0.50** if you want most alarms to be real. Do **not** just use 0.5
because it is the default — decide from the table.

Full quantisation details, if you need them:

```
input  scale 0.003921568859368563   zero_point -128
output scale 0.00390625             zero_point -128
```

### 1.9 Verify before you trust it

Before drawing any conclusion from field results, confirm the device computes the same
features as the training pipeline. Put a few known WAV files on the SD card, run them
through the device, and compare the bird scores against what Python produces for the
same files (`evaluate_mybad.py` does this side). They should agree closely.

Skipping this step is how you end up with a device that reports confident nonsense. It
has already happened in this project's history more than once.

---

## Part 2 — Edge Impulse

**Edge Impulse cannot run this model as-is, and it is important to understand why
before spending a day on it.**

A model is only half of an audio classifier; the other half is the feature extraction
that turns sound into a spectrogram. SparrowNet expects a very specific spectrogram:
16 kHz, `n_fft=512`, hop 256, 16 mel bands from 100–8000 Hz, converted to decibels
relative to the clip's peak with an 80 dB floor, then scaled to [0,1].

Edge Impulse's audio blocks (MFE / MFCC) compute *their own* features with their own
parameters and their own normalisation ("noise floor" in dB). Those features are not the
ones SparrowNet was trained on. Upload the `.tflite` via Edge Impulse's
"Bring Your Own Model" and it will run — and produce meaningless output, because the
numbers going in are on a different scale from the numbers it learned. It will not error.
That failure mode is silent, which is what makes it dangerous.

So there are two legitimate options:

### Option A — retrain inside Edge Impulse (recommended if you want to use EI)

Treat this as building a *separate* model rather than deploying SparrowNet.

1. Get the MyBAD WAV files (`/Volumes/Evo/mybad0/{positive,negative}/`, 56,000 files —
   ask Muneim for a copy or a subset).
2. Upload to Edge Impulse as two classes, `bird` and `nobird`.
3. **Split by source recording, not randomly.** This matters enormously: MyBAD's 28,000
   positive clips come from only 14,896 source recordings, so many clips are different
   3-second segments of the *same* recording. A random split puts segments of one
   recording in both training and test, and the reported accuracy comes out far too
   high — measured at **48.7% of test clips leaking** on this dataset. Filenames encode
   the source: `xc216946_1.wav` and `xc216946_2.wav` are the same recording, so keep
   all files sharing a prefix (everything before the final `_N`) on the same side.
   Edge Impulse's own advice on this is its "data explorer / manual split" — do not use
   the automatic 80/20.
4. Use an MFE block and a small conv network, then deploy as an Arduino library.
5. Expect a different accuracy number from 0.971 — not comparable, different features
   and a different model. Judge it on its own held-out split.

Seeed has a worked tutorial for exactly this hardware and workflow, already bookmarked
in this project: <https://wiki.seeedstudio.com/Wio-Terminal-TinyML-EI-3/>

### Option B — replicate our features inside Edge Impulse (hard, not recommended)

Possible via a custom DSP block that reimplements the exact frontend above, then BYOM
with our `.tflite`. This is strictly more work than Part 1 and gains nothing, because
Part 1's sketch already has a working, profiled implementation of that frontend. Only
worth it if Edge Impulse's device management is a hard requirement for you.

---

## Part 3 — Mistakes this project has already made

Offered so you don't repeat them. Every one produced confident, plausible, wrong numbers
rather than an error.

1. **Feeding differently-scaled features.** MyBAD's published `.npy` spectrograms use a
   different recipe from ours (`log1p` instead of dB, `n_fft=512`, `fmin=20`,
   percentile clipping). Scoring our model on them gave a believable AUC that meant
   nothing. Always confirm the features match.
2. **Random train/test splits on segmented audio.** See the 48.7% leak above.
3. **Trusting an unmeasured latency figure.** An earlier document in this project
   claimed 0.1–0.3 ms inference on a 48 MHz Cortex-M4. The real measured figure on a
   *faster* 120 MHz part was 97 ms — about 300× off. Measure on hardware.
4. **Evaluating only in float32.** The `log1p` frontend loses barely any accuracy in
   float32 but **0.14 AUC after INT8 quantisation**, because its values bunch up near
   zero and waste the INT8 range. If you change the frontend, re-check accuracy *after*
   quantisation, never before.
5. **Assuming a graph will run on a microcontroller.** SparrowNet's original output head
   produced dynamic-shape tensors that TensorFlow Lite Micro cannot allocate. It
   converted and ran fine on a PC. The fix is already in the model you have.

---

## Part 4 — Checklist

```
[ ] Seeed SAMD board package installed, "Seeeduino Wio Terminal" selected
[ ] Chirale_TensorFlowLite, Seeed_Arduino_Mic, TFT_eSPI, Arduino_CMSIS-DSP installed
[ ] Copied wt_drongonet_micro_bench -> wt_sparrownet_bench
[ ] sparrownet-micro.h copied in and #included; GetModel() updated
[ ] Op resolver: Mul, Conv2D, DepthwiseConv2D, ReduceMax, Softmax  (size 5)
[ ] 80 dB floor added to the mel normalisation
[ ] FFT size decided (1024 model as-is, or regenerate tables for 512)
[ ] Flashed; serial shows mel + inference timings and arena usage
[ ] Live mic capture working at 16 kHz, no high-pass filter
[ ] Parity checked against Python on the same WAV files
[ ] Threshold chosen deliberately from the table in 1.8
```

---

## Where things are

| | |
|---|---|
| Model + header | `~/Dropbox/Conda/sparrownet/deploy/n_fft_1024/` (start here), `deploy/` (n_fft=512) |
| Full metadata | `deploy/deploy_info.json` (ops, quantisation, threshold sweep) |
| Regenerate artifacts | `python deploy/finalize.py results/<run>` |
| Size / MAC / RAM check | `python deploy/footprint.py <model.tflite> --clock 120` |
| Existing Wio sketches | `~/Dropbox/Conda/argus/wt_drongonet_*_bench/` |
| Measured Wio latency | `~/Dropbox/Conda/argus/WIO_TERMINAL_DRONGONET_LATENCY.md` |
| Preprocessing bug history | `~/Dropbox/Conda/argus/MEL_NORMALISATION_FINDING.md` |
| Training / evaluation code | `~/Dropbox/Conda/sparrownet/` (see its `README.md`) |

Questions that this guide cannot answer are probably answered in
`sparrownet/README.md`, which records why the architecture and frontend are what they
are.
