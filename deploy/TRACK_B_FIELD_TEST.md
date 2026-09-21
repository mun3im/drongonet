# Track B — field testing SparrowNet on the Wio Terminal

Your track. Read [`../INTERN_WORKPLAN.md`](../INTERN_WORKPLAN.md) first for how your
work meets Track A's; this document is everything you actually do.

**You have the only board that goes outside.** The Portenta H7 that Track A uses is
faster but less robust in the field, so all field measurement is yours.

**Your job in one line:** find out what the detector actually gets wrong outdoors, with
numbers, so Track A knows what to record.

---

## You are not waiting for anything

The model is already built and ready to flash. It is tempting to think field testing
should wait until Track A produces better data — it should not. Nobody knows what fails
outdoors yet. Your first week of measurements is what tells Track A which recordings are
worth making, and that is worth more than a month of guessing.

So: flash what exists, go outside, write down what happens.

---

## Week 1 — get it running and take it out

### Step 1: flash it

Follow [`WIO_TERMINAL_GUIDE.md`](WIO_TERMINAL_GUIDE.md). It covers board setup,
libraries, the four code changes, and the gotchas. Two things from it that catch people:

- Use **`deploy/n_fft_1024/sparrownet-micro.h`**. It works with the existing
  `mel_tables.h` untouched. Do not use the other one unless you also regenerate the mel
  tables — a mismatch runs without error and produces nonsense.
- The op resolver needs `AddDepthwiseConv2D` and `AddReduceMax`, which the old sketch
  does not register. Without them it fails at startup.

### Step 2: live microphone capture

The existing sketch feeds **synthetic** audio, because it was written to measure speed
rather than to detect anything. Getting the real microphone working is **the one real
coding task in this project**, and it is yours. Section 1.7 of the guide covers it: the
`Seeed_Arduino_Mic` library does DMA capture and its `mic_serial_recording` example
already records exactly 3 seconds at 16 kHz, which is precisely the input the model
wants.

Two traps:

- **Don't buffer all 3 seconds if you can avoid it.** 48,000 samples × 2 bytes = 96 KB,
  half the Wio Terminal's RAM. Better: compute one mel frame every 256 samples (16 ms) as
  audio arrives, keeping a rolling 184 × 16 buffer, about 12 KB.
- **Don't apply a high-pass filter.** That example uses `FilterBuHp`. Training used no
  filter, so adding one changes what the model sees.

### Step 3: verify before you trust it

Before drawing any conclusion from field results, confirm the device computes the same
features as the training pipeline: put known WAV files on the SD card, run them through
the device, and compare the bird scores against what Python gives for the same files.
Ask Muneim to run the Python side.

This step is not optional bureaucracy. This project has produced confident, plausible,
completely wrong numbers from preprocessing mismatches more than once. An hour here can
save a week of chasing a phantom.

### Step 4: go outside

See [the protocol](#the-field-test-protocol) below.

## Week 2 — more sites and conditions

Broaden it: different habitats, times of day, and weather. Rain and dusk are especially
worth catching. Deepen the failure list rather than just adding repetitions of the same
observation.

## Week 3 — slack week

Track A is ingesting and retraining. Use the time to extend site coverage, or help Track
A record — an extra pair of hands doubles their throughput and you already know what the
detector gets wrong.

## Week 4 — re-test and compare

Track A hands you a new `sparrownet-micro.h`. Reflash and repeat the week 1 test **at
the same sites, at the same times of day, in similar weather.** Otherwise you are
comparing the weather, not the models.

Then answer one question: **did the specific failures from week 1 get better?** If it
fired 23 times in 10 minutes of cicadas before, what is it now? That is the result — not
an accuracy percentage.

---

## The field test protocol

You need to know what the device *claimed* and what was *actually there*. Two methods;
use the first if you can.

**Method A — device plus recorder (better).** Run the Wio Terminal and a separate audio
recorder side by side, both started at a noted clock time. Write down the time of every
detection. Afterwards, open the recording at those timestamps and check whether a bird
was really there. This gives you defensible numbers, and the recordings are useful to
Track A afterwards.

**Method B — sit and watch (faster, cruder).** Sit with the device in 10-minute blocks.
Every time it says "bird", note whether you heard one; also note obvious birds it missed.
Less rigorous, but you will learn the main failure mode in a single afternoon.

### Log one row per session

| column | example |
|---|---|
| `site`, `date`, `start_time` | `ulugombak`, `2026-10-02`, `18:30` |
| `duration_min` | `10` |
| `dominant_sound` | `cicadas, very loud` |
| `n_detections` | `23` |
| `n_correct` | `2` |
| `n_obvious_birds_missed` | `1` |
| `threshold` | `0.50` |
| `notes` | `fires on every cicada surge; quiet between them` |

`n_detections` against `n_correct` is your false-alarm rate. That pair of numbers,
measured before and after Track A's retrain, is the whole point of the project.

### What you are looking for

Not a percentage — **specific, nameable failures**:

- "It says bird every time the cicadas start."
- "Heavy rain sets it off constantly."
- "It ignores the bulbul calling 20 m away that I can clearly hear."
- "It fires on motorbikes."
- "Fine in the day, useless at dusk."

Each points at a different fix, and the distinction that matters most is **false alarms
versus missed birds**:

- **False alarms** → Track A records more of that sound as a negative. Straightforward.
- **Missed birds** → a different and harder problem that more negatives will not fix.
  **Tell Muneim before Track A collects more**, or they will spend a week on recordings
  that cannot help.

### Threshold

The detector has a sensitivity dial, τ. Measured trade-offs (from
`deploy/deploy_info.json`):

| τ | finds this share of birds | share of alarms that are real |
|---|---|---|
| 0.30 | 98% | 77% |
| 0.40 | 97% | 87% |
| 0.50 | 93% | 92% |

**Start at 0.50** and record which τ you used in every log row. Those percentages come
from clean recordings and will be worse outdoors, so the less sensitive setting gives a
more informative first look. If it still fires constantly at 0.50, that is a real finding
— write it down rather than turning the dial until the problem hides.

---

## Deliver at the end of week 1

1. A one-page list of named failures with rough rates — the example block in
   [`../INTERN_WORKPLAN.md`](../INTERN_WORKPLAN.md) shows the shape.
2. Your session log.
3. Any audio you recorded alongside — useful to Track A directly.
4. Anything that went wrong or that you were unsure about. Problems you report cost
   nothing; problems discovered later cost a lot.

---

## Set your expectations before you start

The model scores **0.97 AUC** on the test set, and you should expect to measure something
noticeably worse outdoors. That is not a fault and not your mistake.

Its bird examples are overwhelmingly Xeno-canto recordings: someone deliberately pointing
a good microphone at a bird, close and clear. Your device hears a distant bird, quietly,
through wind and insects. A set of more incidental recordings in the same dataset scores
**0.80** rather than 0.94, and field audio probably resembles those more.

Measuring 0.80-ish behaviour, or a pile of cicada false alarms, is the **expected
result** — and it is a useful one, because it is the first honest measurement anyone on
this project will have.
