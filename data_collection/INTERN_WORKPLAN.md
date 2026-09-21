# SparrowNet intern workplan — field test, collect, retrain, repeat

Read this first. The recording protocol
([`COLLECTING_MALAYSIAN_NEGATIVES.md`](COLLECTING_MALAYSIAN_NEGATIVES.md)) is the
detailed how-to for one step of this plan; this document is the order to do things in
and why.

---

## The single most important thing in this document

**Field test the model we already have before collecting any data.**

The tempting order is: collect data → retrain → test. That is backwards, and it wastes
weeks. We do not actually know what fails outdoors. We have guesses (cicadas, rain,
distant birds), but they are guesses. One afternoon outdoors with the existing model
tells you more than a month of guessing, and it tells you *which* recordings are worth
collecting.

So: **test first, then collect what the test says is missing.**

---

## The loop

```
Week 1   Flash existing model -> take it outside -> write down what it gets wrong
Week 2   Record 4-5 hours of exactly the sounds that fooled it
Week 3   Build minimal dataset -> retrain -> check it improved on the test set
Week 4   Take it outside again -> compare against Week 1
         then repeat 2-4 if needed
```

Each loop is about a week. Two loops beat one big effort, because the second loop is
aimed at a real measured problem instead of a predicted one.

---

## Week 1 — Field test what already exists

Nothing to train. The model is built and ready.

1. Follow [`../deploy/WIO_TERMINAL_GUIDE.md`](../deploy/WIO_TERMINAL_GUIDE.md) to flash
   the Wio Terminal. Start with `deploy/n_fft_1024/` as that guide explains — it needs
   no changes to the mel tables.
2. Get it running on live microphone input (Section 1.7 of that guide). This is the one
   real piece of coding in Week 1.
3. **Go outside and write down what happens.** See [Field test protocol](#field-test-protocol)
   below.

### What you are looking for

Not a percentage. You are looking for **specific, nameable failures**:

- "It says bird every time the cicadas start."
- "Heavy rain sets it off constantly."
- "It ignores the bulbul calling 20 m away that I can clearly hear."
- "It fires on motorbikes."
- "It is fine in the day and useless at dusk."

Each of those points at a different fix. Cicada false alarms mean we need cicada
recordings as negatives. Missing a distant bird means something different and harder —
tell Muneim, because that may need new *positive* data rather than negative.

**Deliver at end of Week 1:** a one-page list of what it got wrong, with rough
frequencies ("fired maybe 20 times in 10 minutes of cicadas"), plus the audio you
recorded alongside.

---

## Week 2 — Collect the minimal negative set

Now you know what to record. Follow
[`COLLECTING_MALAYSIAN_NEGATIVES.md`](COLLECTING_MALAYSIAN_NEGATIVES.md) for the how —
settings, naming, logging, the no-birds rule.

**But collect much less than that document's 10–15 hour target for this first loop.**

| | |
|---|---|
| **Target for loop 1** | **4–5 hours of usable audio** |
| Why that number | 5 hours ≈ 6,000 three-second clips, which is enough to replace the entire non-Malaysian negative side of a minimal training set (see [the numbers](#appendix--why-those-numbers)) |
| Priority | Whatever fooled the device in Week 1, first and most |

Split it roughly:

- **60% the specific sounds that caused false alarms in Week 1** — if it was cicadas,
  go record cicadas at dusk in several places.
- 40% general coverage: rain, wind, traffic, night, midday forest.

The 15-hour target in the other document is the eventual goal across several loops, not
a prerequisite for the first retrain. Do not wait until you have 15 hours to try
retraining.

---

## Week 3 — Build the minimal dataset and retrain

### Using the positive WAVs

Muneim will give you `/Volumes/Evo/mybad0/positive` — 28,000 three-second WAVs of
Malaysian birds, already cut and ready.

**Three rules:**

1. **Do not re-cut, re-extract, or re-download them.** They are finished. The originals
   they were cut from no longer exist, and you do not need them.
2. **Do not rename them.** The filenames carry information the training code depends on:
   `xc216946_1.wav` and `xc216946_2.wav` are two pieces of the *same* original
   recording, and the code uses the shared `xc216946` prefix to keep them on the same
   side of the train/test split. Rename them and our accuracy numbers silently become
   wrong.
3. **You do not need all 28,000.** We measured this. About **25% of them is enough** —
   see the table below. Using a quarter makes every experiment four times faster, which
   matters far more than the last 0.004 of accuracy while you are still iterating.

### The minimal dataset

| part | how many clips | where from |
|---|---|---|
| Positives (bird) | **~5,000** | 25% of the folder Muneim gives you — the training script picks them for you with one flag |
| Negatives (no bird) | **~5,000** | your Week 2 recordings, cut into 3s pieces |
| | **~10,000 total** | roughly balanced, which is what you want |

That is a dataset you can retrain in **under 10 minutes** on your laptop's GPU (about
7 minutes on CPU alone — I measured it, so a GPU is a convenience, not a requirement).

### Retraining

First turn your recordings into training clips. Check what it *would* do before it
does it:

```bash
cd ~/Dropbox/Conda/sparrownet

# dry run: reports clips per take and what it will skip, writes nothing
python ingest_field_negatives.py --audio-dir /path/to/your/takes \
    --log /path/to/collection_log.csv --dry-run

# then for real
python ingest_field_negatives.py --audio-dir /path/to/your/takes \
    --log /path/to/collection_log.csv
```

It only ingests takes whose log row says `bird_heard = no`. Anything marked `maybe`,
or missing from the log, is listed as skipped and left out — go back and re-listen to
those, then re-log them as `no` if they are clean. It also tells you how far you are
from the ~5,000-clip target and how many more minutes you need.

Then build the features and retrain:

```bash
# ~6 min, only needed when audio has been added
python mybad_cache.py --n-fft 512

# retrain on the minimal set
python train_mybad.py --frontend db --n-fft 512 \
    --pos-source-frac 0.25 \
    --seed 42 --tag LOOP1_minimal

# build the flashable model
python deploy/finalize.py results/LOOP1_minimal
```

Then compare against what exists:

```bash
python -c "
import json
for t in ['G1_sparrow_db_fft512_s42','LOOP1_minimal']:
    d=json.load(open(f'results/{t}/summary.json'))
    print(f\"{t:28s} MyBAD test AUC {d['mybad_test_auc_int8']:.4f}\")"
```

**What to expect, and this is important:** the MyBAD test AUC may go *down* slightly,
and that can still be a success. The MyBAD test set is mostly UK/US negatives, so a
model that got better at rejecting Malaysian cicadas is not rewarded by it. The number
that matters is Week 4's field test, not this one. Report both.

Two things to also try, each one flag, each ~10 minutes:

```bash
# drongonet-micro instead: smaller (5.98 KB vs 8.57 KB), was 0.966 vs 0.971
python train_mybad.py --arch drongonet_micro --frontend db --n-fft 512 --seed 42 --tag LOOP1_micro

# all the positives instead of a quarter, to see if it matters with your new negatives
python train_mybad.py --frontend db --n-fft 512 --seed 42 --tag LOOP1_allpos
```

---

## Week 4 — Test again and compare

Reflash with your retrained model and repeat the Week 1 field test **at the same places,
at the same times of day, in similar weather**. Otherwise you are comparing the weather,
not the models.

Then answer one question: **did the specific failures from Week 1 get better?** If it
fired 20 times per 10 minutes of cicadas before, what is it now? That is the result.
Not AUC.

If some failures remain, that is your Week 2 list for the next loop.

---

## Field test protocol

You need to know what the device claimed *and* what was actually there. Two ways, use
the first if you can:

**Method A — device plus recorder (better).** Run the Wio Terminal and a separate audio
recorder side by side, both started at a noted clock time. Watch the device and write
down the time of every detection. Afterwards, open the recording at those timestamps and
check whether a bird was really there. This gives you real numbers.

**Method B — sit and watch (faster, cruder).** Sit with the device for 10-minute blocks.
Every time it says "bird", note whether you heard one. Also note obvious birds it
missed. Less rigorous, but you will learn the main failure mode in one afternoon.

### Log per session

| column | example |
|---|---|
| `site`, `date`, `start_time` | `ulugombak`, `2026-10-02`, `18:30` |
| `duration_min` | `10` |
| `dominant_sound` | `cicadas, very loud` |
| `n_detections` | `23` |
| `n_correct` | `2` |
| `n_obvious_birds_missed` | `1` |
| `notes` | `fires on every cicada surge; quiet between them` |

`n_detections` vs `n_correct` is your false-alarm rate. That single pair of numbers,
measured before and after retraining, is the whole point of this project.

### Threshold

The detector has a sensitivity dial (τ). Defaults from `deploy/deploy_info.json`:

| τ | finds this share of birds | this share of alarms are real |
|---|---|---|
| 0.30 | 98% | 77% |
| 0.40 | 97% | 87% |
| 0.50 | 93% | 92% |

Start at **0.50** in the field. Those percentages come from clean recordings and will be
worse outdoors, so the less sensitive setting gives you a more informative first look.
If it is still firing constantly at 0.50, that is a real finding — write it down rather
than turning the dial to hide it.

---

## What success looks like

Not a number in a paper. For this stage:

- [ ] The device runs on live microphone input, outdoors, unattended for 10 minutes
- [ ] You can state its false-alarm rate in a named situation ("in dusk cicadas: 23
      alarms in 10 min, 2 real")
- [ ] After one retrain with Malaysian negatives, that rate measurably improves
- [ ] We know which remaining failure is the biggest

If you get all four in a month, that is a very good month.

---

## Appendix — why those numbers

**"25% of the positives is enough."** We trained on different fractions of the positive
*source recordings*, keeping the test set fixed, and measured on 8,412 held-out clips:

| positive sources used | positive clips | MyBAD test AUC |
|---|---|---|
| 12.5% (1,303 sources) | 2,447 | 0.9579 |
| **25% (2,607 sources)** | **4,889** | **0.9678** |
| 50% (5,214 sources) | 9,791 | 0.9679 |
| 100% (14,896 sources) | 19,566 | 0.9715 |

It flattens after 25%: doubling from 25% to 50% bought **0.0001**. The last 0.004 needs
four times the data. So while you are iterating, use a quarter. (One seed so far; a
second is running and may shift these by a few thousandths, not more.)

**"5 hours of negatives is enough for loop 1."** 5 hours ≈ 6,000 three-second clips,
which covers the ~5,000 negatives a balanced minimal set needs. For comparison, MyBAD
currently has only **923 Southeast Asian negative clips out of 28,000** — so 5 hours of
your recordings would multiply the Malaysian negative material by roughly six.

**Why negatives and not more positives.** MyBAD's negatives are 70% UK and US
recordings; only 3.3% are Southeast Asian. Meanwhile the positives are already Malaysian
and, per the table above, already past the point of diminishing returns. The detector
knows what Malaysian birds sound like. It has never been told what Malaysian *everything
else* sounds like. That asymmetry is the whole reason this task exists.

---

Questions to Muneim. If any command here fails, send the exact error rather than
working around it — several things in this pipeline fail silently rather than loudly,
and a wrong-but-plausible result is much more expensive than a crash.
