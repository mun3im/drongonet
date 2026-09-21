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
| **Record for loop 1** | **6 hours of raw audio** |
| Which yields | ~5,000 usable 3-second clips — enough to match the positives on the negative side |
| Why 6 and not 4 | 5,000 clips is 4.2 h of *usable* audio; budget ~30% loss to birds, silence and `maybe` flags, so record 6 h to land 5,000 |
| Priority | Whatever fooled the device in Week 1, first and most |

**Hand over the first hour before recording the other five.** Auto-gain left on, or the
wrong file format, would silently spoil the whole batch, and the only way to catch it is
for someone to look at real files early.

Split it roughly:

- **60% the specific sounds that caused false alarms in Week 1** — if it was cicadas,
  go record cicadas at dusk in several places.
- 40% general coverage: rain, wind, traffic, night, midday forest.

The 15-hour target in the other document is the eventual goal across several loops, not
a prerequisite for the first retrain. Do not wait until you have 15 hours to try
retraining.

---

## Week 3 — Build the minimal dataset and retrain

### Using the positive WAVs — you already have them, and there is nothing to do

You have `/Volumes/Evo/mybad0/positive`: 28,000 three-second WAVs of Malaysian birds,
already cut and ready. **Use all of them, as they are.** Three rules, all "don't":

1. **Don't re-cut, re-extract, or re-download.** They are finished. The full-length
   originals they were cut from no longer exist, and you do not need them.
2. **Don't rename them.** The filenames are load-bearing. The trailing `_1`/`_2` is a
   clip index and everything before it is the source recording, so `xc216946_1.wav` and
   `xc216946_2.wav` are two pieces of the *same* recording. The training code reads that
   shared `xc216946` prefix to keep both on the same side of the train/test split. Rename
   them and our accuracy numbers silently become wrong — no error, just wrong.
3. **Don't hand-pick a "best" subset.** If you ever need fewer for speed, use the
   `--pos-source-frac` flag, which samples whole source recordings at random with a
   fixed seed. Choosing by loudness or "clarity" would re-introduce exactly the bias
   that already makes this dataset optimistic (see [the numbers](#appendix--why-those-numbers)).

**You do not need to subset at all.** A full run takes about 10 minutes on your GPU
(~7 minutes on CPU — measured, so the GPU is a convenience, not a requirement). Earlier
guidance here said "25% is enough"; a second training seed showed that was noise, and
25% actually costs about 0.007 AUC. Not much, but there is no reason to pay it when the
full run is ten minutes.

### "Both clips come from one recording — won't that inflate the results?"

Good question, and the honest answer is: it *would*, badly, if the split were done
naively — but it is not, so **keep both clips**.

A split that picked random *clips* would put `xc216946_1` in training and `xc216946_2`
in test. Those are two slices of the same bird in the same recording with the same
background, so the model would effectively be tested on its training data and the score
would come out flattering and meaningless. On this dataset we measured that a naive
clip-level split would leak **48.7% of the test set** — nearly half.

The code splits by **source recording** instead, so both clips of a recording always
land on the same side. Verified on the real data: of the 18,864 sources that have
exactly 2 clips, **0** had their clips end up in different splits. `train_mybad.py`
also re-checks this on every run and aborts if it is ever violated, so it cannot
quietly regress.

Because there is no leak to fix, halving the data would only throw away real training
signal: one clip per recording scores 0.9689 against 0.9716 for both. Keep both.

There is one place this concern still bites, and it is on **your** side: two takes
recorded at the same site an hour apart are *not* the same recording by filename, but
they do share place, weather and insects. Section 7 of the collection protocol explains
the naming that lets us group them properly at ingest.

### The minimal dataset

| part | how many clips | where from |
|---|---|---|
| Positives (bird) | **28,000** — all of them | the folder you already have |
| Negatives (no bird) | **~5,000** | your Week 2 recordings, cut into 3s pieces |

The positives already outnumber what you will collect, and that is fine — the script
handles the imbalance. Your job is entirely the negative side.

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

Then build the features and retrain.

**Use `--n-fft 1024`, and this matters.** Week 1 flashed the `n_fft_1024` model because
it works with the device's existing `mel_tables.h` untouched. The model and the device's
mel settings must agree: flash a 512-trained model onto the 1024 mel pipeline and it
will run happily and produce nonsense, with no error to warn you. Train at 1024, keep
the same tables, change one thing at a time. (1024 vs 512 is worth 0.0012 AUC — noise.)

```bash
# ~6 min, only needed when audio has been added
python mybad_cache.py --n-fft 1024

# retrain: all positives + your new negatives. Three seeds, not one -- see below.
for S in 42 100 786; do
  python train_mybad.py --frontend db --n-fft 1024 --seed $S --tag LOOP1_s$S
done

# build the flashable model from whichever seed you flash
python deploy/finalize.py results/LOOP1_s42
```

**Run three seeds and compare the mean.** A single run is not a result. Twice in this
project a single-seed number sent us down the wrong path — once picking the wrong
receptive field, once concluding "a quarter of the data is enough" when it was not.
Different random seeds on the same data can differ by more than the effect you are
trying to measure.

```bash
python -c "
import glob, json, numpy as np
for pat in ['results/G1_sparrow_db_fft1024_s*','results/LOOP1_s*']:
    a=[json.load(open(f+'/summary.json'))['mybad_test_auc_int8'] for f in glob.glob(pat)]
    print(f'{pat:36s} {np.mean(a):.4f} +- {np.std(a):.4f}  (n={len(a)})')"
```

Also sanity-check each run's own output as it goes: it prints the split sizes and
`(no overlap)`, the clip counts it actually used, and the float32-vs-INT8 gap. If the
INT8 number is much worse than float32, stop and tell Muneim — that is a frontend
problem, not a training problem.

**What to expect, and this is important:** the MyBAD test AUC may go *down* slightly,
and that can still be a success. The MyBAD test set is mostly UK/US negatives, so a
model that got better at rejecting Malaysian cicadas is not rewarded by it. The number
that matters is Week 4's field test, not this one. Report both.

Two things to also try, each one flag, each ~10 minutes:

```bash
# drongonet-micro instead: smaller (5.98 KB vs 8.57 KB), was 0.966 vs 0.971
python train_mybad.py --arch drongonet_micro --frontend db --n-fft 1024 --seed 42 --tag LOOP1_micro

# a quarter of the positives, to check they are not the limiting factor any more
python train_mybad.py --frontend db --n-fft 1024 --pos-source-frac 0.25 --seed 42 --tag LOOP1_quarterpos
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

**How much do the positives matter?** We trained on different fractions of the positive
*source recordings*, keeping the test set fixed, and scored 8,412 held-out clips. Two
seeds each:

| positive sources used | positive clips | MyBAD test AUC |
|---|---|---|
| 12.5% (1,303 sources) | 2,445 | 0.9578 ± 0.0001 |
| 25% (2,607 sources) | 4,887 | 0.9648 ± 0.0029 |
| 50% (5,214 sources) | 9,797 | 0.9685 ± 0.0006 |
| **100% (all sources)** | **19,611** | **0.9716 ± 0.0012** |
| 1 clip per source, all sources | 10,427 | 0.9689 ± 0.0005 |

Diminishing but never flat — each doubling still buys about 0.003. An earlier draft of
this document said it "flattens after 25%", which came from a single seed where 25% and
50% happened to land 0.0001 apart; the second seed showed that was luck. Corrected: use
all of them, since a full run is ten minutes.

The last row is the interesting one. Using one clip from *every* recording (10,427
clips) scores the same as two clips from *half* the recordings (9,797 clips) — 0.9689
vs 0.9685. At matched clip count, recording variety and clip count are worth about the
same here, so there is no clever subset to find. More data helps a little; nothing else
about the positives is a lever.

**Why 6 hours of negatives.** 5,000 clips is 4.2 h of usable audio, and roughly 30% of
raw recording is lost to birds, silence and `maybe` flags — hence 6 h. For scale, MyBAD
currently holds only **923 Southeast Asian negative clips out of 28,000**, so 6 hours
multiplies the Malaysian negative material by about six.

**Species coverage does not matter (tested).** 236 of the positives come from the
Macaulay Library, covering 3 species Xeno-canto blocks. We trained with those clips
entirely removed and then measured detection on them:

| | recall on those 236 clips | overall AUC |
|---|---|---|
| trained on them | 0.804 | 0.9716 |
| never saw them | 0.777 | 0.9707 |

Having heard a species is worth about 2.7 points of recall on it. The detector is
learning what a bird *sounds like*, not which bird it is, so it finds unfamiliar species
from the general pattern. **Practical consequence: never go hunting for more species
coverage.** Nothing about the bird side of this dataset needs your attention.

That same test surfaced something more useful, and it is the reason to distrust the
headline number: those Macaulay clips score ~0.80 while Xeno-canto clips score ~0.94,
*whether or not* the model trained on them. Macaulay holds more incidental and
soundscape recordings; Xeno-canto skews to deliberate close-microphone captures. If
field audio resembles Macaulay more, then **0.80 predicts your field recall better than
0.97 does**. Which is the whole reason Week 1 is a field test and not another
experiment.

**Why negatives and not more positives.** MyBAD's negatives are 70% UK and US
recordings; only 3.3% are Southeast Asian. Meanwhile the positives are already Malaysian
and, per the table above, already past the point of diminishing returns. The detector
knows what Malaysian birds sound like. It has never been told what Malaysian *everything
else* sounds like. That asymmetry is the whole reason this task exists.

---

Questions to Muneim. If any command here fails, send the exact error rather than
working around it — several things in this pipeline fail silently rather than loudly,
and a wrong-but-plausible result is much more expensive than a crash.
