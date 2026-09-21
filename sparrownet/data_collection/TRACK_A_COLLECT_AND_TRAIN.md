# Track A — collect Malaysian negatives, retrain SparrowNet

Your track. Read [`../INTERN_WORKPLAN.md`](../INTERN_WORKPLAN.md) first for how your
work meets Track B's; this document is everything you actually do.

Your hardware is the **Portenta H7**, used as a development and bench platform. It does
**not** go into the field — the Wio Terminal is the field unit because it is more robust
outdoors. Field measurements are Track B's job.

**Your job in one line:** the detector has never been told what Malaysia sounds like
when there is no bird, and you are going to fix that.

---

## Why negatives, and only negatives

The detector learns from examples of both kinds: bird, and not-bird. Right now:

| | where it comes from |
|---|---|
| Its 28,000 **bird** examples | ~97% Xeno-canto recordings of Malaysian species — genuinely Malaysian |
| Its 28,000 **no-bird** examples | 70% UK and US recordings, 13% generic sound effects, and only **923 clips (3.3%) from Southeast Asia** |

So it knows Malaysian birds and it knows what an English field sounds like. It has never
heard a Malaysian cicada and been told "that is not a bird" — and cicadas sit in the same
pitch range as many bird calls. We expect false alarms on cicadas, rain and geckos, and
your recordings are the fix.

**Do not spend any effort on the bird side.** It is measurably not the bottleneck:

- More positives barely help any more. Halving them costs ~0.003 AUC, doubling gains
  ~0.003. See [the numbers](#appendix--the-measurements-behind-this).
- Species coverage does not matter. We tested it: training on a species is worth about
  2.7 points of recall on that species, because the detector learns what a bird *sounds
  like*, not which bird it is. Never go hunting for more species.

---

## Week 1 — general recording, and get your settings checked

Track B is outdoors measuring failures this week, and their findings will steer your
week 2. So week 1 is for the recordings that are useful regardless: rain, wind, night
insects, midday forest, roadside.

Follow [`COLLECTING_MALAYSIAN_NEGATIVES.md`](COLLECTING_MALAYSIAN_NEGATIVES.md) for
settings, naming, logging and the no-birds rule. Read it properly — it is short and every
rule in it exists because breaking it costs real work.

**Hand your first hour to Muneim before recording any more.** Auto-gain left on, or the
wrong file format, would silently spoil everything you record, and looking at real files
is the only way to catch it.

## Week 2 — targeted recording

Track B gives you a list of what actually fooled the detector. Aim **~60%** of your
remaining recording at those specific sounds, and 40% at general coverage.

**Total target: 6 hours of raw audio.** That yields about 5,000 usable 3-second clips
after roughly 30% is lost to birds, silence, and takes you flagged as `maybe`. Six hours
across a fortnight is unhurried; variety across sites, times and weather matters much
more than piling up hours in one place.

## Week 3 — ingest, retrain, hand over a model

### Step 1: turn your recordings into clips

Check what it will do before it does it:

```bash
cd ~/Dropbox/Conda/sparrownet

# dry run: reports clips per take and what it will skip, writes nothing
python ingest_field_negatives.py --audio-dir /path/to/your/takes \
    --log /path/to/collection_log.csv --dry-run

# then for real
python ingest_field_negatives.py --audio-dir /path/to/your/takes \
    --log /path/to/collection_log.csv
```

It only ingests takes whose log row says `bird_heard = no`. Anything marked `maybe`, or
missing from the log, is reported as skipped and left out — re-listen to those and re-log
them as `no` if they are clean. It also tells you how far you are from ~5,000 clips.

### Step 2: the positive clips — use all of them, untouched

You have `/Volumes/Evo/mybad0/positive`: 28,000 three-second WAVs, already cut. Three
rules, all "don't":

1. **Don't re-cut, re-extract or re-download.** They are finished, and the full-length
   originals no longer exist.
2. **Don't rename them.** The names are load-bearing. The trailing `_1`/`_2` is a clip
   index and everything before it is the source recording, so `xc216946_1.wav` and
   `xc216946_2.wav` are two slices of the *same* recording. The training code reads that
   shared prefix to keep both on the same side of the train/test split. Rename them and
   the accuracy numbers silently become wrong — no error, just wrong.
3. **Don't hand-pick a "best" subset.** Selecting by loudness or clarity would
   re-introduce the very bias that already makes this dataset optimistic. If you ever
   need fewer for speed, use `--pos-source-frac`, which samples whole recordings at
   random with a fixed seed.

**Keep both clips per recording.** You may reasonably worry that two slices of one
recording landing on opposite sides of the split would inflate the results. It would —
badly — but the split is by *source recording*, not by clip, so it cannot happen. Checked
on the real data: of the 18,864 recordings that have exactly 2 clips, **0** had their
clips separated, and `train_mybad.py` re-verifies this every run and aborts if it is ever
violated. A careless clip-level split would have leaked 48.7% of the test set, which is
why the guard exists. Since there is no leak, halving would only discard real signal.

### Step 3: retrain

**Use `--n-fft 1024`.** Track B's device is running the matching `mel_tables.h`, and the
model and the device's mel settings must agree — flash a 512-trained model onto a 1024
pipeline and it runs happily and outputs nonsense, with no error. Change one thing at a
time. (1024 versus 512 is worth 0.0012 AUC, i.e. noise.)

```bash
# ~6 min, only needed after new audio has been ingested
python mybad_cache.py --n-fft 1024

# three seeds, not one
for S in 42 100 786; do
  python train_mybad.py --frontend db --n-fft 1024 --seed $S --tag LOOP1_s$S
done

# build the header Track B will flash
python deploy/finalize.py results/LOOP1_s42
```

About 10 minutes per run on your GPU, ~7 on CPU.

**Three seeds, and compare means.** A single run is not a result. Twice in this project a
single-seed number sent the work down the wrong path — once picking the wrong receptive
field, once concluding "a quarter of the data is enough" when it was not. Seeds differ by
more than some of the effects you will be looking for.

```bash
python -c "
import glob, json, numpy as np
for pat in ['results/G1_sparrow_db_fft1024_s*','results/LOOP1_s*']:
    a=[json.load(open(f+'/summary.json'))['mybad_test_auc_int8'] for f in glob.glob(pat)]
    print(f'{pat:36s} {np.mean(a):.4f} +- {np.std(a):.4f}  (n={len(a)})')"
```

### Step 4: read the result correctly

**The MyBAD test AUC may go down, and that can still be a success.** That test set is 70%
UK/US negatives, so it does not reward a model that got better at rejecting Malaysian
cicadas. Track B's field re-test is the number that matters. Report both, and do not
"fix" a small AUC drop by reverting your new data.

Also check each run's own output as it goes:

- the split line ending `(no overlap)` — the leakage guard passing
- the clip counts it actually used — confirms your negatives were picked up
- the float32 versus INT8 gap. If INT8 is much worse, **stop and tell Muneim**: that is a
  frontend problem, not a training problem, and it has bitten this project before.

Two optional experiments, one flag each:

```bash
# drongonet-micro instead: smaller (5.98 KB vs 8.57 KB), scored 0.966 vs 0.971
python train_mybad.py --arch drongonet_micro --frontend db --n-fft 1024 --seed 42 --tag LOOP1_micro

# a quarter of the positives, to confirm they are still not the limiting factor
python train_mybad.py --frontend db --n-fft 1024 --pos-source-frac 0.25 --seed 42 --tag LOOP1_quarterpos
```

## Week 4 and beyond

Track B re-tests your model. Whatever still fails is your next collection list. Expect
two or three loops; the eventual negative target is 10–15 hours of audio, reached a few
hours at a time with each batch aimed at a measured failure rather than a guess.

---

## Appendix — the measurements behind this

**Positives are past diminishing returns.** Trained on fractions of the positive *source
recordings* with the test set held fixed, two seeds each, scored on 8,412 held-out clips:

| positive sources used | positive clips | MyBAD test AUC |
|---|---|---|
| 12.5% | 2,445 | 0.9578 ± 0.0001 |
| 25% | 4,887 | 0.9648 ± 0.0029 |
| 50% | 9,797 | 0.9685 ± 0.0006 |
| **100%** | **19,611** | **0.9716 ± 0.0012** |
| 1 clip per recording, all recordings | 10,427 | 0.9689 ± 0.0005 |

Diminishing but never flat — roughly 0.003 per doubling. The last row says there is no
clever subset to find: one clip from every recording scores the same as two clips from
half of them (0.9689 vs 0.9685 at almost equal clip counts).

**Species coverage does not matter.** 236 positives come from the Macaulay Library,
covering 3 species Xeno-canto blocks. Training with them entirely removed, then measuring
detection on them:

| | recall on those 236 clips | overall AUC |
|---|---|---|
| trained on them | 0.804 | 0.9716 |
| never saw them | 0.777 | 0.9707 |

**And the more useful half of that result:** those clips score ~0.80 while Xeno-canto
clips score ~0.94 *whether or not* the model trained on them. Macaulay holds more
incidental and soundscape recordings; Xeno-canto skews to deliberate close-microphone
captures. If field audio resembles Macaulay more, **0.80 predicts Track B's results
better than 0.97 does.** Which is why Track B goes outside in week 1 rather than waiting
for a better number on paper.

**Why 6 hours.** 5,000 clips is 4.2 h of usable audio; ~30% of raw recording is lost to
birds, silence and `maybe` flags. For scale, MyBAD currently holds 923 Southeast Asian
negative clips, so 6 hours multiplies the Malaysian negative material by about six.
