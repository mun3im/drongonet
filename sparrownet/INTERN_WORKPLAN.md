# SparrowNet — two-person work plan

Two interns, two tracks, running **in parallel**:

| | Track A | Track B |
|---|---|---|
| Person | data + training | field testing |
| Hardware | **Portenta H7** | **Wio Terminal** |
| Job | collect Malaysian negatives, retrain the model | flash the detector, take it outside, measure what it gets wrong |
| Your document | [`data_collection/TRACK_A_COLLECT_AND_TRAIN.md`](data_collection/TRACK_A_COLLECT_AND_TRAIN.md) | [`deploy/TRACK_B_FIELD_TEST.md`](deploy/TRACK_B_FIELD_TEST.md) |

Read your own track's document. This page is only the part you both need: what the other
person is doing, and where you hand things over.

---

## Why field testing does not wait for better data

Track B starts immediately with the model that already exists. It is tempting to think
field testing should wait until Track A has produced something better — it should not.

Nobody knows what actually fails outdoors. We have strong suspicions (cicadas, rain,
distant birds) but they are suspicions. Track B's first week of measurements is what
tells Track A **which recordings are worth making**, and that is worth far more than a
week of guessing. So Track B goes out with the current model in week 1, while Track A
starts on general-purpose recording.

---

## The two handoffs

Everything else is independent. These are the only two points where you need each other.

### Handoff 1 — B to A, end of week 1: "here is what it gets wrong"

Track B delivers a short list of named failures with rough rates, e.g.:

```
dusk cicadas, Ulu Gombak:  23 alarms in 10 min, 2 real
heavy rain, under shelter: fires almost continuously
distant bulbul ~20 m:      missed it, I could hear it clearly
motorbikes:                no false alarms, fine
```

Track A then aims ~60% of recording at whatever caused the false alarms, and stops
guessing.

**If Track B reports missed birds rather than false alarms, tell Muneim before
collecting more negatives.** Missed birds are a different problem and more negatives
will not fix it.

### Handoff 2 — A to B, end of week 3: "here is a new model"

Track A delivers a `sparrownet-micro.h` built by `deploy/finalize.py`. Track B reflashes
and repeats the week 1 test **at the same sites, times of day and weather** — otherwise
you are comparing the weather rather than the models.

**The header must match the device's mel settings.** Track A trains with
`--n-fft 1024` for exactly this reason: it matches the `mel_tables.h` already on the Wio
Terminal. A mismatch here runs without error and produces nonsense, so if Track A ever
changes the frontend, both tracks change together and deliberately.

---

## Rough schedule

```
Week 1   B: flash, get live mic working, go outside, measure     A: general recording, 1st hour to Muneim for checking
Week 2   B: more sites and conditions, deepen the failure list   A: targeted recording aimed at B's findings
Week 3   B: (slack — help A record, or extend site coverage)     A: ingest, retrain 3 seeds, hand over a header
Week 4   B: re-test at the week 1 sites, compare                 A: whatever the comparison says to try next
```

Track A needs about **6 hours of raw audio** total, which is a fortnight of unhurried
work, not a full-time job. Track B's bottleneck is week 1: getting live microphone
capture running is the only real coding task in the project.

---

## What "done" looks like for this stage

- [ ] The detector runs on live microphone input, outdoors, unattended for 10 minutes
- [ ] We can state its false-alarm rate in a named situation, with numbers
- [ ] One retrain with Malaysian negatives measurably improves that rate
- [ ] We know which remaining failure is the biggest

Four bullets in a month is a good month.

---

## Which board goes outside, and which does not

**All field testing is on the Wio Terminal.** The Portenta H7 is the less robust of the
two in the field, so it is not going outdoors at this stage. That is a deliberate
decision, not an oversight.

So the two boards have different jobs:

| | Wio Terminal (Track B) | Portenta H7 (Track A) |
|---|---|---|
| Role | the field unit | bench and development platform |
| Goes outside | yes | **no, not yet** |
| Why | robust enough to take out | faster and roomier, but not field-hardened |

The practical consequence for Track A: your Portenta is for development, training work
and bench measurements — not for taking into a forest. If you want to check how the
model behaves on hardware, that is a bench exercise, and this project already has
measurements to compare against (`argus/DRONGONET_SPARSE_MEL_BENCHMARK.md`: drongonet-micro
runs in 75 ms on the Portenta's M4 at 240 MHz and 27 ms on its M7 at 480 MHz, against
238 ms on the Wio Terminal).

It also means the **10 KB / 16 KB model budget stands**. The budget exists for the
Cortex-M4 class part that actually goes outside, and the Portenta's much larger memory
does not relax it. Both tracks use the same 8.57 KB model, which keeps every number from
the two tracks directly comparable.

---

## Shared background, if you want it

- [`README.md`](README.md) — what SparrowNet is and why the architecture and frontend
  are what they are
- The headline accuracy is **0.97 AUC**, and both tracks should treat that as an upper
  bound rather than an expectation. It is measured on Xeno-canto recordings, which are
  deliberate close-microphone captures of a targeted bird. A comparable set of more
  incidental recordings in the same dataset scores **0.80**, and field audio is likely
  to resemble the latter. Track B measuring something much worse than 0.97 is the
  expected outcome, not a sign that something is broken.
