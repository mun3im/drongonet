# Collecting the "Malaysian nothing" dataset

A field recording task. No machine learning knowledge needed — if you can operate a
recorder and keep a tidy spreadsheet, you can do this well.

> **Read [`INTERN_WORKPLAN.md`](INTERN_WORKPLAN.md) first.** It sets out the order of
> work — field test the existing detector *before* collecting, so that what you record
> is aimed at a failure you have actually seen. This document is the how-to for the
> recording step itself.

---

## 1. What you are collecting, in one sentence

**Recordings of Malaysia with no bird sounds in them.**

That is genuinely it. Forest with no birds. Rain. Insects at dusk. A roadside. Wind in
leaves. An empty field at 2pm when everything is too hot to sing.

## 2. Why this matters (worth reading — it changes how you record)

We have a bird detector that listens to 3 seconds of audio and says "bird" or "no
bird". It learns by example, so it needs both kinds: sounds that *are* birds, and sounds
that *are not*.

Right now it has plenty of Malaysian birds, but almost all of its "not a bird" examples
come from **England and New York State**. Out of 28,000 "no bird" examples, only **923
are from Southeast Asia** — about 3%.

So the detector has never really been told what Malaysia sounds like when there is no
bird. It has never been shown a cicada and told "that is not a bird". Cicadas make a
loud continuous noise in exactly the same pitch range as many bird calls. We fully
expect the detector to cry "bird!" at cicadas, rain, and geckos, because nobody has
ever taught it otherwise.

**Your recordings are the fix.** This is the single most valuable thing anyone can do
for this project right now — more valuable than any improvement to the model itself.

**The practical consequence:** we do not want clean, quiet, pretty recordings. We want
the noisy, annoying, cluttered sounds that a detector might mistake for a bird. Loud
insects are a *great* recording. Heavy rain is a *great* recording. Silence is nearly
useless to us.

## 3. The one rule that matters most

> **If there is a bird in it, we cannot use it.**

A recording labelled "no bird" that actually contains a bird is worse than no recording
at all — it actively teaches the detector to ignore birds, which is the exact opposite
of what we want. One bad file does more damage than ten good files do good.

This includes:

- distant, faint birds you can only just hear
- a single chirp in an otherwise empty 10 minutes
- chickens, ducks, pigeons, mynas, crows — **poultry and city birds are birds**
- caged birds, someone's pet bird, a bird on a TV or radio nearby

If you are unsure whether something is a bird, **flag it** in the log (Section 8) rather
than guessing. We would much rather throw away a doubtful recording than train on it.

Not birds, and very welcome: insects, cicadas, crickets, frogs, geckos, rain, thunder,
wind, leaves, traffic, motorbikes, aircraft, human speech, music, dogs, cats, cows,
goats, machinery, construction, mosque loudspeakers, running water.

## 4. Equipment

**Best option — record on the device we will actually deploy.** An AudioMoth, or the
Wio Terminal. Its microphone is cheap and has its own character, and a detector trained
on audio from that exact microphone performs better than one trained on audio from a
studio mic. Ask Muneim which unit to take.

**Also fine:** a handheld recorder (Zoom H1n / Tascam DR-05 type), or even a phone with
a recording app that can save uncompressed WAV.

**If you have both, record both at once** — put them side by side. Doubles the value of
every trip at no extra effort.

### Settings

| Setting | Value | Why |
|---|---|---|
| Format | **WAV** (not MP3 / not AAC / not M4A) | Compression invents and discards detail that matters |
| Sample rate | **48 kHz or 44.1 kHz** | We reduce to 16 kHz ourselves. Record high, we can always go down |
| Channels | Mono is fine; stereo is fine | Either works |
| Bit depth | 16-bit | Plenty |
| Gain | Fixed, mid-level. **Turn OFF auto-gain / AGC** | Auto-gain pumps the volume up and down and distorts the loudness patterns the detector learns from |
| Filters | **All OFF** — no noise reduction, no low-cut, no "wind filter" | We need the raw sound, warts and all |

Auto-gain and noise reduction are the two settings most likely to quietly ruin a whole
day's recording. Check them before every session.

## 5. How much, and how varied

**Eventual target: 10–15 hours of usable audio** — roughly 12,000–18,000 three-second
examples, enough to roughly double the "no bird" side of our dataset with Malaysian
content.

**But do not collect all of that before the first retrain.** Per
[`INTERN_WORKPLAN.md`](INTERN_WORKPLAN.md), the plan is: field test the existing
detector first, record **4–5 hours** aimed at whatever actually fooled it, retrain, test
again, then decide what else is worth recording. 4–5 hours is already enough for a
useful first retrain, and recordings chosen after seeing a real failure are worth far
more than recordings chosen by guessing. Treat 15 hours as where you end up after two or
three rounds, not as a gate.

**Variety beats volume.** 10 hours from 12 different places, times and weather
conditions is worth far more than 30 hours from behind your house. Once the detector has
heard one hour of one location, the second hour there teaches it almost nothing new.

Aim to cover a spread like this (adjust to what you can actually reach safely):

| Place | Roughly how much |
|---|---|
| Forest / forest reserve interior | 3 h |
| Forest edge, plantation (oil palm, rubber) | 2 h |
| Park, garden, campus grounds | 2 h |
| Roadside / urban / residential | 2 h |
| Open field, scrub, paddy | 1 h |
| Near water — stream, drain, lake, coast | 1 h |
| Indoors / under shelter during heavy rain | 1 h |

And across these conditions:

- **Times:** midday heat (birds quiet — easiest to get clean negatives), mid-afternoon,
  dusk (insects loudest), night (frogs, geckos, insects), and *late* morning. Dawn and
  early morning are the hardest, because everything is singing — you may get very little
  usable audio then, and that is expected.
- **Weather:** dry, light rain, heavy rain, windy. Rain and wind recordings are
  especially valuable and nobody ever collects enough of them.

**Night and midday are your most productive windows.** Fewest birds, plenty of other
noise. If time is short, prioritise those.

## 6. How to record

1. Pick a spot and **leave the recorder running**. Do not walk around with it, do not
   point it at things, do not start and stop it constantly.
2. **Long continuous takes: 10–30 minutes each.** Do not cut them into short clips
   yourself — we do the cutting, with a script, so that it is done identically every
   time. Hand us the long files.
3. **Put it where the real device would go.** Strapped to a tree trunk or a post, about
   1.5 m up, not held in your hand and not lying on the ground.
4. **Then walk away**, or sit still and quiet at a distance. If you must be nearby, that
   is fine — your footsteps and voice are legitimate "not a bird" sounds — but note it
   in the log.
5. Say the site name and date out loud at the *start* of each take. It is the most
   reliable way to keep track, and we cut the first few seconds off anyway.

## 7. File naming — please follow exactly

```
myneg_<site>_<YYYYMMDD>_<HHMM>_<take>.wav
```

Examples:

```
myneg_ulugombak_20260925_1430_01.wav
myneg_upmcampus_20260926_2130_03.wav
myneg_roadside_kajang_20260927_1200_01.wav
```

Rules:

- lowercase, no spaces — use underscores
- `<site>` a short consistent nickname; **use the same spelling every time** for the
  same place
- `<take>` counts up within a session: `01`, `02`, `03`
- **one file per continuous recording.** If the recorder splits a long take into
  numbered parts automatically, keep those parts and name them `_01a`, `_01b` — just
  tell us they belong together.

Why so fussy: our training code groups files by name to make sure two pieces of the
*same* recording never end up on opposite sides of a train/test split. If that grouping
breaks, our accuracy numbers silently become wrong — this has already bitten this
project once. The naming is how we prevent it.

## 8. The log — as important as the audio

One row per recording, in a spreadsheet (`collection_log.csv` or Google Sheets, either
is fine). Please fill it in on the day, not from memory a week later.

| column | example | notes |
|---|---|---|
| `filename` | `myneg_ulugombak_20260925_1430_01.wav` | must match exactly |
| `site` | `ulugombak` | same nickname as in the filename |
| `location` | `Ulu Gombak forest reserve, trail head` | plain description |
| `gps` | `3.3218, 101.7574` | phone compass/maps app is fine; blank if unavailable |
| `habitat` | `forest interior` | forest / forest edge / plantation / park / urban / field / water |
| `date` | `2026-09-25` | |
| `start_time` | `14:30` | 24-hour |
| `duration_min` | `22` | |
| `weather` | `dry, still` / `heavy rain` / `windy` | |
| `device` | `audiomoth_02` / `zoom_h1n` / `wio_01` | which recorder |
| `dominant_sounds` | `cicadas, distant traffic` | what you actually hear |
| `bird_heard` | `no` / `maybe` / `yes` | **the critical column — see below** |
| `notes` | `motorbike passed at ~4 min` | anything odd |

**`bird_heard` is the column that decides whether we can use the file.**

- `no` — you listened back and are confident there are no birds
- `maybe` — you think you heard something, or you were not paying full attention
- `yes` — there is definitely a bird in there

Mark `maybe` freely. It is not a failure; it is useful information, and it costs us
nothing to set those aside. Marking something `no` when it was really `maybe` is the
only way to actually damage this dataset.

## 9. Checking your own recordings

Please listen back to each recording before handing it over. You do not need to listen
to every second of a 20-minute file, but do this:

1. **Skim it with headphones** — scrub through, listen to 10 seconds every minute or so.
2. **Look at it in Audacity** (free). Open the file, switch the track to **Spectrogram**
   view (click the track name → Spectrogram). Bird calls show up as bright curved or
   hooked marks in the upper half. Insects look like flat continuous bands. Rain looks
   like an even grey fuzz. After ten minutes of practice you will spot bird marks faster
   than you can hear them.
3. If you find birds in only a small part of an otherwise good recording, **do not edit
   it out** — just note the time in the `notes` column (`bird at 03:10–03:25`) and mark
   `bird_heard = maybe`. We will cut around it.

## 10. Before you go out

- **Permissions.** Forest reserves, state parks and private land generally need
  permission to enter, and research recording may need a permit. Check with Muneim
  before the first trip — do not sort this out yourself on the day.
- **Safety first, data second.** Do not go into forest alone. Tell someone your route
  and expected return. Watch for weather, terrain, traffic on roadsides, and snakes.
  Leeches are likely; bring salt and covered shoes. **No recording is worth an injury —
  if a place feels unsafe, leave.**
- **Privacy.** You will inevitably record people talking. Do not deliberately record
  conversations, do not record inside private property without asking, and note it in
  the log if a recording contains a lot of clear speech so we can handle it carefully.
- **Batteries and cards.** Bring spares of both. Check free space before each session.
- **Back up the same day.** Copy the files off the card to a laptop *and* one other
  place before you wipe anything. Field recordings cannot be re-taken.

## 11. What to hand over

1. The **WAV files**, original and unedited, correctly named.
2. The **completed log** as CSV or a shared sheet.
3. A short note on anything that went wrong — settings you were unsure about, a session
   where the gain was wrong, a site where you suspect birds throughout. Problems you
   report cost us nothing; problems we discover later cost us a lot.

Hand over in batches — after your first 1–2 hours, send those over before doing more.
We will check the settings and naming and confirm you are on track. **Do not record for
days before anyone looks at the first file**, in case something about the gain or format
needs changing.

## 12. First session — a suggested start

A single 2-hour trip that tells us everything we need to know about your setup:

1. Pick somewhere easy and close. Campus grounds or a park is perfect.
2. Go at **midday** (fewest birds).
3. Record **four takes of 15 minutes** in different spots — under trees, near a road,
   open grass, near water or a drain.
4. Name them, log them, listen back, check the spectrograms.
5. Send those four files plus the log.

Once we confirm the format is right, scale up to the variety table in Section 5.

---

## Quick reference

```
[ ] WAV, 48 kHz or 44.1 kHz, 16-bit
[ ] Auto-gain OFF, noise reduction OFF, wind filter OFF
[ ] Long continuous takes, 10-30 min — do not pre-cut
[ ] Recorder mounted ~1.5 m up, not handheld
[ ] Named myneg_<site>_<YYYYMMDD>_<HHMM>_<take>.wav
[ ] Every file has a log row, bird_heard filled in honestly
[ ] Listened back and spectrogram-checked
[ ] Backed up in two places
[ ] NO BIRDS — if unsure, mark "maybe"
```

**The two things that would make this collection fail:** auto-gain left on, and
recordings marked "no bird" that have birds in them. Everything else we can work with.

Questions: ask Muneim. If something in this document does not match what you see on your
recorder, ask rather than guessing — a wrong assumption repeated across 15 hours is
expensive, and a question costs a minute.

---

## Appendix — for whoever ingests this (not the intern)

Naming checked against `mybad_cache.group_id()`: segmenting a delivered take into
`..._<take>_<segment>.wav` groups all segments of that take together, and separates
different takes. Verified.

**One residual leakage risk to handle at ingest.** Two takes from the *same session* —
`..._1430_01` and `..._1430_02` — become two different groups, so one could land in
train and the other in test despite being the same place, same hour, same weather, and
the same insects. That is a milder version of the segment leakage that cost us 48.7% of
a test split before. Group these files at **session** level instead:

```python
# for myneg_* files, strip BOTH the segment and the take index
re.sub(r'_\d+_\d+$', '', stem)   # myneg_ulugombak_20260925_1430_01_0007 -> myneg_ulugombak_20260925_1430
```

Better still, group by `site` alone for a held-out test, so the test set measures
performance at *locations never trained on* — which is the question that actually
matters for deployment. That is only possible if the intern keeps site nicknames
consistent, which is why Section 7 insists on it.

Also at ingest: resample to 16 kHz, segment to 3 s to match `N_FRAMES=184`, drop any
take whose `bird_heard` is not `no`, and record `origin="malaysia_field"` so these are
distinguishable from the DCASE-derived negatives (see `mybad_cache.DCASE_DERIVED`).
