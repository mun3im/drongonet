"""
mybad_cache.py — build a MyBAD cache from the RAW WAVs using our own frontend.

Why not use the shipped .npy spectrograms: MyBAD v0.8.0's generator
(mybad-curation/zenodo_compiler/create_zenodo_spectrograms.py, verified to reproduce
the shipped arrays to 3e-07) uses n_fft=512, fmin=20, center=True, np.log1p, and a
p1/p99 percentile clip. Measured on MyBAD's own audio, 99.1% of power-mel values fall
below 0.1, where log1p(S) ~= S to within 5% -- so that "log" spectrogram applies
essentially no compression and is a linear power spectrogram in disguise, which
crushes faint calls into the bottom of the range. We control the firmware frontend, so
there is no reason to inherit it.

Two label hazards this module exists to handle:

1. SEGMENT LEAKAGE. 28,000 positive clips come from only 14,896 source recordings
   (~1.9 segments each); negatives, 22,240 sources. Two 3s segments of one xeno-canto
   recording share bird, recorder and background, so a clip-level split leaks and
   inflates test scores. `group_id()` recovers the source and `grouped_split()` splits
   on it.

2. DCASE OVERLAP. Every one of the 5,754 `ff-*` source ids in MyBAD's negatives is a
   freefield1010 itemid, negative in both datasets -- i.e. essentially all of
   freefield1010's negatives are reused here (~10% of all MyBAD clips). Any
   DCASE-trained model evaluated on MyBAD has therefore already seen them. `origin()`
   labels each clip so those can be excluded or reported separately.
"""

import os
import re
import glob
import argparse
import time
from multiprocessing import Pool

import numpy as np
import librosa

from config import (
    CACHE_BASE, SAMPLE_RATE, HOP_LENGTH, N_MELS, FMIN, FMAX, CLIP_SAMPLES, N_FRAMES,
)

MYBAD_WAV_ROOT = "/Volumes/Evo/mybad0"

_SEG_SUFFIX = re.compile(r"_\d+$")


def group_id(filename):
    """Source recording for a clip: strip the trailing _<segment> index.

    xc216946_2.wav        -> xc216946
    ff-64486_1.wav        -> ff-64486
    esc-1-100032-A-0_1.wav-> esc-1-100032-A-0
    23_12301_1.wav        -> 23_12301
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _SEG_SUFFIX.sub("", stem)


def origin(filename):
    """Which upstream corpus a clip came from, inferred from its name."""
    g = group_id(filename)
    if g.startswith("ff-"):
        return "freefield1010"     # overlaps our DCASE training data
    if g.startswith("esc-"):
        return "esc50"
    if g.startswith("xc"):
        return "xenocanto"
    return "other"


def list_clips(root=MYBAD_WAV_ROOT):
    clips = []
    for cls, label in (("positive", 1), ("negative", 0)):
        for path in sorted(glob.glob(os.path.join(root, cls, "*.wav"))):
            clips.append({"path": path, "label": label,
                          "group": group_id(path), "origin": origin(path)})
    return clips


def grouped_split(groups, labels, fractions=(0.70, 0.15, 0.15), seed=42):
    """Split so that every clip of a source recording lands in exactly one split.

    Groups are bucketed by their majority label first so the class balance survives;
    within a class, whole groups are shuffled and dealt out by target proportion.
    Returns (train_idx, val_idx, test_idx) over the ORIGINAL clip order.
    """
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)

    uniq = np.unique(groups)
    # a group's label: clips of one recording share a label in MyBAD, but take the
    # majority rather than assume it
    g_label = {}
    for g in uniq:
        m = groups == g
        g_label[g] = int(round(labels[m].mean()))

    out = [[], [], []]
    for cls in (0, 1):
        gs = np.array([g for g in uniq if g_label[g] == cls])
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(n * fractions[0]))
        n_va = int(round(n * fractions[1]))
        chunks = [gs[:n_tr], gs[n_tr:n_tr + n_va], gs[n_tr + n_va:]]
        for slot, chunk in enumerate(chunks):
            sel = np.isin(groups, chunk)
            out[slot].append(np.where(sel)[0])

    idx = [np.concatenate(o) for o in out]
    for a in idx:
        rng.shuffle(a)
    return tuple(idx)


# ------------------------------------------------------------------ features

_FB = {}


def _fb(n_mels, n_fft):
    key = (n_mels, n_fft)
    if key not in _FB:
        _FB[key] = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_mels,
                                       fmin=FMIN, fmax=FMAX)
    return _FB[key]


def power_mel(path, n_mels=N_MELS, n_fft=1024, n_frames=N_FRAMES):
    """Our frontend: 16 kHz, center=False, fmin/fmax as config, raw POWER mel.

    dB-vs-log1p is applied at train time, not here, so one cache serves both arms of
    the frontend ablation (same split used for the DCASE cache).
    """
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = np.pad(y, (0, CLIP_SAMPLES - len(y))) if len(y) < CLIP_SAMPLES \
        else y[:CLIP_SAMPLES]
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH, center=False)) ** 2
    mel = (_fb(n_mels, n_fft) @ S).T
    if mel.shape[0] != n_frames:          # n_fft=512 yields 186 frames; crop to match
        out = np.zeros((n_frames, n_mels), dtype=np.float32)
        k = min(mel.shape[0], n_frames)
        out[:k] = mel[:k]
        mel = out
    return mel.astype(np.float32)


def _worker(a):
    i, path, n_mels, n_fft = a
    try:
        return i, power_mel(path, n_mels, n_fft), None
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"


def cache_paths(n_mels=N_MELS, n_fft=1024):
    sfx = f"_m{n_mels}_fft{n_fft}"
    return (os.path.join(CACHE_BASE, f"mybad_mel{sfx}.npy"),
            os.path.join(CACHE_BASE, f"mybad_meta{sfx}.npz"))


def build(n_mels=N_MELS, n_fft=1024, workers=8, force=False):
    os.makedirs(CACHE_BASE, exist_ok=True)
    mel_path, meta_path = cache_paths(n_mels, n_fft)
    clips = list_clips()
    n = len(clips)

    if os.path.exists(mel_path) and os.path.exists(meta_path) and not force:
        e = np.load(mel_path, mmap_mode="r")
        if e.shape == (n, N_FRAMES, n_mels):
            print(f"cache present {e.shape}, skipping (use --force)")
            return
    print(f"building MyBAD cache: {n} clips, n_mels={n_mels}, n_fft={n_fft} -> {mel_path}")

    mm = np.lib.format.open_memmap(mel_path, mode="w+", dtype=np.float32,
                                   shape=(n, N_FRAMES, n_mels))
    fails, t0 = [], time.time()
    with Pool(workers) as pool:
        for done, (i, mel, err) in enumerate(pool.imap_unordered(
                _worker, [(i, c["path"], n_mels, n_fft) for i, c in enumerate(clips)],
                chunksize=32), start=1):
            if err is None:
                mm[i] = mel
            else:
                fails.append((clips[i]["path"], err))
            if done % 5000 == 0 or done == n:
                r = done / (time.time() - t0)
                print(f"  {done}/{n} ({r:.0f}/s, ETA {(n-done)/r/60:.1f} min)", flush=True)
    mm.flush(); del mm

    np.savez(meta_path,
             labels=np.array([c["label"] for c in clips], dtype=np.int8),
             groups=np.array([c["group"] for c in clips]),
             origins=np.array([c["origin"] for c in clips]),
             paths=np.array([c["path"] for c in clips]))
    print(f"done in {(time.time()-t0)/60:.1f} min, "
          f"{os.path.getsize(mel_path)/1024**3:.2f} GB, {len(fails)} failures")
    for p, e in fails[:5]:
        print("   FAILED", p, e)


def load(n_mels=N_MELS, n_fft=1024, mmap=True):
    mel_path, meta_path = cache_paths(n_mels, n_fft)
    mel = np.load(mel_path, mmap_mode="r" if mmap else None)
    m = np.load(meta_path, allow_pickle=False)
    return mel, m["labels"], m["groups"], m["origins"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mels", type=int, default=N_MELS)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--survey", action="store_true", help="report composition only")
    args = ap.parse_args()

    if args.survey:
        clips = list_clips()
        import collections
        print(f"MyBAD raw wavs: {len(clips)} clips")
        for cls in (1, 0):
            sub = [c for c in clips if c["label"] == cls]
            name = "positive" if cls else "negative"
            print(f"  {name}: {len(sub)} clips, {len({c['group'] for c in sub})} sources")
            for o, k in collections.Counter(c["origin"] for c in sub).most_common():
                print(f"      {o:14s} {k:6d} clips")
        ff = [c for c in clips if c["origin"] == "freefield1010"]
        print(f"  freefield1010-derived (overlaps DCASE training): {len(ff)} clips "
              f"= {100*len(ff)/len(clips):.1f}% of MyBAD")
    else:
        build(args.n_mels, args.n_fft, args.workers, args.force)
