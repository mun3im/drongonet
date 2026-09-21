"""
build_cache.py — precompute mel spectrograms for all three DCASE corpora.

Caches the **raw power mel spectrogram** (not dB, not normalized), one fixed-width
(FULL_N_FRAMES, N_MELS) array per clip. The expensive part (decode + resample to
16kHz + STFT + mel filterbank) is done once; the cheap per-window part
(power_to_db(ref=max) + per-window min-max) is applied at train time.

That split matters for faithfulness: `power_to_db` applies `top_db=80` clipping
relative to the window's own max, so a 3s crop normalized inside its own window is
NOT the same as a slice of a 10s-normalized spectrogram. Caching pre-dB keeps the
3s-crop path bit-identical to drongonet-micro's pipeline while still letting the
whole-clip path share one cache.

Output (in CACHE_BASE), per corpus:
    {corpus}_mel.npy      float32 (N, FULL_N_FRAMES, N_MELS)   memmap-friendly
    {corpus}_labels.npy   int8    (N,)
    {corpus}_itemids.npy  unicode (N,)

Usage:
    python build_cache.py                 # all corpora
    python build_cache.py --corpus warblr
    python build_cache.py --workers 8
"""

import os
import argparse
import time
from multiprocessing import Pool

import numpy as np
import librosa

from config import (
    CACHE_BASE, DCASE_CORPORA,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
    FULL_CLIP_SAMPLES, FULL_N_FRAMES,
)
from data import load_corpus

# Built once per worker process rather than per clip, keyed by n_mels.
_MEL_FB = {}


def _mel_filterbank(n_mels: int = N_MELS):
    if n_mels not in _MEL_FB:
        _MEL_FB[n_mels] = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels, fmin=FMIN, fmax=FMAX
        )
    return _MEL_FB[n_mels]


def cache_paths(corpus: str, n_mels: int = N_MELS):
    """16 mels keeps the original unsuffixed filenames; other resolutions get a suffix."""
    sfx = "" if n_mels == N_MELS else str(n_mels)
    return (os.path.join(CACHE_BASE, f"{corpus}_mel{sfx}.npy"),
            os.path.join(CACHE_BASE, f"{corpus}_labels{sfx}.npy"),
            os.path.join(CACHE_BASE, f"{corpus}_itemids{sfx}.npy"))


def _mel_power_for_clip(path: str, n_mels: int = N_MELS) -> np.ndarray:
    """Load one wav -> raw power mel, padded/truncated to (FULL_N_FRAMES, n_mels)."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) < FULL_CLIP_SAMPLES:
        y = np.pad(y, (0, FULL_CLIP_SAMPLES - len(y)))
    elif len(y) > FULL_CLIP_SAMPLES:
        y = y[:FULL_CLIP_SAMPLES]

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False)
    power = np.abs(stft) ** 2
    mel = _mel_filterbank(n_mels) @ power    # (n_mels, frames)
    mel = mel.T                               # (frames, n_mels)

    if mel.shape[0] != FULL_N_FRAMES:         # belt and braces
        out = np.zeros((FULL_N_FRAMES, n_mels), dtype=np.float32)
        n = min(mel.shape[0], FULL_N_FRAMES)
        out[:n] = mel[:n]
        mel = out
    return mel.astype(np.float32)


def _worker(args):
    idx, path, n_mels = args
    try:
        return idx, _mel_power_for_clip(path, n_mels), None
    except Exception as e:
        return idx, None, f"{type(e).__name__}: {e}"


def build_corpus_cache(corpus: str, workers: int = 8, force: bool = False,
                       n_mels: int = N_MELS) -> dict:
    os.makedirs(CACHE_BASE, exist_ok=True)
    mel_path, lab_path, ids_path = cache_paths(corpus, n_mels)

    clips = load_corpus(corpus)
    n = len(clips)

    if os.path.exists(mel_path) and not force:
        existing = np.load(mel_path, mmap_mode="r")
        if existing.shape == (n, FULL_N_FRAMES, n_mels):
            print(f"{corpus}: cache present and correctly shaped {existing.shape}, skipping "
                  f"(use --force to rebuild)")
            return {"corpus": corpus, "n": n, "skipped": True, "failed": 0}
        print(f"{corpus}: cache shape {existing.shape} != expected "
              f"{(n, FULL_N_FRAMES, n_mels)}, rebuilding")

    print(f"{corpus}: building cache ({n} clips, n_mels={n_mels}, {workers} workers) "
          f"-> {mel_path}")
    mel_mm = np.lib.format.open_memmap(
        mel_path, mode="w+", dtype=np.float32, shape=(n, FULL_N_FRAMES, n_mels)
    )

    failures = []
    t0 = time.time()
    with Pool(workers) as pool:
        for done, (idx, mel, err) in enumerate(
            pool.imap_unordered(_worker, [(i, c.path, n_mels) for i, c in enumerate(clips)],
                                chunksize=16), start=1):
            if err is None:
                mel_mm[idx] = mel
            else:
                failures.append((clips[idx].itemid, err))
            if done % 2000 == 0 or done == n:
                rate = done / (time.time() - t0)
                eta = (n - done) / rate if rate else 0
                print(f"  {corpus}: {done}/{n} ({rate:.0f} clips/s, ETA {eta/60:.1f} min)",
                      flush=True)

    mel_mm.flush()
    del mel_mm

    np.save(lab_path, np.array([c.label for c in clips], dtype=np.int8))
    np.save(ids_path, np.array([c.itemid for c in clips]))

    size_gb = os.path.getsize(mel_path) / 1024 ** 3
    print(f"{corpus}: done in {(time.time()-t0)/60:.1f} min, {size_gb:.2f} GB, "
          f"{len(failures)} failures")
    for itemid, err in failures[:10]:
        print(f"    FAILED {itemid}: {err}")

    return {"corpus": corpus, "n": n, "skipped": False, "failed": len(failures)}


def load_cache(corpus: str, mmap: bool = True, n_mels: int = N_MELS):
    """Returns (mel_power, labels, itemids). mel_power is (N, FULL_N_FRAMES, n_mels)."""
    mel_path, lab_path, ids_path = cache_paths(corpus, n_mels)
    mode = "r" if mmap else None
    return np.load(mel_path, mmap_mode=mode), np.load(lab_path), np.load(ids_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, choices=list(DCASE_CORPORA),
                    help="build just one corpus (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    ap.add_argument("--n-mels", type=int, default=N_MELS,
                    help="mel resolution; 80 is needed for the bulbul/sparrow baselines")
    args = ap.parse_args()

    targets = [args.corpus] if args.corpus else list(DCASE_CORPORA)
    results = [build_corpus_cache(c, workers=args.workers, force=args.force,
                                  n_mels=args.n_mels) for c in targets]

    print("\n=== summary ===")
    for r in results:
        state = "skipped" if r["skipped"] else f"{r['failed']} failures"
        print(f"  {r['corpus']:16s} n={r['n']:6d}  {state}")
