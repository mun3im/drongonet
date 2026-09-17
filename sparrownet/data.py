"""
data.py — DCASE 2018 Bird Audio Detection loader for SparrowNet.

Corpora: birdvox (BirdVox-DCASE-20k), freefield1010, warblr. Each is a set of
10s / 44.1kHz mono wavs with a single clip-level weak label (hasbird: 0/1) —
the standard "MI assumption": positive iff at least one bird event occurs
somewhere in the clip.

Two spectrogram modes, matching two different uses:
  - "crop"      fixed 3s / 184-frame window (identical to drongonet-micro's
                frontend) — for the drongonet-micro-on-DCASE baseline check.
  - "full_clip" whole 10s clip as one variable-length mel spectrogram — for
                SparrowNet's local-receptive-field + global-max-pool design,
                which needs no fixed crop (see PLAN in README.md).

Both share the same per-clip preprocessing (resample -> mel -> dB -> per-clip
min-max normalize) so results are comparable across scripts.
"""

import os
import csv
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import librosa

from config import (
    DATASET_PATH, DCASE_CORPORA,
    SAMPLE_RATE, CLIP_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
)


@dataclass
class Clip:
    path: str
    label: int          # 0 = no bird, 1 = bird
    corpus: str          # "birdvox" | "freefield1010" | "warblr"
    itemid: str


def load_corpus(corpus: str, dataset_path: str = DATASET_PATH) -> List[Clip]:
    """Read one corpus's CSV + wav dir into a list of Clip records."""
    if corpus not in DCASE_CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}', expected one of {list(DCASE_CORPORA)}")
    spec = DCASE_CORPORA[corpus]
    csv_path = os.path.join(dataset_path, spec["csv"])
    wav_dir = os.path.join(dataset_path, spec["wav_dir"])

    clips = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            itemid = row["itemid"]
            label = int(row["hasbird"])
            wav_path = os.path.join(wav_dir, f"{itemid}.wav")
            clips.append(Clip(path=wav_path, label=label, corpus=corpus, itemid=itemid))
    return clips


def load_all_corpora(dataset_path: str = DATASET_PATH) -> Dict[str, List[Clip]]:
    return {name: load_corpus(name, dataset_path) for name in DCASE_CORPORA}


def verify_clips(clips: List[Clip], sample: int = 0) -> Tuple[int, int]:
    """Check wav files referenced by the CSV actually exist. Returns (found, missing).
    Pass sample>0 to only check a random subset (faster sanity check on large corpora)."""
    check = clips if sample <= 0 else random.sample(clips, min(sample, len(clips)))
    missing = sum(1 for c in check if not os.path.exists(c.path))
    return len(check) - missing, missing


# --------------------------------------------------------------------------
# Cross-corpus splits (standard DCASE protocol: train on two corpora with
# k-fold CV, test cross-corpus on the held-out third).
# --------------------------------------------------------------------------

def cross_corpus_split(held_out: str, dataset_path: str = DATASET_PATH) -> Tuple[List[Clip], List[Clip]]:
    """held_out corpus becomes the test set; the other two are pooled for train/CV."""
    all_corpora = load_all_corpora(dataset_path)
    if held_out not in all_corpora:
        raise ValueError(f"held_out must be one of {list(DCASE_CORPORA)}")
    test_clips = all_corpora[held_out]
    train_clips = [c for name, clips in all_corpora.items() if name != held_out for c in clips]
    return train_clips, test_clips


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y


def _mel_db_normalize(y: np.ndarray) -> np.ndarray:
    """audio -> log-mel -> per-clip min-max normalize to [0,1]. Shared by both modes."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=False,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T  # (time, freq)
    lo, hi = mel_db.min(), mel_db.max()
    if hi > lo:
        mel_db = (mel_db - lo) / (hi - lo)
    else:
        mel_db = np.zeros_like(mel_db)
    return mel_db[..., np.newaxis].astype(np.float32)  # (time, n_mels, 1)


def compute_mel_crop(path: str, crop_samples: int = CLIP_SAMPLES, rng: random.Random = None) -> np.ndarray:
    """Fixed-length crop mode (drongonet-micro compatible): pad/truncate to
    crop_samples before the mel transform, giving a constant (184, N_MELS, 1)
    output regardless of source clip length. Random crop offset if the source
    is longer (rng given) else center crop; zero-pad if shorter."""
    y = _load_audio(path)
    if len(y) > crop_samples:
        max_start = len(y) - crop_samples
        start = rng.randint(0, max_start) if rng is not None else max_start // 2
        y = y[start:start + crop_samples]
    elif len(y) < crop_samples:
        y = np.pad(y, (0, crop_samples - len(y)))
    return _mel_db_normalize(y)


def compute_mel_full(path: str) -> np.ndarray:
    """Whole-clip mode: no cropping, variable time dimension. Used by SparrowNet's
    fully-convolutional local-window + global-max-pool architecture."""
    y = _load_audio(path)
    return _mel_db_normalize(y)


if __name__ == "__main__":
    # Quick sanity check: counts, class balance, and that a few wavs actually load.
    for name in DCASE_CORPORA:
        clips = load_corpus(name)
        pos = sum(c.label for c in clips)
        found, missing = verify_clips(clips, sample=50)
        print(f"{name:16s} n={len(clips):6d}  positive={pos:6d} ({100*pos/len(clips):.1f}%)  "
              f"sample-check: {found} found / {missing} missing")
        example = clips[0]
        mel_crop = compute_mel_crop(example.path)
        mel_full = compute_mel_full(example.path)
        print(f"  {example.itemid}: crop shape={mel_crop.shape}, full shape={mel_full.shape}")
