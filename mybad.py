"""
mybad.py — loader for MyBAD v0.8.0 (Malaysian Bird Activity Detection).

Zenodo 10.5281/zenodo.17791820: 28,000 positive + 28,000 negative 3-second clips at
16 kHz, distributed as precomputed mel-spectrogram .npy files in five resolutions
(16/32/48/64/80 x 184 frames). The shipped metadata CSV is header-only in this
release, so labels come from the archive's directory layout.

The 16x184 variant matches our own geometry exactly (3s at 16 kHz, n_fft=1024,
hop=256, 16 mels -> 184 frames), which is why it can be fed to DCASE-trained models
at all.

IMPORTANT: these spectrograms were computed by someone else's pipeline. Whether they
are raw power, log power, or already normalized is NOT documented in the record, and
feeding a differently-scaled array into our models would produce meaningless numbers
rather than an error. `inspect()` characterizes the arrays and `describe_scaling()`
reports which convention they match, so the decision is made on evidence.
"""

import os
import glob
import zipfile

import numpy as np

MYBAD_DIR = "/Volumes/Evo/datasets/mybad"
MYBAD_ZIP_16 = os.path.join(MYBAD_DIR, "mybad-spectro-16x184.zip")
MYBAD_EXTRACT = os.path.join(MYBAD_DIR, "spectro-16x184", "mybad-spectro-16x184")

POSITIVE_HINTS = ("positive", "pos", "bird")
NEGATIVE_HINTS = ("negative", "neg", "nobird")


def extract(zip_path=MYBAD_ZIP_16, dest=MYBAD_EXTRACT, force=False):
    if os.path.isdir(dest) and not force and os.listdir(dest):
        print(f"already extracted: {dest}")
        return dest
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        print(f"{os.path.basename(zip_path)}: {len(names)} entries; extracting -> {dest}")
        z.extractall(dest)
    return dest


def list_archive(zip_path=MYBAD_ZIP_16, limit=20):
    """Peek inside without extracting — shows the layout that labels depend on."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    dirs = sorted({os.path.dirname(n) for n in names})
    return names[:limit], dirs, len(names)


def _label_from_path(path):
    low = path.lower()
    parts = low.replace("\\", "/").split("/")
    for p in parts:
        if any(h in p for h in NEGATIVE_HINTS):
            return 0
        if any(h in p for h in POSITIVE_HINTS):
            return 1
    return None


def _is_real_npy(path):
    """Reject macOS AppleDouble resource forks. The archive ships a __MACOSX tree
    whose members are named ._<original>.npy — they are metadata, not arrays, and a
    plain recursive glob happily collects all 28k of them."""
    base = os.path.basename(path)
    return not base.startswith("._") and "__MACOSX" not in path


def find_files(root=MYBAD_EXTRACT):
    files = sorted(f for f in glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
                   if _is_real_npy(f))
    labeled = [(f, _label_from_path(os.path.relpath(f, root))) for f in files]
    unlabeled = [f for f, l in labeled if l is None]
    labeled = [(f, l) for f, l in labeled if l is not None]
    return labeled, unlabeled


def inspect(root=MYBAD_EXTRACT, n=200, seed=0):
    """Characterize the shipped arrays: shape, dtype, and value distribution."""
    labeled, unlabeled = find_files(root)
    if not labeled:
        raise RuntimeError(f"no labeled .npy under {root}; layout may differ, "
                           f"check list_archive()")
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(labeled), size=min(n, len(labeled)), replace=False)
    arrs = [np.load(labeled[i][0]) for i in pick]
    a = np.stack(arrs) if len({x.shape for x in arrs}) == 1 else None

    info = {
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "n_positive": sum(l for _, l in labeled),
        "shapes": sorted({x.shape for x in arrs}),
        "dtype": str(arrs[0].dtype),
        "min": float(min(x.min() for x in arrs)),
        "max": float(max(x.max() for x in arrs)),
        "mean": float(np.mean([x.mean() for x in arrs])),
        "any_negative_values": bool(any((x < 0).any() for x in arrs)),
        "median": float(np.median(np.concatenate([x.ravel() for x in arrs]))),
        "per_clip_min_is_0": None,
        "per_clip_max_is_1": None,
    }
    if a is not None:
        pmin = a.min(axis=(1, 2))
        pmax = a.max(axis=(1, 2))
        info["per_clip_min_is_0"] = bool(np.allclose(pmin, 0, atol=1e-5))
        info["per_clip_max_is_1"] = bool(np.allclose(pmax, 1, atol=1e-5))
    return info, labeled


def describe_scaling(info):
    """Say which convention the arrays match, so the caller knows what to apply.

    The [0,1] range alone is NOT enough to decide, and getting this wrong produces a
    plausible AUC rather than an error. MyBAD v0.8.0 is per-clip min-max in [0,1] but
    on a LINEAR amplitude scale, whose signature is a median near zero (measured:
    p50=0.004, mean=0.068). Our own features are min-max of *dB*, which spreads out
    (p50=0.34, mean=0.38). Feeding MyBAD in as-shipped would present the models with a
    distribution nothing like their training data.
    """
    if info["per_clip_min_is_0"] and info["per_clip_max_is_1"]:
        median = info.get("median")
        if median is not None and median < 0.05:
            return ("linear_minmax",
                    "per-clip min-max in [0,1] but LINEAR scale (median near 0) — "
                    "square to power, then power_to_db(ref=max) and per-clip min-max "
                    "to match our dB convention. NOT usable as shipped")
        return ("db_minmax",
                "per-clip min-max in [0,1] with a spread-out median — already matches "
                "our dB convention; feed straight in")
    if info["any_negative_values"] and info["min"] < -1:
        return ("log_db",
                "log/dB scale with negatives — needs per-clip min-max only, NOT "
                "power_to_db again")
    if info["min"] >= 0 and info["max"] > 2:
        return ("raw_power",
                "non-negative with range > 2 — looks like raw power mel; apply our "
                "full normalize_np (power_to_db ref=max then per-clip min-max)")
    return ("unknown",
            "does not match a known convention; inspect before trusting any AUC")


def to_time_major(a):
    """MyBAD ships (n_mels, 184) = (freq, time); our models take (time, freq).

    Transposing is mandatory and silent if forgotten: a (16,184) array fed to a
    (184,16) input is a shape error only when n_mels != 184, which it always is here,
    but a batch of them reshaped carelessly would not be. Detects by matching the
    184-frame axis.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        return a.T if a.shape[0] < a.shape[1] else a
    if a.ndim == 3:
        return np.transpose(a, (0, 2, 1)) if a.shape[1] < a.shape[2] else a
    raise ValueError(f"expected 2 or 3 dims, got {a.shape}")


def load_all(root=MYBAD_EXTRACT, labeled=None, dtype=np.float32):
    """Load the whole set as (N, 184, n_mels) TIME-MAJOR, plus labels."""
    if labeled is None:
        labeled, _ = find_files(root)
    first = to_time_major(np.load(labeled[0][0]))
    out = np.empty((len(labeled),) + first.shape, dtype=dtype)
    labels = np.empty(len(labeled), dtype=np.int8)
    for i, (path, lab) in enumerate(labeled):
        out[i] = to_time_major(np.load(path))
        labels[i] = lab
    return out, labels


if __name__ == "__main__":
    if os.path.exists(MYBAD_ZIP_16):
        head, dirs, total = list_archive()
        print(f"archive entries: {total}")
        print(f"directories: {dirs[:10]}")
        print("first entries:")
        for n in head[:10]:
            print("  ", n)
    else:
        print(f"not downloaded yet: {MYBAD_ZIP_16}")
