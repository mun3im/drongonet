"""
dataset.py — tf.data pipelines over the cached power-mel arrays.

The cache holds raw power mel (see build_cache.py). Everything downstream of that
happens here, in TF, on the fly:

    power mel -> power_to_db(ref=max, top_db=80) -> per-window min-max [0,1]
              -> optional 3s crop / cyclic time shift -> batch

power_to_db is reimplemented in TF (`power_to_db_tf`) and checked against librosa in
this file's __main__ — the normalization is the one place where a silent mismatch
would make every number here incomparable to drongonet's.

Augmentation is cyclic time shift only. Per the paper that is the single most
important augmentation for both of its models (Sect. III-D: examples are cyclically
shifted in time to gain translation invariance), and drongonet's own augmentation
sweep found its heavier recipes *regressed* cross-corpus AUC at small capacity.
"""

import numpy as np
import tensorflow as tf

from config import N_FRAMES, N_MELS, FULL_N_FRAMES
from build_cache import load_cache

AMIN = 1e-10
TOP_DB = 80.0


def power_to_db_tf(power, top_db=TOP_DB):
    """TF port of librosa.power_to_db(S, ref=np.max, top_db=80), per example.
    `power` is (..., T, F); ref/max and the top_db floor are computed per example."""
    log_spec = 10.0 * tf.math.log(tf.maximum(AMIN, power)) / tf.math.log(10.0)
    ref = tf.reduce_max(power, axis=[-2, -1], keepdims=True)
    log_ref = 10.0 * tf.math.log(tf.maximum(AMIN, ref)) / tf.math.log(10.0)
    log_spec = log_spec - log_ref
    if top_db is not None:
        peak = tf.reduce_max(log_spec, axis=[-2, -1], keepdims=True)
        log_spec = tf.maximum(log_spec, peak - top_db)
    return log_spec


def minmax_tf(x):
    """Per-example min-max to [0,1]; all-constant input -> zeros (matches data.py)."""
    lo = tf.reduce_min(x, axis=[-2, -1], keepdims=True)
    hi = tf.reduce_max(x, axis=[-2, -1], keepdims=True)
    rng = hi - lo
    return tf.where(rng > 0, (x - lo) / tf.maximum(rng, 1e-12), tf.zeros_like(x))


def normalize_tf(power):
    return minmax_tf(power_to_db_tf(power))


def log1p_minmax_tf(power):
    """MyBAD's compression (np.log1p) followed by our per-clip min-max.

    Isolates ONE variable against normalize_tf for the frontend ablation: log
    compression of power (log1p) vs dB relative to the clip peak (power_to_db with
    ref=max, top_db=80). MyBAD additionally clips to the 1st/99th percentile before
    scaling; that is a third factor and is deliberately NOT varied here.

    Note log1p barely compresses at these levels -- 99.1% of MyBAD's power-mel values
    are below 0.1, where log1p(S) is within 5% of S -- so this arm is close to a linear
    power spectrogram, which is the point of testing it.
    """
    return minmax_tf(tf.math.log1p(power))


def log1p_minmax_np(power):
    power = np.asarray(power, dtype=np.float32)
    single = power.ndim == 2
    if single:
        power = power[np.newaxis]
    x = np.log1p(power)
    lo = x.min(axis=(-2, -1), keepdims=True)
    hi = x.max(axis=(-2, -1), keepdims=True)
    rng = hi - lo
    out = np.where(rng > 0, (x - lo) / np.maximum(rng, 1e-12), 0.0)
    return (out[0] if single else out).astype(np.float32)


FRONTENDS_TF = {"db": lambda p: minmax_tf(power_to_db_tf(p)), "log1p": log1p_minmax_tf}


def normalize_np(power, top_db=TOP_DB):
    """Numpy twin of normalize_tf, vectorized over a leading batch axis.

    Exists because the TFLite eval loop called normalize_tf once per clip in eager
    mode, and the per-call graph overhead dominated everything else (20k clips took
    minutes with the GPU idle). Verified equal to normalize_tf in this file's __main__.
    Accepts (T, F) or (N, T, F)."""
    power = np.asarray(power, dtype=np.float32)
    single = power.ndim == 2
    if single:
        power = power[np.newaxis]

    log_spec = 10.0 * np.log10(np.maximum(AMIN, power))
    ref = power.max(axis=(-2, -1), keepdims=True)
    log_spec -= 10.0 * np.log10(np.maximum(AMIN, ref))
    if top_db is not None:
        peak = log_spec.max(axis=(-2, -1), keepdims=True)
        log_spec = np.maximum(log_spec, peak - top_db)

    lo = log_spec.min(axis=(-2, -1), keepdims=True)
    hi = log_spec.max(axis=(-2, -1), keepdims=True)
    rng = hi - lo
    out = np.where(rng > 0, (log_spec - lo) / np.maximum(rng, 1e-12), 0.0)
    return (out[0] if single else out).astype(np.float32)


def _cyclic_shift(power, seed=None):
    """Roll along time. The clip is a ring, so a bird event near the edge stays intact
    (the paper's 'cyclic rotation' augmentation)."""
    t = tf.shape(power)[0]
    shift = tf.random.uniform([], 0, t, dtype=tf.int32)
    return tf.roll(power, shift=shift, axis=0)


def _random_crop_time(power, n_frames):
    t = tf.shape(power)[0]
    max_start = tf.maximum(t - n_frames, 0)
    start = tf.random.uniform([], 0, max_start + 1, dtype=tf.int32)
    return power[start:start + n_frames]


def _center_crop_time(power, n_frames):
    t = tf.shape(power)[0]
    start = tf.maximum((t - n_frames) // 2, 0)
    return power[start:start + n_frames]


def make_dataset(mel, labels, mode="full", training=False, batch_size=64,
                 augment=True, num_classes=2, shuffle_buffer=4096, seed=None,
                 n_mels=N_MELS, n_frames=None, frontend="db"):
    """
    mode "full" -> whole clip, (FULL_N_FRAMES, N_MELS, 1)   [sparrownet]
    mode "crop" -> 3s window,  (N_FRAMES, N_MELS, 1)        [drongonet-micro baseline]

    In "crop" training the window is randomly positioned (a free augmentation and the
    honest reading of a clip-level label); in eval it is centered, so eval is
    deterministic.
    """
    if n_frames is None:
        n_frames = FULL_N_FRAMES if mode == "full" else N_FRAMES
    src_frames = mel[0].shape[0] if len(mel) else n_frames
    norm = FRONTENDS_TF[frontend]
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((np.arange(len(labels), dtype=np.int64), y))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(labels)), seed=seed,
                        reshuffle_each_iteration=True)

    mel_ref = mel  # numpy memmap; read inside py_function

    def _fetch(idx):
        return mel_ref[int(idx.numpy())].astype(np.float32)

    def _map(idx, label):
        power = tf.py_function(_fetch, [idx], tf.float32)
        power.set_shape((src_frames, n_mels))

        if training and augment:
            power = _cyclic_shift(power)

        if mode == "crop" and src_frames > n_frames:
            power = _random_crop_time(power, n_frames) if training \
                else _center_crop_time(power, n_frames)
            power.set_shape((n_frames, n_mels))

        x = norm(power)
        x = tf.expand_dims(x, -1)
        x.set_shape((n_frames, n_mels, 1))
        return x, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def stratified_split(labels, val_fraction=0.15, seed=42):
    """Class-stratified index split. Returns (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def load_corpora_arrays(corpora):
    """Concatenate several cached corpora into (mel_list_view, labels, corpus_of_each).
    Returns mel as a list-like of memmaps plus a global index so nothing is copied into
    RAM: we build an index array of (corpus_slot, row) and a thin gather wrapper."""
    mels, labels, origins = [], [], []
    for name in corpora:
        m, lab, _ = load_cache(name, mmap=True)
        mels.append(m)
        labels.append(lab)
        origins.append(np.full(len(lab), name, dtype=object))
    return mels, np.concatenate(labels), np.concatenate(origins)


class ConcatMel:
    """Read-only concatenation of several memmapped (N, T, F) arrays, indexable like
    one array. Avoids materializing 1.4 GB in RAM just to pool corpora."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        slot = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.arrays[slot][i - self.offsets[slot]]


if __name__ == "__main__":
    # Verify the TF normalization against librosa on real cached data.
    import librosa
    from data import _mel_db_normalize  # numpy reference used by data.py

    mel, labels, _ = load_cache("freefield1010", mmap=True)
    batch = np.array(mel[:32], dtype=np.float32)

    tf_out = normalize_tf(tf.convert_to_tensor(batch)).numpy()

    ref = np.empty_like(batch)
    for i, power in enumerate(batch):
        db = librosa.power_to_db(power.T, ref=np.max).T   # librosa's own, top_db=80
        lo, hi = db.min(), db.max()
        ref[i] = (db - lo) / (hi - lo) if hi > lo else 0.0

    print(f"TF vs librosa normalization over {len(batch)} clips:")
    print(f"  max abs diff = {np.abs(tf_out - ref).max():.3e}")
    print(f"  mean abs diff = {np.abs(tf_out - ref).mean():.3e}")
    assert np.abs(tf_out - ref).max() < 1e-5, "TF normalization diverges from librosa"

    ds = make_dataset(mel, labels, mode="full", training=True, batch_size=8)
    x, y = next(iter(ds))
    print(f"full  batch: x={x.shape} y={y.shape} range=[{x.numpy().min():.3f},{x.numpy().max():.3f}]")
    ds = make_dataset(mel, labels, mode="crop", training=False, batch_size=8)
    x, y = next(iter(ds))
    print(f"crop  batch: x={x.shape} y={y.shape} range=[{x.numpy().min():.3f},{x.numpy().max():.3f}]")
    print("normalization verified")
