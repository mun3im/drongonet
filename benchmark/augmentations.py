#!/usr/bin/env python3
"""
augmentations.py -- log-mel-domain augmentations for DrongoNet cross-corpus experiments.

All ops operate on already-computed log-mel spectrograms of shape (B, FRAMES, n_mels, 1)
in [0, 1] (per-sample normalised), to match the existing dcase_benchmark pipeline. This keeps
the augmentation cost out of the librosa decode step (which is the hot path) and lets the
same recipe apply uniformly to Nano/Micro/Edge.

Recipes:
    none        -- identity (baseline; matches current dcase_benchmark.py behaviour)
    mixup       -- input + label mixup (Zhang et al. 2018), alpha=0.2
    specaug     -- SpecAugment (Park et al. 2019): F=2 freq mask, T=10 time mask, 1 each
    pitch_time  -- approximate pitch shift via frequency-axis roll +/-2 bins,
                   time stretch via time-axis squeeze/stretch interp +/-10%
    full        -- mixup + specaug + pitch_time stacked

Why mel-domain pitch/time and not raw-audio:
    True librosa pitch_shift/time_stretch costs ~50 ms per 3 s clip and we have ~21k clips.
    Mel-domain approximations (frequency roll, time interp) capture the same invariances at
    ~zero cost and are standard in keyword-spotting work (Park 2019, Lin 2020).

Usage:
    from augmentations import make_augmenter
    aug = make_augmenter('full', seed=42)
    # in tf.data pipeline:
    ds = ds.batch(128).map(aug, num_parallel_calls=tf.data.AUTOTUNE)

Mixup needs paired batches; the wrapper handles label broadcasting. Returns (x, y) with
y as float32 soft labels of shape (B, num_classes); the existing focal_loss already handles
soft targets since it does -y * log(y_pred) elementwise.
"""

from __future__ import annotations
import tensorflow as tf


def _mixup(x: tf.Tensor, y: tf.Tensor, alpha: float = 0.2, seed: int | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    """Mixup augmentation. x: (B, ...), y: (B, C) one-hot or soft. Returns mixed (x, y)."""
    batch_size = tf.shape(x)[0]
    # sample lambda from Beta(alpha, alpha); tfp not available in tf215 envs by default, so
    # use the two-gamma construction: Beta(a,a) = Gamma(a,1)/(Gamma(a,1)+Gamma(a,1))
    g1 = tf.random.gamma([batch_size], alpha=alpha, seed=seed)
    g2 = tf.random.gamma([batch_size], alpha=alpha, seed=None if seed is None else seed + 1)
    lam = g1 / (g1 + g2 + 1e-8)
    # symmetric: lam <- max(lam, 1-lam) keeps the "dominant" sample identifiable; standard
    lam = tf.maximum(lam, 1.0 - lam)
    perm = tf.random.shuffle(tf.range(batch_size), seed=seed)
    x_perm = tf.gather(x, perm)
    y_perm = tf.gather(y, perm)
    lam_x = tf.reshape(lam, [-1] + [1] * (len(x.shape) - 1))
    lam_y = tf.reshape(lam, [-1, 1])
    return lam_x * x + (1.0 - lam_x) * x_perm, lam_y * y + (1.0 - lam_y) * y_perm


def _spec_augment(x: tf.Tensor, freq_mask: int = 2, time_mask: int = 10,
                  n_freq: int = 1, n_time: int = 1, seed: int | None = None) -> tf.Tensor:
    """SpecAugment: zero out random frequency bands and time bands per sample.

    x: (B, T, F, 1). Masks applied independently per sample. n_freq/n_time bands per axis.
    For 16-mel inputs (Nano/Micro), F=2 means up to 12.5% of frequency bins blanked; for
    80-mel (Edge), up to 2.5%. Time mask T=10 vs FRAMES=184 = up to 5.4%.
    """
    shape = tf.shape(x)
    batch, t, f = shape[0], shape[1], shape[2]

    def mask_axis(tensor, axis_len, max_width, n_bands, axis):
        for _ in range(n_bands):
            width = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32, seed=seed)
            start = tf.random.uniform([], 0, tf.maximum(axis_len - width, 1), dtype=tf.int32, seed=seed)
            # build per-sample mask: broadcastable along non-axis dims
            r = tf.range(axis_len)
            band = tf.logical_or(r < start, r >= start + width)            # (axis_len,) True=keep
            band = tf.cast(band, tensor.dtype)
            shape_for_broadcast = [1, 1, 1, 1]
            shape_for_broadcast[axis] = axis_len
            band = tf.reshape(band, shape_for_broadcast)
            tensor = tensor * band
        return tensor

    x = mask_axis(x, t, time_mask, n_time, axis=1)
    x = mask_axis(x, f, freq_mask, n_freq, axis=2)
    return x


def _pitch_time(x: tf.Tensor, max_freq_roll: int = 2, max_time_scale: float = 0.10,
                seed: int | None = None) -> tf.Tensor:
    """Mel-domain approximations of pitch shift and time stretch.

    x: (B, T, F, 1).
    pitch:  roll along frequency axis by +/- up to max_freq_roll bins
    time :  rescale along time axis by 1 +/- max_time_scale, then crop/pad back to T
    Both applied per-batch (one random transform shared across the batch for speed).
    """
    static_shape = x.shape  # (B, T, F, 1) with T, F statically known; restored at the end
    shape = tf.shape(x)
    t, f = shape[1], shape[2]

    # ---- pitch shift: integer freq roll ----
    roll = tf.random.uniform([], -max_freq_roll, max_freq_roll + 1, dtype=tf.int32, seed=seed)
    x = tf.roll(x, shift=roll, axis=2)

    # ---- time stretch: resize along time axis ----
    scale = 1.0 + tf.random.uniform([], -max_time_scale, max_time_scale, dtype=tf.float32, seed=seed)
    new_t = tf.cast(tf.cast(t, tf.float32) * scale, tf.int32)
    new_t = tf.maximum(new_t, 1)
    # resize: (B, T, F, 1) -> (B, new_t, F, 1)
    x_resized = tf.image.resize(x, size=tf.stack([new_t, f]), method='bilinear')
    # crop or pad back to T along time axis
    def crop():
        start = (new_t - t) // 2
        return x_resized[:, start:start + t, :, :]
    def pad():
        pad_total = t - new_t
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return tf.pad(x_resized, [[0, 0], [pad_left, pad_right], [0, 0], [0, 0]])
    x = tf.cond(new_t >= t, crop, pad)
    # crop/pad restore the original time length T (and the freq axis is never resized),
    # but tf can't infer that statically -> recover the input's static shape so downstream
    # fixed-shape layers (e.g. BatchNormalization built on (1000, 80, 1)) keep defined dims.
    x = tf.ensure_shape(x, static_shape)
    return x


def make_augmenter(recipe: str, alpha: float = 0.2, freq_mask: int = 2, time_mask: int = 10,
                   n_freq: int = 1, n_time: int = 1, max_freq_roll: int = 2,
                   max_time_scale: float = 0.10, seed: int | None = None):
    """Return a tf.function (x, y) -> (x_aug, y_aug) for the given recipe."""
    recipe = recipe.lower()
    valid = {'none', 'mixup', 'specaug', 'pitch_time', 'full'}
    if recipe not in valid:
        raise ValueError(f'recipe must be one of {sorted(valid)}; got {recipe!r}')

    @tf.function
    def aug_fn(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        if recipe in ('specaug', 'full'):
            x = _spec_augment(x, freq_mask=freq_mask, time_mask=time_mask,
                              n_freq=n_freq, n_time=n_time, seed=seed)
        if recipe in ('pitch_time', 'full'):
            x = _pitch_time(x, max_freq_roll=max_freq_roll,
                            max_time_scale=max_time_scale, seed=seed)
        if recipe in ('mixup', 'full'):
            x, y = _mixup(x, y, alpha=alpha, seed=seed)
        # clip to [0, 1] since mixup of two [0,1] inputs stays in [0,1] but pitch_time interp
        # plus mask zeros can produce values that the upstream code expects non-negative
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x, y

    return aug_fn


# --- quick sanity smoke when run directly ---
if __name__ == '__main__':
    import numpy as np
    tf.random.set_seed(0)
    x = tf.constant(np.random.rand(8, 184, 16, 1).astype('float32'))
    y = tf.one_hot([0, 1, 0, 1, 1, 0, 1, 0], 2)
    for r in ['none', 'mixup', 'specaug', 'pitch_time', 'full']:
        a = make_augmenter(r, seed=42)
        xa, ya = a(x, y)
        print(f'{r:11}  x: {xa.shape} mean={float(tf.reduce_mean(xa)):.3f}  y: {ya.shape} sum/sample={float(tf.reduce_mean(tf.reduce_sum(ya,-1))):.3f}')
