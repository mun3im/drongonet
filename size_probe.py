"""
size_probe.py — measure INT8 .tflite size for architecture variants WITHOUT training.

Phase C established that nothing in the receptive-field x width grid clears the 10 KB
soft target: even a 795-parameter model converts to 10.78 KB, because size is dominated
by per-layer FlatBuffer overhead rather than weights. Training a config just to learn
its size wastes ~8 minutes per data point, so this probes the size space directly with
random weights and a synthetic calibration set.

Size is independent of the trained values, so these numbers are exact for a given
architecture -- only the AUC would need training.

Usage:
    python size_probe.py
"""

import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from config import (
    N_MELS, FULL_N_FRAMES, SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES,
)
from models import build_sparrownet


def randomize_all_weights(model, seed=0):
    """Replace every weight with a distinct random value.

    This is required for the size numbers to mean anything, and the reason is
    non-obvious. A freshly built model has many all-zero buffers (Keras initializes
    conv biases to zeros, BN beta/moving_mean to zeros), and FlatBuffers *deduplicates
    identical buffers* -- so those collapse into one, and the blob comes out ~1 KB
    smaller than the same architecture trained. Measured on st4_w8-8: random init
    9.82 KB vs trained 10.78 KB, with byte-identical structure (39 tensors, same op
    counts), which is what ruled out any structural explanation.

    Variances are kept positive so BN folding stays well defined.
    """
    rng = np.random.RandomState(seed)
    for layer in model.layers:
        w = layer.get_weights()
        if not w:
            continue
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = w
            layer.set_weights([
                rng.uniform(0.5, 1.5, gamma.shape).astype(gamma.dtype),
                rng.uniform(-0.5, 0.5, beta.shape).astype(beta.dtype),
                rng.uniform(-0.5, 0.5, mean.shape).astype(mean.dtype),
                rng.uniform(0.5, 1.5, var.shape).astype(var.dtype),
            ])
        else:
            layer.set_weights([
                rng.uniform(-0.5, 0.5, a.shape).astype(a.dtype) for a in w
            ])
    return model


def convert_size(model, n_calib=8):
    """INT8-convert with synthetic calibration; returns blob size in bytes."""
    randomize_all_weights(model)
    shape = model.input_shape[1:]

    def rep():
        for _ in range(n_calib):
            yield [np.random.rand(1, *shape).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return len(conv.convert())


def verdict(size):
    if size <= SIZE_SOFT_LIMIT_BYTES:
        return "OK"
    return "OVER_SOFT" if size <= SIZE_HARD_LIMIT_BYTES else "OVER_HARD"


if __name__ == "__main__":
    print(f"budget: soft {SIZE_SOFT_LIMIT_BYTES/1024:.0f} KB / "
          f"hard {SIZE_HARD_LIMIT_BYTES/1024:.0f} KB\n")
    print(f"{'config':40s} {'params':>7s} {'KB':>7s} {'verdict':>10s}")
    print("-" * 68)

    variants = []
    for stages in (2, 3, 4, 5):
        for width in ((8, 8), (8, 12), (8, 16), (6, 8)):
            for bn in (True, False):
                variants.append((stages, width, bn))

    rows = []
    for stages, width, bn in variants:
        tf.keras.backend.clear_session()
        m = build_sparrownet(input_shape=(FULL_N_FRAMES, N_MELS, 1),
                             n_time_stages=stages, width=width, batch_norm=bn)
        size = convert_size(m)
        name = f"st{stages}_w{width[0]}-{width[1]}_{'bn' if bn else 'nobn'}"
        rows.append((name, m.count_params(), size, m.receptive_field_frames))
        print(f"{name:40s} {m.count_params():7d} {size/1024:7.2f} {verdict(size):>10s}")

    print("\nunder the soft target:")
    ok = [r for r in rows if r[2] <= SIZE_SOFT_LIMIT_BYTES]
    if not ok:
        print("  NONE -- the 10 KB target is unreachable for this topology family")
        best = min(rows, key=lambda r: r[2])
        print(f"  smallest: {best[0]} at {best[2]/1024:.2f} KB ({best[1]} params)")
    else:
        for name, params, size, rf in sorted(ok, key=lambda r: r[2]):
            print(f"  {name:38s} {size/1024:6.2f} KB  {params:5d} params  RF={rf}f")
