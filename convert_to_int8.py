#!/usr/bin/env python3
"""
Convert SEABADNet-Micro (6a) and SEABADNet-Edge (6b) to INT8 TFLite.

Micro has a custom FrequencyEmphasis layer that blocks INT8 quantization.
We fold it into a constant Lambda before converting.

Usage:
    conda run -n tf215_gpu python3 convert_to_int8.py
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
SEABAD_DIR   = Path('/Volumes/Evo/seabad')
MICRO_DIR    = Path('results/6a_micro_final_fft1024_m16_s42')
EDGE_DIR     = Path('results/6b_edge_final_fft1024_m64_s42')
REPR_SAMPLES = 500

# ── custom layer (needed to load the .keras file) ──────────────────────────
class FrequencyEmphasis(tf.keras.layers.Layer):
    def __init__(self, freq_bins=16, init_center=0.4, init_width=0.2, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins

    def build(self, input_shape):
        self.freq_weights = self.add_weight(
            name='frequency_weights',
            shape=(1, 1, self.freq_bins, 1),
            initializer=tf.constant_initializer(1.0),
            trainable=True, dtype=tf.float32)
        self.scale = self.add_weight(
            name='scale', shape=(1,),
            initializer=tf.constant_initializer(3.0),
            trainable=True, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, training=None):
        weight_map = tf.math.sigmoid(self.freq_weights * self.scale)
        return inputs * weight_map

    def get_config(self):
        config = super().get_config()
        config.update({'freq_bins': self.freq_bins})
        return config


# ── spectrogram helper ──────────────────────────────────────────────────────
def load_spec(wav_path, n_mels, n_fft=1024, hop=256, sr=16000,
              target_len=48000, freq_min=100.0, freq_max=8000.0):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop,
        fmin=freq_min, fmax=freq_max, center=False)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalise to [0, 1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.T.astype(np.float32)   # (time, freq)


def build_repr_dataset(n_mels, n_samples=500):
    """Collect representative samples from SEABAD for calibration."""
    wavs = []
    for split in ('positive', 'negative'):
        d = SEABAD_DIR / split
        wavs += sorted(d.glob('*.wav'))
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(wavs), size=min(n_samples, len(wavs)), replace=False)

    specs = []
    for idx in chosen:
        try:
            s = load_spec(wavs[idx], n_mels=n_mels)
            specs.append(s[..., np.newaxis])   # (T, F, 1)
        except Exception as e:
            print(f"  skip {wavs[idx].name}: {e}")
    print(f"  Calibration samples: {len(specs)}")
    return np.stack(specs, axis=0)   # (N, T, F, 1)


# ── fold FrequencyEmphasis into a constant Lambda ──────────────────────────
def fold_freq_emphasis(model):
    """
    Replace the FrequencyEmphasis layer with a fixed constant multiply.
    Returns a new functional model with identical behaviour but no custom ops.
    """
    fe_layer = None
    for layer in model.layers:
        if isinstance(layer, FrequencyEmphasis):
            fe_layer = layer
            break

    if fe_layer is None:
        print("  No FrequencyEmphasis layer found — model returned as-is.")
        return model

    # Compute constant weight map from trained weights
    fw = fe_layer.freq_weights.numpy()       # (1, 1, freq_bins, 1)
    sc = fe_layer.scale.numpy()              # (1,)
    weight_map = (1.0 / (1.0 + np.exp(-fw * sc))).astype(np.float32)
    print(f"  FrequencyEmphasis weight_map shape: {weight_map.shape}, "
          f"range [{weight_map.min():.3f}, {weight_map.max():.3f}]")

    # Build new model substituting FrequencyEmphasis with a Lambda
    inp = tf.keras.layers.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[1:]:   # skip InputLayer
        if isinstance(layer, FrequencyEmphasis):
            const = tf.constant(weight_map, dtype=tf.float32)
            x = tf.keras.layers.Lambda(
                lambda t, c=const: t * c,
                name='freq_emphasis_folded')(x)
        else:
            x = layer(x)

    new_model = tf.keras.Model(inputs=inp, outputs=x,
                               name=model.name + '_folded')

    # Copy weights for all non-FrequencyEmphasis layers
    for new_layer, old_layer in zip(new_model.layers, model.layers):
        if isinstance(old_layer, FrequencyEmphasis):
            continue
        if old_layer.get_weights():
            new_layer.set_weights(old_layer.get_weights())

    return new_model


# ── INT8 conversion ─────────────────────────────────────────────────────────
def convert_int8(model, calib_data, output_path):
    def representative_dataset():
        for i in range(len(calib_data)):
            yield [calib_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  Saved: {output_path}  ({size_kb:.2f} KB)")
    return size_kb


# ── verify dtype ────────────────────────────────────────────────────────────
def verify_dtype(tflite_path):
    interp = tf.lite.Interpreter(str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  Input  dtype={inp['dtype'].__name__}, quant={inp['quantization']}")
    print(f"  Output dtype={out['dtype'].__name__}, quant={out['quantization']}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).parent

    # ── Micro (6a) ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SEABADNet-Micro (6a)  n_mels=16")
    print("=" * 60)
    micro_dir = script_dir / MICRO_DIR

    print("Loading model …")
    model_micro = tf.keras.models.load_model(
        str(micro_dir / 'best_model.keras'),
        custom_objects={'FrequencyEmphasis': FrequencyEmphasis},
        compile=False)
    model_micro.summary(print_fn=lambda s: None)   # silent

    print("Folding FrequencyEmphasis …")
    model_micro_folded = fold_freq_emphasis(model_micro)

    print("Building calibration dataset …")
    calib_micro = build_repr_dataset(n_mels=16, n_samples=REPR_SAMPLES)

    print("Converting to INT8 …")
    convert_int8(model_micro_folded, calib_micro,
                 micro_dir / 'model_int8.tflite')
    verify_dtype(micro_dir / 'model_int8.tflite')

    # ── Edge (6b) ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SEABADNet-Edge (6b)  n_mels=64")
    print("=" * 60)
    edge_dir = script_dir / EDGE_DIR

    # Edge already has model_int8.tflite — re-convert from best_model.keras
    # to ensure full INT8 (input+output) with current TF version
    print("Loading model …")
    model_edge = tf.keras.models.load_model(str(edge_dir / 'best_model.keras'), compile=False)

    print("Building calibration dataset …")
    calib_edge = build_repr_dataset(n_mels=64, n_samples=REPR_SAMPLES)

    print("Converting to INT8 …")
    convert_int8(model_edge, calib_edge,
                 edge_dir / 'model_int8_full.tflite')
    verify_dtype(edge_dir / 'model_int8_full.tflite')

    print()
    print("Done.")
    print(f"  Micro INT8: {MICRO_DIR}/model_int8.tflite")
    print(f"  Edge  INT8: {EDGE_DIR}/model_int8_full.tflite")
    print()
    print("Note: the decision threshold (τ) is a post-inference setting,")
    print("not part of the TFLite model. Apply it after dequantising the")
    print("output score:  predicted_positive = (score > τ)")
    print("Paper optimal: τ* = 0.264 (95% recall, 68% precision).")


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()
