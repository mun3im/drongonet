#!/usr/bin/env python3
"""
7a_micro_distill.py: DrongoNet-Micro + knowledge distillation (Phase 7)

Trains the EXACT locked 6b Micro architecture (FrequencyEmphasis → Conv2D(6,3×3)
→ MaxPool → Conv2D(12,3×3) → Conv2D(12,1×1) → GAP → Dropout(0.1) → Dense(2);
919 params) with a knowledge-distillation loss:

    L = kd_alpha · L_focal(y, p_s)  +  (1 − kd_alpha) · T² · KL(t_soft ‖ p_s^(1/T))

following the WrenNet recipe (Ciapponi et al., ICASSP 2026: L = 0.6·focal +
0.4·soft-KL, T = 3.0; they distilled from BirdNET — we distill from our own
DrongoNet-Edge, float AUC ≈ 0.999).

Zero inference cost: the exported INT8 TFLite is bit-identical in structure to
6b (same 6.56 KB flash, same 0.10 ms latency) — only the weights differ.

Teacher handling:
  The teacher (Edge) consumes n_mels=80 input; the student consumes n_mels=16.
  Both caches are generated from the same file list with the same split seed,
  so sample i in the m16 cache is the same clip as sample i in the m80 cache.
  This is VERIFIED at runtime by asserting exact equality of the two caches'
  label sequences per split — the run aborts if they do not match.
  Teacher soft labels are precomputed once per (teacher, split) and cached in
  results/kd_teacher_cache/.

  The teacher input layout is inferred from the loaded model's input shape, so
  any 2-class .keras model works (Edge m80, Matchbox m16, 3f m16, ...) via
  --teacher_model + --teacher_cache.

7b variant: pass --specaug to add SpecAugment masking (8a settings) on top of
the repo-standard noise + time-shift augmentation. Output dir stem then becomes
7b_micro_distill_specaug automatically.

Usage:
  conda run -n tf215_gpu python 7a_micro_distill.py --random_seed 42 --use_cache
  conda run -n tf215_gpu python 7a_micro_distill.py --random_seed 42 --use_cache --specaug   # = 7b

Output dir: results/7{a,b}_micro_distill[_specaug]_fft{n_fft}_m{n_mels}_s{seed}_{platform}/
"""

import os
import logging
import time
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve
)
from tqdm import tqdm
import random
import librosa

tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config import DATASET_PATH, RESULTS_BASE, CACHE_BASE

TIME_STEPS = 184  # 3s @ 16kHz, n_fft=1024, hop=256


@dataclass
class TrainingConfig:
    epochs: int = 100
    fraction: float = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    target_sr: int = 16000
    target_length: int = 16000 * 3
    n_mels: int = 16
    n_fft: int = 1024
    hop_length: int = 256
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-5
    early_stopping_patience: int = 15
    random_seed: int = 42
    dataset_path: str = DATASET_PATH
    output_dir: str = f'{RESULTS_BASE}/7a_micro_distill'
    cache_dir: str = f'{CACHE_BASE}_fft1024_m16'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    mel_fmin: float = 100.0
    mel_fmax: float = 8000.0
    # Distillation
    kd_alpha: float = 0.6         # weight on the hard focal term (WrenNet: 0.6)
    kd_temperature: float = 3.0   # softmax temperature (WrenNet: 3.0)
    teacher_model: str = ''       # resolved at runtime if empty
    teacher_cache: str = ''       # resolved from teacher input shape if empty
    use_specaug: bool = False     # --specaug → 7b variant


# Default teacher search order: same-platform Edge s42, then the Dropbox-synced
# Linux one, then the legacy untagged dir.
def default_teacher_candidates():
    tag = 'macos' if platform.system() == 'Darwin' else 'linux'
    return [
        f'results/drongonet_edge_fft1024_m80_s42_{tag}/best_model.keras',
        'results/drongonet_edge_fft1024_m80_s42_linux/best_model.keras',
        'results/drongonet_edge_fft1024_m80_s42/best_model.keras',
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DrongoNet-Micro (locked 6b arch) with knowledge distillation')
    parser.add_argument('--repr_samples', type=int, default=500)
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--force-reprocess', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--n_mels', type=int, default=16)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--kd_alpha', type=float, default=0.6,
                        help='Weight on hard focal loss; soft KD gets 1-alpha (default: 0.6)')
    parser.add_argument('--kd_temperature', type=float, default=3.0,
                        help='Distillation temperature (default: 3.0)')
    parser.add_argument('--teacher_model', type=str, default='',
                        help='Path to teacher .keras model (default: auto-discover Edge s42)')
    parser.add_argument('--teacher_cache', type=str, default='',
                        help='Mel cache dir matching the teacher input '
                             '(default: inferred from teacher input shape)')
    parser.add_argument('--force_teacher_recompute', action='store_true',
                        help='Recompute cached teacher soft labels')
    parser.add_argument('--specaug', action='store_true',
                        help='Add SpecAugment masking → 7b variant (changes output dir stem)')
    parser.add_argument('--output-dir', type=str, default=None)
    return parser.parse_args()


args = parse_args()

if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================================
# STUDENT MODEL — verbatim copy of the locked 6b architecture
# ============================================================================

class FrequencyEmphasis(tf.keras.layers.Layer):
    """
    Learnable frequency weighting (copied verbatim from 6b_micro_final.py).
    Adds freq_bins + 1 parameters.
    """

    def __init__(self, freq_bins=16, init_center=0.4, init_width=0.2, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins
        self.freq_weights = self.add_weight(
            name='frequency_weights',
            shape=(1, 1, freq_bins, 1),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            dtype=tf.float32
        )
        self.scale = self.add_weight(
            name='scale',
            shape=(1,),
            initializer=tf.constant_initializer(3.0),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, training=None):
        weight_map = tf.math.sigmoid(self.freq_weights * self.scale)
        return inputs * weight_map

    def get_config(self):
        config = super().get_config()
        config.update({'freq_bins': self.freq_bins})
        return config


def build_cnn_mel_low_power_optimized(input_shape=(184, 16, 1), num_classes=2):
    """DrongoNet-Micro locked architecture — verbatim copy of 6b_micro_final.py."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = FrequencyEmphasis(freq_bins=input_shape[1], name='frequency_emphasis')(inputs)

    x = tf.keras.layers.Conv2D(
        6, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        12, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Conv2D(
        12, (1, 1),
        padding='same',
        activation='relu',
        name='pointwise_conv'
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="DrongoNet_Micro_Distilled")


def get_optimizer(learning_rate: float):
    system, machine = platform.system(), platform.machine()
    if system == 'Darwin' and machine == 'arm64':
        logger.info("Apple Silicon — legacy Adam")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    logger.info(f"{system} {machine} — AdamW")
    return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


# ============================================================================
# TEACHER — load, verify alignment, precompute soft labels
# ============================================================================

class ChannelFrequencyEmphasis(tf.keras.layers.Layer):
    """Custom-object stub so a 8a Matchbox teacher can be deserialized."""

    def __init__(self, freq_bins=16, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins
        self.freq_weights = self.add_weight(
            name='frequency_weights', shape=(1, 1, 1, freq_bins),
            initializer=tf.constant_initializer(1.0), trainable=True, dtype=tf.float32)
        self.scale = self.add_weight(
            name='scale', shape=(1,),
            initializer=tf.constant_initializer(3.0), trainable=True, dtype=tf.float32)

    def call(self, inputs, training=None):
        return inputs * tf.math.sigmoid(self.freq_weights * self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({'freq_bins': self.freq_bins})
        return config


def resolve_teacher_path(explicit: str) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"--teacher_model not found: {explicit}")
        return explicit
    for cand in default_teacher_candidates():
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "No teacher checkpoint found. Searched:\n  " +
        "\n  ".join(default_teacher_candidates()) +
        "\nTrain 6c_edge_final.py first, or pass --teacher_model explicitly.")


def load_teacher(path: str) -> tf.keras.Model:
    custom = {'FrequencyEmphasis': FrequencyEmphasis,
              'ChannelFrequencyEmphasis': ChannelFrequencyEmphasis}
    model = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    logger.info(f"Teacher loaded: {path}")
    logger.info(f"  Input shape: {model.input_shape}, params: {model.count_params():,}")
    return model


def teacher_cache_dir_for(model: tf.keras.Model, explicit: str, n_fft: int) -> str:
    """Infer the mel cache the teacher needs from its input shape."""
    if explicit:
        return explicit
    shape = model.input_shape  # (None, time, A, B)
    if shape[3] == 1:
        t_mels = shape[2]        # 2D layout (time, n_mels, 1) — 6b/6c family
    else:
        t_mels = shape[3]        # channel layout (time, 1, n_mels) — 8a family
    return f'{CACHE_BASE}_fft{n_fft}_m{t_mels}'


def layout_mels_for(model: tf.keras.Model, mels: np.ndarray) -> np.ndarray:
    """Reshape cached (n, time, n_mels) arrays to the teacher's input layout."""
    shape = model.input_shape
    if shape[3] == 1:
        return mels[..., np.newaxis]          # (n, time, n_mels, 1)
    return mels[:, :, np.newaxis, :]          # (n, time, 1, n_mels)


def compute_teacher_probs(teacher: tf.keras.Model, teacher_cache: str, split: str,
                          student_labels: np.ndarray, cache_key: str,
                          force: bool = False) -> np.ndarray:
    """
    Return teacher P(positive-class softmax vector) for every sample in `split`,
    aligned with the student's m16 cache. Verifies alignment via exact label-
    sequence equality between the two caches. Results cached on disk.
    """
    kd_cache = Path('results/kd_teacher_cache') / cache_key
    kd_cache.mkdir(parents=True, exist_ok=True)
    cache_file = kd_cache / f'{split}.npz'

    if cache_file.exists() and not force:
        d = np.load(cache_file)
        if np.array_equal(d['labels'], student_labels):
            logger.info(f"Teacher probs loaded from cache: {cache_file}")
            return d['probs']
        logger.warning(f"Cached teacher probs stale (label mismatch) — recomputing {split}")

    t_cache_file = Path(teacher_cache) / split / 'mels.npz'
    if not t_cache_file.exists():
        raise FileNotFoundError(
            f"Teacher mel cache not found: {t_cache_file}\n"
            f"Generate it by running the teacher's training script once, or point "
            f"--teacher_cache at an existing cache.")
    d = np.load(t_cache_file)
    t_mels, t_labels = d['mels'], d['labels']

    # HARD alignment check: same clips in the same order in both caches
    if not np.array_equal(t_labels, student_labels):
        raise RuntimeError(
            f"Cache alignment FAILED for split '{split}': the label sequence in "
            f"{teacher_cache} differs from the student cache. The two caches were "
            f"generated with different split seeds or dataset states — regenerate "
            f"one of them so both use the same seed on the same dataset.")

    logger.info(f"Computing teacher probs for {split} ({len(t_mels)} samples)...")
    x = layout_mels_for(teacher, t_mels)
    probs = teacher.predict(x, batch_size=256, verbose=0)  # (n, 2) softmax
    np.savez_compressed(cache_file, probs=probs.astype(np.float32), labels=t_labels)
    logger.info(f"  Saved to {cache_file}")
    return probs.astype(np.float32)


def soften(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-soften teacher softmax probs: softmax(log(p)/T)."""
    z = np.log(np.clip(probs, 1e-7, 1.0)) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ============================================================================
# DISTILLATION TRAINER
# ============================================================================

def focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.5):
    """Repo-standard focal loss (γ=2, α=0.5) on softmax outputs."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred)
    pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    focal_weight = tf.pow(1 - pt, gamma)
    alpha_weight = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)
    return tf.reduce_mean(
        alpha_weight * focal_weight * tf.reduce_sum(cross_entropy, axis=-1, keepdims=True))


class DistillTrainer(tf.keras.Model):
    """
    Wraps the student with a combined focal + soft-KD loss.
    Datasets must yield (x, (y_onehot, teacher_soft)).
    """

    def __init__(self, student, kd_alpha, temperature, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self._metrics_list = []

    def compile(self, optimizer, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self._metrics_list = metrics or []

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _combined_loss(self, y, t_soft, p_s):
        hard = focal_loss_fn(y, p_s)
        # Soften the student the same way as the teacher: softmax(log p / T).
        # The log-softmax shift constant cancels, so this equals softmax(z/T).
        z_s = tf.math.log(tf.clip_by_value(p_s, 1e-7, 1.0)) / self.temperature
        p_sT = tf.nn.softmax(z_s, axis=-1)
        soft = -tf.reduce_mean(
            tf.reduce_sum(t_soft * tf.math.log(tf.clip_by_value(p_sT, 1e-7, 1.0)), axis=-1))
        soft = soft * (self.temperature ** 2)
        return self.kd_alpha * hard + (1.0 - self.kd_alpha) * soft, hard, soft

    def train_step(self, data):
        x, (y, t_soft) = data
        with tf.GradientTape() as tape:
            p_s = self.student(x, training=True)
            loss, hard, soft = self._combined_loss(y, t_soft, p_s)
            loss = loss + tf.add_n(self.student.losses) if self.student.losses else loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        results = {'loss': loss, 'hard_loss': hard, 'soft_loss': soft}
        for m in self._metrics_list:
            m.update_state(y, p_s)
            results[m.name] = m.result()
        return results

    def test_step(self, data):
        x, (y, t_soft) = data
        p_s = self.student(x, training=False)
        loss, hard, soft = self._combined_loss(y, t_soft, p_s)
        results = {'loss': loss, 'hard_loss': hard, 'soft_loss': soft}
        for m in self._metrics_list:
            m.update_state(y, p_s)
            results[m.name] = m.result()
        return results

    @property
    def metrics(self):
        return self._metrics_list


# ============================================================================
# DATASET — 6b mel pipeline + attached teacher soft labels
# ============================================================================

def load_cached_mels(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    cache_file = Path(config.cache_dir) / split / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run 6b once to build it.")
    data = np.load(cache_file)
    logger.info(f"Loaded {len(data['mels'])} cached mels for {split}")
    return data['mels'], data['labels']


def create_distill_dataset(split: str, config: TrainingConfig, teacher_soft: np.ndarray,
                           augment: bool = False) -> tf.data.Dataset:
    """
    Yields (mel, (y_onehot, teacher_soft)). Augmentation is the exact 6b recipe
    (noise + time-shift on axis 0); --specaug adds 8a-style masks adapted to
    the (time, freq, 1) layout. Teacher labels stay attached to the clean clip.
    """
    mel_specs, labels = load_cached_mels(split, config)
    mel_specs = mel_specs[..., np.newaxis]  # (n, 184, 16, 1)
    assert len(teacher_soft) == len(labels), "teacher/student sample count mismatch"

    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels, teacher_soft))

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(mel_specs), seed=config.random_seed)

    def pack(mel, label, t_soft):
        return mel, (tf.one_hot(label, depth=2), t_soft)

    dataset = dataset.map(pack, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and split == 'train':
        n_mels = config.n_mels
        use_specaug = config.use_specaug

        def augment_mel(mel, y):
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.02)
            mel = tf.clip_by_value(mel + noise, 0.0, 1.0)

            should_shift = tf.random.uniform(()) > 0.5

            def time_shift(m):
                shift = tf.random.uniform((), minval=-10, maxval=10, dtype=tf.int32)
                return tf.roll(m, shift=shift, axis=0)

            mel = tf.cond(should_shift, lambda: time_shift(mel), lambda: mel)

            if use_specaug:
                # 8a settings adapted to (time, freq, 1): 2 time masks (≤20
                # frames) + 1 freq mask (≤3 bins), each applied with p=0.5
                def time_mask(m):
                    w = tf.random.uniform((), 1, 21, dtype=tf.int32)
                    t0 = tf.random.uniform((), 0, TIME_STEPS - w, dtype=tf.int32)
                    mask = tf.concat([
                        tf.ones((t0, 1, 1)),
                        tf.zeros((w, 1, 1)),
                        tf.ones((TIME_STEPS - t0 - w, 1, 1))
                    ], axis=0)
                    return m * mask

                def freq_mask(m):
                    w = tf.random.uniform((), 1, 4, dtype=tf.int32)
                    f0 = tf.random.uniform((), 0, n_mels - w, dtype=tf.int32)
                    mask = tf.concat([
                        tf.ones((1, f0, 1)),
                        tf.zeros((1, w, 1)),
                        tf.ones((1, n_mels - f0 - w, 1))
                    ], axis=1)
                    return m * mask

                for _ in range(2):
                    mel = tf.cond(tf.random.uniform(()) > 0.5,
                                  lambda: time_mask(mel), lambda: mel)
                mel = tf.cond(tf.random.uniform(()) > 0.5,
                              lambda: freq_mask(mel), lambda: mel)

            mel = tf.ensure_shape(mel, (TIME_STEPS, n_mels, 1))
            return mel, y

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================================
# EVALUATOR (same protocol as 8a/8b: float32 + INT8 + threshold sweep)
# ============================================================================

class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(history['loss']) + 1)
        panels = [('loss', 'val_loss', 'Combined Loss'),
                  ('auc', 'val_auc', 'AUC'),
                  ('hard_loss', 'val_hard_loss', 'Hard (Focal) Loss'),
                  ('soft_loss', 'val_soft_loss', 'Soft (KD) Loss')]
        for ax, (tr, va, title) in zip(axes.flat, panels):
            if tr in history:
                ax.plot(epochs, history[tr], label='Train', linewidth=2)
            if va in history:
                ax.plot(epochs, history[va], label='Val', linewidth=2)
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

    def evaluate_model(self, model, test_dataset, prefix=''):
        preds, probs, true_labels = [], [], []
        for x, (y, _t) in test_dataset:
            out = model(x, training=False)
            preds.extend(np.argmax(out, axis=1))
            probs.extend(out[:, 1].numpy())
            true_labels.extend(np.argmax(y.numpy(), axis=1))
        preds, probs = np.array(preds), np.array(probs)
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probs)
        acc = np.mean(preds == true_labels)
        logger.info(f"{prefix}Acc={acc:.4f}, AUC={auc:.4f}")
        self._plot_confusion_matrix(true_labels, preds, prefix)
        self._plot_roc_curve(true_labels, probs, prefix)
        self._save_classification_report(true_labels, preds, prefix)
        return auc

    def evaluate_tflite(self, tflite_path, test_dataset):
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        odet = interp.get_output_details()[0]
        isc, izp = idet['quantization']
        osc, ozp = odet['quantization']

        preds, probs, true_labels, times = [], [], [], []
        for inputs, (labels, _t) in tqdm(test_dataset, desc='TFLite eval'):
            inputs_np, labels_np = inputs.numpy(), labels.numpy()
            if isc != 0.0:
                inputs_q = np.round(inputs_np / isc + izp).astype(idet['dtype'])
            else:
                inputs_q = inputs_np.astype(idet['dtype'])
            for i in range(inputs_np.shape[0]):
                t0 = time.perf_counter()
                interp.set_tensor(idet['index'], inputs_q[i:i + 1])
                interp.invoke()
                o = interp.get_tensor(odet['index']).astype(np.float32)
                times.append((time.perf_counter() - t0) * 1000)
                if osc != 0.0:
                    o = (o - ozp) * osc
                probs.append(float(o[0, 1]))
                preds.append(int(np.argmax(o, axis=1)[0]))
                true_labels.append(int(np.argmax(labels_np[i])))

        preds, probs = np.array(preds), np.array(probs)
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probs)
        acc = np.mean(preds == true_labels)
        avg_ms = np.mean(times)
        logger.info(f"TFLite Acc={acc:.4f}, AUC={auc:.4f}, {avg_ms:.2f}ms/sample")
        self._plot_confusion_matrix(true_labels, preds, 'tflite_')
        self._plot_roc_curve(true_labels, probs, 'tflite_')
        self._save_classification_report(true_labels, preds, 'tflite_')
        return acc, auc, avg_ms, probs, true_labels

    def threshold_sweep(self, probs, true_labels, taus=(0.25, 0.30, 0.34, 0.40, 0.50)):
        lines = ["tau | recall | precision | f1 | fpr | tp | fp | fn | tn"]
        for tau in taus:
            p = (probs >= tau).astype(int)
            tp = int(np.sum((p == 1) & (true_labels == 1)))
            fp = int(np.sum((p == 1) & (true_labels == 0)))
            fn = int(np.sum((p == 0) & (true_labels == 1)))
            tn = int(np.sum((p == 0) & (true_labels == 0)))
            rec = tp / (tp + fn) if tp + fn else 0
            prec = tp / (tp + fp) if tp + fp else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            fpr = fp / (fp + tn) if fp + tn else 0
            lines.append(f"{tau:.2f} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {fpr:.4f} "
                         f"| {tp} | {fp} | {fn} | {tn}")
        (self.output_dir / 'threshold_sweep.txt').write_text("\n".join(lines) + "\n")
        for line in lines:
            logger.info(f"  {line}")

    def _plot_confusion_matrix(self, true_labels, preds, prefix):
        cm = confusion_matrix(true_labels, preds)
        ConfusionMatrixDisplay(cm, display_labels=['Neg', 'Pos']).plot(
            cmap=plt.cm.Blues, values_format='d')
        plt.title(f'{prefix}Confusion Matrix')
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_roc_curve(self, true_labels, probs, prefix):
        fpr, tpr, _ = roc_curve(true_labels, probs)
        auc = roc_auc_score(true_labels, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC={auc:.4f}', linewidth=3, color='darkblue')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png', dpi=150)
        plt.close()

    def _save_classification_report(self, true_labels, preds, prefix):
        report = classification_report(true_labels, preds,
                                       target_names=['Negative', 'Positive'],
                                       digits=4, zero_division=0)
        (self.output_dir / f'{prefix}classification_report.txt').write_text(report)


# ============================================================================
# UTILITIES
# ============================================================================

def format_time(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    return f"{h}h {m:02d}m {s % 60:05.2f}s" if h else f"{m}m {s % 60:05.2f}s"


def get_git_hash():
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def save_config(config, output_dir, system_info, teacher_path, teacher_val_auc,
                model_summary=""):
    path = Path(output_dir) / 'config.txt'
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DRONGONET-MICRO DISTILLATION TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("script=7a_micro_distill.py\n")
        f.write(f"git_hash={get_git_hash()}\n\n")
        f.write("Distillation:\n")
        f.write(f"  teacher_model: {teacher_path}\n")
        f.write(f"  teacher_val_auc: {teacher_val_auc:.4f}\n")
        f.write(f"  kd_alpha (hard weight): {config.kd_alpha}\n")
        f.write(f"  kd_temperature: {config.kd_temperature}\n")
        f.write(f"  specaugment: {config.use_specaug}\n\n")
        f.write("Student: locked 6b architecture (919 params)\n")
        f.write(f"  n_mels: {config.n_mels}\n  n_fft: {config.n_fft}\n")
        f.write(f"  seed: {config.random_seed}\n\n")
        f.write("System:\n")
        for k, v in system_info.items():
            f.write(f"  {k}: {v}\n")
        if model_summary:
            f.write("\nModel Summary:\n" + model_summary + "\n")
    logger.info(f"Config saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.n_fft = args.n_fft
    config.n_mels = args.n_mels
    config.kd_alpha = args.kd_alpha
    config.kd_temperature = args.kd_temperature
    config.use_specaug = args.specaug
    config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'

    stem = '7b_micro_distill_specaug' if config.use_specaug else '7a_micro_distill'
    platform_tag = 'macos' if platform.system() == 'Darwin' else 'linux'
    config.output_dir = (args.output_dir or
                         f'results/{stem}_fft{config.n_fft}_m{config.n_mels}'
                         f'_s{config.random_seed}_{platform_tag}')

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    system_info = {
        'platform': platform.system(), 'machine': platform.machine(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': [d.name for d in tf.config.list_physical_devices('GPU')],
    }

    logger.info("=" * 60)
    logger.info(f"DrongoNet-Micro DISTILLATION ({stem})")
    logger.info(f"kd_alpha={config.kd_alpha}, T={config.kd_temperature}, "
                f"SpecAug={config.use_specaug}, seed={config.random_seed}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    try:
        # ── Student cache check ────────────────────────────────────────────
        if not (Path(config.cache_dir) / 'cache_info.pkl').exists():
            raise FileNotFoundError(
                f"Student mel cache not found: {config.cache_dir}. "
                f"Run 6b_micro_final.py once to build it.")
        logger.info(f"Student cache: {config.cache_dir}")

        # ── Teacher: load, align, precompute soft labels ───────────────────
        teacher_path = resolve_teacher_path(args.teacher_model)
        teacher = load_teacher(teacher_path)
        teacher_cache = teacher_cache_dir_for(teacher, args.teacher_cache, config.n_fft)
        logger.info(f"Teacher cache: {teacher_cache}")

        cache_key = Path(teacher_path).parent.name  # e.g. drongonet_edge_..._s42_linux
        _, train_labels = load_cached_mels('train', config)
        _, val_labels = load_cached_mels('val', config)

        t_probs_train = compute_teacher_probs(
            teacher, teacher_cache, 'train', train_labels, cache_key,
            force=args.force_teacher_recompute)
        t_probs_val = compute_teacher_probs(
            teacher, teacher_cache, 'val', val_labels, cache_key,
            force=args.force_teacher_recompute)

        teacher_val_auc = roc_auc_score(val_labels, t_probs_val[:, 1])
        logger.info(f"Teacher val AUC (sanity): {teacher_val_auc:.4f}")
        if teacher_val_auc < 0.99:
            logger.warning(
                f"Teacher val AUC {teacher_val_auc:.4f} < 0.99 — this teacher is weak "
                f"(Mac-trained Edge models converge poorly). Consider --teacher_model "
                f"pointing at a Linux-trained Edge checkpoint.")

        # Free teacher GPU memory before student training
        del teacher
        tf.keras.backend.clear_session()
        tf.random.set_seed(config.random_seed)

        t_soft_train = soften(t_probs_train, config.kd_temperature)
        t_soft_val = soften(t_probs_val, config.kd_temperature)

        # ── Datasets ───────────────────────────────────────────────────────
        train_ds = create_distill_dataset('train', config, t_soft_train, augment=True)
        val_ds = create_distill_dataset('val', config, t_soft_val, augment=False)
        # Test set needs no teacher labels; pass zeros (loss unused at eval)
        _, test_labels = load_cached_mels('test', config)
        test_ds = create_distill_dataset(
            'test', config, np.zeros((len(test_labels), 2), np.float32), augment=False)

        # ── Student model ──────────────────────────────────────────────────
        student = build_cnn_mel_low_power_optimized(
            input_shape=(TIME_STEPS, config.n_mels, 1), num_classes=2)

        model_summary_lines = []
        student.summary(print_fn=lambda x: (logger.info(x), model_summary_lines.append(x)))
        model_summary_str = "\n".join(model_summary_lines)
        (output_dir / 'model_summary.txt').write_text(model_summary_str)
        save_config(config, output_dir, system_info, teacher_path,
                    teacher_val_auc, model_summary_str)

        trainer = DistillTrainer(student, config.kd_alpha, config.kd_temperature)
        trainer.compile(
            optimizer=get_optimizer(config.learning_rate),
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            ])

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=config.lr_reduction_factor,
                patience=config.lr_patience, mode='max', min_lr=config.min_lr),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=config.early_stopping_patience,
                mode='max', restore_best_weights=True),
        ]

        logger.info("Training student with distillation loss...")
        t_train = time.time()
        history = trainer.fit(train_ds, validation_data=val_ds,
                              epochs=config.epochs, callbacks=callbacks, verbose=1)
        times = {'training': time.time() - t_train}
        logger.info(f"Training done in {format_time(times['training'])}")

        evaluator = ModelEvaluator(output_dir)
        evaluator.plot_training_history(history.history)

        logger.info("Evaluating float32 student...")
        float_auc = evaluator.evaluate_model(student, test_ds, prefix='float_')
        student.save(str(output_dir / 'best_model.keras'))

        # ── TFLite INT8 (same conversion as 6b) ────────────────────────────
        logger.info("Converting to TFLite INT8...")

        def representative_dataset():
            count = 0
            for inputs, _y in val_ds:
                if count >= args.repr_samples:
                    break
                for i in range(inputs.shape[0]):
                    if count >= args.repr_samples:
                        break
                    yield [inputs[i:i + 1]]
                    count += 1

        tflite_model, strategy_success = None, "none"
        for name, params in [
            ("Default quantization", {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'representative_dataset': representative_dataset}),
            ("No quantization", {})
        ]:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(student)
                for k, v in params.items():
                    setattr(converter, k, v)
                tflite_model = converter.convert()
                strategy_success = name
                logger.info(f"{name} succeeded")
                break
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

        tflite_path = output_dir / 'model.tflite'
        tflite_path.write_bytes(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"TFLite: {tflite_size_kb:.2f} KB")

        tflite_acc, tflite_auc, tflite_ms, tflite_probs, tflite_labels = \
            evaluator.evaluate_tflite(tflite_path, test_ds)
        evaluator.threshold_sweep(tflite_probs, tflite_labels)

        total_params = student.count_params()
        total_time = time.time() - start_time

        with open(output_dir / 'results_summary.txt', 'w') as f:
            f.write("=" * 60 + "\nDRONGONET-MICRO DISTILLATION RESULTS SUMMARY\n" + "=" * 60 + "\n\n")
            f.write(f"script=7a_micro_distill.py\n")
            f.write(f"variant={stem}\n")
            f.write(f"n_mels={config.n_mels}\nn_fft={config.n_fft}\nseed={config.random_seed}\n")
            f.write(f"teacher_model={teacher_path}\n")
            f.write(f"teacher_val_auc={teacher_val_auc:.4f}\n")
            f.write(f"kd_alpha={config.kd_alpha}\n")
            f.write(f"kd_temperature={config.kd_temperature}\n")
            f.write(f"specaugment={config.use_specaug}\n\n")
            f.write(f"float_auc={float_auc:.4f}\n")
            f.write(f"tflite_strategy={strategy_success}\n")
            f.write(f"tflite_accuracy={tflite_acc:.4f}\n")
            f.write(f"tflite_auc={tflite_auc:.4f}\n")
            f.write(f"tflite_latency_ms={tflite_ms:.2f}\n")
            f.write(f"tflite_size_kb={tflite_size_kb:.2f}\n")
            f.write(f"total_params={total_params}\n")
            deg = float_auc - tflite_auc
            f.write(f"auc_degradation={deg:.4f}\n")
            f.write(f"auc_degradation_pct={deg / float_auc * 100:.2f}\n")
            f.write("\n" + "=" * 60 + "\nTiming\n" + "=" * 60 + "\n")
            f.write(f"Training: {format_time(times['training'])}\n")
            f.write(f"Total: {format_time(total_time)}\n")

        logger.info("=" * 60)
        logger.info(f"DONE: {stem} s{config.random_seed}")
        logger.info(f"  Params: {total_params:,}  Size: {tflite_size_kb:.2f} KB")
        logger.info(f"  Float AUC: {float_auc:.4f}  TFLite AUC: {tflite_auc:.4f}")
        logger.info(f"  (6b control, no KD: ~0.9803 Linux / 0.9820 Mac)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        elapsed = time.time() - start_time
        (output_dir / 'elapsed.txt').write_text(
            f"Total: {format_time(elapsed)}\nSeconds: {elapsed:.3f}\n")


if __name__ == '__main__':
    main()
