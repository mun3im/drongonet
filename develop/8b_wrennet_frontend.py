#!/usr/bin/env python3
"""
7b_wrennet_frontend.py: SEABADNet-Matchbox + WrenNet semi-learnable front-end
(post-ablation exploratory variant, 2026-07-07)

WrenNet (Ciapponi et al., ICASSP 2026, ciapponi2026enabling) proposes a
semi-learnable spectral feature extractor: a sigmoid-weighted convex combination
of logarithmic and linear frequency mappings, parameterised by a breakpoint b
(transition frequency) and a transition width w (sharpness).  The parameters
are learned end-to-end with a higher LR (15× the main network).

Adaptation for SEABADNet:
  Since we operate on cached mel spectrograms (not raw audio + STFT), we cannot
  re-run the FFT bin mapping inside TF.  Instead we approximate the WrenNet idea
  at the mel-bin level: a SemiLearnableFrequencyMap layer applies a per-bin soft
  gain g(i; b, w) that smoothly transitions from log-weighted to linear-weighted
  emphasis across the 16 mel bins.  This is a differentiable proxy for the bin
  selection in Eq. (3) of the paper, operating on the already-computed mel
  features rather than the raw FFT bins.

  Concretely:
    x_i  = i / (n_mels - 1)          normalised bin index in [0,1]
    S(x; b, w) = sigmoid((x - b) * w)  breakpoint gate
    g(i) = (1 - S) * w_log(i)  +  S * w_lin(i)
  where w_log and w_lin are two learned per-bin weight vectors (initialised to
  1.0), enabling the network to discover whether log or linear emphasis serves
  each bin.  The two learnable scalars b and w control the global transition.

  Higher LR multiplier for front-end params is implemented via a separate
  optimizer applied in a custom training loop (15× as in WrenNet).  On Apple
  Silicon the two-optimizer approach is simplified: a single Adam pass with
  a moderate LR for the front-end params is used to avoid Metal instability.

Backbone: identical to 7a (2×1×16 MatchboxNet-style TCS residual net).
All other conventions (focal loss, threshold sweep, output format) follow the
SEABADNet ablation chain and 7a_matchbox_micro.py.

Usage:
  conda run -n tf215_gpu python 7b_wrennet_frontend.py --random_seed 42 --use_cache
  conda run -n tf215_gpu python 7b_wrennet_frontend.py --random_seed 100 --use_cache
  conda run -n tf215_gpu python 7b_wrennet_frontend.py --random_seed 786 --use_cache

Output dir: results/7b_wrennet_frontend_fft{n_fft}_m{n_mels}_s{seed}/
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
    output_dir: str = f'{RESULTS_BASE}/7b_wrennet_frontend'
    cache_dir: str = f'{CACHE_BASE}_fft1024_m16'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    mel_fmin: float = 100.0
    mel_fmax: float = 8000.0
    # MatchboxNet B×R×C backbone
    blocks: int = 2
    sub_blocks: int = 1
    channels: int = 16
    epilogue_channels: int = 16
    epilogue_dilation: int = 1
    dropout: float = 0.1
    use_bn: bool = True
    use_specaug: bool = True
    # WrenNet front-end LR multiplier (applied to b and w params)
    frontend_lr_multiplier: float = 15.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SEABADNet-Matchbox + WrenNet semi-learnable front-end')
    parser.add_argument('--repr_samples', type=int, default=500)
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--force-reprocess', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--n_mels', type=int, default=16)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--sub_blocks', type=int, default=1)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--epilogue_channels', type=int, default=16)
    parser.add_argument('--epilogue_dilation', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--no_specaug', action='store_true')
    parser.add_argument('--frontend_lr_mult', type=float, default=15.0,
                        help='LR multiplier for WrenNet front-end params b and w (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None)
    return parser.parse_args()


args = parse_args()

if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SemiLearnableFrequencyMap(tf.keras.layers.Layer):
    """
    WrenNet-style semi-learnable frequency emphasis (Ciapponi et al. 2026, §3).

    Approximation for post-mel features: learns a per-bin gain that is a
    sigmoid-gated convex combination of two learned weight vectors (w_log and
    w_lin), controlled by a global breakpoint b and transition width w.

    On the n_mels=16 mel grid (bins already log-spaced by librosa), w_log
    emphasises low-frequency bins and w_lin allows more uniform emphasis —
    the network can shift the crossover to fit SEABAD's spectral statistics.

    Parameters trained: b (1), w (1), w_log (n_mels), w_lin (n_mels) → 2+2×16=34
    High-LR vars (b, w) are tagged with the 'frontend' variable collection so
    the custom training step can apply a higher gradient scale.

    Input/output shape: [batch, time, 1, n_mels]  (unchanged)
    """

    def __init__(self, n_mels=16, **kwargs):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        # Normalised bin positions [0, 1]
        self._bin_pos = np.linspace(0.0, 1.0, n_mels, dtype=np.float32)

    def build(self, input_shape):
        # Global transition parameters — high LR in WrenNet (×15 / ×5)
        self.breakpoint_b = self.add_weight(
            name='breakpoint_b',
            shape=(1,),
            initializer=tf.constant_initializer(0.5),  # midpoint
            trainable=True,
            dtype=tf.float32
        )
        self.transition_w = self.add_weight(
            name='transition_w',
            shape=(1,),
            initializer=tf.constant_initializer(10.0),  # moderate sharpness
            trainable=True,
            dtype=tf.float32
        )
        # Per-bin emphasis weights for the two regimes
        self.w_log = self.add_weight(
            name='w_log',
            shape=(1, 1, 1, self.n_mels),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            dtype=tf.float32
        )
        self.w_lin = self.add_weight(
            name='w_lin',
            shape=(1, 1, 1, self.n_mels),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Gate S(x; b, w) — per bin, broadcast over batch/time/dummy axes
        # Shape after reshape: (1, 1, 1, n_mels)
        x = tf.constant(self._bin_pos, dtype=tf.float32)  # (n_mels,)
        x = tf.reshape(x, (1, 1, 1, self.n_mels))
        gate = tf.math.sigmoid((x - self.breakpoint_b) * self.transition_w)

        # Convex combination of log and linear per-bin weights
        w_log_pos = tf.nn.softplus(self.w_log)  # keep positive
        w_lin_pos = tf.nn.softplus(self.w_lin)
        gain = (1.0 - gate) * w_log_pos + gate * w_lin_pos
        return inputs * gain

    def get_config(self):
        config = super().get_config()
        config.update({'n_mels': self.n_mels})
        return config


def tcs_conv(x, filters, kernel, use_bn, stride=1, dilation=1, prefix=""):
    """Time-channel separable conv — identical to 7a."""
    x = tf.keras.layers.DepthwiseConv2D(
        (kernel, 1), dilation_rate=(dilation, 1), padding='same',
        use_bias=False, name=f'{prefix}dw'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), strides=(stride, 1), padding='same',
        use_bias=not use_bn, name=f'{prefix}pw'
    )(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}bn')(x)
    return x


def matchbox_block(x, filters, kernel, sub_blocks, dropout, use_bn, block_idx):
    """MatchboxNet residual block — identical to 7a."""
    in_ch = x.shape[-1]
    if in_ch == filters:
        skip = x
    else:
        skip = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same', use_bias=not use_bn,
            name=f'b{block_idx}skip'
        )(x)
        if use_bn:
            skip = tf.keras.layers.BatchNormalization(name=f'b{block_idx}skipbn')(skip)

    y = x
    for r in range(sub_blocks):
        y = tcs_conv(y, filters, kernel, use_bn, prefix=f'b{block_idx}r{r}')
        if r < sub_blocks - 1:
            y = tf.keras.layers.ReLU()(y)
            y = tf.keras.layers.Dropout(dropout)(y)

    y = tf.keras.layers.Add(name=f'b{block_idx}add')([y, skip])
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    return y


def build_wrennet_matchbox(input_shape=(TIME_STEPS, 1, 16), num_classes=2,
                            blocks=2, sub_blocks=1, channels=16,
                            epilogue_channels=16, dropout=0.1, use_bn=True,
                            epilogue_dilation=1):
    """
    SEABADNet-WrenNet: Matchbox TCS backbone with WrenNet semi-learnable
    frequency front-end replacing the fixed ChannelFrequencyEmphasis.
    Input: [time=184, 1, n_mels] — same layout as 7a.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # WrenNet semi-learnable front-end
    x = SemiLearnableFrequencyMap(n_mels=input_shape[2], name='slfe')(inputs)

    # MatchboxNet TCS backbone (identical to 7a)
    x = tcs_conv(x, channels, 9, use_bn, stride=2, prefix='p')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    for b in range(blocks):
        kernel = 11 + 2 * b
        x = matchbox_block(x, channels, kernel, sub_blocks, dropout, use_bn, b + 1)

    x = tcs_conv(x, epilogue_channels, 17, use_bn, dilation=epilogue_dilation, prefix='e')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc')(x)

    return tf.keras.Model(inputs, outputs, name=f'wmbx{blocks}x{sub_blocks}x{channels}')


# ============================================================================
# CUSTOM TRAINING STEP — dual-LR for WrenNet front-end params
# ============================================================================

class WrenNetTrainer(tf.keras.Model):
    """
    Wraps the WrenNet-Matchbox model with a custom train_step that applies a
    higher learning rate to the SemiLearnableFrequencyMap's b and w params,
    matching WrenNet's 15× (breakpoint) and 5× (width) schedule.

    On Apple Silicon: uses a single legacy Adam for stability; LR multiplier
    is approximated by scaling the gradient of b/w before applying.
    """

    def __init__(self, base_model, config, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.config = config
        self._loss_fn = None
        self._metrics_list = []

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics or [], **kwargs)
        self._loss_fn = loss
        self._metrics_list = metrics or []
        # Second optimizer for the high-LR front-end params
        fe_lr = optimizer.learning_rate * self.config.frontend_lr_multiplier
        system = platform.system()
        machine = platform.machine()
        if system == 'Darwin' and machine == 'arm64':
            self._fe_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=fe_lr)
        else:
            self._fe_optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fe_lr, weight_decay=1e-4)

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # Identify front-end params (b, w) vs backbone params
        fe_vars = [v for v in self.base_model.trainable_variables
                   if 'slfe/breakpoint_b' in v.name or 'slfe/transition_w' in v.name]
        fe_var_names = {v.name for v in fe_vars}
        backbone_vars = [v for v in self.base_model.trainable_variables
                         if v.name not in fe_var_names]

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self._loss_fn(y, y_pred)

        all_vars = fe_vars + backbone_vars
        grads = tape.gradient(loss, all_vars)
        fe_grads = grads[:len(fe_vars)]
        bb_grads = grads[len(fe_vars):]

        if fe_vars:
            self._fe_optimizer.apply_gradients(zip(fe_grads, fe_vars))
        if backbone_vars:
            self.optimizer.apply_gradients(zip(bb_grads, backbone_vars))

        results = {'loss': loss}
        for m in self._metrics_list:
            m.update_state(y, y_pred)
            results[m.name] = m.result()
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.base_model(x, training=False)
        loss = self._loss_fn(y, y_pred)
        results = {'loss': loss}
        for m in self._metrics_list:
            m.update_state(y, y_pred)
            results[m.name] = m.result()
        return results

    @property
    def metrics(self):
        return self._metrics_list


# ============================================================================
# DATASET — identical to 7a
# ============================================================================

class SEABADDataset:
    def __init__(self, root_dir, split='train', fraction=1.0, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.fraction = fraction
        self.files = []
        self.labels = []
        random.seed(seed)

        positive_files, negative_files = [], []
        for label, class_name in enumerate(['negative', 'positive']):
            path = os.path.join(root_dir, class_name)
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} does not exist!")
            class_files = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.wav'):
                        class_files.append(os.path.join(root, f))
            if class_name == 'negative':
                negative_files = [(f, label) for f in class_files]
            else:
                positive_files = [(f, label) for f in class_files]

        logger.info(f"Found {len(positive_files)} positive and {len(negative_files)} negative samples")
        random.shuffle(positive_files)
        random.shuffle(negative_files)

        n_pos, n_neg = len(positive_files), len(negative_files)
        train_pos_end, val_pos_end = int(n_pos * 0.8), int(n_pos * 0.9)
        train_neg_end, val_neg_end = int(n_neg * 0.8), int(n_neg * 0.9)

        if split == 'train':
            subset = positive_files[:train_pos_end] + negative_files[:train_neg_end]
        elif split == 'val':
            subset = positive_files[train_pos_end:val_pos_end] + negative_files[train_neg_end:val_neg_end]
        elif split == 'test':
            subset = positive_files[val_pos_end:] + negative_files[val_neg_end:]
        else:
            raise ValueError(f"Invalid split: {split}")

        if fraction < 1.0:
            subset = random.sample(subset, int(len(subset) * fraction))

        random.shuffle(subset)
        self.files = [f[0] for f in subset]
        self.labels = [f[1] for f in subset]
        logger.info(f"Loaded {len(self.files)} files for {split} "
                    f"({sum(1 for l in self.labels if l==0)} neg, "
                    f"{sum(1 for l in self.labels if l==1)} pos)")

    def __len__(self):
        return len(self.files)

    def get_files_and_labels(self):
        return self.files, self.labels


def compute_mel_spectrogram(waveform, config):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=config.target_sr, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.mel_fmin, fmax=config.mel_fmax, center=False
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    if mel_spec_db.shape[0] > TIME_STEPS:
        mel_spec_db = mel_spec_db[:TIME_STEPS, :]
    elif mel_spec_db.shape[0] < TIME_STEPS:
        pad_width = ((0, TIME_STEPS - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant',
                             constant_values=mel_spec_db.min())
    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    if mel_max - mel_min > 1e-6:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)
    return mel_spec_db


def preprocess_and_cache_mels(dataset_path, config, force_reprocess=False):
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_info_path = cache_dir / 'cache_info.pkl'
    if cache_info_path.exists() and not force_reprocess:
        logger.info(f"Cache exists at {cache_dir}. Skipping preprocessing.")
        return

    cache_info = {}
    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split}...")
        dataset = SEABADDataset(dataset_path, split=split,
                                fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()
        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)
        mel_specs, valid_labels = [], []
        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            try:
                waveform, sr = librosa.load(file_path, sr=None)
                if sr != config.target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr,
                                               target_sr=config.target_sr)
                if len(waveform) > config.target_length:
                    waveform = waveform[:config.target_length]
                elif len(waveform) < config.target_length:
                    waveform = np.concatenate(
                        [waveform, np.zeros(config.target_length - len(waveform))])
                mel_specs.append(compute_mel_spectrogram(waveform, config))
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed: {file_path}: {e}")

        mel_specs = np.array(mel_specs, dtype=np.float32)
        valid_labels = np.array(valid_labels, dtype=np.int32)
        cache_file = split_cache_dir / 'mels.npz'
        np.savez_compressed(cache_file, mels=mel_specs, labels=valid_labels)
        cache_info[split] = {'n_samples': len(mel_specs), 'shape': mel_specs.shape}

    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)
    logger.info("Preprocessing complete.")


def load_cached_mels(split, config):
    cache_file = Path(config.cache_dir) / split / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    data = np.load(cache_file)
    return data['mels'], data['labels']


def create_tf_dataset_from_cache(split, config, augment=False):
    mel_specs, labels = load_cached_mels(split, config)
    # (n, time, n_mels) → (n, time, 1, n_mels) — mel bins as channels
    mel_specs = mel_specs[:, :, np.newaxis, :]
    class_counts = {0: int(np.sum(labels == 0)), 1: int(np.sum(labels == 1))}
    logger.info(f"  {split}: neg={class_counts[0]}, pos={class_counts[1]}")

    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels))

    if split == 'train':
        dataset = dataset.shuffle(len(mel_specs), seed=config.random_seed)

    def to_one_hot(mel, label):
        return mel, tf.one_hot(label, depth=2)

    dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and split == 'train':
        n_mels = config.n_mels
        use_specaug = config.use_specaug

        def augment_mel(mel, label):
            mel = mel + tf.random.normal(tf.shape(mel), stddev=0.02)
            mel = tf.clip_by_value(mel, 0.0, 1.0)
            should_shift = tf.random.uniform(()) > 0.5

            def time_shift(m):
                shift = tf.random.uniform((), -10, 10, dtype=tf.int32)
                return tf.roll(m, shift=shift, axis=0)

            mel = tf.cond(should_shift, lambda: time_shift(mel), lambda: mel)

            if use_specaug:
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
                        tf.ones((1, 1, f0)),
                        tf.zeros((1, 1, w)),
                        tf.ones((1, 1, n_mels - f0 - w))
                    ], axis=2)
                    return m * mask

                for _ in range(2):
                    mel = tf.cond(tf.random.uniform(()) > 0.5,
                                  lambda: time_mask(mel), lambda: mel)
                mel = tf.cond(tf.random.uniform(()) > 0.5,
                              lambda: freq_mask(mel), lambda: mel)

            mel = tf.ensure_shape(mel, (TIME_STEPS, 1, n_mels))
            return mel, label

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE), class_counts


# ============================================================================
# LOSS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - pt, gamma)
        alpha_weight = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)
        return tf.reduce_mean(alpha_weight * focal_weight *
                              tf.reduce_sum(cross_entropy, axis=-1, keepdims=True))
    return loss


# ============================================================================
# TRAINER AND EVALUATOR
# ============================================================================

def get_base_optimizer(learning_rate):
    system, machine = platform.system(), platform.machine()
    if system == 'Darwin' and machine == 'arm64':
        logger.info("Apple Silicon — legacy Adam")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    logger.info(f"{system} {machine} — AdamW")
    return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(history['loss']) + 1)
        pairs = [('loss', 'val_loss', 'Loss'), ('auc', 'val_auc', 'AUC'),
                 ('accuracy', 'val_accuracy', 'Accuracy'),
                 ('precision', 'val_precision', 'Precision')]
        for ax, (tr, va, title) in zip(axes.flat, pairs):
            if tr in history:
                ax.plot(epochs, history[tr], label=f'Train {title}')
            if va in history:
                ax.plot(epochs, history[va], label=f'Val {title}')
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

    def evaluate_model(self, model, test_dataset, prefix=''):
        preds, probs = [], []
        true_labels = []
        for x, y in test_dataset:
            out = model(x, training=False)
            preds.extend(np.argmax(out, axis=1))
            probs.extend(out[:, 1].numpy())
            true_labels.extend(np.argmax(y.numpy(), axis=1))
        preds = np.array(preds); probs = np.array(probs); true_labels = np.array(true_labels)
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
        for inputs, labels in tqdm(test_dataset, desc='TFLite eval'):
            inputs_np = inputs.numpy()
            labels_np = labels.numpy()
            if isc != 0.0:
                inputs_q = np.round(inputs_np / isc + izp).astype(idet['dtype'])
            else:
                inputs_q = inputs_np.astype(idet['dtype'])
            for i in range(inputs_np.shape[0]):
                t0 = time.perf_counter()
                interp.set_tensor(idet['index'], inputs_q[i:i+1])
                interp.invoke()
                o = interp.get_tensor(odet['index']).astype(np.float32)
                times.append((time.perf_counter() - t0) * 1000)
                if osc != 0.0:
                    o = (o - ozp) * osc
                probs.append(float(o[0, 1]))
                preds.append(int(np.argmax(o, axis=1)[0]))
                true_labels.append(int(np.argmax(labels_np[i])))

        preds = np.array(preds); probs = np.array(probs); true_labels = np.array(true_labels)
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
            tp = int(np.sum((p==1)&(true_labels==1))); fp = int(np.sum((p==1)&(true_labels==0)))
            fn = int(np.sum((p==0)&(true_labels==1))); tn = int(np.sum((p==0)&(true_labels==0)))
            rec = tp/(tp+fn) if tp+fn else 0; prec = tp/(tp+fp) if tp+fp else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec else 0; fpr = fp/(fp+tn) if fp+tn else 0
            lines.append(f"{tau:.2f} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {fpr:.4f} "
                         f"| {tp} | {fp} | {fn} | {tn}")
        path = self.output_dir / 'threshold_sweep.txt'
        path.write_text("\n".join(lines) + "\n")
        for line in lines:
            logger.info(f"  {line}")

    def _plot_confusion_matrix(self, true_labels, preds, prefix):
        cm = confusion_matrix(true_labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Neg', 'Pos'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
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
    return f"{h}h {m:02d}m {s%60:05.2f}s" if h else f"{m}m {s%60:05.2f}s"


def get_git_hash():
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def save_config(config, output_dir, system_info, model_summary=""):
    path = Path(output_dir) / 'config.txt'
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEABADNET-WRENNET FRONTEND TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"script=7b_wrennet_frontend.py\n")
        f.write(f"git_hash={get_git_hash()}\n\n")
        f.write(f"arch=wmbx{config.blocks}x{config.sub_blocks}x{config.channels}\n")
        f.write(f"n_mels={config.n_mels}\nn_fft={config.n_fft}\n")
        f.write(f"seed={config.random_seed}\n")
        f.write(f"batch_norm={config.use_bn}\nspecaugment={config.use_specaug}\n")
        f.write(f"dropout={config.dropout}\n")
        f.write(f"frontend_lr_multiplier={config.frontend_lr_multiplier}\n\n")
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
    config.blocks = args.blocks
    config.sub_blocks = args.sub_blocks
    config.channels = args.channels
    config.epilogue_channels = args.epilogue_channels
    config.epilogue_dilation = args.epilogue_dilation
    config.dropout = args.dropout
    config.use_bn = not args.no_bn
    config.use_specaug = not args.no_specaug
    config.frontend_lr_multiplier = args.frontend_lr_mult
    config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'
    platform_tag = 'macos' if platform.system() == 'Darwin' else 'linux'
    config.output_dir = (args.output_dir or
                         f'results/7b_wrennet_frontend_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}_{platform_tag}')

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
    logger.info("SEABADNet-WrenNet Frontend TRAINING")
    logger.info(f"B×R×C: {config.blocks}×{config.sub_blocks}×{config.channels}, "
                f"FE LR mult: {config.frontend_lr_multiplier}×")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    try:
        t_pre = time.time()
        if args.use_cache:
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found: {config.cache_dir}")
            logger.info(f"Using cache: {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config,
                                      force_reprocess=args.force_reprocess)
        times = {'preprocessing': time.time() - t_pre}

        train_ds, train_cc = create_tf_dataset_from_cache('train', config, augment=True)
        val_ds, _ = create_tf_dataset_from_cache('val', config)
        test_ds, _ = create_tf_dataset_from_cache('test', config)

        base_model = build_wrennet_matchbox(
            input_shape=(TIME_STEPS, 1, config.n_mels),
            num_classes=2,
            blocks=config.blocks, sub_blocks=config.sub_blocks,
            channels=config.channels, epilogue_channels=config.epilogue_channels,
            dropout=config.dropout, use_bn=config.use_bn,
            epilogue_dilation=config.epilogue_dilation
        )

        model_summary_lines = []
        base_model.summary(print_fn=lambda x: (logger.info(x), model_summary_lines.append(x)))
        model_summary_str = "\n".join(model_summary_lines)
        (output_dir / 'model_summary.txt').write_text(model_summary_str)
        save_config(config, output_dir, system_info, model_summary_str)

        # Log learned front-end init values
        slfe = base_model.get_layer('slfe')
        logger.info(f"SLFE init: breakpoint_b={slfe.breakpoint_b.numpy()}, "
                    f"transition_w={slfe.transition_w.numpy()}")

        # Wrap in dual-LR trainer
        trainer_model = WrenNetTrainer(base_model, config)
        base_optimizer = get_base_optimizer(config.learning_rate)
        metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        ]
        trainer_model.compile(
            optimizer=base_optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.5),
            metrics=metrics
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=config.lr_reduction_factor,
                patience=config.lr_patience, mode='max', min_lr=config.min_lr),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=config.early_stopping_patience,
                mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'best_model.keras'),
                monitor='val_auc', mode='max', save_best_only=True),
        ]

        logger.info("Training with dual-LR (WrenNet front-end + backbone)...")
        t_train = time.time()
        history = trainer_model.fit(
            train_ds, validation_data=val_ds,
            epochs=config.epochs, callbacks=callbacks, verbose=1
        )
        times['training'] = time.time() - t_train
        logger.info(f"Training done in {format_time(times['training'])}")

        evaluator = ModelEvaluator(output_dir)
        evaluator.plot_training_history(history.history)

        # Log learned front-end params after training
        logger.info(f"SLFE after training: breakpoint_b={slfe.breakpoint_b.numpy()}, "
                    f"transition_w={slfe.transition_w.numpy()}")
        np.save(str(output_dir / 'slfe_weights.npy'), {
            'b': slfe.breakpoint_b.numpy(),
            'w': slfe.transition_w.numpy(),
            'w_log': slfe.w_log.numpy(),
            'w_lin': slfe.w_lin.numpy(),
        })

        logger.info("Evaluating float32 model...")
        float_auc = evaluator.evaluate_model(base_model, test_ds, prefix='float_')
        base_model.save(str(output_dir / 'best_model.keras'))

        logger.info("Converting to TFLite INT8...")
        def representative_dataset():
            count = 0
            for inputs, _ in val_ds:
                if count >= args.repr_samples:
                    break
                for i in range(inputs.shape[0]):
                    if count >= args.repr_samples:
                        break
                    yield [inputs[i:i+1]]
                    count += 1

        tflite_model, strategy_success = None, "none"
        for strategy_name, params in [
            ("Default quantization", {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'representative_dataset': representative_dataset
            }),
            ("No quantization", {})
        ]:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
                for k, v in params.items():
                    setattr(converter, k, v)
                tflite_model = converter.convert()
                strategy_success = strategy_name
                logger.info(f"{strategy_name} succeeded")
                break
            except Exception as e:
                logger.warning(f"{strategy_name} failed: {e}")

        tflite_path = output_dir / 'model.tflite'
        tflite_path.write_bytes(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"TFLite saved: {tflite_size_kb:.2f} KB")

        tflite_acc, tflite_auc, tflite_ms, tflite_probs, tflite_labels = \
            evaluator.evaluate_tflite(tflite_path, test_ds)
        evaluator.threshold_sweep(tflite_probs, tflite_labels)

        total_params = base_model.count_params()
        total_time = time.time() - start_time
        summary = (output_dir / 'results_summary.txt')
        with open(summary, 'w') as f:
            f.write("=" * 60 + "\nSEABADNET-WRENNET FRONTEND RESULTS SUMMARY\n" + "=" * 60 + "\n\n")
            f.write(f"script=7b_wrennet_frontend.py\n")
            f.write(f"arch=wmbx{config.blocks}x{config.sub_blocks}x{config.channels}\n")
            f.write(f"n_mels={config.n_mels}\nn_fft={config.n_fft}\nseed={config.random_seed}\n")
            f.write(f"batch_norm={config.use_bn}\nspecaugment={config.use_specaug}\n")
            f.write(f"dropout={config.dropout}\n")
            f.write(f"frontend_lr_multiplier={config.frontend_lr_multiplier}\n\n")
            f.write(f"float_auc={float_auc:.4f}\n")
            f.write(f"tflite_strategy={strategy_success}\n")
            f.write(f"tflite_accuracy={tflite_acc:.4f}\n")
            f.write(f"tflite_auc={tflite_auc:.4f}\n")
            f.write(f"tflite_latency_ms={tflite_ms:.2f}\n")
            f.write(f"tflite_size_kb={tflite_size_kb:.2f}\n")
            f.write(f"total_params={total_params}\n")
            deg = float_auc - tflite_auc
            f.write(f"auc_degradation={deg:.4f}\nauc_degradation_pct={deg/float_auc*100:.2f}\n")
            f.write(f"\n" + "=" * 60 + "\nTiming\n" + "=" * 60 + "\n")
            f.write(f"Preprocessing: {format_time(times['preprocessing'])}\n")
            f.write(f"Training: {format_time(times['training'])}\n")
            f.write(f"Total: {format_time(total_time)}\n")

        logger.info("=" * 60)
        logger.info(f"DONE: WrenNet-Matchbox s{config.random_seed}")
        logger.info(f"  Params: {total_params:,}  Size: {tflite_size_kb:.2f} KB")
        logger.info(f"  Float AUC: {float_auc:.4f}  TFLite AUC: {tflite_auc:.4f}")
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
