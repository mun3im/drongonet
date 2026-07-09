#!/usr/bin/env python3
"""
7c_wrennet_matchbox_fusion.py: SEABADNet-WrenMatch — WrenNet + MatchboxNet fusion
(post-ablation exploratory variant, 2026-07-07)

Combines the distinctive ideas from both papers:

  From WrenNet (Ciapponi et al., ICASSP 2026, ciapponi2026enabling):
    1. Semi-learnable frequency front-end (SemiLearnableFrequencyMap from 7b)
    2. Squeeze-and-Excitation channel recalibration inside each residual block
       (WrenNet's SE Module, PhiNet-inspired; Hu et al. CVPR 2018 [16] in paper)
    3. Causal depthwise convolutions for streaming compatibility — implemented as
       explicit left-only zero-padding before DepthwiseConv2D with padding='valid',
       so the model has no lookahead (online inference safe)

  From MatchboxNet (Majumdar & Ginsburg, Interspeech 2020, majumdar2020matchboxnet):
    4. B×R×C residual TCS (time-channel separable) blocks with skip connections
    5. Growing kernel sizes across blocks (11, 13, ...)
    6. Wide-kernel epilogue conv (k=17) for large temporal receptive field
    7. SpecAugment-style time and frequency masking

  From SEABADNet Micro chain:
    8. Focal loss (γ=2, α=0.5) — repo standard
    9. GAP classifier head (single Dense(2) + softmax)
    10. Threshold sweep written inline, same τ grid as all variants

Architecture: WrenFE → Causal-Prologue(k=9, stride=2)
              → B × [Causal-TCS-ResBlock + SE]
              → Causal-Epilogue(k=17) → GAP → Dense(2)

SE ratio: 4 (compress channels by 4× → ReLU → restore → sigmoid gate).
Causal padding: (kernel-1) zeros prepended on the time axis before conv;
DepthwiseConv2D uses padding='valid' so no future frames leak.

Note on TFLite size: causal padding adds a PAD op per depthwise layer, pushing
the op count higher than 7a/7b.  The SE block adds 2 Dense + 2 activations.
Expected size ~15–20 KB INT8 (still well under Edge's 33 KB).

Usage:
  conda run -n tf215_gpu python 7c_wrennet_matchbox_fusion.py --random_seed 42 --use_cache
  conda run -n tf215_gpu python 7c_wrennet_matchbox_fusion.py --random_seed 100 --use_cache
  conda run -n tf215_gpu python 7c_wrennet_matchbox_fusion.py --random_seed 786 --use_cache

Output dir: results/7c_wrennet_matchbox_fusion_fft{n_fft}_m{n_mels}_s{seed}/
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

TIME_STEPS = 184


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
    output_dir: str = f'{RESULTS_BASE}/7c_wrennet_matchbox_fusion'
    cache_dir: str = f'{CACHE_BASE}_fft1024_m16'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    mel_fmin: float = 100.0
    mel_fmax: float = 8000.0
    blocks: int = 2
    sub_blocks: int = 1
    channels: int = 16
    epilogue_channels: int = 16
    epilogue_dilation: int = 1
    dropout: float = 0.1
    use_bn: bool = True
    use_specaug: bool = True
    se_ratio: int = 4          # SE squeeze ratio
    use_causal: bool = True    # causal depthwise convs (streaming safe)
    frontend_lr_multiplier: float = 15.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='WrenNet + MatchboxNet fusion variant for SEABADNet')
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
    parser.add_argument('--se_ratio', type=int, default=4,
                        help='SE squeeze ratio (default: 4)')
    parser.add_argument('--no_causal', action='store_true',
                        help='Disable causal padding (use bidirectional convs like 7a)')
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--no_specaug', action='store_true')
    parser.add_argument('--frontend_lr_mult', type=float, default=15.0)
    parser.add_argument('--output-dir', type=str, default=None)
    return parser.parse_args()


args = parse_args()

if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================================
# MODEL ARCHITECTURE — WrenNet + MatchboxNet fusion
# ============================================================================

class SemiLearnableFrequencyMap(tf.keras.layers.Layer):
    """
    WrenNet semi-learnable frequency emphasis (Ciapponi et al. 2026, §3).
    Identical to 7b.  Learns per-bin gain as a sigmoid-gated convex combination
    of log-emphasis and linear-emphasis weight vectors.
    Input/output: [batch, time, 1, n_mels]
    """

    def __init__(self, n_mels=16, **kwargs):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self._bin_pos = np.linspace(0.0, 1.0, n_mels, dtype=np.float32)

    def build(self, input_shape):
        self.breakpoint_b = self.add_weight(
            'breakpoint_b', shape=(1,),
            initializer=tf.constant_initializer(0.5), trainable=True)
        self.transition_w = self.add_weight(
            'transition_w', shape=(1,),
            initializer=tf.constant_initializer(10.0), trainable=True)
        self.w_log = self.add_weight(
            'w_log', shape=(1, 1, 1, self.n_mels),
            initializer=tf.constant_initializer(1.0), trainable=True)
        self.w_lin = self.add_weight(
            'w_lin', shape=(1, 1, 1, self.n_mels),
            initializer=tf.constant_initializer(1.0), trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.reshape(
            tf.constant(self._bin_pos, dtype=tf.float32),
            (1, 1, 1, self.n_mels)
        )
        gate = tf.math.sigmoid((x - self.breakpoint_b) * self.transition_w)
        gain = (1.0 - gate) * tf.nn.softplus(self.w_log) + gate * tf.nn.softplus(self.w_lin)
        return inputs * gain

    def get_config(self):
        config = super().get_config()
        config.update({'n_mels': self.n_mels})
        return config


def se_block(x, filters, ratio=4, prefix=""):
    """
    Squeeze-and-Excitation channel recalibration (WrenNet's SE Module;
    Hu et al. CVPR 2018).  Adapted for (time, 1, channels) layout:
    global average over time+width → Dense(filters//ratio) → ReLU
    → Dense(filters) → sigmoid → channel-wise scale.
    Adds 2 Dense layers: (C×C/r) + (C/r×C) params per block.
    """
    squeezed = tf.keras.layers.GlobalAveragePooling2D(keepdims=False,
                                                       name=f'{prefix}se_gap')(x)
    excitation = tf.keras.layers.Dense(
        max(1, filters // ratio), activation='relu',
        use_bias=True, name=f'{prefix}se_fc1'
    )(squeezed)
    excitation = tf.keras.layers.Dense(
        filters, activation='sigmoid',
        use_bias=True, name=f'{prefix}se_fc2'
    )(excitation)
    # Reshape to [batch, 1, 1, filters] for broadcasting
    excitation = tf.reshape(excitation, (-1, 1, 1, filters))
    return x * excitation


def causal_tcs_conv(x, filters, kernel, use_bn, stride=1, dilation=1,
                    use_causal=True, prefix=""):
    """
    Time-channel separable conv with optional causal padding.

    Causal mode (WrenNet §2): prepend (kernel-1)*dilation zeros on the time
    axis, then use padding='valid' so no future frame is seen.  This makes the
    conv streaming-safe (no lookahead).

    Non-causal mode (fallback, same as 7a): padding='same'.

    Stride still lives on the pointwise conv (TF DepthwiseConv2D rejects
    unequal strides at execution).
    """
    if use_causal and kernel > 1:
        pad = (kernel - 1) * dilation
        x = tf.keras.layers.ZeroPadding2D(
            padding=((pad, 0), (0, 0)), name=f'{prefix}pad'
        )(x)
        dw_padding = 'valid'
    else:
        dw_padding = 'same'

    x = tf.keras.layers.DepthwiseConv2D(
        (kernel, 1), dilation_rate=(dilation, 1),
        padding=dw_padding, use_bias=False, name=f'{prefix}dw'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), strides=(stride, 1), padding='same',
        use_bias=not use_bn, name=f'{prefix}pw'
    )(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}bn')(x)
    return x


def wren_matchbox_block(x, filters, kernel, sub_blocks, dropout, use_bn,
                         se_ratio, use_causal, block_idx):
    """
    WrenMatch residual block: MatchboxNet TCS residual + WrenNet SE.
    Structure per block:
      skip = identity (or 1×1 projection)
      for r in sub_blocks:
          causal TCSConv → BN → ReLU → Dropout
      Add(y, skip) → ReLU → SE recalibration → Dropout
    SE is applied after the skip-add, recalibrating the merged representation
    (same position as in WrenNet's Conv Block diagram).
    """
    in_ch = x.shape[-1]
    if in_ch == filters:
        skip = x
    else:
        skip = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same', use_bias=not use_bn,
            name=f'b{block_idx}skip'
        )(x)
        if use_bn:
            skip = tf.keras.layers.BatchNormalization(
                name=f'b{block_idx}skipbn')(skip)

    y = x
    for r in range(sub_blocks):
        y = causal_tcs_conv(y, filters, kernel, use_bn,
                             use_causal=use_causal,
                             prefix=f'b{block_idx}r{r}')
        if r < sub_blocks - 1:
            y = tf.keras.layers.ReLU()(y)
            y = tf.keras.layers.Dropout(dropout)(y)

    y = tf.keras.layers.Add(name=f'b{block_idx}add')([y, skip])
    y = tf.keras.layers.ReLU()(y)

    # SE channel recalibration (WrenNet SE Module)
    y = se_block(y, filters, ratio=se_ratio, prefix=f'b{block_idx}')

    y = tf.keras.layers.Dropout(dropout)(y)
    return y


def build_wren_matchbox(input_shape=(TIME_STEPS, 1, 16), num_classes=2,
                         blocks=2, sub_blocks=1, channels=16,
                         epilogue_channels=16, dropout=0.1, use_bn=True,
                         epilogue_dilation=1, se_ratio=4, use_causal=True):
    """
    SEABADNet-WrenMatch fusion model.

    Front-end: WrenNet semi-learnable frequency map
    Prologue:  causal TCS conv (k=9, stride=2) — halves time axis
    Backbone:  B × causal TCS residual blocks with SE recalibration
    Epilogue:  causal TCS conv (k=17, wide receptive field)
    Head:      GAP → Dropout → Dense(2, softmax)

    The causal padding ensures no lookahead: at inference each frame only
    attends to itself and past frames, enabling streaming without buffering.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # WrenNet semi-learnable front-end
    x = SemiLearnableFrequencyMap(n_mels=input_shape[2], name='slfe')(inputs)

    # Causal prologue
    x = causal_tcs_conv(x, channels, 9, use_bn, stride=2,
                         use_causal=use_causal, prefix='p')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # WrenMatch residual blocks (MatchboxNet TCS + WrenNet SE)
    for b in range(blocks):
        kernel = 11 + 2 * b
        x = wren_matchbox_block(x, channels, kernel, sub_blocks, dropout,
                                  use_bn, se_ratio, use_causal, b + 1)

    # Causal epilogue (wide kernel for long-range temporal context)
    x = causal_tcs_conv(x, epilogue_channels, 17, use_bn,
                         dilation=epilogue_dilation,
                         use_causal=use_causal, prefix='e')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc')(x)

    causal_tag = 'c' if use_causal else 'nc'
    return tf.keras.Model(
        inputs, outputs,
        name=f'wmbx{blocks}x{sub_blocks}x{channels}_se{se_ratio}_{causal_tag}'
    )


# ============================================================================
# CUSTOM TRAINING STEP — dual-LR for WrenNet front-end params (same as 7b)
# ============================================================================

class WrenMatchTrainer(tf.keras.Model):
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
        fe_lr = optimizer.learning_rate * self.config.frontend_lr_multiplier
        system, machine = platform.system(), platform.machine()
        if system == 'Darwin' and machine == 'arm64':
            self._fe_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=fe_lr)
        else:
            self._fe_optimizer = tf.keras.optimizers.AdamW(
                learning_rate=fe_lr, weight_decay=1e-4)

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        fe_vars = [v for v in self.base_model.trainable_variables
                   if 'slfe/breakpoint_b' in v.name or 'slfe/transition_w' in v.name]
        fe_var_names = {v.name for v in fe_vars}
        backbone_vars = [v for v in self.base_model.trainable_variables
                         if v.name not in fe_var_names]
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self._loss_fn(y, y_pred)
        grads = tape.gradient(loss, fe_vars + backbone_vars)
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
# DATASET — identical to 7a/7b
# ============================================================================

class SEABADDataset:
    def __init__(self, root_dir, split='train', fraction=1.0, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.files, self.labels = [], []
        random.seed(seed)
        positive_files, negative_files = [], []
        for label, class_name in enumerate(['negative', 'positive']):
            path = os.path.join(root_dir, class_name)
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} does not exist!")
            files = []
            for root, dirs, fs in os.walk(path):
                for f in fs:
                    if f.endswith('.wav'):
                        files.append(os.path.join(root, f))
            if class_name == 'negative':
                negative_files = [(f, label) for f in files]
            else:
                positive_files = [(f, label) for f in files]
        logger.info(f"Found {len(positive_files)} pos, {len(negative_files)} neg")
        random.shuffle(positive_files); random.shuffle(negative_files)
        n_pos, n_neg = len(positive_files), len(negative_files)
        tp, vp = int(n_pos*0.8), int(n_pos*0.9)
        tn, vn = int(n_neg*0.8), int(n_neg*0.9)
        if split == 'train':
            subset = positive_files[:tp] + negative_files[:tn]
        elif split == 'val':
            subset = positive_files[tp:vp] + negative_files[tn:vn]
        else:
            subset = positive_files[vp:] + negative_files[vn:]
        if fraction < 1.0:
            subset = random.sample(subset, int(len(subset)*fraction))
        random.shuffle(subset)
        self.files = [f[0] for f in subset]; self.labels = [f[1] for f in subset]
        logger.info(f"{split}: {len(self.files)} files "
                    f"({sum(1 for l in self.labels if l==0)} neg, "
                    f"{sum(1 for l in self.labels if l==1)} pos)")

    def __len__(self): return len(self.files)
    def get_files_and_labels(self): return self.files, self.labels


def compute_mel_spectrogram(waveform, config):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=config.target_sr, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        fmin=config.mel_fmin, fmax=config.mel_fmax, center=False)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    if mel_spec_db.shape[0] > TIME_STEPS:
        mel_spec_db = mel_spec_db[:TIME_STEPS]
    elif mel_spec_db.shape[0] < TIME_STEPS:
        mel_spec_db = np.pad(mel_spec_db, ((0, TIME_STEPS-mel_spec_db.shape[0]), (0,0)),
                             mode='constant', constant_values=mel_spec_db.min())
    mn, mx = mel_spec_db.min(), mel_spec_db.max()
    return (mel_spec_db - mn) / (mx - mn) if mx - mn > 1e-6 else np.zeros_like(mel_spec_db)


def preprocess_and_cache_mels(dataset_path, config, force_reprocess=False):
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    info_path = cache_dir / 'cache_info.pkl'
    if info_path.exists() and not force_reprocess:
        logger.info(f"Cache at {cache_dir} — skipping.")
        return
    cache_info = {}
    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split}...")
        ds = SEABADDataset(dataset_path, split=split,
                           fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = ds.get_files_and_labels()
        split_dir = cache_dir / split
        split_dir.mkdir(exist_ok=True)
        mels, lbls = [], []
        for fp, lbl in tqdm(zip(file_paths, labels), total=len(file_paths)):
            try:
                wav, sr = librosa.load(fp, sr=None)
                if sr != config.target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=config.target_sr)
                tgt = config.target_length
                if len(wav) > tgt: wav = wav[:tgt]
                elif len(wav) < tgt: wav = np.pad(wav, (0, tgt-len(wav)))
                mels.append(compute_mel_spectrogram(wav, config))
                lbls.append(lbl)
            except Exception as e:
                logger.warning(f"Failed {fp}: {e}")
        mels_arr = np.array(mels, dtype=np.float32)
        lbls_arr = np.array(lbls, dtype=np.int32)
        np.savez_compressed(split_dir / 'mels.npz', mels=mels_arr, labels=lbls_arr)
        cache_info[split] = {'n_samples': len(mels_arr), 'shape': mels_arr.shape}
    with open(info_path, 'wb') as f:
        pickle.dump(cache_info, f)
    logger.info("Cache complete.")


def load_cached_mels(split, config):
    cache_file = Path(config.cache_dir) / split / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    d = np.load(cache_file)
    return d['mels'], d['labels']


def create_tf_dataset_from_cache(split, config, augment=False):
    mels, labels = load_cached_mels(split, config)
    mels = mels[:, :, np.newaxis, :]  # (n, time, 1, n_mels)
    class_counts = {0: int(np.sum(labels==0)), 1: int(np.sum(labels==1))}
    logger.info(f"  {split}: neg={class_counts[0]}, pos={class_counts[1]}")
    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.from_tensor_slices((mels, labels))
    if split == 'train':
        ds = ds.shuffle(len(mels), seed=config.random_seed)

    def to_one_hot(mel, label):
        return mel, tf.one_hot(label, depth=2)
    ds = ds.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and split == 'train':
        n_mels = config.n_mels
        use_specaug = config.use_specaug

        def augment_mel(mel, label):
            mel = tf.clip_by_value(
                mel + tf.random.normal(tf.shape(mel), stddev=0.02), 0.0, 1.0)
            mel = tf.cond(
                tf.random.uniform(()) > 0.5,
                lambda: tf.roll(mel, tf.random.uniform((), -10, 10, tf.int32), axis=0),
                lambda: mel)
            if use_specaug:
                def time_mask(m):
                    w = tf.random.uniform((), 1, 21, dtype=tf.int32)
                    t0 = tf.random.uniform((), 0, TIME_STEPS - w, dtype=tf.int32)
                    mask = tf.concat([tf.ones((t0,1,1)), tf.zeros((w,1,1)),
                                      tf.ones((TIME_STEPS-t0-w,1,1))], axis=0)
                    return m * mask
                def freq_mask(m):
                    w = tf.random.uniform((), 1, 4, dtype=tf.int32)
                    f0 = tf.random.uniform((), 0, n_mels - w, dtype=tf.int32)
                    mask = tf.concat([tf.ones((1,1,f0)), tf.zeros((1,1,w)),
                                      tf.ones((1,1,n_mels-f0-w))], axis=2)
                    return m * mask
                for _ in range(2):
                    mel = tf.cond(tf.random.uniform(()) > 0.5,
                                  lambda: time_mask(mel), lambda: mel)
                mel = tf.cond(tf.random.uniform(()) > 0.5,
                              lambda: freq_mask(mel), lambda: mel)
            mel = tf.ensure_shape(mel, (TIME_STEPS, 1, n_mels))
            return mel, label
        ds = ds.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE), class_counts


# ============================================================================
# LOSS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        fw = tf.pow(1 - pt, gamma)
        aw = y_true[:,1:2] * alpha + y_true[:,0:1] * (1 - alpha)
        return tf.reduce_mean(aw * fw * tf.reduce_sum(ce, axis=-1, keepdims=True))
    return loss


# ============================================================================
# EVALUATOR — identical to 7b
# ============================================================================

class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(history['loss']) + 1)
        for ax, (tr, va, title) in zip(axes.flat, [
            ('loss','val_loss','Loss'), ('auc','val_auc','AUC'),
            ('accuracy','val_accuracy','Accuracy'), ('precision','val_precision','Precision')
        ]):
            if tr in history: ax.plot(epochs, history[tr], label=f'Train')
            if va in history: ax.plot(epochs, history[va], label=f'Val')
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

    def evaluate_model(self, model, test_ds, prefix=''):
        preds, probs, true_labels = [], [], []
        for x, y in test_ds:
            out = model(x, training=False)
            preds.extend(np.argmax(out, axis=1))
            probs.extend(out[:,1].numpy())
            true_labels.extend(np.argmax(y.numpy(), axis=1))
        preds = np.array(preds); probs = np.array(probs); true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probs)
        acc = np.mean(preds == true_labels)
        logger.info(f"{prefix}Acc={acc:.4f}, AUC={auc:.4f}")
        self._plot_confusion_matrix(true_labels, preds, prefix)
        self._plot_roc_curve(true_labels, probs, prefix)
        self._save_classification_report(true_labels, preds, prefix)
        return auc

    def evaluate_tflite(self, tflite_path, test_ds):
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        odet = interp.get_output_details()[0]
        isc, izp = idet['quantization']; osc, ozp = odet['quantization']
        preds, probs, true_labels, times = [], [], [], []
        for inputs, labels in tqdm(test_ds, desc='TFLite eval'):
            inp = inputs.numpy(); lbl = labels.numpy()
            inq = np.round(inp/isc+izp).astype(idet['dtype']) if isc != 0.0 else inp.astype(idet['dtype'])
            for i in range(inp.shape[0]):
                t0 = time.perf_counter()
                interp.set_tensor(idet['index'], inq[i:i+1])
                interp.invoke()
                o = interp.get_tensor(odet['index']).astype(np.float32)
                times.append((time.perf_counter()-t0)*1000)
                if osc != 0.0: o = (o - ozp) * osc
                probs.append(float(o[0,1]))
                preds.append(int(np.argmax(o,axis=1)[0]))
                true_labels.append(int(np.argmax(lbl[i])))
        preds = np.array(preds); probs = np.array(probs); true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probs); acc = np.mean(preds==true_labels)
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
            tp=int(np.sum((p==1)&(true_labels==1))); fp=int(np.sum((p==1)&(true_labels==0)))
            fn=int(np.sum((p==0)&(true_labels==1))); tn=int(np.sum((p==0)&(true_labels==0)))
            rec=tp/(tp+fn) if tp+fn else 0; prec=tp/(tp+fp) if tp+fp else 0
            f1=2*prec*rec/(prec+rec) if prec+rec else 0; fpr=fp/(fp+tn) if fp+tn else 0
            lines.append(f"{tau:.2f} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {fpr:.4f} "
                         f"| {tp} | {fp} | {fn} | {tn}")
        (self.output_dir / 'threshold_sweep.txt').write_text("\n".join(lines)+"\n")
        for line in lines: logger.info(f"  {line}")

    def _plot_confusion_matrix(self, true_labels, preds, prefix):
        cm = confusion_matrix(true_labels, preds)
        ConfusionMatrixDisplay(cm, display_labels=['Neg','Pos']).plot(
            cmap=plt.cm.Blues, values_format='d')
        plt.title(f'{prefix}Confusion Matrix')
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_roc_curve(self, true_labels, probs, prefix):
        fpr, tpr, _ = roc_curve(true_labels, probs)
        auc = roc_auc_score(true_labels, probs)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'AUC={auc:.4f}', linewidth=3, color='darkblue')
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True,alpha=0.3)
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png', dpi=150)
        plt.close()

    def _save_classification_report(self, true_labels, preds, prefix):
        report = classification_report(true_labels, preds,
                                       target_names=['Negative','Positive'],
                                       digits=4, zero_division=0)
        (self.output_dir / f'{prefix}classification_report.txt').write_text(report)


# ============================================================================
# UTILITIES
# ============================================================================

def format_time(s):
    h, m = int(s//3600), int((s%3600)//60)
    return f"{h}h {m:02d}m {s%60:05.2f}s" if h else f"{m}m {s%60:05.2f}s"


def get_git_hash():
    try:
        import subprocess
        return subprocess.check_output(
            ['git','rev-parse','--short','HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def get_base_optimizer(lr):
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        logger.info("Apple Silicon — legacy Adam")
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    logger.info("AdamW")
    return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)


def save_config(config, output_dir, system_info, model_summary=""):
    path = Path(output_dir) / 'config.txt'
    with open(path, 'w') as f:
        f.write("=" * 60 + "\nSEABADNET-WRENMATCH TRAINING CONFIGURATION\n" + "=" * 60 + "\n\n")
        f.write(f"script=7c_wrennet_matchbox_fusion.py\ngit_hash={get_git_hash()}\n\n")
        f.write(f"blocks={config.blocks}\nsub_blocks={config.sub_blocks}\n")
        f.write(f"channels={config.channels}\nepilog_channels={config.epilogue_channels}\n")
        f.write(f"se_ratio={config.se_ratio}\nuse_causal={config.use_causal}\n")
        f.write(f"dropout={config.dropout}\nbatch_norm={config.use_bn}\n")
        f.write(f"specaugment={config.use_specaug}\n")
        f.write(f"frontend_lr_multiplier={config.frontend_lr_multiplier}\n")
        f.write(f"n_mels={config.n_mels}\nn_fft={config.n_fft}\nseed={config.random_seed}\n\n")
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
    config.se_ratio = args.se_ratio
    config.use_causal = not args.no_causal
    config.use_bn = not args.no_bn
    config.use_specaug = not args.no_specaug
    config.frontend_lr_multiplier = args.frontend_lr_mult
    config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'
    platform_tag = 'macos' if platform.system() == 'Darwin' else 'linux'
    config.output_dir = (args.output_dir or
                         f'results/7c_wrennet_matchbox_fusion_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}_{platform_tag}')

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

    causal_str = 'causal' if config.use_causal else 'bidirectional'
    logger.info("=" * 60)
    logger.info("SEABADNet-WrenMatch (WrenNet + MatchboxNet fusion)")
    logger.info(f"B×R×C: {config.blocks}×{config.sub_blocks}×{config.channels}, "
                f"SE ratio: {config.se_ratio}, Convs: {causal_str}")
    logger.info(f"FE LR mult: {config.frontend_lr_multiplier}×, "
                f"BN: {config.use_bn}, SpecAug: {config.use_specaug}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)

    try:
        t_pre = time.time()
        if args.use_cache:
            info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not info_path.exists():
                raise FileNotFoundError(f"Cache not found: {config.cache_dir}")
            logger.info(f"Using cache: {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config,
                                      force_reprocess=args.force_reprocess)
        times = {'preprocessing': time.time() - t_pre}

        train_ds, train_cc = create_tf_dataset_from_cache('train', config, augment=True)
        val_ds, _ = create_tf_dataset_from_cache('val', config)
        test_ds, _ = create_tf_dataset_from_cache('test', config)

        base_model = build_wren_matchbox(
            input_shape=(TIME_STEPS, 1, config.n_mels),
            num_classes=2,
            blocks=config.blocks, sub_blocks=config.sub_blocks,
            channels=config.channels, epilogue_channels=config.epilogue_channels,
            dropout=config.dropout, use_bn=config.use_bn,
            epilogue_dilation=config.epilogue_dilation,
            se_ratio=config.se_ratio, use_causal=config.use_causal
        )

        model_summary_lines = []
        base_model.summary(
            print_fn=lambda x: (logger.info(x), model_summary_lines.append(x)))
        model_summary_str = "\n".join(model_summary_lines)
        (output_dir / 'model_summary.txt').write_text(model_summary_str)
        save_config(config, output_dir, system_info, model_summary_str)

        slfe = base_model.get_layer('slfe')
        logger.info(f"SLFE init: b={slfe.breakpoint_b.numpy()}, "
                    f"w={slfe.transition_w.numpy()}")

        trainer_model = WrenMatchTrainer(base_model, config)
        base_opt = get_base_optimizer(config.learning_rate)
        metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        ]
        trainer_model.compile(optimizer=base_opt, loss=focal_loss(2.0, 0.5),
                               metrics=metrics)

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

        t_train = time.time()
        history = trainer_model.fit(
            train_ds, validation_data=val_ds,
            epochs=config.epochs, callbacks=callbacks, verbose=1)
        times['training'] = time.time() - t_train
        logger.info(f"Training done in {format_time(times['training'])}")

        evaluator = ModelEvaluator(output_dir)
        evaluator.plot_training_history(history.history)

        logger.info(f"SLFE after training: b={slfe.breakpoint_b.numpy()}, "
                    f"w={slfe.transition_w.numpy()}")
        np.save(str(output_dir / 'slfe_weights.npy'), {
            'b': slfe.breakpoint_b.numpy(), 'w': slfe.transition_w.numpy(),
            'w_log': slfe.w_log.numpy(), 'w_lin': slfe.w_lin.numpy()
        })

        logger.info("Evaluating float32...")
        float_auc = evaluator.evaluate_model(base_model, test_ds, prefix='float_')
        base_model.save(str(output_dir / 'best_model.keras'))

        logger.info("Converting to TFLite INT8...")
        def representative_dataset():
            count = 0
            for inputs, _ in val_ds:
                if count >= args.repr_samples: break
                for i in range(inputs.shape[0]):
                    if count >= args.repr_samples: break
                    yield [inputs[i:i+1]]; count += 1

        tflite_model, strategy_success = None, "none"
        for name, params in [
            ("Default quantization", {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'representative_dataset': representative_dataset}),
            ("No quantization", {})
        ]:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
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

        total_params = base_model.count_params()
        total_time = time.time() - start_time

        with open(output_dir / 'results_summary.txt', 'w') as f:
            f.write("=" * 60 + "\nSEABADNET-WRENMATCH RESULTS SUMMARY\n" + "=" * 60 + "\n\n")
            f.write(f"script=7c_wrennet_matchbox_fusion.py\n")
            f.write(f"arch=wrenmatch_{config.blocks}x{config.sub_blocks}x{config.channels}"
                    f"_se{config.se_ratio}_{'c' if config.use_causal else 'nc'}\n")
            f.write(f"n_mels={config.n_mels}\nn_fft={config.n_fft}\nseed={config.random_seed}\n")
            f.write(f"batch_norm={config.use_bn}\nspecaugment={config.use_specaug}\n")
            f.write(f"dropout={config.dropout}\nse_ratio={config.se_ratio}\n")
            f.write(f"use_causal={config.use_causal}\n")
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
        logger.info(f"DONE: WrenMatch s{config.random_seed}")
        logger.info(f"  Params: {total_params:,}  Size: {tflite_size_kb:.2f} KB")
        logger.info(f"  Float AUC: {float_auc:.4f}  TFLite AUC: {tflite_auc:.4f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        elapsed = time.time() - start_time
        (output_dir / 'elapsed.txt').write_text(
            f"Total: {format_time(elapsed)}\nSeconds: {elapsed:.3f}\n")


if __name__ == '__main__':
    main()
