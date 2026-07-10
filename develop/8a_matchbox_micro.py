#!/usr/bin/env python3
"""
7a_matchbox_micro.py: SEABADNet-Matchbox (exploratory, post-ablation)
MatchboxNet-inspired variant (Majumdar & Ginsburg, Interspeech 2020) scaled down
~40x from MatchboxNet-3x1x64. Positioned between Micro (6.23 KB) and Edge
(33 KB): TFLite flatbuffer overhead (~3.5 KB base + ~0.3 KB/op) puts even a
B=1 residual net at 9.3 KB, so the Micro ≤8 KB budget is unreachable for this
op-heavy topology; the default 2x1x16 lands at ~11.7 KB INT8, <1 ms.
Success criterion: beat Micro's INT8 AUC 0.9803 (ideally approach 3f's 0.9846
in-sweep ceiling) at ~1/3 of Edge's flash cost, with ≥0.98 recall at τ.

MatchboxNet features adopted:
  1. 1D time-channel separable convolutions — the n_mels mel bins are treated as
     channels of a 1D time sequence (like MatchboxNet's 64 MFCCs), replacing the
     2D convs of 6b_micro_final.
  2. B×R×C residual blocks with skip connections (default B=2, R=1, C=16 —
     2.1K params vs MatchboxNet-3x1x64's 77K).
  3. Dilated epilogue conv (kernel 17, dilation 2) for a wide temporal receptive
     field before GAP.
  4. SpecAugment-style time/frequency masking on top of the repo-standard
     Gaussian noise + time-shift augmentation (--no_specaug to disable).

Deliberate deviations from the paper, to stay comparable with the ablation chain:
  - Focal loss (γ=2, α=0.5) instead of cross-entropy (repo standard).
  - Adam/AdamW + ReduceLROnPlateau instead of NovoGrad + warmup-hold-decay.
  - Learnable per-mel-bin channel emphasis retained (1D analogue of the
    FrequencyEmphasis layer that the Micro chain locked in).

Note on BatchNorm: CLAUDE.md rejects BN for Micro-scale 2D models (no AUC gain,
quantisation sensitivity), but residual stacks train poorly without it and BN
folds into conv weights at TFLite conversion (zero flash cost). BN is therefore
ON by default here; ablate with --no_bn.

Compatible with both macOS (Metal) and Linux (CUDA).
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
from config import DATASET_PATH, RESULTS_BASE, CACHE_BASE

# Suppress TensorFlow warnings and configure GPU BEFORE importing TensorFlow
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

TIME_STEPS = 184  # 3s @ 16kHz, n_fft=1024, hop=256


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
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
    output_dir: str = f'{RESULTS_BASE}/7a_matchbox_micro'
    cache_dir: str = f'{CACHE_BASE}_fft1024_m16'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    mel_fmin: float = 100.0
    mel_fmax: float = 8000.0
    # MatchboxNet B×R×C structure
    blocks: int = 2          # B
    sub_blocks: int = 1      # R
    channels: int = 16       # C
    epilogue_channels: int = 16
    epilogue_dilation: int = 1
    dropout: float = 0.1
    use_bn: bool = True
    use_specaug: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train SEABADNet-Matchbox (MatchboxNet-inspired Micro variant)')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Number of representative samples for TFLite quantization (default: 500)')
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
                        help='Path to SEABAD dataset directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all mel spectrograms even if cache exists')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached mel spectrograms and skip preprocessing (fails if cache does not exist)')
    parser.add_argument('--n_mels', type=int, default=16,
                        help='Number of mel frequency bins = 1D channel count (default: 16)')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    parser.add_argument('--blocks', type=int, default=2,
                        help='B: number of residual blocks (default: 2)')
    parser.add_argument('--sub_blocks', type=int, default=1,
                        help='R: sub-blocks per residual block (default: 1)')
    parser.add_argument('--channels', type=int, default=16,
                        help='C: channels per block (default: 16)')
    parser.add_argument('--epilogue_channels', type=int, default=16,
                        help='Channels in the epilogue conv (default: 16)')
    parser.add_argument('--epilogue_dilation', type=int, default=1,
                        help='Epilogue conv dilation; 2 = paper-faithful but adds '
                             '4 TFLite ops (~2 KB flash) via SpaceToBatchND (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in all sub-blocks (default: 0.1, Gate 4B)')
    parser.add_argument('--no_bn', action='store_true',
                        help='Disable BatchNorm (ablation; residual training may degrade)')
    parser.add_argument('--no_specaug', action='store_true',
                        help='Disable SpecAugment time/freq masking augmentation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: results/7a_matchbox_micro_fft{n_fft}_m{n_mels}_s{seed})')
    return parser.parse_args()


args = parse_args()

if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================================
# MODEL ARCHITECTURE — MatchboxNet-style 1D time-channel separable residual net
# ============================================================================

class ChannelFrequencyEmphasis(tf.keras.layers.Layer):
    """
    1D analogue of the FrequencyEmphasis layer from the Micro chain: a learnable
    sigmoid gate per mel bin (= per channel). Adds n_mels + 1 parameters.
    Input shape: [batch, time, 1, n_mels]
    """

    def __init__(self, freq_bins=16, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins
        self.freq_weights = self.add_weight(
            name='frequency_weights',
            shape=(1, 1, 1, freq_bins),
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


def tcs_conv(x, filters, kernel, use_bn, stride=1, dilation=1, prefix=""):
    """
    Time-channel separable conv on the (time, 1, channels) layout:
    DepthwiseConv2D over time + 1x1 pointwise Conv2D, then BN.
    Expressed as native 2D ops (not SeparableConv1D) so TFLite converts to a
    single DEPTHWISE_CONV_2D + CONV_2D with no reshape/space-to-batch overhead
    — the flatbuffer metadata otherwise triples the model size at this scale.
    """
    # Stride lives on the pointwise conv: TF's DepthwiseConv2D rejects unequal
    # strides like (2,1) at execution time (TFLite's kernel would accept them).
    x = tf.keras.layers.DepthwiseConv2D(
        (kernel, 1),
        dilation_rate=(dilation, 1),
        padding='same',
        use_bias=False,
        name=f'{prefix}dw'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1),
        strides=(stride, 1),
        padding='same',
        use_bias=not use_bn,
        name=f'{prefix}pw'
    )(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}bn')(x)
    return x


def matchbox_block(x, filters, kernel, sub_blocks, dropout, use_bn, block_idx):
    """
    MatchboxNet residual block: R × (TCSConv-BN-ReLU-Dropout), with a skip
    connection added before the final ReLU-Dropout. Deviation from the paper:
    the skip is identity when channel counts match (saves a 1x1 conv + BN per
    block); a 1x1-conv(+BN) projection is used only on channel change.
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


def build_matchbox(input_shape=(TIME_STEPS, 1, 16), num_classes=2,
                   blocks=2, sub_blocks=1, channels=16, epilogue_channels=16,
                   dropout=0.1, use_bn=True, epilogue_dilation=1):
    """
    SEABADNet-Matchbox-BxRxC. Structure mirrors MatchboxNet (Table 1 of the
    paper) scaled to the Micro budget:
      ChannelEmphasis → Prologue TCSConv(k=9, stride 2)
      → B × [residual block, kernel 11, 13, 15, …]
      → Epilogue TCSConv(k=17) → GAP → Dropout → Dense
    Input: [time=184, 1, n_mels] — mel bins are the channels; the dummy width
    axis makes the time convs native 2D ops (single DEPTHWISE_CONV_2D each).
    Deviations from the paper, both driven by flatbuffer overhead (~0.5 KB per
    op against the 8 KB budget):
      - Conv3 (1x1, 128ch) omitted — the epilogue pointwise conv + Dense
        already mix channels.
      - Epilogue dilation defaults to 1 — TF lowers dilated depthwise via
        SpaceToBatchND (4 extra ops); kernel 17 on the stride-2 time base
        already spans ~0.55 s. Restore the paper's rate with
        epilogue_dilation=2 to trade size for receptive field.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = ChannelFrequencyEmphasis(freq_bins=input_shape[2], name='fe')(inputs)

    # Prologue (MatchboxNet Conv1): stride 2 halves the time axis
    x = tcs_conv(x, channels, 9, use_bn, stride=2, prefix='p')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Residual blocks with growing kernels (MatchboxNet uses 13/15/17)
    for b in range(blocks):
        kernel = 11 + 2 * b
        x = matchbox_block(x, channels, kernel, sub_blocks, dropout, use_bn, b + 1)

    # Epilogue (MatchboxNet Conv2): wide receptive field
    x = tcs_conv(x, epilogue_channels, 17, use_bn, dilation=epilogue_dilation, prefix='e')
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='fc')(x)

    name = f"mbx{blocks}x{sub_blocks}x{channels}"
    return tf.keras.Model(inputs, outputs, name=name)


def get_optimizer(learning_rate: float):
    """
    Apple Silicon: legacy Adam (avoid Metal performance issues).
    Linux/others: AdamW with weight_decay=1e-4.
    """
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info("Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        logger.info(f"Detected {system} {machine} - using AdamW optimizer with weight_decay=1e-4")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


# ============================================================================
# DATASET AND PREPROCESSING (identical mel pipeline to 6b_micro_final)
# ============================================================================

class SEABADDataset:
    """Dataset class for SEABAD that returns file paths only, splits data 80:10:10"""

    def __init__(self, root_dir: str, split='train', fraction=1.0, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.fraction = fraction
        self.files = []
        self.labels = []

        random.seed(seed)

        positive_files = []
        negative_files = []

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

        n_pos = len(positive_files)
        n_neg = len(negative_files)

        train_pos_end = int(n_pos * 0.8)
        val_pos_end = int(n_pos * 0.9)

        train_neg_end = int(n_neg * 0.8)
        val_neg_end = int(n_neg * 0.9)

        if split == 'train':
            subset = positive_files[:train_pos_end] + negative_files[:train_neg_end]
        elif split == 'val':
            subset = positive_files[train_pos_end:val_pos_end] + negative_files[train_neg_end:val_neg_end]
        elif split == 'test':
            subset = positive_files[val_pos_end:] + negative_files[val_neg_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        if fraction < 1.0:
            target_size = int(len(subset) * fraction)
            subset = random.sample(subset, target_size)

        random.shuffle(subset)
        self.files = [f[0] for f in subset]
        self.labels = [f[1] for f in subset]

        logger.info(f"Loaded {len(self.files)} files for {split} "
                    f"({len([l for l in self.labels if l == 0])} negative, "
                    f"{len([l for l in self.labels if l == 1])} positive)")

    def __len__(self) -> int:
        return len(self.files)

    def get_files_and_labels(self) -> Tuple[List[str], List[int]]:
        return self.files, self.labels


def compute_mel_spectrogram(waveform: np.ndarray, config: TrainingConfig) -> np.ndarray:
    """Compute mel spectrogram from waveform with bird-specific settings"""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.target_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.mel_fmin,
        fmax=config.mel_fmax,
        center=False
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T  # (time_steps, n_mels)

    if mel_spec_db.shape[0] > TIME_STEPS:
        mel_spec_db = mel_spec_db[:TIME_STEPS, :]
    elif mel_spec_db.shape[0] < TIME_STEPS:
        pad_width = ((0, TIME_STEPS - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=mel_spec_db.min())

    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if mel_max - mel_min > 1e-6:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)

    return mel_spec_db


def preprocess_and_cache_mels(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """Preprocess all audio files and cache mel spectrograms."""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    cache_info_path = cache_dir / 'cache_info.pkl'
    if cache_info_path.exists() and not force_reprocess:
        logger.info(f"Cache already exists at {cache_dir}. Skipping preprocessing.")
        logger.info("Use --force-reprocess to regenerate cache.")
        return

    logger.info("=" * 60)
    logger.info(f"Preprocessing audio files with n_mels={config.n_mels}...")
    logger.info(f"Frequency range: {config.mel_fmin}-{config.mel_fmax}Hz")
    logger.info("=" * 60)

    splits = ['train', 'val', 'test']
    cache_info = {}

    for split in splits:
        logger.info(f"Processing {split} split...")

        dataset = SEABADDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        mel_specs = []
        valid_labels = []

        for i, (file_path, label) in enumerate(
                tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
            try:
                waveform, sr = librosa.load(file_path, sr=None)

                if sr != config.target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.target_sr)

                if len(waveform) > config.target_length:
                    waveform = waveform[:config.target_length]
                elif len(waveform) < config.target_length:
                    pad = np.zeros(config.target_length - len(waveform))
                    waveform = np.concatenate([waveform, pad])

                mel_spec = compute_mel_spectrogram(waveform, config)

                mel_specs.append(mel_spec)
                valid_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        mel_specs = np.array(mel_specs, dtype=np.float32)
        valid_labels = np.array(valid_labels, dtype=np.int32)

        logger.info(f"  Processed {len(mel_specs)} samples")
        logger.info(f"  Mel spectrogram shape: {mel_specs.shape}")

        cache_file = split_cache_dir / 'mels.npz'
        np.savez_compressed(cache_file, mels=mel_specs, labels=valid_labels)
        logger.info(f"  Saved cache to {cache_file}")

        cache_info[split] = {
            'n_samples': len(mel_specs),
            'shape': mel_specs.shape,
            'cache_file': str(cache_file)
        }

    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


def load_cached_mels(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached mel spectrograms for a split."""
    cache_dir = Path(config.cache_dir)
    cache_file = cache_dir / split / 'mels.npz'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run preprocessing first.")

    data = np.load(cache_file)
    mel_specs = data['mels']
    labels = data['labels']

    logger.info(f"Loaded {len(mel_specs)} cached mel spectrograms for {split}")
    logger.info(f"  Shape: {mel_specs.shape}")

    return mel_specs, labels


def create_tf_dataset_from_cache(split: str, config: TrainingConfig,
                                 augment: bool = False) -> Tuple[tf.data.Dataset, Dict[int, int]]:
    """
    Create tf.data.Dataset from cached mel spectrograms.
    NOTE: unlike the 2D scripts, the cached (time, n_mels) arrays get a dummy
    width axis → (time, 1, n_mels), putting mel bins on the channel axis so the
    model's time convs are native 2D ops.
    """
    mel_specs, labels = load_cached_mels(split, config)

    # (n_samples, time, n_mels) -> (n_samples, time, 1, n_mels)
    mel_specs = mel_specs[:, :, np.newaxis, :]

    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    logger.info(f"  Class distribution - Negative: {class_counts[0]}, Positive: {class_counts[1]}")

    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels))

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(mel_specs), seed=config.random_seed)

    def to_one_hot(mel, label):
        label_onehot = tf.one_hot(label, depth=2)
        return mel, label_onehot

    dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and split == 'train':
        n_mels = config.n_mels
        use_specaug = config.use_specaug

        def augment_mel(mel, label):
            """
            Repo-standard noise + time shift, plus MatchboxNet-style SpecAugment
            masking. Unbatched tensor shape: (time, 1, freq) = (184, 1, n_mels).
            """
            # Add small Gaussian noise
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.02)
            mel = mel + noise
            mel = tf.clip_by_value(mel, 0.0, 1.0)

            # Random time shift along axis 0 (time)
            should_shift = tf.random.uniform(()) > 0.5

            def time_shift(m):
                shift = tf.random.uniform((), minval=-10, maxval=10, dtype=tf.int32)
                return tf.roll(m, shift=shift, axis=0)

            mel = tf.cond(should_shift, lambda: time_shift(mel), lambda: mel)

            if use_specaug:
                # SpecAugment: 2 time masks (≤20 frames) + 1 freq mask (≤3 bins,
                # scaled from the paper's 15/64 to n_mels=16), each applied p=0.5
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

    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.5):
    """Focal loss for handling class imbalance. alpha=0.5 for balanced datasets"""

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)

        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - pt, gamma)

        alpha_weight = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)

        loss = alpha_weight * focal_weight * tf.reduce_sum(cross_entropy, axis=-1, keepdims=True)
        return tf.reduce_mean(loss)

    return loss


# ============================================================================
# TRAINER AND EVALUATOR
# ============================================================================

class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig, class_weights: Dict[int, float] = None):
        self.model = model
        self.config = config
        self.class_weights = class_weights

        self.model.compile(
            optimizer=get_optimizer(config.learning_rate),
            loss=focal_loss(gamma=2.0, alpha=0.5),
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            ]
        )

        if class_weights:
            logger.info(f"Using class weights: {class_weights}")

        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=config.lr_reduction_factor,
                patience=config.lr_patience,
                mode='max',
                min_lr=config.min_lr
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=config.early_stopping_patience,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(Path(config.output_dir) / 'best_model.keras'),
                monitor='val_auc',
                mode='max',
                save_best_only=True
            )
        ]

    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> Dict[str, List[float]]:
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=self.callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        return history.history


class ModelEvaluator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history: Dict[str, List[float]]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(history['loss']) + 1)

        axes[0, 0].plot(epochs, history['loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['auc'], label='Train AUC', color='green', linewidth=2)
        axes[0, 1].plot(epochs, history['val_auc'], label='Val AUC', color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Training and Validation AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, history['accuracy'], label='Train Accuracy', color='purple', linewidth=2)
        axes[1, 0].plot(epochs, history['val_accuracy'], label='Val Accuracy', color='brown', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training and Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, history['precision'], label='Train Precision', color='cyan', linewidth=2)
        axes[1, 1].plot(epochs, history['recall'], label='Train Recall', color='magenta', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Training Precision and Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

    def evaluate_model(self, model: tf.keras.Model, test_dataset: tf.data.Dataset, prefix: str = '') -> float:
        predictions, probabilities = self._get_predictions(model, test_dataset)
        true_labels = self._get_labels(test_dataset)
        self._plot_confusion_matrix(true_labels, predictions, prefix)
        auc = self._plot_roc_curve(true_labels, probabilities, prefix)
        self._save_classification_report(true_labels, predictions, prefix)

        accuracy = np.mean(predictions == true_labels)
        logger.info(f"{prefix}Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

        return auc

    def evaluate_tflite(self, tflite_path: Path, test_dataset: tf.data.Dataset):
        """
        Evaluate TFLite model. Returns: (accuracy, auc, inference_time_ms,
        probabilities, true_labels) — probabilities kept for threshold sweep.
        """
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        if 'quantization' in input_details:
            input_scale, input_zero_point = input_details['quantization']
        else:
            input_scale, input_zero_point = 0.0, 0

        if 'quantization' in output_details:
            output_scale, output_zero_point = output_details['quantization']
        else:
            output_scale, output_zero_point = 0.0, 0

        logger.info(f"TFLite Input: dtype={input_details['dtype']}, scale={input_scale}, zero_point={input_zero_point}")
        logger.info(
            f"TFLite Output: dtype={output_details['dtype']}, scale={output_scale}, zero_point={output_zero_point}")

        predictions = []
        probabilities = []
        true_labels = []
        inference_times = []

        logger.info("Evaluating TFLite model...")
        for inputs, labels in tqdm(test_dataset, desc="TFLite inference"):
            inputs = inputs.numpy()
            labels = labels.numpy()
            batch_size = inputs.shape[0]

            if input_scale != 0.0:
                inputs_quantized = np.round(inputs / input_scale + input_zero_point).astype(input_details['dtype'])
            else:
                inputs_quantized = inputs.astype(input_details['dtype'])

            for i in range(batch_size):
                input_data = inputs_quantized[i:i + 1]

                start_time = time.perf_counter()
                interpreter.set_tensor(input_details['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                inference_times.append((time.perf_counter() - start_time) * 1000)

                if output_scale != 0.0:
                    output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
                else:
                    output_float = output_data.astype(np.float32)

                prob_positive = float(output_float[0, 1])
                pred = int(np.argmax(output_float, axis=1)[0])

                predictions.append(pred)
                probabilities.append(prob_positive)
                true_labels.append(int(np.argmax(labels[i])))

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)

        auc = roc_auc_score(true_labels, probabilities)
        acc = np.mean(predictions == true_labels)
        avg_inference_time = np.mean(inference_times)

        logger.info(f"TFLite Test Acc: {acc:.4f}, AUC: {auc:.4f}")
        logger.info(f"TFLite Avg Inference Time: {avg_inference_time:.2f}ms per sample")

        self._plot_confusion_matrix(true_labels, predictions, prefix='tflite_')
        self._plot_roc_curve(true_labels, probabilities, prefix='tflite_')
        self._save_classification_report(true_labels, predictions, prefix='tflite_')

        return acc, auc, avg_inference_time, probabilities, true_labels

    def threshold_sweep(self, probabilities: np.ndarray, true_labels: np.ndarray,
                        taus=(0.25, 0.30, 0.34, 0.40, 0.50)):
        """GPU Validation Protocol sweep: recall/precision/f1/fpr at each τ."""
        lines = ["tau | recall | precision | f1 | fpr | tp | fp | fn | tn"]
        rows = []
        for tau in taus:
            preds = (probabilities >= tau).astype(int)
            tp = int(np.sum((preds == 1) & (true_labels == 1)))
            fp = int(np.sum((preds == 1) & (true_labels == 0)))
            fn = int(np.sum((preds == 0) & (true_labels == 1)))
            tn = int(np.sum((preds == 0) & (true_labels == 0)))
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            lines.append(f"{tau:.2f} | {recall:.4f} | {precision:.4f} | {f1:.4f} | {fpr:.4f} "
                         f"| {tp} | {fp} | {fn} | {tn}")
            rows.append((tau, recall, precision, f1, fpr))

        sweep_path = self.output_dir / 'threshold_sweep.txt'
        with open(sweep_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Threshold sweep saved to {sweep_path}")
        for line in lines:
            logger.info(f"  {line}")
        return rows

    def _get_predictions(self, model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        probabilities = []
        for inputs, _ in dataset:
            outputs = model(inputs, training=False)
            predictions.extend(np.argmax(outputs, axis=1))
            probabilities.extend(outputs[:, 1].numpy())
        return np.array(predictions), np.array(probabilities)

    def _get_labels(self, dataset: tf.data.Dataset) -> np.ndarray:
        labels = []
        for _, lbl in dataset:
            lbl_indices = np.argmax(lbl.numpy(), axis=1)
            labels.extend(lbl_indices)
        return np.array(labels)

    def _plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray, prefix: str):
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'{prefix}Confusion Matrix')
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_roc_curve(self, true_labels: np.ndarray, probabilities: np.ndarray, prefix: str) -> float:
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=3, color='darkblue')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{prefix}Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png', dpi=150)
        plt.close()
        return auc

    def _save_classification_report(self, true_labels: np.ndarray, predictions: np.ndarray, prefix: str):
        report = classification_report(
            true_labels, predictions,
            target_names=['Negative', 'Positive'],
            digits=4,
            zero_division=0
        )
        with open(self.output_dir / f'{prefix}classification_report.txt', 'w') as f:
            f.write(report)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:05.2f}s"
    else:
        return f"{minutes}m {secs:05.2f}s"


def save_elapsed_time(start_time: float, output_dir: Path):
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_str = f"Total execution time: {format_time(elapsed_seconds)}\n"
    elapsed_str += f"Total seconds: {elapsed_seconds:.3f}\n"
    with open(output_dir / 'elapsed.txt', 'w') as f:
        f.write(elapsed_str)
    logger.info(f"Script completed in {format_time(elapsed_seconds)}")


def get_git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def save_config(config: TrainingConfig, output_dir: Path, args, system_info: dict, model_summary: str = ""):
    """Save training configuration and system information as text"""
    config_path = output_dir / 'config.txt'

    with open(config_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEABADNET-MATCHBOX TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("script=7a_matchbox_micro.py\n")
        f.write(f"git_hash={get_git_hash()}\n\n")

        f.write("Model Information (MatchboxNet BxRxC):\n")
        f.write(f"  blocks (B): {config.blocks}\n")
        f.write(f"  sub_blocks (R): {config.sub_blocks}\n")
        f.write(f"  channels (C): {config.channels}\n")
        f.write(f"  epilogue_channels: {config.epilogue_channels}\n")
        f.write(f"  epilogue_dilation: {config.epilogue_dilation}\n")
        f.write(f"  dropout: {config.dropout}\n")
        f.write(f"  batch_norm: {config.use_bn}\n")
        f.write(f"  specaugment: {config.use_specaug}\n")
        f.write(f"  n_mels (= 1D channels): {config.n_mels}\n")
        f.write(f"  Frequency Range: {config.mel_fmin}-{config.mel_fmax} Hz\n")
        f.write("\n")

        f.write("System Information:\n")
        f.write(f"  Platform: {system_info['platform']}\n")
        f.write(f"  Machine: {system_info['machine']}\n")
        f.write(f"  Python Version: {system_info['python_version']}\n")
        f.write(f"  TensorFlow Version: {system_info['tensorflow_version']}\n")
        f.write(f"  GPU Available: {system_info['gpu_available']}\n")
        if system_info['gpu_devices']:
            f.write(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}\n")
        f.write("\n")

        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  Random Seed: {config.random_seed}\n")
        f.write("\n")

        f.write("Audio Configuration:\n")
        f.write(f"  Target Sample Rate: {config.target_sr} Hz\n")
        f.write(f"  Target Length: {config.target_length} samples ({config.target_length / config.target_sr:.1f}s)\n")
        f.write(f"  n_fft: {config.n_fft}\n")
        f.write(f"  hop_length: {config.hop_length}\n")
        f.write("\n")

        f.write("Dataset Information:\n")
        f.write(f"  Dataset: SEABAD\n")
        f.write(f"  Path: {config.dataset_path}\n")
        f.write(f"  Split: {config.train_ratio:.0%} train, {config.val_ratio:.0%} val, {config.test_ratio:.0%} test\n")
        f.write("\n")

        if model_summary:
            f.write("Model Summary:\n")
            f.write(model_summary)
            f.write("\n")

    logger.info(f"Saved configuration to {config_path}")


# ============================================================================
# MAIN FUNCTION
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

    config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'
    platform_tag = 'macos' if platform.system() == 'Darwin' else 'linux'
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f'results/7a_matchbox_micro_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}_{platform_tag}'

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    system_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'librosa_version': librosa.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')],
    }

    logger.info("=" * 60)
    logger.info("SEABADNet-Matchbox TRAINING")
    logger.info("=" * 60)
    logger.info(f"B x R x C: {config.blocks} x {config.sub_blocks} x {config.channels}")
    logger.info(f"n_mels: {config.n_mels}, BN: {config.use_bn}, SpecAug: {config.use_specaug}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 60)

    try:
        preprocess_start = time.time()
        if args.use_cache:
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found at {config.cache_dir}. Cannot use --use_cache.")
            logger.info(f"Using cached mel spectrograms from {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config, force_reprocess=args.force_reprocess)
        times = {'preprocessing': time.time() - preprocess_start}
        logger.info(f"Preprocessing completed in {format_time(times['preprocessing'])}")

        logger.info("Creating tf.data datasets...")
        train_dataset, train_class_counts = create_tf_dataset_from_cache('train', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset_from_cache('val', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset_from_cache('test', config, augment=False)

        class_weights = None
        logger.info(f"Train class distribution: {train_class_counts}")

        model = build_matchbox(
            input_shape=(TIME_STEPS, 1, config.n_mels),
            num_classes=2,
            blocks=config.blocks,
            sub_blocks=config.sub_blocks,
            channels=config.channels,
            epilogue_channels=config.epilogue_channels,
            dropout=config.dropout,
            use_bn=config.use_bn,
            epilogue_dilation=config.epilogue_dilation
        )

        logger.info("=" * 60)
        logger.info("Model Architecture:")
        model_summary_lines = []
        model.summary(print_fn=lambda x: (logger.info(x), model_summary_lines.append(x)))
        logger.info("=" * 60)

        model_summary_str = "\n".join(model_summary_lines)

        with open(output_dir / 'model_summary.txt', 'w') as f:
            f.write(model_summary_str)

        save_config(config, output_dir, args, system_info, model_summary_str)

        trainer = ModelTrainer(model, config, class_weights)
        evaluator = ModelEvaluator(output_dir)

        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        evaluator.plot_training_history(history)

        logger.info("=" * 60)
        logger.info("Evaluating Float32 Model:")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')

        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved TensorFlow model to {output_dir / 'best_model.keras'}")

        logger.info("=" * 60)
        logger.info("Converting to TFLite...")

        def representative_dataset():
            count = 0
            for inputs, _ in val_dataset:
                if count >= args.repr_samples:
                    break
                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    if count >= args.repr_samples:
                        break
                    yield [inputs[i:i + 1]]
                    count += 1

        tflite_model = None
        conversion_strategies = [
            ("Default quantization", {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'representative_dataset': representative_dataset
            }),
            ("Float16 quantization", {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'representative_dataset': representative_dataset,
                'supported_types': [tf.float16]
            }),
            ("No quantization", {})
        ]

        for strategy_name, converter_params in conversion_strategies:
            try:
                logger.info(f"Trying {strategy_name}...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                for key, value in converter_params.items():
                    setattr(converter, key, value)

                tflite_model = converter.convert()
                strategy_success = strategy_name
                logger.info(f"{strategy_name} succeeded!")
                break
            except Exception as e:
                logger.warning(f"{strategy_name} failed: {e}")
                continue

        if tflite_model is None:
            logger.error("All TFLite conversion strategies failed!")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            strategy_success = "Fallback (no quantization)"

        tflite_path = output_dir / "model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"Saved TFLite model ({strategy_success}) to {tflite_path} ({tflite_size_kb:.2f} KB)")

        logger.info("=" * 60)
        logger.info("Evaluating TFLite Model:")
        tflite_acc, tflite_auc, tflite_time, tflite_probs, tflite_labels = \
            evaluator.evaluate_tflite(tflite_path, test_dataset)

        logger.info("=" * 60)
        logger.info("Threshold sweep (TFLite probabilities):")
        evaluator.threshold_sweep(tflite_probs, tflite_labels)

        total_time = time.time() - start_time
        total_params = model.count_params()
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SEABADNET-MATCHBOX RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("script=7a_matchbox_micro.py\n")
            f.write(f"arch=matchbox_{config.blocks}x{config.sub_blocks}x{config.channels}\n")
            f.write(f"n_mels={config.n_mels}\n")
            f.write(f"n_fft={config.n_fft}\n")
            f.write(f"seed={config.random_seed}\n")
            f.write(f"batch_norm={config.use_bn}\n")
            f.write(f"specaugment={config.use_specaug}\n")
            f.write(f"dropout={config.dropout}\n\n")

            f.write(f"float_auc={float_auc:.4f}\n")
            f.write(f"tflite_strategy={strategy_success}\n")
            f.write(f"tflite_accuracy={tflite_acc:.4f}\n")
            f.write(f"tflite_auc={tflite_auc:.4f}\n")
            f.write(f"tflite_latency_ms={tflite_time:.2f}\n")
            f.write(f"tflite_size_kb={tflite_size_kb:.2f}\n")
            f.write(f"total_params={total_params}\n")

            if float_auc > 0:
                auc_degradation = float_auc - tflite_auc
                f.write(f"auc_degradation={auc_degradation:.4f}\n")
                f.write(f"auc_degradation_pct={auc_degradation / float_auc * 100:.2f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Timing Breakdown\n")
            f.write("=" * 60 + "\n")
            if 'preprocessing' in times:
                f.write(f"Preprocessing: {format_time(times['preprocessing'])}\n")
            if 'training' in times:
                f.write(f"Training: {format_time(times['training'])}\n")
            f.write(f"Total: {format_time(total_time)}\n")

        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY:")
        logger.info(f"  Model: SEABADNet-Matchbox {config.blocks}x{config.sub_blocks}x{config.channels}")
        logger.info(f"  Params: {total_params:,}")
        logger.info(f"  Float32 AUC: {float_auc:.4f}")
        logger.info(f"  TFLite AUC: {tflite_auc:.4f}")
        logger.info(f"  TFLite Size: {tflite_size_kb:.2f} KB")
        logger.info(f"  TFLite Inference: {tflite_time:.2f}ms")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        save_elapsed_time(start_time, output_dir)


if __name__ == '__main__':
    main()
