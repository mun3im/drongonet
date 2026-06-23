#!/usr/bin/env python3
"""
6a_nano_final.py: SEABADNet-Nano
Smallest Micro variant (5.41 KB INT8, 763 params). Use when size matters
more than hitting the 0.98 recall target. Superseded by 6b_micro_final
for deployments that require ≥0.98 recall.
- FrequencyEmphasis → Conv(6) → MaxPool → Conv(12) → GAP → Dropout → Dense
- Focal loss, dropout=0.1, n_mels=16, n_fft=1024
Compatible with both macOS (Metal) and Linux (CUDA)
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
from config import DATASET_PATH, TINYCHIRP_PATH, RESULTS_BASE, CACHE_BASE

# Suppress TensorFlow warnings and configure GPU BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

import tensorflow as tf

# Configure GPU memory growth BEFORE any TensorFlow operations
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    pass  # Silently ignore if already initialized or GPU not available

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter issues
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve
)
from tqdm import tqdm
import random
import librosa

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    epochs: int = 100
    fraction: float = 1  # Percentage of dataset to use
    batch_size: int = 32
    learning_rate: float = 0.001
    target_sr: int = 16000
    target_length: int = 16000 * 3  # 3 seconds @ 16kHz = 48000 samples
    # Mel spectrogram parameters - OPTIMIZED FOR BIRDS
    n_mels: int = 16  # REDUCED from 64 to 16 for bigger frequency emphasis impact!
    n_fft: int = 1024  # FFT window size
    hop_length: int = 256  # Hop length for STFT (corrected to 256, crop to 184 frames)
    # Learning rate schedule parameters
    lr_patience: int = 5  # Patience for learning rate reduction
    lr_reduction_factor: float = 0.5  # Factor to reduce LR by
    min_lr: float = 1e-5  # Minimum learning rate
    early_stopping_patience: int = 15  # Early stopping patience (3x LR patience)
    random_seed: int = 42
    # Path configurations
    dataset_path: str = DATASET_PATH
    output_dir: str = f'{RESULTS_BASE}/6a_nano_final'
    cache_dir: str = f'{CACHE_BASE}_fft512_m16'
    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Bird-specific frequency range
    mel_fmin: float = 100.0  # Include dove frequencies
    mel_fmax: float = 8000.0  # Bird upper limit


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SEABAD Low Power models')
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
                        help='Number of mel frequency bins (default: 16 for SEABAD_LP')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    parser.add_argument('--version', type=str, default='Opt',
                        choices=['LP', 'FE', 'Opt'],
                        help='Model version: LP (original), FE (with frequency emphasis), Opt (optimized)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: results/seabadnet_nano_fft{n_fft}_m{n_mels}_s{seed})')
    return parser.parse_args()


args = parse_args()

# Suppress TensorFlow warnings and configure GPU BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================================
# FREQUENCY EMPHASIS LAYERS
# ============================================================================

class FrequencyEmphasis(tf.keras.layers.Layer):
    """
    Learnable frequency weighting for bird sounds
    Learnable frequency weighting: teaches model which frequency bands matter.
    Adds only ~16-64 parameters.
    """

    def __init__(self, freq_bins=16, init_center=0.4, init_width=0.2, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins

        # Learnable frequency weights (VERY SMALL!)
        # Shape: [1, 1, freq_bins, 1] to broadcast across batch and time
        # CHANGED: Renamed from 'weights' to 'freq_weights' to avoid conflict
        self.freq_weights = self.add_weight(
            name='frequency_weights',
            shape=(1, 1, freq_bins, 1),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            dtype=tf.float32
        )

        # Learnable scaling factor (how "confident" the weights are)
        self.scale = self.add_weight(
            name='scale',
            shape=(1,),
            initializer=tf.constant_initializer(3.0),
            trainable=True,
            dtype=tf.float32
        )

        logger.info(f"FrequencyEmphasis layer created with {freq_bins} frequency bins")
        logger.info(f"  Adds only {freq_bins + 1} parameters (~{freq_bins + 1} total)")

    def call(self, inputs, training=None):
        """
        Apply frequency weighting to input spectrogram
        inputs shape: [batch, time_steps, freq_bins, channels]
        returns: weighted spectrogram
        """
        # Apply learned weights with sigmoid activation
        # Sigmoid ensures weights stay in reasonable range (0-1 after scaling)
        weight_map = tf.math.sigmoid(self.freq_weights * self.scale)

        # Broadcast and apply
        return inputs * weight_map

    def get_config(self):
        config = super().get_config()
        config.update({
            'freq_bins': self.freq_bins,
        })
        return config

    def plot_weights(self, output_path):
        """Visualize learned frequency weights"""
        try:
            weights = self.freq_weights.numpy().flatten()  # CHANGED: self.freq_weights
            scale = self.scale.numpy()[0]

            # Apply sigmoid to get actual weights
            actual_weights = 1 / (1 + np.exp(-weights * scale))

            plt.figure(figsize=(10, 4))

            # Plot weights
            plt.subplot(1, 2, 1)
            bins = np.arange(len(actual_weights))
            plt.bar(bins, actual_weights, alpha=0.7, color='steelblue')
            plt.xlabel('Frequency Bin (0=lowest, 15=highest)')
            plt.ylabel('Weight (importance)')
            plt.title('Learned Frequency Weights')
            plt.grid(True, alpha=0.3)

            # Add frequency labels (assuming 0-8000Hz, 16 bins = 500Hz/bin)
            if len(actual_weights) == 16:
                # Show Hz values for key bins
                xticks = [0, 3, 7, 11, 15]  # 0Hz, 1500Hz, 3500Hz, 5500Hz, 8000Hz
                xtick_labels = ['0Hz', '1.5k', '3.5k', '5.5k', '8k']
                plt.xticks(xticks, xtick_labels)

            # Plot as heatmap for intuition
            plt.subplot(1, 2, 2)
            heatmap = actual_weights.reshape(1, -1)
            plt.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Weight')
            plt.title('Frequency Emphasis Heatmap')
            plt.xlabel('Frequency Bin')
            plt.yticks([])

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            logger.info(f"Frequency weights saved to {output_path}")
            logger.info(f"Scale factor: {scale:.3f}")
            logger.info(f"Max weight: {np.max(actual_weights):.3f} at bin {np.argmax(actual_weights)}")
            logger.info(f"Min weight: {np.min(actual_weights):.3f} at bin {np.argmin(actual_weights)}")

            # Save weights to file
            np.save(str(output_path).replace('.png', '.npy'), actual_weights)

        except Exception as e:
            logger.warning(f"Could not plot frequency weights: {e}")


class AdaptiveFrequencyEmphasis(tf.keras.layers.Layer):
    def __init__(self, freq_bins=32, num_bands=3, **kwargs):
        super().__init__(**kwargs)
        # ... rest of init ...

        # CHANGED: All these should use different names
        self.band_centers_param = self.add_weight(  # CHANGED
            name='band_centers',
            shape=(num_bands,),
            initializer=tf.constant_initializer(initial_centers),
            trainable=True
        )

        self.band_widths_param = self.add_weight(  # CHANGED
            name='band_widths',
            shape=(num_bands,),
            initializer=tf.constant_initializer(0.15),
            trainable=True
        )

        self.band_weights_param = self.add_weight(  # CHANGED
            name='band_weights',
            shape=(num_bands,),
            initializer=tf.constant_initializer(1.0),
            trainable=True
        )

    def call(self, inputs, training=None):
        # Create frequency positions [0, 1] across bins
        freq_positions = tf.linspace(0.0, 1.0, self.freq_bins)
        freq_positions = tf.reshape(freq_positions, [1, 1, self.freq_bins, 1])

        # Create Gaussian-like bands
        combined = tf.zeros_like(freq_positions)
        for i in range(self.num_bands):
            center = self.band_centers_param[i]  # CHANGED
            width = self.band_widths_param[i]  # CHANGED
            weight = self.band_weights_param[i]  # CHANGED

            # Gaussian band: exp(-(x - center)² / (2 * width²))
            band = tf.exp(-tf.square(freq_positions - center) / (2 * tf.square(width + 1e-6)))
            combined += weight * band

        # Apply with sigmoid
        emphasis = tf.math.sigmoid(combined)

        return inputs * emphasis

def analyze_frequency_weights(model, output_dir: Path):
    """
    Extract and visualize learned frequency weights from the model
    """
    try:
        # Find frequency emphasis layer
        for layer in model.layers:
            if hasattr(layer, 'freq_weights'):  # CHANGED: Check for freq_weights
                weights = layer.freq_weights.numpy().flatten()  # CHANGED
                scale = layer.scale.numpy()[0]

                # Apply sigmoid to get actual weights
                actual_weights = 1 / (1 + np.exp(-weights * scale))

                plt.figure(figsize=(10, 4))

                # Plot weights
                bins = np.arange(len(actual_weights))
                plt.bar(bins, actual_weights, alpha=0.7, color='steelblue')
                plt.xlabel('Frequency Bin (0=lowest, 15=highest)')
                plt.ylabel('Weight (importance)')
                plt.title('Learned Frequency Weights')
                plt.grid(True, alpha=0.3)

                # Add frequency labels
                if len(actual_weights) == 16:
                    xticks = [0, 3, 7, 11, 15]
                    xtick_labels = ['0Hz', '1.5k', '3.5k', '5.5k', '8k']
                    plt.xticks(xticks, xtick_labels)

                plt.tight_layout()
                weights_path = output_dir / 'frequency_weights.png'
                plt.savefig(weights_path, dpi=150)
                plt.close()

                logger.info(f"Frequency weights saved to {weights_path}")
                logger.info(f"Scale factor: {scale:.3f}")
                logger.info(f"Max weight: {np.max(actual_weights):.3f} at bin {np.argmax(actual_weights)}")
                logger.info(f"Min weight: {np.min(actual_weights):.3f} at bin {np.argmin(actual_weights)}")

                # Save weights to file
                np.save(str(weights_path).replace('.png', '.npy'), actual_weights)
                break
        else:
            logger.info("No frequency emphasis layer found in model")
    except Exception as e:
        logger.warning(f"Could not analyze frequency weights: {e}")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_cnn_mel_low_power(input_shape=(184, 16, 1), num_classes=2):
    """
    SEABADNet-Nano base: ultra-lightweight Conv2D model.
    
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Lightweight block 1
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # → (92, 8, 8)

    # Lightweight block 2
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # → (16,)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="SEABAD_Low_Power")


def build_cnn_mel_low_power_freq_emph(input_shape=(184, 16, 1), num_classes=2):
    """
    SEABADNet-Nano with frequency emphasis.
    
    
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # 1. Frequency Emphasis (ONLY change from V1.0)
    x = FrequencyEmphasis(freq_bins=input_shape[1], name='frequency_emphasis')(inputs)

    # 2. Rest is exactly the same as V1.0
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # → (92, 8, 8)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # → (16,)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="SEABAD_Low_Power_Freq_Emph")
    return model


def build_cnn_mel_low_power_optimized(input_shape=(184, 16, 1), num_classes=2):
    """
    SEABADNet-Nano (final): SeparableConv2D + frequency emphasis + GAP + focal loss.
    
    
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # 1. Frequency Emphasis with visualization capability
    x = FrequencyEmphasis(freq_bins=input_shape[1], name='frequency_emphasis')(inputs)

    # 2. First conv block - optimized for n_mels=16
    # With only 16 frequency bins, we can use fewer filters
    x = tf.keras.layers.Conv2D(
        6,  # Reduced from 8 (saves parameters)
        (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # → (92, 8, 6)

    # 3. Second conv block - slightly more filters
    x = tf.keras.layers.Conv2D(
        12,  # Balanced between capacity and parameters
        (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    # 4. Global pooling and dropout for regularization
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # → (12,)
    x = tf.keras.layers.Dropout(0.1)(x)

    # 5. Final classification
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="SEABAD_Low_Power_Optimized")
    return model


def get_optimizer(learning_rate: float):
    """
    Get appropriate optimizer based on platform.
    - Apple Silicon: legacy Adam (avoid Metal performance issues)
    - Linux/others: AdamW with weight_decay=1e-4 (better regularization)
    """
    system = platform.system()
    machine = platform.machine()

    # Check if running on Apple Silicon (arm64)
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info(f"Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        logger.info(f"Detected {system} {machine} - using AdamW optimizer with weight_decay=1e-4")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


# ============================================================================
# DATASET AND PREPROCESSING (UPDATED FOR BIRD-SPECIFIC MEL)
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

        # SEABAD uses 'positive' and 'negative' folders
        positive_files = []
        negative_files = []

        for label, class_name in enumerate(['negative', 'positive']):
            path = os.path.join(root_dir, class_name)
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} does not exist!")
            # Recursively find all .wav files in subdirectories
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

        # Shuffle before splitting
        random.shuffle(positive_files)
        random.shuffle(negative_files)

        # Split 80:10:10 for each class
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
        """Return lists of file paths and labels"""
        return self.files, self.labels


def compute_mel_spectrogram(waveform: np.ndarray, config: TrainingConfig) -> np.ndarray:
    """
    Compute mel spectrogram from waveform with BIRD-SPECIFIC settings
    """
    # Compute mel spectrogram using librosa with bird-specific frequency range
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.target_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.mel_fmin,  # Include low frequencies for doves
        fmax=config.mel_fmax,  # Upper limit for birds
        center=False
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Transpose to (time, freq) format and normalize
    mel_spec_db = mel_spec_db.T  # Now shape is (time_steps, n_mels)

    # Ensure exactly 184 time steps (for 3s @ 16kHz with hop_length=256)
    if mel_spec_db.shape[0] > 184:
        mel_spec_db = mel_spec_db[:184, :]
    elif mel_spec_db.shape[0] < 184:
        pad_width = ((0, 184 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=mel_spec_db.min())

    # Normalize to [0, 1] range per sample
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    if mel_max - mel_min > 1e-6:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)

    return mel_spec_db


def preprocess_and_cache_mels(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """
    Preprocess all audio files and cache mel spectrograms.
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Check if cache already exists
    cache_info_path = cache_dir / 'cache_info.pkl'
    if cache_info_path.exists() and not force_reprocess:
        logger.info(f"Cache already exists at {cache_dir}. Skipping preprocessing.")
        logger.info(f"Use --force-reprocess to regenerate cache.")
        return

    logger.info("=" * 60)
    logger.info(f"Preprocessing audio files with n_mels={config.n_mels}...")
    logger.info(f"Frequency range: {config.mel_fmin}-{config.mel_fmax}Hz")
    logger.info("=" * 60)

    splits = ['train', 'val', 'test']
    cache_info = {}

    for split in splits:
        logger.info(f"Processing {split} split...")

        # Get file paths
        dataset = SEABADDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        # Create split cache directory
        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        mel_specs = []
        valid_labels = []

        for i, (file_path, label) in enumerate(
                tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
            try:
                # Load audio
                waveform, sr = librosa.load(file_path, sr=None)

                # Resample if needed
                if sr != config.target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.target_sr)

                # Pad or truncate to target length
                if len(waveform) > config.target_length:
                    waveform = waveform[:config.target_length]
                elif len(waveform) < config.target_length:
                    pad = np.zeros(config.target_length - len(waveform))
                    waveform = np.concatenate([waveform, pad])

                # Compute mel spectrogram with bird-specific settings
                mel_spec = compute_mel_spectrogram(waveform, config)

                mel_specs.append(mel_spec)
                valid_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        # Convert to numpy arrays
        mel_specs = np.array(mel_specs, dtype=np.float32)  # Shape: (n_samples, time_steps, n_mels)
        valid_labels = np.array(valid_labels, dtype=np.int32)

        logger.info(f"  Processed {len(mel_specs)} samples")
        logger.info(f"  Mel spectrogram shape: {mel_specs.shape}")

        # Save to cache
        cache_file = split_cache_dir / 'mels.npz'
        np.savez_compressed(cache_file, mels=mel_specs, labels=valid_labels)
        logger.info(f"  Saved cache to {cache_file}")

        cache_info[split] = {
            'n_samples': len(mel_specs),
            'shape': mel_specs.shape,
            'cache_file': str(cache_file)
        }

    # Save cache info
    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


def load_cached_mels(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached mel spectrograms for a split.
    """
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
    """
    # Load cached data
    mel_specs, labels = load_cached_mels(split, config)

    # Add channel dimension: (n_samples, time, freq) -> (n_samples, time, freq, 1)
    mel_specs = mel_specs[..., np.newaxis]

    # Count samples per class
    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    logger.info(f"  Class distribution - Negative: {class_counts[0]}, Positive: {class_counts[1]}")

    # Create dataset on CPU to avoid GPU memory issues
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels))

    # Shuffle if training
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(mel_specs), seed=config.random_seed)

    # Convert labels to one-hot encoding
    def to_one_hot(mel, label):
        label_onehot = tf.one_hot(label, depth=2)
        return mel, label_onehot

    dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation if requested (for training)
    if augment and split == 'train':
        def augment_mel(mel, label):
            """
            SIMPLIFIED augmentation - noise + time shift.
            Unbatched tensor shape: (time, freq, channel) = (184, 16, 1).
            """
            # Add small Gaussian noise
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.02)
            mel = mel + noise
            mel = tf.clip_by_value(mel, 0.0, 1.0)

            # Random time shift along axis 0 (time). Previously axis=1 (freq) by mistake.
            should_shift = tf.random.uniform(()) > 0.5

            def time_shift(mel):
                shift = tf.random.uniform((), minval=-10, maxval=10, dtype=tf.int32)
                return tf.roll(mel, shift=shift, axis=0)

            mel = tf.cond(should_shift,
                          lambda: time_shift(mel),
                          lambda: mel)

            return mel, label

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.5):
    """
    Focal loss for handling class imbalance.
    alpha=0.5 for balanced datasets
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal weight
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - pt, gamma)

        # Apply alpha weighting
        alpha_weight = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)

        # Combine focal weight and alpha weight
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

        # Compile with focal loss
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

        # Callbacks
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

        # Loss
        axes[0, 0].plot(epochs, history['loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # AUC
        axes[0, 1].plot(epochs, history['auc'], label='Train AUC', color='green', linewidth=2)
        axes[0, 1].plot(epochs, history['val_auc'], label='Val AUC', color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Training and Validation AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Accuracy
        axes[1, 0].plot(epochs, history['accuracy'], label='Train Accuracy', color='purple', linewidth=2)
        axes[1, 0].plot(epochs, history['val_accuracy'], label='Val Accuracy', color='brown', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training and Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Precision/Recall
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

        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels)
        logger.info(f"{prefix}Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

        return auc

    def evaluate_tflite(self, tflite_path: Path, test_dataset: tf.data.Dataset) -> Tuple[float, float, float]:
        """
        Evaluate TFLite model with batched inference for efficiency.
        Returns: (accuracy, auc, inference_time_ms)
        """
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Get quantization parameters if they exist
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

            # Quantize inputs if needed
            if input_scale != 0.0:
                inputs_quantized = np.round(inputs / input_scale + input_zero_point).astype(input_details['dtype'])
            else:
                inputs_quantized = inputs.astype(input_details['dtype'])

            # Process each sample
            for i in range(batch_size):
                input_data = inputs_quantized[i:i + 1]

                # Measure inference time
                start_time = time.perf_counter()
                interpreter.set_tensor(input_details['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                inference_times.append((time.perf_counter() - start_time) * 1000)

                # Dequantize output if needed
                if output_scale != 0.0:
                    output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
                else:
                    output_float = output_data.astype(np.float32)

                prob_positive = float(output_float[0, 1])
                pred = int(np.argmax(output_float, axis=1)[0])

                predictions.append(pred)
                probabilities.append(prob_positive)
                true_labels.append(int(np.argmax(labels[i])))

        # Compute metrics
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)

        auc = roc_auc_score(true_labels, probabilities)
        acc = np.mean(predictions == true_labels)
        avg_inference_time = np.mean(inference_times)

        logger.info(f"TFLite Test Acc: {acc:.4f}, AUC: {auc:.4f}")
        logger.info(f"TFLite Avg Inference Time: {avg_inference_time:.2f}ms per sample")

        # Save detailed metrics
        self._plot_confusion_matrix(true_labels, predictions, prefix='tflite_')
        self._plot_roc_curve(true_labels, probabilities, prefix='tflite_')
        self._save_classification_report(true_labels, predictions, prefix='tflite_')

        return acc, auc, avg_inference_time

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
    """Format seconds into human-readable time string"""
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


def save_config(config: TrainingConfig, output_dir: Path, args, system_info: dict, model_summary: str = ""):
    """Save training configuration and system information as text"""
    config_path = output_dir / 'config.txt'

    with open(config_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEABADNET-NANO TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model Information:\n")
        f.write(f"  Version: {args.version}\n")
        f.write(f"  n_mels: {config.n_mels}\n")
        f.write(f"  Frequency Emphasis: {'YES' if args.version != 'v1_0' else 'NO'}\n")
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

        f.write("CLI Arguments:\n")
        f.write(f"  Version: {args.version}\n")
        f.write(f"  n_mels: {args.n_mels}\n")
        f.write(f"  Force Reprocess: {args.force_reprocess}\n")
        f.write(f"  Use Cache: {args.use_cache}\n")
        f.write("\n")

        if model_summary:
            f.write("Model Summary:\n")
            f.write(model_summary)
            f.write("\n")

    logger.info(f"Saved configuration to {config_path}")


def analyze_frequency_weights(model, output_dir: Path):
    """
    Extract and visualize learned frequency weights from the model
    """
    try:
        # Find frequency emphasis layer
        for layer in model.layers:
            if hasattr(layer, 'plot_weights') and callable(layer.plot_weights):
                layer.plot_weights(output_dir / 'frequency_weights.png')
                break
        else:
            logger.info("No frequency emphasis layer found in model")
    except Exception as e:
        logger.warning(f"Could not analyze frequency weights: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    start_time = time.time()

    # Initialize configuration
    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.n_fft = args.n_fft
    config.n_mels = args.n_mels

    # Update cache and output directories based on version and n_mels
    version_suffix = f"_{args.version}" # if args.version != 'Opt' else ""
    config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f'results/seabadnet_nano_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'

    # Set random seeds
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Collect system information
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
    logger.info("SEABAD Low Power TRAINING")
    logger.info("=" * 60)
    logger.info(f"Version: {args.version}")
    logger.info(f"n_mels: {config.n_mels}")
    logger.info(f"Frequency range: {config.mel_fmin}-{config.mel_fmax}Hz")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 60)

    try:
        # Preprocess and cache mel spectrograms
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

        # Create datasets from cache
        logger.info("Creating tf.data datasets...")
        train_dataset, train_class_counts = create_tf_dataset_from_cache('train', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset_from_cache('val', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset_from_cache('test', config, augment=False)

        # Dataset is balanced, no class weights needed
        class_weights = None
        logger.info(f"Train class distribution: {train_class_counts}")

        # Build model based on version
        logger.info(f"Building model version: {args.version}")
        if args.version == 'LP':
            model = build_cnn_mel_low_power(
                input_shape=(184, config.n_mels, 1),
                num_classes=2
            )
        elif args.version == 'FE':
            model = build_cnn_mel_low_power_freq_emph(
                input_shape=(184, config.n_mels, 1),
                num_classes=2
            )
        else:  # Optimized
            model = build_cnn_mel_low_power_optimized(
                input_shape=(184, config.n_mels, 1),
                num_classes=2
            )

        # Print and save model summary
        logger.info("=" * 60)
        logger.info("Model Architecture:")
        model_summary_lines = []
        model.summary(print_fn=lambda x: (logger.info(x), model_summary_lines.append(x)))
        logger.info("=" * 60)

        model_summary_str = "\n".join(model_summary_lines)

        # Save model summary to file
        with open(output_dir / 'model_summary.txt', 'w') as f:
            f.write(model_summary_str)

        # Save configuration
        save_config(config, output_dir, args, system_info, model_summary_str)

        # Train model
        trainer = ModelTrainer(model, config, class_weights)
        evaluator = ModelEvaluator(output_dir)

        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        # Plot training history
        evaluator.plot_training_history(history)

        # Analyze frequency weights (for V1.1 models)
        if args.version != 'v1_0':
            analyze_frequency_weights(model, output_dir)

        # Evaluate float model
        logger.info("=" * 60)
        logger.info("Evaluating Float32 Model:")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')

        # Save best model
        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved TensorFlow model to {output_dir / 'best_model.keras'}")

        # Convert to TFLite
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

        # Try multiple conversion strategies
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
            ("No quantization", {
                # No optimizations for baseline
            })
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
            # Create a simple unquantized model as fallback
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            strategy_success = "Fallback (no quantization)"

        # Save TFLite model
        tflite_path = output_dir / "model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"Saved TFLite model ({strategy_success}) to {tflite_path} ({tflite_size_kb:.2f} KB)")

        # Evaluate TFLite model
        logger.info("=" * 60)
        logger.info("Evaluating TFLite Model:")
        tflite_acc, tflite_auc, tflite_time = evaluator.evaluate_tflite(tflite_path, test_dataset)

        # Save final summary
        total_time = time.time() - start_time
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SEABADNET-NANO RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Version: {args.version}\n")
            f.write(f"n_mels: {config.n_mels}\n")
            f.write(f"Frequency Range: {config.mel_fmin}-{config.mel_fmax} Hz\n\n")

            f.write(f"Float32 Model:\n")
            f.write(f"  AUC: {float_auc:.4f}\n\n")

            f.write(f"TFLite Model ({strategy_success}):\n")
            f.write(f"  Accuracy: {tflite_acc:.4f}\n")
            f.write(f"  AUC: {tflite_auc:.4f}\n")
            f.write(f"  Avg Inference Time: {tflite_time:.2f}ms\n")
            f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n\n")
            total_params = model.count_params()
            f.write(f"  Total Params (Keras): {total_params:,}\n")  # Added this line

            if float_auc > 0:
                auc_degradation = float_auc - tflite_auc
                f.write(f"AUC Degradation: {auc_degradation:.4f} ({auc_degradation / float_auc * 100:.2f}%)\n")

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
        logger.info(f"  Model: SEABADNet-Nano (version={args.version})")
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