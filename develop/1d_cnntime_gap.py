#!/usr/bin/env python3
"""
1d_cnntime_gap.py: CNN-Time + GAP on SEABAD (Phase 1 parallel reference)
CNN-Time with GlobalAveragePooling replacing Flatten.
Run once (seed=42) for the paper comparison table. Not an ablation step.
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SEABAD CNN-1D GAP model')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Number of representative samples for TFLite quantization (default: 500)')
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
                        help='Path to SEABAD dataset directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all waveforms even if cache exists')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    return parser.parse_args()

args = parse_args()

# Suppress TensorFlow warnings and configure GPU BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    # Learning rate schedule parameters
    lr_patience: int = 5  # Patience for learning rate reduction
    lr_reduction_factor: float = 0.5  # Factor to reduce LR by
    min_lr: float = 1e-5  # Minimum learning rate
    early_stopping_patience: int = 15  # Early stopping patience (3x LR patience)
    random_seed: int = 42
    # Path configurations
    dataset_path: str = DATASET_PATH
    output_dir: str = 'results/1d_cnntime_gap'
    cache_dir: str = f'{CACHE_BASE}_waveforms'


import platform
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def get_optimizer(learning_rate: float):
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'
    if is_apple_silicon:
        logger.info("Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=3e-4,clipnorm=1.0)
    elif system == 'Linux':
        logger.info("Detected Linux - using AdamW optimizer with weight_decay=1e-5")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    else:
        logger.info(f"Detected {system} {machine} - using standard Adam optimizer")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_time_model_gap(
    input_shape=(48000, 1),
    num_classes=2,
    use_receptive_enhancer=True
):
    """
    Optimized 1D CNN for raw audio waveform classification.
    Uses GlobalAveragePooling1D to eliminate massive dense layers.
    Designed for ultra-low memory (<10 KB INT8) and fast inference on Cortex-M7.

    Architecture:
      - Conv1D(4, k=3) → ReLU → MaxPool(2)
      - Conv1D(8, k=3) → ReLU → Dropout → AvgPool(2)
      - [Optional] Conv1D(16, k=9, dilation=2) → ReLU → Dropout
      - GlobalAveragePooling1D → Dense(32) → Dense(num_classes)

    Args:
        input_shape: (48000, 1) for 3s @ 16kHz
        num_classes: Number of output classes
        use_receptive_enhancer: Add lightweight dilated conv for better temporal modeling

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1: Initial feature extraction
    x = layers.Conv1D(filters=4, kernel_size=3, padding="same", name="conv1")(inputs)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)  # (24000, 4)

    # Block 2: Refinement
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same", name="conv2")(x)
    x = layers.ReLU(name="relu2")(x)
    # x = layers.Dropout(0.25, name="dropout1")(x)
    x = layers.AveragePooling1D(pool_size=2, name="avgpool1")(x)  # (12000, 8)

    # Optional: Lightweight receptive field enhancer (adds ~1.1K params)
    if use_receptive_enhancer:
        x = layers.Conv1D(
            filters=16,
            kernel_size=9,
            padding="same",
            dilation_rate=2,
            name="conv3_dilated"
        )(x)
        x = layers.ReLU(name="relu3")(x)
        # x = layers.Dropout(0.25, name="dropout2")(x)  # (12000, 16)

    # ✅ GLOBAL AVERAGE POOLING: replaces Flatten()
    x = layers.GlobalAveragePooling1D(name="gap")(x)  # (16,) or (8,) if enhancer off

    # Compact classifier head
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="TinyChirp_CNNTime_GAP")
    return model


def create_dataset_splits(root_dir: str, test_size=0.1, val_size=0.1, seed=42):
    """
    Create 80:10:10 dataset splits from positive/negative folders.

    Args:
        root_dir: Path to dataset root containing 'positive' and 'negative' folders
        test_size: Proportion for test set (default 0.1 = 10%)
        val_size: Proportion for validation set (default 0.1 = 10%)
        seed: Random seed for reproducible splits

    Returns:
        Dictionary with keys 'training', 'validation', 'testing', each containing
        lists of (file_path, label) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    splits = {'training': [], 'validation': [], 'testing': []}

    # Define class mapping
    class_map = {'positive': 1, 'negative': 0}  # positive=1, negative=0

    for class_name, label in class_map.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Directory {class_dir} does not exist!")

        # Get all .wav files in the class directory
        class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.wav')]
        random.shuffle(class_files)

        # Calculate split sizes
        total_samples = len(class_files)
        test_count = int(total_samples * test_size)
        val_count = int(total_samples * val_size)
        train_count = total_samples - test_count - val_count

        # Split the data
        train_files = class_files[:train_count]
        val_files = class_files[train_count:train_count + val_count]
        test_files = class_files[train_count + val_count:]

        # Add to splits with labels
        splits['training'].extend([(f, label) for f in train_files])
        splits['validation'].extend([(f, label) for f in val_files])
        splits['testing'].extend([(f, label) for f in test_files])

    # Shuffle each split to mix classes
    for split_name in splits:
        random.shuffle(splits[split_name])
        logger.info(f"Split {split_name}: {len(splits[split_name])} files "
                    f"({sum(1 for _, label in splits[split_name] if label == 0)} negative, "
                    f"{sum(1 for _, label in splits[split_name] if label == 1)} positive)")

    return splits


class SEABADDataset:
    """Dataset class for SEABAD that returns file paths only, no file verification during init"""

    def __init__(self, root_dir: str, split='training', fraction=1.0, test_size=0.1, val_size=0.1,
                 seed=42):
        self.root_dir = root_dir
        self.split = split
        self.fraction = fraction
        self.files = []
        self.labels = []

        # Create splits
        splits_dict = create_dataset_splits(root_dir, test_size=test_size, val_size=val_size, seed=seed)

        # Get the requested split
        split_data = splits_dict[split]

        if fraction < 1.0:
            # Take a fraction of the data while maintaining class balance
            split_size = int(len(split_data) * fraction)

            # Separate by class to maintain balance
            pos_samples = [(f, l) for f, l in split_data if l == 1]
            neg_samples = [(f, l) for f, l in split_data if l == 0]

            # Calculate how many samples per class we need
            pos_needed = int(len(pos_samples) * fraction)
            neg_needed = int(len(neg_samples) * fraction)

            # Take samples from each class
            selected_pos = random.sample(pos_samples, min(pos_needed, len(pos_samples)))
            selected_neg = random.sample(neg_samples, min(neg_needed, len(neg_samples)))

            subset = selected_pos + selected_neg
            random.shuffle(subset)
        else:
            subset = split_data

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


def process_waveform(waveform: np.ndarray, config: TrainingConfig) -> np.ndarray:
    """
    Process raw waveform: resample, pad/truncate, and normalize.

    Args:
        waveform: Audio waveform array
        config: Training configuration

    Returns:
        Processed waveform array of shape (target_length,)
    """
    # Pad or truncate to target length
    if len(waveform) > config.target_length:
        waveform = waveform[:config.target_length]
    elif len(waveform) < config.target_length:
        pad = np.zeros(config.target_length - len(waveform))
        waveform = np.concatenate([waveform, pad])

    # Normalize to [-1, 1] range
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    return waveform


def preprocess_and_cache_waveforms(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """
    Preprocess all audio files and cache raw waveforms.

    Args:
        dataset_path: Path to dataset root
        config: Training configuration
        force_reprocess: If True, reprocess all files even if cache exists
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
    logger.info("Preprocessing audio files and caching waveforms...")
    logger.info("=" * 60)

    splits = ['training', 'validation', 'testing']
    cache_info = {}

    for split in splits:
        logger.info(f"Processing {split} split...")

        # Get file paths
        dataset = SEABADDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        # Create split cache directory
        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        waveforms = []
        valid_labels = []

        for i, (file_path, label) in enumerate(
                tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
            try:
                # Load audio
                waveform, sr = librosa.load(file_path, sr=None)

                # Resample if needed
                if sr != config.target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.target_sr)

                # Process waveform (pad/truncate and normalize)
                waveform = process_waveform(waveform, config)

                waveforms.append(waveform)
                valid_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        # Convert to numpy arrays
        waveforms = np.array(waveforms, dtype=np.float32)  # Shape: (n_samples, target_length)
        valid_labels = np.array(valid_labels, dtype=np.int32)

        logger.info(f"  Processed {len(waveforms)} samples")
        logger.info(f"  Waveform shape: {waveforms.shape}")

        # Save to cache
        cache_file = split_cache_dir / 'waveforms.npz'
        np.savez_compressed(cache_file, waveforms=waveforms, labels=valid_labels)
        logger.info(f"  Saved cache to {cache_file}")

        cache_info[split] = {
            'n_samples': len(waveforms),
            'shape': waveforms.shape,
            'cache_file': str(cache_file)
        }

    # Save cache info
    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


def load_cached_waveforms(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached waveforms for a split.

    Args:
        split: Dataset split name
        config: Training configuration

    Returns:
        Tuple of (waveforms, labels)
    """
    cache_dir = Path(config.cache_dir)
    cache_file = cache_dir / split / 'waveforms.npz'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run preprocessing first.")

    data = np.load(cache_file)
    waveforms = data['waveforms']
    labels = data['labels']

    logger.info(f"Loaded {len(waveforms)} cached waveforms for {split}")
    logger.info(f"  Shape: {waveforms.shape}")

    return waveforms, labels


def create_tf_dataset_from_cache(split: str, config: TrainingConfig,
                                 augment: bool = False) -> Tuple[tf.data.Dataset, Dict[int, int]]:
    """
    Create tf.data.Dataset from cached waveforms.

    Args:
        split: Dataset split name
        config: Training configuration
        augment: Whether to apply augmentation

    Returns:
        Tuple of (dataset, class_counts)
    """
    # Load cached data
    waveforms, labels = load_cached_waveforms(split, config)

    # Add channel dimension: (n_samples, time) -> (n_samples, time, 1)
    waveforms = waveforms[..., np.newaxis]

    # Count samples per class
    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    logger.info(f"  Class distribution - Negative: {class_counts[0]}, Positive: {class_counts[1]}")

    # Create dataset on CPU to avoid GPU memory issues
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((waveforms, labels))

    # Shuffle if training
    if split == 'training':
        dataset = dataset.shuffle(buffer_size=len(waveforms), seed=config.random_seed)

    # Convert labels to one-hot encoding
    def to_one_hot(waveform, label):
        label_onehot = tf.one_hot(label, depth=2)
        return waveform, label_onehot

    dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation if requested
    if augment:
        def augment_waveform(waveform, label):
            # Random amplitude scaling (0.9 to 1.1) — less aggressive
            scale = tf.random.uniform([], minval=0.9, maxval=1.1)
            waveform = waveform * scale
            # Add small white noise (reduced from 0.005 → 0.001)
            noise = tf.random.normal(tf.shape(waveform), mean=0.0, stddev=0.001)
            waveform = waveform + noise
            # Random time shift (±0.1 sec = ±1600 samples)
            shift_amount = tf.random.uniform([], minval=-1600, maxval=1600, dtype=tf.int32)
            waveform = tf.roll(waveform, shift=shift_amount, axis=0)
            # Clip to valid range
            waveform = tf.clip_by_value(waveform, -1.0, 1.0)
            return waveform, label

        dataset = dataset.map(augment_waveform, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts


# Add this function before ModelTrainer class:


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig):
        self.model = model
        self.config = config

        self.model.compile(
            optimizer=get_optimizer(config.learning_rate),
            loss='categorical_crossentropy',  # ← standard CE for balanced data
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(Path(config.output_dir) / 'best_model.keras'),
                monitor='val_auc', mode='max', save_best_only=True
            )
        ]


    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> Dict[str, List[float]]:
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        return history.history


class ModelEvaluator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history: Dict[str, List[float]]):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(history['loss']) + 1)

        ax1.plot(epochs, history['loss'], label='Train Loss', color='blue')
        ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(epochs, history['auc'], label='Train AUC', color='green', linestyle='--')
        ax2.plot(epochs, history['val_auc'], label='Val AUC', color='orange', linestyle='--')
        ax2.set_ylabel('AUC', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title('Training and Validation Metrics')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()

    def evaluate_model(self, model: tf.keras.Model, test_dataset: tf.data.Dataset, prefix: str = '') -> float:
        predictions, probabilities = self._get_predictions(model, test_dataset)
        true_labels = self._get_labels(test_dataset)
        self._plot_confusion_matrix(true_labels, predictions, prefix)
        auc = self._plot_roc_curve(true_labels, probabilities, prefix)
        self._save_classification_report(true_labels, predictions, prefix)
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
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']

        # Log quantization info
        logger.info(f"TFLite Input: dtype={input_details['dtype']}, scale={input_scale}, zero_point={input_zero_point}")
        logger.info(
            f"TFLite Output: dtype={output_details['dtype']}, scale={output_scale}, zero_point={output_zero_point}")

        predictions = []
        probabilities = []
        true_labels = []
        inference_times = []

        logger.info("Evaluating TFLite model...")
        for inputs, labels in tqdm(test_dataset, desc="TFLite inference"):
            # Convert to numpy once per batch
            inputs = inputs.numpy()
            labels = labels.numpy()
            batch_size = inputs.shape[0]

            # Batch quantize inputs if needed
            if input_scale != 0.0:
                inputs_quantized = np.round(inputs / input_scale + input_zero_point).astype(input_details['dtype'])
            else:
                inputs_quantized = inputs.astype(input_details['dtype'])

            # Process each sample (TFLite interpreter doesn't support batching)
            for i in range(batch_size):
                input_data = inputs_quantized[i:i + 1]  # Shape: (1, 48000, 1)
                # Measure inference time
                start_time = time.perf_counter()
                interpreter.set_tensor(input_details['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])  # shape: (1, 2)
                inference_times.append((time.perf_counter() - start_time) * 1000)  # Convert to ms

                # Dequantize output if needed
                if output_scale != 0.0:
                    output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
                else:
                    output_float = output_data.astype(np.float32)

                # Output is already softmax from model
                prob_positive = float(output_float[0, 1])
                pred = int(np.argmax(output_float, axis=1)[0])

                predictions.append(pred)
                probabilities.append(prob_positive)
                # Convert one-hot encoded label to class index
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
            outputs = model(inputs, training=False)  # Already has softmax
            predictions.extend(np.argmax(outputs, axis=1))
            probabilities.extend(outputs[:, 1].numpy())
        return np.array(predictions), np.array(probabilities)

    def _get_labels(self, dataset: tf.data.Dataset) -> np.ndarray:
        labels = []
        for _, lbl in dataset:
            # Convert one-hot encoded labels back to class indices
            lbl_indices = np.argmax(lbl.numpy(), axis=1)
            labels.extend(lbl_indices)
        return np.array(labels)

    def _plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray, prefix: str):
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{prefix}Confusion Matrix')
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png')
        plt.close()

    def _plot_roc_curve(self, true_labels: np.ndarray, probabilities: np.ndarray, prefix: str) -> float:
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{prefix}Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png')
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


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string (xxh xxm xxs.ss or xxm xxs.ss)"""
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


def save_config(config: TrainingConfig, output_dir: Path, args, system_info: dict):
    """Save training configuration and system information as text"""
    config_path = output_dir / 'config.txt'

    with open(config_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING CONFIGURATION - MYBAD CNN-TIME\n")
        f.write("=" * 60 + "\n\n")

        f.write("System Information:\n")
        f.write(f"  Platform: {system_info['platform']}\n")
        f.write(f"  Machine: {system_info['machine']}\n")
        f.write(f"  Python Version: {system_info['python_version']}\n")
        f.write(f"  TensorFlow Version: {system_info['tensorflow_version']}\n")
        f.write(f"  GPU Available: {system_info['gpu_available']}\n")
        if system_info['gpu_devices']:
            f.write(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}\n")
        f.write("\n")

        f.write("Model Architecture:\n")
        f.write(f"  Model: CNN-Time\n")
        f.write(f"  Input Shape: (48000, 1)\n")
        f.write("\n")

        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  LR Patience: {config.lr_patience}\n")
        f.write(f"  LR Reduction Factor: {config.lr_reduction_factor}\n")
        f.write(f"  Min LR: {config.min_lr}\n")
        f.write(f"  Early Stopping Patience: {config.early_stopping_patience}\n")
        f.write(f"  Random Seed: {config.random_seed}\n")
        f.write(f"  Dataset Fraction: {config.fraction}\n")
        f.write("\n")

        f.write("Audio Configuration:\n")
        f.write(f"  Target Sample Rate: {config.target_sr} Hz\n")
        f.write(f"  Target Length: {config.target_length} samples\n")
        f.write(f"  Target Duration: {config.target_length / config.target_sr:.1f} seconds\n")
        f.write("\n")

        f.write("Paths:\n")
        f.write(f"  Dataset Path: {config.dataset_path}\n")
        f.write(f"  Output Dir: {config.output_dir}\n")
        f.write(f"  Cache Dir: {config.cache_dir}\n")
        f.write("\n")

        f.write("CLI Arguments:\n")
        f.write(f"  Representative Samples: {args.repr_samples}\n")
        f.write(f"  Force Reprocess: {args.force_reprocess}\n")
        f.write("\n")

    logger.info(f"Saved configuration to {config_path}")


def main():
    start_time = time.time()

    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    platform_tag = 'macos' if platform.system() == 'Darwin' else 'linux'

    config.output_dir = f'{RESULTS_BASE}/1d_cnntime_gap_r{config.random_seed}_{platform_tag}'

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

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
    logger.info("DrongoNet CNN-Time Training")
    logger.info("=" * 60)
    logger.info("System Information:")
    logger.info(f"  Platform: {system_info['platform']} {system_info['machine']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  TensorFlow: {system_info['tensorflow_version']}")
    logger.info(f"  Librosa: {system_info['librosa_version']}")
    logger.info(f"  GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_devices']:
        logger.info(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}")
    logger.info("=" * 60)
    logger.info(f"Configuration: output_dir={config.output_dir}")
    logger.info(f"Dataset path: {config.dataset_path}")
    logger.info(f"Representative samples for quantization: {args.repr_samples}")

    # Save configuration
    save_config(config, output_dir, args, system_info)

    # Time tracking
    times = {}

    try:
        # Preprocess and cache waveforms
        preprocess_start = time.time()
        preprocess_and_cache_waveforms(config.dataset_path, config, force_reprocess=args.force_reprocess)
        times['preprocessing'] = time.time() - preprocess_start
        if times['preprocessing'] > 1.0:  # Only log if actually did preprocessing
            logger.info(f"Preprocessing completed in {format_time(times['preprocessing'])}")

        # Create datasets from cache
        logger.info("Creating tf.data datasets from cached waveforms...")

        # Create datasets
        train_dataset, _ = create_tf_dataset_from_cache('training', config, augment=True)
        val_dataset, _ = create_tf_dataset_from_cache('validation', config, augment=False)
        test_dataset, _ = create_tf_dataset_from_cache('testing', config, augment=False)

        # NO class weights — balanced data + standard CE
        logger.info("Using standard categorical crossentropy (balanced dataset, no class weights)")

        model = build_cnn_time_model_gap(
            input_shape=(48000, 1),  # 3s @ 16kHz
            num_classes=2,  # [non-bird, bird]
            use_receptive_enhancer=True
        )

        logger.info("=" * 60)
        logger.info("Model Architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("=" * 60)

        # Save model summary to file
        with open(output_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # 2. Remove class weights when using focal loss
        trainer = ModelTrainer(model, config)  # ← no class_weights arg
        evaluator = ModelEvaluator(output_dir)

        # Training
        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        evaluator.plot_training_history(history)

        # Evaluate float model
        logger.info("=" * 60)
        logger.info("Evaluating Float32 Model:")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')
        logger.info(f'Floating-Point Test AUC: {float_auc:.4f}')

        # Save best model in Keras format
        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved TensorFlow model to {output_dir / 'best_model.keras'}")

        # Convert to int8 TFLite with multiple fallback strategies
        logger.info("=" * 60)
        logger.info("Converting to TFLite int8...")

        # Provide representative dataset for quantization
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

        logger.info(f"Using {args.repr_samples} samples for quantization calibration...")

        tflite_model = None
        conversion_success = False

        # Strategy 1: Try with new converter and quantizer
        try:
            logger.info("Attempting Strategy 1: New converter with int8 input/output...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = representative_dataset

            # Try enabling experimental features if available
            try:
                converter.experimental_new_converter = True
                converter.experimental_new_quantizer = True
            except AttributeError:
                pass  # These flags might not exist in all TF versions

            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]

            tflite_model = converter.convert()
            conversion_success = True
            logger.info("Strategy 1 succeeded!")
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")

        # Strategy 2: Try with float32 output (int8 activations)
        if not conversion_success:
            try:
                logger.info("Attempting Strategy 2: int8 input, float32 output...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.float32  # Keep output as float32
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

                tflite_model = converter.convert()
                conversion_success = True
                logger.info("Strategy 2 succeeded! (Output is float32)")
            except Exception as e:
                logger.warning(f"Strategy 2 failed: {e}")

        # Strategy 3: Try with float32 input/output (internal int8 only)
        if not conversion_success:
            try:
                logger.info("Attempting Strategy 3: float32 input/output, int8 weights...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset

                tflite_model = converter.convert()
                conversion_success = True
                logger.info("Strategy 3 succeeded! (Input/output are float32)")
            except Exception as e:
                logger.warning(f"Strategy 3 failed: {e}")

        if not conversion_success or tflite_model is None:
            raise RuntimeError("All TFLite conversion strategies failed!")

        # Save TFLite model
        tflite_path = output_dir / "model_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"Saved int8 TFLite model to {tflite_path} ({tflite_size_kb:.2f} KB)")

        # Evaluate TFLite model
        logger.info("=" * 60)
        logger.info("Evaluating TFLite int8 Model:")
        tflite_acc, tflite_auc, tflite_time = evaluator.evaluate_tflite(tflite_path, test_dataset)

        # Save final summary with timing breakdown
        total_time = time.time() - start_time
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DrongoNet CNN-Time Training Results Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Float32 Model AUC: {float_auc:.4f}\n")
            f.write(f"TFLite int8 Model:\n")
            f.write(f"  Accuracy: {tflite_acc:.4f}\n")
            f.write(f"  AUC: {tflite_auc:.4f}\n")
            f.write(f"  Avg Inference Time: {tflite_time:.2f}ms\n")
            f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n")
            f.write(
                f"\nAUC Degradation: {(float_auc - tflite_auc):.4f} ({(float_auc - tflite_auc) / float_auc * 100:.2f}%)\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Timing Breakdown\n")
            f.write("=" * 60 + "\n")
            if 'preprocessing' in times and times['preprocessing'] > 1.0:
                f.write(f"Preprocessing: {format_time(times['preprocessing'])}\n")
            if 'training' in times:
                f.write(f"Training: {format_time(times['training'])}\n")
            f.write(f"Total: {format_time(total_time)}\n")

        logger.info("=" * 60)
        logger.info("Results Summary:")
        logger.info(f"  Float32 AUC: {float_auc:.4f}")
        logger.info(f"  TFLite int8 Acc: {tflite_acc:.4f}, AUC: {tflite_auc:.4f}")
        logger.info(f"  TFLite Inference Time: {tflite_time:.2f}ms")
        logger.info(f"  TFLite Model Size: {tflite_size_kb:.2f} KB")
        logger.info(
            f"  AUC Degradation: {(float_auc - tflite_auc):.4f} ({(float_auc - tflite_auc) / float_auc * 100:.2f}%)")
        logger.info("=" * 60)
        logger.info("Timing:")
        if 'preprocessing' in times and times['preprocessing'] > 1.0:
            logger.info(f"  Preprocessing: {format_time(times['preprocessing'])}")
        if 'training' in times:
            logger.info(f"  Training: {format_time(times['training'])}")
        logger.info(f"  Total: {format_time(total_time)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise
    finally:
        save_elapsed_time(start_time, output_dir)


if __name__ == '__main__':
    main()