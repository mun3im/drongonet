#!/usr/bin/env python3
"""
0d_tinychirp_squeezenettime.py: TinyChirp SqueezeNet-Time baseline on TinyChirp dataset
Phase 0 reference — SqueezeNet architecture on raw waveforms.
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
    # Learning rate schedule parameters
    lr_patience: int = 5  # Patience for learning rate reduction
    lr_reduction_factor: float = 0.5  # Factor to reduce LR by
    min_lr: float = 1e-5  # Minimum learning rate
    early_stopping_patience: int = 15  # Early stopping patience (3x LR patience)
    random_seed: int = 42
    # Path configurations
    dataset_path: str = TINYCHIRP_PATH
    output_dir: str = 'results_0d_tinychirp_squeezenettime'
    cache_dir: str = f'{CACHE_BASE}_tinychirp_waveforms'

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train TinyChirp SqueezeNet-Time model')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Number of representative samples for TFLite quantization (default: 500)')
    parser.add_argument('--dataset-path', type=str, default=TINYCHIRP_PATH,
                        help='Path to TinyChirp dataset directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all waveforms even if cache exists')
    return parser.parse_args()

import platform
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def get_optimizer(learning_rate: float):
    """
    Get appropriate optimizer based on platform.
    - Uses legacy Adam optimizer on Apple Silicon (M1/M2/M3/M4) Macs to avoid performance issues.
    - Uses AdamW optimizer on Linux systems.
    - Defaults to standard Adam optimizer otherwise.

    Args:
        learning_rate: Learning rate for optimizer

    Returns:
        TensorFlow optimizer instance
    """
    system = platform.system()
    machine = platform.machine()

    # Check if running on Apple Silicon (arm64)
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info("Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif system == 'Linux':
        logger.info("Detected Linux - using AdamW optimizer")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        logger.info(f"Detected {system} {machine} - using standard Adam optimizer")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


class SingleHeadAttention(tf.keras.layers.Layer):
    """Single-head self-attention layer"""
    def __init__(self, d_model, **kwargs):
        super(SingleHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        # inputs: (batch, time, features)
        query = self.query_dense(inputs)  # (batch, time, d_model)
        key = self.key_dense(inputs)      # (batch, time, d_model)
        value = self.value_dense(inputs)  # (batch, time, d_model)

        # Compute attention scores
        score = tf.matmul(query, key, transpose_b=True)  # (batch, time, time)
        score = score / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch, time, time)

        # Apply attention to values
        output = tf.matmul(attention_weights, value)  # (batch, time, d_model)

        return output

import tensorflow as tf

def fire_module_1d(x, squeeze_filters, expand_filters):
    """
    Fire module for 1D convolutions:
    - Squeeze: Conv1D with fewer filters
    - Expand: parallel Conv1D with 1x1 and 3x3 kernels
    """
    # Squeeze
    squeeze = tf.keras.layers.Conv1D(squeeze_filters, kernel_size=1, padding="same", activation="relu")(x)

    # Expand 1x1
    expand1x1 = tf.keras.layers.Conv1D(expand_filters, kernel_size=1, padding="same", activation="relu")(squeeze)

    # Expand 3x3
    expand3x3 = tf.keras.layers.Conv1D(expand_filters, kernel_size=3, padding="same", activation="relu")(squeeze)

    # Concatenate expand outputs
    return tf.keras.layers.Concatenate()([expand1x1, expand3x3])


def build_squeezenet_time_model(
    input_shape=(48000, 1),
    num_classes=2
):
    """
    SqueezeT-Time model for raw waveform input.

    Layers (per Table II description):
      - Conv1D(kernel=3, filters=8, same) + ReLU -> (48000, 8)
      - MaxPooling1D(pool_size=2)                -> (24000, 8)
      - Fire module (squeeze=4, expand=8)        -> (24000, 16)
      - Dropout(0.25)                            -> (24000, 16)
      - AveragePooling1D(pool_size=2)            -> (12000, 16)
      - Flatten                                  -> 192000
      - Dense(64) + ReLU                         -> 64
      - Dense(num_classes) + Softmax             -> num_classes
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial Conv
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(inputs)  # (48000, 8)

    # MaxPool
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)  # (24000, 8)

    # Fire module
    x = fire_module_1d(x, squeeze_filters=4, expand_filters=8)  # (24000, 16)

    # Dropout
    x = tf.keras.layers.Dropout(0.25)(x)

    # AveragePool
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)  # (12000, 16)

    # Flatten
    x = tf.keras.layers.Flatten()(x)  # 12000*16 = 192000

    # FC + ReLU
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # FC + Softmax
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs, name="TinyChirp_SqueezeTTime")

    return model


class TinyChirpDataset:
    """Dataset class for TinyChirp that returns file paths only, no file verification during init"""
    def __init__(self, root_dir: str, split='training', fraction=1.0, test_size=0.1, val_size=0.1,
                 seed=42):
        self.root_dir = root_dir
        self.split = split
        self.fraction = fraction
        self.files = []
        self.labels = []

        random.seed(seed)

        # TinyChirp uses 'target' and 'non_target' instead of 'positive' and 'negative'
        target_files = []
        non_target_files = []

        # Map split names
        split_dir = os.path.join(root_dir, split)

        for label, class_name in enumerate(['non_target', 'target']):
            path = os.path.join(split_dir, class_name)
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} does not exist!")
            class_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            if class_name == 'non_target':
                non_target_files = [(f, label) for f in class_files]
            else:
                target_files = [(f, label) for f in class_files]

        random.shuffle(target_files)
        random.shuffle(non_target_files)

        # Use all available data from the split
        subset = target_files + non_target_files

        if fraction < 1.0:
            target_size = int(len(subset) * fraction)
            target_size_per_class = target_size // 2
            pos_subset = random.sample(target_files, min(target_size_per_class, len(target_files)))
            neg_subset = random.sample(non_target_files, min(target_size_per_class, len(non_target_files)))
            subset = pos_subset + neg_subset

        random.shuffle(subset)
        self.files = [f[0] for f in subset]
        self.labels = [f[1] for f in subset]

        logger.info(f"Loaded {len(self.files)} files for {split} "
              f"({len([l for l in self.labels if l == 0])} non_target, "
              f"{len([l for l in self.labels if l == 1])} target)")

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

    logger.info("="*60)
    logger.info("Preprocessing audio files and caching waveforms...")
    logger.info("="*60)

    splits = ['training', 'validation', 'testing']
    cache_info = {}

    for split in splits:
        logger.info(f"Processing {split} split...")

        # Get file paths
        dataset = TinyChirpDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        # Create split cache directory
        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        waveforms = []
        valid_labels = []

        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
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

    logger.info("="*60)
    logger.info("Preprocessing complete!")
    logger.info("="*60)

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
    logger.info(f"  Class distribution - Non-target: {class_counts[0]}, Target: {class_counts[1]}")

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
            # Random amplitude scaling (0.8 to 1.2)
            scale = tf.random.uniform([], minval=0.8, maxval=1.2)
            waveform = waveform * scale

            # Add small white noise
            noise = tf.random.normal(tf.shape(waveform), mean=0.0, stddev=0.005)
            waveform = waveform + noise

            # Random time shift (circular shift up to 0.1 seconds = 1600 samples)
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
def focal_loss(gamma=2.0, alpha=0.75):  # ← increase alpha for positive class
    def loss(y_true, y_pred):
        # Cast y_true to float32 to match y_pred's dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = tf.where(y_true == 1, pt[:, 1:2], pt[:, 0:1])  # positive class prob
        focal_weight = tf.pow(1 - pt, gamma)
        return -tf.reduce_mean(alpha * focal_weight * tf.math.log(pt))
    return loss



class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig, class_weights: Dict[int, float] = None):
        self.model = model
        self.config = config
        self.class_weights = class_weights

        # Compile with per-class recall metrics
        self.model.compile(
            optimizer=get_optimizer(config.learning_rate),
            loss=focal_loss(gamma=2.0, alpha=0.75),  # ← this line
            metrics=[
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        if class_weights:
            logger.info(f"Using class weights: {class_weights}")

        # Monitor val_loss and val_accuracy
        # 3. Change callbacks to monitor validation AUC
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
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
            class_weight=self.class_weights,  # Apply class weights during training
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
        logger.info(f"TFLite Output: dtype={output_details['dtype']}, scale={output_scale}, zero_point={output_zero_point}")

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
                input_data = inputs_quantized[i:i + 1]  # (1, 48000, 1)

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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-target', 'Target'])
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
            target_names=['Non-target', 'Target'],
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
        f.write("="*60 + "\n")
        f.write("TRAINING CONFIGURATION - TINYCHIRP SQUEEZENET-TIME\n")
        f.write("="*60 + "\n\n")

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
        f.write(f"  Model: SqueezeNet-Time\n")
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
    args = parse_args()
    start_time = time.time()

    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.output_dir = f'{RESULTS_BASE}/0d_tinychirp_squeezenet_r{config.random_seed}_{platform.system().lower()}'

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

    logger.info("="*60)
    logger.info("TinyChirp SqueezeNet-Time Training")
    logger.info("="*60)
    logger.info("System Information:")
    logger.info(f"  Platform: {system_info['platform']} {system_info['machine']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  TensorFlow: {system_info['tensorflow_version']}")
    logger.info(f"  Librosa: {system_info['librosa_version']}")
    logger.info(f"  GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_devices']:
        logger.info(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}")
    logger.info("="*60)
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
        train_dataset, train_class_counts = create_tf_dataset_from_cache('training', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset_from_cache('validation', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset_from_cache('testing', config, augment=False)

        # Calculate class weights using sklearn's balanced approach
        # class_weight = n_samples / (n_classes * n_samples_per_class)
        total_samples = sum(train_class_counts.values())
        n_classes = len(train_class_counts)

        class_weights = {
            0: 0.5,
            1: 2.0  # Start here, adjust if needed
        }

        # class_weights = None
        logger.info("="*60)
        logger.info(f"Calculated class weights: {class_weights}")
        logger.info(f"  Non-target weight: {class_weights[0]:.3f}")
        logger.info(f"  Target weight: {class_weights[1]:.3f}")
        logger.info("="*60)

        # Build model and print summary
        model = build_squeezenet_time_model(input_shape=(48000, 1), num_classes=2)

        logger.info("="*60)
        logger.info("Model Architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("="*60)

        # Save model summary to file
        with open(output_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # 2. Remove class weights when using focal loss
        trainer = ModelTrainer(model, config, class_weights=None)  # ← None!
        evaluator = ModelEvaluator(output_dir)

        # Training
        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        evaluator.plot_training_history(history)

        # Evaluate float model
        logger.info("="*60)
        logger.info("Evaluating Float32 Model:")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')
        logger.info(f'Floating-Point Test AUC: {float_auc:.4f}')

        # Save best model in Keras format
        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved TensorFlow model to {output_dir / 'best_model.keras'}")

        # Convert to int8 TFLite with multiple fallback strategies
        logger.info("="*60)
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
                    yield [inputs[i:i+1]]
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
        logger.info("="*60)
        logger.info("Evaluating TFLite int8 Model:")
        tflite_acc, tflite_auc, tflite_time = evaluator.evaluate_tflite(tflite_path, test_dataset)

        # Save final summary with timing breakdown
        total_time = time.time() - start_time
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TinyChirp SqueezeNet-Time Training Results Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Float32 Model AUC: {float_auc:.4f}\n")
            f.write(f"TFLite int8 Model:\n")
            f.write(f"  Accuracy: {tflite_acc:.4f}\n")
            f.write(f"  AUC: {tflite_auc:.4f}\n")
            f.write(f"  Avg Inference Time: {tflite_time:.2f}ms\n")
            f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n")
            f.write(f"\nAUC Degradation: {(float_auc - tflite_auc):.4f} ({(float_auc - tflite_auc)/float_auc*100:.2f}%)\n")
            f.write("\n" + "="*60 + "\n")
            f.write("Timing Breakdown\n")
            f.write("="*60 + "\n")
            if 'preprocessing' in times and times['preprocessing'] > 1.0:
                f.write(f"Preprocessing: {format_time(times['preprocessing'])}\n")
            if 'training' in times:
                f.write(f"Training: {format_time(times['training'])}\n")
            f.write(f"Total: {format_time(total_time)}\n")

        logger.info("="*60)
        logger.info("Results Summary:")
        logger.info(f"  Float32 AUC: {float_auc:.4f}")
        logger.info(f"  TFLite int8 Acc: {tflite_acc:.4f}, AUC: {tflite_auc:.4f}")
        logger.info(f"  TFLite Inference Time: {tflite_time:.2f}ms")
        logger.info(f"  TFLite Model Size: {tflite_size_kb:.2f} KB")
        logger.info(f"  AUC Degradation: {(float_auc - tflite_auc):.4f} ({(float_auc - tflite_auc)/float_auc*100:.2f}%)")
        logger.info("="*60)
        logger.info("Timing:")
        if 'preprocessing' in times and times['preprocessing'] > 1.0:
            logger.info(f"  Preprocessing: {format_time(times['preprocessing'])}")
        if 'training' in times:
            logger.info(f"  Training: {format_time(times['training'])}")
        logger.info(f"  Total: {format_time(total_time)}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise
    finally:
        save_elapsed_time(start_time, output_dir)

if __name__ == '__main__':
    main()
