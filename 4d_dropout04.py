#!/usr/bin/env python3
"""
Option A: Add Dropout(0.4) before final classifier
Ablation Study Model: 4d_dropout04
Based on 1a_baseline2d.py with specific architectural modification
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
    # Mel spectrogram parameters
    n_mels: int = 64  # Using optimal n_mels from baseline sweep  # Number of mel bands
    n_fft: int = 1024  # FFT window size
    hop_length: int = 256  # Hop length for STFT (corrected to 256, crop to 184 frames)  # Hop length for STFT (adjusted to produce 184 time steps)
    # Learning rate schedule parameters
    lr_patience: int = 5  # Patience for learning rate reduction
    lr_reduction_factor: float = 0.5  # Factor to reduce LR by
    min_lr: float = 1e-5  # Minimum learning rate
    early_stopping_patience: int = 15  # Early stopping patience (3x LR patience)
    random_seed: int = 42
    # Path configurations
    dataset_path: str = '/Volumes/Evo/seabad'
    output_dir: str = 'results_4d_dropout'
    cache_dir: str = '/Volumes/Evo/cache_seabad_mels'
    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SEABAD CNN-Mel model')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Number of representative samples for TFLite quantization (default: 500)')
    parser.add_argument('--dataset-path', type=str, default='/Volumes/Evo/seabad',
                        help='Path to dataset directory (default: /Volumes/Evo/seabad)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all mel spectrograms even if cache exists')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached mel spectrograms and skip preprocessing (fails if cache does not exist)')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='Number of mel frequency bins (default: 80)')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')
    return parser.parse_args()

def get_optimizer(learning_rate: float):
    """
    Get appropriate Adam optimizer based on platform.
    Uses legacy optimizer on M1/M2/M3/M4 Macs to avoid performance issues.

    Args:
        learning_rate: Learning rate for optimizer

    Returns:
        TensorFlow Adam optimizer instance
    """
    system = platform.system()
    machine = platform.machine()

    # Check if running on Apple Silicon (arm64)
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info(f"Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        logger.info(f"Detected {system} {machine} - using standard Adam optimizer")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def build_cnn_mel_model_table2(
    input_shape=(184, 64, 1),
    num_classes=2
):
    """
    CNN-Mel + Dropout (Option A)
    Adds Dropout(0.4) before final classifier to reduce overfitting
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1: 3x3 valid conv -> ReLU
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="valid")(inputs)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: 3x3 valid conv -> ReLU
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # FC + ReLU -> 8
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    
    # MODIFICATION: Add Dropout
    x = tf.keras.layers.Dropout(0.4)(x)

    # FC + Softmax -> 2
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs, name=f"SEABADNet_4d_dropout04_m{input_shape[1]}")

    return model

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
    Compute mel spectrogram from waveform.

    Args:
        waveform: Audio waveform array
        config: Training configuration

    Returns:
        Mel spectrogram array of shape (time_steps, n_mels)
    """
    # Compute mel spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.target_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=0.0,
        fmax=config.target_sr / 2.0,
        center=False  # Add this to disable padding
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Transpose to (time, freq) format and normalize
    mel_spec_db = mel_spec_db.T  # Now shape is (time_steps, n_mels)

    # Ensure exactly 184 time steps
    if mel_spec_db.shape[0] > 184:
        mel_spec_db = mel_spec_db[:184, :]
    elif mel_spec_db.shape[0] < 184:
        pad_width = ((0, 184 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=0)

    # Normalize to [0, 1] range
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_db

def preprocess_and_cache_mels(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """
    Preprocess all audio files and cache mel spectrograms.

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
    logger.info("Preprocessing audio files and caching mel spectrograms...")
    logger.info("="*60)

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

        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
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

                # Compute mel spectrogram
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

    logger.info("="*60)
    logger.info("Preprocessing complete!")
    logger.info("="*60)

def load_cached_mels(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached mel spectrograms for a split.

    Args:
        split: Dataset split name
        config: Training configuration

    Returns:
        Tuple of (mel_specs, labels)
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

    Args:
        split: Dataset split name
        config: Training configuration
        augment: Whether to apply augmentation

    Returns:
        Tuple of (dataset, class_counts)
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

    # Apply augmentation if requested
    if augment:
        def augment_mel(mel, label):
            # Add small Gaussian noise
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.01)
            mel = mel + noise
            mel = tf.clip_by_value(mel, 0.0, 1.0)
            return mel, label

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts


# Add this function before ModelTrainer class:
def focal_loss(gamma=2.0, alpha=0.5):  # Use 0.5 for balanced datasets, 0.75 for imbalanced
    def loss(y_true, y_pred):
        # Cast y_true to float32 to match y_pred's dtype
        y_true = tf.cast(y_true, tf.float32)
        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate cross-entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal weight: (1 - pt)^gamma where pt is the prob of the true class
        # For true class, pt = y_pred; for false class, pt = 1 - y_pred
        # pt can be computed as: sum(y_true * y_pred, axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - pt, gamma)

        # Apply alpha weighting (weight positive class more)
        # alpha is applied to positive class (index 1)
        alpha_weight = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)

        # Combine focal weight and alpha weight
        loss = alpha_weight * focal_weight * tf.reduce_sum(cross_entropy, axis=-1, keepdims=True)
        return tf.reduce_mean(loss)
    return loss



class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig, class_weights: Dict[int, float] = None):
        self.model = model
        self.config = config
        self.class_weights = class_weights

        # Compile with standard categorical cross-entropy for balanced dataset
        self.model.compile(
            optimizer=get_optimizer(config.learning_rate),
            loss='categorical_crossentropy',
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
                input_data = inputs_quantized[i:i + 1]  # (1, 184, 80, 1)

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
        f.write("="*60 + "\n")
        f.write("TRAINING CONFIGURATION - MYBADV2 CNN-MEL\n")
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
        f.write(f"  Model: CNN-Mel (Table II)\n")
        f.write(f"  Input Shape: (184, 80, 1)\n")
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

        f.write("Mel Spectrogram Configuration:\n")
        f.write(f"  N Mels: {config.n_mels}\n")
        f.write(f"  N FFT: {config.n_fft}\n")
        f.write(f"  Hop Length: {config.hop_length}\n")
        f.write("\n")

        f.write("Dataset Configuration:\n")
        f.write(f"  Dataset: SEABAD\n")
        f.write(f"  Total Samples: 50k (25k positive + 25k negative)\n")
        f.write(f"  Split Ratios: {config.train_ratio:.0%} train / {config.val_ratio:.0%} val / {config.test_ratio:.0%} test\n")
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
    config.n_mels = args.n_mels
    config.cache_dir = f'/Volumes/Evo/cache_seabad_m{config.n_mels}'
    config.output_dir = f'results/4d_dropout04_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'

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
    logger.info("SEABAD CNN-Mel Training")
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
        # Preprocess and cache mel spectrograms
        preprocess_start = time.time()
        if args.use_cache:
            # Verify cache exists
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found at {config.cache_dir}. Cannot use --use_cache.")
            logger.info(f"Using cached mel spectrograms from {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config, force_reprocess=args.force_reprocess)
        times['preprocessing'] = time.time() - preprocess_start
        if times['preprocessing'] > 1.0:  # Only log if actually did preprocessing
            logger.info(f"Preprocessing completed in {format_time(times['preprocessing'])}")

        # Create datasets from cache
        logger.info("Creating tf.data datasets from cached mel spectrograms...")
        train_dataset, train_class_counts = create_tf_dataset_from_cache('train', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset_from_cache('val', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset_from_cache('test', config, augment=False)

        # For balanced datasets, no class weights needed
        class_weights = None

        logger.info("="*60)
        logger.info("Dataset is balanced - not using class weights")
        logger.info(f"  Train class distribution: {train_class_counts}")
        logger.info("="*60)

        # Build model and print summary
        model = build_cnn_mel_model_table2(input_shape=(184, config.n_mels, 1), num_classes=2)

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
            f.write("SEABAD CNN-Mel Training Results Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Float32 Model AUC: {float_auc:.4f}\n")
            f.write(f"TFLite int8 Model:\n")
            f.write(f"  Accuracy: {tflite_acc:.4f}\n")
            f.write(f"  AUC: {tflite_auc:.4f}\n")
            f.write(f"  Avg Inference Time: {tflite_time:.2f}ms\n")
            f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n")
            f.write(f"  (Experimental) Total Params: {len(tflite_model):,}\n")
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
