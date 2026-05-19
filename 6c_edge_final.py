#!/usr/bin/env python3
"""
6b_edge_final.py: SEABADNet-Edge Final Candidate
Deeper GlobalAveragePooling Architecture for SBC deployment
- 3 Conv blocks (16→32→64 filters)
- GlobalAveragePooling2D (efficient, no flatten bottleneck)
- Target: >95% accuracy, <35 KB model size
Compatible with both macOS (Metal) and Linux (CUDA)
"""

for i in range(5): print("🟦"*i)
for i in range(5,0,-1): print("🟦"*i)

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# Configure GPU memory growth BEFORE any TensorFlow operations
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
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

@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    epochs: int = 100
    fraction: float = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    target_sr: int = 16000
    target_length: int = 16000 * 3  # 3 seconds @ 16kHz = 48000 samples
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-5
    early_stopping_patience: int = 15
    random_seed: int = 42
    dataset_path: str = '/Volumes/Evo/seabad'
    output_dir: str = 'results/6b_edge_final'
    cache_dir: str = '/Volumes/Evo/cache_seabad_mels'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Deeper GAP model')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Number of representative samples for TFLite quantization (default: 500)')
    parser.add_argument('--dataset-path', type=str, default='/Volumes/Evo/seabad',
                        help='Path to dataset directory (default: /Volumes/Evo/seabad)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of all mel spectrograms even if cache exists')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached mel spectrograms and skip preprocessing')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='Number of mel frequency bins (default: 80)')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    return parser.parse_args()

args = parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_optimizer(learning_rate: float):
    """Get appropriate optimizer based on platform."""
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info(f"Detected Apple Silicon Mac - using legacy Adam optimizer")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        logger.info(f"Detected {system} {machine} - using AdamW optimizer with weight_decay=1e-4")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


def build_deeper_gap(input_shape=(184, 80, 1), num_classes=2):
    """
    Deeper GlobalAveragePooling Architecture
    - 3 Conv blocks with BatchNormalization
    - Filter progression: 16 → 32 → 64
    - GlobalAveragePooling eliminates flatten bottleneck
    - Expected: ~20K params, ~20 KB model size
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1: 16 filters
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # → (92, 40, 16)

    # Block 2: 32 filters
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # → (46, 20, 32)

    # Block 3: 64 filters
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # No pooling here - let GlobalAveragePooling handle it

    # GlobalAveragePooling - outputs (64,) regardless of spatial dims
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Optional: small hidden layer for more expressiveness
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="DeeperGAP")


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
    """Compute mel spectrogram from waveform."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.target_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=0.0,
        fmax=config.target_sr / 2.0,
        center=False
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T

    if mel_spec_db.shape[0] > 184:
        mel_spec_db = mel_spec_db[:184, :]
    elif mel_spec_db.shape[0] < 184:
        pad_width = ((0, 184 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant', constant_values=0)

    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_db


def preprocess_and_cache_mels(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """Preprocess all audio files and cache mel spectrograms."""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

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

        dataset = SEABADDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        mel_specs = []
        valid_labels = []

        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}")):
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

    logger.info("="*60)
    logger.info("Preprocessing complete!")
    logger.info("="*60)


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
    """Create tf.data.Dataset from cached mel spectrograms."""
    mel_specs, labels = load_cached_mels(split, config)
    mel_specs = mel_specs[..., np.newaxis]

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

    if augment:
        def augment_mel(mel, label):
            # Gaussian noise
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.01)
            mel = mel + noise
            mel = tf.clip_by_value(mel, 0.0, 1.0)
            return mel, label

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, config: TrainingConfig, class_weights: Dict[int, float] = None):
        self.model = model
        self.config = config
        self.class_weights = class_weights

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
            class_weight=self.class_weights,
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
        """Evaluate TFLite model. Returns: (accuracy, auc, inference_time_ms)"""
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']

        logger.info(f"TFLite Input: dtype={input_details['dtype']}, scale={input_scale}, zero_point={input_zero_point}")
        logger.info(f"TFLite Output: dtype={output_details['dtype']}, scale={output_scale}, zero_point={output_zero_point}")

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


def save_config(config: TrainingConfig, output_dir: Path, args, system_info: dict):
    """Save training configuration and system information as text"""
    config_path = output_dir / 'config.txt'

    with open(config_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING CONFIGURATION - DEEPER GAP\n")
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
        f.write(f"  Model: Deeper GAP (3-block CNN + BatchNorm + GlobalAvgPool)\n")
        f.write(f"  Input Shape: (184, {config.n_mels}, 1)\n")
        f.write(f"  Filter Progression: 16 → 32 → 64\n")
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
    start_time = time.time()

    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.n_mels = args.n_mels
    config.n_fft = args.n_fft
    config.cache_dir = f'/Volumes/Evo/cache_seabad_m{config.n_mels}'
    config.output_dir = f'results/6b_edge_final_fft{config.n_fft}_m{config.n_mels}_s{config.random_seed}'

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    logger.info("Deeper GAP Architecture Training")
    logger.info("="*60)
    logger.info("System Information:")
    logger.info(f"  Platform: {system_info['platform']} {system_info['machine']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  TensorFlow: {system_info['tensorflow_version']}")
    logger.info(f"  GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_devices']:
        logger.info(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}")
    logger.info("="*60)
    logger.info(f"Configuration: output_dir={config.output_dir}")
    logger.info(f"Dataset path: {config.dataset_path}")
    logger.info(f"Representative samples for quantization: {args.repr_samples}")

    save_config(config, output_dir, args, system_info)

    times = {}

    try:
        preprocess_start = time.time()
        if args.use_cache:
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found at {config.cache_dir}. Cannot use --use_cache.")
            logger.info(f"Using cached mel spectrograms from {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config, force_reprocess=args.force_reprocess)
        times['preprocessing'] = time.time() - preprocess_start
        if times['preprocessing'] > 1.0:
            logger.info(f"Preprocessing completed in {format_time(times['preprocessing'])}")

        logger.info("Creating tf.data datasets from cached mel spectrograms...")
        train_dataset, train_class_counts = create_tf_dataset_from_cache('train', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset_from_cache('val', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset_from_cache('test', config, augment=False)

        class_weights = None

        logger.info("="*60)
        logger.info("Dataset is balanced - not using class weights")
        logger.info(f"  Train class distribution: {train_class_counts}")
        logger.info("="*60)

        model = build_deeper_gap(input_shape=(184, config.n_mels, 1), num_classes=2)

        logger.info("="*60)
        logger.info("Model Architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("="*60)

        with open(output_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        trainer = ModelTrainer(model, config, class_weights=None)
        evaluator = ModelEvaluator(output_dir)

        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        evaluator.plot_training_history(history)

        logger.info("="*60)
        logger.info("Evaluating Float32 Model:")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')
        logger.info(f'Floating-Point Test AUC: {float_auc:.4f}')

        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved TensorFlow model to {output_dir / 'best_model.keras'}")

        logger.info("="*60)
        logger.info("Converting to TFLite int8...")

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

        # Strategy 1: Full int8
        try:
            logger.info("Attempting Strategy 1: New converter with int8 input/output...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = representative_dataset

            try:
                converter.experimental_new_converter = True
                converter.experimental_new_quantizer = True
            except AttributeError:
                pass

            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]

            tflite_model = converter.convert()
            conversion_success = True
            logger.info("Strategy 1 succeeded!")
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")

        # Strategy 2: int8 input, float32 output
        if not conversion_success:
            try:
                logger.info("Attempting Strategy 2: int8 input, float32 output...")
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.float32
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

                tflite_model = converter.convert()
                conversion_success = True
                logger.info("Strategy 2 succeeded! (Output is float32)")
            except Exception as e:
                logger.warning(f"Strategy 2 failed: {e}")

        # Strategy 3: float32 input/output
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

        tflite_path = output_dir / "model_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"Saved int8 TFLite model to {tflite_path} ({tflite_size_kb:.2f} KB)")

        logger.info("="*60)
        logger.info("Evaluating TFLite int8 Model:")
        tflite_acc, tflite_auc, tflite_time = evaluator.evaluate_tflite(tflite_path, test_dataset)

        total_time = time.time() - start_time
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Deeper GAP Training Results Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Float32 Model AUC: {float_auc:.4f}\n")
            f.write(f"TFLite int8 Model:\n")
            f.write(f"  Accuracy: {tflite_acc:.4f}\n")
            f.write(f"  AUC: {tflite_auc:.4f}\n")
            f.write(f"  Avg Inference Time: {tflite_time:.2f}ms\n")
            f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n")
            f.write(f"  (Experimental) Total Params: {model.count_params():,}\n")
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
