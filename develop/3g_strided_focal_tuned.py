#!/usr/bin/env python3
"""
3g_strided_focal_tuned.py: Strided conv + tuned focal loss (Phase 3 strided investigation)
Replaces MaxPool with strided Conv2D; focal loss α=0.75, γ=1.5.
Strided aside — no gate, result reported for completeness.
Compatible with both macOS (Metal) and Linux (CUDA)
"""

import os
import logging
import time
import argparse
import platform
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import tensorflow as tf
import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve
from tqdm import tqdm

import os
import argparse
from config import DATASET_PATH, TINYCHIRP_PATH, RESULTS_BASE, CACHE_BASE

# Set log level early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse --force_cpu before importing TensorFlow
def _parse_early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    args, _ = parser.parse_known_args()
    return args

_early_args = _parse_early_args()

if _early_args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Now import TensorFlow
import tensorflow as tf

# Enable memory growth for GPUs if available and not disabled
if not _early_args.force_cpu:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Must be called before any GPU initialization
        print(f"GPU memory growth setup failed: {e}")
       
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    fraction: float = 1.0
    random_seed: int = 42
    target_sr: int = 16000
    target_length: int = 48000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-5
    early_stopping_patience: int = 15
    dataset_path: str = DATASET_PATH
    output_dir: str = 'results/3g_strided_focal_tuned'
    cache_dir: str = f'{CACHE_BASE}_fft1024_m64'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def update_derived_paths(self):
        self.cache_dir = f'{CACHE_BASE}_fft{self.n_fft}_m{self.n_mels}'
        self.output_dir = f'{RESULTS_BASE}/3g_strided_focal_tuned_fft{self.n_fft}_m{self.n_mels}_s{self.random_seed}_{self.platform_tag}'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SEABAD CNN-Mel model')
    parser.add_argument('--repr_samples', type=int, default=500)
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--force-reprocess', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--force_cpu', action='store_true')
    return parser.parse_args()


class FrequencyEmphasis(tf.keras.layers.Layer):
    """Frequency emphasis layer - trainable per-frequency weighting"""
    def __init__(self, freq_bins: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins

    def build(self, input_shape):
        self.freq_weights = self.add_weight(name='frequency_weights', shape=(1, 1, self.freq_bins, 1),
                                      initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.scale = self.add_weight(name='scale', shape=(1,), initializer=tf.keras.initializers.Constant(3.0),
                                    trainable=True)
        self.per_freq_gain = self.add_weight(name='per_freq_gain', shape=(1, 1, self.freq_bins, 1),
                                            initializer=tf.keras.initializers.Constant(1.0), trainable=False)

    def call(self, inputs, training=None):
        weighted = inputs * tf.math.sigmoid(self.freq_weights * self.scale)
        weighted = weighted * self.per_freq_gain
        return weighted

    def get_config(self):
        config = super().get_config()
        config.update({'freq_bins': self.freq_bins})
        return config


def build_model(input_shape=(184, 80, 1), num_classes=2):
    """
    Build model with:
    - Frequency Emphasis layer
    - Strided Conv2D (no MaxPool) - TIER-1 optimization
    - 1x1 Pointwise convolution before GAP
    - Global Average Pooling (GAP)
    - Focal loss with α=0.75, γ=1.5 (applied during compilation)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Frequency emphasis - trainable per-frequency weighting
    x = FrequencyEmphasis(freq_bins=input_shape[1])(inputs)

    # Strided convolution (replaces Conv + MaxPool)
    x = tf.keras.layers.Conv2D(8, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)

    # 1x1 pointwise convolution before GAP
    x = tf.keras.layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="DrongoNet_3g")


def focal_loss(gamma=1.5, alpha=0.75):  # Tuned for recall: overweight false negatives
    """Focal loss optimized for high recall (γ=1.5, α=0.75)"""
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


def get_optimizer(learning_rate):
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'
    if is_apple_silicon:
        logger.info("Using legacy Adam optimizer (Apple Silicon)")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        logger.info(f"Using AdamW optimizer ({system} {machine})")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)


class SEABADDataset:
    def __init__(self, root_dir, split='train', fraction=1.0, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.fraction = fraction
        self.files = []
        self.labels = []
        random.seed(seed)
        positive_files, negative_files = self._load_files()
        subset = self._split_data(positive_files, negative_files)
        if fraction < 1.0:
            subset = random.sample(subset, int(len(subset) * fraction))
        random.shuffle(subset)
        self.files = [f[0] for f in subset]
        self.labels = [f[1] for f in subset]
        self._log_stats()

    def _load_files(self):
        positive_files = []
        negative_files = []
        for label, class_name in enumerate(['negative', 'positive']):
            path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(path):
                raise ValueError(f"Directory {path} does not exist!")
            class_files = []
            for root, _, files in os.walk(path):
                class_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
            labeled_files = [(f, label) for f in class_files]
            if class_name == 'negative':
                negative_files = labeled_files
            else:
                positive_files = labeled_files
        logger.info(f"Found {len(positive_files)} positive and {len(negative_files)} negative samples")
        return positive_files, negative_files

    def _split_data(self, positive_files, negative_files):
        random.shuffle(positive_files)
        random.shuffle(negative_files)
        n_pos = len(positive_files)
        n_neg = len(negative_files)
        train_pos_end = int(n_pos * 0.8)
        val_pos_end = int(n_pos * 0.9)
        train_neg_end = int(n_neg * 0.8)
        val_neg_end = int(n_neg * 0.9)
        splits = {
            'train': positive_files[:train_pos_end] + negative_files[:train_neg_end],
            'val': positive_files[train_pos_end:val_pos_end] + negative_files[train_neg_end:val_neg_end],
            'test': positive_files[val_pos_end:] + negative_files[val_neg_end:]
        }
        if self.split not in splits:
            raise ValueError(f"Invalid split: {self.split}")
        return splits[self.split]

    def _log_stats(self):
        n_neg = sum(1 for l in self.labels if l == 0)
        n_pos = sum(1 for l in self.labels if l == 1)
        logger.info(f"Loaded {len(self.files)} files for {self.split} ({n_neg} negative, {n_pos} positive)")

    def __len__(self):
        return len(self.files)

    def get_files_and_labels(self):
        return self.files, self.labels


def compute_mel_spectrogram(waveform, config):
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=config.target_sr, n_fft=config.n_fft,
                                              hop_length=config.hop_length, n_mels=config.n_mels, fmin=0.0,
                                              fmax=config.target_sr / 2.0, center=False)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    if mel_spec_db.shape[0] > 184:
        mel_spec_db = mel_spec_db[:184, :]
    elif mel_spec_db.shape[0] < 184:
        pad_width = ((0, 184 - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_db


def preprocess_audio(file_path, config):
    try:
        waveform, sr = librosa.load(file_path, sr=None)
        if sr != config.target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.target_sr)
        if len(waveform) > config.target_length:
            waveform = waveform[:config.target_length]
        elif len(waveform) < config.target_length:
            pad = np.zeros(config.target_length - len(waveform))
            waveform = np.concatenate([waveform, pad])
        return compute_mel_spectrogram(waveform, config)
    except Exception as e:
        logger.warning(f"Failed to process {file_path}: {e}")
        return None


def preprocess_and_cache_mels(dataset_path, config, force_reprocess=False):
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_info_path = cache_dir / 'cache_info.pkl'
    if cache_info_path.exists() and not force_reprocess:
        logger.info(f"Cache exists at {cache_dir}. Use --force-reprocess to regenerate.")
        return
    logger.info("=" * 60)
    logger.info("Preprocessing audio files and caching mel spectrograms...")
    logger.info("=" * 60)
    cache_info = {}
    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split} split...")
        dataset = SEABADDataset(dataset_path, split=split, fraction=config.fraction, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()
        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)
        mel_specs = []
        valid_labels = []
        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}"):
            mel_spec = preprocess_audio(file_path, config)
            if mel_spec is not None:
                mel_specs.append(mel_spec)
                valid_labels.append(label)
        mel_specs = np.array(mel_specs, dtype=np.float32)
        valid_labels = np.array(valid_labels, dtype=np.int32)
        logger.info(f"  Processed {len(mel_specs)} samples - Shape: {mel_specs.shape}")
        cache_file = split_cache_dir / 'mels.npz'
        np.savez_compressed(cache_file, mels=mel_specs, labels=valid_labels)
        logger.info(f"  Saved cache to {cache_file}")
        cache_info[split] = {'n_samples': len(mel_specs), 'shape': mel_specs.shape, 'cache_file': str(cache_file)}
    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)


def load_cached_mels(split, config):
    cache_file = Path(config.cache_dir) / split / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    data = np.load(cache_file)
    logger.info(f"Loaded {len(data['mels'])} cached mels for {split} - Shape: {data['mels'].shape}")
    return data['mels'], data['labels']


def create_tf_dataset(split, config, augment=False):
    mel_specs, labels = load_cached_mels(split, config)
    mel_specs = mel_specs[..., np.newaxis]
    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    logger.info(f"  Class distribution - Negative: {class_counts[0]}, Positive: {class_counts[1]}")
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_specs, labels))
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(mel_specs), seed=config.random_seed)
    dataset = dataset.map(lambda mel, label: (mel, tf.one_hot(label, depth=2)), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        def augment_mel(mel, label):
            noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.01)
            mel = tf.clip_by_value(mel + noise, 0.0, 1.0)
            return mel, label

        dataset = dataset.map(augment_mel, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, class_counts


class ModelTrainer:
    def __init__(self, model, config, class_weights=None):
        self.model = model
        self.config = config
        self.class_weights = class_weights
        self.model.compile(optimizer=get_optimizer(config.learning_rate), loss=focal_loss(gamma=1.5, alpha=0.75),
                          metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'),
                                   tf.keras.metrics.Recall(name='recall')])
        output_path = Path(config.output_dir)
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(output_path / 'best_model.keras'), monitor='val_auc', mode='max',
                                              save_best_only=True)
        ]

    def train(self, train_dataset, val_dataset):
        logger.info(f"Starting training for up to {self.config.epochs} epochs")
        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.config.epochs,
                                callbacks=self.callbacks, class_weight=self.class_weights, verbose=1)
        return history.history


class ModelEvaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history):
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

    def evaluate_model(self, model, test_dataset, prefix=''):
        predictions, probabilities = self._get_predictions(model, test_dataset)
        true_labels = self._get_labels(test_dataset)
        self._plot_confusion_matrix(true_labels, predictions, prefix)
        auc = self._plot_roc_curve(true_labels, probabilities, prefix)
        self._save_classification_report(true_labels, predictions, prefix)
        return auc

    def evaluate_tflite(self, tflite_path, test_dataset):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']
        logger.info(f"TFLite quantization - Input: scale={input_scale}, zp={input_zero_point}")
        logger.info(f"TFLite quantization - Output: scale={output_scale}, zp={output_zero_point}")
        predictions, probabilities, true_labels, inference_times = [], [], [], []
        for inputs, labels in tqdm(test_dataset, desc="TFLite inference"):
            inputs_np = inputs.numpy()
            labels_np = labels.numpy()
            if input_scale != 0.0:
                inputs_quantized = np.round(inputs_np / input_scale + input_zero_point).astype(input_details['dtype'])
            else:
                inputs_quantized = inputs_np.astype(input_details['dtype'])
            for i in range(inputs_np.shape[0]):
                start = time.perf_counter()
                interpreter.set_tensor(input_details['index'], inputs_quantized[i:i + 1])
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])
                inference_times.append((time.perf_counter() - start) * 1000)
                if output_scale != 0.0:
                    output_float = (output.astype(np.float32) - output_zero_point) * output_scale
                else:
                    output_float = output.astype(np.float32)
                probabilities.append(float(output_float[0, 1]))
                predictions.append(int(np.argmax(output_float, axis=1)[0]))
                true_labels.append(int(np.argmax(labels_np[i])))
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)
        auc = roc_auc_score(true_labels, probabilities)
        acc = np.mean(predictions == true_labels)
        avg_time = np.mean(inference_times)
        logger.info(f"TFLite Test Acc: {acc:.4f}, AUC: {auc:.4f}")
        logger.info(f"TFLite Avg Inference Time: {avg_time:.2f}ms")
        self._plot_confusion_matrix(true_labels, predictions, 'tflite_')
        self._plot_roc_curve(true_labels, probabilities, 'tflite_')
        self._save_classification_report(true_labels, predictions, 'tflite_')
        return acc, auc, avg_time

    def _get_predictions(self, model, dataset):
        predictions, probabilities = [], []
        for inputs, _ in dataset:
            outputs = model(inputs, training=False)
            predictions.extend(np.argmax(outputs, axis=1))
            probabilities.extend(outputs[:, 1].numpy())
        return np.array(predictions), np.array(probabilities)

    def _get_labels(self, dataset):
        labels = []
        for _, lbl in dataset:
            labels.extend(np.argmax(lbl.numpy(), axis=1))
        return np.array(labels)

    def _plot_confusion_matrix(self, true_labels, predictions, prefix):
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{prefix}Confusion Matrix')
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png')
        plt.close()

    def _plot_roc_curve(self, true_labels, probabilities, prefix):
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{prefix}ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png')
        plt.close()
        return auc

    def _save_classification_report(self, true_labels, predictions, prefix):
        report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], digits=4,
                                      zero_division=0)
        with open(self.output_dir / f'{prefix}classification_report.txt', 'w') as f:
            f.write(report)


def convert_to_tflite(model, val_dataset, output_path, repr_samples=500):
    logger.info("=" * 60)
    logger.info("Converting to TFLite int8...")

    def representative_dataset():
        count = 0
        for inputs, _ in val_dataset:
            if count >= repr_samples:
                break
            for i in range(inputs.shape[0]):
                if count >= repr_samples:
                    break
                yield [inputs[i:i + 1]]
                count += 1

    logger.info(f"Using {repr_samples} samples for quantization calibration...")
    strategies = [
        ("New converter with int8 I/O", {'inference_input_type': tf.int8, 'inference_output_type': tf.int8,
                                        'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                                          tf.lite.OpsSet.TFLITE_BUILTINS]}),
        ("Int8 input, float32 output", {'inference_input_type': tf.int8, 'inference_output_type': tf.float32,
                                       'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS]}),
        ("Float32 I/O, int8 weights", {'supported_ops': None})
    ]
    for strategy_name, params in strategies:
        try:
            logger.info(f"Attempting: {strategy_name}...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            converter.representative_dataset = representative_dataset
            if params.get('inference_input_type'):
                converter.inference_input_type = params['inference_input_type']
            if params.get('inference_output_type'):
                converter.inference_output_type = params['inference_output_type']
            if params.get('supported_ops'):
                converter.target_spec.supported_ops = params['supported_ops']
            tflite_model = converter.convert()
            logger.info(f"{strategy_name} succeeded!")
            return tflite_model
        except Exception as e:
            logger.warning(f"{strategy_name} failed: {e}")
    raise RuntimeError("All TFLite conversion strategies failed!")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:05.2f}s"
    return f"{minutes}m {secs:05.2f}s"


def get_system_info():
    return {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'librosa_version': librosa.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': [d.name for d in tf.config.list_physical_devices('GPU')],
    }


def save_config(config, output_dir, args, system_info):
    config_path = output_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("System Information:\n")
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTraining Parameters:\n")
        for key, value in asdict(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\nCLI Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
    logger.info(f"Saved configuration to {config_path}")


def save_results_summary(output_dir, float_auc, tflite_acc, tflite_auc, tflite_time, tflite_size_kb, times):
    """Save results summary to file"""
    summary_path = output_dir / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Float32 Model:\n")
        f.write(f"  AUC: {float_auc:.4f}\n\n")
        f.write("TFLite Int8 Model:\n")
        f.write(f"  Accuracy: {tflite_acc:.4f}\n")
        f.write(f"  AUC: {tflite_auc:.4f}\n")
        f.write(f"  Inference Time: {tflite_time:.2f}ms\n")
        f.write(f"  Model Size: {tflite_size_kb:.2f} KB\n")
        f.write(
            f"  AUC Degradation: {(float_auc - tflite_auc):.4f} ({(float_auc - tflite_auc) / float_auc * 100:.2f}%)\n\n")
        f.write("Timing Breakdown:\n")
        for phase, duration in times.items():
            if duration > 1.0:
                f.write(f"  {phase.capitalize()}: {format_time(duration)}\n")


def verify_model_features(model):
    """Verify that the model implements required features"""
    logger.info("=" * 60)
    logger.info("VERIFYING MODEL FEATURES")
    logger.info("=" * 60)

    # Check for GAP
    has_gap = any('global_average_pooling' in layer.name.lower() for layer in model.layers)
    logger.info(f"✓ Global Average Pooling (GAP): {'PRESENT' if has_gap else 'MISSING'}")
    assert has_gap, "Model must have GlobalAveragePooling2D layer"

    # Check for Frequency Emphasis
    has_freq_emphasis = any('frequencyemphasis' in layer.__class__.__name__.lower() for layer in model.layers)
    logger.info(f"✓ Frequency Emphasis: {'PRESENT' if has_freq_emphasis else 'MISSING'}")
    assert has_freq_emphasis, "Model must have FrequencyEmphasis layer"

    # Check for 1x1 convolution
    has_1x1_conv = False
    for layer in model.layers:
        if 'conv2d' in layer.__class__.__name__.lower():
            config = layer.get_config()
            kernel_size = config.get('kernel_size', None)
            if kernel_size == (1, 1) or kernel_size == [1, 1]:
                has_1x1_conv = True
                break
    logger.info(f"✓ 1x1 Pointwise Convolution: {'PRESENT' if has_1x1_conv else 'MISSING'}")
    assert has_1x1_conv, "Model must have 1x1 convolution before GAP"

    logger.info("✓ Focal Loss: WILL BE APPLIED (configured in model compilation)")
    logger.info("=" * 60)
    logger.info("All required features verified successfully!")
    logger.info("=" * 60)


def main():
    """Main training pipeline"""
    print("🟥" * 30)
    print(os.path.basename(__file__))
    print("🟥" * 30)

    start_time = time.time()
    args = parse_arguments()

    # Configure environment
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Setup configuration
    config = TrainingConfig()
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.n_fft = args.n_fft
    config.n_mels = args.n_mels
    config.update_derived_paths()

    # Set seeds
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # System info
    system_info = get_system_info()
    logger.info("=" * 60)
    logger.info("7e: Strided Conv + Focal Loss Tuned (α=0.75, γ=1.5)")
    logger.info("=" * 60)
    logger.info(f"Platform: {system_info['platform']} {system_info['machine']}")
    logger.info(f"TensorFlow: {system_info['tensorflow_version']}")
    logger.info(f"GPU Available: {system_info['gpu_available']}")
    logger.info("=" * 60)

    save_config(config, output_dir, args, system_info)

    times = {}

    try:
        # Preprocessing
        preprocess_start = time.time()
        if args.use_cache:
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found. Cannot use --use_cache.")
            logger.info(f"Using cached spectrograms from {config.cache_dir}")
        else:
            preprocess_and_cache_mels(config.dataset_path, config, args.force_reprocess)
        times['preprocessing'] = time.time() - preprocess_start

        # Load datasets
        logger.info("Creating datasets from cache...")
        train_dataset, train_counts = create_tf_dataset('train', config, augment=True)
        val_dataset, _ = create_tf_dataset('val', config, augment=False)
        test_dataset, _ = create_tf_dataset('test', config, augment=False)

        logger.info("=" * 60)
        logger.info(f"Train distribution: {train_counts}")
        logger.info("=" * 60)

        # Build model
        model = build_model(input_shape=(184, config.n_mels, 1), num_classes=2)

        # Verify model has all required features
        verify_model_features(model)

        logger.info("=" * 60)
        logger.info("Model Architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("=" * 60)

        with open(output_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Training
        trainer = ModelTrainer(model, config)
        evaluator = ModelEvaluator(output_dir)

        train_start = time.time()
        history = trainer.train(train_dataset, val_dataset)
        times['training'] = time.time() - train_start
        logger.info(f"Training completed in {format_time(times['training'])}")

        evaluator.plot_training_history(history)

        # Evaluate float model
        logger.info("=" * 60)
        logger.info("Evaluating Float32 Model")
        float_auc = evaluator.evaluate_model(model, test_dataset, prefix='float_')
        logger.info(f"Float32 AUC: {float_auc:.4f}")

        model.save(str(output_dir / 'best_model.keras'))

        # Convert to TFLite
        tflite_model = convert_to_tflite(model, val_dataset, output_dir, args.repr_samples)

        tflite_path = output_dir / "model_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        logger.info(f"Saved TFLite model: {tflite_path} ({tflite_size_kb:.2f} KB)")

        # Evaluate TFLite
        logger.info("=" * 60)
        logger.info("Evaluating TFLite Int8 Model")
        tflite_acc, tflite_auc, tflite_time = evaluator.evaluate_tflite(tflite_path, test_dataset)

        # Save results
        save_results_summary(output_dir, float_auc, tflite_acc, tflite_auc,
                            tflite_time, tflite_size_kb, times)

        logger.info("=" * 60)
        logger.info(f"Float32 AUC: {float_auc:.4f}")
        logger.info(f"TFLite Acc: {tflite_acc:.4f}, AUC: {tflite_auc:.4f}")
        logger.info(f"TFLite Inference: {tflite_time:.2f}ms")
        logger.info(f"TFLite Size: {tflite_size_kb:.2f} KB")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        elapsed_path = output_dir / 'elapsed.txt'
        with open(elapsed_path, 'w') as f:
            f.write(f"Total execution time: {format_time(total_time)}\n")
            f.write(f"Total seconds: {total_time:.3f}\n")
        logger.info(f"Completed in {format_time(total_time)}")


if __name__ == '__main__':
    main()
