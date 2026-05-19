#!/usr/bin/env python3
"""
Standard CNN Architecture Training - VGG16, ResNet50, EfficientNetB0, MobileNetV3-Small
Transfer learning from ImageNet pretrained weights on MyBAD dataset
For dataset validation (Table 10 in paper)
Compatible with Linux (Ubuntu 22.04, Python 3.10, TensorFlow 2.15)
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
import shutil

# Parse --force_cpu early, before TensorFlow import
def _parse_early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    args, _ = parser.parse_known_args()
    return args

_early_args = _parse_early_args()

# Disable GPU if requested (must be done before TensorFlow import)
if _early_args.force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

for i in range(101):
    w=shutil.get_terminal_size().columns-9
    print(f"\r\033[92m{'█'*int(i*w/100)}\033[90m{'░'*(w-int(i*w/100))}\033[0m {i:3}%",end='',flush=1)
    time.sleep(0.03)
print()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# Configure GPU memory growth
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

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    target_sr: int = 16000
    target_length: int = 16000 * 3  # 3 seconds
    # Mel spectrogram parameters for standard architectures
    img_height: int = 224
    img_width: int = 224
    n_mels: int = 224  # Match image height
    n_fft: int = 2048  # Higher for better frequency resolution
    hop_length: int = 256
    # Learning rate schedule
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 15
    random_seed: int = 42
    # Path configurations
    dataset_path: str = '/Volumes/Evo/mybad'
    output_dir: str = 'results/standard_arch'
    cache_dir: str = '/Volumes/Evo/cache_mybad_224x224'
    # Train/val/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Architecture
    architecture: str = 'vgg16'  # vgg16, resnet50, efficientnetb0, mobilenetv3small

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train standard CNN architectures on MyBAD')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['vgg16', 'resnet50', 'efficientnetb0', 'mobilenetv3small'],
                        help='Architecture to train')
    parser.add_argument('--dataset-path', type=str, default='/Volumes/Evo/mybad',
                        help='Path to dataset directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing even if cache exists')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached spectrograms (fails if cache does not exist)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force use of CPU instead of GPU')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for spectrograms')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    return parser.parse_args()

def get_optimizer(learning_rate: float):
    """Get optimizer based on platform"""
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == 'Darwin' and machine == 'arm64'

    if is_apple_silicon:
        logger.info("Detected Apple Silicon - using legacy Adam")
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif system == 'Linux':
        logger.info("Detected Linux - using AdamW with weight_decay=0.01")
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    else:
        logger.info(f"Detected {system} - using standard Adam")
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def build_model(architecture: str, num_classes: int = 2):
    """
    Build standard CNN architecture with ImageNet pretrained weights

    Args:
        architecture: One of 'vgg16', 'resnet50', 'efficientnetb0', 'mobilenetv3small'
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    input_shape = (224, 224, 3)  # Standard ImageNet input size

    logger.info(f"Building {architecture.upper()} model...")

    if architecture == 'vgg16':
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        model_name = 'VGG16'

    elif architecture == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        model_name = 'ResNet50'

    elif architecture == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        model_name = 'EfficientNetB0'

    elif architecture == 'mobilenetv3small':
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg',
            minimalistic=False
        )
        model_name = 'MobileNetV3-Small'
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Build full model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Freeze base initially
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs, name=model_name)

    # Initially freeze base model
    base_model.trainable = False

    logger.info(f"Built {model_name} with {base_model.count_params():,} base parameters")
    logger.info(f"Base model frozen for initial training")

    return model, base_model

class MyBADv2Dataset:
    """Dataset class for MyBADv2 - returns file paths and labels"""
    def __init__(self, root_dir: str, split='train', seed=42):
        self.root_dir = root_dir
        self.split = split
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
            raise ValueError(f"Invalid split: {split}")

        random.shuffle(subset)
        self.files = [f[0] for f in subset]
        self.labels = [f[1] for f in subset]

        logger.info(f"Loaded {len(self.files)} files for {split} "
              f"({len([l for l in self.labels if l == 0])} negative, "
              f"{len([l for l in self.labels if l == 1])} positive)")

    def __len__(self):
        return len(self.files)

    def get_files_and_labels(self):
        return self.files, self.labels

def compute_mel_spectrogram_image(waveform: np.ndarray, config: TrainingConfig) -> np.ndarray:
    """
    Compute mel spectrogram as 224x224 RGB image for standard architectures

    Args:
        waveform: Audio waveform
        config: Training configuration

    Returns:
        224x224x3 RGB image
    """
    # Compute mel spectrogram
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

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to 224x224
    from scipy.ndimage import zoom
    current_shape = mel_spec_db.shape
    zoom_factors = (config.img_height / current_shape[0], config.img_width / current_shape[1])
    mel_spec_resized = zoom(mel_spec_db, zoom_factors, order=1)

    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min() + 1e-8)

    # Convert to RGB by repeating across 3 channels
    mel_spec_rgb = np.stack([mel_spec_norm, mel_spec_norm, mel_spec_norm], axis=-1)

    return mel_spec_rgb.astype(np.float32)

def preprocess_and_cache(dataset_path: str, config: TrainingConfig, force_reprocess: bool = False):
    """Preprocess audio and cache 224x224 mel spectrograms"""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    cache_info_path = cache_dir / 'cache_info.pkl'
    if cache_info_path.exists() and not force_reprocess:
        logger.info(f"Cache exists at {cache_dir}. Skipping preprocessing.")
        return

    logger.info("="*60)
    logger.info("Preprocessing audio and caching 224x224 mel spectrograms...")
    logger.info("="*60)

    splits = ['train', 'val', 'test']
    cache_info = {}

    for split in splits:
        logger.info(f"Processing {split} split...")

        dataset = MyBADv2Dataset(dataset_path, split=split, seed=config.random_seed)
        file_paths, labels = dataset.get_files_and_labels()

        split_cache_dir = cache_dir / split
        split_cache_dir.mkdir(exist_ok=True)

        mel_images = []
        valid_labels = []

        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc=f"Processing {split}"):
            try:
                # Load audio
                waveform, sr = librosa.load(file_path, sr=None)

                # Resample if needed
                if sr != config.target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.target_sr)

                # Pad or truncate
                if len(waveform) > config.target_length:
                    waveform = waveform[:config.target_length]
                elif len(waveform) < config.target_length:
                    pad = np.zeros(config.target_length - len(waveform))
                    waveform = np.concatenate([waveform, pad])

                # Compute mel spectrogram as RGB image
                mel_image = compute_mel_spectrogram_image(waveform, config)

                mel_images.append(mel_image)
                valid_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        # Convert to numpy arrays
        mel_images = np.array(mel_images, dtype=np.float32)
        valid_labels = np.array(valid_labels, dtype=np.int32)

        logger.info(f"  Processed {len(mel_images)} samples")
        logger.info(f"  Shape: {mel_images.shape}")

        # Save to cache
        cache_file = split_cache_dir / 'mels_224x224.npz'
        np.savez_compressed(cache_file, mels=mel_images, labels=valid_labels)
        logger.info(f"  Saved to {cache_file}")

        cache_info[split] = {
            'n_samples': len(mel_images),
            'shape': mel_images.shape,
            'cache_file': str(cache_file)
        }

    # Save cache info
    with open(cache_info_path, 'wb') as f:
        pickle.dump(cache_info, f)

    logger.info("="*60)
    logger.info("Preprocessing complete!")
    logger.info("="*60)

def load_cached_data(split: str, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached 224x224 mel spectrograms"""
    cache_dir = Path(config.cache_dir)
    cache_file = cache_dir / split / 'mels_224x224.npz'

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    data = np.load(cache_file)
    mel_images = data['mels']
    labels = data['labels']

    logger.info(f"Loaded {len(mel_images)} cached samples for {split}")
    logger.info(f"  Shape: {mel_images.shape}")

    return mel_images, labels

def create_tf_dataset(split: str, config: TrainingConfig, augment: bool = False) -> Tuple[tf.data.Dataset, Dict[int, int]]:
    """Create tf.data.Dataset from cached data"""
    mel_images, labels = load_cached_data(split, config)

    # Count samples per class
    class_counts = {0: np.sum(labels == 0), 1: np.sum(labels == 1)}
    logger.info(f"  Class distribution - Negative: {class_counts[0]}, Positive: {class_counts[1]}")

    # Create dataset
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((mel_images, labels))

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(mel_images), seed=config.random_seed)

    # Convert labels to one-hot
    def to_one_hot(img, label):
        label_onehot = tf.one_hot(label, depth=2)
        return img, label_onehot

    dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation if requested
    if augment:
        def augment_image(img, label):
            # Random brightness
            img = tf.image.random_brightness(img, max_delta=0.1)
            # Random contrast
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            # Clip to valid range
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_counts

class ModelTrainer:
    """Handle model training with two-stage approach"""
    def __init__(self, model: tf.keras.Model, base_model, config: TrainingConfig):
        self.model = model
        self.base_model = base_model
        self.config = config

        # Compile with frozen base
        self.model.compile(
            optimizer=get_optimizer(config.learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(Path(config.output_dir) / 'best_model.keras'),
                monitor='val_auc', mode='max', save_best_only=True
            )
        ]

    def train_stage1_frozen(self, train_dataset, val_dataset, epochs: int = 20):
        """Stage 1: Train with frozen base"""
        logger.info("="*60)
        logger.info("STAGE 1: Training with frozen base model")
        logger.info("="*60)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        return history.history

    def train_stage2_unfrozen(self, train_dataset, val_dataset, epochs: int):
        """Stage 2: Unfreeze and fine-tune"""
        logger.info("="*60)
        logger.info("STAGE 2: Fine-tuning with unfrozen base model")
        logger.info("="*60)

        # Unfreeze base model
        self.base_model.trainable = True
        logger.info(f"Unfroze base model - now has {self.base_model.count_params():,} trainable parameters")

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=get_optimizer(self.config.learning_rate * 0.1),  # 10x lower LR
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        return history.history

class ModelEvaluator:
    """Handle model evaluation and visualization"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(history['loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # AUC
        axes[1, 0].plot(epochs, history['auc'], 'b-', label='Train AUC')
        axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Val AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # F1 (calculated from precision and recall)
        if 'precision' in history and 'recall' in history:
            train_f1 = 2 * (np.array(history['precision']) * np.array(history['recall'])) / (np.array(history['precision']) + np.array(history['recall']) + 1e-8)
            val_f1 = 2 * (np.array(history['val_precision']) * np.array(history['val_recall'])) / (np.array(history['val_precision']) + np.array(history['val_recall']) + 1e-8)
            axes[1, 1].plot(epochs, train_f1, 'b-', label='Train F1')
            axes[1, 1].plot(epochs, val_f1, 'r-', label='Val F1')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

    def evaluate_model(self, model, test_dataset):
        """Evaluate model and generate visualizations"""
        predictions, probabilities = self._get_predictions(model, test_dataset)
        true_labels = self._get_labels(test_dataset)

        # Compute metrics
        acc = np.mean(predictions == true_labels)
        auc = roc_auc_score(true_labels, probabilities)

        # Generate visualizations
        self._plot_confusion_matrix(true_labels, predictions, 'test_')
        self._plot_roc_curve(true_labels, probabilities, 'test_')
        self._save_classification_report(true_labels, predictions, 'test_')

        return acc, auc

    def _get_predictions(self, model, dataset):
        predictions = []
        probabilities = []
        for inputs, _ in dataset:
            outputs = model(inputs, training=False)
            predictions.extend(np.argmax(outputs, axis=1))
            probabilities.extend(outputs[:, 1].numpy())
        return np.array(predictions), np.array(probabilities)

    def _get_labels(self, dataset):
        labels = []
        for _, lbl in dataset:
            lbl_indices = np.argmax(lbl.numpy(), axis=1)
            labels.extend(lbl_indices)
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

    def _save_classification_report(self, true_labels, predictions, prefix):
        report = classification_report(
            true_labels, predictions,
            target_names=['Negative', 'Positive'],
            digits=4,
            zero_division=0
        )
        with open(self.output_dir / f'{prefix}classification_report.txt', 'w') as f:
            f.write(report)

def format_time(seconds: float) -> str:
    """Format seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:05.2f}s"
    else:
        return f"{minutes}m {secs:05.2f}s"

def save_config(config, output_dir, args, system_info):
    """Save configuration to file"""
    config_path = output_dir / 'config.txt'

    with open(config_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"STANDARD CNN ARCHITECTURE TRAINING - {config.architecture.upper()}\n")
        f.write("="*60 + "\n\n")

        f.write("System Information:\n")
        f.write(f"  Platform: {system_info['platform']}\n")
        f.write(f"  Machine: {system_info['machine']}\n")
        f.write(f"  Python: {system_info['python_version']}\n")
        f.write(f"  TensorFlow: {system_info['tensorflow_version']}\n")
        f.write(f"  GPU Available: {system_info['gpu_available']}\n")
        if system_info['gpu_devices']:
            f.write(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}\n")
        f.write("\n")

        f.write("Model Configuration:\n")
        f.write(f"  Architecture: {config.architecture}\n")
        f.write(f"  Input Shape: (224, 224, 3)\n")
        f.write(f"  Pretrained Weights: ImageNet\n")
        f.write("\n")

        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {config.epochs}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  Random Seed: {config.random_seed}\n")
        f.write("\n")

        f.write("Paths:\n")
        f.write(f"  Dataset: {config.dataset_path}\n")
        f.write(f"  Output: {config.output_dir}\n")
        f.write(f"  Cache: {config.cache_dir}\n")
        f.write("\n")

    logger.info(f"Saved configuration to {config_path}")

def main():
    args = parse_args()
    start_time = time.time()

    # Setup configuration
    config = TrainingConfig()
    config.architecture = args.architecture
    config.random_seed = args.random_seed
    config.dataset_path = args.dataset_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    if args.cache_dir:
        config.cache_dir = args.cache_dir
    else:
        config.cache_dir = f'/Volumes/Evo/cache_mybad_224x224'

    config.output_dir = f'results/standard_{args.architecture}_s{config.random_seed}'

    # Set random seeds
    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # System info
    system_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')],
    }

    logger.info("="*60)
    logger.info(f"Standard CNN Architecture Training - {args.architecture.upper()}")
    logger.info("="*60)
    logger.info("System Information:")
    logger.info(f"  Platform: {system_info['platform']} {system_info['machine']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  TensorFlow: {system_info['tensorflow_version']}")
    logger.info(f"  GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_devices']:
        logger.info(f"  GPU Devices: {', '.join(system_info['gpu_devices'])}")
    logger.info("="*60)

    # Save configuration
    save_config(config, output_dir, args, system_info)

    try:
        # Preprocess and cache
        if args.use_cache:
            cache_info_path = Path(config.cache_dir) / 'cache_info.pkl'
            if not cache_info_path.exists():
                raise FileNotFoundError(f"Cache not found at {config.cache_dir}")
            logger.info(f"Using cached data from {config.cache_dir}")
        else:
            preprocess_and_cache(config.dataset_path, config, force_reprocess=args.force_reprocess)

        # Create datasets
        logger.info("Creating tf.data datasets...")
        train_dataset, train_class_counts = create_tf_dataset('train', config, augment=True)
        val_dataset, val_class_counts = create_tf_dataset('val', config, augment=False)
        test_dataset, test_class_counts = create_tf_dataset('test', config, augment=False)

        # Build model
        model, base_model = build_model(config.architecture, num_classes=2)

        logger.info("="*60)
        logger.info("Model Architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info("="*60)

        # Save model summary
        with open(output_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Train
        trainer = ModelTrainer(model, base_model, config)
        evaluator = ModelEvaluator(output_dir)

        # Stage 1: Frozen base (20 epochs)
        train_start = time.time()
        history1 = trainer.train_stage1_frozen(train_dataset, val_dataset, epochs=20)

        # Stage 2: Unfrozen fine-tuning (remaining epochs)
        history2 = trainer.train_stage2_unfrozen(train_dataset, val_dataset, epochs=config.epochs - 20)

        # Combine histories
        history = {}
        for key in history1.keys():
            history[key] = history1[key] + history2[key]

        train_time = time.time() - train_start
        logger.info(f"Training completed in {format_time(train_time)}")

        evaluator.plot_training_history(history)

        # Evaluate
        logger.info("="*60)
        logger.info("Evaluating on test set...")
        test_acc, test_auc = evaluator.evaluate_model(model, test_dataset)

        # Save model
        model.save(str(output_dir / 'best_model.keras'))
        logger.info(f"Saved model to {output_dir / 'best_model.keras'}")

        # Save results summary
        total_time = time.time() - start_time
        summary_path = output_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"Standard Architecture Results - {config.architecture.upper()}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            f.write(f"Total Parameters: {model.count_params():,}\n")
            f.write(f"\nTraining Time: {format_time(train_time)}\n")
            f.write(f"Total Time: {format_time(total_time)}\n")

        logger.info("="*60)
        logger.info("Results Summary:")
        logger.info(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"  Test AUC: {test_auc:.4f}")
        logger.info(f"  Total Parameters: {model.count_params():,}")
        logger.info(f"  Training Time: {format_time(train_time)}")
        logger.info(f"  Total Time: {format_time(total_time)}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
    finally:
        with open(output_dir / 'elapsed.txt', 'w') as f:
            elapsed = time.time() - start_time
            f.write(f"Total execution time: {format_time(elapsed)}\n")
            f.write(f"Total seconds: {elapsed:.3f}\n")

if __name__ == '__main__':
    main()
