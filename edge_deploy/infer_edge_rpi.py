#!/usr/bin/env python3
"""
infer_edge_rpi.py — SEABADNet-Edge inference on Raspberry Pi.

Evaluates pre-trained SEABADNet-Edge (33.06 KB INT8) on SEABAD test set.
Lightweight deployment: loads TFLite model and SEABAD dataset only.

Usage:
    python deploy/infer_edge_rpi.py \\
        --dataset-path /path/to/seabad \\
        --model seabadnet_edge_int8.tflite \\
        --threshold 0.60

Output:
    Prints AUC, accuracy, recall, precision, F1, and per-sample predictions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_curve
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SEABADNetInference:
    """TFLite inference engine for SEABADNet-Edge."""

    def __init__(self, model_path: str, n_mels: int = 80, n_fft: int = 1024):
        """Initialize TFLite interpreter and mel parameters."""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = 256
        self.target_sr = 16000
        self.target_length = 48000  # 3 seconds @ 16 kHz

        logger.info(f"✓ Model loaded: {Path(model_path).name}")
        logger.info(f"  Input shape: {self.input_details[0]['shape']}")
        logger.info(f"  Output shape: {self.output_details[0]['shape']}")

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        # Ensure correct length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Reshape for model: (n_mels, time_steps, 1)
        mel_spec = np.expand_dims(S_dB, axis=-1).astype(np.float32)
        return mel_spec

    def predict(self, mel_spec: np.ndarray) -> float:
        """Run inference on mel spectrogram; returns bird-positive probability in [0, 1]."""
        input_data = np.expand_dims(mel_spec, axis=0).astype(
            self.input_details[0]['dtype']
        )

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Output is [batch_size, 2] int8 (2-class softmax: [no_bird, bird]).
        # Dequantize column 1 (bird class) using the model's output scale/zero-point.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        return float(scale * (int(output_data[0, 1]) - zero_point))


def load_seabad_test_set(dataset_path: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load SEABAD test set (10% of data)."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    positive_dir = dataset_path / "positive"
    negative_dir = dataset_path / "negative"

    if not positive_dir.exists() or not negative_dir.exists():
        raise FileNotFoundError(
            f"Expected 'positive' and 'negative' subdirs in {dataset_path}"
        )

    positive_files = sorted(positive_dir.glob("*.wav"))
    negative_files = sorted(negative_dir.glob("*.wav"))

    # Split: train 80%, val 10%, test 10%
    pos_split = int(len(positive_files) * 0.9)  # First 90% for train+val
    test_pos = positive_files[pos_split:]

    neg_split = int(len(negative_files) * 0.9)
    test_neg = negative_files[neg_split:]

    logger.info(f"Loading test set: {len(test_pos)} positive, {len(test_neg)} negative")

    audio_list = []
    labels = []
    filenames = []

    # Load positive examples
    for fpath in test_pos:
        try:
            audio, sr = librosa.load(str(fpath), sr=16000, mono=True)
            audio_list.append(audio)
            labels.append(1)
            filenames.append(fpath.name)
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    # Load negative examples
    for fpath in test_neg:
        try:
            audio, sr = librosa.load(str(fpath), sr=16000, mono=True)
            audio_list.append(audio)
            labels.append(0)
            filenames.append(fpath.name)
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    return audio_list, labels, filenames


def main():
    parser = argparse.ArgumentParser(
        description="SEABADNet-Edge inference on Raspberry Pi"
    )
    parser.add_argument('--dataset-path', required=True,
                        help='Path to SEABAD dataset root')
    parser.add_argument('--model', default='deploy/seabadnet_edge_int8.tflite',
                        help='Path to TFLite model (default: deploy/seabadnet_edge_int8.tflite)')
    parser.add_argument('--threshold', type=float, default=0.60,
                        help='Decision threshold for binary classification (default: 0.60)')
    parser.add_argument('--output', default=None,
                        help='Output CSV file for results (optional)')

    args = parser.parse_args()

    # Load model
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    engine = SEABADNetInference(args.model)

    # Load test set
    try:
        audio_list, labels, filenames = load_seabad_test_set(args.dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if len(audio_list) == 0:
        logger.error("No audio files loaded")
        sys.exit(1)

    # Run inference
    logger.info(f"Running inference on {len(audio_list)} test samples...")
    predictions = []

    for i, audio in enumerate(audio_list):
        mel_spec = engine.preprocess(audio)
        pred_prob = engine.predict(mel_spec)
        predictions.append(pred_prob)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1} / {len(audio_list)} ✓")

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Compute metrics
    pred_binary = (predictions >= args.threshold).astype(int)

    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, pred_binary)
    recall = recall_score(labels, pred_binary)
    precision = precision_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)

    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Print results
    print("\n" + "=" * 70)
    print("SEABADNet-Edge Evaluation Results")
    print("=" * 70)
    print(f"Model:          seabadnet_edge_int8.tflite")
    print(f"Dataset:        SEABAD test set ({len(audio_list)} samples)")
    print(f"Threshold (τ):  {args.threshold}")
    print("-" * 70)
    print(f"AUC:            {auc:.4f}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Specificity:    {specificity:.4f}")
    print(f"FPR:            {fpr:.4f}")
    print("-" * 70)
    print(f"Confusion Matrix:")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")
    print("=" * 70)

    # Save results if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'label', 'pred_prob', 'pred_binary',
                'correct'
            ])
            writer.writeheader()
            for fname, lbl, pred_p, pred_b in zip(
                filenames, labels, predictions, pred_binary
            ):
                writer.writerow({
                    'filename': fname,
                    'label': lbl,
                    'pred_prob': f"{pred_p:.4f}",
                    'pred_binary': pred_b,
                    'correct': int(lbl == pred_b)
                })
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
