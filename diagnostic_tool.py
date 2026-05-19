#!/usr/bin/env python3
"""
Dataset Diagnostic Tool for Gatekeeper Model

Analyzes:
1. Class distribution and imbalance ratio
2. Sample predictions to understand model behavior
3. Confusion matrix analysis
4. Threshold sensitivity analysis
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose gatekeeper model issues')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model (.keras file)')
    parser.add_argument('--cache_dir', type=str, default='/Volumes/Evo/cache_mybad_waveforms',
                        help='Path to cached waveforms')
    parser.add_argument('--output_dir', type=str, default='diagnostics',
                        help='Output directory for diagnostic plots')
    return parser.parse_args()

args = parse_args()

# Load test data
cache_dir = Path(args.cache_dir)
test_cache = cache_dir / 'testing' / 'waveforms.npz'

print("=" * 60)
print("GATEKEEPER MODEL DIAGNOSTICS")
print("=" * 60)

data = np.load(test_cache)
X_test = data['waveforms']
y_test = data['labels']

print(f"\nTest set size: {len(X_test)}")
print(f"Negative samples (class 0): {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"Positive samples (class 1): {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
print(f"Imbalance ratio: {max(np.sum(y_test == 0), np.sum(y_test == 1)) / min(np.sum(y_test == 0), np.sum(y_test == 1)):.2f}:1")

# Load model
print(f"\nLoading model from {args.model_path}...")
model = tf.keras.models.load_model(args.model_path)

# Add channel dimension
X_test = X_test[..., np.newaxis]

# Get predictions
print("Generating predictions...")
predictions_proba = model.predict(X_test, verbose=0)
predictions_class = np.argmax(predictions_proba, axis=1)
predictions_positive_proba = predictions_proba[:, 1]

print("\n" + "=" * 60)
print("PREDICTION ANALYSIS")
print("=" * 60)

# Basic stats
print(f"\nPredictions:")
print(f"  Predicted negative: {np.sum(predictions_class == 0)} ({np.sum(predictions_class == 0)/len(predictions_class)*100:.1f}%)")
print(f"  Predicted positive: {np.sum(predictions_class == 1)} ({np.sum(predictions_class == 1)/len(predictions_class)*100:.1f}%)")

print(f"\nPrediction probabilities for positive class:")
print(f"  Min: {predictions_positive_proba.min():.4f}")
print(f"  Max: {predictions_positive_proba.max():.4f}")
print(f"  Mean: {predictions_positive_proba.mean():.4f}")
print(f"  Median: {np.median(predictions_positive_proba):.4f}")
print(f"  Std: {predictions_positive_proba.std():.4f}")

# Analyze by true class
print(f"\nProbabilities by TRUE class:")
pos_probs = predictions_positive_proba[y_test == 1]
neg_probs = predictions_positive_proba[y_test == 0]

print(f"  TRUE POSITIVE samples:")
print(f"    Mean prob: {pos_probs.mean():.4f}")
print(f"    Median prob: {np.median(pos_probs):.4f}")
print(f"    Min prob: {pos_probs.min():.4f}")
print(f"    Max prob: {pos_probs.max():.4f}")

print(f"  TRUE NEGATIVE samples:")
print(f"    Mean prob: {neg_probs.mean():.4f}")
print(f"    Median prob: {np.median(neg_probs):.4f}")
print(f"    Min prob: {neg_probs.min():.4f}")
print(f"    Max prob: {neg_probs.max():.4f}")

# Confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX (threshold=0.5)")
print("=" * 60)
cm = confusion_matrix(y_test, predictions_class)
tn, fp, fn, tp = cm.ravel()

print(f"\n  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}  ⚠️ MISSED BIRDS")
print(f"  True Positives (TP):  {tp}")

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"\n  Recall: {recall:.4f}")
print(f"  Precision: {precision:.4f}")

# Threshold analysis
print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
print("-" * 60)

for thresh in thresholds:
    preds = (predictions_positive_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{thresh:<12.2f} {recall:<10.4f} {precision:<12.4f} {tp:<8} {fp:<8} {fn:<8}")

best_thresh = None
best_precision = 0
target_recall = 0.98

# Find best threshold for 95% recall
print("\n" + "=" * 60)
print(f"FINDING OPTIMAL THRESHOLD FOR {target_recall*100}% RECALL")
print("=" * 60)


for thresh in np.linspace(0, 1, 1000):
    preds = (predictions_positive_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    if recall >= target_recall and precision > best_precision:
        best_thresh = thresh
        best_precision = precision

if best_thresh is not None:
    preds = (predictions_positive_proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nOptimal threshold: {best_thresh:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
else:
    print("\n⚠️  Cannot achieve 95% recall at any threshold!")
    print("This indicates the model is fundamentally not learning to detect positive class.")

# Sample analysis
print("\n" + "=" * 60)
print("SAMPLE ANALYSIS: Missed Detections (False Negatives)")
print("=" * 60)

fn_indices = np.where((y_test == 1) & (predictions_class == 0))[0]
if len(fn_indices) > 0:
    fn_probs = predictions_positive_proba[fn_indices]
    print(f"\nTotal false negatives: {len(fn_indices)}")
    print(f"Predicted probabilities for missed birds:")
    print(f"  Mean: {fn_probs.mean():.4f}")
    print(f"  Median: {np.median(fn_probs):.4f}")
    print(f"  Min: {fn_probs.min():.4f}")
    print(f"  Max: {fn_probs.max():.4f}")
    print(f"\nTop 10 highest-confidence misses:")
    top_indices = fn_indices[np.argsort(fn_probs)[-10:]]
    for idx in reversed(top_indices):
        print(f"  Index {idx}: prob={predictions_positive_proba[idx]:.4f}")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

if recall < 0.3:
    print("\n🔴 CRITICAL: Model is severely biased toward negative class!")
    print("   Possible causes:")
    print("   1. Class imbalance not properly handled")
    print("   2. Model architecture too simple for the task")
    print("   3. Training data quality issues")
    print("   4. Loss function not appropriate for imbalanced data")
    print("\n   Recommendations:")
    print("   ✓ Use class weights or focal loss")
    print("   ✓ Check if positive samples are actually different from negative")
    print("   ✓ Try oversampling positive class")
    print("   ✓ Increase model capacity")
    
elif recall < 0.7:
    print("\n🟡 WARNING: Model has poor recall")
    print("   Recommendations:")
    print("   ✓ Adjust decision threshold (see threshold analysis above)")
    print("   ✓ Use class weights")
    print("   ✓ Try focal loss")
    
elif recall < target_recall:
    print(f"\n🟡 Model recall ({recall:.2%}) below target ({target_recall:.2%})")
    print("   Recommendations:")
    print("   ✓ Adjust decision threshold for deployment")
    print(f"   ✓ Use threshold {best_thresh:.4f} to achieve target recall")
else:
    print(f"\n🟢 Model meets target recall!")

print("\n" + "=" * 60)
