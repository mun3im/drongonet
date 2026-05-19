#!/usr/bin/env python3
"""
Combine existing ROC curve images into comparison figures
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

FIGURES_DIR = Path('figures_publication')
RESULTS_DIR = Path('results')
RESULTS_LINUX_DIR = Path('results_linux')

def combine_roc_key_models():
    """Combine ROC curves for key models into a grid"""
    print("\nGenerating Combined ROC Curves Figure...")

    # Key models to include
    models = [
        ('1a_baseline2d_fft512_m80_s42', '1a Baseline m80\n(AUC=0.9833)'),
        ('4a_baseline_gap_fft512_m48_s42', '4a GAP m48\n(AUC=0.9568)'),
        ('7e_strided_focal_tuned_fft512_m16_s42', '7e Strided m16\n(AUC=0.9685)'),
        ('7e_strided_focal_tuned_fft512_m80_s42', '7e Strided m80\n(AUC=0.9952)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (model_dir, title) in enumerate(models):
        roc_path = RESULTS_DIR / model_dir / 'tflite_roc_curve.png'
        if roc_path.exists():
            img = mpimg.imread(str(roc_path))
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontweight='bold', fontsize=12)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'ROC not found:\n{model_dir}',
                          ha='center', va='center', fontsize=10)
            axes[idx].axis('off')

    plt.suptitle('XiaoChirp TFLite int8 ROC Curves Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curves_comparison.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'roc_curves_comparison.pdf')
    print(f"  Saved: {FIGURES_DIR}/roc_curves_comparison.png/.pdf")
    plt.close()

def combine_nmels_progression():
    """Show ROC curve progression with n_mels for 7e model"""
    print("\nGenerating n_mels ROC Progression Figure...")

    # 7e models across n_mels values
    nmels_values = [16, 32, 48, 64, 80]
    auc_values = [0.9685, 0.9764, 0.9792, 0.9796, 0.9952]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (n_mels, auc_val) in enumerate(zip(nmels_values, auc_values)):
        model_dir = f'7e_strided_focal_tuned_fft512_m{n_mels}_s42'
        roc_path = RESULTS_DIR / model_dir / 'tflite_roc_curve.png'

        if roc_path.exists():
            img = mpimg.imread(str(roc_path))
            axes[idx].imshow(img)
            axes[idx].set_title(f'm{n_mels} (AUC={auc_val:.4f})', fontweight='bold', fontsize=11)
        else:
            axes[idx].text(0.5, 0.5, f'Not found', ha='center', va='center')

        axes[idx].axis('off')

    plt.suptitle('7e Strided Model: ROC Curves Across n_mels Values (fft512)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_nmels_progression.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'roc_nmels_progression.pdf')
    print(f"  Saved: {FIGURES_DIR}/roc_nmels_progression.png/.pdf")
    plt.close()

def combine_confusion_matrices():
    """Combine confusion matrices for key models"""
    print("\nGenerating Combined Confusion Matrices Figure...")

    models = [
        ('7e_strided_focal_tuned_fft512_m16_s42', 'Gatekeeper (m16)'),
        ('7e_strided_focal_tuned_fft512_m80_s42', 'Best 7e (m80)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (model_dir, title) in enumerate(models):
        cm_path = RESULTS_DIR / model_dir / 'tflite_confusion_matrix.png'
        if cm_path.exists():
            img = mpimg.imread(str(cm_path))
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontweight='bold', fontsize=12)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'Not found', ha='center', va='center')
            axes[idx].axis('off')

    plt.suptitle('XiaoChirp TFLite int8 Confusion Matrices', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrices_comparison.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'confusion_matrices_comparison.pdf')
    print(f"  Saved: {FIGURES_DIR}/confusion_matrices_comparison.png/.pdf")
    plt.close()

def combine_training_histories():
    """Combine training history plots"""
    print("\nGenerating Combined Training Histories Figure...")

    models = [
        ('1a_baseline2d_fft512_m32_s42', '1a Baseline m32'),
        ('7e_strided_focal_tuned_fft512_m32_s42', '7e Strided m32'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for idx, (model_dir, title) in enumerate(models):
        hist_path = RESULTS_DIR / model_dir / 'training_history.png'
        if hist_path.exists():
            img = mpimg.imread(str(hist_path))
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontweight='bold', fontsize=12)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'Not found', ha='center', va='center')
            axes[idx].axis('off')

    plt.suptitle('Training History Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_histories_comparison.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'training_histories_comparison.pdf')
    print(f"  Saved: {FIGURES_DIR}/training_histories_comparison.png/.pdf")
    plt.close()

def main():
    """Generate all combined figures"""
    print("\n" + "="*70)
    print("COMBINING EXISTING ROC/CM FIGURES")
    print("="*70)

    FIGURES_DIR.mkdir(exist_ok=True)

    combine_roc_key_models()
    combine_nmels_progression()
    combine_confusion_matrices()
    combine_training_histories()

    print("\n" + "="*70)
    print("Combined figures generated!")
    print("="*70)

if __name__ == '__main__':
    main()
