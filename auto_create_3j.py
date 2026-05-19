#!/usr/bin/env python3
"""
Auto-create variant 3j after experiments complete
Monitors experiments, finds best dropout rate for depthwise, creates 3j variant
"""

import time
import subprocess
import pandas as pd
import re
from pathlib import Path

def wait_for_experiments():
    """Wait for all 50 experiments to complete"""
    print("Monitoring experiment progress...")

    while True:
        try:
            latest_log = sorted(Path('ablation_logs').glob('master_*.log'),
                              key=lambda x: x.stat().st_mtime)[-1]

            with open(latest_log) as f:
                content = f.read()
                success_count = content.count('✓ SUCCESS')

            if success_count >= 49:
                print(f"\n✓ Experiments completed ({success_count} successful)!")
                return True
            else:
                print(f"\rProgress: {success_count}/50 completed", end='', flush=True)
                time.sleep(30)  # Check every 30 seconds

        except Exception as e:
            print(f"\nWaiting for experiments to start... ({e})")
            time.sleep(30)

def analyze_depthwise_dropout_variants():
    """Analyze 3f, 3g, 3h, 3i to find best dropout rate"""
    print("\n\nAnalyzing depthwise + dropout variants...")

    variants = {
        'depthwise': ('3f', 0.0),
        'depthwise_d02': ('3g', 0.2),
        'depthwise_d03': ('3h', 0.3),
        'depthwise_d04': ('3i', 0.4)
    }

    results = []

    for prefix, (variant_code, dropout_rate) in variants.items():
        for n_mels in [16, 32, 48, 64, 80]:
            result_dir = Path(f'results_{prefix}_m{n_mels}_s42')
            summary_file = result_dir / 'results_summary.txt'

            if not summary_file.exists():
                continue

            content = summary_file.read_text()

            # Extract TFLite metrics
            tflite_match = re.search(
                r'TFLite int8 Model:.*?AUC:\s+([\d.]+).*?Model Size:\s+([\d.]+)\s+KB',
                content, re.DOTALL
            )

            if tflite_match:
                results.append({
                    'variant': variant_code,
                    'dropout': dropout_rate,
                    'n_mels': n_mels,
                    'auc': float(tflite_match.group(1))
                })

    df = pd.DataFrame(results)

    # Find best dropout rate (average across all n_mels)
    by_dropout = df.groupby('dropout')['auc'].agg(['mean', 'std']).round(4)
    print("\nDepthwise + Dropout Performance:")
    print(by_dropout)

    best_dropout = by_dropout['mean'].idxmax()
    best_variant = df[df['dropout'] == best_dropout].iloc[0]['variant']

    print(f"\n🏆 Best dropout rate: {best_dropout}")
    print(f"   Mean AUC: {by_dropout.loc[best_dropout, 'mean']:.4f}")
    print(f"   Variant: {best_variant}")

    return best_dropout

def create_variant_3j(best_dropout):
    """Create 3j: depthwise + batchnorm + best dropout"""
    print(f"\nCreating variant 3j (depthwise + batchnorm + dropout {best_dropout})...")

    # Copy from depthwise base
    subprocess.run(['cp', '3f_depthwise.py', '3j_depthwise_bn_drop.py'])

    # Read the file
    with open('3j_depthwise_bn_drop.py', 'r') as f:
        content = f.read()

    # Update docstring
    content = content.replace(
        '''"""
Option F: Depthwise Separable Convolutions
Ablation Study Model: 3a_f_depthwise
Replaces standard Conv2D with SeparableConv2D for parameter efficiency
Compatible with both macOS (Metal) and Linux (CUDA)
"""''',
        f'''"""
Option J: Depthwise Separable + BatchNorm + Dropout({best_dropout})
Ablation Study Model: 3j_depthwise_bn_drop
Combines parameter efficiency, training stability, and regularization
Compatible with both macOS (Metal) and Linux (CUDA)
"""'''
    )

    # Update model function to add BatchNorm after convs and Dropout before classifier
    old_model = '''    # Block 1: 3x3 depthwise separable conv -> ReLU
    x = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(3, 3), padding="valid")(inputs)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: 3x3 depthwise separable conv -> ReLU
    x = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # FC + ReLU -> 8
    x = tf.keras.layers.Dense(8, activation="relu")(x)

    # FC + Softmax -> 2
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs, name=f"MyBAD_depthwise_m{input_shape[1]}")'''

    new_model = f'''    # Block 1: 3x3 depthwise separable conv -> BatchNorm -> ReLU
    x = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(3, 3), padding="valid")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: 3x3 depthwise separable conv -> BatchNorm -> ReLU
    x = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=(3, 3), padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # MaxPool 2x2
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # FC + ReLU -> 8
    x = tf.keras.layers.Dense(8, activation="relu")(x)

    # MODIFICATION: Add Dropout({best_dropout})
    x = tf.keras.layers.Dropout({best_dropout})(x)

    # FC + Softmax -> 2
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs, name=f"MyBAD_depthwise_bn_drop_m{{input_shape[1]}}")'''

    content = content.replace(old_model, new_model)

    # Update output directory
    content = content.replace(
        "config.output_dir = f'results_depthwise_m{config.n_mels}_s{config.random_seed}'",
        "config.output_dir = f'results_depthwise_bn_drop_m{config.n_mels}_s{config.random_seed}'"
    )

    # Write updated file
    with open('3j_depthwise_bn_drop.py', 'w') as f:
        f.write(content)

    print("✓ Created 3j_depthwise_bn_drop.py")
    return True

def run_variant_3j():
    """Run variant 3j experiments"""
    print("\nRunning variant 3j experiments (5 n_mels × seed 42)...")

    for n_mels in [16, 32, 48, 64, 80]:
        print(f"\n  Running 3j with n_mels={n_mels}...")
        result = subprocess.run([
            'python', '3j_depthwise_bn_drop.py',
            '--dataset-path', '/Volumes/Evo/mybad',
            '--n_mels', str(n_mels),
            '--random_seed', '42'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Completed n_mels={n_mels}")
        else:
            print(f"  ✗ Failed n_mels={n_mels}")
            print(f"     Error: {result.stderr[:200]}")

    print("\n✓ All 3j experiments completed!")

def main():
    print("="*80)
    print("AUTONOMOUS 3J VARIANT CREATION")
    print("="*80)
    print("\nWaiting for experiments to complete...")

    # Wait for all experiments
    wait_for_experiments()

    # Analyze results
    best_dropout = analyze_depthwise_dropout_variants()

    # Create variant 3j
    create_variant_3j(best_dropout)

    # Run variant 3j
    run_variant_3j()

    print("\n" + "="*80)
    print("3J VARIANT COMPLETE - Ready for final analysis!")
    print("="*80)

if __name__ == '__main__':
    main()
