#!/usr/bin/env python3
"""
Pareto Frontier Analysis for XiaoChirp Architecture Sweep Results

Creates Pareto frontier plots of Accuracy vs Model Size for each architecture,
comparing different n_fft configurations on the same plot.
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RESULTS_DIR = Path("results")

def parse_results_summary(filepath):
    """Parse results_summary.txt to extract accuracy and model size."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract accuracy
    acc_match = re.search(r'Accuracy:\s*([\d.]+)', content)
    accuracy = float(acc_match.group(1)) if acc_match else None

    # Extract model size
    size_match = re.search(r'Model Size:\s*([\d.]+)\s*KB', content)
    model_size = float(size_match.group(1)) if size_match else None

    # Extract inference time
    time_match = re.search(r'Avg Inference Time:\s*([\d.]+)ms', content)
    inference_time = float(time_match.group(1)) if time_match else None

    # Extract AUC
    auc_match = re.search(r'TFLite int8 Model:[\s\S]*?AUC:\s*([\d.]+)', content)
    auc = float(auc_match.group(1)) if auc_match else None

    return {
        'accuracy': accuracy,
        'model_size_kb': model_size,
        'inference_time_ms': inference_time,
        'auc': auc
    }

def parse_config(filepath):
    """Parse config.txt to extract n_mels and n_fft."""
    with open(filepath, 'r') as f:
        content = f.read()

    n_mels_match = re.search(r'N Mels:\s*(\d+)', content)
    n_mels = int(n_mels_match.group(1)) if n_mels_match else None

    n_fft_match = re.search(r'N FFT:\s*(\d+)', content)
    n_fft = int(n_fft_match.group(1)) if n_fft_match else None

    return {'n_mels': n_mels, 'n_fft': n_fft}

def get_architecture_from_dirname(dirname):
    """Extract architecture name from directory name."""
    if dirname.startswith('1_baseline'):
        return '1_baseline'
    elif dirname.startswith('1b_'):
        return '1b_cnntime'
    elif dirname.startswith('2_depthwise'):
        return '2_depthwise'
    elif dirname.startswith('7_low_power'):
        return '7_low_power'
    elif dirname.startswith('13_tiny'):
        return '13_tiny'
    return None

def compute_pareto_frontier(points):
    """
    Compute Pareto frontier for accuracy (maximize) vs model_size (minimize).
    Returns indices of Pareto-optimal points.
    """
    if len(points) == 0:
        return []

    points = np.array(points)
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Point j dominates point i if:
                # j has higher accuracy AND smaller or equal model size
                # OR j has higher or equal accuracy AND smaller model size
                if (points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1])):
                    is_pareto[i] = False
                    break

    return np.where(is_pareto)[0]

def collect_all_results():
    """Collect all results from the results directory."""
    all_results = defaultdict(lambda: defaultdict(list))  # arch -> n_fft -> list of results

    for result_dir in RESULTS_DIR.iterdir():
        if not result_dir.is_dir():
            continue

        arch = get_architecture_from_dirname(result_dir.name)
        if arch is None or arch == '1b_cnntime':  # Skip CNN-Time (different input)
            continue

        results_file = result_dir / "results_summary.txt"
        config_file = result_dir / "config.txt"

        if not results_file.exists() or not config_file.exists():
            continue

        results = parse_results_summary(results_file)
        config = parse_config(config_file)

        if results['accuracy'] is None or results['model_size_kb'] is None:
            continue
        if config['n_mels'] is None or config['n_fft'] is None:
            continue

        entry = {
            'dirname': result_dir.name,
            'accuracy': results['accuracy'],
            'model_size_kb': results['model_size_kb'],
            'inference_time_ms': results['inference_time_ms'],
            'auc': results['auc'],
            'n_mels': config['n_mels'],
            'n_fft': config['n_fft']
        }

        all_results[arch][config['n_fft']].append(entry)

    return all_results

def plot_pareto_for_architecture(arch, data_by_fft, save_path):
    """Create Pareto frontier plot for one architecture."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {512: 'tab:blue', 1024: 'tab:orange'}
    markers = {512: 'o', 1024: 's'}

    all_points = []  # For setting axis limits

    for n_fft, results in sorted(data_by_fft.items()):
        accuracies = [r['accuracy'] for r in results]
        sizes = [r['model_size_kb'] for r in results]
        n_mels_list = [r['n_mels'] for r in results]

        # Collect points for Pareto computation: (accuracy, size)
        points = list(zip(accuracies, sizes))
        all_points.extend(points)

        # Plot all points
        ax.scatter(sizes, accuracies,
                   c=colors.get(n_fft, 'gray'),
                   marker=markers.get(n_fft, 'o'),
                   s=100, alpha=0.7,
                   label=f'n_fft={n_fft}')

        # Annotate with n_mels
        for acc, sz, n_mels in zip(accuracies, sizes, n_mels_list):
            ax.annotate(f'm{n_mels}', (sz, acc),
                        textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        # Compute and plot Pareto frontier
        pareto_idx = compute_pareto_frontier(points)
        if len(pareto_idx) > 0:
            pareto_points = np.array([points[i] for i in pareto_idx])
            # Sort by model size for line plot
            sort_idx = np.argsort(pareto_points[:, 1])
            pareto_sorted = pareto_points[sort_idx]
            ax.plot(pareto_sorted[:, 1], pareto_sorted[:, 0],
                    c=colors.get(n_fft, 'gray'),
                    linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Model Size (KB)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Pareto Frontier: {arch}\nAccuracy vs Model Size (n_mels/n_fft sweep)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Set nice axis limits
    if all_points:
        all_points = np.array(all_points)
        x_margin = (all_points[:, 1].max() - all_points[:, 1].min()) * 0.1
        y_margin = (all_points[:, 0].max() - all_points[:, 0].min()) * 0.1
        ax.set_xlim(all_points[:, 1].min() - x_margin, all_points[:, 1].max() + x_margin)
        ax.set_ylim(all_points[:, 0].min() - y_margin, all_points[:, 0].max() + y_margin)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def print_summary_table(all_results):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("SUMMARY OF ALL RESULTS")
    print("="*100)

    for arch in sorted(all_results.keys()):
        print(f"\n{'='*50}")
        print(f"Architecture: {arch}")
        print(f"{'='*50}")
        print(f"{'n_fft':>6} {'n_mels':>6} {'Accuracy':>10} {'Size(KB)':>10} {'Infer(ms)':>10} {'AUC':>8}")
        print("-"*60)

        for n_fft in sorted(all_results[arch].keys()):
            results = sorted(all_results[arch][n_fft], key=lambda x: x['n_mels'])
            for r in results:
                print(f"{r['n_fft']:>6} {r['n_mels']:>6} {r['accuracy']:>10.4f} "
                      f"{r['model_size_kb']:>10.2f} {r['inference_time_ms']:>10.2f} "
                      f"{r['auc']:>8.4f}")

def identify_best_configs(all_results):
    """Identify the best configurations for each architecture."""
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS BY ARCHITECTURE")
    print("="*100)

    for arch in sorted(all_results.keys()):
        all_entries = []
        for n_fft in all_results[arch]:
            all_entries.extend(all_results[arch][n_fft])

        if not all_entries:
            continue

        # Best by accuracy
        best_acc = max(all_entries, key=lambda x: x['accuracy'])
        # Best by size (smallest)
        best_size = min(all_entries, key=lambda x: x['model_size_kb'])
        # Best by efficiency (accuracy / size)
        best_eff = max(all_entries, key=lambda x: x['accuracy'] / x['model_size_kb'])

        print(f"\n{arch}:")
        print(f"  Best Accuracy: {best_acc['accuracy']:.4f} @ n_fft={best_acc['n_fft']}, n_mels={best_acc['n_mels']} ({best_acc['model_size_kb']:.2f} KB)")
        print(f"  Smallest Size: {best_size['model_size_kb']:.2f} KB @ n_fft={best_size['n_fft']}, n_mels={best_size['n_mels']} (acc={best_size['accuracy']:.4f})")
        print(f"  Best Efficiency: {best_eff['accuracy']:.4f}/{best_eff['model_size_kb']:.2f}KB @ n_fft={best_eff['n_fft']}, n_mels={best_eff['n_mels']}")

def main():
    print("Collecting results from all experiments...")
    all_results = collect_all_results()

    # Print summary
    print_summary_table(all_results)
    identify_best_configs(all_results)

    # Create output directory for plots
    plots_dir = Path("pareto_plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate Pareto plots for each architecture
    print("\n" + "="*100)
    print("GENERATING PARETO FRONTIER PLOTS")
    print("="*100)

    architectures_to_plot = ['1_baseline', '2_depthwise', '7_low_power', '13_tiny']

    for arch in architectures_to_plot:
        if arch in all_results:
            plot_pareto_for_architecture(
                arch,
                all_results[arch],
                plots_dir / f"pareto_{arch}.png"
            )
        else:
            print(f"Warning: No results found for {arch}")

    # Create combined comparison plot
    print("\nCreating combined comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {512: 'tab:blue', 1024: 'tab:orange'}
    markers = {512: 'o', 1024: 's'}

    for idx, arch in enumerate(architectures_to_plot):
        ax = axes[idx]
        if arch not in all_results:
            ax.text(0.5, 0.5, f'No data for {arch}', ha='center', va='center')
            ax.set_title(arch)
            continue

        data_by_fft = all_results[arch]

        for n_fft, results in sorted(data_by_fft.items()):
            accuracies = [r['accuracy'] for r in results]
            sizes = [r['model_size_kb'] for r in results]
            n_mels_list = [r['n_mels'] for r in results]

            ax.scatter(sizes, accuracies,
                       c=colors.get(n_fft, 'gray'),
                       marker=markers.get(n_fft, 'o'),
                       s=80, alpha=0.7,
                       label=f'n_fft={n_fft}')

            # Annotate with n_mels
            for acc, sz, n_mels in zip(accuracies, sizes, n_mels_list):
                ax.annotate(f'm{n_mels}', (sz, acc),
                            textcoords="offset points",
                            xytext=(3, 3), fontsize=7)

            # Pareto frontier
            points = list(zip(accuracies, sizes))
            pareto_idx = compute_pareto_frontier(points)
            if len(pareto_idx) > 0:
                pareto_points = np.array([points[i] for i in pareto_idx])
                sort_idx = np.argsort(pareto_points[:, 1])
                pareto_sorted = pareto_points[sort_idx]
                ax.plot(pareto_sorted[:, 1], pareto_sorted[:, 0],
                        c=colors.get(n_fft, 'gray'),
                        linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Model Size (KB)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{arch}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Pareto Frontiers: Accuracy vs Model Size\n(n_mels/n_fft sweep for each architecture)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plots_dir / 'pareto_combined.png'}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
