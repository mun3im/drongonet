import matplotlib.pyplot as plt
import numpy as np

# Data from the tables
n_mels = [16, 32, 48, 64, 80]
int8_auc_1024 = [0.9706, 0.9787, 0.9860, 0.9846, 0.9836]
fp32_auc_1024 = [0.9706, 0.9791, 0.9860, 0.9848, 0.9840]
int8_auc_512 = [0.9749, 0.9815, 0.9841, 0.9832, 0.9817]
fp32_auc_512 = [0.9753, 0.9824, 0.9845, 0.9832, 0.9818]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot with different colors and line styles
plt.plot(n_mels, int8_auc_1024, 'o-', color='blue', linewidth=2, markersize=8, label='INT8 n_fft=1024', linestyle='-')
plt.plot(n_mels, fp32_auc_1024, 'o--', color='red', linewidth=2, markersize=8, label='FP32 n_fft=1024', linestyle='--')
plt.plot(n_mels, int8_auc_512, 's-', color='green', linewidth=2, markersize=8, label='INT8 n_fft=512', linestyle='-')
plt.plot(n_mels, fp32_auc_512, 's--', color='orange', linewidth=2, markersize=8, label='FP32 n_fft=512', linestyle='--')

# Formatting
plt.title('INT8 AUC and FP32 AUC vs Mel Bins for Different FFT Window Sizes', fontweight='bold', fontsize=14)
plt.xlabel('n_mels')
plt.ylabel('AUC', fontweight='bold')
plt.xticks(n_mels)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig( 'auc_vs_mels.png')