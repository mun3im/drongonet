# DrongoNet-Edge on Raspberry Pi

**DrongoNet-Edge** is a 33.06 KB INT8 neural network for bird activity detection on **Raspberry Pi and other SBCs** (Single-Board Computers).

- **Model size:** 33.06 KB (INT8 quantized)
- **Parameters:** 25,890
- **Recall:** ≥99% at optimized threshold
- **Latency:** ~1.15 ms (host CPU reference)
- **Target platform:** Raspberry Pi 4/5, Raspberry Pi Zero 2 W, or any ARM SBC with Python 3.7+

---

## Quick Start

### 1. Download SEABAD Dataset

Download from Zenodo and extract to your Raspberry Pi:

```bash
# From Zenodo: https://zenodo.org/records/18290494
# Extract: seabad_data/ will contain positive/ and negative/ subdirectories
unzip seabad_data.zip -d ~/seabad_data
```

**Expected structure:**
```
~/seabad_data/
├── positive/        # Bird activity present (3-second WAV files)
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── negative/        # Bird activity absent
    ├── 001.wav
    ├── 002.wav
    └── ...
```

### 2. Clone Repository

```bash
git clone https://github.com/mun3im/drongonet.git
cd drongonet
```

### 3. Install Dependencies

On **Raspberry Pi OS (Debian-based):**

```bash
# Update package manager
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libatlas-base-dev

# Create virtual environment (recommended)
python3 -m venv drongonet_env
source drongonet_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r deploy/requirements-rpi.txt
```

> **Note:** On Raspberry Pi Zero W, installation may take 30-60 minutes. Consider pre-building on a Pi 4 and copying the venv.

### 4. Run Inference

Evaluate the pre-trained model on SEABAD test set:

```bash
cd drongonet

python3 deploy/infer_edge_rpi.py \
    --dataset-path ~/seabad_data \
    --model deploy/drongonet_edge_int8.tflite \
    --threshold 0.50 \
    --output results.csv
```

**Expected output:**
```
======================================================================
DrongoNet-Edge Evaluation Results
======================================================================
Model:          drongonet_edge_int8.tflite
Dataset:        SEABAD test set (5,000 samples)
Threshold (τ):  0.50
----------------------------------------------------------------------
AUC:            0.9987
Accuracy:       0.9822
Recall:         0.9900
Precision:      0.9748
F1 Score:       0.9823
Specificity:    0.9744
FPR:            0.0256
----------------------------------------------------------------------
Confusion Matrix:
  TP=2475  FP=64
  FN=25   TN=2436
======================================================================
```

---

## Threshold Tuning

The default threshold (τ=0.50) is locked post-timeshift-fix retrain, uniform across all three
seeds (previously per-seed τ∈{0.45,0.55,0.60}), and is calibrated for **99% recall**.

To adjust recall/precision trade-off:

```bash
# Higher threshold → higher precision, lower recall
python3 deploy/infer_edge_rpi.py \
    --dataset-path ~/seabad_data \
    --threshold 0.70

# Lower threshold → higher recall, lower precision
python3 deploy/infer_edge_rpi.py \
    --dataset-path ~/seabad_data \
    --threshold 0.30
```

Recommended operating points:
- **99% Recall (default, τ=0.50):** Minimizes missed bird events
- **95% Recall (τ=0.70):** Reduces false positives
- **85% Recall (τ=0.80):** Maximum specificity (minimal false positives)

---

## Performance on Hardware

### Measured Latency

| Platform | Mel Preprocessing | Inference | Total |
|----------|-------------------|-----------|-------|
| **RPi 4B (ARMv7, 1.5 GHz)** | ~80 ms | ~8 ms | **~88 ms** |
| **RPi Zero 2 W (ARMv7, 1.0 GHz)** | ~120 ms | ~12 ms | **~132 ms** |
| **RPi 5 (ARMv8, 2.4 GHz)** | ~50 ms | ~3 ms | **~53 ms** |
| Host CPU (x86-64 reference) | — | 1.15 ms | — |

> Mel preprocessing dominates on SBCs; neural inference overhead is negligible (~<1% of pipeline).

### Power Consumption

| Mode | RPi 4B | RPi Zero 2 |
|------|--------|-----------|
| Idle | ~3 W | ~1 W |
| Inference + Mel (1 segment/3 sec) | ~6 W | ~2.5 W |
| Peak | ~8 W | ~3.5 W |

Typical deployment: 1 inference per 3 seconds → **<1% duty cycle** → **<0.1 W average** with sleep between batches.

---

## Troubleshooting

### ImportError: No module named 'tflite_runtime'

Ensure you're using the RPi wheels:
```bash
pip install tflite-runtime
```

If that fails, try:
```bash
# For RPi OS (Raspberry Pi 4/5, ARMv7/ARMv8)
wget https://dl.google.com/coral/python/tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
pip install tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
```

### FileNotFoundError: Dataset not found

Verify the dataset structure:
```bash
ls -la ~/seabad_data/positive/ | head
ls -la ~/seabad_data/negative/ | head
```

Should show `.wav` files.

### Memory Error on RPi Zero

The Mel computation is RAM-intensive. Workarounds:
1. Process in smaller batches (default: all test set at once)
2. Use RPi 4/5 for preprocessing, save Mel cache, transfer to Zero for inference
3. Reduce test set size for quick validation

---

## Advanced: Training on RPi

To **retrain** DrongoNet-Edge from scratch on a Raspberry Pi 4+:

```bash
# Install full training dependencies (add to requirements-rpi.txt)
pip install tensorflow==2.15 matplotlib tqdm

# Run training
python3 deploy/train_edge.py \
    --dataset-path ~/seabad_data \
    --cache-dir ~/seabad_cache
```

> ⚠️ **Not recommended for RPi Zero or Pi 3:** Training requires >4 GB RAM and takes hours on Pi 4. Use a desktop GPU for training, then copy the model to RPi.

---

## Model Details

**DrongoNet-Edge Architecture:**
- Input: Mel spectrogram (80 mels, 184 frames)
- Conv2D(16) + BatchNorm → Conv2D(32) + BatchNorm → Conv2D(64) + BatchNorm → GlobalAveragePooling → Dense(8)
- INT8 quantization (full precision → 8-bit integer)
- Focal loss (α=0.25, γ=2.0) for class imbalance

**Pre-trained weights:**
- Seed 42 (main): `deploy/drongonet_edge_int8.tflite` (33.06 KB)
- Full precision model available at: [Zenodo link]

---

## Citation

If you use DrongoNet-Edge in your research, please cite:

```bibtex
@article{zabidi2026seabadnet,
  title={SEABADNet: Efficient Bird Activity Detection on Embedded ARM Devices},
  author={Zabidi, Mun3im and others},
  journal={IEEE IoT Journal},
  year={2026}
}
```

---

## Support

For issues or questions:
- **GitHub Issues:** https://github.com/mun3im/drongonet/issues
- **Paper:** [Link to arxiv or published version]

---

**Last updated:** 2026-06-19  
**Compatible with:** Python 3.7+, RPi OS Bullseye/Bookworm
