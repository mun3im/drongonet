# DrongoNet-Edge: Raspberry Pi Deployment

Self-contained package for **DrongoNet-Edge** inference on Raspberry Pi and Linux SBCs.

- **Model:** 33.06 KB INT8 (full integer quantization)
- **Parameters:** 25,890
- **Recall:** ≥99% at optimized threshold τ=0.50
- **AUC (INT8, 3 seeds):** 0.9990 ± 0.0002
- **Latency:** ~88 ms end-to-end (RPi 4B)
- **Power:** <1 W average (with duty cycling)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download Dataset from Zenodo

```bash
# Download SEABAD (~4.2 GB)
wget -O seabad.zip https://zenodo.org/records/18290494/files/mybad.zip

# Extract
unzip seabad.zip -d ~/seabad_data
```

Expected structure:
```
~/seabad_data/
├── positive/        # 3-second WAV clips with bird activity
├── negative/        # 3-second WAV clips without bird activity
└── metadata.csv
```

### Step 2: Clone Repository

```bash
git clone https://github.com/mun3im/drongonet.git
cd drongonet/edge_deploy
```

### Step 3: Install & Run

**On Raspberry Pi OS (Bullseye/Bookworm):**

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get install -y \
    python3-pip python3-venv \
    libopenblas-dev liblapack-dev libatlas-base-dev

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements-rpi.txt

# Run inference
python infer_edge_rpi.py \
    --dataset-path ~/seabad_data \
    --threshold 0.50 \
    --output results.csv
```

**Expected output (on RPi 4B, ~5-10 minutes for 5000 samples):**
```
======================================================================
DrongoNet-Edge Evaluation Results
======================================================================
Model:          drongonet_edge_int8.tflite  (post-timeshift-fix, seed 42)
Dataset:        SEABAD test set (5000 samples)
Threshold (τ):  0.50
----------------------------------------------------------------------
AUC:            0.9987   (INT8, seed 42; 0.9990 ± 0.0002 across 3 seeds)
Accuracy:       0.9822
Recall:         0.9900
Precision:      0.9748
F1 Score:       0.9823
Specificity:    0.9744
FPR:            0.0256
======================================================================
```

---

## 📦 What's Included

| File | Purpose |
|------|---------|
| `infer_edge_rpi.py` | Lightweight TFLite inference engine (no TensorFlow needed) |
| `drongonet_edge_int8.tflite` | Pre-trained INT8 model (33.06 KB) |
| `requirements-rpi.txt` | Minimal dependencies (librosa, tflite-runtime, scikit-learn) |
| `train_edge.py` | Retraining script (for advanced users) |
| `README.md` | This file |

---

## ⚙️ Configuration & Thresholds

### Default Operating Point
- **Threshold τ = 0.50** (single value, identical across seeds 42 / 100 / 786)
- **Recall = 99.0%** (minimizes missed bird events)
- **Precision = 97.5%**
- **FPR = 2.6%**

### Tuning for Your Use Case

```bash
# Default (≥99% recall)
python infer_edge_rpi.py --dataset-path ~/seabad_data --threshold 0.50

# Higher precision (fewer false positives, slightly fewer recalls)
python infer_edge_rpi.py --dataset-path ~/seabad_data --threshold 0.70

# Highest precision (rare-event monitoring; recall drops to ~97%)
python infer_edge_rpi.py --dataset-path ~/seabad_data --threshold 0.85
```

Recommended thresholds (from the seed-42 post-fix sweep on the SEABAD test set):

| Use Case | τ | Recall | Precision | F1 | FPR | Notes |
|----------|---|--------|-----------|----|-----|-------|
| Maximum sensitivity | 0.30 | 99.48% | 96.47% | 0.980 | 3.64% | Catches near-everything |
| **Default (target recall)** | **0.50** | **99.00%** | **97.48%** | **0.982** | **2.56%** | **Single τ that meets ≥0.99 recall across all 3 seeds** |
| Higher precision | 0.70 | 98.28% | 98.56% | 0.984 | 1.44% | Fewer false alarms |
| Rare-event monitoring | 0.85 | 97.12% | 99.06% | 0.981 | 0.92% | Maximum precision |

---

## 📊 Performance

### Measured Latency on Hardware

| Platform | Clock | Mel (ms) | Inference (ms) | Total (ms) |
|----------|-------|----------|----------------|-----------|
| **RPi 4B** | 1.5 GHz | 80 | 8 | **88** |
| **RPi Zero 2 W** | 1.0 GHz | 120 | 12 | **132** |
| **RPi 5** | 2.4 GHz | 50 | 3 | **53** |
| Host CPU (x86-64) | — | — | 1.15 | — |

> ℹ️ Mel preprocessing dominates on SBCs. Neural inference is <10% of total pipeline.

### Power Consumption

Measured with INA219 power monitor (typical):

| State | RPi 4B | RPi Zero 2 |
|-------|--------|-----------|
| Idle (no activity) | 3 W | 1 W |
| Active (1 segment/3s) | 6 W | 2.5 W |
| Peak (continuous) | 8 W | 3.5 W |

**Field deployment:** 1 inference per 3 seconds → **<1% duty cycle** → **<0.1 W average** with sleep.

### Model Size Comparison

| Model | Size (KB) | Params | τ | Recall | Target |
|-------|-----------|--------|---|--------|--------|
| Nano (ref) | 5.09 | 763 | — | — | MCU |
| **Micro** | 6.23 | 919 | 0.30 | 98.7% | **MCU (AudioMoth)** |
| **Edge** | 33.06 | 25,890 | 0.50 | 99.0% | **SBC (RPi)** ← You are here |

---

## 🔧 Troubleshooting

### ImportError: No module named 'tflite_runtime'

```bash
# Ensure you have the correct wheel for your RPi
pip install tflite-runtime

# If that fails, try direct download:
# For RPi 4/5 (ARMv7/ARMv8)
wget https://dl.google.com/coral/python/tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
pip install tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
```

### FileNotFoundError: Dataset not found

Verify dataset structure:
```bash
ls ~/seabad_data/positive/ | head -5
ls ~/seabad_data/negative/ | head -5
```

Should show `.wav` files in both directories.

### MemoryError on RPi Zero

The Mel computation uses significant RAM. Workarounds:

1. **Process in smaller batches** (future enhancement)
2. **Precompute Mel cache** on RPi 4, transfer to Zero
3. **Use RPi 4** for deployment (recommended; only ~$35 vs Zero at $15)

### Slow Performance on RPi Zero

Processing 5000 samples takes ~1-2 hours on Zero 2 W (vs ~10 min on RPi 4).
This is expected due to CPU speed. Options:

- Run on RPi 4 or later
- Reduce test set size for quick validation
- Deploy model and run inference in real-time (not batch evaluation)

---

## 🎓 Advanced: Retraining

To **retrain DrongoNet-Edge** from scratch:

```bash
# Install full training deps (GPU highly recommended)
pip install tensorflow==2.15 matplotlib tqdm

# Run training
python train_edge.py \
    --dataset-path ~/seabad_data \
    --cache-dir ~/seabad_cache
```

> ⚠️ **Not recommended on RPi Zero/3:** Requires >4 GB RAM. Use RPi 4+ or desktop GPU.
> Training takes ~2-4 hours on RPi 4 without GPU; use GPU for faster iterations.

### Locked Configuration (Do Not Change)
```
n_mels = 80
n_fft = 1024
hop_length = 256
target_sr = 16000
architecture = Conv(16) + BN → Conv(32) + BN → Conv(64) + BN → GAP → Dense(8)
loss = Focal Loss (α=0.25, γ=2.0)
optimizer = Adam (lr=0.001)
```

Results land in `results/seabadnet_edge_s{seed}/`.

---

## 📋 Architecture Details

**DrongoNet-Edge Architecture:**
```
Input: Mel spectrogram (184 frames × 80 mels × 1 channel), int8

  Conv2D(16, 3×3, stride=2) + BatchNorm + ReLU
  Conv2D(32, 3×3, stride=2) + BatchNorm + ReLU
  Conv2D(64, 3×3, stride=2) + BatchNorm + ReLU
  GlobalAveragePooling2D
  Dense(32) + ReLU
  Dense(8)  + ReLU
  Dense(2)  + Softmax       → [no_bird, bird], int8
```

Output is a 2-class softmax (column 0 = no-bird, column 1 = bird), quantized to int8.
Dequantize column 1 with the model's output `scale` and `zero_point` to recover the
bird-positive probability in [0, 1] before thresholding. `infer_edge_rpi.py:predict()` does
this automatically.

**Quantization:** Full INT8 (int8 input → int8 output)
- Representative dataset: 500 SEABAD validation samples
- Post-training INT8: Symmetric, per-axis (where supported)
- Trained with Focal Loss (γ=2.0, α=0.25)

---

## 📚 Model Details & Metrics

### Operating Threshold (x86-64 INT8 calibration)

A single τ=0.50 meets the ≥0.99 recall target for all three training seeds. Per-seed
metrics at τ=0.50 on the SEABAD test set:

| Seed | τ | Recall | Precision | F1 | FPR | Notes |
|------|---|--------|-----------|----|----|-------|
| 42 (provided) | 0.50 | 0.9900 | 0.9748 | 0.9823 | 0.0256 | **Default — shipped model** |
| 100 | 0.50 | 0.9912 | 0.9888 | 0.9900 | 0.0112 | — |
| 786 | 0.50 | 0.9900 | 0.9884 | 0.9892 | 0.0116 | — |

> ℹ️ Re-calibration is recommended if deploying on different hardware (ARM SBC, TPU, etc.).
> Threshold should be tuned on your specific platform.

### AUC (Test Set, ~5000 SEABAD clips)
- **Float32:** 0.9990 ± 0.0002 (mean ± std, 3 seeds)
- **INT8:** 0.9985 ± 0.0001 (quantization degradation < 0.05%)

---

## 🗂️ Output Files

When you run inference with `--output results.csv`:

```csv
filename,label,pred_prob,pred_binary,correct
001.wav,1,0.9823,1,1
002.wav,0,0.1204,0,1
...
```

Columns:
- `filename`: Input WAV file name
- `label`: Ground truth (1 = bird, 0 = no bird)
- `pred_prob`: Raw model output [0, 1]
- `pred_binary`: Thresholded prediction (1 if pred_prob ≥ τ)
- `correct`: Whether prediction matches label



---

## ✅ Checklist: Deploying to Production

- [ ] Downloaded SEABAD dataset from Zenodo
- [ ] Extracted to `~/seabad_data/` with correct structure
- [ ] Created Python venv: `python3 -m venv venv && source venv/bin/activate`
- [ ] Installed dependencies: `pip install -r requirements-rpi.txt`
- [ ] Ran test inference: `python infer_edge_rpi.py --dataset-path ~/seabad_data`
- [ ] Verified AUC ≥ 0.999 and Recall ≥ 0.99
- [ ] (Optional) Tuned threshold for your use case
- [ ] (Optional) Exported predictions: `--output results.csv`

---

**Last updated:** 2026-06-24  
**Compatible with:** Python 3.11+, Raspberry Pi OS (Bullseye/Bookworm), any ARM Linux SBC
