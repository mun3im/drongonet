# benchmark/

External-dataset benchmarks for SEABADNet cross-dataset generalization (sibling of `pre-ablation/`, `develop/`, `deploy/`).

## Benchmark Overview

Two independent protocols demonstrate that SEABADNet is not overfit to SEABAD:

1. **DCASE-2018 BAD** — Temperate bird species on BirdVox-DCASE-20k (standard leaderboard)
2. **TinyChirp Corn Bunting** — Single tropical species, directly comparable to TinyChirp

Both train fresh models from scratch. Variants:
- **Nano** — 763 params, ultra-compact (ARM Cortex-M4 target)
- **Micro** — 919 params, optimized (AudioMoth target, SEABAD primary)
- **Edge** — 25,890 params, higher-capacity (SBC target)

---

## DCASE-2018 Bird Audio Detection — `dcase_benchmark.py`

Retrains SEABADNet **from scratch** on the standard DCASE-2018 BAD protocol. Reports clip-level 
ROC-AUC on the held-out **BirdVox-DCASE-20k** test set. This directly answers: "Is SEABADNet overfit 
to SEABAD?" — a fresh model on an independent temperate dataset, directly comparable to the 
DCASE-2018 leaderboard (~0.85–0.89 AUC).

### Dataset

| Split | Sources | Clips | Notes |
|-------|---------|-------|-------|
| train | ff1010bird (7,690) + warblrb10k (8,000) | 15,690 | Mixed species, temperate |
| test  | BirdVox-DCASE-20k | 20,000 | Primary leaderboard benchmark |

**Leakage note:** SEABAD reuses BirdVox's 9,983 no-bird clips in training, but this benchmark 
trains on ff1010bird+warblrb10k only. BirdVox is completely unseen — fair cross-dataset test.

### Protocol

Each 10 s clip is split into **6 evenly-spaced 3 s windows** (hop 1.4 s, 53% overlap) to fairly 
apply the 3 s model to 10 s clips:

- Clip starts: {0, 1.4, 2.8, 4.2, 5.6, 7.0} s
- Train: each window inherits the clip's binary label (multiple-instance learning style)
- Test: model scores all 6 windows; final score = MAX bird-probability over windows ("is there a bird anywhere?")

**Mel spectrogram:** n_fft=1024, hop=256, fmin=100, fmax=8000, center=False → 184 frames, 
per-sample min-max [0,1] normalization. Built in RAM (disk near-full).

### Usage

```bash
# Single variant
conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant nano --seeds 42 100 786
conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant micro --seeds 42 100 786
conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant edge --seeds 42 100 786

# All variants
bash benchmark/run_dcase_benchmark.sh
```

**Output:** `results4arxiv/dcase_benchmark_{variant}_r{seed}/summary.json`  
Each contains: test_auc, variant, params, n_mels, seed, protocol description.

---

## TinyChirp Corn Bunting Generalization — `tinychirp_generalization.py`

Retrains SEABADNet **from scratch** on the TinyChirp single-species Corn Bunting dataset. 
This is a *different ecological domain* from SEABAD (tropical mixed-flock vs. single-species 
temperate); demonstrates architectural robustness beyond the SEABAD training set.

### Dataset

| Split | Source | Clips | Species | Notes |
|-------|--------|-------|---------|-------|
| train | TinyChirp/training | ~5,500 | Corn Bunting | Binary: Corn Bunting present/absent |
| val   | TinyChirp/validation | ~1,300 | Corn Bunting | Held-out for early stopping |
| test  | TinyChirp/testing | ~1,300 | Corn Bunting | Evaluation set |

**Data:** `/Volumes/Evo/TinyChirp/{training,validation,testing}/{target,non_target}/` (16 kHz, 3 s clips)

### Protocol

Direct clip-to-prediction (no sliding-window aggregation, unlike DCASE). Each clip is mel-processed 
once and fed to the model.

**Mel spectrogram:** 16 kHz, 3 s clips, padded/truncated to 48,000 samples. Per-variant config:
- Nano: n_fft=512, n_mels=16
- Micro: n_fft=1024, n_mels=16  
- Edge: n_fft=1024, n_mels=80

Common: hop=256, fmin=100, fmax=8000, center=False → 184 frames, per-sample [0,1] normalization.

### Usage

```bash
# Single variant
conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant nano --seeds 42 100 786
conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant micro --seeds 42 100 786
conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant edge --seeds 42 100 786

# All variants in sequence
for v in nano micro edge; do
  conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant $v --seeds 42 100 786
done
```

**Output:** `results4arxiv/tinychirp_benchmark_{variant}_r{seed}/summary.json`  
Each contains: test_auc, variant, n_fft, n_mels, params, seed, protocol description.

---

## Training Configuration (Both Protocols)

Identical across DCASE and TinyChirp:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW(lr=3e-4, weight_decay=1e-4) | Standard SEABADNet config |
| Loss | Focal Loss (γ=2.0, α=0.5) | Handles class imbalance |
| Epochs | 50 | Training ceiling |
| Early stopping | val_accuracy, patience=10 | Restore best weights |
| LR plateau | val_loss, factor=0.5, patience=5 | Reduce LR on plateau |
| Batch size | DCASE: 128, TinyChirp: 64 | Memory constraints (DCASE uses RAM mels) |

---

## Expected Results

| Protocol | Variant | Expected AUC | Notes |
|----------|---------|--------------|-------|
| DCASE-2018 | Nano | ~0.83 | Smaller architecture, larger test |
| DCASE-2018 | Micro | ~0.85 | Primary production model |
| DCASE-2018 | Edge | ~0.87 | Larger capacity |
| TinyChirp | Nano | ~0.95 | Simpler dataset, single species |
| TinyChirp | Micro | ~0.97 | Strong generalization |
| TinyChirp | Edge | ~0.98 | High-capacity ceiling |

(DCASE ~0.85–0.89 aligns with published leaderboard. TinyChirp higher due to single-species simplicity.)
