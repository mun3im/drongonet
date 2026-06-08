# deploy/

End-user artefacts for SEABADNet: pre-trained INT8 TFLite models, training wrappers with locked configs, and a firmware conversion tool.

## Dataset

SEABAD is hosted on Zenodo at [doi.org/10.5281/zenodo.18290494](https://doi.org/10.5281/zenodo.18290494).
The archive is approximately 4.2 GB — ensure you have at least 9 GB free (compressed + extracted).
The file is named `mybad.zip` on Zenodo (development name retained at upload time); rename it to `seabad.zip` as shown below.

```bash
# Option A: browser
# Go to https://zenodo.org/records/18290494
# Download mybad.zip (~4.2 GB), then:
mv mybad.zip seabad.zip && unzip seabad.zip -d seabad/

# Option B: wget
wget --content-disposition -O seabad.zip \
     https://zenodo.org/records/18290494/files/mybad.zip
unzip seabad.zip -d seabad/

# Option C: curl
curl -L -o seabad.zip \
     https://zenodo.org/records/18290494/files/mybad.zip
unzip seabad.zip -d seabad/
```

Expected layout after extraction:

```
seabad/
  positives/      bird vocalisations (~25,000 clips)
  negatives/      ambient audio — rain, wind, insects, urban (~25,000 clips)
  metadata.csv    clip id, label, species, source recording
  README.md
```

## Pre-trained models

All models are **full INT8** (input/output int8, `OpsSet.TFLITE_BUILTINS_INT8`), compatible with
TensorFlow Lite Micro and CMSIS-NN deployment on ARM Cortex-M.

| File | Model | Size | τ | Recall | Seed |
|---|---|---|---|---|---|
| `seabadnet_nano_int8.tflite` | SEABADNet-Nano | 5.10 KB | — | — | 42 |
| `seabadnet_micro_int8.tflite` | **SEABADNet-Micro** | 6.26 KB | 0.35 | ≥0.98 | 42 |
| `seabadnet_edge_int8.tflite` | SEABADNet-Edge | 33.06 KB | per-seed¹ | ≥0.99 | 42 |

> ¹ Edge operating threshold is calibrated per seed on x86-64 INT8 inference: τ=0.60 (seed 42),
> τ=0.55 (seed 100), τ=0.45 (seed 786). Re-calibration is recommended for ARM deployment.
> The seed-42 model provided here uses τ=0.60.

> **Tensor arena requirements (TFLite Micro, Cortex-M7):** Nano/Micro ≈23 kB (minimum viable
> 25.8 kB); Edge ≈290 kB (minimum viable 325 kB — requires external SDRAM on MCUs).

## Training wrappers

To retrain a model from scratch, only two arguments are required — all hyperparameters are locked:

```bash
# SEABADNet-Nano  (5.10 KB, no recall target)
python deploy/train_nano.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft512_m16

# SEABADNet-Micro  (6.26 KB, ≥0.98 recall @ τ=0.35)  ← primary model
python deploy/train_micro.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m16

# SEABADNet-Edge  (33.06 KB, ≥0.99 recall @ per-seed τ)
python deploy/train_edge.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m80
```

Optional: `--random_seed INT` (default: 42). Results land in `results/seabadnet_{model}_fft{n_fft}_m{n_mels}_s{seed}/`.

## convert_xxd.sh

Wraps `xxd -i` to produce an `alignas(8)` C array and matching header, ready to drop into a Portenta H7 (or any bare-metal TFLM) firmware project.

```bash
bash deploy/convert_xxd.sh <model.tflite> <out_base> <symbol_prefix>
```

**Example — SEABADNet-Micro:**

```bash
bash deploy/convert_xxd.sh \
    deploy/seabadnet_micro_int8.tflite \
    src/seabadnet_micro_model_data \
    g_seabadnet_model_data
```

Produces `src/seabadnet_micro_model_data.h` and `.cc` with `alignas(8)` so the array satisfies TFLM's alignment requirement.
