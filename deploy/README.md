# deploy/

End-user artefacts for SEABADNet: pre-trained INT8 TFLite models, training wrappers with locked configs, and a firmware conversion tool.

## Pre-trained models

| File | Model | Size | τ | Recall | Seed |
|---|---|---|---|---|---|
| `seabadnet_nano_int8.tflite` | SEABADNet-Nano | 5.41 KB | — | — | 42 |
| `seabadnet_micro_int8.tflite` | **SEABADNet-Micro** | 6.56 KB | 0.35 | ≥0.98 | 42 |
| `seabadnet_edge_int8.tflite` | SEABADNet-Edge | 33.06 KB | 0.50 | ≥0.99 | 42 |

## Training wrappers

To retrain a model from scratch, only two arguments are required — all hyperparameters are locked:

```bash
# SEABADNet-Nano  (5.41 KB, no recall target)
python deploy/train_nano.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft512_m16

# SEABADNet-Micro  (6.56 KB, ≥0.98 recall @ τ=0.35)  ← primary model
python deploy/train_micro.py \
    --dataset-path /path/to/seabad \
    --cache-dir    /path/to/cache_fft1024_m16

# SEABADNet-Edge  (33 KB, ≥0.99 recall @ τ=0.50)
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
