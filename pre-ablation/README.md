# pre-ablation/

Phase 0 scripts: the **starting point** for the DrongoNet ablation chain. Two related but
distinct experiments live here.

1. **Native TinyChirp baselines** (`0a`–`0e`) — Reproduce the five published TinyChirp
   architectures on the TinyChirp Corn-Bunting dataset to anchor "where we started from".
2. **Zero-shot SEABAD** (`0f`–`0j`) — Take each of those Corn-Bunting-trained models *as-is*
   (no retraining, no fine-tuning) and evaluate on the SEABAD test set. This quantifies the
   train-test domain gap that motivates DrongoNet.

Phase 1 onwards lives in **`develop/`** — see `develop/README.md`.

## Native TinyChirp baselines — `0a`–`0e`

Each script trains the published TinyChirp architecture on `TinyChirp/{training,validation,testing}/`
and writes the float32 + INT8 metrics to `results4arxiv/{script}_s{seed}/`.

| Script | Architecture |
|---|---|
| `0a_tinychirp_cnnmel.py` | CNN-Mel — primary baseline (the architecture DrongoNet derives from) |
| `0b_tinychirp_cnntime.py` | CNN-Time (raw waveform 1D conv) |
| `0c_tinychirp_transformer.py` | Transformer-Time |
| `0d_tinychirp_squeezenettime.py` | SqueezeNet-Time |
| `0e_tinychirp_squeezenetmel.py` | SqueezeNet-Mel |

## Zero-shot SEABAD — `0f`–`0j`

Thin wrappers that load the matching `0a`–`0e` Corn-Bunting INT8 model and evaluate, with
no retraining, on the fixed SEABAD test split (seed=42 split). All five share one SEABAD
test split so their AUCs are directly comparable.

| Script | Loads from | Input transform |
|---|---|---|
| `0f_tinychirp_cnnmel_zeroshot.py` | `0a` CNN-Mel | mel (n_fft=1024, n_mels=80) |
| `0g_tinychirp_cnntime_zeroshot.py` | `0b` CNN-Time | raw waveform |
| `0h_tinychirp_transformer_zeroshot.py` | `0c` Transformer | raw waveform |
| `0i_tinychirp_squeezenettime_zeroshot.py` | `0d` SqueezeNet-Time | raw waveform |
| `0j_tinychirp_squeezenetmel_zeroshot.py` | `0e` SqueezeNet-Mel | mel |

Shared harness: `_zeroshot_common.py`. Run any wrapper from the repo root:

```bash
conda run -n tf215_gpu python pre-ablation/0f_tinychirp_cnnmel_zeroshot.py --seed 42
```

## Multi-seed aggregation

```bash
conda run -n tf215_gpu python pre-ablation/aggregate_phase0_multiseed.py
```

Aggregates native TinyChirp test AUC (`0a`–`0e`) and zero-shot SEABAD AUC (`0f`–`0j`) across
seeds 42/100/786, reporting mean ± std per architecture.

A `run_phase0_multiseed.sh` runner is provided for local convenience — it iterates over the
three seeds and the ten scripts.

## Support files

| File | Purpose |
|---|---|
| `config.py` | Shared dataset/cache path constants for the Phase 0 scripts |
| `_zeroshot_common.py` | Shared harness: SEABAD test-split loader, mel/raw transforms, INT8 inference, AUC + JSON output |
