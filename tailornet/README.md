# TailorNet

A multi-species garden-bird classifier built as a variant of **DrongoNet-Edge**,
not a separate architecture: it reuses Edge's `build_deeper_gap` conv backbone
verbatim as a feature extractor, feeds it DrongoNet-Micro's exact mel
front-end (16 mels / 1024-pt FFT / 184 frames, so the same tensor a
DrongoNet-Micro gate already computes can be reused with zero second mel
pass), and closes with a MatchboxNet-style fully-convolutional epilogue
adapted from MynaNet 1o. Named for the Common Tailorbird, one of the
MyGardenBird species it classifies.

Trained and evaluated on **MyGardenBird**, a flat_dir + CSV-split multi-species
dataset (12- and 14-species variants), 80/10/10 split, same convention as
MynaNet's loader.

Full component-by-component provenance and design rationale is documented in
the [`tailornet.py`](tailornet.py) module docstring.

| Variant | Params | INT8 size | FP32 test acc. | INT8 test acc. |
|---|---|---|---|---|
| TailorNet-12sp (seed 42) | 35,340 | 47.2 KB | 93.75% | 93.89% |
| TailorNet-14sp (seed 42) | 35,598 | 47.5 KB | 92.38% | 92.38% |

Both are ~3.4x fewer parameters and ~4.1x smaller INT8 flash than the
reference MynaNet 1o (120,500 params, 193.4 KB INT8).

## Repository layout

```
tailornet.py                       model definition, dataset loader, training/eval CLI
eval_tailornet_seed7.py             reproduce a seed's INT8 test accuracy from its saved model
plot_tailornet_history.py           plot train/val accuracy + loss curves from a training_history.csv
plot_tailornet_cm.py                render a confusion-matrix figure from eval_tailornet_seed7.py's output
retrain_tailornet_seed7_history.py  retrain one seed with a combined warmup+finetune epoch history log
results_tailornet/                 one representative run per species count (seed 42):
                                    INT8 TFLite model, classification report, summary JSON
```

## Quickstart

Pre-trained INT8 TFLite models (seed 42) are in `results_tailornet/` — use
them directly for deployment, no training required:

```
results_tailornet/tailornet_12sp_epi128_drop0.2_mixup0.2_rand42/model_int8.tflite
results_tailornet/tailornet_14sp_epi128_drop0.2_mixup0.2_rand42/model_int8.tflite
```

Input tensor is `(184, 16, 1)` INT8-quantized mel-spectrogram (16 mels /
1024-pt FFT / 184 frames — DrongoNet-Micro's exact front-end, see
[`tailornet.py`](tailornet.py) for `compute_mel_spectrogram`); output is a
softmax over the classes listed in that run's `classification_report_int8.txt`,
alphabetically sorted.

To retrain from scratch instead:

```bash
python tailornet.py \
    --flat_dir   /path/to/mygardenbird16khz \
    --splits_csv /path/to/splits_mip_80_10_10.csv \
    --random_seed 42
```

Only `--flat_dir` and `--splits_csv` are required; class count is inferred
from the species subdirectories present under `flat_dir`. See
`python tailornet.py --help` for the full set of training knobs (dropout,
mixup alpha, warmup/finetune epochs and learning rates, epilogue width).
Results land in a directory containing FP32 + INT8 TFLite evaluation, a
classification report, and a parseable `tailornet_summary.json`.

To reproduce a specific seed's reported INT8 accuracy from its saved model
(i.e. a `--output_dir` produced by the `tailornet.py` run above):

```bash
python eval_tailornet_seed7.py \
    --flat_dir   /path/to/mygardenbird16khz \
    --splits_csv /path/to/splits_mip_80_10_10.csv \
    --result_dir /path/to/that/run/output_dir
```

## Requirements

- Python 3.10+, TensorFlow 2.15, `tf_keras`
- librosa, numpy, scikit-learn, matplotlib

## Relation to DrongoNet

TailorNet is downstream of this repository's own `develop/6b_micro_final.py`
(mel front-end) and `develop/6c_edge_final.py` (backbone) — see those files
for the components it reuses verbatim.
