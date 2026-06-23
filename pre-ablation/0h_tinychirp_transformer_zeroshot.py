#!/usr/bin/env python3
"""0h — TinyChirp Transformer ZERO-SHOT on SEABAD (Phase 0 / pre-ablation).

Corn-Bunting-trained Transformer-Time (0c), raw-waveform model, no retraining, on the
fixed SEABAD test split. Seed-configurable via --seed (42/100/786). Of interest because
this architecture transfers above chance (~0.75-0.80), unlike the others.

Run:  conda run -n tf215_gpu python pre-ablation/0h_tinychirp_transformer_zeroshot.py --seed 42
"""
from _zeroshot_common import main_cli, waveform_input

if __name__ == '__main__':
    main_cli('0c_tinychirp_transformer', '0h_tinychirp_transformer_zeroshot', waveform_input,
             'TinyChirp-Transformer_zeroshot_on_SEABAD',
             'Corn-Bunting INT8 model, raw waveform (peak-normalised, 48000 samples), no retraining')
