#!/usr/bin/env python3
"""0f — TinyChirp CNN-Mel ZERO-SHOT on SEABAD (Phase 0 / pre-ablation).

Corn-Bunting-trained CNN-Mel (0a), no retraining, on the fixed SEABAD test split.
Seed-configurable: --seed picks which training-seed model (42/100/786) to evaluate.

Run:  conda run -n tf215_gpu python pre-ablation/0f_tinychirp_cnnmel_zeroshot.py --seed 42
"""
from _zeroshot_common import main_cli, mel_input

if __name__ == '__main__':
    main_cli('0a_tinychirp_cnnmel', '0f_tinychirp_cnnmel_zeroshot', mel_input,
             'TinyChirp-CNNMel_zeroshot_on_SEABAD',
             'Corn-Bunting INT8 model, TinyChirp mel cfg (fmin=0,center=False), no retraining')
