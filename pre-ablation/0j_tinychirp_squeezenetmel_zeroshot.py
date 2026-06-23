#!/usr/bin/env python3
"""0j — TinyChirp SqueezeNet-Mel ZERO-SHOT on SEABAD (Phase 0 / pre-ablation).

Corn-Bunting-trained SqueezeNet-Mel (0e), mel-input model, no retraining, on the
fixed SEABAD test split. Seed-configurable via --seed (42/100/786).

Run:  conda run -n tf215_gpu python pre-ablation/0j_tinychirp_squeezenetmel_zeroshot.py --seed 42
"""
from _zeroshot_common import main_cli, mel_input

if __name__ == '__main__':
    main_cli('0e_tinychirp_squeezenetmel', '0j_tinychirp_squeezenetmel_zeroshot', mel_input,
             'TinyChirp-SqueezeNetMel_zeroshot_on_SEABAD',
             'Corn-Bunting INT8 model, TinyChirp mel cfg (fmin=0,center=False), no retraining')
