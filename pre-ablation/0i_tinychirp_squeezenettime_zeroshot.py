#!/usr/bin/env python3
"""0i — TinyChirp SqueezeNet-Time ZERO-SHOT on SEABAD (Phase 0 / pre-ablation).

Corn-Bunting-trained SqueezeNet-Time (0d), raw-waveform model, no retraining, on the
fixed SEABAD test split. Seed-configurable via --seed (42/100/786).
Note: training output dir prefix is 0d_tinychirp_squeezenet (no "time").

Run:  conda run -n tf215_gpu python pre-ablation/0i_tinychirp_squeezenettime_zeroshot.py --seed 42
"""
from _zeroshot_common import main_cli, waveform_input

if __name__ == '__main__':
    main_cli('0d_tinychirp_squeezenet', '0i_tinychirp_squeezenettime_zeroshot', waveform_input,
             'TinyChirp-SqueezeNetTime_zeroshot_on_SEABAD',
             'Corn-Bunting INT8 model, raw waveform (peak-normalised, 48000 samples), no retraining')
