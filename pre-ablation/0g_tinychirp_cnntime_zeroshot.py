#!/usr/bin/env python3
"""0g — TinyChirp CNN-Time ZERO-SHOT on SEABAD (Phase 0 / pre-ablation).

Corn-Bunting-trained CNN-Time (0b), raw-waveform 1D model, no retraining, on the
fixed SEABAD test split. Seed-configurable via --seed (42/100/786).

Run:  conda run -n tf215_gpu python pre-ablation/0g_tinychirp_cnntime_zeroshot.py --seed 42
"""
from _zeroshot_common import main_cli, waveform_input

if __name__ == '__main__':
    main_cli('0b_tinychirp_cnntime', '0g_tinychirp_cnntime_zeroshot', waveform_input,
             'TinyChirp-CNNTime_zeroshot_on_SEABAD',
             'Corn-Bunting INT8 model, raw waveform (peak-normalised, 48000 samples), no retraining')
