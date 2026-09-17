"""
config.py — canonical path configuration for SparrowNet experiments.
All scripts import from here. Edit this file to change locations.
"""

DATASET_PATH = "/Volumes/Evo/datasets"
RESULTS_BASE = "results"

# Cache lives OUTSIDE Dropbox and off the root partition, on /home (225 GB free).
# drongonet's PICKUP.md: an 11 GB mel cache on /tmp filled the root partition and
# caused repeated training crashes. A Dropbox-resident cache would also sync ~1.4 GB.
CACHE_BASE = "/home/muneim/.cache/sparrownet"

# Conda env used for all runs (matches drongonet, so numbers stay comparable):
#   /home/muneim/miniconda3/envs/tf215_gpu/bin/python   (TF 2.15.0, tfmot 0.8.0)
CONDA_PYTHON = "/home/muneim/miniconda3/envs/tf215_gpu/bin/python"

# Model size budget (INT8 .tflite, bytes)
SIZE_SOFT_LIMIT_BYTES = 10 * 1024   # 10 KB — target
SIZE_HARD_LIMIT_BYTES = 16 * 1024   # 16 KB — reject anything over this

# The three DCASE 2018 Bird Audio Detection corpora, sitting under DATASET_PATH.
# Each has: {name}/{csv} with columns itemid,datasetid,hasbird and {name}/wav/{itemid}.wav
DCASE_CORPORA = {
    "birdvox": {
        "csv": "birdvox/BirdVoxDCASE20k_csvpublic.csv",
        "wav_dir": "birdvox/wav",
    },
    "freefield1010": {
        "csv": "freefield1010/ff1010bird_metadata_2018.csv",
        "wav_dir": "freefield1010/wav",
    },
    "warblr": {
        "csv": "warblr/warblrb10k_public_metadata_2018.csv",
        "wav_dir": "warblr/wav",
    },
}

# Frontend — identical to drongonet-micro's, so results are directly comparable.
SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 48000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 16
FMIN = 100.0
FMAX = 8000.0
N_FRAMES = 1 + (CLIP_SAMPLES - N_FFT) // HOP_LENGTH  # 184

# Whole-clip mode: DCASE clips are 10s. Cache is a fixed-width array, so clips are
# padded/truncated to this many frames (warblr runs ~10.03s -> 626 frames; dropping
# 4 frames off the tail of a 10s clip is immaterial).
FULL_CLIP_SECONDS = 10.0
FULL_CLIP_SAMPLES = int(SAMPLE_RATE * FULL_CLIP_SECONDS)  # 160000
FULL_N_FRAMES = 1 + (FULL_CLIP_SAMPLES - N_FFT) // HOP_LENGTH  # 622
