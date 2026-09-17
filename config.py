"""
config.py — canonical path configuration for SparrowNet experiments.
All scripts import from here. Edit this file to change locations.
"""

DATASET_PATH = "/Volumes/Evo/datasets"
RESULTS_BASE = "results"
CACHE_BASE = "cache"

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
