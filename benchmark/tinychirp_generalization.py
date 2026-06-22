#!/usr/bin/env python3
"""
tinychirp_generalization.py — Train SEABADNet FROM SCRATCH on the TinyChirp single-species
Corn Bunting dataset to demonstrate cross-dataset generalization (not overfit to SEABAD).

  dataset : /Volumes/Evo/TinyChirp/{training,validation,testing}/{target,non_target}
            (binary Corn Bunting present/absent, 16 kHz / 3 s clips)
  metric  : test ROC-AUC per seed, reported as mean ± std

Each variant uses its deployed architecture + mel config:
  Nano  : 6a_nano_final, build_cnn_mel_low_power_optimized (763 params),  n_fft=512,  n_mels=16
  Micro : 6b_micro_final, build_cnn_mel_low_power_optimized (919 params),  n_fft=1024, n_mels=16
  Edge  : 6c_edge_final, build_deeper_gap (25,890 params),                n_fft=1024, n_mels=80

SEABADNet mel preprocessing: n_fft, n_mels as per config above; fmin=100, fmax=8000,
center=False, 184 frames, per-sample [0,1] normalization.

Results -> results4arxiv/tinychirp_benchmark_{variant}_r{seed}/summary.json

Usage
  conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant micro --seeds 42 100 786
  conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant nano  --seeds 42 100 786
  conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant edge  --seeds 42 100 786
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

TINYCHIRP = Path('/Volumes/Evo/TinyChirp')
SR, CLIP, HOP, FRAMES = 16000, 48000, 256, 184
FMIN, FMAX = 100.0, 8000.0

SEABADNET_DIR = Path(__file__).resolve().parent.parent / 'develop'
VARIANTS = {
    # variant : (script, build_fn, n_fft, n_mels)
    'nano':  ('6a_nano_final.py',  'build_cnn_mel_low_power_optimized', 512,  16),
    'micro': ('6b_micro_final.py', 'build_cnn_mel_low_power_optimized', 1024, 16),
    'edge':  ('6c_edge_final.py',  'build_deeper_gap',                  1024, 80),
}


def load_split(split):
    """Return (wav_path, label) for a TinyChirp split. non_target=0, target=1."""
    items = []
    for label, cls in enumerate(['non_target', 'target']):
        d = TINYCHIRP / split / cls
        items += [(str(f), label) for f in d.glob('*.wav')]
    return items


def clip_to_mel(wav_path, n_fft, n_mels):
    import librosa
    y, _ = librosa.load(wav_path, sr=SR)
    if len(y) >= CLIP:
        y = y[:CLIP]
    else:
        y = np.pad(y, (0, CLIP - len(y)))
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=n_fft, hop_length=HOP, n_mels=n_mels,
                                       fmin=FMIN, fmax=FMAX, center=False)
    m = librosa.power_to_db(m, ref=np.max).T
    if m.shape[0] > FRAMES:
        m = m[:FRAMES]
    elif m.shape[0] < FRAMES:
        m = np.pad(m, ((0, FRAMES - m.shape[0]), (0, 0)), mode='constant', constant_values=m.min())
    return ((m - m.min()) / (m.max() - m.min() + 1e-8)).astype(np.float32)


def build_split_mels(split, n_fft, n_mels):
    items = load_split(split)
    X = np.empty((len(items), FRAMES, n_mels, 1), dtype=np.float32)
    y = np.empty(len(items), dtype=np.int32)
    for i, (wav, label) in enumerate(items):
        X[i, ..., 0] = clip_to_mel(wav, n_fft, n_mels)
        y[i] = label
    return X, y


def load_build_fn(script, fn_name):
    if str(SEABADNET_DIR) not in sys.path:
        sys.path.insert(0, str(SEABADNET_DIR))
    spec = importlib.util.spec_from_file_location('seabadnet_mod', SEABADNET_DIR / script)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [saved[0]]                 # dev scripts early-parse sys.argv at import
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return getattr(mod, fn_name)


def focal_loss(gamma=2.0, alpha=0.5):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)
    return loss


def run_variant(variant, seeds, epochs, batch_size):
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    script, fn_name, n_fft, n_mels = VARIANTS[variant]
    build_fn = load_build_fn(script, fn_name)

    print(f'\n=== {variant.upper()} (n_fft={n_fft}, n_mels={n_mels}) — building TinyChirp mels ===')
    t0 = time.time()
    Xtr, ytr = build_split_mels('training', n_fft, n_mels)
    Xva, yva = build_split_mels('validation', n_fft, n_mels)
    Xte, yte = build_split_mels('testing', n_fft, n_mels)
    print(f'  train={len(Xtr)} val={len(Xva)} test={len(Xte)} (built in {time.time()-t0:.0f}s)')

    aucs = []
    for seed in seeds:
        tf.random.set_seed(seed); np.random.seed(seed)
        model = build_fn(input_shape=(FRAMES, n_mels, 1), num_classes=2)
        params = int(model.count_params())
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                      loss=focal_loss(2.0, 0.5), metrics=['accuracy'])
        model.fit(Xtr, tf.keras.utils.to_categorical(ytr, 2),
                  validation_data=(Xva, tf.keras.utils.to_categorical(yva, 2)),
                  epochs=epochs, batch_size=batch_size, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                              restore_best_weights=True, verbose=0),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                  patience=5, verbose=0)])
        probs = model.predict(Xte, batch_size=256, verbose=0)[:, 1]
        auc = float(roc_auc_score(yte, probs))
        aucs.append(auc)

        out_dir = Path(f'results4arxiv/tinychirp_benchmark_{variant}_r{seed}')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'SEABADNet-{variant}_TinyChirp_Corn-Bunting',
            'protocol': 'train=TinyChirp/training, test=TinyChirp/testing (Corn Bunting)',
            'variant': variant, 'n_fft': n_fft, 'n_mels': n_mels, 'seed': seed,
            'test_auc': auc, 'params': params,
        }, indent=2))
        print(f'  [seed {seed}] TinyChirp test AUC = {auc:.4f}')

    return {'tag': f'{variant.capitalize()}_n_mels{n_mels}', 'variant': variant,
            'n_fft': n_fft, 'n_mels': n_mels, 'params': params,
            'seed_aucs': aucs, 'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs))}


def main():
    ap = argparse.ArgumentParser(description='SEABADNet generalization on TinyChirp')
    ap.add_argument('--variant', choices=list(VARIANTS), default='micro')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    print('\n' + '=' * 64)
    print(f'SEABADNet-{args.variant} | TinyChirp Corn Bunting | seeds {args.seeds}')
    print('=' * 64)

    result = run_variant(args.variant, args.seeds, args.epochs, args.batch_size)

    print('\n' + '=' * 64)
    print(f'SEABADNet-{args.variant} | TinyChirp Corn Bunting | test AUC (seeds {args.seeds})')
    print(f"  {result['variant']:6} ({result['params']:6} params): {result['auc_mean']:.4f} +/- {result['auc_std']:.4f}"
          f"   {[round(a,4) for a in result['seed_aucs']]}")
    print(f'  saved -> results4arxiv/tinychirp_benchmark_{result["variant"]}_r*/summary.json')
    print('=' * 64)


if __name__ == '__main__':
    main()
