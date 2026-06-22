#!/usr/bin/env python3
"""
dcase_benchmark.py — Retrain SEABADNet under the DCASE-2018 Bird Audio Detection protocol,
with sliding-window aggregation (the fair way to apply a 3 s model to 10 s clips).

External-dataset benchmark for the "is SEABADNet overfit to SEABAD?" question. Trains the
SEABADNet architecture from scratch on the DCASE-2018 BAD split and reports clip-level
ROC-AUC on the held-out BirdVox-DCASE-20k test set (cf. DCASE-2018 leaderboard ~0.85-0.89).

  train : ff1010bird (7,690) + warblrb10k (8,000) = 15,690 clips
  test  : BirdVox-DCASE-20k (20,000)
  metric: clip-level ROC-AUC (binary bird / no-bird)

Sliding windows (~50% overlap, full 10 s coverage)
  Each 10 s clip -> 16 kHz -> SIX 3 s windows evenly spaced at starts {0,1.4,2.8,4.2,5.6,7.0}s
  (hop 1.4 s, 53% overlap) -> SEABADNet log-mel (n_fft=1024, hop=256, fmin=100, fmax=8000,
  center=False, 184 frames, per-sample [0,1]).
  Train : every window inherits the clip label (multiple-instance style).
  Test  : the model scores all 6 windows; the clip score is the MAX bird-probability
          ("is there a bird anywhere in the 10 s?").

Window-mels are built in RAM (no large /tmp cache; the host disk is near-full). All seeds for
a variant are trained in one invocation so the wav decode happens once.

Usage
  conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant nano  --seeds 42 100 786
  conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant micro --seeds 42 100 786
  conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant edge  --seeds 42 100 786
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path('/Volumes/Evo/datasets')
SOURCES = {
    'ff1010bird': (DATA_ROOT / 'freefield1010/ff1010bird_metadata_2018.csv', DATA_ROOT / 'freefield1010/wav'),
    'warblrb10k': (DATA_ROOT / 'warblr/warblrb10k_public_metadata_2018.csv', DATA_ROOT / 'warblr/wav'),
    'birdvox':    (DATA_ROOT / 'birdvox/BirdVoxDCASE20k_csvpublic.csv',       DATA_ROOT / 'birdvox/wav'),
}
TRAIN_SOURCES = ['ff1010bird', 'warblrb10k']
TEST_SOURCES = ['birdvox']

SR = 16000
WIN = SR * 3                 # 3 s analysis window = 48000 samples
CLIP_LEN = SR * 10           # pad/truncate clips to 10 s = 160000 samples
WINDOWS_PER_CLIP = 6         # ~50% overlap (hop 1.4 s, 53%), full coverage
WIN_STARTS = np.linspace(0, CLIP_LEN - WIN, WINDOWS_PER_CLIP).astype(int)
N_FFT, HOP, FRAMES = 1024, 256, 184
FMIN, FMAX = 100.0, 8000.0

SEABADNET_DIR = Path(__file__).resolve().parent.parent / 'develop'
VARIANTS = {
    'nano':  ('6a_nano_final.py',  'build_cnn_mel_low_power_optimized', 16),
    'micro': ('6b_micro_final.py', 'build_cnn_mel_low_power_optimized', 16),
    'edge':  ('6c_edge_final.py',  'build_deeper_gap',                  80),
}


def read_source(name):
    csv_path, wav_dir = SOURCES[name]
    items = []
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
        ii, hi = header.index('itemid'), header.index('hasbird')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(ii, hi):
                continue
            wav = wav_dir / f'{parts[ii]}.wav'
            if wav.exists():
                items.append((str(wav), int(parts[hi])))
    return items


def six_window_mels(wav_path, n_mels):
    """Decode -> pad/truncate to 10 s -> 6 evenly-spaced 3 s log-mel windows (FRAMES, n_mels)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=SR)
    if len(y) >= CLIP_LEN:
        y = y[:CLIP_LEN]
    else:
        y = np.pad(y, (0, CLIP_LEN - len(y)))
    out = np.empty((WINDOWS_PER_CLIP, FRAMES, n_mels), dtype=np.float32)
    for k, s in enumerate(WIN_STARTS):
        m = librosa.feature.melspectrogram(
            y=y[s:s + WIN].astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=n_mels, fmin=FMIN, fmax=FMAX, center=False)
        m = librosa.power_to_db(m, ref=np.max).T
        if m.shape[0] > FRAMES:
            m = m[:FRAMES]
        elif m.shape[0] < FRAMES:
            m = np.pad(m, ((0, FRAMES - m.shape[0]), (0, 0)), mode='constant', constant_values=m.min())
        out[k] = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return out


def build_split(split, sources, n_mels):
    """Return X (n_clips*6, FRAMES, n_mels, 1), clip_labels (n_clips,). Windows are clip-ordered."""
    items = []
    for s in sources:
        items += read_source(s)
    n = len(items)
    X = np.empty((n * WINDOWS_PER_CLIP, FRAMES, n_mels, 1), dtype=np.float32)
    labels = np.empty(n, dtype=np.int32)
    print(f'[build] {split}: {n} clips x {WINDOWS_PER_CLIP} windows, n_mels={n_mels}')
    t0 = time.time()
    for ci, (wav_path, label) in enumerate(items):
        if ci % 2000 == 0:
            print(f'  {ci}/{n}  ({time.time()-t0:.0f}s)')
        mels = six_window_mels(wav_path, n_mels)
        X[ci * WINDOWS_PER_CLIP:(ci + 1) * WINDOWS_PER_CLIP, ..., 0] = mels
        labels[ci] = label
    print(f'[build] {split} done in {time.time()-t0:.0f}s')
    return X, labels


def load_build_fn(variant):
    import sys
    script, fn_name, n_mels = VARIANTS[variant]
    if str(SEABADNET_DIR) not in sys.path:
        sys.path.insert(0, str(SEABADNET_DIR))
    spec = importlib.util.spec_from_file_location(f'seabadnet_{variant}', SEABADNET_DIR / script)
    mod = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]            # the dev scripts early-parse sys.argv at import
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv
    return getattr(mod, fn_name), n_mels


def focal_loss(gamma=2.0, alpha=0.5):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)
    return loss


def main():
    ap = argparse.ArgumentParser(description='DCASE-2018 BAD benchmark for SEABADNet (sliding window)')
    ap.add_argument('--variant', choices=list(VARIANTS), default='micro')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--val_frac', type=float, default=0.1)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    build_fn, n_mels = load_build_fn(args.variant)

    # ---- data (built once; reused across seeds) ----
    Xtr_w, tr_lab = build_split('train', TRAIN_SOURCES, n_mels)   # window-level X, clip-level labels
    Xte_w, te_lab = build_split('test', TEST_SOURCES, n_mels)
    ytr_w = np.repeat(tr_lab, WINDOWS_PER_CLIP)                    # window labels = clip label
    print(f'train windows={len(Xtr_w)} (clips {len(tr_lab)})  test windows={len(Xte_w)} (clips {len(te_lab)})  '
          f'train pos {tr_lab.mean():.3f}  test pos {te_lab.mean():.3f}')

    aucs = []
    for seed in args.seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # clip-level train/val split (keep a clip's 6 windows together)
        rng = np.random.RandomState(seed)
        clip_perm = rng.permutation(len(tr_lab))
        n_val = int(len(clip_perm) * args.val_frac)
        val_clips, tr_clips = set(clip_perm[:n_val]), set(clip_perm[n_val:])
        wmask_tr = np.array([ (i // WINDOWS_PER_CLIP) in tr_clips for i in range(len(Xtr_w)) ])
        Xtr, ytr = Xtr_w[wmask_tr], ytr_w[wmask_tr]
        Xva, yva = Xtr_w[~wmask_tr], ytr_w[~wmask_tr]

        model = build_fn(input_shape=(FRAMES, n_mels, 1), num_classes=2)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                      loss=focal_loss(2.0, 0.5), metrics=['accuracy'])
        model.fit(Xtr, tf.keras.utils.to_categorical(ytr, 2),
                  validation_data=(Xva, tf.keras.utils.to_categorical(yva, 2)),
                  epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                              restore_best_weights=True, verbose=0),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                  patience=5, verbose=0)])

        # ---- sliding-window MAX aggregation on the test clips ----
        win_probs = model.predict(Xte_w, batch_size=512, verbose=0)[:, 1]
        clip_probs = win_probs.reshape(len(te_lab), WINDOWS_PER_CLIP).max(axis=1)
        auc = float(roc_auc_score(te_lab, clip_probs))
        aucs.append(auc)

        out_dir = Path(f'results4arxiv/dcase_benchmark_{args.variant}_r{seed}')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'SEABADNet-{args.variant}_DCASE2018_slidingwin',
            'protocol': 'train=ff1010bird+warblrb10k, test=BirdVox-DCASE-20k, 6x3s windows (max-agg)',
            'variant': args.variant, 'n_mels': n_mels, 'seed': seed,
            'windows_per_clip': WINDOWS_PER_CLIP, 'test_auc': auc,
            'params': int(model.count_params()),
        }, indent=2))
        print(f'  [seed {seed}] clip-level test AUC = {auc:.4f}')

    print('\n' + '=' * 64)
    print(f'SEABADNet-{args.variant} | DCASE-2018 BAD | 6x3s sliding-window (max-agg)')
    print(f'  test AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   per-seed {[round(a,4) for a in aucs]}')
    print(f'  (cf. DCASE-2018 leaderboard ~0.85-0.89)')
    print('=' * 64)


if __name__ == '__main__':
    main()
