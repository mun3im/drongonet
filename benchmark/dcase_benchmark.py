#!/usr/bin/env python3
"""
dcase_benchmark.py — Retrain SEABADNet on the DCASE-2018 Bird Audio Detection data to show
the architecture is NOT overfit to SEABAD and ADAPTS to a different clip length.

SEABAD/TinyChirp clips are 3 s; DCASE clips are 10 s. The same SEABADNet 3 s model is applied
to 10 s clips via sliding-window aggregation, retrained from scratch on a pooled in-domain
split of all three DCASE corpora, and evaluated on a held-out portion of the same pool.

  data   : ff1010bird (7,690) + warblrb10k (8,000) + BirdVox-DCASE-20k (20,000) = 35,690 clips
  split  : stratified clip-level train / val / test (in-domain; all corpora in every split)
  metric : clip-level ROC-AUC (binary bird / no-bird)

This answers "does SEABADNet generalize beyond SEABAD, including to a different clip length?"
It is NOT the official DCASE-2018 cross-corpus transfer task (train ff1010+warblr -> test
BirdVox), which measures zero-adaptation domain transfer and collapses to chance for a small
from-scratch model (see [[transformer-zeroshot-transfers]] in project memory).

Sliding windows (~50% overlap, full 10 s coverage)
  Each 10 s clip -> 16 kHz -> SIX 3 s windows evenly spaced at starts {0,1.4,2.8,4.2,5.6,7.0}s
  (hop 1.4 s, 53% overlap) -> SEABADNet log-mel (n_fft=1024, hop=256, fmin=100, fmax=8000,
  center=False, 184 frames, per-sample [0,1]).
  Train : every window inherits the clip label (multiple-instance style).
  Test  : the model scores all 6 windows; the clip score is the MAX bird-probability
          ("is there a bird anywhere in the 10 s?").

Window-mels are built once in RAM, then re-split per seed (host disk is near-full).

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
ALL_SOURCES = ['ff1010bird', 'warblrb10k', 'birdvox']   # pooled in-domain (different from official cross-corpus task)

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


def build_all(n_mels, sources):
    """Pool the given DCASE corpora -> X (n_clips*W, FRAMES, n_mels, 1), clip_labels (n_clips,).
    Windows are clip-ordered: clip i occupies rows [i*W : (i+1)*W]."""
    items = []
    for s in sources:
        items += read_source(s)
    n = len(items)
    X = np.empty((n * WINDOWS_PER_CLIP, FRAMES, n_mels, 1), dtype=np.float32)
    labels = np.empty(n, dtype=np.int32)
    print(f'[build] DCASE {"+".join(sources)}: {n} clips x {WINDOWS_PER_CLIP} windows, n_mels={n_mels}')
    t0 = time.time()
    for ci, (wav_path, label) in enumerate(items):
        if ci % 2000 == 0:
            print(f'  {ci}/{n}  ({time.time()-t0:.0f}s)')
        mels = six_window_mels(wav_path, n_mels)
        X[ci * WINDOWS_PER_CLIP:(ci + 1) * WINDOWS_PER_CLIP, ..., 0] = mels
        labels[ci] = label
    print(f'[build] pooled DCASE done in {time.time()-t0:.0f}s')
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


def windows_for(clip_idx):
    """Row indices (clip-ordered window layout) for the given clip indices."""
    clip_idx = np.asarray(clip_idx)
    base = clip_idx[:, None] * WINDOWS_PER_CLIP + np.arange(WINDOWS_PER_CLIP)[None, :]
    return base.reshape(-1)


def main():
    ap = argparse.ArgumentParser(description='In-domain DCASE benchmark for SEABADNet (sliding window)')
    ap.add_argument('--variant', choices=list(VARIANTS), default='micro')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--val_frac', type=float, default=0.1)   # fraction of the train+val pool
    ap.add_argument('--windows', type=int, default=6,
                    help='3 s windows per 10 s clip (full coverage at any N>=4; default 6)')
    ap.add_argument('--sources', nargs='+', default=ALL_SOURCES, choices=ALL_SOURCES,
                    help='corpora to pool for the in-domain split (default: all three). '
                         'Use "ff1010bird warblrb10k" to match the bulbul DCASE dev set.')
    ap.add_argument('--run-tag', default='', dest='run_tag',
                    help='suffix for output dir / summary tag, e.g. "dev" for the ff1010+warblr protocol')
    args = ap.parse_args()

    # window count is configurable; functions read these module globals at call time
    global WINDOWS_PER_CLIP, WIN_STARTS
    WINDOWS_PER_CLIP = args.windows
    WIN_STARTS = np.linspace(0, CLIP_LEN - WIN, WINDOWS_PER_CLIP).astype(int)

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    build_fn, n_mels = load_build_fn(args.variant)

    # ---- data: build pooled mels once; re-split per seed ----
    X_all, lab_all = build_all(n_mels, args.sources)
    n_clips = len(lab_all)
    clip_ids = np.arange(n_clips)
    print(f'pooled clips={n_clips}  windows={len(X_all)}  pos frac={lab_all.mean():.3f}')

    aucs = []
    for seed in args.seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # stratified clip-level split: test held out; remainder -> train/val
        trv_ids, te_ids = train_test_split(clip_ids, test_size=args.test_frac,
                                           stratify=lab_all, random_state=seed)
        tr_ids, va_ids = train_test_split(trv_ids, test_size=args.val_frac,
                                          stratify=lab_all[trv_ids], random_state=seed)

        wtr, wva = windows_for(tr_ids), windows_for(va_ids)
        Xtr, ytr = X_all[wtr], np.repeat(lab_all[tr_ids], WINDOWS_PER_CLIP)
        Xva, yva = X_all[wva], np.repeat(lab_all[va_ids], WINDOWS_PER_CLIP)

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

        # ---- sliding-window MAX aggregation on the held-out test clips ----
        wte = windows_for(te_ids)
        te_lab = lab_all[te_ids]
        win_probs = model.predict(X_all[wte], batch_size=512, verbose=0)[:, 1]
        clip_probs = win_probs.reshape(len(te_lab), WINDOWS_PER_CLIP).max(axis=1)
        auc = float(roc_auc_score(te_lab, clip_probs))
        aucs.append(auc)

        # dir name: optional _<run_tag> (protocol), then _w{N} for non-canonical window counts
        rtag = f'_{args.run_tag}' if args.run_tag else ''
        wtag = '' if args.windows == 6 else f'_w{args.windows}'
        out_dir = Path(f'results4arxiv/dcase_benchmark_{args.variant}{rtag}{wtag}_r{seed}')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'SEABADNet-{args.variant}_DCASE2018_indomain_{args.run_tag or "pool"}_w{args.windows}',
            'protocol': f'in-domain split of [{"+".join(args.sources)}], {args.windows}x3s windows (max-agg); '
                        'demonstrates cross-dataset + different-clip-length (10s) adaptation',
            'variant': args.variant, 'n_mels': n_mels, 'seed': seed,
            'windows_per_clip': WINDOWS_PER_CLIP,
            'n_train_clips': int(len(tr_ids)), 'n_val_clips': int(len(va_ids)), 'n_test_clips': int(len(te_ids)),
            'test_auc': auc, 'params': int(model.count_params()),
        }, indent=2))
        print(f'  [seed {seed}] in-domain clip-level test AUC = {auc:.4f}')

    print('\n' + '=' * 64)
    print(f'SEABADNet-{args.variant} | DCASE-2018 (in-domain pool) | 6x3s sliding-window (max-agg)')
    print(f'  test AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   per-seed {[round(a,4) for a in aucs]}')
    print(f'  (3 s model applied to 10 s clips; trained from scratch on DCASE, not SEABAD)')
    print('=' * 64)


if __name__ == '__main__':
    main()
