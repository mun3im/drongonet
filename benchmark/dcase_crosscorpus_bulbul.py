"""dcase_crosscorpus_bulbul.py — bulbul architecture under our cross-corpus protocol.

Same corpora (train ff1010bird+warblrb10k -> zero-shot test BirdVox-DCASE-20k), same seeds, same
focal loss, same augmentation recipes as dcase_crosscorpus.py, same 80-mel / n_fft=1024 / 16 kHz
front-end (fmin=100, fmax=8000). The ONLY thing that changes vs SEABADNet is the architecture and
its native input: bulbul reads the full 10 s clip as one contiguous 1000x80 log-mel spectrogram
(hop=160 = 10 ms/frame -> 1000 frames), exactly its Table-I input, with one prediction per clip.
At 1000 frames the model is ~373,202 params -- bulbul's literal capacity.

The AUC measured here is what we cite for bulbul -- NOT its published 0.887/0.855, which came
from a different pipeline (22.05 kHz, mean-subtraction, 5-fold ensemble, bulbul's own aug).

Run:  python benchmark/dcase_crosscorpus_bulbul.py --aug none --seeds 42
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import dcase_benchmark as D
from augmentations import make_augmenter
from bulbul_arch import build_bulbul

N_MELS = 80
HOP_BULBUL = 160          # 10 ms/frame at 16 kHz
BULBUL_FRAMES = 1000      # full 10 s clip -> bulbul's Table-I 1000x80 input


def full_clip_mel(wav_path):
    """Decode -> pad/truncate to 10 s -> single contiguous 1000x80 log-mel (matches the log+
    per-clip [0,1] normalisation used by dcase_benchmark.six_window_mels)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=D.SR)
    y = y[:D.CLIP_LEN] if len(y) >= D.CLIP_LEN else np.pad(y, (0, D.CLIP_LEN - len(y)))
    m = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=D.SR, n_fft=D.N_FFT, hop_length=HOP_BULBUL,
        n_mels=N_MELS, fmin=D.FMIN, fmax=D.FMAX, center=False)
    m = librosa.power_to_db(m, ref=np.max).T          # (frames, n_mels)
    if m.shape[0] > BULBUL_FRAMES:
        m = m[:BULBUL_FRAMES]
    elif m.shape[0] < BULBUL_FRAMES:
        m = np.pad(m, ((0, BULBUL_FRAMES - m.shape[0]), (0, 0)), mode='constant', constant_values=m.min())
    return ((m - m.min()) / (m.max() - m.min() + 1e-8)).astype(np.float32)


CACHE = Path('/home/muneim/.cache/bulbul_mel_cache')   # on /home (308 GB free); root is near-full. ~11 GB, deletable, not in Dropbox


def build_full(sources):
    """-> X (n_clips, 1000, 80, 1), labels (n_clips,). Cached to /tmp so per-seed processes
    don't recompute the 35k-clip mel set each time."""
    key = '_'.join(sources)
    CACHE.mkdir(parents=True, exist_ok=True)
    xp, yp = CACHE / f'{key}_X.npy', CACHE / f'{key}_y.npy'
    if xp.exists() and yp.exists():
        print(f'[build] cache hit: {key}', flush=True)
        return np.load(xp), np.load(yp)
    items = []
    for s in sources:
        items += D.read_source(s)
    n = len(items)
    X = np.empty((n, BULBUL_FRAMES, N_MELS, 1), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for i, (wav, lab) in enumerate(items):
        X[i, ..., 0] = full_clip_mel(wav)
        y[i] = lab
        if (i + 1) % 2000 == 0:
            print(f'  [build] {"+".join(sources)}: {i+1}/{n}', flush=True)
    print(f'[build] {"+".join(sources)}: {n} clips x {BULBUL_FRAMES}x{N_MELS}')
    np.save(xp, X); np.save(yp, y)
    return X, y


def main():
    ap = argparse.ArgumentParser(description='bulbul under cross-corpus protocol')
    ap.add_argument('--aug', choices=['none', 'mixup', 'specaug', 'pitch_time', 'full'], default='none')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)   # 1000-frame input -> smaller batch (bulbul used 64)
    ap.add_argument('--val_frac', type=float, default=0.10)
    ap.add_argument('--mixup_alpha', type=float, default=0.2)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    Xtr_all, ytr_all = build_full(['ff1010bird', 'warblrb10k'])
    Xte_all, yte_all = build_full(['birdvox'])
    n_train_clips, n_test_clips = len(ytr_all), len(yte_all)
    print(f'[data] train clips={n_train_clips}  test clips={n_test_clips}  '
          f'frames={BULBUL_FRAMES}  train pos frac={ytr_all.mean():.3f}  test pos frac={yte_all.mean():.3f}')

    def predict_pos(model, X, batch=256):
        out = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), batch):
            out[i:i + batch] = model.predict_on_batch(X[i:i + batch])[:, 1]
        return out

    results = []
    for seed in args.seeds:
        t_seed = time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        tr_ids, va_ids = train_test_split(np.arange(n_train_clips), test_size=args.val_frac,
                                          stratify=ytr_all, random_state=seed)
        Xtr, ytr = Xtr_all[tr_ids], ytr_all[tr_ids]
        Xva, yva = Xtr_all[va_ids], ytr_all[va_ids]
        ytr_oh = tf.keras.utils.to_categorical(ytr, 2).astype('float32')
        yva_oh = tf.keras.utils.to_categorical(yva, 2).astype('float32')

        ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr_oh))
                 .shuffle(min(len(Xtr), 20000), seed=seed, reshuffle_each_iteration=True)
                 .batch(args.batch_size, drop_remainder=True))
        aug = make_augmenter(args.aug, alpha=args.mixup_alpha, seed=seed)
        ds_tr = ds_tr.map(aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        ds_va = (tf.data.Dataset.from_tensor_slices((Xva, yva_oh))
                 .batch(args.batch_size).prefetch(tf.data.AUTOTUNE))

        model = build_bulbul(input_shape=(BULBUL_FRAMES, N_MELS, 1), num_classes=2)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                      loss=D.focal_loss(2.0, 0.5), metrics=['accuracy'])
        model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                              restore_best_weights=True, verbose=0),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                  patience=5, verbose=0)])

        in_auc = float(roc_auc_score(yva, predict_pos(model, Xva)))
        cx_auc = float(roc_auc_score(yte_all, predict_pos(model, Xte_all)))
        results.append((seed, in_auc, cx_auc))
        print(f'  [seed {seed}] in-domain val AUC = {in_auc:.4f}   '
              f'cross-corpus BirdVox AUC = {cx_auc:.4f}   ({time.time()-t_seed:.0f}s)')

        out_dir = Path(f'results4arxiv/dcase_crosscorpus_bulbul_{args.aug}_r{seed}')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'bulbul_DCASE2018_crosscorpus_{args.aug}',
            'protocol': 'train ff1010bird+warblrb10k -> test BirdVox-DCASE-20k (zero-shot); '
                        'bulbul arch on full 10 s clip (1000x80, hop=160), single per-clip prediction',
            'variant': 'bulbul', 'n_mels': N_MELS, 'seed': seed, 'aug': args.aug,
            'frames': BULBUL_FRAMES, 'hop': HOP_BULBUL,
            'n_train_clips': int(len(tr_ids)), 'n_val_clips': int(len(va_ids)),
            'n_test_clips': int(n_test_clips),
            'indomain_val_auc': in_auc, 'crosscorpus_birdvox_auc': cx_auc,
            'params': int(model.count_params()), 'epochs_run': args.epochs,
            'mixup_alpha': args.mixup_alpha if args.aug in ('mixup', 'full') else None,
        }, indent=2))

    cx = np.array([r[2] for r in results]); ind = np.array([r[1] for r in results])
    print('\n' + '=' * 72)
    print(f'bulbul | aug={args.aug} | cross-corpus (train dev -> test BirdVox)')
    print(f'  in-domain val AUC = {ind.mean():.4f} +/- {ind.std():.4f}   {[round(x,4) for x in ind]}')
    print(f'  cross-corpus  AUC = {cx.mean():.4f} +/- {cx.std():.4f}   {[round(x,4) for x in cx]}')
    print('=' * 72)


if __name__ == '__main__':
    main()
