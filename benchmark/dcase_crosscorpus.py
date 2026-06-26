#!/usr/bin/env python3
"""
dcase_crosscorpus.py -- official DCASE-2018 cross-corpus task with optional augmentation.

Trains SEABADNet on ff1010bird + warblrb10k (the DCASE-2018 dev set, ~15.7k clips), then
evaluates ZERO-SHOT on the held-out BirdVox-DCASE-20k corpus (20k clips), with NO domain
adaptation. Reports clip-level AUC for both the in-domain val split (sanity check) and
the cross-corpus BirdVox test.

This is the protocol bulbul reports as 0.887/0.855 (with/without augmentation). Our previous
no-augmentation result: Edge 0.646, Micro 0.488, Nano 0.446. This script adds an --aug flag
to drive recipes from augmentations.py and measure how much of the gap closes.

Reuses dcase_benchmark.py for data loading, model loading, focal loss, and constants -- the
only new things here are (i) corpus-disjoint train/test, and (ii) the augmentation hook.

Usage
  conda run -n tf215_gpu python benchmark/dcase_crosscorpus.py \
      --variant micro --aug full --seeds 42 100 786

Outputs
  results4arxiv/dcase_crosscorpus_{variant}_{aug}_r{seed}/summary.json
    {variant, n_mels, seed, aug, indomain_val_auc, crosscorpus_birdvox_auc, params, ...}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# reuse the existing pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcase_benchmark as D
from augmentations import make_augmenter


def build_disjoint(n_mels):
    """Build train pool (ff1010+warblr) and test pool (birdvox) separately. Returns
    (X_train_windows, train_clip_labels, X_test_windows, test_clip_labels). Windows are
    clip-ordered: clip i -> rows [i*W : (i+1)*W]."""
    print('[build] train pool = ff1010bird + warblrb10k (DCASE-2018 dev set)')
    Xtr, ytr = D.build_all(n_mels, sources=['ff1010bird', 'warblrb10k'])
    print('[build] test  pool = birdvox (held-out, zero-shot)')
    Xte, yte = D.build_all(n_mels, sources=['birdvox'])
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser(description='DCASE-2018 cross-corpus (ff1010+warblr -> BirdVox)')
    ap.add_argument('--variant', choices=list(D.VARIANTS), default='micro')
    ap.add_argument('--aug', choices=['none', 'mixup', 'specaug', 'pitch_time', 'full'], default='none',
                    help='augmentation recipe (see benchmark/augmentations.py)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--val_frac', type=float, default=0.10, help='fraction of train pool reserved for val')
    ap.add_argument('--mixup_alpha', type=float, default=0.2)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    build_fn, n_mels = D.load_build_fn(args.variant)

    # build mels once; re-split per seed
    Xtr_all, ytr_all, Xte_all, yte_all = build_disjoint(n_mels)
    n_train_clips = len(ytr_all)
    n_test_clips = len(yte_all)
    print(f'[data] train clips={n_train_clips}  test clips={n_test_clips}  '
          f'train pos frac={ytr_all.mean():.3f}  test pos frac={yte_all.mean():.3f}')
    print(f'[data] train windows={len(Xtr_all)}  test windows={len(Xte_all)}')

    results = []
    for seed in args.seeds:
        t_seed = time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        clip_ids = np.arange(n_train_clips)
        tr_ids, va_ids = train_test_split(clip_ids, test_size=args.val_frac,
                                          stratify=ytr_all, random_state=seed)
        wtr, wva = D.windows_for(tr_ids), D.windows_for(va_ids)
        Xtr, ytr_w = Xtr_all[wtr], np.repeat(ytr_all[tr_ids], D.WINDOWS_PER_CLIP)
        Xva, yva_w = Xtr_all[wva], np.repeat(ytr_all[va_ids], D.WINDOWS_PER_CLIP)

        ytr_oh = tf.keras.utils.to_categorical(ytr_w, 2).astype('float32')
        yva_oh = tf.keras.utils.to_categorical(yva_w, 2).astype('float32')

        # --- training tf.data pipeline with augmentation ---
        # augmentation is applied AFTER batch, on log-mel inputs (which are in [0,1])
        ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr_oh))
                 .shuffle(min(len(Xtr), 20000), seed=seed, reshuffle_each_iteration=True)
                 .batch(args.batch_size, drop_remainder=True))
        aug = make_augmenter(args.aug, alpha=args.mixup_alpha, seed=seed)
        ds_tr = ds_tr.map(aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        ds_va = (tf.data.Dataset.from_tensor_slices((Xva, yva_oh))
                 .batch(args.batch_size).prefetch(tf.data.AUTOTUNE))

        model = build_fn(input_shape=(D.FRAMES, n_mels, 1), num_classes=2)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
                      loss=D.focal_loss(2.0, 0.5), metrics=['accuracy'])
        model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                              restore_best_weights=True, verbose=0),
                             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                  patience=5, verbose=0)])

        # ---- in-domain val sanity (clip-level MAX agg over windows) ----
        # all val windows -> probs -> per-clip max
        va_win_probs = model.predict(Xva, batch_size=512, verbose=0)[:, 1]
        va_clip_probs = va_win_probs.reshape(len(va_ids), D.WINDOWS_PER_CLIP).max(axis=1)
        in_auc = float(roc_auc_score(ytr_all[va_ids], va_clip_probs))

        # ---- cross-corpus zero-shot on BirdVox ----
        te_win_probs = model.predict(Xte_all, batch_size=512, verbose=0)[:, 1]
        te_clip_probs = te_win_probs.reshape(n_test_clips, D.WINDOWS_PER_CLIP).max(axis=1)
        cx_auc = float(roc_auc_score(yte_all, te_clip_probs))

        results.append((seed, in_auc, cx_auc))
        elapsed = time.time() - t_seed
        print(f'  [seed {seed}] in-domain val AUC = {in_auc:.4f}   cross-corpus BirdVox AUC = {cx_auc:.4f}   ({elapsed:.0f}s)')

        out_dir = Path(f'results4arxiv/dcase_crosscorpus_{args.variant}_{args.aug}_r{seed}')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'SEABADNet-{args.variant}_DCASE2018_crosscorpus_{args.aug}',
            'protocol': 'train ff1010bird+warblrb10k -> test BirdVox-DCASE-20k (zero-shot)',
            'variant': args.variant, 'n_mels': n_mels, 'seed': seed, 'aug': args.aug,
            'windows_per_clip': D.WINDOWS_PER_CLIP,
            'n_train_clips': int(len(tr_ids)), 'n_val_clips': int(len(va_ids)),
            'n_test_clips': int(n_test_clips),
            'indomain_val_auc': in_auc, 'crosscorpus_birdvox_auc': cx_auc,
            'params': int(model.count_params()), 'epochs_run': args.epochs,
            'mixup_alpha': args.mixup_alpha if args.aug in ('mixup', 'full') else None,
        }, indent=2))

    print('\n' + '=' * 72)
    cx = np.array([r[2] for r in results])
    ind = np.array([r[1] for r in results])
    print(f'SEABADNet-{args.variant} | aug={args.aug} | DCASE-2018 cross-corpus (train dev -> test BirdVox)')
    print(f'  in-domain val   AUC = {ind.mean():.4f} +/- {ind.std():.4f}   per-seed {[round(x,4) for x in ind]}')
    print(f'  cross-corpus    AUC = {cx.mean():.4f} +/- {cx.std():.4f}   per-seed {[round(x,4) for x in cx]}')
    print('=' * 72)


if __name__ == '__main__':
    main()
