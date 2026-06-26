#!/usr/bin/env python3
"""
_dcase_diagnose.py — one-shot diagnostic for the low DCASE benchmark AUC.

Reuses dcase_benchmark's data pipeline + model loader, trains ONE nano model (seed 42),
and reports a breakdown that separates the candidate causes:

  in-domain  window-level AUC  (val split of ff1010+warblr)   -> is training itself OK?
  cross      window-level AUC  (BirdVox, no aggregation)       -> raw transfer
  cross      clip   MAX-agg    (current protocol)              -> what we shipped
  cross      clip   MEAN-agg                                   -> robust aggregation
  cross      clip   MEAN over top-2 windows                    -> compromise

Run: conda run -n tf215_gpu python benchmark/_dcase_diagnose.py
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcase_benchmark as D  # reuse SOURCES, build_split, load_build_fn, focal_loss, constants


def main():
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    build_fn, n_mels = D.load_build_fn('nano')
    Xtr_w, tr_lab = D.build_split('train', D.TRAIN_SOURCES, n_mels)
    Xte_w, te_lab = D.build_split('test', D.TEST_SOURCES, n_mels)
    ytr_w = np.repeat(tr_lab, D.WINDOWS_PER_CLIP)

    seed = 42
    tf.random.set_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    clip_perm = rng.permutation(len(tr_lab))
    n_val = int(len(clip_perm) * 0.1)
    val_clips = set(clip_perm[:n_val]); tr_clips = set(clip_perm[n_val:])
    wmask_tr = np.array([(i // D.WINDOWS_PER_CLIP) in tr_clips for i in range(len(Xtr_w))])
    Xtr, ytr = Xtr_w[wmask_tr], ytr_w[wmask_tr]
    Xva, yva = Xtr_w[~wmask_tr], ytr_w[~wmask_tr]
    va_clip_idx = np.array(sorted(val_clips))
    yva_clip = tr_lab[va_clip_idx]

    model = build_fn(input_shape=(D.FRAMES, n_mels, 1), num_classes=2)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
                  loss=D.focal_loss(2.0, 0.5), metrics=['accuracy'])
    model.fit(Xtr, tf.keras.utils.to_categorical(ytr, 2),
              validation_data=(Xva, tf.keras.utils.to_categorical(yva, 2)),
              epochs=40, batch_size=128, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)])

    # in-domain window-level AUC (val windows)
    va_win_p = model.predict(Xva, batch_size=512, verbose=0)[:, 1]
    indom_win_auc = roc_auc_score(yva, va_win_p)
    # in-domain clip-level (mean agg over the val clip's windows)
    va_clip_p_mean = va_win_p.reshape(len(va_clip_idx), D.WINDOWS_PER_CLIP).mean(axis=1)
    va_clip_p_max = va_win_p.reshape(len(va_clip_idx), D.WINDOWS_PER_CLIP).max(axis=1)
    indom_clip_mean = roc_auc_score(yva_clip, va_clip_p_mean)
    indom_clip_max = roc_auc_score(yva_clip, va_clip_p_max)

    # cross-domain
    te_win_p = model.predict(Xte_w, batch_size=512, verbose=0)[:, 1]
    cross_win_auc = roc_auc_score(np.repeat(te_lab, D.WINDOWS_PER_CLIP), te_win_p)
    P = te_win_p.reshape(len(te_lab), D.WINDOWS_PER_CLIP)
    cross_max = roc_auc_score(te_lab, P.max(axis=1))
    cross_mean = roc_auc_score(te_lab, P.mean(axis=1))
    cross_top2 = roc_auc_score(te_lab, np.sort(P, axis=1)[:, -2:].mean(axis=1))

    print('\n' + '=' * 60)
    print('DCASE diagnostic — nano, seed 42')
    print('=' * 60)
    print(f'in-domain  window AUC      = {indom_win_auc:.4f}   (training signal quality)')
    print(f'in-domain  clip MEAN AUC   = {indom_clip_mean:.4f}')
    print(f'in-domain  clip MAX  AUC   = {indom_clip_max:.4f}')
    print('-' * 60)
    print(f'cross      window AUC      = {cross_win_auc:.4f}   (raw transfer, no agg)')
    print(f'cross      clip MAX  AUC   = {cross_max:.4f}   <- current shipped protocol')
    print(f'cross      clip MEAN AUC   = {cross_mean:.4f}')
    print(f'cross      clip top-2 AUC  = {cross_top2:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
