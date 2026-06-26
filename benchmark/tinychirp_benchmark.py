#!/usr/bin/env python3
"""
tinychirp_benchmark.py — Retrain the SEABADNet variants from scratch on the
TinyChirp dataset and report test-set ROC-AUC, parameter count, and INT8 size.

Counterpart of dcase_benchmark.py — the "is SEABADNet overfit to SEABAD?"
question, this time against TinyChirp's published train/val/test splits.

  train : TinyChirp/training
  val   : TinyChirp/validation
  test  : TinyChirp/testing
  metric: clip-level ROC-AUC (binary target / non_target)

Variants (architecture loaded from develop/6{a,b,c}_*.py):
  nano  : SeparableConv2D-ish, FrequencyEmphasis, n_fft=512,  n_mels=16
  micro : Conv2D + 1x1 pointwise, FrequencyEmphasis, n_fft=1024, n_mels=16
  edge  : 3-block Conv2D + BN, n_fft=1024, n_mels=80

Each variant decodes the ~17k TinyChirp clips once and caches the log-mel
arrays under /tmp (or --cache-root). Per-variant cache is keyed by
(n_fft, n_mels) so swapping variants only triggers a re-mel when the params change.

Usage:
  conda run -n tf215_gpu python benchmarking/tinychirp_benchmark.py --variant micro --seeds 42 100 786
  conda run -n tf215_gpu python benchmarking/tinychirp_benchmark.py --variant edge  --seeds 42
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path('/Volumes/Evo/TinyChirp')
SPLITS = ('training', 'validation', 'testing')
CLASSES = ('non_target', 'target')        # label 0 / 1

SR = 16000
TARGET_LEN = SR * 3                       # 3 s @ 16 kHz = 48000 samples
HOP, FRAMES = 256, 184
FMIN, FMAX = 0.0, SR / 2.0                # TinyChirp config: full band, fmin=0

SEABADNET_DIR = Path(__file__).resolve().parent.parent / 'develop'

# (script, build_fn, n_mels, n_fft)
VARIANTS = {
    'nano':  ('6a_nano_final.py',  'build_cnn_mel_low_power_optimized', 16, 512),
    'micro': ('6b_micro_final.py', 'build_cnn_mel_low_power_optimized', 16, 1024),
    'edge':  ('6c_edge_final.py',  'build_deeper_gap',                  80, 1024),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_split(split):
    """Return a list of (wav_path, label) for one TinyChirp split."""
    items = []
    for label, cls in enumerate(CLASSES):       # 0=non_target, 1=target
        wav_dir = DATA_ROOT / split / cls
        if not wav_dir.is_dir():
            raise FileNotFoundError(f'TinyChirp split missing: {wav_dir}')
        items += [(str(p), label) for p in sorted(wav_dir.iterdir()) if p.suffix == '.wav']
    return items


def clip_mel(wav_path, n_mels, n_fft):
    """Decode -> pad/truncate to 3 s -> log-mel (FRAMES, n_mels), per-sample [0,1]."""
    import librosa
    y, _ = librosa.load(wav_path, sr=SR)
    if len(y) >= TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    m = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=SR, n_fft=n_fft, hop_length=HOP,
        n_mels=n_mels, fmin=FMIN, fmax=FMAX, center=False)
    m = librosa.power_to_db(m, ref=np.max).T
    if m.shape[0] > FRAMES:
        m = m[:FRAMES]
    elif m.shape[0] < FRAMES:
        m = np.pad(m, ((0, FRAMES - m.shape[0]), (0, 0)),
                   mode='constant', constant_values=m.min())
    return ((m - m.min()) / (m.max() - m.min() + 1e-8)).astype(np.float32)


def build_or_load_split(split, n_mels, n_fft, cache_root):
    """Mel cache per (n_fft, n_mels); returns X (N, FRAMES, n_mels, 1), y (N,)."""
    cache_dir = Path(cache_root) / f'tinychirp_fft{n_fft}_m{n_mels}' / split
    cache_npz = cache_dir / 'mels.npz'
    if cache_npz.exists():
        d = np.load(cache_npz)
        X = d['mels'][..., np.newaxis]
        return X, d['labels']

    cache_dir.mkdir(parents=True, exist_ok=True)
    items = list_split(split)
    n = len(items)
    print(f'[build] {split}: {n} clips, n_fft={n_fft}, n_mels={n_mels}')
    X = np.empty((n, FRAMES, n_mels), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)
    t0 = time.time()
    for i, (wav, label) in enumerate(items):
        if i and i % 1000 == 0:
            print(f'  {i}/{n}  ({time.time() - t0:.0f}s)')
        X[i] = clip_mel(wav, n_mels, n_fft)
        y[i] = label
    np.savez_compressed(cache_npz, mels=X, labels=y)
    print(f'[build] {split} done in {time.time() - t0:.0f}s -> {cache_npz}')
    return X[..., np.newaxis], y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_build_fn(variant):
    """Import develop/6{a,b,c}_*.py and return its build function.

    The dev scripts early-parse sys.argv at import time, so we stub argv first."""
    script, fn_name, _, _ = VARIANTS[variant]
    if str(SEABADNET_DIR) not in sys.path:
        sys.path.insert(0, str(SEABADNET_DIR))
    spec = importlib.util.spec_from_file_location(f'seabadnet_{variant}',
                                                  SEABADNET_DIR / script)
    mod = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv
    return getattr(mod, fn_name)


def focal_loss(gamma=2.0, alpha=0.5):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)
    return loss


def quantize_int8(model, Xrepr, n_samples=500):
    """Full INT8 quant with a representative dataset; returns tflite bytes and size_kb."""
    import tensorflow as tf

    def rep_data():
        for i in range(min(n_samples, len(Xrepr))):
            yield [Xrepr[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = rep_data
    tflite = converter.convert()
    return tflite, len(tflite) / 1024.0


def tflite_eval(tflite_bytes, X, y):
    """Run an INT8 TFLite model and return (acc, auc, mean_latency_ms)."""
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    probs = np.empty(len(X), dtype=np.float32)
    preds = np.empty(len(X), dtype=np.int32)
    lat_ms = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        x = X[i:i + 1]
        q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        interp.set_tensor(inp_d['index'], q)
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])
        lat_ms[i] = (time.perf_counter() - t0) * 1000.0
        prob_pos = (raw[0, 1].astype(np.float32) - out_zp) * out_scale
        probs[i] = prob_pos
        preds[i] = int(np.argmax(raw[0]))
    acc = float(np.mean(preds == y))
    auc = float(roc_auc_score(y, probs))
    return acc, auc, float(np.mean(lat_ms))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='TinyChirp benchmark for SEABADNet variants (nano/micro/edge)')
    ap.add_argument('--variant', choices=list(VARIANTS), default='micro')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--cache-root', default='/tmp/tinychirp_bench_cache',
                    help='Where to cache per-(n_fft,n_mels) mels (default /tmp/...)')
    ap.add_argument('--results-root', default='results4arxiv',
                    help='Where to write per-seed summary.json (default results4arxiv/)')
    ap.add_argument('--repr-samples', type=int, default=500)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    script, fn_name, n_mels, n_fft = VARIANTS[args.variant]
    build_fn = load_build_fn(args.variant)
    input_shape = (FRAMES, n_mels, 1)
    print(f'[variant] {args.variant}  arch={fn_name}  n_fft={n_fft}  n_mels={n_mels}')

    # ---- data (built once per (n_fft, n_mels); reused across seeds) ----
    Xtr, ytr = build_or_load_split('training',   n_mels, n_fft, args.cache_root)
    Xva, yva = build_or_load_split('validation', n_mels, n_fft, args.cache_root)
    Xte, yte = build_or_load_split('testing',    n_mels, n_fft, args.cache_root)
    print(f'train={len(ytr)}  val={len(yva)}  test={len(yte)}  '
          f'train pos {ytr.mean():.3f}  test pos {yte.mean():.3f}')

    ytr_oh = tf.keras.utils.to_categorical(ytr, 2)
    yva_oh = tf.keras.utils.to_categorical(yva, 2)

    aucs_f32, aucs_int8, accs_int8, sizes_kb, lats_ms, params = [], [], [], [], [], None

    for seed in args.seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_fn(input_shape=input_shape, num_classes=2)
        if params is None:
            params = int(model.count_params())
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
            loss=focal_loss(2.0, 0.5),
            metrics=['accuracy'])
        model.fit(
            Xtr, ytr_oh,
            validation_data=(Xva, yva_oh),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=10,
                    restore_best_weights=True, verbose=0),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, verbose=0),
            ])

        # float32 test AUC
        probs_f32 = model.predict(Xte, batch_size=512, verbose=0)[:, 1]
        auc_f32 = float(roc_auc_score(yte, probs_f32))

        # INT8 quantise (representative = first 500 val samples) + eval
        tflite_bytes, size_kb = quantize_int8(model, Xva, n_samples=args.repr_samples)
        acc_int8, auc_int8, lat_ms = tflite_eval(tflite_bytes, Xte, yte)

        aucs_f32.append(auc_f32)
        aucs_int8.append(auc_int8)
        accs_int8.append(acc_int8)
        sizes_kb.append(size_kb)
        lats_ms.append(lat_ms)

        out_dir = Path(args.results_root) / f'tinychirp_benchmark_{args.variant}_r{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'model_int8.tflite').write_bytes(tflite_bytes)
        (out_dir / 'summary.json').write_text(json.dumps({
            'tag': f'SEABADNet-{args.variant}_TinyChirp',
            'protocol': 'train=TinyChirp/training, val=TinyChirp/validation, test=TinyChirp/testing',
            'variant': args.variant,
            'arch_fn': fn_name,
            'n_fft': n_fft,
            'n_mels': n_mels,
            'seed': seed,
            'params': params,
            'size_int8_kb': size_kb,
            'latency_int8_ms': lat_ms,
            'test_auc_f32': auc_f32,
            'test_auc_int8': auc_int8,
            'test_acc_int8': acc_int8,
        }, indent=2))
        print(f'  [seed {seed}] f32 AUC={auc_f32:.4f}  int8 AUC={auc_int8:.4f}  '
              f'acc={acc_int8:.4f}  size={size_kb:.2f}KB  lat={lat_ms:.3f}ms')

    print('\n' + '=' * 72)
    print(f'SEABADNet-{args.variant} | TinyChirp | {len(args.seeds)} seed(s)')
    print(f'  params           = {params}')
    print(f'  test AUC f32     = {np.mean(aucs_f32):.4f} +/- {np.std(aucs_f32):.4f}   '
          f'per-seed {[round(a, 4) for a in aucs_f32]}')
    print(f'  test AUC int8    = {np.mean(aucs_int8):.4f} +/- {np.std(aucs_int8):.4f}  '
          f'per-seed {[round(a, 4) for a in aucs_int8]}')
    print(f'  test acc  int8   = {np.mean(accs_int8):.4f} +/- {np.std(accs_int8):.4f}')
    print(f'  size_int8 (KB)   = {np.mean(sizes_kb):.2f}')
    print(f'  latency_int8 ms  = {np.mean(lats_ms):.3f}')
    print(f'  (TinyChirp CNN-Mel baseline from paper: ~96.0% accuracy)')
    print('=' * 72)


if __name__ == '__main__':
    main()
