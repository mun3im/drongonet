#!/usr/bin/env python3
"""
generate_int8_models.py
=======================
Generate proper full-INT8 TFLite models for Nano and Micro,
then print a paper-ready numbers table alongside Edge float32 AUC.

Nano/Micro -> MCU target -> full INT8 (OpsSet.TFLITE_BUILTINS_INT8)
Edge       -> SBC target -> float32 AUC from existing results4arxiv

Saves new models to results_int8/{nano,micro}_s{seed}/.
Does NOT touch results4arxiv/.
"""

import sys, os, re, time, importlib.util
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── helpers ──────────────────────────────────────────────────────────────────

BASE = Path('/home/muneim/Dropbox/Conda/seabadnet')
CACHE_BASE = '/Volumes/Evo/cache4arxiv'
RESULTS_INT8 = BASE / 'results_int8'
RESULTS_INT8.mkdir(exist_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_config(mod, n_fft, n_mels, seed):
    cfg = mod.TrainingConfig()
    cfg.n_fft = n_fft
    cfg.n_mels = n_mels
    cfg.hop_length = 256
    cfg.random_seed = seed
    cfg.cache_dir = f'{CACHE_BASE}_fft{n_fft}_m{n_mels}'
    return cfg


def build_datasets(mod, cfg, batch=32):
    tf.random.set_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    cfg.batch_size = batch
    tr_ds, _ = mod.create_tf_dataset_from_cache('train', cfg)
    va_ds, _ = mod.create_tf_dataset_from_cache('val',   cfg)
    te_ds, _ = mod.create_tf_dataset_from_cache('test',  cfg)
    return tr_ds, va_ds, te_ds


def train_model(model, tr_ds, va_ds, mod, cfg, out_dir):
    """Train with early stopping on val AUC; save best weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = mod.focal_loss(gamma=2.0, alpha=0.5)
    lr = 3e-4
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / 'best.weights.h5'),
            monitor='val_accuracy', save_best_only=True,
            save_weights_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=0),
    ]
    model.fit(tr_ds, validation_data=va_ds, epochs=50,
              callbacks=callbacks, verbose=1)
    return model


def evaluate_float32(model, te_ds):
    """Return float32 AUC on test set."""
    probs, labels = [], []
    for x, y in te_ds:
        p = model(x, training=False).numpy()
        probs.append(p[:, 1])
        y_np = y.numpy()
        # labels may be one-hot (batch, 2) or integer (batch,)
        labels.append(np.argmax(y_np, axis=1) if y_np.ndim == 2 else y_np)
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels)
    return float(roc_auc_score(labels, probs))


def convert_to_int8(model, va_ds, repr_samples=500):
    """Convert Keras model to full INT8 TFLite (input and output int8)."""
    count = [0]

    def representative_dataset():
        for x, _ in va_ds:
            for i in range(x.shape[0]):
                if count[0] >= repr_samples:
                    return
                yield [x[i:i+1].numpy()]
                count[0] += 1

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_int8(tflite_bytes, te_ds):
    """Return INT8 TFLite AUC and mean latency (ms) on test set."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp   = inp_d['quantization']
    out_scale, out_zp = out_d['quantization']

    probs, labels, times = [], [], []
    for x_batch, y_batch in te_ds:
        x_np = x_batch.numpy()
        y_np = y_batch.numpy()
        # labels may be one-hot (batch, 2) or integer (batch,)
        if y_np.ndim == 2:
            y_np = np.argmax(y_np, axis=1)
        for i in range(x_np.shape[0]):
            x_q = np.clip(np.round(x_np[i:i+1] / in_scale + in_zp),
                          -128, 127).astype(np.int8)
            interp.set_tensor(inp_d['index'], x_q)
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)
            out_q = interp.get_tensor(out_d['index'])
            p1 = (out_q[0, 1].astype(np.float32) - out_zp) * out_scale
            probs.append(p1)
            labels.append(y_np[i])

    return float(roc_auc_score(labels, probs)), float(np.mean(times))


# ── Edge float32 reader ───────────────────────────────────────────────────────

def read_edge_float32():
    results = {}
    for seed in [42, 100, 786]:
        d = BASE / f'results4arxiv/6c_edge_final_fft1024_m80_s{seed}/results_summary.txt'
        if not d.exists():
            continue
        txt = d.read_text()
        m = re.search(r'Float32 Model AUC:\s*([0-9.]+)', txt)
        if m:
            results[seed] = float(m.group(1))
    return results


# ── main ─────────────────────────────────────────────────────────────────────

VARIANTS = [
    ('Nano',  '6a_nano_final.py',  512,  16),
    ('Micro', '6b_micro_final.py', 1024, 16),
]
SEEDS = [42, 100, 786]

all_results = {}

for vname, script, n_fft, n_mels in VARIANTS:
    print(f'\n{"="*60}')
    print(f'{vname}  (n_fft={n_fft}, n_mels={n_mels})')
    print('='*60)

    mod = load_module(vname.lower(), BASE / script)
    seed_results = {}

    for seed in SEEDS:
        print(f'\n--- seed {seed} ---')
        out_dir = RESULTS_INT8 / f'{vname.lower()}_s{seed}'

        cfg = make_config(mod, n_fft, n_mels, seed)

        # ── data ──
        tr_ds, va_ds, te_ds = build_datasets(mod, cfg)

        # ── model ──
        input_shape = (184, n_mels, 1)
        model = mod.build_cnn_mel_low_power_optimized(
            input_shape=input_shape, num_classes=2)
        model.summary(print_fn=lambda s: None)

        # ── train ──
        model = train_model(model, tr_ds, va_ds, mod, cfg, out_dir)

        # ── float32 eval ──
        auc_f32 = evaluate_float32(model, te_ds)
        print(f'Float32 AUC: {auc_f32:.4f}')

        # ── INT8 conversion ──
        print('Converting to full INT8...')
        tflite_bytes = convert_to_int8(model, va_ds)
        size_kb = len(tflite_bytes) / 1024

        # check dtypes
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        in_dtype  = interp.get_input_details()[0]['dtype'].__name__
        out_dtype = interp.get_output_details()[0]['dtype'].__name__
        print(f'INT8 model: {size_kb:.2f} KB  I/O={in_dtype}/{out_dtype}')

        # save
        (out_dir / 'model_int8.tflite').write_bytes(tflite_bytes)

        # ── INT8 eval ──
        print('Evaluating INT8 model...')
        auc_int8, lat_ms = evaluate_int8(tflite_bytes, te_ds)
        print(f'INT8 AUC: {auc_int8:.4f}  latency: {lat_ms:.3f} ms')

        seed_results[seed] = dict(
            auc_f32=auc_f32, auc_int8=auc_int8,
            size_kb=size_kb, lat_ms=lat_ms,
            in_dtype=in_dtype, out_dtype=out_dtype,
        )

    all_results[vname] = seed_results

# ── summary ───────────────────────────────────────────────────────────────────

print('\n' + '='*70)
print('PAPER-READY NUMBERS  (Nano/Micro = full INT8;  Edge = float32)')
print('='*70)

for vname, sresults in all_results.items():
    f32s   = [r['auc_f32']  for r in sresults.values()]
    i8s    = [r['auc_int8'] for r in sresults.values()]
    sizes  = [r['size_kb']  for r in sresults.values()]
    lats   = [r['lat_ms']   for r in sresults.values()]
    deltas = [abs(a-b) for a, b in zip(f32s, i8s)]

    print(f'\n{vname} (full INT8, I/O int8):')
    print(f'  AUC float32 : {np.mean(f32s):.4f} ± {np.std(f32s):.4f}  '
          f'{[f"{v:.4f}" for v in f32s]}')
    print(f'  AUC INT8    : {np.mean(i8s):.4f} ± {np.std(i8s):.4f}  '
          f'{[f"{v:.4f}" for v in i8s]}')
    print(f'  Degradation : {np.mean(deltas)*100:.4f}% ± {np.std(deltas)*100:.4f}%  '
          f'max={max(deltas)*100:.4f}%')
    print(f'  Size (KB)   : {np.mean(sizes):.2f} ± {np.std(sizes):.4f}')
    print(f'  Latency(ms) : {np.mean(lats):.3f} ± {np.std(lats):.4f}')

edge = read_edge_float32()
if edge:
    vals = list(edge.values())
    print(f'\nEdge (float32 from results4arxiv, SBC deployment):')
    print(f'  AUC float32 : {np.mean(vals):.4f} ± {np.std(vals):.4f}  '
          f'{[f"{v:.4f}" for v in vals]}')
    print(f'  (INT8 file for reference: 33.06 kB — not used for SBC)')

print()
