#!/usr/bin/env python3
"""
convert_newseeds_int8.py
========================
Generate full-INT8 model_int8.tflite for the 5-seed expansion seeds (7, 1234) of
Nano/Micro/Edge, converting the ALREADY-TRAINED best_model.keras in each
results4arxiv seed dir (NO retraining — preserves the exact weights add_seeds trained).

Full INT8: OpsSet.TFLITE_BUILTINS_INT8, inference_input/output_type = int8 — identical
recipe to analysis/generate_int8_models.py::convert_to_int8, so the new seeds match the
paper's INT8 deployment basis. Idempotent: skips a dir that already has model_int8.tflite.
"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

BASE = Path('/home/muneim/Dropbox/Conda/drongonet')
# variant scripts do `from config import ...`; config.py lives in develop/
sys.path.insert(0, str(BASE / 'develop'))
CACHE_BASE = '/Volumes/Evo/cache4arxiv'
SEEDS = [7, 1234]

# variant: (module script, result-dir template, n_fft, n_mels)
VARIANTS = [
    ('6a_nano_final.py',  '6a_nano_final_fft512_m16_s{s}',  512,  16),
    ('6b_micro_final.py', '6b_micro_final_fft1024_m16_s{s}', 1024, 16),
    ('6c_edge_final.py',  '6c_edge_final_fft1024_m80_s{s}',  1024, 80),
]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def custom_objects(mod):
    co = {}
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, tf.keras.layers.Layer):
            co[name] = obj
    if hasattr(mod, 'focal_loss'):
        co['focal_loss'] = mod.focal_loss
    return co


def convert_to_int8(model, va_ds, repr_samples=500):
    count = [0]

    def representative_dataset():
        for x, _ in va_ds:
            for i in range(x.shape[0]):
                if count[0] >= repr_samples:
                    return
                yield [x[i:i + 1].numpy()]
                count[0] += 1

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def eval_int8(tflite_bytes, te_ds):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    in_s, in_z = ind['quantization']
    out_s, out_z = outd['quantization']
    probs, labels = [], []
    for xb, yb in te_ds:
        xn, yn = xb.numpy(), yb.numpy()
        if yn.ndim == 2:
            yn = np.argmax(yn, axis=1)
        for i in range(xn.shape[0]):
            xq = np.clip(np.round(xn[i:i+1] / in_s + in_z), -128, 127).astype(np.int8)
            interp.set_tensor(ind['index'], xq)
            interp.invoke()
            oq = interp.get_tensor(outd['index'])
            probs.append((oq[0, 1].astype(np.float32) - out_z) * out_s)
            labels.append(yn[i])
    return float(roc_auc_score(labels, probs))


def main():
    for script, dtmpl, n_fft, n_mels in VARIANTS:
        mod = load_module(BASE / 'develop' / script)
        co = custom_objects(mod)
        for s in SEEDS:
            rd = BASE / 'results4arxiv' / dtmpl.format(s=s)
            out = rd / 'model_int8.tflite'
            keras = rd / 'best_model.keras'
            if out.exists():
                print(f"SKIP {rd.name} (model_int8.tflite exists)", flush=True)
                continue
            if not keras.exists():
                print(f"MISS {rd.name} (no best_model.keras)", flush=True)
                continue
            print(f"\n=== {rd.name} (n_fft={n_fft} n_mels={n_mels}) ===", flush=True)
            cfg = mod.TrainingConfig()
            cfg.n_fft, cfg.n_mels, cfg.hop_length = n_fft, n_mels, 256
            cfg.random_seed, cfg.batch_size = s, 32
            cfg.cache_dir = f'{CACHE_BASE}_fft{n_fft}_m{n_mels}'
            va_ds, _ = mod.create_tf_dataset_from_cache('val', cfg)
            te_ds, _ = mod.create_tf_dataset_from_cache('test', cfg)
            model = tf.keras.models.load_model(str(keras), custom_objects=co, compile=False)
            tb = convert_to_int8(model, va_ds)
            out.write_bytes(tb)
            auc = eval_int8(tb, te_ds)
            it = tf.lite.Interpreter(model_content=tb); it.allocate_tensors()
            d = it.get_input_details()[0]
            print(f"  wrote {out.name}  {len(tb)/1024:.2f}KB  in={d['dtype'].__name__} "
                  f"scale={d['quantization'][0]:.6g}  INT8 test AUC={auc:.4f}", flush=True)


if __name__ == '__main__':
    main()
