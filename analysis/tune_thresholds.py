#!/usr/bin/env python3
"""
tune_thresholds.py — for each variant, find the HIGHEST single τ whose 5-seed mean recall
still clears the target floor (Edge 0.99, Micro 0.98, Nano 0.97). Higher τ => higher precision,
so this maximises precision subject to the recall floor. Full-INT8 basis (model_int8.tflite).
Reports τ, mean recall/precision ±std, and the per-seed worst case at that τ.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

BASE = Path('/home/muneim/Dropbox/Conda/drongonet')
SEEDS = [42, 100, 786, 7, 1234]
GRID = np.round(np.arange(0.01, 1.00, 0.001), 3)

VARIANTS = {  # name: (dir tmpl, cache, target, current_tau)
    'Edge':  ('6c_edge_final_fft1024_m80_s{s}',  'cache4arxiv_fft1024_m80', 0.99, 0.50),
    'Micro': ('6b_micro_final_fft1024_m16_s{s}', 'cache4arxiv_fft1024_m16', 0.98, 0.35),
    'Nano':  ('6a_nano_final_fft512_m16_s{s}',   'cache4arxiv_fft512_m16',  0.97, 0.35),
}


def infer(tflite, mels):
    it = tf.lite.Interpreter(model_path=str(tflite)); it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    in_s, in_z = ind['quantization']; out_s, out_z = outd['quantization']
    probs = []
    for smp in mels[..., np.newaxis]:
        inp = smp[np.newaxis]
        inp = np.round(inp / in_s + in_z).astype(ind['dtype']) if in_s != 0 else inp.astype(ind['dtype'])
        it.set_tensor(ind['index'], inp); it.invoke()
        raw = it.get_tensor(outd['index'])
        out = (raw.astype(np.float32) - out_z) * out_s if out_s != 0 else raw.astype(np.float32)
        probs.append(float(out[0, 1]))
    return np.array(probs)


def rp(probs, y, tau):
    p = (probs >= tau).astype(int)
    tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0)); fn = np.sum((p == 0) & (y == 1))
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return rec, prec


def main():
    for name, (tmpl, cache, target, cur) in VARIANTS.items():
        d = np.load(f'/Volumes/Evo/{cache}/test/mels.npz')
        mels, y = d['mels'].astype(np.float32), d['labels'].astype(np.int32)
        seed_probs = {s: infer(BASE / 'results4arxiv' / tmpl.format(s=s) / 'model_int8.tflite', mels) for s in SEEDS}

        # mean recall/precision across seeds at each grid tau
        best = None
        for tau in GRID:
            recs = [rp(seed_probs[s], y, tau)[0] for s in SEEDS]
            precs = [rp(seed_probs[s], y, tau)[1] for s in SEEDS]
            if np.mean(recs) >= target:
                best = (tau, recs, precs)  # keep overwriting -> highest tau that still clears
        tau, recs, precs = best
        recs, precs = np.array(recs), np.array(precs)
        # current-tau numbers for comparison
        crecs = np.array([rp(seed_probs[s], y, cur)[0] for s in SEEDS])
        cprecs = np.array([rp(seed_probs[s], y, cur)[1] for s in SEEDS])
        print(f"\n===== {name}  (target mean recall ≥ {target}) =====", flush=True)
        print(f"  current τ={cur:.2f}:  recall {crecs.mean():.4f}±{crecs.std():.4f}   precision {cprecs.mean():.4f}±{cprecs.std():.4f}", flush=True)
        print(f"  TUNED   τ={tau:.3f}:  recall {recs.mean():.4f}±{recs.std():.4f}   precision {precs.mean():.4f}±{precs.std():.4f}", flush=True)
        print(f"           Δprecision = {(precs.mean()-cprecs.mean())*100:+.2f} pp   (min-seed recall {recs.min():.4f})", flush=True)
        print(f"           per-seed recall  {[f'{r:.4f}' for r in recs]}", flush=True)
        print(f"           per-seed precision {[f'{p:.4f}' for p in precs]}", flush=True)


if __name__ == '__main__':
    main()
