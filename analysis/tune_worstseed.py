#!/usr/bin/env python3
"""Highest single τ where EVERY seed's recall clears the floor (worst-seed basis).
Micro ≥0.98, Nano ≥0.97. Edge left at τ=0.50 (reported for reference). Full-INT8 basis."""
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE = Path('/home/muneim/Dropbox/Conda/drongonet')
SEEDS = [42, 100, 786, 7, 1234]
GRID = np.round(np.arange(0.01, 1.00, 0.001), 3)
VARIANTS = {  # name: (tmpl, cache, target, current, worst_seed_basis)
    'Edge':  ('6c_edge_final_fft1024_m80_s{s}',  'cache4arxiv_fft1024_m80', 0.99, 0.50, False),
    'Micro': ('6b_micro_final_fft1024_m16_s{s}', 'cache4arxiv_fft1024_m16', 0.98, 0.35, True),
    'Nano':  ('6a_nano_final_fft512_m16_s{s}',   'cache4arxiv_fft512_m16',  0.97, 0.35, True),
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
    return (tp/(tp+fn) if tp+fn else 0.0), (tp/(tp+fp) if tp+fp else 0.0)


for name, (tmpl, cache, target, cur, worst) in VARIANTS.items():
    d = np.load(f'/Volumes/Evo/{cache}/test/mels.npz')
    mels, y = d['mels'].astype(np.float32), d['labels'].astype(np.int32)
    sp = {s: infer(BASE/'results4arxiv'/tmpl.format(s=s)/'model_int8.tflite', mels) for s in SEEDS}

    def stats(tau):
        r = np.array([rp(sp[s], y, tau)[0] for s in SEEDS]); p = np.array([rp(sp[s], y, tau)[1] for s in SEEDS])
        return r, p
    cr, cp = stats(cur)
    print(f"\n===== {name} (floor {target}) =====", flush=True)
    print(f"  current τ={cur:.2f}: recall {cr.mean():.4f}±{cr.std():.4f} (min {cr.min():.4f})  precision {cp.mean():.4f}", flush=True)
    if worst:
        chosen = None
        for tau in GRID:
            r, _ = stats(tau)
            if r.min() >= target:
                chosen = tau  # highest τ with EVERY seed ≥ target
        r, p = stats(chosen)
        print(f"  TUNED  τ={chosen:.3f}: recall {r.mean():.4f}±{r.std():.4f} (min {r.min():.4f})  precision {p.mean():.4f}±{p.std():.4f}", flush=True)
        print(f"         Δprecision {(p.mean()-cp.mean())*100:+.2f} pp", flush=True)
        print(f"         per-seed recall    {dict(zip(SEEDS,[round(v,4) for v in r]))}", flush=True)
        print(f"         per-seed precision {dict(zip(SEEDS,[round(v,4) for v in p]))}", flush=True)
    else:
        print(f"         (kept at τ=0.50 per request; per-seed recall {dict(zip(SEEDS,[round(v,4) for v in cr]))})", flush=True)
