#!/usr/bin/env python3
"""Write per-variant locked-threshold files at the CHOSEN operating τ (5-seed, full INT8):
Nano τ=0.37 (min-seed ≥0.97), Micro τ=0.35, Edge τ=0.50. Emits:
  results4arxiv/{nano,micro,edge}_threshold_locked.txt   — locked summary at chosen τ
and re-marks the chosen τ inside results4arxiv/{nano,micro,edge}_threshold_sweep_5seed.txt.
"""
from pathlib import Path
import numpy as np, tensorflow as tf

BASE = Path('/home/muneim/Dropbox/Conda/drongonet'); SEEDS = [42, 100, 786, 7, 1234]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
CFG = {  # name: (tmpl, cache, pretty, floor, chosen_tau, basis_note)
    'nano':  ('6a_nano_final_fft512_m16_s{s}',  'cache4arxiv_fft512_m16',  'SEABADNet-Nano',  0.97, 0.37, 'operating point (min-seed recall ≥0.97); Nano has no formal recall target'),
    'micro': ('6b_micro_final_fft1024_m16_s{s}', 'cache4arxiv_fft1024_m16', 'SEABADNet-Micro', 0.98, 0.35, 'design target ≥0.98 (mean); min-seed 0.9788'),
    'edge':  ('6c_edge_final_fft1024_m80_s{s}',  'cache4arxiv_fft1024_m80', 'SEABADNet-Edge',  0.99, 0.425, 'design target ≥0.99 (mean); τ=0.425 = highest τ with 5-seed mean recall ≥0.99'),
}


def infer(t, m):
    it = tf.lite.Interpreter(model_path=str(t)); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    iss, izz = i['quantization']; oss, ozz = o['quantization']; pr = []
    for s in m[..., None]:
        x = s[None]; x = np.round(x/iss+izz).astype(i['dtype']) if iss != 0 else x.astype(i['dtype'])
        it.set_tensor(i['index'], x); it.invoke(); r = it.get_tensor(o['index'])
        pr.append(float(((r.astype(np.float32)-ozz)*oss if oss != 0 else r)[0, 1]))
    return np.array(pr)


def metrics(p, y, t):
    q = (p >= t).astype(int)
    tp = np.sum((q == 1) & (y == 1)); fp = np.sum((q == 1) & (y == 0)); fn = np.sum((q == 0) & (y == 1)); tn = np.sum((q == 0) & (y == 0))
    r = tp/(tp+fn); pr = tp/(tp+fp); f = 2*pr*r/(pr+r); fpr = fp/(fp+tn)
    return r, pr, f, fpr


for name, (tmpl, cache, pretty, floor, tau, note) in CFG.items():
    d = np.load(f'/Volumes/Evo/{cache}/test/mels.npz'); m, y = d['mels'].astype(np.float32), d['labels'].astype(np.int32)
    sp = {s: infer(BASE/'results4arxiv'/tmpl.format(s=s)/'model_int8.tflite', m) for s in SEEDS}
    aucs = [__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y, sp[s]) for s in SEEDS]
    R = np.array([metrics(sp[s], y, tau)[0] for s in SEEDS]); P = np.array([metrics(sp[s], y, tau)[1] for s in SEEDS])
    F = np.array([metrics(sp[s], y, tau)[2] for s in SEEDS]); FPR = np.array([metrics(sp[s], y, tau)[3] for s in SEEDS])
    L = [f"{pretty} — Locked Threshold (5-seed, full INT8)", "=" * 52,
         f"basis         = model_int8.tflite, cache {cache}", f"seeds         = {SEEDS}",
         f"note          = {note}", "",
         f"LOCKED τ      = {tau:g}",
         f"recall        = {R.mean():.4f} ± {R.std():.4f}   (min-seed {R.min():.4f})",
         f"precision     = {P.mean():.4f} ± {P.std():.4f}",
         f"f1            = {F.mean():.4f} ± {F.std():.4f}",
         f"fpr           = {FPR.mean():.4f} ± {FPR.std():.4f}",
         f"auc           = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", "",
         "per-seed @ τ:"]
    for s, r, p, f in zip(SEEDS, R, P, F):
        L.append(f"  s{s:<5} recall={r:.4f}  precision={p:.4f}  f1={f:.4f}")
    out = BASE/'results4arxiv'/f'{name}_threshold_locked.txt'
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out.name}: τ={tau:.2f} recall {R.mean():.4f}±{R.std():.4f} (min {R.min():.4f}) prec {P.mean():.4f}")

    # re-mark chosen τ inside the combined 5-seed sweep table
    sweep = BASE/'results4arxiv'/f'{name}_threshold_sweep_5seed.txt'
    if sweep.exists():
        lines = sweep.read_text().splitlines()
        newl = []
        for ln in lines:
            ln = ln.replace("  <-- LOCKED", "").replace(" <-- LOCKED", "")
            parts = ln.split()
            if parts and parts[0].replace('.', '', 1).isdigit():
                try:
                    if abs(float(parts[0]) - tau) < 1e-9:
                        ln = ln + "   <-- OPERATING τ"
                except ValueError:
                    pass
            if ln.startswith("LOCKED τ ="):
                ln = f"OPERATING τ = {tau:g}   ({note})"
            newl.append(ln)
        sweep.write_text("\n".join(newl) + "\n")
        print(f"  re-marked operating τ in {sweep.name}")
