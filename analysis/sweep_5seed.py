#!/usr/bin/env python3
"""
sweep_5seed.py — uniform full-INT8 threshold sweep across all 5 seeds (42,100,786,7,1234)
for Nano / Micro / Edge. Reads model_int8.tflite from each seed dir + the matching mel cache,
computes a per-seed threshold table (written into each seed dir as threshold_sweep.txt, same
format as the original per-variant scripts) and a 5-seed mean±std + locked-τ summary per variant.

All three variants use the SAME basis (full INT8, model_int8.tflite) so per-variant 5-seed
mean±std is valid. Supersedes the earlier mixed basis (nano/edge INT8 but micro float32).
"""
import sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

BASE = Path('/home/muneim/Dropbox/Conda/drongonet')
SEEDS = [42, 100, 786, 7, 1234]
THRESHOLDS = np.round(np.arange(0.005, 0.9951, 0.005), 3)  # fine grid: 0.05-multiples are a subset; op τ (0.35/0.37/0.425) exact

# name: (dir template, cache, target_recall, pretty)
VARIANTS = {
    'nano':  ('6a_nano_final_fft512_m16_s{s}',  'cache4arxiv_fft512_m16',  0.98, 'DrongoNet-Nano'),
    'micro': ('6b_micro_final_fft1024_m16_s{s}', 'cache4arxiv_fft1024_m16', 0.98, 'DrongoNet-Micro'),
    'edge':  ('6c_edge_final_fft1024_m80_s{s}',  'cache4arxiv_fft1024_m80', 0.99, 'DrongoNet-Edge'),
}


def load_test(cache):
    d = np.load(f'/Volumes/Evo/{cache}/test/mels.npz')
    return d['mels'].astype(np.float32), d['labels'].astype(np.int32)


def infer(tflite, mels):
    it = tf.lite.Interpreter(model_path=str(tflite)); it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    in_s, in_z = ind['quantization']; out_s, out_z = outd['quantization']
    x4 = mels[..., np.newaxis]
    probs = []
    for smp in x4:
        inp = smp[np.newaxis]
        inp = (np.round(inp / in_s + in_z).astype(ind['dtype']) if in_s != 0 else inp.astype(ind['dtype']))
        it.set_tensor(ind['index'], inp); it.invoke()
        raw = it.get_tensor(outd['index'])
        out = ((raw.astype(np.float32) - out_z) * out_s) if out_s != 0 else raw.astype(np.float32)
        probs.append(float(out[0, 1]))
    return np.array(probs)


def sweep(probs, y):
    rows = []
    for tau in THRESHOLDS:
        p = (probs >= tau).astype(int)
        tp = int(np.sum((p == 1) & (y == 1))); fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1))); tn = int(np.sum((p == 0) & (y == 0)))
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        rows.append(dict(tau=tau, recall=rec, precision=prec, f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn))
    return rows


def write_per_seed(rows, path, auc, seed, pretty, target):
    L = [f"{pretty} — Threshold Sweep  (seed={seed})", f"Target recall : ≥{target}",
         f"AUC           : {auc:.4f}  [full INT8]", "",
         f"{'τ':>6}  {'Recall':>8}  {'Precision':>10}  {'F1':>7}  {'FPR':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}",
         "-" * 72]
    for r in rows:
        L.append(f"{r['tau']:>6.3f}  {r['recall']:>8.4f}  {r['precision']:>10.4f}  {r['f1']:>7.4f}  "
                 f"{r['fpr']:>7.4f}  {r['tp']:>6}  {r['fp']:>6}  {r['fn']:>6}  {r['tn']:>6}")
    path.write_text("\n".join(L) + "\n")


def main():
    for name, (tmpl, cache, target, pretty) in VARIANTS.items():
        mels, y = load_test(cache)
        all_rows, aucs = {}, []
        print(f"\n===== {pretty}  (5-seed full-INT8 sweep, cache {cache}) =====", flush=True)
        for s in SEEDS:
            rd = BASE / 'results4arxiv' / tmpl.format(s=s)
            tfl = rd / 'model_int8.tflite'
            if not tfl.exists():
                print(f"  s{s}: MISSING {tfl}", flush=True); continue
            probs = infer(tfl, mels)
            auc = roc_auc_score(y, probs)
            rows = sweep(probs, y)
            all_rows[s] = rows; aucs.append(auc)
            write_per_seed(rows, rd / 'threshold_sweep.txt', auc, s, pretty, target)
            r035 = next(r for r in rows if abs(r['tau'] - (0.50 if name == 'edge' else 0.35)) < 1e-9)
            tag = 'τ0.50' if name == 'edge' else 'τ0.35'
            print(f"  s{s}: AUC={auc:.4f}  recall@{tag}={r035['recall']:.4f}  prec={r035['precision']:.4f}", flush=True)
        # 5-seed mean/std per tau, locked tau = highest tau with mean recall >= target
        seeds_ok = list(all_rows.keys())
        mean_rows = [{k: float(np.mean([all_rows[s][i][k] for s in seeds_ok])) for k in ('tau', 'recall', 'precision', 'f1', 'fpr')}
                     for i in range(len(THRESHOLDS))]
        std_rows = [{k: float(np.std([all_rows[s][i][k] for s in seeds_ok])) for k in ('recall', 'precision', 'f1')}
                    for i in range(len(THRESHOLDS))]
        cand = [i for i, m in enumerate(mean_rows) if m['recall'] >= target]
        li = max(cand, key=lambda i: mean_rows[i]['tau']) if cand else int(np.argmax([m['recall'] for m in mean_rows]))
        m, sd = mean_rows[li], std_rows[li]
        combined = (BASE / 'results4arxiv' / f'{name}_threshold_sweep_5seed.txt')
        out = [f"{pretty} — 5-seed full-INT8 threshold sweep", f"seeds = {seeds_ok}",
               f"AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
               f"LOCKED τ = {m['tau']:.2f}  (highest τ with mean recall ≥ {target})",
               f"  recall    = {m['recall']:.4f} ± {sd['recall']:.4f}",
               f"  precision = {m['precision']:.4f} ± {sd['precision']:.4f}",
               f"  f1        = {m['f1']:.4f} ± {sd['f1']:.4f}", "",
               f"{'τ':>6}  {'mean_rec':>9}  {'std_rec':>8}  {'mean_prec':>10}  {'std_prec':>9}"]
        for i in range(len(THRESHOLDS)):
            mk = "  <-- LOCKED" if i == li else ""
            out.append(f"{THRESHOLDS[i]:>6.3f}  {mean_rows[i]['recall']:>9.4f}  {std_rows[i]['recall']:>8.4f}  "
                       f"{mean_rows[i]['precision']:>10.4f}  {std_rows[i]['precision']:>9.4f}{mk}")
        combined.write_text("\n".join(out) + "\n")
        print(f"  LOCKED τ={m['tau']:.2f}  recall={m['recall']:.4f}±{sd['recall']:.4f}  "
              f"prec={m['precision']:.4f}±{sd['precision']:.4f}  → {combined.name}", flush=True)


if __name__ == '__main__':
    main()
