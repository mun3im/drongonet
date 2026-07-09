#!/usr/bin/env python3
"""Produce ALL 5-seed numbers needed to migrate the paper to a 5-seed mean-basis.
INT8 recall/precision/specificity/F1/F2 (deployed basis) + float32 AUC (from results_summary.txt).
Operating τ: Micro 0.35, Edge 0.425, Nano 0.37. Sweep grid matches paper (0.05)."""
import re
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score

BASE = Path('/home/muneim/Dropbox/Conda/drongonet'); SEEDS = [42, 100, 786, 7, 1234]
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
V = {  # name:(tmpl,cache,op_tau,floor)
 'Nano':('6a_nano_final_fft512_m16_s{s}','cache4arxiv_fft512_m16',0.37,0.97),
 'Micro':('6b_micro_final_fft1024_m16_s{s}','cache4arxiv_fft1024_m16',0.35,0.98),
 'Edge':('6c_edge_final_fft1024_m80_s{s}','cache4arxiv_fft1024_m80',0.425,0.99)}

def infer(t,m):
    it=tf.lite.Interpreter(model_path=str(t));it.allocate_tensors()
    i,o=it.get_input_details()[0],it.get_output_details()[0]
    iss,izz=i['quantization'];oss,ozz=o['quantization'];pr=[]
    for s in m[...,None]:
        x=s[None];x=np.round(x/iss+izz).astype(i['dtype']) if iss!=0 else x.astype(i['dtype'])
        it.set_tensor(i['index'],x);it.invoke();r=it.get_tensor(o['index'])
        pr.append(float(((r.astype(np.float32)-ozz)*oss if oss!=0 else r)[0,1]))
    return np.array(pr)

def cm(p,y,t):
    q=(p>=t).astype(int)
    tp=np.sum((q==1)&(y==1));fp=np.sum((q==1)&(y==0));fn=np.sum((q==0)&(y==1));tn=np.sum((q==0)&(y==0))
    rec=tp/(tp+fn);pre=tp/(tp+fp);spec=tn/(tn+fp)
    f1=2*pre*rec/(pre+rec);f2=5*pre*rec/(4*pre+rec)
    return dict(rec=rec,pre=pre,spec=spec,f1=f1,f2=f2,tp=tp,fp=fp,fn=fn,tn=tn)

def f32auc(name,s):
    tmpl=V[name][0]; f=BASE/'results4arxiv'/tmpl.format(s=s)/'results_summary.txt'
    m=re.search(r'Float32 Model[:\s]*\n?\s*AUC:\s*([0-9.]+)',f.read_text())
    return float(m.group(1)) if m else None

for name,(tmpl,cache,op,floor) in V.items():
    d=np.load(f'/Volumes/Evo/{cache}/test/mels.npz');m,y=d['mels'].astype(np.float32),d['labels'].astype(np.int32)
    sp={s:infer(BASE/'results4arxiv'/tmpl.format(s=s)/'model_int8.tflite',m) for s in SEEDS}
    fa=[f32auc(name,s) for s in SEEDS]; ia=[roc_auc_score(y,sp[s]) for s in SEEDS]
    print(f"\n############### {name}  (op τ={op}, floor {floor}) ###############")
    print(f"  AUC float32 5-seed = {np.mean(fa):.4f} ± {np.std(fa):.4f}   per-seed {[round(x,4) for x in fa]}")
    print(f"  AUC int8    5-seed = {np.mean(ia):.4f} ± {np.std(ia):.4f}")
    # full sweep table (mean over seeds)
    print(f"  {'τ':>5} {'recall':>16} {'precision':>16} {'specificity':>16} {'F1':>14} {'F2':>14}")
    for t in GRID:
        R=np.array([cm(sp[s],y,t)['rec'] for s in SEEDS]);P=np.array([cm(sp[s],y,t)['pre'] for s in SEEDS])
        S=np.array([cm(sp[s],y,t)['spec'] for s in SEEDS]);F1=np.array([cm(sp[s],y,t)['f1'] for s in SEEDS]);F2=np.array([cm(sp[s],y,t)['f2'] for s in SEEDS])
        star=' *OP' if abs(t-op)<1e-9 else ''
        print(f"  {t:>5.2f} {R.mean():.4f}±{R.std():.4f} {P.mean():.4f}±{P.std():.4f} {S.mean():.4f}±{S.std():.4f} {F1.mean():.4f}±{F1.std():.4f} {F2.mean():.4f}±{F2.std():.4f}{star}")
    # operating point (may be off-grid, e.g. Edge 0.425)
    Rop=np.array([cm(sp[s],y,op)['rec'] for s in SEEDS]);Pop=np.array([cm(sp[s],y,op)['pre'] for s in SEEDS])
    Sop=np.array([cm(sp[s],y,op)['spec'] for s in SEEDS]);F1o=np.array([cm(sp[s],y,op)['f1'] for s in SEEDS]);F2o=np.array([cm(sp[s],y,op)['f2'] for s in SEEDS])
    tp=np.mean([cm(sp[s],y,op)['tp'] for s in SEEDS]);fp=np.mean([cm(sp[s],y,op)['fp'] for s in SEEDS])
    fn=np.mean([cm(sp[s],y,op)['fn'] for s in SEEDS]);tn=np.mean([cm(sp[s],y,op)['tn'] for s in SEEDS])
    print(f"  >> OP τ={op}: recall {Rop.mean():.4f}±{Rop.std():.4f} (min {Rop.min():.4f}) | prec {Pop.mean():.4f}±{Pop.std():.4f} | spec {Sop.mean():.4f} | F1 {F1o.mean():.4f} | F2 {F2o.mean():.4f}")
    print(f"     per-seed recall {dict(zip(SEEDS,[round(v,4) for v in Rop]))}")
    print(f"     confusion (mean/5000): TP={tp:.0f} FN={fn:.0f} FP={fp:.0f} TN={tn:.0f}  FPR={fp/(fp+tn):.4f}")
