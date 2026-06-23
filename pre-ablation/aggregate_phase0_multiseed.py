#!/usr/bin/env python3
"""Aggregate Phase 0 multi-seed results: native TinyChirp test AUC (0a-0e) and
zero-shot SEABAD AUC (0f-0j), mean +/- std across seeds 42/100/786."""
import json
import re
import numpy as np
from pathlib import Path

SEEDS = [42, 100, 786]
ARCH = [
    ('CNN-Mel',         '0a_tinychirp_cnnmel',       '0f_tinychirp_cnnmel_zeroshot'),
    ('CNN-Time',        '0b_tinychirp_cnntime',      '0g_tinychirp_cnntime_zeroshot'),
    ('Transformer',     '0c_tinychirp_transformer',  '0h_tinychirp_transformer_zeroshot'),
    ('SqueezeNet-Time', '0d_tinychirp_squeezenet',   '0i_tinychirp_squeezenettime_zeroshot'),
    ('SqueezeNet-Mel',  '0e_tinychirp_squeezenetmel','0j_tinychirp_squeezenetmel_zeroshot'),
]
BASE = Path('results4arxiv')


def native_auc(prefix, seed):
    f = BASE / f'{prefix}_r{seed}_linux' / 'results_summary.txt'
    if not f.exists():
        return None
    m = re.findall(r'AUC[:=]\s*([0-9.]+)', f.read_text())
    return float(m[0]) if m else None      # first match = float32 native AUC


def zs_auc(prefix, seed):
    f = BASE / f'{prefix}_r{seed}' / 'summary.json'
    if not f.exists():
        return None
    return json.load(open(f))['auc']


def fmt(vals):
    v = [x for x in vals if x is not None]
    per = '/'.join('--' if x is None else f'{x:.4f}' for x in vals)
    if not v:
        return f'n/a            ({per})'
    return f'{np.mean(v):.4f}+/-{np.std(v):.4f}  ({per})'


print('\n' + '=' * 104)
print('PHASE 0 MULTI-SEED  (seeds 42 / 100 / 786)')
print('=' * 104)
print(f'{"Architecture":16} | {"TinyChirp native AUC":34} | {"SEABAD zero-shot AUC":34}')
print('-' * 104)
for name, tp, zp in ARCH:
    print(f'{name:16} | {fmt([native_auc(tp, s) for s in SEEDS]):34} | {fmt([zs_auc(zp, s) for s in SEEDS]):34}')
print('=' * 104)
print('Note: zero-shot uses the FIXED SEABAD test split (seed 42); only the model training seed varies.')
