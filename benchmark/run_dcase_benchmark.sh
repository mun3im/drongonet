#!/bin/bash
# run_dcase_benchmark.sh — SEABADNet on DCASE-2018 BAD (6x3s sliding-window, max-agg).
# All seeds for a variant run in one invocation (wav decode once; mels in RAM).
set -u
ROOT="/home/muneim/Dropbox/Conda/seabadnet"
cd "$ROOT"
ENV=tf215_gpu
LOG="benchmark_dcase_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== SEABADNet on DCASE-2018 BAD benchmark: nano+micro+edge, seeds 42/100/786 ==="
for v in nano micro edge; do
  log "[run] $v seeds 42 100 786"
  if conda run -n $ENV python benchmark/dcase_benchmark.py --variant "$v" --seeds 42 100 786 >> "$LOG" 2>&1; then
    log "  ok: $v"
  else
    log "  FAIL: $v"
  fi
done

log "=== Results ==="
conda run -n $ENV python -c "
import json, numpy as np
print('\nSEABADNet on DCASE-2018 BAD (BirdVox-20k, 6x3s sliding-window, clip-level MAX aggregation)')
print('variant      test AUC (mean ± std)    per-seed')
for v in ['nano','micro','edge']:
    aucs=[]
    for s in [42,100,786]:
        try: aucs.append(json.load(open(f'results4arxiv/dcase_benchmark_{v}_r{s}/summary.json'))['test_auc'])
        except Exception: pass
    if aucs:
        print(f'{v:8}    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}        {[round(a,4) for a in aucs]}')
" 2>&1 | tee -a "$LOG"
log "=== DONE -> $LOG ==="
