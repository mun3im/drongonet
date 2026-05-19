# Threshold Sweep — Linux TODO

## Edge

```bash
python threshold_sweep_edge.py \
    --results-dir results/6b_edge_final_fft1024_m80_s42 \
    --cache-dir   /data/cache_seabad_m80
```

Outputs into `results/6b_edge_final_fft1024_m80_s42/`:
- `threshold_sweep.txt` — full table
- `threshold_locked.txt` — locked τ + recall / precision / f1 / AUC / size / latency
- `threshold_sweep_pr.png` — PR curve with locked point

Expected: locked τ ≈ 0.35–0.45, recall ≥ 0.99 achievable (τ=0.5 already 0.9896).

## Micro

```bash
python threshold_sweep_micro.py \
    --results-base results \
    --cache-dir    /data/cache_seabad_m16 \
    --out-dir      results/micro_threshold_sweep
```

Outputs per-seed into `results/6b_micro_improved_fft1024_m16_s{42,100,786}/`:
- `threshold_sweep.txt`
- `threshold_sweep_pr.png`

Combined outputs into `results/micro_threshold_sweep/`:
- `threshold_sweep_combined.txt` — all-seed table with mean ± std per τ
- `threshold_locked.txt` — locked τ + mean ± std metrics (copy into paper)
- `threshold_sweep_pr_combined.png` — all three seed PR curves overlaid

**Warning:** AUC 0.9741 with τ=0.5 recall ~0.926 (mean, 3 seeds). The script
will warn if no threshold achieves mean recall ≥ 0.98 and fall back to the
best available. If that happens, the architecture needs revisiting before
the paper can claim the 0.98 target.

## After running both

1. Read `threshold_locked.txt` for each model and fill paper values:
   - SEABADNet-Edge: τ, recall, precision, F1, AUC, size (KB), latency (ms)
   - SEABADNet-Micro: same, reported as mean ± std across 3 seeds
2. If Micro fails to reach 0.98 recall at any τ, flag for architecture review
   before finalising the paper claim
