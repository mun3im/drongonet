# Pre-Closure TODO — Linux GPU

Three tasks must complete before journal finalisation. Run in order.

---

## 1. 6c Edge — seed × n_mels sweep (seeds 100 & 786)

Seed 42 at m80 is already done (AUC 0.9994). Need seeds 100 and 786 to
report mean ± std for the paper.

```bash
bash run_6c_edge_sweep.sh
```

Outputs into `results/6c_edge_final_fft1024_m80_s100/` and `…_s786/`.

---

## 2. 6c Edge — threshold sweep

Run after step 1. Operates on seed 42 result (single-seed is sufficient
for Edge; add seeds 100/786 if time allows).

```bash
bash run_threshold_sweep_edge.sh
```

Outputs into `results/6c_edge_final_fft1024_m80_s42/`:
- `threshold_sweep.txt` — full table
- `threshold_locked.txt` — locked τ + recall / precision / F1 / AUC / size / latency
- `threshold_sweep_pr.png` — PR curve with locked point

Expected: locked τ ≈ 0.35–0.45, recall ≥ 0.99 achievable (τ=0.5 already 0.9896).

---

## 3. 6b Micro — threshold sweep

Operates on existing seeds 42 / 100 / 786 results. No new training needed.

```bash
bash run_threshold_sweep_micro.sh
```

Outputs per-seed into `results/6b_micro_final_fft1024_m16_s{42,100,786}/`:
- `threshold_sweep.txt`
- `threshold_sweep_pr.png`

Combined outputs into `results/micro_threshold_sweep/`:
- `threshold_sweep_combined.txt` — all-seed table with mean ± std per τ
- `threshold_locked.txt` — locked τ + mean ± std metrics (copy into paper)
- `threshold_sweep_pr_combined.png` — all three seed PR curves overlaid

**Warning:** AUC 0.9741 with τ=0.5 recall ~0.926 (mean, 3 seeds). Script will
warn if no threshold reaches mean recall ≥ 0.98 and fall back to best available.
If that happens, flag for architecture review before finalising the paper claim.

---

## After all three

1. Fill paper values from `threshold_locked.txt` for each model:
   - SEABADNet-Edge: τ, recall, precision, F1, AUC, size (KB), latency (ms)
   - SEABADNet-Micro: same, reported as mean ± std across 3 seeds
2. Update CLAUDE.md experiment status tables with seed 100/786 Edge AUC values
3. If Micro misses 0.98 recall target, flag before submitting
