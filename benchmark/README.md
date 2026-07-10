# benchmark/

Comparisons for DrongoNet against other datasets and architectures. Most scripts are
cross-dataset generalization benchmarks (retrain fresh from scratch — no SEABAD weights
transferred); `8a`–`8c` are a same-dataset (SEABAD) architecture comparison instead
(moved from `develop/` 2026-07-10 — not part of the ablation chain, not in the paper;
see CLAUDE.md's "Post-Ablation Exploration" and "Final decision" notes for why).

## Scripts

| File | Purpose |
|------|---------|
| `dcase_benchmark.py` | In-domain DCASE-2018 (ff1010+warblr+BirdVox pooled split) |
| `dcase_crosscorpus.py` | Cross-corpus augmentation sweep: train ff1010+warblr → test BirdVox |
| `augmentations.py` | Augmentation recipes: `none / mixup / specaug / pitch_time / full` |
| `tinychirp_generalization.py` | TinyChirp Corn Bunting (paper-reported, f32 AUC only) |
| `tinychirp_benchmark.py` | TinyChirp + INT8 quantization eval + mel caching (extended version) |
| `bulbul_arch.py` | bulbul (Grill et al.) reference architecture, our pipeline |
| `dcase_crosscorpus_bulbul.py` | bulbul cross-corpus comparison, matched protocol |
| `8a_matchbox_micro.py` | MatchboxNet-inspired variant on SEABAD, n_mels=16 (same-dataset, not cross-corpus) |
| `8b_wrennet_frontend.py` | Matchbox + WrenNet semi-learnable frequency front-end |
| `8c_wrennet_matchbox_fusion.py` | WrenMatch: WrenNet + MatchboxNet fusion (SE + causal convs) |
| `_dcase_diagnose.py` | Stale diagnostic — ignore |
| `run_dcase_benchmark.sh` | Batch runner for `dcase_benchmark.py` |
| `run_augmentation_sweep.sh` | Batch runner for the augmentation sweep (default: 27 runs) |
| `run_window_sweep.sh` | Window-count robustness sweep (N=4/5/6) |
| `run_bulbul_match.sh` | bulbul-matched protocol (ff1010+warblr dev only, seeds 123/456/789) |

---

## DCASE-2018 In-Domain — `dcase_benchmark.py`

Pool all three DCASE corpora (ff1010bird 7,690 + warblrb10k 8,000 + BirdVox-DCASE-20k 20,000 = 35,690 clips), stratified clip-level split ≈72/8/20%. Every corpus appears in every split so there is no domain gap — this tests whether the architecture can learn DCASE bird detection at all, not cross-corpus transfer.

Each 10 s clip is scored via **6 evenly-spaced 3 s windows** (starts 0/1.4/2.8/4.2/5.6/7.0 s); clip score = MAX window probability.

```bash
bash benchmark/run_dcase_benchmark.sh                                  # all variants
conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant micro --seeds 42 100 786
```

**Output:** `results4arxiv/dcase_benchmark_{variant}_r{seed}/summary.json`

---

## DCASE-2018 Cross-Corpus Augmentation Sweep — `dcase_crosscorpus.py`

**The question:** Does log-mel-domain augmentation close the cross-corpus gap?

Trains on ff1010+warblr only; evaluates zero-shot on held-out BirdVox-DCASE-20k. Augmentation recipes applied per batch during training (see `augmentations.py`):

| Recipe | What it does |
|--------|-------------|
| `none` | Baseline (no augmentation) |
| `mixup` | Linear interpolation of two random samples (α=0.2) |
| `specaug` | Zero-out random freq band (≤2 mel bins) + time band (≤10 frames) |
| `pitch_time` | Mel-domain freq roll ±2 bins + bilinear time-axis resize ±10% |
| `full` | specaug → pitch_time → mixup stacked |

**Decision gate (after Micro × {mixup, specaug}, ~2.5 h):**

| Cross-corpus AUC | Action |
|-----------------|--------|
| ≥ 0.60 | Proceed to full sweep (all variants × all recipes) |
| 0.55–0.60 | Run `full` Micro only; hold nano/edge |
| < 0.55 | Abort — confirms MCU-scale architectural floor; report as §10 evidence |

```bash
# Step 1 — cheapest signal first
bash benchmark/run_augmentation_sweep.sh --recipes "mixup specaug" --variants "micro"

# Step 3 (if not aborted) — full sweep
bash benchmark/run_augmentation_sweep.sh --recipes "mixup specaug" --variants "nano edge"
bash benchmark/run_augmentation_sweep.sh --recipes "full" --variants "nano micro edge"

# View results anytime
conda run -n tf215_gpu python3 - <<'PY'
import json, os, numpy as np
print(f'{"variant":7}  {"recipe":11}  {"cross-corpus AUC":22}  per-seed')
for v in ('nano','micro','edge'):
    for r in ('none','mixup','specaug','pitch_time','full'):
        cx = []
        for s in (42,100,786):
            p = f'results4arxiv/dcase_crosscorpus_{v}_{r}_r{s}/summary.json'
            if os.path.exists(p):
                cx.append(json.load(open(p))['crosscorpus_birdvox_auc'])
        if cx:
            print(f'{v:7}  {r:11}  {np.mean(cx):.4f} +/- {np.std(cx):.4f}  {[round(x,4) for x in cx]}')
PY
```

**Output:** `results4arxiv/dcase_crosscorpus_{variant}_{aug}_r{seed}/summary.json`

---

## TinyChirp Corn Bunting — `tinychirp_generalization.py` / `tinychirp_benchmark.py`

Retrains DrongoNet on TinyChirp's single-species Corn Bunting dataset (different region, single species, native 3 s clips). Uses published train/val/test splits.

- `tinychirp_generalization.py` — paper-reported version (f32 AUC, no INT8 eval)
- `tinychirp_benchmark.py` — extended version with INT8 quantization, mel caching, per-seed TFLite eval

**Data:** `/Volumes/Evo/TinyChirp/{training,validation,testing}/{target,non_target}/` (16 kHz, 3 s)

**Mel config:** hop=256, fmin=100, fmax=8000, center=False → 184 frames, per-sample [0,1] norm. Per-variant n_fft/n_mels: Nano (512/16), Micro (1024/16), Edge (1024/80).

```bash
conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant micro --seeds 42 100 786
```

**Output:** `results4arxiv/tinychirp_benchmark_{variant}_r{seed}/summary.json`

---

## Training Config (All Scripts)

AdamW lr=3e-4, weight_decay=1e-4 · Focal Loss γ=2.0, α=0.5 · 50 epochs · EarlyStopping(val_accuracy, patience=10) · ReduceLROnPlateau(factor=0.5, patience=5)

---

## Results

### TinyChirp (primary generalization evidence)

| Variant | Test AUC | Seeds |
|---------|----------|-------|
| Nano (763 params) | 0.9664 ± 0.0084 | 42/100/786 |
| Micro (919 params) | 0.9757 ± 0.0085 | 42/100/786 |
| Edge (25,890 params) | 0.9997 ± 0.0004 | 42/100/786 |

Micro matches TinyChirp's published CNN-Mel baseline (0.9985) at 28× fewer parameters.

### DCASE-2018 (secondary evidence — different clip length, different corpora)

| Variant | In-domain AUC | Cross-corpus → BirdVox (no aug) |
|---------|--------------|---------------------------------|
| Nano | 0.821 ± 0.002 | 0.446 (chance) |
| Micro | 0.847 ± 0.011 | 0.488 (chance) |
| Edge | 0.938 ± 0.010 | 0.646 |
| **bulbul** (ref, 373k params) | ~0.96 (5-fold CV) | 0.887 / 0.855 (with/without aug) |

In-domain seeds: 123/456/789 (bulbul-matched). Cross-corpus seeds: 42/100/786.

Edge (80-mel) is bulbul's architectural match: 0.938 vs bulbul's 0.96 at **14× fewer parameters and no augmentation**. Cross-corpus collapse is the same train↔test domain gap bulbul documents (Pearson r=0.40), amplified by 14–489× fewer parameters.

### Cross-corpus augmentation sweep (does aug close the gap at MCU scale?)

**Single protocol** (`dcase_crosscorpus.py`): train ff1010+warblr → zero-shot test on
BirdVox-DCASE-20k, 10% held-out val, 50 epochs, 6-window MAX aggregation, seeds 42/100/786.
All numbers below are *within this protocol* — do **not** compare against the bulbul-match
DCASE table above (different split/seeds/training config; e.g. that table's Micro no-aug=0.488,
this protocol's Micro no-aug=0.527).

Micro has the full augmentation ladder; Edge/Nano have the no-aug baseline and the strongest
(`full`) recipe, so per-variant deltas are now valid within this protocol (P0 baselines completed
2026-06-29).

| Variant | Params | none | mixup | specaug | full | Δ(full−none) |
|---------|--------|------|-------|---------|------|--------------|
| Nano  | 763    | 0.4904 ± 0.0485 | 0.4809 ± 0.0207 | 0.4476 ± 0.0340 | 0.5622 ± 0.0399 | **+7.2 pp** |
| Micro | 919    | 0.5271 ± 0.0193 | 0.5250 ± 0.0317 | 0.4855 ± 0.0370 | 0.4587 ± 0.0341 | **−6.8 pp** |
| Edge  | 25,890 | **0.6799 ± 0.0158** | 0.6513 ± 0.0168 | 0.6516 ± 0.0225 | 0.5909 ± 0.0346 | **−8.9 pp** |

Full ladder now complete for all three (P0 mixup/specaug cells for Nano/Edge done 2026-06-30).
For Edge **every** rung is below the no-aug baseline (0.680) — the regression is recipe-robust,
not a `full`-stack artefact.

**Verdict (H3):** Augmentation helps **only the smallest model** (Nano +7.2 pp) and *regresses*
both Micro (−6.8 pp) and Edge (−8.9 pp). For Micro, no recipe beats the no-aug baseline (0.527):
mixup is flat (0.525), specaug and full are below (0.486, 0.459), and in-domain val AUC drops
0.865→0.811 under `full` — the 919-param model trades training-fit for augmentation it cannot
exploit. Without augmentation, cross-corpus transfer rises monotonically with capacity
(Nano 0.490 < Micro 0.527 < **Edge 0.680**): the un-augmented Edge is the best transfer model and
comes closest to the ≥0.70 "gap closed" threshold, and `full` *erodes* it. **Capacity, not
augmentation, is the operative variable.** In contrast to bulbul (augmentation +3 pp at 373k
params), across our tested range — an order of magnitude below that capacity — stacked mel-domain
augmentation regresses every model above the smallest. This *strengthens* the in-region deployment
scope: at sub-1 kB budgets, target-domain retraining (not augmentation) is the remedy. Written up
in paper §discussion (`tab:aug_sweep`). See
`DCASE_AUGMENTATION_PLAN.md` (Paper3_SEABADnet_ApplAcoust) for full hypotheses.
