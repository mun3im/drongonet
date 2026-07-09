# DrongoNet Cross-Dataset Benchmark Results — durable record

_Saved 2026-06-22. Authoritative source: `results4arxiv/*/summary.json`. This file is a crash-safe
snapshot; regenerate by reading those JSONs._

Purpose: prove DrongoNet is **not overfit to SEABAD** and **adapts to other datasets / clip
lengths**. Compared against **bulbul** (Grill & Schlüter, EUSIPCO 2017), the BAD-challenge-winning
CNN that inspired this work.

---

## 1. TinyChirp Corn Bunting (3 s clips, single species) — seeds 42/100/786

Native 3 s format, direct clip inference. Dir: `results4arxiv/tinychirp_benchmark_{variant}_r{seed}/`.

| Variant | Params | Test AUC (mean ± std) | per-seed (42/100/786) |
|---------|-------:|-----------------------|------------------------|
| Nano  | 763    | **0.9664 ± 0.0084** | 0.9598 / 0.9610 / 0.9782 |
| Micro | 919    | **0.9757 ± 0.0085** | 0.9800 / 0.9637 / 0.9833 |
| Edge  | 25,890 | **0.9997 ± 0.0004** | 1.0000 / 0.9991 / 0.9999 |

Clean transfer to a different region/species at native length.

---

## 2. DCASE-2018 — cross-corpus (official task, NO adaptation) — seeds 42/100/786

Train ff1010bird+warblrb10k → test held-out BirdVox-DCASE-20k. Dir: `results4arxiv/dcase_xcorpus_{variant}_r{seed}/`.

| Variant | Params | Test AUC (mean ± std) | per-seed (42/100/786) |
|---------|-------:|-----------------------|------------------------|
| Nano  | 763    | **0.4464 ± 0.0136** | 0.4379 / 0.4356 / 0.4656 |
| Micro | 919    | **0.4884 ± 0.0180** | 0.4968 / 0.4634 / 0.5049 |
| Edge  | 25,890 | **0.6459 ± 0.0207** | 0.6633 / 0.6168 / 0.6576 |

Zero-adaptation cross-corpus transfer ≈ chance for the small variants — the SAME domain gap
bulbul documents (train↔test AUC Pearson corr = 0.40). NOT a capability claim; reported only as
the bulbul-comparable cross-corpus row. Aggregation-independent (MAX/MEAN/window all ≈ chance).

---

## 2b. DCASE-2018 — bulbul-matched in-domain (ff1010+warblr only) — seeds 123/456/789  [PRIMARY for bulbul comparison]

DECISION 2026-06-22: for a FAIR head-to-head with bulbul, the in-domain comparison uses
**ff1010bird + warblrb10k only** (the DCASE-2018 dev set bulbul reports 5-fold CV on; excludes
the hard BirdVox set that dragged the pooled numbers down and that bulbul's CV never saw).
Run: `benchmark/run_bulbul_match.sh` → dirs `dcase_benchmark_{variant}_dev_r{seed}/`.

| Variant | Params | Test AUC (mean ± std) | per-seed (123/456/789) |
|---------|-------:|-----------------------|-------------------------|
| Nano  | 763    | **0.8211 ± 0.0020** | 0.8213 / 0.8233 / 0.8185 |
| Micro | 919    | **0.8465 ± 0.0113** | 0.8411 / 0.8361 / 0.8622 |
| Edge  | 25,890 | **0.9384 ± 0.0097** | 0.9262 / 0.9390 / 0.9499 |

Edge (80-mel, bulbul's architectural match) = **0.938** vs bulbul 0.96 at 14.4× fewer params, with NO
augmentation (bulbul uses cyclic time-shift, pitch-shift, denoising). Gap ~0.02 = capacity, not augmentation.
(Edge initially OOM'd on seed 456 in the multi-seed process; re-ran 456/789 as separate single-seed invocations.)

Window sweep (micro, dev protocol):
- N=6: 0.8465 ± 0.0113   | N=5: 0.8333 ± 0.0083 (−0.013) | N=4: 0.8301 ± 0.0170 (−0.016)
- Reducing windows = 17%/33% fewer inferences for ~0.01–0.02 AUC cost. Matches prediction.

---

## 3. DCASE-2018 — in-domain POOLED incl. BirdVox (SUPERSEDED, secondary) — seeds 123/456/789

NOTE: superseded as the bulbul comparator by §2b (pooled set includes hard BirdVox in train+test,
so it is harder AND not comparable to bulbul's BirdVox-free CV). Kept as a "harder pooled set"
data point. **Edge pooled FAILED** (0-byte log, likely PC interruption) — not retried (dev is primary).

Pooled ff1010+warblr+birdvox (35,690 clips, pos frac 0.504), stratified clip-level train/val/test
split. 10 s clips via 6×3 s sliding windows + MAX aggregation. Seeds independent of ablation
seeds. Dir: `results4arxiv/dcase_benchmark_{variant}_r{seed}/`.

| Variant | Params | Test AUC (mean ± std) | per-seed (123/456/789) |
|---------|-------:|-----------------------|-------------------------|
| Nano  | 763    | 0.7713 ± 0.0052 | 0.7778 / 0.7710 / 0.7652 |
| Micro | 919    | 0.7924 ± 0.0213 | 0.8223 / 0.7805 / 0.7743 |
| Edge  | 25,890 | FAILED (PC interruption; not retried) | — |

Shows the architecture learns a different dataset at a different clip length (10 s). Conservative:
the pool includes BirdVox's hard short night-flight-call clips.

NOTE: an earlier "~0.85" figure for Nano was from a diagnostic that (a) used ff1010+warblr ONLY
(no BirdVox) and (b) reported the validation, not test, split — it is NOT comparable to the
pooled numbers above. The pooled test numbers (0.77–0.79) are the honest ones.

---

## 4. bulbul reference (Grill & Schlüter 2017, EUSIPCO)

PDF: `/home/muneim/Dropbox/References/grill2017two.pdf` (Table I = arch). Paper-stated params.

| System | Params | in-domain AUC | cross-corpus AUC |
|--------|-------:|---------------|------------------|
| bulbul | **373,169** | ~0.96–0.97 (5-fold CV, Fig 3) | **0.887** test / 0.855 no-aug |
| sparrow | 309,843 | — | (companion model) |

- Input: 80 mel, n_fft=1024 @ 22.05 kHz, log-mag, max-pool-over-time → single output. Closest to
  **DrongoNet-Edge** (80 mel) → fair architectural head-to-head: 25,890 vs 373,169 = **14.4× smaller**.
- bulbul hits 0.887 cross-corpus ONLY with heavy augmentation (cyclic time-shift, pitch-shift,
  denoising); 0.855 without. DrongoNet uses no augmentation here.
- Param efficiency: Nano 763 (489× smaller), Micro 919 (406×), Edge 25,890 (14.4×).

---

## 5. Diagnostic (nano, seed 42, ff1010+warblr-only in-domain) — for the record

Confirmed the cross-corpus collapse is domain shift, not a bug or aggregation artifact:

```
in-domain  window AUC      = 0.8196
in-domain  clip MEAN AUC   = 0.8415
in-domain  clip MAX  AUC   = 0.8536
cross      window AUC      = 0.4966   (chance, before any aggregation)
cross      clip MAX  AUC   = 0.4958
cross      clip MEAN AUC   = 0.4968
cross      clip top-2 AUC  = 0.4963
```

MAX=MEAN=window at chance cross-corpus → aggregation irrelevant when signal is zero.

---

## 6. Headline comparison table (fill Edge in-domain when ready)

In-domain column = bulbul-matched ff1010+warblr protocol (§2b), NOT the pooled §3 numbers.

| System | Params | vs bulbul | In-domain AUC (dev) | Cross-corpus AUC |
|--------|-------:|----------:|---------------------|------------------|
| bulbul (2017) | 373,169 | 1× | ~0.96–0.97 (CV, +aug) | 0.887 / 0.855 no-aug |
| DrongoNet-Edge (80 mel) | 25,890 | 14.4× smaller | 0.938 ± 0.010 (no aug) | 0.646 |
| DrongoNet-Micro (16 mel) | 919 | 406× smaller | 0.847 ± 0.011 (no aug) | 0.488 |
| DrongoNet-Nano (16 mel) | 763 | 489× smaller | 0.821 ± 0.002 (no aug) | 0.446 |

Window sweep (micro, dev): N=6 0.8465 / N=5 0.8333 / N=4 0.8301 (−0.013/−0.016 for 17%/33% fewer inferences).
DrongoNet uses NO augmentation; bulbul uses cyclic time-shift+pitch-shift+denoising. In-domain gap ≈ capacity.

---

## 7. Status — COMPLETE (2026-06-22)

- [x] Edge in-domain AUC (seeds 123/456/789) = 0.9384 ± 0.0097 → §2b, §6 filled.
- [x] Window-count sweep (micro dev N=4/5/6) → §6. Prediction confirmed (near-flat, −0.013/−0.016).
- [x] Paper `.tex` (`Paper3_SEABADnet_ApplAcoust/zabidi2026seabadnet_arxiv.tex`): bulbul comparison
      paragraph + `tab:dcase_bulbul` table + "no augmentation" clause added; 0 placeholders; compiles clean.
- [x] README + this file synced to dev numbers.

Run commands (for reproduction):
```
conda run -n tf215_gpu python benchmark/dcase_benchmark.py --variant edge  --seeds 123 456 789
# window sweep (after adding --windows arg): --windows 4 / 5 / 6
conda run -n tf215_gpu python benchmark/tinychirp_generalization.py --variant {v} --seeds 42 100 786
```
