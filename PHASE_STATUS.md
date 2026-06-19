# Ablation Experiment Status — 2026-06-19

## Phase 0: TinyChirp Baseline (Reference)
**Status:** ⏭️ SKIPPED (reference work, not critical path)

**Rationale:**
- Phase 0 scripts exist (`0a`–`0e`) for training TinyChirp models on TinyChirp dataset
- Results would establish baseline AUC/size for comparison
- **Not required** for SEABADNet ablation chain (Micro/Edge)
- Can be run later for paper supplementary material

**To run if needed:**
```bash
python develop/0a_tinychirp_cnnmel.py --dataset-path /path/to/tinychirp
```

---

## Phase 1: n_mels Sweep on SEABAD (CNN-Mel Baseline)
**Status:** ✅ **COMPLETE** (15/15 runs)

**Completed runs:**
- Script: `1a_baseline2d.py`
- n_mels: 16, 32, 48, 64, 80 (5 values)
- Seeds: 42, 100, 786 (3 seeds)
- Results location: `results4arxiv/1a_baseline2d_fft1024_m{16..80}_s{42,100,786}/`

**Sample result (n_mels=80, seed=42):**
```
Float32 AUC: 0.9844
TFLite INT8 AUC: 0.9841
Accuracy: 0.9402
Model Size: 29.30 KB
Inference: 0.33 ms
```

**Gate 1 Decision:** Pick n_mels that maximizes validation AUC across all seeds
- Expected winner: **n_mels=64** (from prior literature)
- Confirm from aggregated results

---

## Phase 2: GAP Variants (Pooling Comparison)
**Status:** ⏹️ PENDING (ready to run)

**Scripts:**
- `2a_baseline_gap.py` — Plain GAP (expected winner)
- `2b_baseline_gap_learned.py` — GAP + learned pooling
- `2c_baseline_gap_1x1.py` — GAP + 1×1 bottleneck

**To run:**
```bash
# Requires: gate1_n_mels from Phase 1
python develop/run_phase1_sweep.py  # or manually invoke 2a/2b/2c
```

---

## Phase 3–6: Ablation Chain (Future)
**Status:** ⏹️ PENDING

The full pipeline (Phases 2–6) requires:
1. Completion of earlier phases
2. Gates decisions at each step
3. ~65 hours GPU time for all experiments

**Critical path:** Phase 1 → Phase 2 → Phase 3A/3B (Edge/Micro split) → Phase 4–6

---

## Summary Table

| Phase | Purpose | Status | Runs | Location |
|-------|---------|--------|------|----------|
| **0** | TinyChirp baseline (reference) | Skipped | 0/5 | — |
| **1** | n_mels sweep on SEABAD | ✅ Done | 15/15 | `results4arxiv/1a_baseline2d_*` |
| **2** | GAP vs Flatten | ⏹️ Pending | 0/9 | — |
| **3** | Conv type, filters, loss | ⏹️ Pending | 0/30 | — |
| **4** | Dropout sweep | ⏹️ Pending | 0/24 | — |
| **5** | Micro filter count | ⏹️ Pending | 0/2 | — |
| **6** | Final candidates (Micro/Edge) | ✅ Done | 6/6 | `results4arxiv/6b_micro_final_*`, `6c_edge_final_*` |

**Note:** Phase 6 was run out-of-sequence (final validation); represents locked configurations.

---

## Next Steps

**Option A — Full Ablation (recommended for paper):**
```bash
# Run Phases 2–5 in sequence, apply gates between each
# Total: ~60 hours GPU
python develop/run_phase2_gap_variants.py  # placeholder, write script
```

**Option B — Quick Validation:**
```bash
# Aggregate Phase 1 results to confirm Gate 1 decision
python develop/analyze_phase1_results.py
```

**Option C — Skip to Phase 6 (already done):**
Phase 6 results (SEABADNet-Micro and Edge) are already in `results4arxiv/`.
These are the **final locked configurations** for the paper.

---

## Files

- `retrain_phase0_on_seabad.py` — Attempted Phase 0 retraining (broken: missing source models)
- `run_phase1_sweep.py` — Phase 1 orchestrator (functional)

---

**Last updated:** 2026-06-19 23:45 UTC
