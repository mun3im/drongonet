# Ablation Pipeline Checkpoint — 2026-06-19

## Status Summary

| Task | Status | Details |
|------|--------|---------|
| **1. Clean up untracked scripts** | ✅ DONE | Committed `retrain_phase0_on_seabad.py` and `run_phase1_sweep.py` to GitHub |
| **2. Debug & rerun Phase 0** | ℹ️ CLARIFIED | Phase 0 is reference work (TinyChirp on TinyChirp). Not required for SEABADNet ablation chain. Skipped. |
| **3. Execute Phase 1 sweep** | ✅ DONE | All 15 runs (5 n_mels × 3 seeds) complete with Gate 1 decision applied. |

---

## Phase 1: Complete Results & Gate 1 Decision

**Configuration:**
- Script: `1a_baseline2d.py` (CNN-Mel on SEABAD)
- n_fft: 1024
- Dropout: 0
- Dense: 8
- Seeds: 42, 100, 786
- Results: `results4arxiv/1a_baseline2d_fft1024_m{16,32,48,64,80}_s{42,100,786}/`

**Results by n_mels:**
```
n_mels=16: AUC 0.9706 ± 0.0011  (variable, lower)
n_mels=32: AUC 0.9786 ± 0.0008  (good)
n_mels=48: AUC 0.9797 ± 0.0024  (good, higher variance)
n_mels=64: AUC 0.9823 ± 0.0003  ← WINNER (most stable, highest AUC)
n_mels=80: AUC 0.9822 ± 0.0017  (tied, higher variance)
```

**Gate 1 Decision:**
✅ **n_mels=64** advances to Phase 2
- Highest mean AUC: 0.9823
- Lowest variance: ±0.0003 (most robust)
- Excellent consistency across seeds (0.9825 / 0.9826 / 0.9819)

**Also preserved for Micro branch:**
- n_mels=16 results (AUC 0.9706) → baseline for Micro track

---

## Ablation Pipeline State

### Locked Decisions
- Phase 1: n_mels=64 → proceed to Phase 2
- Phase 1: n_mels=16 preserved → separate Micro ablation track

### Ready for Next Phase
- **Phase 2** (GAP vs Flatten): 3 scripts ready
  - `2a_baseline_gap.py` (expected winner)
  - `2b_baseline_gap_learned.py`
  - `2c_baseline_gap_1x1.py`

### Already Completed (Out-of-Sequence)
- **Phase 6** (Final candidates): 6/6 runs complete
  - `6a_nano_final` (5.41 KB, AUC 0.9715)
  - `6b_micro_final` (6.56 KB, AUC 0.9743, recall ≥98.1%)
  - `6c_edge_final` (33.06 KB, AUC 0.9992, recall ≥99.0%)

---

## Recommended Next Steps

### Option A: Continue Ablation Chain (Full Investigation)
Run Phases 2–5 in sequence to document design decisions:
```bash
# Phase 2: GAP variants
python develop/2a_baseline_gap.py --dataset-path ... --n_mels 64

# Phase 3: Conv type, filters, loss
# Phase 4: Dropout sweep
# Phase 5: Micro filter count

# Expected time: ~60 hours GPU
```

**Output:** Complete ablation tables for paper (rationale for each design choice)

### Option B: Fast-Forward (Already Have Answer)
Skip to Phase 6 results, which are final locked configurations:
```bash
# SEABADNet-Micro: 6b_micro_final (6.56 KB, 98.1% recall)
# SEABADNet-Edge: 6c_edge_final (33.06 KB, 99.0% recall)
```

**Output:** Minimal time; paper uses final results directly

### Option C: Hybrid (Recommended for Paper)
Run only critical phases (2–3) to show design narrative:
- Phase 2: Justify GAP pooling (size reduction)
- Phase 3: Justify focal loss + depthwise separable (recall improvement)
- Phase 6: Final validation (already done)

**Output:** ~20 hours GPU; addresses key design questions

---

## Files Committed Today

- `edge_deploy/` — Complete SEABADNet-Edge package for Raspberry Pi (1232 lines added)
- `develop/retrain_phase0_on_seabad.py` — Phase 0 utility script
- `develop/run_phase1_sweep.py` — Phase 1 orchestrator
- Updated `deploy/README.md` with link to RPi deployment

---

## What's NOT Needed

❌ Phase 0 (TinyChirp baseline) — reference only, not critical path  
❌ Retraining Phase 0 models on SEABAD — conceptual mismatch  
❌ Re-running Phase 1 — complete, validated, Gate 1 applied  

---

## To Resume Ablation

When ready to continue:

1. **Decision point:** Choose Option A, B, or C above
2. **Checkpoint:** Gate 1 = n_mels=64
3. **Script:** Use `run_phase1_sweep.py` as template for orchestration
4. **Validation:** Aggregate results with `analyze_phase1.py` logic

---

**Generated:** 2026-06-19 23:47 UTC  
**Ready for:** Paper writing, Phase 2–3 experiments, or publication with Phase 6 results
