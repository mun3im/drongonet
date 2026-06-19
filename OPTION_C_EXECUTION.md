# Option C Execution Plan — SEABADNet Ablation Pipeline

**Status:** ✅ Started 2026-06-19 23:58 UTC  
**Estimated completion:** 2026-06-20 06:58 UTC (~7 hours)

---

## 🎯 What is Option C?

Run **Phases 2–3** to document key design decisions (GAP pooling, focal loss, architecture variants), then validate with **Phase 6** results (already complete).

This balances:
- **Justification:** Why each architectural choice matters
- **Efficiency:** ~20 hours GPU instead of 60 hours
- **Completeness:** Enough ablations to support paper narrative

---

## 📊 Execution Timeline

### ✅ Already Complete (This Session)

| Phase | Status | Runs | Location |
|-------|--------|------|----------|
| **Phase 0** | Skipped (reference) | 0 | — |
| **Phase 1** | ✅ Complete | 15 | `results4arxiv/1a_baseline2d_*` |
| **Phase 1 Gate** | ✅ Locked | — | n_mels=64 |

### ⏳ Running Now (Phases 2–3)

| Phase | Purpose | Scripts | Runs | Time | Location |
|-------|---------|---------|------|------|----------|
| **Phase 2** | GAP vs Flatten | 2a, 2b, 2c | 9 | ~3h | `results4arxiv/2*_m64_s*` |
| **Phase 3** | Conv/Loss variants | 3a–3i | 9 | ~4h | `results4arxiv/3*_m64_s*` |

### ✅ Will Validate (Phase 6 — Complete)

| Model | Size | AUC | Recall | Location |
|-------|------|-----|--------|----------|
| **SEABADNet-Micro** | 6.26 KB | 0.9743 | ≥98.1% | `results4arxiv/6b_micro_final_*` |
| **SEABADNet-Edge** | 33.06 KB | 0.9992 | ≥99.0% | `results4arxiv/6c_edge_final_*` |

---

## 🔍 Phase 2: GAP vs Flatten Pooling

**Question:** Does Global Average Pooling reduce parameters without sacrificing AUC?

**Configuration (locked):**
- Baseline: CNN-Mel from Phase 1
- n_mels: 64 (Gate 1 winner)
- Conv: 4 filters
- n_fft: 1024
- Seeds: 42, 100, 786

**Scripts:**
1. `2a_baseline_gap.py` — Plain GAP (expected winner)
2. `2b_baseline_gap_learned.py` — GAP + learned pooling (marginal?)
3. `2c_baseline_gap_1x1.py` — GAP + 1×1 bottleneck (marginal?)

**Expected Finding:**
- Plain GAP reduces parameters significantly (Flatten → GAP often 10–50× smaller on final FC)
- AUC loss < 0.01 (acceptable)
- **Gate 2 Decision:** Lock GAP for all subsequent phases

**Paper Contribution:**
- Table: AUC vs parameter count (demonstrates efficiency tradeoff)
- Figure: Architecture comparison

---

## 🔬 Phase 3: Conv Type, Filters, Loss Function

**Question:** Which architectural variants best serve Micro and Edge branches?

**Configuration (locked):**
- Pooling: GAP (from Phase 2)
- n_mels: 64 (from Phase 1)
- n_fft: 1024
- Seeds: 42 only (main scripts); investigation scripts also 42

**Main Scripts (6):**
1. `3a_depthwise.py` — SeparableConv2D, 4 filters (Micro direction)
2. `3b_filters8.py` — Conv2D, 8 filters (Edge direction)
3. `3c_gap_focal_loss.py` — GAP + Focal Loss (α=0.25, γ=2.0)
4. `3d_gap_freq_emphasis.py` — + Frequency emphasis augmentation
5. `3e_gap_freq_emph_ds.py` — Freq emphasis + depthwise sep
6. `3f_gap_focal_loss_freq_emph_pointwise.py` — GAP + FL + FE + pointwise (Micro ceiling)

**Investigation Scripts (3, reference):**
- `3g_strided_focal_tuned.py` — Strided conv + focal loss
- `3h_strided_focal_no1x1.py` — Simpler strided arch
- `3i_strided_focal_depthwise.py` — Strided + depthwise

**Expected Findings:**
- **Gate 3A (Edge):** Conv2D-8f + GAP + Focal Loss (highest AUC for larger models)
- **Gate 3B (Micro):** Depthwise-4f + GAP + Focal Loss (best size/accuracy tradeoff)
- Focal loss improves recall without hurting AUC
- Frequency emphasis helps depthwise variants
- Strided convolutions: interesting but not winning path

**Paper Contribution:**
- Ablation table: Conv type, filters, loss impact on AUC
- Architecture diagrams: Micro vs Edge final designs
- Discussion: Why depthwise is critical for MCU deployment

---

## ✅ Phase 6: Final Validation (Already Complete)

**Results Available Now:**

### SEABADNet-Micro
```
Configuration: Depthwise-sep, 4 filters, n_mels=16, GAP, focal loss
Size: 6.26 KB INT8
AUC: 0.9743 ± 0.0011 (3 seeds)
Recall: ≥98.1% @ τ=0.35
Latency: ~0.10 ms (CPU), ~530 ms (AudioMoth est.)
Target: ARM Cortex-M4 (AudioMoth)
```

### SEABADNet-Edge
```
Configuration: Conv2D-3block, 8 filters, n_mels=80, GAP, focal loss, BN
Size: 33.06 KB INT8
AUC: 0.9992 ± 0.0002 (3 seeds)
Recall: ≥99.0% @ τ=0.60 (seed-dependent: 0.45–0.60)
Latency: ~1.15 ms (CPU), ~88 ms (RPi 4B est.)
Target: Linux SBC (Raspberry Pi 4/5)
```

**Paper Integration:**
- Final metrics table (AUC, recall, size, latency for both models)
- Comparison with baselines (TinyChirp, BirdNET)
- Deployment section (AudioMoth vs Raspberry Pi)

---

## 📈 How to Monitor Progress

### Check Status
```bash
# See last 50 lines of log
tail -50 phase2_phase3.log

# Watch in real-time
tail -f phase2_phase3.log

# Count completed runs
grep -c "✓" phase2_phase3.log   # Successful
grep -c "✗" phase2_phase3.log   # Failed

# Check GPU usage
watch -n 1 nvidia-smi
```

### When Complete
```bash
# Results will be in:
ls results4arxiv/2a_* results4arxiv/2b_* results4arxiv/2c_*  # Phase 2
ls results4arxiv/3*_m64_s42*                                 # Phase 3

# Each run produces:
# - results_summary.txt (AUC, accuracy, size, latency)
# - model_int8.tflite (quantized model)
# - Metrics plots (ROC, confusion matrix, etc.)
```

---

## 🎓 Paper Structure Using Option C

With Phase 2–3 + Phase 6 results, the paper narrative is:

**Section 5 — Architecture Design:**
1. Phase 1: Frequency resolution (n_mels sweep) → Gate 1: n_mels=64 (Edge baseline)
2. Phase 2: Pooling (GAP vs Flatten) → Gate 2: lock GAP
3. Phase 3: Loss function (Focal loss) → detect improvements for recall
4. Phase 3: Model variants (Depthwise vs Conv2D) → Gate 3A/3B: split Micro/Edge

**Section 6 — Final Models:**
1. Phase 6 results: SEABADNet-Micro (6.26 KB, 98% recall) and Edge (33.06 KB, 99% recall)
2. Comparison: vs TinyChirp baselines, vs BirdNET zero-shot
3. Deployment: AudioMoth latency estimates, Raspberry Pi measurements

**Supplementary:**
- Full ablation tables (Phases 1–6)
- Threshold analysis (per-seed operating points)
- Hardware benchmarks

---

## 🚨 If Something Goes Wrong

**Script errors:**
- Check `phase2_phase3.log` for tracebacks
- Re-run individual scripts manually:
  ```bash
  conda run -n tf215_gpu python develop/2a_baseline_gap.py --n_mels 64 --n_fft 1024 --random_seed 42
  ```

**GPU out of memory:**
- Check `nvidia-smi` for memory usage
- Reduce batch size in scripts (if available)
- Run scripts sequentially instead of parallel

**Incomplete runs:**
- Failed runs will be listed in log
- Re-run with same seed after fixing
- Results are cumulative (no need to re-run successful runs)

---

## 📋 Deliverables After Option C

✅ **Code commits:** All orchestration scripts, fixes  
✅ **Results:** 18 new experiments (Phases 2–3)  
✅ **Tables:** Phase 2/3 ablation summaries (auto-generated)  
✅ **Paper-ready:** Enough data to write results + discussion sections  
✅ **Reproducibility:** All configs logged in `config.txt` for each run  

**NOT included in Option C:**
- Phases 4–5 (dropout sweep, filter count) — supplement if time
- Full per-seed validation for Phases 2–3 — single seed sufficient for design decisions
- Rerun of Phase 6 — existing results are definitive

---

## Next Steps After This Completes

1. **Aggregate results:**
   ```bash
   # Auto-generated by orchestrators
   python develop/analyze_phase2_results.py
   python develop/analyze_phase3_results.py
   ```

2. **Write paper results section** with Phase 2–3 findings

3. **Decision:** Proceed to publication with Phase 6 results, or run Phases 4–5 for additional ablations?

---

**Started:** 2026-06-19 23:58 UTC  
**Expected end:** 2026-06-20 06:58 UTC  
**Log file:** `phase2_phase3.log` (updated in real-time)  
**Status updates:** Watch `grep "✅\|Gate" phase2_phase3.log`
