#!/usr/bin/env bash
# Phase B: diagnostics + full cross-corpus protocol + QAT + paper baselines.
#
# B0  in-domain sanity check per corpus. Phase A showed cross-corpus AUC at chance on
#     held-out birdvox for BOTH architectures; this separates "birdvox is unlearnable
#     with these features" (a pipeline bug) from "birdvox is unreachable from the other
#     two corpora" (genuine domain shift).
# B1  all three cross-corpus folds x 3 seeds for the winning config.
# B2  QAT versions of the same.
# B3  the 80-mel published baselines (sparrow / bulbul / drongonet-edge).
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
EPOCHS=40
STAGES="${STAGES:-4}"          # winning config from phase A, override on the CLI
SEEDS="${SEEDS:-42 100 786}"

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train.py --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$"
  echo
}

# ---- B0: in-domain sanity, one seed, both architectures on birdvox
for C in birdvox freefield1010 warblr; do
  run "B0_indomain_${C}_sparrow"  --arch sparrownet --stages "$STAGES" --single-corpus "$C" --seed 42
done
run "B0_indomain_birdvox_micro"   --arch drongonet_micro --single-corpus birdvox --seed 42

# ---- B1: full cross-corpus protocol, 3 folds x 3 seeds
for HO in birdvox freefield1010 warblr; do
  for S in $SEEDS; do
    run "B1_sparrow_st${STAGES}_ho-${HO}_s${S}" \
        --arch sparrownet --stages "$STAGES" --held-out "$HO" --seed "$S"
    run "B1_micro_ho-${HO}_s${S}" \
        --arch drongonet_micro --held-out "$HO" --seed "$S"
  done
done

# ---- B2: QAT (drongonet lesson: PTQ is the top recall killer)
for HO in birdvox freefield1010 warblr; do
  run "B2_sparrow_st${STAGES}_qat_ho-${HO}_s42" \
      --arch sparrownet --stages "$STAGES" --held-out "$HO" --seed 42 --qat
done

# ---- B3: published baselines at their native 80-mel resolution
for HO in birdvox freefield1010 warblr; do
  for A in sparrow bulbul drongonet_edge; do
    run "B3_${A}_m80_ho-${HO}_s42" \
        --arch "$A" --n-mels 80 --held-out "$HO" --seed 42 --batch-size 32
  done
done

echo "=== PHASE B COMPLETE ==="
