#!/usr/bin/env bash
# Phase C: the real architecture selection, with batch norm throughout.
#
# Phase A ran without batch norm and every stages>=6 model collapsed to a constant
# output (AUC exactly 0.5000) -- the signal vanishes through six undamped depthwise
# stages, so the receptive-field question was never actually answered past stages=5.
# BN is also what the paper does for this architecture ("In sparrow, we also apply
# batch normalization to all layers"), and TFLite folds it into the preceding conv,
# so it is close to free at inference.
#
# Phase A's numbers are therefore NOT comparable to these and are kept only as the
# evidence for the collapse. This phase is one grid so nothing depends on a
# mid-sweep judgement call: receptive field x width, then two ablations.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
HO=birdvox
SEED=42
EPOCHS=40

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train.py --held-out "$HO" --seed "$SEED" --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$"
  echo
}

# Receptive field x width grid. Width trims target the 10 KB soft budget; the stem
# stays at 8 channels in all of these because drongonet's ablations found narrowing
# the EARLY layer is the one change that reliably breaks recall.
for ST in 4 5 6 7; do
  for W in "8,16" "8,12" "8,8"; do
    run "C1_sparrow_st${ST}_w${W/,/-}" --arch sparrownet --stages "$ST" --width "$W"
  done
done

# Ablations at a mid receptive field, to test the two claims we inherited rather than
# assume them: global-max over global-avg (paper Fig. 4), and cyclic-shift augmentation.
run "C2_sparrow_st5_w8-12_avg"   --arch sparrownet --stages 5 --width 8,12 --pool avg
run "C2_sparrow_st5_w8-12_noaug" --arch sparrownet --stages 5 --width 8,12 --no-augment

echo "=== PHASE C COMPLETE ==="
