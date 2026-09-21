#!/usr/bin/env bash
# Phase F: models for the MyBAD (Malaysian) external test.
#
# Trained on ALL THREE DCASE corpora -- for an external test set, holding out a DCASE
# fold just discards training data. MyBAD is never seen: not for training, not for
# INT8 calibration, not for threshold selection.
#
# Both SparrowNet geometries are trained because MyBAD clips are 3s / 184 frames:
#   --mode crop   184 frames natively, geometry-matched to MyBAD
#   --mode full   622 frames (the deployed config); weights still apply at 184 frames
#                 but padding="same" distributes padding by input length, so the trunk
#                 differs slightly from training geometry. Running both shows the cost.
# drongonet-micro is natively 184 frames and needs no variant.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
EPOCHS=40
STAGES=3
WIDTH=8,8

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train.py --held-out all --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$"
  echo
}

for S in 42 100 786; do
  run "F1_sparrow_crop_all_s${S}" --arch sparrownet --stages $STAGES --width $WIDTH \
      --mode crop --seed "$S"
  run "F2_sparrow_full_all_s${S}" --arch sparrownet --stages $STAGES --width $WIDTH \
      --mode full --seed "$S"
  run "F3_micro_all_s${S}"        --arch drongonet_micro --seed "$S"
done

echo "=== PHASE F COMPLETE ==="
