#!/usr/bin/env bash
# Phase A2: get under the 10 KB soft target.
#
# A1_sparrow_st4 landed at 10.91 KB -- inside the 16 KB hard ceiling but over the
# 10 KB soft target. Only ~1.2k of those bytes are weights; the rest is per-layer
# FlatBuffer overhead (quantization metadata, op descriptors), which is why a 1219-param
# model costs 10.9 KB while 919-param drongonet-micro costs 6.0 KB. So the lever is
# layer count and width, not parameter count alone.
#
# drongonet's lesson list says reducing EARLY layer width is the one change that
# reliably breaks recall, so the stem stays at 8 and only the pointwise width moves --
# except the last arm, which tests that rule directly rather than assuming it.
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

run "A2_sparrow_st4_w8-12"  --arch sparrownet --stages 4 --width 8,12
run "A2_sparrow_st4_w8-8"   --arch sparrownet --stages 4 --width 8,8
run "A2_sparrow_st3_w8-16"  --arch sparrownet --stages 3 --width 8,16
run "A2_sparrow_st4_w6-12"  --arch sparrownet --stages 4 --width 6,12

echo "=== PHASE A2 COMPLETE ==="
