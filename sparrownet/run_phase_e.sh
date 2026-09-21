#!/usr/bin/env bash
# Phase E: what does the 10 KB soft target actually cost?
#
# The corrected size probe (size_probe.py, now reproducing trained sizes exactly)
# says that WITH batch norm -- which is not optional, deep stacks collapse without it
# -- only stages<=3 fits under 10 KB:
#
#     st3_w6-8_bn   9.39 KB   531 params   RF 15f (0.24s)
#     st3_w8-8_bn   9.67 KB   603 params   RF 15f (0.24s)
#     st4_w8-8_bn  10.78 KB   795 params   RF 31f (0.50s)   over soft
#     st5_w8-12_bn 13.98 KB  1539 params   RF 63f (1.01s)   over soft, best xcorp so far
#
# But phase C suggested cross-corpus transfer *wants* a longer receptive field. So the
# soft budget and generalization pull in opposite directions, and the question is what
# the 10 KB constraint costs in AUC.
#
# Phase C ran one seed per config and its cross-corpus numbers swung 0.47-0.70 -- far
# too noisy to choose on (a width change of 16->12 at stages=5 moved it 0.22). So this
# phase runs three seeds per candidate and compares means.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
HO=birdvox
EPOCHS=40

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train.py --held-out "$HO" --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$"
  echo
}

for S in 42 100 786; do
  run "E1_sparrow_st3_w8-8_s${S}"  --arch sparrownet --stages 3 --width 8,8  --seed "$S"
  run "E1_sparrow_st4_w8-8_s${S}"  --arch sparrownet --stages 4 --width 8,8  --seed "$S"
  run "E1_sparrow_st5_w8-12_s${S}" --arch sparrownet --stages 5 --width 8,12 --seed "$S"
done

echo "=== PHASE E COMPLETE ==="
