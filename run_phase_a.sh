#!/usr/bin/env bash
# Phase A: architecture selection on a single fold (held out birdvox, seed 42).
# Cheap enough to run the whole ablation, so the expensive 3-fold x 3-seed phase
# only runs the configuration that actually won. Sequential: one GPU.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
HO=birdvox
SEED=42
EPOCHS=40

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then
    echo "SKIP ${tag} (summary.json exists)"; return 0
  fi
  echo "=== RUN ${tag} ==="
  $PY train.py --held-out "$HO" --seed "$SEED" --epochs "$EPOCHS" --tag "$tag" "$@" \
    2>&1 | grep -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|fully_quantize|Summary on the non|^-*$|^$|UserWarning|warnings.warn"
  echo
}

# Baseline: is the poor ASEAN result a dataset problem or an architecture problem?
run "A0_micro_baseline"        --arch drongonet_micro

# SparrowNet receptive-field sweep. 4->31f(0.5s) 5->63f(1.0s) 6->127f(2.0s) 7->255f(4.1s)
# brackets the paper's 103 frames (1.5s) and drongonet's 184-frame (3s) window.
for ST in 4 5 6 7; do
  run "A1_sparrow_st${ST}"     --arch sparrownet --stages "$ST" --pool max
done

# Does global MAX actually beat global AVG, as the paper's Fig. 4 claims?
run "A2_sparrow_st6_avg"       --arch sparrownet --stages 6 --pool avg

# Is the cyclic-shift augmentation earning its place?
run "A3_sparrow_st6_noaug"     --arch sparrownet --stages 6 --pool max --no-augment

echo "=== PHASE A COMPLETE ==="
