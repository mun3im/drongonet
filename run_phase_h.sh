#!/usr/bin/env bash
# Phase H: how many positive recordings does SparrowNet actually need?
#
# Answers "how many samples do I need to keep/regenerate" with a learning curve over
# positive SOURCE RECORDINGS (not clips), plus a test of whether the 2nd-choice
# (weaker-energy) segment per recording earns its place.
#
# Training subsets only. Val and test keep every clip, so all points are comparable.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
run() {
  local tag="$1"; shift
  [ -f "results/${tag}/summary.json" ] && { echo "SKIP ${tag}"; return 0; }
  echo "=== RUN ${tag} ==="
  $PY train_mybad.py --epochs 40 --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -E "subset:|MyBAD:|float32:|INT8:|INT8 size|wrote"
  echo
}
for S in 42 100; do
  for F in 0.125 0.25 0.50 1.0; do
    run "H1_frac${F}_s${S}" --frontend db --n-fft 512 --pos-source-frac "$F" --seed "$S"
  done
  # does the weaker 2nd segment per recording add anything?
  run "H2_seg1only_s${S}" --frontend db --n-fft 512 --segments-per-source 1 --seed "$S"
done
echo "=== PHASE H COMPLETE ==="
