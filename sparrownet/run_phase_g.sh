#!/usr/bin/env bash
# Phase G: train on MyBAD, and settle the frontend empirically.
#
# Objective: acceptable accuracy for MALAYSIAN bird detection. So MyBAD is the training
# and primary test set; DCASE is a secondary check only (and excludes freefield1010,
# whose negatives are inside MyBAD).
#
# The frontend grid tests two things we should not assume:
#   db vs log1p   - MyBAD ships log1p, which at its own signal levels (99.1% of power-mel
#                   values below 0.1) compresses almost nothing. dB relative to the clip
#                   peak should give faint calls proportional range. Predicted: dB wins.
#   1024 vs 512   - drongonet's LESSONS_LEARNT calls n_fft=1024 "non-negotiable" after a
#                   37% AUC collapse at 512, but that was measured at n_mels=64. With 16
#                   mel bands the STFT detail is averaged into wide bands anyway, so the
#                   penalty may be far smaller. MyBAD itself ships 512.
#
# Every split is grouped by source recording (train_mybad.py asserts zero overlap).
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
EPOCHS=40
SEEDS="${SEEDS:-42 100 786}"

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train_mybad.py --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$|Ignored |saved_model|MLIR|mlir"
  echo
}

# G1: SparrowNet frontend grid
for S in $SEEDS; do
  for FE in db log1p; do
    for FFT in 1024 512; do
      run "G1_sparrow_${FE}_fft${FFT}_s${S}" --arch sparrownet --stages 3 --width 8,8 \
          --frontend "$FE" --n-fft "$FFT" --seed "$S"
    done
  done
done

# G2: drongonet-micro, to check the frontend effect is not architecture-specific
for S in $SEEDS; do
  for FE in db log1p; do
    run "G2_micro_${FE}_fft1024_s${S}" --arch drongonet_micro \
        --frontend "$FE" --n-fft 1024 --seed "$S"
  done
done

echo "=== PHASE G COMPLETE ==="
