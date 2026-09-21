#!/usr/bin/env bash
# Phase D: the reportable protocol. Run after phase C picks STAGES/WIDTH.
#
#   STAGES=5 WIDTH=8,12 bash run_phase_d.sh
#
# D0  in-domain sanity per corpus: if a held-out corpus scores far below its own
#     in-domain ceiling, that gap is domain shift rather than an unlearnable target.
# D1  drongonet-micro CONTROL with batch norm. Batch norm alone was worth ~0.13
#     cross-corpus AUC on SparrowNet, so the headline comparison has to hold it
#     constant or the topology gets credit that normalization earned.
# D2  full cross-corpus protocol: 3 held-out folds x 3 seeds, SparrowNet + faithful micro.
# D3  QAT, since PTQ is drongonet's documented top recall killer.
# D4  published baselines at their native 80-mel resolution.
set -u
PY=/home/muneim/miniconda3/envs/tf215_gpu/bin/python
EPOCHS=40
STAGES="${STAGES:-5}"
WIDTH="${WIDTH:-8,12}"
SEEDS="${SEEDS:-42 100 786}"
WTAG="${WIDTH/,/-}"

run() {
  local tag="$1"; shift
  if [ -f "results/${tag}/summary.json" ]; then echo "SKIP ${tag}"; return 0; fi
  echo "=== RUN ${tag} ==="
  $PY train.py --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "oneDNN|cuDNN|cuFFT|cuBLAS|TF-TRT|NUMA|external/local_xla|computation placer|SSE4|rebuild TensorFlow|subprocess.cc|absl::InitializeLog|device_compiler|XNNPACK|Non-Converted Ops|Accepted dialects|arith.constant|f32:|i32:|fully_quantize|Summary on the non|^ *$|UserWarning|warnings.warn|^-+$"
  echo
}

SP="--arch sparrownet --stages $STAGES --width $WIDTH"

# ---- D0: in-domain ceilings
for C in birdvox freefield1010 warblr; do
  run "D0_indomain_${C}" $SP --single-corpus "$C" --seed 42
done

# ---- D1: BN control for the baseline
for HO in birdvox freefield1010 warblr; do
  run "D1_micro_bn_ho-${HO}_s42" --arch drongonet_micro --bn --held-out "$HO" --seed 42
done

# ---- D2: full protocol
for HO in birdvox freefield1010 warblr; do
  for S in $SEEDS; do
    run "D2_sparrow_st${STAGES}_w${WTAG}_ho-${HO}_s${S}" $SP --held-out "$HO" --seed "$S"
    run "D2_micro_ho-${HO}_s${S}" --arch drongonet_micro --held-out "$HO" --seed "$S"
  done
done

# ---- D3: QAT
for HO in birdvox freefield1010 warblr; do
  run "D3_sparrow_st${STAGES}_w${WTAG}_qat_ho-${HO}_s42" $SP --held-out "$HO" --seed 42 --qat
done

# ---- D4: published baselines (80 mel, their native resolution)
for HO in birdvox freefield1010 warblr; do
  for A in sparrow bulbul drongonet_edge; do
    run "D4_${A}_m80_ho-${HO}_s42" --arch "$A" --n-mels 80 --held-out "$HO" \
        --seed 42 --batch-size 32
  done
done

echo "=== PHASE D COMPLETE ==="
