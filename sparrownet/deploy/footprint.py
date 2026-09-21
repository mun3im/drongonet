"""
footprint.py — Cortex-M4 deployment footprint for a converted INT8 model.

Reports, from the .tflite FlatBuffer itself (not from the Keras model, so what is
measured is what would actually ship):
  * flash    the model blob size, against the 10 KB soft / 16 KB hard budget
  * RAM      tensor-arena estimates (peak-op proxy and an all-live upper bound)
  * MACs     multiply-accumulates per inference, per op and in total
  * latency  ESTIMATED from MACs and a CMSIS-NN throughput assumption

IMPORTANT: the latency figure is an *estimate*, not a measurement. Nothing here runs
on an AudioMoth; verifying it needs the board (drongonet's edge_deploy/ sketches are
the harness for that). The estimate is stated with its assumptions so it can be
checked rather than trusted.

Usage:
    python deploy/footprint.py results/<tag>/sparrownet_int8.tflite
    python deploy/footprint.py results/*/*.tflite --clock 48
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from config import SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES

# CALIBRATED against a real measurement, not assumed. argus/WIO_TERMINAL_DRONGONET_
# LATENCY.md reports drongonet-micro at 97.47 ms inference on a Wio Terminal
# (ATSAMD51P19A, Cortex-M4F @ 120 MHz, N=100, reproduced across three flashes).
# That model is 741,912 MACs, so:
#     741,912 MACs / (0.09747 s * 120e6 Hz) = 0.0634 MACs/cycle
# i.e. TFLite Micro INT8 on this part sustains ~1/16th of the naive 1 MAC/cycle
# figure once interpreter dispatch, im2col and memory stalls are counted. The old
# 1.0 default under-predicted that measurement by 15x.
#
# Caveat: one data point from one op mix. Depthwise convolutions generally achieve
# lower MAC efficiency than dense ones, so a depthwise-heavy graph like SparrowNet
# may run slower per MAC than this. Treat the output as an order-of-magnitude guide
# and measure on hardware with the existing bench sketch.
MACS_PER_CYCLE_M4 = 0.0634
AUDIOMOTH_CLOCK_MHZ = 48.0
WIO_TERMINAL_CLOCK_MHZ = 120.0


def _dtype_bytes(dtype):
    return int(np.dtype(dtype).itemsize)


def analyze_tflite(path, clock_mhz=AUDIOMOTH_CLOCK_MHZ, macs_per_cycle=MACS_PER_CYCLE_M4):
    size = os.path.getsize(path)
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()

    tensors = {t["index"]: t for t in interp.get_tensor_details()}

    # Constant tensors have data; variable (activation) tensors are what the arena holds.
    const_idx = set()
    for idx, t in tensors.items():
        try:
            interp.get_tensor(idx)
            const_idx.add(idx)
        except (ValueError, RuntimeError):
            pass

    def nbytes(idx):
        t = tensors[idx]
        shape = t["shape"]
        if len(shape) == 0:
            return _dtype_bytes(t["dtype"])
        return int(np.prod(shape)) * _dtype_bytes(t["dtype"])

    ops = interp._get_ops_details()
    per_op, total_macs, peak_op_bytes = [], 0, 0

    for op in ops:
        ins = [i for i in op["inputs"] if i >= 0]
        outs = [o for o in op["outputs"] if o >= 0]
        name = op["op_name"]

        macs = 0
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(ins) >= 2:
            w = tensors[ins[1]]["shape"]          # (out_c, kh, kw, in_c) / (1, kh, kw, ch)
            o = tensors[outs[0]]["shape"]         # (n, h, w, c)
            if len(w) == 4 and len(o) == 4:
                out_positions = int(o[1]) * int(o[2])
                if name == "CONV_2D":
                    macs = out_positions * int(w[0]) * int(w[1]) * int(w[2]) * int(w[3])
                else:
                    macs = out_positions * int(w[1]) * int(w[2]) * int(w[3])
        elif name == "FULLY_CONNECTED" and len(ins) >= 2:
            w = tensors[ins[1]]["shape"]
            if len(w) == 2:
                macs = int(w[0]) * int(w[1])

        live = sum(nbytes(i) for i in ins if i not in const_idx) + \
               sum(nbytes(o) for o in outs)
        peak_op_bytes = max(peak_op_bytes, live)
        total_macs += macs
        per_op.append({"op": name, "macs": int(macs), "live_bytes": int(live)})

    arena_upper = sum(nbytes(i) for i in tensors if i not in const_idx)
    cycles = total_macs / macs_per_cycle
    latency_ms = cycles / (clock_mhz * 1e6) * 1000.0

    within_soft = size <= SIZE_SOFT_LIMIT_BYTES
    within_hard = size <= SIZE_HARD_LIMIT_BYTES

    return {
        "path": path,
        "flash_bytes": size,
        "flash_kb": round(size / 1024, 2),
        "within_soft_limit": within_soft,
        "within_hard_limit": within_hard,
        "verdict": "OK" if within_soft else ("OVER_SOFT" if within_hard else "OVER_HARD"),
        "input_shape": [int(v) for v in interp.get_input_details()[0]["shape"]],
        "total_macs": int(total_macs),
        "arena_peak_op_kb": round(peak_op_bytes / 1024, 2),
        "arena_all_live_kb": round(arena_upper / 1024, 2),
        "n_ops": len(ops),
        "estimated_latency_ms": round(latency_ms, 3),
        "latency_assumptions": {
            "clock_mhz": clock_mhz, "macs_per_cycle": macs_per_cycle,
            "note": "ESTIMATE from MAC count; not measured on hardware",
        },
        "per_op": per_op,
    }


def fmt(r):
    lines = [
        f"{os.path.basename(r['path'])}",
        f"  input            {r['input_shape']}",
        f"  flash            {r['flash_kb']:.2f} KB  -> {r['verdict']} "
        f"(soft {SIZE_SOFT_LIMIT_BYTES/1024:.0f} / hard {SIZE_HARD_LIMIT_BYTES/1024:.0f} KB)",
        f"  MACs             {r['total_macs']:,}  ({r['n_ops']} ops)",
        f"  RAM peak-op      {r['arena_peak_op_kb']:.2f} KB   (proxy for tensor arena)",
        f"  RAM all-live     {r['arena_all_live_kb']:.2f} KB   (upper bound)",
        f"  latency EST      {r['estimated_latency_ms']:.2f} ms @ "
        f"{r['latency_assumptions']['clock_mhz']:.0f} MHz, "
        f"{r['latency_assumptions']['macs_per_cycle']} MAC/cycle  [NOT measured]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--clock", type=float, default=AUDIOMOTH_CLOCK_MHZ)
    ap.add_argument("--macs-per-cycle", type=float, default=MACS_PER_CYCLE_M4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = []
    for path in args.models:
        r = analyze_tflite(path, args.clock, args.macs_per_cycle)
        out.append(r)
        if not args.json:
            print(fmt(r))
            print()
    if args.json:
        print(json.dumps(out, indent=2))
