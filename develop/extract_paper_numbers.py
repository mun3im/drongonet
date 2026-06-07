#!/usr/bin/env python3
"""
extract_paper_numbers.py
Reads results4arxiv/ for Nano, Micro, Edge (3 seeds each) and prints:
  - actual TFLite file sizes
  - tensor dtypes (float32 I/O vs int8 I/O)
  - AUC float32, AUC TFLite, latency, per seed
  - mean ± std across seeds
Outputs a paper-ready summary.
"""

import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path

RESULTS = Path("results4arxiv")

VARIANTS = {
    "Nano":  {"dirs": [f"6a_nano_final_fft512_m16_s{s}"  for s in [42, 100, 786]], "tflite": "model.tflite"},
    "Micro": {"dirs": [f"6b_micro_final_fft1024_m16_s{s}" for s in [42, 100, 786]], "tflite": "model.tflite"},
    "Edge":  {"dirs": [f"6c_edge_final_fft1024_m80_s{s}"  for s in [42, 100, 786]], "tflite": "model_int8.tflite"},
}

def parse_results_summary(path: Path) -> dict:
    """Pull key=value pairs and labelled sections from results_summary.txt."""
    d = {}
    if not path.exists():
        return d
    text = path.read_text()

    # Float32 AUC — two formats:
    # "Float32 Model:\n  AUC: X" and "Float32 Model AUC: X"
    m = re.search(r"Float32 Model.*?AUC:\s*([0-9.]+)", text, re.DOTALL)
    if m:
        d["auc_float32"] = float(m.group(1))

    # TFLite AUC
    m = re.search(r"TFLite[^\n]*Model[^\n]*:\s*\n.*?AUC:\s*([0-9.]+)", text, re.DOTALL)
    if m:
        d["auc_tflite"] = float(m.group(1))

    # Inference time
    m = re.search(r"Avg Inference Time:\s*([0-9.]+)\s*ms", text)
    if m:
        d["latency_ms"] = float(m.group(1))

    # Model size from summary (may differ slightly from file)
    m = re.search(r"Model Size:\s*([0-9.]+)\s*KB", text)
    if m:
        d["size_summary_kb"] = float(m.group(1))

    # Strategy name
    m = re.search(r"TFLite Model \(([^)]+)\)", text)
    if m:
        d["strategy"] = m.group(1)
    elif "TFLite int8 Model" in text:
        d["strategy"] = "INT8 (TFLITE_BUILTINS_INT8)"

    return d

def inspect_tflite(path: Path) -> dict:
    """Get actual file size and tensor dtypes from a TFLite model."""
    d = {}
    if not path.exists():
        return d
    d["file_size_kb"] = path.stat().st_size / 1024

    try:
        interp = tf.lite.Interpreter(str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        d["input_dtype"]  = inp["dtype"].__name__
        d["output_dtype"] = out["dtype"].__name__
    except Exception as e:
        d["error"] = str(e)
    return d

# ── collect ──────────────────────────────────────────────────────────────────
rows = {}
for name, cfg in VARIANTS.items():
    seeds = [42, 100, 786]
    data = []
    for seed, d in zip(seeds, cfg["dirs"]):
        rdir = RESULTS / d
        summary = parse_results_summary(rdir / "results_summary.txt")
        tflite  = inspect_tflite(rdir / cfg["tflite"])
        row = {"seed": seed, **summary, **tflite}
        data.append(row)
        fmt = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else str(v)
        print(f"  [{name} s{seed}] "
              f"AUC_f32={fmt(row.get('auc_float32','?'), '.4f')}  "
              f"AUC_tflite={fmt(row.get('auc_tflite','?'), '.4f')}  "
              f"size={fmt(row.get('file_size_kb','?'), '.2f')}KB  "
              f"lat={fmt(row.get('latency_ms','?'), '.2f')}ms  "
              f"I/O={row.get('input_dtype','?')}/{row.get('output_dtype','?')}  "
              f"strategy={row.get('strategy','?')}")
    rows[name] = data

# ── aggregate ─────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("PAPER-READY SUMMARY (mean ± std across seeds 42, 100, 786)")
print("="*72)

for name, data in rows.items():
    f32   = [r["auc_float32"]  for r in data if "auc_float32"  in r]
    tfl   = [r["auc_tflite"]   for r in data if "auc_tflite"   in r]
    lat   = [r["latency_ms"]   for r in data if "latency_ms"   in r]
    sizes = [r["file_size_kb"] for r in data if "file_size_kb" in r]
    strat = data[0].get("strategy", "?") if data else "?"
    io    = f'{data[0].get("input_dtype","?")} I/O' if data else "?"

    print(f"\n{name}  ({io})  quantization: {strat}")
    if f32:
        print(f"  AUC float32 : {np.mean(f32):.4f} ± {np.std(f32):.4f}  "
              f"(seeds: {[f'{v:.4f}' for v in f32]})")
    if tfl:
        print(f"  AUC TFLite  : {np.mean(tfl):.4f} ± {np.std(tfl):.4f}  "
              f"(seeds: {[f'{v:.4f}' for v in tfl]})")
    if f32 and tfl:
        deltas = [abs(a - b) for a, b in zip(f32, tfl)]
        print(f"  AUC degradation: {np.mean(deltas)*100:.4f}% ± {np.std(deltas)*100:.4f}%")
    if sizes:
        print(f"  File size (KB): {np.mean(sizes):.2f} ± {np.std(sizes):.4f}  "
              f"(seeds: {[f'{v:.2f}' for v in sizes]})")
    if lat:
        print(f"  Latency (ms): {np.mean(lat):.2f} ± {np.std(lat):.4f}")

print("\n" + "="*72)
print("QUANTIZATION NOTES")
print("="*72)
for name, data in rows.items():
    io = data[0].get("input_dtype", "?") if data else "?"
    strat = data[0].get("strategy", "?") if data else "?"
    full_int8 = (io == "int8")
    label = "Full INT8 (CMSIS-NN compatible)" if full_int8 else \
            "Dynamic range quantization (weights int8, I/O float32 — NOT full INT8)"
    print(f"  {name}: {label}")
    print(f"         Strategy logged: '{strat}'")
