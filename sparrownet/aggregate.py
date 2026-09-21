"""
aggregate.py — collect results/*/summary.json into one table.

Usage:
    python aggregate.py                  # everything, as a text table
    python aggregate.py --prefix A       # only phase-A tags
    python aggregate.py --group          # mean +/- std across seeds of the same config
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np

from config import RESULTS_BASE, SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES


def load_all(prefix=None):
    rows = []
    if not os.path.isdir(RESULTS_BASE):
        return rows
    for tag in sorted(os.listdir(RESULTS_BASE)):
        if prefix and not tag.startswith(prefix):
            continue
        path = os.path.join(RESULTS_BASE, tag, "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    return rows


def fmt_table(rows):
    if not rows:
        return "(no results yet)"
    hdr = (f"{'tag':34s} {'params':>7s} {'RF':>7s} {'val':>7s} {'xcorp':>7s} "
           f"{'i8 val':>7s} {'i8 xcorp':>8s} {'KB':>6s} {'verdict':>9s}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        rf = f"{r['receptive_field_frames']}f" if r.get("receptive_field_frames") else "-"
        lines.append(
            f"{r['tag']:34s} {r['params']:7d} {rf:>7s} "
            f"{r['float32']['val_auc']:7.4f} {r['float32']['test_auc']:7.4f} "
            f"{r['int8']['val_auc']:7.4f} {r['int8']['test_auc']:8.4f} "
            f"{r['int8']['size_kb']:6.2f} {r['int8']['verdict']:>9s}")
    return "\n".join(lines)


def group_by_config(rows):
    """Group seeds of the same configuration; report mean +/- std."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["arch"], r["mode"], r["held_out"], r.get("stages"), r.get("pool"),
               r.get("head_kernel"), r.get("augment"))
        groups[key].append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arch, mode, ho, stages, pool, hk, aug = key
        name = (f"{arch}"
                + (f"_st{stages}" if stages else "")
                + (f"_{pool}" if pool else "")
                + (f"_hk{hk}" if hk and hk > 1 else "")
                + ("" if aug else "_noaug")
                + f"_ho-{ho}")
        out.append({
            "config": name, "n_seeds": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "params": rs[0]["params"],
            "size_kb": rs[0]["int8"]["size_kb"],
            "verdict": rs[0]["int8"]["verdict"],
            "val_auc_mean": float(np.mean([r["float32"]["val_auc"] for r in rs])),
            "val_auc_std": float(np.std([r["float32"]["val_auc"] for r in rs])),
            "test_auc_mean": float(np.mean([r["float32"]["test_auc"] for r in rs])),
            "test_auc_std": float(np.std([r["float32"]["test_auc"] for r in rs])),
            "int8_test_auc_mean": float(np.mean([r["int8"]["test_auc"] for r in rs])),
            "int8_test_auc_std": float(np.std([r["int8"]["test_auc"] for r in rs])),
        })
    return out


def fmt_groups(groups):
    if not groups:
        return "(no results yet)"
    hdr = (f"{'config':44s} {'n':>2s} {'params':>7s} {'KB':>6s} "
           f"{'val AUC':>15s} {'xcorp AUC':>15s} {'i8 xcorp':>15s}")
    lines = [hdr, "-" * len(hdr)]
    for g in groups:
        lines.append(
            f"{g['config']:44s} {g['n_seeds']:2d} {g['params']:7d} {g['size_kb']:6.2f} "
            f"{g['val_auc_mean']:.4f}+-{g['val_auc_std']:.4f} "
            f"{g['test_auc_mean']:.4f}+-{g['test_auc_std']:.4f} "
            f"{g['int8_test_auc_mean']:.4f}+-{g['int8_test_auc_std']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = load_all(args.prefix)
    if args.group:
        groups = group_by_config(rows)
        print(json.dumps(groups, indent=2) if args.json else fmt_groups(groups))
    else:
        print(json.dumps(rows, indent=2) if args.json else fmt_table(rows))
        print(f"\nbudget: soft {SIZE_SOFT_LIMIT_BYTES/1024:.0f} KB / "
              f"hard {SIZE_HARD_LIMIT_BYTES/1024:.0f} KB | {len(rows)} run(s)")
