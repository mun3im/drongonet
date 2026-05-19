#!/usr/bin/env python3
"""
update_status.py — Periodically rewrites the '## Experiment Status' section
of CLAUDE.md to reflect which runs have completed results_summary.txt.
"""

import re
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/muneim/Dropbox/Conda/SEABADNet/results")
CLAUDE_MD   = Path("/home/muneim/Dropbox/Conda/SEABADNet/CLAUDE.md")

N_MELS   = [16, 32, 48, 64, 80]
SEED     = 42

# ── All tracked runs ──────────────────────────────────────────────────────────

PHASE1 = [("1a_baseline2d", 1024, m) for m in N_MELS]

PHASE2 = [
    (s, 1024, m)
    for s in ["2a_baseline_gap", "2b_baseline_gap_learned", "2c_baseline_gap_1x1"]
    for m in N_MELS
]

PHASE3 = [
    (s, 1024, m)
    for s in [
        "3a_depthwise", "3b_filters8", "3c_gap_focal_loss",
        "3d_gap_freq_emphasis", "3e_gap_freq_emph_ds",
        "3f_gap_focal_loss_freq_emph_pointwise",
        "3g_strided_focal_tuned", "3h_strided_focal_no1x1", "3i_strided_focal_depthwise",
    ]
    for m in N_MELS
]

PHASE4 = [
    (s, 1024, m)
    for s in [
        "4a_dropout01", "4b_dropout02", "4c_dropout03", "4d_dropout04",
        "4e_depthwise_drop01", "4f_depthwise_drop02",
        "4g_depthwise_drop03", "4h_depthwise_drop04",
    ]
    for m in N_MELS
]

PHASE5 = (
    [(s, 1024, m) for s in ["5a_depthwise_f6"] for m in N_MELS]
    + [(s, 512, m) for s in ["5b_depthwise_f5"] for m in N_MELS]
)

PHASE6 = [
    (s, 1024, m)
    for s in ["6a_micro_final", "6b_edge_final"]
    for m in N_MELS
]

PHASE6B = [
    ("6b_micro_improved", 1024, m)
    for m in N_MELS
]

SEEDS = [42, 100, 786]

FINAL_6B = [
    ("6b_micro_improved", 1024, m, s)
    for m in N_MELS for s in SEEDS
]

FINAL_3F = [
    ("3f_gap_focal_loss_freq_emph_pointwise", 1024, m, s)
    for m in N_MELS for s in SEEDS
]

ALL_RUNS = PHASE1 + PHASE2 + PHASE3 + PHASE4 + PHASE5 + PHASE6 + PHASE6B

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_done(script: str, n_fft: int, n_mels: int) -> bool:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{SEED}"
    return (d / "results_summary.txt").exists()


def auc_for(script: str, n_fft: int, n_mels: int) -> str:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{SEED}"
    p = d / "results_summary.txt"
    if not p.exists():
        return "—"
    text = p.read_text()
    for line in text.splitlines():
        m = re.search(r"AUC[:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return "?"


def size_for(script: str, n_fft: int, n_mels: int) -> str:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{SEED}"
    p = d / "results_summary.txt"
    if not p.exists():
        return "—"
    text = p.read_text()
    for line in text.splitlines():
        m = re.search(r"(?:Model Size|Size)[:\s]+([0-9.]+)\s*KB", line, re.IGNORECASE)
        if m:
            return f"{m.group(1)} KB"
    return "—"


def phase_table(runs: list[tuple]) -> str:
    # Group by script
    by_script: dict[str, list] = {}
    for script, n_fft, n_mels in runs:
        by_script.setdefault(script, []).append((n_fft, n_mels))

    header = "| Script | n_fft | " + " | ".join(f"m{m}" for m in N_MELS) + " |"
    sep    = "|--------|-------|" + "|".join(["-------"] * len(N_MELS)) + "|"
    rows   = [header, sep]

    for script, entries in by_script.items():
        n_fft = entries[0][0]
        cells = []
        for m in N_MELS:
            if is_done(script, n_fft, m):
                auc = auc_for(script, n_fft, m)
                cells.append(f"✅ {auc}")
            else:
                cells.append("⏳")
        rows.append(f"| `{script}` | {n_fft} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def is_done_seed(script: str, n_fft: int, n_mels: int, seed: int) -> bool:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
    return (d / "results_summary.txt").exists()


def auc_for_seed(script: str, n_fft: int, n_mels: int, seed: int) -> str:
    d = RESULTS_DIR / f"{script}_fft{n_fft}_m{n_mels}_s{seed}"
    p = d / "results_summary.txt"
    if not p.exists():
        return "—"
    for line in p.read_text().splitlines():
        m = re.search(r"AUC[:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return "?"


def final_table(runs: list[tuple]) -> str:
    """Table for multi-seed runs: rows = (script, n_mels), cols = seeds."""
    by_key: dict[tuple, int] = {}
    for script, n_fft, n_mels, seed in runs:
        by_key[(script, n_fft, n_mels)] = n_fft  # collect unique (script, mels)

    seeds_used = sorted({s for _, _, _, s in runs})
    header = "| Script | n_mels | " + " | ".join(f"s{s}" for s in seeds_used) + " |"
    sep    = "|--------|--------|" + "|".join(["--------"] * len(seeds_used)) + "|"
    rows   = [header, sep]

    seen = set()
    for script, n_fft, n_mels, _ in runs:
        key = (script, n_fft, n_mels)
        if key in seen:
            continue
        seen.add(key)
        cells = []
        for s in seeds_used:
            if is_done_seed(script, n_fft, n_mels, s):
                auc = auc_for_seed(script, n_fft, n_mels, s)
                cells.append(f"✅ {auc}")
            else:
                cells.append("⏳")
        rows.append(f"| `{script}` | m{n_mels} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def count_done(runs):
    if runs and len(runs[0]) == 4:
        return sum(1 for s, f, m, sd in runs if is_done_seed(s, f, m, sd))
    return sum(1 for s, f, m in runs if is_done(s, f, m))


def build_status_section() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(ALL_RUNS)
    done  = count_done(ALL_RUNS)
    pct   = 100 * done // total

    lines = [
        "## Experiment Status",
        "",
        f"_Last updated: {now} — {done}/{total} runs complete ({pct}%)_",
        "",
    ]

    # Overall sweep progress bar
    filled = done * 40 // total
    bar = "█" * filled + "░" * (40 - filled)
    lines += [f"`[{bar}]` {done}/{total}", ""]

    for phase_name, runs in [
        ("Phase 1 — n_mels sweep (baseline)",    PHASE1),
        ("Phase 2 — GAP variants",               PHASE2),
        ("Phase 3 — Conv type / loss",           PHASE3),
        ("Phase 4 — Dropout sweep",              PHASE4),
        ("Phase 5 — Filter count (Micro)",       PHASE5),
        ("Phase 6 — Final candidates",           PHASE6),
        ("Phase 6b — Micro improved (+ pointwise conv)", PHASE6B),
    ]:
        pd = count_done(runs)
        pt = len(runs)
        status = "✅ complete" if pd == pt else f"⏳ {pd}/{pt}"
        lines += [f"### {phase_name} — {status}", ""]
        lines += [phase_table(runs), ""]

    # Final multi-seed validation (separate seed-aware tables)
    for phase_name, runs in [
        ("Final validation — 6b Micro (seeds 42/100/786)", FINAL_6B),
        ("Final validation — 3f Edge (seeds 42/100/786)",  FINAL_3F),
    ]:
        pd = count_done(runs)
        pt = len(runs)
        status = "✅ complete" if pd == pt else f"⏳ {pd}/{pt}"
        lines += [f"### {phase_name} — {status}", ""]
        lines += [final_table(runs), ""]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    text = CLAUDE_MD.read_text()
    new_section = build_status_section()

    # Replace existing status section, or insert before first ## heading
    pattern = r"## Experiment Status\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, re.DOTALL):
        updated = re.sub(pattern, new_section, text, flags=re.DOTALL)
    else:
        # Insert at top, right after the title line
        updated = re.sub(r"(# SEABADNet[^\n]*\n)", r"\1\n" + new_section + "\n\n", text, count=1)

    CLAUDE_MD.write_text(updated)
    done  = count_done(ALL_RUNS)
    total = len(ALL_RUNS)
    print(f"[{datetime.now():%H:%M}] CLAUDE.md updated — {done}/{total} done")


if __name__ == "__main__":
    main()
