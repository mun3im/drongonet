"""
Read results_summary.txt files directly from experiment output directories
for experiments where collect_all_results.py returned NaN.
"""
import os
from pathlib import Path

RESULTS_ROOT = Path('results')

TARGETS = [
    '6b_micro_improved_fft1024_m16_s42',
    '6b_micro_improved_fft1024_m32_s42',
    '6b_micro_improved_fft1024_m16_s100',
    '6b_micro_improved_fft1024_m16_s786',
    '6b_micro_improved_fft1024_m32_s100',
    '6b_micro_improved_fft1024_m32_s786',
    '6a_micro_final_fft1024_m16_s42',
    '6a_micro_final_fft1024_m32_s42',
    '3f_gap_focal_loss_freq_emph_pointwise_fft1024_m80_s42',
    '3f_gap_focal_loss_freq_emph_pointwise_fft1024_m80_s100',
    '3f_gap_focal_loss_freq_emph_pointwise_fft1024_m80_s786',
    '6b_edge_final_fft1024_m80_s42',
]


def find_dir(name):
    """Try exact match first, then partial match."""
    exact = RESULTS_ROOT / name
    if exact.exists():
        return exact
    for d in RESULTS_ROOT.iterdir():
        if d.is_dir() and name in d.name:
            return d
    return None


def read_summary(exp_dir):
    for fname in ['results_summary.txt', 'summary.txt', 'metrics.txt']:
        f = exp_dir / fname
        if f.exists():
            return f.read_text()
    return None


def find_any_txt(exp_dir):
    txts = list(exp_dir.glob('*.txt'))
    if txts:
        return txts[0].name, txts[0].read_text()
    return None, None


if not RESULTS_ROOT.exists():
    print(f"ERROR: '{RESULTS_ROOT}' directory not found.")
    print("Contents of current directory:")
    for p in Path('.').iterdir():
        print(f"  {p}")
    exit(1)

print(f"Results root: {RESULTS_ROOT.resolve()}")
print(f"Subdirectories found: {sum(1 for d in RESULTS_ROOT.iterdir() if d.is_dir())}")
print()

for name in TARGETS:
    exp_dir = find_dir(name)
    print(f"{'='*60}")
    print(f"EXPERIMENT: {name}")
    if exp_dir is None:
        print(f"  [NOT FOUND] No directory matching '{name}'")
        # Show partial matches
        matches = [d.name for d in RESULTS_ROOT.iterdir()
                   if d.is_dir() and any(part in d.name for part in name.split('_')[:3])]
        if matches:
            print(f"  Partial matches: {matches[:5]}")
        continue

    print(f"  Dir: {exp_dir}")
    summary = read_summary(exp_dir)
    if summary:
        print(f"  results_summary.txt:")
        for line in summary.strip().splitlines():
            print(f"    {line}")
    else:
        fname, content = find_any_txt(exp_dir)
        if fname:
            print(f"  No results_summary.txt — found {fname}:")
            for line in (content or '').strip().splitlines()[:20]:
                print(f"    {line}")
        else:
            print(f"  No .txt files found. Contents:")
            for f in list(exp_dir.iterdir())[:10]:
                print(f"    {f.name}")
    print()
