#!/usr/bin/env python3
"""
Fix output directory naming to match rerun script expectations
"""

import re
from pathlib import Path

# Script name -> (pattern to find, replacement)
FIXES = {
    '6_best_accuracy.py': (
        r"config\.output_dir = f'results/6_best_fft",
        r"config.output_dir = f'results/6_best_accuracy_fft"
    ),
    '7_hybrid.py': (
        r"config\.output_dir = f'results/7_hybrid_m",
        r"config.output_dir = f'results/7_hybrid_fft{config.n_fft}_m"
    ),
    '8a_dropout01.py': (
        r"config\.output_dir = f'results/8a_dropout01_m",
        r"config.output_dir = f'results/8a_dropout01_fft{config.n_fft}_m"
    ),
    '8b_dropout02.py': (
        r"config\.output_dir = f'results/8b_dropout02_m",
        r"config.output_dir = f'results/8b_dropout02_fft{config.n_fft}_m"
    ),
    '8c_dropout03.py': (
        r"config\.output_dir = f'results/8c_dropout03_m",
        r"config.output_dir = f'results/8c_dropout03_fft{config.n_fft}_m"
    ),
    '8d_dropout04.py': (
        r"config\.output_dir = f'results/8d_dropout04_m",
        r"config.output_dir = f'results/8d_dropout04_fft{config.n_fft}_m"
    ),
    '9a_depthwise_drop01.py': (
        r"config\.output_dir = f'results/9a_depthwise_drop01_m",
        r"config.output_dir = f'results/9a_depthwise_drop01_fft{config.n_fft}_m"
    ),
    '9b_depthwise_drop02.py': (
        r"config\.output_dir = f'results/9b_depthwise_drop02_m",
        r"config.output_dir = f'results/9b_depthwise_drop02_fft{config.n_fft}_m"
    ),
    '9c_depthwise_drop03.py': (
        r"config\.output_dir = f'results/9c_depthwise_drop03_m",
        r"config.output_dir = f'results/9c_depthwise_drop03_fft{config.n_fft}_m"
    ),
    '9d_depthwise_drop04.py': (
        r"config\.output_dir = f'results/9d_depthwise_drop04_m",
        r"config.output_dir = f'results/9d_depthwise_drop04_fft{config.n_fft}_m"
    ),
}

def fix_output_dir(filepath: Path, pattern: str, replacement: str):
    """Fix the output_dir naming"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already fixed
    if replacement in content:
        return False, "already fixed"

    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        return False, "pattern not found"

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    return True, "fixed output directory naming"

def main():
    print("="*80)
    print("FIXING OUTPUT DIRECTORY NAMING")
    print("="*80)
    print()

    updated = 0
    failed = 0

    for script_name, (pattern, replacement) in FIXES.items():
        filepath = Path(script_name)
        if not filepath.exists():
            print(f"⚠️  {script_name:<30} NOT FOUND")
            failed += 1
            continue

        success, msg = fix_output_dir(filepath, pattern, replacement)
        if success:
            print(f"✓ {script_name:<30} {msg}")
            updated += 1
        else:
            print(f"  {script_name:<30} {msg}")
            if "already fixed" not in msg:
                failed += 1

    print()
    print("="*80)
    print(f"SUMMARY: {updated} fixed, {failed} failed")
    print("="*80)

if __name__ == "__main__":
    main()
