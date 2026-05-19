#!/usr/bin/env python3
"""
Remove the extra ')' at the end of --n_fft help string
"""

import re
from pathlib import Path

SCRIPTS = [
    '6_best_accuracy.py',
    '7_hybrid.py',
    '8a_dropout01.py',
    '8b_dropout02.py',
    '8c_dropout03.py',
    '8d_dropout04.py',
    '9a_depthwise_drop01.py',
    '9b_depthwise_drop02.py',
    '9c_depthwise_drop03.py',
    '9d_depthwise_drop04.py',
    '10_depthwise_f6.py',
]

def fix_extra_paren(filepath: Path):
    """Remove extra ')' at end of --n_fft help string"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace the extra ')' pattern
    # From: help='FFT window size (default: 1024)')')
    # To:   help='FFT window size (default: 1024)')
    pattern = r"(help=['\"]FFT window size \(default: 1024\)['\"])\)['\"]"
    replacement = r"\1"

    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        return False, "pattern not found"

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    return True, "removed extra ')'"

def main():
    print("="*80)
    print("REMOVING EXTRA PARENTHESIS")
    print("="*80)
    print()

    updated = 0
    failed = 0

    for script_name in SCRIPTS:
        filepath = Path(script_name)
        if not filepath.exists():
            print(f"⚠️  {script_name:<30} NOT FOUND")
            failed += 1
            continue

        success, msg = fix_extra_paren(filepath)
        if success:
            print(f"✓ {script_name:<30} {msg}")
            updated += 1
        else:
            print(f"  {script_name:<30} {msg}")
            failed += 1

    print()
    print("="*80)
    print(f"SUMMARY: {updated} fixed, {failed} failed")
    print("="*80)

if __name__ == "__main__":
    main()
