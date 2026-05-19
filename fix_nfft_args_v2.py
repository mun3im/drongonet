#!/usr/bin/env python3
"""
Fix the syntax errors introduced by fix_nfft_args.py
Properly close --n_mels argument and add --n_fft argument
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

def fix_nfft_argument(filepath: Path):
    """Fix the broken --n_fft argument insertion"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match the broken structure
    # Looking for: help='...(default: 80)\n    parser.add_argument('--n_fft'...)')'
    broken_pattern = r"(parser\.add_argument\(['\"]--n_mels['\"],.*?help=['\"])([^'\"]*?)(\n\s+parser\.add_argument\(['\"]--n_fft['\"],.*?help=['\"])([^'\"]*?)(['\"])\)"

    # Check if we have the broken pattern
    match = re.search(broken_pattern, content, re.DOTALL)
    if not match:
        return False, "pattern not found"

    # Reconstruct properly
    # The correct format should be:
    # parser.add_argument('--n_mels', type=int, default=48,
    #                     help='Number of mel frequency bins (default: 80)')
    # parser.add_argument('--n_fft', type=int, default=1024,
    #                     help='FFT window size (default: 1024)')

    fixed_section = """parser.add_argument('--n_mels', type=int, default=48,
                        help='Number of mel frequency bins (default: 80)')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')"""

    new_content = re.sub(broken_pattern, fixed_section, content, flags=re.DOTALL)

    if new_content == content:
        return False, "replacement failed"

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    return True, "fixed --n_fft argument"

def main():
    print("="*80)
    print("FIXING BROKEN --n_fft ARGUMENTS")
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

        success, msg = fix_nfft_argument(filepath)
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
