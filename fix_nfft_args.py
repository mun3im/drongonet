#!/usr/bin/env python3
"""
Add missing --n_fft argument to scripts 6-10
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

def add_nfft_argument(filepath: Path):
    """Add --n_fft argument to parse_args if missing"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already has --n_fft
    if "--n_fft" in content or '--n_fft' in content:
        return False, "already has --n_fft"

    # Find the n_mels argument and add n_fft after it
    pattern = r"(parser\.add_argument\(['\"]--n_mels['\"],.*?\n.*?help=.*?\))"

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return False, "couldn't find --n_mels argument"

    n_mels_arg = match.group(1)

    # Create n_fft argument
    nfft_arg = """    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')"""

    # Insert after n_mels
    new_content = content.replace(n_mels_arg, n_mels_arg + "\n" + nfft_arg)

    if new_content == content:
        return False, "replacement failed"

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    return True, "added --n_fft argument"

def main():
    print("="*80)
    print("ADDING MISSING --n_fft ARGUMENT")
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

        success, msg = add_nfft_argument(filepath)
        if success:
            print(f"✓ {script_name:<30} {msg}")
            updated += 1
        else:
            print(f"  {script_name:<30} {msg}")
            if "already has" not in msg:
                failed += 1

    print()
    print("="*80)
    print(f"SUMMARY: {updated} updated, {failed} failed")
    print("="*80)

if __name__ == "__main__":
    main()
