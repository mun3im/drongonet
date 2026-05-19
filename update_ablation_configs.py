#!/usr/bin/env python3
"""
Batch update scripts 2-10 for rerun with optimal hyperparameters
Updates: n_fft=1024, n_mels=48 (from baseline sweep results)
"""

import re
from pathlib import Path

# Scripts to update
SCRIPTS = [
    '2_depthwise.py',
    '3_batchnorm.py',
    '4_dense.py',
    '5_filters.py',
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

def update_script(filepath: Path):
    """Update hyperparameters in a single script"""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # 1. Update n_mels default in TrainingConfig
    pattern = r'(n_mels:\s*int\s*=\s*)\d+'
    if re.search(pattern, content):
        new_content = re.sub(pattern, r'\g<1>48', content)
        if new_content != content:
            changes.append('n_mels: X → 48')
            content = new_content

    # 2. Update n_fft default in TrainingConfig
    pattern = r'(n_fft:\s*int\s*=\s*)\d+'
    if re.search(pattern, content):
        new_content = re.sub(pattern, r'\g<1>1024', content)
        if new_content != content:
            changes.append('n_fft: X → 1024')
            content = new_content

    # 3. Update n_mels in parse_args default
    pattern = r"(parser\.add_argument\(['\"]--n_mels['\"],.*?default=)\d+"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, r'\g<1>48', content)
        if new_content != content:
            changes.append('parse_args n_mels: X → 48')
            content = new_content

    # 4. Update n_fft in parse_args default
    pattern = r"(parser\.add_argument\(['\"]--n_fft['\"],.*?default=)\d+"
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, r'\g<1>1024', content)
        if new_content != content:
            changes.append('parse_args n_fft: X → 1024')
            content = new_content

    # 5. Update model input_shape default (if using hardcoded 80)
    pattern = r'(input_shape=\(184,\s*)80(\s*,\s*1\))'
    if re.search(pattern, content):
        new_content = re.sub(pattern, r'\g<1>48\g<2>', content)
        if new_content != content:
            changes.append('input_shape: (184, 80, 1) → (184, 48, 1)')
            content = new_content

    # Check if any changes were made
    if content != original:
        # Write back
        with open(filepath, 'w') as f:
            f.write(content)
        return True, changes
    return False, []

def main():
    print("="*80)
    print("UPDATING ABLATION SCRIPTS FOR RERUN")
    print("="*80)
    print(f"Target configuration: n_fft=1024, n_mels=48")
    print(f"Updating {len(SCRIPTS)} scripts")
    print("="*80)
    print()

    updated_count = 0
    unchanged_count = 0

    for script_name in SCRIPTS:
        filepath = Path(script_name)

        if not filepath.exists():
            print(f"⚠️  {script_name:<30} NOT FOUND")
            continue

        updated, changes = update_script(filepath)

        if updated:
            print(f"✓ {script_name:<30} UPDATED")
            for change in changes:
                print(f"  - {change}")
            updated_count += 1
        else:
            print(f"  {script_name:<30} No changes needed")
            unchanged_count += 1

    print()
    print("="*80)
    print(f"SUMMARY: {updated_count} updated, {unchanged_count} unchanged")
    print("="*80)

    if updated_count > 0:
        print()
        print("Next steps:")
        print("1. Review changes: git diff")
        print("2. Run rerun script: ./rerun_ablations_2_10.sh")
        print("3. Monitor progress: watch -n 60 'ls -ltr results/ | tail -20'")

if __name__ == "__main__":
    main()
