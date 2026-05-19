#!/usr/bin/env python3
"""
Update ablation scripts 2-10 to use new optimal hyperparameters from V3 dataset
Changes:
- n_mels: 48 → 64
- input_shape: (184, 48, 1) → (184, 64, 1)
- n_fft: 1024 (unchanged)
"""

import re
from pathlib import Path

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
    """Update n_mels and input_shape in ablation script"""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Update n_mels default in TrainingConfig
    pattern1 = r'(n_mels:\s*int\s*=\s*)48'
    if re.search(pattern1, content):
        content = re.sub(pattern1, r'\g<1>64', content)
        changes.append("n_mels default: 48 → 64")

    # 2. Update input_shape from (184, 48, 1) to (184, 64, 1)
    pattern2 = r'input_shape\s*=\s*\(184,\s*48,\s*1\)'
    if re.search(pattern2, content):
        content = re.sub(pattern2, 'input_shape=(184, 64, 1)', content)
        changes.append("input_shape: (184, 48, 1) → (184, 64, 1)")

    # 3. Update parse_args default if exists
    pattern3 = r"(parser\.add_argument\(['\"]--n_mels['\"],\s*type=int,\s*default=)48"
    if re.search(pattern3, content):
        content = re.sub(pattern3, r'\g<1>64', content)
        changes.append("parse_args n_mels: 48 → 64")

    # 4. Update help text to reflect new default
    pattern4 = r"(help=['\"]Number of mel frequency bins \(default:\s*)48\)"
    if re.search(pattern4, content):
        content = re.sub(pattern4, r'\g<1>64)', content)
        changes.append("help text: default 48 → 64")

    # 5. Update cache directory pattern (if hardcoded)
    pattern5 = r"cache_mybad_m48"
    if re.search(pattern5, content):
        content = re.sub(pattern5, 'cache_mybad_m64', content)
        changes.append("cache dir: m48 → m64")

    if content == original_content:
        return False, "no changes needed"

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    return True, ", ".join(changes)

def main():
    print("="*80)
    print("UPDATING ABLATION SCRIPTS TO USE n_mels=64 (V3 Dataset Optimal)")
    print("="*80)
    print()
    print("Changes to make:")
    print("  • n_mels default: 48 → 64")
    print("  • input_shape: (184, 48, 1) → (184, 64, 1)")
    print("  • n_fft: 1024 (unchanged)")
    print()
    print("="*80)
    print()

    updated = 0
    failed = 0
    skipped = 0

    for script_name in SCRIPTS:
        filepath = Path(script_name)
        if not filepath.exists():
            print(f"⚠️  {script_name:<30} NOT FOUND")
            failed += 1
            continue

        success, msg = update_script(filepath)
        if success:
            print(f"✓ {script_name:<30} {msg}")
            updated += 1
        else:
            print(f"  {script_name:<30} {msg}")
            skipped += 1

    print()
    print("="*80)
    print(f"SUMMARY: {updated} updated, {skipped} skipped, {failed} failed")
    print("="*80)
    print()

    if updated > 0:
        print("Next steps:")
        print("1. Backup old results (m48-based):")
        print("   mv results results_v2_m48_$(date +%Y%m%d)")
        print()
        print("2. Rerun ablation studies with new config:")
        print("   ./rerun_ablations_2_10.sh")
        print()
        print("3. Or test single experiment first:")
        print("   python3 3_batchnorm.py --n_mels 64 --n_fft 1024 --use_cache")
        print()

    return updated, failed

if __name__ == "__main__":
    updated, failed = main()
    exit(0 if failed == 0 else 1)
