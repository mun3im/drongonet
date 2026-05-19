#!/usr/bin/env python3
"""
Standardize results folder naming conventions
Removes redundant prefixes and suffixes for consistency
"""

import os
import re
from pathlib import Path
import shutil

def standardize_name(name):
    """
    Standardize folder name according to conventions:
    1. Remove 'results_' prefix (redundant in results/ folder)
    2. Remove '_linux' suffix (platform info stored in metadata)
    3. Keep meaningful suffixes like '_balanced'

    Args:
        name: Original folder name

    Returns:
        Standardized folder name
    """
    original = name

    # Remove 'results_' prefix
    if name.startswith('results_'):
        name = name[8:]  # Remove 'results_'

    # Remove '_linux' suffix (but preserve _balanced and other meaningful suffixes)
    if name.endswith('_linux'):
        name = name[:-6]  # Remove '_linux'

    return name, original != name  # Return new name and whether it changed

def main():
    """Main function to rename all result folders"""
    results_dir = Path("results")

    if not results_dir.exists():
        print("❌ results/ directory not found!")
        return

    # Collect all directories that need renaming
    renames = []

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue

        old_name = d.name
        new_name, changed = standardize_name(old_name)

        if changed:
            renames.append((old_name, new_name))

    if not renames:
        print("✓ All folder names are already standardized!")
        return

    print("="*80)
    print(f"PROPOSED RENAMES ({len(renames)} folders)")
    print("="*80)

    for old, new in renames:
        print(f"{old:<60} → {new}")

    print("\n" + "="*80)
    response = input(f"Proceed with renaming {len(renames)} folders? (yes/no): ").strip().lower()

    if response != 'yes':
        print("❌ Renaming cancelled")
        return

    # Perform renames
    print("\n" + "="*80)
    print("RENAMING FOLDERS")
    print("="*80)

    success_count = 0
    error_count = 0

    for old_name, new_name in renames:
        old_path = results_dir / old_name
        new_path = results_dir / new_name

        try:
            # Check if target already exists
            if new_path.exists():
                print(f"⚠️  {old_name:<60} SKIP: target exists")
                error_count += 1
                continue

            # Rename
            old_path.rename(new_path)
            print(f"✓ {old_name:<60} → {new_name}")
            success_count += 1

        except Exception as e:
            print(f"❌ {old_name:<60} ERROR: {e}")
            error_count += 1

    print("\n" + "="*80)
    print(f"SUMMARY: {success_count} renamed, {error_count} errors")
    print("="*80)

if __name__ == "__main__":
    main()
