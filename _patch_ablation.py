#!/usr/bin/env python3
"""
_patch_ablation.py — one-time patcher for all numbered ablation scripts.

Applies four targeted substitutions to every [0-9]*.py script:
  1. Injects  `from config import DATASET_PATH, TINYCHIRP_PATH, RESULTS_BASE, CACHE_BASE`
     after the last standard-library/framework import line.
  2. Replaces dataset-path argparse defaults with config constants.
  3. Replaces hardcoded cache_dir derivations with CACHE_BASE + n_fft + n_mels.
  4. Redirects output_dir assignments from 'results/' to RESULTS_BASE.

Idempotent — skips files already containing `from config import`.
Run once before the full ablation sweep:
    python _patch_ablation.py
"""

import re
import sys
from pathlib import Path

SCRIPTS = sorted(Path(".").glob("[0-9]*.py"))

CONFIG_IMPORT = "from config import DATASET_PATH, TINYCHIRP_PATH, RESULTS_BASE, CACHE_BASE\n"

# ------------------------------------------------------------------
# Substitution rules  (applied in order, each a (pattern, replacement) pair)
# ------------------------------------------------------------------

SUBS = [
    # --- dataset path defaults ---
    (
        r"default='/Volumes/Evo/seabad'",
        "default=DATASET_PATH",
    ),
    (
        r'default="/Volumes/Evo/seabad"',
        "default=DATASET_PATH",
    ),
    (
        r"default='/Volumes/Evo/TinyChirp'",
        "default=TINYCHIRP_PATH",
    ),
    (
        r'default="/Volumes/Evo/TinyChirp"',
        "default=TINYCHIRP_PATH",
    ),

    # --- cache_dir derivation: config.cache_dir = f'..._m{config.n_mels}' ---
    (
        r"config\.cache_dir = f'/Volumes/Evo/cache_seabad_m\{config\.n_mels\}'",
        "config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'",
    ),
    (
        r'config\.cache_dir = f"/Volumes/Evo/cache_seabad_m\{config\.n_mels\}"',
        "config.cache_dir = f'{CACHE_BASE}_fft{config.n_fft}_m{config.n_mels}'",
    ),

    # --- cache_dir derivation: self.cache_dir = f'..._m{self.n_mels}' ---
    (
        r"self\.cache_dir = f'/Volumes/Evo/cache_seabad_m\{self\.n_mels\}'",
        "self.cache_dir = f'{CACHE_BASE}_fft{self.n_fft}_m{self.n_mels}'",
    ),
    (
        r'self\.cache_dir = f"/Volumes/Evo/cache_seabad_m\{self\.n_mels\}"',
        "self.cache_dir = f'{CACHE_BASE}_fft{self.n_fft}_m{self.n_mels}'",
    ),

    # --- dataclass field default (static fallback; always overridden at runtime) ---
    (
        r"cache_dir: str = '/Volumes/Evo/cache_seabad_mels'",
        "cache_dir: str = f'{CACHE_BASE}_fft1024_m64'",
    ),
    (
        r"cache_dir: str = '/Volumes/Evo/cache_seabad_m64'",
        "cache_dir: str = f'{CACHE_BASE}_fft1024_m64'",
    ),

    # --- output_dir assignment: config.output_dir = f'results/... ---
    (
        r"config\.output_dir = f'results/",
        "config.output_dir = f'{RESULTS_BASE}/",
    ),
    (
        r'config\.output_dir = f"results/',
        'config.output_dir = f"{RESULTS_BASE}/',
    ),
]


def find_import_insertion_point(lines: list[str]) -> int:
    """
    Return the line index AFTER the last top-level import block.
    We look for the last consecutive `import X` / `from X import Y` line
    before any class/def/if/__name__ block.
    """
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(import |from \S+ import )', stripped):
            last_import_idx = i
        elif stripped and not stripped.startswith('#') and last_import_idx >= 0:
            # First non-import, non-blank, non-comment line after imports
            if re.match(r'^(class |def |if |@|logger |logging\.|tf\.|os\.environ)', stripped):
                break
    return last_import_idx + 1  # insert AFTER the last import line


def patch(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text()

    # Idempotency guard
    if "from config import" in text:
        print(f"  SKIP  {path.name}  (already patched)")
        return False

    lines = text.splitlines(keepends=True)

    # 1. Inject config import
    insert_at = find_import_insertion_point(lines)
    lines.insert(insert_at, CONFIG_IMPORT)
    text = "".join(lines)

    # 2. Apply substitution rules
    change_count = 0
    for pattern, replacement in SUBS:
        new_text, n = re.subn(pattern, replacement, text)
        if n:
            change_count += n
            text = new_text

    if dry_run:
        print(f"  DRY   {path.name}  ({change_count} substitutions + import injected)")
        return True

    path.write_text(text)
    print(f"  PATCH {path.name}  ({change_count} substitutions + import injected)")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN — no files will be written\n")

    patched = skipped = 0
    for script in SCRIPTS:
        if script.name.startswith("_"):
            continue
        result = patch(script, dry_run=dry_run)
        if result:
            patched += 1
        else:
            skipped += 1

    print(f"\nDone: {patched} patched, {skipped} skipped.")
    if dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
