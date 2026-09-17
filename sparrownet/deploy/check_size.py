"""
check_size.py — enforce the SparrowNet INT8 size budget.

Soft limit 10 KB (target), hard limit 16 KB (reject). Training scripts should call
check_size() right after TFLite conversion so an over-budget model fails loudly
instead of quietly landing in the results table.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIZE_SOFT_LIMIT_BYTES, SIZE_HARD_LIMIT_BYTES


class SizeBudgetExceeded(Exception):
    pass


def check_size(tflite_path: str, raise_on_hard: bool = True) -> dict:
    """Returns {"bytes", "kb", "within_soft", "within_hard"}.
    Raises SizeBudgetExceeded if over the hard limit and raise_on_hard."""
    size = os.path.getsize(tflite_path)
    result = {
        "bytes": size,
        "kb": size / 1024.0,
        "within_soft": size <= SIZE_SOFT_LIMIT_BYTES,
        "within_hard": size <= SIZE_HARD_LIMIT_BYTES,
    }

    name = os.path.basename(tflite_path)
    if not result["within_hard"]:
        msg = (f"{name}: {result['kb']:.2f} KB exceeds HARD limit "
               f"{SIZE_HARD_LIMIT_BYTES / 1024:.0f} KB")
        if raise_on_hard:
            raise SizeBudgetExceeded(msg)
        print(f"REJECT  {msg}")
    elif not result["within_soft"]:
        print(f"WARN    {name}: {result['kb']:.2f} KB over soft target "
              f"{SIZE_SOFT_LIMIT_BYTES / 1024:.0f} KB, under hard limit "
              f"{SIZE_HARD_LIMIT_BYTES / 1024:.0f} KB")
    else:
        print(f"OK      {name}: {result['kb']:.2f} KB (soft target "
              f"{SIZE_SOFT_LIMIT_BYTES / 1024:.0f} KB)")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 check_size.py <model.tflite> [...]")
        sys.exit(1)
    over_hard = False
    for path in sys.argv[1:]:
        r = check_size(path, raise_on_hard=False)
        over_hard = over_hard or not r["within_hard"]
    sys.exit(1 if over_hard else 0)
