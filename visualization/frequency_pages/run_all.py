"""
visualization/frequency_pages/run_all.py

Runs both population/feature-level figure scripts in one call: the
population feature plots (numeric histograms + KDE, categorical bar
charts) and the real-vs-synthetic frequency/KDE grid, one page per feature.
"""

import os
import sys
import traceback

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../run_all.py`, an IDE "Run" button, or
# `python -m visualization.frequency_pages.run_all` from the repo root --
# only the last of those already has the repo root on sys.path).
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    # sys.path.insert(0, _p)
    sys.path.append(_p)
from visualization.frequency_pages import population_features, frequency_pages

MODULES = [population_features, frequency_pages]


def main():
    failed = []
    for mod in MODULES:
        print(f"\n{'='*60}\n{mod.__name__}\n{'='*60}")
        try:
            mod.main()
        except Exception:
            print(f"  [FAIL] {mod.__name__}")
            traceback.print_exc()
            failed.append(mod.__name__)

    if failed:
        print(f"\n[WARN] {len(failed)} module(s) failed: {failed}")
    print("\nAll frequency-page figures done.")


if __name__ == "__main__":
    main()
