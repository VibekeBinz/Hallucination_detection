"""
visualization/pooled_trajectories/run_all.py

Runs all four pooled-trajectory figure scripts in one call: metric-family
trajectories, the hallucination-rate trajectory, the metric overlay, and
the bin-resolution figure. Use this for a full regeneration; run any one
module directly (`python -m visualization.pooled_trajectories.metric_family_trajectories`)
to iterate on a single figure family.
"""

import os
import sys
import traceback

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../run_all.py`, an IDE "Run" button, or
# `python -m visualization.pooled_trajectories.run_all` from the repo root --
# only the last of those already has the repo root on sys.path).
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

from visualization.pooled_trajectories import (
    metric_family_trajectories,
    hallucination_rate_trajectory,
    metric_overlay,
    bin_resolution,
)

MODULES = [
    metric_family_trajectories,
    hallucination_rate_trajectory,
    metric_overlay,
    bin_resolution,
]


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
    print("\nAll pooled-trajectory figures done.")


if __name__ == "__main__":
    main()
