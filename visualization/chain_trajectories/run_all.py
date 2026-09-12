"""
visualization/chain_trajectories/run_all.py

Runs every chain-trajectory collector in one call: TSTR AUROC + SD quality
metrics, the four hallucination metrics (HR_b5, Expert_any, LLM_only,
Expert_plus_LLM), and the readmission minority-share trajectory.
"""

import os
import sys
import traceback


_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

from visualization.chain_trajectories import (
    collect_tstr_sd,
    collect_hallucination,
    collect_minority_share,
)

MODULES = [collect_tstr_sd, collect_hallucination, collect_minority_share]


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
    print("\nAll chain-trajectory figures done.")


if __name__ == "__main__":
    main()
