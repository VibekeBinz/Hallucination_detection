"""
Per-chain onset detection for the three QA/generation-quality metrics that
measure mode collapse and degeneration -- beta-recall, alpha-precision,
authenticity. Step 2 of the results stage. This is the "results" step for
these metrics, computed once here and read from its output file by
whichever claim-testing script needs it
(claim2_hallucination_collapse.py), the same relationship
1_hallucination_onset.py and 3_failure_events.py already have with
claim1_hallucination_utility.py: results are computed once, in one place,
before any claim is checked against them -- not recomputed inline as part
of testing a specific claim.

Onset is onset_utils.compute_incline_onset_table's shared spike
definition -- the first generation at which a chain's single-step change,
in the metric's own degradation direction, exceeds a threshold T, scanned
in generation order. This is the exact function and convention
1_hallucination_onset.py uses for the hallucination metrics, so "onset"
means the same thing across every metric in this pipeline.

The (source file, threshold, direction) for each of the three metrics
lives in metric_config.py's QA_METRICS dict, not here -- that's the
single source of truth claim2_hallucination_collapse.py also reads the
calibration from, since this file's own name (2_qametric_onset.py) starts
with a digit and so can never be `import`ed by another script under that
name (see metric_config.py's docstring for why). Thresholds: beta_recall
T=0.0242 (5% of median gen0=0.4832), alpha_precision T=0.0478 (5% of
median gen0=0.9566), authenticity T=0.0254 (5% of median gen0=0.5088) --
each calibrated separately since the three don't share a natural absolute
scale with each other or with hallucination.

Inputs: one per-chain file per metric in PER_CHAIN_DIR (already long
format: generator, dseed, mseed, chain, generation, value). All three QA
files already cover all 120 chains directly (unlike the hallucination
step4 files, which can be missing a chain entirely for a generator that
never fired), but they are still reindexed against the full chain
registry (tstr_auroc_per_chain.csv) for consistency with every other
onset table in this pipeline.

Output: qa_metric_onset.csv, one row per (metric, generator, dseed,
mseed) -- metric, generator, dseed, mseed, chain, onset_generation,
step_size, crossed, n_observed_generations. Same schema as
hallucination_onset.csv, so a downstream script can treat both onset
tables identically.
"""

import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

import pandas as pd

from config.pipeline_config import PER_CHAIN_DIR, RESULTS_ROOT
from onset_utils import load_chain_registry, compute_incline_onset_table
from metric_config import QA_METRICS

try:
    from config.pipeline_config import STATISTICS_DIR
    OUT_DIR = STATISTICS_DIR
except ImportError:
    OUT_DIR = os.path.join(RESULTS_ROOT, "statistics")

OUT_FILE = os.path.join(OUT_DIR, "qa_metric_onset.csv")


def load_metric(filename):
    path = os.path.join(PER_CHAIN_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARN] missing input file: {path}")
        return pd.DataFrame(columns=["generator", "dseed", "mseed", "chain", "generation", "value"])
    df = pd.read_csv(path, usecols=["generator", "dseed", "mseed", "chain", "generation", "value"])
    print(f"  loaded {filename}: {len(df)} rows")
    return df


def main():
    registry = load_chain_registry(PER_CHAIN_DIR)
    n_registry_chains = len(registry)
    print(f"Chain registry: {n_registry_chains} chains "
          f"({registry['generator'].nunique()} generators x "
          f"{n_registry_chains // registry['generator'].nunique()} dseed/mseed pairs)")

    all_results = []
    for metric, cfg in QA_METRICS.items():
        print(f"\n{metric}:")
        df = load_metric(cfg["file"])
        result = compute_incline_onset_table(df, registry, threshold=cfg["threshold"], direction=cfg["direction"])
        result.insert(0, "metric", metric)
        all_results.append(result)

        n_crossed = int(result["crossed"].sum())
        print(f"  [SUMMARY] {n_crossed}/{len(result)} chains had a single-step "
              f"{cfg['direction']} >= T={cfg['threshold']}")
        breakdown = result.groupby("generator")["crossed"].agg(["sum", "count"])
        for generator, r in breakdown.iterrows():
            print(f"    {generator}: {int(r['sum'])}/{int(r['count'])} crossed")

        insufficient = result[result["n_observed_generations"] < 2]
        if not insufficient.empty:
            print(f"  [CAVEAT] {len(insufficient)} chain(s) have fewer than 2 observed "
                  f"generations for {metric} -- no single-step change could be computed, "
                  f"so they are reported as non-spiking:")
            for _, r in insufficient.iterrows():
                print(f"    {r['generator']} {r['chain']} (n_observed_generations={r['n_observed_generations']})")

    out = pd.concat(all_results, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}  ({len(out)} rows)")


if __name__ == "__main__":
    main()