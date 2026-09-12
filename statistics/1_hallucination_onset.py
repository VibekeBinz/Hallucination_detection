"""
Per-chain hallucination-onset detection. Step 1 of the results stage.

Onset is the hallucination-spike definition shared with Claim 2
(claim2_hallucination_collapse.py): the first generation at which a chain's
single-step (generation-over-generation) increase reaches or exceeds a
threshold T -- not a cumulative rise from baseline, and not the single
largest jump anywhere in the run, but literally the first real jump,
scanned in generation order. T=0.05 (a 5-percentage-point one-step rise),
chosen from the empirical gap in the pooled single-step-increase
distribution across all 8 generator/variant tokens: the six generators
other than DDPM/DDPM_CORE top out at a max one-step increase of 0.0427,
DDPM_CORE's chains start at 0.0904, and nothing falls in between --
0.05 sits in that gap. Both hallucination metrics are run through the
same detector for consistency, though hr_b5 is not expected to give a
meaningful onset signal (see extract_qa_metrics for why); T=0.05 was
calibrated on expert_any specifically.

The (glob pattern, threshold, direction) for each metric lives in
metric_config.py's HALLUCINATION_METRICS dict, not here -- that's the
single source of truth other scripts (claim2_hallucination_collapse.py)
read the calibration from, since this file's own name
(1_hallucination_onset.py) starts with a digit and so can never be
`import`ed by another script under that name (see metric_config.py's
docstring for why).

Inputs: every file matching step4_<metric>_per_chain_<GENERATOR>.csv in
PER_CHAIN_DIR, one file per (metric, generator/variant) combination
(generator already spelled the GENERATOR_VARIANTS way, e.g. "CTGAN",
"DDPM_CORE", inside each file's own generator column -- no cross-file
token translation needed here, unlike 3_failure_events.py's
minority_share join).

CHAIN REGISTRY: a generator/chain is sometimes entirely absent from a
metric's per-chain file, or has fewer than 2 observed generations. Either
way that means no real single-step jump was observed for that chain --
not a gap in the analysis -- so it is reported as non-spiking
(crossed=False, onset_generation=None) via the full chain registry
(tstr_auroc_per_chain.csv, populated for every chain regardless of
hallucination activity) rather than dropped or left undefined.

Output: hallucination_onset.csv, one row per (metric, generator, dseed,
mseed) -- the onset generation (or blank if no single-step increase ever
reached T within the observed generations), the size of that qualifying
step, whether it crossed at all, and how many generations were actually
observed.

The onset-detection and chain-registry logic itself lives in
onset_utils.py (load_chain_registry / compute_incline_onset_table) --
shared with claim1_hallucination_utility.py and
claim2_hallucination_collapse.py so hallucination onset means the same
thing in every downstream analysis.
"""

import glob
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
from metric_config import HALLUCINATION_METRICS

try:
    from config.pipeline_config import STATISTICS_DIR
    OUT_DIR = STATISTICS_DIR
except ImportError:
    OUT_DIR = os.path.join(RESULTS_ROOT, "statistics")

OUT_FILE = os.path.join(OUT_DIR, "hallucination_onset.csv")


def load_metric(pattern):
    paths = sorted(glob.glob(os.path.join(PER_CHAIN_DIR, pattern)))
    if not paths:
        print(f"  [WARN] no files matched {pattern} in {PER_CHAIN_DIR}")
        return pd.DataFrame(columns=["generator", "dseed", "mseed", "chain", "generation", "value"])
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=["generator", "dseed", "mseed", "chain", "generation", "value"])
        frames.append(df)
        print(f"  loaded {os.path.basename(p)}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def main():
    registry = load_chain_registry(PER_CHAIN_DIR)
    n_registry_chains = len(registry)
    print(f"Chain registry: {n_registry_chains} chains "
          f"({registry['generator'].nunique()} generators x "
          f"{n_registry_chains // registry['generator'].nunique()} dseed/mseed pairs)")

    all_results = []
    for metric, cfg in HALLUCINATION_METRICS.items():
        print(f"\n{metric}:")
        df = load_metric(cfg["glob"])
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
                  f"generations for {metric} -- no single-step increase could be computed, "
                  f"so they are reported as non-spiking:")
            for _, r in insufficient.iterrows():
                print(f"    {r['generator']} {r['chain']} (n_observed_generations={r['n_observed_generations']})")

    out = pd.concat(all_results, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}  ({len(out)} rows)")


if __name__ == "__main__":
    main()