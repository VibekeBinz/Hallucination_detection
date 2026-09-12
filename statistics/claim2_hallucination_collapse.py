"""
Claim 2: beta-recall (delta_coverage), alpha-precision (delta_precision),
and authenticity "spikes" coincide with the hallucination spike,
specifically for DDPM (DDPM and DDPM_CORE).

"Specifically for DDPM" is a claim about specificity, not just about
DDPM in isolation -- it says the other generators should NOT show the
same coincidence (or shouldn't have a comparison to make at all, if they
never really hallucinate). By default this script checks every
generator/variant token, not just DDPM/DDPM_CORE, and reports a
per-generator breakdown so that can actually be checked rather than
assumed. Pass --generators to restrict to a subset (e.g. the original
DDPM DDPM_CORE) if only the original two-variant comparison is wanted.

This script tests the claim; it does not compute onset. Onset for all
four metrics is computed once, upstream, by two results scripts, and read
from their output here:
  - hallucination_onset.csv  (1_hallucination_onset.py) -- metric=expert_any
  - qa_metric_onset.csv      (2_qametric_onset.py)       -- beta_recall,
                                                             alpha_precision,
                                                             authenticity
Both must be run first. This mirrors claim1_hallucination_utility.py, which reads
failure_events.csv and hallucination_onset.csv rather than recomputing
either -- results and claim-testing are two separate steps, so "onset"
is computed exactly once per metric, the same way, regardless of which
claim ends up checked against it.

ALL FOUR METRICS share the same underlying onset definition
(onset_utils.compute_incline_onset_table, used by both upstream results
scripts): the first generation at which a chain's single-step change, in
that metric's own degradation direction, exceeds a threshold T -- scanned
in generation order, not a cumulative rise and not the single largest
jump anywhere in the run. Each metric has its own (threshold, direction)
pair, calibrated on its own scale, and that calibration lives in
metric_config.py (imported here, not duplicated) rather than in either
numbered results script -- 1_hallucination_onset.py and
2_qametric_onset.py can't be `import`ed by filename (Python module names
can't start with a digit), so metric_config.py is the shared,
unprefixed home for every metric's (threshold, direction), and this
script cannot drift out of sync with the calibration the onset tables
were actually built with.

Output: CLAIM2_OUT_FILE (per-chain spike-generation table for all four
metrics, for whichever generators were selected) plus:
  - a pooled comparison across all selected generators together (labeled
    clearly as pooled, since mixing a generator with real spikes and one
    with noise-floor-only "spikes" can distort a combined tau/Wilcoxon)
  - a per-generator breakdown, each generator's ~15 chains tested on its
    own -- this is the actual specificity check
  - leave-one-dseed-out / leave-one-mseed-out sensitivity, run on the
    pooled set only (per-generator n is already only 15; dropping one
    seed there adds little and multiplies output substantially)
"""

import argparse
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
from scipy.stats import kendalltau, wilcoxon

from config.pipeline_config import RESULTS_ROOT, GENERATOR_VARIANTS
from metric_config import HALLUCINATION_METRICS, QA_METRICS as QA_METRICS_CONFIG

HALLUCINATION_T = HALLUCINATION_METRICS["expert_any"]["threshold"]

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

# Default: every generator/variant token, so the DDPM-specificity part of
# the claim is actually checked rather than assumed. Pass e.g.
# --generators DDPM DDPM_CORE to reproduce the original DDPM-only scope.
GENERATORS = list(GENERATOR_VARIANTS.keys())

QA_METRICS = ["beta_recall", "alpha_precision", "authenticity"]

HALLUCINATION_ONSET_FILE = os.path.join(STATS_DIR, "hallucination_onset.csv")
QA_ONSET_FILE = os.path.join(STATS_DIR, "qa_metric_onset.csv")
OUT_FILE = os.path.join(STATS_DIR, "claim2_spike_generations.csv")


def load_onset_tables(generators):
    """
    Reads the two upstream results files and restricts each metric's
    onset table to the selected generators. Raises with a clear message
    if either file is missing -- that means the results step hasn't been
    run yet, not that this script should silently recompute it.
    """
    if not os.path.exists(HALLUCINATION_ONSET_FILE):
        raise SystemExit(f"Missing {HALLUCINATION_ONSET_FILE} -- run 1_hallucination_onset.py first.")
    if not os.path.exists(QA_ONSET_FILE):
        raise SystemExit(f"Missing {QA_ONSET_FILE} -- run 2_qametric_onset.py first.")

    hallu = pd.read_csv(HALLUCINATION_ONSET_FILE)
    hallu = hallu[hallu["metric"] == "expert_any"]
    if hallu.empty:
        raise SystemExit(f"No metric='expert_any' rows in {HALLUCINATION_ONSET_FILE}.")

    qa = pd.read_csv(QA_ONSET_FILE)

    tables = {}
    hallu_tbl = hallu[hallu["generator"].isin(generators)].copy()
    hallu_tbl = hallu_tbl.rename(columns={"onset_generation": "spike_generation", "step_size": "spike_size"})
    tables["hallucination_expert_any"] = hallu_tbl

    for label in QA_METRICS:
        tbl = qa[(qa["metric"] == label) & (qa["generator"].isin(generators))].copy()
        if tbl.empty:
            raise SystemExit(f"No metric={label!r} rows in {QA_ONSET_FILE}.")
        tbl = tbl.rename(columns={"onset_generation": "spike_generation", "step_size": "spike_size"})
        tables[label] = tbl

    return tables


def paired_generation_tests(a, b, label_a, label_b):
    """
    a, b: two spike-generation DataFrames (generator, dseed, mseed, chain,
    spike_generation, ...), already restricted to the same chain set.
    Reports Kendall's tau + Wilcoxon on the paired spike-generation
    differences, restricted further to chains where BOTH have a defined
    spike_generation (a real single-step jump was found in both).
    """
    merged = a.merge(b, on=["generator", "dseed", "mseed", "chain"], suffixes=(f"_{label_a}", f"_{label_b}"))
    col_a, col_b = f"spike_generation_{label_a}", f"spike_generation_{label_b}"
    both = merged.dropna(subset=[col_a, col_b])
    n = len(both)
    if n < 2:
        print(f"    n={n}: too few chains with a defined spike in both metrics for a test")
        return merged, {"n": n}

    diffs = both[col_b] - both[col_a]
    tau, tau_p = kendalltau(both[col_a], both[col_b])
    result = {"n": n, "tau": tau, "tau_p": tau_p, "median_offset_generations": diffs.median()}
    if (diffs != 0).any():
        w_stat, w_p = wilcoxon(diffs)
        result["wilcoxon_stat"], result["wilcoxon_p"] = w_stat, w_p
        print(f"    n={n}  Kendall's tau={tau:.3f} (p={tau_p:.4f})  "
              f"Wilcoxon on ({label_b} - {label_a} spike generation): stat={w_stat:.2f} (p={w_p:.4f})  "
              f"median offset={diffs.median():.1f} generations")
    else:
        print(f"    n={n}  Kendall's tau={tau:.3f} (p={tau_p:.4f})  "
              f"Wilcoxon skipped (all paired differences are exactly 0 -- perfect coincidence)")
    return merged, result


def per_generator_report(spike_tables, generators):
    """
    The actual specificity check: run the same paired test separately for
    each generator token (n~=15 chains each) instead of pooling them.
    Every metric's spike_generation is threshold-gated (its own
    calibrated T, sourced from metric_config.py),
    so "n had a qualifying step" is a direct, comparable count for all
    four -- no separate noise-floor diagnostic needed.
    """
    print("\n\n=== Per-generator breakdown (does the coincidence hold generator-by-generator?) ===")
    hallu = spike_tables["hallucination_expert_any"]
    for generator in generators:
        g_hallu = hallu[hallu["generator"] == generator]
        n_total = len(g_hallu)
        n_defined = int(g_hallu["spike_generation"].notna().sum())
        print(f"\n-- {generator} ({n_total} chains) --")
        print(f"  hallucination_expert_any: {n_defined}/{n_total} chains had a qualifying step (T={HALLUCINATION_T})")
        for qa in QA_METRICS:
            g_qa = spike_tables[qa][spike_tables[qa]["generator"] == generator]
            n_qa_defined = int(g_qa["spike_generation"].notna().sum())
            print(f"  vs {qa} ({n_qa_defined}/{n_total} had a qualifying step, "
                  f"T={QA_METRICS_CONFIG[qa]['threshold']}):")
            paired_generation_tests(g_hallu, g_qa, "hallucination_expert_any", qa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+", default=GENERATORS,
                         help=f"Which generator tokens to include (default: {GENERATORS}).")
    args = parser.parse_args()

    spike_tables = load_onset_tables(args.generators)

    hallu_tbl = spike_tables["hallucination_expert_any"]
    n_defined = int(hallu_tbl["spike_generation"].notna().sum())
    print(f"hallucination_expert_any: {n_defined}/{len(hallu_tbl)} chains had a qualifying single-step "
          f"increase >= T={HALLUCINATION_T} (1_hallucination_onset.py)")
    for label in QA_METRICS:
        tbl = spike_tables[label]
        cfg = QA_METRICS_CONFIG[label]
        n_defined = int(tbl["spike_generation"].notna().sum())
        print(f"{label}: {n_defined}/{len(tbl)} chains had a qualifying single-step "
              f"{cfg['direction']} >= T={cfg['threshold']} (2_qametric_onset.py)")

    combined = hallu_tbl[["generator", "dseed", "mseed", "chain"]].copy()
    for label, tbl in spike_tables.items():
        combined = combined.merge(
            tbl[["generator", "dseed", "mseed", "spike_generation", "spike_size"]]
            .rename(columns={"spike_generation": f"{label}_spike_generation",
                              "spike_size": f"{label}_spike_size"}),
            on=["generator", "dseed", "mseed"],
        )
    os.makedirs(STATS_DIR, exist_ok=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}  ({len(combined)} chains)")

    def run_comparisons(hallu, qa_tables, header):
        print(header)
        for qa in QA_METRICS:
            print(f"  vs {qa}:")
            paired_generation_tests(hallu, qa_tables[qa], "hallucination_expert_any", qa)

    print("\n\n=== Full population (pooled across all selected generators -- "
          "see per-generator breakdown below for the specificity check) ===")
    run_comparisons(hallu_tbl, spike_tables, "(hallucination spike generation vs each QA metric's spike generation)")

    per_generator_report(spike_tables, args.generators)

    all_dseeds = sorted(set(hallu_tbl["dseed"]))
    all_mseeds = sorted(set(hallu_tbl["mseed"]))

    print("\n\n=== Leave-one-dseed-out sensitivity (pooled set) ===")
    for dseed in all_dseeds:
        print(f"\n-- excluding dseed={dseed} --")
        for qa in QA_METRICS:
            a = hallu_tbl[hallu_tbl["dseed"] != dseed]
            b = spike_tables[qa][spike_tables[qa]["dseed"] != dseed]
            print(f"  vs {qa}:")
            paired_generation_tests(a, b, "hallucination_expert_any", qa)

    print("\n\n=== Leave-one-mseed-out sensitivity (pooled set) ===")
    for mseed in all_mseeds:
        print(f"\n-- excluding mseed={mseed} --")
        for qa in QA_METRICS:
            a = hallu_tbl[hallu_tbl["mseed"] != mseed]
            b = spike_tables[qa][spike_tables[qa]["mseed"] != mseed]
            print(f"  vs {qa}:")
            paired_generation_tests(a, b, "hallucination_expert_any", qa)


if __name__ == "__main__":
    main()