"""
Claim 1: hallucination onset does NOT coincide with Failure-of-utility
onset (a dissociation claim), tested on real per-chain event tables --
not eyeballed.

Inputs (both already produced by this pipeline, read from STATISTICS_DIR):
  - failure_events.csv       (3_failure_events.py)
        generator, dseed, mseed, chain, time, event, mechanism,
        n_tstr_datapoints, insufficient_tstr_data
  - hallucination_onset.csv  (1_hallucination_onset.py)
        metric, generator, dseed, mseed, chain, onset_generation, step_size,
        crossed, n_observed_generations
        (onset_generation = first generation where a chain's single-step
        increase reaches T=0.05 -- see onset_utils.compute_incline_onset_table;
        the same definition claim2_hallucination_collapse.py uses for the hallucination
        side of Claim 2, so "hallucination onset" means the same thing in
        both claims)

Only one hallucination metric is used per run (default: expert_any -- the
metric this pipeline treats as the informative one; hr_b5 can be passed
instead via --metric hr_b5, e.g. to confirm the claim is not an artifact
of which metric is chosen).

What "dissociation" means here, and how it's tested:

1. Per-chain categorical breakdown (always computed, no dependencies
   beyond pandas): each chain is one of
     - both:                Failure occurred AND hallucination crossed T
                             (further split by whether hallucination onset
                             came before / at / after the Failure onset)
     - failure_only:        Failure occurred, hallucination never crossed T
     - hallucination_only:  hallucination crossed T, chain never Failed
                             (within the observed generations)
     - neither
   If dissociation holds, "both, hallucination strictly before Failure at
   the same generation" should be the minority, not the modal case -- and
   failure_only / hallucination_only should both be common (either event
   can occur without the other).

2. Among "both" chains only: a paired Wilcoxon signed-rank test on
   (failure_time - hallucination_onset_generation), testing whether the
   two onsets are systematically offset in time (evidence for temporal
   dissociation) rather than coincident. Also Kendall's tau between the
   two onset generations across chains, which answers a different
   question -- whether chains that hallucinate earlier also tend to fail
   earlier in rank -- and is reported separately so the two are not
   conflated.

3. Optional: a Cox proportional-hazards model with hallucination-onset
   as a binary time-varying covariate (0 before the chain's hallucination
   onset generation, 1 from that generation on), stratified by dseed AND
   generator (generator identity is a real confound: the generators most
   likely to cross hallucination onset are not a random sample of
   generators with respect to Failure risk), so the estimated hazard
   ratio answers "does crossing the hallucination threshold change the
   instantaneous Failure hazard from that point on"
   -- the most direct test of dissociation vs. causal association. This
   needs the `lifelines` package (pip install lifelines); if it isn't
   installed, this section is skipped with a message rather than failing
   the whole script, since parts 1-2 don't need it.

4. Leave-one-dseed-out and leave-one-mseed-out sensitivity: parts 2-3
   are re-run once per dseed/mseed value held out, so a result driven by
   a single seed rather than the population is visible rather than
   hidden inside one pooled number.

Output: CLAIM1_OUT_FILE (per-chain merged table with the dissociation
category, for inspection) plus everything above printed to stdout. This
script only reports what the data show; it does not decide the claim.
"""

import argparse
import itertools
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

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

from config.pipeline_config import RESULTS_ROOT

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

FAILURE_FILE = os.path.join(STATS_DIR, "failure_events.csv")
HALLUCINATION_FILE = os.path.join(STATS_DIR, "hallucination_onset.csv")
OUT_FILE = os.path.join(STATS_DIR, "claim1_dissociation_chains.csv")


def load_and_merge(metric):
    failure = pd.read_csv(FAILURE_FILE)
    onset = pd.read_csv(HALLUCINATION_FILE)
    onset = onset[onset["metric"] == metric].copy()
    if onset.empty:
        raise SystemExit(
            f"No rows for metric={metric!r} in {HALLUCINATION_FILE}. "
            f"Available metrics: {sorted(pd.read_csv(HALLUCINATION_FILE)['metric'].unique())}"
        )

    merged = failure.merge(
        onset[["generator", "dseed", "mseed", "onset_generation", "crossed", "n_observed_generations"]],
        on=["generator", "dseed", "mseed"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        print(f"[WARN] {len(unmatched)} chain(s) present in only one of the two input files "
              f"-- check that failure_events.csv and hallucination_onset.csv were built from "
              f"the same chain set:")
        print(unmatched[["generator", "dseed", "mseed", "_merge"]].to_string(index=False))
    merged = merged.drop(columns="_merge")

    merged = merged.rename(columns={
        "time": "failure_time",
        "event": "failure_event",
        "onset_generation": "hallucination_onset_generation",
        "crossed": "hallucination_crossed",
    })
    return merged


def categorize(row):
    failed = row["failure_event"] == 1
    crossed = bool(row["hallucination_crossed"])
    if failed and crossed:
        h, f = row["hallucination_onset_generation"], row["failure_time"]
        if h < f:
            return "both__hallucination_before_failure"
        elif h == f:
            return "both__same_generation"
        else:
            return "both__hallucination_after_failure"
    elif failed and not crossed:
        return "failure_only"
    elif not failed and crossed:
        return "hallucination_only"
    else:
        return "neither"


def paired_timing_tests(sub, label=""):
    """
    sub: rows already restricted to chains where both events occurred.
    Runs the two paired-timing tests and prints them. Returns the numbers
    so leave-one-out loops can report without re-printing everything, and
    so table-building scripts (appendix_tables.py, manuscript_tables.py)
    can put every field into a table without re-deriving anything from the
    printed text.
    """
    n = len(sub)
    if n < 2:
        print(f"  {label}n={n}: too few paired chains for a statistical test")
        return {"n": n}

    diffs = sub["failure_time"] - sub["hallucination_onset_generation"]
    tau, tau_p = kendalltau(sub["hallucination_onset_generation"], sub["failure_time"])

    result = {"n": n, "tau": tau, "tau_p": tau_p, "median_offset_generations": diffs.median()}
    if (diffs != 0).any():
        w_stat, w_p = wilcoxon(diffs)
        result["wilcoxon_stat"], result["wilcoxon_p"] = w_stat, w_p
        print(f"  {label}n={n}  Kendall's tau={tau:.3f} (p={tau_p:.4f})  "
              f"Wilcoxon on (failure_time - hallu_onset): stat={w_stat:.2f} (p={w_p:.4f})  "
              f"median offset={diffs.median():.1f} generations")
    else:
        print(f"  {label}n={n}  Kendall's tau={tau:.3f} (p={tau_p:.4f})  "
              f"Wilcoxon skipped (all paired differences are exactly 0)")
    return result


def try_cox_time_varying(merged):
    try:
        from lifelines import CoxTimeVaryingFitter
    except ImportError:
        print("\n[Cox model skipped] `lifelines` is not installed. "
              "Run `pip install lifelines` and re-run this script to get the "
              "hallucination-onset hazard ratio (stratified by dseed and generator).")
        return None

    rows = []
    for _, r in merged.iterrows():
        start, stop = 0.0, float(r["failure_time"])
        if stop <= start:
            continue  # a chain that "fails" at generation 0 has no risk interval to model
        h = r["hallucination_onset_generation"]
        event = int(r["failure_event"])

        if pd.isna(h) or h >= stop:
            rows.append({"generator": r["generator"], "dseed": r["dseed"], "mseed": r["mseed"],
                         "start": start, "stop": stop, "event": event, "hallucination_active": 0})
        elif h <= start:
            rows.append({"generator": r["generator"], "dseed": r["dseed"], "mseed": r["mseed"],
                         "start": start, "stop": stop, "event": event, "hallucination_active": 1})
        else:
            rows.append({"generator": r["generator"], "dseed": r["dseed"], "mseed": r["mseed"],
                         "start": start, "stop": h, "event": 0, "hallucination_active": 0})
            rows.append({"generator": r["generator"], "dseed": r["dseed"], "mseed": r["mseed"],
                         "start": h, "stop": stop, "event": event, "hallucination_active": 1})

    long = pd.DataFrame(rows)
    if long.empty or long["event"].sum() == 0:
        print("\n[Cox model skipped] no usable risk intervals / no events after filtering.")
        return None

    # Stratified by dseed AND generator, not dseed alone: generator identity is
    # a real confound here -- DDPM/DDPM_CORE are simultaneously the chains most
    # likely to cross hallucination onset and, independently, among the least
    # likely to ever fail at all (5/15 and 9/15 Failure events, versus up to
    # 15/15 for RTVAE/RTVAE_CORE). Stratifying by dseed only cannot separate
    # "hallucinating changes this chain's hazard" from "the generator that
    # tends to hallucinate was already a different-hazard generator for
    # unrelated reasons" -- adding generator as a second stratum controls for
    # that directly, at the cost of finer (dseed, generator) strata with fewer
    # chains each.
    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(
            long, id_col=None, event_col="event", start_col="start", stop_col="stop",
            strata=["dseed", "generator"], formula="hallucination_active",
        )
    except TypeError:
        # older lifelines versions take covariate columns positionally rather than a formula
        long = long[["dseed", "generator", "start", "stop", "event", "hallucination_active"]]
        ctv.fit(long, event_col="event", start_col="start", stop_col="stop", strata=["dseed", "generator"])

    print("\nCox time-varying model (hallucination_active, stratified by dseed and generator):")
    preferred_cols = ["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    cols = [c for c in preferred_cols if c in ctv.summary.columns]
    if len(cols) < len(preferred_cols):
        # column names have varied slightly across lifelines versions -- fall
        # back to the full summary table rather than silently dropping the
        # confidence interval columns this was added for.
        print(ctv.summary.to_string())
    else:
        print(ctv.summary[cols].to_string())
        print("(exp(coef) is the hazard ratio; the two 'exp(coef) ... 95%' columns "
              "are its 95% confidence interval -- a wide interval means the point "
              "estimate is imprecise, not necessarily that there's no effect.)")
    return ctv


def run_full_analysis(merged, metric, verbose_header=""):
    if verbose_header:
        print(verbose_header)

    counts = merged["dissociation_category"].value_counts()
    print("Per-chain dissociation breakdown:")
    for cat in ["both__hallucination_before_failure", "both__same_generation",
                "both__hallucination_after_failure", "failure_only",
                "hallucination_only", "neither"]:
        print(f"  {cat}: {int(counts.get(cat, 0))}")

    both = merged[merged["dissociation_category"].str.startswith("both__", na=False)]
    print("\nPaired timing tests (chains where both Failure and hallucination onset occurred):")
    result = paired_timing_tests(both)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="expert_any", choices=["expert_any", "hr_b5"],
                         help="Which hallucination_onset.csv metric to test against "
                              "Failure onset (default: expert_any).")
    args = parser.parse_args()

    merged = load_and_merge(args.metric)
    merged["dissociation_category"] = merged.apply(categorize, axis=1)

    os.makedirs(STATS_DIR, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"[OK] Saved per-chain table: {OUT_FILE}  ({len(merged)} chains)\n")

    run_full_analysis(merged, args.metric, verbose_header=f"=== Full population (metric={args.metric}) ===")
    try_cox_time_varying(merged)

    print("\n\n=== Leave-one-dseed-out sensitivity ===")
    for dseed in sorted(merged["dseed"].unique()):
        sub = merged[merged["dseed"] != dseed]
        both = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        print(f"\n-- excluding dseed={dseed} ({len(sub)} chains remain) --")
        paired_timing_tests(both)

    print("\n\n=== Leave-one-mseed-out sensitivity ===")
    for mseed in sorted(merged["mseed"].unique()):
        sub = merged[merged["mseed"] != mseed]
        both = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        print(f"\n-- excluding mseed={mseed} ({len(sub)} chains remain) --")
        paired_timing_tests(both)


if __name__ == "__main__":
    main()