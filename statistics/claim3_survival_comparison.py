"""
Claim 3: do the generators' time-to-failure distributions actually differ,
and does that difference disappear once RTVAE is excluded?

This is a different statistical question from claim3_timetofailure_ranking.py's
Kendall's W: W measures whether the 15 seed-pairs agree on a *relative order*
of the generators, while this script asks whether the underlying
time-to-failure *distributions* (the full survival curves, censoring
included) differ at all. The two are related but not interchangeable, so a
claim about "survival curves differing" needs a survival-curve test behind
it, not a ranking-concordance number cited in its place.

Two tests are run per group, both from failure_events.csv's (generator,
dseed, mseed, time, event) rows, and neither pools GROUPS together:

  1. Log-rank test (omnibus multivariate + pairwise, Holm-adjusted).
     This is the standard test for "do these groups' survival curves
     differ", but it assumes every chain is an independent observation.
     That assumption is false here: each (dseed, mseed) seed-pair supplies
     one chain per generator, so a seed-pair that happens to run "fast"
     or "slow" pushes all of that seed-pair's generators together,
     correlating observations within a seed-pair. The log-rank p-values
     below are reported anyway (they are what the omnibus/pairwise
     comparison conventionally looks like), but flagged as anticonservative
     for that reason.

  2. Stratified Cox proportional-hazards likelihood-ratio test, strata =
     the (dseed, mseed) seed-pair itself. Stratifying on the seed-pair
     restricts the comparison to within-seed-pair contrasts only, which is
     exactly the unit that is NOT independent of itself -- so this is the
     correction for the log-rank test's blind spot above, not a second,
     unrelated test. Its likelihood-ratio p-value is the one that should be
     cited for "these generators' survival curves differ", with the
     log-rank result kept alongside for comparison rather than as the
     primary evidence.

RTVAE and RTVAE_CORE fail in all 15/15 chains (see
claim3_timetofailure_ranking.py's rank summary), which is complete
separation for a covariate in the Cox model: every RTVAE chain has
event=1, so the partial likelihood has no finite maximum along that
covariate's direction. Groups containing RTVAE/RTVAE_CORE therefore get a
second Cox run automatically, excluding whichever generators in that group
fail in 100% of their chains, alongside the primary run (which is left to
fail visibly with a caught, printed warning rather than silently omitted).
This mirrors how the ranking script's base_excl_rtvae/core_excl_rtvae
groups exist for the same reason.

Leave-one-dseed-out / leave-one-mseed-out sensitivity is reported for both
tests, the same convention used throughout this project's statistics
scripts.

Inputs: failure_events.csv (produced by 3_failure_events.py); the same four
groups (base, core, base_excl_rtvae, core_excl_rtvae) defined once in
claim3_timetofailure_ranking.py and imported here, not redefined.

Output: claim3_survival_comparison.csv, long format -- group, subset, test,
comparison, statistic, df, p, p_holm (one row per test per subset per
group; p_holm is only populated for the pairwise log-rank rows).
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

import numpy as np
import pandas as pd

from config.pipeline_config import RESULTS_ROOT
import claim3_timetofailure_ranking as c3

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

OUT_FILE = os.path.join(STATS_DIR, "claim3_survival_comparison.csv")


def holm(pvals):
    """Holm-Bonferroni step-down adjustment, order preserved."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def _subset_label(excluded_dseed=None, excluded_mseed=None):
    if excluded_dseed is None and excluded_mseed is None:
        return "Full population"
    if excluded_dseed is not None:
        return f"Excluding data seed {excluded_dseed}"
    return f"Excluding model seed {excluded_mseed}"


def logrank_rows(sub, group_generators, group_label, subset_label):
    """
    Omnibus multivariate log-rank test across group_generators, plus
    Holm-adjusted pairwise log-rank tests. sub must already be restricted
    to group_generators and to whichever dseed/mseed subset is being
    tested.
    """
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    rows = []
    present = [g for g in group_generators if g in set(sub["generator"])]
    if len(present) < 2:
        rows.append({
            "Group": group_label, "Subset": subset_label, "Test": "Omnibus log-rank",
            "Comparison": "n/a (fewer than 2 generators present in this subset)",
            "Statistic": np.nan, "df": np.nan, "p": np.nan, "p_holm": np.nan,
        })
        return rows

    res = multivariate_logrank_test(sub["time"], sub["generator"], sub["event"])
    rows.append({
        "Group": group_label, "Subset": subset_label, "Test": "Omnibus log-rank",
        "Comparison": " vs ".join(present), "Statistic": float(res.test_statistic),
        "df": int(len(present) - 1), "p": float(res.p_value), "p_holm": np.nan,
    })

    pairs, stats, dfs, ps = [], [], [], []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            two = sub[sub["generator"].isin([a, b])]
            r = logrank_test(
                two.loc[two["generator"] == a, "time"], two.loc[two["generator"] == b, "time"],
                two.loc[two["generator"] == a, "event"], two.loc[two["generator"] == b, "event"],
            )
            pairs.append(f"{a} vs {b}")
            stats.append(float(r.test_statistic))
            dfs.append(1)
            ps.append(float(r.p_value))

    for pair, stat, df_, p, p_holm in zip(pairs, stats, dfs, ps, holm(ps)):
        rows.append({
            "Group": group_label, "Subset": subset_label, "Test": "Pairwise log-rank",
            "Comparison": pair, "Statistic": stat, "df": df_, "p": p, "p_holm": p_holm,
        })
    return rows


def _fit_stratified_cox(sub, generators_for_model, group_label, subset_label, note):
    """
    One stratified Cox PH likelihood-ratio test, strata = the (dseed,
    mseed) seed-pair. Returns one result row, or an "n/a" row with the
    caught exception's message if the fit fails (e.g. residual separation).
    """
    from lifelines import CoxPHFitter

    cox = sub[sub["generator"].isin(generators_for_model)].copy()
    cox["duration"] = cox["time"] + 1  # Cox needs strictly positive time
    cox["seed_pair"] = cox["dseed"].astype(str) + "_" + cox["mseed"].astype(str)
    design = pd.get_dummies(
        cox[["duration", "event", "generator", "seed_pair"]],
        columns=["generator"], drop_first=True, dtype=float,
    )
    n_cov = design.shape[1] - 3  # duration, event, seed_pair are not covariates
    if n_cov < 1:
        return {
            "Group": group_label, "Subset": subset_label,
            "Test": f"Stratified Cox LRT ({note})",
            "Comparison": " vs ".join(generators_for_model), "Statistic": np.nan,
            "df": np.nan, "p": np.nan, "p_holm": np.nan,
        }
    try:
        cph = CoxPHFitter()
        cph.fit(design, duration_col="duration", event_col="event", strata=["seed_pair"])
        lrt = cph.log_likelihood_ratio_test()
        return {
            "Group": group_label, "Subset": subset_label,
            "Test": f"Stratified Cox LRT ({note})",
            "Comparison": " vs ".join(generators_for_model),
            "Statistic": float(lrt.test_statistic), "df": n_cov,
            "p": float(lrt.p_value), "p_holm": np.nan,
        }
    except Exception as err:
        print(f"  [warn] stratified Cox ({note}) failed for {group_label}/{subset_label}: "
              f"{type(err).__name__}: {err}")
        return {
            "Group": group_label, "Subset": subset_label,
            "Test": f"Stratified Cox LRT ({note})",
            "Comparison": " vs ".join(generators_for_model),
            "Statistic": np.nan, "df": np.nan,
            "p": f"n/a (fit failed: {type(err).__name__})", "p_holm": np.nan,
        }


def cox_rows(sub, group_generators, group_label, subset_label):
    """
    Primary stratified Cox LRT across every generator present, plus an
    automatic supplementary run excluding any generator that fails in
    100% of its chains in this subset (complete separation) -- see module
    docstring.
    """
    present = [g for g in group_generators if g in set(sub["generator"])]
    if len(present) < 2:
        return [{
            "Group": group_label, "Subset": subset_label, "Test": "Stratified Cox LRT (all generators)",
            "Comparison": "n/a (fewer than 2 generators present in this subset)",
            "Statistic": np.nan, "df": np.nan, "p": np.nan, "p_holm": np.nan,
        }]

    rows = [_fit_stratified_cox(sub, present, group_label, subset_label, "all generators")]

    always_fail = [g for g in present if sub.loc[sub["generator"] == g, "event"].all()]
    remaining = [g for g in present if g not in always_fail]
    if always_fail and len(remaining) >= 2:
        rows.append(_fit_stratified_cox(
            sub, remaining, group_label, subset_label,
            f"excluding {', '.join(always_fail)}",
        ))
    return rows


def run_group_tests(df, group_generators, group_label, skip_loo=False):
    sub_full = df[df["generator"].isin(group_generators)].copy()
    all_rows = []

    print(f"\n{'=' * 70}\n{group_label}: {group_generators}\n{'=' * 70}")

    def do_subset(sub, subset_label):
        rows = logrank_rows(sub, group_generators, group_label, subset_label)
        rows += cox_rows(sub, group_generators, group_label, subset_label)
        for r in rows:
            p_disp = r["p"] if not isinstance(r["p"], float) else (
                f"{r['p']:.4g}" if not np.isnan(r["p"]) else "n/a")
            print(f"    [{subset_label}] {r['Test']} ({r['Comparison']}): "
                  f"statistic={r['Statistic']}, df={r['df']}, p={p_disp}")
        return rows

    all_rows += do_subset(sub_full, _subset_label())

    if not skip_loo:
        for dseed in sorted(sub_full["dseed"].unique()):
            all_rows += do_subset(sub_full[sub_full["dseed"] != dseed],
                                   _subset_label(excluded_dseed=dseed))
        for mseed in sorted(sub_full["mseed"].unique()):
            all_rows += do_subset(sub_full[sub_full["mseed"] != mseed],
                                   _subset_label(excluded_mseed=mseed))

    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-loo", action="store_true",
                         help="skip the leave-one-dseed/mseed-out sensitivity checks")
    args = parser.parse_args()

    df = pd.read_csv(c3.FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    all_rows = []
    for label, group_generators in c3.GROUPS.items():
        all_rows += run_group_tests(df, group_generators, label, skip_loo=args.skip_loo)

    out = pd.DataFrame(all_rows)
    os.makedirs(STATS_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
