"""
Generator failure-speed ranking and cross-seed consistency (Claim 3).

Question: how fast do the generators fail, and is that ranking consistent
across seed-pairs or effectively random?

Four rankings are computed, never pooled together: one across the 4 base
generators (ARF, CTGAN, DDPM, RTVAE), one across their 4 _CORE
counterparts (ARF_CORE, CTGAN_CORE, DDPM_CORE, RTVAE_CORE), and one more
of each kind with RTVAE excluded (base_excl_rtvae, core_excl_rtvae).
RTVAE fails fastest in nearly every seed-pair, so a 4-generator ranking
mostly tests whether RTVAE separates from the other three rather than
whether ARF, CTGAN, and DDPM have a consistent order among themselves;
the 3-generator groups isolate that second question. Each of the 15
(dseed, mseed) seed-pairs is treated as one "rater" that orders the
generators in a group by time-to-failure (failure_events.csv's time
column); Kendall's coefficient of concordance (W) then measures how much
the 15 seed-pairs agree with each other. W=1 means every seed-pair ranks
the generators identically; W=0 means the seed-pairs' rankings are no more
similar to each other than random orderings would be. The corresponding
Friedman chi-square test (df = n_generators-1) tests the null hypothesis
that a seed-pair's ranking is unrelated to the generator identities (pure
noise); the 15 seed-pairs are not fully independent sampling units for
this test, since they form a crossed 3-data-seed x 5-model-seed design
rather than 15 free replicates, which is disclosed separately as a
limitation on the resulting p-values rather than folded into the null
hypothesis wording here.

Ranking convention within a seed-pair/group: a generator that failed
(event=1) is ranked by its time-to-failure, fastest first, with the usual
mid-rank for exact ties. A generator that never failed within the run
(event=0, censored) is always ranked slower than every generator that did
fail in that seed-pair -- not by its censoring time, since "never failed"
is not a point on the same timeline as an observed failure. Multiple
censored generators in the same seed-pair/group tie for the remaining
rank slot(s) and share the mid-rank of that block, the same convention
used for tied failure times. A censored chain whose own follow-up is
incomplete (time < LAST_GENERATION, i.e. it ran out of data before
generation 19 rather than surviving to the end of the run) is flagged as
a caveat rather than silently ranked -- its "did not fail" status is
based on less evidence than a censored chain that was followed the whole
way through.

Inputs: failure_events.csv (produced by 3_failure_events.py) -- generator,
dseed, mseed, chain, time, event, mechanism, n_tstr_datapoints,
insufficient_tstr_data. Chains flagged insufficient_tstr_data=True are
NOT excluded here (failure_events.py's own censoring already reflects the
limited data it had to work with); they are carried through like any
other censored chain, since dropping them would silently shrink the
denominator the same way an unindexed metric file would.

Output: claim3_ranking.csv, long format -- group, dseed, mseed, generator,
rank (one row per generator per seed-pair per group). Console output adds,
per group: Kendall's W and its Friedman significance test, each
generator's mean/median/min/max rank across seed-pairs and how often it
was ranked fastest-to-fail, and leave-one-dseed-out / leave-one-mseed-out
sensitivity of W (consistent with the sensitivity checks used elsewhere
in this project).
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
from scipy.stats import chi2, rankdata

from config.pipeline_config import RESULTS_ROOT
from metric_config import LAST_GENERATION

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

FAILURE_EVENTS_FILE = os.path.join(STATS_DIR, "failure_events.csv")
OUT_FILE = os.path.join(STATS_DIR, "claim3_ranking.csv")

GROUPS = {
    "base": ["ARF", "CTGAN", "DDPM", "RTVAE"],
    "core": ["ARF_CORE", "CTGAN_CORE", "DDPM_CORE", "RTVAE_CORE"],
    "base_excl_rtvae": ["ARF", "CTGAN", "DDPM"],
    "core_excl_rtvae": ["ARF_CORE", "CTGAN_CORE", "DDPM_CORE"],
}


def rank_one_seed_pair(sub):
    """
    sub: DataFrame of exactly len(group) rows (one per generator in the
    group) for one (dseed, mseed) pair, columns include 'time' and
    'event'. Returns a Series of ranks aligned to sub's index: 1 = failed
    fastest. See module docstring for the censored-chain convention.
    """
    n = len(sub)
    failed = sub[sub["event"] == 1]
    censored = sub[sub["event"] == 0]

    ranks = pd.Series(index=sub.index, dtype=float)
    if len(failed):
        ranks.loc[failed.index] = rankdata(failed["time"].values, method="average")
    if len(censored):
        k = len(failed)
        block = np.arange(k + 1, n + 1)
        ranks.loc[censored.index] = block.mean()
    return ranks


def build_rank_matrix(df, group_generators):
    """
    df: failure_events.csv rows, already restricted to group_generators.
    Returns (rank_matrix, meta): rank_matrix has one row per (dseed,
    mseed) seed-pair and one column per generator; meta records, per
    seed-pair, whether all group_generators were present (should always
    be true given the pipeline's chain registry, but checked rather than
    assumed) and whether any censored chain there has incomplete
    follow-up.
    """
    rank_rows = {}
    meta_rows = []

    for (dseed, mseed), sub in df.groupby(["dseed", "mseed"], sort=False):
        sub = sub.drop_duplicates(subset="generator").set_index("generator").reindex(group_generators)
        missing = sub["time"].isna()

        if missing.any():
            meta_rows.append({
                "dseed": dseed, "mseed": mseed, "complete": False,
                "missing_generators": ", ".join(sub.index[missing]),
                "incomplete_censoring": False,
            })
            rank_rows[(dseed, mseed)] = pd.Series({g: np.nan for g in group_generators})
            continue

        sub = sub.reset_index()
        r = rank_one_seed_pair(sub)
        r.index = sub["generator"].values
        rank_rows[(dseed, mseed)] = r.reindex(group_generators)

        incomplete = bool(((sub["event"] == 0) & (sub["time"] < LAST_GENERATION)).any())
        meta_rows.append({
            "dseed": dseed, "mseed": mseed, "complete": True,
            "missing_generators": "", "incomplete_censoring": incomplete,
        })

    rank_matrix = pd.DataFrame.from_dict(rank_rows, orient="index")
    rank_matrix.index = pd.MultiIndex.from_tuples(rank_matrix.index, names=["dseed", "mseed"])
    rank_matrix = rank_matrix[group_generators]
    meta = pd.DataFrame(meta_rows)
    return rank_matrix, meta


def kendalls_w(rank_matrix):
    """
    rank_matrix: raters (seed-pairs) x items (generators) of ranks
    (ties as mid-ranks). Rows with any NaN (an incomplete seed-pair) are
    dropped, not treated as zero. Tie-corrected per the standard formula:
    W = 12*S / (m^2*(n^3-n) - m*sum(t^3-t)), summed over each rater's tied
    groups. chi2 = m*(n-1)*W with df=n-1 is the Friedman-equivalent test
    of independence across raters.
    """
    rm = rank_matrix.dropna(axis=0, how="any")
    m, n = rm.shape
    if m == 0 or n < 2:
        return {"W": float("nan"), "m": m, "n": n, "chi2": float("nan"), "df": max(n - 1, 0), "p": float("nan")}

    R = rm.sum(axis=0).values
    S = float(((R - R.mean()) ** 2).sum())

    tie_term = 0.0
    for _, row in rm.iterrows():
        _, counts = np.unique(row.values, return_counts=True)
        tie_term += float(((counts ** 3) - counts).sum())

    denom = (m ** 2) * (n ** 3 - n) - m * tie_term
    W = (12 * S / denom) if denom > 0 else float("nan")

    df = n - 1
    chi2_stat = m * df * W if not np.isnan(W) else float("nan")
    p = float(chi2.sf(chi2_stat, df)) if not np.isnan(chi2_stat) else float("nan")

    return {"W": W, "m": m, "n": n, "chi2": chi2_stat, "df": df, "p": p}


def summarize_group(label, rank_matrix, meta):
    print(f"\n{'=' * 70}\n{label} generators: {list(rank_matrix.columns)}\n{'=' * 70}")

    incomplete = meta[~meta["complete"]]
    if not incomplete.empty:
        print(f"  [WARN] {len(incomplete)} seed-pair(s) missing a generator in this "
              f"group -- excluded from the ranking below:")
        for _, r in incomplete.iterrows():
            print(f"    d{r['dseed']} / m{r['mseed']}: missing {r['missing_generators']}")

    flagged = meta[meta["incomplete_censoring"]]
    if not flagged.empty:
        print(f"  [CAVEAT] {len(flagged)} seed-pair(s) have a censored chain with "
              f"less than full follow-up (time < {LAST_GENERATION}) in this group -- "
              f"its 'did not fail' rank rests on partial data:")
        for _, r in flagged.iterrows():
            print(f"    d{r['dseed']} / m{r['mseed']}")

    result = kendalls_w(rank_matrix)
    print(f"\n  Kendall's W = {result['W']:.4f}  (m={result['m']} seed-pairs, n={result['n']} generators)")
    print(f"  Friedman chi2 = {result['chi2']:.4f}, df={result['df']}, p = {result['p']:.4g}")
    if not np.isnan(result["W"]):
        if result["p"] < 0.05:
            print("  -> ranking is significantly more consistent across seed-pairs than chance.")
        else:
            print("  -> not significantly different from a random/inconsistent ranking at alpha=0.05.")

    rm = rank_matrix.dropna(axis=0, how="any")
    print("\n  Per-generator rank across seed-pairs (1 = fails fastest):")
    agg = rm.agg(["mean", "median", "min", "max"]).T
    agg["n_ranked_1st"] = (rm == 1).sum(axis=0)
    agg = agg.sort_values("mean")
    for generator, r in agg.iterrows():
        print(f"    {generator:12s} mean={r['mean']:.2f}  median={r['median']:.1f}  "
              f"range=[{r['min']:.1f},{r['max']:.1f}]  ranked-fastest in "
              f"{int(r['n_ranked_1st'])}/{len(rm)}")

    return result, agg


def leave_one_out_sensitivity(df, group_generators, label):
    dseeds = sorted(df["dseed"].unique())
    mseeds = sorted(df["mseed"].unique())

    print(f"\n  Leave-one-dseed-out sensitivity ({label}):")
    for d in dseeds:
        rm, _ = build_rank_matrix(df[df["dseed"] != d], group_generators)
        res = kendalls_w(rm)
        print(f"    excl dseed={d}: W={res['W']:.4f}  p={res['p']:.4g}  (m={res['m']})")

    print(f"\n  Leave-one-mseed-out sensitivity ({label}):")
    for ms in mseeds:
        rm, _ = build_rank_matrix(df[df["mseed"] != ms], group_generators)
        res = kendalls_w(rm)
        print(f"    excl mseed={ms}: W={res['W']:.4f}  p={res['p']:.4g}  (m={res['m']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-loo", action="store_true",
                         help="skip the leave-one-dseed/mseed-out sensitivity checks")
    args = parser.parse_args()

    df = pd.read_csv(FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    all_out = []
    for label, group_generators in GROUPS.items():
        sub = df[df["generator"].isin(group_generators)].copy()
        rank_matrix, meta = build_rank_matrix(sub, group_generators)
        summarize_group(label, rank_matrix, meta)

        long = rank_matrix.reset_index().melt(
            id_vars=["dseed", "mseed"], var_name="generator", value_name="rank"
        )
        long.insert(0, "group", label)
        all_out.append(long)

        if not args.skip_loo:
            leave_one_out_sensitivity(sub, group_generators, label)

    out = pd.concat(all_out, ignore_index=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved: {OUT_FILE}  ({len(out)} rows)")


if __name__ == "__main__":
    main()