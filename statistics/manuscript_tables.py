"""
Manuscript-ready result tables for Claims 1, 2, and 3 -- clean,
reader-facing tables meant to be pasted directly into the paper and its
Supplementary Information, as distinct from appendix_tables.py's raw,
full-disclosure tables.

Every number here comes from calling the exact same functions
claim1_hallucination_utility.py, claim2_hallucination_collapse.py, and
claim3_timetofailure_ranking.py already use to compute and print their
results (imported here as modules, not reimplemented), the same way
appendix_tables.py does -- so a manuscript number can never drift from
what running those scripts directly reports. This script only reshapes
and relabels two things:

  1. Repeated sensitivity tests. Each claim's full-population test and its
     leave-one-dseed-out / leave-one-mseed-out re-tests are the same
     statistical test run on different subsets, so they are merged into
     one table per claim with a "Subset" column identifying each row
     ("Full population", "Excluding data seed 14930352", ...) instead of
     being left as three or four separate tables.

  2. Column and category names. Code-facing tokens (tau_p, wilcoxon_stat,
     cmp to, -log2(p), both__hallucination_before_failure, expert_any, ...)
     are renamed to plain-language labels a reader sees in the paper
     (Kendall's tau p-value, Wilcoxon statistic, Hazard ratio, "Both
     observed -- hallucination onset before failure", "Hallucination rate
     (Expert_any)", ...). No value is altered, dropped, or recomputed by
     this renaming.

Must be run after the three results scripts (1_hallucination_onset.py,
2_qametric_onset.py, 3_failure_events.py), the same prerequisite
appendix_tables.py has, since it imports the same claim modules that read
those results files.

Tables produced (one CSV each, in STATISTICS_TABLES_DIR, plus
a combined .xlsx workbook if openpyxl is available):
  claim1_dissociation_categories   -- per-chain category counts, relabeled
  claim1_paired_timing             -- merged Subset table (full population +
                                       leave-one-dseed-out +
                                       leave-one-mseed-out), Kendall's tau +
                                       Wilcoxon, relabeled
  claim1_cox_model                 -- Cox time-varying hazard ratio,
                                       relabeled, with the excluded-chain
                                       count and stratification carried as
                                       columns so the table is self-
                                       contained without a separate footnote
  claim2_metric_crossing_counts    -- n crossed / total per metric, relabeled
  claim2_paired_timing             -- merged Subset table (pooled full
                                       population + leave-one-dseed-out +
                                       leave-one-mseed-out) per QA metric
  claim2_per_generator_breakdown   -- the same pooled comparison broken out
                                       per generator x QA metric, relabeled
  claim3_rank_summary              -- per-generator mean/median/min/max
                                       rank, base + core groups, relabeled
  claim3_kendalls_w                -- merged Subset table (full population +
                                       leave-one-dseed-out +
                                       leave-one-mseed-out) of Kendall's W,
                                       base + core groups
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

from config.pipeline_config import STATISTICS_TABLES_DIR

OUT_DIR = STATISTICS_TABLES_DIR

import claim1_hallucination_utility as c1
import claim2_hallucination_collapse as c2
import claim3_timetofailure_ranking as c3
import claim3_survival_comparison as c3s

# ============================================================
# Reader-facing labels shared across tables
# ============================================================

METRIC_LABELS = {
    "expert_any": "Hallucination rate (Expert_any)",
    "hr_b5": "Hallucination rate (HR_b5)",
    "hallucination_expert_any": "Hallucination rate (Expert_any)",
    "beta_recall": "Beta-recall",
    "alpha_precision": "Alpha-precision",
    "authenticity": "Authenticity",
}

DIRECTION_LABELS = {"increase": "Increase", "decrease": "Decrease"}

CATEGORY_LABELS = {
    "both__hallucination_before_failure": "Both observed -- hallucination onset before failure",
    "both__same_generation": "Both observed -- same generation",
    "both__hallucination_after_failure": "Both observed -- hallucination onset after failure",
    "failure_only": "Failure only (no hallucination onset detected)",
    "hallucination_only": "Hallucination onset only (no failure observed)",
    "neither": "Neither event observed",
}

# Shared across Claim 1's paired timing test and Claim 2's paired
# generation test -- both return the same shape of result dict from
# their respective claim modules.
TIMING_COLUMN_ORDER = [
    "Subset", "N chains", "Kendall's tau", "Kendall's tau p-value",
    "Wilcoxon statistic", "Wilcoxon p-value", "Median offset (generations)",
]

COX_COLUMN_LABELS = {
    "covariate": "Covariate",
    "exp(coef)": "Hazard ratio",
    "exp(coef) lower 95%": "95% CI (lower)",
    "exp(coef) upper 95%": "95% CI (upper)",
    "p": "p-value",
}

COX_COLUMN_ORDER = [
    "Covariate", "Hazard ratio", "95% CI (lower)", "95% CI (upper)", "p-value",
    "Chains in model", "Exclusion note", "Stratification",
]


def _subset_label(excluded_dseed=None, excluded_mseed=None):
    if excluded_dseed is None and excluded_mseed is None:
        return "Full population"
    if excluded_dseed is not None:
        return f"Excluding data seed {excluded_dseed}"
    return f"Excluding model seed {excluded_mseed}"


# A cell in the timing tables can be empty for two structurally different
# reasons, and a blank cell alone doesn't tell a reader which one applies:
# too few chains to test at all (no pair to compare), or enough chains but
# a genuine mathematical degeneracy (Kendall's tau is undefined when one
# of the two variables being correlated has no variance across chains;
# the paired Wilcoxon test is not run at all when every paired offset is
# exactly 0, since there is nothing to rank). Each case gets its own
# explanatory string instead of a blank, so the table is self-explanatory
# without a separate footnote.
NO_PAIR_LABEL = "n/a (n<2 chains with both onsets defined)"
NO_VARIANCE_LABEL = "n/a (no variance in onset generation)"
ZERO_OFFSET_LABEL = "n/a (all offsets = 0 generations)"


def _format_timing_result(result_dict):
    """
    Reshapes a paired_timing_tests() / paired_generation_tests() result
    dict into reader-facing columns. See the module-level comment above
    for what each placeholder string means.
    """
    out = {"N chains": result_dict.get("n")}

    if "tau" not in result_dict:
        # n < 2: the claim module returned immediately with only {"n": n}
        # -- no statistic was attempted.
        out["Kendall's tau"] = NO_PAIR_LABEL
        out["Kendall's tau p-value"] = NO_PAIR_LABEL
        out["Wilcoxon statistic"] = NO_PAIR_LABEL
        out["Wilcoxon p-value"] = NO_PAIR_LABEL
        out["Median offset (generations)"] = NO_PAIR_LABEL
        return out

    tau, tau_p = result_dict["tau"], result_dict["tau_p"]
    if pd.isna(tau):
        # n >= 2, but one of the two paired variables is constant across
        # every chain in this subset -- Kendall's tau has no defined
        # value here, not merely an uncomputed one.
        out["Kendall's tau"] = NO_VARIANCE_LABEL
        out["Kendall's tau p-value"] = NO_VARIANCE_LABEL
    else:
        out["Kendall's tau"] = tau
        out["Kendall's tau p-value"] = tau_p

    if "wilcoxon_stat" not in result_dict:
        # Every paired offset is exactly 0 -- the claim module skips the
        # Wilcoxon test by design in this case.
        out["Wilcoxon statistic"] = ZERO_OFFSET_LABEL
        out["Wilcoxon p-value"] = ZERO_OFFSET_LABEL
    else:
        out["Wilcoxon statistic"] = result_dict["wilcoxon_stat"]
        out["Wilcoxon p-value"] = result_dict["wilcoxon_p"]

    out["Median offset (generations)"] = result_dict["median_offset_generations"]
    return out


def _reorder(df, preferred_order):
    cols = [c for c in preferred_order if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


# ============================================================
# Claim 1
# ============================================================

def claim1_dissociation_categories_table(merged):
    counts = merged["dissociation_category"].value_counts()
    order = ["both__hallucination_before_failure", "both__same_generation",
              "both__hallucination_after_failure", "failure_only",
              "hallucination_only", "neither"]
    return pd.DataFrame({
        "Category": [CATEGORY_LABELS[c] for c in order],
        "N chains": [int(counts.get(c, 0)) for c in order],
    })


def claim1_paired_timing_table(merged):
    rows = []

    both = merged[merged["dissociation_category"].str.startswith("both__", na=False)]
    result = c1.paired_timing_tests(both)
    rows.append({"Subset": _subset_label(), **_format_timing_result(result)})

    for dseed in sorted(merged["dseed"].unique()):
        sub = merged[merged["dseed"] != dseed]
        both_sub = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        r = c1.paired_timing_tests(both_sub)
        rows.append({"Subset": _subset_label(excluded_dseed=dseed), **_format_timing_result(r)})

    for mseed in sorted(merged["mseed"].unique()):
        sub = merged[merged["mseed"] != mseed]
        both_sub = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        r = c1.paired_timing_tests(both_sub)
        rows.append({"Subset": _subset_label(excluded_mseed=mseed), **_format_timing_result(r)})

    return _reorder(pd.DataFrame(rows), TIMING_COLUMN_ORDER)


def claim1_cox_model_table(merged):
    ctv = c1.try_cox_time_varying(merged)
    if ctv is None:
        return pd.DataFrame([{
            "Note": "Cox model not available -- `lifelines` was not installed "
                    "in the environment this table was last generated in. "
                    "Install it (`pip install lifelines`) and re-run this "
                    "script for the hazard-ratio table."
        }])

    summary = ctv.summary.reset_index().rename(columns=COX_COLUMN_LABELS)

    n_at_risk = int((merged["failure_time"] > 0).sum())
    n_total = len(merged)
    n_excluded = n_total - n_at_risk
    summary["Chains in model"] = f"{n_at_risk} of {n_total}"
    summary["Exclusion note"] = (
        f"{n_excluded} chain(s) excluded (failure observed at generation 0 "
        f"leaves no risk interval to model)" if n_excluded else "none excluded"
    )
    summary["Stratification"] = "Data seed and generator"

    # Select only the reader-facing columns -- lifelines' summary carries
    # several more (coef, se(coef), cmp to, z, -log2(p), ...) that duplicate
    # or derive from the ones kept here; the full set is already disclosed
    # in appendix_tables.py's claim1_cox_model table.
    return summary[[c for c in COX_COLUMN_ORDER if c in summary.columns]]


def build_claim1_tables():
    merged = c1.load_and_merge("expert_any")
    merged["dissociation_category"] = merged.apply(c1.categorize, axis=1)
    return {
        "claim1_dissociation_categories": claim1_dissociation_categories_table(merged),
        "claim1_paired_timing": claim1_paired_timing_table(merged),
        "claim1_cox_model": claim1_cox_model_table(merged),
    }


# ============================================================
# Claim 2
# ============================================================

def claim2_metric_crossing_counts_table(spike_tables):
    hallu_tbl = spike_tables["hallucination_expert_any"]
    rows = [{
        "Metric": METRIC_LABELS["hallucination_expert_any"],
        "Threshold": c2.HALLUCINATION_T,
        "Direction of change at onset": DIRECTION_LABELS["increase"],
        "N chains crossed": int(hallu_tbl["spike_generation"].notna().sum()),
        "N chains total": len(hallu_tbl),
    }]
    for label in c2.QA_METRICS:
        cfg = c2.QA_METRICS_CONFIG[label]
        tbl = spike_tables[label]
        rows.append({
            "Metric": METRIC_LABELS[label],
            "Threshold": cfg["threshold"],
            "Direction of change at onset": DIRECTION_LABELS[cfg["direction"]],
            "N chains crossed": int(tbl["spike_generation"].notna().sum()),
            "N chains total": len(tbl),
        })
    return pd.DataFrame(rows)


def claim2_paired_timing_table(spike_tables, generators):
    hallu_tbl = spike_tables["hallucination_expert_any"]
    dseeds = sorted(set(hallu_tbl["dseed"]))
    mseeds = sorted(set(hallu_tbl["mseed"]))

    rows = []
    for qa in c2.QA_METRICS:
        _, r = c2.paired_generation_tests(hallu_tbl, spike_tables[qa],
                                           "hallucination_expert_any", qa)
        rows.append({"QA metric": METRIC_LABELS[qa], "Subset": _subset_label(),
                     **_format_timing_result(r)})

    for dseed in dseeds:
        a = hallu_tbl[hallu_tbl["dseed"] != dseed]
        for qa in c2.QA_METRICS:
            b = spike_tables[qa][spike_tables[qa]["dseed"] != dseed]
            _, r = c2.paired_generation_tests(a, b, "hallucination_expert_any", qa)
            rows.append({"QA metric": METRIC_LABELS[qa],
                         "Subset": _subset_label(excluded_dseed=dseed),
                         **_format_timing_result(r)})

    for mseed in mseeds:
        a = hallu_tbl[hallu_tbl["mseed"] != mseed]
        for qa in c2.QA_METRICS:
            b = spike_tables[qa][spike_tables[qa]["mseed"] != mseed]
            _, r = c2.paired_generation_tests(a, b, "hallucination_expert_any", qa)
            rows.append({"QA metric": METRIC_LABELS[qa],
                         "Subset": _subset_label(excluded_mseed=mseed),
                         **_format_timing_result(r)})

    df = pd.DataFrame(rows)
    # "N chains" here specifically means chains where both the
    # hallucination spike and this QA metric's spike were defined -- kept
    # distinct from claim2_metric_crossing_counts's "N chains crossed" /
    # "N chains total", which count each metric separately.
    df = df.rename(columns={"N chains": "N with both onsets defined"})
    order = ["QA metric", "Subset", "N with both onsets defined"] + TIMING_COLUMN_ORDER[2:]
    return _reorder(df, order)


def claim2_per_generator_breakdown_table(spike_tables, generators):
    hallu_tbl = spike_tables["hallucination_expert_any"]
    rows = []
    for generator in generators:
        g_hallu = hallu_tbl[hallu_tbl["generator"] == generator]
        for qa in c2.QA_METRICS:
            g_qa = spike_tables[qa][spike_tables[qa]["generator"] == generator]
            _, r = c2.paired_generation_tests(g_hallu, g_qa, "hallucination_expert_any", qa)
            rows.append({
                "Generator": generator,
                "QA metric": METRIC_LABELS[qa],
                "N chains total": len(g_hallu),
                "N with hallucination onset": int(g_hallu["spike_generation"].notna().sum()),
                "N with QA-metric onset": int(g_qa["spike_generation"].notna().sum()),
                **_format_timing_result(r),
            })
    df = pd.DataFrame(rows)
    df = df.rename(columns={"N chains": "N with both onsets defined"})
    order = ["Generator", "QA metric", "N chains total", "N with hallucination onset",
             "N with QA-metric onset", "N with both onsets defined"] + TIMING_COLUMN_ORDER[2:]
    return _reorder(df, order)


def build_claim2_tables():
    generators = c2.GENERATORS
    spike_tables = c2.load_onset_tables(generators)
    return {
        "claim2_metric_crossing_counts": claim2_metric_crossing_counts_table(spike_tables),
        "claim2_paired_timing": claim2_paired_timing_table(spike_tables, generators),
        "claim2_per_generator_breakdown": claim2_per_generator_breakdown_table(spike_tables, generators),
    }


# ============================================================
# Claim 3
# ============================================================

GROUP_LABELS = {
    "base": "Full dataset",
    "core": "Core dataset",
    "base_excl_rtvae": "Full dataset, excluding RTVAE",
    "core_excl_rtvae": "Core dataset, excluding RTVAE",
}

RANK_SUMMARY_COLUMN_LABELS = {
    "generator": "Generator",
    "mean": "Mean rank",
    "median": "Median rank",
    "min": "Min rank",
    "max": "Max rank",
    "n_ranked_1st": "N ranked fastest-to-fail",
    "n_seed_pairs": "N seed-pairs",
}

W_COLUMN_LABELS = {
    "W": "Kendall's W",
    "m": "N seed-pairs",
    "n": "N generators",
    "chi2": "Chi-square statistic",
    "df": "Degrees of freedom",
    "p": "p-value",
}

# kendalls_w() always returns every key, but W/chi2/p can be nan (m=0, no
# complete seed-pairs in this subset, or n<2 generators to rank) -- same
# "undefined, not merely uncomputed" situation as Kendall's tau above.
NO_RANKING_LABEL = "n/a (fewer than 2 seed-pairs or generators with a complete ranking)"


def _format_w_result(result_dict):
    out = {W_COLUMN_LABELS.get(k, k): v for k, v in result_dict.items()}
    if pd.isna(result_dict.get("W")):
        for col in ("Kendall's W", "Chi-square statistic", "p-value"):
            out[col] = NO_RANKING_LABEL
    return out


def claim3_rank_summary_table():
    df = pd.read_csv(c3.FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    rows = []
    for label, group_generators in c3.GROUPS.items():
        sub = df[df["generator"].isin(group_generators)].copy()
        rank_matrix, _meta = c3.build_rank_matrix(sub, group_generators)
        rm = rank_matrix.dropna(axis=0, how="any")
        agg = rm.agg(["mean", "median", "min", "max"]).T
        agg["n_ranked_1st"] = (rm == 1).sum(axis=0)
        agg["n_seed_pairs"] = len(rm)
        agg = agg.reset_index().rename(columns={"index": "generator"})
        agg = agg.rename(columns=RANK_SUMMARY_COLUMN_LABELS)
        agg.insert(0, "Group", GROUP_LABELS[label])
        rows.append(agg)

    return pd.concat(rows, ignore_index=True)


def claim3_kendalls_w_table():
    df = pd.read_csv(c3.FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    rows = []
    for label, group_generators in c3.GROUPS.items():
        sub = df[df["generator"].isin(group_generators)].copy()

        rank_matrix, _meta = c3.build_rank_matrix(sub, group_generators)
        result = c3.kendalls_w(rank_matrix)
        rows.append({"Group": GROUP_LABELS[label], "Subset": _subset_label(),
                     **_format_w_result(result)})

        for dseed in sorted(sub["dseed"].unique()):
            rm_d, _ = c3.build_rank_matrix(sub[sub["dseed"] != dseed], group_generators)
            r = c3.kendalls_w(rm_d)
            rows.append({"Group": GROUP_LABELS[label],
                         "Subset": _subset_label(excluded_dseed=dseed),
                         **_format_w_result(r)})

        for mseed in sorted(sub["mseed"].unique()):
            rm_m, _ = c3.build_rank_matrix(sub[sub["mseed"] != mseed], group_generators)
            r = c3.kendalls_w(rm_m)
            rows.append({"Group": GROUP_LABELS[label],
                         "Subset": _subset_label(excluded_mseed=mseed),
                         **_format_w_result(r)})

    df_out = pd.DataFrame(rows)
    order = ["Group", "Subset", "Kendall's W", "N seed-pairs", "N generators",
             "Chi-square statistic", "Degrees of freedom", "p-value"]
    return _reorder(df_out, order)


def claim3_survival_comparison_table():
    df = pd.read_csv(c3.FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    rows = []
    for label, group_generators in c3.GROUPS.items():
        rows += c3s.run_group_tests(df, group_generators, GROUP_LABELS[label])

    out = pd.DataFrame(rows)
    order = ["Group", "Subset", "Test", "Comparison", "Statistic", "df", "p", "p_holm"]
    return _reorder(out, order)


def build_claim3_tables():
    return {
        "claim3_rank_summary": claim3_rank_summary_table(),
        "claim3_kendalls_w": claim3_kendalls_w_table(),
        "claim3_survival_comparison": claim3_survival_comparison_table(),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    all_tables = {}
    all_tables.update(build_claim1_tables())
    all_tables.update(build_claim2_tables())
    all_tables.update(build_claim3_tables())

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n=== Writing manuscript tables to {OUT_DIR} ===")
    for name, tbl in all_tables.items():
        path = os.path.join(OUT_DIR, f"{name}.csv")
        tbl.to_csv(path, index=False)
        print(f"[OK] Saved: {path}  ({len(tbl)} rows)")

    try:
        import openpyxl  # noqa: F401
        xlsx_path = os.path.join(OUT_DIR, "manuscript_tables.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, tbl in all_tables.items():
                sheet_name = name[:31]  # Excel's 31-character sheet-name limit
                tbl.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[OK] Saved combined workbook: {xlsx_path}")
    except ImportError:
        print("[NOTE] `openpyxl` not installed -- the CSVs above are still "
              "complete; install openpyxl and re-run this script for one "
              "combined .xlsx workbook as well.")


if __name__ == "__main__":
    main()