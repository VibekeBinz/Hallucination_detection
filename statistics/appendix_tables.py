"""
Appendix result tables for the manuscript -- every number behind Claims
1-3 exported as plain tables (CSV, plus a combined .xlsx workbook if
openpyxl is available) rather than left in console output.

This script does not recompute any statistic itself. Every number here
comes from calling the exact same functions
claim1_hallucination_utility.py, claim2_hallucination_collapse.py, and
claim3_timetofailure_ranking.py already use to print their results (their
filenames are valid Python identifiers, so they're imported here as
modules rather than re-implemented) -- this only collects their return
values into tables instead of printing them, so an appendix number can
never drift from what running those scripts directly reports.

Must be run after the three results scripts (1_hallucination_onset.py,
2_qametric_onset.py, 3_failure_events.py), since that's what this script's
imports read from (STATISTICS_DIR/failure_events.csv, hallucination_onset.csv,
qa_metric_onset.csv). It does NOT require claim1/2/3's own output CSVs to
already exist -- it calls their loading/analysis functions directly.

Tables produced (one CSV each, in STATISTICS_DIR/appendix_tables/):
  metric_calibration                 -- every metric's onset threshold
                                         and direction (methods reference)
  claim1_dissociation_counts         -- per-chain category counts
  claim1_paired_timing_full          -- Kendall's tau + Wilcoxon, full pop
  claim1_cox_model                   -- Cox time-varying model summary
                                         (skipped/noted if lifelines isn't
                                         installed in the environment this
                                         is run in)
  claim1_leave_one_dseed_out         -- paired timing test per excluded dseed
  claim1_leave_one_mseed_out         -- paired timing test per excluded mseed
  claim2_metric_crossing_counts      -- n crossed / total per metric
  claim2_pooled_paired_tests         -- hallucination vs each QA metric, pooled
  claim2_per_generator_breakdown     -- same, broken out per generator x QA metric
  claim2_leave_one_dseed_out         -- pooled paired test per excluded dseed x QA metric
  claim2_leave_one_mseed_out         -- pooled paired test per excluded mseed x QA metric
  claim3_per_generator_rank_summary  -- mean/median/min/max rank, base + core groups
  claim3_kendalls_w                  -- Kendall's W + Friedman test, base + core groups
  claim3_leave_one_dseed_out         -- W per excluded dseed x group
  claim3_leave_one_mseed_out         -- W per excluded mseed x group
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

from config.pipeline_config import RESULTS_ROOT
from metric_config import HALLUCINATION_METRICS, QA_METRICS

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

OUT_DIR = os.path.join(STATS_DIR, "appendix_tables")

import claim1_hallucination_utility as c1
import claim2_hallucination_collapse as c2
import claim3_timetofailure_ranking as c3


def calibration_table():
    rows = []
    for metric, cfg in HALLUCINATION_METRICS.items():
        rows.append({"metric": metric, "threshold": cfg["threshold"], "direction": cfg["direction"]})
    for metric, cfg in QA_METRICS.items():
        rows.append({"metric": metric, "threshold": cfg["threshold"], "direction": cfg["direction"]})
    return {"metric_calibration": pd.DataFrame(rows)}


def claim1_tables():
    tables = {}

    merged = c1.load_and_merge("expert_any")
    merged["dissociation_category"] = merged.apply(c1.categorize, axis=1)

    counts = merged["dissociation_category"].value_counts()
    cats = ["both__hallucination_before_failure", "both__same_generation",
            "both__hallucination_after_failure", "failure_only",
            "hallucination_only", "neither"]
    tables["claim1_dissociation_counts"] = pd.DataFrame(
        {"category": cats, "n_chains": [int(counts.get(c, 0)) for c in cats]}
    )

    both = merged[merged["dissociation_category"].str.startswith("both__", na=False)]
    result = c1.paired_timing_tests(both)
    tables["claim1_paired_timing_full"] = pd.DataFrame([{"scope": "full population", **result}])

    ctv = c1.try_cox_time_varying(merged)
    if ctv is not None:
        summary = ctv.summary.reset_index()
        summary.insert(0, "scope", "full population, stratified by dseed+generator")
        tables["claim1_cox_model"] = summary
    else:
        tables["claim1_cox_model"] = pd.DataFrame(
            [{"note": "Cox model not available -- `lifelines` was not installed "
                       "in the environment this script last ran in. Install it "
                       "and re-run for this table."}]
        )

    loo_dseed_rows = []
    for dseed in sorted(merged["dseed"].unique()):
        sub = merged[merged["dseed"] != dseed]
        both_sub = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        r = c1.paired_timing_tests(both_sub)
        loo_dseed_rows.append({"excluded_dseed": dseed, **r})
    tables["claim1_leave_one_dseed_out"] = pd.DataFrame(loo_dseed_rows)

    loo_mseed_rows = []
    for mseed in sorted(merged["mseed"].unique()):
        sub = merged[merged["mseed"] != mseed]
        both_sub = sub[sub["dissociation_category"].str.startswith("both__", na=False)]
        r = c1.paired_timing_tests(both_sub)
        loo_mseed_rows.append({"excluded_mseed": mseed, **r})
    tables["claim1_leave_one_mseed_out"] = pd.DataFrame(loo_mseed_rows)

    return tables


def claim2_tables():
    tables = {}
    generators = c2.GENERATORS
    spike_tables = c2.load_onset_tables(generators)
    hallu_tbl = spike_tables["hallucination_expert_any"]

    crossing_rows = [{
        "metric": "hallucination_expert_any",
        "threshold": c2.HALLUCINATION_T, "direction": "increase",
        "n_crossed": int(hallu_tbl["spike_generation"].notna().sum()),
        "n_total": len(hallu_tbl),
    }]
    for label in c2.QA_METRICS:
        cfg = c2.QA_METRICS_CONFIG[label]
        tbl = spike_tables[label]
        crossing_rows.append({
            "metric": label, "threshold": cfg["threshold"], "direction": cfg["direction"],
            "n_crossed": int(tbl["spike_generation"].notna().sum()), "n_total": len(tbl),
        })
    tables["claim2_metric_crossing_counts"] = pd.DataFrame(crossing_rows)

    pooled_rows = []
    for qa in c2.QA_METRICS:
        _, r = c2.paired_generation_tests(hallu_tbl, spike_tables[qa], "hallucination_expert_any", qa)
        pooled_rows.append({"qa_metric": qa, **r})
    tables["claim2_pooled_paired_tests"] = pd.DataFrame(pooled_rows)

    per_gen_rows = []
    for generator in generators:
        g_hallu = hallu_tbl[hallu_tbl["generator"] == generator]
        for qa in c2.QA_METRICS:
            g_qa = spike_tables[qa][spike_tables[qa]["generator"] == generator]
            _, r = c2.paired_generation_tests(g_hallu, g_qa, "hallucination_expert_any", qa)
            per_gen_rows.append({
                "generator": generator, "qa_metric": qa,
                "n_total": len(g_hallu),
                "n_hallucination_defined": int(g_hallu["spike_generation"].notna().sum()),
                "n_qa_defined": int(g_qa["spike_generation"].notna().sum()),
                **r,
            })
    tables["claim2_per_generator_breakdown"] = pd.DataFrame(per_gen_rows)

    dseeds = sorted(set(hallu_tbl["dseed"]))
    mseeds = sorted(set(hallu_tbl["mseed"]))

    loo_dseed_rows = []
    for dseed in dseeds:
        a = hallu_tbl[hallu_tbl["dseed"] != dseed]
        for qa in c2.QA_METRICS:
            b = spike_tables[qa][spike_tables[qa]["dseed"] != dseed]
            _, r = c2.paired_generation_tests(a, b, "hallucination_expert_any", qa)
            loo_dseed_rows.append({"excluded_dseed": dseed, "qa_metric": qa, **r})
    tables["claim2_leave_one_dseed_out"] = pd.DataFrame(loo_dseed_rows)

    loo_mseed_rows = []
    for mseed in mseeds:
        a = hallu_tbl[hallu_tbl["mseed"] != mseed]
        for qa in c2.QA_METRICS:
            b = spike_tables[qa][spike_tables[qa]["mseed"] != mseed]
            _, r = c2.paired_generation_tests(a, b, "hallucination_expert_any", qa)
            loo_mseed_rows.append({"excluded_mseed": mseed, "qa_metric": qa, **r})
    tables["claim2_leave_one_mseed_out"] = pd.DataFrame(loo_mseed_rows)

    return tables


def claim3_tables():
    tables = {}

    df = pd.read_csv(c3.FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    rank_summary_rows = []
    w_rows = []
    loo_dseed_rows = []
    loo_mseed_rows = []

    for label, group_generators in c3.GROUPS.items():
        sub = df[df["generator"].isin(group_generators)].copy()
        rank_matrix, _meta = c3.build_rank_matrix(sub, group_generators)
        result = c3.kendalls_w(rank_matrix)
        w_rows.append({"group": label, **result})

        rm = rank_matrix.dropna(axis=0, how="any")
        agg = rm.agg(["mean", "median", "min", "max"]).T
        agg["n_ranked_1st"] = (rm == 1).sum(axis=0)
        agg["n_seed_pairs"] = len(rm)
        agg = agg.reset_index().rename(columns={"index": "generator"})
        agg.insert(0, "group", label)
        rank_summary_rows.append(agg)

        for dseed in sorted(sub["dseed"].unique()):
            rm_d, _ = c3.build_rank_matrix(sub[sub["dseed"] != dseed], group_generators)
            r = c3.kendalls_w(rm_d)
            loo_dseed_rows.append({"group": label, "excluded_dseed": dseed, **r})

        for mseed in sorted(sub["mseed"].unique()):
            rm_m, _ = c3.build_rank_matrix(sub[sub["mseed"] != mseed], group_generators)
            r = c3.kendalls_w(rm_m)
            loo_mseed_rows.append({"group": label, "excluded_mseed": mseed, **r})

    tables["claim3_per_generator_rank_summary"] = pd.concat(rank_summary_rows, ignore_index=True)
    tables["claim3_kendalls_w"] = pd.DataFrame(w_rows)
    tables["claim3_leave_one_dseed_out"] = pd.DataFrame(loo_dseed_rows)
    tables["claim3_leave_one_mseed_out"] = pd.DataFrame(loo_mseed_rows)

    return tables


def main():
    all_tables = {}
    all_tables.update(calibration_table())
    all_tables.update(claim1_tables())
    all_tables.update(claim2_tables())
    all_tables.update(claim3_tables())

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n=== Writing appendix tables to {OUT_DIR} ===")
    for name, tbl in all_tables.items():
        path = os.path.join(OUT_DIR, f"{name}.csv")
        tbl.to_csv(path, index=False)
        print(f"[OK] Saved: {path}  ({len(tbl)} rows)")

    try:
        import openpyxl  # noqa: F401
        xlsx_path = os.path.join(OUT_DIR, "appendix_tables.xlsx")
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