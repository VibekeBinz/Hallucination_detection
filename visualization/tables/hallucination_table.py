"""
visualization/tables/hallucination_table.py

Builds the hallucination summary table:
- Rows: metric name + generator
- Columns: gen_0 ... gen_19
- Cells: mean% +/- CI%, except the "Expert rules + LLM %" row, which reports
  median% (min%-max%) across the per-chain values because n = 3 makes a
  t-interval uninformative (and can span implausible ranges).

Reads:
  {HALLUCINATION_PIPELINE_SUMMARY_DIR}/
    step1_summary_<GENERATOR>.csv
    step2_summary_<GENERATOR>.csv
  {STEP4_PER_CHAIN_DIR}/step4_expert_plus_llm_per_chain_<GENERATOR>.csv
    (raw per-chain values for the "Expert rules + LLM %" row)
  {HALLUCINATION_SUMMARIES}/merged_hallucination_summary_<GENERATOR>.csv
    (mean +/- CI for the "LLM only %" row, and the fallback for
    "Expert rules + LLM %" on any generation step4 doesn't cover)

Writes:
  {TABLE_OUTPUT_DIR}/hallucination_table_for_article_full.csv

Per-chain LLM values
---------------------
step4_expert_plus_llm_per_chain_<GENERATOR>.csv is the per-chain source for
"Expert rules + LLM %": one row per (generation, chain), value already a
fraction (0-1) -- no percent conversion needed. It's only written for a
handful of generations per chain (it's the expensive step), so most gen_N
columns in that row will legitimately be blank there and fall back to the
merged mean +/- CI. That's expected, not a bug.

"LLM only %" (the LLM check on its own, not combined with expert rules) has
no per-chain source at all -- it's read directly from
merged_hallucination_summary_<GENERATOR.csv> as mean +/- CI for every
generation.
"""

import os
import statistics
import sys

import pandas as pd

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../hallucination_table.py`, an IDE "Run" button, or
# `python -m visualization.tables.hallucination_table` from the repo root --
# only the last of those already has the repo root on sys.path).
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)
_REPO_ROOT = _p

from config.pipeline_config import (
    BIN_LIST, GENERATOR_VARIANTS, HALLUC_SUMMARY_DIR, HALLUCINATION_SUMMARIES,
    PER_CHAIN_DIR, RESULTS_TABLES_DIR,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

HALLUCINATION_PIPELINE_SUMMARY_DIR = HALLUC_SUMMARY_DIR
STEP4_PER_CHAIN_DIR = PER_CHAIN_DIR
TABLE_OUTPUT_DIR = RESULTS_TABLES_DIR
OUTPUT_CSV_NAME = "hallucination_table_for_article_full.csv"

# Every generator/variant combination config.pipeline_config knows about
# (ARF, ARF_CORE, CTGAN, CTGAN_CORE, ...) -- one shared list instead of a
# second copy of the same eight names.
GENERATORS = list(GENERATOR_VARIANTS.keys())

GENERATIONS = [f"gen_{i}" for i in range(20)]  # gen_0 ... gen_19

# "Expert rules + LLM %" rests on three dseed/mseed chains, so a t-interval
# spans implausible ranges. Median (min, max) is reported instead -- for
# exactly 3 chains this shows all three raw values (sorted), just more
# compactly than listing them.
VALUE_DECIMALS = 1

PRETTY = {
    "HR_pilgram": "HR",
    "HR_adjusted": "HR*",
    "Expert_any": "Expert rules (all) %",
    "Expert_impossible": "% impossible expert rules",
    "LLM": "LLM only %",
    "LLM_accumulated": "Expert rules + LLM %",
}


# -------------------------------------------------------------------
# FORMATTING
# -------------------------------------------------------------------

def format_mean_ci_percent(mean: float, low: float, high: float) -> str:
    """Return 'mean% +/- CI%' with CI = high - mean, converted to percentages."""
    mean_pct = mean * 100
    ci_pct = (high - mean) * 100
    return f"{mean_pct:.1f}% ± {ci_pct:.1f}%"


def format_median_min_max_percent(values) -> str:
    """
    Return 'median% (min%-max%)' computed directly from the individual
    per-chain values -- used for the "Expert rules + LLM %" row, where n=3
    makes a t-based mean +/- CI uninformative.

    Values are expected as FRACTIONS (0-1); this multiplies by 100 to
    display, same convention as format_mean_ci_percent.

    With a single chain the range collapses to a point, so just the value is
    shown rather than the noisy 'x% (x%-x%)'.
    """
    vals = sorted(float(v) * 100 for v in values if pd.notna(v))
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.{VALUE_DECIMALS}f}%"
    median_pct = statistics.median(vals)
    return (
        f"{median_pct:.{VALUE_DECIMALS}f}% "
        f"({vals[0]:.{VALUE_DECIMALS}f}%–{vals[-1]:.{VALUE_DECIMALS}f}%)"
    )


def _normalize_generation_labels(series: pd.Series) -> pd.Series:
    """'10' / 10 / 'gen_10' -> 'gen_10', matching the GENERATIONS convention."""
    def norm(v):
        s = str(v)
        return s if s.startswith("gen_") else f"gen_{int(v)}"
    return series.apply(norm)


# -------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------

def load_step1(generator: str) -> pd.DataFrame:
    """Load step1_summary_<GENERATOR>.csv (all bins; filtering happens later)."""
    path = os.path.join(HALLUCINATION_PIPELINE_SUMMARY_DIR, f"step1_summary_{generator}.csv")
    return pd.read_csv(path).copy()


def load_step2(generator: str) -> pd.DataFrame:
    """Load step2_summary_<GENERATOR>.csv."""
    path = os.path.join(HALLUCINATION_PIPELINE_SUMMARY_DIR, f"step2_summary_{generator}.csv")
    return pd.read_csv(path)


def load_merged(generator: str) -> pd.DataFrame | None:
    """
    merged_hallucination_summary_<GENERATOR>.csv -- mean +/- CI (plus
    min/max/n where n=3 chains ran) for both LLM rows. Source for
    "LLM only %" (always) and the fallback source for "Expert rules + LLM %"
    on any generation step4_expert_plus_llm_per_chain doesn't cover.
    """
    path = os.path.join(HALLUCINATION_SUMMARIES, f"merged_hallucination_summary_{generator}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] no merged summary for {generator}: {path}")
        return None
    df = pd.read_csv(path)
    # Keyed on (generation, bin), but the LLM_*/LLM_accumulated_* values are
    # broadcast identically across every bin -- one row per generation is
    # enough.
    return df.drop_duplicates(subset="generation", keep="first")


def load_step4_expert_plus_llm_per_chain(generator: str) -> dict:
    """
    Return {generation: [rate_chain1, rate_chain2, ...]} for the "Expert
    rules + LLM %" row, from step4_expert_plus_llm_per_chain_<GENERATOR>.csv
    (one row per (generation, chain), 'value' already a fraction 0-1).
    Only a handful of generations are covered per chain -- {} for any
    generation not present, so the caller falls back to merged's mean +/- CI.
    """
    path = os.path.join(STEP4_PER_CHAIN_DIR, f"step4_expert_plus_llm_per_chain_{generator}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] no step4 expert+LLM per-chain file for {generator}: {path}")
        return {}

    df = pd.read_csv(path)
    if "generation" not in df.columns or "value" not in df.columns:
        print(
            f"  [WARN] {path} is missing 'generation' or 'value' - "
            f"columns present: {list(df.columns)}"
        )
        return {}

    df = df.copy()
    df["generation"] = _normalize_generation_labels(df["generation"])
    grouped = df.groupby("generation")["value"].apply(lambda s: [float(v) for v in s if pd.notna(v)])
    return {g: v for g, v in grouped.items() if v}


# -------------------------------------------------------------------
# ROW EXTRACTORS
# -------------------------------------------------------------------

def extract_step1_metric_row(df_step1, metric_name: str, bin_value: int) -> dict:
    if metric_name == "HR_pilgram":
        mean_col, low_col, high_col = (
            "Pilgram_HR_mean", "Pilgram_HR_low", "Pilgram_HR_high"
        )
    elif metric_name == "HR_adjusted":
        mean_col, low_col, high_col = (
            "HR_adjusted_mean", "HR_adjusted_low", "HR_adjusted_high"
        )
    else:
        raise ValueError(f"Unknown Step 1 metric: {metric_name}")

    out = {"metric": f"{PRETTY.get(metric_name, metric_name)} (bin={bin_value})"}
    df_bin = df_step1[df_step1["bin"] == bin_value]

    for gen in GENERATIONS:
        row = df_bin[df_bin["generation"] == gen]
        if row.empty:
            out[gen] = ""
            continue
        out[gen] = format_mean_ci_percent(
            float(row[mean_col].iloc[0]),
            float(row[low_col].iloc[0]),
            float(row[high_col].iloc[0]),
        )
    return out


def extract_expert_any(df_step2) -> dict:
    """Expert_any as percentage +/- CI."""
    out = {"metric": "Expert rules (all) %"}
    for gen in GENERATIONS:
        row = df_step2[df_step2["generation"] == gen]
        if row.empty:
            out[gen] = ""
            continue
        out[gen] = format_mean_ci_percent(
            float(row["Expert_any_mean"].iloc[0]),
            float(row["Expert_any_low"].iloc[0]),
            float(row["Expert_any_high"].iloc[0]),
        )
    return out


def extract_expert_distribution(df_step2) -> dict:
    """% impossible expert rules as a share of all expert-rule flags."""
    out = {"metric": "% impossible expert rules"}
    for gen in GENERATIONS:
        row = df_step2[df_step2["generation"] == gen]
        if row.empty:
            out[gen] = ""
            continue
        any_mean = float(row["Expert_any_mean"].iloc[0])
        imp_mean = float(row["Expert_impossible_mean"].iloc[0])
        # A zero denominator is not 0% - no rule fired, so the share is undefined.
        out[gen] = f"{100 * imp_mean / any_mean:.1f}%" if any_mean > 0 else "NA"
    return out


def extract_llm_only(df_merged) -> dict:
    """
    "LLM only %" -- no per-chain source exists for this metric, so every
    generation is read directly from merged_hallucination_summary's
    LLM_mean/LLM_high as mean +/- CI.
    """
    out = {"metric": PRETTY["LLM"]}
    for gen in GENERATIONS:
        if df_merged is None or "LLM_mean" not in df_merged.columns:
            out[gen] = ""
            continue
        row = df_merged[df_merged["generation"] == gen]
        if row.empty or pd.isna(row["LLM_mean"].iloc[0]):
            out[gen] = ""
            continue
        out[gen] = format_mean_ci_percent(
            float(row["LLM_mean"].iloc[0]), None, float(row["LLM_high"].iloc[0])
        )
    return out


def extract_expert_plus_llm(df_merged, per_chain: dict) -> dict:
    """
    "Expert rules + LLM %" -- median (min-max) across the raw per-chain
    values from step4_expert_plus_llm_per_chain, for whichever generations
    that file covers (n=3 chains, so a t-interval would be uninformative).

    Falls back to merged_hallucination_summary's mean +/- CI (clearly
    tagged) for any generation step4 didn't cover -- most of them, since
    step4 is only run for a handful of generations per chain.
    """
    out = {"metric": PRETTY["LLM_accumulated"]}
    mean_col, high_col, n_col = "LLM_accumulated_mean", "LLM_accumulated_high", "LLM_accumulated_n"

    for gen in GENERATIONS:
        if gen in per_chain:
            out[gen] = format_median_min_max_percent(per_chain[gen])
            continue

        if df_merged is None or mean_col not in df_merged.columns:
            out[gen] = ""
            continue
        row = df_merged[df_merged["generation"] == gen]
        if row.empty or pd.isna(row[mean_col].iloc[0]):
            out[gen] = ""
            continue
        cell = format_mean_ci_percent(
            float(row[mean_col].iloc[0]), None, float(row[high_col].iloc[0])
        )
        cell += " [mean±CI - no raw chain values found]"
        if n_col in df_merged.columns:
            n = row[n_col].iloc[0]
            if pd.notna(n):
                cell += f" (n={int(n)})"
        out[gen] = cell
    return out


# -------------------------------------------------------------------
# MAIN TABLE BUILD
# -------------------------------------------------------------------

def build_summary_table() -> pd.DataFrame:
    rows = []

    for generator in GENERATORS:
        df1 = load_step1(generator)
        df2 = load_step2(generator)
        dfm = load_merged(generator)
        per_chain = load_step4_expert_plus_llm_per_chain(generator)

        for bin_value in BIN_LIST:
            for metric in ("HR_pilgram", "HR_adjusted"):
                r = extract_step1_metric_row(df1, metric, bin_value)
                r["generator"] = generator
                rows.append(r)

        r = extract_expert_any(df2)
        r["generator"] = generator
        rows.append(r)

        r = extract_expert_distribution(df2)
        r["generator"] = generator
        rows.append(r)

        r = extract_llm_only(dfm)
        r["generator"] = generator
        rows.append(r)

        r = extract_expert_plus_llm(dfm, per_chain)
        r["generator"] = generator
        rows.append(r)

    cols = ["metric", "generator"] + GENERATIONS
    return pd.DataFrame(rows)[cols]


def main():
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    df = build_summary_table()
    out_path = os.path.join(TABLE_OUTPUT_DIR, OUTPUT_CSV_NAME)
    # utf-8-sig so Excel on Windows renders the +/- sign correctly
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table to: {out_path}")


if __name__ == "__main__":
    main()