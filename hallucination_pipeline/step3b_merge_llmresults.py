"""
hallucination_pipeline/step3b_merge_llm.py

STEP 3b — Write LLM results into the merged hallucination summaries.

Standalone. Run after step3_llm_filter.py has produced its per-folder
summaries. Makes no API calls and recomputes nothing — it only aggregates
summary_files/hallucination_pipeline/summary/step3_summary_{GENERATOR}.csv
across seed folders and merges the per-generation statistics into

    {HALLUCINATION_SUMMARIES}/merged_hallucination_summary_{GENERATOR}.csv

Columns written (each block has _mean, _var, _low, _high, _min, _max, _n):

    LLM_*              Claude's flags ONLY. Expert-flagged rows are removed
                       in step 3's process_file before the LLM sees them, so
                       llm_rate is step 3's own incremental contribution and
                       nothing else.
    LLM_accumulated_*  Expert rules + LLM (step3 combined_rate).

Note: LLM_accumulated is taken directly from combined_rate — do NOT try to
reconstruct it as (Expert_any_mean already in the merged file) + LLM_mean.
Those Expert_any_* columns may be averaged over more seed folders than the
LLM ran on, so they are not a matched baseline.

n = number of dseed/mseed folder combinations that completed for that
generation. It is per-generation, not per-generator, because coverage
varies (in the ARF summaries gen_0 has 5 folders while gen_3/5/15 have 3).

The merged files are keyed on (generation, bin). LLM results are
bin-independent, so a left-merge on generation alone copies the same values
onto every bin row — the same way the Expert_* columns already behave.

Step 3's output lives in one shared, flat
summary_files/hallucination_pipeline/summary/ folder, so discover_generators()
below (the glob over step3_summary_*.csv) finds every generator that has
been run, not just the one GENERATOR currently set in the config -- that's
the point of it, not a side effect to work around.

Usage:
    python step3b_merge_llm.py                 # every generator found
    python step3b_merge_llm.py ARF ARF_CORE    # only these
    python step3b_merge_llm.py --dry-run       # print, write nothing
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import t
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.pipeline_config import HALLUC_SUMMARY_DIR, HALLUCINATION_SUMMARIES

# ============================================================
# SETTINGS
# ============================================================

# Rates in step3_summary_*.csv are stored as percentages (5.5 == 5.5%).
# The merged summaries store fractions (0.055). Divide to match.
RATE_SCALE = 100.0

# Column blocks this script owns. Dropped before merging so re-runs overwrite
# cleanly instead of producing _x / _y suffixes. Expert_any_llmsubset_ is no
# longer written but stays here so earlier runs get cleaned out.
OWNED_PREFIXES = ("LLM_", "Expert_any_llmsubset_")

STEP3_GLOB = os.path.join(HALLUC_SUMMARY_DIR, "step3_summary_*.csv")


def merged_path(generator: str) -> str:
    return os.path.join(HALLUCINATION_SUMMARIES,
                        f"merged_hallucination_summary_{generator}.csv")


# ============================================================
# STATISTICS
# ============================================================

def stats_full(arr) -> dict:
    """mean, variance, min, max, n and 95% t-CI for one metric."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "var": np.nan, "low": np.nan,
                "high": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    mean = arr.mean()
    var  = arr.var(ddof=1) if n > 1 else 0.0
    sd   = arr.std(ddof=1) if n > 1 else 0.0
    ci   = t.ppf(0.975, df=n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "var": var, "low": mean - ci, "high": mean + ci,
            "min": arr.min(), "max": arr.max(), "n": n}


# ============================================================
# LOAD + VALIDATE
# ============================================================

def load_step3_summary(generator: str) -> pd.DataFrame:
    path = os.path.join(HALLUC_SUMMARY_DIR, f"step3_summary_{generator}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Step 3 summary not found: {path}")

    df = pd.read_csv(path)
    required = {"generation", "folder", "llm_rate", "combined_rate", "expert_any_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Resumed checkpoints can append a folder twice; keep the last run.
    before = len(df)
    df = df.drop_duplicates(subset=["generation", "folder"], keep="last")
    if len(df) != before:
        print(f"  [WARN] dropped {before - len(df)} duplicate (generation, folder) rows")

    return df


def validate_rates(df: pd.DataFrame, generator: str):
    """
    Sanity checks. These are cheap and catch the two failure modes that
    would silently corrupt the merged file.
    """
    # 1. Units. Percentages, not fractions.
    if df["combined_rate"].max() <= 1.0:
        print(f"  [WARN] {generator}: all rates <= 1.0 — these may already be "
              f"fractions, in which case RATE_SCALE should be 1.0.")

    # 2. Disjointness. expert + llm should equal combined, because step 3
    #    removes expert-flagged rows before calling the LLM.
    resid = (df["expert_any_rate"] + df["llm_rate"] - df["combined_rate"]).abs()
    bad = df[resid > 0.02]
    if not bad.empty:
        print(f"  [WARN] {generator}: expert_any_rate + llm_rate != combined_rate "
              f"on {len(bad)} row(s). The LLM may be re-flagging expert rows — "
              f"check the folder/generation keys in the step 2 ID file.")
        print(bad[["generation", "folder", "expert_any_rate",
                   "llm_rate", "combined_rate"]].to_string(index=False))


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_by_generation(df: pd.DataFrame) -> pd.DataFrame:
    """One row per generation, aggregated across dseed/mseed folders."""
    blocks = [("LLM", "llm_rate"), ("LLM_accumulated", "combined_rate")]

    rows = []
    for gen, grp in df.groupby("generation"):
        gen_label = str(gen) if str(gen).startswith("gen_") else f"gen_{gen}"
        row = {"generation": gen_label}
        for prefix, col in blocks:
            values = grp[col].values.astype(float) / RATE_SCALE
            for stat_name, value in stats_full(values).items():
                row[f"{prefix}_{stat_name}"] = value
        rows.append(row)

    out = pd.DataFrame(rows)
    out["_gen_num"] = out["generation"].str.extract(r"(\d+)").astype(int)
    return out.sort_values("_gen_num").drop(columns="_gen_num").reset_index(drop=True)


# ============================================================
# MERGE + WRITE
# ============================================================

def update_merged_summary(generator: str, dry_run: bool = False) -> bool:
    print(f"\n=== {generator} ===")

    try:
        step3 = load_step3_summary(generator)
    except (FileNotFoundError, ValueError) as e:
        print(f"  [SKIP] {e}")
        return False

    validate_rates(step3, generator)
    agg = aggregate_by_generation(step3)

    print(f"  {len(step3)} folder-rows -> {len(agg)} generations")
    show = ["generation", "LLM_mean", "LLM_low", "LLM_high", "LLM_min",
            "LLM_max", "LLM_n", "LLM_accumulated_mean", "LLM_accumulated_n"]
    print(agg[show].to_string(index=False,
                              float_format=lambda v: f"{v:.6f}"))

    target = merged_path(generator)
    if not os.path.exists(target):
        print(f"  [SKIP] merged summary not found: {target}")
        return False

    merged = pd.read_csv(target)
    merged["generation"] = merged["generation"].astype(str)

    stale = [c for c in merged.columns if c.startswith(OWNED_PREFIXES)]
    if stale:
        merged = merged.drop(columns=stale)

    missing = sorted(set(agg["generation"]) - set(merged["generation"]))
    if missing:
        print(f"  [WARN] generations in step 3 but not in the merged file: {missing}")
    empty = sorted(set(merged["generation"]) - set(agg["generation"]))
    if empty:
        print(f"  [INFO] no LLM run for {len(empty)} generation(s) — left blank: {empty}")

    merged = merged.merge(agg, on="generation", how="left")

    merged["_gen_num"] = merged["generation"].str.extract(r"(\d+)").astype(int)
    sort_cols = ["_gen_num"] + (["bin"] if "bin" in merged.columns else [])
    merged = merged.sort_values(sort_cols).drop(columns="_gen_num")

    if dry_run:
        print(f"  [DRY RUN] would write {len(merged)} rows to {target}")
        return True

    merged.to_csv(target, index=False)
    print(f"  Written: {target}  ({len(merged)} rows, {len(merged.columns)} columns)")
    return True


# ============================================================
# MAIN
# ============================================================

def discover_generators() -> list:
    found = []
    for path in sorted(glob.glob(STEP3_GLOB)):
        m = re.match(r"step3_summary_(.+)\.csv$", os.path.basename(path))
        if m:
            found.append(m.group(1))
    return found


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv

    generators = args or discover_generators()
    if not generators:
        print(f"No step 3 summaries found matching {STEP3_GLOB}")
        return

    print(f"Step 3 summaries : {HALLUC_SUMMARY_DIR}")
    print(f"Merged summaries : {HALLUCINATION_SUMMARIES}")
    print(f"Generators       : {generators}")
    if dry_run:
        print("Mode             : DRY RUN (nothing will be written)")

    ok = [g for g in generators if update_merged_summary(g, dry_run)]

    print(f"\nDone — {len(ok)}/{len(generators)} generator(s) updated.")
    skipped = [g for g in generators if g not in ok]
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()