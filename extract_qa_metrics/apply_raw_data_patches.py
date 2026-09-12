"""
extract_qa_metrics/apply_raw_data_patches.py

Overlays one-off corrections from the upstream pipeline onto the already-
extracted qa_metrics output, rather than teaching extract_sd_quality_metrics.py
a second raw-data shape for what is a stopgap fix, not the new normal.

Source: long-format CSVs dropped directly into RAW_DATA_ROOT (not per
chain) -- corrected mmd values, and computational-cost fields
(training_time, training_time_cumulative, generation_time, model_size_mb)
that the original per-chain SD-metrics CSVs had wrong or missing.
Columns: variant, dseed, mseed, model, generation, metric_name,
metric_value. 'variant' here uses yet another spelling (pf_all /
pf_pilgram) -- resolved via VARIANT_TOKEN_MAP in config/pipeline_config.py,
same as every other raw-side variant spelling.

Run this AFTER extract_sd_quality_metrics.py (run_extraction.py already
does, in that order). Safe to rerun: it always patches from the same
source files onto a freshly-extracted per-chain frame, never onto its own
previous output, so running it twice in a row is a no-op.

Delete this script once the upstream pipeline re-exports mmd and
computational cost correctly in its normal per-chain CSV -- at that point
extract_sd_quality_metrics.py alone is sufficient again.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import (
    PER_CHAIN_DIR,
    QA_METRICS_POOLED_SUMMARY_DIR,
    RAW_DATA_ROOT,
    generator_value,
    variant_from_token,
)
from extract_qa_metrics._common import stats

POOLED_DIR = QA_METRICS_POOLED_SUMMARY_DIR

# Filenames as dropped into RAW_DATA_ROOT. A missing file is skipped with
# a warning, not an error -- a partial patch set (e.g. mmd only) is still
# worth applying.
PATCH_FILES = [
    "metrics_long_pfa_computational_cost.csv",
    "metrics_long_pfa_mmd_corrected.csv",
    "metrics_long_pfp_computational_cost.csv",
    "metrics_long_pfp_mmd_corrected.csv",
]

# mmd is a genuine fix to the raw data, not a parallel "corrected" variant
# to keep alongside the original -- so it overwrites the 'mmd' column in
# place, same as the computational-cost columns below. No rename.


def load_patch_long(filename: str):
    path = os.path.join(RAW_DATA_ROOT, filename)
    if not os.path.isfile(path):
        print(f"  [SKIP] not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"variant", "dseed", "mseed", "model", "generation",
                "metric_name", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def to_wide_by_generator(long_df: pd.DataFrame) -> dict:
    """{GENERATOR: DataFrame(dseed, mseed, generation, <metric columns>)},
    one entry per (model, variant) combination actually present."""
    long_df = long_df.copy()
    long_df["variant_label"] = long_df["variant"].apply(variant_from_token)
    long_df["GENERATOR"] = [
        generator_value(model, variant)
        for model, variant in zip(long_df["model"], long_df["variant_label"])
    ]

    out = {}
    for generator_name, sub in long_df.groupby("GENERATOR"):
        wide = sub.pivot_table(
            index=["dseed", "mseed", "generation"],
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
        ).reset_index()
        out[generator_name] = wide
    return out


def build_pooled_summary(per_chain: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for gen, sub in per_chain.groupby("generation"):
        row = {"generation": gen}
        for col in cols:
            mean, var, low, high = stats(sub[col].values)
            row[f"{col}_mean"] = mean
            row[f"{col}_var"] = var
            row[f"{col}_low"] = low
            row[f"{col}_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("generation")


def patch_generator(generator_name: str, patch_wide: pd.DataFrame):
    per_chain_path = os.path.join(PER_CHAIN_DIR, f"sd_quality_{generator_name}.csv")
    if not os.path.isfile(per_chain_path):
        print(f"  [WARN] {per_chain_path} does not exist yet -- run "
              f"extract_sd_quality_metrics.py first. Skipping {generator_name}.")
        return

    per_chain = pd.read_csv(per_chain_path)
    patch_cols = [c for c in patch_wide.columns if c not in ("dseed", "mseed", "generation")]

    merged = per_chain.merge(
        patch_wide, on=["dseed", "mseed", "generation"], how="left", suffixes=("", "_patch")
    )

    match_counts = {}
    for col in patch_cols:
        patch_col = f"{col}_patch" if f"{col}_patch" in merged.columns else col
        match_counts[col] = int(merged[patch_col].notna().sum())
        if patch_col != col:
            merged[col] = merged[patch_col].where(merged[patch_col].notna(), merged.get(col))
            merged.drop(columns=[patch_col], inplace=True)

    merged.to_csv(per_chain_path, index=False)
    counts_str = ", ".join(f"{c}: {n}/{len(merged)}" for c, n in match_counts.items())
    print(f"  [OK] {generator_name}: patched {counts_str}")

    pooled = build_pooled_summary(merged, patch_cols)
    pooled_path = os.path.join(POOLED_DIR, f"sd_quality_summary_{generator_name}.csv")
    if os.path.isfile(pooled_path):
        existing = pd.read_csv(pooled_path)
        keep_cols = [c for c in existing.columns
                     if not any(c.startswith(f"{p}_") for p in patch_cols)]
        existing = existing[keep_cols]
        pooled = existing.merge(pooled, on="generation", how="outer")
    pooled.to_csv(pooled_path, index=False)
    print(f"  [OK] {generator_name}: pooled summary refreshed for {patch_cols}")


def main():
    print("\n=== qa_metrics: applying raw-data patches (mmd, computational cost) ===")
    for filename in PATCH_FILES:
        print(f"\n{filename}")
        long_df = load_patch_long(filename)
        if long_df is None:
            continue
        by_generator = to_wide_by_generator(long_df)
        for generator_name, patch_wide in by_generator.items():
            patch_generator(generator_name, patch_wide)


if __name__ == "__main__":
    main()