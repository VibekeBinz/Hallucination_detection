"""
extract_qa_metrics/extract_sd_quality_metrics.py

One pass over the raw per-chain SD-metrics CSV
(<chain_dir>/<chain_dir_name>.csv -- one file per chain, every generation
as a row) -> a per-chain long table and a pooled per-generation summary.

The file already has 'generation' as a column and every metric (mmd,
wasserstein, prdc, detection scores, alpha-precision, SDMetrics scores,
...) as its own column, so this extractor is mostly concatenation with
generator/dseed/mseed attached -- the duplication it removes is in the
readers, not the source data.

mmd and the computational-cost columns (training_time, etc.) are known
incomplete/wrong in the current raw export -- see apply_raw_data_patches.py,
which runs after this and overlays the corrected values. Don't hand-edit
this script to "fix" those columns; fix the patch step instead.

Output:
  summary_files/per_chain/sd_quality_<GENERATOR>.csv
  summary_files/qa_metrics/pooled_summary/sd_quality_summary_<GENERATOR>.csv
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import (
    GENERATOR_VARIANTS, PER_CHAIN_DIR, QA_METRICS_POOLED_SUMMARY_DIR, RESULTS_ROOT,
)
from extract_qa_metrics._common import check_chain_count, find_chain_dirs, seed_pair, stats

POOLED_DIR = QA_METRICS_POOLED_SUMMARY_DIR
os.makedirs(PER_CHAIN_DIR, exist_ok=True)
os.makedirs(POOLED_DIR, exist_ok=True)


def find_sd_metrics_csv(chain_dir: str):
    """The one CSV holding SD quality metrics for every generation of this
    chain -- named after the chain folder itself. Falls back to any single
    CSV directly inside the folder, and raises rather than guessing if
    that fallback is ambiguous."""
    name = os.path.basename(chain_dir)
    direct = os.path.join(chain_dir, f"{name}.csv")
    if os.path.isfile(direct):
        return direct
    matches = sorted(glob.glob(os.path.join(chain_dir, "*.csv")))
    if not matches:
        return None
    if len(matches) > 1:
        listing = "\n  ".join(os.path.basename(m) for m in matches)
        raise RuntimeError(
            f"{len(matches)} CSV files found directly in {chain_dir}:\n  {listing}\n"
            f"Expected exactly one SD-metrics CSV per chain."
        )
    return matches[0]


def collect_generator(generator_name: str) -> pd.DataFrame:
    chain_dirs = find_chain_dirs(generator_name)
    check_chain_count(chain_dirs, generator_name)
    frames = []
    for chain_dir in chain_dirs:
        pair = seed_pair(chain_dir)
        if pair is None:
            print(f"  [WARN] could not parse dseed/mseed from {chain_dir}")
            continue
        dseed, mseed = pair
        csv_path = find_sd_metrics_csv(chain_dir)
        if csv_path is None:
            print(f"  [WARN] no SD-metrics CSV in {os.path.basename(chain_dir)}")
            continue
        try:
            sd_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] could not read {csv_path}: {e}")
            continue
        if "generation" not in sd_df.columns:
            print(f"  [WARN] no 'generation' column in {csv_path}")
            continue
        sd_df = sd_df.copy()
        sd_df.insert(0, "generator", generator_name)
        sd_df.insert(1, "dseed", dseed)
        sd_df.insert(2, "mseed", mseed)
        frames.append(sd_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_pooled_summary(per_chain: pd.DataFrame) -> pd.DataFrame:
    id_cols = {"generator", "dseed", "mseed", "generation"}
    value_cols = [c for c in per_chain.columns
                  if c not in id_cols and pd.api.types.is_numeric_dtype(per_chain[c])]
    out_rows = []
    for gen, sub in per_chain.groupby("generation"):
        row = {"generation": gen}
        for col in value_cols:
            mean, var, low, high = stats(sub[col].values)
            row[f"{col}_mean"] = mean
            row[f"{col}_var"] = var
            row[f"{col}_low"] = low
            row[f"{col}_high"] = high
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("generation")


def run(generator_name: str):
    print(f"\n=== extract_sd_quality_metrics: {generator_name} ===")
    per_chain = collect_generator(generator_name)
    if per_chain.empty:
        print(f"  [WARN] no data for {generator_name}, skipping")
        return

    per_chain = per_chain.sort_values(["dseed", "mseed", "generation"])
    per_chain_path = os.path.join(PER_CHAIN_DIR, f"sd_quality_{generator_name}.csv")
    per_chain.to_csv(per_chain_path, index=False)
    print(f"  saved: {per_chain_path} ({len(per_chain)} rows, {len(per_chain.columns)} columns)")

    pooled = build_pooled_summary(per_chain)
    pooled_path = os.path.join(POOLED_DIR, f"sd_quality_summary_{generator_name}.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"  saved: {pooled_path} ({len(pooled)} rows)")


def main():
    for generator_name in GENERATOR_VARIANTS:
        run(generator_name)


if __name__ == "__main__":
    main()