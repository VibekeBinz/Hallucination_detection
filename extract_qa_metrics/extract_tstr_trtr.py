"""
extract_qa_metrics/extract_tstr_trtr.py

One pass over the raw TSTR/TRTR JSONs
(<chain_dir>/metrics/tstr_..._gen_<g>_....json), all scalar metrics and
confusion-matrix cells read from each file once -> a per-chain long table
and a pooled per-generation summary, in the same run. This is the one
canonical reader of these files; anything downstream that needs TSTR/TRTR
values reads this output rather than re-parsing the raw JSONs itself.

Output:
  PER_CHAIN_DIR/tstr_trtr_<GENERATOR>.csv
  QA_METRICS_POOLED_SUMMARY_DIR/tstr_trtr_summary_<GENERATOR>.csv
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import (
    GENERATOR_VARIANTS, PER_CHAIN_DIR, QA_METRICS_POOLED_SUMMARY_DIR,
)
from extract_qa_metrics._common import GENERATIONS, check_chain_count, find_chain_dirs, seed_pair, stats

METRICS_SUBDIR = "metrics"
GEN_FILE_GLOB = "tstr_*_gen_{g}_*.json"

TSTR_BLOCK = "tstr_metrics"
TRTR_BLOCK = "trtr_metrics"
SCALAR_METRICS = ["auroc", "accuracy", "precision", "recall", "f1_score", "threshold"]
# Output column name != JSON key for this one -- PRETTY (pipeline_config.py)
# already labels a 'tstr_f1' column for plots/tables, so the extractor
# writes that name directly instead of leaving 'f1_score' for something
# downstream to rename.
OUTPUT_METRIC_NAME = {"f1_score": "f1"}
CM_CELLS = ["tn", "fp", "fn", "tp"]

POOLED_DIR = QA_METRICS_POOLED_SUMMARY_DIR
os.makedirs(PER_CHAIN_DIR, exist_ok=True)
os.makedirs(POOLED_DIR, exist_ok=True)


def find_gen_file(metrics_dir: str, g: int):
    """The single JSON for generation g. Raises on more than one match --
    an ambiguous file is not something you'd catch by eye in a per-chain
    long table of thousands of rows."""
    matches = sorted(glob.glob(os.path.join(metrics_dir, GEN_FILE_GLOB.format(g=g))))
    if not matches:
        return None
    if len(matches) > 1:
        listing = "\n  ".join(os.path.basename(m) for m in matches)
        raise RuntimeError(f"{len(matches)} files match gen_{g} in {metrics_dir}:\n  {listing}")
    return matches[0]


def flatten_json(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(flatten_json(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(flatten_json(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def _safe_ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN-safe and division-by-zero-safe. A NaN
    input propagates to a NaN denominator automatically (nan + x = nan),
    so no separate isfinite check is needed on the inputs."""
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def extract_one_file(path: str) -> dict:
    """Every scalar TSTR/TRTR metric and confusion-matrix cell from one
    JSON, read once, plus fpr/fnr/tnr derived from the confusion matrix
    (PRETTY labels these for plots/tables; the JSON itself doesn't carry
    them as rates, only as raw tn/fp/fn/tp counts), and utility_gap
    (trtr_auroc - tstr_auroc)."""
    with open(path, "r", encoding="utf-8") as fh:
        flat = flatten_json(json.load(fh))
    row = {}
    for block, tag in ((TSTR_BLOCK, "tstr"), (TRTR_BLOCK, "trtr")):
        for m in SCALAR_METRICS:
            out_name = OUTPUT_METRIC_NAME.get(m, m)
            row[f"{tag}_{out_name}"] = flat.get(f"{block}.{m}", np.nan)

        cm = {cell: flat.get(f"{block}.confusion_matrix.{cell}", np.nan) for cell in CM_CELLS}
        for cell in CM_CELLS:
            row[f"{tag}_{cell}"] = cm[cell]

        row[f"{tag}_fpr"] = _safe_ratio(cm["fp"], cm["fp"] + cm["tn"])
        row[f"{tag}_fnr"] = _safe_ratio(cm["fn"], cm["fn"] + cm["tp"])
        row[f"{tag}_tnr"] = _safe_ratio(cm["tn"], cm["tn"] + cm["fp"])

    row["utility_gap"] = row["trtr_auroc"] - row["tstr_auroc"]
    return row


def collect_generator(generator_name: str) -> list:
    chain_dirs = find_chain_dirs(generator_name)
    check_chain_count(chain_dirs, generator_name)
    rows = []
    for chain_dir in chain_dirs:
        pair = seed_pair(chain_dir)
        if pair is None:
            print(f"  [WARN] could not parse dseed/mseed from {chain_dir}")
            continue
        dseed, mseed = pair
        metrics_dir = os.path.join(chain_dir, METRICS_SUBDIR)
        if not os.path.isdir(metrics_dir):
            print(f"  [WARN] no '{METRICS_SUBDIR}' folder in {os.path.basename(chain_dir)}")
            continue
        for g in GENERATIONS:
            fp = find_gen_file(metrics_dir, g)
            if fp is None:
                continue
            row = extract_one_file(fp)
            row.update(generator=generator_name, dseed=dseed, mseed=mseed, generation=g)
            rows.append(row)
    return rows


def build_pooled_summary(per_chain: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in per_chain.columns
                  if c not in ("generator", "dseed", "mseed", "generation")]
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
    print(f"\n=== extract_tstr_trtr: {generator_name} ===")
    rows = collect_generator(generator_name)
    if not rows:
        print(f"  [WARN] no data for {generator_name}, skipping")
        return

    per_chain = pd.DataFrame(rows).sort_values(["dseed", "mseed", "generation"])
    per_chain_path = os.path.join(PER_CHAIN_DIR, f"tstr_trtr_{generator_name}.csv")
    per_chain.to_csv(per_chain_path, index=False)
    print(f"  saved: {per_chain_path} ({len(per_chain)} rows)")

    pooled = build_pooled_summary(per_chain)
    pooled_path = os.path.join(POOLED_DIR, f"tstr_trtr_summary_{generator_name}.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"  saved: {pooled_path} ({len(pooled)} rows)")


def main():
    for generator_name in GENERATOR_VARIANTS:
        run(generator_name)


if __name__ == "__main__":
    main()