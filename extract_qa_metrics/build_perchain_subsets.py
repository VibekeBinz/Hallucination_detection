"""
extract_qa_metrics/build_perchain_subsets.py

Reshapes a handful of already-extracted per-chain metrics into tidy,
one-metric-per-file trajectory tables -- one row per chain per generation,
generator/dseed/mseed/chain identifying the row, ready for the same kind
of cross-generator comparison and threshold calibration the hallucination
trajectory files already support.

Read-only against extract_sd_quality_metrics.py's and extract_tstr_trtr.py's
output -- no new computation, no re-reading the raw per-chain CSVs. Run
those two scripts first; this one just picks columns out of what they
already wrote.

Metrics covered:
  tstr_auroc                 <- per_chain/tstr_trtr_{GENERATOR}.csv
  alpha_delta_precision_OC   <- per_chain/sd_quality_{GENERATOR}.csv
  alpha_delta_coverage_OC    <- per_chain/sd_quality_{GENERATOR}.csv
  alpha_authenticity_OC      <- per_chain/sd_quality_{GENERATOR}.csv
  detection_avg              <- per_chain/sd_quality_{GENERATOR}.csv
  detection_gmm              <- per_chain/sd_quality_{GENERATOR}.csv
  detection_xgb              <- per_chain/sd_quality_{GENERATOR}.csv
  detection_mlp               <- per_chain/sd_quality_{GENERATOR}.csv
  detection_linear            <- per_chain/sd_quality_{GENERATOR}.csv

Output: one file per metric, every generator variant pooled together
(full and CORE alike -- "generator" already spells out which is which,
e.g. "ARF" vs "ARF_CORE", same convention as everywhere else in this
pipeline). Written into the shared PER_CHAIN_DIR (config/pipeline_config.py)
alongside sd_quality_{GENERATOR}.csv, tstr_trtr_{GENERATOR}.csv, and
hallucination_pipeline's Step 4 files -- named by metric rather than
generator, so nothing collides with those:

  summary_files/per_chain/<metric>_per_chain.csv
      columns: generator, dseed, mseed, chain, generation, value
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import GENERATOR_VARIANTS, PER_CHAIN_DIR

OUT_DIR = PER_CHAIN_DIR
os.makedirs(OUT_DIR, exist_ok=True)

SD_QUALITY_METRICS = [
    "alpha_delta_precision_OC",
    "alpha_delta_coverage_OC",
    "alpha_authenticity_OC",
    "detection_avg",
    "detection_gmm",
    "detection_xgb",
    "detection_mlp",
    "detection_linear",
]
TSTR_TRTR_METRICS = ["tstr_auroc"]


def _tidy(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One generator's wide per-chain frame -> tidy rows for one metric."""
    if metric not in df.columns:
        return pd.DataFrame()
    out = df[["generator", "dseed", "mseed", "generation", metric]].rename(
        columns={metric: "value"}
    )
    # Same "d<dseed> / m<mseed>" chain label the hallucination pipeline's
    # per-chain files use, so the two can sit side by side in a plot.
    out["chain"] = "d" + out["dseed"].astype(str) + " / m" + out["mseed"].astype(str)
    return out[["generator", "dseed", "mseed", "chain", "generation", "value"]]


def build_from(source_name: str, metrics: list) -> None:
    frames = {m: [] for m in metrics}
    for generator_name in GENERATOR_VARIANTS:
        path = os.path.join(PER_CHAIN_DIR, f"{source_name}_{generator_name}.csv")
        if not os.path.exists(path):
            print(f"  [SKIP] not found: {path}")
            continue
        df = pd.read_csv(path)
        for m in metrics:
            tidy = _tidy(df, m)
            if not tidy.empty:
                frames[m].append(tidy)
            else:
                print(f"  [WARN] '{m}' not in columns of {path}")

    for m in metrics:
        if not frames[m]:
            print(f"  [SKIP] no data found for '{m}' across any generator")
            continue
        combined = pd.concat(frames[m], ignore_index=True)
        combined = combined.sort_values(["generator", "dseed", "mseed", "generation"])
        out_path = os.path.join(OUT_DIR, f"{m}_per_chain.csv")
        combined.to_csv(out_path, index=False)
        n_chains = combined.groupby(["generator", "dseed", "mseed"]).ngroups
        print(f"  wrote {out_path} ({len(combined)} rows, {n_chains} chains)")


def main():
    print("=== build_perchain_subsets: SD quality metrics "
          "(alpha-precision / beta-recall / authenticity / detection) ===")
    build_from("sd_quality", SD_QUALITY_METRICS)
    print("\n=== build_perchain_subsets: TSTR/TRTR (tstr_auroc) ===")
    build_from("tstr_trtr", TSTR_TRTR_METRICS)
    print("\ndone")


if __name__ == "__main__":
    main()