"""
visualization/tables/all_metrics_table.py

Builds two CSV tables from the pooled per-generation quality/utility metric
summaries:

1. FULL table (appendix): every metric in FULL_METRICS.
2. SIMPLIFIED table (results chapter): only the metrics in SIMPLIFIED_METRICS.

Both cover the same generation subset (GENERATIONS below) and every
generator/variant in config.GENERATOR_VARIANTS (e.g. ARF, ARF_CORE, CTGAN,
CTGAN_CORE, ...).

Reads:  {QA_METRICS_POOLED_DIR}/sd_quality_summary_<GENERATOR>.csv
        {QA_METRICS_POOLED_DIR}/tstr_trtr_summary_<GENERATOR>.csv
  <GENERATOR> is the exact variant key (e.g. "ARF" or "ARF_CORE") -- these
  files are per generator/variant, not per base-generator-plus-core-flag, so
  no separate full/core filename prefix is needed the way the old
  summary_raw_Step7pfa/pfp files required.

Writes: {TABLE_OUTPUT_DIR}/metrics_table_for_article_full.csv
        {TABLE_OUTPUT_DIR}/metrics_table_for_article_simplified.csv
"""

import os
import sys
import pandas as pd

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../all_metrics_table.py`, an IDE "Run" button, or
# `python -m visualization.tables.all_metrics_table` from the repo root --
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

from config.pipeline_config import GENERATOR_VARIANTS, QA_METRICS_POOLED_SUMMARY_DIR, RESULTS_TABLES_DIR

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

QA_METRICS_POOLED_DIR = QA_METRICS_POOLED_SUMMARY_DIR
TABLE_OUTPUT_DIR = RESULTS_TABLES_DIR

# Every generator/variant combination config.pipeline_config knows about
# (ARF, ARF_CORE, CTGAN, CTGAN_CORE, ...) -- one shared list instead of a
# second copy of the same eight names.
GENERATORS = list(GENERATOR_VARIANTS.keys())

# A deliberately trimmed subset of the 20 generations, not the full 0-19 --
# printed page width doesn't fit all of them. Adjust the list directly if
# the cut needs to move.
GENERATIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "19"]

# -------------------------------------------------------------------
# HUMAN-READABLE METRIC NAMES FOR THIS TABLE
# -------------------------------------------------------------------
# Table-specific labels -- deliberately separate from config.PRETTY (which
# the figure scripts use for axis/legend text). Table headers favour plain
# ASCII ("alpha-Precision") over the figures' unicode ("α-Precision") since
# these CSVs get opened in Excel; keep the two lists independent rather than
# merging them.

PRETTY = {
    # Alpha metrics
    "alpha_delta_precision_OC": "alpha-Precision",
    "alpha_delta_coverage_OC": "beta-Recall",
    "alpha_authenticity_OC": "Authenticity",

    # PRDC
    "prdc_avg": "PRDC Average",
    "prdc_precision": "PRDC Precision",
    "prdc_recall": "PRDC Recall",
    "prdc_density": "PRDC Density",
    "prdc_coverage": "PRDC Coverage",

    # Distances
    "tv_complement": "TV Complement",
    "ks_complement": "KS Complement",
    "wasserstein_dist": "Wasserstein Distance",

    # JSD
    "jsd_syndat": "Jensen-Shannon Divergence",

    # Detection
    "detection_avg": "Detector Average",
    "detection_gmm": "GMM Detector",
    "detection_xgb": "XGBoost Detector",
    "detection_mlp": "MLP Detector",
    "detection_linear": "Linear Detector",

    # SDMetrics
    "sdmetrics_column_shapes": "SDMetrics Column Shapes",
    "sdmetrics_column_pair_trends": "SDMetrics Column Pair Trends",

    # Privacy
    "MemorizedFR": "Memorized Fraction",
    "new_row_synthesis": "New Row Synthesis",

    # k-Anonymity
    "k_anonymity_reference": "k-Anonymity (Reference)",
    "k_anonymity_synthetic": "k-Anonymity (Synthetic)",
    "k_ratio_synthetic_reference": "k-Anonymity Ratio synthetic vs reference",

    # Training / resource
    "training_time": "Training Time (s)",
    "generation_time": "Generation Time (s)",
    "model_size_mb": "Model Size (MB)",

    # Fairness
    "readmission_minority_share": "Minority class share (readm)",
    "readmission_collapsed": "Readmission Collapsed",

    # Utility
    "trtr_auroc": "TRTR AUROC",
    "tstr_auroc": "TSTR AUROC",
    "tstr_accuracy": "TSTR Accuracy",
    "tstr_precision": "TSTR Precision",
    "tstr_recall": "TSTR Recall",
    "tstr_f1": "TSTR F1",
    "tstr_fpr": "TSTR FPR",
    "tstr_fnr": "TSTR FNR",
    "tstr_tnr": "TSTR TNR",
}

# -------------------------------------------------------------------
# USER-DEFINED SIMPLIFIED METRICS
# -------------------------------------------------------------------

SIMPLIFIED_METRICS = [
    "trtr_auroc",
    "tstr_auroc",
    "sdmetrics_column_shapes",
    "sdmetrics_column_pair_trends",
]

# -------------------------------------------------------------------
# FULL METRIC SET
# -------------------------------------------------------------------
# mmd_corrected removed: it's only ever produced as a per-chain raw column
# (sd_quality_<GEN>.csv), not in the pooled sd_quality_summary_<GEN>.csv this
# table reads -- there is no *_mean/_low/_high aggregate for it to show.

FULL_METRICS = [
    # Alpha metrics
    "alpha_delta_precision_OC",
    "alpha_delta_coverage_OC",
    "alpha_authenticity_OC",

    # PRDC metrics
    "prdc_precision",
    "prdc_recall",
    "prdc_density",
    "prdc_coverage",

    # Distribution distances
    "tv_complement",
    "ks_complement",
    "wasserstein_dist",

    # JSD metrics
    "jsd_syndat",

    # Detection metrics
    "detection_avg",
    "detection_gmm",
    "detection_xgb",
    "detection_mlp",
    "detection_linear",

    # SDMetrics
    "sdmetrics_column_shapes",
    "sdmetrics_column_pair_trends",

    # Privacy
    "MemorizedFR",
    "new_row_synthesis",
    "k_anonymity_reference",
    "k_anonymity_synthetic",
    "k_ratio_synthetic_reference",

    # Training / resource
    "training_time",
    "generation_time",
    "model_size_mb",

    # Fairness
    "readmission_minority_share",
    "readmission_collapsed",

    # TSTR utility
    "trtr_auroc",
    "tstr_auroc",
    "tstr_accuracy",
    "tstr_precision",
    "tstr_recall",
    "tstr_f1",
    "tstr_fpr",
    "tstr_fnr",
    "tstr_tnr",
]

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def load_pooled_summary(generator: str) -> pd.DataFrame:
    """
    Load and merge the two pooled per-generation summary files for one
    generator/variant (e.g. 'ARF' or 'ARF_CORE'): sd_quality_summary_<GEN>.csv
    (quality/privacy/resource metrics) and tstr_trtr_summary_<GEN>.csv
    (TSTR/TRTR utility metrics). These used to be one combined
    summary_raw_Step7pfa/pfp file; they're now two separate files, merged
    here on 'generation' so extract_metric() can keep working against a
    single frame regardless of which file actually holds a given metric.
    """
    sd_path = os.path.join(QA_METRICS_POOLED_DIR, f"sd_quality_summary_{generator}.csv")
    tstr_path = os.path.join(QA_METRICS_POOLED_DIR, f"tstr_trtr_summary_{generator}.csv")

    df_sd = pd.read_csv(sd_path)
    df_tstr = pd.read_csv(tstr_path)
    return df_sd.merge(df_tstr, on="generation", how="outer", suffixes=("", "_tstr_dup"))


def extract_metric(df: pd.DataFrame, metric: str) -> dict:
    """
    Extract mean +/- CI for selected generations.
    Assumes columns follow the pattern <metric>_mean, <metric>_low,
    <metric>_high (the new pooled-summary column convention -- the old
    summary_raw_Step7 files used _ci95_lo/_ci95_hi instead).
    """
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_low"
    hi_col = f"{metric}_high"

    pretty_name = PRETTY.get(metric, metric)
    out = {"metric": pretty_name}

    for gen in GENERATIONS:
        row = df[df["generation"].astype(str) == gen]
        if row.empty or mean_col not in df.columns:
            out[gen] = ""
            continue

        m = float(row[mean_col].iloc[0])
        lo = float(row[lo_col].iloc[0])
        hi = float(row[hi_col].iloc[0])
        ci = hi - lo

        out[gen] = f"{m:.4f} ± {ci:.4f}"

    return out


# -------------------------------------------------------------------
# BUILD TABLES
# -------------------------------------------------------------------

def build_table(metrics: list) -> pd.DataFrame:
    rows = []

    for generator in GENERATORS:
        df = load_pooled_summary(generator)

        for metric in metrics:
            row = extract_metric(df, metric)
            row["generator"] = generator
            rows.append(row)

    cols = ["metric", "generator"] + GENERATIONS
    return pd.DataFrame(rows)[cols]


def main():
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)

    df_full = build_table(FULL_METRICS)
    df_full.to_csv(os.path.join(TABLE_OUTPUT_DIR, "metrics_table_for_article_full.csv"),
               index=False, encoding="utf-8")
    print("Saved metrics_table_for_article_full.csv")

    df_simple = build_table(SIMPLIFIED_METRICS)
    df_simple.to_csv(os.path.join(TABLE_OUTPUT_DIR, "metrics_table_for_article_simplified.csv"),
                     index=False, encoding="utf-8")
    print("Saved metrics_table_for_article_simplified.csv")


if __name__ == "__main__":
    main()