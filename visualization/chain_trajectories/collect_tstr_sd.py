"""
visualization/chain_trajectories/collect_tstr_sd.py

Consolidation of PLOT3b_TSTR_AUROC_trajectories_chains.py,
PLOT3c_correlation_trajectories_chains.py and PLOT3d_chain_trajectories.py —
those three were ~95% identical (same collect()/build_figure()/
scan_class_collapse()), differing only in which TSTR metric or which SD
column they were pointed at via a handful of module-level constants. That
difference is now just an entry in TSTR_METRICS / SD_METRICS below; the
collection and plotting code is written once, in this module and in
_shared.py.

Two data sources, same chain-folder tree (config.RAW_DATA_ROOT):
  1. TSTR/TRTR classification metrics — one JSON per generation, in
     <chain_dir>/metrics/tstr_..._gen_<g>_....json
  2. SD quality metrics (SDMetrics, alpha-precision, PRDC, detection, etc.)
     — one CSV per chain (all generations as rows), directly in
     <chain_dir>/<chain_dir_name>.csv

Writes into {CHAIN_TRAJECTORIES_FIGURES}:
  tstr_auroc_trajectories_{variant}_grid.html / .csv
  sd_<metric>_trajectories_{variant}_grid.html / .csv
"""

import glob
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)
from config.pipeline_config import PRETTY
from visualization.chain_trajectories._shared import (
    BASE, GENERATORS, GENERATOR_ORDER, VARIANTS, VARIANT_LABELS, GENERATIONS,
    EXPECTED_CHAINS, find_chain_dirs, chain_label, seed_pair,
    scan_class_collapse, attach_class_collapse, build_chain_figure,
    save_chain_outputs,
)

# ============================================================
# METRIC SELECTION
# ============================================================

# TSTR metrics to produce a figure for. "auroc" replaces PLOT3b; add more
# (e.g. "recall" for what PLOT3c effectively varied) by extending this list.
TSTR_METRICS = ["auroc"]

# SD-metrics CSV columns to produce a figure for. Includes what PLOT3d/PLOT3c
# were pointed at (sdmetrics_column_pair_trends, sdmetrics_column_shapes) plus
# the alpha/beta/authenticity trio used for hallucination-onset calibration.
SD_METRICS = [
    "sdmetrics_column_pair_trends",
    "sdmetrics_column_shapes",
    "alpha_delta_precision_OC",
    "alpha_delta_coverage_OC",
    "alpha_authenticity_OC",
    "detection_gmm", 
    "detection_mlp",
    "detection_xgb", 
    "detection_linear" 
]

METRIC_FLOORS = {"auroc": 0.5}
LOW_AUROC_THRESHOLD = 0.5
TSTR_BLOCK = "tstr_metrics"
TRTR_BLOCK = "trtr_metrics"
METRICS_SUBDIR = "metrics"


def pretty_label(name, fallback=None):
    return PRETTY.get(name, fallback if fallback is not None else name)


# ============================================================
# TSTR JSON HELPERS
# ============================================================

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


def numeric_items(flat):
    out = {}
    for k, v in flat.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = float(v)
    return out


def find_gen_file(metrics_dir, g):
    matches = sorted(glob.glob(os.path.join(metrics_dir, f"tstr_*_gen_{g}_*.json")))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"{len(matches)} files match gen_{g} in {metrics_dir}")
    return matches[0]


def collect_tstr(variant, metric):
    run_prefix = f"Step7{variant}"
    metric_key = f"{TSTR_BLOCK}.{metric}"
    baseline_key = f"{TRTR_BLOCK}.{metric}"
    auroc_key = f"{TSTR_BLOCK}.auroc"

    records = []
    missing = []

    for disp, tok in GENERATORS.items():
        chain_dirs = find_chain_dirs(run_prefix, tok)
        if not chain_dirs:
            print(f"  [WARN] no chain folders matched for {disp}")
            continue
        if len(chain_dirs) != EXPECTED_CHAINS:
            print(f"  [WARN] {len(chain_dirs)} chain folders found for {disp}, expected {EXPECTED_CHAINS}")

        for chain_dir in chain_dirs:
            label = chain_label(chain_dir)
            metrics_dir = os.path.join(chain_dir, METRICS_SUBDIR)
            if not os.path.isdir(metrics_dir):
                continue
            pair = seed_pair(os.path.basename(chain_dir))
            dseed, mseed = pair if pair else (None, None)

            for g in GENERATIONS:
                fp = find_gen_file(metrics_dir, g)
                if fp is None:
                    missing.append((disp, label, g))
                    continue
                with open(fp, "r", encoding="utf-8") as fh:
                    flat = numeric_items(flatten_json(json.load(fh)))
                if metric_key not in flat:
                    missing.append((disp, label, g))
                    continue

                auroc = flat.get(auroc_key, np.nan)
                records.append({
                    "generator": disp, "chain": label, "generation": g,
                    "value": flat[metric_key],
                    "value_trtr": flat.get(baseline_key, np.nan),
                    "auroc_tstr": auroc,
                    "low_auroc": bool(np.isfinite(auroc) and auroc <= LOW_AUROC_THRESHOLD),
                    "dseed": dseed, "mseed": mseed, "variant": variant,
                })

    df = pd.DataFrame.from_records(records)
    if missing:
        print(f"  [WARN] {len(missing)} generator/chain/generation combinations had no value")

    if not df.empty:
        collapse_map = scan_class_collapse(variants=[variant])
        df = attach_class_collapse(df, collapse_map)

    return df


def run_tstr(variant, metric):
    print(f"\n{'='*60}\nTSTR  variant={variant}  metric={metric}\n{'='*60}")
    df = collect_tstr(variant, metric)
    if df.empty:
        print(f"  [SKIP] no data for TSTR {metric} / {variant}")
        return None

    y_label = pretty_label(metric, f"TSTR {metric}")
    variant_label = VARIANT_LABELS[variant]
    title = f"{y_label} trajectories — {variant_label} ({df['chain'].nunique()} chains)"

    fig, height, width = build_chain_figure(
        df, y_label, title,
        show_baseline=True, show_floor=(metric in METRIC_FLOORS),
        floor_value=METRIC_FLOORS.get(metric),
        mark_low_auroc=True, low_auroc_threshold=LOW_AUROC_THRESHOLD,
        mark_class_collapse=True,
    )
    out_stem = f"tstr_{metric}_trajectories_{variant}"
    save_chain_outputs(df, fig, out_stem, height, width)
    return df


# ============================================================
# SD QUALITY METRICS (per-chain CSV)
# ============================================================

def find_sd_metrics_csv(chain_dir):
    name = os.path.basename(chain_dir)
    direct = os.path.join(chain_dir, f"{name}.csv")
    if os.path.isfile(direct):
        return direct
    matches = sorted(glob.glob(os.path.join(chain_dir, "*.csv")))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"{len(matches)} CSV files found directly in {chain_dir}")
    return matches[0]


def collect_sd(variant, metric_column):
    run_prefix = f"Step7{variant}"
    records = []
    missing = []

    for disp, tok in GENERATORS.items():
        chain_dirs = find_chain_dirs(run_prefix, tok)
        if not chain_dirs:
            print(f"  [WARN] no chain folders matched for {disp}")
            continue

        for chain_dir in chain_dirs:
            label = chain_label(chain_dir)
            csv_path = find_sd_metrics_csv(chain_dir)
            if csv_path is None:
                continue
            try:
                sd_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  [WARN] could not read {csv_path}: {e}")
                continue
            if "generation" not in sd_df.columns or metric_column not in sd_df.columns:
                continue

            pair = seed_pair(os.path.basename(chain_dir))
            dseed, mseed = pair if pair else (None, None)

            for g in GENERATIONS:
                row = sd_df.loc[sd_df["generation"] == g]
                if row.empty:
                    missing.append((disp, label, g))
                    continue
                value = row.iloc[0][metric_column]
                if pd.isna(value):
                    missing.append((disp, label, g))
                    continue
                records.append({
                    "generator": disp, "chain": label, "generation": g,
                    "value": float(value), "dseed": dseed, "mseed": mseed,
                    "variant": variant,
                })

    df = pd.DataFrame.from_records(records)
    if missing:
        print(f"  [WARN] {len(missing)} generator/chain/generation combinations had no value")

    if not df.empty:
        collapse_map = scan_class_collapse(variants=[variant])
        df = attach_class_collapse(df, collapse_map)

    return df


def run_sd(variant, metric_column):
    print(f"\n{'='*60}\nSD metric  variant={variant}  column={metric_column}\n{'='*60}")
    df = collect_sd(variant, metric_column)
    if df.empty:
        print(f"  [SKIP] no data for SD metric '{metric_column}' / {variant}")
        return None

    y_label = pretty_label(metric_column, metric_column)
    variant_label = VARIANT_LABELS[variant]
    title = f"{y_label} trajectories — {variant_label} ({df['chain'].nunique()} chains)"

    fig, height, width = build_chain_figure(df, y_label, title, mark_class_collapse=True)
    out_stem = f"sd_{metric_column}_trajectories_{variant}"
    save_chain_outputs(df, fig, out_stem, height, width)
    return df


def main():
    failed = []
    for variant in VARIANTS:
        for metric in TSTR_METRICS:
            try:
                run_tstr(variant, metric)
            except Exception:
                print(f"  [FAIL] TSTR variant={variant} metric={metric}")
                traceback.print_exc()
                failed.append(("tstr", variant, metric))

    for variant in VARIANTS:
        for metric_column in SD_METRICS:
            try:
                run_sd(variant, metric_column)
            except Exception:
                print(f"  [FAIL] SD variant={variant} metric={metric_column}")
                traceback.print_exc()
                failed.append(("sd", variant, metric_column))

    if failed:
        print(f"\n[WARN] {len(failed)} combination(s) failed: {failed}")
    print("\nDone.")


if __name__ == "__main__":
    main()
