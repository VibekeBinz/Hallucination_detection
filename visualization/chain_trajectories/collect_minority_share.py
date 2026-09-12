"""
visualization/chain_trajectories/collect_minority_share.py

Synthetic 'readmission' minority-class share (% TRUE) across recursive
generations, read directly from the DECODED synthetic CSVs -- the only place
this information exists (the TSTR JSON's confusion matrix reflects real
prevalence, not synthetic label balance; the SD-metrics CSV holds fidelity/
privacy scores, not label counts).

Data source: the decoded synthetic CSVs under
{EVALUATIONREADY_ROOT}/{GENERATOR_FOLDER}/{CHAIN}/{SD_SUBDIR}/, one file per
(chain, generation). GENERATOR_FOLDER is the generator name itself for
variant "pfa" (full feature set) and "{GENERATOR}_CORE" for variant "pfp"
(core feature set) -- the two variants live in separate top-level folders,
not mixed together under one, so which folder to look in depends on the
variant being collected.

Runs all four generators (ARF, CTGAN, DDPM, RTVAE) by default.

Writes into {CHAIN_TRAJECTORIES_FIGURES}:
  readmission_pct_true_trajectories_{variant}_grid.html / .csv
"""

import os
import sys

import numpy as np
import pandas as pd

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../collect_minority_share.py`, an IDE "Run" button, or
# `python -m visualization.chain_trajectories.collect_minority_share` from
# the repo root -- only the last of those already has the repo root on
# sys.path).
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

from visualization.chain_trajectories._shared import (
    EVALUATIONREADY_ROOT, SD_SUBDIR, VARIANTS, VARIANT_LABELS, GENERATIONS,
    EXPECTED_CHAINS, index_sd_chains, sd_generator_folder, chain_label,
    build_chain_figure, save_chain_outputs,
)

GENERATORS_TO_RUN = ["ARF", "CTGAN", "DDPM", "RTVAE"]

LABEL_COLUMN = "readmission"
TRUE_TOKEN = "true"

SHOW_REAL_BASELINE = True
REAL_PREVALENCE_PCT = 1273 / 10000 * 100   # 12.73, real test-set prevalence


def find_decoded_csv(chain_dir, g):
    """Decoded synthetic CSV for one (chain, generation), matched the same
    way _shared.find_sd_gen_csv matches SD-tree files (a "_gen_{g}_"
    substring under {chain_dir}/{SD_SUBDIR}/), but raising instead of
    silently picking one if more than one file matches."""
    sd_dir = os.path.join(chain_dir, SD_SUBDIR)
    if not os.path.isdir(sd_dir):
        return None
    matches = sorted(
        f for f in os.listdir(sd_dir) if f"_gen_{g}_" in f and f.endswith(".csv")
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"{len(matches)} files match gen_{g} in {sd_dir}: {matches}")
    return os.path.join(sd_dir, matches[0])


def read_label_share(csv_path):
    try:
        col = pd.read_csv(csv_path, usecols=[LABEL_COLUMN])[LABEL_COLUMN]
    except Exception:
        return None
    col = col.dropna()
    if col.empty:
        return {"n_total": 0, "n_true": 0, "pct_true": np.nan, "n_classes": 0}
    norm = col.astype(str).str.strip().str.lower().replace(
        {"1": "true", "0": "false", "yes": "true", "no": "false"})
    n_total = len(norm)
    n_true = int((norm == TRUE_TOKEN).sum())
    return {"n_total": n_total, "n_true": n_true,
            "pct_true": 100.0 * n_true / n_total, "n_classes": int(norm.nunique())}


def collect(variant):
    run_prefix = f"Step7{variant}"
    records = []
    missing = []

    for disp in GENERATORS_TO_RUN:
        folder_disp = sd_generator_folder(disp, variant)
        index = index_sd_chains(EVALUATIONREADY_ROOT, folder_disp, run_prefix)
        if not index:
            print(f"  [WARN] no chain folders matched for {disp}")
            continue
        if len(index) != EXPECTED_CHAINS:
            print(f"  [WARN] {len(index)} chain folders for {disp}, expected {EXPECTED_CHAINS}")

        for (dseed, mseed), chain_dir in index.items():
            label = chain_label(chain_dir)

            for g in GENERATIONS:
                fp = find_decoded_csv(chain_dir, g)
                if fp is None:
                    missing.append((disp, label, g))
                    continue
                stats = read_label_share(fp)
                if stats is None:
                    missing.append((disp, label, g))
                    continue
                records.append({
                    "generator": disp, "chain": label, "dseed": dseed, "mseed": mseed,
                    "generation": g, "value": stats["pct_true"],
                    "n_classes": stats["n_classes"],
                    "class_collapse": bool(stats["n_classes"] <= 1),
                    "variant": variant,
                })

    df = pd.DataFrame.from_records(records)
    if missing:
        print(f"  [WARN] {len(missing)} generator/chain/generation combinations had no value")
    return df


def run(variant):
    print(f"\n{'='*60}\nreadmission minority share  variant={variant}\n{'='*60}")
    df = collect(variant)
    if df.empty:
        print(f"  [SKIP] no data for variant '{variant}'")
        return None

    y_label = f"Synthetic '{LABEL_COLUMN}' — % TRUE"
    variant_label = VARIANT_LABELS[variant]
    title = f"{y_label} trajectories — {variant_label} ({df['chain'].nunique()} chains)"

    fig, height, width = build_chain_figure(
        df, y_label, title, gens_present=GENERATORS_TO_RUN,
        mark_class_collapse=True,
    )
    if SHOW_REAL_BASELINE:
        fig.add_hline(y=REAL_PREVALENCE_PCT, line=dict(color="#444444", width=2, dash="dash"))

    out_stem = f"readmission_pct_true_trajectories_{variant}"
    save_chain_outputs(df, fig, out_stem, height, width)
    return df


def main():
    for variant in VARIANTS:
        run(variant)
    print("\nDone.")


if __name__ == "__main__":
    main()