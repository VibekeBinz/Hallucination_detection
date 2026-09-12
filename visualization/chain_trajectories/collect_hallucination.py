"""
visualization/chain_trajectories/collect_hallucination.py

Consolidation of PLOT3f_chain_trajectories_HR_LLM.py into the shared
chain-trajectory package: chain-by-chain HR (bin=5), Expert rules
(Expert_any), LLM-only, and combined Expert+LLM (true set union, not a
naive sum) trajectories, using build_chain_figure() from _shared.py instead
of its own copy of the grid-layout code.

Data sources (all flat under {HALLUC_IDS_DIR} -- one file per generator,
generator name embedded in the filename rather than a per-generator
subfolder -- one row per FLAGGED record):

  step1_hr_{GENERATOR}[_CORE].csv     metric == "HR_b{bin}", filtered to b5
  step2_expert_{GENERATOR}.csv        metric in {Expert_any, Expert_impossible,
                                       Expert_suspicious}, filtered to Expert_any
  step3_llm_{GENERATOR}[_CORE].csv    metric == "LLM"; only 3/15 chains per
                                       generator, only generations {0,5,10,19}

step2_expert_rule_detail_ARF.csv also lives in this directory (a per-rule
diagnostic export, columns generation/folder/category/rule/ID -- no
"metric" column) and would blow up pd.read_csv's usecols= if it were ever
picked up alongside the real per-generator files. _generator_csvs() filters
it out by requiring an uppercase generator name right after the prefix.

The official step1_summary_{GENERATOR}[_CORE].csv reference used by
cross_check_hr() lives in the sibling {HALLUC_SUMMARY_DIR}, not here.

IMPORTANT — three different "chain absent from file" semantics, preserved
unchanged from the legacy script (this is real, previously-debugged logic —
see the RTVAE bin=5 / generation 19 case in the docstring below — not
something to simplify away):
  HR, Expert : absence at a generation the file covers elsewhere means zero
               flagged records -- filled to rate=0, but ONLY when that
               generation has rows for >= min_coverage of the known chain
               roster (default 50%); otherwise the generation is DROPPED,
               not zero-filled, so an export gap shows as a break in the
               line rather than a fake floor at 0.
  LLM        : absence of a WHOLE CHAIN means that chain was never
               evaluated (only 3/15 were) -- never filled to 0, simply
               omitted from that trace.
  Combined   : Expert_any UNION LLM, true set union of record IDs (not
               summed rates), restricted to the (chain, generation) pairs
               LLM actually covers.

Writes into {CHAIN_TRAJECTORIES_FIGURES}:
  hallucination_HR_b5_trajectories_{variant}_grid.html / .csv
  hallucination_Expert_any_trajectories_{variant}_grid.html / .csv
  hallucination_LLM_only_trajectories_{variant}_grid.html / .csv
  hallucination_Expert_plus_LLM_trajectories_{variant}_grid.html / .csv
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../collect_hallucination.py`, an IDE "Run" button, or
# `python -m visualization.chain_trajectories.collect_hallucination` from
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

from config.pipeline_config import HALLUC_IDS_DIR, HALLUC_SUMMARY_DIR
from visualization.chain_trajectories._shared import (
    GENERATOR_ORDER, VARIANTS, VARIANT_LABELS,
    scan_class_collapse, attach_class_collapse, build_chain_figure,
    save_chain_outputs, chain_label,
)

HR_BIN = 5
N_RECORDS_PER_DATASET = 10_000

FOLDER_RE = re.compile(r"Step7(pfa|pfp)_dseed(\d+)_synthcity_(\w+?)_mseed(\d+)")

STEP1_HR_PREFIX = "step1_hr_"
STEP2_EXPERT_PREFIX = "step2_expert_"
STEP3_LLM_PREFIX = "step3_llm_"


def _generator_csvs(root: Path, prefix: str) -> list:
    """
    Glob root for '{prefix}*.csv' and keep only files whose name continues
    with an uppercase generator name (ARF, ARF_CORE, CTGAN, ...) right
    after the prefix.

    This excludes per-rule diagnostic exports like
    step2_expert_rule_detail_ARF.csv (continues with lowercase "rule...",
    and is missing the "metric" column every loader here usecols= on).
    Filtering is done here in Python with str.isupper() rather than by
    anchoring the glob pattern itself (e.g. "step2_expert_[A-Z]*.csv"):
    pathlib.Path.glob/rglob matches case-insensitively on Windows, so a
    "[A-Z]" character class there also matches lowercase letters and
    silently defeats the whole point of anchoring it.
    """
    files = sorted(root.rglob(f"{prefix}*.csv"))
    return [f for f in files if f.name[len(prefix):len(prefix) + 1].isupper()]


METRIC_LABELS = {
    "HR_b5": f"Hallucination Rate (Pilgram, bin={HR_BIN})",
    "Expert_any": "Expert Rules (any)",
    "LLM_only": "LLM-flagged only",
    "Expert_plus_LLM": "Expert Rules + LLM (combined)",
}


def parse_folder(folder: str):
    m = FOLDER_RE.match(folder)
    if not m:
        return None
    pfx, dseed, generator, mseed = m.groups()
    return pfx, generator.upper(), int(dseed), int(mseed)


# ============================================================
# LOADERS -- HR and Expert (zero-fill semantics)
# ============================================================

def _load_flag_counts(root: Path, prefix: str, metric_filter: str, source_label: str):
    files = _generator_csvs(root, prefix)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.csv files found under {root}")

    counts, chains_seen, gens_seen, chains_per_gen = {}, set(), set(), {}

    for f in files:
        df = pd.read_csv(f, usecols=["generation", "folder", "metric", "ID"])
        for folder in df["folder"].unique():
            parsed = parse_folder(folder)
            if parsed is None:
                print(f"  WARNING: could not parse folder '{folder}' in {f.name}")
                continue
            chains_seen.add(parsed)

        sub = df[df["metric"] == metric_filter]
        gens_seen.update(int(g) for g in sub["generation"].unique())

        grouped = sub.groupby(["folder", "generation"]).size()
        for (folder, gen), n in grouped.items():
            parsed = parse_folder(folder)
            if parsed is None:
                continue
            variant, generator, dseed, mseed = parsed
            gen = int(gen)
            counts[(variant, generator, dseed, mseed, gen)] = int(n)
            chains_per_gen.setdefault(gen, set()).add((variant, generator, dseed, mseed))

    print(f"  [{source_label}] {len(files)} file(s), {len(chains_seen)} chains, "
          f"generations {sorted(gens_seen)}")

    n_chains_total = len(chains_seen)
    for gen in sorted(gens_seen):
        n_covered = len(chains_per_gen.get(gen, set()))
        if n_chains_total and n_covered < n_chains_total:
            pct = 100 * n_covered / n_chains_total
            flag = " <-- LOW COVERAGE" if pct < 50 else ""
            print(f"    generation {gen}: {n_covered}/{n_chains_total} chains "
                  f"have a '{metric_filter}' row ({pct:.0f}%){flag}")

    return counts, chains_seen, gens_seen


def load_rate_trajectory(root: Path, prefix: str, metric_filter: str,
                          source_label: str, min_coverage: float = 0.5) -> pd.DataFrame:
    counts, chains_seen, gens_seen = _load_flag_counts(root, prefix, metric_filter, source_label)
    n_chains_total = len(chains_seen)

    chains_per_gen = {}
    for (variant, generator, dseed, mseed, gen) in counts:
        chains_per_gen.setdefault(gen, set()).add((variant, generator, dseed, mseed))

    dropped_gens = []
    rows = []
    for gen in gens_seen:
        n_covered = len(chains_per_gen.get(gen, set()))
        coverage = n_covered / n_chains_total if n_chains_total else 0
        if coverage < min_coverage:
            dropped_gens.append((gen, n_covered, n_chains_total))
            continue
        for (variant, generator, dseed, mseed) in chains_seen:
            n = counts.get((variant, generator, dseed, mseed, gen), 0)
            rows.append(dict(variant=variant, generator=generator, dseed=dseed,
                             mseed=mseed, generation=gen, value=n / N_RECORDS_PER_DATASET))

    if dropped_gens:
        print(f"  [{source_label}] DROPPED (not zero-filled, coverage < {min_coverage:.0%}): " +
              ", ".join(f"gen {g} ({c}/{t} chains)" for g, c, t in dropped_gens))

    return pd.DataFrame(rows)


def load_llm_trajectory(root: Path) -> pd.DataFrame:
    counts, chains_seen, _ = _load_flag_counts(root, STEP3_LLM_PREFIX, "LLM", "LLM")

    chain_gens = {}
    for f in _generator_csvs(root, STEP3_LLM_PREFIX):
        df = pd.read_csv(f, usecols=["generation", "folder", "metric", "ID"])
        df = df[df["metric"] == "LLM"]
        for folder, gen in df[["folder", "generation"]].drop_duplicates().itertuples(index=False):
            parsed = parse_folder(folder)
            if parsed is None:
                continue
            chain_gens.setdefault(parsed, set()).add(int(gen))

    rows = []
    for chain_key, gens in chain_gens.items():
        variant, generator, dseed, mseed = chain_key
        for gen in gens:
            n = counts.get((variant, generator, dseed, mseed, gen), 0)
            rows.append(dict(variant=variant, generator=generator, dseed=dseed,
                             mseed=mseed, generation=gen, value=n / N_RECORDS_PER_DATASET))
    return pd.DataFrame(rows)


def load_combined_trajectory(root: Path) -> pd.DataFrame:
    expert_ids = {}
    for f in _generator_csvs(root, STEP2_EXPERT_PREFIX):
        df = pd.read_csv(f, usecols=["generation", "folder", "metric", "ID"])
        df = df[df["metric"] == "Expert_any"]
        for (folder, gen), group in df.groupby(["folder", "generation"]):
            parsed = parse_folder(folder)
            if parsed is None:
                continue
            variant, generator, dseed, mseed = parsed
            key = (variant, generator, dseed, mseed, int(gen))
            expert_ids.setdefault(key, set()).update(group["ID"].tolist())

    llm_ids = {}
    for f in _generator_csvs(root, STEP3_LLM_PREFIX):
        df = pd.read_csv(f, usecols=["generation", "folder", "metric", "ID"])
        df = df[df["metric"] == "LLM"]
        for (folder, gen), group in df.groupby(["folder", "generation"]):
            parsed = parse_folder(folder)
            if parsed is None:
                continue
            variant, generator, dseed, mseed = parsed
            key = (variant, generator, dseed, mseed, int(gen))
            llm_ids.setdefault(key, set()).update(group["ID"].tolist())

    rows = []
    for key in llm_ids:
        variant, generator, dseed, mseed, gen = key
        union = expert_ids.get(key, set()) | llm_ids[key]
        rows.append(dict(variant=variant, generator=generator, dseed=dseed,
                         mseed=mseed, generation=gen, value=len(union) / N_RECORDS_PER_DATASET))
    print(f"  [Combined] {len(rows)} (chain, generation) points "
          f"(limited to LLM's coverage)")
    return pd.DataFrame(rows)


# ============================================================
# CROSS-CHECK AGAINST OFFICIAL step1_summary_*.csv
# ============================================================

def load_hr_summary_reference(root: Path, hr_bin: int) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("step1_summary_*.csv")):
        matched_gen = next((g for g in GENERATOR_ORDER if g in f.stem.upper()), None)
        if matched_gen is None:
            continue
        variant = "pfp" if "CORE" in f.stem.upper() else "pfa"
        df = pd.read_csv(f)
        if "generation" not in df.columns or "bin" not in df.columns or "Pilgram_HR_mean" not in df.columns:
            continue
        df["gen_num"] = df["generation"].astype(str).str.extract(r"(\d+)").astype(int)
        sub = df[df["bin"] == hr_bin]
        for _, r in sub.iterrows():
            rows.append(dict(variant=variant, generator=matched_gen,
                             generation=int(r["gen_num"]), official_mean=float(r["Pilgram_HR_mean"])))
    return pd.DataFrame(rows)


def cross_check_hr(hr_df: pd.DataFrame, root: Path, hr_bin: int, tolerance: float = 0.10):
    ref = load_hr_summary_reference(root, hr_bin)
    if ref.empty:
        print("  [summary xcheck] no step1_summary_*.csv files found -- skipping")
        return
    ours = hr_df.groupby(["variant", "generator", "generation"], as_index=False)["value"].mean()
    merged = ours.merge(ref, on=["variant", "generator", "generation"], how="inner")
    merged["diff"] = (merged["value"] - merged["official_mean"]).abs()
    bad = merged[merged["diff"] > tolerance].sort_values("diff", ascending=False)
    if bad.empty:
        print(f"  [summary xcheck] OK -- all {len(merged)} points within {tolerance:.0%}")
    else:
        print(f"  [summary xcheck] WARNING -- {len(bad)}/{len(merged)} points differ "
              f"by more than {tolerance:.0%}")


# ============================================================
# RUN
# ============================================================

def run_one(metric_key: str, variant: str, df_all: pd.DataFrame, collapse_map: dict):
    sub = df_all[df_all["variant"] == variant].copy()
    if sub.empty:
        print(f"  [SKIP] {metric_key} / {variant}: no data")
        return

    sub["chain"] = sub.apply(lambda r: chain_label(f"dseed{r['dseed']}_mseed{r['mseed']}"), axis=1)
    sub = attach_class_collapse(sub, collapse_map)

    y_label = METRIC_LABELS.get(metric_key, metric_key)
    variant_label = VARIANT_LABELS[variant]
    title = f"{y_label} trajectories — {variant_label} ({sub['chain'].nunique()} chains)"

    fig, height, width = build_chain_figure(sub, y_label, title, mark_class_collapse=True)
    out_stem = f"hallucination_{metric_key}_trajectories_{variant}"
    save_chain_outputs(sub, fig, out_stem, height, width)


def main():
    root = Path(HALLUC_IDS_DIR)
    summary_root = Path(HALLUC_SUMMARY_DIR)

    print("Loading HR (bin=5)...")
    hr_df = load_rate_trajectory(root, STEP1_HR_PREFIX, f"HR_b{HR_BIN}", f"HR_b{HR_BIN}")
    cross_check_hr(hr_df, summary_root, HR_BIN)

    print("\nLoading Expert rules (Expert_any)...")
    expert_df = load_rate_trajectory(root, STEP2_EXPERT_PREFIX, "Expert_any", "Expert_any")

    print("\nLoading LLM (LLM-only, not zero-filled)...")
    llm_df = load_llm_trajectory(root)

    print("\nLoading combined Expert+LLM (true union, LLM's coverage only)...")
    combined_df = load_combined_trajectory(root)

    collapse_map = scan_class_collapse(variants=VARIANTS)

    datasets = {
        "HR_b5": hr_df, "Expert_any": expert_df,
        "LLM_only": llm_df, "Expert_plus_LLM": combined_df,
    }

    for metric_key, df_all in datasets.items():
        print(f"\n=== {metric_key} ===")
        for variant in VARIANTS:
            run_one(metric_key, variant, df_all, collapse_map)

    print("\nDone.")


if __name__ == "__main__":
    main()