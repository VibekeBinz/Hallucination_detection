"""
rtvae_tstr_coverage_check.py

Checks whether RTVAE's low TSTR/TRTR row count (46/300 full, 49/300 core,
vs 300/300 for every other generator) is explained by class collapse, or
is a real gap worth investigating further.

For each RTVAE chain: compares the last generation with TSTR/TRTR data
(from tstr_trtr_RTVAE[_CORE].csv) against that chain's class-collapse
generation (from minority_share.csv). If TSTR data stops at-or-just-before
collapse for every chain, the low count is fully explained. Any chain
where TSTR data is missing well before collapse, or continues well after
it, is the one to look at.

Run: python rtvae_tstr_coverage_check.py
"""

import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

import pandas as pd

from config.pipeline_config import PER_CHAIN_DIR

# Sourced from the pipeline's own PER_CHAIN_DIR rather than hardcoded here --
# this used to point at summary_files/qa_metrics/per_chain/, which no
# longer matches where PER_CHAIN_DIR actually resolves to
# (summary_files/per_chain/). Importing it removes that whole class of
# drift: if PER_CHAIN_DIR ever moves again, this script moves with it.
MINORITY_SHARE_PATH = os.path.join(PER_CHAIN_DIR, "minority_share.csv")
TSTR_FULL_PATH = os.path.join(PER_CHAIN_DIR, "tstr_trtr_RTVAE.csv")
TSTR_CORE_PATH = os.path.join(PER_CHAIN_DIR, "tstr_trtr_RTVAE_CORE.csv")


def check_variant(tstr_path, variant_label, minority):
    print(f"\n{'=' * 70}\nRTVAE / {variant_label}\n{'=' * 70}")
    tstr = pd.read_csv(tstr_path)

    last_tstr_gen = (
        tstr.groupby(["dseed", "mseed"])["generation"]
        .max()
        .reset_index(name="last_tstr_generation")
    )

    mshare = minority[(minority["generator"] == "rtvae") & (minority["variant"] == variant_label)]
    collapse_gen = (
        mshare[mshare["minority_share"] == 0]
        .groupby(["dseed", "mseed"])["generation"]
        .min()
        .reset_index(name="collapse_generation")
    )
    max_available_gen = (
        mshare.groupby(["dseed", "mseed"])["generation"]
        .max()
        .reset_index(name="max_generation_in_minority_share")
    )

    merged = last_tstr_gen.merge(collapse_gen, on=["dseed", "mseed"], how="outer")
    merged = merged.merge(max_available_gen, on=["dseed", "mseed"], how="outer")
    merged = merged.sort_values(["dseed", "mseed"])

    merged["diff_vs_collapse"] = merged["last_tstr_generation"] - merged["collapse_generation"]

    print(merged.to_string(index=False))

    print("\nSummary:")
    no_collapse = merged["collapse_generation"].isna()
    if no_collapse.any():
        print(f"  {no_collapse.sum()} chain(s) never collapsed (per minority_share.csv) "
              f"but still have low TSTR coverage -- these need a closer look, "
              f"not explained by collapse.")
        print(merged[no_collapse].to_string(index=False))

    has_both = merged["diff_vs_collapse"].notna()
    if has_both.any():
        d = merged.loc[has_both, "diff_vs_collapse"]
        exact_or_before = (d <= 0).sum()
        after = (d > 0).sum()
        print(f"  chains where last TSTR generation is AT or BEFORE collapse: "
              f"{exact_or_before}/{has_both.sum()}")
        if after:
            print(f"  [FLAG] {after} chain(s) have TSTR data AFTER their collapse "
                  f"generation -- shouldn't be possible if TSTR truly can't be "
                  f"computed post-collapse. Worth checking those rows directly.")
            print(merged.loc[has_both & (merged['diff_vs_collapse'] > 0)].to_string(index=False))


def main():
    minority = pd.read_csv(MINORITY_SHARE_PATH)
    minority["generator"] = minority["generator"].str.lower()
    minority["variant"] = minority["variant"].str.lower()

    check_variant(TSTR_FULL_PATH, "full", minority)
    check_variant(TSTR_CORE_PATH, "core", minority)


if __name__ == "__main__":
    main()