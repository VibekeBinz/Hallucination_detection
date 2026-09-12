"""
visualization/tables/rule_level_hr_overlap.py

Overlap between HR(bin=5) and each individual expert rule, at the record
level -- a per-rule breakdown of the same "of expert-flagged IDs, what
fraction did HR(bin=5) also catch" computation reported in
hr_expert_overlap.py, but split out rule by rule instead of collapsed into
Expert_any / Expert_impossible.

Reads:
  step1_hr_<GENERATOR>.csv          -- generation, folder, metric, ID
                                        (filtered to "HR_b5")
  step2_expert_rule_detail_<GENERATOR>.csv
                                     -- generation, folder, category, rule, ID
                                        (one row per record per individual
                                        rule that fired on it; category is
                                        "impossible" or "suspicious". Produced
                                        by the updated step2_expert_rules.py
                                        -- a generator that hasn't been rerun
                                        since that change won't have this file
                                        yet, and is skipped with a [WARN]
                                        rather than crashing.)

Same ID caveat as hr_expert_overlap.py: "ID" is a row index into that
(generator, folder, generation)'s own synthetic dataset, not a globally
unique record ID. Every join here is done within (generator, folder,
generation), never on ID alone.

For each generator and each (category, rule) pair, pooled across every
(folder, generation) cell where that rule fired at least once:

  rule_in_hr_b5_pooled_ratio = sum(overlap IDs across all cells) / sum(rule-flagged IDs across all cells)

  -- of every record that rule actually flagged (across the whole
  generator, not per cell), what fraction did HR(bin=5) also flag.
  rule_in_hr_b5_pooled_miss_rate is just 1 - that ratio: the fraction of
  genuinely rule-flagged records HR(bin=5) failed to catch. Both are
  record-level counts pooled across cells, not an average of per-cell
  percentages -- a rule that fires 100 times in one chain/generation and
  never elsewhere weighs in proportionally to those 100 records, not as
  one data point equal to a cell with 1 record.

This script does NOT classify rules as "combination-type" vs
"single-column-range" -- that requires knowing what each rule actually
checks, which lives in config/expert_rules.py, not in this data. It only
produces the sortable per-rule overlap table; use the "rule" column together
with your own knowledge of what each rule tests to see whether the
low-overlap rules cluster by type.

Output: {TABLE_OUTPUT_DIR}/rule_level_hr_overlap_for_article.csv -- one row
per (generator, category, rule), sorted by generator then ascending pooled
overlap ratio (the rules HR misses most stand at the top of each
generator's block), plus the underlying cell/ID counts.
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
_REPO_ROOT = _p

import pandas as pd

from config.pipeline_config import GENERATOR_VARIANTS, RESULTS_TABLES_DIR

ID_DIR = os.path.join(_REPO_ROOT, "summary_files", "hallucination_pipeline", "ids")
TABLE_OUTPUT_DIR = RESULTS_TABLES_DIR
OUTPUT_CSV_NAME = "rule_level_hr_overlap_for_article.csv"

GENERATORS = list(GENERATOR_VARIANTS.keys())
HR_METRIC = "HR_b5"


def load_hr_groups(path):
    """{(folder, generation): set(ID)} of HR_b5-flagged IDs. Empty dict if
    the file is missing or has no HR_b5 rows."""
    if not os.path.exists(path):
        print(f"  [WARN] missing file: {path}")
        return {}
    df = pd.read_csv(path)
    df = df[df["metric"] == HR_METRIC]
    if df.empty:
        return {}
    groups = {}
    for (folder, generation), grp in df.groupby(["folder", "generation"]):
        groups[(folder, generation)] = set(grp["ID"])
    return groups


def load_rule_groups(path):
    """
    {(category, rule): {(folder, generation): set(ID)}} from a
    step2_expert_rule_detail_<GEN>.csv file. Returns an empty dict (not an
    error) if the file doesn't exist yet -- the caller decides whether that's
    worth a [WARN].
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = {}
    for (category, rule), rule_grp in df.groupby(["category", "rule"]):
        groups = {}
        for (folder, generation), grp in rule_grp.groupby(["folder", "generation"]):
            groups[(folder, generation)] = set(grp["ID"])
        out[(category, rule)] = groups
    return out


def cell_fractions(numerator_groups, denominator_groups):
    """Same as hr_expert_overlap.py: for every (folder, generation) key with
    at least one flagged ID in denominator_groups, the fraction of those IDs
    also present in numerator_groups. Returns (key, fraction, overlap, denom)."""
    rows = []
    for key, denom_ids in denominator_groups.items():
        if not denom_ids:
            continue
        num_ids = numerator_groups.get(key, set())
        overlap = len(denom_ids & num_ids)
        rows.append((key, overlap / len(denom_ids), overlap, len(denom_ids)))
    return rows


def summarize(rows):
    """
    rows: list of (key, fraction, overlap_count, denom_count) from
    cell_fractions() -- one entry per (folder, generation) cell where this
    rule fired at least once. Pools the raw ID counts across every such
    cell into one record-level ratio (and its complement, the miss rate),
    rather than averaging per-cell percentages.
    """
    n_cells = len(rows)
    pooled_overlap = sum(r[2] for r in rows)
    pooled_denom = sum(r[3] for r in rows)
    pooled_ratio = pooled_overlap / pooled_denom if pooled_denom else float("nan")

    return {
        "n_cells": n_cells,
        "pooled_overlap_ids": pooled_overlap,
        "pooled_denom_ids": pooled_denom,
        "pooled_missed_ids": pooled_denom - pooled_overlap,
        "pooled_count_ratio": pooled_ratio,
        "pooled_miss_rate": (1 - pooled_ratio) if pooled_denom else float("nan"),
    }


def build_report():
    rows_out = []

    for generator in GENERATORS:
        hr_path = os.path.join(ID_DIR, f"step1_hr_{generator}.csv")
        rule_detail_path = os.path.join(ID_DIR, f"step2_expert_rule_detail_{generator}.csv")

        rule_groups_by_key = load_rule_groups(rule_detail_path)
        if rule_groups_by_key is None:
            print(f"{generator}: [WARN] no {os.path.basename(rule_detail_path)} found -- "
                  f"skipped (rerun step2_expert_rules.py for this generator to produce it).")
            continue

        hr_groups = load_hr_groups(hr_path)

        print(f"{generator}: {len(rule_groups_by_key)} distinct (category, rule) pairs found")

        for (category, rule), expert_groups in rule_groups_by_key.items():
            rule_in_hr = cell_fractions(hr_groups, expert_groups)  # denom = rule-flagged
            s = summarize(rule_in_hr)
            rows_out.append({
                "generator": generator,
                "category": category,
                "rule": rule,
                "rule_in_hr_b5_pooled_ratio": s["pooled_count_ratio"],
                "rule_in_hr_b5_pooled_miss_rate": s["pooled_miss_rate"],
                "rule_in_hr_b5_n_cells": s["n_cells"],
                "rule_in_hr_b5_pooled_denom_ids": s["pooled_denom_ids"],
                "rule_in_hr_b5_pooled_missed_ids": s["pooled_missed_ids"],
            })

    df = pd.DataFrame(rows_out)
    if not df.empty:
        # Ascending pooled ratio within each generator: the rules HR misses
        # most (lowest overlap / highest miss rate) sort to the top of each
        # generator's block.
        df.sort_values(["generator", "rule_in_hr_b5_pooled_ratio"], inplace=True)
    return df


def main():
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    df = build_report()
    out_path = os.path.join(TABLE_OUTPUT_DIR, OUTPUT_CSV_NAME)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {out_path}")
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("[WARN] no rows produced -- no step2_expert_rule_detail_<GEN>.csv "
              "files found. Rerun step2_expert_rules.py first.")


if __name__ == "__main__":
    main()