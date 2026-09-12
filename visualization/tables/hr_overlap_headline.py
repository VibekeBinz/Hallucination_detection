"""
visualization/tables/hr_overlap_headline.py

Headline HR(bin=5)-vs-expert-rule overlap numbers for the article, plus the
per-rule "which specific rules did HR miss" breakdown behind each headline
number.

Two scope groups, reported separately rather than pooled together -- a
generator's _CORE variant is missing whole fields (e.g. pregnancy status,
age-linked checks) that several rules and even one whole headline metric
depend on, so blending the two regimes into one number would misrepresent
both, not average them meaningfully:
  full: every non-core generator (ARF, CTGAN, DDPM, RTVAE)
  core: every core-feature-set generator (ARF_CORE, CTGAN_CORE, ...)

Two headline metrics per scope, using the already-deduplicated Expert_any /
Expert_impossible metric from step2_expert_<GENERATOR>.csv -- NOT a sum over
the individual per-rule counts. A record that trips more than one rule at
once must only be counted once toward "how many records did the expert
rules flag"; Expert_any/Expert_impossible already guarantee that, and naively
summing the per-rule table (rule_level_hr_overlap.py's output) would
double-count any record that fired multiple rules simultaneously:

  "clinical" = Expert_any        (impossible + suspicious combined)
  "impossible" = Expert_impossible

  headline_miss_rate = 1 - |expert-flagged IDs (intersect) HR_b5-flagged IDs| / |expert-flagged IDs|

  pooled at the record level across every (generator, folder, generation)
  cell in the scope group -- same convention as hr_expert_overlap.py's "ALL"
  row, just split into the two scope groups instead of merged into one, and
  reported as a pooled ratio/miss-rate rather than a mean-of-cells +/- CI
  (see rule_level_hr_overlap.py for why: the record-level pooled ratio is
  what "X% of the violations were missed" actually means).

Per-rule breakdown: for each scope group, every individual (category, rule)
from step2_expert_rule_detail_<GENERATOR>.csv, pooled across the scope's
generators the same way. The "clinical" list is every rule in the scope
(both categories, ranked by how much HR missed -- most-ignored first); the
"impossible" list is the same rows filtered to category=="impossible".

Cross-check: for each scope, the union of every individual rule's flagged
IDs (from the rule-detail file) is compared against the Expert_any total
(from step2_expert_<GENERATOR>.csv) -- these two are independently computed
and should agree exactly. A mismatch means the rule-detail file and the
Expert_any column were produced by different versions of
step2_expert_rules.py for at least one generator in that scope, and is
printed as a [WARN] rather than silently trusted.

Reads (ID_DIR):
  step1_hr_<GENERATOR>.csv              -- generation, folder, metric, ID
  step2_expert_<GENERATOR>.csv          -- generation, folder, metric, ID
  step2_expert_rule_detail_<GENERATOR>.csv
                                         -- generation, folder, category, rule, ID
                                            (skipped with a [WARN] per generator
                                            if not yet produced for it)

Output: {TABLE_OUTPUT_DIR}/
  hr_overlap_headline_for_article.csv       -- one row per (scope, metric)
  hr_overlap_headline_rules_for_article.csv -- one row per (scope, category, rule)
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

from config.pipeline_config import GENERATOR_VARIANTS, RESULTS_TABLES_DIR, is_core

ID_DIR = os.path.join(_REPO_ROOT, "summary_files", "hallucination_pipeline", "ids")
TABLE_OUTPUT_DIR = RESULTS_TABLES_DIR
HEADLINE_CSV_NAME = "hr_overlap_headline_for_article.csv"
RULES_CSV_NAME = "hr_overlap_headline_rules_for_article.csv"

HR_METRIC = "HR_b5"

# label -> metric string in step2_expert_<GENERATOR>.csv.
HEADLINE_METRICS = {
    "clinical (any)": "Expert_any",
    "impossible": "Expert_impossible",
}

# "full" vs "core" scope groups, derived from the same single source of
# truth the rest of the pipeline uses rather than a second hardcoded list.
SCOPE_GROUPS = {"full": [], "core": []}
for _name in GENERATOR_VARIANTS:
    SCOPE_GROUPS["core" if is_core(_name) else "full"].append(_name)


# -------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------

def load_hr_groups(generator):
    """{(folder, generation): set(ID)} of HR_b5-flagged IDs for one generator."""
    path = os.path.join(ID_DIR, f"step1_hr_{generator}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] missing file: {path}")
        return {}
    df = pd.read_csv(path)
    df = df[df["metric"] == HR_METRIC]
    if df.empty:
        return {}
    return {key: set(grp["ID"]) for key, grp in df.groupby(["folder", "generation"])}


def load_expert_metric_groups(generator, metric):
    """{(folder, generation): set(ID)} for one metric value
    (Expert_any / Expert_impossible) in one generator's step2_expert file."""
    path = os.path.join(ID_DIR, f"step2_expert_{generator}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] missing file: {path}")
        return {}
    df = pd.read_csv(path)
    df = df[df["metric"] == metric]
    if df.empty:
        return {}
    return {key: set(grp["ID"]) for key, grp in df.groupby(["folder", "generation"])}


def load_rule_groups(generator):
    """{(category, rule): {(folder, generation): set(ID)}} for one generator.
    Returns {} (not an error) if the rule-detail file doesn't exist yet for
    this generator -- callers decide whether/how to warn."""
    path = os.path.join(ID_DIR, f"step2_expert_rule_detail_{generator}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] no rule-detail file for {generator} -- excluded from the "
              f"per-rule breakdown (rerun step2_expert_rules.py for this generator).")
        return {}
    df = pd.read_csv(path)
    out = {}
    for (category, rule), rule_grp in df.groupby(["category", "rule"]):
        out[(category, rule)] = {
            key: set(grp["ID"]) for key, grp in rule_grp.groupby(["folder", "generation"])
        }
    return out


def merge_groups(group_dicts):
    """Union multiple {(folder, generation): set(ID)} dicts into one. Safe
    across generators because a chain's folder string already embeds which
    generator it belongs to (e.g. "..._synthcity_arf_..." vs
    "..._synthcity_ctgan_..."), so no (folder, generation) key can collide
    between two different generators."""
    merged = {}
    for d in group_dicts:
        merged.update(d)
    return merged


# -------------------------------------------------------------------
# OVERLAP MATH -- same convention as hr_expert_overlap.py / rule_level_hr_overlap.py
# -------------------------------------------------------------------

def cell_fractions(numerator_groups, denominator_groups):
    rows = []
    for key, denom_ids in denominator_groups.items():
        if not denom_ids:
            continue
        num_ids = numerator_groups.get(key, set())
        overlap = len(denom_ids & num_ids)
        rows.append((key, overlap / len(denom_ids), overlap, len(denom_ids)))
    return rows


def summarize(rows):
    n_cells = len(rows)
    pooled_overlap = sum(r[2] for r in rows)
    pooled_denom = sum(r[3] for r in rows)
    pooled_ratio = pooled_overlap / pooled_denom if pooled_denom else float("nan")
    return {
        "n_cells": n_cells,
        "pooled_overlap_ids": pooled_overlap,
        "pooled_denom_ids": pooled_denom,
        "pooled_missed_ids": pooled_denom - pooled_overlap,
        "pooled_ratio": pooled_ratio,
        "pooled_miss_rate": (1 - pooled_ratio) if pooled_denom else float("nan"),
    }


def union_across_rules(rule_groups_by_key):
    """{(folder, generation): set(ID)} = union of every rule's flagged IDs
    for that cell, from a {(category, rule): {(folder, generation): set(ID)}}
    structure. Used only for the Expert_any cross-check below."""
    union = {}
    for groups in rule_groups_by_key.values():
        for key, ids in groups.items():
            union.setdefault(key, set()).update(ids)
    return union


# -------------------------------------------------------------------
# PER-SCOPE DATA LOADING
# -------------------------------------------------------------------

def load_scope_data(generators):
    hr_groups = merge_groups(load_hr_groups(g) for g in generators)

    expert_metric_groups = {
        label: merge_groups(load_expert_metric_groups(g, metric) for g in generators)
        for label, metric in HEADLINE_METRICS.items()
    }

    per_generator_rule_groups = [load_rule_groups(g) for g in generators]
    all_rule_keys = sorted({k for d in per_generator_rule_groups for k in d})
    rule_groups = {
        key: merge_groups(d.get(key, {}) for d in per_generator_rule_groups)
        for key in all_rule_keys
    }

    return hr_groups, expert_metric_groups, rule_groups


def cross_check_any(expert_any_groups, rule_groups):
    union = union_across_rules(rule_groups)
    any_total = sum(len(ids) for ids in expert_any_groups.values())
    union_total = sum(len(ids) for ids in union.values())
    if any_total != union_total:
        print(f"  [WARN] Expert_any total ({any_total}) != union of every individual "
              f"rule from the rule-detail files ({union_total}) -- the rule-detail "
              f"files and the Expert_any column may be out of sync for at least one "
              f"generator in this scope (different step2_expert_rules.py run). "
              f"The headline 'clinical' number above uses Expert_any directly; the "
              f"per-rule list below uses the rule-detail files -- treat them as "
              f"possibly inconsistent until this is resolved.")
    else:
        print(f"  [OK] Expert_any total matches the union of every individual rule "
              f"({any_total} IDs).")


# -------------------------------------------------------------------
# REPORT BUILDERS
# -------------------------------------------------------------------

def build_headline_table(scope_data):
    rows_out = []
    for scope, (hr_groups, expert_metric_groups, rule_groups) in scope_data.items():
        print(f"\n=== scope: {scope} ===")
        cross_check_any(expert_metric_groups["clinical (any)"], rule_groups)

        for label in HEADLINE_METRICS:
            expert_groups = expert_metric_groups[label]
            n_flagged_cells = sum(1 for ids in expert_groups.values() if ids)
            print(f"  [{label}] {n_flagged_cells} (generator, chain, generation) cells flagged")

            fractions = cell_fractions(hr_groups, expert_groups)  # denom = expert-flagged
            s = summarize(fractions)
            rows_out.append({
                "scope": scope,
                "metric_label": label,
                "expert_metric": HEADLINE_METRICS[label],
                "pooled_ratio": s["pooled_ratio"],
                "pooled_miss_rate": s["pooled_miss_rate"],
                "n_cells": s["n_cells"],
                "pooled_denom_ids": s["pooled_denom_ids"],
                "pooled_missed_ids": s["pooled_missed_ids"],
            })
    return pd.DataFrame(rows_out)


def build_rule_table(scope_data):
    rows_out = []
    for scope, (hr_groups, _, rule_groups) in scope_data.items():
        for (category, rule), expert_groups in rule_groups.items():
            fractions = cell_fractions(hr_groups, expert_groups)
            s = summarize(fractions)
            rows_out.append({
                "scope": scope,
                "category": category,
                "rule": rule,
                "pooled_ratio": s["pooled_ratio"],
                "pooled_miss_rate": s["pooled_miss_rate"],
                "n_cells": s["n_cells"],
                "pooled_denom_ids": s["pooled_denom_ids"],
                "pooled_missed_ids": s["pooled_missed_ids"],
            })

    df = pd.DataFrame(rows_out)
    if not df.empty:
        # Ascending pooled ratio within each scope: the rules HR misses most
        # (lowest overlap / highest miss rate) sort to the top of each block.
        df.sort_values(["scope", "pooled_ratio"], inplace=True)
    return df


def main():
    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)

    scope_data = {scope: load_scope_data(gens) for scope, gens in SCOPE_GROUPS.items()}

    headline_df = build_headline_table(scope_data)
    headline_path = os.path.join(TABLE_OUTPUT_DIR, HEADLINE_CSV_NAME)
    headline_df.to_csv(headline_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {headline_path}")
    print(headline_df.to_string(index=False))

    rules_df = build_rule_table(scope_data)
    rules_path = os.path.join(TABLE_OUTPUT_DIR, RULES_CSV_NAME)
    rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {rules_path}")
    if not rules_df.empty:
        print(rules_df.to_string(index=False))
    else:
        print("[WARN] no rows produced -- no step2_expert_rule_detail_<GEN>.csv files found.")


if __name__ == "__main__":
    main()