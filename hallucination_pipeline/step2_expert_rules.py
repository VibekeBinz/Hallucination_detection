"""
hallucination_pipeline/step2_expert_rules.py

STEP 2 — Expert Rules

Applies a fixed set of clinical plausibility rules (config/expert_rules.py)
to each synthetic row and flags which ones fire: physiologically impossible
combinations (Expert_impossible), merely suspicious ones (Expert_suspicious),
and their union (Expert_any). Reads whatever variables the metadata file
lists for the current GENERATOR, core or full.

Outputs:
  summary_files/hallucination_pipeline/ids/step2_expert_{GENERATOR}.csv
  summary_files/hallucination_pipeline/ids/step2_expert_rule_detail_{GENERATOR}.csv
  summary_files/hallucination_pipeline/summary/step2_summary_{GENERATOR}.csv
  summary_files/hallucination_pipeline/summary/step2_rule_summary_{GENERATOR}.csv
  summary_files/hallucination_pipeline/diagnostics/QA_ONLY_step2_nofire_diagnostics_{GENERATOR}.csv  (only if any file fires nothing -- QA-only, not a metric)
  summary_files/hallucination_summaries/merged_hallucination_summary_{GENERATOR}.csv
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from datetime import datetime
from collections import Counter
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from config.pipeline_config import (
    POPULATION_FILE, METADATA_FILE, GEN_DIR,
    GENERATOR, HALLUCINATION_SUMMARIES, EXPORT_GENERATIONS,
    HALLUC_IDS_DIR, HALLUC_SUMMARY_DIR, HALLUC_DIAGNOSTICS_DIR,
)
from config.expert_rules import run as expert_run, build_population_bounds

os.makedirs(HALLUCINATION_SUMMARIES, exist_ok=True)
os.makedirs(HALLUC_IDS_DIR, exist_ok=True)
os.makedirs(HALLUC_SUMMARY_DIR, exist_ok=True)
os.makedirs(HALLUC_DIAGNOSTICS_DIR, exist_ok=True)


ID_FILE      = os.path.join(HALLUC_IDS_DIR, f"step2_expert_{GENERATOR}.csv")
# Per-ID detail on WHICH rule(s) fired, not just whether one did -- one row
# per (ID, rule) rather than the boolean-membership rows in ID_FILE above.
RULE_ID_FILE = os.path.join(HALLUC_IDS_DIR, f"step2_expert_rule_detail_{GENERATOR}.csv")
STEP2_SUMMARY = os.path.join(HALLUC_SUMMARY_DIR, f"step2_summary_{GENERATOR}.csv")
STEP1_SUMMARY = os.path.join(HALLUC_SUMMARY_DIR, f"step1_summary_{GENERATOR}.csv")
# QA_ONLY: not a metric, only written when a file fires zero expert rules --
# distinguishes real mode collapse from a silent data problem (see
# diagnose_nofire()). Safe to ignore unless you're chasing a suspiciously
# low/zero rate.
NOFIRE_FILE  = os.path.join(HALLUC_DIAGNOSTICS_DIR,
                            f"QA_ONLY_step2_nofire_diagnostics_{GENERATOR}.csv")

RULE_SUMMARY_COLS = ["generation", "rule", "mean", "low", "high", "sd", "n"]

# Columns the rules actually read. Used only for diagnostics.
RULE_INPUT_COLS = [
    "age", "glucose", "sodium", "spo2", "respiratory_rate", "heartrate",
    "systolic_bp", "diastolic_bp", "creatinine", "blood_urea_nitro",
    "potassium", "hemoglobin", "gender", "icd9", "admission_type",
    "first_careunit", "ethnicity", "bmi", "nt-probnp", "cholesterol",
    "albumin", "readmission", "prior_icu",
]

# ============================================================
# HELPERS
# ============================================================

def extract_gen(filename):
    m = re.search(r"gen_(\d+)", filename)
    return int(m.group(1)) if m else None

def stats(arr):
    mean  = arr.mean()
    var   = arr.var(ddof=1) if len(arr) > 1 else 0
    sd    = arr.std(ddof=1) if len(arr) > 1 else 0
    n     = len(arr)
    tcrit = t.ppf(0.975, df=n - 1) if n > 1 else 0
    ci    = tcrit * sd / np.sqrt(n)
    return mean, var, mean - ci, mean + ci

def load_column_meta():
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    columns_meta = metadata["columns"]
    num_cols = [c for c, m in columns_meta.items() if m["sdtype"] == "numerical"]
    cat_cols = [c for c, m in columns_meta.items() if m["sdtype"] in ["categorical", "boolean"]]
    all_cols = ["ID"] + [c for c in num_cols + cat_cols if c != "ID"]
    return num_cols, cat_cols, all_cols

def normalize_for_exact(df, num_cols, cat_cols):
    df2 = df.copy()
    for col in num_cols:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").round(0)
    for col in cat_cols:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.lower()
    return df2

def df_to_tuple_set(df, cols):
    present = [c for c in cols if c in df.columns]
    return set(map(tuple, df[present].values.tolist()))

# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnose_nofire(syn_df, folder, gen):
    """
    Called when a file produces zero rule firings. That is remarkable — other
    generators fire 18-27 distinct rules — and there are two very different
    causes that look identical downstream:

      GENUINE COLLAPSE: the generator emits one (or a few) repeated rows that
      happen to be clinically coherent. Zero firings is then a real result.
      Signature: unique_rows very low, missing_pct low.

      SILENT DATA PROBLEM: the values arrive as NaN or as non-numeric strings,
      so every _safe() guard in the rule module returns False and nothing can
      fire regardless of content. The rate is then meaningless, not zero.
      Signature: missing_pct high, or numeric columns with dtype object.

    Returns one row of evidence per file; written to NOFIRE_FILE.
    """
    present = [c for c in RULE_INPUT_COLS if c in syn_df.columns]
    missing_cols = [c for c in RULE_INPUT_COLS if c not in syn_df.columns]

    n_rows = len(syn_df)
    sub = syn_df[present] if present else syn_df

    # values the rules can actually use
    usable = 0
    total = 0
    nonnumeric = []
    for c in present:
        s = syn_df[c]
        total += n_rows
        if c in ("gender", "icd9", "admission_type", "first_careunit",
                 "ethnicity", "bmi", "nt-probnp", "cholesterol", "albumin",
                 "readmission", "prior_icu"):
            usable += s.notna().sum()
        else:
            num = pd.to_numeric(s, errors="coerce")
            usable += np.isfinite(num).sum()
            # Test coercion, not dtype: pandas 3 reports string columns as
            # "str" rather than "object", so a dtype check misses them.
            if int((num.isna() & s.notna()).sum()) > 0:
                nonnumeric.append(c)

    unique_rows = int(sub.drop_duplicates().shape[0]) if n_rows else 0
    dead = [c for c in present if syn_df[c].isna().all()]

    # Order matters. A single absent column blocks only the rules that need
    # it, so it must NOT outrank evidence about the columns that ARE present,
    # or a single missing categorical would mask the real cause for every
    # other rule.
    if not present:
        cause = "no rule input columns present"
    elif total and usable / total < 0.5:
        cause = "unusable values — rate is NOT a real zero"
    elif nonnumeric:
        cause = f"numeric stored as text ({len(nonnumeric)}) — rules cannot read it"
    elif dead:
        cause = f"all-NaN columns ({len(dead)}) — those rules cannot fire"
    elif n_rows and unique_rows <= max(5, 0.001 * n_rows):
        cause = "mode collapse — few unique rows"
    else:
        cause = ("values usable and varied — no rule triggered"
                 + (f" (note: {len(missing_cols)} input column(s) absent)"
                    if missing_cols else ""))

    return {
        "folder": folder,
        "generation": gen,
        "n_rows": n_rows,
        "unique_rows": unique_rows,
        "unique_row_pct": round(100 * unique_rows / n_rows, 3) if n_rows else np.nan,
        "usable_value_pct": round(100 * usable / total, 3) if total else np.nan,
        "all_nan_columns": ";".join(c for c in present
                                    if syn_df[c].isna().all()),
        "numeric_cols_stored_as_text": ";".join(nonnumeric),
        "missing_columns": ";".join(missing_cols),
        "likely_cause": cause,
    }

# ============================================================
# EXPERT PIPELINE
# ============================================================

def process_folder_expert(args):
    folder, gen_dir, num_cols, cat_cols, all_cols, num_bounds, cat_levels = args

    sd_path = os.path.join(gen_dir, folder, "SD")
    if not os.path.isdir(sd_path):
        return {}, [], [], []

    results = {}
    nofire = []
    id_frames = []
    rule_id_frames = []
    for fname in os.listdir(sd_path):
        if not fname.endswith(".csv"): continue
        gen = extract_gen(fname)
        if gen is None: continue

        syn_df = pd.read_csv(os.path.join(sd_path, fname))
        for col in num_cols:
            if col in syn_df.columns:
                syn_df[col] = pd.to_numeric(syn_df[col], errors="coerce")
        syn_df = syn_df[[c for c in all_cols if c in syn_df.columns]]

        expert_out = expert_run(syn_df, num_bounds, cat_levels)

        # Count total rule violations (all tags, not just rows)
        all_impossible = expert_out["H_impossible"].fillna("").str.split(";")
        all_suspicious = expert_out["H_suspicious"].fillna("").str.split(";")

        imp_flat = [tag.strip() for lst in all_impossible for tag in lst if tag.strip()]
        sus_flat = [tag.strip() for lst in all_suspicious for tag in lst if tag.strip()]

        rule_counts = Counter(imp_flat + sus_flat)
        # Per-row IDs, for step 3's exclusion set. Built as one small frame
        # per metric rather than a dict per row — that is what made the old
        # export slow, not the volume of data.
        if EXPORT_GENERATIONS is None or gen in EXPORT_GENERATIONS:
            if "ID" in expert_out.columns:
                imp_mask = expert_out["H_impossible"].fillna("").str.strip() != ""
                sus_mask = expert_out["H_suspicious"].fillna("").str.strip() != ""
                for metric, mask in (("Expert_impossible", imp_mask),
                                     ("Expert_suspicious", sus_mask),
                                     ("Expert_any",        imp_mask | sus_mask)):
                    if mask.any():
                        id_frames.append(pd.DataFrame({
                            "generation": gen,
                            "folder":     folder,
                            "metric":     metric,
                            "ID":         expert_out.loc[mask, "ID"].values,
                        }))

                # Per-ID rule detail: which exact rule(s) fired for each
                # flagged record, not just whether one did. H_impossible and
                # H_suspicious hold semicolon-joined rule tags per row --
                # explode them into one row per (ID, rule), so a record with
                # three rules firing produces three rows here.
                for category, tag_col, mask in (("impossible", "H_impossible", imp_mask),
                                                 ("suspicious", "H_suspicious", sus_mask)):
                    if mask.any():
                        sub = expert_out.loc[mask, ["ID", tag_col]].copy()
                        sub[tag_col] = sub[tag_col].fillna("").str.split(";")
                        sub = sub.explode(tag_col)
                        sub[tag_col] = sub[tag_col].str.strip()
                        sub = sub[sub[tag_col] != ""]
                        if not sub.empty:
                            rule_id_frames.append(pd.DataFrame({
                                "generation": gen,
                                "folder":     folder,
                                "category":   category,
                                "rule":       sub[tag_col].values,
                                "ID":         sub["ID"].values,
                            }))
        # A file where nothing fires is either a real finding (mode collapse)
        # or a silent data problem. Record the evidence either way rather than
        # letting the two look identical downstream.
        if not rule_counts:
            nofire.append(diagnose_nofire(syn_df, folder, gen))

        total_imp_violations = len(imp_flat)
        total_sus_violations = len(sus_flat)
        total_any_violations = total_imp_violations + total_sus_violations

        imp_rate   = (expert_out["H_impossible"].str.strip() != "").mean()
        sus_rate   = (expert_out["H_suspicious"].str.strip() != "").mean()
        any_rate   = ((expert_out["H_impossible"].str.strip() != "") |
                      (expert_out["H_suspicious"].str.strip() != "")).mean()

        results.setdefault(gen, []).append({
            "imp": imp_rate,
            "sus": sus_rate,
            "any": any_rate,
            "imp_violations": total_imp_violations,
            "sus_violations": total_sus_violations,
            "any_violations": total_any_violations,
            "rule_counts": rule_counts,
        })

    return results, nofire, id_frames, rule_id_frames


def run_expert_pipeline(num_cols, cat_cols, all_cols, num_bounds, cat_levels):
    folders = [f for f in os.listdir(GEN_DIR)
               if os.path.isdir(os.path.join(GEN_DIR, f))
               and os.path.isdir(os.path.join(GEN_DIR, f, "SD"))]
    if not folders:
        raise RuntimeError(f"No folders found in {GEN_DIR}.")

    print(f"Expert pipeline — {len(folders)} folders")

    args = [(f, GEN_DIR, num_cols, cat_cols, all_cols, num_bounds, cat_levels) for f in folders]
    with Pool(min(cpu_count(), len(folders))) as pool:
        results = pool.map(process_folder_expert, args)

    store, nofire, id_frames, rule_id_frames = {}, [], [], []
    for fr, nf, idf, ridf in results:
        for gen, vals in fr.items():
            store.setdefault(gen, []).extend(vals)
        nofire.extend(nf)
        id_frames.extend(idf)
        rule_id_frames.extend(ridf)

    if id_frames:
        ids = pd.concat(id_frames, ignore_index=True)
        ids["ID"] = pd.to_numeric(ids["ID"], errors="coerce").astype("Int64")
        ids.sort_values(["generation", "folder", "metric", "ID"], inplace=True)
        ids.to_csv(ID_FILE, index=False)
        n_any = int((ids["metric"] == "Expert_any").sum())
        print(f"✔ Expert IDs saved: {ID_FILE}  "
              f"({len(ids):,} rows, {n_any:,} Expert_any)")
    else:
        print("[WARN] no expert IDs produced")

    if rule_id_frames:
        rule_ids = pd.concat(rule_id_frames, ignore_index=True)
        rule_ids["ID"] = pd.to_numeric(rule_ids["ID"], errors="coerce").astype("Int64")
        rule_ids.sort_values(["generation", "folder", "category", "rule", "ID"], inplace=True)
        rule_ids.to_csv(RULE_ID_FILE, index=False)
        print(f"✔ Expert rule-level IDs saved: {RULE_ID_FILE}  "
              f"({len(rule_ids):,} rows)")
    else:
        print("[WARN] no expert rule-level IDs produced")

    if nofire:
        nf_df = (pd.DataFrame(nofire)
                 .sort_values(["generation", "folder"])
                 .reset_index(drop=True))
        nf_df.to_csv(NOFIRE_FILE, index=False)
        n_files = sum(len([x for x in os.listdir(os.path.join(GEN_DIR, f, "SD"))
                           if x.endswith(".csv")])
                      for f in os.listdir(GEN_DIR)
                      if os.path.isdir(os.path.join(GEN_DIR, f, "SD")))
        print(f"\n[WARN] {len(nf_df)} of {n_files} file(s) produced ZERO rule "
              f"firings. Diagnostics: {NOFIRE_FILE}")
        print(nf_df["likely_cause"].value_counts().to_string())

        # Which seeds are silent, and from which generation. If most folders
        # fire nothing while a few fire a lot, the mean rate is driven by a
        # minority of runs and its CI will be wide or cross zero.
        by_folder = nf_df.groupby("folder")["generation"].agg(["count", "min"])
        by_folder.columns = ["silent_files", "first_silent_gen"]
        print("\nsilent files per seed folder:")
        print(by_folder.sort_values("silent_files", ascending=False).to_string())
        print()

    return store

# ============================================================
# SUMMARY
# ============================================================

def build_summary(expert_store):
    rows = []
    for gen in sorted(expert_store.keys()):
        vals = expert_store[gen]
        if not vals: continue
        imp_m, imp_v, imp_l, imp_h = stats(np.array([v["imp"] for v in vals]))
        sus_m, sus_v, sus_l, sus_h = stats(np.array([v["sus"] for v in vals]))
        any_m, any_v, any_l, any_h = stats(np.array([v["any"] for v in vals]))
        imp_vio = np.array([v["imp_violations"] for v in vals])
        sus_vio = np.array([v["sus_violations"] for v in vals])
        any_vio = np.array([v["any_violations"] for v in vals])
        rows.append({
            "generation": f"gen_{gen}",
            "Expert_impossible_mean": imp_m, "Expert_impossible_var": imp_v,
            "Expert_impossible_low":  imp_l, "Expert_impossible_high": imp_h,
            "Expert_suspicious_mean": sus_m, "Expert_suspicious_var": sus_v,
            "Expert_suspicious_low":  sus_l, "Expert_suspicious_high": sus_h,
            "Expert_any_mean": any_m, "Expert_any_var": any_v,
            "Expert_any_low":  any_l, "Expert_any_high": any_h,
            "Expert_imp_violations_total": imp_vio.sum(),
            "Expert_sus_violations_total": sus_vio.sum(),
            "Expert_any_violations_total": any_vio.sum(),
        })
    return pd.DataFrame(rows).sort_values("generation")

def build_rule_summary(expert_store):
    rows = []
    for gen, vals in sorted(expert_store.items()):
        all_counts = [v["rule_counts"] for v in vals]
        if not all_counts:
            continue

        # union of all rule names; empty Counters contribute nothing
        all_rules = set().union(*[set(c.keys()) for c in all_counts])
        if not all_rules:
            print(f"[WARN] gen_{gen}: no rules fired in any of "
                  f"{len(all_counts)} run(s)")
            continue

        for rule in sorted(all_rules):
            arr = np.array([c.get(rule, 0) for c in all_counts], dtype=float)
            mean = arr.mean()
            sd = arr.std(ddof=1) if len(arr) > 1 else 0
            n = len(arr)
            tcrit = t.ppf(0.975, df=n-1) if n > 1 else 0
            ci = tcrit * sd / np.sqrt(n)

            rows.append({
                "generation": gen,
                "rule": rule,
                "mean": mean,
                "low": mean - ci,
                "high": mean + ci,
                "sd": sd,
                "n": n,
            })

    # An empty frame has no columns, so sort_values would raise KeyError.
    # Write the header anyway: downstream scripts then read an empty file
    # instead of failing on a missing one.
    if not rows:
        print(f"[WARN] no rules fired anywhere for {GENERATOR} — "
              f"writing an empty rule summary")
        df = pd.DataFrame(columns=RULE_SUMMARY_COLS)
    else:
        df = pd.DataFrame(rows).sort_values(["rule", "generation"])

    out_file = os.path.join(HALLUC_SUMMARY_DIR, f"step2_rule_summary_{GENERATOR}.csv")
    df.to_csv(out_file, index=False)
    print(f"✔ Rule summary saved: {out_file}  ({len(df)} rows)")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()
    print(f"\n=== STEP 2: Expert Rules ({GENERATOR}) ===\n")

    num_cols, cat_cols, all_cols = load_column_meta()
    num_bounds, cat_levels       = build_population_bounds(POPULATION_FILE)

    expert_store = run_expert_pipeline(num_cols, cat_cols, all_cols, num_bounds, cat_levels)

    summary = build_summary(expert_store)
    summary.to_csv(STEP2_SUMMARY, index=False)
    print(f"✔ Step 2 summary saved: {STEP2_SUMMARY}")

    print(f"\nStep 2 done — elapsed: {datetime.now() - t0}")

    MERGED_FILE = os.path.join(HALLUCINATION_SUMMARIES, f"merged_hallucination_summary_{GENERATOR}.csv")

    # Build expert stats per generation (compute stats once per metric)
    exp_rows = []
    for gen, vals in sorted(expert_store.items()):
        row = {"generation": f"gen_{gen}"}
        for key, prefix in [("imp", "Expert_impossible"),
                            ("sus", "Expert_suspicious"),
                            ("any", "Expert_any")]:
            m, v, l, h = stats(np.array([x[key] for x in vals]))
            row[f"{prefix}_mean"] = m
            row[f"{prefix}_var"]  = v
            row[f"{prefix}_low"]  = l
            row[f"{prefix}_high"] = h
        exp_rows.append(row)
    exp_df = pd.DataFrame(exp_rows)

    if os.path.exists(MERGED_FILE):
        merged = pd.read_csv(MERGED_FILE)
    elif os.path.exists(STEP1_SUMMARY):
        print(f"Merged summary not found — creating it from {STEP1_SUMMARY}")
        merged = pd.read_csv(STEP1_SUMMARY)
    else:
        raise FileNotFoundError(
            f"Neither {MERGED_FILE} nor {STEP1_SUMMARY} exists — "
            f"run Step 1 for {GENERATOR} first.")

    expert_cols = [c for c in merged.columns if c.startswith("Expert_")]
    merged = merged.drop(columns=expert_cols)
    merged = merged.merge(exp_df, on="generation", how="left")
    merged.sort_values(["generation", "bin"], inplace=True)
    merged.to_csv(MERGED_FILE, index=False)
    print(f"✔ Merged summary saved: {MERGED_FILE}")

    build_rule_summary(expert_store)


if __name__ == "__main__":
    main()