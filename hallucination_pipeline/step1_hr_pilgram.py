"""
hallucination_pipeline/step1_hr_pilgram.py

STEP 1 — Pilgram HR + HR_adjusted + Exact Copy Rate

Computes the Pilgram hallucination-rate (HR) statistic and the HR_adjusted
variant (incorporating the opportunity-space term and stabilizing constant
V) for each synthetic row, plus an exact-copy-rate check against the
reference population. Runs across the generations and bins configured in
EXPORT_GENERATIONS / BIN_LIST for whichever GENERATOR is set in
config.pipeline_config.

Reruns merge into the existing merged_hallucination_summary_{GENERATOR}.csv
rather than overwriting it: only this step's own columns
(STEP1_OWNED_PREFIXES) are replaced, so anything Step 2/3/3b already wrote
into that file survives a Step 1 rerun.

Outputs:
  summary_files/hallucination_pipeline/ids/step1_hr_{GENERATOR}.csv
  summary_files/hallucination_pipeline/ids/step1_copy_{GENERATOR}.csv
  summary_files/hallucination_pipeline/summary/step1_summary_{GENERATOR}.csv
  summary_files/hallucination_summaries/merged_hallucination_summary_{GENERATOR}.csv
"""

import math
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
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from config.pipeline_config import (
    POPULATION_FILE, METADATA_FILE, GEN_DIR,
    GENERATOR, BIN_LIST, HALLUCINATION_SUMMARIES, V, EXPORT_GENERATIONS,
    HALLUC_IDS_DIR, HALLUC_SUMMARY_DIR,
    is_core, CORE_COLS,
)

os.makedirs(HALLUCINATION_SUMMARIES, exist_ok=True)
os.makedirs(HALLUC_IDS_DIR, exist_ok=True)
os.makedirs(HALLUC_SUMMARY_DIR, exist_ok=True)

HR_ID_FILE   = os.path.join(HALLUC_IDS_DIR, f"step1_hr_{GENERATOR}.csv")
COPY_ID_FILE = os.path.join(HALLUC_IDS_DIR, f"step1_copy_{GENERATOR}.csv")
SUMMARY_FILE = os.path.join(HALLUC_SUMMARY_DIR, f"step1_summary_{GENERATOR}.csv")

# ============================================================
# HELPERS
# ============================================================

def extract_gen(filename):
    m = re.search(r"gen_(\d+)", filename)
    return int(m.group(1)) if m else None

def normalize_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Missing"
    s = str(v).strip().lower()
    if s in ["true", "1"]:  return "1"
    if s in ["false", "0"]: return "0"
    if s in ["nan", "null", "none", ""]: return "Missing"
    return str(v)

def bin_numeric_column(series, mn, mx, n_bins):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(["Missing"] * len(s), index=s.index)
    mask = s.notna()

    if mx == mn:
        out[mask] = "1"
        return out

    # Use floor() instead of astype(int)
    scaled = np.floor((s[mask] - mn) / (mx - mn) * n_bins).astype(int) + 1

    # Clamp to [1, n_bins]
    scaled = scaled.clip(1, n_bins)

    out[mask] = scaled.astype(str)
    return out

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

def stats(arr):
    mean  = arr.mean()
    var   = arr.var(ddof=1) if len(arr) > 1 else 0
    sd    = arr.std(ddof=1) if len(arr) > 1 else 0
    n     = len(arr)
    tcrit = t.ppf(0.975, df=n - 1) if n > 1 else 0
    ci    = tcrit * sd / np.sqrt(n)
    return mean, var, mean - ci, mean + ci

# ============================================================
# SHARED RESOURCES
# ============================================================

def load_shared_resources():
    pop_df = pd.read_csv(POPULATION_FILE)

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    columns_meta = metadata["columns"]
    all_num_cols = [c for c, m in columns_meta.items() if m["sdtype"] == "numerical"]
    all_cat_cols = [c for c, m in columns_meta.items() if m["sdtype"] in ["categorical", "boolean"]]

    # For CORE generators: restrict to columns that exist in CORE_COLS
    if is_core(GENERATOR):
        num_cols = [c for c in all_num_cols if c in CORE_COLS]
        cat_cols = [c for c in all_cat_cols if c in CORE_COLS]
    else:
        num_cols = all_num_cols
        cat_cols = all_cat_cols

    all_cols = ["ID"] + [c for c in num_cols + cat_cols if c != "ID"]

    for col in num_cols:
        if col in pop_df.columns:
            pop_df[col] = pd.to_numeric(pop_df[col], errors="coerce")

    # Only keep columns that actually exist in population
    all_cols = [c for c in all_cols if c in pop_df.columns]
    pop_df   = pop_df[all_cols]

    bin_stats = {}
    for col in num_cols:
        if col == "ID" or col not in pop_df.columns:
            continue
        s = pop_df[col].dropna()
        bin_stats[col] = (s.min(), s.max())

    merge_cols = [c for c in all_cols if c != "ID"]
    pop_sets = {}
    for b in BIN_LIST:
        df = pop_df.copy()
        for col in num_cols:
            if col == "ID" or col not in df.columns:
                continue
            mn, mx = bin_stats[col]
            df[col] = bin_numeric_column(df[col], mn, mx, b)
        for col in num_cols + cat_cols:
            if col == "ID" or col not in df.columns:
                continue
            df[col] = df[col].apply(normalize_value)
        pop_sets[b] = df_to_tuple_set(df, merge_cols)

    cat_card = {}
    for col in cat_cols:
        cat_card[col] = pop_df[col].nunique(dropna=True) if col in pop_df.columns else 0

    print(f"  CORE mode: {is_core(GENERATOR)}")
    print(f"  Numeric cols ({len(num_cols)}): {num_cols}")
    print(f"  Categorical cols ({len(cat_cols)}): {cat_cols}")

    return {
        "pop_df": pop_df, "pop_sets": pop_sets, "bin_stats": bin_stats,
        "merge_cols": merge_cols, "num_cols": num_cols, "cat_cols": cat_cols,
        "all_cols": all_cols, "cat_card": cat_card,
        "P": len(pop_df), "N": len(num_cols),
    }

# ============================================================
# PILGRAM + COPY PIPELINE
# ============================================================

def process_folder_pilgram(args):
    folder, gen_dir, shared = args
    pop_sets   = shared["pop_sets"]
    bin_stats  = shared["bin_stats"]
    merge_cols = shared["merge_cols"]
    num_cols   = shared["num_cols"]
    cat_cols   = shared["cat_cols"]
    all_cols   = shared["all_cols"]

    sd_path = os.path.join(gen_dir, folder, "SD")
    rd_path = os.path.join(gen_dir, folder, "RD")
    if not os.path.isdir(sd_path):
        return {}

    # --- Training hashes for copy detection ---
    hash_cols    = [c for c in all_cols if c != "ID"]
    train_hashes = set()
    candidates   = []
    if os.path.isdir(rd_path):
        candidates = [f for f in os.listdir(rd_path)
                      if f.startswith("reference_") and f.endswith("_clin.csv")]
    if candidates:
        gen0       = [f for f in candidates if "gen_0" in f]
        train_df   = pd.read_csv(os.path.join(rd_path, gen0[0] if gen0 else candidates[0]))
        train_exact  = normalize_for_exact(train_df, num_cols, cat_cols)
        train_hashes = df_to_tuple_set(train_exact, hash_cols)
    else:
        print(f"WARNING: No training file in {rd_path}, copy detection skipped.")

    results = {}

    for fname in os.listdir(sd_path):
        if not fname.endswith(".csv"): continue
        gen = extract_gen(fname)
        if gen is None: continue

        syn_df = pd.read_csv(os.path.join(sd_path, fname))
        for col in num_cols:
            if col in syn_df.columns:
                syn_df[col] = pd.to_numeric(syn_df[col], errors="coerce")
        # Only keep columns present in both the metadata and the file
        syn_df = syn_df[[c for c in all_cols if c in syn_df.columns]]

        # --- Copy rate ---
        syn_exact  = normalize_for_exact(syn_df, num_cols, cat_cols)
        syn_common = [c for c in hash_cols if c in syn_exact.columns]
        syn_tuples = list(map(tuple, syn_exact[syn_common].astype(str).values.tolist()))
        copy_rate  = sum(1 for r in syn_tuples if r in train_hashes) / len(syn_tuples) if syn_tuples else 0.0

        # --- Pilgram HR per bin ---
        for b in BIN_LIST:
            df = syn_df.copy()
            for col in num_cols:
                if col == "ID" or col not in df.columns: continue
                mn, mx = bin_stats[col]
                df[col] = bin_numeric_column(df[col], mn, mx, b)
            for col in num_cols + cat_cols:
                if col == "ID" or col not in df.columns: continue
                df[col] = df[col].apply(normalize_value)

            present        = [c for c in merge_cols if c in df.columns]
            rows_as_tuples = list(map(tuple, df[present].values.tolist()))
            not_in_pop     = sum(1 for row in rows_as_tuples if row not in pop_sets[b])
            pilgram_hr     = not_in_pop / len(rows_as_tuples) if rows_as_tuples else 0.0

            results.setdefault(gen, {}).setdefault(b, []).append({
                "pilgram":   pilgram_hr,
                "copy_rate": copy_rate,
            })

    return results


def run_pilgram_pipeline(shared):
    folders = [f for f in os.listdir(GEN_DIR) if os.path.isdir(os.path.join(GEN_DIR, f))]
    if not folders:
        raise RuntimeError(f"No folders found in {GEN_DIR}.")
    print(f"Pilgram pipeline — {len(folders)-1} folders ")

    # Build argument list
    args = [(f, GEN_DIR, shared) for f in folders]

    # Run sequentially (Windows‑safe)
    results = [process_folder_pilgram(a) for a in args]

    # Merge results
    store = {}
    for fr in results:
        for gen, bins in fr.items():
            for b, vals in bins.items():
                store.setdefault(gen, {}).setdefault(b, []).extend(vals)
    return store


# ============================================================
# HR_ADJUSTED
# ============================================================

def compute_hr_adjusted(pilgram_store, shared):
    P        = shared["P"]
    N        = shared["N"]
    cat_card = shared["cat_card"]

    O_by_bin = {}
    for b in BIN_LIST:
        cat_product = 1
        for c in cat_card.values():
            cat_product *= c
        O_by_bin[b] = (b ** N) * cat_product

    adjusted_store = {}
    for gen, bins in pilgram_store.items():
        for b, vals in bins.items():
            O        = O_by_bin[b]
            ln_ratio = math.log(P) / math.log(O) if O > 1 else 1.0
            v_eff    = 0.0 if ln_ratio > 1 else V
            factor   = (1 - v_eff) + (v_eff * ln_ratio)
            for val in vals:
                adjusted_store.setdefault(gen, {}).setdefault(b, []).append(
                    {"hr_adjusted": val["pilgram"] * factor}
                )
    return adjusted_store

# Column prefixes Step 1 owns in the merged file. Used both to build this
# step's contribution and, in main(), to know which columns are safe to
# drop-and-replace on a rerun without touching anything Step 2/3/3b wrote.
STEP1_OWNED_PREFIXES = ("Pilgram_HR_", "HR_adjusted_", "Copy_")


def build_merged_summary(pilgram_store, adjusted_store):
    """
    Builds this run's Step 1 (HR pilgram, HR adjusted, Copy) contribution to
    merged_hallucination_summary_{GENERATOR}.csv -- Step 1's columns only.
    No placeholders for Expert_*/LLM_* here: main() merges this into
    whatever already exists in the file rather than overwriting it, so any
    columns Step 2/3/3b already wrote are preserved as-is, not blanked out.
    """
    rows = []
    for gen in sorted(pilgram_store.keys()):
        for b in BIN_LIST:
            pil_vals = pilgram_store[gen].get(b, [])
            adj_vals = adjusted_store.get(gen, {}).get(b, [])
            if not pil_vals:
                continue

            pil_arr  = np.array([v["pilgram"]     for v in pil_vals])
            adj_arr  = np.array([v["hr_adjusted"] for v in adj_vals]) if adj_vals else np.zeros_like(pil_arr)
            copy_arr = np.array([v["copy_rate"]   for v in pil_vals])

            pil_m,  pil_v,  pil_l,  pil_h  = stats(pil_arr)
            adj_m,  adj_v,  adj_l,  adj_h  = stats(adj_arr)
            copy_m, copy_v, copy_l, copy_h = stats(copy_arr)

            rows.append({
                "generation": f"gen_{gen}",
                "bin": b,
                "Pilgram_HR_mean":    pil_m,  "Pilgram_HR_var":    pil_v,
                "Pilgram_HR_low":     pil_l,  "Pilgram_HR_high":   pil_h,
                "HR_adjusted_mean":   adj_m,  "HR_adjusted_var":   adj_v,
                "HR_adjusted_low":    adj_l,  "HR_adjusted_high":  adj_h,
                "Copy_mean":          copy_m, "Copy_var":          copy_v,
                "Copy_low":           copy_l, "Copy_high":         copy_h,
            })

    return pd.DataFrame(rows).sort_values(["generation", "bin"])

# ============================================================
# SUMMARY
# ============================================================

def build_summary(pilgram_store, adjusted_store):
    rows = []
    for gen in sorted(pilgram_store.keys()):
        for b in BIN_LIST:
            pil_vals = pilgram_store[gen].get(b, [])
            adj_vals = adjusted_store.get(gen, {}).get(b, [])
            if not pil_vals: continue

            pil_arr  = np.array([v["pilgram"]     for v in pil_vals])
            adj_arr  = np.array([v["hr_adjusted"] for v in adj_vals]) if adj_vals else np.zeros_like(pil_arr)
            copy_arr = np.array([v["copy_rate"]   for v in pil_vals])

            pil_m,  pil_v,  pil_l,  pil_h  = stats(pil_arr)
            adj_m,  adj_v,  adj_l,  adj_h  = stats(adj_arr)
            copy_m, copy_v, copy_l, copy_h = stats(copy_arr)

            rows.append({
                "generation": f"gen_{gen}", "bin": b,
                "Pilgram_HR_mean":   pil_m,  "Pilgram_HR_var":   pil_v,
                "Pilgram_HR_low":    pil_l,  "Pilgram_HR_high":  pil_h,
                "HR_adjusted_mean":  adj_m,  "HR_adjusted_var":  adj_v,
                "HR_adjusted_low":   adj_l,  "HR_adjusted_high": adj_h,
                "Copy_mean":         copy_m, "Copy_var":         copy_v,
                "Copy_low":          copy_l, "Copy_high":        copy_h,
            })

    return pd.DataFrame(rows).sort_values(["generation", "bin"])

# ============================================================
# ID EXPORT (long format)
# ============================================================

def export_ids(shared):
    num_cols   = shared["num_cols"]
    cat_cols   = shared["cat_cols"]
    all_cols   = shared["all_cols"]
    pop_sets   = shared["pop_sets"]
    bin_stats  = shared["bin_stats"]
    merge_cols = shared["merge_cols"]
    hash_cols  = [c for c in all_cols if c != "ID"]

    hr_rows   = []
    copy_rows = []
    print(f"Step 1 ID export — generations: {EXPORT_GENERATIONS}")

    for folder in os.listdir(GEN_DIR):
        folder_path = os.path.join(GEN_DIR, folder)
        sd_path     = os.path.join(folder_path, "SD")
        rd_path     = os.path.join(folder_path, "RD")
        if not os.path.isdir(sd_path): continue

        train_hashes = set()
        candidates = []
        if os.path.isdir(rd_path):
            candidates = [f for f in os.listdir(rd_path)
                          if f.startswith("reference_") and f.endswith("_clin.csv")]
        if candidates:
            gen0         = [f for f in candidates if "gen_0" in f]
            train_df     = pd.read_csv(os.path.join(rd_path, gen0[0] if gen0 else candidates[0]))
            train_exact  = normalize_for_exact(train_df, num_cols, cat_cols)
            train_hashes = df_to_tuple_set(train_exact, hash_cols)
        else:
            print(f"WARNING: No training file in {rd_path}, copy detection skipped.")

        for fname in os.listdir(sd_path):
            if not fname.endswith(".csv"): continue
            gen = extract_gen(fname)
            if gen is None or gen not in EXPORT_GENERATIONS: continue

            df = pd.read_csv(os.path.join(sd_path, fname))
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[[c for c in all_cols if c in df.columns]]
            if "ID" not in df.columns: continue
            df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

            # Copy IDs
            syn_exact  = normalize_for_exact(df, num_cols, cat_cols)
            syn_common = [c for c in hash_cols if c in syn_exact.columns]
            syn_tuples = list(map(tuple, syn_exact[syn_common].astype(str).values.tolist()))
            copy_mask  = [r in train_hashes for r in syn_tuples]
            for id_val in df.loc[copy_mask, "ID"].tolist():
                copy_rows.append({"generation": gen, "folder": folder,
                                   "metric": "Copy", "ID": id_val})

            # HR IDs per bin
            for b in BIN_LIST:
                df_b = df.copy()
                for col in num_cols:
                    if col == "ID" or col not in df_b.columns: continue
                    mn, mx = bin_stats[col]
                    df_b[col] = bin_numeric_column(df_b[col], mn, mx, b)
                for col in num_cols + cat_cols:
                    if col == "ID" or col not in df_b.columns: continue
                    df_b[col] = df_b[col].apply(normalize_value)

                present    = [c for c in merge_cols if c in df_b.columns]
                tuples     = list(map(tuple, df_b[present].values.tolist()))
                not_in_pop = [row not in pop_sets[b] for row in tuples]
                for id_val in df.loc[not_in_pop, "ID"].tolist():
                    hr_rows.append({"generation": gen, "folder": folder,
                                    "metric": f"HR_b{b}", "ID": id_val})

    def save(rows, path):
        if not rows:
            print(f"WARNING: No rows for {path}")
            return
        out = pd.DataFrame(rows, columns=["generation", "folder", "metric", "ID"])
        out["ID"] = pd.to_numeric(out["ID"], errors="coerce").astype("Int64")
        out.sort_values(["generation", "folder", "metric", "ID"], inplace=True)
        out.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"✔ Saved: {path}  ({len(out):,} rows, {size_mb:.1f} MB)")

    save(hr_rows,   HR_ID_FILE)
    save(copy_rows, COPY_ID_FILE)

# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()
    print(f"\n=== STEP 1: Pilgram HR + HR_adjusted + Copy Rate ({GENERATOR}) ===\n")

    shared         = load_shared_resources()
    print(f"Population: {shared['P']} rows, {shared['N']} numeric cols, "
          f"{len(shared['cat_cols'])} categorical cols\n")

    pilgram_store  = run_pilgram_pipeline(shared)
    adjusted_store = compute_hr_adjusted(pilgram_store, shared)

    summary = build_summary(pilgram_store, adjusted_store)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"✔ Step 1 summary saved: {SUMMARY_FILE}")

    # Merged summary — written to the consolidated summary directory
    # (HALLUCINATION_SUMMARIES), never inside the data tree.
    #
    # This MERGES rather than overwrites: a rerun of Step 1 must never wipe
    # out Expert_*/LLM_*/LLM_accumulated_* columns that Step 2/3/3b already
    # wrote. This run's fresh Step 1 columns are the row basis (that's the
    # (generation, bin) grid Step 1 itself defines); every other column
    # already in the file is carried over untouched. Only genuinely stale
    # Step1-owned columns (STEP1_OWNED_PREFIXES) are dropped before the
    # merge, so a rerun cleanly replaces its own numbers without producing
    # _x/_y suffix duplicates.
    MERGED_FILE = os.path.join(HALLUCINATION_SUMMARIES, f"merged_hallucination_summary_{GENERATOR}.csv")
    step1_cols = build_merged_summary(pilgram_store, adjusted_store)

    if os.path.exists(MERGED_FILE):
        existing = pd.read_csv(MERGED_FILE)
        existing["generation"] = existing["generation"].astype(str)
        stale = [c for c in existing.columns if c.startswith(STEP1_OWNED_PREFIXES)]
        other_cols = existing.drop(columns=stale)
        merged = step1_cols.merge(other_cols, on=["generation", "bin"], how="left")
    else:
        merged = step1_cols

    merged.sort_values(["generation", "bin"], inplace=True)
    merged.to_csv(MERGED_FILE, index=False)
    print(f"✔ Merged summary saved: {MERGED_FILE}")

    export_ids(shared)

    print(f"\nStep 1 done — elapsed: {datetime.now() - t0}")

if __name__ == "__main__":
    main()