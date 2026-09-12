import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.append(_p)

import pandas as pd

from config.pipeline_config import RAW_DATA_ROOT, ROOT

# ============================================================
# CONFIG
# ============================================================

GENERATORS = [
    "ARF", "ARF_CORE",
    "RTVAE", "RTVAE_CORE",
    "DDPM", "DDPM_CORE",
    "CTGAN", "CTGAN_CORE",
]

# Population dataset: single reference file, not per-run. POP_FILE is a
# sibling folder to the generator output folders (ROOT/ARF, ROOT/DDPM, ...);
# population.csv must be placed there manually since it has no per-run
# export to pull from automatically. Clinicified output goes to a separate
# POPULATION folder so the manually-provided raw file is never overwritten.
POP_FILE_DIR = os.path.join(ROOT, "POP_FILE")
POP_FILE_NAME = "population.csv"
POPULATION_OUT_DIR = os.path.join(ROOT, "POPULATION")
POPULATION_OUT_NAME = "population.csv"

# ============================================================
# FEATURE GROUPS
# ============================================================

BOOLEAN_COLS     = ["readmission", "prior_icu"]
DEMOGRAPHIC_COLS = ["age"]
CATEGORICAL_COLS = [
    "admission_type", "first_careunit",
    "ethnicity", "icd9", "gender", "insurance",
    "language", "marital_status", "religion"
]
ORDINAL_COLS     = ["bmi", "nt-probnp", "cholesterol", "albumin"]
MEASUREMENT_COLS = [
    "glucose", "sodium", "spo2", "respiratory_rate",
    "heartrate", "systolic_bp", "diastolic_bp",
    "blood_urea_nitro"
]
FLOAT64_COLS = ["creatinine", "potassium", "hemoglobin"]

# ============================================================
# ORDINAL MAPPINGS
# ============================================================

BMI_MAP     = {-1: "Missing", 0: "Underweight", 1: "Normal", 2: "High", 3: "Obese", 4: "Morbidly obese"}
ALBUMIN_MAP = {-1: "Missing", 0: "Low",         1: "Normal", 2: "High"}
CHOL_MAP    = {-1: "Missing", 0: "Lowrisk",     1: "Intermediate", 2: "Highrisk"}
NT_MAP      = {-1: "Missing", 0: "Normal",      1: "Acute",        2: "Critical"}

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

FINAL_ORDER = [
    "ID", "readmission", "prior_icu", "age", "gender", "icd9", "ethnicity",
    "admission_type", "first_careunit", "bmi", "nt-probnp", "cholesterol",
    "albumin", "blood_urea_nitro", "glucose", "sodium", "respiratory_rate",
    "heartrate", "systolic_bp", "diastolic_bp", "spo2", "creatinine",
    "potassium", "hemoglobin"
]

# ============================================================
# TRANSFORM
# ============================================================

def transform(df):
    df.insert(0, "ID", range(1, len(df) + 1))

    for col in BOOLEAN_COLS:
        if col in df:
            df[col] = df[col].astype(int).astype(bool)

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.upper().map(
            {"FALSE": "M", "TRUE": "F", "0": "M", "1": "F", "M": "M", "F": "F"}
        )

    if "bmi"         in df: df["bmi"]         = df["bmi"].map(BMI_MAP)
    if "albumin"     in df: df["albumin"]      = df["albumin"].map(ALBUMIN_MAP)
    if "cholesterol" in df: df["cholesterol"]  = df["cholesterol"].map(CHOL_MAP)
    if "nt-probnp"   in df: df["nt-probnp"]   = df["nt-probnp"].map(NT_MAP)

    for col in FLOAT64_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)

    for col in MEASUREMENT_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")

    for col in DEMOGRAPHIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")

    df = df[[c for c in FINAL_ORDER if c in df.columns]]
    return df

# ============================================================
# PER-GENERATOR PROCESSING
# ============================================================

def process_generator(generator):
    is_core  = generator.endswith("_CORE")
    base_gen = generator.replace("_CORE", "").lower()
    prefix   = "Step7pfp" if is_core else "Step7pfa"

    # Raw/output roots come from config.pipeline_config's RAW_DATA_ROOT and
    # ROOT constants, keeping every script's raw/output layout in one place.
    raw_root = os.path.join(RAW_DATA_ROOT, f"{prefix}_{base_gen}_export")
    out_root = os.path.join(ROOT, generator)

    print(f"\n=== BATCH CLINICIFICATION: {generator} ===")
    print(f"    RAW_ROOT : {raw_root}")
    print(f"    OUT_ROOT : {out_root}\n")

    if not os.path.isdir(raw_root):
        print(f"  [MISSING] RAW_ROOT not found: {raw_root}")
        print(f"  -> Add the raw export folder for '{generator}' manually at this path, then re-run.")
        return

    os.makedirs(out_root, exist_ok=True)

    for folder in os.listdir(raw_root):
        full_folder = os.path.join(raw_root, folder)
        if not os.path.isdir(full_folder):
            continue

        decoded_path = os.path.join(full_folder, "data", "decoded")
        if not os.path.isdir(decoded_path):
            print(f"  ⚠  No data/decoded/ in {folder}, skipping.")
            continue

        print(f"Processing: {folder}")

        out_folder = os.path.join(out_root, folder)
        rd_folder  = os.path.join(out_folder, "RD")
        sd_folder  = os.path.join(out_folder, "SD")
        os.makedirs(rd_folder, exist_ok=True)
        os.makedirs(sd_folder, exist_ok=True)

        csvs        = [f for f in os.listdir(decoded_path) if f.lower().endswith(".csv")]
        train_files = [f for f in csvs if f.startswith("reference_") and "gen_0" in f]
        syn_files   = [f for f in csvs if f.startswith("synthetic_")]

        # Training file → RD/
        if len(train_files) != 1:
            print(f"  ⚠  Expected 1 training file (reference_*gen_0*), found {len(train_files)}")
        else:
            src = os.path.join(decoded_path, train_files[0])
            dst = os.path.join(rd_folder, train_files[0].replace(".csv", "_clin.csv"))
            pd.read_csv(src).pipe(transform).to_csv(dst, index=False)
            print(f"    ✔ RD → {os.path.basename(dst)}")

        # Synthetic files → SD/
        for syn in syn_files:
            src = os.path.join(decoded_path, syn)
            dst = os.path.join(sd_folder, syn.replace(".csv", "_clin.csv"))
            pd.read_csv(src).pipe(transform).to_csv(dst, index=False)
            print(f"    ✔ SD → {os.path.basename(dst)}")

# ============================================================
# POPULATION PROCESSING
# ============================================================

def process_population():
    print(f"\n=== BATCH CLINICIFICATION: POPULATION ===\n")

    os.makedirs(POP_FILE_DIR, exist_ok=True)
    src = os.path.join(POP_FILE_DIR, POP_FILE_NAME)

    if not os.path.isfile(src):
        print("  [MISSING] Population data file population.csv must be manually "
              "added to folder ...Data\\SD_evaluationready\\POP_file.")
        return

    os.makedirs(POPULATION_OUT_DIR, exist_ok=True)
    dst = os.path.join(POPULATION_OUT_DIR, POPULATION_OUT_NAME)
    pd.read_csv(src).pipe(transform).to_csv(dst, index=False)
    print(f"    ✔ POPULATION → {os.path.basename(dst)}")

# ============================================================
# MAIN
# ============================================================

def main():
    failed = []

    for generator in GENERATORS:
        try:
            process_generator(generator)
        except Exception:
            import traceback
            print(f"  FAILED on '{generator}':")
            traceback.print_exc()
            failed.append(generator)

    try:
        process_population()
    except Exception:
        import traceback
        print("  FAILED on 'POPULATION':")
        traceback.print_exc()
        failed.append("POPULATION")

    print("\n=== DONE ===")
    if failed:
        print(f"[WARN] {len(failed)} item(s) failed: {failed}")


if __name__ == "__main__":
    main()