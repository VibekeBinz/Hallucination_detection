"""
hallucination_pipeline/step3_llm_filter.py

STEP 3 — LLM Hallucination Detection

Reads Step 2's expert IDs and sends only the UNFLAGGED rows to Claude for
a coherence review that a fixed expert rule can't express -- e.g. a value
that's technically within range but clinically incongruous given the rest
of the row. Run step3b_merge_llm.py after this to merge the results into
the shared summary file. Runs only for the generators listed in
LLM_GENERATORS_TO_RUN (config); this makes real, billable API calls, so it
is never driven automatically -- run it by hand, one generator at a time,
with an eye on MAX_SPEND_USD.

Outputs:
  summary_files/hallucination_pipeline/ids/step3_llm_{GENERATOR}.csv
  summary_files/hallucination_pipeline/summary/step3_summary_{GENERATOR}.csv
  hallucination_pipeline/checkpoints/step3_checkpoint_{GENERATOR}.json
      (beside the code, not under summary_files/ -- this is resume/process
      state, not a metric, the same category as run_pipeline.py's logs/)

Payload
-------
All 23 variables go to Claude, including readmission (the prediction target).
The system prompts define units for the numeric variables and levels for the
categoricals — without those, values like readmission=True or albumin=Low
carry no meaning and the model cannot judge coherence.

Resuming / skipping already-done work
--------------------------------------
Before processing a (folder, generation) file, the script checks whether
results for that exact folder+generation pair already exist in
STEP3_SUMMARY. If so, it is skipped entirely (no API calls). This is keyed
on (folder, generation) — not generation alone — so if you ever run more
than one seed folder per generator, results for one folder never cause a
different folder's same-numbered generation to be skipped.

Both output files (STEP3_SUMMARY and ID_FILE3) are updated by merging new
rows into whatever already exists on disk, never by overwriting the file
wholesale. So asking for generations [0, 5, 10, 19] when 10 was already
run previously will: skip 10 (no cost), run 0/5/19, and the resulting CSVs
will contain 0/5/10/19 — the old gen-10 rows are preserved untouched.

Safety on interruption
----------------------
The checkpoint is written after every COMPLETE file, so a crash costs at most
the file in progress and a rerun resumes from where it stopped. Four guards
protect against the worse failure — recording an INCOMPLETE file as complete:

  1. call_claude raises at once on errors retrying cannot fix (credit
     exhaustion, bad key, 400s) instead of working through the backoff ladder
     on every remaining batch.
  2. A shared stop flag cancels the rest of a file's batches after the first
     failure. Without it, all ~49 batches are already queued and each burns
     MAX_RETRIES billable calls whose results are then thrown away.
  3. process_file reports how many batches failed. If any did, main() does NOT
     record the file and stops, so the partial result is never checkpointed
     and the rerun redoes that file cleanly.
  4. MAX_SPEND_USD turns "ran out of credits" into a controlled stop at a
     threshold you choose, with the checkpoint intact.

On a clean finish the checkpoint is renamed to *.done, so a later run with
different LLM_GENERATIONS does not silently skip every file.
"""

import os
import re
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
from scipy.stats import t
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import Anthropic

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.pipeline_config import (
    POPULATION_FILE, METADATA_FILE, GEN_DIR, GENERATOR,
    LLM_GENERATORS_TO_RUN, LLM_GENERATIONS, LLM_SEED_FOLDERS,
    BATCH_SIZE, MODEL, MAX_WORKERS, MAX_RETRIES, BASE_BACKOFF, INTER_CALL_DELAY,
    PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M,
    PRICE_CACHE_WRITE_PER_1M, PRICE_CACHE_READ_PER_1M,
    EXPORT_GENERATIONS,
    HALLUC_IDS_DIR, HALLUC_SUMMARY_DIR,
    is_core,
)
from config.clinical_ranges import NORMAL_RANGES

load_dotenv()
os.makedirs(HALLUC_IDS_DIR, exist_ok=True)
os.makedirs(HALLUC_SUMMARY_DIR, exist_ok=True)

# Checkpoint state lives beside the code, not under summary_files/ -- it's
# process/resume state (same category as run_pipeline.py's logs/), not a
# metric anyone reads or analyzes.
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

STEP3_SUMMARY   = os.path.join(HALLUC_SUMMARY_DIR, f"step3_summary_{GENERATOR}.csv")
ID_FILE2        = os.path.join(HALLUC_IDS_DIR, f"step2_expert_{GENERATOR}.csv")
ID_FILE3        = os.path.join(HALLUC_IDS_DIR, f"step3_llm_{GENERATOR}.csv")
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_DIR, f"step3_checkpoint_{GENERATOR}.json")

# ============================================================
# SPEND CAP
# ============================================================
MAX_SPEND_USD = 20.00

NON_RETRYABLE = (
    "credit balance", "quota", "billing", "insufficient",
    "authentication", "invalid x-api-key", "permission", "not_found_error",
    "invalid_request_error",
)

# ============================================================
# CLIENT
# ============================================================

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not found.")
client = Anthropic(api_key=api_key)

# ============================================================
# SYSTEM PROMPTS — full and CORE variants
# ============================================================

SYSTEM_PROMPT_FULL = """
You are a conservative clinical validation and hallucinations detection
system evaluating SYNTHETIC patient records from an adult ICU cohort
including critically ill and dying patients.

Numeric variables and units:
  age                 years
  heartrate           bpm
  systolic_bp         mmHg
  diastolic_bp        mmHg
  respiratory_rate    bpm
  spo2                %
  glucose             mg/dL
  potassium           mEq/L
  sodium              mEq/L
  hemoglobin          g/dL
  creatinine          mg/dL
  blood_urea_nitro    mg/dL

Categorical variables and levels:
  readmission         True / False — whether this admission was a readmission
  prior_icu           True / False — previous ICU stay
  gender              M / F
  bmi                 Missing / Underweight / Normal / High / Obese / Morbidly obese
  nt-probnp           Missing / Normal / Acute / Critical (cardiac strain marker)
  cholesterol         Missing / Lowrisk / Intermediate / Highrisk
  albumin             Missing / Low / Normal / High (nutritional status)
  icd9                ICD-9 diagnosis chapter
  first_careunit      CCU / CSRU / MICU / SICU / TSICU
  admission_type      ELECTIVE / EMERGENCY / URGENT
  ethnicity           as given

These rows have already passed an automated impossible-value filter.
Flag ONLY rows with combinations or multivariate patterns that are
physiologically impossible or clinically illogical in real human
physiology, even in an ICU setting.

CRITICAL OUTPUT RULES:
- Output ONLY the numeric ID values
- Return them as a single comma-separated string
- No explanations, no reasoning, no bullet points, no other text
- If no rows are flagged, output exactly: NONE

Example correct output:
42,187,3901
""".strip()

SYSTEM_PROMPT_CORE = """
You are a conservative clinical validation and hallucinations detection
system evaluating SYNTHETIC patient records from an adult ICU cohort
including critically ill and dying patients.

Numeric variables and units:
  age                 years
  heartrate           bpm
  systolic_bp         mmHg
  diastolic_bp        mmHg
  respiratory_rate    bpm
  potassium           mEq/L
  creatinine          mg/dL
  blood_urea_nitro    mg/dL

Categorical variables and levels:
  readmission         True / False — whether this admission was a readmission
  nt-probnp           Missing / Normal / Acute / Critical (cardiac strain marker)
  cholesterol         Missing / Lowrisk / Intermediate / Highrisk
  admission_type      ELECTIVE / EMERGENCY / URGENT
  ethnicity           as given

These rows have already passed an automated impossible-value filter.
Flag ONLY rows with combinations or multivariate patterns that are
physiologically impossible or clinically illogical in real human
physiology, even in an ICU setting.

CRITICAL OUTPUT RULES:
- Output ONLY the numeric ID values
- Return them as a single comma-separated string
- No explanations, no reasoning, no bullet points, no other text
- If no rows are flagged, output exactly: NONE

Example correct output:
42,187,3901
""".strip()

CACHED_SYSTEM_FULL = [
    {"type": "text", "text": SYSTEM_PROMPT_FULL, "cache_control": {"type": "ephemeral"}}
]
CACHED_SYSTEM_CORE = [
    {"type": "text", "text": SYSTEM_PROMPT_CORE, "cache_control": {"type": "ephemeral"}}
]

def get_cached_system(core: bool) -> list:
    return CACHED_SYSTEM_CORE if core else CACHED_SYSTEM_FULL

# ============================================================
# TOKEN TRACKER
# ============================================================

class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = self.output_tokens = 0
        self.cache_creation_tokens = self.cache_read_tokens = 0
        self.n_calls = 0

    def add(self, usage):
        with self._lock:
            self.input_tokens          += getattr(usage, "input_tokens",                0)
            self.output_tokens         += getattr(usage, "output_tokens",               0)
            self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
            self.cache_read_tokens     += getattr(usage, "cache_read_input_tokens",     0)
            self.n_calls               += 1

    def cost(self):
        with self._lock:
            return (self.input_tokens          / 1_000_000 * PRICE_INPUT_PER_1M
                  + self.output_tokens         / 1_000_000 * PRICE_OUTPUT_PER_1M
                  + self.cache_creation_tokens / 1_000_000 * PRICE_CACHE_WRITE_PER_1M
                  + self.cache_read_tokens     / 1_000_000 * PRICE_CACHE_READ_PER_1M)

    def report(self):
        print(f"\n[TOKENS] calls={self.n_calls}  in={self.input_tokens:,}  "
              f"out={self.output_tokens:,}  cache_write={self.cache_creation_tokens:,}  "
              f"cache_read={self.cache_read_tokens:,}  cost=${self.cost():.4f}")

tracker = TokenTracker()

# ============================================================
# COLUMN FILTER
# ============================================================

LLM_PAYLOAD_COLS = [
    "ID",
    "age", "gender", "ethnicity", "icd9",
    "heartrate", "systolic_bp", "diastolic_bp", "respiratory_rate", "spo2",
    "glucose", "potassium", "sodium", "hemoglobin", "creatinine",
    "blood_urea_nitro",
    "nt-probnp", "cholesterol", "albumin", "bmi",
    "first_careunit", "admission_type", "readmission", "prior_icu",
]

# ============================================================
# SUSPICIOUS ROW FILTER
# ============================================================

def is_worth_llm_review(row) -> bool:
    def safe_float(col):
        try:
            val = row.get(col)
            if val is None:
                return None
            f = float(val)
            return None if pd.isna(f) else f
        except (ValueError, TypeError):
            return None

    for col, bounds in NORMAL_RANGES.items():
        v = safe_float(col)
        if v is None:
            continue
        if bounds.get("low")  is not None and v < bounds["low"]:
            return True
        if bounds.get("high") is not None and v > bounds["high"]:
            return True
    return False

# ============================================================
# HELPERS
# ============================================================

def extract_gen(filename):
    m = re.search(r"gen_(\d+)", filename)
    return int(m.group(1)) if m else None

def load_column_meta():
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    columns_meta = metadata["columns"]
    num_cols = [c for c, m in columns_meta.items() if m["sdtype"] == "numerical"]
    cat_cols = [c for c, m in columns_meta.items() if m["sdtype"] in ["categorical", "boolean"]]
    all_cols = ["ID"] + [c for c in num_cols + cat_cols if c != "ID"]
    return num_cols, cat_cols, all_cols

def load_expert_ids_df() -> pd.DataFrame:
    if not os.path.exists(ID_FILE2):
        raise FileNotFoundError(
            f"Step 2 ID file not found: {ID_FILE2}\n"
            f"Please run step2_expert_rules.py first (with the ID export enabled)."
        )
    df = pd.read_csv(ID_FILE2)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    return df

def stats(arr):
    mean  = arr.mean()
    var   = arr.var(ddof=1) if len(arr) > 1 else 0
    sd    = arr.std(ddof=1) if len(arr) > 1 else 0
    n     = len(arr)
    tcrit = t.ppf(0.975, df=n - 1) if n > 1 else 0
    ci    = tcrit * sd / np.sqrt(n)
    return mean, var, mean - ci, mean + ci

# ---------------------------------------------------------
# LOAD EXISTING STEP 3 SUMMARY TO SKIP COMPLETED (folder, generation) PAIRS
# ---------------------------------------------------------
# Keyed on (folder, generation) -- the same key the checkpoint's done_files
# already uses -- so completing one folder's generation N never causes a
# different folder's generation N to be skipped too.
completed_keys = set()

if os.path.exists(STEP3_SUMMARY):
    try:
        s3 = pd.read_csv(STEP3_SUMMARY)
        if {"generation", "folder"}.issubset(s3.columns):
            completed_keys = set(
                zip(s3["folder"].astype(str), s3["generation"].astype(int))
            )
            print(f"[INFO] Already-completed (folder, generation) pairs "
                  f"(from existing summary): {sorted(completed_keys)}")
    except Exception:
        print("[WARN] Could not read existing step3 summary; assuming none completed.")

# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"id_rows": [], "summary_rows": [], "done_files": []}

def save_checkpoint(data: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def retire_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        done = CHECKPOINT_FILE + ".done"
        os.replace(CHECKPOINT_FILE, done)
        print(f"  Checkpoint retired: {os.path.basename(done)}")

# ============================================================
# LLM HELPERS
# ============================================================

class StopRun(Exception):
    """Raised when continuing would be pointless or costly."""

def call_claude(user_prompt: str, core: bool) -> str:
    if MAX_SPEND_USD is not None and tracker.cost() > MAX_SPEND_USD:
        raise StopRun(f"Spend cap ${MAX_SPEND_USD:.2f} reached "
                      f"(${tracker.cost():.2f}).")

    cached_system = get_cached_system(core)
    attempt = 0
    while True:
        try:
            time.sleep(INTER_CALL_DELAY)
            response = client.messages.create(
                model=MODEL, max_tokens=600, temperature=0,
                system=cached_system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            tracker.add(response.usage)
            return response.content[0].text.strip()
        except StopRun:
            raise
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in NON_RETRYABLE):
                raise StopRun(f"Not retryable: {e}") from e

            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(f"Max retries exceeded: {e}")
            sleep = BASE_BACKOFF * (2 ** (attempt - 1)) * (3 if "429" in str(e) else 1)
            print(f"[WARN] Retry {attempt}/{MAX_RETRIES} in {sleep:.1f}s... "
                  f"({type(e).__name__}: {str(e)[:160]})")
            time.sleep(sleep)

def parse_ids(text: str, valid_ids: set) -> set:
    if not text or re.search(r"\bNONE\b", text, re.IGNORECASE):
        return set()
    return {int(tok) for tok in re.findall(r"\b\d+\b", text) if int(tok) in valid_ids}

def build_user_prompt(batch_df: pd.DataFrame) -> str:
    cols = [c for c in LLM_PAYLOAD_COLS if c in batch_df.columns]
    return (
        "Return only the IDs of rows that are physiologically impossible "
        "or clinically illogical.\n\nCSV:\n"
        + batch_df[cols].round(1).to_csv(index=False)
    )

# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(csv_path, gen, folder, all_cols, expert_ids_df, core: bool):
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns:
        print(f"WARNING: No ID column in {csv_path}, skipping.")
        return [], len(df) if not df.empty else 0, 0, 0

    df = df[[c for c in all_cols if c in df.columns]]
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    total_rows = len(df)

    this_expert_ids = set(
        expert_ids_df[
            (expert_ids_df["metric"]     == "Expert_any") &
            (expert_ids_df["generation"] == gen) &
            (expert_ids_df["folder"]     == folder)
        ]["ID"].tolist()
    )
    unflagged_df = df[~df["ID"].isin(this_expert_ids)].reset_index(drop=True)
    n_expert     = len(this_expert_ids)

    borderline_mask = unflagged_df.apply(is_worth_llm_review, axis=1)
    borderline_df   = unflagged_df[borderline_mask].reset_index(drop=True)
    n_normal        = (~borderline_mask).sum()

    print(f"  Expert flagged:       {n_expert:,}")
    print(f"  All-normal (skipped): {n_normal:,}")
    print(f"  Borderline → Claude:  {len(borderline_df):,}  "
          f"({'CORE' if core else 'FULL'} prompt)")

    llm_ids = set()
    n_failed = 0
    if not borderline_df.empty:
        batches = [borderline_df.iloc[i:i + BATCH_SIZE]
                   for i in range(0, len(borderline_df), BATCH_SIZE)]

        stop_flag = threading.Event()

        def run_batch(batch_df):
            if stop_flag.is_set():
                raise StopRun("skipped — an earlier batch in this file failed")
            prompt    = build_user_prompt(batch_df)
            raw       = call_claude(prompt, core)
            valid_ids = set(batch_df["ID"].tolist())
            return parse_ids(raw, valid_ids)

        n_skipped = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(run_batch, b) for b in batches]
            for future in as_completed(futures):
                try:
                    llm_ids.update(future.result())
                except Exception as e:
                    if stop_flag.is_set():
                        n_skipped += 1
                    else:
                        stop_flag.set()
                        n_failed += 1
                        if isinstance(e, StopRun):
                            print(f"[STOP] {e}")
                        else:
                            print(f"[ERROR] Batch failed in {csv_path}: {e}")

        if n_skipped:
            print(f"  [SKIP] {n_skipped} remaining batch(es) not sent "
                  f"(file already aborted)")

    return sorted(llm_ids), total_rows, n_expert, n_failed

# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()
    print(f"\n=== STEP 3: LLM Hallucination Detection ({GENERATOR}) ===\n")

    if GENERATOR not in LLM_GENERATORS_TO_RUN:
        print(f"GENERATOR '{GENERATOR}' is not in LLM_GENERATORS_TO_RUN "
              f"({LLM_GENERATORS_TO_RUN}). Skipping LLM step.")
        return

    missing = sorted(set(LLM_GENERATIONS) - set(EXPORT_GENERATIONS))
    if missing:
        raise SystemExit(
            f"LLM_GENERATIONS {missing} are not in EXPORT_GENERATIONS, so "
            f"step 2 exported no expert IDs for them. Add them to "
            f"EXPORT_GENERATIONS and rerun step 2, or drop them here.")

    num_cols, cat_cols, all_cols = load_column_meta()

    print("\n--- DEBUG ---")
    print(f"GEN_DIR: {GEN_DIR}")
    print(f"LLM_GENERATIONS: {LLM_GENERATIONS}")
    print(f"LLM_SEED_FOLDERS: {LLM_SEED_FOLDERS}")
    all_folders = sorted(os.listdir(GEN_DIR))
    n_matched = 0
    print(f"\nAll folders in GEN_DIR ({len(all_folders)}):")
    for f in all_folders:
        sd = os.path.join(GEN_DIR, f, "SD")
        has_sd = os.path.isdir(sd)
        in_filter = LLM_SEED_FOLDERS is None or f in LLM_SEED_FOLDERS
        if has_sd and in_filter:
            n_matched += 1
        print(f"  {'✓' if (has_sd and in_filter) else '✗'} {f}  "
              f"(SD={has_sd}, in_filter={in_filter})")
        if has_sd and in_filter:
            gens_found = sorted(
                g for g in (extract_gen(x) for x in os.listdir(sd)
                            if x.endswith(".csv")) if g is not None)
            print(f"      Generations found: {gens_found}")
            print(f"      Overlap with LLM_GENERATIONS: "
                  f"{sorted(set(gens_found) & set(LLM_GENERATIONS))}")
    print("--- END DEBUG ---\n")

    if n_matched == 0:
        raise SystemExit(
            f"No folder in GEN_DIR matched LLM_SEED_FOLDERS.\n"
            f"CORE runs use the Step7pfp prefix and carry no _CORE in the "
            f"generator segment — e.g. Step7pfp_dseed1597_synthcity_arf_"
            f"mseed39088169 for ARF_CORE.")

    core = is_core(GENERATOR)
    payload = [c for c in LLM_PAYLOAD_COLS if c in all_cols]
    stripped = [c for c in all_cols if c not in LLM_PAYLOAD_COLS and c != "ID"]

    print(f"CORE mode: {core}")
    print(f"Model: {MODEL}   Spend cap: "
          f"{'none' if MAX_SPEND_USD is None else f'${MAX_SPEND_USD:.2f}'}")
    print(f"Columns stripped from Claude payload: {stripped or 'none'}")
    print(f"Columns sent to Claude ({len(payload)}): {payload}\n")

    print("Loading expert-flagged IDs from step 2...")
    expert_ids_df = load_expert_ids_df()
    expert_any_df = expert_ids_df[expert_ids_df["metric"] == "Expert_any"]
    print(f"  {len(expert_any_df):,} Expert_any rows in the ID file "
          f"({expert_ids_df['generation'].nunique()} generations x "
          f"{expert_ids_df['folder'].nunique()} folders)")

    in_scope = expert_any_df[
        expert_any_df["generation"].isin(LLM_GENERATIONS)
        & (expert_any_df["folder"].isin(LLM_SEED_FOLDERS)
           if LLM_SEED_FOLDERS else True)
    ]
    n_files = len(LLM_GENERATIONS) * n_matched
    print(f"  {len(in_scope):,} of those are in scope for this run "
          f"({n_files} files, ~{len(in_scope)/max(n_files,1):,.0f} per file)\n")

    checkpoint   = load_checkpoint()
    id_rows      = checkpoint["id_rows"]
    summary_rows = checkpoint["summary_rows"]
    done_files   = set(checkpoint["done_files"])
    if done_files:
        print(f"Resuming from checkpoint — {len(done_files)} files already done\n")

    aborted = False

    for folder in sorted(os.listdir(GEN_DIR)):
        if aborted:
            break

        sd_path = os.path.join(GEN_DIR, folder, "SD")
        if not os.path.isdir(sd_path):
            continue

        if LLM_SEED_FOLDERS is not None and folder not in LLM_SEED_FOLDERS:
            continue

        for fname in sorted(os.listdir(sd_path)):
            if not fname.endswith(".csv"):
                continue
            gen = extract_gen(fname)
            if gen is None or gen not in LLM_GENERATIONS:
                continue

            # Skip this exact (folder, generation) if it's already recorded
            # in the on-disk summary from a previous run.
            if (folder, gen) in completed_keys:
                print(f"[SKIP] {folder} | gen={gen} already completed "
                      f"(found in existing summary) — skipping, no API calls.")
                continue

            file_key = f"{folder}|{gen}"
            if file_key in done_files:
                print(f"[SKIP] Already done (checkpoint): {file_key}")
                continue

            csv_path = os.path.join(sd_path, fname)
            print(f"\n[FILE] gen={gen} | {folder} | {fname}")

            llm_ids, total_rows, n_expert, n_failed = process_file(
                csv_path, gen, folder, all_cols, expert_ids_df, core
            )

            if n_failed:
                print(f"  [ABORT] {n_failed} batch(es) failed — this file is "
                      f"NOT recorded. Fix the cause and rerun to resume "
                      f"from here.")
                aborted = True
                break

            print(f"  LLM flagged: {len(llm_ids):,}")

            folder_expert_any = set(
                expert_any_df[
                    (expert_any_df["generation"] == gen) &
                    (expert_any_df["folder"]     == folder)
                ]["ID"].tolist()
            )

            combined_ids  = folder_expert_any | set(llm_ids)
            expert_rate   = len(folder_expert_any) / total_rows if total_rows else 0
            llm_rate      = len(llm_ids)           / total_rows if total_rows else 0
            combined_rate = len(combined_ids)      / total_rows if total_rows else 0

            for id_val in llm_ids:
                id_rows.append({"generation": gen, "folder": folder,
                                "metric": "LLM", "ID": id_val})

            summary_rows.append({
                "generation":      gen,
                "folder":          folder,
                "total_rows":      total_rows,
                "expert_flagged":  n_expert,
                "llm_flagged":     len(llm_ids),
                "expert_any_rate": round(expert_rate   * 100, 4),
                "llm_rate":        round(llm_rate      * 100, 4),
                "combined_rate":   round(combined_rate * 100, 4),
            })

            done_files.add(file_key)
            save_checkpoint({
                "id_rows":      id_rows,
                "summary_rows": summary_rows,
                "done_files":   list(done_files),
            })
            print(f"  ✔ Checkpoint saved ({len(done_files)} files done, "
                  f"${tracker.cost():.2f} spent)")

    if not summary_rows:
        tracker.report()
        raise SystemExit(
            f"\nNo files processed for {GENERATOR}. Check LLM_SEED_FOLDERS "
            f"and LLM_GENERATIONS against the DEBUG listing above.")

    # --------------------------------------------------------------
    # Merge new LLM-flagged IDs into whatever already exists on disk rather
    # than overwriting ID_FILE3 with only this run's rows, so asking for a
    # new set of generations (e.g. 0,5,19 after 10 was already done) keeps
    # the already-saved gen-10 IDs intact.
    # --------------------------------------------------------------
    if id_rows:
        new_ids_df = pd.DataFrame(id_rows, columns=["generation", "folder", "metric", "ID"])
        new_ids_df["ID"] = pd.to_numeric(new_ids_df["ID"], errors="coerce").astype("Int64")

        if os.path.exists(ID_FILE3) and os.path.getsize(ID_FILE3) > 0:
            try:
                old_ids_df = pd.read_csv(ID_FILE3)
            except pd.errors.EmptyDataError:
                old_ids_df = pd.DataFrame(columns=["generation", "folder", "metric", "ID"])
            if not old_ids_df.empty:
                old_ids_df["ID"] = pd.to_numeric(old_ids_df["ID"], errors="coerce").astype("Int64")
            out = pd.concat([old_ids_df, new_ids_df], ignore_index=True)
        else:
            out = new_ids_df

        out = out.drop_duplicates(subset=["generation", "folder", "metric", "ID"])
        out.sort_values(["generation", "folder", "metric", "ID"], inplace=True)
        out.to_csv(ID_FILE3, index=False)
        size_mb = os.path.getsize(ID_FILE3) / 1_048_576
        print(f"\n✔ Step 3 IDs updated: {ID_FILE3}  "
              f"({len(new_ids_df):,} new rows, {len(out):,} total, {size_mb:.1f} MB)")
    else:
        print("\nNo new LLM IDs produced this run (nothing flagged, or nothing left to process).")

    new_summary_df = pd.DataFrame(summary_rows)

    if os.path.exists(STEP3_SUMMARY):
        try:
            old_summary = pd.read_csv(STEP3_SUMMARY)
        except pd.errors.EmptyDataError:
            old_summary = pd.DataFrame()
        combined_summary = pd.concat([old_summary, new_summary_df], ignore_index=True)
        combined_summary = combined_summary.drop_duplicates(
            subset=["generation", "folder"], keep="last"
        )
    else:
        combined_summary = new_summary_df

    combined_summary.to_csv(STEP3_SUMMARY, index=False)
    print(f"✔ Step 3 summary updated: {STEP3_SUMMARY}  ({len(new_summary_df)} new rows, "
          f"{len(combined_summary)} total)")

    tracker.report()

    if aborted:
        print(f"\n[INCOMPLETE] Run stopped early. {len(done_files)} file(s) "
              f"recorded; the checkpoint is kept so a rerun resumes.")
        print(f"Step 3 stopped — elapsed: {datetime.now() - t0}")
        sys.exit(1)

    retire_checkpoint()
    print(f"\nStep 3 done — elapsed: {datetime.now() - t0}")


if __name__ == "__main__":
    main()