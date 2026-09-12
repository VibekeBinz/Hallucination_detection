"""
hallucination_pipeline/step4_per_chain_summary.py

STEP 4 -- Per-chain packaging of Steps 1-3's results.

Read-only against every earlier step's output; writes three new files and
touches nothing else -- no LLM calls, no recomputation of anything Step 3
already paid for. Steps 1-3 already write pooled, per-generation summaries
(mean +/- CI across all chains); this step reshapes their existing
per-(folder, generation) detail into a per-chain, per-generation table, so
plotting and statistics can consume it directly with no further calculation.

Outputs (flat under the shared summary_files/per_chain/ folder --
config/pipeline_config.py's PER_CHAIN_DIR -- one row per chain per
generation, generator embedded in the filename. Shared with
extract_qa_metrics' own per-chain files, e.g. sd_quality_{GENERATOR}.csv
and tstr_trtr_{GENERATOR}.csv, so a cross-package comparison -- like
checking a hallucination-rate threshold against alpha/beta/authenticity --
reads every per-chain series from one folder):

  step4_hr_b5_per_chain_{GENERATOR}.csv
      HR_b5 rate = count(metric == "HR_b5") / N_SYNTHETIC_ROWS, from
      ids/step1_hr_{GENERATOR}.csv.

  step4_expert_any_per_chain_{GENERATOR}.csv
      Expert_any rate = count(metric == "Expert_any") / N_SYNTHETIC_ROWS,
      from ids/step2_expert_{GENERATOR}.csv. The id file only lists rows
      where a rule actually fired, so a (folder, generation) with zero
      firings across every Expert_* metric leaves no row there at all; the
      (folder, generation) universe reported here is taken from the HR id
      file instead (see load_full_universe()) so those generations still
      get an explicit 0.0 rather than being missing from the output.

  step4_expert_plus_llm_per_chain_{GENERATOR}.csv
      combined_rate straight from summary/step3_summary_{GENERATOR}.csv,
      rescaled from a percentage to a 0-1 fraction. Step 3 removes
      expert-flagged rows before the LLM ever sees them, so combined_rate is
      already the right union rate -- not recomputed here, only reshaped.
      Empty/absent for any chain or generation Step 3 hasn't been run on yet
      -- expected, given Step 3 is run by hand per generator, not this one.

N_SYNTHETIC_ROWS = 10,000 is a fixed constant for the whole study (every
synthetic export is a 10,000-row dataset -- see supplementary methods), not
re-derived from the raw decoded CSVs.

No "variant" column: full vs. CORE is already encoded in GENERATOR itself
(e.g. "ARF" vs "ARF_CORE"), same as everywhere else in this pipeline, so a
separate column would just duplicate that.

Self-contained by design: no join against class-collapse / minority-share
info -- that lives in extract_qa_metrics/'s output, a different package.
Whatever consumes these files (statistics, or the future visualization
package) can do that join itself.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import GENERATOR, HALLUC_IDS_DIR, HALLUC_SUMMARY_DIR, PER_CHAIN_DIR

os.makedirs(PER_CHAIN_DIR, exist_ok=True)

N_SYNTHETIC_ROWS = 10_000

HR_ID_FILE     = os.path.join(HALLUC_IDS_DIR, f"step1_hr_{GENERATOR}.csv")
EXPERT_ID_FILE = os.path.join(HALLUC_IDS_DIR, f"step2_expert_{GENERATOR}.csv")
STEP3_SUMMARY  = os.path.join(HALLUC_SUMMARY_DIR, f"step3_summary_{GENERATOR}.csv")

OUT_HR_B5      = os.path.join(PER_CHAIN_DIR, f"step4_hr_b5_per_chain_{GENERATOR}.csv")
OUT_EXPERT_ANY = os.path.join(PER_CHAIN_DIR, f"step4_expert_any_per_chain_{GENERATOR}.csv")
OUT_EXPERT_LLM = os.path.join(PER_CHAIN_DIR, f"step4_expert_plus_llm_per_chain_{GENERATOR}.csv")

# Same folder-naming shape used throughout this pipeline, e.g.
# "Step7pfa_dseed14930352_synthcity_arf_mseed24157817".
FOLDER_RE = re.compile(
    r"^(Step7pf[ap])_dseed(\d+)_synthcity_([A-Za-z0-9]+)_mseed(\d+)$", re.IGNORECASE
)


def split_folder(folder: str):
    """'Step7pfa_dseed14930352_synthcity_arf_mseed24157817' ->
    (dseed, mseed, chain_label), e.g. (14930352, 24157817, 'd14930352 / m24157817').
    (None, None, folder) if it doesn't match the expected shape -- reported,
    not silently dropped."""
    m = FOLDER_RE.match(str(folder))
    if not m:
        return None, None, str(folder)
    _, dseed, _, mseed = m.groups()
    return int(dseed), int(mseed), f"d{dseed} / m{mseed}"


def _attach_chain_cols(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["folder"].apply(split_folder)
    df = df.copy()
    df["dseed"] = [p[0] for p in parsed]
    df["mseed"] = [p[1] for p in parsed]
    df["chain"] = [p[2] for p in parsed]

    unparsed = df[df["dseed"].isna()]
    if not unparsed.empty:
        example = unparsed["folder"].iloc[0]
        print(f"  [WARN] {unparsed['folder'].nunique()} folder(s) didn't match "
              f"the expected naming pattern, e.g. {example!r}")
    return df


def load_full_universe():
    """Every (folder, generation) pair actually evaluated for this generator,
    taken from the HR id file rather than from whichever file
    rate_from_id_file() happens to be reading. Pilgram HR always assigns
    every record to some bin, so every (folder, generation) that was run
    leaves at least one row in step1_hr_{GENERATOR}.csv -- unlike the
    expert-rule id file, where a (folder, generation) with zero rule
    firings across all of Expert_any/Expert_impossible/Expert_suspicious
    leaves no row at all. Returns None if the HR id file itself is
    missing, in which case rate_from_id_file() falls back to inferring
    the universe from its own file.
    """
    if not os.path.exists(HR_ID_FILE):
        return None
    hr = pd.read_csv(HR_ID_FILE, usecols=["folder", "generation"])
    return hr.drop_duplicates()


def rate_from_id_file(id_file: str, metric: str, out_path: str, full_universe=None):
    """Per (folder, generation): count(metric == metric) / N_SYNTHETIC_ROWS.

    The universe of (folder, generation) pairs to report on comes from
    `full_universe` when given (see load_full_universe()) -- a
    (folder, generation) that was evaluated but had zero rows for every
    metric in id_file still gets an explicit 0.0 row rather than being
    silently absent. Without `full_universe`, the pairs are taken from
    whatever appears anywhere in id_file itself, which under-reports any
    (folder, generation) where nothing at all was flagged.
    """
    if not os.path.exists(id_file):
        print(f"  [SKIP] not found: {id_file}")
        return

    df = pd.read_csv(id_file)
    all_keys = full_universe if full_universe is not None else \
        df[["folder", "generation"]].drop_duplicates()

    sub = df[df["metric"] == metric]
    counts = sub.groupby(["folder", "generation"]).size().reset_index(name="n_flagged")

    counts = all_keys.merge(counts, on=["folder", "generation"], how="left")
    counts["n_flagged"] = counts["n_flagged"].fillna(0).astype(int)
    counts["value"] = counts["n_flagged"] / N_SYNTHETIC_ROWS

    counts = _attach_chain_cols(counts)
    out = counts[["generation", "folder", "dseed", "mseed", "chain", "value"]]
    out = out.sort_values(["dseed", "mseed", "generation"])
    out.insert(0, "generator", GENERATOR)

    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(out)} rows, {out['chain'].nunique()} chains)")


def combined_rate_from_step3(out_path: str):
    """Reshape step3_summary_{GENERATOR}.csv's own combined_rate column --
    no recomputation, Step 3 already paid for this."""
    if not os.path.exists(STEP3_SUMMARY):
        print(f"  [SKIP] no Step 3 summary yet (expected if the LLM step "
              f"hasn't been run for {GENERATOR}): {STEP3_SUMMARY}")
        return

    df = pd.read_csv(STEP3_SUMMARY)
    df = _attach_chain_cols(df)

    # combined_rate is stored as a percentage (5.5 == 5.5%) in step3_summary;
    # match the 0-1 fraction scale the other two per-chain files use.
    df["value"] = df["combined_rate"] / 100.0

    out = df[["generation", "folder", "dseed", "mseed", "chain", "value"]]
    out = out.sort_values(["dseed", "mseed", "generation"])
    out.insert(0, "generator", GENERATOR)

    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(out)} rows, {out['chain'].nunique()} chains)")


def main():
    print(f"=== Step 4: per-chain packaging for {GENERATOR} ===")

    full_universe = load_full_universe()
    if full_universe is None:
        print(f"  [WARN] HR id file not found ({HR_ID_FILE}); falling back to "
              f"each output's own file for its (folder, generation) universe, "
              f"which under-reports any generation where nothing was flagged.")

    print("\n1. HR_b5 per chain")
    rate_from_id_file(HR_ID_FILE, "HR_b5", OUT_HR_B5, full_universe)

    print("\n2. Expert_any per chain")
    rate_from_id_file(EXPERT_ID_FILE, "Expert_any", OUT_EXPERT_ANY, full_universe)

    print("\n3. Expert + LLM combined per chain")
    combined_rate_from_step3(OUT_EXPERT_LLM)

    print("\ndone")


if __name__ == "__main__":
    main()