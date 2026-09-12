import os

# ============================================================
# PATHS
# ============================================================
#
# Everything below is relative to REPO_ROOT (this file's own folder, one
# level up from config/) or to DATA_ROOT (the raw-data folder, a SIBLING of
# the repo -- .../DATA/, next to .../Hallucination_rate/, not inside it).
# DATA_ROOT is kept outside the repo deliberately: the raw MIMIC-III-derived
# data falls under a PhysioNet data use agreement and must never end up
# inside a git-tracked folder, even one that's .gitignored.
#
# Both are computed with os.path.join/os.path.dirname rather than hardcoded
# strings, so the pipeline runs unchanged regardless of which machine or
# drive the repo and the sibling DATA folder are checked out under -- the
# only requirement is that the two stay siblings of each other, same as
# today.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.normpath(os.path.join(REPO_ROOT, "..", "DATA"))

GENERATOR = "REAL"
ROOT            = os.path.join(REPO_ROOT, "Data", "SD_evaluationready")
POP_DIR         = os.path.join(ROOT, "POP_file")
POPULATION_FILE = os.path.join(POP_DIR, "population_clin.csv")
METADATA_FILE   = os.path.join(POP_DIR, "metadata_clinical.json")
GEN_DIR = os.path.join(ROOT, GENERATOR)
RESULTS_DIR = os.path.join(GEN_DIR, "results")
ID_DIR       = os.path.join(RESULTS_DIR, "IDs")

SUMMARIES = os.path.join(REPO_ROOT, "summary_files")
METRIC_SUMMARIES        = os.path.join (SUMMARIES, "metric_summaries")

# QA-metrics pooled (per-generation, across-chains) summaries -- current
# home of sd_quality_summary_<GENERATOR>.csv (one file per generator-variant
# token, e.g. sd_quality_summary_ARF.csv / sd_quality_summary_ARF_CORE.csv).
# METRIC_SUMMARIES above still exists for anything else that reads from it,
# but the pooled-trajectory figure scripts (visualization/pooled_trajectories/
# _shared.py) read from here now. Flattened out of the old qa_metrics/
# wrapper folder -- HALLUCINATION_SUMMARIES (hallucination_pipeline/
# merged_summary/) is a separate, unrelated pooled-summary output that
# never joins this one, so a shared "qa_metrics" parent for just this one
# kind of output no longer served a purpose.
QA_METRICS_POOLED_SUMMARY_DIR = os.path.join(SUMMARIES, "pooled_summaries")

# ------------------------------------------------------------
# PRINT_RESULTS_ROOT -- every table and figure meant for print, kept
# together under one folder so it's never mixed up with raw/working
# output (the summary folders above, PER_CHAIN_DIR, or STATISTICS_DIR
# further down). Distinct from RESULTS_ROOT (further down) -- RESULTS_ROOT
# is the whole summary_files/ tree every extractor writes into;
# PRINT_RESULTS_ROOT is specifically the "for the paper" subset of that.
#
# tables/ splits two ways by what kind of table it is, not by which script
# produced it:
#   results_tables/     descriptive summaries, no hypothesis tests --
#                        mean/CI per generation per metric (all_metrics_table.py,
#                        hallucination_table.py) and pooled miss-rate tables
#                        with a bootstrap CI (hr_overlap_headline.py,
#                        hr_overlap_summay.py, rule_level_hr_overlap.py).
#   statistics_tables/  the clean, paper-ready versions of the Claims 1-3
#                        hypothesis tests (manuscript_tables.py) -- reader-
#                        facing labels, one merged table per claim rather
#                        than one file per leave-one-out row. The full-
#                        disclosure raw version of the same tests
#                        (appendix_tables.py) is NOT here -- it's kept in
#                        STATISTICS_DIR/appendix_tables/ further down,
#                        alongside the working data it's a disclosure trail
#                        for, since it isn't meant to be pasted directly
#                        into the paper the way these two folders are.
# ------------------------------------------------------------

PRINT_RESULTS_ROOT         = os.path.join(SUMMARIES, "results")
TABLES_ROOT                = os.path.join(PRINT_RESULTS_ROOT, "tables")
RESULTS_TABLES_DIR         = os.path.join(TABLES_ROOT, "results_tables")
STATISTICS_TABLES_DIR      = os.path.join(TABLES_ROOT, "statistics_tables")
FIGURES_ROOT               = os.path.join(PRINT_RESULTS_ROOT, "figures")
FREQUENCYPLOTS_FIGURES     = os.path.join(FIGURES_ROOT, "frequency_pages")
POOLED_FIGURES             = os.path.join(FIGURES_ROOT, "pooled_trajectories")
CHAIN_TRAJECTORIES_FIGURES = os.path.join(FIGURES_ROOT, "chain_trajectories")
MISC_FIGURES               = os.path.join(FIGURES_ROOT, "misc")

# ============================================================
# STEP 1: HR / PILGRAM + COPY RATE
# ============================================================

BIN_LIST        = [5, 7, 10, 15, 20, 30]
PLOT_BIN        = 5
V               = 1       # stabilizing constant for HR_adjusted

# ============================================================
# STEP 2: EXPERT RULES
# ============================================================

# (no extra config needed — uses EXPORT_GENERATIONS and GENERATOR)

# ============================================================
# STEP 3: LLM
# — runs ONLY for generators listed here (subset of all generators)
# — uses EXPORT_GENERATIONS for which generations to process
# ============================================================

BATCH_SIZE       = 200
MODEL            = "claude-sonnet-4-6"
MAX_WORKERS      = 2
MAX_RETRIES      = 5
BASE_BACKOFF     = 2
INTER_CALL_DELAY = 1.5

PRICE_INPUT_PER_1M       =  3.00
PRICE_OUTPUT_PER_1M      = 15.00
PRICE_CACHE_WRITE_PER_1M =  3.75
PRICE_CACHE_READ_PER_1M  =  0.30

# Step 3 specific controls
LLM_GENERATORS_TO_RUN = {GENERATOR}
LLM_GENERATIONS        = [0,5,10,19]           # can differ from EXPORT_GENERATIONS

_IS_CORE     = GENERATOR.upper().endswith("_CORE")
_GEN_BASE    = GENERATOR.lower().replace("_core", "")   # arf_core -> arf
_ROOT_PREFIX = "Step7pfp" if _IS_CORE else "Step7pfa"

LLM_SEED_FOLDERS = [
    f"{_ROOT_PREFIX}_dseed1597_synthcity_{_GEN_BASE}_mseed39088169",
    f"{_ROOT_PREFIX}_dseed196418_synthcity_{_GEN_BASE}_mseed28657",
    f"{_ROOT_PREFIX}_dseed14930352_synthcity_{_GEN_BASE}_mseed24157817",
]

# ============================================================
# SHARED: which generations and folders to export IDs for
# ============================================================

EXPORT_GENERATIONS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]       # which generations to process across all steps
MAX_SEED_FOLDERS   = None      # None = all seed folders

PRETTY = {
    "training_time_cumulative": "Model Training Time (seconds, cumulative)",
    "new_row_synthesis": "New Row Synthesis",
    "alpha_delta_precision_OC": "α-Precision",
    "alpha_delta_coverage_OC": "β-Recall",
    "alpha_authenticity_OC": "Authenticity",
    "wasserstein_dist": "Wasserstein Distance",
    "mmd_corrected": "Maximum Mean Discrepancy",
    "jsd_syndat": "Jensen–Shannon Divergence (SynDat)",
    "tv_complement": "TV Complement",
    "ks_complement": "KS Complement",
    "sdmetrics_column_shapes": "SDMetrics Column Shapes",
    "sdmetrics_column_pair_trends": "SDMetrics Column Pair Trends",
    "k_ratio_synthetic_reference":
        "K-anonymity ratio synthetic/reference (>1 = synth more private)",
    "utility_gap": "Utility Gap (TRTR − TSTR AUROC)",
    "tstr_auroc": "TSTR AUROC", "trtr_auroc": "TRTR AUROC",
    "tstr_f1": "TSTR F1", "trtr_f1": "TRTR F1",
    "tstr_recall": "TSTR Recall", "trtr_recall": "TRTR Recall",
    "tstr_fpr": "TSTR False Positive Rate",
    "tstr_fnr": "TSTR False Negative Rate",
    "tstr_tnr": "TSTR True Negative Rate",
    "Pilgram_HR": "HR",
    "HR_adjusted": "HR*",
    "Expert_any": "Expert rules",
    "Expert_impossible": "Expert rules — Impossible",
    "Expert_suspicious": "Expert rules — Suspicious",
    "LLM": "LLM only",
    "LLM_accumulated": "Expert rules + LLM",
    "Copy": "Copied rows",
    "readmission_minority_share": "Readmission minority-class share",
    "detection_avg": "Average detection score",
    "detection_gmm": "GMM detector",
    "detection_xgb": "XGBoost detector",
    "detection_mlp": "MLP detector",
    "detection_linear": "Linear detector",
    "prdc_density": "PRDC Density",
    "prdc_coverage": "PRDC Coverage",
    "prdc_precision": "PRDC Precision",
    "prdc_recall": "PRDC Recall",
    "NovelFR": "Novel row fraction",
    "MemorizedFR": "Memorized row fraction",
    "TotalFR": "Total row fraction",
}

# ============================================================
# EXTRACT_QA_METRICS -- paths that package needs
# ============================================================

RAW_DATA_ROOT = os.path.join(DATA_ROOT, "SD_repository_raw")
CLINICIFIED_ROOT = ROOT  # already defined above: Data/SD_evaluationready

# extract_qa_metrics writes into the same consolidated summary_files root
# everything else uses (HALLUCINATION_SUMMARIES, METRIC_SUMMARIES, the
# chain_trajectories and survival subfolders) -- not a separate results/
# folder of its own. Its pooled, per-generation output lands at
# summary_files/qa_metrics/pooled_summary/; its per-chain output lands in
# the shared PER_CHAIN_DIR below, alongside hallucination_pipeline's.
RESULTS_ROOT = SUMMARIES

# ============================================================
# GENERATOR_VARIANTS -- single source of truth for every raw-side spelling
# of "which base generator, full or core" used anywhere in this pipeline.
# ============================================================

BASE_GENERATORS = ["ARF", "CTGAN", "DDPM", "RTVAE"]

# Columns kept for a CORE (13-variable) dataset. Full datasets keep
# everything the metadata file lists as numerical/categorical.
CORE_COLS = {
    "readmission", "age", "ethnicity", "admission_type", "nt-probnp",
    "cholesterol", "blood_urea_nitro", "respiratory_rate", "heartrate",
    "systolic_bp", "diastolic_bp", "creatinine", "potassium",
}

# Every raw-side token for "full" and "core". A new raw source with yet
# another spelling gets one line added here, not a new local mapping in
# whatever script reads it.
VARIANT_TOKEN_MAP = {
    "step7pfa": "full", "pfa": "full", "pf_all": "full",
    "step7pfp": "core", "pfp": "core", "pf_pilgram": "core",
}


def variant_from_token(token: str) -> str:
    """Any known raw-side spelling -> 'full' or 'core'."""
    key = str(token).strip().lower()
    if key not in VARIANT_TOKEN_MAP:
        raise KeyError(
            f"Unrecognised variant token: {token!r}. Add it to VARIANT_TOKEN_MAP "
            f"in config/pipeline_config.py rather than handling it locally."
        )
    return VARIANT_TOKEN_MAP[key]


def generator_value(base_generator: str, variant: str) -> str:
    """('ARF', 'core') -> 'ARF_CORE'; ('ARF', 'full') -> 'ARF'."""
    base = str(base_generator).upper()
    variant = variant.strip().lower()
    if variant not in ("full", "core"):
        raise ValueError(f"variant must be 'full' or 'core', got {variant!r}")
    return f"{base}_CORE" if variant == "core" else base


def _build_generator_variants():
    variants = {}
    for base in BASE_GENERATORS:
        variants[base] = {
            "base_generator": base,
            "base_token": base.lower(),
            "is_core": False,
            "raw_prefix": "Step7pfa",
            "clinicified_folder": base,
        }
        variants[f"{base}_CORE"] = {
            "base_generator": base,
            "base_token": base.lower(),
            "is_core": True,
            "raw_prefix": "Step7pfp",
            "clinicified_folder": f"{base}_CORE",
        }
    return variants


# Key = the GENERATOR value used everywhere in this pipeline (ARF,
# ARF_CORE, CTGAN, CTGAN_CORE, DDPM, DDPM_CORE, RTVAE, RTVAE_CORE).
GENERATOR_VARIANTS = _build_generator_variants()


def is_core(generator_name: str) -> bool:
    return GENERATOR_VARIANTS[generator_name.upper()]["is_core"]


def raw_prefix(generator_name: str) -> str:
    return GENERATOR_VARIANTS[generator_name.upper()]["raw_prefix"]


def base_token(generator_name: str) -> str:
    return GENERATOR_VARIANTS[generator_name.upper()]["base_token"]


# ============================================================
# PER_CHAIN_DIR -- shared per-chain trajectory output
# ============================================================
#
# One row per chain per generation, one file per metric. Owned by neither
# package: extract_qa_metrics writes its per-chain metric files here
# (sd_quality_*, tstr_trtr_*, minority_share.csv, and
# build_perchain_subsets.py's reshaped <metric>_per_chain.csv files), and
# hallucination_pipeline's Step 4 writes its three per-chain files here
# too, so anything doing cross-metric, cross-package comparison -- like
# calibrating a hallucination-spike threshold against alpha/beta/
# authenticity -- has one folder to read from.

PER_CHAIN_DIR = os.path.join(RESULTS_ROOT, "per_chain_summaries")

# ============================================================
# HALLUCINATION_PIPELINE -- output locations
# ============================================================
#
# Flat under RESULTS_ROOT/hallucination_pipeline/, one subfolder per kind
# of output, generator name embedded in every filename. Step 4's per-chain
# files are the one exception -- they're written to the shared
# PER_CHAIN_DIR above instead, alongside extract_qa_metrics' per-chain
# output.
#
# diagnostics/ holds only the QA-only nofire file from Step 2 -- not a
# metric, see that file's QA_ONLY_ filename prefix. Step 3's checkpoint
# JSON is process/resume state, not a metric, so it lives beside the code
# instead, in hallucination_pipeline/checkpoints/ (computed locally in
# step3_llm_filter.py, not defined here).
#
# merged_summary/ holds the cross-step, per-generator merged summaries
# (merged_hallucination_summary_<GENERATOR>.csv, written by
# step3b_merge_llmresults.py) -- named distinctly from summary/ below,
# which holds each STEP's own raw per-generator summary
# (step1_summary_<GENERATOR>.csv, step2_summary_<GENERATOR>.csv, ...)
# rather than the merged cross-step view. Both live under
# hallucination_pipeline/ since both are outputs of that pipeline; nothing
# else in the repo writes either one.

HALLUC_PIPELINE_ROOT     = os.path.join(RESULTS_ROOT, "hallucination_pipeline")
HALLUC_IDS_DIR           = os.path.join(HALLUC_PIPELINE_ROOT, "ids")
HALLUC_SUMMARY_DIR       = os.path.join(HALLUC_PIPELINE_ROOT, "summary")
HALLUC_DIAGNOSTICS_DIR   = os.path.join(HALLUC_PIPELINE_ROOT, "diagnostics")
HALLUCINATION_SUMMARIES  = os.path.join(HALLUC_PIPELINE_ROOT, "merged_summary")
STATISTICS_DIR = os.path.join(RESULTS_ROOT, "statistics_pipeline")
# appendix_tables.py's OUT_DIR is STATISTICS_DIR/appendix_tables -- the
# full-disclosure raw statistical-test tables, kept alongside the working
# claim/onset data rather than in TABLES_ROOT (see the PRINT_RESULTS_ROOT
# comment above for why).