"""
Expert-rule hallucination detection — reusable module version.

This module exposes:
    - check_row(row)
    - run(df, num_bounds=None, cat_levels=None)
    - rule_summary(df)
    - build_population_bounds(reference_csv)
    - check_out_of_population(row, num_bounds, cat_levels)

All file I/O, CLI, printing, and path logic has been removed.
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)



import numpy as np
import pandas as pd
import tqdm
tqdm.tqdm = lambda *args, **kwargs: iter([])


# =============================================================================
# IMPORT CLINICAL RANGES (single source of truth)
# =============================================================================

from config.clinical_ranges import (
    REALITY_LIMITS, NORMAL_RANGES, EXTREME_THRESHOLDS, FATAL_THRESHOLDS,
    CATEGORICAL, RULE_PARAMS, PROTEIN_WASTING_CHAPTERS, ELEVATED_BMI,
)

P = RULE_PARAMS  # shorthand

# =============================================================================
# POPULATION-RANGE CHECK
# =============================================================================

OOR_NUMERIC_COLS = [
    "age", "glucose", "sodium", "spo2", "respiratory_rate", "heartrate",
    "systolic_bp", "diastolic_bp", "creatinine", "blood_urea_nitro",
    "potassium", "hemoglobin",
]

OOR_CATEGORICAL_COLS = [
    "gender", "icd9", "admission_type", "first_careunit", "ethnicity",
    "bmi", "nt-probnp", "cholesterol", "albumin",
]

def _safe(v) -> bool:
    """Return True if v is numeric and finite."""
    try:
        return v is not None and pd.notna(v) and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False

def build_population_bounds(reference_csv: str):
    """Learn numeric min/max and observed categorical levels from the reference file."""
    ref = pd.read_csv(reference_csv)

    num_bounds = {}
    for col in OOR_NUMERIC_COLS:
        if col in ref.columns:
            s = pd.to_numeric(ref[col], errors="coerce").dropna()
            if len(s):
                num_bounds[col] = (float(s.min()), float(s.max()))

    cat_levels = {}
    for col in OOR_CATEGORICAL_COLS:
        if col in ref.columns:
            cat_levels[col] = set(ref[col].dropna().astype(str).unique())

    return num_bounds, cat_levels

def check_out_of_population(row, num_bounds, cat_levels) -> str:
    """Return '; '-joined field names whose value is outside the population range."""
    offenders = []

    for col, (lo, hi) in num_bounds.items():
        v = row.get(col)
        if _safe(v):
            x = float(v)
            if x < lo or x > hi:
                offenders.append(col)

    for col, levels in cat_levels.items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v) not in levels:
            offenders.append(col)

    return "; ".join(offenders)

# =============================================================================
# RULE DESCRIPTIONS
# =============================================================================

RULE_DESCRIPTIONS = {
    # --- impossible ---
    "reality_limit":      "value outside absolute measurable physiological range",
    "bad_category":       "categorical value not in allowed set",
    "dbp_ge_sbp":         "diastolic BP >= systolic BP",
    "pp_zero":            "pulse pressure <=2 with non-extreme vitals",
    "buncr_ratio_impossible": "BUN/Cr ratio < 1.0 (physically impossible)",
    "rr_spo2_impossible": "RR <=3 with SpO2 >=98 (agonal RR + normal sat)",
    "fatal_constellation_impossible": ">=3 fatal-range values with all primary vitals pristine",
    "preg_male":          "Pregnancy diagnosis with gender=M",
    "preg_age":           "Pregnancy diagnosis outside reproductive age range",
    "nut_impossible":     "Underweight + albumin High + Hb>=17 (metabolic contradiction)",

    # --- suspicious ---
    "pp_low":             "pulse pressure <5 with fully normal vitals",
    "pp_wide":            "wide pulse pressure (>150) in younger patient, SBP not extreme",
    "pp_htn_narrow":      "extreme hypertension (SBP>=200) with near-zero pulse pressure",
    "buncr_ratio_high":   "BUN/Cr >100 with high BUN and no anemia",
    "low_cr":             "creatinine implausibly low for adult",
    "comp_anemia":        "severe anemia with no HR or RR compensation despite hypotension",
    "comp_shock":         "severe shock (SBP<50) with no HR or RR compensation",
    "comp_hyperk":        "severe hyperkalemia with normal HR and normal kidneys",
    "comp_hypona":        "severe hyponatremia with shock and abnormal HR",
    "pair_cardiogenic":   "extreme bradycardia + extreme hypotension with preserved respiration",
    "pair_htn_brady":     "extreme hypertension + extreme bradycardia",
    "pair_tachy_htn":     "extreme tachycardia + extreme hypertension",
    "fatal_constellation_suspicious": ">=3 fatal-range values (with disturbed vitals)",
    "extreme_count":      ">=5 features crossing extreme thresholds",
    "elderly_f_highhb":   "elderly female with extreme-high hemoglobin",
    "pregnancy_elective": "Pregnancy diagnosis with elective admission",
    "unknown_extremelabs":"UNKNOWN diagnosis with extreme labs",
    "circ_nomarkers":     "cardiac diagnosis with no cardiac markers and normal vitals",
    "preg_hb":            "Pregnancy with Hb>15",
    "nut_alb_normal":     "albumin Normal despite Underweight + severe anemia",
    "nut_chol_maln":      "Highrisk cholesterol with severe malnutrition pattern",
    "nut_obese_hypo":     "Obese + hypotension + low albumin + normal BNP",
    "alb_obese_anemia":   "low albumin + elevated BMI + severe anemia",
    "alb_high_underwt":   "albumin High with Underweight",
    "alb_high_wasting":   "albumin High with protein-wasting diagnosis",
    "severe_hypok":       "severe hypokalemia (K<=2.0) with normal HR",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _is_extreme(col, v, direction) -> bool:
    if not _safe(v):
        return False
    t = EXTREME_THRESHOLDS.get(col, {})
    if direction == "high":
        return float(v) >= t.get("high", float("inf"))
    return float(v) <= t.get("low", float("-inf"))

def _is_fatal(col, v, direction) -> bool:
    if not _safe(v):
        return False
    t = FATAL_THRESHOLDS.get(col, {})
    if direction == "high":
        return float(v) >= t.get("high", float("inf"))
    return float(v) <= t.get("low", float("-inf"))

def _not_extreme(col, v) -> bool:
    if not _safe(v):
        return False
    t = EXTREME_THRESHOLDS.get(col, {})
    x = float(v)
    return not (x >= t.get("high", float("inf")) or x <= t.get("low", float("-inf")))

def _is_normal(col, v) -> bool:
    if not _safe(v):
        return False
    t = NORMAL_RANGES.get(col, {})
    if not t:
        return False
    return t.get("low", float("-inf")) <= float(v) <= t.get("high", float("inf"))

def _count_fatal_all(row):
    vitals = {"heartrate", "systolic_bp", "respiratory_rate", "spo2"}
    n_total = n_vitals = 0
    for col, dirs in FATAL_THRESHOLDS.items():
        v = row.get(col)
        if not _safe(v):
            continue
        x = float(v)
        crossed = ((dirs.get("low") is not None and x <= dirs["low"]) or
                   (dirs.get("high") is not None and x >= dirs["high"]))
        if crossed:
            n_total += 1
            if col in vitals:
                n_vitals += 1
    return n_total, n_vitals

def _primary_vitals_pristine(row):
    for col in ("heartrate", "systolic_bp", "respiratory_rate", "spo2"):
        v = row.get(col)
        if _safe(v) and not _not_extreme(col, v):
            return False
    return True

def _count_extreme_features(row):
    n = 0
    for col, dirs in EXTREME_THRESHOLDS.items():
        v = row.get(col)
        if not _safe(v):
            continue
        for d in dirs:
            if _is_extreme(col, v, d):
                n += 1
    return n

# =============================================================================
# MAIN RULE CHECK
# =============================================================================

def check_row(row: pd.Series) -> dict:
    """Return dict with H_impossible and H_suspicious rule tags."""
    impossible = []
    suspicious = []

    age=row.get("age"); gl=row.get("glucose"); na=row.get("sodium")
    spo2=row.get("spo2"); rr=row.get("respiratory_rate"); hr=row.get("heartrate")
    sbp=row.get("systolic_bp"); dbp=row.get("diastolic_bp")
    cr=row.get("creatinine"); bun=row.get("blood_urea_nitro")
    k=row.get("potassium"); hb=row.get("hemoglobin")
    gender=row.get("gender"); icd9=row.get("icd9")
    bmi=row.get("bmi"); alb=row.get("albumin"); bnp=row.get("nt-probnp"); chol=row.get("cholesterol")

    # ---- reality limits ----
    for col, (lo, hi) in REALITY_LIMITS.items():
        v = row.get(col)
        if _safe(v) and (float(v) < lo or float(v) > hi):
            impossible.append("reality_limit")
            break

    # ---- categorical ----
    for col, allowed in CATEGORICAL.items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if v not in allowed:
            impossible.append("bad_category")
            break

    # ---- blood pressure / pulse pressure ----
    if _safe(sbp) and _safe(dbp):
        sbp_f, dbp_f = float(sbp), float(dbp)
        if dbp_f >= sbp_f:
            impossible.append("dbp_ge_sbp")
        else:
            pp = sbp_f - dbp_f
            vitals_ne = (_not_extreme("heartrate", hr) and
                         _not_extreme("respiratory_rate", rr) and
                         _not_extreme("spo2", spo2))

            if pp <= P["PP_IMPOSSIBLE_MAX"] and vitals_ne:
                impossible.append("pp_zero")
            elif pp < P["PP_SUSPICIOUS_MAX"] and (
                _is_normal("heartrate", hr) and
                _is_normal("respiratory_rate", rr) and
                _is_normal("spo2", spo2) and
                _is_normal("systolic_bp", sbp)
            ):
                suspicious.append("pp_low")

            if (pp > P["PP_WIDE_MIN"] and sbp_f < P["PP_WIDE_SBP_MAX"]
                    and _safe(age) and float(age) < P["PP_WIDE_AGE_MAX"]):
                suspicious.append("pp_wide")

            if sbp_f >= P["PP_HTN_SBP_MIN"] and pp < P["PP_HTN_PP_MAX"]:
                suspicious.append("pp_htn_narrow")

    # ---- BUN/Cr ----
    if _safe(bun) and _safe(cr) and float(cr) > 0:
        ratio = float(bun) / float(cr)
        if ratio < P["BUNCR_RATIO_IMPOSSIBLE_LOW"]:
            impossible.append("buncr_ratio_impossible")
        if (ratio > P["BUNCR_RATIO_HIGH"] and float(bun) > P["BUNCR_HIGH_BUN_MIN"]
                and _safe(hb) and float(hb) >= P["BUNCR_HIGH_HB_MIN"]):
            suspicious.append("buncr_ratio_high")

    if (_safe(cr) and float(cr) <= P["LOW_CR_MAX"]
            and _safe(age) and float(age) >= 18):
        suspicious.append("low_cr")

    # ---- RR/SpO2 impossible ----
    if _safe(rr) and _safe(spo2) and float(rr) <= 3 and float(spo2) >= 98:
        impossible.append("rr_spo2_impossible")

    # ---- compensation failure ----
    if (_safe(hb) and float(hb) <= P["COMP_ANEMIA_HB_MAX"]
            and _is_normal("heartrate", hr)
            and _safe(sbp) and float(sbp) < P["COMP_ANEMIA_SBP_MAX"]
            and _safe(rr) and float(rr) <= NORMAL_RANGES["respiratory_rate"]["high"]):
        suspicious.append("comp_anemia")

    if (_safe(sbp) and float(sbp) < P["COMP_SHOCK_SBP_MAX"]
            and _is_normal("heartrate", hr)
            and _safe(rr) and float(rr) <= NORMAL_RANGES["respiratory_rate"]["high"]):
        suspicious.append("comp_shock")

    if (_safe(k) and float(k) >= P["COMP_HYPERK_K_MIN"]
            and _is_normal("heartrate", hr)
            and _safe(cr) and float(cr) < P["COMP_HYPERK_CR_MAX"]):
        suspicious.append("comp_hyperk")

    if (_safe(na) and float(na) < P["COMP_HYPONA_NA_MAX"]
            and _safe(sbp) and float(sbp) < P["COMP_HYPONA_SBP_MAX"]
            and _safe(hr) and (float(hr) < 60 or float(hr) > 120)):
        suspicious.append("comp_hypona")

    # ---- pairwise vital incoherence ----
    if (_is_extreme("heartrate", hr, "low") and
        _is_extreme("systolic_bp", sbp, "low") and
        _not_extreme("spo2", spo2) and
        _not_extreme("respiratory_rate", rr)):
        suspicious.append("pair_cardiogenic")

    if _is_extreme("systolic_bp", sbp, "high") and _is_extreme("heartrate", hr, "low"):
        suspicious.append("pair_htn_brady")

    if _is_extreme("heartrate", hr, "high") and _is_extreme("systolic_bp", sbp, "high"):
        suspicious.append("pair_tachy_htn")

    # ---- constellation ----
    n_fatal, n_fatal_vitals = _count_fatal_all(row)
    if n_fatal >= 3 and n_fatal_vitals >= 1:
        if _primary_vitals_pristine(row):
            impossible.append("fatal_constellation_impossible")
        else:
            suspicious.append("fatal_constellation_suspicious")

    if _count_extreme_features(row) >= P["EXTREME_COUNT_MIN"]:
        suspicious.append("extreme_count")

    # ---- age ----
    if (_safe(age) and float(age) > P["ELDERLY_AGE_MIN"] and gender == "F"
            and _safe(hb) and float(hb) >= P["ELDERLY_HB_HIGH_MIN"]):
        suspicious.append("elderly_f_highhb")

    # ---- pregnancy ----
    if icd9 == "Pregnancy":
        if gender == "M":
            impossible.append("preg_male")
        if _safe(age) and (float(age) < P["PREG_AGE_MIN"] or float(age) > P["PREG_AGE_MAX"]):
            impossible.append("preg_age")
        if _safe(hb) and float(hb) > P["PREG_HB_MAX"]:
            suspicious.append("preg_hb")
        if row.get("admission_type") == "ELECTIVE":
            suspicious.append("pregnancy_elective")

    # ---- ICD9 structural ----
    if icd9 == "UNKNOWN" and (
        bnp == "Critical" or
        _is_extreme("creatinine", cr, "high") or
        _is_extreme("potassium", k, "high") or
        _is_extreme("hemoglobin", hb, "low")
    ):
        suspicious.append("unknown_extremelabs")

    if (icd9 == "Circulatory system" and chol == "Lowrisk" and bnp == "Normal"
            and _is_normal("heartrate", hr) and _is_normal("systolic_bp", sbp)):
        suspicious.append("circ_nomarkers")

       # ---- nutritional / albumin combos ----
    if (chol == "Highrisk" and bmi == "Underweight" and alb == "Low"
            and _safe(hb) and float(hb) <= P["NUT_CHOLMALN_HB_MAX"]):
        suspicious.append("nut_chol_maln")

    if (bmi in ("Obese", "Morbidly obese")
            and _safe(sbp) and float(sbp) <= P["NUT_OBESEHYPO_SBP_MAX"]
            and alb == "Low" and bnp == "Normal"):
        suspicious.append("nut_obese_hypo")

    if (alb == "Low" and bmi in ELEVATED_BMI
            and _safe(hb) and float(hb) <= P["ALB_OBESE_ANEMIA_HB_MAX"]
            and icd9 not in PROTEIN_WASTING_CHAPTERS):
        suspicious.append("alb_obese_anemia")

    if alb == "High" and bmi == "Underweight":
        suspicious.append("alb_high_underwt")

    if alb == "High" and icd9 in PROTEIN_WASTING_CHAPTERS:
        suspicious.append("alb_high_wasting")

    # severe hypokalemia
    if (_safe(k) and float(k) <= P["HYPOK_K_MAX"] and _is_normal("heartrate", hr)):
        suspicious.append("severe_hypok")

    # de-duplicate while preserving order
    imp = list(dict.fromkeys(impossible))
    sus = list(dict.fromkeys(suspicious))
    return {"H_impossible": "; ".join(imp), "H_suspicious": "; ".join(sus)}


# =============================================================================
# RUN + SUMMARY
# =============================================================================
from multiprocessing import Pool

# def run(df, num_bounds=None, cat_levels=None):
#     with Pool() as pool:
#         rows = pool.map(check_row, [row for _, row in df.iterrows()])
#     out = pd.DataFrame(rows)
#     out = pd.concat([df.reset_index(drop=True), out], axis=1)
#     ...
#     return out



def run(df: pd.DataFrame, num_bounds=None, cat_levels=None) -> pd.DataFrame:
    """Apply expert rules row-wise and return dataframe with rule columns."""
    res = df.apply(check_row, axis=1, result_type="expand")

    out = df.copy()
    out["H_impossible"] = res["H_impossible"]
    out["H_suspicious"] = res["H_suspicious"]

    if num_bounds is not None and cat_levels is not None:
        out["out_of_population"] = df.apply(
            lambda r: check_out_of_population(r, num_bounds, cat_levels), axis=1)

    return out


def rule_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of each rule fired."""
    n = len(out)
    rows = []

    for col in ("H_impossible", "H_suspicious"):
        for cell in out[col].fillna(""):
            for tag in str(cell).split(";"):
                tag = tag.strip()
                if tag:
                    rows.append({"bucket": col, "rule": tag})

    if not rows:
        return pd.DataFrame(columns=["bucket", "rule", "count", "pct_rows", "description"])

    s = (pd.DataFrame(rows)
         .value_counts(["bucket", "rule"])
         .reset_index(name="count")
         .sort_values(["bucket", "count"], ascending=[True, False]))

    s["pct_rows"] = (100 * s["count"] / n).round(3)
    s["description"] = s["rule"].map(RULE_DESCRIPTIONS).fillna("")

    return s
