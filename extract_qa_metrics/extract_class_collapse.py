"""
extract_qa_metrics/extract_class_collapse.py

Builds the tidy per-chain, per-generation class-balance table that the
survival analysis reads to determine class collapse.

Single pass, self-verifying: DISCOVER_ONLY mode to inspect a folder's files
before committing to name filters, ambiguity-is-error on multiple candidate
files per chain/generation, and sanity checks on whether class balance
actually moves across generations.

Output: summary_files/per_chain/minority_share.csv, one row per
chain per generation:
    generator, variant, dseed, mseed, generation,
    n_rows, n_positive, positive_share, minority_share, n_classes, source_file

minority_share = min(positive_share, 1 - positive_share), so it reaches 0
whenever the readmission label has collapsed in EITHER direction: all-
negative (the RTVAE pattern) or all-positive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import PER_CHAIN_DIR, RAW_DATA_ROOT, variant_from_token

DATA_ROOT = Path(RAW_DATA_ROOT)
OUT_PATH = Path(PER_CHAIN_DIR) / "minority_share.csv"

LABEL_COL = "readmission"
LABEL_COL_ALIASES = ("readmission_30d", "readmit", "readmitted", "label", "target")

INCLUDE_NAME_REGEX = r"^synthetic_synthcity_.*_decoded\.csv$"
EXCLUDE_NAME_PREFIXES = (
    "reference_", "real_", "train_", "test_", "population_", "holdout_",
)
AMBIGUITY_IS_ERROR = True

DISCOVER_ONLY = False
DISCOVER_N_FOLDERS = 2

EXCLUDE_DIRS = {"metrics", "results", "ids", "plots", "figures", "logs"}

MAX_GEN = 19

SEED_DIR_RE = re.compile(
    r"^(Step7pf[ap])_dseed(\d+)_synthcity_([A-Za-z0-9]+)_mseed(\d+)$", re.IGNORECASE
)
GEN_RE = re.compile(r"_gen_(\d+)(?:[_.]|$)")

TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
FALSE_TOKENS = {"0", "false", "no", "n", "f"}


def to_binary(series: pd.Series) -> pd.Series:
    """Map a readmission column to 0/1 regardless of how it was serialised."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)

    norm = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[norm.isin(TRUE_TOKENS)] = 1.0
    out[norm.isin(FALSE_TOKENS)] = 0.0
    unknown = norm[out.isna()].unique()
    if len(unknown):
        raise ValueError(
            f"Unrecognised values in '{LABEL_COL}': {list(unknown)[:5]}. "
            f"Extend TRUE_TOKENS / FALSE_TOKENS."
        )
    return out.astype(int)


_LABEL_CACHE: dict = {}


def read_label(path: Path):
    """Return (label series, header) -- the series is None if no label
    column exists."""
    cached = _LABEL_CACHE.get("name")
    if cached is not None:
        try:
            return pd.read_csv(path, usecols=[cached])[cached], []
        except ValueError:
            pass  # this file uses a different spelling; fall through and resolve

    header = list(pd.read_csv(path, nrows=0).columns)
    lookup = {c.strip().lower(): c for c in header}
    for candidate in (LABEL_COL, *LABEL_COL_ALIASES):
        col = lookup.get(candidate.strip().lower())
        if col is not None:
            _LABEL_CACHE["name"] = col
            return pd.read_csv(path, usecols=[col])[col], header
    return None, header


def find_seed_dirs(root: Path) -> list:
    dirs = [p for p in root.rglob("Step7pf*_dseed*_mseed*") if p.is_dir()]
    return [p for p in dirs if SEED_DIR_RE.match(p.name)]


def is_synthetic_name(name: str) -> bool:
    lower = name.lower()
    if any(lower.startswith(p.lower()) for p in EXCLUDE_NAME_PREFIXES):
        return False
    if INCLUDE_NAME_REGEX and not re.search(INCLUDE_NAME_REGEX, name, re.IGNORECASE):
        return False
    return True


def discover(seed_dirs: list) -> None:
    """Print the filenames present in a few seed folders, then stop."""
    for seed_dir in seed_dirs[:DISCOVER_N_FOLDERS]:
        print(f"\n{seed_dir}")
        names = sorted(p.relative_to(seed_dir) for p in seed_dir.rglob("*.csv"))
        if not names:
            print("   (no csv files)")
        for rel in names[:30]:
            gen = GEN_RE.search(rel.name)
            mark = "KEEP" if (gen and is_synthetic_name(rel.name)) else "drop"
            print(f"   [{mark}] {rel}")
        if len(names) > 30:
            print(f"   ... and {len(names) - 30} more")
    print("\nDISCOVER_ONLY is True -- set it to False once the KEEP/drop split is right.")


def candidate_files(seed_dir: Path) -> dict:
    """Map generation -> synthetic dataset path for one chain."""
    by_gen: dict = {}
    for path in seed_dir.rglob("*.csv"):
        if any(part.lower() in EXCLUDE_DIRS for part in path.relative_to(seed_dir).parts[:-1]):
            continue
        if not is_synthetic_name(path.name):
            continue
        m = GEN_RE.search(path.name)
        if m is None:
            continue
        gen = int(m.group(1))
        if gen > MAX_GEN:
            continue
        by_gen.setdefault(gen, []).append(path)

    chosen: dict = {}
    for gen, paths in by_gen.items():
        paths = sorted(paths)
        if len(paths) > 1:
            listing = "\n".join(f"      {p.name}" for p in paths)
            msg = (
                f"{len(paths)} candidate synthetic files for {seed_dir.name} gen {gen}:\n"
                f"{listing}\n"
                f"   Tighten INCLUDE_NAME_REGEX or EXCLUDE_NAME_PREFIXES, or set "
                f"DISCOVER_ONLY = True to inspect the folder."
            )
            if AMBIGUITY_IS_ERROR:
                raise SystemExit(f"  [stop] {msg}")
            print(f"  [warn] {msg}\n   using {paths[0].name}")
        chosen[gen] = paths[0]
    return chosen


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seed_dirs = find_seed_dirs(DATA_ROOT)
    print(f"1. found {len(seed_dirs)} chain folders under {DATA_ROOT}")
    if not seed_dirs:
        raise SystemExit(
            "No Step7pf*_dseed*_synthcity_*_mseed* folders found. Check RAW_DATA_ROOT."
        )
    if DISCOVER_ONLY:
        discover(seed_dirs)
        return

    rows, missing_label, unreadable = [], [], []
    n_candidates = 0
    for i, seed_dir in enumerate(seed_dirs, start=1):
        prefix, dseed, generator, mseed = SEED_DIR_RE.match(seed_dir.name).groups()
        variant = variant_from_token(prefix)

        files = candidate_files(seed_dir)
        n_candidates += len(files)
        for gen, path in sorted(files.items()):
            try:
                y_raw, header = read_label(path)
            except OSError as err:
                unreadable.append((path, err))
                continue
            if y_raw is None:
                missing_label.append((path, header))
                continue

            y = to_binary(y_raw)
            n_rows = int(len(y))
            n_pos = int(y.sum())
            share = n_pos / n_rows if n_rows else np.nan
            rows.append(
                {
                    "generator": generator.lower(),
                    "variant": variant,
                    "dseed": int(dseed),
                    "mseed": int(mseed),
                    "generation": gen,
                    "n_rows": n_rows,
                    "n_positive": n_pos,
                    "positive_share": share,
                    "minority_share": min(share, 1.0 - share) if n_rows else np.nan,
                    "n_classes": int(y.nunique()),
                    "source_file": str(path),
                }
            )

        if i % 20 == 0:
            print(f"   ... {i}/{len(seed_dirs)} chains")

    if missing_label:
        example, header = missing_label[0]
        print(f"  [warn] {len(missing_label)} files have no recognised label column.")
        print(f"         example: {example.name}")
        print(f"         its columns: {header[:15]}")
    if unreadable:
        print(f"  [warn] {len(unreadable)} files could not be read, e.g. {unreadable[0][0]}")

    if not rows:
        raise SystemExit(
            "  [stop] no usable synthetic datasets found.\n"
            f"         files passing the name filters: {n_candidates}\n"
            f"         of those, missing a label column: {len(missing_label)}\n"
            f"         of those, unreadable: {len(unreadable)}\n"
            "         If the first number is 0, the name filters are wrong: set\n"
            "         DISCOVER_ONLY = True to see the KEEP/drop split per file.\n"
            "         If it is non-zero, the label column is named something else:\n"
            "         add it to LABEL_COL_ALIASES (the columns are printed above)."
        )

    out = pd.DataFrame(rows).sort_values(
        ["variant", "generator", "dseed", "mseed", "generation"]
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"2. wrote {OUT_PATH} ({len(out)} rows, "
          f"{out.groupby(['variant','generator','dseed','mseed']).ngroups} chains, "
          f"label column '{_LABEL_CACHE.get('name', LABEL_COL)}')")

    print("3. sanity check: is the class balance actually moving?")
    spread = (
        out.groupby(["variant", "generator", "dseed", "mseed"])["n_positive"]
        .nunique()
        .reset_index(name="distinct_values")
    )
    frozen = spread[spread["distinct_values"] <= 1]
    if frozen.empty:
        print(f"   varies within every chain (median {int(spread['distinct_values'].median())} "
              f"distinct values across 20 generations)")
    else:
        print(f"   [FAIL] {len(frozen)} chains have an identical positive count at every "
              f"generation. That is what reading a fixed reference/real partition looks "
              f"like. Check INCLUDE_NAME_REGEX and EXCLUDE_NAME_PREFIXES before using "
              f"this file.")
        print(frozen.head(10).to_string(index=False))

    print("4. class collapse (minority_share == 0), first generation per chain:")
    collapsed = out[out["minority_share"] == 0]
    if collapsed.empty:
        print("   none")
    else:
        first = (
            collapsed.sort_values("generation")
            .drop_duplicates(subset=["variant", "generator", "dseed", "mseed"], keep="first")
            .groupby(["variant", "generator"])["generation"]
            .agg(["count", "min", "median", "max"])
        )
        print(first.to_string())

    print("5. chains with fewer than the expected number of generations:")
    counts = out.groupby(["variant", "generator", "dseed", "mseed"]).size()
    short = counts[counts < MAX_GEN + 1]
    if short.empty:
        print("   none")
    else:
        print(short.to_string())
        print("   (a chain that terminates because the class collapsed is expected here;"
              " a chain with an internal gap is not)")

    print("done")


if __name__ == "__main__":
    main()