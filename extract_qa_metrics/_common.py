"""
extract_qa_metrics/_common.py

Shared chain-discovery and stats helpers for every extractor in this
package -- one definition of "what is a chain folder" and "how do we
summarise an array of per-chain values", not one per extractor.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.pipeline_config import GENERATOR_VARIANTS, RAW_DATA_ROOT

LAST_GEN = 19
GENERATIONS = list(range(0, LAST_GEN + 1))
EXPECTED_CHAINS = 15

CHAIN_DIR_RE = re.compile(
    r"^(Step7pf[ap])_dseed(\d+)_synthcity_([A-Za-z0-9]+)_mseed(\d+)$", re.IGNORECASE
)


def find_chain_dirs(generator_name: str) -> list:
    """All dseed/mseed chain folders under RAW_DATA_ROOT for one GENERATOR
    value (e.g. 'DDPM_CORE')."""
    combo = GENERATOR_VARIANTS[generator_name.upper()]
    prefix = combo["raw_prefix"]
    token = combo["base_token"]
    export_dir = os.path.join(RAW_DATA_ROOT, f"{prefix}_{token}_export")
    if not os.path.isdir(export_dir):
        print(f"  [WARN] export folder not found: {export_dir}")
        return []
    pattern = os.path.join(export_dir, f"{prefix}_dseed*mseed*")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def seed_pair(chain_dir: str):
    """(dseed, mseed) as ints, parsed from the chain folder name. None if
    the name doesn't match the expected shape."""
    name = os.path.basename(chain_dir)
    m = CHAIN_DIR_RE.match(name)
    if not m:
        return None
    _, dseed, _, mseed = m.groups()
    return int(dseed), int(mseed)


def stats(arr) -> tuple:
    """mean, var, low (95% CI), high (95% CI) -- same convention as
    hallucination_pipeline's stats() so pooled summaries read the same way
    across every package."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = arr.mean()
    var = arr.var(ddof=1) if n > 1 else 0.0
    sd = arr.std(ddof=1) if n > 1 else 0.0
    tcrit = t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    ci = tcrit * sd / np.sqrt(n)
    return mean, var, mean - ci, mean + ci


def check_chain_count(chain_dirs: list, generator_name: str):
    if len(chain_dirs) != EXPECTED_CHAINS:
        print(f"  WARNING: {generator_name} has {len(chain_dirs)} chain "
              f"folders, expected {EXPECTED_CHAINS}")