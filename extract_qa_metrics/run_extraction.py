"""
extract_qa_metrics/run_extraction.py

Runs every qa_metrics extractor in sequence. Mirrors run_pipeline.py's
role for hallucination_pipeline/ -- a single entry point that duplicates
no extraction logic itself.

Order:
  1. extract_class_collapse.py -- survival_common.py reads its output
     directly, and it surfaces a raw-data problem (wrong RAW_DATA_ROOT,
     missing label column) before the larger per-generator loops run.
  2. extract_tstr_trtr.py
  3. extract_sd_quality_metrics.py
  4. apply_raw_data_patches.py -- must run after (3): it patches
     sd_quality's per-chain output in place. See that script's docstring
     for why this step exists and when to delete it.
"""

import sys
from datetime import datetime
from pathlib import Path

# Needed when this file is run directly (python run_extraction.py) rather
# than as a module -- Python only puts this file's own folder on sys.path
# in that case, not the repo root, so "from extract_qa_metrics import ..." below
# can't find the package one level up without this.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from extract_qa_metrics import (
    apply_raw_data_patches,
    extract_class_collapse,
    extract_sd_quality_metrics,
    extract_tstr_trtr,
)


def main():
    t0 = datetime.now()

    print("\n=== qa_metrics: class collapse ===")
    extract_class_collapse.main()

    print("\n=== qa_metrics: TSTR / TRTR ===")
    extract_tstr_trtr.main()

    print("\n=== qa_metrics: SD quality metrics ===")
    extract_sd_quality_metrics.main()

    apply_raw_data_patches.main()

    print(f"\nqa_metrics extraction done -- elapsed: {datetime.now() - t0}")


if __name__ == "__main__":
    main()