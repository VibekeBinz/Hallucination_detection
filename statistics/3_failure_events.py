"""
Per-chain Failure-of-utility event table.

Failure is defined as the first of two mechanisms to occur, generation by
generation:

  - signal_loss:    TSTR AUROC <= 0.5 for 2 consecutive *available*
                     generations for that chain (a gap in the data does not
                     break the streak -- it is simply skipped over).
  - class_collapse: the readmission minority-class share hits exactly 0.

A chain that never triggers either mechanism by its last observed
generation is censored there (event=0).

Inputs (both already produced by the current pipeline, read from
PER_CHAIN_DIR):
  - tstr_auroc_per_chain.csv   (generator, dseed, mseed, chain, generation, value)
  - minority_share.csv         (generator, variant, dseed, mseed, generation,
                                 n_rows, n_positive, positive_share,
                                 minority_share, n_classes, source_file)

minority_share.csv still spells its generator as a lowercase base token
plus a separate "variant" column (e.g. "arf", "core") rather than the
GENERATOR_VARIANTS-style combined token ("ARF_CORE") tstr_auroc_per_chain.csv
already uses -- config.generator_value() is the pipeline's existing
translation between the two, so it's used here rather than a new local
mapping.

Output: one row per chain -- generator, dseed, mseed, chain, time (onset
generation, or last observed generation if censored), event (1=failed,
0=censored), mechanism (signal_loss / class_collapse / both / none),
plus the diagnostic columns (n_tstr_datapoints, insufficient_tstr_data)
carried over from the crossing-detection step so a censored/insufficient
chain is distinguishable from one that was genuinely observed all the way
through.

NOTE on where this writes to: PER_CHAIN_DIR (see pipeline_config.py) is
documented as per-chain-per-generation trajectory output; this table is
per-chain (one row per chain, not per generation), so it doesn't belong
there. This script writes to STATISTICS_DIR, a new path that does not yet
exist in pipeline_config.py -- add it alongside the other path blocks:

    STATISTICS_DIR = os.path.join(RESULTS_ROOT, "statistics")

until that's added, this script creates the folder itself as a fallback
(see OUT_DIR below) so it still runs.
"""

import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)

import pandas as pd

from config.pipeline_config import PER_CHAIN_DIR, RESULTS_ROOT, generator_value

try:
    from config.pipeline_config import STATISTICS_DIR
    OUT_DIR = STATISTICS_DIR
except ImportError:
    OUT_DIR = os.path.join(RESULTS_ROOT, "statistics")

TSTR_FILE = os.path.join(PER_CHAIN_DIR, "tstr_auroc_per_chain.csv")
MINORITY_FILE = os.path.join(PER_CHAIN_DIR, "minority_share.csv")
OUT_FILE = os.path.join(OUT_DIR, "failure_events.csv")

AUROC_THRESHOLD = 0.5
K_CONSECUTIVE = 2
LAST_GENERATION = 19


def find_signal_loss_onset(gen_values, threshold=AUROC_THRESHOLD, k=K_CONSECUTIVE):
    """
    gen_values: list of (generation, value) sorted ascending by generation,
    already filtered to non-null values. Scans for the first run of k
    consecutive *available* points all <= threshold. A missing generation
    is skipped, not treated as breaking the streak.
    """
    for i in range(len(gen_values) - k + 1):
        window = gen_values[i:i + k]
        if all(v <= threshold for _, v in window):
            return window[0][0]
    return None


def find_class_collapse_onset(gen_values):
    for gen, v in gen_values:
        if v == 0:
            return gen
    return None


def main():
    tstr = pd.read_csv(TSTR_FILE)
    tstr["value"] = pd.to_numeric(tstr["value"], errors="coerce")
    tstr["generation"] = tstr["generation"].astype(int)

    minority = pd.read_csv(MINORITY_FILE)
    minority["generator"] = minority.apply(
        lambda r: generator_value(str(r["generator"]).upper(), str(r["variant"])),
        axis=1,
    )
    minority["generation"] = minority["generation"].astype(int)

    tstr_groups = {
        keys: grp.sort_values("generation")
        for keys, grp in tstr.groupby(["generator", "dseed", "mseed"], sort=False)
    }
    minority_groups = {
        keys: grp.sort_values("generation")
        for keys, grp in minority.groupby(["generator", "dseed", "mseed"], sort=False)
    }

    all_chains = sorted(set(tstr_groups) | set(minority_groups))
    rows = []

    for generator, dseed, mseed in all_chains:
        t_grp = tstr_groups.get((generator, dseed, mseed))
        m_grp = minority_groups.get((generator, dseed, mseed))

        if t_grp is not None:
            t_vals = [(g, v) for g, v in zip(t_grp["generation"], t_grp["value"]) if pd.notna(v)]
            chain_label = t_grp["chain"].iloc[0]
        else:
            t_vals = []
            chain_label = m_grp["chain"].iloc[0] if m_grp is not None and "chain" in m_grp else f"d{dseed} / m{mseed}"

        m_vals = [(g, v) for g, v in zip(m_grp["generation"], m_grp["minority_share"])] if m_grp is not None else []

        n_tstr_datapoints = len(t_vals)
        insufficient_tstr_data = n_tstr_datapoints < K_CONSECUTIVE

        signal_loss_gen = find_signal_loss_onset(t_vals)
        class_collapse_gen = find_class_collapse_onset(m_vals)

        candidates = [g for g in (signal_loss_gen, class_collapse_gen) if g is not None]

        if candidates:
            onset = min(candidates)
            mechanisms = []
            if signal_loss_gen == onset:
                mechanisms.append("signal_loss")
            if class_collapse_gen == onset:
                mechanisms.append("class_collapse")
            mechanism = "+".join(mechanisms)
            event = 1
            time = onset
        else:
            mechanism = ""
            event = 0
            last_observed = max(
                [g for g, _ in t_vals] + [g for g, _ in m_vals] + [LAST_GENERATION]
            )
            time = min(last_observed, LAST_GENERATION)

        rows.append({
            "generator": generator,
            "dseed": dseed,
            "mseed": mseed,
            "chain": chain_label,
            "time": time,
            "event": event,
            "mechanism": mechanism,
            "n_tstr_datapoints": n_tstr_datapoints,
            "insufficient_tstr_data": insufficient_tstr_data,
        })

    out = pd.DataFrame(rows).sort_values(["generator", "dseed", "mseed"])

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"[OK] Saved: {OUT_FILE}  ({len(out)} chains)")

    n_failed = int(out["event"].sum())
    n_insufficient = int(out["insufficient_tstr_data"].sum())
    print(f"[SUMMARY] {n_failed}/{len(out)} chains failed by generation {LAST_GENERATION}.")
    if n_insufficient:
        print(
            f"[SUMMARY] {n_insufficient} chain(s) had fewer than {K_CONSECUTIVE} "
            f"TSTR AUROC datapoints -- signal_loss could not be evaluated for "
            f"these beyond what's available; check n_tstr_datapoints/"
            f"insufficient_tstr_data before trusting a 'censored' verdict on them."
        )

    breakdown = out.groupby("generator")["event"].agg(["sum", "count"])
    print("\nBreakdown by generator:")
    for generator, r in breakdown.iterrows():
        print(f"  {generator}: {int(r['sum'])}/{int(r['count'])} chains failed")


if __name__ == "__main__":
    main()