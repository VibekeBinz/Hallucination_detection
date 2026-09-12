"""
Shared onset-detection helpers.

One canonical implementation of "peak-minus-baseline delta onset" and of
the full-chain registry fill, used by every script that needs to detect a
threshold-crossing event in a per-chain metric trajectory (hallucination
metrics, alpha-precision, beta-recall, authenticity, ...). Having this in
one place instead of copy-pasted per script is the point: every onset
column downstream -- across Claims 1, 2, and 3 -- is guaranteed to have
been computed the exact same way, so cross-claim comparisons are actually
comparable.

Onset definition: for a chain sorted by generation, baseline = the
chain's first available value; onset = the first generation at which the
cumulative running-max minus baseline reaches or exceeds a threshold T.
Both the input generation and the input value are used as given -- no
smoothing, no interpolation.

Chain registry: many per-chain metric files (especially hallucination
counts) simply have no row at all for a chain that never fired -- that is
a real zero-valued chain, not missing data, and dropping it would shrink
the denominator silently. Every script that reports "N/15" or "N/120"
crossing counts should reindex against the full chain list, not just the
chains that happen to appear in one metric's file. tstr_auroc_per_chain.csv
is the canonical registry source: TSTR AUROC is computed every generation
regardless of whether hallucinations were ever detected, so it lists every
(generator, dseed, mseed) chain that exists in the pipeline.
"""

import os

import pandas as pd


def load_chain_registry(per_chain_dir, registry_filename="tstr_auroc_per_chain.csv"):
    """
    Return the full (generator, dseed, mseed, chain) chain list, one row
    per chain, sourced from a file that is populated for every chain
    regardless of the metric being analyzed elsewhere.
    """
    path = os.path.join(per_chain_dir, registry_filename)
    reg = pd.read_csv(path, usecols=["generator", "dseed", "mseed", "chain"])
    return reg.drop_duplicates(subset=["generator", "dseed", "mseed"]).reset_index(drop=True)


def compute_onset_table(df, registry, threshold):
    """
    df: long-format per-chain metric data with columns
        generator, dseed, mseed, chain, generation, value
        (already restricted to a single metric -- callers concatenate
        their own per-generator files before calling this).
    registry: output of load_chain_registry() -- the full chain list to
        reindex against, so a chain absent from df is reported as a
        zero-valued, non-crossing chain rather than dropped.
    threshold: the peak-minus-baseline delta T.

    Returns one row per registry chain: generator, dseed, mseed, chain,
    baseline, peak, max_rise, onset_generation (None if never crossed),
    crossed (bool), and baseline_is_gen0 (False when the chain's first
    available row is not generation 0 -- flagged rather than silently
    treated as if it were).
    """
    df = df.copy()
    if not df.empty:
        df["generation"] = df["generation"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    present = {}
    for (generator, dseed, mseed), grp in df.groupby(["generator", "dseed", "mseed"], sort=False):
        grp = grp.sort_values("generation")
        gens = list(grp["generation"])
        vals = list(grp["value"])
        baseline = vals[0]

        onset_gen = None
        running_max = baseline
        for g, v in zip(gens, vals):
            if pd.isna(v):
                continue
            running_max = max(running_max, v)
            if running_max - baseline >= threshold:
                onset_gen = g
                break

        peak = max((v for v in vals if pd.notna(v)), default=float("nan"))
        present[(generator, dseed, mseed)] = {
            "chain": grp["chain"].iloc[0],
            "baseline": baseline,
            "peak": peak,
            "max_rise": peak - baseline,
            "onset_generation": onset_gen,
            "crossed": onset_gen is not None,
            "baseline_is_gen0": (gens[0] == 0),
        }

    rows = []
    for _, reg_row in registry.iterrows():
        key = (reg_row["generator"], reg_row["dseed"], reg_row["mseed"])
        if key in present:
            rec = present[key]
        else:
            rec = {
                "chain": reg_row["chain"],
                "baseline": 0.0,
                "peak": 0.0,
                "max_rise": 0.0,
                "onset_generation": None,
                "crossed": False,
                "baseline_is_gen0": True,
            }
        rows.append({
            "generator": key[0],
            "dseed": key[1],
            "mseed": key[2],
            **rec,
        })

    return pd.DataFrame(rows).sort_values(["generator", "dseed", "mseed"]).reset_index(drop=True)


def compute_incline_onset_table(df, registry, threshold, direction="increase"):
    """
    The current spike definition, used identically for hallucination in
    both Claim 1 (1_hallucination_onset.py) and Claim 2
    (claim2_hallucination_collapse.py), and for all four metrics compared in Claim 2
    (hallucination_expert_any, beta-recall, alpha-precision, authenticity)
    -- same function, each metric supplying its own calibrated (threshold,
    direction) pair, so "spike" is computed the same way everywhere it's
    used, not a family of similar-but-different definitions.

    Onset here is neither the cumulative peak-minus-baseline rise used by
    compute_onset_table() nor the single largest jump anywhere in the
    chain used by compute_spike_generation_table(). It is the first
    single-step (generation-over-generation) change in the specified
    direction that exceeds a threshold T, scanned in generation order --
    i.e. literally "the start of the spike," the first real jump, rather
    than the point furthest from where the chain began or the single
    biggest jump wherever in the run it happens to fall.

    direction: "increase" (the metric spikes upward as a chain degrades --
        hallucination rate, authenticity) requires
        value[i] - value[i-1] >= threshold; "decrease" (the metric
        collapses downward -- beta-recall, alpha-precision) requires
        value[i] - value[i-1] <= -threshold. Each metric's threshold is
        calibrated on its own scale (see metric_config.py for every
        metric's calibration) -- this is why a direction-specific, threshold-
        gated definition is safe to use here even though the earlier
        direction-agnostic compute_spike_generation_table() was needed
        when no such per-metric threshold existed yet.

    df: long-format per-chain metric data (generator, dseed, mseed, chain,
        generation, value), already restricted to a single metric.
    registry: output of load_chain_registry() -- a chain absent from df,
        or with fewer than 2 observed generations, is reported as
        non-spiking (crossed=False, onset_generation=None) rather than
        dropped or left undefined: no data (or one point) means no jump
        was observed, which is a real non-event for this purpose, not a
        gap in the analysis.
    threshold: the one-step change T (always given as a positive number)
        a chain must exceed, in the given direction, to count.

    Returns one row per registry chain: generator, dseed, mseed, chain,
    onset_generation (the generation immediately after the first
    qualifying step; None if no step ever exceeded T), step_size (the
    signed size of that qualifying step -- negative for direction=
    "decrease"), crossed (bool), and n_observed_generations.
    """
    if direction not in ("increase", "decrease"):
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")

    df = df.copy()
    if not df.empty:
        df["generation"] = df["generation"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    present = {}
    for (generator, dseed, mseed), grp in df.groupby(["generator", "dseed", "mseed"], sort=False):
        grp = grp.sort_values("generation").dropna(subset=["value"])
        gens = list(grp["generation"])
        vals = list(grp["value"])
        n = len(vals)

        onset_gen = None
        step_size = None
        for i in range(1, n):
            diff = vals[i] - vals[i - 1]
            qualifies = (diff >= threshold) if direction == "increase" else (diff <= -threshold)
            if qualifies:
                onset_gen = gens[i]
                step_size = diff
                break

        present[(generator, dseed, mseed)] = {
            "chain": grp["chain"].iloc[0] if n else None,
            "onset_generation": onset_gen,
            "step_size": step_size,
            "crossed": onset_gen is not None,
            "n_observed_generations": n,
        }

    rows = []
    for _, reg_row in registry.iterrows():
        key = (reg_row["generator"], reg_row["dseed"], reg_row["mseed"])
        rec = present.get(key) or {
            "chain": reg_row["chain"],
            "onset_generation": None,
            "step_size": None,
            "crossed": False,
            "n_observed_generations": 0,
        }
        if rec.get("chain") is None:
            rec = {**rec, "chain": reg_row["chain"]}
        rows.append({"generator": key[0], "dseed": key[1], "mseed": key[2], **rec})

    return pd.DataFrame(rows).sort_values(["generator", "dseed", "mseed"]).reset_index(drop=True)


def compute_spike_generation_table(df, registry):
    """
    Threshold-free alternative to compute_onset_table(), for comparing the
    *timing* of a spike across metrics that live on different natural
    scales (e.g. a hallucination rate against alpha-precision, beta-recall,
    authenticity) -- picking one absolute threshold T and applying it to
    every metric silently assumes they are on the same scale, which they
    are not (see 1_hallucination_onset.py's docstring on the CORE-vs-FULL
    scale mismatch between expert_any and hr_b5).

    "Spike generation" here is defined as the generation of the single
    largest one-step CHANGE in the chain's trajectory (argmax of the
    *absolute* first difference) -- scale-invariant, no threshold
    parameter, and direction-agnostic. Direction-agnostic is deliberate,
    not incidental: hallucination_expert_any and authenticity rise as a
    chain degrades, but alpha-precision and beta-recall (delta_coverage)
    fall as a chain degrades -- collapsing from their generation-0 values
    toward ~0 within the first few generations and then sitting flat,
    with only sub-0.002 noise, for the remainder of the run. An earlier
    version of this function took argmax of the signed difference (i.e.
    the single largest *increase*), which is correct for the two metrics
    that rise but silently wrong for the two that fall: for a chain that
    collapses in generations 0-5 and then stays flat, the largest signed
    increase is a meaningless noise-level uptick somewhere in the flat
    noisy tail (observed: generation ~14-18, size ~0.001), not the actual
    collapse -- while the real event, a drop of ~0.85-0.9, is invisible to
    "biggest increase" by construction. Using absolute value fixes this
    for both directions without hardcoding which metrics rise and which
    fall.

    Returns one row per registry chain: generator, dseed, mseed, chain,
    spike_generation (the generation *after* the biggest jump -- i.e. the
    first generation at the new, post-jump level), spike_size (the signed
    size of that jump -- negative for a drop), and n_observed_generations.
    A chain with fewer than 2 observed generations, or one absent from df
    entirely, gets spike_generation=None (there's no single-step change to
    locate) and is still included via the registry rather than dropped.
    """
    df = df.copy()
    if not df.empty:
        df["generation"] = df["generation"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    present = {}
    for (generator, dseed, mseed), grp in df.groupby(["generator", "dseed", "mseed"], sort=False):
        grp = grp.sort_values("generation").dropna(subset=["value"])
        gens = list(grp["generation"])
        vals = list(grp["value"])
        n = len(vals)

        if n < 2:
            present[(generator, dseed, mseed)] = {
                "chain": grp["chain"].iloc[0] if n else None,
                "spike_generation": None,
                "spike_size": None,
                "n_observed_generations": n,
            }
            continue

        diffs = [vals[i] - vals[i - 1] for i in range(1, n)]
        best_i = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
        present[(generator, dseed, mseed)] = {
            "chain": grp["chain"].iloc[0],
            "spike_generation": gens[best_i + 1],
            "spike_size": diffs[best_i],
            "n_observed_generations": n,
        }

    rows = []
    for _, reg_row in registry.iterrows():
        key = (reg_row["generator"], reg_row["dseed"], reg_row["mseed"])
        rec = present.get(key) or {
            "chain": reg_row["chain"],
            "spike_generation": None,
            "spike_size": None,
            "n_observed_generations": 0,
        }
        if rec.get("chain") is None:
            rec = {**rec, "chain": reg_row["chain"]}
        rows.append({"generator": key[0], "dseed": key[1], "mseed": key[2], **rec})

    return pd.DataFrame(rows).sort_values(["generator", "dseed", "mseed"]).reset_index(drop=True)