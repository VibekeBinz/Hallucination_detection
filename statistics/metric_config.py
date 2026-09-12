"""
Single source of truth for the onset threshold and direction of every
metric this pipeline detects a "spike" in.

This lives in its own unprefixed module rather than inside one of the
numbered results scripts (1_hallucination_onset.py, 2_qametric_onset.py,
3_failure_events.py) because Python module names cannot start with a
digit -- a file literally named 1_hallucination_onset.py cannot be
`import`ed by another script under that name at all (it isn't a valid
Python identifier), so any value another script needs to reuse has to
live somewhere without a numeric prefix. Every numbered results script
reads its own metric's entry from here to compute onset; every claim
script that needs a metric's threshold for reporting
(claim2_hallucination_collapse.py) reads the same entry -- so the
calibration is defined exactly once, regardless of how many scripts
need it, and can't drift between the script that computes onset and the
script that tests a claim against it.

HALLUCINATION_METRICS: metric name -> glob pattern (relative to
PER_CHAIN_DIR) for that metric's per-generator step4 files, plus the
shared (threshold, direction). T=0.05 (a 5-percentage-point one-step
rise) sits at approximately the 91st percentile of the pooled
distribution of ALL single-generation changes in Expert_any across the
120 chains (2,280 chain-generation transitions total, every generator
and _CORE variant combined): 205 of those 2,280 transitions meet or
exceed 0.05. The distribution has no natural gap anywhere near this
value -- it is a smooth, long-tailed curve running from about 0.02 up
through 0.36, with more transitions landing in [0.05, 0.06) than in
[0.04, 0.05) -- so T=0.05 is a percentile-based cutoff chosen to isolate
roughly the upper decile of observed one-step jumps, not a threshold
sitting in an empirical gap between two separated clusters. Both
hallucination metrics are run through the same detector for
consistency, though hr_b5 is not expected to give a meaningful onset
signal; T=0.05 was calibrated on expert_any specifically.

QA_METRICS: metric name -> (source file in PER_CHAIN_DIR, threshold,
direction) for the three QA/generation-quality metrics that measure mode
collapse and degeneration. Each threshold is 5% of that metric's own
typical (median, across all 120 chains) generation-0 value -- calibrated
separately per metric since the three don't share a natural absolute
scale with each other or with hallucination:
  beta_recall (alpha_delta_coverage_OC_per_chain.csv):      median gen0 =
    0.4832 -> T=0.0242, direction="decrease" (collapses toward 0 as a
    chain degrades).
  alpha_precision (alpha_delta_precision_OC_per_chain.csv): median gen0 =
    0.9566 -> T=0.0478, direction="decrease".
  authenticity (alpha_authenticity_OC_per_chain.csv):       median gen0 =
    0.5088 -> T=0.0254, direction="increase" (rises as a chain degrades,
    same direction as hallucination).

LAST_GENERATION: the last generation index in the pipeline (0-indexed,
so 19 = 20 generations). Used to flag a censored chain (in
3_failure_events.py's output) whose own follow-up stopped short of the
full run, and by claim3_timetofailure_ranking.py for the same check.
"""

HALLUCINATION_METRICS = {
    "expert_any": {
        "glob": "step4_expert_any_per_chain_*.csv",
        "threshold": 0.05, "direction": "increase",
    },
    "hr_b5": {
        "glob": "step4_hr_b5_per_chain_*.csv",
        "threshold": 0.05, "direction": "increase",
    },
}

QA_METRICS = {
    "beta_recall": {
        "file": "alpha_delta_coverage_OC_per_chain.csv",
        "threshold": 0.0242, "direction": "decrease",
    },
    "alpha_precision": {
        "file": "alpha_delta_precision_OC_per_chain.csv",
        "threshold": 0.0478, "direction": "decrease",
    },
    "authenticity": {
        "file": "alpha_authenticity_OC_per_chain.csv",
        "threshold": 0.0254, "direction": "increase",
    },
}

LAST_GENERATION = 19