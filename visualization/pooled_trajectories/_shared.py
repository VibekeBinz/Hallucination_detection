"""
visualization/pooled_trajectories/_shared.py

Everything the four pooled-trajectory figure scripts (metric_family_
trajectories, hallucination_rate_trajectory, metric_overlay, bin_resolution)
have in common: loading the metric/hallucination summaries, curve
extraction with the band-collapse-below-n rule, the collapse-onset vertical
markers, the TRTR reference lines, and the shared axis/legend formatting.

This is the split of PLOT3_mean_trajectories_gp_pooled_v9.py's single 940-
line file into one script per figure purpose, per Debbie's instruction --
the data plumbing below is unchanged from that script; only the four figure
BUILDERS moved out into their own modules.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)
from config.pipeline_config import (
    METRIC_SUMMARIES, HALLUCINATION_SUMMARIES, POOLED_FIGURES, MISC_FIGURES,
    PLOT_BIN, BIN_LIST, PRETTY as CONFIG_PRETTY,
    QA_METRICS_POOLED_SUMMARY_DIR, generator_value, variant_from_token,
)
from visualization.style import (
    GENERATOR_COLORS, GENERATOR_MARKERS, ROOT_LABEL, ROOT_SUFFIX, ROOT_DASH,
    COLLAPSE_ANNOTATION_COLOR, apply_nature_layout, write_figure, hex_to_rgba,
)

# ============================================================
# CONFIG
# ============================================================

ROOTS = ["Step7pfa", "Step7pfp"]
GENERATORS = ["CTGAN", "ARF", "RTVAE", "DDPM"]

HR_BIN = PLOT_BIN
N_GENERATIONS = 20
GENERATION_X_RANGE = [0, N_GENERATIONS - 1]

BAND_MIN_N = 5    # below this the CI band collapses onto the mean
POINT_MIN_N = 1   # below this the point itself is dropped

BAND_ALPHA = 0.15
PANEL_HEIGHT = 320
# On-screen figure width for every stacked-panel (rows=n, cols=1) pooled-
# trajectory figure. Previously only implicit -- each figure script computed
# an `aspect = height / 900` for write_figure() without ever setting this as
# the figure's actual on-screen width, so the print-DPI export scale had
# nothing real to be computed from. Now apply_nature_layout(fig, height=...,
# width=PANEL_WIDTH) sets it explicitly, matching the 900 already baked into
# every aspect ratio below (no visual change -- this makes that assumption
# real instead of implicit; see write_figure()'s docstring in style.py).
PANEL_WIDTH = 900
SHOW_BANDS = True

OVERLAY_SCALE = 1.0
HR_FIGURE_SCALE = 100.0

HR_METRICS = {
    "Pilgram_HR", "HR_adjusted", "Copy",
    "Expert_any", "Expert_impossible", "Expert_suspicious",
    "LLM", "LLM_accumulated",
}

TRTR_REFERENCE = {
    "tstr_auroc":  "trtr_auroc",
    "tstr_recall": "trtr_recall",
    "tstr_f1":     "trtr_f1",
}

TRTR_REF_STYLE = {
    "Step7pfa": dict(color="#000000", dash="dot"),
    "Step7pfp": dict(color="#7f7f7f", dash="dashdot"),
}

TRTR_EXCLUDE_GENERATORS = set()
TRTR_SPREAD_TOL = 1e-6

IGNORE_METRICS = {
    "summary",
    "HR",   # the metric-summaries HR column -- all hallucination figures
            # use the merged hallucination summaries instead of this one.
    "jsd_nannyml", "jsd_synthcity",
    "alpha_precision", "prdc_avg", "sdmetrics_overall",
    "k_anonymity_synthetic",
    "tstr_accuracy", "tstr_precision",
    "tstr_false_positive_rate", "tstr_false_negative_rate",
    "tstr_true_negative_rate",
}

DIRECTION_OVERRIDES = {
    "utility_gap": "↓ better",
    "MemorizedFR": "↓ better",
    "tstr_fpr": "↓ better",
    "tstr_fnr": "↓ better",
}

BIN_DEPENDENT_METRICS = {"Pilgram_HR", "HR_adjusted"}

# Physical bounds for a metric's CI band -- so a small-n/high-variance
# interval (see load_tstr_summary()'s docstring: TSTR/TRTR has no "_n"
# column, so _curve()'s low-n band-collapse never triggers for these) can
# never be drawn past what the metric can actually take. Only TSTR/TRTR
# rate metrics (bounded [0, 1]) and utility_gap (a difference of two such
# rates, bounded [-1, 1]) have a known physical bound this way; everything
# else routed through get_curve()'s non-HR branch (MMD, Wasserstein,
# training time, ...) is left unclipped, since clipping an unbounded metric
# would silently corrupt real data rather than fix an impossible one.
RATE_METRIC_BOUNDS = (0.0, 1.0)
UTILITY_GAP_BOUNDS = (-1.0, 1.0)


def _metric_bounds(metric):
    """(clip_low, clip_high) for metric, or (None, None) if it has no known
    physical bound."""
    if metric == "utility_gap":
        return UTILITY_GAP_BOUNDS
    if metric.startswith("tstr_") or metric.startswith("trtr_"):
        return RATE_METRIC_BOUNDS
    return None, None


def pretty(metric, with_bin=False):
    label = CONFIG_PRETTY.get(metric, metric.replace("_", " ").title())
    if with_bin and metric in BIN_DEPENDENT_METRICS:
        label = f"{label} ({HR_BIN} bins)"
    return label


def direction(metric, default_dir):
    d = DIRECTION_OVERRIDES.get(metric, default_dir)
    return f"({d})" if d else ""


# ---------------- loading --------------------------------------------------

def load_metric_summary(root, gen):
    """
    root is "Step7pfa"/"Step7pfp", gen is a base generator name (CTGAN, ARF,
    RTVAE, DDPM) -- both converted to the single GENERATOR-variant token
    (e.g. "ARF_CORE") that sd_quality_summary_<token>.csv is actually named
    after, one file per token, in QA_METRICS_POOLED_SUMMARY_DIR.
    """
    generator_token = generator_value(gen, variant_from_token(root))
    path = os.path.join(QA_METRICS_POOLED_SUMMARY_DIR,
                        f"sd_quality_summary_{generator_token}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] missing metric summary: {path}")
        return None
    df = pd.read_csv(path)
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    return df.sort_values("generation").reset_index(drop=True)


def load_tstr_summary(root, gen):
    """Pooled TSTR/TRTR summary -- a separate file from
    sd_quality_summary_<token>.csv, written by extract_tstr_trtr.py to the
    same QA_METRICS_POOLED_SUMMARY_DIR. Merged into the metric-summary df
    in load_all_series() so every existing panel/routing path (get_curve,
    _curve, metric_bases) picks these columns up the same way it already
    does for sd_quality's own columns."""
    generator_token = generator_value(gen, variant_from_token(root))
    path = os.path.join(QA_METRICS_POOLED_SUMMARY_DIR,
                        f"tstr_trtr_summary_{generator_token}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] missing TSTR/TRTR summary: {path}")
        return None
    df = pd.read_csv(path)
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    return df


def load_hr_summary(root, gen):
    """Merged hallucination summary, filtered to HR_BIN."""
    name = f"merged_hallucination_summary_{gen}{ROOT_SUFFIX[root]}.csv"
    path = os.path.join(HALLUCINATION_SUMMARIES, name)
    if not os.path.exists(path):
        print(f"  [WARN] missing hallucination summary: {path}")
        return None

    df = pd.read_csv(path)
    df["generation"] = (df["generation"].astype(str)
                        .str.extract(r"(\d+)").astype(float))
    df = df[df["bin"] == HR_BIN].sort_values("generation").reset_index(drop=True)
    if df.empty:
        print(f"  [WARN] no rows with bin == {HR_BIN} in {path}")
        return None
    return df


# ---------------- curve extraction -----------------------------------------

_missing_metric_warned = set()


def _curve(df, metric, lo_suffix, hi_suffix, scale, clip_low=None, clip_high=None):
    """
    (x, mean, lo, hi) or None.

    Points with n < POINT_MIN_N are dropped. The band collapses onto the
    mean where n < BAND_MIN_N -- a t-CI on 3 runs is meaningless and can run
    negative. Both checks are a no-op (n is treated as None, below) whenever
    the source has no "<metric>_n" column -- true of every QA metric read
    from sd_quality_summary_<GENERATOR>.csv and tstr_trtr_summary_
    <GENERATOR>.csv right now, which both carry mean/var/low/high but no
    per-point sample count. So for those metrics every generation's point
    is always shown and the band is never collapsed, regardless of how few
    chains actually contributed to it -- clip_low/clip_high (see
    _metric_bounds()) is the safety net for exactly that case: a small-n,
    high-variance interval that would otherwise be drawn past a metric's
    known physical bound (e.g. TSTR recall's CI spanning -4.5 to 5.2 when
    recall itself can only ever be in [0, 1]).

    clip_low/clip_high are in the metric's own physical units and are
    scaled by `scale` before being applied, so a bound of e.g. 1.0 still
    means "1.0 in the metric's own units" even when the curve itself is
    plotted at a different scale (e.g. HR_FIGURE_SCALE=100 for a percentage
    axis).
    """
    if df is None:
        return None

    mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        # A metric name that doesn't match any "<name>_mean" column used to
        # fail silently here -- the figure just came out missing that line,
        # with nothing in the console to say why. Warn once per metric name
        # (not once per generator/root -- that would repeat the same line
        # up to 8 times) and list what column names actually are available,
        # so a mismatch is visible on the very next run instead of requiring
        # a manual column dump.
        if metric not in _missing_metric_warned:
            available = sorted(c[:-5] for c in df.columns if c.endswith("_mean"))
            print(f"  [WARN] metric '{metric}': no column '{mean_col}' in "
                  f"the metric summary -- skipped on every panel/generator. "
                  f"Available metrics here: {available}")
            _missing_metric_warned.add(metric)
        return None

    sub = df.dropna(subset=[mean_col])
    if sub.empty:
        return None

    n_col = f"{metric}_n"
    if n_col in sub.columns:
        n = pd.to_numeric(sub[n_col], errors="coerce").fillna(0)
        sub, n = sub[n >= POINT_MIN_N], n[n >= POINT_MIN_N]
        if sub.empty:
            return None
    else:
        n = None

    x = sub["generation"].values
    m = sub[mean_col].values.astype(float) * scale

    lo_col, hi_col = f"{metric}{lo_suffix}", f"{metric}{hi_suffix}"
    lo = (sub[lo_col].values.astype(float) * scale
          if lo_col in sub.columns else m.copy())
    hi = (sub[hi_col].values.astype(float) * scale
          if hi_col in sub.columns else m.copy())
    lo = np.where(np.isnan(lo), m, lo)
    hi = np.where(np.isnan(hi), m, hi)

    if n is not None:
        weak = (n < BAND_MIN_N).values
        lo = np.where(weak, m, lo)
        hi = np.where(weak, m, hi)

    if clip_low is not None:
        lo = np.maximum(lo, clip_low * scale)
    if clip_high is not None:
        hi = np.minimum(hi, clip_high * scale)

    return x, m, lo, hi


def get_curve(series, metric, scale=1.0):
    """Route to the right source. Rates are clipped at zero.

    Both branches use "_low"/"_high" -- that's the suffix convention
    sd_quality_summary_<GENERATOR>.csv and tstr_trtr_summary_<GENERATOR>.csv
    both use for their CI columns (alongside "_mean" and "_var"; neither has
    a "_n" column, so the low-n band-collapse / point-drop logic in
    _curve() never triggers for these metrics -- see the module-level note
    above _curve()). The non-HR branch additionally clips the CI band to
    the metric's known physical range via _metric_bounds() (TSTR/TRTR rates
    -> [0, 1], utility_gap -> [-1, 1]; anything without a known bound is
    left unclipped).
    """
    if metric in HR_METRICS:
        return _curve(series["hr"], metric, "_low", "_high", scale, clip_low=0.0)
    clip_low, clip_high = _metric_bounds(metric)
    return _curve(series["df"], metric, "_low", "_high", scale,
                  clip_low=clip_low, clip_high=clip_high)


def metric_bases(df):
    bases = [c[:-5] for c in df.columns if c.endswith("_mean")]
    return [b for b in bases if b not in IGNORE_METRICS]


def first_collapse(df):
    """First generation where any run had a collapsed readmission class."""
    if df is None or "readmission_collapsed_mean" not in df.columns:
        return None
    sub = df[["generation", "readmission_collapsed_mean"]].dropna()
    hit = sub[sub["readmission_collapsed_mean"] > 0]
    return None if hit.empty else float(hit["generation"].iloc[0])


# ---------------- drawing --------------------------------------------------

def add_curve(fig, curve, name, color, dash, group, row=None, col=1,
              show_band=True, show_legend=True, opacity=1.0,
              band_alpha=BAND_ALPHA, marker=None):
    """Mean line plus optional CI band, band drawn first so it sits behind.

    col defaults to 1 -- every existing single-column figure only ever
    passes row=, so this is a no-op for them. A multi-column figure (see
    metric_overlay.py's joined full+core layout) passes col explicitly.
    """
    x, m, lo, hi = curve
    kw = {} if row is None else dict(row=row, col=col)

    if show_band:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, band_alpha),
            line=dict(width=0), hoverinfo="skip",
            legendgroup=group, showlegend=False), **kw)

    fig.add_trace(go.Scatter(
        x=x, y=m,
        mode="lines" if marker is None else "lines+markers",
        line=dict(color=color, dash=dash, width=2),
        marker=None if marker is None
               else dict(symbol=marker, size=7, color=color),
        opacity=opacity, name=name,
        legendgroup=group, showlegend=show_legend), **kw)


def add_collapse_marks(fig, series_list, x_max, col=None, row="all"):
    """Vertical line per collapsing series, one annotation, shaded tail.

    col=None (default) reproduces the exact prior behaviour -- no row/col
    passed to add_vline/add_annotation/add_vrect at all, as every existing
    single-column caller expects. Pass col explicitly (see metric_overlay.
    py's joined full+core layout) to scope the marks to one column of a
    multi-column figure; row="all" then applies them to every row in it.
    """
    kw = {} if col is None else dict(row=row, col=col)
    points = []
    for s in series_list:
        cg = s.get("collapse")
        if cg is None:
            continue
        points.append(cg)
        fig.add_vline(x=cg, line_color=s["color"], line_dash="dash",
                      line_width=1.3, opacity=0.8, **kw)

    if not points:
        return

    earliest = min(points)
    fig.add_annotation(x=earliest, y=1.005, yref="paper", xref="x",
                       text="RTVAE class collapse", showarrow=False,
                       xanchor="left",
                       font=dict(color=COLLAPSE_ANNOTATION_COLOR, size=10),
                       **kw)
    fig.add_vrect(x0=earliest, x1=x_max + 0.5, fillcolor="red",
                  opacity=0.02, line_width=0, **kw)


def compute_trtr_references(series_list):
    """{(root, ref_metric): value} -- one scalar per root per reference metric."""
    refs = {}
    for root in ROOTS:
        for ref_metric in sorted(set(TRTR_REFERENCE.values())):
            per_generator = {}
            for s in series_list:
                if s["root"] != root or s["gen"] in TRTR_EXCLUDE_GENERATORS:
                    continue
                df = s["df"]
                if df is None or f"{ref_metric}_mean" not in df.columns:
                    continue
                vals = pd.to_numeric(df[f"{ref_metric}_mean"],
                                     errors="coerce").dropna()
                if not vals.empty:
                    per_generator[s["gen"]] = float(vals.mean())

            if not per_generator:
                continue

            arr = np.array(list(per_generator.values()))
            if arr.max() - arr.min() > TRTR_SPREAD_TOL:
                spread = ", ".join(f"{g}={v:.4f}"
                                   for g, v in sorted(per_generator.items()))
                print(f"  [WARN] {ref_metric} ({ROOT_LABEL[root]}) differs "
                      f"across generators — using the mean. {spread}")
            refs[(root, ref_metric)] = float(arr.mean())
    return refs


def add_trtr_reference(fig, refs, ref_metric, row, seen):
    """Horizontal reference line per root, drawn across the full x range."""
    for root in ROOTS:
        value = refs.get((root, ref_metric))
        if value is None:
            continue
        style = TRTR_REF_STYLE.get(root, dict(color="black", dash="dot"))
        fig.add_trace(go.Scatter(
            x=[0, N_GENERATIONS - 1], y=[value, value],
            mode="lines",
            line=dict(color=style["color"], dash=style["dash"], width=2),
            name=f"{pretty(ref_metric)} — {ROOT_LABEL[root]}",
            legendgroup=f"trtr_{ref_metric}_{root}",
            showlegend=(ref_metric, root) not in seen,
            hovertemplate=f"{pretty(ref_metric)} "
                          f"({ROOT_LABEL[root]}): {value:.4f}<extra></extra>",
        ), row=row, col=1)
        seen.add((ref_metric, root))


def apply_generation_xaxis(fig, n_rows, x_range=GENERATION_X_RANGE, col=1):
    """Tick marks/numbers at every generation, on every panel in one column.

    col defaults to 1 (every existing single-column caller is unaffected);
    a multi-column figure calls this once per column with col= explicit.
    """
    for r in range(1, n_rows + 1):
        fig.update_xaxes(
            showticklabels=True,
            tickmode="linear", tick0=0, dtick=1,
            ticks="outside", ticklen=4,
            range=x_range,
            row=r, col=col,
        )
    fig.update_xaxes(title_text="Generation", row=n_rows, col=col)


# ---------------- loading all series ---------------------------------------

def load_all_series():
    series_list = []
    for root in ROOTS:
        for gen in GENERATORS:
            df = load_metric_summary(root, gen)
            tstr_df = load_tstr_summary(root, gen)
            if df is not None and tstr_df is not None:
                df = df.merge(tstr_df, on="generation", how="outer")
            elif df is None:
                df = tstr_df
            hr = load_hr_summary(root, gen)
            if df is None and hr is None:
                continue
            series_list.append(dict(
                root=root, gen=gen, df=df, hr=hr,
                label=f"{gen} ({ROOT_LABEL[root]})",
                color=GENERATOR_COLORS.get(gen, "#333333"),
                marker=GENERATOR_MARKERS.get(gen),
                dash=ROOT_DASH.get(root, "solid"),
                collapse=first_collapse(df),
            ))
    return series_list