"""
visualization/km_time_to_failure.py

Kaplan-Meier estimate of time-to-Failure by generator, built through this
package's shared visual identity and export pipeline (visualization/
style.py): GENERATOR_COLORS / GENERATOR_MARKERS for generator identity,
apply_nature_layout / write_figure for the same font, legend, and
print-DPI handling every other figure in the manuscript uses. Import
style.py rather than redefining any of this locally.

Two panels -- base generators and their _CORE counterparts -- rather
than style.py's single-plot/dash-carries-root convention: for this
figure the two roots stay visually separate (matching how
claim3_timetofailure_ranking.py already treats them as two independent
rankings, never pooled onto one axes), instead of overlaying all 8 lines
as solid/dashed pairs on shared axes. Within each panel, line dash still
follows the package's dash-means-CORE convention (solid in the base
panel, dashed in the CORE panel) so the figure stays consistent with
that rule even though it's split across two panels rather than one.
Marker symbols (GENERATOR_MARKERS) mark the censoring ticks, so
generator identity survives on a third channel (color + dash-per-panel +
marker shape) the way style.py's colorblind/grayscale-safety rationale
asks for.

Kaplan-Meier estimator: the standard product-limit estimator,
S(t) = product over observed failure times t_i <= t of (1 - d_i / n_i),
where n_i is the number of chains still at risk (time >= t_i) and d_i is
the number that failed exactly at t_i. S(t) is the point estimate itself
(the estimated probability a chain from that generator is still
un-failed at generation t) -- not a mean of anything, and not smoothed;
it is a step function that only moves at generations where a failure was
actually observed. Computed directly here (no `lifelines` dependency) --
`lifelines` is only needed for the Cox model in
claim1_hallucination_utility.py, not for this plot. Censored chains
(event=0, i.e. never failed within the observed generations) leave the
risk set at their own time without contributing a death, and are marked
on the curve with the generator's own marker symbol at the survival
level they leave at -- the standard KM convention for showing where
censoring happened without implying survival dropped there.

95% confidence bands use Greenwood's formula for the variance of S(t):
Var[S(t)] = S(t)^2 * sum over t_i <= t of d_i / (n_i * (n_i - d_i)),
giving S(t) +/- 1.96*sqrt(Var[S(t)]), clipped to [0, 1]. This is the
plain (untransformed) Greenwood interval rather than the log-log
transformed interval some software defaults to -- the transformed
version is undefined once S(t) reaches exactly 0 (log(0)), which happens
here for RTVAE/RTVAE_CORE (15/15 chains fail), so the untransformed
interval is used for every curve rather than switching methods only for
the curves that hit 0.

Input: failure_events.csv (3_failure_events.py's output) -- generator,
dseed, mseed, time, event (1=failed, 0=censored).

Output: this script follows the package's standard workflow
(style.write_figure) -- a self-contained HTML file with a camera-button
PNG export wired to Nature's default DPI/print-width settings, rather
than a PNG/PDF written directly by this script.
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
# metric_config.py lives in statistics/, a sibling of this file's own
# visualization/ directory -- not on sys.path by default the way it is
# for a script that lives in statistics/ itself, so it's added explicitly
# rather than duplicating LAST_GENERATION as a second hardcoded constant.
_statistics_dir = os.path.join(_p, "statistics")
if _statistics_dir not in sys.path:
    sys.path.insert(0, _statistics_dir)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.pipeline_config import RESULTS_ROOT, MISC_FIGURES
from metric_config import LAST_GENERATION
from style import (
    GENERATOR_COLORS, GENERATOR_MARKERS, ROOT_DASH,
    apply_nature_layout, write_figure, hex_to_rgba,
)

try:
    from config.pipeline_config import STATISTICS_DIR
    STATS_DIR = STATISTICS_DIR
except ImportError:
    STATS_DIR = os.path.join(RESULTS_ROOT, "statistics")

# Where this figure lives: MISC_FIGURES (config.pipeline_config), the
# same constant every other misc-category figure uses -- previously
# hardcoded independently here instead of importing it, which is exactly
# what let this go stale when the for_article/ tree was renamed.
OUT_DIR = MISC_FIGURES

FAILURE_EVENTS_FILE = os.path.join(STATS_DIR, "failure_events.csv")

# Base generator -> its _CORE counterpart, and which panel + dash each
# root gets. Dash values come straight from style.py's ROOT_DASH (keyed
# by the raw Step7pfa/Step7pfp token) rather than a second local mapping.
PANELS = [
    {"title": "Full dataset", "suffix": "", "dash": ROOT_DASH["Step7pfa"]},
    {"title": "Core dataset", "suffix": "_CORE", "dash": ROOT_DASH["Step7pfp"]},
]

_PLOTLY_DASH = {"solid": "solid", "dash": "dash"}


def kaplan_meier(times, events, max_time):
    """
    Standard product-limit (Kaplan-Meier) estimator, with a pointwise 95%
    confidence interval via Greenwood's formula (see module docstring).

    Returns (steps, censor_points): steps is a list of
    (time, survival, ci_lo, ci_hi) tuples defining the step function --
    extended flat out to max_time if the last observed time is earlier;
    censor_points is a list of (time, survival) marking where a censored
    chain leaves the risk set.
    """
    df = pd.DataFrame({"time": times, "event": events})
    unique_times = sorted(df["time"].unique())
    survival = 1.0
    greenwood_sum = 0.0
    steps = [(0.0, 1.0, 1.0, 1.0)]
    censor_points = []
    for t in unique_times:
        n_at_risk = int((df["time"] >= t).sum())
        d = int(((df["time"] == t) & (df["event"] == 1)).sum())
        c = int(((df["time"] == t) & (df["event"] == 0)).sum())
        if d > 0 and n_at_risk > 0:
            survival *= (1 - d / n_at_risk)
            if n_at_risk > d:
                greenwood_sum += d / (n_at_risk * (n_at_risk - d))
            se = survival * (greenwood_sum ** 0.5)
            ci_lo = max(0.0, survival - 1.96 * se)
            ci_hi = min(1.0, survival + 1.96 * se)
            steps.append((float(t), survival, ci_lo, ci_hi))
        if c > 0:
            censor_points.append((float(t), survival))
    if steps[-1][0] < max_time:
        last = steps[-1]
        steps.append((float(max_time), last[1], last[2], last[3]))
    return steps, censor_points


def add_panel_traces(fig, df, base_generators, suffix, dash, col, showlegend):
    for base in base_generators:
        generator = f"{base}{suffix}"
        sub = df[df["generator"] == generator]
        if sub.empty:
            print(f"  [WARN] no rows for {generator} -- skipped")
            continue

        steps, censor_points = kaplan_meier(sub["time"].values, sub["event"].values, LAST_GENERATION)
        xs = [p[0] for p in steps]
        ys = [p[1] for p in steps]
        los = [p[2] for p in steps]
        his = [p[3] for p in steps]
        color = GENERATOR_COLORS[base]
        marker = GENERATOR_MARKERS[base]
        line_dash = _PLOTLY_DASH.get(dash, "solid")
        n = len(sub)
        n_events = int(sub["event"].sum())

        # Confidence band: invisible upper bound, then lower bound filled
        # up to it -- the standard plotly two-trace pattern for a shaded
        # band, using the package's own hex_to_rgba() for the fill color.
        fig.add_trace(
            go.Scatter(x=xs, y=his, mode="lines", line=dict(width=0),
                       hoverinfo="skip", showlegend=False, legendgroup=generator),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(x=xs, y=los, mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor=hex_to_rgba(color, 0.15),
                       hoverinfo="skip", showlegend=False, legendgroup=generator),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", line_shape="hv",
                line=dict(color=color, dash=line_dash, width=2.0),
                name=f"{generator} ({n_events}/{n} failed)",
                legendgroup=generator, showlegend=showlegend,
            ),
            row=1, col=col,
        )
        if censor_points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in censor_points], y=[p[1] for p in censor_points],
                    mode="markers", marker=dict(symbol=marker, color=color, size=8,
                                                 line=dict(width=1, color=color)),
                    legendgroup=generator, showlegend=False, hoverinfo="skip",
                ),
                row=1, col=col,
            )


def main():
    df = pd.read_csv(FAILURE_EVENTS_FILE)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = df["event"].astype(int)

    base_generators = list(GENERATOR_COLORS.keys())

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                         subplot_titles=[p["title"] for p in PANELS],
                         horizontal_spacing=0.06)

    for col, panel in enumerate(PANELS, start=1):
        add_panel_traces(fig, df, base_generators, panel["suffix"], panel["dash"],
                          col, showlegend=(col == 1))

    margin = 0.4
    for col in (1, 2):
        fig.update_xaxes(title_text="Generation", range=[-margin, LAST_GENERATION + margin],
                          tick0=0, dtick=1, row=1, col=col)
    fig.update_yaxes(title_text="Survival probability (not yet failed)", range=[0, 1.05],
                      row=1, col=1)

    apply_nature_layout(fig, height=460, width=1000, legend=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_figure(fig, OUT_DIR, "km_time_to_failure", panel_kind="multi_panel")


if __name__ == "__main__":
    main()