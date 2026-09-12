"""
visualization/pooled_trajectories/metric_overlay.py

Figure purpose 3 of 4 split out of PLOT3_mean_trajectories_gp_pooled_v9.py:
one page per root (full / core), four stacked panels (one per generator),
every OVERLAY_METRIC overlaid on a shared 0-1 axis. Colour = metric here
(not generator — the panel title already carries the generator, so no
second channel is needed there); every metric also gets its own marker
symbol, so the 8-line overlay stays legible under grayscale printing.

A third, joined page puts full and core side by side instead of on
separate pages: one row per generator, full in the left column, core in
the right, so a reader compares the two without flipping between pages.
Same metrics, same colours/markers, one shared legend.

Writes {POOLED_FIGURES}/metric_overlay_gp_pooled_full.html
       {POOLED_FIGURES}/metric_overlay_gp_pooled_core.html
       {POOLED_FIGURES}/metric_overlay_gp_pooled_joined.html
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
from config.pipeline_config import POOLED_FIGURES
from visualization.style import apply_nature_layout, write_figure

from plotly.subplots import make_subplots

from visualization.pooled_trajectories._shared import (
    add_curve, add_collapse_marks, apply_generation_xaxis, get_curve,
    pretty, load_all_series, PANEL_HEIGHT, PANEL_WIDTH, ROOT_LABEL,
    OVERLAY_SCALE, HR_BIN,
)

OUT_DIR = POOLED_FIGURES
GENERATORS = ["CTGAN", "ARF", "RTVAE", "DDPM"]

OVERLAY_METRICS = [
    "alpha_delta_precision_OC",
    "alpha_delta_coverage_OC",
    "alpha_authenticity_OC",
    "tstr_auroc",
    "Pilgram_HR",
    "HR_adjusted",
    "Expert_any",
    "LLM_accumulated",
]

OVERLAY_METRIC_COLORS = {
    "alpha_delta_precision_OC":     "#af1f7f",
    "alpha_delta_coverage_OC":      "#c0183c",
    "alpha_authenticity_OC":        "#d86806",
    "tstr_auroc":                   "#cabd0a",
    "Pilgram_HR":                   "#1f77b4",
    "HR_adjusted":                  "#4eb4ce",
    "Expert_any":                   "#2ca02c",
    "LLM_accumulated":              "#025522",
}

# Every metric gets its own marker symbol (not just the 4 the legacy script
# bothered with) so all 8 overlaid lines stay distinguishable by shape alone.
OVERLAY_METRIC_MARKERS = {
    "alpha_delta_precision_OC":     "triangle-up",
    "alpha_delta_coverage_OC":      "square",
    "alpha_authenticity_OC":        "star",
    "tstr_auroc":                   "hexagon",
    "Pilgram_HR":                   "circle",
    "HR_adjusted":                  "x",
    "Expert_any":                   "diamond",
    "LLM_accumulated":              "cross",
}

OVERLAY_Y_RANGE = [0.0, 1.0]
OVERLAY_SHOW_BANDS = True


def build_overlay_figure(series_list, root):
    """One page, four stacked panels (one generator each), every
    OVERLAY_METRIC overlaid on a shared 0-100% axis."""
    root_series = [s for s in series_list if s["root"] == root]
    if not root_series:
        print(f"  [SKIP] overlay ({ROOT_LABEL[root]}): no series loaded")
        return

    gens = [g for g in GENERATORS if any(s["gen"] == g for s in root_series)]
    n = len(gens)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=gens, vertical_spacing=0.07)

    x_max = 0
    seen = set()
    for i, gen in enumerate(gens, start=1):
        s = next(x for x in root_series if x["gen"] == gen)
        for metric in OVERLAY_METRICS:
            curve = get_curve(s, metric, scale=OVERLAY_SCALE)
            if curve is None:
                continue
            x_max = max(x_max, curve[0].max())
            add_curve(
                fig, curve,
                name=pretty(metric, with_bin=True),
                color=OVERLAY_METRIC_COLORS.get(metric, "#333333"),
                dash="solid", group=metric, row=i,
                show_band=OVERLAY_SHOW_BANDS, band_alpha=0.08,
                show_legend=(metric not in seen),
                marker=OVERLAY_METRIC_MARKERS.get(metric),
            )
            seen.add(metric)

        fig.update_yaxes(title_text="Value (0–1)", range=OVERLAY_Y_RANGE,
                         row=i, col=1)

    add_collapse_marks(fig, root_series, x_max)
    apply_generation_xaxis(fig, n)

    label = ROOT_LABEL[root]
    # See metric_family_trajectories.py's build_family_figure() for why
    # width= is now passed explicitly (write_figure() needs the figure's
    # real on-screen size to compute a correct print-DPI export scale).
    height = max(600, PANEL_HEIGHT * n)
    apply_nature_layout(fig, height=height, width=PANEL_WIDTH)
    write_figure(fig, OUT_DIR, f"metric_overlay_gp_pooled_{label}",
                panel_kind="multi_panel")


def build_joined_overlay_figure(series_list):
    """One page, full (col 1) and core (col 2) side by side, one row per
    generator -- same metrics/colours/markers as the two per-root pages,
    laid out for direct full-vs-core comparison instead of two pages.

    Uses the fixed GENERATORS list for rows (not "whichever generators
    have data for this root", as build_overlay_figure does) so row i is
    the same generator in both columns; a (generator, root) combination
    with no data just leaves that one panel blank rather than shifting
    the grid.
    """
    root_col = {"Step7pfa": 1, "Step7pfp": 2}
    titles = []
    for gen in GENERATORS:
        for root in ("Step7pfa", "Step7pfp"):
            titles.append(f"{gen} ({ROOT_LABEL[root]})")

    n = len(GENERATORS)
    fig = make_subplots(rows=n, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=titles, vertical_spacing=0.05,
                        horizontal_spacing=0.06)

    x_max_by_col = {1: 0, 2: 0}
    seen = set()

    for row_i, gen in enumerate(GENERATORS, start=1):
        for root, col in root_col.items():
            matches = [s for s in series_list
                      if s["root"] == root and s["gen"] == gen]
            if not matches:
                continue
            s = matches[0]
            for metric in OVERLAY_METRICS:
                curve = get_curve(s, metric, scale=OVERLAY_SCALE)
                if curve is None:
                    continue
                x_max_by_col[col] = max(x_max_by_col[col], curve[0].max())
                add_curve(
                    fig, curve,
                    name=pretty(metric, with_bin=True),
                    color=OVERLAY_METRIC_COLORS.get(metric, "#333333"),
                    dash="solid", group=metric, row=row_i, col=col,
                    show_band=OVERLAY_SHOW_BANDS, band_alpha=0.08,
                    show_legend=(metric not in seen),
                    marker=OVERLAY_METRIC_MARKERS.get(metric),
                )
                seen.add(metric)

        fig.update_yaxes(title_text="Value (0–1)", range=OVERLAY_Y_RANGE,
                         row=row_i, col=1)
        fig.update_yaxes(range=OVERLAY_Y_RANGE, row=row_i, col=2)

    for root, col in root_col.items():
        root_series = [s for s in series_list if s["root"] == root]
        add_collapse_marks(fig, root_series, x_max_by_col[col], col=col)

    for col in (1, 2):
        apply_generation_xaxis(fig, n, col=col)

    height = max(600, PANEL_HEIGHT * n)
    apply_nature_layout(fig, height=height, width=PANEL_WIDTH * 2)
    write_figure(fig, OUT_DIR, "metric_overlay_gp_pooled_joined",
                panel_kind="multi_panel")


def main():
    print("Loading series...")
    series_list = load_all_series()
    if not series_list:
        raise SystemExit("No data found — check METRIC_SUMMARIES / "
                         "HALLUCINATION_SUMMARIES.")

    ROOTS = ["Step7pfa", "Step7pfp"]
    print("\nMetric overlays:")
    for root in ROOTS:
        build_overlay_figure(series_list, root)
    build_joined_overlay_figure(series_list)
    print("\nDone.")


if __name__ == "__main__":
    main()