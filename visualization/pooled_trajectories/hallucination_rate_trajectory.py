"""
visualization/pooled_trajectories/hallucination_rate_trajectory.py

Figure purpose 2 of 4 split out of PLOT3_mean_trajectories_gp_pooled_v9.py:
the hallucination-rate trajectory figure (HR, HR*, Expert rules, LLM, etc.),
one stacked-panel HTML, all four generators x both roots.

Writes {POOLED_FIGURES}/hr_trajectory_gp_pooled.html
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
    pretty, load_all_series, SHOW_BANDS, PANEL_HEIGHT, PANEL_WIDTH,
    HR_FIGURE_SCALE, HR_BIN,
)

OUT_DIR = POOLED_FIGURES

HR_PANELS = [
    ["Pilgram_HR"],
    ["HR_adjusted"],
    ["Expert_any"],
    ["LLM"],
    ["LLM_accumulated"],
    ["Expert_impossible"],
    ["Expert_suspicious"],
]

# Extra marker overrides for the two-metric overlay panels (kept distinct
# from the generator's own marker, which is used whenever a panel plots
# only one metric).
HR_PANEL_MARKERS = {
    "Expert_impossible": "circle",
    "Expert_suspicious": "x",
}


def build_hr_figure(series_list):
    hr_series = [s for s in series_list if s["hr"] is not None]
    if not hr_series:
        print("  [SKIP] HR figure: no hallucination data loaded")
        return

    panels = [p for p in HR_PANELS
              if any(f"{m}_mean" in s["hr"].columns
                     for m in p for s in hr_series)]
    if not panels:
        print("  [SKIP] HR figure: none of the HR metrics found")
        return

    titles = [(f"{pretty(p[0])} (↓ better)" if len(p) == 1
               else " vs ".join(pretty(m) for m in p) + " (↓ better)")
              for p in panels]
    n = len(panels)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=titles,
                        vertical_spacing=min(0.08, 0.35 / n))

    x_max = 0
    seen = set()
    for i, panel in enumerate(panels, start=1):
        for s in hr_series:
            for metric in panel:
                curve = get_curve(s, metric, scale=HR_FIGURE_SCALE)
                if curve is None:
                    continue
                x_max = max(x_max, curve[0].max())
                if len(panel) == 1:
                    group = name = s["label"]
                    marker = s["marker"]
                else:
                    group = f"{s['label']} — {metric}"
                    name = f"{s['label']} — {pretty(metric)}"
                    marker = HR_PANEL_MARKERS.get(metric, s["marker"])
                add_curve(fig, curve, name=name, color=s["color"],
                          dash=s["dash"], group=group, row=i,
                          show_band=SHOW_BANDS,
                          show_legend=(group not in seen), marker=marker)
                seen.add(group)
        fig.update_yaxes(title_text="%", row=i, col=1)

    add_collapse_marks(fig, hr_series, x_max)
    apply_generation_xaxis(fig, n)

    # See metric_family_trajectories.py's build_family_figure() for why
    # width= is now passed explicitly.
    height = max(420, PANEL_HEIGHT * n)
    apply_nature_layout(fig, height=height, width=PANEL_WIDTH)
    write_figure(fig, OUT_DIR, "hr_trajectory_gp_pooled",
                panel_kind="multi_panel")


def main():
    print("Loading series...")
    series_list = load_all_series()
    if not series_list:
        raise SystemExit("No data found — check METRIC_SUMMARIES / "
                         "HALLUCINATION_SUMMARIES.")

    print("\nHallucination rate trajectory:")
    build_hr_figure(series_list)
    print("\nDone.")


if __name__ == "__main__":
    main()