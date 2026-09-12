"""
visualization/pooled_trajectories/metric_family_trajectories.py

Figure purpose 1 of 4 split out of PLOT3_mean_trajectories_gp_pooled_v9.py:
per-metric-family trajectories, one HTML per family, all four generators x
both roots (full/core) on stacked panels.

Writes into {POOLED_FIGURES} (config.pipeline_config):
  fidelity_diversity_generalization_gp_pooled
  detection_trajectory_gp_pooled
  divergence_distance_trajectory_gp_pooled
  prdc_trajectory_gp_pooled
  privacy_trajectory_gp_pooled
  statistical_complement_trajectory_gp_pooled
  tstr_trajectory_gp_pooled
  training_time_gp_pooled

Visual identity: color + marker = generator (visualization/style.py, shared
with every other figure in the manuscript), dash = root (full solid, core
dashed). Markers are drawn on every curve here (the legacy script only put
markers on the HR figure) so the four generators stay distinguishable by
shape alone under grayscale printing or a colorblind reader, not just color.
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
    metric_bases, pretty, direction, load_all_series, compute_trtr_references,
    add_trtr_reference, TRTR_REFERENCE, SHOW_BANDS, PANEL_HEIGHT, PANEL_WIDTH,
)

OUT_DIR = POOLED_FIGURES

FIGURES = [
    dict(name="fidelity_diversity_generalization_gp_pooled",
         title="Fidelity, Diversity, and Generalization",
         panels=[("prefix", "alpha")],
         default_dir="↑ better",
         reverse_panels=True),

    dict(name="detection_trajectory_gp_pooled",
         title="Detection Trajectory",
         panels=[("prefix", "detection")],
         default_dir="↓ better"),

    dict(name="divergence_distance_trajectory_gp_pooled",
         title="Divergence / Distance Trajectory",
         panels=[("metric", "wasserstein_dist"),
                 ("metric", "mmd_corrected"),
                 ("metric", "jsd_syndat")],
         default_dir="↓ better"),

    dict(name="prdc_trajectory_gp_pooled",
         title="PRDC Trajectory",
         panels=[("prefix", "prdc")],
         default_dir="↑ better"),

    dict(name="privacy_trajectory_gp_pooled",
         title="Privacy Trajectory",
         panels=[("metric", "new_row_synthesis")],
         default_dir="↑ better"),

    dict(name="statistical_complement_trajectory_gp_pooled",
         title="Statistical Complement Trajectory",
         panels=[("metric", "tv_complement"),
                 ("metric", "ks_complement"),
                 ("prefix", "sdmetrics")],
         default_dir="↑ better"),

    dict(name="tstr_trajectory_gp_pooled",
         title="TSTR / TRTR Trajectory",
         panels=[("metric", "utility_gap"),
                 ("metric", "tstr_auroc"),
                 ("metric", "tstr_f1"),
                 ("metric", "tstr_recall")],
         default_dir="↑ better",
         # Hardcoded per-panel y-axis ranges for this figure only: the
         # RTVAE-collapse generation (see add_collapse_marks) leaves a
         # handful of generations with very few surviving chains, and
         # tstr_trtr_summary_<GENERATOR>.csv carries no "_n" column (see
         # _curve()'s docstring in _shared.py), so those generations' CI
         # bands are wide even after _curve()'s [0,1]/[-1,1] physical
         # clipping. Fixed ranges keep every other generation readable
         # instead of autorange stretching to fit that one collapsed band.
         # tstr_recall is left out (no override -- autorange already reads
         # well there).
         y_axis_overrides={
             "utility_gap": [0, 0.2],
             "tstr_auroc": [0.45, 0.65],
             "tstr_f1": [0, 0.3],
         }),

    dict(name="training_time_gp_pooled",
         title="Training Time Trajectory (seconds)",
         panels=[("metric", "training_time"),
                 ("metric", "generation_time")],
         default_dir="↓ better"),
]


def resolve_panels(figdef, series_list):
    available = set()
    for s in series_list:
        if s["df"] is not None:
            available.update(metric_bases(s["df"]))

    panels = []
    for kind, target in figdef["panels"]:
        if kind == "prefix":
            hits = sorted(m for m in available if m.startswith(target))
            if not hits:
                print(f"  [WARN] [{figdef['name']}] no metrics with prefix "
                      f"'{target}'")
            panels.extend(hits)
        elif kind == "metric":
            if target in available:
                panels.append(target)
            else:
                print(f"  [WARN] [{figdef['name']}] metric '{target}' not "
                      f"found — panel skipped")
    return panels


def build_family_figure(figdef, series_list, trtr_refs):
    panels = resolve_panels(figdef, series_list)
    if not panels:
        print(f"  [SKIP] {figdef['name']}: no panels resolved")
        return
    if figdef.get("reverse_panels"):
        panels = list(reversed(panels))

    y_axis_overrides = figdef.get("y_axis_overrides", {})

    titles = [f"{pretty(m)} {direction(m, figdef['default_dir'])}"
              for m in panels]
    n = len(panels)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.15)

    x_max = 0
    seen_refs = set()
    for i, metric in enumerate(panels, start=1):
        for s in series_list:
            curve = get_curve(s, metric)
            if curve is None:
                continue
            x_max = max(x_max, curve[0].max())
            add_curve(fig, curve, name=s["label"], color=s["color"],
                      dash=s["dash"], group=s["label"], row=i,
                      show_band=SHOW_BANDS, show_legend=(i == 1),
                      marker=s["marker"])

        ref = TRTR_REFERENCE.get(metric)
        if ref is not None:
            add_trtr_reference(fig, trtr_refs, ref, row=i, seen=seen_refs)

        if metric in y_axis_overrides:
            fig.update_yaxes(range=y_axis_overrides[metric], row=i, col=1)

    add_collapse_marks(fig, series_list, x_max)
    apply_generation_xaxis(fig, n)

    # apply_nature_layout must get BOTH height and width -- write_figure()
    # now reads the figure's actual on-screen size straight off fig.layout
    # to compute the print-DPI export scale (see style.py). PANEL_WIDTH is
    # the same on-screen-width convention every other pooled-trajectory
    # figure uses (previously only implicit, via the height/900 aspect
    # ratio passed to write_figure -- now made explicit instead).
    height = max(420, PANEL_HEIGHT * n)
    apply_nature_layout(fig, height=height, width=PANEL_WIDTH)
    write_figure(fig, OUT_DIR, figdef["name"], panel_kind="multi_panel")


def main():
    print("Loading series...")
    series_list = load_all_series()
    if not series_list:
        raise SystemExit("No data found — check METRIC_SUMMARIES / "
                         "HALLUCINATION_SUMMARIES.")

    trtr_refs = compute_trtr_references(series_list)

    print("\nPer-metric-family trajectories:")
    for figdef in FIGURES:
        build_family_figure(figdef, series_list, trtr_refs)

    print("\nDone.")


if __name__ == "__main__":
    main()