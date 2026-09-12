"""
visualization/pooled_trajectories/bin_resolution.py

HR (solid) vs HR* (dotted) against bin resolution, mean over all 20
generations. Full dataset and core dataset are drawn as two panels of one
figure, full on the left and core on the right, sharing a single y-axis.

Writes {MISC_FIGURES}/hallucination_rate_vs_bin_resolution.html
"""

import os
import sys

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
from config.pipeline_config import HALLUCINATION_SUMMARIES, MISC_FIGURES
from visualization.style import (
    GENERATOR_COLORS, GENERATOR_MARKERS, apply_nature_layout, write_figure,
)

from visualization.pooled_trajectories._shared import (
    pretty, ROOT_LABEL, ROOT_SUFFIX, HR_FIGURE_SCALE, N_GENERATIONS,
)

OUT_DIR = MISC_FIGURES
OUT_NAME = "hallucination_rate_vs_bin_resolution"
GENERATORS = ["CTGAN", "ARF", "RTVAE", "DDPM"]

BIN_RES_METRICS = [
    ("Pilgram_HR",  "solid"),
    ("HR_adjusted", "dot"),
]

# (root, subplot column) -- full dataset in column 1 (left), core dataset in
# column 2 (right).
ROOT_PANELS = [("Step7pfa", 1), ("Step7pfp", 2)]

FIG_HEIGHT = 460
FIG_WIDTH = 1000   # total figure width, both panels plus the shared legend


def add_root_panel(fig, root, col, showlegend):
    """
    Adds one panel's traces (mean over all 20 generations of {metric}_mean,
    per bin resolution, for one root) to `fig` at column `col`. HR solid,
    HR* dotted, both in the generator's colour and marker. Pooling across
    generations mixes early and collapsed generations by design. Returns
    the set of bin values seen in this panel, so the caller can set that
    panel's own x tick values.
    """
    bins_seen = set()
    seen_groups = set()

    for gen in GENERATORS:
        name = f"merged_hallucination_summary_{gen}{ROOT_SUFFIX[root]}.csv"
        path = os.path.join(HALLUCINATION_SUMMARIES, name)
        if not os.path.exists(path):
            print(f"  [WARN] missing hallucination summary: {path}")
            continue

        df = pd.read_csv(path)

        for metric, dash in BIN_RES_METRICS:
            col_name = f"{metric}_mean"
            if col_name not in df.columns:
                print(f"  [WARN] '{col_name}' not in {name}")
                continue

            grouped = (df.groupby("bin")[col_name].mean().sort_index()
                       * HR_FIGURE_SCALE)
            bins_seen.update(grouped.index.tolist())

            group = f"{gen}_{metric}"
            fig.add_trace(go.Scatter(
                x=grouped.index.values, y=grouped.values,
                mode="lines+markers",
                name=f"{gen} — {pretty(metric)}",
                legendgroup=group,
                showlegend=showlegend and group not in seen_groups,
                line=dict(color=GENERATOR_COLORS[gen], dash=dash, width=2),
                marker=dict(symbol=GENERATOR_MARKERS[gen], size=8,
                            color=GENERATOR_COLORS[gen]),
            ), row=1, col=col)
            seen_groups.add(group)

    if not bins_seen:
        print(f"  [WARN] bin resolution panel ({ROOT_LABEL[root]}): no data")
    return bins_seen


def build_bin_resolution_figure():
    titles = [f"{ROOT_LABEL[root].capitalize()} dataset" for root, _ in ROOT_PANELS]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                         subplot_titles=titles, horizontal_spacing=0.06)

    any_data = False
    for root, col in ROOT_PANELS:
        bins_seen = add_root_panel(fig, root, col, showlegend=(col == 1))
        if bins_seen:
            any_data = True
            fig.update_xaxes(title_text="Bin resolution", tickmode="array",
                             tickvals=sorted(bins_seen), ticks="outside",
                             ticklen=4, row=1, col=col)

    if not any_data:
        print("  [SKIP] bin resolution figure: no data for either root")
        return

    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=1)

    # apply_nature_layout must get BOTH height and width -- write_figure()
    # reads the figure's actual on-screen size straight off fig.layout to
    # compute the print-DPI export scale (see style.py).
    apply_nature_layout(fig, height=FIG_HEIGHT, width=FIG_WIDTH)
    write_figure(fig, OUT_DIR, OUT_NAME, panel_kind="multi_panel")


def main():
    print("\nBin resolution:")
    build_bin_resolution_figure()
    print("\nDone.")


if __name__ == "__main__":
    main()