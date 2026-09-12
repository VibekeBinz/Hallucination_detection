"""
visualization/population_features.py

Feature plots for the ICU population dataset — one page for the numeric
features (histogram + KDE) and one for the categorical features (bar chart
of value counts). Uses the same Plotly + Nature-layout pipeline as
frequency_pages.py: apply_nature_layout() for shared font/legend/template
settings, write_figure() for the self-contained HTML with camera-button PNG
export, and the shared style.py identity colors instead of locally chosen
hex codes.

Unlike frequency_pages.py, this script ALSO writes a real PNG to disk at
write time (via kaleido), reusing style.py's png_export_dims() so the PNG
lands at the same print DPI the HTML's camera button would produce. This is
local to this script only -- write_figure() itself is unchanged, so every
other figure family still stays HTML + manual camera-button export.

Reads:   {POPULATION_FILE}   (config.pipeline_config)
Writes:  {FREQUENCYPLOTS_FIGURES}/POP_numeric_features.html
         {FREQUENCYPLOTS_FIGURES}/POP_numeric_features.png
         {FREQUENCYPLOTS_FIGURES}/POP_categorical_features.html
         {FREQUENCYPLOTS_FIGURES}/POP_categorical_features.png

Nature Communications & AI compliance notes:
  - Font: FONT_FAMILY via style.py, same as every other figure family.
  - Combination art (density histograms, shaded KDE lines, bar charts)
    exported at DEFAULT_EXPORT_DPI (600), the same tier as frequency_pages.
  - Single-series content (one dataset, not a generator comparison), so
    REAL_COLOR is the one identity color used throughout, with
    BASELINE_COLOR for the median reference line -- both drawn from
    style.py's shared color roles rather than ad hoc hex codes. No legend
    is drawn (apply_nature_layout(..., legend=False)), since there's only
    one series to label.

Usage:
    python -m visualization.population_features
    python -m visualization.population_features /path/to/other_data.csv
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make the repo root importable regardless of how this script is launched
# (a plain `python .../population_features.py`, an IDE "Run" button, or
# `python -m visualization.frequency_pages.population_features` from the
# repo root -- only the last of those already has the repo root on
# sys.path).
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.append(_p)

from config.pipeline_config import POPULATION_FILE, FREQUENCYPLOTS_FIGURES
from visualization.style import (
    REAL_COLOR, BASELINE_COLOR, apply_nature_layout, write_figure,
    png_export_dims, FONT_FAMILY, FONT_COLOR, LINE_WIDTH_PRIMARY,
    LINE_WIDTH_REFERENCE,
)

# Lettering size for this figure family only, applied after
# apply_nature_layout() so it isn't overwritten by that call's own
# font-size pass -- same pattern and same value (15) as frequency_pages.py,
# since both figure families pack a similarly dense grid of panels onto one
# page and read smaller than a single-panel figure at style.py's shared
# defaults.
POP_FONT_SIZE_AXIS_TITLE = 15
POP_FONT_SIZE_TICK = 15
POP_FONT_SIZE_ANNOTATION = 15

# KDE smoothing / axis-clipping constants -- same values as frequency_pages.py
# so a reader sees the same smoothing behavior across every distribution plot
# in the manuscript.
CLIP_PCTILE = (0.5, 99.5)
KDE_BW_SCALE = 1.5
N_GRID = 400

# ---- Column definitions -----------------------------------------------------
NUMERIC = [
    "blood_urea_nitro", "glucose", "sodium", "respiratory_rate", "heartrate",
    "systolic_bp", "diastolic_bp", "spo2", "creatinine", "potassium",
    "hemoglobin", "age",
]

CATEGORICAL = [
    "readmission", "prior_icu", "gender", "icd9", "ethnicity",
    "admission_type", "first_careunit",
]
# nt-probnp / cholesterol / albumin are often stored as categorical bins
# (e.g. Normal / Low / Lowrisk). Add them to whichever list matches your data.
MAYBE_CATEGORICAL = ["nt-probnp", "cholesterol", "albumin"]

# Bar charts get unreadable past this many categories; we keep the top-N.
TOP_N_CATEGORIES = 20

OUT_DIR = FREQUENCYPLOTS_FIGURES


def _grid(n, ncols):
    """Return (nrows, ncols) for n panels."""
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def safe_kde(values, grid):
    v = pd.to_numeric(values, errors="coerce").dropna().values
    if len(v) < 5 or np.ptp(v) == 0:
        return None
    try:
        kde = gaussian_kde(v)
        kde.set_bandwidth(bw_method=kde.factor * KDE_BW_SCALE)
        return kde(grid)
    except Exception:
        return None


def _bump_fonts(fig):
    """Same local font-size bump pattern as frequency_pages.py's run_feature()."""
    fig.update_xaxes(title_font=dict(family=FONT_FAMILY, size=POP_FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=POP_FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_yaxes(title_font=dict(family=FONT_FAMILY, size=POP_FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=POP_FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_annotations(font=dict(family=FONT_FAMILY, size=POP_FONT_SIZE_ANNOTATION,
                                      color=FONT_COLOR))


def _write_png(fig, out_dir, name, panel_kind="multi_panel"):
    """Real PNG on disk, local to this script only -- frequency_pages.py and
    every other figure family stay HTML + manual camera-button export, since
    write_figure() itself in style.py is untouched. Reuses png_export_dims()
    so this PNG lands at the same print DPI the HTML's camera button would
    produce; requires the kaleido package."""
    if fig.layout.width is None or fig.layout.height is None:
        raise ValueError(f"_write_png({name!r}): fig.layout.width/height are not set.")

    width_px, height_px, scale = png_export_dims(
        fig.layout.width, fig.layout.height, panel_kind=panel_kind
    )
    path = os.path.join(out_dir, f"{name}.png")
    try:
        fig.write_image(path, format="png", width=width_px, height=height_px, scale=scale)
    except Exception as e:
        print(f"  [WARN] PNG export failed for {name} ({e}). "
              f"Install/upgrade kaleido: pip install -U kaleido")
        return
    print(f"  Saved: {path}")


def plot_numeric(df, cols, name):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("No numeric columns found.")
        return
    ncols = 3
    nrows, ncols = _grid(len(cols), ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                         horizontal_spacing=0.06, vertical_spacing=0.10)

    for i, col in enumerate(cols):
        row_index = i // ncols + 1
        col_index = i % ncols + 1

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        lo, hi = np.percentile(s, CLIP_PCTILE)
        if hi <= lo:
            lo, hi = float(s.min()), float(s.max())
            if hi <= lo:
                hi = lo + 1.0

        fig.add_trace(
            go.Histogram(x=s, histnorm="probability density", opacity=0.30,
                         marker=dict(color=REAL_COLOR), nbinsx=40,
                         showlegend=False),
            row=row_index, col=col_index,
        )

        grid = np.linspace(lo, hi, N_GRID)
        ky = safe_kde(s, grid)
        if ky is not None:
            fig.add_trace(
                go.Scatter(x=grid, y=ky, mode="lines",
                           line=dict(color=REAL_COLOR, width=LINE_WIDTH_PRIMARY),
                           showlegend=False),
                row=row_index, col=col_index,
            )

        med = s.median()
        fig.add_vline(x=med, line_width=LINE_WIDTH_REFERENCE, line_dash="dash",
                      line_color=BASELINE_COLOR, row=row_index, col=col_index)
        fig.add_annotation(
            x=0.97, y=0.95, xref="x domain", yref="y domain",
            text=f"n={len(s)}<br>med={med:.1f}",
            showarrow=False, align="right", xanchor="right", yanchor="top",
            bgcolor="white", bordercolor="#B3B3B3", borderwidth=1,
            row=row_index, col=col_index,
        )
        fig.update_xaxes(range=[lo, hi], row=row_index, col=col_index)
        fig.update_yaxes(title_text="Density", row=row_index, col=col_index)

    width = ncols * 450
    height = nrows * 320
    apply_nature_layout(fig, height=height, width=width, legend=False)
    _bump_fonts(fig)
    write_figure(fig, OUT_DIR, name, panel_kind="multi_panel")
    _write_png(fig, OUT_DIR, name, panel_kind="multi_panel")


def plot_categorical(df, cols, name, top_n=TOP_N_CATEGORIES):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("No categorical columns found.")
        return
    ncols = 2
    nrows, ncols = _grid(len(cols), ncols)

    titles = []
    panel_data = []
    for col in cols:
        vc = df[col].astype("string").fillna("<NA>").value_counts()
        truncated = len(vc) > top_n
        vc = vc.head(top_n)
        title = col + (f"  (top {top_n} of {df[col].nunique(dropna=False)})" if truncated else "")
        titles.append(title)
        panel_data.append(vc)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                         horizontal_spacing=0.15, vertical_spacing=0.10)

    for i, vc in enumerate(panel_data):
        row_index = i // ncols + 1
        col_index = i % ncols + 1
        fig.add_trace(
            go.Bar(x=vc.values, y=vc.index, orientation="h",
                   marker=dict(color=REAL_COLOR, opacity=0.85),
                   showlegend=False),
            row=row_index, col=col_index,
        )
        fig.update_xaxes(title_text="count", row=row_index, col=col_index)
        fig.update_yaxes(autorange="reversed", row=row_index, col=col_index)

    width = ncols * 600
    height = nrows * 340
    apply_nature_layout(fig, height=height, width=width, legend=False)
    _bump_fonts(fig)
    write_figure(fig, OUT_DIR, name, panel_kind="multi_panel")
    _write_png(fig, OUT_DIR, name, panel_kind="multi_panel")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else POPULATION_FILE
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} cols")

    cats = list(CATEGORICAL)
    nums = list(NUMERIC)
    for c in MAYBE_CATEGORICAL:
        if c not in df.columns:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() > 0.8:
            nums.append(c)
        else:
            cats.append(c)

    plot_numeric(df, nums, "POP_numeric_features")
    plot_categorical(df, cats, "POP_categorical_features")


if __name__ == "__main__":
    main()