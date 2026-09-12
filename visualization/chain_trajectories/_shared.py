"""
visualization/chain_trajectories/_shared.py

Shared plumbing for every chain-trajectory figure: chain-folder discovery,
seed-pair parsing, and the one figure builder (build_chain_figure) used by
every collector in this package. This consolidates what used to be four
near-duplicate scripts (PLOT3b_TSTR_AUROC_trajectories_chains.py,
PLOT3c_correlation_trajectories_chains.py, PLOT3d_chain_trajectories.py,
PLOT3f_chain_trajectories_HR_LLM.py) — their collect()/build_figure()/
scan_class_collapse() were ~95% identical, differing only in which
metric/column/data-source they pointed at. That difference is now the only
thing each collector module owns; everything else (the grid layout, chain
lines + bold mean, class-collapse marker, TRTR baseline, floor line, output
naming) lives here once.

Visual identity: color + marker = generator, shared with every other figure
in the manuscript via visualization/style.py. The legacy chain-trajectory
scripts had NO markers at all (color was the only channel) — every mean
line here now carries its generator's marker at intervals, and the generator
identity is otherwise identical to the pooled-trajectory figures.
"""

import glob
import os
import re
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
from config.pipeline_config import ROOT, RAW_DATA_ROOT, CHAIN_TRAJECTORIES_FIGURES
from visualization.style import (
    GENERATOR_COLORS, GENERATOR_MARKERS, BASELINE_COLOR, CLASS_COLLAPSE_COLOR,
    apply_nature_layout, write_figure,
)

# ============================================================
# SHARED CONFIG
# ============================================================

# Both data trees the chain scripts read from: TSTR JSONs and per-chain SD
# CSVs live under the raw repository export tree; the ID-based hallucination
# files and the decoded synthetic CSVs (readmission share) live under the
# evaluation-ready tree. Same two roots config/pipeline_config.py already
# names for the rest of the pipeline.
BASE = RAW_DATA_ROOT
EVALUATIONREADY_ROOT = ROOT

GENERATORS = {"ARF": "arf", "CTGAN": "ctgan", "DDPM": "ddpm", "RTVAE": "rtvae"}
GENERATOR_ORDER = ["ARF", "CTGAN", "DDPM", "RTVAE"]
VARIANTS = ["pfa", "pfp"]
VARIANT_LABELS = {"pfa": "full", "pfp": "core"}

GENERATIONS = list(range(0, 20))
EXPECTED_CHAINS = 15

# Fixed x-axis range for EVERY chain-trajectory panel, regardless of how far
# any individual chain's own data actually extends. Without this, plotly
# autoscales each panel's x-axis to the max generation actually present in
# that panel's data -- so a generator/metric where one or more chains lack
# late-generation rows (an export gap, a chain that failed and stopped being
# tracked, a metric with a min-coverage-per-generation cutoff -- see e.g.
# collect_hallucination.py's zero-fill-vs-drop rule) renders with an x-axis
# that silently ends before generation 19, panel by panel, rather than a
# consistent 0-19 range across every figure. This is the specific bug behind
# "some of them were cut short": the pooled-trajectory figures
# (visualization/pooled_trajectories/_shared.py's apply_generation_xaxis)
# already pin this range explicitly; this package never did.
GENERATION_X_RANGE = [GENERATIONS[0], GENERATIONS[-1]]

OUT_DIR = CHAIN_TRAJECTORIES_FIGURES

LAYOUT = "grid"
CHAIN_OPACITY = 0.35
CHAIN_WIDTH = 1.2
MEAN_WIDTH = 3.0
MEAN_MARKER_SIZE = 7
SHARED_Y = True
FIRST_OCCURRENCE_ONLY = True
VERTICAL_SPACING = 0.16
HORIZONTAL_SPACING = 0.10
PANEL_HEIGHT = 420
PANEL_WIDTH = 560
TOP_MARGIN = 110

LABEL_COLUMN = "readmission"
COLLAPSE_MIN_MINORITY = 0.0
SD_SUBDIR = "SD"


# ============================================================
# CHAIN / FOLDER DISCOVERY (raw repository tree: BASE)
# ============================================================

def find_chain_dirs(run_prefix, tok):
    export_dir = os.path.join(BASE, f"{run_prefix}_{tok}_export")
    if not os.path.isdir(export_dir):
        print(f"  [WARN] export folder not found: {export_dir}")
        return []
    pattern = os.path.join(export_dir, f"{run_prefix}_dseed*mseed*")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def chain_label(chain_dir_or_name):
    name = os.path.basename(chain_dir_or_name)
    d = re.search(r"dseed(\d+)", name)
    m = re.search(r"mseed(\d+)", name)
    if d and m:
        return f"d{d.group(1)} / m{m.group(1)}"
    return name


def seed_pair(folder_name):
    d = re.search(r"dseed(\d+)", folder_name)
    m = re.search(r"mseed(\d+)", folder_name)
    return (d.group(1), m.group(1)) if d and m else None


# ============================================================
# CLASS COLLAPSE — synthetic label balance (shared by every collector)
# ============================================================

def sd_generator_folder(generator_disp, variant):
    """Top-level SD_evaluationready folder for a generator+variant pair.

    The "pfp" (core feature set) chains live under a separate top-level
    "{generator_disp}_CORE" folder, not mixed in under "{generator_disp}"
    alongside the "pfa" (full feature set) chains -- e.g.
    Data/SD_evaluationready/ARF/ holds only Step7pfa_... chains, and
    Data/SD_evaluationready/ARF_CORE/ holds only Step7pfp_... chains."""
    return generator_disp if variant == "pfa" else f"{generator_disp}_CORE"


def index_sd_chains(evaluationready_root, generator_disp, run_prefix):
    """{(dseed, mseed): chain_dir} for one generator, under the
    evaluation-ready tree (Data/SD_evaluationready/{GEN}/{run_prefix}_...).

    generator_disp must already be resolved to the right top-level folder
    for the variant being scanned -- see sd_generator_folder()."""
    gen_base = os.path.join(evaluationready_root, generator_disp)
    index = {}
    if not os.path.isdir(gen_base):
        print(f"  [WARN] SD folder not found for {generator_disp}: {gen_base}")
        return index
    for name in os.listdir(gen_base):
        full = os.path.join(gen_base, name)
        if not os.path.isdir(full) or not name.startswith(run_prefix):
            continue
        pair = seed_pair(name)
        if pair:
            index[pair] = full
    return index


def find_sd_gen_csv(sd_chain_dir, g):
    sub = os.path.join(sd_chain_dir, SD_SUBDIR)
    if not os.path.isdir(sub):
        return None
    for fname in os.listdir(sub):
        if f"_gen_{g}_" in fname and fname.endswith(".csv"):
            return os.path.join(sub, fname)
    return None


def label_balance(csv_path):
    """(n_classes, minority_fraction) for LABEL_COLUMN, or (nan, nan)."""
    try:
        col = pd.read_csv(csv_path, usecols=[LABEL_COLUMN])[LABEL_COLUMN]
    except Exception:
        return np.nan, np.nan
    col = col.dropna()
    if col.empty:
        return 0, np.nan
    norm = col.astype(str).str.strip().str.lower().replace(
        {"1": "true", "0": "false", "yes": "true", "no": "false"})
    counts = norm.value_counts(normalize=True)
    return int(counts.size), float(counts.min())


def scan_class_collapse(variants=VARIANTS):
    """{(variant, generator, dseed, mseed, generation): (n_classes, minority_fraction)}"""
    print(f"\nScanning synthetic label balance ('{LABEL_COLUMN}') under {EVALUATIONREADY_ROOT}")
    out = {}
    for disp in GENERATOR_ORDER:
        for variant in variants:
            run_prefix = f"Step7{variant}"
            index = index_sd_chains(EVALUATIONREADY_ROOT, sd_generator_folder(disp, variant), run_prefix)
            for (dseed, mseed), chain_dir in index.items():
                for g in GENERATIONS:
                    fp = find_sd_gen_csv(chain_dir, g)
                    if fp is None:
                        continue
                    n_cls, minority = label_balance(fp)
                    out[(variant, disp, dseed, mseed, g)] = (n_cls, minority)
    return out


def is_collapsed(entry):
    if entry is None:
        return False
    n_cls, minority = entry
    if not np.isfinite(n_cls):
        return False
    if n_cls <= 1:
        return True
    if COLLAPSE_MIN_MINORITY > 0 and np.isfinite(minority):
        return bool(minority < COLLAPSE_MIN_MINORITY)
    return False


def attach_class_collapse(df, collapse_map, variant_col="variant"):
    df = df.copy()
    keys = list(zip(df[variant_col], df["generator"], df["dseed"].astype(str),
                     df["mseed"].astype(str), df["generation"]))
    entries = [collapse_map.get(k) for k in keys]
    df["n_classes"] = [e[0] if e else np.nan for e in entries]
    df["minority_fraction"] = [e[1] if e else np.nan for e in entries]
    df["class_collapse"] = [is_collapsed(e) for e in entries]
    return df


# ============================================================
# SHARED FIGURE BUILDER
# ============================================================

def first_per_chain(sub, flag_col):
    hits = sub[sub[flag_col].fillna(False)]
    if hits.empty or not FIRST_OCCURRENCE_ONLY:
        return hits
    idx = hits.groupby("chain")["generation"].idxmin()
    return hits.loc[idx].sort_values("generation")


def build_chain_figure(df, y_label, title, gens_present=None,
                        show_baseline=False, baseline_col="value_trtr",
                        baseline_name="TRTR (real-trained)",
                        show_floor=False, floor_value=None,
                        mark_low_auroc=False, low_auroc_col="low_auroc",
                        low_auroc_threshold=0.5, low_auroc_value_col="auroc_tstr",
                        mark_class_collapse=True):
    """
    Grid of one panel per generator: thin chain lines + bold marked mean,
    optional TRTR baseline / floor / no-skill ring / class-collapse ring.
    Generalizes build_figure() from PLOT3d/PLOT3f — the same layout, now
    parameterized instead of copy-pasted per script.
    """
    if gens_present is None:
        gens_present = [g for g in GENERATOR_ORDER if g in set(df["generator"])]

    n_cols = 2
    n_rows = int(np.ceil(len(gens_present) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_yaxes=SHARED_Y,
                        shared_xaxes=False, vertical_spacing=VERTICAL_SPACING,
                        horizontal_spacing=HORIZONTAL_SPACING,
                        subplot_titles=gens_present)

    for i, disp in enumerate(gens_present):
        row, col = i // n_cols + 1, i % n_cols + 1
        sub = df[df["generator"] == disp]
        color = GENERATOR_COLORS.get(disp, "#666666")
        marker = GENERATOR_MARKERS.get(disp, "circle")
        first_trace = True

        for label, chain_df in sub.groupby("chain"):
            chain_df = chain_df.sort_values("generation")
            fig.add_trace(go.Scatter(
                x=chain_df["generation"], y=chain_df["value"], mode="lines",
                name=disp, legendgroup=disp, showlegend=first_trace,
                opacity=CHAIN_OPACITY, line=dict(color=color, width=CHAIN_WIDTH),
                hovertemplate=(f"{disp} — {label}<br>gen %{{x}}<br>"
                               f"value %{{y:.4f}}<extra></extra>"),
            ), row=row, col=col)
            first_trace = False

        mean_df = sub.groupby("generation", as_index=False)["value"].mean().sort_values("generation")
        fig.add_trace(go.Scatter(
            x=mean_df["generation"], y=mean_df["value"], mode="lines+markers",
            name=f"{disp} mean", legendgroup=disp, showlegend=True,
            line=dict(color=color, width=MEAN_WIDTH),
            marker=dict(symbol=marker, size=MEAN_MARKER_SIZE, color=color),
            hovertemplate=f"{disp} mean<br>gen %{{x}}<br>value %{{y:.4f}}<extra></extra>",
        ), row=row, col=col)

        if show_baseline and baseline_col in sub.columns and sub[baseline_col].notna().any():
            base_df = sub.groupby("generation", as_index=False)[baseline_col].mean().sort_values("generation")
            fig.add_trace(go.Scatter(
                x=base_df["generation"], y=base_df[baseline_col], mode="lines",
                name=baseline_name, legendgroup="trtr", showlegend=(i == 0),
                line=dict(color=BASELINE_COLOR, width=2, dash="dash"),
                hovertemplate=f"{baseline_name}<br>gen %{{x}}<br>value %{{y:.4f}}<extra></extra>",
            ), row=row, col=col)

        if show_floor and floor_value is not None:
            fig.add_hline(y=floor_value, line=dict(color="#999999", width=1, dash="dot"),
                          row=row, col=col)

        if mark_low_auroc and low_auroc_col in sub.columns:
            low = first_per_chain(sub, low_auroc_col)
            if not low.empty:
                onset_label = "first " if FIRST_OCCURRENCE_ONLY else ""
                fig.add_trace(go.Scatter(
                    x=low["generation"], y=low["value"], mode="markers",
                    name=f"{onset_label}TSTR AUROC ≤ {low_auroc_threshold}",
                    legendgroup="low_auroc", showlegend=(i == 0),
                    marker=dict(color="rgba(0,0,0,0)", size=9,
                                line=dict(color="#d62728", width=1.5)),
                    customdata=low[[low_auroc_value_col]] if low_auroc_value_col in low.columns else None,
                    hovertemplate="no separability<br>gen %{x}<extra></extra>",
                ), row=row, col=col)

        if mark_class_collapse and "class_collapse" in sub.columns:
            coll = first_per_chain(sub, "class_collapse")
            if not coll.empty:
                onset_label = "first " if FIRST_OCCURRENCE_ONLY else ""
                fig.add_trace(go.Scatter(
                    x=coll["generation"], y=coll["value"], mode="markers",
                    name=f"{onset_label}'{LABEL_COLUMN}' class collapsed",
                    legendgroup="class_collapse", showlegend=(i == 0),
                    marker=dict(symbol="circle", color="rgba(0,0,0,0)", size=14,
                                line=dict(color=CLASS_COLLAPSE_COLOR, width=2)),
                    hovertemplate=(f"'{LABEL_COLUMN}' collapsed<br>gen %{{x}}<extra></extra>"),
                ), row=row, col=col)

        # range=GENERATION_X_RANGE is the x-axis-truncation fix -- every
        # panel now spans the same fixed 0-19, instead of autoscaling to
        # whatever generation this panel's own data happens to reach (see
        # GENERATION_X_RANGE's docstring above).
        fig.update_xaxes(title_text="Generation", dtick=2,
                         range=GENERATION_X_RANGE, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    n_rows_used = int(np.ceil(len(gens_present) / 2))
    height = PANEL_HEIGHT * n_rows_used + TOP_MARGIN
    width = PANEL_WIDTH * min(2, len(gens_present)) + 220

    fig.update_layout(height=height, width=width,
                      margin=dict(t=TOP_MARGIN, l=80, r=40, b=70))
    apply_nature_layout(fig, height=height, width=width)
    return fig, height, width


def save_chain_outputs(df, fig, out_stem, height, width):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{out_stem}.csv")
    df.sort_values(["generator", "chain", "generation"]).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    write_figure(fig, OUT_DIR, f"{out_stem}_{LAYOUT}", panel_kind="multi_panel")