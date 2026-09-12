"""
visualization/frequency_pages.py
(rebuild + merge of PLOT2_frequencyplots_pooled_singlefeature.py and its
driver, PLOT2_allfeatures_with_PLOT_frequencyplots_pooled.py)

Real-vs-synthetic frequency/KDE grid, one page per feature: rows = anchor
generations, columns = generators. Same data reading and figure layout as
the legacy pair of scripts (which only differed by "loop every feature" vs
"run one feature"); merged here into one module with a single run_feature()
plus a main() that loops every feature, so there is one script instead of
two, matching how every other rebuilt figure script in this package works.

Reads:  {ROOT}/{GENERATOR}/{RUN_PREFIX}*/RD/*.csv   (real, pooled across runs)
        {ROOT}/{GENERATOR}/{RUN_PREFIX}*/SD/*_gen_{g}_*.csv   (synthetic)
Writes: {FREQUENCYPLOTS_FIGURES}/{feature}_trajectory_multicol.html
        (self-contained HTML, camera-button PNG export)

Nature Communications & AI compliance notes:
  - Real vs Synthetic was previously color-only (blue vs green). Color is
    kept (it's still the fastest cue on screen) but is no longer the only
    channel: KDE lines are also solid (Real) vs dashed (Synthetic), and
    histogram bars carry a diagonal hatch on the Synthetic series, so the
    real/synthetic split survives grayscale printing.
  - Arial/Helvetica lettering, no baked-in figure title, legend included
    (static, non-interactive) per the shared visualization/style.py layout.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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
from config.pipeline_config import ROOT, FREQUENCYPLOTS_FIGURES
from visualization.style import (
    REAL_COLOR, SYNTHETIC_COLOR, apply_nature_layout, write_figure,
    FONT_FAMILY, FONT_COLOR,
)

# Lettering size for this figure family only, applied after
# apply_nature_layout() so it isn't overwritten by that call's own
# font-size pass. Bigger than style.py's shared manuscript-wide defaults
# (12px everywhere) -- these grids pack up to 4 columns x 4 rows of panels
# onto one page, so the shared size reads noticeably smaller here than on
# a single-panel or stacked-panel figure. Changed here rather than in
# style.py so every other figure in the manuscript keeps its current size.
FREQ_FONT_SIZE_AXIS_TITLE = 15
FREQ_FONT_SIZE_TICK = 15
FREQ_FONT_SIZE_ANNOTATION = 15
FREQ_FONT_SIZE_LEGEND = 15

# ============================================================
# CONFIG
# ============================================================

BASE = ROOT
GENERATORS = ["ARF", "CTGAN", "DDPM", "RTVAE"]
OUT_DIR = FREQUENCYPLOTS_FIGURES

GENS_TO_PLOT = [0, 3, 5, 19]
RUN_PREFIX = "Step7pfa"

NUMERIC_FEATURES = [
    "age", "blood_urea_nitro", "glucose", "sodium", "respiratory_rate",
    "heartrate", "systolic_bp", "diastolic_bp", "spo2", "creatinine",
    "potassium", "hemoglobin",
]

CATEGORICAL_FEATURES = [
    "readmission", "prior_icu", "gender", "icd9", "ethnicity",
    "admission_type", "first_careunit", "bmi", "nt-probnp",
    "cholesterol", "albumin",
]

CLIP_PCTILE = (0.5, 99.5)
BARS = 100
KDE_BW_SCALE = 1.5
N_GRID = 400

SYNTHETIC_PATTERN = "/"   # diagonal hatch -- the second (color-independent)
                          # channel that separates Real from Synthetic


def _marker(color, pattern_shape="", opacity=None):
    """A marker dict with an optional pattern fill. Plotly's Pattern object
    has no 'color' property of its own (fgcolor/bgcolor only), so the plain
    'no pattern' case must omit the `pattern` key entirely rather than pass
    it an empty/invalid dict."""
    m = dict(color=color)
    if opacity is not None:
        m["opacity"] = opacity
    if pattern_shape:
        m["pattern"] = dict(shape=pattern_shape, fgcolor=color, bgcolor=color,
                            fillmode="overlay")
    return m

# ============================================================
# HELPERS -- unchanged from the legacy script
# ============================================================

def find_gen_file(sd_folder, gen):
    if not os.path.isdir(sd_folder):
        return None
    for fname in os.listdir(sd_folder):
        if f"_gen_{gen}_" in fname and fname.endswith(".csv"):
            return os.path.join(sd_folder, fname)
    return None


def pooled_feature_values(folders, subdir, feature, gen=None):
    vals = []
    for run_folder in folders:
        sub = os.path.join(run_folder, subdir)
        if not os.path.isdir(sub):
            continue
        if gen is None:
            files = [os.path.join(sub, f) for f in os.listdir(sub)
                     if f.endswith(".csv")]
        else:
            fp = find_gen_file(sub, gen)
            files = [fp] if fp else []
        for fp in files:
            try:
                col = pd.read_csv(fp, usecols=[feature])[feature]
                vals.extend(col.dropna().tolist())
            except ValueError:
                pass
            except Exception as e:
                print(f"    [WARN] error reading {os.path.basename(fp)}: {e}")
    return pd.Series(vals)


def resolve_is_numeric(feature, sample):
    if feature in NUMERIC_FEATURES:
        num = pd.to_numeric(sample, errors="coerce")
        if num.notna().mean() < 0.8:
            print(f"  [WARN] '{feature}' is in NUMERIC_FEATURES but <80% numeric — treating as categorical")
            return False
        return True
    if feature in CATEGORICAL_FEATURES:
        return False
    num = pd.to_numeric(sample, errors="coerce")
    inferred = num.notna().mean() >= 0.8 and sample.nunique() > 15
    print(f"  [WARN] '{feature}' not in lists — inferred {'numeric' if inferred else 'categorical'}")
    return inferred


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


# ============================================================
# MAIN — one page per feature
# ============================================================

def run_feature(feature):
    print(f"\n=== Frequency page for FEATURE = {feature} ===")

    fig = make_subplots(
        rows=len(GENS_TO_PLOT),
        cols=len(GENERATORS),
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
        subplot_titles=[
            f"{gen_name} — Gen {g}"
            for g in GENS_TO_PLOT
            for gen_name in GENERATORS
        ],
    )

    for col_index, gen_name in enumerate(GENERATORS, start=1):
        print(f"\nProcessing generator: {gen_name}")

        gen_base = os.path.join(BASE, gen_name)
        if not os.path.isdir(gen_base):
            print(f"  [WARN] Folder not found: {gen_base}")
            continue

        run_folders = [
            os.path.join(gen_base, f)
            for f in os.listdir(gen_base)
            if os.path.isdir(os.path.join(gen_base, f))
            and f.startswith(RUN_PREFIX)
        ]
        if not run_folders:
            print(f"  [WARN] No run folders (prefix '{RUN_PREFIX}') for {gen_name}")
            continue

        real_vals = pooled_feature_values(run_folders, "RD", feature)
        if real_vals.empty:
            print(f"  [WARN] No real values for '{feature}' — skipping {gen_name}")
            continue

        is_numeric = resolve_is_numeric(feature, real_vals)

        syn_by_gen = {
            g: pooled_feature_values(run_folders, "SD", feature, gen=g)
            for g in GENS_TO_PLOT
        }

        if is_numeric:
            real_num = pd.to_numeric(real_vals, errors="coerce").dropna()
            syn_all = pd.concat(
                [pd.to_numeric(v, errors="coerce").dropna()
                 for v in syn_by_gen.values()]
            ) if syn_by_gen else pd.Series(dtype=float)

            lo_r, hi_r = np.percentile(real_num, CLIP_PCTILE)
            if len(syn_all):
                lo_s, hi_s = np.percentile(syn_all, CLIP_PCTILE)
                lo, hi = min(lo_r, lo_s), max(hi_r, hi_s)
            else:
                lo, hi = lo_r, hi_r
            if hi <= lo:
                pool = pd.concat([real_num, syn_all]) if len(syn_all) else real_num
                lo, hi = float(pool.min()), float(pool.max())
                if hi <= lo:
                    hi = lo + 1.0

        for row_index, g in enumerate(GENS_TO_PLOT, start=1):
            syn_vals = syn_by_gen[g]
            if syn_vals.empty:
                print(f"  [WARN] No synthetic values for gen {g}")
                continue

            first = (row_index == 1)

            if is_numeric:
                syn_num = pd.to_numeric(syn_vals, errors="coerce").dropna()
                real_num = pd.to_numeric(real_vals, errors="coerce").dropna()
                x_grid = np.linspace(lo, hi, N_GRID)

                for vals, name, color, pattern in [
                    (real_num, "Real", REAL_COLOR, ""),
                    (syn_num, "Synthetic", SYNTHETIC_COLOR, SYNTHETIC_PATTERN),
                ]:
                    fig.add_trace(
                        go.Histogram(
                            x=vals,
                            name=name,
                            histnorm="probability density",
                            opacity=0.30,
                            marker=_marker(color, pattern),
                            nbinsx=BARS,
                            showlegend=(first and col_index == 1),
                        ),
                        row=row_index,
                        col=col_index,
                    )

                for vals, name, color, dash in [
                    (real_num, "Real KDE", REAL_COLOR, "solid"),
                    (syn_num, "Synthetic KDE", SYNTHETIC_COLOR, "dash"),
                ]:
                    ky = safe_kde(vals, x_grid)
                    if ky is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=x_grid,
                                y=ky,
                                mode="lines",
                                line=dict(color=color, width=2, dash=dash),
                                name=name,
                                showlegend=(first and col_index == 1),
                            ),
                            row=row_index,
                            col=col_index,
                        )

                fig.update_xaxes(range=[lo, hi], row=row_index, col=col_index)
                fig.update_yaxes(title_text="Density", row=row_index, col=col_index)

            else:
                real_counts = real_vals.astype(str).value_counts(normalize=True)
                syn_counts = syn_vals.astype(str).value_counts(normalize=True)
                categories = sorted(set(real_counts.index) | set(syn_counts.index))

                for counts, name, color, pattern in [
                    (real_counts, "Real", REAL_COLOR, ""),
                    (syn_counts, "Synthetic", SYNTHETIC_COLOR, SYNTHETIC_PATTERN),
                ]:
                    fig.add_trace(
                        go.Bar(
                            x=categories,
                            y=[counts.get(c, 0) for c in categories],
                            name=name,
                            marker=_marker(color, pattern, opacity=0.6),
                            showlegend=(first and col_index == 1),
                        ),
                        row=row_index,
                        col=col_index,
                    )

                fig.update_yaxes(title_text="Proportion", row=row_index, col=col_index)

    fig.update_layout(
        height=len(GENS_TO_PLOT) * 300,
        width=len(GENERATORS) * 450,
        barmode="overlay",
    )
    apply_nature_layout(fig, height=len(GENS_TO_PLOT) * 300,
                        width=len(GENERATORS) * 450)

    # Bump lettering beyond style.py's shared default for this figure only
    # -- see FREQ_FONT_SIZE_* above. Font size here is in on-screen CSS px,
    # same as apply_nature_layout uses; write_figure()'s PNG export
    # re-rasterizes this same layout at higher pixel density (see
    # png_export_dims's docstring in style.py), so a bigger on-screen font
    # here comes out proportionally bigger in the exported PNG too.
    fig.update_xaxes(title_font=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_yaxes(title_font=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_annotations(font=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_ANNOTATION,
                                      color=FONT_COLOR))
    fig.update_layout(legend=dict(font=dict(family=FONT_FAMILY, size=FREQ_FONT_SIZE_LEGEND,
                                             color=FONT_COLOR)))

    name = f"{feature}_trajectory_multicol"
    write_figure(fig, OUT_DIR, name, panel_kind="multi_panel")


def main():
    failed = []
    all_features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    for i, feature in enumerate(all_features, start=1):
        print(f"\n{'='*60}\n[{i}/{len(all_features)}] FEATURE = {feature}\n{'='*60}")
        try:
            run_feature(feature)
        except Exception:
            import traceback
            print(f"  FAILED on '{feature}':")
            traceback.print_exc()
            failed.append(feature)

    print(f"\n{'='*60}\nAll done — {len(all_features) - len(failed)}/"
          f"{len(all_features)} features succeeded.")
    if failed:
        print("Failed features:", failed)


if __name__ == "__main__":
    main()