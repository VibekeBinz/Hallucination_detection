"""
visualization/expert_rule_composition.py

Expert rule composition — one stacked bar per (generator, anchor
generation). Eight bars: four generators x {generation 0, generation 19},
each normalised to 100%, showing which expert rules account for the flags
raised and how that mix narrows as recursive generation degrades the data.

Reads:
  {ROOT}/{GEN}/results/step2_rule_summary_{GEN}.csv
  columns: generation, rule, mean, low, high, sd, n

Writes:
  {MISC_FIGURES}/expert_rule_composition.html
  {MISC_FIGURES}/expert_rule_composition.csv

Full variants only — the CORE datasets carry 13 variables, so 18 of the 36
rules can never fire there; a composition comparison would be measuring the
variable set rather than the data.

Lettering, line weight, legend:
  - The impossible/suspicious buckets were previously color-only (a blue-red
    ramp vs a yellow-green ramp). Both buckets now also carry a bar pattern
    (diagonal hatch for impossible, dot hatch for suspicious, solid for
    "Other") so the bucket split survives grayscale printing, not just the
    lightness gradient within each ramp.
  - Arial/Helvetica lettering, no baked-in figure title, legend included
    (static) via visualization/style.py.
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "config")):
    _next = os.path.dirname(_p)
    if _next == _p:
        raise RuntimeError("repo root not found (no 'config' folder above this file)")
    _p = _next
if _p not in sys.path:
    sys.path.insert(0, _p)
from config.pipeline_config import ROOT, MISC_FIGURES
from visualization.style import apply_nature_layout, write_figure

# ============================================================
# CONFIG
# ============================================================

RULE_BASE = ROOT           # Data/SD_evaluationready -- same root every other
RULE_SUBDIR = "results"     # rebuilt script in this package reads from

OUT_DIR = MISC_FIGURES
OUT_NAME = "expert_rule_composition"

GENERATORS = ["CTGAN", "ARF", "RTVAE", "DDPM"]
ANCHOR_GENERATIONS = [0, 19]

N_ROWS = 10_000

SEGMENT_MIN_SHARE_PCT = 5.0
MAX_SEGMENTS = 12
LARGEST_AT_BOTTOM = True

BAR_WIDTH = 0.62
PAIR_SPACING = 0.70
GROUP_SPACING = 2.20

FIG_HEIGHT = 620
FIG_WIDTH = 1100
MIN_LABEL_PCT = 3.0

IMPOSSIBLE_RULES = {
    "reality_limit", "bad_category", "dbp_ge_sbp", "pp_zero",
    "buncr_ratio_impossible", "rr_spo2_impossible",
    "fatal_constellation_impossible", "preg_male", "preg_age",
    "nut_impossible",
}

IMPOSSIBLE_ENDS = ("#1f4e9c", "#c0183c")   # blue -> red
SUSPICIOUS_ENDS = ("#f2c230", "#157f3c")   # yellow -> green
OTHER_COLOR = "#bdbdbd"

# Bar patterns, on top of the color ramps above -- the second, color-
# independent channel that separates the two buckets under grayscale.
IMPOSSIBLE_PATTERN = "/"     # diagonal hatch
SUSPICIOUS_PATTERN = "."     # dot hatch
OTHER_PATTERN = ""           # solid


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp(start_hex, end_hex, n):
    if n <= 0:
        return []
    if n == 1:
        return [start_hex]
    a, b = _hex_to_rgb(start_hex), _hex_to_rgb(end_hex)
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append("#%02x%02x%02x" % tuple(
            round(a[c] + (b[c] - a[c]) * t) for c in range(3)))
    return out


PRETTY_RULE = {
    "dbp_ge_sbp":             "Diastolic ≥ systolic BP",
    "pp_zero":                "Pulse pressure ≤ 2",
    "pp_low":                 "Pulse pressure < 5, normal vitals",
    "pp_wide":                "Wide pulse pressure",
    "pp_htn_narrow":          "Hypertension, narrow pulse pressure",
    "buncr_ratio_impossible": "BUN/Cr < 1.0",
    "buncr_ratio_high":       "BUN/Cr > 100",
    "low_cr":                 "Creatinine implausibly low",
    "severe_hypok":           "Severe hypokalaemia, normal HR",
    "comp_anemia":            "Anaemia without compensation",
    "comp_shock":             "Shock without compensation",
    "comp_hyperk":            "Hyperkalaemia, normal HR",
    "comp_hypona":            "Hyponatraemia with shock",
    "alb_obese_anemia":       "Low albumin, high BMI, anaemia",
    "alb_high_underwt":       "High albumin, underweight",
    "alb_high_wasting":       "High albumin, wasting diagnosis",
    "nut_obese_hypo":         "Obese, hypotensive, low albumin",
    "elderly_f_highhb":       "Elderly female, extreme haemoglobin",
    "circ_nomarkers":         "Cardiac diagnosis, no markers",
    "unknown_extremelabs":    "UNKNOWN diagnosis, extreme labs",
    "pair_cardiogenic":       "Bradycardia + hypotension",
    "pair_htn_brady":         "Hypertension + bradycardia",
    "pair_tachy_htn":         "Tachycardia + hypertension",
    "preg_male":              "Pregnancy, male",
    "preg_age":               "Pregnancy, age out of range",
    "preg_hb":                "Pregnancy, haemoglobin > 15",
    "pregnancy_elective":     "Pregnancy, elective admission",
    "extreme_count":          "≥ 5 extreme features",
    "reality_limit":          "Outside measurable range",
    "bad_category":           "Category not in allowed set",
    "rr_spo2_impossible":     "Agonal RR with normal SpO₂",
    "fatal_constellation_impossible": "Fatal constellation, vitals pristine",
    "fatal_constellation_suspicious": "Fatal constellation",
}


def pretty_rule(rule):
    return PRETTY_RULE.get(rule, rule.replace("_", " "))


# ============================================================
# LOADING
# ============================================================

def rule_summary_path(gen):
    return os.path.join(RULE_BASE, gen, RULE_SUBDIR, f"step2_rule_summary_{gen}.csv")


def load_rule_summaries():
    frames = []
    for gen in GENERATORS:
        path = rule_summary_path(gen)
        if not os.path.exists(path):
            print(f"  [WARN] missing rule summary: {path}")
            continue
        df = pd.read_csv(path)
        missing = {"generation", "rule", "mean"} - set(df.columns)
        if missing:
            print(f"  [WARN] {path} missing columns {sorted(missing)} — skipped")
            continue
        df = df[df["generation"].isin(ANCHOR_GENERATIONS)].copy()
        if df.empty:
            continue
        df["generator"] = gen
        df["pct"] = df["mean"].astype(float) / N_ROWS * 100
        frames.append(df[["generator", "generation", "rule", "pct"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# COMPOSITION
# ============================================================

def bar_shares(long_df):
    shares = {}
    for gen in GENERATORS:
        for g in ANCHOR_GENERATIONS:
            sub = long_df[(long_df["generator"] == gen) & (long_df["generation"] == g)]
            if sub.empty:
                continue
            by_rule = sub.groupby("rule")["pct"].sum()
            total = by_rule.sum()
            if total > 0:
                shares[(gen, g)] = by_rule / total * 100
    return shares


def choose_segments(long_df):
    shares = bar_shares(long_df)
    peak = pd.DataFrame(shares).max(axis=1).fillna(0).sort_values(ascending=False)
    totals = long_df.groupby("rule")["pct"].sum()
    named = [r for r in peak.index if peak[r] >= SEGMENT_MIN_SHARE_PCT][:MAX_SEGMENTS]
    return named, totals, peak


def build_composition(long_df, segments):
    rows = []
    for gen in GENERATORS:
        for g in ANCHOR_GENERATIONS:
            sub = long_df[(long_df["generator"] == gen) & (long_df["generation"] == g)]
            if sub.empty:
                continue
            by_rule = sub.groupby("rule")["pct"].sum()
            total = by_rule.sum()
            if total <= 0:
                continue
            row = {"generator": gen, "generation": g, "total_pct": total,
                   "n_rules_fired": int((by_rule > 0).sum())}
            named = 0.0
            for seg in segments:
                share = by_rule.get(seg, 0.0) / total * 100
                row[seg] = share
                named += share
            row["Other"] = max(0.0, 100.0 - named)
            rows.append(row)
    return pd.DataFrame(rows)


def segment_colors(segments):
    imp = [r for r in segments if r in IMPOSSIBLE_RULES]
    sus = [r for r in segments if r not in IMPOSSIBLE_RULES]
    colors = {}
    colors.update(dict(zip(imp, ramp(*IMPOSSIBLE_ENDS, len(imp)))))
    colors.update(dict(zip(sus, ramp(*SUSPICIOUS_ENDS, len(sus)))))
    colors["Other"] = OTHER_COLOR
    return colors


def segment_pattern(seg):
    if seg == "Other":
        return OTHER_PATTERN
    return IMPOSSIBLE_PATTERN if seg in IMPOSSIBLE_RULES else SUSPICIOUS_PATTERN


# ============================================================
# FIGURE
# ============================================================

def bar_positions(comp):
    xs, group_centres = [], {}
    for i, gen in enumerate(GENERATORS):
        bars = comp.index[comp["generator"] == gen].tolist()
        if not bars:
            continue
        base = i * GROUP_SPACING
        pos = [base + j * PAIR_SPACING for j in range(len(bars))]
        for b, x in zip(bars, pos):
            xs.append((b, x))
        group_centres[gen] = sum(pos) / len(pos)
    order = dict(xs)
    return [order[i] for i in comp.index], group_centres


def build_figure(comp, segments, n_other_rules):
    colors = segment_colors(segments)
    comp = comp.reset_index(drop=True)
    x, group_centres = bar_positions(comp)

    stack = segments + ["Other"]
    if not LARGEST_AT_BOTTOM:
        stack = stack[::-1]

    fig = go.Figure()
    for seg in stack:
        if seg == "Other":
            label = f"Other ({n_other_rules} rules, each < {SEGMENT_MIN_SHARE_PCT:.0f}%)"
        else:
            bucket = "impossible" if seg in IMPOSSIBLE_RULES else "suspicious"
            label = f"{pretty_rule(seg)}  ({bucket})"
        vals = comp[seg].tolist()
        pattern_shape = segment_pattern(seg)
        fig.add_trace(go.Bar(
            x=x, y=vals, width=BAR_WIDTH, name=label, marker=dict(
                color=colors[seg],
                pattern=dict(shape=pattern_shape, fgcolor="#ffffff", size=6, solidity=0.3)
                if pattern_shape else dict(),
            ),
            text=[f"{v:.0f}" if v >= MIN_LABEL_PCT else "" for v in vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"{label}<br>%{{y:.2f}}% of firings<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=[104] * len(comp), mode="text",
        text=[f"Σ {r.total_pct:.1f}%<br>"
              f"<span style='font-size:9px'>{r.n_rules_fired} rules</span>"
              for r in comp.itertuples()],
        textposition="middle center", textfont=dict(size=10, color="#444"),
        showlegend=False, hoverinfo="skip", cliponaxis=False,
    ))

    for gen, cx in group_centres.items():
        fig.add_annotation(x=cx, y=-0.085, xref="x", yref="paper", text=gen,
                           showarrow=False, font=dict(size=13, color="#222"))

    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="% of rule firings", range=[0, 112],
                   tickvals=[0, 20, 40, 60, 80, 100], ticksuffix="%"),
        xaxis=dict(
            title="", tickmode="array", tickvals=x,
            ticktext=[f"gen {g}" for g in comp["generation"]],
            range=[min(x) - 0.75, max(x) + 0.75],
            showgrid=False, ticks="outside", ticklen=3,
        ),
        legend=dict(traceorder="reversed" if LARGEST_AT_BOTTOM else "normal"),
        margin=dict(t=60, l=70, r=40, b=80),
    )
    # height=/width= must go through apply_nature_layout (not a separate
    # fig.update_layout call) -- write_figure() reads fig.layout.width/
    # height back off the figure to compute the print-DPI export scale, so
    # both need to be set here before write_figure() is called.
    apply_nature_layout(fig, height=FIG_HEIGHT, width=FIG_WIDTH, extra_layout=dict(
        legend=dict(traceorder="reversed" if LARGEST_AT_BOTTOM else "normal")))
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Rule summaries : {RULE_BASE}\\{{GEN}}\\{RULE_SUBDIR}")
    print(f"Figures        : {OUT_DIR}")
    print(f"Anchors        : generations {ANCHOR_GENERATIONS}\n")

    long_df = load_rule_summaries()
    if long_df.empty:
        raise SystemExit("No rule summaries found — check RULE_BASE.")

    absent = [g for g in GENERATORS if g not in set(long_df["generator"])]
    if absent:
        print(f"  [WARN] no bars for {', '.join(absent)}\n")

    segments, totals, peak = choose_segments(long_df)
    n_fired = int((totals > 0).sum())
    n_other = n_fired - len(segments)

    print(f"{n_fired} rules fired across the anchor generations; "
          f"{len(segments)} reach {SEGMENT_MIN_SHARE_PCT:.0f}% in at least one bar")

    comp = build_composition(long_df, segments)
    if comp.empty:
        raise SystemExit("No bars to draw — check ANCHOR_GENERATIONS.")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{OUT_NAME}.csv")
    comp.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    fig = build_figure(comp, segments, n_other)
    write_figure(fig, OUT_DIR, OUT_NAME, panel_kind="multi_panel")


if __name__ == "__main__":
    main()