"""
visualization/style.py

One shared visual identity for every figure in the manuscript, so a reader
can learn "blue square = ARF" once and have it hold across every figure —
pooled trajectories, chain trajectories, expert-rule composition, frequency
pages. Import from here; do not redefine a generator color/marker locally.

Also carries the Nature Communications & AI compliance defaults (font,
line width, legend behaviour, PNG export sizing) as one place to tune them,
rather than four scripts each guessing their own numbers.

--------------------------------------------------------------------------
GENERATOR IDENTITY
--------------------------------------------------------------------------
Colour is never the only channel: every generator also gets its own marker
symbol, so the figures survive both a colorblind reader and a black-and-
white print run. The four colors are drawn from the Okabe-Ito palette
(designed for colorblind-safe figures) and avoid yellow, which has poor
contrast on a white background.

Root/variant (full vs core) is carried by DASH, not by color or marker, and
that convention is shared across every figure family that draws both roots
on the same axes.

This replaces two conflicting mappings that existed in the legacy scripts:
  - PLOT3_mean_trajectories_gp_pooled_v9.py's GENERATOR_COLORS/MARKERS
    (CTGAN=blue/circle, ARF=red/square, RTVAE=green/diamond, DDPM=purple/
    triangle-up)
  - the four chain-trajectory scripts' GEN_COLORS (ARF=blue, CTGAN=red,
    DDPM=green, RTVAE=purple) -- which additionally had NO markers at all,
    i.e. color was the only channel.
Neither survives black-and-white printing or a colorblind reader on its
own; this module is the single replacement for both.
"""

import os

# ============================================================
# GENERATOR IDENTITY -- color + marker + dash, shared everywhere
# ============================================================

GENERATORS = ["ARF", "CTGAN", "DDPM", "RTVAE"]

GENERATOR_COLORS = {
    "ARF":   "#0072B2",   # blue
    "CTGAN": "#D55E00",   # vermillion
    "DDPM":  "#009E73",   # bluish green
    "RTVAE": "#CC79A7",   # reddish purple
}

# Distinguishable by shape alone (filled vs open would add a 4th channel if
# ever needed, but 4 shapes is already enough for 4 generators in grayscale).
GENERATOR_MARKERS = {
    "ARF":   "square",
    "CTGAN": "circle",
    "DDPM":  "diamond",
    "RTVAE": "triangle-up",
}

# matplotlib uses single-character/string marker codes rather than plotly's
# symbol names -- kept as a parallel dict so the matplotlib-based scripts
# (population_features.py) can still draw the same shapes without a
# plotly-name lookup at the call site.
GENERATOR_MARKERS_MPL = {
    "ARF":   "s",   # square
    "CTGAN": "o",   # circle
    "DDPM":  "D",   # diamond
    "RTVAE": "^",   # triangle-up
}

# Root / variant: full vs core. Carried by dash, never by color or marker,
# so it composes cleanly with the generator identity above (e.g. "ARF, full"
# and "ARF, core" are the same blue square, solid vs dashed line).
ROOT_LABEL  = {"Step7pfa": "full",  "Step7pfp": "core"}
ROOT_SUFFIX = {"Step7pfa": "",      "Step7pfp": "_CORE"}
ROOT_DASH   = {"Step7pfa": "solid", "Step7pfp": "dash"}
ROOT_DASH_MPL = {"Step7pfa": "-",   "Step7pfp": "--"}

VARIANT_DASH = {"full": "solid", "core": "dash"}
VARIANT_DASH_MPL = {"full": "-", "core": "--"}

# A small set of non-generator colors used consistently for reference lines,
# collapse markers, etc. across every figure family.
REAL_COLOR = "#1f77b4"          # real data, in the frequency-page figures
SYNTHETIC_COLOR = "#009E73"     # synthetic data, in the frequency-page figures
BASELINE_COLOR = "#444444"      # TRTR / real-prevalence reference lines
CLASS_COLLAPSE_COLOR = "#8c564b"  # brown -- distinct from all 4 generator hues
COLLAPSE_ANNOTATION_COLOR = GENERATOR_COLORS["RTVAE"]

# ============================================================
# Lettering, line weight, legend
# ============================================================

# minimal size variance across a figure. Plotly sizes are
# in px; at typical export scale 1px ~= 0.75pt, so keep the numbers modest.
FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_SIZE_TITLE = 16
FONT_SIZE_AXIS_TITLE = 12
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12
FONT_SIZE_ANNOTATION = 12
FONT_COLOR = "#000000"          # max contrast on a white figure background

# matplotlib equivalents (points, matplotlib's native unit)
MPL_FONT_FAMILY = "Arial"
MPL_FONT_SIZE_TITLE = 12
MPL_FONT_SIZE_AXIS = 10
MPL_FONT_SIZE_TICK = 9

# >=0.3pt line width. Plotly line "width" is in px; at the DPI this module
# targets, 1px line width comfortably clears 0.3pt, so 2px (mean/primary
# lines) and 1.2px (thin chain lines) are both safe floors, not the literal
# minimum.
LINE_WIDTH_PRIMARY = 2.0
LINE_WIDTH_THIN = 1.2
LINE_WIDTH_REFERENCE = 1.5
MPL_LINEWIDTH_MIN_PT = 0.3   # the actual Nature floor, for anything drawn thinner

# Legends are INCLUDED (not omitted) per instruction -- Debbie writes the
# caption's legend explanation separately and crops the on-figure legend out
# herself afterward. Every figure in this package keeps its legend visible
# and non-interactive (itemclick disabled) so what ends up in the PNG
# actually matches what a legend-based caption would describe.
LEGEND_STATIC = dict(itemclick=False, itemdoubleclick=False)

# ============================================================
# PRINT EXPORT SIZING -- HTML + camera-button PNG snapshot
# ============================================================
# Debbie's preferred workflow (kept as-is): every figure below is written as
# a self-contained, embedded HTML page (include_plotlyjs="cdn" only pulls in
# the plotly.js *library* to render the interactive page -- the figure's own
# data and images are always inline, nothing about the figure itself is a
# linked image) with a camera-button PNG export wired to a specific
# width/height/scale so the downloaded PNG lands at a genuine print DPI
# rather than whatever the browser viewport happens to be.
#
# PNG is explicitly listed as an "acceptable" figure format in Nature's own
# list (PDF/EPS/AI/PS are "preferred"; PSD/EPC/TIFF/JPG/GIF/PPT/PPTX/PNG/BMP/
# VSD/CDX/EMF are "other acceptable formats") -- so keeping this workflow is
# compliant, it just needs the export dimensions dialed in to hit a real
# print DPI rather than the ~300dpi the old PNG_SCALE=3 export produced.
#
# DEFAULT_EXPORT_DPI defaults to Nature's "combination art" tier (600 dpi):
# every figure here mixes clean vector-style lines/markers with text and
# (in the frequency pages) shaded density fills, which is combination art
# rather than pure line art. Bump to 1200 for a figure you know will be
# reproduced as pure line art with no shading, or drop toward 300 if a
# resulting PNG is unworkably large for one of the bigger multi-panel grids
# -- print_width_mm/target_dpi are both plain arguments to png_export_dims,
# nothing here is hardwired.
DEFAULT_PRINT_WIDTH_MM = {
    "single_panel": 89,     # one-column figure
    "multi_panel": 183,     # two-column / full-width figure
}
DEFAULT_EXPORT_DPI = 600
MM_PER_INCH = 25.4

# Safety valve for a tall multi-panel stack (e.g. a 7-row figure): scaling
# both dimensions to hit 600dpi at full print width can otherwise produce a
# many-thousand-pixel-tall PNG that is slow for Kaleido/the browser to
# rasterize and needlessly large on disk. If the raw target exceeds this on
# either axis, the export scale is capped so both dimensions are held to it
# together (so DPI drops below the requested value for that one figure, but
# proportionally on width and height alike -- the PNG never comes out
# stretched). Nature's floor for combination art is 600dpi; a figure that
# trips this cap is a signal to either split it into fewer panels or export
# it at a smaller print width rather than silently accepting whatever the
# browser produces.
MAX_EXPORT_PX = 8000


def png_export_dims(fig_width, fig_height, width_mm=None, dpi=DEFAULT_EXPORT_DPI,
                     panel_kind="multi_panel", max_px=MAX_EXPORT_PX):
    """
    (width_px, height_px, scale) for a camera-button PNG export that hits
    `dpi` at `width_mm` printed width, WITHOUT shrinking text/lines/markers
    relative to what's on screen.

    `fig_width`/`fig_height` must be the figure's actual on-screen layout
    size in px (i.e. fig.layout.width / fig.layout.height, as set by
    apply_nature_layout's width=/height= arguments) -- this function returns
    those SAME numbers back as width_px/height_px. That's deliberate: in
    Plotly's toImageButtonOptions, `width`/`height` set the LAYOUT size used
    to render the export, not just a target pixel count -- so an earlier
    version of this function fed it the large final print-DPI pixel count
    directly (e.g. ~4323x1989 for 183mm @ 600dpi) with scale hardcoded to 1.
    That silently re-laid the whole figure out at ~4300px wide while every
    absolute-px font/line/marker constant in apply_nature_layout (e.g.
    FONT_SIZE_TICK=12) stayed the same literal number of pixels -- so text
    that looked normal at the on-screen ~1000px width became ~4x too small,
    relatively, once rendered into a canvas ~4x wider. That's the exact
    "fonts fine in HTML, too small once exported to PNG" symptom.

    `scale` is the fix: it's a pure resolution multiplier (like an @2x /
    retina screenshot) applied AFTER layout -- it re-rasterizes the SAME
    on-screen layout at higher pixel density, so text/lines/markers grow
    together with the canvas and stay in the same proportion they have on
    screen. `scale = target_width_px / fig_width` is exactly the multiplier
    needed to hit `dpi` at `width_mm` printed width from that layout.

    If the resulting scale would push either axis past `max_px`, scale is
    capped instead (both axes still scale together, proportionally) -- see
    MAX_EXPORT_PX above.
    """
    if width_mm is None:
        width_mm = DEFAULT_PRINT_WIDTH_MM[panel_kind]
    width_in = width_mm / MM_PER_INCH
    target_width_px = width_in * dpi
    scale = target_width_px / fig_width

    if max_px and max(fig_width, fig_height) * scale > max_px:
        capped_scale = max_px / max(fig_width, fig_height)
        effective_dpi = dpi * (capped_scale / scale)
        scale = capped_scale
        print(f"  [NOTE] export capped at {max_px}px on the long edge "
              f"(effective ~{effective_dpi:.0f} dpi instead of {dpi} -- "
              f"split this figure into fewer panels to recover full {dpi} dpi)")

    return fig_width, fig_height, scale


def camera_button_config(name, width_px, height_px, scale, dpi=DEFAULT_EXPORT_DPI):
    """toImageButtonOptions block for fig.write_html's `config=`."""
    return dict(
        displaylogo=False,
        toImageButtonOptions=dict(
            format="png", filename=name,
            width=width_px, height=height_px, scale=scale,
        ),
        modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d",
                                 "toggleSpikelines"],
    )


def apply_nature_layout(fig, height=None, width=None, title_text=None,
                         legend=True, extra_layout=None):
    """
    Shared layout pass for every plotly figure in this package: font family/
    size, legend behaviour, and (per Nature's "no titles baked into the
    illustration" rule) NO title drawn on the figure itself -- `title_text`
    is accepted only so callers can pass it through to write_figure() for
    the accompanying manifest/caption note, never to fig.update_layout.

    Always pass height= and width= (even though both are technically
    optional here): write_figure() reads fig.layout.width/height back off
    the figure to compute the print-DPI export scale, so a figure that never
    had its on-screen size set here can't be exported correctly downstream.
    """
    layout = dict(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=FONT_COLOR),
    )
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    if legend:
        layout["legend"] = dict(
            orientation="v",
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_LEGEND, color=FONT_COLOR),
            **LEGEND_STATIC,
        )
    else:
        layout["showlegend"] = False
    if extra_layout:
        layout.update(extra_layout)
    fig.update_layout(**layout)

    # axis titles/ticks: consistent font everywhere, applied after the main
    # layout call so it reaches every subplot axis plotly has created so far.
    fig.update_xaxes(title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_yaxes(title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS_TITLE,
                                      color=FONT_COLOR),
                      tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK,
                                    color=FONT_COLOR))
    fig.update_annotations(font=dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION,
                                      color=FONT_COLOR))
    return fig


def write_figure(fig, out_dir, name, panel_kind="multi_panel",
                  width_mm=None, dpi=DEFAULT_EXPORT_DPI):
    """
    Write the self-contained HTML (camera-button PNG wired to a real print
    DPI) that every figure script in this package ends with.

    Reads the figure's own on-screen size straight off fig.layout.width /
    fig.layout.height (set via apply_nature_layout's height=/width=) rather
    than taking an `aspect` argument -- computing the export scale from the
    figure's REAL layout size, instead of an independently-declared aspect
    ratio, is what keeps text/lines/markers proportioned correctly in the
    exported PNG (see png_export_dims's docstring). Raises if either isn't
    set, since there'd be nothing correct to scale from.
    """
    if fig.layout.width is None or fig.layout.height is None:
        raise ValueError(
            f"write_figure({name!r}): fig.layout.width/height are not set -- "
            "call apply_nature_layout(fig, height=..., width=...) before "
            "write_figure() so the print-DPI export scale can be computed "
            "from the figure's actual on-screen size."
        )
    os.makedirs(out_dir, exist_ok=True)
    width_px, height_px, scale = png_export_dims(
        fig.layout.width, fig.layout.height, width_mm, dpi, panel_kind
    )
    config = camera_button_config(name, width_px, height_px, scale, dpi)
    path = os.path.join(out_dir, f"{name}.html")
    fig.write_html(path, include_plotlyjs="cdn", config=config)
    export_w = int(round(width_px * scale))
    export_h = int(round(height_px * scale))
    print(f"  Saved: {path}  (camera button exports PNG at {export_w}x{export_h}px "
          f"~= {dpi} dpi at {width_mm or DEFAULT_PRINT_WIDTH_MM[panel_kind]}mm width, "
          f"i.e. {width_px}x{height_px} on-screen layout at {scale:.2f}x scale)")
    return path


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"