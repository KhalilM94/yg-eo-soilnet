"""The one place this project decides what a figure looks like.

Every constant below is lifted verbatim from the first cell of ``notebooks/paper_figures.ipynb``,
which is where this design system was worked out. The point of copying it here rather than
re-inventing it is that a run's diagnostic artifacts and the paper's figures then read as one system:
the same warm greys, the same blue/orange project pair, the same y-only grid, the same bold
left-aligned panel letter. If the notebook's palette changes, change it here too - they are meant to
be the same values, and a divergence is a bug rather than a variation.

Style is applied through :func:`styled` / :func:`style_context`, which wrap ``plt.rc_context``, NOT
through a module-level ``rcParams.update``. That matters twice over: these modules get imported
inside notebooks that have set their own style, and inside a test suite that turns warnings into
errors, and a global mutation would leak into both.

``savefig.dpi`` and ``savefig.bbox`` are in :data:`RC_PARAMS` for notebook parity but cannot do any
work here, because the save happens in ``artifacts.log_figure`` long after the context has closed.
That function passes ``dpi=SAVE_DPI, bbox_inches="tight"`` explicitly instead.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter

# --- ink and chrome ---------------------------------------------------------
# The greys are deliberately WARM - they carry a yellow-green cast rather than being neutral - which
# is what keeps them from fighting the blue/orange pair below.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
EMPTY_BAND = "#f0efec"

# The two source projects. Fertimap has three tints because in most figures it is context while Al
# Moutmir is the focus; the full colour is for when they are equals.
PROJECT_COLORS = {"Fertimap": "#2a78d6", "Al Moutmir": "#eb6834"}
FERTIMAP_TINT = "#86b6ef"
FERTIMAP_AREA_FILL = "#b7d3f6"

# Single-hue sequential ramps, light to dark. Both are built so the lightest step still clears ~2:1
# contrast on white, which is what makes the low end readable rather than merely present.
SOM_RAMP = ["#d6a66c", "#b9844a", "#96622f", "#6f4420", "#482a12"]
CARBONATE_RAMP = ["#a7b3c0", "#7d8ea1", "#546a82", "#2c4561"]

# Colours for a categorical series - one model, one line. The notebook has no such palette (it never
# plots more than a handful of named classes), so this is assembled from the project pair plus the
# darker ramp steps, ordered so the first few are maximally distinct.
MODEL_COLORS = [
    PROJECT_COLORS["Al Moutmir"],
    PROJECT_COLORS["Fertimap"],
    "#546a82",
    "#96622f",
    "#9a3b12",
    "#104281",
    "#2c4561",
    "#482a12",
]

# --- geometry ---------------------------------------------------------------
# The notebook uses exactly two widths and no others: 6.5in spans a two-column page, 3.4in is one
# column. Heights vary; widths do not.
FIG_WIDTH_FULL = 6.5
FIG_WIDTH_COLUMN = 3.4

SAVE_DPI = 300

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType so text stays editable in the PDF
}

# Thousands separators are the house rule on every count axis.
THOUSANDS = StrMethodFormatter("{x:,g}")


@contextmanager
def style_context():
    """Draw under this project's rcParams, restoring whatever was set before.

    Use it directly around third-party drawing calls - shap and optuna both build their own figures,
    and this is the only way their artists inherit the palette.
    """
    with plt.rc_context(RC_PARAMS):
        yield


def styled(function):
    """Decorator form of :func:`style_context`, for functions that build a Figure."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with style_context():
            return function(*args, **kwargs)

    return wrapper


def restyle_axes(axis, *, grid_axis: str = "y"):
    """Force the house style onto an axes that some other library has already drawn on.

    Needed because :func:`style_context` cannot defend against a library that calls
    ``plt.style.use`` itself - optuna's plotting helpers do exactly that (``plt.style.use("ggplot")``
    inside ``plot_optimization_history``), which replaces the rcParams mid-draw and leaves a grey
    panel, a white grid and ggplot's font sizes. Setting the properties on the artists afterwards is
    the only thing that outlives it.

    ``grid_axis="x"`` for a horizontal bar chart, where the value runs along x and a y grid would
    rule lines through the bars while marking nothing.
    """
    axis.set_facecolor("none")
    axis.figure.set_facecolor("white")

    # The bottom spine and nothing else, as everywhere in the house style - including on the
    # horizontal bar charts, which carry no left edge either.
    for name, spine in axis.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(BASELINE)
        spine.set_linewidth(0.8)

    axis.grid(False)
    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)

    # x keeps its tick marks, y never has any. That is the house rule (`ytick.major.size: 0`), and
    # it holds whichever way the bars run.
    axis.tick_params(axis="x", color=BASELINE, labelcolor=INK_2, labelsize=9, length=3)
    axis.tick_params(axis="y", color=BASELINE, labelcolor=INK_2, labelsize=9, length=0)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_color(INK_2)
        label.set_fontsize(9)

    # All three title slots, not just the centre one: a library that wrote its own heading may have
    # put it in any of them, and the caller is about to set its own.
    for location in ("left", "center", "right"):
        axis.set_title("", loc=location)

    # Whatever text the library annotated onto the axes - bar-end values, mostly - in secondary ink
    # at the house size rather than ggplot's.
    for text in axis.texts:
        text.set_color(INK_2)
        text.set_fontsize(7.5)
    return axis


def square_panel(axis):
    """Re-dress an axes for a scatter whose two axes mean the same thing.

    The house style drops the left spine and grids on y only, which suits the bar charts and time
    series it was designed for. A predicted-vs-observed panel is the documented exception: its 1:1
    line is the thing being read, and reading a point against that line needs gridlines running both
    ways and a left edge to anchor them. Everything else - colours, fonts, weights - stays.
    """
    axis.spines["left"].set_visible(True)
    axis.spines["left"].set_color(BASELINE)
    axis.grid(True, which="major", axis="both", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    # The house style zeroes y tick length because a y-only grid already marks the values; with both
    # grids on, the ticks are what tie the left edge to them.
    axis.tick_params(axis="y", length=3, color=BASELINE)


def panel_letter(axis, letter: str):
    """The figure's panel label: bold, lower-case, no parentheses, in the left title slot."""
    axis.set_title(letter, loc="left", fontweight="bold", color=INK)


def panel_subtitle(axis, text: str):
    """Context for a panel - which target, how many samples - in the RIGHT title slot.

    Kept separate from :func:`panel_letter` on purpose: matplotlib gives an axes three independent
    title slots, so the letter and its caption can coexist on one line without either being squeezed
    into the other's string.
    """
    axis.set_title(text, loc="right", fontsize=8, color=INK_2)


def metric_box(axis, text: str, *, loc: str = "upper left"):
    """A framed block of numbers inside the axes, matching the notebook's map-legend frame.

    Placed in a corner rather than beside the panel, because the numbers describe the cloud they sit
    on and a reader should not have to look away from it to find them.
    """
    positions = {
        "upper left": (0.04, 0.96, "left", "top"),
        "upper right": (0.96, 0.96, "right", "top"),
        "lower left": (0.04, 0.04, "left", "bottom"),
        "lower right": (0.96, 0.04, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    return axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=ha,
        va=va,
        fontsize=7.5,
        color=INK_2,
        zorder=6,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=BASELINE, linewidth=0.8),
    )


@styled
def message_figure(message: str, *, figsize=(FIG_WIDTH_COLUMN, 1.6)):
    """A figure that says why there is nothing to draw.

    Every plotting module in this project returns one of these instead of ``None`` when its input is
    empty, so a caller never has to branch on the return value. This is the single implementation;
    three near-identical copies preceded it.
    """
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=8, color=MUTED)
    axis.set_axis_off()
    return figure


def sequential_cmap(ramp=CARBONATE_RAMP, name: str = "soilnet_sequential"):
    """A continuous colormap from one of the discrete ramps above.

    Used wherever a colour has to encode a number - predictive sigma on the pred-vs-obs panel, the CV
    score on the parallel-coordinates plot. Single-hue by construction, which is what a magnitude
    deserves; the multi-hue defaults (viridis and friends) imply categories that are not there.
    """
    return LinearSegmentedColormap.from_list(name, ramp)


def minus(text: str) -> str:
    """Hyphen-minus to a real minus sign, so hand-written numbers match the axis ticks."""
    return text.replace("-", "\N{MINUS SIGN}")
