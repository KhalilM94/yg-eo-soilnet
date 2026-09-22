"""The one place this project decides what a figure looks like.

Colours, fonts, spacing and sizes, shared by every figure a run produces and by the paper figures,
so a diagnostic plot and a published one are recognisably the same family.
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
    """Draw in this project's style, putting the previous settings back afterwards.

    Use it around another library's drawing calls, which build their own figures.
        """
    with plt.rc_context(RC_PARAMS):
        yield


def styled(function):
    """The decorator form of :func:`style_context`, for a function that builds a figure."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        """Call the wrapped function in this project's style."""
        with style_context():
            return function(*args, **kwargs)

    return wrapper


def restyle_axes(axis, *, grid_axis: str = "y"):
    """Impose the house style on a figure another library has already drawn.

    Needed because a library that sets its own colours and sizes as it draws cannot be styled in
    advance.
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

    The house style suits bar charts and time series; a predicted-against-measured panel needs both
    spines and a grid on both axes, or its diagonal cannot be read.
        """
    axis.spines["left"].set_visible(True)
    axis.spines["left"].set_color(BASELINE)
    axis.grid(True, which="major", axis="both", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    # The house style zeroes y tick length because a y-only grid already marks the values; with both
    # grids on, the ticks are what tie the left edge to them.
    axis.tick_params(axis="y", length=3, color=BASELINE)


def panel_letter(axis, letter: str):
    """The panel's label - **a**, **b** - in the left title slot."""
    axis.set_title(letter, loc="left", fontweight="bold", color=INK)


def panel_subtitle(axis, text: str):
    """Context for a panel, such as the target and how many points, in the right title slot."""
    axis.set_title(text, loc="right", fontsize=8, color=INK_2)


def metric_box(axis, text: str, *, loc: str = "upper left"):
    """A framed block of numbers inside the axes, describing what is drawn."""
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

    Every plotting function here returns one of these rather than nothing, so no caller has to check.
        """
    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=8, color=MUTED)
    axis.set_axis_off()
    return figure


def sequential_cmap(ramp=CARBONATE_RAMP, name: str = "soilnet_sequential"):
    """A continuous colour ramp, for wherever a colour has to stand for a number."""
    return LinearSegmentedColormap.from_list(name, ramp)


def minus(text: str) -> str:
    """Replace a hyphen with a real minus sign, so numbers match the axis labels."""
    return text.replace("-", "\N{MINUS SIGN}")
