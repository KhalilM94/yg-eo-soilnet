"""How every A1 figure is drawn: sizes, fonts, the models' colours and markers, and axis wording.

The figures are sized for a journal page - one column 90 mm wide, two columns 190 mm - with no
text smaller than 7 pt at that size. Each model keeps one colour and one marker everywhere, and the
main-text models are always listed in the same order. A metric axis says which direction is better
and in what unit; a bias axis marks zero as "no bias" and names its two sides.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL, PER_CLASS_MEASURED, PER_CLASS_PREDICTED

MM = 1 / 25.4
#: A one-column figure, inches.
WIDTH_1COL = 90 * MM
#: A two-column figure, inches.
WIDTH_2COL = 190 * MM
#: The smallest text in any figure, points.
MIN_FONT = 7.0
#: The unit of SOM and of every SOM score that has one.
SOM_UNIT = "g kg⁻¹"

#: The three models of the main-text figures, in legend order.
MAIN_MODELS = (GLOBAL, PER_CLASS_MEASURED, PER_CLASS_PREDICTED)
#: Every model, in legend order: the main ones, then the supplementary and within-basin ones.
MODEL_ORDER = (*MAIN_MODELS, GLOBAL_AUX, LOCAL)
#: One marker per model, so no model is told apart by colour alone.
MODEL_MARKERS = {GLOBAL: "o", PER_CLASS_MEASURED: "s", PER_CLASS_PREDICTED: "^", GLOBAL_AUX: "D", LOCAL: "v"}

#: Font sizes at print size: nothing below :data:`MIN_FONT`.
RC = {
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "figure.titlesize": 8.5,
}


@contextmanager
def a1_style():
    """Draw in the project's style, at the A1 print sizes."""
    with ps.style_context(), plt.rc_context(RC):
        yield


def styled(function):
    """The decorator form of :func:`a1_style`."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        """Call the wrapped function in the A1 style."""
        with a1_style():
            return function(*args, **kwargs)

    return wrapper


def model_color(model: str) -> str:
    """The model's colour (:data:`~yg_eo_soilnet.plot_style.A1_MODEL_COLORS`)."""
    return ps.A1_MODEL_COLORS.get(model, ps.MUTED)


def model_marker(model: str) -> str:
    """The model's marker."""
    return MODEL_MARKERS.get(model, "o")


def class_color(label) -> str:
    """The carbonate class's colour, light sand (< 2 %) to dark brown (>= 25 %)."""
    return ps.A1_CARBONATE_COLORS.get(str(label), ps.MUTED)


def better_label(name: str, direction: Optional[str], unit: Optional[str] = None) -> str:
    """An axis label that says the unit and which way is better.

    Examples
    --------
    >>> better_label("RMSE", "lower", "g kg⁻¹")
    'RMSE (g kg⁻¹), lower is better'
    >>> better_label("R²", "higher")
    'R², higher is better'
    """
    text = f"{name} ({unit})" if unit else name
    return f"{text}, {direction} is better" if direction else text


def bias_axis(axis, *, orientation: str = "y", unit: str = SOM_UNIT) -> None:
    """Mark zero as "no bias" and name the two sides of a bias axis (predicted minus measured)."""
    if orientation == "y":
        axis.axhline(0, color=ps.INK_2, lw=0.8, zorder=1)
        axis.set_ylabel(f"Bias ({unit}), predicted − measured")
        axis.text(
            1.0,
            0.0,
            " no bias",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color=ps.INK_2,
        )
        axis.text(
            0.01, 0.98, "↑ over-predicts", transform=axis.transAxes, ha="left", va="top", fontsize=7, color=ps.INK_2
        )
        axis.text(
            0.01, 0.02, "↓ under-predicts", transform=axis.transAxes, ha="left", va="bottom", fontsize=7, color=ps.INK_2
        )
    else:
        axis.axvline(0, color=ps.INK_2, lw=0.8, zorder=1)
        axis.set_xlabel(f"Bias ({unit}), predicted − measured")


def caption_line(figure, text: str) -> None:
    """A small line under the figure saying what its bars or bands are.

    A strip at the bottom of the figure is kept for it: the layout (constrained) is told to fit
    everything else - axes and legends below them - above the strip, so they cannot run into it on
    any later layout pass. The strip is as tall as the wrapped text needs.
    """
    import textwrap

    width_pt = figure.get_figwidth() * 72
    per_line = max(40, int(width_pt / (MIN_FONT * 0.52)))
    wrapped = "\n".join(textwrap.wrap(text, per_line))
    lines = wrapped.count("\n") + 1
    height = (lines * MIN_FONT * 1.25 + 6) / (figure.get_figheight() * 72)
    engine = figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set") and "rect" in engine.get():
        engine.set(rect=(0, height, 1, 1 - height))
        figure.text(0.005, height - 0.004, wrapped, ha="left", va="top", fontsize=MIN_FONT, color=ps.INK_2)
    else:
        figure.text(0.0, -0.015, wrapped, ha="left", va="top", fontsize=MIN_FONT, color=ps.INK_2)


#: Where :func:`place_labels` tries a label: offset (points), alignment.
LABEL_SPOTS = (
    ((5, 3), "left", "bottom"),
    ((5, -3), "left", "top"),
    ((-5, 3), "right", "bottom"),
    ((-5, -3), "right", "top"),
    ((0, 6), "center", "bottom"),
    ((0, -6), "center", "top"),
    ((5, 14), "left", "bottom"),
    ((-5, 14), "right", "bottom"),
)


def place_labels(axis, xs, ys, texts, *, avoid=(), fontsize: float = MIN_FONT, passes: int = 4) -> list:
    """Label points so no label covers another label, a point, or the ``avoid`` artists.

    Each label may sit at any of :data:`LABEL_SPOTS` around its point. A spot costs one per thing
    it overlaps, three if it leaves the axes, and a half if another point is nearer to it than its
    own (it would read as that point's label). With few labels every combination is tried and the
    cheapest kept; with many, each label in turn moves to its cheapest spot, for a few passes. Call
    it once the layout is final (after colour bars and legends), since it measures on screen.
    """
    import itertools

    figure = axis.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    avoided = [artist.get_window_extent(renderer) for artist in avoid]
    anchors = [axis.transData.transform((x, y)) for x, y in zip(xs, ys)]
    points = [Bbox.from_extents(px - 4, py - 4, px + 4, py + 4) for px, py in anchors]
    # Every candidate box of every label, measured once.
    boxes = []
    for x, y, text in zip(xs, ys, texts):
        options = []
        for offset, ha, va in LABEL_SPOTS:
            label = axis.annotate(
                text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=fontsize
            )
            label.set_in_layout(False)
            options.append(label.get_window_extent(renderer))
            label.remove()
        boxes.append(options)

    def gap(box, anchor):
        px, py = anchor
        return np.hypot(max(box.x0 - px, 0, px - box.x1), max(box.y0 - py, 0, py - box.y1))

    def own_cost(index, spot):
        """What a label costs on its own: points and avoided artists covered, leaving the axes, misleading."""
        box = boxes[index][spot]
        inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
        mine = gap(box, anchors[index])
        misleading = any(gap(box, anchor) < mine for k, anchor in enumerate(anchors) if k != index)
        covered = sum(box.overlaps(other) for other in avoided) + sum(
            box.overlaps(point) for k, point in enumerate(points) if k != index
        )
        return covered + (0 if inside else 3) + (0.5 if misleading else 0)

    alone = [[own_cost(index, spot) for spot in range(len(LABEL_SPOTS))] for index in range(len(boxes))]

    def total(choice):
        clashes = sum(
            boxes[a][choice[a]].overlaps(boxes[b][choice[b]])
            for a in range(len(choice))
            for b in range(a + 1, len(choice))
        )
        return clashes + sum(alone[index][spot] for index, spot in enumerate(choice))

    spots = range(len(LABEL_SPOTS))
    if len(LABEL_SPOTS) ** len(boxes) <= 50_000:
        chosen = list(min(itertools.product(spots, repeat=len(boxes)), key=total))
    else:
        chosen = [int(np.argmin(costs)) for costs in alone]
        for _ in range(passes):
            for index in range(len(boxes)):
                chosen[index] = min(spots, key=lambda spot: total(chosen[:index] + [spot] + chosen[index + 1 :]))
    labels = []
    for x, y, text, spot in zip(xs, ys, texts, chosen):
        offset, ha, va = LABEL_SPOTS[spot]
        label = axis.annotate(
            text,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=ps.INK_2,
            zorder=6,
        )
        # Out of the layout: it was placed on this layout, and must not move it.
        label.set_in_layout(False)
        labels.append(label)
    return labels


def shuffled(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The rows in a fixed random order, so no group of points is always drawn on top."""
    return frame.iloc[np.random.default_rng(seed).permutation(len(frame))]


def letter(axis, text: str) -> None:
    """The panel's letter, bold, in the left title slot."""
    ps.panel_letter(axis, text)


def figure_fonts(figure) -> list:
    """The size of every visible text in the figure (for the tests' >= 7 pt check)."""
    sizes = []
    for text in figure.findobj(match=lambda artist: hasattr(artist, "get_fontsize") and hasattr(artist, "get_text")):
        if text.get_visible() and str(text.get_text()).strip():
            sizes.append(float(text.get_fontsize()))
    return sizes
