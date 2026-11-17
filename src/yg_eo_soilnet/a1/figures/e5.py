"""Figures E5a-E5c: what each input group adds, per design and basin, and the time-series summaries.

Each draws ΔRMSE (variant - reference) with its 95 % paired 10 km block-bootstrap interval
(:func:`yg_eo_soilnet.a1.e5.variant_differences`); a filled marker is Holm-significant (p_Holm <
0.05, within each part and design), an open one is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import e5
from yg_eo_soilnet.a1.figures.core import _missing
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import SOM_UNIT, WIDTH_2COL, caption_line, letter, model_color, styled
from yg_eo_soilnet.a1.stage2 import GLOBAL

#: The designs of figures E5a and E5c, in column order.
MAIN_DESIGNS = (e5.DESIGN_RANDOM, e5.DESIGN_10KM, e5.DESIGN_LOGO)
#: Holm significance.
ALPHA = 0.05
NOTE = "Bars: 95 % paired 10 km block bootstrap (first seed); filled: p_Holm < 0.05 within each part and design."


def variant_label(name: str, n_features=None) -> str:
    """A variant's plain name: ``all − Satellite``, ``R2 (quarterly medians)``, with its inputs when given.

    >>> variant_label("all_minus_legacy_soil")
    'all − Legacy soil maps'
    """
    if name.startswith("all_minus_"):
        text = f"all − {pretty_name(name.removeprefix('all_minus_'))}"
    elif name == "all_plus_xy":
        text = "all + x, y (interpolation check)"
    elif name.startswith("all_plus_"):
        text = f"all + {pretty_name(name.removeprefix('all_plus_'))}"
    elif name == "only_satellite":
        text = f"{pretty_name('satellite')} alone"
    elif name == "satellite_plus_terrain":
        text = f"{pretty_name('satellite')} + {pretty_name('terrain')}"
    elif name.startswith("ts_"):
        rep = name.removeprefix("ts_").removesuffix("_minus_legacy_soil")
        text = rep
    elif name.startswith("only_satellite_"):
        text = name.removeprefix("only_satellite_")
    elif name.startswith("only_"):
        text = f"{pretty_name(name.removeprefix('only_'))} alone"
    else:
        text = name
    return f"{text} ({int(n_features)})" if n_features is not None and np.isfinite(n_features) else text


def _panel(axis, rows: pd.DataFrame, order: list, labels: list, *, markers=None, offsets=None, show_labels=True):
    """One panel: a row per variant, ΔRMSE with its interval, the zero line."""
    y = np.arange(len(order))
    markers = markers or {"": ("o", model_color(GLOBAL))}
    offsets = offsets or {"": 0.0}
    for key, (marker, color) in markers.items():
        part = rows[rows["series"] == key] if "series" in rows else rows
        values = part.set_index("variant").reindex(order)
        estimate = values["estimate"].to_numpy(float)
        errors = np.vstack(
            [
                np.clip(estimate - values["low"].to_numpy(float), 0, None),
                np.clip(values["high"].to_numpy(float) - estimate, 0, None),
            ]
        )
        position = y + offsets.get(key, 0.0)
        axis.errorbar(estimate, position, xerr=errors, fmt="none", ecolor=color, elinewidth=0.9)
        significant = values["p_holm"].to_numpy(float) < ALPHA
        axis.scatter(estimate[significant], position[significant], marker=marker, s=22, color=color, zorder=3)
        axis.scatter(
            estimate[~significant],
            position[~significant],
            marker=marker,
            s=22,
            facecolors="white",
            edgecolors=color,
            zorder=3,
        )
    axis.axvline(0, color=ps.INK_2, lw=0.8, zorder=1)
    axis.set_yticks(y, labels if show_labels else [""] * len(y))
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.grid(False, axis="y")


def _rmse(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["metric"] == "rmse"] if not table.empty else table


@styled
def draw_ablation(table: pd.DataFrame, designs, *, title_of=None):
    """Draw E5a/E5b: the groups (upper row) and the satellite sub-blocks (lower row), one column per design."""
    import matplotlib.pyplot as plt

    rows = _rmse(table)
    groups = [f"all_minus_{g}" for g in e5.PART_A_GROUPS] + ["only_satellite", "satellite_plus_terrain", "all_plus_xy"]
    groups = [g for g in groups if g in set(rows["variant"])]
    blocks = [f"all_minus_{b}" for b in e5.SUBBLOCKS if f"all_minus_{b}" in set(rows["variant"])]
    features = rows.drop_duplicates("variant").set_index("variant")["n_features"].to_dict()
    # A row of panels per list that has variants (the groups, the satellite sub-blocks).
    orders = [order for order in (groups, blocks) if order]
    figure, axes = plt.subplots(
        len(orders),
        len(designs),
        figsize=(WIDTH_2COL, 0.27 * sum(len(order) for order in orders) + 1.6),
        layout="constrained",
        squeeze=False,
        sharey="row",
        gridspec_kw={"height_ratios": [len(order) for order in orders]},
    )
    for column, design in enumerate(designs):
        part = rows[rows["design"] == design]
        for row, order in enumerate(orders):
            axis = axes[row, column]
            _panel(axis, part, order, [variant_label(v, features.get(v)) for v in order], show_labels=column == 0)
            if row == 0:
                axis.set_title(title_of(design) if title_of else design, fontsize=8)
            if row == len(orders) - 1:
                axis.set_xlabel(f"ΔRMSE vs all ({SOM_UNIT})")
        letter(axes[0, column], "abcdefghij"[column])
    caption_line(figure, NOTE + " Positive: the model does worse without the group.")
    return figure


def figure_ablation(table: pd.DataFrame):
    """Figure E5a and its caption: the main designs."""
    rows = _rmse(table)
    if rows.empty:
        return _missing("No Part A comparisons: run run_e5.py --part A, then evaluate_e5.py."), "Not drawn."
    designs = [d for d in MAIN_DESIGNS if d in set(rows["design"])]
    figure = draw_ablation(table, designs)
    worst = rows[rows["design"] == e5.DESIGN_LOGO].sort_values("estimate", ascending=False).head(1)
    caption = (
        "Figure E5a. What each input group adds to Global: ΔRMSE = RMSE without the group − RMSE with every input "
        "(all), on the random split, the 10 km block folds pooled and the five held-out basins pooled; upper: the "
        "covariate groups (and satellite alone, satellite + terrain, all + projected x, y - an interpolation check, "
        "not a transfer model: interpret it on random and 10 km only), lower: the satellite sub-blocks. "
        + NOTE
        + (
            f" Leaving one basin out, removing {variant_label(worst['variant'].iloc[0])} costs most: "
            f"{worst['estimate'].iloc[0]:+.2f} [{worst['low'].iloc[0]:.2f}, {worst['high'].iloc[0]:.2f}] {SOM_UNIT}."
            if not worst.empty
            else ""
        )
    )
    return figure, ps.minus(caption)


def figure_ablation_basins(table: pd.DataFrame):
    """Figure E5b and its caption: each held-out basin."""
    rows = _rmse(table)
    designs = sorted(d for d in set(rows["design"]) if d.startswith("logo ") and d != e5.DESIGN_LOGO)
    if not designs:
        return _missing("No held-out basin in the Part A comparisons."), "Not drawn."
    figure = draw_ablation(table, designs, title_of=lambda d: d.removeprefix("logo "))
    caption = (
        "Figure E5b. As figure E5a, for each held-out basin on its own (Global trained on the other four). " + NOTE
    )
    return figure, ps.minus(caption)


@styled
def draw_representations(table: pd.DataFrame, designs):
    """Draw E5c: one row per representation, one column per design, a marker per reference."""
    import matplotlib.pyplot as plt

    rows = _rmse(table).copy()
    rows["rep"] = [variant_label(v) for v in rows["variant"]]
    rows["series"] = rows["reference"]
    reps = sorted(rows["rep"].unique())
    features = rows[rows["reference"] == "all"].drop_duplicates("rep").set_index("rep")["n_features"].to_dict()
    figure, axes = plt.subplots(
        1, len(designs), figsize=(WIDTH_2COL, 0.4 * len(reps) + 1.5), layout="constrained", sharey=True, squeeze=False
    )
    markers = {
        "all": ("o", model_color(GLOBAL)),
        "all_minus_legacy_soil": ("s", ps.INK_2),
        "only_satellite": ("^", "#1baf7a"),
    }
    offsets = {"all": -0.22, "all_minus_legacy_soil": 0.0, "only_satellite": 0.22}
    for column, design in enumerate(designs):
        axis = axes[0, column]
        part = rows[rows["design"] == design].assign(variant=lambda frame: frame["rep"])
        _panel(
            axis,
            part,
            reps,
            [
                f"{rep} ({int(features[rep])})" if rep in features and np.isfinite(features[rep]) else rep
                for rep in reps
            ],
            markers={k: v for k, v in markers.items() if k in set(part["series"])},
            offsets=offsets,
            show_labels=column == 0,
        )
        axis.set_title(design, fontsize=8)
        axis.set_xlabel(f"ΔRMSE vs R0 ({SOM_UNIT})")
        letter(axis, "abc"[column])
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker=m,
            ls="",
            color=c,
            label={
                "all": "on all inputs",
                "all_minus_legacy_soil": "without the soil-map layers",
                "only_satellite": "satellite alone",
            }[k],
        )
        for k, (m, c) in markers.items()
        if k in set(rows["series"])
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    caption_line(figure, NOTE + " Number of inputs in brackets (on all inputs).")
    return figure


def figure_representations(table: pd.DataFrame):
    """Figure E5c and its caption."""
    rows = _rmse(table)
    if rows.empty:
        return _missing("No Part B comparisons: run run_e5.py --part B, then evaluate_e5.py."), "Not drawn."
    designs = [d for d in MAIN_DESIGNS if d in set(rows["design"])]
    figure = draw_representations(table, designs)
    described = "; ".join(f"{rep}: {text}" for rep, text in e5.REPRESENTATIONS.items() if rep != "R0")
    caption = (
        "Figure E5c. Time-series summaries against R0, the whole-record percentiles of the monthly composites "
        "(2017-2025, years after sampling included): ΔRMSE = RMSE with the summary − RMSE with R0, every static input "
        "kept; on every input (circles), without the soil-map layers (squares, against R0 without them) and, for R2 "
        f"and R6, satellite alone (triangles). {described}. R6 is a bare-month filter on monthly composites, a weaker "
        "version of the scene-level rule of Dvorakova et al. (2023). " + NOTE
    )
    return figure, ps.minus(caption)
