"""Figures C4-C7: the robustness checks of the transfer mechanism (:mod:`yg_eo_soilnet.a1.robustness`).

C4  where SOM relates differently to the covariates within a carbonate class, and how often a basin
    still shows it when cut to a common size (the power check);
C5  SOM's level per measured class, each basin against every other point;
C6  how far k local samples correct the transferred model, against the local model;
C7  the cells: the transfer loss against the relationship change, the bias against the level shift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import _missing
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import (
    SOM_UNIT,
    WIDTH_2COL,
    caption_line,
    letter,
    model_color,
    styled,
)
from yg_eo_soilnet.a1.robustness import ALPHA, METHODS, local_text
from yg_eo_soilnet.a1.stage2 import GLOBAL

#: One colour per basin (figure C7), from the validated categorical palette of the covariate groups.
BASIN_COLORS = tuple(ps.A1_GROUP_COLORS.values())
#: One marker per measured carbonate class (figure C7).
CLASS_MARKERS = ("o", "s", "^", "D")
#: Line style of each correction (figure C6).
METHOD_STYLE = {
    "uncorrected": (ps.INK_2, "-", "No correction"),
    "offset": (model_color(GLOBAL), "-", "Offset"),
    "linear": ("#1baf7a", "--", "Linear"),
    "per_class_offset": ("#eb6834", ":", "Offset per class (diagnostic)"),
}
LOCAL_COLOR = model_color("local_global")


def basin_order(basins, failing) -> list:
    """The basins, those whose transfer fails first, each group in its given order."""
    failing = [basin for basin in basins if basin in set(failing)]
    return failing + [basin for basin in basins if basin not in set(failing)]


def _basin_label(basin: str, failing) -> str:
    return f"{basin} (fails)" if basin in set(failing) else basin


# --- C4 ------------------------------------------------------------------------------------------------


def within_class_grid(relations: pd.DataFrame, primary, labels) -> pd.DataFrame:
    """The within-class rows of ``basin_relations.csv`` for the covariates (CaCO3 and CEC left out)."""
    from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES

    part = relations[(relations["class"] != ALL_CLASSES) & ~relations["variable"].isin(list(primary))]
    part = part.assign(order=part["class"].map({label: position for position, label in enumerate(labels)}))
    return part.sort_values(["order"]).drop(columns="order")


def figure_within_class(
    relations: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    primary,
    labels,
    failing=(),
    sizes=(240, 55),
    n_draws: int = 100,
    block_km: float = 10.0,
):
    """Figure C4 and its caption, from ``basin_relations.csv`` and ``basin_relations_subsample.csv``."""
    grid = within_class_grid(relations, primary, labels)
    if grid.empty:
        return _missing("No within-class relations: run basin_relations.py first."), "Not drawn: no within-class rows."
    basins = basin_order(list(dict.fromkeys(relations["basin"])), failing)
    bars = summary[summary["level"] == "basin"] if not summary.empty else summary
    figure = draw_within_class(grid, bars, basins=basins, labels=labels, failing=failing, sizes=sizes)
    significant = grid[grid["p_holm"] < ALPHA].groupby("basin").size()
    found = "; ".join(f"{basin} {int(count)}" for basin, count in significant.items()) or "none"
    parts = []
    for size in sizes:
        for basin in basins:
            row = bars[(bars["basin"] == basin) & (bars["n_sub"] == size)] if not bars.empty else bars
            if row.empty or not bool(row["testable"].iloc[0]):
                continue
            row = row.iloc[0]
            parts.append(
                f"{basin} at {size}: ≥ 1 in {row['share_any']:.0%} of draws, mean {row['mean_significant']:.1f} "
                f"({row['q05_significant']:.0f}-{row['q95_significant']:.0f})"
            )
    caption = (
        "Figure C4. Within each measured carbonate class, the basin − rest difference in Spearman ρ(SOM, covariate) "
        "for the 15 covariates of figure 3; outlined: p_Holm < 0.05 (Holm within each basin over the exploratory "
        f"family). Significant within-class cells: {found}. Side bars: the power check - each class of the basin "
        f"cut to n points by whole {block_km:g} km blocks, {n_draws} draws, the share of draws with at least one "
        "significant within-class difference (CaCO₃ and CEC counted), for n = "
        + " and ".join(str(size) for size in sizes)
        + "; 'not testable' where no class of the basin has n points. "
        + ("; ".join(parts) + "." if parts else "")
    )
    return figure, ps.minus(caption)


@styled
def draw_within_class(grid: pd.DataFrame, bars: pd.DataFrame, *, basins, labels, failing=(), sizes=(240, 55)):
    """Draw figure C4: the heatmap of basins by class x covariate, and the power-check bars beside it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    variables = list(dict.fromkeys(grid["variable"]))
    classes = [label for label in labels if label in set(grid["class"])]
    columns = [(label, variable) for label in classes for variable in variables]
    values = np.full((len(basins), len(columns)), np.nan)
    marked = np.zeros_like(values, dtype=bool)
    lookup = grid.set_index(["basin", "class", "variable"])
    for i, basin in enumerate(basins):
        for j, (label, variable) in enumerate(columns):
            if (basin, label, variable) in lookup.index:
                row = lookup.loc[(basin, label, variable)]
                values[i, j] = row["difference"]
                marked[i, j] = bool(row["p_holm"] < ALPHA)
    limit = max(0.1, float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else 1.0
    figure = plt.figure(figsize=(WIDTH_2COL, 0.32 * len(basins) + 2.7), layout="constrained")
    heat_axis, bar_axis = figure.subplots(1, 2, width_ratios=[6.0, 1.0], sharey=True)
    image = heat_axis.imshow(
        values, aspect="auto", cmap=ps.diverging_cmap().reversed(), vmin=-limit, vmax=limit, interpolation="nearest"
    )
    for i, j in zip(*np.nonzero(marked)):
        heat_axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ps.INK, lw=1.1))
    heat_axis.set_yticks(np.arange(len(basins)), [_basin_label(basin, failing) for basin in basins])
    heat_axis.set_xticks(
        np.arange(len(columns)), [pretty_name(variable) for _, variable in columns], rotation=90, fontsize=7
    )
    for position in range(1, len(classes)):
        heat_axis.axvline(position * len(variables) - 0.5, color="white", lw=2.0)
    for position, label in enumerate(classes):
        heat_axis.text(
            (position + 0.5) * len(variables) - 0.5,
            -0.75,
            f"CaCO₃ {pretty_name(label)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    heat_axis.grid(False)  # the style's grid would cross the cells
    heat_axis.tick_params(length=0)
    for spine in heat_axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=heat_axis, location="bottom", shrink=0.4, pad=0.02, aspect=30)
    colorbar.set_label("ρ(SOM, covariate): basin − rest; outlined: p_Holm < 0.05")
    colorbar.outline.set_visible(False)
    letter(heat_axis, "a")

    height = 0.8 / max(1, len(sizes))
    for position, size in enumerate(sizes):
        offset = (position - (len(sizes) - 1) / 2) * height
        for i, basin in enumerate(basins):
            row = bars[(bars["basin"] == basin) & (bars["n_sub"] == size)] if not bars.empty else bars
            y = i + offset
            if row.empty or not bool(row["testable"].iloc[0]):
                bar_axis.text(0.02, y, f"n = {size}: not testable", va="center", fontsize=7, color=ps.MUTED)
                continue
            share = float(row["share_any"].iloc[0])
            color = ps.INK_2 if position == 0 else ps.MUTED
            bar_axis.barh(y, share, height=height * 0.9, color=color)
            bar_axis.text(min(share, 0.98) + 0.02, y, f"n = {size}", va="center", fontsize=7, color=ps.INK_2)
    bar_axis.grid(False)
    bar_axis.set_xlim(0, 1.45)
    bar_axis.set_xticks([0, 0.5, 1.0], ["0", "50", "100 %"])
    bar_axis.set_xlabel("Draws with ≥ 1\nsignificant cell")
    bar_axis.tick_params(axis="y", labelleft=False, length=0)
    letter(bar_axis, "b")
    return figure


# --- C5 ------------------------------------------------------------------------------------------------


def figure_som_profiles(profiles: pd.DataFrame, *, labels, failing=()):
    """Figure C5 and its caption, from :func:`~yg_eo_soilnet.a1.robustness.som_profiles`."""
    if profiles.empty:
        return _missing("No SOM profiles."), "Not drawn: no points."
    basins = basin_order(list(dict.fromkeys(profiles["basin"])), failing)
    figure = draw_som_profiles(profiles, basins=basins, labels=labels, failing=failing)
    basin_side = profiles[profiles["side"] == "basin"].set_index(["basin", "class"])["som_median"]
    rest = profiles[profiles["side"] == "complement"].set_index(["basin", "class"])["som_median"]
    shift = (basin_side - rest).dropna()
    largest = shift.abs().idxmax() if not shift.empty else None
    caption = (
        f"Figure C5. Median SOM ({SOM_UNIT}) per measured carbonate class in each held-out basin (filled) and in "
        "every other point (open); bars: interquartile range; same y axis in every panel. "
        + (
            f"The largest shift is {largest[0]}, class {pretty_name(largest[1])}: "
            f"{shift[largest]:+.1f} {SOM_UNIT} against the rest."
            if largest is not None
            else ""
        )
    )
    return figure, ps.minus(caption)


@styled
def draw_som_profiles(profiles: pd.DataFrame, *, basins, labels, failing=()):
    """Draw figure C5: one panel per basin, x the measured classes, basin and rest side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = min(5, len(basins))
    rows = int(np.ceil(len(basins) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, 2.2 * rows + 0.6), layout="constrained", squeeze=False, sharey=True
    )
    x = np.arange(len(labels))
    for position, basin in enumerate(basins):
        axis = axes[position // columns, position % columns]
        for side, offset, face in (("basin", -0.12, model_color(GLOBAL)), ("complement", 0.12, "white")):
            part = (
                profiles[(profiles["basin"] == basin) & (profiles["side"] == side)].set_index("class").reindex(labels)
            )
            median = part["som_median"].to_numpy(float)
            errors = np.vstack(
                [
                    np.clip(median - part["som_q25"].to_numpy(float), 0, None),
                    np.clip(part["som_q75"].to_numpy(float) - median, 0, None),
                ]
            )
            axis.errorbar(
                x + offset,
                median,
                yerr=errors,
                fmt="o",
                ms=3.5,
                color=model_color(GLOBAL) if side == "basin" else ps.INK_2,
                mfc=face,
                lw=0.9,
                capsize=1.5,
            )
        axis.set_title(_basin_label(basin, failing), fontsize=7)
        axis.set_xticks(x, [pretty_name(label) for label in labels], rotation=30, ha="right", rotation_mode="anchor")
        axis.set_xlabel("Measured CaCO₃")
        if position % columns == 0:
            axis.set_ylabel(f"SOM ({SOM_UNIT}), median and IQR")
    for empty in range(len(basins), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Line2D([], [], color=model_color(GLOBAL), marker="o", ls="none", ms=3.5, label="Held-out basin"),
        Line2D([], [], color=ps.INK_2, marker="o", mfc="white", ls="none", ms=3.5, label="Every other point"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return figure


# --- C6 ------------------------------------------------------------------------------------------------


def figure_bias_correction(
    summary: pd.DataFrame,
    *,
    local_model: str,
    failing=(),
    e2: Optional[pd.DataFrame] = None,
    n_draws: int = 100,
    block_km: float = 10.0,
):
    """Figure C6 and its caption, from ``bias_correction.csv`` (and the E2 curves when given)."""
    if summary.empty:
        return _missing("No basin leaves enough test points for any k."), "Not drawn: no correction draws."
    basins = basin_order(list(dict.fromkeys(summary["basin"])), failing)
    e2 = e2 if e2 is not None else pd.DataFrame()
    figure = draw_bias_correction(summary, basins=basins, failing=failing, e2=e2)
    parts = []
    for basin in basins:
        part = summary[summary["basin"] == basin].sort_values("k")
        if part.empty:
            continue
        last = part.iloc[-1]
        parts.append(
            f"{basin}, k = {int(last['k'])}: {last['rmse_uncorrected']:.2f} uncorrected, {last['rmse_offset']:.2f} with "
            f"the offset, {last['rmse_local']:.2f} local"
        )
    caption = (
        f"Figure C6. RMSE ({SOM_UNIT}) on each held-out basin's remaining points when k of its points are used as "
        "local samples to correct the model trained on the other basins (Global, no refitting): no correction; "
        "minus the samples' mean residual (offset); measured regressed on predicted (linear, k ≥ 25); one offset "
        "per measured carbonate class (a diagnostic: it needs the measured class of every point). Samples drawn by "
        f"whole {block_km:g} km blocks, the rest of the last block set aside; {n_draws} draws per k, line the mean, "
        f"band 5-95 %. Dashed: the {local_text(local_model)} on the same test points. A k is shown when at least 100 "
        "points remain. The mechanism predicts that an offset fixes Bouregreg, partly Oum Er Rbia, and not Sebou; "
        "the panels show whether it does. "
        + ("; ".join(parts) + ". " if parts else "")
        + (
            "Grey squares and diamonds: E2 retraining (control: the moved points alone; mixed: the other basins plus "
            "them), mean over repeats against the number of moved points - on each mixed plan's own remaining points, "
            "not the test points of the correction draws."
            if not e2.empty
            else ""
        )
    )
    return figure, ps.minus(caption)


@styled
def draw_bias_correction(summary: pd.DataFrame, *, basins, failing=(), e2: Optional[pd.DataFrame] = None):
    """Draw figure C6: one panel per basin, RMSE against k on a log axis."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    columns = min(5, len(basins))
    rows = int(np.ceil(len(basins) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, 2.3 * rows + 0.8), layout="constrained", squeeze=False, sharey=True
    )
    e2 = e2 if e2 is not None else pd.DataFrame()
    for position, basin in enumerate(basins):
        axis = axes[position // columns, position % columns]
        part = summary[summary["basin"] == basin].sort_values("k")
        for method in METHODS:
            color, style, _ = METHOD_STYLE[method]
            line = part.dropna(subset=[f"rmse_{method}"])
            if line.empty:
                continue
            axis.fill_between(
                line["k"], line[f"rmse_{method}_q05"], line[f"rmse_{method}_q95"], color=color, alpha=0.12, lw=0
            )
            axis.plot(line["k"], line[f"rmse_{method}"], color=color, ls=style, lw=1.1, marker="o", ms=2.5)
        axis.plot(part["k"], part["rmse_local"], color=LOCAL_COLOR, ls="--", lw=1.0)
        curves = e2[e2["basin"] == basin] if not e2.empty else e2
        if not curves.empty:
            axis.plot(curves["n_moved"], curves["rmse_control"], color=ps.MUTED, marker="s", ms=3, lw=0.8)
            axis.plot(curves["n_moved"], curves["rmse_mixed"], color=ps.MUTED, marker="D", ms=3, lw=0.8, ls=":")
        axis.set_xscale("log")
        axis.set_title(_basin_label(basin, failing), fontsize=7)
        axis.set_xlabel("Local samples, k")
        if position % columns == 0:
            axis.set_ylabel(f"RMSE ({SOM_UNIT}), remaining points")
    for empty in range(len(basins), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Line2D([], [], color=color, ls=style, lw=1.1, label=label) for color, style, label in METHOD_STYLE.values()
    ]
    handles += [
        Line2D([], [], color=LOCAL_COLOR, ls="--", lw=1.0, label="Local model"),
        Patch(color=ps.MUTED, alpha=0.3, label="5-95 % over draws"),
    ]
    if not e2.empty:
        handles += [
            Line2D([], [], color=ps.MUTED, marker="s", ms=3, lw=0.8, label="E2 control"),
            Line2D([], [], color=ps.MUTED, marker="D", ms=3, lw=0.8, ls=":", label="E2 mixed"),
        ]
    figure.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    caption_line(figure, "Same test points for every line of one draw; samples by whole blocks.")
    return figure


# --- C7 ------------------------------------------------------------------------------------------------


C7_PANELS = (
    ("relationship_change", "loss", "Mean |Δρ| over the 15 covariates", f"Transfer loss, RMSE − local ({SOM_UNIT})"),
    (
        "level_shift",
        "bias_transferred",
        f"Level shift, median SOM − rest ({SOM_UNIT})",
        f"Transferred bias ({SOM_UNIT})",
    ),
)


def figure_cells(cells: pd.DataFrame, correlations: pd.DataFrame, *, labels, local_model: str, scope: str):
    """Figure C7 and its caption: the cells of one ``seed`` scope (the first seed, or ``mean``)."""
    part = cells[cells["seed"] == scope] if not cells.empty else cells
    if part.empty:
        return _missing("No cells to draw."), "Not drawn: no cells."
    rho = correlations[correlations["seed"] == scope] if not correlations.empty else correlations
    figure = draw_cells(part, rho, labels=labels)
    parts = []
    for row in rho.itertuples():
        parts.append(
            f"{row.y} ~ {row.x}: ρ = {row.rho:.2f} [{row.low:.2f}, {row.high:.2f}] ({row.share_skipped:.0%} of draws skipped)"
        )
    caption = (
        f"Figure C7. The {len(part)} cells (held-out basin × measured carbonate class), "
        + ("mean over the seeds" if scope == "mean" else f"seed {scope}")
        + f": (a) transfer loss (Global's RMSE minus the {local_text(local_model)}'s, same points) against the "
        "relationship change (mean |basin − rest difference| of ρ(SOM, covariate) over the 15 covariates); "
        "(b) Global's bias (predicted − measured) against the level shift (the cell's median SOM minus that of the "
        "same class elsewhere): a cell richer in SOM than the training data is under-predicted, so the expected sign "
        "is negative. "
        "Colour: basin; marker: class. Spearman ρ with a 95 % interval from resampling the 5 basins (1000 draws; a "
        "draw with fewer than 3 distinct basins is skipped). Exploratory: 20 cells from 5 basins. "
        + "; ".join(parts)
        + "."
    )
    return figure, ps.minus(caption)


@styled
def draw_cells(cells: pd.DataFrame, correlations: pd.DataFrame, *, labels):
    """Draw figure C7: two scatter panels of the cells."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    basins = list(dict.fromkeys(cells["basin"]))
    colors = {basin: BASIN_COLORS[position % len(BASIN_COLORS)] for position, basin in enumerate(basins)}
    markers = {label: CLASS_MARKERS[position % len(CLASS_MARKERS)] for position, label in enumerate(labels)}
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 3.2), layout="constrained")
    for code, axis, (x, y, xlabel, ylabel) in zip("ab", axes, C7_PANELS):
        for _, row in cells.iterrows():
            axis.scatter(
                row[x],
                row[y],
                color=colors[row["basin"]],
                marker=markers.get(row["class"], "o"),
                s=26,
                edgecolor="white",
                lw=0.4,
                zorder=3,
            )
        axis.axhline(0, color=ps.BASELINE, lw=0.8, zorder=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        found = correlations[(correlations["x"] == x) & (correlations["y"] == y)] if not correlations.empty else None
        if found is not None and not found.empty:
            row = found.iloc[0]
            axis.set_title(
                ps.minus(f"Spearman ρ = {row['rho']:.2f} [{row['low']:.2f}, {row['high']:.2f}]"),
                fontsize=7,
                loc="right",
            )
        letter(axis, code)
    handles = [Line2D([], [], color=colors[basin], marker="o", ls="none", ms=4, label=basin) for basin in basins]
    handles += [
        Line2D([], [], color=ps.INK_2, marker=markers[label], ls="none", ms=4, label=f"CaCO₃ {pretty_name(label)}")
        for label in labels
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    return figure
