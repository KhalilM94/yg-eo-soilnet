"""Figure 6 and supplementary figures S1-S2: measured against predicted SOM on the test points.

Each panel shows one model under one validation design - or, for leave one basin out, one held-out
basin, since basins whose SOM ranges differ should not be pooled: every test point in grey, the 1:1
line, the least-squares line through all the points and one per measured carbonate class (named at
its right end). A slope under
1 and a spread ratio SD(predicted)/SD(measured) under 1 say the model pulls its predictions towards
the mean - high SOM under-predicted, low SOM over-predicted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name, pretty_split
from yg_eo_soilnet.a1.figures.style import (
    MAIN_MODELS,
    SOM_UNIT,
    WIDTH_1COL,
    WIDTH_2COL,
    caption_line,
    class_color,
    letter,
    shuffled,
    styled,
)
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import GLOBAL
from yg_eo_soilnet.datamodules.splitting import TEST
from yg_eo_soilnet.metrics import regression_metrics

#: The upper axis limit of every panel: this percentile of the values shown.
UPPER_PERCENTILE = 99.5
#: A per-class least-squares line is drawn only with at least this many points.
MIN_CLASS_POINTS = 10


def obs_pred_stats(observed, predicted) -> dict:
    """The numbers in a measured-against-predicted panel.

    ``n, r2, rmse, rpiq, bias`` from :func:`~yg_eo_soilnet.metrics.regression_metrics`; ``slope``
    of the least-squares line of predicted on measured; ``sd_ratio`` = SD(predicted) / SD(measured)
    (both with one degree of freedom removed).

    Examples
    --------
    >>> stats = obs_pred_stats([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    >>> round(stats["slope"], 6), round(stats["sd_ratio"], 6), round(stats["rmse"], 6)
    (1.0, 1.0, 0.0)
    """
    pairs = pd.DataFrame(
        {"observed": np.asarray(observed, dtype=float), "predicted": np.asarray(predicted, dtype=float)}
    ).dropna()
    scores = regression_metrics(pairs["observed"].to_numpy(), pairs["predicted"].to_numpy()) if len(pairs) else {}
    enough = len(pairs) >= 2 and pairs["observed"].std(ddof=1) > 0
    return {
        "n": int(len(pairs)),
        "r2": float(scores.get("r2_test", np.nan)),
        "rmse": float(scores.get("rmse_test", np.nan)),
        "rpiq": float(scores.get("rpiq_test", np.nan)),
        "bias": float(scores.get("bias_test", np.nan)),
        "slope": float(np.polyfit(pairs["observed"], pairs["predicted"], 1)[0]) if enough else np.nan,
        "sd_ratio": float(pairs["predicted"].std(ddof=1) / pairs["observed"].std(ddof=1)) if enough else np.nan,
    }


def test_points(inputs: FigureInputs, model: str, kind: str, block_size_m=None) -> pd.DataFrame:
    """One model's test predictions on every input (``combined``) under one design.

    Spatial folds of one block size are pooled (each point is tested once); leave-one-basin-out
    rows keep their ``held_out`` basin.
    """
    frame = inputs.predictions
    rows = frame[
        (frame["split"] == TEST)
        & (frame["covariate_group"] == COMBINED_GROUP)
        & (frame["model"] == model)
        & (frame["kind"] == kind)
    ]
    if kind == SPATIAL_KFOLD and block_size_m is not None:
        rows = rows[rows["block_size_m"] == float(block_size_m)]
    columns = ["point_id", "held_out", "observed", "predicted", "carbonate_class_measured"]
    return rows.reindex(columns=columns).dropna(subset=["observed", "predicted"])


def panels_frame(
    inputs: FigureInputs, models: Sequence[str], designs: Sequence[tuple], *, split_basins: bool = True
) -> pd.DataFrame:
    """The test points of every panel, long: ``model, kind, block_size_m, design, panel, ...``.

    ``panel`` is the design's name, or with ``split_basins`` the held-out basin's for each basin
    of a leave-one-basin-out design (one panel per basin).
    """
    parts = []
    for model in models:
        for kind, size in designs:
            part = test_points(inputs, model, kind, size)
            if part.empty:
                continue
            design = pretty_split(kind, size)
            part = part.assign(
                model=model, kind=kind, block_size_m=float(size) if size is not None else np.nan, design=design
            )
            if kind == LOGO and split_basins:
                part = part.assign(panel=part["held_out"].astype(str))
            else:
                part = part.assign(panel=design)
            parts.append(part)
    columns = [
        "model",
        "kind",
        "block_size_m",
        "design",
        "panel",
        "held_out",
        "point_id",
        "observed",
        "predicted",
        "carbonate_class_measured",
    ]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


def stats_frame(points: pd.DataFrame, by=("model", "design", "panel")) -> pd.DataFrame:
    """:func:`obs_pred_stats` of every panel of :func:`panels_frame` (or of any grouping ``by``)."""
    rows = []
    for key, part in points.groupby(list(by), sort=False):
        rows.append({**dict(zip(by, key)), **obs_pred_stats(part["observed"], part["predicted"])})
    return pd.DataFrame(rows)


def _main_designs(inputs: FigureInputs) -> list:
    """Figure 6's designs, those of them the run has: random, the main block size, leave one basin out."""
    kinds = set(inputs.predictions["kind"])
    designs = [(RANDOM, None)] if RANDOM in kinds else []
    if inputs.primary_block_size is not None:
        designs.append((SPATIAL_KFOLD, inputs.primary_block_size))
    if LOGO in kinds:
        designs.append((LOGO, None))
    return designs


def _caption_numbers(stats: pd.DataFrame) -> str:
    return "; ".join(
        f"{row.design}: R² = {row.r2:.2f}, slope = {row.slope:.2f}, SD ratio = {row.sd_ratio:.2f}"
        for row in stats.itertuples()
    )


def figure_obs_pred(inputs: FigureInputs):
    """Figure 6: the global model's measured against predicted SOM, one panel per design (per basin for LOBO)."""
    points = panels_frame(inputs, [GLOBAL], _main_designs(inputs))
    if points.empty:
        return _missing("No test predictions of the global model: run run_a1.py with random, spatial or logo plans.")
    stats = stats_frame(points)
    pooled = stats_frame(points, by=("model", "design"))
    figure = draw_obs_pred(points, labels=inputs.class_labels)
    caption = (
        "Figure 6. Measured against predicted SOM, global model, test points"
        + (", one panel per held-out basin" if LOGO in set(points["kind"]) else "")
        + ". Pooled: "
        + _caption_numbers(pooled)
        + ". A slope and SD ratio below 1 mean predictions are pulled towards the mean: high SOM is "
        "under-predicted and low SOM over-predicted."
    )
    return attach(figure, {"fig06_obs_pred_points": points, "fig06_obs_pred_stats": stats}, caption)


def figure_obs_pred_blocks(inputs: FigureInputs):
    """Figure S1: the global model under the designs figure 6 leaves out (the other block sizes)."""
    shown = {(kind, size) for kind, size in _main_designs(inputs)}
    extra = [(SPATIAL_KFOLD, size) for size in inputs.block_sizes if (SPATIAL_KFOLD, size) not in shown]
    if len(extra) < 2:
        return _missing(
            f"This run has {len(extra) or 'no'} validation design{'s' if len(extra) != 1 else ''} beyond those of "
            "figure 6; figure S1 shows the other spatial block sizes and needs at least two."
        )
    points = panels_frame(inputs, [GLOBAL], extra)
    stats = stats_frame(points)
    figure = draw_obs_pred(points, labels=inputs.class_labels)
    caption = "Figure S1. As figure 6, under the block sizes figure 6 leaves out. " + _caption_numbers(stats) + "."
    return attach(figure, {"figS1_obs_pred_blocks_stats": stats}, caption)


def figure_obs_pred_models(inputs: FigureInputs):
    """Figure S2: figure 6 for each main model - rows models, columns designs (or basins, LOBO alone)."""
    designs = _main_designs(inputs)
    only_logo = [kind for kind, _ in designs] == [LOGO]
    points = panels_frame(inputs, MAIN_MODELS, designs, split_basins=only_logo)
    if points.empty:
        return _missing("No test predictions of the main models.")
    stats = stats_frame(points)
    figure = draw_obs_pred(points, labels=inputs.class_labels, by_model=True)
    caption = (
        "Figure S2. As figure 6, one row per model. "
        + "; ".join(
            f"{pretty_name(row.model)}, {row.panel}: R² = {row.r2:.2f}, slope = {row.slope:.2f}"
            for row in stats.itertuples()
        )
        + "."
    )
    return attach(figure, {"figS2_obs_pred_models_stats": stats}, caption)


def _upper_limit(points: pd.DataFrame) -> float:
    values = pd.concat([points["observed"], points["predicted"]]).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, UPPER_PERCENTILE)) if len(values) else 1.0


def _fit(observed, predicted):
    if len(observed) < 2 or np.std(observed) == 0:
        return None
    return np.polyfit(observed, predicted, 1)


def _end_labels(axis, items, limit: float, min_gap_pt: float = 8.0) -> None:
    """Each class's name just right of the panel, level with its line's end, nudged so none overlap.

    Anchored to the data (the nudge alone is in points), so a later change of layout moves the
    labels with their lines.
    """
    from matplotlib.transforms import blended_transform_factory, offset_copy

    if not items:
        return
    figure = axis.figure
    to_points = 72.0 / figure.dpi
    ys = np.array([axis.transData.transform((limit, y))[1] * to_points for _, y, _ in items])
    placed = ys.copy()
    order = np.argsort(ys)
    for rank in range(1, len(order)):  # push each up until it clears the one below
        below, here = order[rank - 1], order[rank]
        placed[here] = max(placed[here], placed[below] + min_gap_pt)
    top = axis.transAxes.transform((0, 1))[1] * to_points
    if placed.max() > top:  # ran over the top: stack down from it instead
        for rank, here in enumerate(order[::-1]):
            ceiling = top if rank == 0 else placed[order[::-1][rank - 1]] - min_gap_pt
            placed[here] = min(placed[here], ceiling)
    margin = blended_transform_factory(axis.transAxes, axis.transData)
    for (text, y, color), nudge in zip(items, placed - ys):
        axis.text(
            1.02,
            y,
            text,
            transform=offset_copy(margin, fig=figure, x=0, y=nudge, units="points"),
            ha="left",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _panel(axis, part: pd.DataFrame, labels, limit: float, *, end_labels: bool = True) -> None:
    order = shuffled(part)
    axis.scatter(
        order["observed"], order["predicted"], s=4, color=ps.MUTED, alpha=0.35, lw=0, rasterized=True, zorder=2
    )
    axis.plot([0, limit], [0, limit], color=ps.INK_2, lw=0.8, ls="--", zorder=3)
    x = np.array([0.0, limit])
    ends = []
    for label in labels:
        group = part[part["carbonate_class_measured"].astype(str) == str(label)]
        fit = (
            _fit(group["observed"].to_numpy(), group["predicted"].to_numpy())
            if len(group) >= MIN_CLASS_POINTS
            else None
        )
        if fit is None:
            continue
        axis.plot(x, fit[1] + fit[0] * x, color=class_color(label), lw=1.5, zorder=4)
        ends.append((pretty_name(label), float(np.clip(fit[1] + fit[0] * limit, 0, limit)), class_color(label)))
    overall = _fit(part["observed"].to_numpy(), part["predicted"].to_numpy())
    if overall is not None:
        axis.plot(x, overall[1] + overall[0] * x, color=ps.INK, lw=1.8, zorder=5)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    stats = obs_pred_stats(part["observed"], part["predicted"])
    text = (
        f"n = {stats['n']:,}\nR² = {stats['r2']:.2f}\nRMSE = {stats['rmse']:.2f}\nRPIQ = {stats['rpiq']:.2f}\n"
        f"bias = {stats['bias']:.2f}\nslope = {stats['slope']:.2f}\nSD ratio = {stats['sd_ratio']:.2f}"
    )
    # Lower right: empty when the slope is under 1, as it is here.
    axis.text(
        0.97,
        0.03,
        ps.minus(text),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=ps.INK,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ps.BASELINE, lw=0.5, alpha=0.75),
    )
    if end_labels:
        axis.figure.draw_without_rendering()
        _end_labels(axis, ends, limit)


@styled
def draw_obs_pred(points: pd.DataFrame, *, labels, columns: int = 3, by_model: bool = False):
    """Draw measured against predicted SOM: one panel per ``panel``; with ``by_model``, one row per model.

    The figure is sized to its panels: one column (90 mm) for a single panel, two (190 mm) otherwise.
    """
    import matplotlib.pyplot as plt

    if points.empty:
        return _missing("No test predictions to draw.")
    panels = list(dict.fromkeys(points["panel"]))
    models = list(dict.fromkeys(points["model"])) if by_model else [None]
    columns = len(panels) if by_model else min(columns, len(panels))
    rows = len(models) if by_model else int(np.ceil(len(panels) / columns))
    width = WIDTH_1COL if columns == 1 else WIDTH_2COL
    # The row labels of figure S2 take about an inch on the left.
    panel_size = (width - (1.9 if by_model else 0.6)) / columns
    mosaic = [[f"{r}_{c}" for c in range(columns)] for r in range(rows)] + [["legend"] * columns]
    figure, grid = plt.subplot_mosaic(
        mosaic,
        figsize=(width, panel_size * rows + 0.8),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0] * rows + [0.3 / panel_size]},  # a legend row about 0.3 inch tall
    )
    limit = _upper_limit(points)
    cells = [(model, panel) for model in models for panel in panels]
    for position, (model, panel) in enumerate(cells):
        r, c = divmod(position, columns)
        axis = grid[f"{r}_{c}"]
        part = points[(points["panel"] == panel) & ((points["model"] == model) if model else True)]
        _panel(axis, part, labels, limit, end_labels=c == columns - 1 or not by_model)
        design = part["design"].iloc[0]
        title = panel if panel == design else f"{panel}, n = {len(part):,}"
        if not by_model or r == 0:
            axis.set_title(title, fontsize=8)  # the design, when not in the title, is in the caption
        if c == 0:
            axis.set_ylabel(f"Predicted SOM ({SOM_UNIT})")
            if by_model:
                # The model's name as a row label, left of the axis title.
                axis.annotate(
                    pretty_name(model).replace(", ", ",\n"),
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-34, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=ps.INK,
                    rotation=0,
                )
        if position >= len(cells) - columns:
            axis.set_xlabel(f"Measured SOM ({SOM_UNIT})")
        letter(axis, "abcdefghijklmnopqrstuvwxyz"[position])
    for position in range(len(cells), rows * columns):
        r, c = divmod(position, columns)
        grid[f"{r}_{c}"].set_axis_off()
    legend_axis = grid["legend"]
    legend_axis.set_axis_off()
    handles = [
        Line2D([], [], color=ps.INK_2, lw=0.8, ls="--", label="1:1"),
        Line2D([], [], color=ps.INK, lw=1.8, label="Least squares, all points"),
        Line2D([], [], color=class_color(labels[-1]), lw=1.5, label="Least squares per CaCO₃ class (named at its end)"),
    ]
    legend_axis.legend(handles=handles, loc="center", ncol=2 if columns == 1 else 3, frameon=False)
    caption_line(
        figure,
        f"Axes from 0 to the {UPPER_PERCENTILE:g}th percentile of the values shown; a few larger values fall outside.",
    )
    return figure
