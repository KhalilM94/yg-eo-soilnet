"""Figure 6 and supplementary figures S1-S2: measured against predicted SOM on the test points.

Each panel shows one model under one validation design: every test point in grey, the 1:1 line,
the least-squares line through all the points and one per measured carbonate class. A slope under
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
    WIDTH_2COL,
    caption_line,
    class_color,
    letter,
    model_color,
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
    pools the basins.
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
    return rows[["point_id", "observed", "predicted", "carbonate_class_measured"]].dropna(
        subset=["observed", "predicted"]
    )


def panels_frame(inputs: FigureInputs, models: Sequence[str], designs: Sequence[tuple]) -> pd.DataFrame:
    """The test points of every (model, design) panel, long: ``model, kind, block_size_m, design, ...``."""
    parts = []
    for model in models:
        for kind, size in designs:
            part = test_points(inputs, model, kind, size)
            if part.empty:
                continue
            parts.append(
                part.assign(
                    model=model,
                    kind=kind,
                    block_size_m=float(size) if size is not None else np.nan,
                    design=pretty_split(kind, size),
                )
            )
    columns = [
        "model",
        "kind",
        "block_size_m",
        "design",
        "point_id",
        "observed",
        "predicted",
        "carbonate_class_measured",
    ]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


def stats_frame(points: pd.DataFrame) -> pd.DataFrame:
    """:func:`obs_pred_stats` of every panel of :func:`panels_frame`."""
    rows = []
    for (model, design), part in points.groupby(["model", "design"], sort=False):
        rows.append({"model": model, "design": design, **obs_pred_stats(part["observed"], part["predicted"])})
    return pd.DataFrame(rows)


def _main_designs(inputs: FigureInputs) -> list:
    designs = [(RANDOM, None)]
    if inputs.primary_block_size is not None:
        designs.append((SPATIAL_KFOLD, inputs.primary_block_size))
    designs.append((LOGO, None))
    return designs


def _caption_numbers(stats: pd.DataFrame) -> str:
    return "; ".join(
        f"{row.design}: R² = {row.r2:.2f}, slope = {row.slope:.2f}, SD ratio = {row.sd_ratio:.2f}"
        for row in stats.itertuples()
    )


def figure_obs_pred(inputs: FigureInputs):
    """Figure 6: the global model's measured against predicted SOM, random | primary blocks | basins."""
    points = panels_frame(inputs, [GLOBAL], _main_designs(inputs))
    if points.empty:
        return _missing("No test predictions of the global model: run run_a1.py with random, spatial or logo plans.")
    stats = stats_frame(points)
    figure = draw_obs_pred(points, labels=inputs.class_labels)
    caption = (
        "Figure 6. Measured against predicted SOM, global model, test points. "
        + _caption_numbers(stats)
        + ". A slope and SD ratio below 1 mean predictions are pulled towards the mean: high SOM is "
        "under-predicted and low SOM over-predicted."
    )
    return attach(figure, {"fig06_obs_pred_points": points, "fig06_obs_pred_stats": stats}, caption)


def figure_obs_pred_blocks(inputs: FigureInputs):
    """Figure S1: the global model under every design the run has (each block size)."""
    designs = [(RANDOM, None), *((SPATIAL_KFOLD, size) for size in inputs.block_sizes), (LOGO, None)]
    points = panels_frame(inputs, [GLOBAL], designs)
    if points.empty:
        return _missing("No test predictions of the global model.")
    stats = stats_frame(points)
    figure = draw_obs_pred(points, labels=inputs.class_labels, columns=3)
    caption = "Figure S1. As figure 6, under every validation design. " + _caption_numbers(stats) + "."
    return attach(figure, {"figS1_obs_pred_blocks_stats": stats}, caption)


def figure_obs_pred_models(inputs: FigureInputs):
    """Figure S2: figure 6 for each main model, one row per model."""
    points = panels_frame(inputs, MAIN_MODELS, _main_designs(inputs))
    if points.empty:
        return _missing("No test predictions of the main models.")
    stats = stats_frame(points)
    figure = draw_obs_pred(points, labels=inputs.class_labels, by_model=True)
    caption = (
        "Figure S2. As figure 6, one row per model. "
        + "; ".join(
            f"{pretty_name(row.model)}, {row.design}: R² = {row.r2:.2f}, slope = {row.slope:.2f}"
            for row in stats.itertuples()
        )
        + "."
    )
    return attach(figure, {"figS2_obs_pred_models_stats": stats}, caption)


def _upper_limit(points: pd.DataFrame) -> float:
    values = pd.concat([points["observed"], points["predicted"]]).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, UPPER_PERCENTILE)) if len(values) else 1.0


def _least_squares(axis, observed, predicted, limit, **style) -> None:
    if len(observed) < 2 or np.std(observed) == 0:
        return
    slope, intercept = np.polyfit(observed, predicted, 1)
    x = np.array([0.0, limit])
    axis.plot(x, intercept + slope * x, **style)


def _panel(axis, part: pd.DataFrame, labels, limit: float, color: str) -> None:
    order = shuffled(part)
    axis.scatter(order["observed"], order["predicted"], s=4, color=color, alpha=0.35, lw=0, rasterized=True, zorder=2)
    axis.plot([0, limit], [0, limit], color=ps.INK_2, lw=0.8, ls="--", zorder=3)
    for label in labels:
        group = part[part["carbonate_class_measured"].astype(str) == str(label)]
        if len(group) >= MIN_CLASS_POINTS:
            _least_squares(
                axis,
                group["observed"].to_numpy(),
                group["predicted"].to_numpy(),
                limit,
                color=class_color(label),
                lw=1.0,
                zorder=4,
            )
    _least_squares(
        axis, part["observed"].to_numpy(), part["predicted"].to_numpy(), limit, color=ps.INK, lw=1.6, zorder=5
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    stats = obs_pred_stats(part["observed"], part["predicted"])
    text = (
        f"n = {stats['n']:,}\nR² = {stats['r2']:.2f}\nRMSE = {stats['rmse']:.2f}\nRPIQ = {stats['rpiq']:.2f}\n"
        f"bias = {stats['bias']:.2f}\nslope = {stats['slope']:.2f}\nSD ratio = {stats['sd_ratio']:.2f}"
    )
    axis.text(
        0.03,
        0.97,
        ps.minus(text),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=ps.INK,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ps.BASELINE, lw=0.5, alpha=0.9),
    )


@styled
def draw_obs_pred(points: pd.DataFrame, *, labels, columns: int = 3, by_model: bool = False):
    """Draw measured against predicted SOM, one panel per design (and one row per model with ``by_model``)."""
    import matplotlib.pyplot as plt

    if points.empty:
        return _missing("No test predictions to draw.")
    designs = list(dict.fromkeys(points["design"]))
    models = list(dict.fromkeys(points["model"])) if by_model else [None]
    panels = [(model, design) for model in models for design in designs]
    columns = len(designs) if by_model else min(columns, len(designs))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, 2.75 * rows + 1.0), layout="constrained", squeeze=False
    )
    limit = _upper_limit(points)
    for position, (axis, (model, design)) in enumerate(zip(axes.flat, panels)):
        part = points[(points["design"] == design) & ((points["model"] == model) if model else True)]
        color = model_color(model) if by_model else ps.MUTED
        _panel(axis, part, labels, limit, color)
        axis.set_title(design if not by_model else f"{pretty_name(model)}\n{design}", fontsize=8)
        if position % columns == 0:
            axis.set_ylabel(f"Predicted SOM ({SOM_UNIT})")
        if position >= len(panels) - columns:
            axis.set_xlabel(f"Measured SOM ({SOM_UNIT})")
        letter(axis, "abcdefghijkl"[position])
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_axis_off()
    handles = [
        Line2D([], [], color=ps.INK_2, lw=0.8, ls="--", label="1:1"),
        Line2D([], [], color=ps.INK, lw=1.6, label="Least squares, all points"),
        *(
            Line2D([], [], color=class_color(label), lw=1.0, label=f"Least squares, {pretty_name(label)} CaCO₃")
            for label in labels
        ),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    caption_line(
        figure,
        f"Axes from 0 to the {UPPER_PERCENTILE:g}th percentile of the values shown; a few larger values fall outside.",
    )
    return figure
