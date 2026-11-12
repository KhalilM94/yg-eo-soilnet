"""Figure 7, the RPIQ-by-class table and supplementary figure S3: scores within each carbonate class.

The intervals are a block bootstrap per design, model and measured class over the run's test
predictions on every input; ``evaluation/scores_by_class.csv`` holds the estimates but not the
intervals, so they are computed here, once, and shared by the three outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name, pretty_split
from yg_eo_soilnet.a1.figures.style import (
    MAIN_MODELS,
    MODEL_ORDER,
    SOM_UNIT,
    WIDTH_2COL,
    better_label,
    bias_axis,
    caption_line,
    letter,
    model_color,
    model_marker,
    styled,
)
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import LOCAL
from yg_eo_soilnet.datamodules.splitting import TEST

#: The models scored per class: every stage-2 model but the within-basin one.
CLASS_MODELS = tuple(model for model in MODEL_ORDER if model != LOCAL)


def class_intervals(inputs: FigureInputs) -> pd.DataFrame:
    """RPIQ and bias with 95 % block-bootstrap intervals per design, model and measured class.

    The designs are random, the primary spatial block size and leave one basin out (basins
    pooled). Computed once per :class:`FigureInputs`.

    Returns
    -------
    pandas.DataFrame
        ``design, kind, model, carbonate_class_measured, metric, estimate, low, high, n, small``.
    """

    def build():
        frame = inputs.predictions
        rows = frame[
            (frame["split"] == TEST)
            & (frame["covariate_group"] == COMBINED_GROUP)
            & frame["model"].isin(CLASS_MODELS)
            & frame["kind"].isin([RANDOM, SPATIAL_KFOLD, LOGO])
        ].dropna(subset=["carbonate_class_measured"])
        primary = inputs.primary_block_size
        rows = rows[(rows["kind"] != SPATIAL_KFOLD) | (rows["block_size_m"] == primary)]
        if rows.empty:
            return pd.DataFrame(
                columns=[
                    "design",
                    "kind",
                    "model",
                    "carbonate_class_measured",
                    "metric",
                    "estimate",
                    "low",
                    "high",
                    "n",
                    "small",
                ]
            )
        rows = rows.assign(design=[pretty_split(kind, size) for kind, size in zip(rows["kind"], rows["block_size_m"])])
        result = evaluation.block_bootstrap(
            rows,
            ["design", "kind", "model", "carbonate_class_measured"],
            block_km=inputs.block_km,
            n=inputs.n_boot,
            seed=inputs.settings.seed,
        )
        keep = result.intervals[result.intervals["metric"].isin(["rpiq", "bias"])]
        order = {kind: position for position, kind in enumerate([RANDOM, SPATIAL_KFOLD, LOGO])}
        return (
            keep.assign(_order=keep["kind"].map(order))
            .sort_values(["_order", "model"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    return inputs._cached("class_intervals", build)


def rpiq_table(intervals: pd.DataFrame, labels) -> pd.DataFrame:
    """RPIQ by class, one row per design and model, "estimate [low, high]" per class (the supplementary table)."""
    rpiq = intervals[intervals["metric"] == "rpiq"]
    rows = []
    for (design, model), part in rpiq.groupby(["design", "model"], sort=False):
        part = part.set_index("carbonate_class_measured")
        row = {"Design": design, "Model": pretty_name(model)}
        for label in labels:
            if label in part.index:
                value = part.loc[label]
                row[f"{pretty_name(label)} CaCO₃"] = (
                    f"{value['estimate']:.2f} [{value['low']:.2f}, {value['high']:.2f}]"
                )
            else:
                row[f"{pretty_name(label)} CaCO₃"] = "–"
        rows.append(row)
    return pd.DataFrame(rows)


def figure_class_bias(inputs: FigureInputs):
    """Figure 7: the main models' bias within each carbonate class, one panel per design."""
    intervals = class_intervals(inputs)
    bias = intervals[(intervals["metric"] == "bias") & intervals["model"].isin(MAIN_MODELS)]
    if bias.empty:
        return _missing("No per-class scores: run run_a1.py with random, spatial or logo plans.")
    labels = inputs.class_labels
    figure = draw_class_bias(bias, labels=labels, block_km=inputs.block_km)
    worst = bias.loc[bias["estimate"].abs().idxmax()]
    caption = (
        "Figure 7. Bias of predicted SOM within each measured CaCO₃ class (predicted − measured). "
        f"The largest is {worst['estimate']:+.2f} {SOM_UNIT} ({pretty_name(worst['model'])}, {worst['design']}, "
        f"{pretty_name(worst['carbonate_class_measured'])} CaCO₃). RPIQ by class is in the supplementary table."
    )
    return attach(figure, {"fig07_class_bias": bias, "table_rpiq_by_class": rpiq_table(intervals, labels)}, caption)


def figure_class_scores(inputs: FigureInputs):
    """Figure S3: RPIQ and bias within each class for the four stage-2 models."""
    intervals = class_intervals(inputs)
    if intervals.empty:
        return _missing("No per-class scores: run run_a1.py with random, spatial or logo plans.")
    figure = draw_class_scores(intervals, labels=inputs.class_labels, block_km=inputs.block_km)
    caption = (
        "Figure S3. RPIQ (top) and bias (bottom) per measured CaCO₃ class, all four SOM models, per validation design."
    )
    return attach(figure, {"figS3_class_scores": intervals}, caption)


def _points(axis, part: pd.DataFrame, labels, models) -> None:
    offsets = dict(zip(models, np.linspace(-0.24, 0.24, len(models)) if len(models) > 1 else [0.0]))
    for model in models:
        chosen = part[part["model"] == model].set_index("carbonate_class_measured").reindex(labels)
        x = np.arange(len(labels)) + offsets[model]
        errors = np.vstack(
            [(chosen["estimate"] - chosen["low"]).clip(lower=0), (chosen["high"] - chosen["estimate"]).clip(lower=0)]
        )
        color = model_color(model)
        axis.errorbar(
            x,
            chosen["estimate"],
            yerr=np.nan_to_num(errors),
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=0,
            zorder=2,
        )
        small = (
            chosen["small"].fillna(True).astype(bool).to_numpy() if "small" in chosen else np.zeros(len(labels), bool)
        )
        marker = model_marker(model)
        axis.scatter(
            x[~small], chosen["estimate"][~small], s=18, marker=marker, color=color, edgecolor="white", lw=0.4, zorder=3
        )
        axis.scatter(
            x[small],
            chosen["estimate"][small],
            s=18,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            lw=0.9,
            zorder=3,
        )
    axis.set_xticks(range(len(labels)), [pretty_name(label) for label in labels])
    axis.set_xlim(-0.5, len(labels) - 0.5)


def _legend(figure, models) -> None:
    handles = [
        Line2D([], [], ls="", marker=model_marker(m), color=model_color(m), label=pretty_name(m)) for m in models
    ]
    handles.append(
        Line2D(
            [], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=ps.MUTED, label="Fewer than 30 points"
        )
    )
    figure.legend(
        handles=handles, loc="outside upper center", ncol=len(handles) if len(handles) <= 4 else 3, frameon=False
    )


@styled
def draw_class_bias(bias: pd.DataFrame, *, labels, block_km: float = 10):
    """Draw figure 7: bias per class, one panel per design, the main models side by side."""
    import matplotlib.pyplot as plt

    if bias.empty:
        return _missing("No per-class bias to draw.")
    designs = list(dict.fromkeys(bias["design"]))
    figure, axes = plt.subplots(
        1, len(designs), figsize=(WIDTH_2COL, 2.9), layout="constrained", sharey=True, squeeze=False
    )
    for position, (axis, design) in enumerate(zip(axes.flat, designs)):
        _points(axis, bias[bias["design"] == design], labels, [m for m in MAIN_MODELS if m in set(bias["model"])])
        bias_axis(axis)
        if position:
            axis.set_ylabel("")
        axis.set_xlabel("Measured CaCO₃ class")
        axis.set_title(design, fontsize=8)
        letter(axis, "abc"[position])
    _legend(figure, [m for m in MAIN_MODELS if m in set(bias["model"])])
    caption_line(figure, f"Error bars: 95 % block-bootstrap intervals ({block_km:g} km blocks).")
    return figure


@styled
def draw_class_scores(intervals: pd.DataFrame, *, labels, block_km: float = 10):
    """Draw figure S3: RPIQ (top row) and bias (bottom row) per class, one column per design."""
    import matplotlib.pyplot as plt

    if intervals.empty:
        return _missing("No per-class scores to draw.")
    designs = list(dict.fromkeys(intervals["design"]))
    models = [m for m in CLASS_MODELS if m in set(intervals["model"])]
    figure, axes = plt.subplots(
        2, len(designs), figsize=(WIDTH_2COL, 4.6), layout="constrained", sharey="row", squeeze=False
    )
    for column, design in enumerate(designs):
        for row, metric in enumerate(("rpiq", "bias")):
            axis = axes[row, column]
            _points(axis, intervals[(intervals["design"] == design) & (intervals["metric"] == metric)], labels, models)
            if metric == "bias":
                bias_axis(axis)
                axis.set_xlabel("Measured CaCO₃ class")
            else:
                axis.set_ylabel(better_label("RPIQ", "higher"))
                axis.set_title(design, fontsize=8)
            if column:
                axis.set_ylabel("")
            letter(axis, "abcdefgh"[row * len(designs) + column])
    _legend(figure, models)
    caption_line(figure, f"Error bars: 95 % block-bootstrap intervals ({block_km:g} km blocks).")
    return figure
