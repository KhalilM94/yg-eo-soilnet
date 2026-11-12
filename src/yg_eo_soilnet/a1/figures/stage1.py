"""Figure 4: how well stage 1 predicts CaCO3 and CEC, and the carbonate class, on the random split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach, class_confusion, stage1_points
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, class_color, letter, shuffled, styled
from yg_eo_soilnet.a1.plans import RANDOM
from yg_eo_soilnet.metrics import regression_metrics


def figure_stage1(inputs: FigureInputs, *, plan_id: str = RANDOM):
    """Figure 4: measured against out-of-fold predicted CaCO3 and CEC; the class confusion matrix."""
    try:
        points = stage1_points(inputs, plan_id)
    except FileNotFoundError:
        return _missing(f"No stage-1 audit for plan {plan_id!r}: run run_a1.py with that plan first.")
    names = {inputs.carbonate_column: ("CaCO₃", "%")}
    if inputs.cec_column:
        names[inputs.cec_column] = ("CEC", "meq 100 g⁻¹")
    labels = inputs.class_labels
    counts = class_confusion(points["class_measured"], points["class_predicted"], labels)
    figure = draw_stage1(points, names, counts, labels=labels)
    scores = {
        name: regression_metrics(points[f"observed_{target}"], points[f"predicted_{target}"])
        for target, (name, _) in names.items()
    }
    correct = np.trace(counts.to_numpy()) / max(int(counts.to_numpy().sum()), 1)
    caption = (
        "Figure 4. Stage 1 on the random split: each training point's CaCO₃ and CEC predicted by a model fitted "
        "without it (out of fold). "
        + "; ".join(
            f"{name}: R² = {score.get('r2_test', np.nan):.2f}, RMSE = {score.get('rmse_test', np.nan):.2f}"
            for name, score in scores.items()
        )
        + f". (c) The predicted carbonate class is the measured one for {correct:.0%} of points; rows are the measured class."
    )
    confusion = (counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
    return attach(
        figure,
        {
            "fig04_stage1_points": points.reset_index(),
            "fig04_confusion_percent": confusion.rename_axis("measured").reset_index(),
        },
        caption,
    )


@styled
def draw_stage1(points: pd.DataFrame, names: dict, counts: pd.DataFrame, *, labels):
    """Draw figure 4: (a, b) measured against out-of-fold predicted; (c) the confusion matrix in row %."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(names) + 1, figsize=(WIDTH_2COL, 2.9), layout="constrained")
    for code, axis, (target, (name, unit)) in zip("abc", axes, names.items()):
        part = shuffled(
            points[[f"observed_{target}", f"predicted_{target}", "class_measured"]].dropna(
                subset=[f"observed_{target}", f"predicted_{target}"]
            )
        )
        observed, predicted = part[f"observed_{target}"], part[f"predicted_{target}"]
        colors = [class_color(label) for label in part["class_measured"].astype(object)]
        axis.scatter(observed, predicted, s=5, c=colors, edgecolor=ps.INK_2, lw=0.15, alpha=0.6, rasterized=True)
        if len(part):
            low, high = float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))
            axis.plot([low, high], [low, high], color=ps.INK_2, lw=0.8, ls="--", label="1:1")
        ps.square_panel(axis)
        axis.set_xlabel(f"Measured {name} ({unit})")
        axis.set_ylabel(f"Predicted {name}, out of fold ({unit})")
        scores = regression_metrics(observed, predicted)
        if scores:
            ps.metric_box(
                axis,
                ps.minus(
                    f"n = {int(scores['n_test']):,}\nR² = {scores.get('r2_test', np.nan):.2f}\nRMSE = {scores['rmse_test']:.2f} {unit}"
                ),
            )
        letter(axis, code)
    handles = [
        Patch(facecolor=class_color(label), edgecolor=ps.INK_2, lw=0.3, label=pretty_name(label)) for label in labels
    ]
    figure.legend(handles=handles, title="Measured CaCO₃", loc="outside upper center", ncol=4, frameon=False)

    matrix = axes[-1]
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    matrix.imshow(
        shares.to_numpy(dtype=float), cmap=ps.sequential_cmap(["#ffffff", *ps.CARBONATE_RAMP]), vmin=0, vmax=100
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            share = shares.iat[row, column]
            text = f"{share:.0f} %" if np.isfinite(share) else "–"
            matrix.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(share) and share > 55 else ps.INK,
                fontweight="bold" if row == column else "normal",
            )
        matrix.add_patch(Rectangle((row - 0.5, row - 0.5), 1, 1, fill=False, ec=ps.INK, lw=1.4))
    pretty = [pretty_name(label) for label in labels]
    matrix.set_xticks(range(len(labels)), pretty, rotation=30, ha="right")
    matrix.set_yticks(range(len(labels)), pretty)
    matrix.set_xlabel("Predicted CaCO₃ class")
    matrix.set_ylabel("Measured CaCO₃ class")
    matrix.grid(False)
    matrix.tick_params(length=0)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    total = int(counts.to_numpy().sum())
    correct = np.trace(counts.to_numpy()) / total if total else np.nan
    matrix.set_title(f"Right class: {correct:.0%}", fontsize=8, loc="right")
    letter(matrix, "c")
    caption_line(
        figure,
        "Training points of the random split, each predicted by a stage-1 model that did not see it. (c) Each row sums to 100 %; outlined: the right class.",
    )
    return figure
