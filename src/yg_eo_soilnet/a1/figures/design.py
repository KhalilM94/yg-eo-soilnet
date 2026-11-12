"""Figure 1: the design of the experiment, as boxes and arrows - no data."""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import FigureInputs, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import MODEL_ORDER, WIDTH_2COL, caption_line, model_color, styled
from yg_eo_soilnet.a1.stage2 import LOCAL


def design_frame(inputs: FigureInputs) -> pd.DataFrame:
    """What the schematic states: the inputs, the models and the validation designs of this run."""
    settings = inputs.settings
    sizes = inputs.block_sizes or list(settings.plans.spatial_kfold.block_sizes_m())
    sizes = [size for size in sizes if size is not None]
    blocks = ", ".join(f"{size / 1000:g}" for size in sizes) + " km" if sizes else "spatial clusters"
    rows = [
        ("inputs", "Satellite, climate, terrain, management indicators and legacy soil maps"),
        ("stage 1", f"CaCO₃ and CEC predicted for every point, cross-fitted in {settings.inner_cv_folds} folds"),
        *(("stage 2", pretty_name(model)) for model in MODEL_ORDER if model != LOCAL),
        ("validation", "Random split"),
        ("validation", f"Spatial blocks ({blocks})"),
        ("validation", f"Leave one basin out ({len(inputs.held_out_names) or 'each'} basins)"),
    ]
    return pd.DataFrame(rows, columns=["step", "text"])


def figure_design(inputs: FigureInputs):
    """Figure 1: inputs -> stage 1 -> stage 2 -> the three validation designs."""
    frame = design_frame(inputs)
    caption = (
        "Figure 1. Design of the experiment. Stage 1 predicts CaCO₃ and CEC for every point from the "
        "covariates; each training point's prediction comes from a model fitted without it (cross-fitting). "
        "Stage 2 predicts SOM with four models, which are scored under three validation designs of "
        "increasing distance between training and test points: "
        + "; ".join(frame.loc[frame["step"] == "validation", "text"])
        + "."
    )
    return attach(draw_design(frame), {"fig01_design": frame}, caption)


def _box(axis, x, y, width, height, text, *, face="#f4f3ef", edge=ps.INK_2, bold=False):
    import textwrap

    # About 105 characters of 7.5 pt text across the full 190 mm width.
    text = "\n".join(textwrap.fill(line, max(12, int(width * 105))) for line in text.split("\n"))
    axis.add_patch(
        FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01,rounding_size=0.015", fc=face, ec=edge, lw=0.8)
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.5 if not bold else 8,
        fontweight="bold" if bold else "normal",
        color=ps.INK,
    )


def _arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=ps.INK_2))


@styled
def draw_design(frame: pd.DataFrame):
    """Draw figure 1 from :func:`design_frame`."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(WIDTH_2COL, 3.0), layout="constrained")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    inputs = frame.loc[frame["step"] == "inputs", "text"].iloc[0]
    stage1 = frame.loc[frame["step"] == "stage 1", "text"].iloc[0]
    models = frame.loc[frame["step"] == "stage 2", "text"].tolist()
    designs = frame.loc[frame["step"] == "validation", "text"].tolist()

    axis.text(0.11, 0.97, "Inputs", ha="center", fontsize=8, fontweight="bold")
    _box(axis, 0.01, 0.38, 0.2, 0.42, inputs)
    axis.text(0.36, 0.97, "Stage 1", ha="center", fontsize=8, fontweight="bold")
    _box(axis, 0.26, 0.38, 0.2, 0.42, stage1)
    axis.text(0.62, 0.97, "Stage 2: SOM models", ha="center", fontsize=8, fontweight="bold")
    height = 0.8 / max(len(models), 1)
    for position, (model, text) in enumerate(zip(MODEL_ORDER, models)):
        y = 0.86 - (position + 1) * height
        _box(axis, 0.51, y + 0.02, 0.22, height - 0.04, text, face="white", edge=model_color(model))
    axis.text(0.89, 0.97, "Validation designs", ha="center", fontsize=8, fontweight="bold")
    height = 0.8 / max(len(designs), 1)
    for position, text in enumerate(designs):
        y = 0.86 - (position + 1) * height
        _box(axis, 0.78, y + 0.02, 0.21, height - 0.04, text.replace(" (", "\n("))
    _arrow(axis, (0.21, 0.59), (0.26, 0.59))
    _arrow(axis, (0.46, 0.59), (0.51, 0.59))
    _arrow(axis, (0.73, 0.59), (0.78, 0.59))
    axis.text(
        0.36,
        0.30,
        "Predicted CaCO₃/CEC feed the\nper-class models and\n“Global + predicted CaCO₃/CEC”",
        ha="center",
        va="top",
        fontsize=7,
        color=ps.INK_2,
    )
    caption_line(
        figure,
        "Schematic; no data. Distance between training and test points grows from top to bottom of the validation designs.",
    )
    return figure
