"""Figure 1: the design of the experiment, as boxes and arrows - no data.

The validation column lists only the designs this run holds (read from its prediction files), so
the schematic and its caption never promise a design the run does not have.
"""

from __future__ import annotations

import textwrap

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import FigureInputs, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, model_color, styled
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, PER_CLASS_MEASURED, PER_CLASS_PREDICTED

#: The stage-2 models of the schematic, top to bottom; Global first, the one that skips stage 1.
DESIGN_MODELS = (GLOBAL, PER_CLASS_MEASURED, PER_CLASS_PREDICTED, GLOBAL_AUX)


def _km(size_m: float) -> str:
    return f"{size_m / 1000:g}"


def validation_designs(kinds, block_sizes, held_out) -> list:
    """The validation designs a run holds, in order of distance between training and test points.

    Examples
    --------
    >>> validation_designs(["logo", "mixed", "control"], [], ["Oum Er Rbia", "Sebou"])
    ['Leave one basin out (2 basins)', 'Mixed: share of the basin moved into training', 'Control: the moved points alone']
    >>> validation_designs(["random", "spatial_kfold"], [5000.0, 50000.0], [])
    ['Random split', 'Spatial blocks, 5-50 km grid']
    """
    kinds = set(kinds)
    designs = []
    if RANDOM in kinds:
        designs.append("Random split")
    if SPATIAL_KFOLD in kinds:
        sizes = sorted(float(size) for size in block_sizes)
        if not sizes:
            designs.append("Spatial blocks (clusters)")
        elif len(sizes) == 1:
            designs.append(f"Spatial blocks, {_km(sizes[0])} km grid")
        else:
            designs.append(f"Spatial blocks, {_km(sizes[0])}-{_km(sizes[-1])} km grid")
    if LOGO in kinds:
        count = len(held_out)
        designs.append(f"Leave one basin out ({count} basin{'s' if count != 1 else ''})")
    if MIXED in kinds:
        designs.append("Mixed: share of the basin moved into training")
    if CONTROL in kinds:
        designs.append("Control: the moved points alone")
    return designs


def design_frame(inputs: FigureInputs) -> pd.DataFrame:
    """What the schematic states: the inputs, the models and the validation designs of this run."""
    kinds = inputs.predictions["kind"].dropna().unique()
    rows = [
        ("inputs", "Satellite, climate, terrain, management indicators and legacy soil maps"),
        ("stage 1", "CaCO₃ and CEC predicted for every point, cross-fitted (folds grouped by block or basin)"),
        *(("stage 2", pretty_name(model)) for model in DESIGN_MODELS),
        *(("validation", text) for text in validation_designs(kinds, inputs.block_sizes, inputs.held_out_names)),
    ]
    return pd.DataFrame(rows, columns=["step", "text"])


def figure_design(inputs: FigureInputs):
    """Figure 1: inputs -> stage 1 -> stage 2 -> the validation designs of this run."""
    frame = design_frame(inputs)
    designs = frame.loc[frame["step"] == "validation", "text"].tolist()
    caption = (
        "Figure 1. Design of the experiment. Stage 1 predicts CaCO₃ and CEC for every point from the "
        "covariates; each training point's prediction comes from a model fitted without it (cross-fitting, "
        "folds grouped by spatial block or basin). The global model uses the covariates only; the other three "
        "SOM models also use the predicted CaCO₃/CEC. "
        + (
            f"The models are scored under {len(designs)} validation design{'s' if len(designs) != 1 else ''}: "
            + "; ".join(designs)
            + "."
            if designs
            else ""
        )
    )
    return attach(draw_design(frame), {"fig01_design": frame}, caption)


def _box(axis, x, y, width, height, text, *, face="#f4f3ef", edge=ps.INK_2, size=7.5):
    axis.add_patch(
        FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01,rounding_size=0.015", fc=face, ec=edge, lw=0.8)
    )
    # About 105 characters of 7.5 pt text across the full 190 mm width.
    text = "\n".join(textwrap.fill(line, max(12, int(width * 100))) for line in text.split("\n"))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size, color=ps.INK)


def _arrow(axis, start, end, *, rad=0.0, color=ps.INK_2):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=0.9,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


@styled
def draw_design(frame: pd.DataFrame):
    """Draw figure 1 from :func:`design_frame`."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(WIDTH_2COL, 3.1), layout="constrained")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    inputs = frame.loc[frame["step"] == "inputs", "text"].iloc[0]
    stage1 = frame.loc[frame["step"] == "stage 1", "text"].iloc[0]
    models = frame.loc[frame["step"] == "stage 2", "text"].tolist()
    designs = frame.loc[frame["step"] == "validation", "text"].tolist()

    for x, title in (
        (0.105, "Inputs"),
        (0.35, "Stage 1"),
        (0.61, "Stage 2: SOM models"),
        (0.895, "Validation designs"),
    ):
        axis.text(x, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold", color=ps.INK)

    # Inputs and stage 1 sit level with the three models that read stage 1's predictions.
    _box(axis, 0.01, 0.12, 0.19, 0.52, inputs)
    _box(axis, 0.26, 0.12, 0.18, 0.52, stage1)
    top, bottom = 0.88, 0.04
    step = (top - bottom) / max(len(models), 1)
    centres = []
    for position, (model, text) in enumerate(zip(DESIGN_MODELS, models)):
        y = top - (position + 1) * step
        _box(axis, 0.51, y + 0.015, 0.2, step - 0.03, text, face="white", edge=model_color(model))
        centres.append(y + step / 2)

    # Inputs -> stage 1, and inputs -> Global over the top of stage 1 (Global skips it).
    _arrow(axis, (0.205, 0.38), (0.253, 0.38))
    axis.plot([0.105, 0.105, 0.475], [0.64, centres[0], centres[0]], color=ps.INK_2, lw=0.9, solid_capstyle="butt")
    _arrow(axis, (0.475, centres[0]), (0.503, centres[0]))
    axis.text(0.29, centres[0] + 0.012, "covariates only", ha="center", va="bottom", fontsize=7, color=ps.INK_2)
    # Stage 1 -> the three models that use predicted CaCO3/CEC.
    for centre in centres[1:]:
        _arrow(axis, (0.447, 0.38), (0.503, centre))
    axis.text(0.35, 0.07, "+ predicted CaCO₃/CEC", ha="center", va="center", fontsize=7, color=ps.INK_2)

    # A bracket spanning the four models, and from it one arrow to the designs.
    low, high = centres[-1] - step / 2 + 0.02, centres[0] + step / 2 - 0.02
    axis.plot([0.718, 0.735, 0.735, 0.718], [high, high, low, low], color=ps.INK_2, lw=0.9)
    middle = (low + high) / 2
    _arrow(axis, (0.735, middle), (0.782, middle))

    if designs:
        step = (top - bottom) / len(designs)
        for position, text in enumerate(designs):
            y = top - (position + 1) * step
            _box(axis, 0.79, y + 0.015, 0.2, step - 0.03, text.replace(" (", "\n(").replace(": ", ":\n"))
    caption_line(
        figure,
        "Schematic; no data. Validation designs ordered by distance between training and test points, nearest first.",
    )
    return figure
