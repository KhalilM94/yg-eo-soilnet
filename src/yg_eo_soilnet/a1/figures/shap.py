"""Figure 11 and supplementary figure S4: what the global model relies on, within each carbonate class.

The global model of one plan (the random split by default) is fitted again and explained with SHAP
on its test points (:func:`~yg_eo_soilnet.a1.figures.core.shap_by_class`). Figure 11 adds the
mean |SHAP| of the inputs of each covariate group, one stacked bar per class; figure S4 shows the
single inputs that matter most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach, shap_by_class
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, class_color, letter, styled
from yg_eo_soilnet.a1.plans import RANDOM
from yg_eo_soilnet.a1.stage2 import GLOBAL

#: Inputs per class in figure S4.
N_FEATURES = 10


def global_shap(inputs: FigureInputs, plan_id: str = RANDOM) -> pd.DataFrame:
    """:func:`shap_by_class` of the global model, computed once per plan (the refit is slow)."""
    return inputs._cached(("shap", plan_id), lambda: shap_by_class(inputs, plan_id=plan_id, model=GLOBAL))


def group_of(inputs: FigureInputs) -> dict:
    """Each input column's covariate group (the first group listing it)."""
    resolved = inputs.a1.resolve_groups(inputs.table)
    owner: dict = {}
    for group, columns in resolved.groups.items():
        if group == AUXILIARY_GROUP:
            continue
        for column in columns:
            owner.setdefault(column, group)
    return owner


def shap_groups(values: pd.DataFrame, owner: dict) -> pd.DataFrame:
    """Mean |SHAP| summed per covariate group and class: ``class, group, mean_abs_shap, share, n``."""
    if values.empty:
        return pd.DataFrame(columns=["class", "group", "mean_abs_shap", "share", "n"])
    frame = values.assign(group=values["feature"].map(owner).fillna("other"))
    summed = (
        frame.groupby(["class", "group"], sort=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), n=("n", "first"))
        .reset_index()
    )
    summed["share"] = summed["mean_abs_shap"] / summed.groupby("class")["mean_abs_shap"].transform("sum")
    return summed


def figure_shap_groups(inputs: FigureInputs, *, plan_id: str = RANDOM):
    """Figure 11: the global model's mean |SHAP| per covariate group, one stacked bar per class."""
    try:
        values = global_shap(inputs, plan_id)
    except FileNotFoundError as error:
        return _missing(f"No SHAP values: {error}")
    groups = shap_groups(values, group_of(inputs))
    if groups.empty:
        return _missing("No SHAP values: no test point has a carbonate class.")
    labels = [label for label in inputs.class_labels if label in set(groups["class"])]
    order = groups.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).index.tolist()
    figure = draw_shap_groups(groups, labels=labels, order=order, plan=plan_id)
    top = groups.loc[groups.groupby("class")["share"].idxmax()]
    caption = (
        "Figure 11. What the global SOM model relies on in each measured CaCO₃ class: mean |SHAP| of its inputs, "
        f"added up per covariate group, on the test points of the {pretty_name(plan_id)} split. "
        + "; ".join(
            f"{pretty_name(row['class'])}: {pretty_name(row['group'])} {row['share']:.0%}" for _, row in top.iterrows()
        )
        + "."
    )
    return attach(figure, {"fig11_shap_groups": groups}, caption)


@styled
def draw_shap_groups(groups: pd.DataFrame, *, labels, order, plan: str = RANDOM):
    """Draw figure 11 from :func:`shap_groups`."""
    import matplotlib.pyplot as plt

    if groups.empty:
        return _missing("No SHAP values to draw.")
    figure, axis = plt.subplots(figsize=(WIDTH_2COL, 0.45 * len(labels) + 1.4), layout="constrained")
    positions = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for group in order:
        values = (
            groups[groups["group"] == group].set_index("class")["mean_abs_shap"].reindex(labels).fillna(0).to_numpy()
        )
        axis.barh(
            positions,
            values,
            left=left,
            height=0.65,
            color=ps.A1_GROUP_COLORS.get(group, ps.MUTED),
            edgecolor="white",
            lw=0.5,
            label=pretty_name(group),
        )
        left += values
    counts = groups.drop_duplicates("class").set_index("class")["n"]
    axis.set_yticks(positions, [f"{pretty_name(label)} (n = {int(counts.get(label, 0)):,})" for label in labels])
    ps.restyle_axes(axis, grid_axis="x")
    axis.tick_params(axis="both", labelsize=7)
    axis.set_ylabel("Measured CaCO₃ class")
    axis.set_xlabel("Mean |SHAP|, summed over the group's inputs (model units)")
    axis.legend(title="Covariate group", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False)
    letter(axis, "a")
    caption_line(
        figure, f"Global model refitted on the {pretty_name(plan)} split's training points; SHAP on its test points."
    )
    return figure


def figure_shap_features(inputs: FigureInputs, *, plan_id: str = RANDOM, n_features: int = N_FEATURES):
    """Figure S4: the global model's most relied-on single inputs, per carbonate class."""
    try:
        values = global_shap(inputs, plan_id)
    except FileNotFoundError as error:
        return _missing(f"No SHAP values: {error}")
    figure = draw_shap(values, labels=inputs.class_labels, n_features=n_features)
    caption = f"Figure S4. The {n_features} inputs with the largest mean |SHAP| in the global model, per measured CaCO₃ class."
    return attach(figure, {"figS4_shap_features": values}, caption)


@styled
def draw_shap(values: pd.DataFrame, *, labels, n_features: int = N_FEATURES):
    """Draw figure S4 from :func:`~yg_eo_soilnet.a1.figures.core.shap_by_class`."""
    import matplotlib.pyplot as plt

    present = [label for label in labels if label in set(values.get("class", []))]
    if not present:
        return _missing("No SHAP values: no test point has a carbonate class.")
    columns = min(2, len(present))
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, rows * (0.2 * n_features + 0.9)), layout="constrained", squeeze=False
    )
    for position, (axis, label) in enumerate(zip(axes.flat, present)):
        part = values[values["class"] == label].nlargest(n_features, "mean_abs_shap")[::-1]
        axis.barh(
            range(len(part)), part["mean_abs_shap"], color=class_color(label), edgecolor=ps.INK_2, lw=0.3, height=0.7
        )
        ps.restyle_axes(axis, grid_axis="x")
        axis.set_yticks(range(len(part)), [pretty_name(feature) for feature in part["feature"]])
        axis.tick_params(axis="both", labelsize=7)
        axis.set_title(f"{pretty_name(label)} CaCO₃, n = {int(part['n'].iloc[0]):,}", fontsize=7.5, loc="right")
        axis.set_xlabel("Mean |SHAP| (model units)")
        letter(axis, "abcd"[position])
    for axis in list(axes.flat)[len(present) :]:
        axis.set_axis_off()
    caption_line(figure, "Global model, random split, test points.")
    return figure
