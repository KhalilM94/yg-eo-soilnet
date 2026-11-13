"""Figure 11 and supplementary figure S4: what the global model relies on, within each carbonate class.

The global model of a plan is fitted again and explained with SHAP on its test points
(:func:`~yg_eo_soilnet.a1.figures.core.shap_by_class`). Which plan: ``--shap-plan`` when given;
otherwise the random split, or - in a run without one - each leave-one-basin-out plan, one panel
per held-out basin (:func:`shap_plans`). Figure 11 adds the mean |SHAP| of the inputs of each
covariate group, one stacked bar per class; figure S4 shows the single inputs that matter most.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach, shap_by_class
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.layout import ran_plan
from yg_eo_soilnet.a1.figures.style import WIDTH_1COL, WIDTH_2COL, caption_line, class_color, letter, styled
from yg_eo_soilnet.a1.plans import LOGO, RANDOM
from yg_eo_soilnet.a1.stage2 import GLOBAL

logger = logging.getLogger(__name__)

#: Inputs per panel in figure S4.
N_FEATURES = 12


def shap_plans(inputs: FigureInputs, plan_id: Optional[str] = None) -> list:
    """The plans figures 11 and S4 explain.

    ``plan_id`` when given (``--shap-plan``); otherwise ``random`` when the run has it, and each
    leave-one-basin-out plan when it does not. Only plans the run actually ran count.
    """
    if plan_id:
        return [plan_id]
    index = inputs.index
    ran = index[[ran_plan(inputs.run_dir, plan) for plan in index["plan_id"]]]
    if RANDOM in set(ran["plan_id"]):
        return [RANDOM]
    logo = ran[ran["kind"] == LOGO].sort_values("plan_id")
    if logo.empty:
        return [RANDOM]  # nothing to fall back on: the note will say what is missing
    plans = logo["plan_id"].tolist()
    logger.info(f"No random plan in this run: figures 11 and S4 explain the global model of {', '.join(plans)}.")
    return plans


def plan_title(inputs: FigureInputs, plan_id: str) -> str:
    """A plan's name in a panel title: "Random split", or "<basin> held out"."""
    index = inputs.index.set_index("plan_id")
    if plan_id in index.index and index.loc[plan_id, "kind"] == LOGO:
        return f"{index.loc[plan_id, 'held_out']} held out"
    return "Random split" if plan_id == RANDOM else str(plan_id)


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


def _values(inputs: FigureInputs, plans: list) -> pd.DataFrame:
    """:func:`global_shap` of every plan, stacked, with a ``panel`` title per plan."""
    parts = [global_shap(inputs, plan).assign(plan_id=plan, panel=plan_title(inputs, plan)) for plan in plans]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def figure_shap_groups(inputs: FigureInputs, *, plan_id: Optional[str] = None):
    """Figure 11: the global model's mean |SHAP| per covariate group, one stacked bar per class, per plan."""
    plans = shap_plans(inputs, plan_id)
    try:
        values = _values(inputs, plans)
    except FileNotFoundError as error:
        return _missing(f"No SHAP values: {error}")
    owner = group_of(inputs)
    groups = (
        pd.concat(
            [shap_groups(part, owner).assign(panel=panel) for panel, part in values.groupby("panel", sort=False)],
            ignore_index=True,
        )
        if not values.empty
        else pd.DataFrame()
    )
    if groups.empty:
        return _missing("No SHAP values: no test point has a carbonate class.")
    labels = [label for label in inputs.class_labels if label in set(groups["class"])]
    order = groups.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).index.tolist()
    figure = draw_shap_groups(groups, labels=labels, order=order)
    top = groups.loc[groups.groupby(["panel", "class"])["share"].idxmax()]
    names = ", ".join(plan_title(inputs, plan) for plan in plans)
    caption = (
        "Figure 11. What the global SOM model relies on in each measured CaCO₃ class: mean |SHAP| of its inputs, "
        f"added up per covariate group, on the test points of: {names}"
        + (" (the run has no random split; one panel per held-out basin)" if len(plans) > 1 else "")
        + ". Largest group: "
        + "; ".join(
            f"{row['panel']}, {pretty_name(row['class'])}: {pretty_name(row['group'])} {row['share']:.0%}"
            for _, row in top.iterrows()
        )
        + "."
    )
    return attach(figure, {"fig11_shap_groups": groups}, caption)


@styled
def draw_shap_groups(groups: pd.DataFrame, *, labels, order, plan: str = ""):
    """Draw figure 11 from :func:`shap_groups` (one panel per ``panel`` when the table has that column)."""
    import matplotlib.pyplot as plt

    if groups.empty:
        return _missing("No SHAP values to draw.")
    if "panel" not in groups:
        groups = groups.assign(panel=plan or "")
    panels = list(dict.fromkeys(groups["panel"]))
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(WIDTH_2COL, (0.45 * len(labels) + 0.9) * len(panels) + 0.5),
        layout="constrained",
        squeeze=False,
        sharex=True,
    )
    widest = groups.groupby(["panel", "class"])["mean_abs_shap"].sum().max()
    for position, (axis, panel) in enumerate(zip(axes.flat, panels)):
        part = groups[groups["panel"] == panel]
        positions = np.arange(len(labels))[::-1]
        left = np.zeros(len(labels))
        for group in order:
            values = (
                part[part["group"] == group].set_index("class")["mean_abs_shap"].reindex(labels).fillna(0).to_numpy()
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
        counts = part.drop_duplicates("class").set_index("class")["n"]
        axis.set_yticks(positions, [f"{pretty_name(label)} (n = {int(counts.get(label, 0)):,})" for label in labels])
        ps.restyle_axes(axis, grid_axis="x")
        axis.tick_params(axis="both", labelsize=7)
        axis.set_xlim(0, widest * 1.03)
        axis.set_ylabel("Measured CaCO₃ class")
        if panel:
            axis.set_title(panel, fontsize=8, loc="right")
        letter(axis, "abcdefghij"[position])
    axes.flat[-1].set_xlabel("Mean |SHAP|, summed over the group's inputs (model units)")
    axes.flat[0].legend(
        title="Covariate group", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=False
    )
    caption_line(
        figure, "Global model refitted on each plan's training points, as the run fitted it; SHAP on its test points."
    )
    return figure


def figure_shap_features(inputs: FigureInputs, *, plan_id: Optional[str] = None, n_features: int = N_FEATURES):
    """Figure S4: the global model's most relied-on single inputs, one panel per plan, a bar per class."""
    plans = shap_plans(inputs, plan_id)
    try:
        values = _values(inputs, plans)
    except FileNotFoundError as error:
        return _missing(f"No SHAP values: {error}")
    figure = draw_shap(values, labels=inputs.class_labels, n_features=n_features)
    names = ", ".join(plan_title(inputs, plan) for plan in plans)
    caption = (
        f"Figure S4. The {n_features} inputs with the largest mean |SHAP| in the global model ({names}); for each, one "
        "bar per measured CaCO₃ class."
    )
    return attach(figure, {"figS4_shap_features": values}, caption)


@styled
def draw_shap(values: pd.DataFrame, *, labels, n_features: int = N_FEATURES):
    """Draw figure S4: per panel (plan), the inputs strongest on average over the classes, one thin bar per class."""
    import matplotlib.pyplot as plt

    present = [label for label in labels if label in set(values.get("class", []))]
    if not present:
        return _missing("No SHAP values: no test point has a carbonate class.")
    if "panel" not in values:
        values = values.assign(panel="")
    panels = list(dict.fromkeys(values["panel"]))
    columns = min(2, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(WIDTH_2COL if columns > 1 else WIDTH_1COL * 1.0, rows * (0.24 * n_features + 1.0) + 0.4),
        layout="constrained",
        squeeze=False,
    )
    height = 0.8 / len(present)
    for position, (axis, panel) in enumerate(zip(axes.flat, panels)):
        part = values[values["panel"] == panel]
        strongest = part.groupby("feature")["mean_abs_shap"].mean().nlargest(n_features).index[::-1]
        for rank, label in enumerate(present):
            bars = part[part["class"] == label].set_index("feature")["mean_abs_shap"].reindex(strongest).fillna(0)
            y = (
                np.arange(len(strongest)) + ((len(present) - 1) / 2 - rank) * height
            )  # first class on top, as in the legend
            axis.barh(y, bars.to_numpy(), height=height, color=class_color(label), edgecolor="none")
        ps.restyle_axes(axis, grid_axis="x")
        axis.set_yticks(range(len(strongest)), [pretty_name(feature) for feature in strongest])
        axis.tick_params(axis="both", labelsize=7)
        if panel:
            axis.set_title(panel, fontsize=8, loc="right")
        axis.set_xlabel("Mean |SHAP| (model units)")
        letter(axis, "abcdefghij"[position])
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_axis_off()
    handles = [Patch(facecolor=class_color(label), label=pretty_name(label)) for label in present]
    figure.legend(handles=handles, title="Measured CaCO₃", loc="outside upper center", ncol=len(handles), frameon=False)
    caption_line(figure, "Inputs ordered by their mean |SHAP| over the classes; test points of each plan.")
    return figure
