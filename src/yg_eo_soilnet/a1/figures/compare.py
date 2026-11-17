"""Figures C1 and C2 of the round-3 comparisons (:mod:`yg_eo_soilnet.a1.compare`).

``figC1_mixed_arms``
    One column per held-out basin: (upper) test RMSE against the number of the basin's points moved
    into training, for control, mixed and each arm, with the model with no local points and the
    local model as reference lines; (lower) each arm minus control, paired, with its interval.
``figC2_logo_matched``
    One panel per held-out basin, ordered by the unmatched RMSE: RMSE against the training size -
    each matched draw a point, the mean over draws a line ending at the unmatched plan, at its own
    training size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import _missing
from yg_eo_soilnet.a1.figures.labels import load_labels, pretty_name
from yg_eo_soilnet.a1.figures.style import (
    SOM_UNIT,
    WIDTH_1COL,
    WIDTH_2COL,
    better_label,
    caption_line,
    direction_axis,
    letter,
    model_color,
    styled,
)
from yg_eo_soilnet.a1.plans import CONTROL, MIXED, MIXED_INDICATOR, MIXED_WEIGHTED
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL_GLOBAL, PER_CLASS_MEASURED, PER_CLASS_PREDICTED

#: The most basins figure C1 shows, one column each.
MAX_BASINS = 5
#: Colour and marker of each mixed kind: control as the local model, mixed as the global one.
KIND_STYLE = {
    CONTROL: (model_color(LOCAL_GLOBAL), "v"),
    MIXED: (model_color(GLOBAL), "D"),
    MIXED_WEIGHTED: (model_color(PER_CLASS_MEASURED), "s"),
    MIXED_INDICATOR: (model_color(PER_CLASS_PREDICTED), "^"),
}
ARMS = (MIXED, MIXED_WEIGHTED, MIXED_INDICATOR)


def kind_label(kind: str) -> str:
    """The plain name of a mixed kind (``labels.yml``, ``kinds``)."""
    return (load_labels().get("kinds") or {}).get(kind) or pretty_name(kind)


def basin_title(basin: str, status: pd.DataFrame) -> str:
    """The basin's name, and whether its transfer fails, when the main run's status is known."""
    if status is None or status.empty or basin not in set(status["basin"]):
        return str(basin)
    fails = bool(status.loc[status["basin"] == basin, "transfer_fails"].iloc[0])
    return f"{basin}\n" + ("transfer fails" if fails else "no significant transfer loss")


def draw_all(tables: dict, unmatched: pd.DataFrame, *, model: str = GLOBAL) -> dict:
    """Both figures and their captions, ``{name: (figure, caption)}``; a note where a figure has no data."""
    return {
        "figC1_mixed_arms": figure_mixed_arms(
            tables["mixed_arms"], tables["mixed_arms_paired"], tables["transfer_status"], model=model
        ),
        "figC2_logo_matched": figure_logo_matched(tables["logo_matched_draws"], unmatched, model=model),
    }


# --- C1 -----------------------------------------------------------------------------------------------


def arms_summary(arms: pd.DataFrame) -> pd.DataFrame:
    """Per run, basin, share and kind: moved points and RMSE, mean and min-max over the repeats."""
    rows = []
    for (run, basin, fraction), part in arms.groupby(["run", "basin", "fraction"], sort=True):
        for kind in ("logo", *KIND_STYLE, "local"):
            values = part[f"rmse_{kind}"].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "run": run,
                    "basin": basin,
                    "fraction": fraction,
                    "kind": kind,
                    "moved": float(part["n_moved"].mean()),
                    "mean": float(values.mean()),
                    "low": float(values.min()),
                    "high": float(values.max()),
                    "n_repeats": len(values),
                }
            )
    return pd.DataFrame(rows, columns=["run", "basin", "fraction", "kind", "moved", "mean", "low", "high", "n_repeats"])


def figure_mixed_arms(arms: pd.DataFrame, paired: pd.DataFrame, status: pd.DataFrame, *, model: str = GLOBAL):
    """Figure C1 and its caption."""
    if arms.empty:
        return _missing("No mixed plans in the compared runs."), "Not drawn: no mixed plans in the compared runs."
    # One run per basin: the first that has its mixed plans.
    first_run = arms.drop_duplicates("basin").set_index("basin")["run"]
    basins = sorted(first_run.index)[:MAX_BASINS]
    shown = arms[arms["basin"].isin(basins) & (arms["run"] == arms["basin"].map(first_run))]
    summary = arms_summary(shown)
    to_control = paired[(paired["b"] == CONTROL) & paired["basin"].isin(basins)] if not paired.empty else paired
    if not to_control.empty:
        to_control = to_control[to_control["run"] == to_control["basin"].map(first_run)]
    figure = draw_mixed_arms(summary, to_control, status, basins, model=model)

    parts = []
    for basin in basins:
        rows = to_control[to_control["basin"] == basin] if not to_control.empty else to_control
        texts = []
        for arm in ARMS:
            arm_rows = rows[rows["a"] == arm] if not rows.empty else rows
            if arm_rows.empty:
                continue
            better = int((arm_rows["high"] < 0).sum())
            texts.append(
                f"{kind_label(arm).lower()} − control {arm_rows['mean_difference'].mean():+.2f} {SOM_UNIT} on average, "
                f"interval below 0 at {better} of {len(arm_rows)} shares"
            )
        if texts:
            parts.append(ps.minus(f"{basin}: " + "; ".join(texts)))
    sources = sorted({run for run in shown["local_run"].dropna().astype(str) if run})
    local_text = (
        "Dashed: the local model of the whole basin on the same inputs as the curves' model (for Global: every "
        "covariate, no predicted CaCO₃/CEC) - fitted on the basin's own points only, each point predicted "
        "out of fold by a model that never saw its 10 km block - scored on the same remaining test points as the "
        "curves, not on the whole basin; it does not depend on the share moved, so it moves only with the test "
        f"points (from {', '.join(sources)})."
        if sources
        else "No compared run has the local model on these inputs, so there is no dashed line."
    )
    rule = (
        "A basin's title says 'transfer fails' when, in the main run (results/a1_xgb, plain leave-one-basin-out, "
        f"{pretty_name(model)} model, every input, against the local model on the same inputs), the transferred "
        "minus the local RMSE is above 0 with its 95 % "
        "block-bootstrap interval above 0 and a Holm-corrected p-value (across basins) below 0.05."
        if status is not None and not status.empty
        else "The main run (results/a1_xgb) is not among the compared runs, so the titles give the basin only."
    )
    caption = (
        f"Figure C1. {pretty_name(model)} model, every input. Upper row: test RMSE on the basin's remaining points "
        "against the number of its points moved into training, for the moved points alone (control), the other "
        "basins plus the moved points (mixed), the same with the moved points weighted up (weighted) or flagged by "
        "an input (indicator); mean over repeats, band min-max. Dotted: the model with no local points, on the same "
        "test points. " + local_text + " Lower row: each arm minus control on the same "
        "test points, mean over repeats, with a 95 % interval from resampling 10 km blocks of the test points of "
        "every repeat. " + rule + (" " + "; ".join(parts) + "." if parts else "")
    )
    return figure, caption


@styled
def draw_mixed_arms(summary: pd.DataFrame, paired: pd.DataFrame, status: pd.DataFrame, basins: list, *, model=GLOBAL):
    """Draw figure C1: one column per basin, RMSE on top, arm minus control below."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = len(basins)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(WIDTH_2COL if columns > 1 else WIDTH_1COL, 4.6),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 0.6]},
        squeeze=False,
        sharex="col",
    )
    drawn = set()
    for column, basin in enumerate(basins):
        upper, lower = axes[0, column], axes[1, column]
        part = summary[summary["basin"] == basin]
        for kind, (color, marker) in KIND_STYLE.items():
            line = part[part["kind"] == kind].sort_values("moved")
            if line.empty:
                continue
            upper.fill_between(line["moved"], line["low"], line["high"], color=color, alpha=0.15, lw=0)
            upper.plot(line["moved"], line["mean"], color=color, marker=marker, ms=3.2, lw=1.2, mec="white", mew=0.4)
            drawn.add(kind)
        for kind, style, color in (("logo", ":", ps.INK_2), ("local", "--", model_color(LOCAL_GLOBAL))):
            line = part[part["kind"] == kind].sort_values("moved")
            if not line.empty:
                upper.plot(line["moved"], line["mean"], color=color, ls=style, lw=1.0)
                drawn.add(kind)
        upper.set_xscale("log")
        upper.set_title(basin_title(basin, status), fontsize=8)
        upper.set_ylabel(better_label("RMSE", "lower", SOM_UNIT) if column == 0 else "")

        rows = paired[paired["basin"] == basin] if not paired.empty else paired
        offsets = {MIXED: 0.94, MIXED_WEIGHTED: 1.0, MIXED_INDICATOR: 1.06}
        for arm in ARMS:
            arm_rows = rows[rows["a"] == arm].sort_values("n_moved") if not rows.empty else rows
            if arm_rows.empty:
                continue
            color, marker = KIND_STYLE[arm]
            x = arm_rows["n_moved"].to_numpy(float) * offsets[arm]
            y = arm_rows["mean_difference"].to_numpy(float)
            errors = np.vstack(
                [
                    np.clip(y - arm_rows["low"].to_numpy(float), 0, None),
                    np.clip(arm_rows["high"].to_numpy(float) - y, 0, None),
                ]
            )
            lower.errorbar(
                x, y, yerr=errors, color=color, marker=marker, ms=3.0, lw=0.9, capsize=1.5, mec="white", mew=0.4
            )
        direction_axis(
            lower, above="control better", below="arm better", zero="no difference", labels=column == columns - 1
        )
        lower.set_xscale("log")
        lower.set_xlabel("Points moved into training")
        lower.set_ylabel(f"Arm − control RMSE ({SOM_UNIT})" if column == 0 else "")
        letter(upper, "abcdefghij"[column])
    handles = [
        Line2D([], [], color=color, marker=marker, lw=1.2, ms=3.5, label=kind_label(kind))
        for kind, (color, marker) in KIND_STYLE.items()
        if kind in drawn
    ]
    if "logo" in drawn:
        handles.append(Line2D([], [], color=ps.INK_2, ls=":", lw=1.0, label="No local points"))
    if "local" in drawn:
        handles.append(
            Line2D(
                [],
                [],
                color=model_color(LOCAL_GLOBAL),
                ls="--",
                lw=1.0,
                label=pretty_name(LOCAL_GLOBAL) + ", same inputs",
            )
        )
    figure.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 6), frameon=False)
    caption_line(
        figure, f"{pretty_name(model)} model. Bands: min-max over repeats. Bars: 95 % block-bootstrap intervals."
    )
    return figure


# --- C2 -----------------------------------------------------------------------------------------------


def figure_logo_matched(draws: pd.DataFrame, unmatched: pd.DataFrame, *, model: str = GLOBAL):
    """Figure C2 and its caption."""
    if draws.empty:
        return (
            _missing("No matched leave-one-basin-out plans in the compared runs."),
            "Not drawn: no matched leave-one-basin-out plans in the compared runs.",
        )
    order = list(unmatched.sort_values("rmse")["basin"]) if not unmatched.empty else []
    basins = [basin for basin in order if basin in set(draws["basin"])]
    basins += sorted(set(draws["basin"]) - set(basins))
    figure = draw_logo_matched(draws, unmatched, basins, model=model)
    parts = []
    plain = unmatched.set_index("basin") if not unmatched.empty else unmatched
    for basin in basins:
        texts = []
        for size, part in draws[draws["basin"] == basin].groupby("training_size"):
            texts.append(
                f"{part['rmse'].mean():.2f} ({part['rmse'].min():.2f}–{part['rmse'].max():.2f}) at {int(size):,} points"
            )
        if not plain.empty and basin in plain.index:
            reference = plain.loc[basin]
            texts.append(f"{reference['rmse']:.2f} unmatched, at {int(reference['training_size']):,}")
        parts.append(f"{basin}: " + ", ".join(texts))
    caption = (
        f"Figure C2. {pretty_name(model)} model, every input, leave one basin out. Test RMSE of each held-out basin "
        "against the size of the training set: each matched draw (training cut to the same size for every basin, "
        "whole 10 km blocks drawn at random) a point, the mean over draws a line, which ends at the unmatched plan "
        "(every other basin's points) at its own training size. Basins ordered by unmatched RMSE. "
        + "; ".join(parts)
        + "."
    )
    return figure, ps.minus(caption)


@styled
def draw_logo_matched(draws: pd.DataFrame, unmatched: pd.DataFrame, basins: list, *, model=GLOBAL):
    """Draw figure C2: per basin, RMSE against the training size."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = min(len(basins), 5)
    rows = int(np.ceil(len(basins) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(WIDTH_2COL if columns > 1 else WIDTH_1COL, 2.2 * rows + 0.8),
        layout="constrained",
        squeeze=False,
    )
    color = model_color(model)
    plain = unmatched.set_index("basin") if not unmatched.empty else unmatched
    for position, basin in enumerate(basins):
        axis = axes[position // columns, position % columns]
        part = draws[draws["basin"] == basin]
        jitter = np.random.default_rng(position).uniform(-0.01, 0.01, len(part)) * part["training_size"].to_numpy(float)
        axis.scatter(part["training_size"] + jitter, part["rmse"], s=9, color=color, alpha=0.55, lw=0)
        means = part.groupby("training_size")["rmse"].mean().sort_index()
        x, y = list(means.index.astype(float)), list(means.to_numpy())
        if not plain.empty and basin in plain.index and np.isfinite(plain.loc[basin, "training_size"]):
            x.append(float(plain.loc[basin, "training_size"]))
            y.append(float(plain.loc[basin, "rmse"]))
            axis.plot(x[-1], y[-1], marker="*", ms=8, color=ps.INK_2, ls="none", zorder=4)
        axis.plot(x, y, color=color, lw=1.2, marker="o", ms=3, zorder=3)
        axis.set_title(str(basin), fontsize=8)
        axis.set_xlabel("Training points")
        axis.set_ylabel(better_label("RMSE", "lower", SOM_UNIT) if position % columns == 0 else "")
        letter(axis, "abcdefghijklmnop"[position])
    for empty in range(len(basins), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Line2D([], [], color=color, marker="o", ls="none", alpha=0.55, ms=3.5, label="Matched draw"),
        Line2D([], [], color=color, marker="o", lw=1.2, ms=3, label="Mean over draws"),
        Line2D([], [], color=ps.INK_2, marker="*", ls="none", ms=7, label="Unmatched (every other basin)"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    caption_line(figure, f"{pretty_name(model)} model, every input. Points: draws; line: mean over draws.")
    return figure
