"""Figures 8 and 9: what is lost when a model is used in a basin it never saw, and what local samples win back.

Figure 8 compares, per held-out basin, the model trained on the other basins ("transferred") with a
model cross-validated on the basin's own points ("local"), both on the same points, with every input
and with each covariate group alone. Figure 9 moves a growing number of the basin's points into
training ("mixed") and compares with a model trained on those moved points alone ("control").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import (
    SOM_UNIT,
    WIDTH_1COL,
    WIDTH_2COL,
    better_label,
    caption_line,
    place_labels,
    letter,
    model_color,
    model_marker,
    styled,
)
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED
from yg_eo_soilnet.a1.stage2 import GLOBAL_AUX, LOCAL
from yg_eo_soilnet.datamodules.splitting import TEST
from yg_eo_soilnet.metrics import regression_metrics

#: Line colours of figure 9.
MIXED_COLOR = model_color(GLOBAL_AUX)
CONTROL_COLOR = model_color(LOCAL)


# --- figure 8 ---------------------------------------------------------------------------------------


def transfer_points(inputs: FigureInputs) -> pd.DataFrame:
    """Per held-out basin, every input: local and transferred RMSE, n, and the share of training removed.

    The share removed is the basin's points over all points of its leave-one-basin-out plan,
    ``n_test / (n_train + n_test)`` from the plan index.
    """
    transfer = inputs.transfer
    if transfer.empty:
        return pd.DataFrame()
    logo = transfer[
        (transfer["kind"] == LOGO) & transfer["has_local"] & (transfer["covariate_group"] == COMBINED_GROUP)
    ]
    index = inputs.index
    plans = index[index["kind"] == LOGO].assign(region=lambda frame: frame["held_out"].astype(str))
    plans = plans.assign(share_removed=plans["n_test"] / (plans["n_train"] + plans["n_test"]))
    keep = ["region", "n", "rmse_local", "rmse_transferred", "rmse_drop", "rmse_drop_pct"]
    return logo.assign(region=logo["region"].astype(str))[keep].merge(
        plans[["region", "share_removed"]], on="region", how="left"
    )


def transfer_grid(inputs: FigureInputs) -> pd.DataFrame:
    """The RMSE change (transferred - local) per held-out basin (rows) and covariate group (columns)."""
    transfer = inputs.transfer
    if transfer.empty:
        return pd.DataFrame()
    logo = transfer[(transfer["kind"] == LOGO) & transfer["has_local"]]
    if logo.empty:
        return pd.DataFrame()
    grid = logo.pivot_table(index="region", columns="covariate_group", values="rmse_drop", aggfunc="mean")
    groups = [COMBINED_GROUP] + [group for group in grid.columns if group != COMBINED_GROUP]
    return grid[[group for group in groups if group in grid.columns]]


def figure_transfer(inputs: FigureInputs):
    """Figure 8: (a) transferred against local RMSE per basin; (b) the RMSE change per basin and group."""
    points = transfer_points(inputs)
    if points.empty:
        return _missing(
            "No transfer scores: run run_a1.py with --kinds logo --groups (the local model needs --groups)."
        )
    grid = transfer_grid(inputs)
    figure = draw_transfer(points, grid)
    worse = int((points["rmse_drop"] > 0).sum())
    caption = (
        f"Figure 8. (a) RMSE of the model trained on the other basins against a model trained on the basin's own points, "
        f"same test points, every input. Transfer is worse in {worse} of {len(points)} basins; the change ranges from "
        + ps.minus(f"{points['rmse_drop'].min():+.2f} to {points['rmse_drop'].max():+.2f}")
        + f" {SOM_UNIT}. (b) The same change for each "
        "covariate group alone: each cell compares two models given the same inputs, so it shows how well what the "
        "group teaches carries over to an unseen basin."
    )
    return attach(
        figure,
        {"fig08_transfer_points": points, "fig08_transfer_grid": grid.rename_axis("basin").reset_index()},
        caption,
    )


@styled
def draw_transfer(points: pd.DataFrame, grid: pd.DataFrame):
    """Draw figure 8 from :func:`transfer_points` and :func:`transfer_grid`."""
    import matplotlib.pyplot as plt

    if points.empty:
        return _missing("No transfer scores to draw.")
    figure, (scatter, heat) = plt.subplots(
        1, 2, figsize=(WIDTH_2COL, 3.4), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1.1]}
    )
    values = pd.concat([points["rmse_local"], points["rmse_transferred"]]).dropna()
    low, high = float(values.min()) * 0.9, float(values.max()) * 1.1
    scatter.fill_between([low, high], [low, high], [high, high], color=ps.MUTED, alpha=0.12, lw=0, zorder=0)
    scatter.text(
        low + 0.04 * (high - low),
        high - 0.04 * (high - low),
        "worse when transferred",
        ha="left",
        va="top",
        fontsize=7,
        color=ps.INK_2,
    )
    scatter.text(
        high - 0.04 * (high - low),
        low + 0.04 * (high - low),
        "better when transferred",
        ha="right",
        va="bottom",
        fontsize=7,
        color=ps.INK_2,
    )
    scatter.plot([low, high], [low, high], color=ps.INK_2, lw=0.8, ls="--", zorder=1)
    scatter.scatter(
        points["rmse_local"],
        points["rmse_transferred"],
        s=26,
        color=model_color(GLOBAL_AUX),
        marker=model_marker(GLOBAL_AUX),
        edgecolor="white",
        lw=0.5,
        zorder=3,
    )
    scatter.set_xlim(low, high)
    scatter.set_ylim(low, high)
    scatter.set_aspect("equal", adjustable="box")
    scatter.set_xlabel(f"Local RMSE ({SOM_UNIT}), basin's own points")
    scatter.set_ylabel(f"Transferred RMSE ({SOM_UNIT}), other basins")
    letter(scatter, "a")

    if grid.empty:
        heat.set_axis_off()
    else:
        matrix = grid.to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        image = heat.imshow(matrix, cmap=ps.diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    heat.text(
                        column,
                        row,
                        ps.minus(f"{value:+.1f}"),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.6 * limit else ps.INK,
                        fontweight="bold" if grid.columns[column] == COMBINED_GROUP else "normal",
                    )
        heat.set_xticks(range(matrix.shape[1]), [pretty_name(group) for group in grid.columns], rotation=30, ha="right")
        heat.set_yticks(range(matrix.shape[0]), [str(region) for region in grid.index])
        heat.tick_params(length=0)
        heat.grid(False)
        for spine in heat.spines.values():
            spine.set_visible(False)
        heat.set_xlabel("Inputs")
        colorbar = figure.colorbar(image, ax=heat, shrink=0.85, pad=0.02, aspect=25)
        colorbar.set_label(f"RMSE change ({SOM_UNIT}), transferred − local\n> 0: worse when transferred")
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0)
        letter(heat, "b")
    # Last, on the final layout: the colour bar of panel b reflows both panels.
    texts = [
        f"{row.region}\nn = {int(row.n):,}" + (f", {row.share_removed:.0%}" if pd.notna(row.share_removed) else "")
        for row in points.itertuples()
    ]
    place_labels(scatter, points["rmse_local"], points["rmse_transferred"], texts, avoid=list(scatter.texts))
    caption_line(
        figure,
        "(a) Labels: basin, its points (n) and their share of all points - what holding it out removes from training. "
        "Model: Global + predicted CaCO₃/CEC, trained on the other basins, against the same inputs fitted inside the basin (local).",
    )
    return figure


# --- figure 9 ---------------------------------------------------------------------------------------


def mixed_curves(inputs: FigureInputs, *, group: str = COMBINED_GROUP) -> pd.DataFrame:
    """Test RMSE of every mixed and control plan, by basin and number of moved samples.

    Model ``global_aux`` on ``group``'s inputs, scored on the basin's points left out of training.
    The number of moved samples is the matching control plan's ``n_train`` (same basin, share and
    repeat); the share is used when the plan index does not hold it.

    Returns
    -------
    pandas.DataFrame
        ``region, kind, fraction, repeat, moved, rmse, n``.
    """
    frame = inputs.predictions
    rows = frame[
        (frame["split"] == TEST)
        & frame["kind"].isin([MIXED, CONTROL])
        & (frame["model"] == GLOBAL_AUX)
        & (frame["covariate_group"] == group)
    ]
    if rows.empty:
        return pd.DataFrame(columns=["region", "kind", "fraction", "repeat", "moved", "rmse", "n"])
    scores = []
    for (plan_id, kind, region, fraction, repeat), part in rows.groupby(
        ["plan_id", "kind", "held_out", "fraction", "repeat"]
    ):
        metrics = regression_metrics(part["observed"].to_numpy(dtype=float), part["predicted"].to_numpy(dtype=float))
        scores.append(
            {
                "plan_id": plan_id,
                "region": str(region),
                "kind": kind,
                "fraction": float(fraction),
                "repeat": repeat,
                "rmse": metrics.get("rmse_test", np.nan),
                "n": int(metrics.get("n_test", 0)),
            }
        )
    scores = pd.DataFrame(scores)
    index = inputs.index
    control = index[index["kind"] == CONTROL][["held_out", "fraction", "repeat", "n_train"]].rename(
        columns={"held_out": "region", "n_train": "moved"}
    )
    control = control.assign(region=control["region"].astype(str), fraction=control["fraction"].astype(float))
    scores = scores.merge(control, on=["region", "fraction", "repeat"], how="left")
    scores["moved"] = scores["moved"].fillna(scores["fraction"])
    return scores[["region", "kind", "fraction", "repeat", "moved", "rmse", "n"]].sort_values(
        ["region", "kind", "moved", "repeat"]
    )


def mixed_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max of the test RMSE over repeats, per basin, kind and number moved."""
    return (
        curves.groupby(["region", "kind", "moved"])["rmse"]
        .agg(mean="mean", low="min", high="max", repeats="count")
        .reset_index()
    )


def figure_mixed(inputs: FigureInputs):
    """Figure 9: test RMSE against the number of the basin's points moved into training."""
    curves = mixed_curves(inputs)
    if curves.empty:
        return _missing("No mixed-training plans: run run_a1.py with --kinds mixed (it brings the control plans).")
    summary = mixed_summary(curves)
    references = transfer_points(inputs)
    figure = draw_mixed(summary, references)
    first = summary[summary["kind"] == MIXED].sort_values("moved").groupby("region").first()
    caption = (
        "Figure 9. Test RMSE on the basin's remaining points as more of its points are moved into training, "
        "every input. Mixed: the other basins plus the moved points; control: the moved points alone. "
        + (
            "; ".join(
                f"{region}: {row['mean']:.2f} {SOM_UNIT} with {int(row['moved'])} moved"
                for region, row in first.iterrows()
            )
        )
        + ". Horizontal lines: no local points (leave one basin out) and the local model on all the basin's points."
    )
    return attach(figure, {"fig09_mixed_repeats": curves, "fig09_mixed_summary": summary}, caption)


@styled
def draw_mixed(summary: pd.DataFrame, references: pd.DataFrame):
    """Draw figure 9: one panel per basin, mixed and control lines with their min-max band over repeats."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return _missing("No mixed-training plans to draw.")
    regions = sorted(summary["region"].unique())
    columns = min(3, len(regions))
    rows = int(np.ceil(len(regions) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(WIDTH_2COL if columns > 1 else WIDTH_1COL, 2.4 * rows + 0.9),
        layout="constrained",
        squeeze=False,
    )
    references = references.set_index("region") if not references.empty else references
    for position, (axis, region) in enumerate(zip(axes.flat, regions)):
        for kind, color, marker in ((MIXED, MIXED_COLOR, "D"), (CONTROL, CONTROL_COLOR, "v")):
            line = summary[(summary["region"] == region) & (summary["kind"] == kind)].sort_values("moved")
            if line.empty:
                continue
            axis.fill_between(line["moved"], line["low"], line["high"], color=color, alpha=0.18, lw=0)
            axis.plot(
                line["moved"],
                line["mean"],
                color=color,
                lw=1.4,
                marker=marker,
                ms=3.5,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
        if not references.empty and region in references.index:
            reference = references.loc[region]
            axis.axhline(reference["rmse_transferred"], color=MIXED_COLOR, lw=0.9, ls=":")
            axis.axhline(reference["rmse_local"], color=CONTROL_COLOR, lw=0.9, ls="--")
        axis.set_title(region, fontsize=8)
        if position % columns == 0:
            axis.set_ylabel(better_label("Test RMSE", "lower", SOM_UNIT).replace(", lower", ",\nlower"))
        axis.set_xlabel("Basin points moved into training")
        letter(axis, "abcdefghijkl"[position])
    for axis in list(axes.flat)[len(regions) :]:
        axis.set_axis_off()
    handles = [
        Line2D([], [], color=MIXED_COLOR, lw=1.4, marker="D", ms=3.5, label="Mixed: other basins + moved points"),
        Line2D([], [], color=CONTROL_COLOR, lw=1.4, marker="v", ms=3.5, label="Control: moved points alone"),
        Patch(facecolor=ps.MUTED, alpha=0.3, label="Range over repeats"),
        Line2D([], [], color=MIXED_COLOR, lw=0.9, ls=":", label="No local points (leave one basin out)"),
        Line2D([], [], color=CONTROL_COLOR, lw=0.9, ls="--", label="Local model, all the basin's points"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3 if columns > 1 else 1, frameon=False)
    caption_line(
        figure, "Lines: mean over repeats; bands: min to max. Model: Global + predicted CaCO₃/CEC, every input."
    )
    return figure
