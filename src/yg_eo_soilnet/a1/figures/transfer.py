"""Figures 8 and 9: what is lost when a model is used in a basin it never saw, and what local samples win back.

Figure 8 compares, per held-out basin, the model trained on the other basins ("transferred") with a
model cross-validated on the basin's own points ("local"), both on the same points, with every input
and with each covariate group alone. Figure 9 moves a growing number of the basin's points into
training ("mixed") and compares with a model trained on those moved points alone ("control").
"""

from __future__ import annotations

import numpy as np
from matplotlib import patheffects
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import (
    SOM_UNIT,
    direction_axis,
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
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL
from yg_eo_soilnet.datamodules.splitting import TEST
from yg_eo_soilnet.metrics import regression_metrics

#: Line colours of figure 9: mixed in the global model's colour, control in the local one's.
MIXED_COLOR = model_color(GLOBAL)
CONTROL_COLOR = model_color(LOCAL)


# --- figure 8 ---------------------------------------------------------------------------------------


def transfer_points(inputs: FigureInputs) -> pd.DataFrame:
    """Per held-out basin, every input: local and transferred RMSE and R², n, and the share of training removed.

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
    keep = ["region", "n", "rmse_local", "rmse_transferred", "r2_local", "r2_transferred", "rmse_drop", "rmse_drop_pct"]
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


def _no_transfer_message(inputs: FigureInputs) -> str:
    """Why figure 8 cannot be drawn, true for this run."""
    if LOGO not in set(inputs.predictions["kind"]):
        return "This run has no leave-one-basin-out plans: run run_a1.py with --kinds logo."
    if LOCAL not in set(inputs.predictions["model"]):
        return (
            "No local model in this run: its leave-one-basin-out plans were run before the local model was "
            "always fitted, or every held-out basin has fewer than 3 blocks. Run run_a1.py again (it resumes)."
        )
    return "No held-out basin has both a transferred and a local score."


def figure_transfer(inputs: FigureInputs):
    """Figure 8: (a) transferred against local RMSE per basin; (b) per covariate group, when the run has them."""
    points = transfer_points(inputs)
    if points.empty:
        return _missing(_no_transfer_message(inputs))
    grid = transfer_grid(inputs)
    if list(grid.columns) == [COMBINED_GROUP]:
        grid = pd.DataFrame()  # every input only: panel a says it all
    figure = draw_transfer(points, grid)
    worse = int((points["rmse_drop"] > 0).sum())
    caption = (
        f"Figure 8. (a) RMSE of the model trained on the other basins against a model trained on the basin's own points, "
        f"same test points, every input. Transfer is worse in {worse} of {len(points)} basins; the change ranges from "
        + ps.minus(f"{points['rmse_drop'].min():+.2f} to {points['rmse_drop'].max():+.2f}")
        + f" {SOM_UNIT}."
        + (
            " (b) The same change for each covariate group alone: each cell compares two models given the same "
            "inputs, so it shows how well what the group teaches carries over to an unseen basin."
            if not grid.empty
            else ""
        )
    )
    data = {"fig08_transfer_points": points}
    if not grid.empty:
        data["fig08_transfer_grid"] = grid.rename_axis("basin").reset_index()
    return attach(figure, data, caption)


@styled
def draw_transfer(points: pd.DataFrame, grid: pd.DataFrame):
    """Draw figure 8: (a) from :func:`transfer_points`; (b) from :func:`transfer_grid` when not empty."""
    import matplotlib.pyplot as plt

    if points.empty:
        return _missing("No transfer scores to draw.")
    if grid.empty:
        figure, scatter = plt.subplots(figsize=(WIDTH_1COL, 3.5), layout="constrained")
        heat = None
    else:
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

    if heat is not None:
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

#: The shares named on figure 9's top axis, when the run has them.
SHARE_TICKS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
#: Figure 9's "plateau": the mean mixed RMSE over the shares at least this large.
PLATEAU_FROM = 0.5


def _rmse(part: pd.DataFrame) -> float:
    return regression_metrics(part["observed"].to_numpy(dtype=float), part["predicted"].to_numpy(dtype=float)).get(
        "rmse_test", np.nan
    )


def mixed_curves(inputs: FigureInputs, *, model: str = GLOBAL, group: str = COMBINED_GROUP) -> pd.DataFrame:
    """Test RMSE of every mixed and control plan of ``model``, by basin and number of moved samples.

    Scored on the basin's points left out of training. The number of moved samples is the matching
    control plan's ``n_train`` (same basin, share and repeat); the share when the plan index does
    not hold it.

    Returns
    -------
    pandas.DataFrame
        ``region, kind, fraction, repeat, moved, rmse, n``.
    """
    frame = inputs.predictions
    rows = frame[
        (frame["split"] == TEST)
        & frame["kind"].isin([MIXED, CONTROL])
        & (frame["model"] == model)
        & (frame["covariate_group"] == group)
    ]
    columns = ["region", "kind", "fraction", "repeat", "moved", "rmse", "n"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    scores = [
        {
            "plan_id": plan_id,
            "region": str(region),
            "kind": kind,
            "fraction": float(fraction),
            "repeat": repeat,
            "rmse": _rmse(part),
            "n": len(part),
        }
        for (plan_id, kind, region, fraction, repeat), part in rows.groupby(
            ["plan_id", "kind", "held_out", "fraction", "repeat"]
        )
    ]
    scores = pd.DataFrame(scores)
    index = inputs.index
    control = index[index["kind"] == CONTROL][["held_out", "fraction", "repeat", "n_train"]].rename(
        columns={"held_out": "region", "n_train": "moved"}
    )
    control = control.assign(region=control["region"].astype(str), fraction=control["fraction"].astype(float))
    scores = scores.merge(control, on=["region", "fraction", "repeat"], how="left")
    scores["moved"] = scores["moved"].fillna(scores["fraction"])
    return scores[columns].sort_values(["region", "kind", "moved", "repeat"])


def mixed_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max of the test RMSE over repeats, per basin, kind and number moved; mean test points."""
    grouped = curves.groupby(["region", "kind", "moved"])
    summary = grouped["rmse"].agg(mean="mean", low="min", high="max", repeats="count")
    summary["n"] = grouped["n"].mean()
    summary["fraction"] = grouped["fraction"].first()
    return summary.reset_index()


def mixed_paired(curves: pd.DataFrame) -> pd.DataFrame:
    """RMSE(mixed) - RMSE(control) per basin, share and repeat: the two share their test points.

    Returns
    -------
    pandas.DataFrame
        ``region, fraction, moved, repeat, rmse_mixed, rmse_control, difference, n``; a negative
        difference means the other basins help.
    """
    keys = ["region", "fraction", "repeat"]
    mixed = curves[curves["kind"] == MIXED].set_index(keys)
    control = curves[curves["kind"] == CONTROL].set_index(keys)
    both = mixed[["rmse", "n"]].join(control[["rmse", "moved"]], lsuffix="_mixed", rsuffix="_control", how="inner")
    both = both.rename(columns={"rmse_mixed": "rmse_mixed", "rmse_control": "rmse_control"})
    both["difference"] = both["rmse_mixed"] - both["rmse_control"]
    out = both.reset_index()[["region", "fraction", "moved", "repeat", "rmse_mixed", "rmse_control", "difference", "n"]]
    return out.sort_values(["region", "moved", "repeat"])


def mixed_references(inputs: FigureInputs, *, model: str = GLOBAL) -> pd.DataFrame:
    """Per held-out basin: ``model``'s RMSE with no local points (leave one basin out) and the local model's.

    Both on every point of the basin (every input).
    """
    frame = inputs.predictions
    logo = frame[(frame["split"] == TEST) & (frame["kind"] == LOGO) & (frame["covariate_group"] == COMBINED_GROUP)]
    rows = []
    for region, part in logo.groupby("held_out"):
        transferred, local = part[part["model"] == model], part[part["model"] == LOCAL]
        rows.append(
            {
                "region": str(region),
                "rmse_transferred": _rmse(transferred) if len(transferred) else np.nan,
                "rmse_local": _rmse(local) if len(local) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["region", "rmse_transferred", "rmse_local"])


def figure_mixed(inputs: FigureInputs, *, model: str = GLOBAL):
    """Figure 9: test RMSE against the number of the basin's points moved into training, and mixed - control."""
    curves = mixed_curves(inputs, model=model)
    if curves.empty:
        if MIXED in set(inputs.predictions["kind"]):
            return _missing(f"No mixed or control predictions of the {pretty_name(model)} model in this run.")
        return _missing("This run has no mixed plans: run run_a1.py with --kinds mixed (it brings the control plans).")
    summary = mixed_summary(curves)
    paired = mixed_paired(curves)
    references = mixed_references(inputs, model=model)
    figure = draw_mixed(summary, references, paired, model=model)
    parts = []
    for region, part in paired.groupby("region"):
        reference = references.set_index("region").reindex([region]).iloc[0] if not references.empty else None
        mixed = summary[(summary["region"] == region) & (summary["kind"] == MIXED)]
        plateau = mixed[mixed["fraction"] >= PLATEAU_FROM]["mean"].mean()
        text = (
            f"{region}: control beats mixed in {(part['difference'] > 0).mean():.0%} of {len(part)} paired repeats; "
            f"mean mixed − control {part['difference'].mean():+.2f} {SOM_UNIT}"
        )
        if reference is not None and np.isfinite(reference["rmse_transferred"]):
            text += f"; {reference['rmse_transferred']:.2f} with no local points"
        if np.isfinite(plateau):
            text += f", {plateau:.2f} at the plateau (shares ≥ {PLATEAU_FROM:.0%})"
        parts.append(ps.minus(text))
    caption = (
        f"Figure 9. {pretty_name(model)} model, every input. Upper row: test RMSE on the basin's remaining points as "
        "more of its points are moved into training - mixed (the other basins plus the moved points) and control "
        "(the moved points alone); mean over repeats, band min-max. Each share tests a different set of points, so "
        "the upper row compares test sets as well as models. Lower row: the paired difference mixed − control on "
        "the same test points of each repeat, the valid comparison. " + "; ".join(parts) + "."
    )
    return attach(
        figure,
        {"fig09_mixed_repeats": curves, "fig09_mixed_summary": summary, "fig09_mixed_paired": paired},
        caption,
    )


def _share_axis(axis, summary: pd.DataFrame) -> None:
    """A top axis naming the share of the basin moved, at the same places as the moved counts."""
    pairs = summary.drop_duplicates("moved").sort_values("moved")
    moved, share = pairs["moved"].to_numpy(dtype=float), pairs["fraction"].to_numpy(dtype=float)
    if len(moved) < 2:
        return
    log_moved = np.log(moved)

    def forward(values):
        return np.interp(np.log(np.clip(values, 1e-9, None)), log_moved, share)

    def backward(values):
        return np.exp(np.interp(values, share, log_moved))

    top = axis.secondary_xaxis("top", functions=(forward, backward))
    ticks = [value for value in SHARE_TICKS if share.min() - 1e-9 <= value <= share.max() + 1e-9]
    from matplotlib.ticker import NullLocator

    top.set_xticks(ticks, [f"{value:.0%}" for value in ticks])
    top.xaxis.set_minor_locator(NullLocator())  # only the shares named, no automatic log-scale labels
    top.tick_params(labelsize=7, length=2)
    top.set_xlabel("Share of the basin moved", fontsize=7)


@styled
def draw_mixed(summary: pd.DataFrame, references: pd.DataFrame, paired: pd.DataFrame, *, model: str = GLOBAL):
    """Draw figure 9: per basin, (upper) mixed and control RMSE with their range; (lower) the paired difference."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    if summary.empty:
        return _missing("No mixed-training plans to draw.")
    regions = sorted(summary["region"].unique())
    columns = min(3, len(regions))
    blocks = int(np.ceil(len(regions) / columns))
    mosaic, ratios = [], []
    for block in range(blocks):
        mosaic += [[f"rmse_{block}_{c}" for c in range(columns)], [f"diff_{block}_{c}" for c in range(columns)]]
        ratios += [1.0, 0.55]
    figure, grid = plt.subplot_mosaic(
        mosaic,
        figsize=(WIDTH_2COL if columns > 1 else WIDTH_1COL, 3.6 * blocks + 0.9),
        layout="constrained",
        gridspec_kw={"height_ratios": ratios},
    )
    references = references.set_index("region") if not references.empty else references
    drawn = {"mixed": False, "control": False, "transferred": False, "local": False}
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for position, region in enumerate(regions):
        block, c = divmod(position, columns)
        upper, lower = grid[f"rmse_{block}_{c}"], grid[f"diff_{block}_{c}"]
        lower.sharex(upper)
        part = summary[summary["region"] == region]
        y_values = [part["low"], part["high"]]
        for kind, color, marker in ((MIXED, MIXED_COLOR, "D"), (CONTROL, CONTROL_COLOR, "v")):
            line = part[part["kind"] == kind].sort_values("moved")
            if line.empty:
                continue
            upper.fill_between(line["moved"], line["low"], line["high"], color=color, alpha=0.18, lw=0)
            upper.plot(
                line["moved"],
                line["mean"],
                color=color,
                lw=1.4,
                marker=marker,
                ms=3.5,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
            drawn["mixed" if kind == MIXED else "control"] = True
        if not references.empty and region in references.index:
            reference = references.loc[region]
            if np.isfinite(reference["rmse_transferred"]):
                upper.axhline(reference["rmse_transferred"], color=MIXED_COLOR, lw=0.9, ls=":")
                y_values.append(pd.Series([reference["rmse_transferred"]]))
                drawn["transferred"] = True
            if np.isfinite(reference["rmse_local"]):
                upper.axhline(reference["rmse_local"], color=CONTROL_COLOR, lw=0.9, ls="--")
                y_values.append(pd.Series([reference["rmse_local"]]))
                drawn["local"] = True
        # Every line drawn lies inside the limits, reference lines included.
        values = pd.concat(y_values).to_numpy(dtype=float)
        low, high = np.nanmin(values), np.nanmax(values)
        pad = 0.08 * (high - low if high > low else 1.0)
        upper.set_ylim(low - pad, high + pad * 2.5)
        # The test points left, at the first and the last share.
        mixed = part[part["kind"] == MIXED].sort_values("moved")
        for row, ha in ((mixed.iloc[0], "left"), (mixed.iloc[-1], "right")) if len(mixed) else ():
            upper.annotate(
                f"{int(round(row['n'])):,} test points",
                (row["moved"], row["high"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=7,
                color=ps.INK_2,
                path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
            )
        upper.set_xscale("log")
        upper.set_title(region, fontsize=8, pad=18)
        upper.set_ylabel(better_label("Test RMSE", "lower", SOM_UNIT).replace(", lower", ",\nlower") if c == 0 else "")
        upper.tick_params(labelbottom=False)
        _share_axis(upper, part)
        letter(upper, next(letters))

        pairs = paired[paired["region"] == region]
        lower.scatter(pairs["moved"], pairs["difference"], s=7, color=ps.MUTED, alpha=0.7, lw=0, zorder=2)
        means = pairs.groupby("moved")["difference"].mean()
        lower.plot(means.index, means.to_numpy(), color=ps.INK, lw=1.3, zorder=3)
        direction_axis(lower, above="control better", below="mixed better", zero="", labels=c == columns - 1)
        lower.set_ylabel(f"Mixed − control\nRMSE ({SOM_UNIT})" if c == 0 else "")
        moved = sorted(part["moved"].unique())
        lower.xaxis.set_minor_locator(NullLocator())
        # Every count gets a tick; a label only where it clears the last one (log scale crowds the top).
        shown, last = [], -np.inf
        for value in moved:
            keep = np.log10(value) - last >= 0.11
            shown.append(f"{int(value):,}" if keep else "")
            last = np.log10(value) if keep else last
        lower.set_xticks(moved, shown, rotation=45, ha="right", rotation_mode="anchor")
        lower.set_xlabel("Basin points moved into training (log scale)")
        letter(lower, next(letters))
    for position in range(len(regions), blocks * columns):
        block, c = divmod(position, columns)
        grid[f"rmse_{block}_{c}"].set_axis_off()
        grid[f"diff_{block}_{c}"].set_axis_off()
    handles = []
    if drawn["mixed"]:
        handles.append(
            Line2D([], [], color=MIXED_COLOR, lw=1.4, marker="D", ms=3.5, label="Mixed: other basins + moved points")
        )
    if drawn["control"]:
        handles.append(
            Line2D([], [], color=CONTROL_COLOR, lw=1.4, marker="v", ms=3.5, label="Control: moved points alone")
        )
    handles.append(Patch(facecolor=ps.MUTED, alpha=0.3, label="Range over repeats"))
    if drawn["transferred"]:
        handles.append(Line2D([], [], color=MIXED_COLOR, lw=0.9, ls=":", label="No local points (leave one basin out)"))
    if drawn["local"]:
        handles.append(
            Line2D([], [], color=CONTROL_COLOR, lw=0.9, ls="--", label="Local model, all the basin's points")
        )
    handles.append(Line2D([], [], ls="", marker="o", color=ps.MUTED, ms=3, label="Paired difference, one repeat"))
    figure.legend(handles=handles, loc="outside upper center", ncol=2 if columns > 1 else 1, frameon=False)
    caption_line(
        figure,
        f"Model: {pretty_name(model)}, every input. Upper: mean over repeats, band min to max. Lower: one point per "
        "repeat, line their mean; mixed and control share their test points within a repeat.",
    )
    return figure
