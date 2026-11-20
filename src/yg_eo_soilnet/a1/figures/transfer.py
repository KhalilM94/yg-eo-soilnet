"""Figures 8 and 9: what is lost when a model is used in a basin it never saw, and what local samples win back.

Figure 8 compares, per held-out basin, the model trained on the other basins ("transferred") with a
model cross-validated on the basin's own points ("local"), both on the same points, with every input
and with each covariate group alone. Figure 9 moves a growing number of the basin's points into
training ("mixed") and compares with a model trained on those moved points alone ("control").
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.ticker import NullLocator
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import (
    FALLBACK_NOTE,
    SOM_UNIT,
    WIDTH_1COL,
    WIDTH_2COL,
    basin_colors,
    caption_line,
    place_labels,
    letter,
    model_color,
    model_marker,
    styled,
)
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, MIXED_INDICATOR, MIXED_WEIGHTED, is_matched
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, LOCAL_GLOBAL

#: Figure 9's E4 panel: each basin's points shifted by this share of the x range, so their intervals do not overlap.
E4_OFFSET = 0.02
#: Line colours of figure 9: mixed in the global model's colour, control in the local one's.
MIXED_COLOR = model_color(GLOBAL)
CONTROL_COLOR = model_color(LOCAL_GLOBAL)


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
    if not {LOCAL_GLOBAL, LOCAL} & set(inputs.predictions["model"]):
        return (
            f"No {LOCAL_GLOBAL} model in this run (the local model on Global's inputs): its leave-one-basin-out plans "
            "were run before it was fitted, or every held-out basin has fewer than 3 blocks. Run run_a1.py again "
            "(it resumes, fitting only what is missing)."
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
        f"Figure 8. (a) RMSE of Global trained on the other basins against the local model on the same inputs "
        f"(every covariate, no predicted CaCO₃/CEC), trained on the basin's own points, same test points. Transfer is worse in {worse} of {len(points)} basins; the change ranges from "
        + ps.minus(f"{points['rmse_drop'].min():+.2f} to {points['rmse_drop'].max():+.2f}")
        + f" {SOM_UNIT}."
        + (
            " (b) The same change: first column as in (a); then each covariate group alone, Global + predicted "
            "CaCO₃/CEC and the local model both fitted on the group's inputs only, so each cell compares two "
            "models given the same inputs and shows how well what the group teaches carries over to an unseen basin."
            if not grid.empty
            else ""
        )
        + (FALLBACK_NOTE if evaluation.fell_back(inputs.transfer) else "")
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
        color=model_color(GLOBAL),
        marker=model_marker(GLOBAL),
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
        "Model: Global (every covariate, no predicted CaCO₃/CEC), trained on the other basins, against the same inputs "
        "fitted inside the basin (local).",
    )
    return figure


# --- figure 9 ---------------------------------------------------------------------------------------

#: The kinds figure 9 draws, in legend order (whichever the run has).
FIG9_KINDS = (CONTROL, MIXED, MIXED_WEIGHTED, MIXED_INDICATOR)
#: Line style of the model with no local points (the plain logo plan) on each share's test points.
TRANSFERRED_STYLE = (ps.INK_2, ":")
#: The note under figure 9.
FIG9_NOTE = "Bands: 95 % 10 km block bootstrap over test points, repeats stacked; dots: individual repeats."


def moved_points(inputs: FigureInputs) -> pd.DataFrame:
    """``region, fraction, moved``: the mean number of basin points moved into training per share.

    The matching control plans' ``n_train`` (the moved points alone), averaged over the repeats;
    the share itself when the plan index does not hold it.
    """
    index = inputs.index
    control = index[index["kind"] == CONTROL] if not index.empty else index
    if control.empty:
        return pd.DataFrame(columns=["region", "fraction", "moved"])
    table = (
        control.assign(region=control["held_out"].astype(str), fraction=control["fraction"].astype(float).round(6))
        .groupby(["region", "fraction"])["n_train"]
        .mean()
        .rename("moved")
        .reset_index()
    )
    return table


def _with_moved(frame: pd.DataFrame, moved: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"held_out": "region"}).assign(
        region=lambda part: part["region"].astype(str), fraction=lambda part: part["fraction"].astype(float).round(6)
    )
    frame = frame.merge(moved, on=["region", "fraction"], how="left")
    frame["moved"] = frame["moved"].fillna(frame["fraction"])
    return frame


def e4_sizes(logo_runs, *, model: str = GLOBAL, block_km: float = 10.0, n: int = 1000, seed: int = 42) -> tuple:
    """The E4 panel's data: the transferred RMSE per basin and number of external training points.

    From the run folders given (the matched-size runs and the main run): each matched size, repeats
    stacked, and the main run's full leave-one-basin-out model, each with a 95 % interval
    (:func:`~yg_eo_soilnet.a1.compare.matched_intervals`); and every matched draw on its own.
    Two empty tables when no folder holds matched plans.
    """
    from pathlib import Path

    from yg_eo_soilnet.a1 import compare

    empty = pd.DataFrame(), pd.DataFrame()
    folders = [Path(path) for path in (logo_runs or ()) if (Path(path) / "manifest.json").exists()]
    if not folders:
        return empty
    runs = [compare.read_run(folder, model=model, kinds=[LOGO]) for folder in folders]
    reference = compare.pick_transfer_run(runs, None)
    if reference is None:
        plain = [run for run in runs if (~is_matched(run.predictions["plan_id"])).any()]
        reference = plain[0] if plain else None
    stack, _, n_train = compare.matched_stack(runs, reference, model=model)
    stack = stack[stack["model"] != compare.LOCAL_LABEL]
    if not compare._sizes(stack):
        return empty
    intervals = compare.matched_intervals(stack, n_train, block_km=block_km, n=n, seed=seed)
    intervals = intervals[intervals["metric"] == "rmse"]
    repeats = compare.matched_repeats(stack)
    repeats["n_train"] = [
        n_train.get((basin, label), np.nan) for basin, label in zip(repeats["basin"], repeats["model"])
    ]
    return intervals.reset_index(drop=True), repeats


#: The figure each draw of the moved points gets: block draws the main figure 9, point draws figure S7.
DRAW_FIGURES = {"blocks": ("fig09_mixed", "Figure 9"), "points": ("figS7_mixed_points", "Figure S7")}


def mixed_draw(inputs: FigureInputs) -> str:
    """How this run's mixed plans drew their moved points: ``blocks`` when the plan index says so, else ``points``."""
    index = inputs.index
    if index.empty or "draw" not in index.columns:
        return "points"
    draws = set(index.loc[index["kind"] == MIXED, "draw"].dropna().astype(str))
    return "blocks" if "blocks" in draws else "points"


def _compare_rows(inputs: FigureInputs, name: str) -> pd.DataFrame:
    """This run's rows of a ``compare.py`` table, from the newest ``<results>/a1_compare*/`` folder that has some."""
    from pathlib import Path

    if getattr(inputs, "run_dir", None) is None:
        return pd.DataFrame()
    run_dir = Path(inputs.run_dir)
    output = run_dir.parent.parent
    run_name = output.name + ("" if run_dir.name == "default" else f"/{run_dir.name}")
    found = sorted(output.parent.glob(f"a1_compare*/{name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in found:
        table = pd.read_csv(path)
        if "run" in table and (table["run"] == run_name).any():
            return table[table["run"] == run_name]
    return pd.DataFrame()


def sharing_range(inputs: FigureInputs) -> Optional[tuple]:
    """The min and max, over basins and shares, of the mean share of test points beside a moved point
    (``block_sharing.csv``); None without that table."""
    table = _compare_rows(inputs, "block_sharing.csv")
    if table.empty:
        return None
    means = table.groupby(["basin", "fraction"])["share_test_sharing_block"].mean()
    return float(means.min()), float(means.max())


def compare_rings(inputs: FigureInputs, moved: pd.DataFrame) -> pd.DataFrame:
    """Figure 9's RMSE rings from ``compare.py``'s ``mixed_arms_paired.csv`` (each arm minus control), when it has this run."""
    table = _compare_rows(inputs, "mixed_arms_paired.csv")
    if table.empty:
        return table
    rows = table[table["b"] == CONTROL]
    found = pd.DataFrame(
        {
            "held_out": rows["basin"],
            "fraction": rows["fraction"],
            "a": rows["a"],
            "metric": "rmse",
            "p_holm": rows["p_holm"],
        }
    )
    return _with_moved(found, moved)


def figure_mixed(inputs: FigureInputs, *, model: str = GLOBAL, logo_runs=None, draw: Optional[str] = None):
    """Figure 9 (block draws) or S7 (point draws): test RMSE and bias against the local points in training.

    Each kind (control, mixed, and the arms the run has) and the model with no local points are
    scored on the same test points of each share, the repeats stacked, with a 95 % block-bootstrap
    band; a ring marks an arm that differs from control (paired, Holm across the shares) - read from
    ``compare.py``'s ``mixed_arms_paired.csv`` for the RMSE when it has this run, computed the same
    way otherwise. With ``logo_runs``, a small panel adds E4. ``draw`` (``blocks`` for figure 9,
    ``points`` for S7) draws only a run whose mixed plans were drawn that way; None draws any run.
    """
    stacked = evaluation.mixed_stack(inputs.predictions, model=model)
    if stacked.empty:
        if MIXED in set(inputs.predictions["kind"]):
            return _missing(f"No mixed or control predictions of the {pretty_name(model)} model in this run.")
        return _missing("This run has no mixed plans: run run_a1.py with --kinds mixed (it brings the control plans).")
    run_draw = mixed_draw(inputs)
    if draw is not None and run_draw != draw:
        other, _ = DRAW_FIGURES[run_draw]
        moves = "whole 10 km blocks" if run_draw == "blocks" else "single points"
        return _missing(f"This run's mixed plans move {moves}: its figure is {other}.")
    seed = inputs.settings.seed
    moved = moved_points(inputs)
    intervals = _with_moved(
        evaluation.mixed_intervals(stacked, block_km=inputs.block_km, n=inputs.n_boot, seed=seed), moved
    )
    repeats = _with_moved(evaluation.mixed_repeat_scores(stacked), moved)
    rows = inputs.predictions[inputs.predictions["model"] == model]
    differences = evaluation.mixed_differences(rows, block_km=inputs.block_km, n=inputs.n_boot, seed=seed)
    differences = _with_moved(differences.rename(columns={"region": "held_out"}), moved)
    from_compare = compare_rings(inputs, moved)
    if not from_compare.empty:
        differences = pd.concat(
            [
                differences[differences["metric"] != "rmse"],
                from_compare[differences.columns.intersection(from_compare.columns)],
            ],
            ignore_index=True,
        )
    e4, e4_repeats = e4_sizes(logo_runs, model=model, block_km=inputs.block_km, n=inputs.n_boot, seed=seed)
    if not e4.empty and not set(e4["basin"]) & set(intervals["region"]):
        e4, e4_repeats = pd.DataFrame(), pd.DataFrame()  # another dataset's basins (the demo, say): no panel
    figure = draw_mixed(intervals, repeats, differences, e4, e4_repeats, model=model)

    parts = []
    rmse = intervals[intervals["metric"] == "rmse"]
    for region, part in rmse.groupby("region"):
        last = part["fraction"].max()
        at = part[part["fraction"] == last].set_index("kind")
        text = f"{region}, {int(round(at['moved'].iloc[0])):,} local points: " + ", ".join(
            f"{_kind_name(kind).lower()} {at.loc[kind, 'estimate']:.2f}"
            for kind in (*FIG9_KINDS, evaluation.TRANSFERRED)
            if kind in at.index
        )
        flagged = differences[
            (differences["region"] == region) & (differences["metric"] == "rmse") & (differences["p_holm"] < 0.05)
        ]
        if not flagged.empty:
            text += "; differs from control (p_Holm < 0.05): " + ", ".join(
                f"{_kind_name(kind).lower()} at {count} share(s)" for kind, count in flagged.groupby("a").size().items()
            )
        parts.append(ps.minus(text))
    _, title = DRAW_FIGURES[run_draw]
    if run_draw == "blocks":
        how = (
            "Whole 10 km blocks are moved, the last one cut to the exact number and the rest of it left out, so no "
            "test point shares a block with a moved point. "
        )
    else:
        shared = sharing_range(inputs)
        how = "Single points moved (design of Broeg et al. 2023); test points share 10 km blocks with moved points" + (
            f" ({100 * shared[0]:.0f}–{100 * shared[1]:.0f} %, block_sharing.csv). " if shared is not None else ". "
        )
    rings = "mixed_arms_paired.csv" if not from_compare.empty else "computed here"
    caption = (
        f"{title}. {pretty_name(model)} model, every input. RMSE (upper row) and bias, predicted − measured (lower "
        "row), on the basin's points left out of training, against the number of the basin's points in training "
        "(log scale): control (those points alone), mixed (the other basins plus them, pooled) and each arm the run "
        "has (weighted: the moved points weigh more; indicator: an input flags them), and, dotted, the model trained "
        "on the other basins alone on the same test points. "
        + how
        + "Each share tests a different set of points; within a share every line is scored on the same points. "
        f"Rings: an arm differs from control (paired block bootstrap, Holm across the shares of one basin; RMSE "
        f"{rings}). "
        + FIG9_NOTE
        + (
            " Right: E4, the transferred RMSE against the number of external training points - the matched draws "
            "(2,000 and 3,600 points, repeats stacked) and the full leave-one-basin-out model (one fit: no dots)."
            if not e4.empty
            else ""
        )
        + " "
        + "; ".join(parts)
        + "."
    )
    stem = DRAW_FIGURES[run_draw][0].split("_")[0]  # fig09 or figS7
    data = {
        f"{stem}_mixed_intervals": intervals,
        f"{stem}_mixed_repeats": repeats,
        f"{stem}_mixed_vs_control": differences,
    }
    if not e4.empty:
        data[f"{stem}_e4_intervals"] = e4
        data[f"{stem}_e4_repeats"] = e4_repeats
    return attach(figure, data, caption)


def _kind_name(kind: str) -> str:
    from yg_eo_soilnet.a1.figures.compare import kind_label

    return "No local points" if kind == evaluation.TRANSFERRED else kind_label(kind)


@styled
def draw_mixed(
    intervals: pd.DataFrame,
    repeats: pd.DataFrame,
    differences: pd.DataFrame,
    e4: Optional[pd.DataFrame] = None,
    e4_repeats: Optional[pd.DataFrame] = None,
    *,
    model: str = GLOBAL,
):
    """Draw figure 9: per basin, RMSE (upper) and bias (lower) against the local points; the E4 panel when given."""
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.a1.figures.compare import KIND_STYLE

    if intervals.empty:
        return _missing("No mixed-training plans to draw.")
    e4 = e4 if e4 is not None else pd.DataFrame()
    e4_repeats = e4_repeats if e4_repeats is not None else pd.DataFrame()
    regions = sorted(intervals["region"].unique())
    names = [f"r{position}" for position in range(len(regions))]
    upper_row = [f"rmse_{name}" for name in names] + (["e4"] if not e4.empty else [])
    lower_row = [f"bias_{name}" for name in names] + (["key"] if not e4.empty else [])
    widths = [1.0] * len(regions) + ([0.75] if not e4.empty else [])
    figure, grid = plt.subplot_mosaic(
        [upper_row, lower_row],
        figsize=(WIDTH_2COL if len(upper_row) > 1 else WIDTH_1COL, 4.6),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 0.65], "width_ratios": widths},
    )
    styles = {**{kind: (KIND_STYLE[kind][0], "-", KIND_STYLE[kind][1]) for kind in FIG9_KINDS}}
    styles[evaluation.TRANSFERRED] = (TRANSFERRED_STYLE[0], TRANSFERRED_STYLE[1], "o")
    present = [kind for kind in (*FIG9_KINDS, evaluation.TRANSFERRED) if kind in set(intervals["kind"])]
    for region, name in zip(regions, names):
        for metric, axis in (("rmse", grid[f"rmse_{name}"]), ("bias", grid[f"bias_{name}"])):
            for kind in present:
                color, style, marker = styles[kind]
                line = intervals[
                    (intervals["region"] == region) & (intervals["kind"] == kind) & (intervals["metric"] == metric)
                ].sort_values("moved")
                if line.empty:
                    continue
                axis.fill_between(line["moved"], line["low"], line["high"], color=color, alpha=0.14, lw=0)
                axis.plot(line["moved"], line["estimate"], color=color, ls=style, lw=1.2)
                dots = repeats[(repeats["region"] == region) & (repeats["kind"] == kind)]
                axis.scatter(dots["moved"], dots[metric], s=3, color=color, alpha=0.55, lw=0, zorder=3)
                flagged = differences[
                    (differences["region"] == region)
                    & (differences["a"] == kind)
                    & (differences["metric"] == metric)
                    & (differences["p_holm"] < 0.05)
                ]
                if not flagged.empty:
                    at = line.set_index("fraction").reindex(flagged["fraction"])
                    axis.scatter(
                        at["moved"], at["estimate"], s=34, facecolors="none", edgecolors=color, lw=1.0, zorder=4
                    )
            axis.set_xscale("log")
            axis.xaxis.set_minor_locator(NullLocator())
            if metric == "rmse":
                axis.set_title(region, fontsize=8)
                axis.tick_params(labelbottom=False)
                if name == names[0]:
                    axis.set_ylabel(f"RMSE ({SOM_UNIT})")
            else:
                axis.axhline(0, color=ps.BASELINE, lw=0.8, zorder=1)
                axis.set_xlabel("Basin points in training (log scale)")
                if name == names[0]:
                    axis.set_ylabel(f"Bias ({SOM_UNIT})")
        grid[f"bias_{name}"].sharex(grid[f"rmse_{name}"])
    # Letters along the rows: the RMSE panels (and E4), then the bias panels.
    for code, key in zip("abcdefghijklmnop", [*upper_row, *[key for key in lower_row if key != "key"]]):
        letter(grid[key], code)
    handles = [
        Line2D([], [], color=styles[kind][0], ls=styles[kind][1], lw=1.2, label=_kind_name(kind)) for kind in present
    ]
    handles += [
        Patch(facecolor=ps.MUTED, alpha=0.3, label="95 % block bootstrap"),
        Line2D([], [], ls="", marker="o", color=ps.MUTED, ms=2.5, label="One repeat"),
        Line2D([], [], ls="", marker="o", mfc="none", color=ps.INK_2, ms=5, label="Differs from control (Holm)"),
    ]
    if not e4.empty:
        axis = grid["e4"]
        basins = sorted(e4["basin"].unique())
        colors = basin_colors(basins)
        span = float(e4["n_train"].max() - e4["n_train"].min()) or 1000.0
        basin_handles = []
        for position, basin in enumerate(basins):
            color = colors[basin]
            # Each basin a little to the side, so the intervals of one size do not hide each other.
            shift = (position - (len(basins) - 1) / 2) * E4_OFFSET * span
            line = e4[e4["basin"] == basin].sort_values("n_train")
            x = line["n_train"].to_numpy(float) + shift
            errors = np.vstack(
                [(line["estimate"] - line["low"]).clip(lower=0), (line["high"] - line["estimate"]).clip(lower=0)]
            )
            axis.errorbar(
                x, line["estimate"], yerr=errors, color=color, lw=1.0, marker="o", ms=3, capsize=0, elinewidth=0.9
            )
            dots = e4_repeats[e4_repeats["basin"] == basin] if not e4_repeats.empty else e4_repeats
            if not dots.empty:
                axis.scatter(dots["n_train"] + shift, dots["rmse"], s=3, color=color, alpha=0.55, lw=0, zorder=3)
            basin_handles.append(Line2D([], [], color=color, marker="o", ms=3, lw=1.0, label=basin))
        axis.set_title("E4: training size", fontsize=8)
        axis.set_xlabel("External training points")
        axis.set_ylabel(f"Transferred RMSE ({SOM_UNIT})")
        # The basins just under the panel, clear of its bars; the kinds at the foot of the slot below.
        axis.legend(
            handles=basin_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            frameon=False,
            fontsize=7,
            ncol=3,
            handlelength=1.2,
            columnspacing=0.8,
        )
        key = grid["key"]
        key.set_axis_off()
        key.legend(handles=handles, loc="lower center", frameon=False, fontsize=7)
    else:
        figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    caption_line(figure, FIG9_NOTE)
    return figure
