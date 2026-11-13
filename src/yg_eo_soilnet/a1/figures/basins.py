"""Figure 10 and supplementary figure S5: how different each held-out basin is from the training basins.

Figure 10 is a table, one row per basin: how many points, how much of the data leaving it out
removes, how many of its points fall outside the training data's area of applicability, how its
carbonate mix and its SOM correlations differ from training, and the RMSE change on transfer. It is
also written as CSV and LaTeX. Figure S5 plots the RMSE change against two of those differences.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach, dissimilarity_frame
from yg_eo_soilnet.a1.figures.style import SOM_UNIT, WIDTH_2COL, caption_line, letter, place_labels, styled

#: The table's columns: the plain heading of each, short enough for one or two lines in a cell.
HEADINGS = {
    "basin": "Basin",
    "n": "Points",
    "share_removed": "Removed (%)",
    "share_outside": "Outside AoA (%)",
    "carbonate_tvd": "CaCO₃-mix difference (TVD)",
    "rho_caco3": "ρ SOM–CaCO₃, basin / training",
    "rho_cec": "ρ SOM–CEC, basin / training",
    "rmse_transferred": "RMSE, transferred",
    "rmse_local": "RMSE, local",
    "rmse_drop": "RMSE change",
    "r2_transferred": "R², leave one basin out",
}


def _logo_scores(inputs: FigureInputs) -> pd.DataFrame:
    """Per held-out basin, every input: transferred and local RMSE, their change, and the transferred R²."""
    transfer = inputs.transfer
    columns = ["basin", "rmse_transferred", "rmse_local", "rmse_drop", "r2_transferred"]
    if transfer.empty:
        return pd.DataFrame(columns=columns)
    logo = transfer[(transfer["kind"] == "logo") & (transfer["covariate_group"] == "combined")]
    return logo.assign(basin=logo["region"].astype(str))[columns]


def basin_table(inputs: FigureInputs) -> pd.DataFrame:
    """One row per held-out basin, numeric: the columns of :data:`HEADINGS` (rho as basin and training)."""
    summary = inputs.dissimilarity_summary
    if summary.empty:
        return pd.DataFrame()
    frame = summary.assign(basin=summary["held_out"].astype(str)).merge(_logo_scores(inputs), on="basin", how="left")
    index = inputs.index
    plans = index[index["kind"] == "logo"].assign(basin=lambda part: part["held_out"].astype(str))
    removed = (plans["n_test"] / (plans["n_train"] + plans["n_test"])).set_axis(plans["basin"])
    out = pd.DataFrame(
        {
            "basin": frame["basin"],
            "n": frame["n_test"].astype(int),
            "share_removed": frame["basin"].map(removed),
            "share_outside": frame.get("share_outside", np.nan),
            "carbonate_tvd": frame.get("carbonate_tvd", np.nan),
            "rho_caco3_basin": frame.get("spearman_som_caco3_region", np.nan),
            "rho_caco3_train": frame.get("spearman_som_caco3_train", np.nan),
            "rho_cec_basin": frame.get("spearman_som_cec_region", np.nan),
            "rho_cec_train": frame.get("spearman_som_cec_train", np.nan),
            "rmse_transferred": frame["rmse_transferred"],
            "rmse_local": frame["rmse_local"],
            "rmse_drop": frame["rmse_drop"],
            "r2_transferred": frame["r2_transferred"],
        }
    )
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def printable(table: pd.DataFrame) -> pd.DataFrame:
    """:func:`basin_table` as text, under the plain headings - the form drawn and written to LaTeX."""

    def pct(value):
        return f"{100 * value:.1f}" if pd.notna(value) else "–"

    def pair(a, b):
        return ps.minus(f"{a:.2f} / {b:.2f}") if pd.notna(a) and pd.notna(b) else "–"

    def number(value, signed=False):
        if pd.isna(value):
            return "–"
        return ps.minus(f"{value:+.2f}" if signed else f"{value:.2f}")

    rows = []
    for row in table.itertuples():
        rows.append(
            {
                HEADINGS["basin"]: row.basin,
                HEADINGS["n"]: f"{row.n:,}",
                HEADINGS["share_removed"]: pct(row.share_removed),
                HEADINGS["share_outside"]: pct(row.share_outside),
                HEADINGS["carbonate_tvd"]: number(row.carbonate_tvd),
                HEADINGS["rho_caco3"]: pair(row.rho_caco3_basin, row.rho_caco3_train),
                HEADINGS["rho_cec"]: pair(row.rho_cec_basin, row.rho_cec_train),
                HEADINGS["rmse_transferred"]: number(row.rmse_transferred),
                HEADINGS["rmse_local"]: number(row.rmse_local),
                HEADINGS["rmse_drop"]: number(row.rmse_drop, signed=True),
                HEADINGS["r2_transferred"]: number(row.r2_transferred),
            }
        )
    return pd.DataFrame(rows)


def figure_basins(inputs: FigureInputs):
    """Figure 10: the held-out basins, as a table."""
    table = basin_table(inputs)
    if table.empty:
        return _missing("No dissimilarity summary: run run_a1.py with --kinds logo.")
    text = printable(table)
    figure = draw_basins(text)
    widest = table.loc[table["carbonate_tvd"].idxmax()] if table["carbonate_tvd"].notna().any() else None
    caption = (
        f"Figure 10. The {len(table)} held-out basins. Removed: the basin's share of all points, left out of training "
        "when it is held out. Area of applicability: points whose inputs are further from the training data than the "
        "training points are from each other. TVD: half the summed absolute difference between the basin's and the "
        "training data's class shares (0 same mix, 1 no overlap). RMSE in "
        f"{SOM_UNIT}: transferred (Global + predicted CaCO₃/CEC, trained on the other basins), local (the same inputs, "
        "fitted inside the basin), and the change, transferred − local. "
        + (
            f"{widest['basin']} differs most in carbonate mix (TVD = {widest['carbonate_tvd']:.2f}). "
            if widest is not None
            else ""
        )
    )
    return attach(figure, {"table_basins": text, "fig10_basins": table}, caption)


@styled
def draw_basins(text: pd.DataFrame):
    """Draw figure 10: the table of :func:`printable`."""
    import matplotlib.pyplot as plt

    if text.empty:
        return _missing("No basins to tabulate.")
    figure, axis = plt.subplots(figsize=(WIDTH_2COL, 0.32 * len(text) + 1.0), layout="constrained")
    axis.set_axis_off()
    # The words wrapped to the cell, the unit kept whole on a line of its own.
    headings = []
    for heading in text.columns:
        words, _, unit = heading.partition(" (")
        headings.append("\n".join(textwrap.wrap(words, 13, break_long_words=False) + ([f"({unit}"] if unit else [])))
    # Each column as wide as its longest line, so no text runs over a cell edge.
    # Bold heading lines run about a fifth wider than plain cell text.
    longest = [
        max([1.2 * len(line) for line in heading.split("\n")] + [len(str(value)) for value in text.iloc[:, position]])
        for position, heading in enumerate(headings)
    ]
    widths = [max(5, length) + 2 for length in longest]
    table = axis.table(
        cellText=text.to_numpy(),
        colLabels=headings,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[width / sum(widths) for width in widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(ps.BASELINE)
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=ps.INK)
            cell.set_facecolor("#f4f3ef")
            cell.set_height(cell.get_height() * 2.6)
        if column == 0 and row:
            cell.set_text_props(fontweight="bold")
    caption_line(
        figure,
        f"AoA: area of applicability. ρ: Spearman correlation with SOM, on the basin's points and on the training "
        f"points. RMSE in {SOM_UNIT}; change = transferred − local.",
    )
    return figure


@styled
def draw_dissimilarity(frame: pd.DataFrame):
    """Draw figure S5: transfer against the dissimilarity index and the SOM-CEC correlation change.

    The y axis is the RMSE change on transfer (transferred - local) when the local model exists, and
    the transferred RMSE otherwise.
    """
    import matplotlib.pyplot as plt

    if frame.empty:
        return _missing("No dissimilarity summary: this run has no leave-one-basin-out plans (run_a1.py --kinds logo).")
    use_drop = "rmse_drop" in frame and frame["rmse_drop"].notna().any()
    y_column = "rmse_drop" if use_drop else "rmse_transferred"
    y_label = (
        f"RMSE change on transfer ({SOM_UNIT})\n> 0: worse when transferred"
        if use_drop
        else f"Transferred RMSE ({SOM_UNIT}),\nlower is better"
    )
    panels = [("median_di", "Median dissimilarity index (0: like the training data)")]
    if "spearman_som_cec_diff" in frame:
        panels.append(("spearman_som_cec_diff", "Change in ρ(SOM, CEC), basin − training"))
    figure, axes = plt.subplots(1, len(panels), figsize=(WIDTH_2COL, 3.0), layout="constrained", squeeze=False)
    color_values = frame["carbonate_tvd"] if "carbonate_tvd" in frame else pd.Series(0.0, index=frame.index)
    points = None
    for code, axis, (column, label) in zip("ab", axes.flat, panels):
        points = axis.scatter(
            frame[column],
            frame[y_column],
            c=color_values,
            cmap=ps.sequential_cmap(),
            vmin=0,
            vmax=max(0.01, float(color_values.max())),
            s=30,
            edgecolor=ps.INK_2,
            lw=0.3,
            zorder=3,
        )
        axis.margins(0.12)  # no point on the frame
        if use_drop:
            axis.axhline(0, color=ps.BASELINE, lw=0.8)
        axis.set_xlabel(label)
        if code == "a":
            axis.set_ylabel(y_label)
        pairs = frame[[column, y_column]].dropna()
        if len(pairs) >= 3:
            rho = pairs[column].corr(pairs[y_column], method="spearman")
            axis.set_title(ps.minus(f"Spearman ρ = {rho:.2f}, n = {len(pairs)} basins"), fontsize=7.5, loc="right")
        else:
            axis.set_title(f"n = {len(pairs)} basins", fontsize=7.5, loc="right")
        letter(axis, code)
    colorbar = figure.colorbar(points, ax=list(axes.flat), shrink=0.8, pad=0.02, aspect=25)
    colorbar.set_label("CaCO₃-mix difference from training (TVD)")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    # Last: the colour bar reflows the panels, and labels are placed on the final layout.
    for axis, (column, _) in zip(axes.flat, panels):
        place_labels(axis, frame[column], frame[y_column], [str(region) for region in frame["region"]])
    caption_line(
        figure,
        "One point per held-out basin; every input, Global + predicted CaCO₃/CEC"
        + (" against the local model." if use_drop else " trained on the other basins."),
    )
    return figure


def figure_dissimilarity(inputs: FigureInputs):
    """Figure S5: transfer against how different the basin is (the files figure 10 reads)."""
    frame = dissimilarity_frame(inputs.transfer, inputs.dissimilarity_summary)
    figure = draw_dissimilarity(frame)
    use_drop = not frame.empty and frame["rmse_drop"].notna().any()
    what = "RMSE change on transfer (transferred − local)" if use_drop else "transferred RMSE"
    caption = (
        f"Figure S5. The {what} of each held-out basin against (a) the median dissimilarity index of its points and "
        "(b) the change in the SOM-CEC correlation; colour: carbonate-mix difference."
        + (f" With {len(frame)} basins, no correlation is reported." if 0 < len(frame) < 3 else "")
    )
    return attach(figure, {"figS5_dissimilarity": frame}, caption)
