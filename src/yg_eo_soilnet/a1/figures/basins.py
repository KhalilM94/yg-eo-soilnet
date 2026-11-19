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
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.figures.core import (
    FigureInputs,
    _missing,
    attach,
    dissimilarity_frame,
    legacy_soil_sensitivity,
)
from yg_eo_soilnet.a1.figures.style import (
    FALLBACK_NOTE,
    SOM_UNIT,
    WIDTH_2COL,
    caption_line,
    letter,
    place_labels,
    styled,
)
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, LOCAL_GLOBAL

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
    "r2_transferred": "R², transferred",
    "r2_local": "R², local [95 % CI]",
    "bias_transferred": "Bias, transferred [95 % CI]",
}


def region_interval(inputs: FigureInputs, model: str, metric: str) -> pd.DataFrame:
    """``basin, estimate, low, high`` of one model's score per held-out basin, from ``intervals_by_region.csv``.

    The plain leave-one-basin-out plans of the seed the figures show; empty when the run has none.
    """
    frame = inputs.intervals_by_region
    columns = ["basin", "estimate", "low", "high"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    keep = (frame["kind"] == "logo") & (frame["model"] == model) & (frame["metric"] == metric)
    if "seed" in frame and not inputs.predictions.empty:
        keep &= frame["seed"] == inputs.fit_seed
    part = frame[keep]
    return part.assign(basin=part["held_out"].astype(str))[columns].reset_index(drop=True)


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
    local = evaluation.region_local_model(inputs.intervals_by_region)
    for name, model, metric in (("r2_local", local, "r2"), ("bias_transferred", GLOBAL, "bias")):
        found = region_interval(inputs, model, metric).set_index("basin")
        for column, suffix in (("estimate", ""), ("low", "_low"), ("high", "_high")):
            out[name + suffix] = out["basin"].map(found[column]) if not found.empty else np.nan
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

    def interval(value, low, high, signed=False):
        if pd.isna(value):
            return "–"
        bounds = ps.minus(f"[{low:.2f}, {high:.2f}]") if pd.notna(low) and pd.notna(high) else ""
        return number(value, signed) + (f"\n{bounds}" if bounds else "")

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
                HEADINGS["r2_local"]: interval(row.r2_local, row.r2_local_low, row.r2_local_high),
                HEADINGS["bias_transferred"]: interval(
                    row.bias_transferred, row.bias_transferred_low, row.bias_transferred_high, signed=True
                ),
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
        f"{SOM_UNIT}: transferred (Global - every covariate, no predicted CaCO₃/CEC - trained on the other basins), "
        "local (the same inputs, fitted inside the basin), and the change, transferred − local. R², local: the local "
        f"model's R² on the basin's points. Bias, transferred: Global's mean predicted − measured ({SOM_UNIT}). "
        f"Brackets: 95 % block-bootstrap intervals ({inputs.block_km:g} km blocks). "
        + (
            f"{widest['basin']} differs most in carbonate mix (TVD = {widest['carbonate_tvd']:.2f}). "
            if widest is not None
            else ""
        )
        + (FALLBACK_NOTE if evaluation.fell_back(inputs.transfer) else "")
        + legacy_soil_sensitivity(inputs)
    )
    return attach(figure, {"table_basins": text, "fig10_basins": table}, caption)


#: Font size of figure 10's table, points.
TABLE_FONT = 7.0
#: Space between a cell's text and its edges, points.
CELL_PAD = 2.5


def _wrapped_heading(heading: str, width: int = 8) -> str:
    """A heading wrapped to the cell, its unit or interval note - "(%)", "[95 % CI]" - kept whole on its own line."""
    for opener in (" (", " ["):
        words, found, unit = heading.partition(opener)
        if found:
            return "\n".join(
                textwrap.wrap(words, width, break_long_words=False, break_on_hyphens=False) + [opener.strip() + unit]
            )
    return "\n".join(textwrap.wrap(heading, width, break_long_words=False, break_on_hyphens=False))


def _text_width(text: str, *, bold: bool, size: float = TABLE_FONT) -> float:
    """The width of the widest line of ``text`` in points, at ``size``, as matplotlib will draw it."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(weight="bold" if bold else "normal", size=size)
    return max(TextPath((0, 0), line or " ", size=size, prop=prop).get_extents().width for line in text.split("\n"))


def column_widths(text: pd.DataFrame, headings: list) -> list:
    """Each column's width in points: its widest heading line or cell (bold in the basin column), padded."""
    widths = []
    for position, heading in enumerate(headings):
        cells = [_text_width(str(value), bold=position == 0) for value in text.iloc[:, position]]
        widths.append(max([_text_width(heading, bold=False), *cells]) + 2 * CELL_PAD)
    return widths


@styled
def draw_basins(text: pd.DataFrame):
    """Draw figure 10: the table of :func:`printable`, each column as wide as its widest text."""
    return draw_table(
        text,
        f"AoA: area of applicability. ρ: Spearman correlation with SOM, on the basin's points and on the training "
        f"points. RMSE and bias in {SOM_UNIT}; change = transferred − local; brackets: 95 % intervals.",
    )


@styled
def draw_table(text: pd.DataFrame, note: str):
    """A table figure of ``text`` (one string per cell), each column as wide as its widest text, ``note`` under it."""
    import matplotlib.pyplot as plt

    if text.empty:
        return _missing("No basins to tabulate.")
    headings = [_wrapped_heading(heading) for heading in text.columns]
    widths = column_widths(text, headings)
    line = TABLE_FONT * 1.3
    heading_lines = max(heading.count("\n") + 1 for heading in headings)
    row_lines = [max(str(value).count("\n") + 1 for value in row) for row in text.to_numpy()]
    heights = [heading_lines * line + 2 * CELL_PAD] + [n * line + 2 * CELL_PAD for n in row_lines]
    # The table at its natural size (never squeezed below it), plus room for the note under it.
    width_in = max(WIDTH_2COL, (sum(widths) + 12) / 72)
    figure, axis = plt.subplots(figsize=(width_in, sum(heights) / 72 + 0.55), layout="constrained")
    axis.set_axis_off()
    table = axis.table(
        cellText=text.to_numpy(),
        colLabels=headings,
        cellLoc="center",
        colLoc="center",
        colWidths=[width / sum(widths) for width in widths],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(heights[row] / sum(heights))
        cell.set_edgecolor(ps.BASELINE)
        cell.set_linewidth(0.4)
        cell.PAD = 0.0  # the padding is in the widths already
        if row == 0:
            # Plain weight on the shaded row: bold headings would not fit thirteen columns in two page columns.
            cell.set_text_props(color=ps.INK)
            cell.set_facecolor("#f4f3ef")
        if column == 0 and row:
            cell.set_text_props(fontweight="bold")
    caption_line(figure, note)
    return figure


# --- supplementary table S6 -------------------------------------------------------------------------

#: Table S6's columns: the plain heading of each.
ERROR_HEADINGS = {
    "basin": "Basin",
    "n": "Points",
    "rmse_transferred": "RMSE, transferred [95 % CI]",
    "bias_transferred": "Bias, transferred [95 % CI]",
    "rmse_debiased_transferred": "Debiased RMSE, transferred [95 % CI]",
    "rmse_local": "RMSE, local [95 % CI]",
}


def basin_errors(inputs: FigureInputs) -> pd.DataFrame:
    """:func:`~yg_eo_soilnet.a1.evaluation.basin_error_table` of the seed the figures show, largest basin first."""
    table = evaluation.basin_error_table(inputs.intervals_by_region)
    if "seed" in table and not table.empty:
        table = table[table["seed"] == inputs.fit_seed].drop(columns="seed")
    return table.sort_values("n", ascending=False).reset_index(drop=True)


def printable_errors(table: pd.DataFrame) -> pd.DataFrame:
    """:func:`basin_errors` as text: each score over its interval, under :data:`ERROR_HEADINGS`."""

    def cell(row, name, signed=False):
        value, low, high = row[name], row[f"{name}_low"], row[f"{name}_high"]
        if pd.isna(value):
            return "–"
        text = ps.minus(f"{value:+.2f}" if signed else f"{value:.2f}")
        return text + (ps.minus(f"\n[{low:.2f}, {high:.2f}]") if pd.notna(low) and pd.notna(high) else "")

    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                ERROR_HEADINGS["basin"]: row["basin"],
                ERROR_HEADINGS["n"]: f"{int(row['n']):,}",
                ERROR_HEADINGS["rmse_transferred"]: cell(row, "rmse_transferred"),
                ERROR_HEADINGS["bias_transferred"]: cell(row, "bias_transferred", signed=True),
                ERROR_HEADINGS["rmse_debiased_transferred"]: cell(row, "rmse_debiased_transferred"),
                ERROR_HEADINGS["rmse_local"]: cell(row, "rmse_local"),
            }
        )
    return pd.DataFrame(rows, columns=list(ERROR_HEADINGS.values()))


@styled
def draw_basin_errors(text: pd.DataFrame):
    """Draw table S6 from :func:`printable_errors`."""
    return draw_table(
        text,
        f"In {SOM_UNIT}. Debiased RMSE = √(RMSE² − bias²): the RMSE once the constant offset is removed. "
        "Brackets: 95 % block-bootstrap intervals.",
    )


def figure_basin_errors(inputs: FigureInputs):
    """Supplementary table S6: per basin, how much of the transferred error is a constant offset."""
    table = basin_errors(inputs)
    if table.empty:
        return _missing("No leave-one-basin-out intervals: run run_a1.py with --kinds logo, then evaluate.py.")
    figure = draw_basin_errors(printable_errors(table))
    has_local = table["rmse_local"].notna().any()
    closer = table[table["rmse_debiased_transferred"] < table["rmse_local"]] if has_local else table.iloc[0:0]
    caption = (
        f"Table S6. The {len(table)} held-out basins: RMSE and bias of Global (every covariate, no predicted CaCO₃/CEC) "
        "trained on the other basins, its debiased RMSE √(RMSE² − bias²) - what is left once its constant offset in "
        "the basin is removed - and the RMSE of the local model on the same inputs, fitted on the basin's own points. "
        f"95 % intervals from resampling {inputs.block_km:g} km blocks of the basin's points. "
        + (
            f"Once debiased, the transferred model is below the local RMSE in {len(closer)} of {len(table)} basins."
            if has_local
            else f"The local model on Global's inputs ({LOCAL_GLOBAL}) is not in this run, so its column is empty."
        )
        + (FALLBACK_NOTE if has_local and (table["local_model"] == LOCAL).any() else "")
    )
    return attach(figure, {"table_basin_errors": printable_errors(table), "figS6_basin_errors": table}, caption)


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
        "One point per held-out basin; Global, every covariate (no predicted CaCO₃/CEC),"
        + (" against the local model on the same inputs." if use_drop else " trained on the other basins."),
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
        + (FALLBACK_NOTE if use_drop and evaluation.fell_back(inputs.transfer) else "")
    )
    return attach(figure, {"figS5_dissimilarity": frame}, caption)
