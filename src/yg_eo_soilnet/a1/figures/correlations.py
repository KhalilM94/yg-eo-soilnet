"""Figure 3: how SOM correlates with the covariates within each carbonate class, and the bare-soil spectra."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.figures.core import (
    BAREST_BAND,
    FigureInputs,
    _missing,
    _numeric_covariates,
    attach,
    bare_soil_spectra,
    spearman_by_class,
)
from yg_eo_soilnet.a1.figures.labels import (
    load_labels,
    percentile_of,
    pretty_name,
    source_of,
    source_order,
    variable_of,
)
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, class_color, letter, styled

#: Rows of the heatmap at most.
MAX_ROWS = 15
#: Another percentile replaces the median only when its strongest |rho| beats the median's by this.
STRONGER_BY = 0.10


def choose_rows(correlations: pd.DataFrame, *, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """One column per variable, the strongest variables first, grouped by source.

    Per variable the median (``q50``) is kept, unless another percentile's strongest |rho| over the
    classes beats the median's by :data:`STRONGER_BY`. The ``max_rows`` strongest variables are kept,
    then ordered by source (``labels.yml``) and, within a source, by strength.

    Returns
    -------
    pandas.DataFrame
        ``column, variable, source, strength``, in drawing order.
    """
    strength = correlations.groupby("covariate")["rho"].apply(lambda values: values.abs().max()).dropna()
    rows = []
    for variable, columns in strength.groupby(lambda column: variable_of(column)):
        columns = columns.droplevel(0) if isinstance(columns.index, pd.MultiIndex) else columns
        median = [column for column in columns.index if percentile_of(column) in ("q50", "p50")]
        chosen = median[0] if median else columns.idxmax()
        best = columns.idxmax()
        if best != chosen and columns[best] >= columns[chosen] + STRONGER_BY:
            chosen = best
        rows.append({"column": chosen, "variable": variable, "source": source_of(chosen), "strength": columns[chosen]})
    frame = pd.DataFrame(rows, columns=["column", "variable", "source", "strength"])
    frame = frame.dropna(subset=["source"]).nlargest(max_rows, "strength")
    order = {name: position for position, name in enumerate(source_order())}
    frame["_source"] = frame["source"].map(order)
    return (
        frame.sort_values(["_source", "strength"], ascending=[True, False])
        .drop(columns="_source")
        .reset_index(drop=True)
    )


def figure_correlations(inputs: FigureInputs, *, n: int = 1000):
    """Figure 3: (a) Spearman rho of SOM with the strongest covariate of each kind, per class; (b) spectra."""
    table = inputs.table
    frame = table.frame
    classes = inputs.measured_classes()
    covariates = [column for column in _numeric_covariates(table) if source_of(column) is not None]
    if not covariates:
        return _missing("No covariate of the figure-3 sources (configs/a1/labels.yml) is an input.")
    coordinates = table.coordinates.rename_axis("point_id").reset_index()
    blocks = pd.Series(evaluation.spatial_blocks(coordinates, inputs.block_km).to_numpy(), index=frame.index)
    correlations = spearman_by_class(
        frame,
        inputs.settings.target,
        covariates,
        classes,
        labels=inputs.class_labels,
        blocks=blocks,
        n=n,
        seed=inputs.settings.seed,
    )
    rows = choose_rows(correlations)
    shown = (
        correlations[correlations["covariate"].isin(rows["column"])]
        .merge(rows[["column", "source"]], left_on="covariate", right_on="column")
        .drop(columns="column")
    )
    spectra = bare_soil_spectra(frame, classes, inputs.class_labels)
    figure = draw_correlations(shown, rows, spectra, labels=inputs.class_labels, n_boot=n, block_km=inputs.block_km)
    top = shown.loc[shown["rho"].abs().idxmax()] if not shown.empty else None
    caption = (
        "Figure 3. (a) Spearman correlation of SOM with the covariates, within each measured CaCO₃ class; one "
        f"percentile per variable, the {len(rows)} strongest. "
        + (
            f"The strongest is {pretty_name(top['covariate'])} in the {pretty_name(top['class'])} class (ρ = {top['rho']:.2f}). "
            if top is not None
            else ""
        )
        + f"{int(shown['significant'].sum())} of {len(shown)} cells differ from 0 (block bootstrap, Holm). "
        "(b) Median bare-soil Sentinel-2 reflectance per class, with its interquartile range."
    )
    return attach(figure, {"fig03_correlations": shown, "fig03_spectra": spectra}, caption)


@styled
def draw_correlations(
    shown: pd.DataFrame, rows: pd.DataFrame, spectra: pd.DataFrame, *, labels, n_boot: int = 1000, block_km: float = 10
):
    """Draw figure 3 from the chosen correlations (:func:`choose_rows`) and the spectra."""
    import matplotlib.pyplot as plt

    if rows.empty:
        return _missing("No correlations to draw: no class has three points with SOM.")
    columns = list(rows["column"])
    grid = shown.pivot(index="covariate", columns="class", values="rho").reindex(index=columns, columns=labels)
    marks = shown.pivot(index="covariate", columns="class", values="significant").reindex(index=columns, columns=labels)
    height = max(3.2, 0.24 * len(columns) + 1.4)
    figure, (names_axis, heat, spectrum) = plt.subplots(
        1, 3, figsize=(WIDTH_2COL, height), layout="constrained", gridspec_kw={"width_ratios": [0.2, 1.0, 0.95]}
    )
    values = grid.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    image = heat.imshow(values, cmap=ps.diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    for row in range(len(columns)):
        for column in range(len(labels)):
            value = values[row, column]
            if np.isfinite(value):
                heat.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.6 * limit else ps.INK,
                )
            if bool(marks.iloc[row, column]) if pd.notna(marks.iloc[row, column]) else False:
                heat.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, ec=ps.INK, lw=1.4))
    heat.set_xticks(range(len(labels)), [pretty_name(label) for label in labels])
    heat.set_yticks(range(len(columns)), [pretty_name(column) for column in columns])
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)
    heat.grid(False)
    heat.set_xlabel("Measured CaCO₃ class")
    # The source of each run of rows, in a column of its own left of the row labels, with a rule between sources.
    names_axis.set_ylim(heat.get_ylim())
    names_axis.set_xlim(0, 1)
    names_axis.set_axis_off()
    sources = list(rows["source"])
    start = 0
    for position in range(1, len(sources) + 1):
        if position == len(sources) or sources[position] != sources[start]:
            names_axis.text(
                1.0,
                (start + position - 1) / 2,
                sources[start].replace(" ", "\n", 1) if len(sources[start]) > 10 else sources[start],
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=ps.INK_2,
            )
            if position < len(sources):
                heat.axhline(position - 0.5, color="white", lw=2.0)
            start = position
    colorbar = figure.colorbar(image, ax=heat, orientation="horizontal", shrink=0.8, pad=0.02, aspect=30)
    colorbar.set_label("Spearman ρ with SOM (outlined: differs from 0)")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7)
    letter(heat, "a")

    if spectra.empty:
        spectrum.text(0.5, 0.5, "No bare-soil Sentinel-2 bands in the data", ha="center", va="center", color=ps.MUTED)
        spectrum.set_axis_off()
    else:
        bands = load_labels().get("bands") or {}
        ticks = spectra.drop_duplicates("band").sort_values("wavelength_nm")
        place = dict(zip(ticks["band"], range(len(ticks))))
        for label in labels:
            part = spectra[spectra["class"] == label]
            if part.empty:
                continue
            color = class_color(label)
            spectrum.fill_between(part["band"].map(place), part["q25"], part["q75"], color=color, alpha=0.2, lw=0)
            spectrum.plot(
                part["band"].map(place),
                part["median"],
                color=color,
                lw=1.5,
                marker="o",
                ms=3,
                markeredgecolor=ps.INK_2,
                markeredgewidth=0.3,
            )
        names = [
            bands.get(BAREST_BAND.match(band).group(1).lower(), band) if BAREST_BAND.match(band) else band
            for band in ticks["band"]
        ]
        spectrum.set_xticks(range(len(ticks)), names, fontsize=7, rotation=45, ha="right", rotation_mode="anchor")
        spectrum.set_xlabel("Sentinel-2 band, by wavelength (evenly spaced)")
        spectrum.set_ylabel("Bare-soil reflectance (median and IQR)")
        handles = [
            Patch(facecolor=class_color(label), edgecolor=ps.INK_2, lw=0.3, label=pretty_name(label))
            for label in labels
        ]
        spectrum.legend(
            handles=handles, title="Measured CaCO₃", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0
        )
    letter(spectrum, "b")
    caption_line(
        figure,
        f"Outlined cells: ρ differs from 0 at the 5 % level after Holm correction (block bootstrap, {block_km:g} km blocks, {n_boot:,} draws).",
    )
    return figure
