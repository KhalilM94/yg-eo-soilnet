"""Figure C3: Spearman rho of SOM with each variable, in each held-out basin against its complement."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import _missing
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, model_color, styled
from yg_eo_soilnet.a1.stage2 import GLOBAL

#: Panels per row of the figure.
COLUMNS = 5
BASIN_COLOR = model_color(GLOBAL)
COMPLEMENT_COLOR = ps.INK_2


def figure_basin_relations(relations: pd.DataFrame, *, block_km: float = 10.0, n_boot: int = 1000):
    """Figure C3 and its caption, from ``basin_relations.csv`` (every point of a side, not per class)."""
    from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES, PRIMARY, p_method_text, variables_in_order

    overall = relations[relations["class"] == ALL_CLASSES]
    if overall.empty:
        return _missing("No held-out basin to relate."), "Not drawn: no held-out basin to relate."
    variables = variables_in_order(overall)
    basins = list(dict.fromkeys(overall["basin"]))
    figure = draw_basin_relations(overall, variables, basins, block_km=block_km)
    parts = []
    for row in overall[overall["family"] == PRIMARY].itertuples():
        parts.append(
            f"{row.basin}, {pretty_name(row.variable)}: ρ {row.rho_basin:.2f} against {row.rho_complement:.2f} in the "
            f"rest (difference {row.difference:+.2f}, p_Holm {row.p_holm:.3f})"
        )
    flagged = overall[(overall["family"] != PRIMARY) & (overall["p_holm"] < 0.05)]
    caption = (
        "Figure C3. Spearman correlation of SOM with CaCO₃, CEC and the 15 covariates of figure 3, in each held-out "
        "basin (filled) and in every other point - the training data when that basin is held out (open); bars are "
        f"95 % intervals from {n_boot} resamples of {block_km:g} km blocks, each side resampled on its own. "
        "Basin − rest differences are Holm-corrected within each basin in two families: primary (CaCO₃ and CEC) and "
        "exploratory (the covariates, and every variable within each carbonate class, in basin_relations.csv)"
        + (f"; {p_method_text(relations)}. " if p_method_text(relations) else ". ")
        + ("; ".join(parts) + ". " if parts else "")
        + f"{len(flagged)} exploratory difference(s) of {int((overall['family'] != PRIMARY).sum())} drawn have p_Holm < 0.05."
    )
    return figure, ps.minus(caption)


@styled
def draw_basin_relations(overall: pd.DataFrame, variables: list, basins: list, *, block_km: float = 10.0):
    """Draw figure C3: one panel per variable, the basins as rows, basin and complement side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = min(COLUMNS, len(variables))
    rows = int(np.ceil(len(variables) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(WIDTH_2COL, 0.32 * len(basins) * rows + 0.55 * rows + 0.9),
        layout="constrained",
        squeeze=False,
        sharey=True,
    )
    y = np.arange(len(basins))
    for position, variable in enumerate(variables):
        axis = axes[position // columns, position % columns]
        part = overall[overall["variable"] == variable].set_index("basin").reindex(basins)
        for side, offset, color, face in (
            ("basin", -0.15, BASIN_COLOR, BASIN_COLOR),
            ("complement", 0.15, COMPLEMENT_COLOR, "white"),
        ):
            rho = part[f"rho_{side}"].to_numpy(float)
            low = np.clip(rho - part[f"low_{side}"].to_numpy(float), 0, None)
            high = np.clip(part[f"high_{side}"].to_numpy(float) - rho, 0, None)
            axis.errorbar(
                rho, y + offset, xerr=np.vstack([low, high]), fmt="o", ms=3, color=color, mfc=face, lw=0.8, capsize=1.2
            )
        axis.axvline(0, color=ps.INK_2, lw=0.6, zorder=0)
        axis.set_title(textwrap.fill(pretty_name(variable), 26), fontsize=7)
        axis.set_xlim(-1, 1)
        if position == 0:
            axis.set_yticks(y, basins)
            axis.invert_yaxis()
        axis.tick_params(axis="y", labelleft=position % columns == 0)
        if position // columns == rows - 1:
            axis.set_xlabel("Spearman ρ")
    for empty in range(len(variables), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Line2D([], [], color=BASIN_COLOR, marker="o", ls="none", ms=3.5, label="Held-out basin"),
        Line2D([], [], color=COMPLEMENT_COLOR, marker="o", mfc="white", ls="none", ms=3.5, label="Every other point"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    caption_line(figure, f"Bars: 95 % intervals from resampling {block_km:g} km blocks.")
    return figure
