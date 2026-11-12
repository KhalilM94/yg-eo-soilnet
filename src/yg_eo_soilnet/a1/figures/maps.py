"""Figure 2: where the points are, by carbonate class, and how many of each class every basin holds."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, _region_outlines, attach, class_counts
from yg_eo_soilnet.a1.figures.labels import pretty_name
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL, caption_line, class_color, letter, shuffled, styled


def map_points(inputs: FigureInputs) -> pd.DataFrame:
    """One row per point: ``lat, lon, region, class`` (its measured carbonate class)."""
    table = inputs.table
    return pd.DataFrame(
        {
            "lat": table.frame[table.lat_column].to_numpy(dtype=float),
            "lon": table.frame[table.lon_column].to_numpy(dtype=float),
            "region": table.frame[table.region_column].to_numpy() if table.region_column else None,
            "class": inputs.measured_classes().astype(object).to_numpy(),
        }
    ).dropna(subset=["lat", "lon"])


def figure_map(inputs: FigureInputs, *, relief: bool = True, cache_dir: Optional[str] = None):
    """Figure 2: the points by carbonate class on the basins (held-out ones outlined thicker), and their mix."""
    points = map_points(inputs)
    held_out = list(inputs.settings.plans.logo.held_out or inputs.held_out_names)
    counts = (
        class_counts(points["region"], points["class"], inputs.class_labels) if points["region"].notna().any() else None
    )
    figure = draw_map(
        points, inputs.regions, labels=inputs.class_labels, held_out=held_out, relief=relief, cache_dir=cache_dir
    )
    if counts is None:
        return figure
    biggest = counts.sum(axis=1).idxmax()
    shares = counts.div(counts.sum(axis=1), axis=0)
    caption = (
        f"Figure 2. The {len(points):,} sample points coloured by measured CaCO₃ class, on the hydraulic basins "
        f"(held out in turn, thick outline: {', '.join(held_out) or 'none'}). (b) Points per basin and class: "
        f"{biggest} holds the most ({int(counts.loc[biggest].sum()):,}); the share of points in the ≥ 25 % class "
        f"ranges from {shares.iloc[:, -1].min():.0%} to {shares.iloc[:, -1].max():.0%} across basins."
    )
    return attach(figure, {"fig02_map_counts": counts.rename_axis("basin").reset_index()}, caption)


@styled
def draw_map(
    points: pd.DataFrame,
    regions=None,
    *,
    labels: Sequence[str],
    held_out: Sequence[str] = (),
    relief: bool = True,
    cache_dir: Optional[str] = None,
):
    """Draw figure 2: (a) the map, (b) points per basin and class, beside it - never over the points."""
    import matplotlib.pyplot as plt

    from yg_eo_soilnet import map_style
    from yg_eo_soilnet.plot_utils import residual_bounds

    if points.empty:
        return _missing("No points to map.")
    bounds = residual_bounds(points["lat"].to_numpy(), points["lon"].to_numpy())
    figure, (axis, bars) = plt.subplots(
        1, 2, figsize=(WIDTH_2COL, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    cache_dir = cache_dir or map_style.DEFAULT_CACHE_DIR
    drawn = map_style.draw_basemap(axis, bounds, cache_dir=cache_dir, relief=relief)

    outlines = _region_outlines(points, regions)
    if outlines is not None and not outlines.empty:
        projected = outlines.to_crs("EPSG:3857")
        held = projected["region"].astype(str).isin([str(name) for name in held_out])
        if (~held).any():
            projected[~held].boundary.plot(ax=axis, color=ps.INK_2, lw=0.5, zorder=3)
        if held.any():
            projected[held].boundary.plot(ax=axis, color=ps.INK, lw=1.6, zorder=3)
        _label_regions(axis, projected, points, held)

    # Every point in one call, in a fixed random order: no class is always drawn on top.
    order = shuffled(
        points.assign(color=[class_color(label) if pd.notna(label) else "none" for label in points["class"]])
    )
    measured = order[order["class"].notna()]
    x, y = map_style.to_mercator(measured["lon"].to_numpy(), measured["lat"].to_numpy())
    axis.scatter(
        x, y, s=5, c=measured["color"].tolist(), edgecolor=ps.INK_2, lw=0.15, alpha=0.85, zorder=4, rasterized=True
    )
    handles = [
        Patch(
            facecolor=class_color(label),
            edgecolor=ps.INK_2,
            lw=0.3,
            label=f"{pretty_name(label)} (n = {int((points['class'] == label).sum()):,})",
        )
        for label in labels
    ]
    unknown = order[order["class"].isna()]
    if len(unknown):
        x, y = map_style.to_mercator(unknown["lon"].to_numpy(), unknown["lat"].to_numpy())
        axis.scatter(x, y, s=4, facecolors="none", edgecolors=ps.MUTED, lw=0.3, zorder=4, rasterized=True)
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=ps.MUTED,
                label=f"CaCO₃ not measured (n = {len(unknown):,})",
            )
        )
    if len(held_out):
        handles.append(Line2D([], [], color=ps.INK, lw=1.6, label="Basin held out in turn"))
    axis.legend(
        handles=handles,
        title="Measured CaCO₃",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        frameon=False,
    )
    scale_bar_at = map_style.scale_bar_placement(bounds)
    map_style.finish_map(
        axis,
        bounds,
        scale_bar_at=scale_bar_at,
        north_arrow_at=map_style.north_arrow_placement(bounds, scale_bar_at),
        credit=map_style.CREDIT if (drawn["relief"] or drawn["borders"]) else None,
    )
    for text in axis.texts:  # the map chrome's small print, raised to the 7 pt floor
        if text.get_fontsize() < 7:
            text.set_fontsize(7)
    letter(axis, "a")

    if points["region"].notna().any():
        counts = class_counts(points["region"], points["class"], labels)
        positions = np.arange(len(counts))[::-1]
        left = np.zeros(len(counts))
        for label in labels:
            values = counts[label].to_numpy(dtype=float)
            bars.barh(positions, values, left=left, color=class_color(label), edgecolor="white", lw=0.4, height=0.75)
            left += values
        bars.set_yticks(positions, [str(name) for name in counts.index])
        ps.restyle_axes(bars, grid_axis="x")
        bars.tick_params(axis="both", labelsize=7)
        bars.set_xlabel("Number of points")
        letter(bars, "b")
    else:
        bars.set_axis_off()
    caption_line(figure, "Thick outlines: the basins held out in the leave-one-basin-out validation.")
    return figure


def _label_regions(axis, projected, points: pd.DataFrame, held) -> None:
    """Each basin's name at a point inside it, bold for a held-out basin; only basins with points."""
    named = set(points["region"].dropna().astype(str))
    for (_, row), is_held in zip(projected.iterrows(), held):
        if str(row["region"]) not in named or row.geometry is None or row.geometry.is_empty:
            continue
        spot = row.geometry.representative_point()
        axis.text(
            spot.x,
            spot.y,
            str(row["region"]),
            ha="center",
            va="center",
            fontsize=7,
            color=ps.INK,
            fontweight="bold" if is_held else "normal",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
