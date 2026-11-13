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


#: A basin label is moved off its basin's centre when more points than this lie under it.
MAX_POINTS_UNDER_LABEL = 15


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
    classed = int(counts.loc[biggest].sum())
    everything = int((points["region"].astype(str) == str(biggest)).sum())
    unclassed = everything - classed
    shares = counts.div(counts.sum(axis=1), axis=0)
    caption = (
        f"Figure 2. The {len(points):,} sample points coloured by measured CaCO₃ class, on the hydraulic basins "
        f"(held out in turn, thick outline: {', '.join(held_out) or 'none'}). (b) Points per basin and class: "
        f"{biggest} holds the most ({classed:,} with measured CaCO₃"
        + (f"; {everything:,} in all, the {unclassed:,} others have no CaCO₃ measurement" if unclassed else "")
        + f"); the share of points in the ≥ 25 % class ranges from {shares.iloc[:, -1].min():.0%} to "
        f"{shares.iloc[:, -1].max():.0%} across basins."
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
    """Draw figure 2: (a) the map, (b) points per basin and class beside it, one legend under both."""
    import matplotlib.pyplot as plt

    from yg_eo_soilnet import map_style
    from yg_eo_soilnet.plot_utils import residual_bounds

    if points.empty:
        return _missing("No points to map.")
    bounds = residual_bounds(points["lat"].to_numpy(), points["lon"].to_numpy())
    figure, grid = plt.subplot_mosaic(
        [["map", "bars"], ["legend", "legend"]],
        figsize=(WIDTH_2COL, 4.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.6, 1.0], "height_ratios": [1.0, 0.07]},
    )
    axis, bars, legend_axis = grid["map"], grid["bars"], grid["legend"]
    cache_dir = cache_dir or map_style.DEFAULT_CACHE_DIR
    drawn = map_style.draw_basemap(axis, bounds, cache_dir=cache_dir, relief=relief)

    outlines = _region_outlines(points, regions)
    projected, held = None, None
    if outlines is not None and not outlines.empty:
        projected = outlines.to_crs("EPSG:3857")
        held = projected["region"].astype(str).isin([str(name) for name in held_out])
        if (~held).any():
            projected[~held].boundary.plot(ax=axis, color=ps.INK_2, lw=0.5, zorder=3)
        if held.any():
            projected[held].boundary.plot(ax=axis, color=ps.INK, lw=1.6, zorder=3)

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
            label=f"{pretty_name(label)} ({int((points['class'] == label).sum()):,})",
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
                label=f"not measured ({len(unknown):,})",
            )
        )
    # One row under both panels, so neither the map nor the bars lose room to it.
    legend_axis.set_axis_off()
    heading = Patch(facecolor="none", edgecolor="none", label="Measured CaCO₃ (points):")
    legend_axis.legend(
        handles=[heading, *handles],
        loc="center",
        ncol=len(handles) + 1,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
        handlelength=1.2,
    )

    lon, lat, marks = map_style.scale_bar_placement(bounds, margin=0.16)  # room for "km" after the last mark
    marks = (0, marks[-1] / 2, marks[-1])  # three marks: the middle one never collides with its neighbours
    map_style.finish_map(
        axis,
        bounds,
        scale_bar_at=(lon, lat, marks),
        north_arrow_at=map_style.north_arrow_placement(bounds, (lon, lat, marks)),
        credit=None,
    )
    for text in axis.texts:  # the map chrome's small print, raised to the 7 pt floor
        if text.get_fontsize() < 7:
            text.set_fontsize(7)
    _clear_scale_unit(axis, marks)
    if drawn["relief"] or drawn["borders"]:
        # Under the degree labels, outside the frame: it never covers the map or a tick label.
        axis.annotate(
            map_style.CREDIT,
            xy=(1.0, 0.0),
            xycoords="axes fraction",
            xytext=(0, -17),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color=ps.MUTED,
        )
    if projected is not None:
        _label_regions(axis, projected, points, held)
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
        bars.set_xlabel("Number of points", fontsize=7)
        letter(bars, "b")
    else:
        bars.set_axis_off()
    caption_line(figure, "Thick outlines: the basins held out in the leave-one-basin-out validation.")
    return figure


def _label_regions(axis, projected, points: pd.DataFrame, held) -> None:
    """Each basin's name inside it, with a white halo, bold for a held-out basin; only basins with points.

    A label goes at the basin's representative point unless more than
    :data:`MAX_POINTS_UNDER_LABEL` points would lie under it; then it moves, by whole label sizes,
    to the nearby spot inside the basin with the fewest points under it.
    """
    from matplotlib import patheffects
    from shapely.geometry import Point

    from yg_eo_soilnet import map_style

    named = set(points["region"].dropna().astype(str))
    px, py = map_style.to_mercator(points["lon"].to_numpy(), points["lat"].to_numpy())
    figure = axis.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    on_screen = axis.transData.transform(np.column_stack([px, py]))
    placed = []  # the boxes of the labels already placed: a later one must not cover them
    for (_, row), is_held in zip(projected.iterrows(), held):
        if str(row["region"]) not in named or row.geometry is None or row.geometry.is_empty:
            continue
        spot = row.geometry.representative_point()
        label = axis.text(
            spot.x,
            spot.y,
            str(row["region"]),
            ha="center",
            va="center",
            fontsize=7,
            color=ps.INK,
            fontweight="bold" if is_held else "normal",
            zorder=5,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
        )
        label.set_in_layout(False)
        box = label.get_window_extent(renderer)

        def under(cx, cy, box=box):
            sx, sy = axis.transData.transform((cx, cy))
            half_w, half_h = box.width / 2, box.height / 2
            inside = (np.abs(on_screen[:, 0] - sx) <= half_w) & (np.abs(on_screen[:, 1] - sy) <= half_h)
            return int(inside.sum())

        def clashes(cx, cy, box=box):
            sx, sy = axis.transData.transform((cx, cy))
            moved = box.translated(sx - (box.x0 + box.x1) / 2, sy - (box.y0 + box.y1) / 2)
            return sum(moved.overlaps(other) for other in placed)

        frame = axis.get_window_extent(renderer)

        def outside(cx, cy, box=box, frame=frame):
            sx, sy = axis.transData.transform((cx, cy))
            half_w, half_h = box.width / 2, box.height / 2
            return int(
                sx - half_w < frame.x0 or sx + half_w > frame.x1 or sy - half_h < frame.y0 or sy + half_h > frame.y1
            )

        if (
            under(spot.x, spot.y) <= MAX_POINTS_UNDER_LABEL
            and not clashes(spot.x, spot.y)
            and not outside(spot.x, spot.y)
        ):
            placed.append(label.get_window_extent(renderer))
            continue
        # Candidate centres a whole label width or height away, kept when still inside the basin.
        inverse = axis.transData.inverted()
        sx, sy = axis.transData.transform((spot.x, spot.y))
        best = (outside(spot.x, spot.y), clashes(spot.x, spot.y), under(spot.x, spot.y), 0.0, spot.x, spot.y)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                cx, cy = inverse.transform((sx + dx * box.width, sy + dy * box.height * 1.2))
                inside_view = (
                    axis.get_xlim()[0] < cx < axis.get_xlim()[1] and axis.get_ylim()[0] < cy < axis.get_ylim()[1]
                )
                if not inside_view or not row.geometry.contains(Point(cx, cy)):
                    continue
                candidate = (outside(cx, cy), clashes(cx, cy), under(cx, cy), abs(dx) + abs(dy), cx, cy)
                best = min(best, candidate)
        label.set_position((best[4], best[5]))
        placed.append(label.get_window_extent(renderer))


def _clear_scale_unit(axis, marks) -> None:
    """Move the scale bar's "km" just right of its last mark's number, which it would otherwise touch."""
    texts = {text.get_text(): text for text in axis.texts}
    unit, last = texts.get("km"), texts.get(f"{marks[-1]:g}")
    if unit is None or last is None:
        return
    figure = axis.figure
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    right = last.get_window_extent(renderer).x1 + 3  # 3 pixels of air
    x, _ = axis.transData.inverted().transform((right, 0))
    unit.set_x(x)
    unit.set_ha("left")
