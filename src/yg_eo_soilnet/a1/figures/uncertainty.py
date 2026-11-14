"""Figures U1-U4: the prediction intervals of the quantile model (``global_q``), and how well they cover.

``figU1_bands``
    One panel per design - Random; the 10 km block folds pooled; each held-out basin - with the test
    points sorted by their predicted median: the conformal 80 % band, the median, and the measured
    SOM, orange inside the band and red outside.
``figU2_reliability``
    Observed against nominal coverage (50, 80, 90 %) per design, raw and conformal.
``figU3_width``
    (a) Each test point's 80 % interval width against its dissimilarity index (DI), per design; (b)
    PICP and mean width per measured carbonate class. Does the band widen where the model
    extrapolates?
``figU4_mixed_coverage``
    In a mixed run: PICP 80 % against the number of moved points, per basin, for each kind present.

Drawn only for a run with ``global_q`` (:func:`has_quantiles`); a run without draws none of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import uncertainty as uq
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import load_labels, pretty_name, pretty_split
from yg_eo_soilnet.a1.figures.style import (
    SOM_UNIT,
    WIDTH_2COL,
    caption_line,
    class_color,
    letter,
    model_color,
    styled,
)
from yg_eo_soilnet.a1.plans import CONTROL, DIVIDED_KINDS, LOGO, MIXED, MIXED_INDICATOR, MIXED_WEIGHTED, RANDOM
from yg_eo_soilnet.a1.plans import SPATIAL_KFOLD, is_matched
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, PER_CLASS_MEASURED, PER_CLASS_PREDICTED
from yg_eo_soilnet.datamodules.splitting import TEST

#: The uncertainty figures, drawn only for a run with ``global_q``.
UNCERTAINTY_FIGURES = ("figU1_bands", "figU2_reliability", "figU3_width", "figU4_mixed_coverage")
#: The interval the band figures show.
BAND_LEVEL = 0.8
INSIDE_COLOR = model_color(PER_CLASS_MEASURED)  # orange
OUTSIDE_COLOR = "#b2182b"  # red; also told apart by its marker
BAND_COLOR = model_color(GLOBAL)
METHOD_MARKERS = {uq.RAW: "o", uq.CQR: "s"}
KIND_STYLE = {
    CONTROL: (model_color(LOCAL), "v"),
    MIXED: (model_color(GLOBAL), "D"),
    MIXED_WEIGHTED: (model_color(PER_CLASS_MEASURED), "s"),
    MIXED_INDICATOR: (model_color(PER_CLASS_PREDICTED), "^"),
}


def has_quantiles(inputs: FigureInputs) -> bool:
    """Whether the run has the quantile model's predictions."""
    try:
        return not inputs.quantile_predictions.empty
    except FileNotFoundError:
        return False


def band_method(frame: pd.DataFrame, level: float = BAND_LEVEL) -> str:
    """The conformal interval when the run wrote it, the raw one otherwise."""
    return uq.CQR if uq.CQR in uq.methods_of(frame, level) else uq.RAW


def designs(inputs: FigureInputs) -> list[tuple[str, pd.DataFrame]]:
    """The test rows of each design: Random, the spatial folds of ``cell_size_m`` pooled, each held-out basin."""
    frame = inputs.quantile_predictions
    test = frame[frame["split"] == TEST]
    chosen = []
    random = test[test["kind"] == RANDOM]
    if not random.empty:
        chosen.append((pretty_split(RANDOM), random))
    cell = inputs.settings.plans.spatial_kfold.cell_size_m
    folds = test[(test["kind"] == SPATIAL_KFOLD) & np.isclose(test["block_size_m"].astype(float), float(cell))]
    if not folds.empty:
        chosen.append((pretty_split(SPATIAL_KFOLD, cell), folds))
    logo = test[(test["kind"] == LOGO) & ~is_matched(test["plan_id"])]
    for basin, part in logo.groupby("held_out", sort=True):
        chosen.append((f"{pretty_split(LOGO)}: {basin}", part))
    return chosen


def _coverage(part: pd.DataFrame, level: float, method: str, n_boot: int, seed: int, block_km: float):
    """PICP and mean width of one design, with block-bootstrap intervals of the PICP."""
    from yg_eo_soilnet.a1 import evaluation

    low, high = uq.bound_columns(level, method)
    y = part["observed"].to_numpy(float)
    inside = (y >= part[low].to_numpy(float)) & (y <= part[high].to_numpy(float))
    width = (part[high] - part[low]).to_numpy(float)
    blocks = evaluation.spatial_blocks(part, block_km).to_numpy()
    bounds = uq._bootstrap(part, blocks, [level], n_boot, seed, ("figure", str(part["plan_id"].iloc[0])))
    picp_low, picp_high = bounds.get((method, level, "picp"), (np.nan, np.nan))
    return float(inside.mean()), picp_low, picp_high, float(width.mean()), inside


# --- U1 -----------------------------------------------------------------------------------------------


def figure_bands(inputs: FigureInputs):
    """Figure U1: the conformal 80 % band along the test points sorted by predicted median, per design."""
    chosen = designs(inputs)
    if not chosen:
        return _missing("No test predictions of the quantile model (global_q) in a design figure U1 shows.")
    rows, summaries = [], []
    for name, part in chosen:
        method = band_method(part)
        low, high = uq.bound_columns(BAND_LEVEL, method)
        picp, picp_low, picp_high, width, inside = _coverage(
            part, BAND_LEVEL, method, inputs.n_boot, inputs.settings.seed, inputs.block_km
        )
        ordered = part.assign(inside=inside).sort_values("predicted").reset_index(drop=True)
        rows.append(
            ordered.assign(design=name, rank=np.arange(len(ordered)))[
                ["design", "rank", "plan_id", "point_id", "observed", "predicted", low, high, "inside"]
            ].rename(columns={low: "band_low", high: "band_high"})
        )
        summaries.append(
            {
                "design": name,
                "method": method,
                "n": len(part),
                "picp": picp,
                "picp_low": picp_low,
                "picp_high": picp_high,
                "width_mean": width,
            }
        )
    data, summary = pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)
    figure = draw_bands(data, summary)
    parts = [
        f"{row.design}: {row.picp:.0%} ({row.picp_low:.0%}–{row.picp_high:.0%}) inside, mean width {row.width_mean:.1f} {SOM_UNIT}"
        for row in summary.itertuples()
    ]
    method = "conformal (CQR)" if (summary["method"] == uq.CQR).all() else "raw quantile"
    caption = (
        f"Figure U1. Prediction intervals of SOM ({pretty_name('global_q')}), test points of each design sorted by "
        f"their predicted median. Band: the {method} 80 % interval (10th to 90th percentile"
        + (
            ", after conformal calibration on the training points' out-of-fold predictions"
            if method.startswith("conf")
            else ""
        )
        + "); line: the predicted median; points: measured SOM, orange inside the band, red outside. "
        "Titles: the share inside (PICP) with its 95 % block-bootstrap interval, and the mean width. "
        + "; ".join(parts)
        + "."
    )
    return attach(figure, {"figU1_bands": data, "figU1_bands_summary": summary}, ps.minus(caption))


@styled
def draw_bands(data: pd.DataFrame, summary: pd.DataFrame):
    """Draw figure U1."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names = list(summary["design"])
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, 2.3 * rows + 0.8), layout="constrained", squeeze=False, sharey=True
    )
    values = pd.concat([data["band_low"], data["band_high"], data["observed"]]).to_numpy(float)
    low, high = np.nanpercentile(values, 0.5), np.nanpercentile(values, 99.5)
    pad = 0.04 * (high - low)
    for position, name in enumerate(names):
        axis = axes[position // columns, position % columns]
        part = data[data["design"] == name]
        size = 3 if len(part) > 1000 else 7
        axis.fill_between(part["rank"], part["band_low"], part["band_high"], color=BAND_COLOR, alpha=0.2, lw=0)
        axis.plot(part["rank"], part["predicted"], color=BAND_COLOR, lw=1.0)
        inside, outside = part[part["inside"]], part[~part["inside"]]
        axis.scatter(inside["rank"], inside["observed"], s=size, color=INSIDE_COLOR, lw=0, alpha=0.8, zorder=3)
        axis.scatter(
            outside["rank"], outside["observed"], s=size * 1.6, color=OUTSIDE_COLOR, marker="x", lw=0.6, zorder=4
        )
        row = summary.set_index("design").loc[name]
        # The panel letter leads the title: a separate letter would sit on its first line.
        axis.set_title(
            f"({'abcdefghijklmnop'[position]}) {name}\nn = {int(row['n'])}, PICP80 {row['picp']:.0%} "
            f"({row['picp_low']:.0%}–{row['picp_high']:.0%}), width {row['width_mean']:.1f}",
            fontsize=7,
        )
        axis.set_ylim(low - pad, high + pad)
        axis.set_xlabel("Test points, by predicted median")
        if position % columns == 0:
            axis.set_ylabel(f"SOM ({SOM_UNIT})")
    for empty in range(len(names), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Patch(color=BAND_COLOR, alpha=0.2, label="80 % interval"),
        Line2D([], [], color=BAND_COLOR, lw=1.0, label="Predicted median"),
        Line2D([], [], color=INSIDE_COLOR, marker="o", ls="none", ms=3.5, label="Measured, inside"),
        Line2D([], [], color=OUTSIDE_COLOR, marker="x", ls="none", ms=3.5, label="Measured, outside"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    caption_line(figure, "Same SOM limits in every panel (0.5th to 99.5th percentile of every value drawn).")
    return figure


# --- U2 -----------------------------------------------------------------------------------------------


def figure_reliability(inputs: FigureInputs):
    """Figure U2: observed against nominal coverage per design, raw and conformal."""
    chosen = designs(inputs)
    if not chosen:
        return _missing("No test predictions of the quantile model (global_q) in a design figure U2 shows.")
    rows = []
    for name, part in chosen:
        for level in uq.levels_of(part):
            for method in uq.methods_of(part, level):
                picp, low, high, width, _ = _coverage(
                    part, level, method, inputs.n_boot, inputs.settings.seed, inputs.block_km
                )
                rows.append(
                    {
                        "design": name,
                        "level": level,
                        "method": method,
                        "picp": picp,
                        "low": low,
                        "high": high,
                        "n": len(part),
                    }
                )
    data = pd.DataFrame(rows)
    figure = draw_reliability(data)
    parts = []
    for name, part in data.groupby("design", sort=False):
        text = ", ".join(
            f"{int(round(row.level * 100))} %: {row.picp:.0%}"
            for row in part[part["method"] == part["method"].iloc[-1]].itertuples()
        )
        parts.append(f"{name} {text}")
    caption = (
        "Figure U2. Observed coverage (share of measured SOM inside the interval, PICP) against nominal coverage, "
        "per design: circles the raw quantile intervals, squares the conformal ones; bars 95 % block-bootstrap "
        "intervals; on the 1:1 line an interval covers what it promises. Conformal coverage is guaranteed only when "
        "calibration and test points are exchangeable, which spatial blocks and held-out basins are not; there it "
        "is measured. "
        + ("Conformal coverage - " if (data["method"] == uq.CQR).any() else "Raw coverage - ")
        + "; ".join(parts)
        + "."
    )
    return attach(figure, {"figU2_reliability": data}, ps.minus(caption))


@styled
def draw_reliability(data: pd.DataFrame):
    """Draw figure U2."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list(dict.fromkeys(data["design"]))
    columns = min(5, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_2COL, 2.0 * rows + 0.8), layout="constrained", squeeze=False, sharey=True
    )
    for position, name in enumerate(names):
        axis = axes[position // columns, position % columns]
        axis.plot([0.4, 1.0], [0.4, 1.0], color=ps.INK_2, lw=0.8, ls="--", zorder=1)
        for method, offset in ((uq.RAW, -0.008), (uq.CQR, 0.008)):
            part = data[(data["design"] == name) & (data["method"] == method)].sort_values("level")
            if part.empty:
                continue
            x = part["level"].to_numpy(float) + offset
            y = part["picp"].to_numpy(float)
            errors = np.vstack([np.clip(y - part["low"], 0, None), np.clip(part["high"] - y, 0, None)])
            face = "white" if method == uq.RAW else BAND_COLOR
            axis.errorbar(
                x, y, yerr=errors, fmt=METHOD_MARKERS[method], color=BAND_COLOR, mfc=face, ms=3.5, lw=0.8, capsize=1.5
            )
        axis.set_xlim(0.4, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks([0.5, 0.8, 0.9], ["50", "80", "90"])
        axis.set_title(f"({'abcdefghijklmnop'[position]}) {name.replace(': ', ':' + chr(10))}", fontsize=7)
        axis.set_xlabel("Nominal coverage (%)")
        if position % columns == 0:
            axis.set_ylabel("Observed coverage (PICP)")
    for empty in range(len(names), rows * columns):
        axes[empty // columns, empty % columns].set_visible(False)
    handles = [
        Line2D([], [], color=BAND_COLOR, marker="o", mfc="white", ls="none", ms=3.5, label="Raw quantiles"),
        Line2D([], [], color=BAND_COLOR, marker="s", ls="none", ms=3.5, label="Conformal (CQR)"),
        Line2D([], [], color=ps.INK_2, ls="--", lw=0.8, label="1:1, as promised"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    caption_line(figure, "Bars: 95 % block-bootstrap intervals of the observed coverage.")
    return figure


# --- U3 -----------------------------------------------------------------------------------------------


def figure_width(inputs: FigureInputs):
    """Figure U3: (a) interval width against the dissimilarity index per design; (b) PICP and width per class."""
    chosen = designs(inputs)
    if not chosen:
        return _missing("No test predictions of the quantile model (global_q) in a design figure U3 shows.")
    di = inputs.point_dissimilarity
    points, classes = [], []
    for name, part in chosen:
        method = band_method(part)
        low, high = uq.bound_columns(BAND_LEVEL, method)
        y = part["observed"].to_numpy(float)
        frame = part.assign(
            design=name,
            width=(part[high] - part[low]).to_numpy(float),
            inside=(y >= part[low].to_numpy(float)) & (y <= part[high].to_numpy(float)),
        )
        if not di.empty:
            frame = frame.merge(di[["plan_id", "point_id", "di"]], on=["plan_id", "point_id"], how="left")
        else:
            frame = frame.assign(di=np.nan)
        points.append(frame[["design", "plan_id", "point_id", "di", "width", "inside", "carbonate_class_measured"]])
        for label, chosen_class in frame.groupby("carbonate_class_measured", observed=True):
            if len(chosen_class) < inputs.settings.min_cell_size:
                continue
            classes.append(
                {
                    "design": name,
                    "class": str(label),
                    "n": len(chosen_class),
                    "picp": float(chosen_class["inside"].mean()),
                    "width_mean": float(chosen_class["width"].mean()),
                }
            )
    data = pd.concat(points, ignore_index=True)
    by_class = pd.DataFrame(classes, columns=["design", "class", "n", "picp", "width_mean"])
    figure = draw_width(data, by_class, labels=list(inputs.class_labels))
    parts = []
    for name, part in data.dropna(subset=["di"]).groupby("design", sort=False):
        if len(part) >= 3:
            rho = part[["di", "width"]].rank().corr().iloc[0, 1]
            parts.append(f"{name}: Spearman ρ(width, DI) = {rho:.2f}")
    caption = (
        "Figure U3. Does the band widen where the model extrapolates? (a) Width of each test point's 80 % interval "
        "against its dissimilarity index (DI: distance to the nearest training point in the model's weighted "
        "inputs, over the mean distance between training points), orange inside the band, red outside; the DI is "
        "computed for the random plan, the 10 km folds and each held-out basin. (b) PICP 80 % and mean width per "
        f"measured carbonate class (classes with at least {inputs.settings.min_cell_size} test points). "
        + ("; ".join(parts) + "." if parts else "No dissimilarity file was found for these plans.")
    )
    return attach(figure, {"figU3_width": data, "figU3_width_by_class": by_class}, ps.minus(caption))


@styled
def draw_width(data: pd.DataFrame, by_class: pd.DataFrame, *, labels: list):
    """Draw figure U3."""
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(data["design"]))
    columns = min(5, len(names))
    figure = plt.figure(figsize=(WIDTH_2COL, 4.6), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[1.0, 0.9])
    axes = top.subplots(1, columns, squeeze=False, sharey=True)[0]
    for position, name in enumerate(names[:columns]):
        axis = axes[position]
        part = data[data["design"] == name]
        for inside, color, marker in ((True, INSIDE_COLOR, "o"), (False, OUTSIDE_COLOR, "x")):
            chosen = part[part["inside"] == inside]
            axis.scatter(
                chosen["di"], chosen["width"], s=5, color=color, marker=marker, lw=0.5 if not inside else 0, alpha=0.7
            )
        axis.set_title(name, fontsize=7)
        axis.set_xlabel("Dissimilarity index")
        if position == 0:
            axis.set_ylabel(f"80 % width ({SOM_UNIT})")
        letter(axis, "a" if position == 0 else "")
    left, right = bottom.subplots(1, 2)
    width = 0.8 / max(1, len(labels))
    for position, label in enumerate(labels):
        part = by_class[by_class["class"] == label].set_index("design").reindex(names)
        x = np.arange(len(names)) + (position - (len(labels) - 1) / 2) * width
        left.bar(x, part["picp"], width=width, color=class_color(label), label=pretty_name(label))
        right.bar(x, part["width_mean"], width=width, color=class_color(label))
    left.axhline(BAND_LEVEL, color=ps.INK_2, lw=0.8, ls="--")
    for axis, ylabel in ((left, "PICP 80 %"), (right, f"Mean 80 % width ({SOM_UNIT})")):
        # A held-out basin's design is named by the basin alone under the bars; the panels above say the rest.
        axis.set_xticks(np.arange(len(names)), [name.split(": ")[-1] for name in names], fontsize=7)
        axis.set_ylabel(ylabel)
    left.set_ylim(0, 1.02)
    letter(left, "b")
    handles, texts = left.get_legend_handles_labels()
    bottom.legend(handles, texts, title="Measured CaCO₃", loc="outside upper center", ncol=len(texts), frameon=False)
    caption_line(figure, "Dashed: the nominal 80 %. Orange: inside the band; red crosses: outside.")
    return figure


# --- U4 -----------------------------------------------------------------------------------------------


def figure_mixed_coverage(inputs: FigureInputs):
    """Figure U4: PICP 80 % against the number of moved points, per basin, for each mixed kind present."""
    frame = inputs.quantile_predictions
    test = frame[(frame["split"] == TEST) & frame["kind"].isin(DIVIDED_KINDS)]
    if test.empty:
        return _missing("No mixed or control plans with the quantile model (global_q) in this run.")
    index = inputs.index.set_index("plan_id")
    moved = {}
    for plan_id in test["plan_id"].unique():
        row = index.loc[plan_id] if plan_id in index.index else None
        moved[plan_id] = float(row["n_train"]) if row is not None and row["kind"] == CONTROL else np.nan
    controls = test[test["kind"] == CONTROL].drop_duplicates("plan_id")
    moved_by_key = {
        (r.held_out, round(float(r.fraction), 6), r.repeat): moved[r.plan_id] for r in controls.itertuples()
    }
    rows = []
    for (kind, basin, fraction), part in test.groupby(["kind", "held_out", "fraction"]):
        method = band_method(part)
        low, high = uq.bound_columns(BAND_LEVEL, method)
        y = part["observed"].to_numpy(float)
        inside = (y >= part[low].to_numpy(float)) & (y <= part[high].to_numpy(float))
        counts = [
            moved_by_key.get((basin, round(float(fraction), 6), repeat), np.nan) for repeat in part["repeat"].unique()
        ]
        rows.append(
            {
                "basin": basin,
                "kind": kind,
                "fraction": float(fraction),
                "moved": float(np.nanmean(counts)) if np.isfinite(counts).any() else np.nan,
                "picp": float(inside.mean()),
                "method": method,
                "n_repeats": int(part["repeat"].nunique()),
                "n": len(part),
            }
        )
    data = pd.DataFrame(rows)
    figure = draw_mixed_coverage(data)
    parts = []
    for (basin, kind), part in data.groupby(["basin", "kind"]):
        parts.append(f"{basin}, {_kind_label(kind).lower()}: {part['picp'].min():.0%}–{part['picp'].max():.0%}")
    caption = (
        "Figure U4. Share of measured SOM inside the "
        + ("conformal " if (data["method"] == uq.CQR).all() else "")
        + "80 % interval (PICP) of the quantile model against the number of the basin's points moved into "
        "training, per basin, for each kind of mixed plan present. Each PICP per basin and share is the average "
        "over the repeats' test points: the test points of every repeat of the share pooled, each counted once per "
        "repeat. Dashed: the nominal 80 %. Range over the shares - " + "; ".join(parts) + "."
    )
    return attach(figure, {"figU4_mixed_coverage": data}, ps.minus(caption))


def _kind_label(kind: str) -> str:
    return (load_labels().get("kinds") or {}).get(kind) or pretty_name(kind)


@styled
def draw_mixed_coverage(data: pd.DataFrame):
    """Draw figure U4."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    basins = sorted(data["basin"].unique())
    columns = min(5, len(basins))
    figure, axes = plt.subplots(1, columns, figsize=(WIDTH_2COL, 2.6), layout="constrained", squeeze=False, sharey=True)
    drawn = set()
    for position, basin in enumerate(basins[:columns]):
        axis = axes[0, position]
        for kind, (color, marker) in KIND_STYLE.items():
            part = data[(data["basin"] == basin) & (data["kind"] == kind)].sort_values("moved")
            if part.empty:
                continue
            axis.plot(part["moved"], part["picp"], color=color, marker=marker, ms=3.2, lw=1.1, mec="white", mew=0.4)
            drawn.add(kind)
        axis.axhline(BAND_LEVEL, color=ps.INK_2, lw=0.8, ls="--")
        axis.set_xscale("log")
        axis.set_ylim(0, 1.02)
        axis.set_title(str(basin), fontsize=8)
        axis.set_xlabel("Points moved into training")
        if position == 0:
            axis.set_ylabel("PICP 80 %")
        letter(axis, "abcdefghij"[position])
    handles = [
        Line2D([], [], color=color, marker=marker, lw=1.1, ms=3.5, label=_kind_label(kind))
        for kind, (color, marker) in KIND_STYLE.items()
        if kind in drawn
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    caption_line(figure, "PICP per basin and share over the repeats' test points pooled. Dashed: the nominal 80 %.")
    return figure
