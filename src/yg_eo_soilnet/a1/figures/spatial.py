"""Figure 5: how accuracy changes as test points move further from the training points.

The validation designs are put in order of that distance - random, spatial blocks from the smallest
to the largest, leave one basin out - and each panel follows one score along them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1.config import stage1_prediction_column
from yg_eo_soilnet.a1.figures.core import FigureInputs, _missing, attach
from yg_eo_soilnet.a1.figures.labels import pretty_name, pretty_split
from yg_eo_soilnet.a1.figures.style import (
    MAIN_MODELS,
    SOM_UNIT,
    WIDTH_2COL,
    better_label,
    bias_axis,
    caption_line,
    letter,
    model_color,
    model_marker,
    styled,
)
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.metrics import regression_metrics

#: The kinds figure 5 orders along its x axis.
DESIGN_KINDS = (RANDOM, SPATIAL_KFOLD, LOGO)
#: The scores of panels a-d: (metric, label, unit, which way is better).
PANELS = (
    ("r2", "R²", None, "higher"),
    ("rmse", "RMSE", SOM_UNIT, "lower"),
    ("rpiq", "RPIQ", None, "higher"),
    ("bias", "Bias", SOM_UNIT, None),
)


def design_key(kind: str, block_size_m) -> float:
    """Where a design sits along figure 5's x axis: random first, leave one basin out last."""
    if kind == RANDOM:
        return 0.0
    if kind == LOGO:
        return 1e9
    return float(block_size_m) if pd.notna(block_size_m) else 1e8


def designs_present(frame: pd.DataFrame) -> pd.DataFrame:
    """The designs a table holds (``kind``, ``block_size_m``), in x order, with their labels."""
    rows = frame[frame["kind"].isin(DESIGN_KINDS)][["kind", "block_size_m"]].drop_duplicates()
    rows = rows.assign(key=[design_key(k, s) for k, s in zip(rows["kind"], rows["block_size_m"])])
    rows = rows.sort_values("key").drop(columns="key").reset_index(drop=True)
    rows["design"] = [pretty_split(k, s) for k, s in zip(rows["kind"], rows["block_size_m"])]
    rows["x"] = np.arange(len(rows))
    return rows


def spatial_scores(intervals: pd.DataFrame) -> pd.DataFrame:
    """The main models' scores per design, with intervals, from ``intervals.csv``."""
    keep = intervals[intervals["kind"].isin(DESIGN_KINDS) & intervals["model"].isin(MAIN_MODELS)]
    keep = keep[keep["metric"].isin([metric for metric, *_ in PANELS])]
    if "block_size_m" not in keep:
        keep = keep.assign(block_size_m=np.nan)
    designs = designs_present(keep)
    return keep.merge(designs, on=["kind", "block_size_m"], how="inner").sort_values(["x", "model", "metric"])


def stage1_out_of_fold(inputs: FigureInputs) -> pd.DataFrame:
    """Stage 1's out-of-fold R2 of each property, pooled per design.

    Read from each plan's stage-1 audit (``stage1/<plan>.parquet``): the training points, whose
    prediction came from a fold that left them out (``predicted_by_* = oof_fold_k``), against
    their measured values. Pooled over the folds of a design, a spatial training point counts once
    per fold it trains in.
    """
    index = inputs.index
    index = index[index["kind"].isin(DESIGN_KINDS)]
    measured = inputs.table.frame
    pooled: dict = {}
    for row in index.itertuples():
        path = inputs.run_dir / "stage1" / f"{row.plan_id}.parquet"
        if not path.exists():
            continue
        audit = pd.read_parquet(path).set_index("point_id")
        for target in inputs.settings.auxiliary.targets:
            oof = audit[audit[f"predicted_by_{target}"].astype(str).str.startswith("oof")]
            pair = pd.DataFrame(
                {
                    "observed": measured[target].reindex(oof.index).to_numpy(dtype=float),
                    "predicted": oof[stage1_prediction_column(target)].to_numpy(dtype=float),
                }
            )
            key = (row.kind, row.block_size_m if row.kind == SPATIAL_KFOLD else np.nan, target)
            pooled.setdefault(key, []).append(pair)
    rows = []
    for (kind, size, target), parts in pooled.items():
        both = pd.concat(parts, ignore_index=True)
        scores = regression_metrics(both["observed"], both["predicted"])
        rows.append(
            {
                "kind": kind,
                "block_size_m": size,
                "target": target,
                "r2": scores.get("r2_test", np.nan),
                "n": int(scores.get("n_test", 0)),
            }
        )
    frame = pd.DataFrame(rows, columns=["kind", "block_size_m", "target", "r2", "n"])
    if frame.empty:
        return frame
    return frame.merge(designs_present(frame), on=["kind", "block_size_m"], how="inner").sort_values(["x", "target"])


def figure_spatial(inputs: FigureInputs):
    """Figure 5: R2, RMSE, RPIQ and bias of the main models, and stage 1's R2, along the designs."""
    scores = spatial_scores(inputs.intervals)
    if scores.empty:
        return _missing(
            "No scores per validation design: run evaluate.py, or run_a1.py with random, spatial or logo plans."
        )
    stage1 = stage1_out_of_fold(inputs)
    figure = draw_spatial(scores, stage1, labels=inputs.settings.auxiliary.targets, block_km=inputs.block_km)
    rmse = scores[(scores["metric"] == "rmse") & (scores["model"] == MAIN_MODELS[0])].sort_values("x")
    caption = (
        "Figure 5. Accuracy against the distance between training and test points. "
        + (
            f"The global model's RMSE rises from {rmse['estimate'].iloc[0]:.2f} {SOM_UNIT} ({rmse['design'].iloc[0]}) to "
            f"{rmse['estimate'].iloc[-1]:.2f} {SOM_UNIT} ({rmse['design'].iloc[-1]}). "
            if len(rmse) >= 2
            else ""
        )
        + "Spatial-block scores pool every fold, so each point is tested once. (e) Stage 1's out-of-fold R² for "
        "CaCO₃ and CEC on the same designs."
    )
    return attach(figure, {"fig05_spatial_scores": scores, "fig05_stage1_r2": stage1}, caption)


@styled
def draw_spatial(scores: pd.DataFrame, stage1: pd.DataFrame, *, labels=(), block_km: float = 10):
    """Draw figure 5: panels a-d one score each, e stage 1's out-of-fold R2."""
    import matplotlib.pyplot as plt

    designs = scores.drop_duplicates("x").sort_values("x")
    figure, axes = plt.subplots(2, 3, figsize=(WIDTH_2COL, 4.8), layout="constrained", sharex=True)
    offsets = dict(zip(MAIN_MODELS, np.linspace(-0.12, 0.12, len(MAIN_MODELS))))
    for code, axis, (metric, name, unit, better) in zip("abcd", axes.flat, PANELS):
        part = scores[scores["metric"] == metric]
        for model in MAIN_MODELS:
            line = part[part["model"] == model].sort_values("x")
            if line.empty:
                continue
            x = line["x"].to_numpy() + offsets[model]
            errors = np.vstack(
                [(line["estimate"] - line["low"]).clip(lower=0), (line["high"] - line["estimate"]).clip(lower=0)]
            )
            axis.errorbar(
                x,
                line["estimate"],
                yerr=errors,
                color=model_color(model),
                marker=model_marker(model),
                ms=4,
                lw=1.2,
                elinewidth=0.9,
                capsize=0,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
        if metric == "bias":
            bias_axis(axis)
        else:
            axis.set_ylabel(better_label(name, better, unit))
        letter(axis, code)
    stage_axis = axes.flat[4]
    if stage1.empty:
        stage_axis.text(
            0.5, 0.5, "No stage-1 audit files", ha="center", va="center", color=ps.MUTED, transform=stage_axis.transAxes
        )
    else:
        styles = {target: (ps.INK, "o") if position == 0 else (ps.MUTED, "s") for position, target in enumerate(labels)}
        for target, line in stage1.groupby("target"):
            line = line.sort_values("x")
            color, marker = styles.get(target, (ps.INK_2, "^"))
            stage_axis.plot(line["x"], line["r2"], color=color, marker=marker, ms=4, lw=1.2, label=pretty_name(target))
        stage_axis.legend(title="Stage 1, out of fold", loc="lower left")
    stage_axis.set_ylabel(better_label("R²", "higher"))
    letter(stage_axis, "e")
    legend_axis = axes.flat[5]
    legend_axis.set_axis_off()
    handles = [
        Line2D([], [], color=model_color(m), marker=model_marker(m), ms=5, lw=1.2, label=pretty_name(m))
        for m in MAIN_MODELS
    ]
    legend_axis.legend(handles=handles, title="SOM model", loc="center", frameon=False)
    for axis in axes[1]:
        axis.set_xticks(designs["x"], list(designs["design"]), rotation=35, ha="right", rotation_mode="anchor")
    for axis in axes[0]:
        axis.tick_params(labelbottom=False)
    figure.supxlabel("Validation design, by distance between training and test points →", fontsize=7.5)
    caption_line(
        figure,
        f"Error bars: 95 % block-bootstrap intervals ({block_km:g} km blocks), on the test points of every fold pooled.",
    )
    return figure
