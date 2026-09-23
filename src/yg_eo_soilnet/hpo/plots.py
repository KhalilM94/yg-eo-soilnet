"""The figures a finished study leaves behind, and its table of trials.

As everywhere else here, each function builds a figure and returns it; the caller saves it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import optuna

# Explicit: `import optuna` alone does not bind the visualization submodule, so
# `optuna.visualization.matplotlib.plot_*` raises AttributeError without this. The matplotlib
# backend is the only usable one here - the default one needs plotly, which is not installed.
from matplotlib.figure import Figure
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from yg_eo_soilnet.hpo.tracker import COMPLETE, ObjectiveTracker
from yg_eo_soilnet.plot_style import (
    BASELINE,
    FIG_WIDTH_FULL,
    INK_2,
    MODEL_COLORS,
    PROJECT_COLORS,
    SAVE_DPI,
    message_figure as _message_figure,
    panel_subtitle,
    restyle_axes,
    style_context,
)

ARTIFACT_PATH = "optuna"


def _completed(study: optuna.Study) -> int:
    """The trials that actually finished."""
    return sum(1 for trial in study.trials if trial.state.name == COMPLETE)


def _dress(axes, subtitle: str, *, horizontal_bars: bool = False) -> Figure:
    """Bring a figure the tuning library drew into this project's style."""
    figure = axes.figure
    figure.set_size_inches(FIG_WIDTH_FULL, 3.0)
    # Not optional, and not redundant with the style_context around the draw: optuna calls
    # plt.style.use("ggplot") inside plot_optimization_history, which replaces the rcParams while it
    # draws. Only setting the properties on the finished artists survives that.
    restyle_axes(axes, grid_axis="x" if horizontal_bars else "y")
    # restyle_axes clears optuna's own heading from every title slot; the house style says what the
    # plot is in the right slot at the smaller size, leaving the left slot free for a panel letter.
    panel_subtitle(axes, subtitle)
    # Recoloured series by series rather than all to one colour: the history plot draws the trial
    # values and the running best as two separate lines, and painting both the same colour would
    # merge the two things the plot exists to contrast.
    for index, line in enumerate(axes.get_lines()):
        line.set_color(MODEL_COLORS[index % len(MODEL_COLORS)])
        line.set_linewidth(1.6)
    # The individual trials, which optuna draws as a scatter. Context, so grey.
    for collection in axes.collections:
        collection.set_color(BASELINE)
    # The importance bars, which are patches rather than a collection.
    for patch in axes.patches:
        patch.set_color(PROJECT_COLORS["Fertimap"])
    legend = axes.get_legend()
    if legend is not None:
        # Rebuilt rather than restyled in place: optuna parks it outside the axes on the right,
        # which costs about a third of a 6.5in figure's width for two entries.
        handles, labels = axes.get_legend_handles_labels()
        legend.remove()
        if handles:
            axes.legend(handles, labels, loc="best", fontsize=7.5, labelcolor=INK_2)
    figure.tight_layout()
    return figure


def optimization_history(study: optuna.Study) -> Figure:
    """Each trial's score, with the best so far - does the search improve?"""
    if _completed(study) < 1:
        return _message_figure("No completed trials yet", figsize=(FIG_WIDTH_FULL, 1.6))
    with style_context():
        # These return an Axes, not a Figure - every saver in this repo calls fig.savefig, so the
        # .figure hop is what makes them usable at all.
        return _dress(plot_optimization_history(study), "objective per trial")


def param_importances(study: optuna.Study) -> Figure:
    """Which settings actually moved the score, and which made no difference."""
    # fANOVA needs at least two completed trials and something that varies between them; below that
    # get_param_importances raises rather than returning an empty result.
    if _completed(study) < 2:
        return _message_figure("Need at least two completed trials for importances", figsize=(FIG_WIDTH_FULL, 1.6))
    with style_context():
        try:
            axes = plot_param_importances(study)
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            return _message_figure(f"Importances unavailable: {exc}", figsize=(FIG_WIDTH_FULL, 1.6))
        return _dress(axes, "hyperparameter importance", horizontal_bars=True)


def write_study_artifacts(
    study: optuna.Study,
    tracker: ObjectiveTracker,
    output_dir: str | Path,
    *,
    log_to_mlflow: bool = True,
    logger: Any = None,
) -> list[Path]:
    """Write ``trials.csv`` and the two figures, and attach them to the run when there is one.

    Written to a real folder first, so a run with recording switched off still leaves the same files
    behind.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    frame = tracker.trials_frame(study)
    csv_path = output_dir / "trials.csv"
    frame.to_csv(csv_path, index=False)
    written.append(csv_path)

    for name, builder in (("optimization_history", optimization_history), ("param_importances", param_importances)):
        try:
            figure = builder(study)
        except Exception as exc:  # pragma: no cover - a plot must never sink a finished study
            if logger is not None:
                logger.warning(f"Could not build the {name} figure: {exc!r}")
            continue
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    if log_to_mlflow and mlflow.active_run() is not None:
        for path in written:
            mlflow.log_artifact(str(path), artifact_path=ARTIFACT_PATH)

    return written
