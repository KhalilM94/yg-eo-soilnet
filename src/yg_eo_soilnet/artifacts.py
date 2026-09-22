"""Where each file a run produces is stored, and the helpers that write them.

Nothing is written to a results folder: every table, figure and summary is written to a temporary
file and uploaded to the run, where :doc:`/outputs` describes how to read it. Both model families
use the same layout, so a scikit-learn run and a deep-learning run can be compared file by file::

    <run>/
      meta/           run_summary.json
      eval_results/   eval_results.csv, split_summary.json
      plots/          pred_obs.png, cv_val_curve.png
      explain/        shap_beeswarm.png, shap_bar.png, shap_values.parquet, shap_summary.json
      uncertainty/    reliability.png, sigma_vs_error.png, uncertainty_summary.json
      predictions/    point_predictions.csv
      cv/             cv_results.csv        (scikit-learn only)
      checkpoints/    best.ckpt             (deep learning only)

The saved model itself is the exception: MLflow keeps it outside this tree, addressed by the name
:meth:`ArtifactLayout.logged_model_name` gives it.

The file names do not carry the target or the model, because MLflow compares two runs by matching
file paths, and names that differed between runs would leave nothing to compare. The run's own name
and tags say which target and model it is. Runs recorded before that change are still readable:
:data:`LEGACY_ARTIFACT_PATHS` lists where their files used to sit.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Iterable

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from yg_eo_soilnet.plot_style import SAVE_DPI


class ArtifactLayout:
    """Where each kind of file lives inside a run, and what it is called.

    The class attributes are the folder names and the file names; the methods build a path for one
    target when a run covers several.
    """

    META = "meta"
    EVAL_RESULTS = "eval_results"
    PLOTS = "plots"
    EXPLAIN = "explain"
    UNCERTAINTY = "uncertainty"
    PREDICTIONS = "predictions"
    CV = "cv"
    CHECKPOINTS = "checkpoints"

    # On the main run only.
    LEADERBOARD_PLOTS = "leaderboard_plots"
    DATA_SPLITS = "data_splits"

    # The same file names in every run, so two runs can be compared file by file.
    EVAL_RESULTS_FILE = "eval_results.csv"
    SPLIT_SUMMARY_FILE = "split_summary.json"
    RUN_SUMMARY_FILE = "run_summary.json"
    # Says why a model has no explanation - switched off, skipped, too slow, or it failed.
    EXPLAIN_SUMMARY_FILE = "explain_summary.json"
    CV_RESULTS_FILE = "cv_results.csv"
    PRED_OBS_FILE = "pred_obs.png"
    # Lightning names its own checkpoints after the epoch they come from, which differs between
    # runs; the original name is kept in the run summary and as a tag instead.
    CHECKPOINT_FILE = "best.ckpt"

    SHAP_BEESWARM_FILE = "shap_beeswarm.png"
    SHAP_BAR_FILE = "shap_bar.png"
    SHAP_BLOCK_BAR_FILE = "shap_block_bar.png"
    SHAP_VALUES_FILE = "shap_values.parquet"
    SHAP_SUMMARY_FILE = "shap_summary.json"

    # The error bars themselves are drawn on pred_obs.png; these two figures say whether those
    # bars are honest, and are kept together for a reader checking exactly that.
    RELIABILITY_FILE = "reliability.png"
    SIGMA_ERROR_FILE = "sigma_vs_error.png"
    UNCERTAINTY_SUMMARY_FILE = "uncertainty_summary.json"

    # Per-point predictions: one model's in its own sub-run, every model's combined on the main
    # run. See yg_eo_soilnet.predictions_export.
    POINT_PREDICTIONS_FILE = "point_predictions.csv"
    POINT_PREDICTIONS_WIDE_FILE = "point_predictions_wide.csv"
    POINT_PREDICTIONS_LONG_FILE = "point_predictions_long.csv"

    @classmethod
    def explain_path(cls, target: Any = None) -> str:
        """Where the SHAP figures go: ``explain``, or ``explain/<target>`` with several targets."""
        return cls._per_target(cls.EXPLAIN, target)

    @classmethod
    def uncertainty_path(cls, target: Any = None) -> str:
        """Where the uncertainty figures go, per target when a run covers several."""
        return cls._per_target(cls.UNCERTAINTY, target)

    @classmethod
    def plots_path(cls, target: Any = None) -> str:
        """Where the figures go, per target when a run covers several."""
        return cls._per_target(cls.PLOTS, target)

    @classmethod
    def _per_target(cls, root: str, target: Any = None) -> str:
        """One folder for a single target, a folder per target for several.

        A single-target run keeps the plain path, so it can be compared with any other; a run
        covering several must separate them, or its targets would overwrite each other's files.
        """
        return f"{root}/{cls.safe(target)}" if target else root

    @staticmethod
    def safe(component: Any) -> str:
        """Make a name safe to use in a path: anything unusual becomes an underscore.

        Target names come from the configuration and may carry spaces or slashes, which would
        otherwise break the upload or quietly create a folder.

        Examples
        --------
        >>> ArtifactLayout.safe("organic matter (g/kg)")
        'organic_matter_g_kg'
        """
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(component)).strip("_")
        return cleaned or "unnamed"

    # --- the older file names ----------------------------------------------
    # Nothing writes these any more. They are kept so the leaderboard can still read runs recorded
    # before the names were made the same in every run.

    @classmethod
    def filename(cls, kind: str, target: Any, model_name: Any, extension: str) -> str:
        """The older ``<kind>_<target>_<model>.<ext>`` name. For reading old runs only."""
        suffix = extension.lstrip(".")
        return f"{kind}_{cls.safe(target)}_{cls.safe(model_name)}.{suffix}"

    @classmethod
    def eval_results_filename(cls, target: Any, model_name: Any) -> str:
        """The older name of the results table. Write :attr:`EVAL_RESULTS_FILE` instead."""
        return cls.filename("eval_results", target, model_name, "csv")

    @classmethod
    def split_summary_filename(cls, target: Any) -> str:
        """The older name of the split summary. Write :attr:`SPLIT_SUMMARY_FILE` instead."""
        return f"split_summary_{cls.safe(target)}.json"

    @classmethod
    def run_summary_filename(cls, target: Any, model_name: Any) -> str:
        """The older name of the run summary. Write :attr:`RUN_SUMMARY_FILE` instead."""
        return cls.filename("run_summary", target, model_name, "json")

    @classmethod
    def cv_results_filename(cls, target: Any, model_name: Any) -> str:
        """The older name of the search results. Write :attr:`CV_RESULTS_FILE` instead."""
        return cls.filename("cv_results", target, model_name, "csv")

    @classmethod
    def logged_model_name(cls, target: Any, model_name: Any) -> str:
        """The name a saved model is stored under: ``<target>_<model>``, for both families.

        A name, not a path: MLflow keeps saved models outside the run's own files.

        Examples
        --------
        >>> ArtifactLayout.logged_model_name("clay_pct", "soil_cnn")
        'clay_pct_soil_cnn'
        """
        return f"{cls.safe(target)}_{cls.safe(model_name)}"


#: Where each kind of file used to be kept, for reading runs recorded earlier. Newest first;
#: an empty string means the run's top level. Nothing writes to these.
LEGACY_ARTIFACT_PATHS: dict[str, tuple[str, ...]] = {
    ArtifactLayout.META: ("lightning_metadata",),
    ArtifactLayout.PLOTS: ("eval_plots", ""),
    ArtifactLayout.CV: ("cv_results",),
    # The eval CSV has always lived under eval_results/, but older runs also wrote it at the run
    # root and the parent-run leaderboard still has to find those.
    ArtifactLayout.EVAL_RESULTS: ("",),
}


def candidate_artifact_paths(artifact_path: str, *filenames: str) -> list[str]:
    """Every place a reader should look for one file, the current layout first.

    Runs recorded at different times keep their files in different places and under different
    names, so a reader passes the current name first and the older ones after it.

    Parameters
    ----------
    artifact_path : str
        The folder in the current layout.
    *filenames : str
        The file names to try, current first.

    Returns
    -------
    list of str
        Paths to try in order.

    Examples
    --------
    >>> candidate_artifact_paths("cv", "cv_results.csv")
    ['cv/cv_results.csv', 'cv_results/cv_results.csv']
    """
    directories = [artifact_path, *LEGACY_ARTIFACT_PATHS.get(artifact_path, ())]

    candidates: list[str] = []
    for filename in filenames:
        if not filename:
            continue
        for directory in directories:
            candidate = f"{directory}/{filename}" if directory else filename
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def log_table(frame: pd.DataFrame, filename: str, artifact_path: str) -> None:
    """Write a table as CSV and upload it to the current run.

    Parameters
    ----------
    frame : pandas.DataFrame
        What to write.
    filename : str
        The file name, such as ``eval_results.csv``.
    artifact_path : str
        The folder inside the run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        frame.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_parquet(frame: pd.DataFrame, filename: str, artifact_path: str) -> None:
    """Write a table as Parquet and upload it, for tables too wide for CSV to be sensible.

    Used for the full SHAP contributions, one number per point per input.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        frame.to_parquet(path, index=False)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_json(payload: dict, filename: str, artifact_path: str) -> None:
    """Write a summary as JSON and upload it to the current run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_figure(figure, filename: str, artifact_path: str) -> None:
    """Save a figure, upload it to the current run, and close it.

    Closing it here is the point: a run that draws dozens of figures and forgets to close them
    fills memory with them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or None
        The figure; None does nothing.
    filename : str
        The file name, such as ``pred_obs.png``.
    artifact_path : str
        The folder inside the run.
    """
    if figure is None:
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            figure.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
            mlflow.log_artifact(path, artifact_path=artifact_path)
    finally:
        plt.close(figure)


def log_figures(figures: Iterable[tuple], artifact_path: str) -> None:
    """Upload several figures: ``(figure, filename)`` pairs, skipping any that is None."""
    for figure, filename in figures:
        log_figure(figure, filename, artifact_path)
