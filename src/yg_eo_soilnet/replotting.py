"""Redraw a finished run's figures from what it already recorded; the engine behind ``replot.py``.

Nothing here reads the source data or a model. It does not need to: every figure a run produces can
be redrawn from the two tables it already wrote - the test-point results, and the search results.
So the figures of a run that finished months ago can be redrawn after a change to how they look,
without retraining anything.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

import mlflow
import pandas as pd

from yg_eo_soilnet.artifacts import ArtifactLayout, candidate_artifact_paths
from yg_eo_soilnet.logger.mlflow_loggers import (
    ChildRunLogger,
    ParentRunLogger,
    _eval_results_artifact_paths,
    _scoring_runs,
)
from yg_eo_soilnet.targets import split_target_names
from yg_eo_soilnet.uncertainty.columns import sigma_column
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.intervals import (
    CONFORMAL,
    GAUSSIAN,
    SIGMA,
    GaussianInterval,
    SigmaInterval,
    describe as describe_interval,
    normalize_method,
)

# What a caller can ask to be redrawn. `leaderboard` is the only parent-run entry; the rest are
# per-child.
CHILD_KINDS = ("pred_obs", "uncertainty", "cv")
PARENT_KINDS = ("leaderboard",)
ALL_KINDS = CHILD_KINDS + PARENT_KINDS


def _wanted(only: Optional[Iterable[str]], kind: str) -> bool:
    """Whether a figure of this kind was asked for."""
    return kind in (set(only) if only else set(ALL_KINDS))


def _download(run_id: str, artifact_path: str):
    """Fetch one file from a run, or None when it is not there."""
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    except Exception:
        return None


def _read_first(run_id: str, artifact_paths: Iterable[str]) -> Optional[pd.DataFrame]:
    """Read the first of several possible paths that exists.

    Runs recorded at different times keep their files in different places.
    """
    for artifact_path in artifact_paths:
        local_path = _download(run_id, artifact_path)
        if local_path is None:
            continue
        try:
            return pd.read_csv(local_path)
        except Exception:
            continue
    return None


def _interval_estimator(run: Any, target: str) -> Any:
    """Rebuild the kind of interval this run's error bars came from, for the label.

    Only the label needs it - "±1σ" against "conformal 95%" - but a figure that says the wrong thing
    about what its bars mean is worse than one with no label.
    """
    params = dict(run.data.params)
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if f"interval_method_{target}" not in params and parent_id:
        try:
            params.update(mlflow.tracking.MlflowClient().get_run(parent_id).data.params)
        except Exception:
            return None

    method = params.get(f"interval_method_{target}")
    if not method:
        return None
    try:
        resolved = normalize_method(method)
    except ValueError:
        return None

    if resolved == SIGMA:
        return SigmaInterval(k=float(params.get(f"interval_k_{target}", 1.0)))
    if resolved == GAUSSIAN:
        coverage = float(params.get(f"interval_nominal_coverage_{target}", 0.95))
        return GaussianInterval(alpha=1.0 - coverage)
    if resolved == CONFORMAL:
        # No q in the params - it is only in uncertainty_summary.json, which _calibrator_for reads.
        # Returning a coverage-carrying stand-in is enough for `describe`, which is all this is for.
        coverage = float(params.get(f"interval_nominal_coverage_{target}", 0.95))
        return ConformalCalibrator(q=float("nan"), alpha=1.0 - coverage, n_calib=0)
    return None


def _calibrator_for(run_id: str, artifact_path: str) -> Optional[ConformalCalibrator]:
    """Rebuild this run's own interval calibrator from what it recorded.

    With it, the reliability figure grades the procedure the run actually used rather than a
    reconstruction of it.
    """
    local_path = _download(run_id, f"{artifact_path}/{ArtifactLayout.UNCERTAINTY_SUMMARY_FILE}")
    if local_path is None:
        return None
    try:
        with open(local_path, encoding="utf-8") as handle:
            return ConformalCalibrator.from_dict(json.load(handle))
    except (OSError, ValueError):
        return None


def regenerate_child_figures(
    run: Any,
    *,
    only: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Redraw one model's figures.

    Parameters
    ----------
    run : mlflow.entities.Run
        The run to redraw; its tags say which target and model it is.
    kinds : sequence of str, optional
        Which figures to redraw; all of them unless given.
    dry_run : bool, default False
        Report what would be written without writing anything.

    Returns
    -------
    dict
        What was written, or why nothing was.
    """
    run_id = run.info.run_id
    target = run.data.tags.get("target")
    model_name = run.data.tags.get("model_name")
    outcome: dict[str, Any] = {"run_id": run_id, "run_name": run.info.run_name, "written": []}

    if not target or not model_name:
        # A model run in a joint fit carries both tags; a container run that scored nothing may not.
        outcome["skipped"] = "run carries no target/model_name tag"
        return outcome

    if len(split_target_names(target) or [target]) > 1:
        # A joint fit's MODEL run: it holds the fitted model and the joint eval frame, but it draws
        # no pred-vs-obs of its own - each of its per-target children does, from its own slice.
        # Reached only by --run-id; the tree walk returns the children instead.
        outcome["skipped"] = "joint model run; its per-target children hold the figures"
        return outcome

    outcome.update(target=target, model_name=model_name)

    evaluation_df = None
    if _wanted(only, "pred_obs") or _wanted(only, "uncertainty"):
        evaluation_df = _read_first(run_id, _eval_results_artifact_paths(target, model_name))
        if evaluation_df is None:
            outcome["skipped"] = "no eval_results.csv"
            return outcome

    cv_results = None
    if _wanted(only, "cv"):
        cv_results = _read_first(run_id, candidate_artifact_paths(ArtifactLayout.CV, ArtifactLayout.CV_RESULTS_FILE))

    if evaluation_df is None and cv_results is None:
        outcome["skipped"] = "nothing to redraw for the requested kinds"
        return outcome

    logger = ChildRunLogger()
    frame = _own_target_frame(logger, evaluation_df, target, model_name)
    if evaluation_df is not None and frame is None:
        outcome["skipped"] = f"eval_results.csv holds no column for {target}"
        return outcome

    has_sigma = frame is not None and sigma_column(frame, target) is not None

    if dry_run:
        outcome["would_write"] = _planned(has_sigma, cv_results, only)
        return outcome

    # start_run(run_id=...) on a FINISHED run flips it to RUNNING; the context manager sets it back
    # to FINISHED on the way out, so the run's terminal status survives a replot.
    with mlflow.start_run(run_id=run_id):
        if frame is not None:
            if _wanted(only, "pred_obs") and logger._log_pred_obs_artifact(
                frame,
                target=target,
                model_name=model_name,
                artifact_path=ArtifactLayout.plots_path(),
                interval_label=describe_interval(_interval_estimator(run, target)),
            ):
                outcome["written"].append(f"{ArtifactLayout.PLOTS}/{ArtifactLayout.PRED_OBS_FILE}")

            if _wanted(only, "uncertainty") and has_sigma:
                written = logger._log_uncertainty_artifacts(
                    frame=frame,
                    target=target,
                    multi_target=False,
                    calibrator=_calibrator_for(run_id, ArtifactLayout.UNCERTAINTY),
                )
                outcome["written"].extend(written.get("artifacts", []))

        if cv_results is not None:
            outcome["written"].extend(_log_cv_figures(logger, cv_results, target, model_name))

    return outcome


def _own_target_frame(
    logger: ChildRunLogger, evaluation_df: Optional[pd.DataFrame], target: str, model_name: str
) -> Optional[pd.DataFrame]:
    """The results this run is responsible for, chosen by its own target tag.

    A run covers exactly one target, even when its model predicted several.
    """
    if evaluation_df is None:
        return None
    frames = list(logger._iter_target_eval_frames(evaluation_df, target, model_name))
    for frame, target_name, _column in frames:
        if target_name == target:
            return frame
    return frames[0][0] if len(frames) == 1 else None


def _planned(has_sigma: bool, cv_results, only) -> list[str]:
    """The files a real run would write, listed without writing any of them."""
    planned: list[str] = []
    if _wanted(only, "pred_obs"):
        planned.append(f"{ArtifactLayout.PLOTS}/{ArtifactLayout.PRED_OBS_FILE}")
    if _wanted(only, "uncertainty") and has_sigma:
        planned.append(f"{ArtifactLayout.UNCERTAINTY}/{ArtifactLayout.RELIABILITY_FILE}")
        planned.append(f"{ArtifactLayout.UNCERTAINTY}/{ArtifactLayout.SIGMA_ERROR_FILE}")
    if cv_results is not None and _cv_plot_name(cv_results) is not None:
        planned.append(f"{ArtifactLayout.PLOTS}/{_cv_plot_name(cv_results)}.png")
    return planned


def _cv_plot_name(cv_results: pd.DataFrame) -> Optional[str]:
    """Which search figure this model gets, or None when it searched nothing.

    The same choice the trainer makes: a line for one searched setting, a parallel-coordinates figure
    for several.
    """
    param_columns = [column for column in cv_results.columns if str(column).startswith("param_")]
    if not param_columns:
        return None
    return "cv_val_curve" if len(param_columns) == 1 else "cv_parallel_coordinates"


def _log_cv_figures(logger: ChildRunLogger, cv_results: pd.DataFrame, target, model_name) -> list[str]:
    """Redraw and upload the search figures for one model."""
    name = _cv_plot_name(cv_results)
    if name is None:
        return []
    logger._log_plots({f"yg_eo_soilnet.plot_utils.{name}": {"args": [cv_results]}}, target, model_name)
    return [f"{ArtifactLayout.PLOTS}/{name}.png"]


def regenerate_parent_figures(
    parent_run_id: str,
    *,
    only: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Redraw a run's leaderboard figure and its combined predicted-against-measured figure."""
    outcome: dict[str, Any] = {"run_id": parent_run_id, "written": []}
    if not _wanted(only, "leaderboard"):
        outcome["skipped"] = "leaderboard not among the requested kinds"
        return outcome

    planned = [
        f"{ArtifactLayout.LEADERBOARD_PLOTS}/leaderboard.png",
        f"{ArtifactLayout.LEADERBOARD_PLOTS}/pred_error_plot.png",
        "leaderboard.csv",
    ]
    if dry_run:
        outcome["would_write"] = planned
        return outcome

    with mlflow.start_run(run_id=parent_run_id):
        # The same method training calls, so the two paths cannot drift into drawing different
        # figures from the same data.
        ParentRunLogger().log_parent_figures(parent_run_id)
    outcome["written"] = planned
    return outcome


def scoring_descendants(parent_run_id: str) -> list:
    """Every run under a main run that carries results.

    Walks the same two levels the leaderboard does, since a model predicting several targets keeps its
    results a level deeper.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_run(parent_run_id).info.experiment_id

    def children_of(run_id: str):
        """The sub-runs of one run."""
        return client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        )

    runs = []
    for child in children_of(parent_run_id):
        grandchildren = _scoring_runs(children_of(child.info.run_id))
        runs.extend(grandchildren or [child])
    return runs


def regenerate_tree(
    parent_run_id: str,
    *,
    only: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> list[dict]:
    """Redraw every figure under one main run: each model's, then the run's own.

    The run's own go last, because they are built from the models' tables.
    """
    outcomes = [regenerate_child_figures(run, only=only, dry_run=dry_run) for run in scoring_descendants(parent_run_id)]
    outcomes.append(regenerate_parent_figures(parent_run_id, only=only, dry_run=dry_run))
    return outcomes
