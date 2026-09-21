"""Redraw a finished run's figures from the artifacts it already holds.

Nothing here loads the source data, a checkpoint, or a model. It does not need to: every figure a
run emits is a pure function of two CSVs it already wrote. ``eval_results/eval_results.csv`` carries
the observations, the predictions and - when the run had uncertainty on - sigma and the interval
bounds; ``cv/cv_results.csv`` carries the sklearn hyper-parameter sweep. That is the whole input.

This is what makes a style change affordable. Restyling the plotters would otherwise leave every
run trained before the change holding the old picture, with retraining as the only way to refresh
it - which would also give a different model, so the figure and the metrics beside it would no
longer describe the same fit.

Figures are re-logged INTO the run they came from, replacing the file at the same artifact path. A
run whose artifacts cannot be read is skipped and counted, never raised on: a tree of 800 runs
always contains some that never got far enough to write an eval CSV, and one of those must not stop
the other 799 from being refreshed.
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
    return kind in (set(only) if only else set(ALL_KINDS))


def _download(run_id: str, artifact_path: str):
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    except Exception:
        return None


def _read_first(run_id: str, artifact_paths: Iterable[str]) -> Optional[pd.DataFrame]:
    """The first of several candidate artifact paths that resolves, read as a CSV.

    Always goes through the candidate list rather than a hardcoded path: three generations of
    artifact layout coexist in ``mlruns/``, and a run from any of them should replot.
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
    """The interval this run's bars actually came from, rebuilt from the params that recorded it.

    Only the LABEL on the plot needs this - "bar: ±1σ" against "bar: conformal 95%" - but that label
    is what makes the PICP printed beside it interpretable, because the same picture means different
    things under the three methods.

    The params live on the MODEL run, not on the per-target child, so this walks up one level. A
    single-target run is its own model run and the walk is a no-op.
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
    """The run's own conformal calibrator, rebuilt from ``uncertainty_summary.json``.

    Worth the extra download. Handed a calibrator, the reliability curve grades the procedure the
    run actually used; handed nothing, it falls back to Gaussian z-multiples of the raw sigma, which
    grades a different claim and draws a different line - so a replot without this would not match
    the figure it replaces even though neither is wrong.
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
    """Redraw one run's per-target figures. Returns what was written, or why nothing was.

    ``run`` is an ``mlflow.entities.Run``, because the target and model name come from its tags and
    fetching it again by id would be a second round trip for something the caller already has.
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
        cv_results = _read_first(
            run_id, candidate_artifact_paths(ArtifactLayout.CV, ArtifactLayout.CV_RESULTS_FILE)
        )

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
                outcome["written"].append(
                    f"{ArtifactLayout.PLOTS}/{ArtifactLayout.PRED_OBS_FILE}"
                )

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
    """The single frame this RUN is responsible for, selected by its own ``target`` tag.

    A run scopes exactly one target - ``_log_per_target_runs`` opens a child run per target as soon
    as a group has more than one - but the eval CSV it holds is the JOINT frame, carrying a
    ``prediction_<t>`` column for every target in the group and a ``target_names`` column listing
    them all. So iterating that frame the way the training code does yields every target in the
    group, and replotting all of them here would write six pictures into a run that owns one, five
    of them belonging to sibling runs.

    The fallback matters for the legacy single-target frames that carry no ``target_name`` at all:
    there is exactly one frame in that case and it is this run's.
    """
    if evaluation_df is None:
        return None
    frames = list(logger._iter_target_eval_frames(evaluation_df, target, model_name))
    for frame, target_name, _column in frames:
        if target_name == target:
            return frame
    return frames[0][0] if len(frames) == 1 else None


def _planned(has_sigma: bool, cv_results, only) -> list[str]:
    """The artifact paths a non-dry run would write, without writing any of them.

    Flat paths throughout, because that is what both training paths write: ``plots_path()`` is
    called with no target on either family, and every run in ``mlruns/`` has a flat ``plots/`` and
    ``uncertainty/``. Replotting has to land on the file it is replacing, not beside it.
    """
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
    """Which CV figure this sweep gets, or ``None`` for an estimator with nothing swept.

    The same three-way choice ``sklearn_trainer`` makes when training: several swept parameters get
    parallel coordinates, one gets a validation curve, and NONE gets no figure at all - a model
    fitted at fixed hyper-parameters has no sweep to draw. That last case is not hypothetical; it
    produced a degenerate one-axis parallel-coordinates plot and a matplotlib warning about
    identical axis limits before it was handled here.
    """
    param_columns = [column for column in cv_results.columns if str(column).startswith("param_")]
    if not param_columns:
        return None
    return "cv_val_curve" if len(param_columns) == 1 else "cv_parallel_coordinates"


def _log_cv_figures(logger: ChildRunLogger, cv_results: pd.DataFrame, target, model_name) -> list[str]:
    name = _cv_plot_name(cv_results)
    if name is None:
        return []
    logger._log_plots(
        {f"yg_eo_soilnet.plot_utils.{name}": {"args": [cv_results]}}, target, model_name
    )
    return [f"{ArtifactLayout.PLOTS}/{name}.png"]


def regenerate_parent_figures(
    parent_run_id: str,
    *,
    only: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Redraw the leaderboard scatter and the combined pred-vs-obs overlay on a parent run."""
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
    """Every run under a parent that holds a score, flattened.

    Walks the same two levels the leaderboard collector does - a joint fit keeps its per-target
    evaluation frames in grandchildren, and its model run holds the joint frame those were split
    from - and drops ensemble member runs, which carry their parent's target and model name but no
    test metrics of their own.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_run(parent_run_id).info.experiment_id

    def children_of(run_id: str):
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
    """Every figure under one parent: each scoring child, then the parent's own two.

    The parent goes LAST because its overlay is built from the children's eval CSVs, and doing it
    last means a reader comparing the parent figure against a child's is looking at two pictures
    drawn by the same code in the same pass.
    """
    outcomes = [
        regenerate_child_figures(run, only=only, dry_run=dry_run)
        for run in scoring_descendants(parent_run_id)
    ]
    outcomes.append(regenerate_parent_figures(parent_run_id, only=only, dry_run=dry_run))
    return outcomes
