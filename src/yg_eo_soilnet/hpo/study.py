"""Create, resume and run a :term:`study`, and record it.

Studies are kept in a small database file, so a search can be stopped and resumed, run from several
processes at once, and looked at afterwards. MLflow gets one run per study rather than one per
trial: a few hundred runs would bury every training run in the experiment.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import mlflow
import optuna

from yg_eo_soilnet.hpo.plots import write_study_artifacts
from yg_eo_soilnet.hpo.search_space import SearchSpace
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker, best_value_or_none

HPO_EXPERIMENT_NAME = "Soil_HPO_Experiment"
DEFAULT_STORAGE = "sqlite:///optuna_studies/soilnet.db"

# Study user-attr holding the fingerprint of the search space its trials were drawn from.
FINGERPRINT_ATTR = "space_fingerprint"

# MLflow rejects param values past this length.
_MAX_PARAM_CHARS = 500


def ensure_storage_directory(storage: str) -> str:
    """Create the folder the study's database goes in, so the first run does not fail."""
    prefix = "sqlite:///"
    if storage.startswith(prefix) and not storage.endswith(":memory:"):
        Path(storage[len(prefix) :]).parent.mkdir(parents=True, exist_ok=True)
    return storage


def default_study_name(entry: str, space: SearchSpace) -> str:
    """The study's name: the model, plus a digest of the search space it is tuned with.

    Resuming works by name, so an edited search space starts a clean study rather than mixing trials
    run under different rules. A name looks like ``soil_cnn-a3f19c``.

    Parameters
    ----------
    entry : str
        The model being tuned.
    space : SearchSpace
        Its search space, whose :meth:`~yg_eo_soilnet.hpo.search_space.SearchSpace.fingerprint`
        becomes the suffix.

    Returns
    -------
    str
        """
    return f"{entry}-{space.fingerprint()}"


def create_or_load_study(space: SearchSpace, study_name: str, storage: str) -> optuna.Study:
    """Open the study, resuming it when it exists - which is what running the same command twice does."""
    study = optuna.create_study(
        study_name=study_name,
        storage=ensure_storage_directory(storage),
        direction=space.objective.direction,
        sampler=space.make_sampler(),
        pruner=space.make_pruner(),
        load_if_exists=True,
    )

    # Optuna keeps the STORED direction when it loads an existing study and silently discards the
    # one asked for - no warning. Editing an objective from maximize to minimize and re-running
    # under the same --study-name would resume the old study still maximizing, i.e. selecting the
    # worst model, and report it as best. Refuse instead.
    stored = study.direction.name.lower()
    if stored != space.objective.direction:
        raise ValueError(
            f"Study {study_name!r} already exists in {storage} with direction {stored!r}, but the "
            f"search space asks to {space.objective.direction} {space.objective.metric!r}. Optuna "
            f"would silently keep {stored!r} and optimize the wrong way. Use a new --study-name for "
            f"the new objective, or delete the old study with --reset."
        )

    # The fingerprint catches what the naming scheme cannot: an explicit --study-name held across an
    # edit to the space. Direction alone is far too weak a guard - two `maximize` studies on
    # different metrics, or on different bounds, pass it and then rank against each other.
    fingerprint = space.fingerprint()
    stored_fingerprint = study.user_attrs.get(FINGERPRINT_ATTR)
    if stored_fingerprint is None:
        study.set_user_attr(FINGERPRINT_ATTR, fingerprint)
    elif stored_fingerprint != fingerprint:
        raise ValueError(
            f"Study {study_name!r} in {storage} holds {len(study.trials)} trial(s) drawn from search "
            f"space {stored_fingerprint!r}, but {space.entry!r} now fingerprints as {fingerprint!r}. "
            f"Resuming would rank trials from two different spaces against each other and export a "
            f"winner from either. Drop --study-name to get a study named for the new space, or pass "
            f"--reset to discard the old one."
        )
    return study


def study_state(study: optuna.Study) -> str:
    """A short line saying whether this study is new or being resumed, and how many trials it holds."""
    existing = len(study.trials)
    return "(new)" if existing == 0 else f"(resuming, {existing} trials on record)"


def reset_study(study_name: str, storage: str, logger: Any = None) -> None:
    """Delete a study so the next run starts clean. A study that is not there is not an error."""
    try:
        optuna.delete_study(study_name=study_name, storage=ensure_storage_directory(storage))
    except KeyError:
        if logger is not None:
            logger.info(f"--reset: no study {study_name!r} in {storage} to delete")
        return
    if logger is not None:
        logger.info(f"--reset: deleted study {study_name!r} from {storage}")


def set_hpo_experiment(name: str = HPO_EXPERIMENT_NAME) -> None:
    """Point MLflow at the tuning experiment, and close any run left open.

    Called before the data is prepared, and even when the study itself is not being recorded, so
    nothing lands in the training experiment by accident.
        """
    from yg_eo_soilnet.tracking import configure_tracking

    # Through configure_tracking so HPO records into the same tracking root as training, and so the
    # MLflow 3.14 file-store opt-in is applied here too rather than depending on the launcher.
    configure_tracking(experiment_name=name)
    if mlflow.active_run():
        mlflow.end_run()


def _log_params(params: dict[str, Any]) -> None:
    """Record a batch of settings on the study's run, as text."""
    mlflow.log_params({key: str(value)[:_MAX_PARAM_CHARS] for key, value in params.items()})


def _mlflow_trial_callback() -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Build the callback that records each trial's score on the study's run."""
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Record one finished trial's score."""
        if trial.value is not None:
            mlflow.log_metric("objective", float(trial.value), step=trial.number)
        # best_value_or_none, not `study.best_value is not None`: Optuna raises when no trial has
        # completed, so the plain guard blows up before it compares - which a study whose first
        # trial is pruned hits on trial 0.
        best_value = best_value_or_none(study)
        if best_value is not None:
            mlflow.log_metric("objective_best", best_value, step=trial.number)

    return callback


def summarize(study: optuna.Study, space: SearchSpace) -> dict[str, Any]:
    """A short report of how the study went: the best value, which trial, and how many ran."""
    states = [trial.state.name for trial in study.trials]
    summary = {
        "study": study.study_name,
        "trials": len(study.trials),
        "complete": states.count("COMPLETE"),
        "pruned": states.count("PRUNED"),
        "failed": states.count("FAIL"),
        f"best_{space.objective.metric}": None,
        "best_trial": None,
    }
    best_value = best_value_or_none(study)
    if best_value is not None:
        summary[f"best_{space.objective.metric}"] = best_value
        summary["best_trial"] = study.best_trial.number
    return summary


def run_study(
    objective: Callable[[optuna.Trial], float],
    space: SearchSpace,
    *,
    study_name: str,
    storage: str = DEFAULT_STORAGE,
    n_trials: int = 50,
    timeout: float | None = None,
    use_mlflow: bool = True,
    extra_params: dict[str, Any] | None = None,
    logger: Any = None,
    tracker: ObjectiveTracker | None = None,
    progress: Any = None,
    artifact_dir: str | Path | None = None,
) -> optuna.Study:
    """Run the study: draw, train and score trials until the budget is used up.

    Parameters
    ----------
    study : optuna.Study
        The study to run.
    objective : TrialObjective
        What one trial does.
    n_trials : int
        How many trials to run in this session.
    callbacks : list, optional
        Extra things to call when a trial finishes, such as the progress display.

    Returns
    -------
    optuna.Study
        """
    study = create_or_load_study(space, study_name, storage)
    # Read before optimize() adds any: n_trials is an increment on whatever is already stored, and
    # a resumed study's sparkline and counts include that history.
    if logger is not None:
        logger.info(f"Study '{study_name}' {study_state(study)}: running {n_trials} trials")
    if tracker is not None:
        tracker.prime(study)

    # One flow rather than a --no-mlflow short circuit: the tracker and the progress display must
    # be registered either way, and previously that branch returned before any callback was added.
    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
    if tracker is not None:
        callbacks.append(tracker.record)
    if progress is not None:
        # After the tracker's, so the bar's postfix reflects this trial.
        callbacks.append(progress.on_trial_end)
    if use_mlflow:
        set_hpo_experiment()
        callbacks.append(_mlflow_trial_callback())

    run_context = mlflow.start_run(run_name=study_name) if use_mlflow else nullcontext()
    progress_context = progress if progress is not None else nullcontext()

    with run_context, progress_context:
        if use_mlflow:
            _log_params(
                {
                    "study_name": study_name,
                    "storage": storage,
                    "space_fingerprint": space.fingerprint(),
                    "objective_metric": space.objective.metric,
                    "objective_direction": space.objective.direction,
                    "n_trials_requested": n_trials,
                    **space.describe(),
                    **(extra_params or {}),
                }
            )
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
        finally:
            summary = summarize(study, space)
            best_value = best_value_or_none(study)
            if use_mlflow:
                _log_params({f"result.{key}": value for key, value in summary.items()})
                if best_value is not None:
                    _log_params({f"best.{key}": value for key, value in study.best_trial.params.items()})
                    mlflow.log_metric(f"best_{space.objective.metric}", best_value)
            if artifact_dir is not None and tracker is not None:
                write_study_artifacts(
                    study, tracker, artifact_dir, log_to_mlflow=use_mlflow, logger=logger
                )
            if logger is not None:
                logger.info(f"Study summary: {summary}")
    return study
