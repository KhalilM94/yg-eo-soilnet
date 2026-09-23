"""Search for the best hyperparameters of a deep-learning model, with Optuna.

Run it from the repository root::

    python tune.py --entry soil_cnn --n-trials 200

``--entry`` names a model in the deep-learning :term:`model list <model registry>` and its
:term:`search space` in ``configs/lightning/search_spaces/``. Each :term:`trial` trains the model
once with one combination of hyperparameters, on the same data and split as ``main.py``, and is
scored on the validation set (``val_loss`` by default). Unpromising trials are stopped early.

The best trial is written as a ready-to-train model-list file, named after the :term:`study`::

    LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py

A study is named ``<entry>-<fingerprint>``, where the :term:`fingerprint` summarises the search
space. Running the same command again adds trials to the same study; editing the search space
starts a new one. ``--reset`` deletes a study; ``--study-name`` chooses the name yourself.
Studies are stored in ``optuna_studies/soilnet.db`` and logged to the MLflow experiment
``Soil_HPO_Experiment``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import mlflow

from config import Config
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.hpo.data import build_lightning_input
from yg_eo_soilnet.hpo.export import export_best_config
from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.progress import MODES, StudyProgress
from yg_eo_soilnet.hpo.rerank import choose_winner, rerank, rerank_frame
from yg_eo_soilnet.hpo.search_space import SearchSpace
from yg_eo_soilnet.hpo.study import (
    DEFAULT_STORAGE,
    create_or_load_study,
    default_study_name,
    reset_study,
    run_study,
    set_hpo_experiment,
    summarize,
)
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker, best_value_or_none
from yg_eo_soilnet.hpo.trial_runner import UnrecoverableAcceleratorError, silence_lightning
from yg_eo_soilnet.logger.training_logger import TrainingLogger


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Search for the best hyperparameters of a deep-learning model, with Optuna."
    )
    parser.add_argument(
        "--entry", required=True, help="The model to tune, as named in the deep-learning model list, e.g. soil_cnn."
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main configuration file (default: configs/main_config.yml); the data and split come from it.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Which target group to tune, e.g. clay_pct. Needed only with MULTI_TARGET_MODE: "
            "per_target, where each target has its own model: a study tunes one model at a time."
        ),
    )
    parser.add_argument(
        "--search-spaces",
        default="configs/lightning/search_spaces",
        help=(
            "Where the search spaces are (default: configs/lightning/search_spaces): a folder with "
            "one file per model, or a single YAML file holding several."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="How many trials to run now (default 50); they are added to the study if it already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop starting new trials after this many seconds; the running trial is finished first.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Study name (default: <entry>-<fingerprint of the search space>, so editing the search "
        "space starts a new study). A name you choose is kept across edits; an incompatible edit "
        "is refused.",
    )
    parser.add_argument(
        "--storage",
        default=DEFAULT_STORAGE,
        help=f"Where studies are stored, as an Optuna storage URL (default: {DEFAULT_STORAGE}).",
    )
    parser.add_argument("--reset", action="store_true", help="Delete the study, and all its trials, before running.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for the trials (default: RANDOM_SEED from the config)."
    )
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=1,
        help="Train each trial this many times with different seeds and score the average (default 1).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs per trial, overriding the search space. This changes the study's "
        "fingerprint, so it starts a new study.",
    )
    parser.add_argument(
        "--cache-datamodules",
        action="store_true",
        help="Reuse the prepared data between trials whose data settings (such as batch size) are "
        "the same; saves time.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop with the error when a trial fails, instead of marking it failed and going on.",
    )
    parser.add_argument(
        "--progress",
        choices=MODES,
        default="auto",
        help="How progress is shown: bar, plain (one line per trial), none, or auto (bars in a "
        "terminal, plain when the output goes to a file; the default).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="How many of the best trials to list at the end (default 10)."
    )
    parser.add_argument("--verbose", action="store_true", help="Show Lightning's own training output for every trial.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Do not log anything to MLflow; trials are still saved in the study storage.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Write the tuned config for the best trial of an existing study and stop, without "
        "loading data or running trials.",
    )
    parser.add_argument(
        "--rerank-top",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Re-train the K best trials of an existing study with several seeds each and export "
            "the one with the best average score, instead of the single best trial. Runs no new "
            "trials; costs K x --rerank-seeds training runs."
        ),
    )
    parser.add_argument(
        "--rerank-seeds",
        type=int,
        default=3,
        metavar="N",
        help="Seeds per trial when re-ranking (default 3); the first is the trial's own seed.",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Where to write the tuned config (default: configs/lightning/tuned/<study name>_best.yml, "
        "or _reranked.yml after --rerank-top).",
    )
    return parser.parse_args()


def main() -> None:
    """Run (or continue) a study, then export the best trial as a tuned config."""
    args = parse_args()
    if not args.verbose:
        silence_lightning()

    config = Config(config_path=args.config_path)
    logger = TrainingLogger(
        name=f"AlMoutmir HPO - {args.entry}",
        log_filename=f"hpo_{args.entry}",
        enable_file_logging=config.MAIN_FILE_LOGGING_ENABLED,
    ).get_logger()

    space = SearchSpace.from_yaml(args.search_spaces, args.entry)
    if args.max_epochs is not None:
        space.fixed["trainer.max_epochs"] = args.max_epochs

    if args.entry not in config.LIGHTNING_MODEL_REGISTRY:
        available = ", ".join(sorted(config.LIGHTNING_MODEL_REGISTRY))
        raise SystemExit(f"No entry '{args.entry}' in {config.lightning_registry_path}. Available: {available}")
    registry_entry = config.LIGHTNING_MODEL_REGISTRY[args.entry]

    study_name = args.study_name or default_study_name(args.entry, space)
    if args.reset:
        reset_study(study_name, args.storage, logger)

    if args.export_only:
        # Re-exporting needs only the stored study and the model entry, not the data.
        study = create_or_load_study(space, study_name, args.storage)
        tracker = ObjectiveTracker(space.objective)
        tracker.prime(study)
        logger.info(f"Re-exporting study '{study_name}' ({len(study.trials)} trials) without running any")
        report_and_export(study, space, tracker, deepcopy(registry_entry), args, config, logger)
        return

    # Data preparation logs the split to MLflow, so it is done inside a named run of the tuning
    # experiment. Under --no-mlflow it still has somewhere to write - the splitter logs the split
    # whoever calls it - so tracking is pointed at a scratch directory instead of being skipped:
    # skipping it left the split's own artifacts to open an unnamed run in the real store, which is
    # the thing the flag is meant to prevent. The directory is left for the OS to reap, as the run
    # logs are.
    record = not args.no_mlflow
    if record:
        set_hpo_experiment(config=config)
    else:
        scratch = Path(tempfile.mkdtemp(prefix="yg_eo_soilnet_no_mlflow_"))
        logger.info(f"--no-mlflow: nothing is recorded; MLflow calls go to {scratch}")
        set_hpo_experiment(config=SimpleNamespace(MLFLOW_TRACKING_URI=scratch.as_uri()))

    # Load and split the data exactly as main.py does, once; every trial reuses it.
    stage_start = time.perf_counter()
    with mlflow.start_run(run_name=f"{study_name}_data") if record else nullcontext():
        scikit_datamodule = ScikitDataModule(config, logger, DataManager(config, logger))
        split_data = scikit_datamodule.prepare()
        # Every trial uses this same split, so trials are compared on the same validation points.
        # (That is also why the search space may not change the split.)
        if record:
            mlflow.log_params(split_data["split_plan"].describe())
        data = build_lightning_input(
            args.entry, registry_entry, config, split_data, logger=logger, data_manager=scikit_datamodule.data_manager
        )
    logger.info(f"Data prepared in {time.perf_counter() - stage_start:.2f}s")

    context = ObjectiveContext.from_config(
        args.entry,
        config,
        data=data,
        target=args.target,
        logger=logger,
        data_manager=scikit_datamodule.data_manager,
        datamodule_cache={} if args.cache_datamodules else None,
    )
    tracker = ObjectiveTracker(space.objective)
    progress = StudyProgress(
        n_trials=args.n_trials,
        objective=space.objective,
        tracker=tracker,
        mode=args.progress,
        logger=logger,
    )
    objective = TrialObjective(
        context,
        space,
        seed=args.seed if args.seed is not None else int(config.RANDOM_SEED),
        seed_repeats=args.seed_repeats,
        fail_fast=args.fail_fast,
        progress=progress,
    )

    if args.rerank_top is not None:
        rerank_and_export(objective, space, args, config, logger, study_name)
        return

    logger.info(f"Tuning '{args.entry}' to {space.objective.direction} {space.objective.metric}")
    try:
        study = run_study(
            objective,
            space,
            study_name=study_name,
            storage=args.storage,
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_mlflow=not args.no_mlflow,
            extra_params={"entry": args.entry, "target": context.target, "seed_repeats": args.seed_repeats},
            logger=logger,
            tracker=tracker,
            progress=progress,
            artifact_dir=Path("optuna_studies") / study_name,
            config=config,
        )
    except BaseException as error:
        # Also on Ctrl-C: the best trial so far is still exported.
        handle_study_abort(error, space, tracker, context, args, config, logger, study_name)
        raise

    report_and_export(study, space, tracker, context.registry_entry, args, config, logger)


def rerank_and_export(objective, space, args, config, logger, study_name: str) -> None:
    """Re-train the best trials with several seeds and export the one with the best average score.

    A study's best score is the best of many noisy trials, so a retrain usually does a little worse.
    Re-training the shortlist with several seeds picks the settings that hold up, and reports the
    score to expect from a retrain. See :term:`rerank`.

    Parameters
    ----------
    objective : yg_eo_soilnet.hpo.objective.TrialObjective
        Trains and scores one set of hyperparameters.
    space : yg_eo_soilnet.hpo.search_space.SearchSpace
        The study's search space.
    args : argparse.Namespace
        The command-line options (``--rerank-top``, ``--rerank-seeds``, ``--export-path``, ...).
    config : config.Config
        The run configuration.
    logger : logging.Logger
        Where progress is reported.
    study_name : str
        The study to re-rank; it must already exist.

    Raises
    ------
    SystemExit
        If none of the shortlisted trials could be re-trained.
    """
    study = create_or_load_study(space, study_name, args.storage)
    logger.info(
        f"Re-ranking the top {args.rerank_top} of {len(study.trials)} trials in '{study_name}' "
        f"over {args.rerank_seeds} seed(s) each - {args.rerank_top * args.rerank_seeds} training runs"
    )

    results = rerank(objective, study, top_k=args.rerank_top, seeds=args.rerank_seeds, logger=logger)
    if not results:
        raise SystemExit("No shortlisted trial could be re-run, so there is nothing to export.")

    frame = rerank_frame(results, metric=space.objective.metric)
    logger.info(f"Re-ranked shortlist:\n{frame.to_string(index=False)}")

    not_reproduced = [r.trial_number for r in results if r.reproduced is False]
    if not_reproduced:
        # The first seed of each trial is its own, so its score should come out the same.
        logger.warning(
            f"Trials {not_reproduced} did not reproduce at their own seed - runs are not "
            f"reproducible from their recorded seed, so treat the re-ranked means as noisy."
        )

    winner = choose_winner(results, space.objective.direction)
    logger.info(
        f"Winner: trial {winner.trial_number} | headline {winner.original_value:.6f} -> "
        f"re-ranked {winner.mean:.6f} +- {winner.std:.6f}"
    )
    if winner.trial_number != study.best_trial.number:
        logger.info(
            f"That is NOT the study's best trial ({study.best_trial.number}, "
            f"{study.best_value:.6f}) - the headline ranking did not survive re-running."
        )

    artifact_dir = Path("optuna_studies") / study_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifact_dir / "rerank.csv", index=False)

    export_path = Path(args.export_path or f"configs/lightning/tuned/{study_name}_reranked.yml")
    export_best_config(
        study,
        args.entry,
        deepcopy(config.LIGHTNING_MODEL_REGISTRY[args.entry]),
        space.objective,
        export_path,
        registry_path=config.lightning_registry_path,
        rerank=winner,
    )
    logger.info(f"Exported to {export_path}")
    print(f"\nRe-ranked winner: trial {winner.trial_number}")
    print(f"  headline   {space.objective.metric} = {winner.original_value:.6f}  (best of {len(study.trials)} trials)")
    print(f"  expect     {space.objective.metric} = {winner.mean:.6f} +- {winner.std:.6f}  on a retrain")
    print(f"Tuned config: {export_path}")
    print(f"  rerank table: {artifact_dir / 'rerank.csv'}")


def handle_study_abort(error, space, tracker, context, args, config, logger, study_name: str) -> None:
    """Export the best completed trial of a study that stopped early, then let the caller re-raise.

    Every finished trial is already saved in the study storage, so a study interrupted by Ctrl-C or
    a crash still exports its best result. Failing to export is logged, not raised, so the original
    error stays visible. If the GPU stopped working, the run ends with a short message saying how to
    recover instead of a long traceback.
    """
    try:
        study = create_or_load_study(space, study_name, args.storage)
        if best_value_or_none(study) is None:
            logger.warning("Study aborted with no completed trial, so there is nothing to export.")
        else:
            logger.warning(f"Study aborted - exporting the best of its {len(study.trials)} trials anyway.")
            report_and_export(study, space, tracker, context.registry_entry, args, config, logger)
    except Exception as export_error:
        logger.error(f"Could not export after the study aborted: {export_error!r}", exc_info=True)

    if isinstance(error, UnrecoverableAcceleratorError):
        raise SystemExit(str(error)) from error


def report_and_export(study, space, tracker, registry_entry, args, config, logger) -> None:
    """Log the best trials and their hyperparameters, and write the tuned config file.

    Raises
    ------
    SystemExit
        If no trial of the study completed.
    """
    summary = summarize(study, space)
    logger.info(f"Study finished: {summary}")
    if summary["best_trial"] is None:
        raise SystemExit("No trial completed, so there is nothing to export.")

    top_frame = tracker.top_frame(study, n=args.top_n)
    logger.info(f"Top {len(top_frame)} trials:\n{top_frame.to_string(index=False)}")
    best_params = "\n".join(f"  {key} = {value}" for key, value in sorted(tracker.best_params(study).items()))
    logger.info(f"Best trial {study.best_trial.number} parameters:\n{best_params}")

    # Named after the study, not the model, so studies of the same model never overwrite each
    # other's tuned file.
    export_path = Path(args.export_path or f"configs/lightning/tuned/{study.study_name}_best.yml")
    export_best_config(
        study,
        args.entry,
        registry_entry,
        space.objective,
        export_path,
        registry_path=config.lightning_registry_path,
    )
    logger.info(f"Best {space.objective.metric}={study.best_value:.6f} (trial {study.best_trial.number})")
    logger.info(f"Exported to {export_path}")
    print(f"\nBest {space.objective.metric} = {study.best_value:.6f}  (trial {study.best_trial.number})")
    print(f"Tuned config: {export_path}")
    print(f"Train it with:\n  LIGHTNING_MODEL_REGISTRY_PATH={export_path} python main.py")


if __name__ == "__main__":
    main()
