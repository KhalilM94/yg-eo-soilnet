"""Train every enabled model and record the results in MLflow.

The project's main entry point. Run it from the repository root::

    python main.py                                   # reads configs/main_config.yml
    python main.py --config-path examples/demo_config/main_config.yml

One run:

1. reads the configuration (:class:`config.Config`);
2. loads the static, target and time-series files and selects the usable features;
3. assigns every point once to the training, validation or test set - one :term:`split` shared by
   every model, so all test scores are comparable;
4. resolves the :term:`target groups <target group>` (one model for all targets, or one per target);
5. trains every enabled scikit-learn model (hyperparameters chosen by cross-validation), then every
   enabled deep-learning model;
6. logs each model's test scores, figures and saved model to MLflow, and a :term:`leaderboard`
   comparing them.

Everything is logged under one MLflow :term:`main run` named ``Run_<date>_<time>``, with one sub-run
per trained model. Browse the results with ``pixi run mlflow``.
"""

from yg_eo_soilnet.trainers.sklearn_trainer import ModelTrainer
from yg_eo_soilnet.logger.training_logger import TrainingLogger
from yg_eo_soilnet.models.config_fatories.model_config_factory import ModelConfigFactory
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.utils import LogTransformer
from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger
from config import Config
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.seeding import seed_everything
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer
from yg_eo_soilnet.tracking import (
    close_stale_runs,
    configure_tracking,
    install_run_signal_handlers,
    log_params_once,
    repair_corrupt_runs,
    resolve_local_tracking_root,
    run_owner_tags,
    tracking_settings,
)
from yg_eo_soilnet.targets import join_target_names, resolve_target_groups
import mlflow
import datetime
import logging
import time
import shutil
import re
from pathlib import Path
from typing import Any, Dict
import argparse


try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _describe_rows_cols(value: Any) -> str:
    """Describe a table's size for the log, e.g. ``"rows=300 | cols=14"``."""
    rows = len(value) if hasattr(value, "__len__") else "n/a"
    columns = len(value.columns) if hasattr(value, "columns") else "n/a"
    return f"rows={rows} | cols={columns}"


def _describe_shapes(value: Any, x_key: str = "X", y_key: str = "y") -> str:
    """Describe the input (X) and target (y) table sizes held in a dict, for the log."""
    x_shape = getattr(value.get(x_key), "shape", "n/a") if isinstance(value, dict) else "n/a"
    y_shape = getattr(value.get(y_key), "shape", "n/a") if isinstance(value, dict) else "n/a"
    return f"X_shape={x_shape} | y_shape={y_shape}"


def _sanitize_path_component(value: str) -> str:
    """Make a name safe to use as a folder name: letters, digits, ``.``, ``_`` and ``-`` only."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "experiment"


def _resolve_main_relative_path(path_value: str) -> Path:
    """Return an absolute path unchanged, and a relative one relative to this file's folder."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _seed_everything(seed: int) -> None:
    """Set every random-number generator from ``seed``, so a run can be repeated exactly.

    A thin wrapper around :func:`yg_eo_soilnet.seeding.seed_everything`.
    """
    seed_everything(seed)


def _export_mlflow_run_folder(trainer, experiment_id: str, run_id: str, run_name: str) -> Path | None:
    """Copy this run's MLflow folder elsewhere, when ``MLFLOW_EXPERIMENT_EXPORT_ENABLED`` is on.

    The copy goes to ``<MLFLOW_EXPERIMENT_EXPORT_PATH>/<experiment name>/<run name>/`` and replaces
    any earlier copy of the same run. It only works when MLflow stores its runs in a local folder.

    Returns
    -------
    pathlib.Path or None
        Where the copy was written, or ``None`` if nothing was copied.
    """
    if not getattr(trainer.config, "MLFLOW_EXPERIMENT_EXPORT_ENABLED", False):
        return None

    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        return None

    tracking_root = resolve_local_tracking_root(mlflow.get_tracking_uri())
    if tracking_root is None:
        return None

    source_dir = tracking_root / experiment_id / run_id
    if not source_dir.exists():
        return None

    export_root = _resolve_main_relative_path(trainer.config.MLFLOW_EXPERIMENT_EXPORT_PATH)
    experiment_folder = _sanitize_path_component(experiment.name)
    run_folder = _sanitize_path_component(run_name)
    destination_dir = export_root / experiment_folder / run_folder
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    return destination_dir


class SoilModelTraining:
    """Everything one training run needs: configuration, data loading, the split and the model builders.

    Creating it reads the configuration, seeds every random-number generator and sets up the
    loggers, the data loader, the shared split and the two model factories. Nothing is loaded or
    trained until :func:`main` runs the steps in turn.

    Parameters
    ----------
    run_name : str
        Name of the run; also used to name the log files.
    config_path : str
        Main configuration file (``configs/main_config.yml`` by default).

    Examples
    --------
    >>> trainer = SoilModelTraining(config_path="examples/demo_config/main_config.yml")  # doctest: +SKIP
    >>> raw = trainer.scikit_datamodule.load_frame()                                    # doctest: +SKIP
    """

    def __init__(
        self,
        run_name: str = "Soil_Model_Training",
        config_path: str = "configs/main_config.yml",
    ):
        self.run_name = run_name
        self.config = Config(
            config_path=config_path,
        )
        _seed_everything(int(self.config.RANDOM_SEED))
        self.log_transformer = LogTransformer()
        self.logger_wrapper = TrainingLogger(
            name='AlMoutmir Soil Models Training',
            log_filename=self.run_name,
            enable_file_logging=self.config.MAIN_FILE_LOGGING_ENABLED,
        )
        self.logger = self.logger_wrapper.get_logger()
        self.sklearn_logger_wrapper = TrainingLogger(
            name='AlMoutmir Soil Models Training - sklearn',
            log_filename=f'{self.run_name}_sklearn',
            enable_file_logging=self.config.SKLEARN_FILE_LOGGING_ENABLED,
        )
        self.sklearn_logger = self.sklearn_logger_wrapper.get_logger()
        # Stated in the log because repeating a run means repeating this number.
        self.logger.info("Random seed for this run: %s", self.config.RANDOM_SEED)

        self.data_manager = DataManager(self.config, self.logger)
        # One split for the whole run, shared by both model families, so a point in the test set
        # is never a training point for the other family.
        self.split_plan_provider = SplitPlanProvider(self.config, self.logger, self.data_manager)
        self.scikit_datamodule = ScikitDataModule(
            self.config, self.logger, self.data_manager, split_plan_provider=self.split_plan_provider
        )
        self.model_configs = ModelConfigFactory(self.config.MODEL_REGISTRY, random_state=self.config.RANDOM_SEED)
        self.lightning_model_configs = LightningConfigFactory(
            self.config.LIGHTNING_MODEL_REGISTRY,
            self.config,
            logger=self.logger,
            data_manager=self.data_manager,
        )
        self.lightning_trainer = LightningTrainer(config=self.config, logger=self.logger)

    def train_models(self, data: Dict):
        """Train every enabled model, target group by target group.

        The scikit-learn models are trained first, then the deep-learning models. Each model
        records its own results in MLflow as it finishes; nothing is returned.

        Parameters
        ----------
        data : dict
            The split data from :meth:`ScikitDataModule.split
            <yg_eo_soilnet.datamodules.scikit.scikit_datamodule.ScikitDataModule.split>`: the input
            and target tables for each part of the split, the point ids, and the split itself.
        """
        self.logger.info("Starting full training process for all targets...")
        X_key = "X_train" if "X_train" in data else "X"
        trainer = ModelTrainer(
            config=self.config,
            columns_to_transform=self.config.COLUMNS_TO_TRANSFORM,
            enable_clustering =self.config.ENABLE_CLUSTERING,
            split_strategy=self.config.SPLIT_STRATEGY,
            seed=self.config.RANDOM_SEED,
            logger=self.sklearn_logger,
        )
        lightning_input = dict(data)

        # MULTI_TARGET_MODE decides whether one model predicts every target or each target gets its
        # own model. A scikit-learn model can only predict several targets at once if its entry
        # declares it can (`multi_target: native`); a deep-learning model always can.
        model_pipelines = self.model_configs.build_model_configs(
            num_features=data[X_key].shape[1],
            default_seed=self.config.RANDOM_SEED,
        )
        # Decided model by model: PLS and Ridge can predict several targets at once, TabICL cannot.
        # A model that cannot falls back to one model per target, with a warning.
        #
        # Models that end up with the same grouping are trained together, in one call per group,
        # so FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET can see every model tried for a target.
        sklearn_groups: dict[tuple, dict] = {}
        for model_name, pipeline in (model_pipelines or {}).items():
            groups = resolve_target_groups(
                self.config,
                self.config.MODEL_REGISTRY.get(model_name, {}),
                require_joint_support=True,
                logger=self.logger,
                entry_name=model_name,
            )
            for target_group in groups:
                sklearn_groups.setdefault(tuple(target_group), {})[model_name] = pipeline

        # The same decision, model by model, for the deep-learning models.
        lightning_groups: dict[tuple, list[str]] = {}
        for entry_name, spec in self.config.LIGHTNING_MODEL_REGISTRY.items():
            if not spec.get("enabled", False):
                continue
            for target_group in resolve_target_groups(self.config, spec):
                lightning_groups.setdefault(tuple(target_group), []).append(entry_name)

        # Recorded on the main run before training starts, so the run always says what it set out
        # to fit, even if it is stopped part-way.
        self._log_target_plan(sklearn_groups, lightning_groups)

        for index, (target_group, pipelines) in enumerate(sklearn_groups.items(), start=1):
            label = join_target_names(list(target_group))
            # A progress line per group: a slow model can take many minutes, and a silent log is
            # hard to tell apart from a stuck one.
            self.logger.info(f"[sklearn group {index}/{len(sklearn_groups)}] {label} - starting")
            started = time.perf_counter()
            trainer.train(
                target=label,
                targets=list(target_group),
                data=data,
                model_pipelines=pipelines,
            )
            self.logger.info(
                f"[sklearn group {index}/{len(sklearn_groups)}] {label} - "
                f"done in {(time.perf_counter() - started) / 60:.1f}min"
            )

        for index, (target_group, entry_names) in enumerate(lightning_groups.items(), start=1):
            # Each model is built from a fixed random seed, so two runs of the same config - and a
            # tuned config and the tuning trial it came from - start from the same point.
            label = join_target_names(list(target_group))
            self.logger.info(f"[lightning group {index}/{len(lightning_groups)}] {label} - starting")
            started = time.perf_counter()
            def build_bundles(seed: int, _label=label, _entries=entry_names):
                """Build this group's deep-learning models, starting from random seed ``seed``.

                Called once per copy of the model when uncertainty is switched on (each copy starts
                from a different seed); otherwise once. The data files are read only once and kept
                in memory, but the model inputs are rebuilt on every call.
                """
                return self.lightning_model_configs.build_lightning_configs(
                    target=_label,
                    data=lightning_input,
                    seed=seed,
                    entries=_entries,
                )

            lightning_model_bundles = build_bundles(int(self.config.RANDOM_SEED))
            if lightning_model_bundles:
                self.lightning_trainer.train(
                    target=label,
                    data=lightning_input,
                    model_bundles=lightning_model_bundles,
                    bundle_builder=build_bundles,
                )
            self.logger.info(
                f"[lightning group {index}/{len(lightning_groups)}] {label} - "
                f"done in {(time.perf_counter() - started) / 60:.1f}min"
            )

    def _log_target_plan(self, sklearn_groups: Dict, lightning_groups: Dict) -> None:
        """Record on the main MLflow run which targets each model family will predict.

        Seeing ``MULTI_TARGET_MODE: joint`` next to one-target scikit-learn groups shows that a
        model could not predict several targets at once and fell back to one model per target.

        Parameters
        ----------
        sklearn_groups, lightning_groups : dict
            Target groups as keys (tuples of target names), for each model family.
        """
        def describe(groups) -> str:
            """The target groups as one readable line, for the run's settings."""
            return " | ".join(join_target_names(list(group)) for group in groups) or "(none)"

        params = {
            "MULTI_TARGET_MODE": getattr(self.config, "MULTI_TARGET_MODE", "joint"),
            "TARGET_COLUMNS": ",".join(self.config.TARGET_COLUMNS) or "(none)",
            "sklearn_target_groups": describe(sklearn_groups),
            "lightning_target_groups": describe(lightning_groups),
        }
        self.logger.info(
            f"Target plan: mode={params['MULTI_TARGET_MODE']} | "
            f"sklearn={params['sklearn_target_groups']} | "
            f"lightning={params['lightning_target_groups']}"
        )
        try:
            # The only place TARGET_COLUMNS is recorded on the main run.
            log_params_once(params, logger=self.logger)
        except Exception as exc:  # pragma: no cover - never worth failing a run over
            self.logger.warning(f"Could not log the target plan: {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options (just ``--config-path``)."""
    parser = argparse.ArgumentParser(
        description=(
            "Train every enabled model on the configured data and record the results in MLflow."
        )
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help=(
            "Main configuration file to read (default: configs/main_config.yml). The other "
            "configuration files it names are looked up next to it."
        ),
    )
    return parser.parse_args()


def main():
    """Run the whole training pipeline once, as described at the top of this file."""
    args = parse_args()
    mlflow.enable_system_metrics_logging()
    # Where MLflow stores the runs, and under which experiment name. Set before any run starts.
    experiment_name = configure_tracking(tracking_settings(args.config_path))

    if mlflow.active_run():
        mlflow.end_run()

    # Tidy up after earlier runs that were killed part-way (for example when the computer ran out
    # of memory): they stay marked "running" forever otherwise. Pressing Ctrl-C on this run marks it
    # "killed" instead. This happens before the new run starts, with a simple logger, because the
    # run's own logger is only created further down.
    startup_logger = logging.getLogger(__name__)
    install_run_signal_handlers(startup_logger)
    # A run killed while writing can leave a damaged record that stops MLflow from listing runs, so
    # damaged records are repaired first.
    repair_corrupt_runs(experiment_name, startup_logger)
    close_stale_runs(experiment_name, startup_logger)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Run_{timestamp}"
    mlflow_logger = ParentRunLogger()
    with mlflow.start_run(run_name=run_name) as main_run:
        # Which computer and process own this run, so a later run can tell "still running" from
        # "abandoned".
        mlflow.set_tags(run_owner_tags())
        trainer = SoilModelTraining(
            run_name=run_name,
            config_path=args.config_path,
        )
        mlflow.log_param("CONFIG_PATH", trainer.config.config_path)
        for param_name, param_value in (
            ("DATA_SPEC_PATH", getattr(trainer.config, "data_spec_path", None)),
            ("SKLEARN_CONFIG_PATH", getattr(trainer.config, "sklearn_config_path", None)),
            ("LIGHTNING_CONFIG_PATH", getattr(trainer.config, "lightning_config_path", None)),
        ):
            if param_value is not None:
                mlflow.log_param(param_name, param_value)
        mlflow.log_param("REGISTRY_PATH", trainer.config.registry_path)
        lightning_registry_path = getattr(trainer.config, "lightning_registry_path", None)
        if lightning_registry_path is not None:
            mlflow.log_param("LIGHTNING_REGISTRY_PATH", lightning_registry_path)
        try:
            # Load the data files and keep the usable columns.
            stage_start = time.perf_counter()
            raw_data = trainer.scikit_datamodule.load_frame()
            trainer.logger.info(
                f"load_dataset completed in {time.perf_counter() - stage_start:.2f}s | {_describe_rows_cols(raw_data)}"
            )

            stage_start = time.perf_counter()
            processed_data = trainer.scikit_datamodule.preprocess(raw_data)
            trainer.logger.info(
                "preprocess_data completed in "
                f"{time.perf_counter() - stage_start:.2f}s | {_describe_shapes(processed_data)}"
            )
            trainer.logger.info("Running in full training mode...")

            # Decide the train/validation/test split once, for both model families, so every model
            # is scored on the same test points.
            stage_start = time.perf_counter()
            split_plan = trainer.split_plan_provider.plan()
            mlflow.log_params(split_plan.describe())
            trainer.logger.info(
                f"split plan built in {time.perf_counter() - stage_start:.2f}s | {split_plan.counts()}"
            )

            stage_start = time.perf_counter()
            split_data = trainer.scikit_datamodule.split(processed_data, split_plan)
            trainer.logger.info(
                f"split_data completed in {time.perf_counter() - stage_start:.2f}s | X_train={getattr(split_data.get('X_train'), 'shape', 'n/a')} | X_test={getattr(split_data.get('X_test'), 'shape', 'n/a')}"
            )

            stage_start = time.perf_counter()
            # Train every model. The deep-learning models build their own inputs from the data
            # files, but use the same split, which travels inside `split_data`.
            trainer.train_models(split_data)
            trainer.logger.info(f"train_models completed in {time.perf_counter() - stage_start:.2f}s")

            # The leaderboard and summary plots. Every model is already trained and recorded by now,
            # so a failure here is logged and noted on the run (tag `parent_summary_error`) rather
            # than stopping it.
            try:
                mlflow_logger.log_parent_summary(main_run.info.run_id, trainer)
            except Exception as summary_error:
                trainer.logger.error(
                    f"Parent summary failed after training completed: "
                    f"{type(summary_error).__name__}: {summary_error}",
                    exc_info=True,
                )
                mlflow.set_tag(
                    "parent_summary_error",
                    f"{type(summary_error).__name__}: {summary_error}"[:500],
                )

        except Exception as e:
            trainer.logger.error(f"An error occurred during training: {e}")
            raise
        if trainer.logger_wrapper.log_file:
            mlflow.log_artifact(trainer.logger_wrapper.log_file)

    experiment_id = getattr(main_run.info, "experiment_id", None)
    run_id = getattr(main_run.info, "run_id", None)
    if experiment_id is not None and run_id is not None:
        _export_mlflow_run_folder(trainer, experiment_id, run_id, run_name)

if __name__ == "__main__":
    main()