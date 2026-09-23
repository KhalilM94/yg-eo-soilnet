"""Train the deep-learning models: fit, score, and record the results."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping

import mlflow
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover
    LightningCallback = object  # type: ignore[assignment]

from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningModelBundle
from yg_eo_soilnet.targets import join_target_names
from yg_eo_soilnet.tracking import start_child_run
from yg_eo_soilnet.predictions_export import export_enabled_for
from yg_eo_soilnet.uncertainty.intervals import (
    build_interval_estimators,
    needs_calibration_set,
    normalize_method,
)
from yg_eo_soilnet.uncertainty import (
    aggregate,
    attach_uncertainty_columns,
    fit_calibrators,
    member_seeds,
    uncertainty_enabled_for,
)

#: Tag marking a sub-run as one :term:`ensemble` member rather than one target's results, as on
#: the scikit-learn side. The leaderboard reads it.
MEMBER_RUN_KIND = "ensemble_member"


class _LightningMlflowEpochMetricCallback(LightningCallback):
    """Record the training, validation and test loss once per epoch, so the curves can be plotted."""

    def __init__(self):
        self._last_logged_epoch: dict[str, int] = {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Return a score as a plain number, or None when it is not a single value."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item() if getattr(value, "ndim", 0) == 0 else value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = float(value.reshape(-1)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _log_metrics(self, trainer, metric_names: tuple[str, ...]) -> None:
        """Record these scores for the current epoch, once each."""
        if getattr(trainer, "sanity_checking", False):
            return

        current_epoch = int(getattr(trainer, "current_epoch", 0))
        for metric_name in metric_names:
            if self._last_logged_epoch.get(metric_name) == current_epoch:
                continue
            metric_value = trainer.callback_metrics.get(metric_name)
            metric_float = self._to_float(metric_value)
            if metric_float is None:
                continue
            mlflow.log_metric(metric_name, metric_float, step=current_epoch)
            self._last_logged_epoch[metric_name] = current_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Record the training loss."""
        self._log_metrics(trainer, ("train_loss",))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Record the validation loss."""
        self._log_metrics(trainer, ("val_loss",))

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Record the test loss."""
        self._log_metrics(trainer, ("test_loss",))


@dataclass
class LightningRunResult:
    """What one trained deep-learning model left behind.

    Attributes
    ----------
    model_name : str
        The model's name in the model list.
    target : str
        The :term:`target group` it predicts.
    validation_metrics, test_metrics : dict of str to float
        Its scores. Note these are on the model's own :term:`training scale`, not the target's
        units; the scores in the target's units are computed by the logger.
    best_model_path : str or None
        The :term:`checkpoint` file kept - the best epoch, not the last.
    """

    model_name: str
    target: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_model_path: str | None


class LightningTrainer:
    """Train every deep-learning model switched on, one :term:`target group` at a time.

    Each model is trained inside a sub-run of its own, early-stopped on the validation points, then
    scored on the test points using its best epoch. With uncertainty on it trains an
    :term:`ensemble` of differently seeded models instead and calibrates the intervals.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger, optional
        Where progress messages go.
    mlflow_logger : ChildRunLogger, optional
        What records the results; a new one unless given.

    Examples
    --------
    >>> trainer = LightningTrainer(config, logger=logger)                  # doctest: +SKIP
    >>> results = trainer.train("clay_pct", data, bundles)                 # doctest: +SKIP
    >>> results["soil_cnn"].test_metrics["test_loss"]                      # doctest: +SKIP
    0.41
    """

    def __init__(self, config, logger=None, mlflow_logger=None):
        self.config = config
        self.logger = logger
        self.mlflow_logger = mlflow_logger or ChildRunLogger()

    def train(
        self,
        target: str,
        data: Mapping[str, Any],
        model_bundles: Mapping[str, LightningModelBundle],
        bundle_builder: Any = None,
    ):
        """Train every model :term:`bundle` for one :term:`target group`.

        Parameters
        ----------
        target : str
            The group's name.
        data : mapping
            The prepared data and the run's shared split.
        model_bundles : mapping of str to LightningModelBundle
            The built models, from
            :meth:`LightningConfigFactory.build_lightning_configs
            <yg_eo_soilnet.models.config_fatories.lightning_config_factory.LightningConfigFactory.build_lightning_configs>`.
        bundle_builder : callable, optional
            ``builder(seed) -> {model name: bundle}``. Needed for an :term:`ensemble`: a model's
            starting weights are drawn when it is built, so each member has to be built afresh at
            its own seed.

        Returns
        -------
        dict of str to LightningRunResult
        """
        results: dict[str, LightningRunResult] = {}

        for model_name, bundle in model_bundles.items():
            if bundle_builder is not None and uncertainty_enabled_for(self.config, model_name):
                results[model_name] = self._train_ensemble(
                    target=target,
                    model_name=model_name,
                    bundle_builder=bundle_builder,
                )
                continue
            # No seeding here: it would be too late to reach the weights, which were drawn when
            # the model was built, and would make a tuned configuration miss the trial that chose
            # it. The factory seeds just before building each model instead.
            run_name = f"{target}_{model_name}"
            with start_child_run(run_name):
                trainer = self._build_trainer(bundle)
                bundle.datamodule.setup("fit")
                # After the statistics are fitted and before training, so every checkpoint the
                # run writes carries them and the saved model can read raw data by itself.
                self._attach_preprocessing_state(bundle)
                trainer.fit(bundle.model, datamodule=bundle.datamodule)

                best_model_path = self._resolve_best_checkpoint(trainer)

                validation_metrics = self._normalize_metrics(
                    self._call_trainer_method(
                        trainer,
                        "validate",
                        bundle.model,
                        bundle.datamodule,
                        ckpt_path=best_model_path,
                    )
                )
                test_metrics = self._normalize_metrics(
                    self._call_trainer_method(
                        trainer,
                        "test",
                        bundle.model,
                        bundle.datamodule,
                        ckpt_path=best_model_path,
                    )
                )

                evaluation_df = self._build_evaluation_frame(bundle, trainer, target, ckpt_path=best_model_path)

                # One call whatever the number of targets: the logger keeps the model, the curves
                # and the overall scores here, and opens one sub-run per target when there are
                # several.
                self.mlflow_logger.log_lightning_child_run(
                    config=self.config,
                    target=target,
                    model_name=model_name,
                    evaluation_df=evaluation_df,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    best_model_path=best_model_path,
                    extra_params=self._serialize_params(bundle),
                    plot_functions={},
                    bundle=bundle,
                    model=bundle.model,
                )

                results[model_name] = LightningRunResult(
                    model_name=model_name,
                    target=target,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    best_model_path=best_model_path,
                )

        return results

    # --- uncertainty --------------------------------------------------------

    def _train_ensemble(
        self,
        *,
        target: str,
        model_name: str,
        bundle_builder: Any,
    ) -> LightningRunResult:
        """Train ``n_members`` models from different seeds and record them as one :term:`ensemble`.

        The members keep every training row: unlike a linear model, two networks started from
        different weights already end up in different places, so resampling the rows would cost
        accuracy to buy a spread they have anyway.

        Returns
        -------
        LightningRunResult
        """
        n_members = int(getattr(self.config, "UNCERTAINTY_N_MEMBERS", 5))
        stride = int(getattr(self.config, "UNCERTAINTY_SEED_STRIDE", 1000))
        seeds = member_seeds(int(getattr(self.config, "RANDOM_SEED", 42)), n_members, stride)

        run_name = f"{target}_{model_name}"
        with start_child_run(run_name):
            mlflow.log_params(
                {
                    "uncertainty_n_members": n_members,
                    "uncertainty_member_seeds": ",".join(str(seed) for seed in seeds),
                    "uncertainty_bootstrapped": False,
                    "uncertainty_calibration_source": getattr(self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val"),
                }
            )

            members = []
            for index, seed in enumerate(seeds):
                with start_child_run(
                    f"{run_name}_member{index}",
                    tags={
                        "run_kind": MEMBER_RUN_KIND,
                        "target": target,
                        "model_name": model_name,
                        "ensemble_member": str(index),
                        "ensemble_seed": str(seed),
                    },
                ):
                    members.append(
                        self._fit_member(
                            bundle=bundle_builder(seed)[model_name],
                            index=index,
                            seed=seed,
                        )
                    )

            return self._log_ensemble(target=target, model_name=model_name, members=members, seeds=seeds)

    def _fit_member(self, *, bundle: LightningModelBundle, index: int, seed: int) -> dict:
        """Train one ensemble member and collect everything the ensemble needs from it.

        It predicts on the test points and on the validation points, the latter being what the
        intervals are calibrated against. Those points cost nothing here - this family never trains
        on them - though early stopping did read them, so the calibration is slightly optimistic.

        Returns
        -------
        dict
            The member's model, scores, checkpoint and predictions.
        """
        trainer = self._build_trainer(bundle)
        bundle.datamodule.setup("fit")
        self._attach_preprocessing_state(bundle)
        trainer.fit(bundle.model, datamodule=bundle.datamodule)

        best_model_path = self._resolve_best_checkpoint(trainer)
        validation_metrics = self._normalize_metrics(
            self._call_trainer_method(trainer, "validate", bundle.model, bundle.datamodule, ckpt_path=best_model_path)
        )
        test_metrics = self._normalize_metrics(
            self._call_trainer_method(trainer, "test", bundle.model, bundle.datamodule, ckpt_path=best_model_path)
        )

        mlflow.log_params(
            {
                "ensemble_member": index,
                "ensemble_seed": seed,
                # Where this member's checkpoint was written.
                "best_model_path": best_model_path,
            }
        )
        # Every member keeps its own checkpoint, so the ensemble can be rebuilt later from the run
        # alone. That is one checkpoint per member rather than one per model.
        if best_model_path:
            self.mlflow_logger._log_checkpoint(best_model_path)

        test_predictions, test_sigmas = self._predict_split(bundle, trainer, best_model_path, "predict")
        calibration_predictions, calibration_sigmas = self._predict_split(bundle, trainer, best_model_path, "val")

        return {
            "full_predictions": self._predict_full_population(bundle),
            "bundle": bundle,
            "trainer": trainer,
            "best_model_path": best_model_path,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_predictions": test_predictions,
            "test_sigmas": test_sigmas,
            "calibration_predictions": calibration_predictions,
            "calibration_sigmas": calibration_sigmas,
        }

    def _predict_full_population(self, bundle) -> np.ndarray | None:
        """This member's prediction for every point, for the per-point export.

        Only an ensemble needs this per member: the exported number is the members' average, which
        no single model computes. Skipped when the export is switched off, since it is a full extra
        pass over the data per member.

        Returns
        -------
        numpy.ndarray or None
        """
        if not export_enabled_for(self.config, bundle.name):
            return None

        datamodule = getattr(bundle, "datamodule", None)
        sequence_bundle = getattr(datamodule, "sequence_bundle", None)
        if sequence_bundle is None:
            return None

        try:
            from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

            return SoilSequencePredictor(bundle.model).predict(sequence_bundle)
        except Exception as exc:  # pragma: no cover - the logger reports and keeps the run
            if self.logger is not None:
                self.logger.warning(
                    f"Full-population prediction failed for one member of {bundle.name}: "
                    f"{type(exc).__name__}: {exc}. The ensemble export will be skipped."
                )
            return None

    def _predict_split(self, bundle, trainer, ckpt_path, split: str):
        """Predict over one split, in the target's own units.

        Returns
        -------
        means : numpy.ndarray or None
        sigmas : numpy.ndarray or None
            Only from a model with a :term:`variance head`.
        """
        predict_method = getattr(trainer, "predict", None)
        if predict_method is None:
            return None, None

        try:
            if split == "val":
                dataloader = bundle.datamodule.val_dataloader()
                if dataloader is None:
                    return None, None
                predictions = predict_method(bundle.model, dataloaders=dataloader, ckpt_path=ckpt_path)
            else:
                predictions = predict_method(bundle.model, datamodule=bundle.datamodule, ckpt_path=ckpt_path)
        except (TypeError, RuntimeError, ValueError):
            return None, None

        return self._flatten_predictions_with_sigma(predictions)

    def _log_ensemble(self, *, target, model_name, members, seeds) -> LightningRunResult:
        """Average the members, calibrate the intervals, and record the ensemble as one model."""
        reference = members[0]
        bundle = reference["bundle"]
        datamodule = bundle.datamodule
        target_names = list(getattr(datamodule, "target_names", []) or [])

        test_stack, test_sigmas = self._stack_member_outputs(members, "test")
        prediction = aggregate(test_stack, test_sigmas) if test_stack else None

        calibrators = self._calibrate_ensemble(members, datamodule, target_names)

        evaluation_df = self._build_evaluation_frame(
            bundle, reference["trainer"], target, ckpt_path=reference["best_model_path"]
        )
        if evaluation_df is not None and prediction is not None and target_names:
            # What the run reports must be what the ensemble predicts, not one member's numbers.
            self._overwrite_predictions(evaluation_df, prediction, target_names)
            attach_uncertainty_columns(evaluation_df, prediction, target_names, calibrators)

        # Averaged across members, so the losses describe the ensemble rather than one member.
        validation_metrics = self._average_metrics([m["validation_metrics"] for m in members])
        test_metrics = self._average_metrics([m["test_metrics"] for m in members])

        self.mlflow_logger.log_lightning_child_run(
            config=self.config,
            target=target,
            model_name=model_name,
            evaluation_df=evaluation_df,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_model_path=reference["best_model_path"],
            extra_params=self._serialize_params(bundle),
            plot_functions={},
            bundle=bundle,
            model=bundle.model,
            calibrators=calibrators,
            full_population_predictions=self._ensemble_full_population(members),
        )

        return LightningRunResult(
            model_name=model_name,
            target=target,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_model_path=reference["best_model_path"],
        )

    def _calibrate_ensemble(self, members, datamodule, target_names) -> dict:
        """Build one :term:`prediction interval` estimator per target.

        ``sigma`` and ``gaussian`` need no data. ``conformal`` compares the ensemble's predictions
        for the validation points with their measurements; without usable validation predictions the
        run reports a spread but no interval, and says so.

        Returns
        -------
        dict of str to object
        """
        method = normalize_method(getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal"))
        if not needs_calibration_set(method):
            return build_interval_estimators(
                method,
                target_names,
                alpha=float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05)),
                k=float(getattr(self.config, "UNCERTAINTY_INTERVAL_K", 1.0)),
            )

        y_val = getattr(datamodule, "y_val_frame_", None)
        stack, sigmas = self._stack_member_outputs(members, "calibration")
        if y_val is None or not len(y_val) or not stack or not target_names:
            if self.logger is not None:
                self.logger.warning(
                    "No usable validation split for conformal calibration; the ensemble reports a "
                    "standard deviation but no calibrated interval."
                )
            return {}

        calibration = aggregate(stack, sigmas)
        if calibration.mean.shape[0] != len(y_val):
            if self.logger is not None:
                self.logger.warning(
                    f"Validation predictions ({calibration.mean.shape[0]} rows) do not line up with "
                    f"y_val_frame_ ({len(y_val)} rows); skipping calibration rather than pairing "
                    "residuals with the wrong observations."
                )
            return {}

        return fit_calibrators(
            calibration,
            y_val,
            target_names,
            alpha=float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05)),
            logger=self.logger,
        )

    @staticmethod
    def _ensemble_full_population(members) -> np.ndarray | None:
        """The ensemble's average prediction for every point, or None if any member is missing one.

        All or nothing: averaging over the members that happened to work would export a number that
        is neither one member's prediction nor the ensemble's.
        """
        full = [member["full_predictions"] for member in members if member.get("full_predictions") is not None]
        if not full or len(full) != len(members):
            return None
        return aggregate(full).mean

    @staticmethod
    def _stack_member_outputs(members, split: str):
        """Collect every member's predictions for one split.

        The spreads are returned only if every member reported one: averaging over some of them
        would treat the rest as certain and understate the uncertainty.
        """
        means = [member[f"{split}_predictions"] for member in members if member.get(f"{split}_predictions") is not None]
        sigmas = [member[f"{split}_sigmas"] for member in members if member.get(f"{split}_sigmas") is not None]
        return means, (sigmas if means and len(sigmas) == len(means) else None)

    @staticmethod
    def _overwrite_predictions(evaluation_df, prediction, target_names) -> None:
        """Put the ensemble's average into the results table, in place of one member's."""
        multi_target = len(target_names) > 1
        for index, target_name in enumerate(target_names):
            column = f"prediction_{target_name}" if multi_target else "prediction"
            if column in evaluation_df.columns:
                evaluation_df[column] = prediction.mean[:, index]

    @staticmethod
    def _average_metrics(metric_dicts: list[dict]) -> dict[str, float]:
        """Average each score across the members, over the scores they all report."""
        if not metric_dicts:
            return {}
        shared = set(metric_dicts[0])
        for metrics in metric_dicts[1:]:
            shared &= set(metrics)
        return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in sorted(shared)}

    @staticmethod
    def _attach_preprocessing_state(bundle: LightningModelBundle) -> None:
        """Copy the fitted input statistics onto the model, so its checkpoint carries them."""
        state_source = getattr(bundle.datamodule, "preprocessing_state", None)
        attach = getattr(bundle.model, "attach_preprocessing_state", None)
        if not callable(state_source) or not callable(attach):
            return
        attach(state_source())

    def _build_trainer(self, bundle: LightningModelBundle):
        """Build Lightning's trainer from this bundle's training settings."""
        lightning = self._get_lightning_module()
        callbacks = self._build_callbacks(bundle.callback_specs)

        trainer_kwargs = dict(bundle.trainer_kwargs)
        trainer_kwargs["callbacks"] = callbacks
        if not getattr(self.config, "LIGHTNING_ENABLE_DEFAULT_LOGGER", True):
            trainer_kwargs.setdefault("logger", False)
        return lightning.Trainer(**trainer_kwargs)

    def _build_callbacks(self, callback_specs: Mapping[str, Any]):
        """Build the per-epoch recording, the early stopping and the checkpoint saving."""
        lightning = self._get_lightning_module()
        callbacks = [_LightningMlflowEpochMetricCallback()]

        early_stopping = callback_specs.get("early_stopping")
        if early_stopping:
            callbacks.append(lightning.callbacks.EarlyStopping(**early_stopping))

        checkpoint = callback_specs.get("checkpoint")
        if checkpoint:
            # Where checkpoints are written comes from the model list if it says; otherwise
            # Lightning chooses.
            callbacks.append(lightning.callbacks.ModelCheckpoint(**dict(checkpoint)))

        return callbacks

    def _get_lightning_module(self):
        """Import PyTorch Lightning, with a clear message when it is not installed."""
        try:
            return importlib.import_module("lightning.pytorch")
        except ImportError as exc:  # pragma: no cover - exercised only when lightning is absent
            raise ImportError("lightning.pytorch is required to execute LightningTrainer.train().") from exc

    def _call_trainer_method(self, trainer, method_name: str, model, datamodule, ckpt_path: str | None = None):
        """Call ``validate`` or ``test`` on the trainer, whichever arguments its version takes."""
        method = getattr(trainer, method_name, None)
        if method is None:
            return []
        try:
            if ckpt_path is not None:
                return method(model, datamodule=datamodule, verbose=False, ckpt_path=ckpt_path)
            return method(model, datamodule=datamodule, verbose=False)
        except TypeError:
            if ckpt_path is not None:
                return method(model, datamodule=datamodule, ckpt_path=ckpt_path)
            return method(model, datamodule=datamodule)

    def _resolve_best_checkpoint(self, trainer) -> str | None:
        """Where the best epoch's :term:`checkpoint` was written, or None if none was kept."""
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", None)
        if best_model_path:
            return best_model_path

        for callback in getattr(trainer, "callbacks", []):
            best_model_path = getattr(callback, "best_model_path", None)
            if best_model_path:
                return best_model_path

        return None

    def _build_evaluation_frame(
        self,
        bundle: LightningModelBundle,
        trainer,
        target: str,
        ckpt_path: str | None = None,
    ) -> pd.DataFrame | None:
        """Build the table of test-point results: covariates, measurements and predictions.

        Returns
        -------
        pandas.DataFrame or None
            One row per test point. Saved with the run as ``eval_results/eval_results.csv``.
        """
        datamodule = bundle.datamodule
        if getattr(datamodule, "X_test_frame_", None) is None or getattr(datamodule, "y_test_frame_", None) is None:
            return None

        predict_method = getattr(trainer, "predict", None)
        if predict_method is None:
            return None

        try:
            if ckpt_path is not None:
                predictions = predict_method(bundle.model, datamodule=datamodule, ckpt_path=ckpt_path)
            else:
                predictions = predict_method(bundle.model, datamodule=datamodule)
        except TypeError:
            if ckpt_path is not None:
                predictions = predict_method(bundle.model, ckpt_path=ckpt_path)
            else:
                predictions = predict_method(bundle.model)

        if predictions is None:
            return None

        prediction_values = self._flatten_predictions(predictions)
        if prediction_values is None:
            return None

        eval_df = datamodule.X_test_frame_.copy()
        target_frame = datamodule.y_test_frame_.copy()
        target_names = list(getattr(datamodule, "target_names", []) or target_frame.columns.tolist())

        for column in target_frame.columns:
            eval_df[column] = target_frame[column].to_numpy()

        is_single_output = prediction_values.ndim == 1 or (
            prediction_values.ndim == 2 and prediction_values.shape[1] == 1
        )

        if is_single_output:
            eval_df["prediction"] = prediction_values.reshape(-1)
        else:
            target_columns = list(target_names)
            for column_index in range(prediction_values.shape[1]):
                column_name = target_columns[column_index] if column_index < len(target_columns) else str(column_index)
                eval_df[f"prediction_{column_name}"] = prediction_values[:, column_index]
            # No plain `prediction` column with several targets: it would hold the first target's
            # predictions under a name that claims to be the run's. Readers use prediction_<target>.

        if len(target_names) > 1:
            encoded = join_target_names(target_names)
            eval_df["target_name"] = encoded
            eval_df["target_names"] = encoded
        else:
            eval_df["target_name"] = target
        return eval_df

    def _flatten_predictions(self, predictions) -> np.ndarray | None:
        """The predictions alone, for callers that do not need the spread."""
        return self._flatten_predictions_with_sigma(predictions)[0]

    def _flatten_predictions_with_sigma(self, predictions) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Join a list of per-batch predictions into one array, with the spreads when there are any.

        Returns
        -------
        means : numpy.ndarray or None
        sigmas : numpy.ndarray or None
            None unless every batch reported one.
        """
        means, sigmas = [], []
        for batch in predictions:
            if batch is None:
                continue
            if isinstance(batch, tuple):
                mean_batch, sigma_batch = batch[0], batch[1] if len(batch) > 1 else None
            else:
                mean_batch, sigma_batch = batch, None

            means.append(self._as_2d_array(mean_batch))
            if sigma_batch is not None:
                sigmas.append(self._as_2d_array(sigma_batch))

        if not means:
            return None, None

        stacked_means = np.concatenate(means, axis=0)
        # All or nothing: a partial set would pair some points' uncertainty with other points'
        # predictions.
        stacked_sigmas = np.concatenate(sigmas, axis=0) if len(sigmas) == len(means) else None
        return stacked_means, stacked_sigmas

    @staticmethod
    def _as_2d_array(batch) -> np.ndarray:
        """Return one batch of predictions as a rows-by-targets array."""
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        else:
            batch = np.asarray(batch)

        if batch.ndim == 0:
            return batch.reshape(1, 1)
        if batch.ndim == 1:
            return batch.reshape(-1, 1)
        return batch

    def _normalize_metrics(self, metrics) -> dict[str, float]:
        """Return what Lightning reported as a plain dict of numbers."""
        if not metrics:
            return {}

        if isinstance(metrics, list):
            metrics = metrics[0] if metrics else {}

        normalized = {}
        for key, value in dict(metrics).items():
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                continue

        return normalized

    def _serialize_params(self, bundle: LightningModelBundle) -> dict[str, Any]:
        """The settings to record with the run: the model's, its data's and its training's."""
        params = {
            "model_name": bundle.name,
            "target": bundle.target,
            "modeltype": bundle.registry_entry.get("modeltype"),
        }
        params.update(bundle.trainer_kwargs)

        init_args = bundle.registry_entry.get("init_args", {})
        datamodule_args = bundle.registry_entry.get("datamodule_init_args", {})
        for prefix, values in (("model", init_args), ("datamodule", datamodule_args)):
            for key, value in values.items():
                params[f"{prefix}.{key}"] = value

        return params
