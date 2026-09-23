"""Take the scikit-learn family's rows out of the run's shared split."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Dict, Optional

import mlflow
import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.splitting import SplitPlan, TEST, TRAIN, VAL


class SklearnDataSplitter:
    """Take this family's rows out of the run's shared split.

    It does not decide the split - :mod:`yg_eo_soilnet.datamodules.splitting` does that, before
    either family sees the data, so both hold out the same points.

    One thing to know when reading what :meth:`split_data` returns: **``X_train`` is the**
    :term:`fit pool` - the training *and* validation points together. The scikit-learn models choose
    their hyperparameters by cross-validation inside that pool, so they have no use for a separate
    validation set, while the deep-learning model early-stops on one. Both are then scored on the
    same ``X_test``. ``X_train_only`` and ``X_val`` are returned as well, for the saved split files
    only.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where the split sizes go.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def split_data(
        self,
        processed_data: Dict[str, Any],
        *,
        sanitize_features: Callable[[pd.DataFrame, list[str]], pd.DataFrame] | None = None,
        model_config_factory: Any = None,
        split_plan: Optional[SplitPlan] = None,
    ) -> Dict[str, Any]:
        """Return the covariates and targets of each split, and save the split with the run.

        Parameters
        ----------
        processed_data : dict
            What
            :meth:`TabularPreprocessor.preprocess_data
            <yg_eo_soilnet.datamodules.scikit.tabular_preprocessor.TabularPreprocessor.preprocess_data>`
            returned: ``X``, ``y``, ``lat``, ``lon`` and ``point_ids``.
        sanitize_features : callable, optional
            Applied to ``X`` first, to drop anything that is not a model input.
        model_config_factory : object, optional
            Unused; kept so older callers still work.
        split_plan : SplitPlan
            The run's shared split. Required.

        Returns
        -------
        dict
            ``X_train``/``y_train`` (the :term:`fit pool`), ``X_test``/``y_test``, the coordinates of
            each, ``X_train_only``/``X_val`` and their targets, ``point_ids``, ``split_labels``,
            ``split_plan``, ``X_all`` (every point, for the per-point export) and, for a spatial
            split, ``groups_train``/``groups_test``.

        Raises
        ------
        ValueError
            If no split plan is given, or the plan leaves this family no test rows.
        """
        X: pd.DataFrame = processed_data["X"]
        y: pd.DataFrame = processed_data["y"]
        lat = processed_data["lat"]
        lon = processed_data["lon"]
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))

        if sanitize_features is not None:
            X = sanitize_features(X, target_columns)

        if split_plan is None:
            raise ValueError(
                "SklearnDataSplitter needs the run's split_plan. Build it with "
                "SplitPlanProvider(config, logger, data_manager).plan() before splitting, so this "
                "family and the Lightning families hold out the same points."
            )

        point_ids = self._point_ids(processed_data, X)
        labels = split_plan.labels_for(point_ids).to_numpy()

        train_pos = np.flatnonzero(labels == TRAIN)
        val_pos = np.flatnonzero(labels == VAL)
        test_pos = np.flatnonzero(labels == TEST)
        # Training and validation points together, kept in the table's own order.
        fit_pos = np.sort(np.concatenate([train_pos, val_pos])) if len(val_pos) else train_pos
        unassigned = int(len(labels) - (len(train_pos) + len(val_pos) + len(test_pos)))

        self.logger.info(
            f"Applying the shared split plan ({split_plan.strategy}, "
            f"policy={split_plan.population_policy}): fit pool={len(fit_pos)} "
            f"(train={len(train_pos)} + val={len(val_pos)}) | test={len(test_pos)} rows"
        )
        if unassigned:
            self.logger.info(
                f"{unassigned} row(s) are outside the shared split population and are used by no "
                f"split. That is population_policy={split_plan.population_policy} doing its job."
            )
        if len(test_pos) == 0:
            raise ValueError(
                "The shared split plan assigned no test rows to the sklearn family. Check "
                "split.test_size and split.population_policy."
            )

        split_data: Dict[str, Any] = {}

        X_train, y_train = X.iloc[fit_pos], y.iloc[fit_pos]
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]
        lat_train, lat_test = lat.iloc[fit_pos], lat.iloc[test_pos]
        lon_train, lon_test = lon.iloc[fit_pos], lon.iloc[test_pos]

        if split_plan.clusters is not None:
            # Cross-validation inside the fit pool keeps these groups whole too, so its folds
            # respect the same spatial grouping the test points were held out by.
            groups = split_plan.clusters.reindex(pd.Index(point_ids))
            groups.index = X.index
            split_data["groups_train"] = groups.iloc[fit_pos]
            split_data["groups_test"] = groups.iloc[test_pos]
            self.logger.info(
                f"Fit-pool group distribution:\n{split_data['groups_train'].value_counts().sort_index().to_string()}"
            )
            self.logger.info(
                f"Test group distribution:\n{split_data['groups_test'].value_counts().sort_index().to_string()}"
            )

        point_id_series = pd.Series(np.asarray(point_ids), index=X.index, name="point_id")
        self._log_split_artifacts(
            split_plan,
            point_ids=point_id_series,
            frames={
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            },
        )

        split_data["X_train"] = X_train
        split_data["X_test"] = X_test
        split_data["y_train"] = y_train
        split_data["y_test"] = y_test
        split_data["lat_train"] = lat_train
        split_data["lat_test"] = lat_test
        split_data["lon_train"] = lon_train
        split_data["lon_test"] = lon_test
        # For the saved files only, so the fit pool can be taken apart afterwards. Training reads
        # neither: cross-validation inside the pool is what chooses the hyperparameters.
        split_data["X_train_only"] = X.iloc[train_pos]
        split_data["y_train_only"] = y.iloc[train_pos]
        split_data["X_val"] = X.iloc[val_pos]
        split_data["y_val"] = y.iloc[val_pos]
        split_data["point_ids"] = point_id_series
        split_data["split_labels"] = pd.Series(labels, index=X.index, name="split")
        split_data["split_plan"] = split_plan
        # Every point, not just the holdout: read by the per-point prediction export. The same
        # object the splits were taken from, so its row numbers still match `point_ids`.
        split_data["X_all"] = X

        return split_data

    def _point_ids(self, processed_data: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """The point id of every covariate row, in order; the split is keyed on it."""
        point_ids = processed_data.get("point_ids")
        if point_ids is None:
            raise KeyError(
                "processed_data carries no 'point_ids'. TabularPreprocessor.preprocess_data adds "
                "it; a hand-built dict must too, because the shared split is keyed on point id."
            )
        if isinstance(point_ids, pd.Series):
            # Matched on row labels rather than assumed to line up: a mismatch here would key the
            # whole split to the wrong points.
            return point_ids.reindex(X.index).to_numpy()
        point_ids = np.asarray(point_ids)
        if len(point_ids) != len(X):
            raise ValueError(
                f"processed_data['point_ids'] has {len(point_ids)} entries for {len(X)} feature "
                f"rows; the shared split cannot be keyed."
            )
        return point_ids

    def _log_split_artifacts(
        self, split_plan: SplitPlan, *, point_ids: pd.Series, frames: Dict[str, pd.DataFrame]
    ) -> None:
        """Save the split with the run, under ``data_splits/``.

        One file per split, each with a ``point_id`` column, plus ``split_assignments.parquet``:
        the whole plan, which is what lets a finished run be matched back to its source data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            split_start = time.perf_counter()
            for name, frame in frames.items():
                keyed = frame.copy()
                keyed.insert(0, "point_id", point_ids.reindex(frame.index).to_numpy())
                keyed.to_parquet(os.path.join(tmpdir, f"{name}.parquet"), index=False)
            split_plan.to_frame().to_parquet(os.path.join(tmpdir, "split_assignments.parquet"), index=False)
            self.logger.info(f"split_data parquet writes completed in {time.perf_counter() - split_start:.2f}s")

            artifact_start = time.perf_counter()
            mlflow.log_artifacts(tmpdir, artifact_path="data_splits")
            self.logger.info(
                f"split_data mlflow.log_artifacts completed in {time.perf_counter() - artifact_start:.2f}s"
            )
