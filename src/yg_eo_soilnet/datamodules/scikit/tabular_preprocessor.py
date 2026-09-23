"""Turn the loaded table into the covariates and targets the scikit-learn models read."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.frame_cleaning import assert_columns_are_dense_enough


class TabularPreprocessor:
    """Turn the loaded table into the covariates and targets the scikit-learn models read.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where the column counts and warnings go.
    data_manager : DataManager
        Used to pick out the columns that may be model inputs.
    """

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager

    def preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Split the table into covariates, targets, coordinates and point ids.

        Columns that are blank everywhere are dropped; no row is dropped. A covariate missing on too
        many rows stops the run (see :meth:`assert_covariates_are_dense_enough`).

        Parameters
        ----------
        data : pandas.DataFrame
            One row per point, covariates and targets together.

        Returns
        -------
        dict
            ``X`` the covariates, ``y`` the targets, ``lat`` and ``lon`` the coordinates, and
            ``point_ids``.

        Raises
        ------
        KeyError
            If the coordinate columns are missing.
        """
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))
        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")

        if lat_col not in data.columns or lon_col not in data.columns:
            raise KeyError(f"Tabular data must contain coordinate columns '{lat_col}' and '{lon_col}'")

        candidate_feature_columns = self.data_manager.filter_schema(data, target_columns).columns.tolist()

        valid_feature_columns = data[candidate_feature_columns].dropna(axis=1, how="all").columns.tolist()
        original_feature_count = len(candidate_feature_columns)
        retained_feature_count = len(valid_feature_columns)
        if original_feature_count > 0 and retained_feature_count < original_feature_count:
            dropped_ratio = (original_feature_count - retained_feature_count) / float(original_feature_count)
            max_drop_ratio = float(getattr(self.config, "MAX_FEATURE_DROP_RATIO_WARNING", 0.9))
            if dropped_ratio > max_drop_ratio:
                self.logger.warning(
                    f"Dropped feature ratio for preprocessing is {dropped_ratio:.2%} "
                    f"(threshold {max_drop_ratio:.2%}); {original_feature_count - retained_feature_count} of {original_feature_count} features were removed."
                )

        min_feature_count = int(getattr(self.config, "MIN_FEATURE_COUNT", 10))
        if retained_feature_count < min_feature_count:
            self.logger.warning(
                f"Only {retained_feature_count} usable features remain after preprocessing "
                f"(minimum recommended: {min_feature_count})."
            )

        X = data[valid_feature_columns]
        categorical_cols = [
            col for col in self.config.CATEGORICAL_FEATURES if col in X.columns and col not in self.config.EXCLUDE_CATEGORICAL
        ]
        # The same check the deep-learning side makes, so a column too empty to fill in stops both
        # families. Numeric columns only: a missing category is a code of its own, not a filled gap.
        self.assert_covariates_are_dense_enough(
            data, [column for column in valid_feature_columns if column not in categorical_cols]
        )
        if categorical_cols:
            self.logger.info(
                "Categorical encoding will be applied per model in the training pipeline "
                "(one-hot for linear models, ordinal encoding for tree-based models)."
            )

        data_cleaned = data.loc[X.index]
        return {
            "X": X,
            "y": data_cleaned[target_columns],
            "lat": data_cleaned[lat_col],
            "lon": data_cleaned[lon_col],
            # The id is not a covariate, so it is carried separately: it is what the shared split
            # is keyed on.
            "point_ids": self._point_ids(data_cleaned),
        }

    def assert_covariates_are_dense_enough(self, data: pd.DataFrame, columns) -> None:
        """Stop the run if one of these covariates is missing on too many rows.

        Uses ``common.data_quality`` - ``max_missing_column_ratio``, ``allow_sparse_columns`` and
        ``fail_on_sparse_columns``.

        Parameters
        ----------
        data : pandas.DataFrame
            The table to check.
        columns : iterable of str
            The numeric covariates.
        """
        assert_columns_are_dense_enough(
            data,
            columns,
            max_missing_ratio=float(getattr(self.config, "MAX_MISSING_COLUMN_RATIO", 0.2)),
            label="the sklearn feature matrix",
            logger=self.logger,
            allow=getattr(self.config, "ALLOW_SPARSE_COLUMNS", ()) or (),
            fail=bool(getattr(self.config, "FAIL_ON_SPARSE_COLUMNS", True)),
        )

    def usable_point_ids(self, data: pd.DataFrame) -> pd.Index:
        """The points this family can use - all of them, since no row is ever dropped here.

        The split asks every family the same question; see
        :class:`~yg_eo_soilnet.datamodules.split_plan_provider.SplitPlanProvider`.

        Parameters
        ----------
        data : pandas.DataFrame
            One row per point.

        Returns
        -------
        pandas.Index
        """
        return pd.Index(self._point_ids(data).to_numpy())

    def _point_ids(self, data: pd.DataFrame) -> pd.Series:
        """The point id of every row, or its row number when the data has no id column."""
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        if point_col in data.columns:
            return data[point_col]
        # With no id column, row numbers still key the split: both families read the same file in
        # the same order, so they agree on what each number means.
        return pd.Series(np.arange(len(data)), index=data.index, name=point_col)
