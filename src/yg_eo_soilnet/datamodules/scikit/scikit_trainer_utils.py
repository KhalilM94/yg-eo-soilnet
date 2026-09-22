"""The parts a scikit-learn model is assembled from: its pipeline, its folds, its row filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.model_selection import GroupKFold, KFold

from yg_eo_soilnet.utils import LogTransformer


@dataclass
class CVSplitter:
    """Make the cross-validation folds the hyperparameter search uses.

    Attributes
    ----------
    cv_strategy : {"kfold", "groupkfold"}, default "kfold"
        ``groupkfold`` keeps each spatial group inside one fold, matching a spatial split.
    n_splits : int, default 5
        How many folds.
    random_state : int, default 42
        The random seed.
    shuffle : bool, default True
        Shuffle the rows before making the folds.
    """

    cv_strategy: str = 'kfold'
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True

    def create_splits(self, X, y=None, groups=None):
        """Return the folds as a list of ``(train positions, validation positions)`` pairs.

        Parameters
        ----------
        X : array-like
            The covariates of the :term:`fit pool`.
        y : array-like, optional
            The targets.
        groups : array-like, optional
            One group per row; required for ``groupkfold``.

        Returns
        -------
        list of tuple

        Raises
        ------
        ValueError
            If ``groupkfold`` is asked for without groups, or the strategy is unknown.
        """
        strategy = self.cv_strategy.lower()
        if strategy == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            return list(cv.split(X))
        if strategy == 'groupkfold':
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            return list(cv.split(X, y, groups))
        raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")


class TargetNanFilter(BaseEstimator, TransformerMixin):
    """Drop the rows whose target was not measured, so a model only sees usable rows."""

    def fit(self, X, y=None):
        """Nothing to learn; returns itself."""
        return self

    def transform(self, X, y=None, groups=None):
        """Return ``(X, y, groups)`` with the unmeasured rows removed.

        When ``y`` holds several targets - one model predicting them all - a row is kept only where
        every one of them was measured. That is the cost of fitting one model to several targets;
        one model per target keeps every row it has a measurement for.
        """
        if y is None:
            return X
        # Several targets: one row of covariates serves them all, so a row is usable only where
        # every target was measured.
        if isinstance(y, pd.DataFrame):
            mask = y.notna().all(axis=1)
        else:
            mask = pd.notna(y)
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        groups_clean = groups.loc[mask] if groups is not None else None
        return X_clean, y_clean, groups_clean


@dataclass
class PipelineBuilder:
    """Wrap an estimator in the steps that prepare its inputs.

    The result is one scikit-learn :class:`~sklearn.pipeline.Pipeline`: fill gaps, scale, encode
    categories, then the model. Saved as one object, so new points are prepared exactly like the
    training points were. Every step is fitted inside each cross-validation fold, so nothing learned
    from held-out rows reaches the model.

    Attributes
    ----------
    seed : int, default 42
        The random seed.
    tree_categorical_encoding : {"ordinal", "onehot"}, default "ordinal"
        How categories are encoded for tree-based models. Linear models always get one-hot columns.
    tree_onehot_max_categories : int, optional
        With ``onehot``, keep this many categories and group the rest together.
    """

    seed: int = 42
    tree_categorical_encoding: str = "ordinal"
    tree_onehot_max_categories: Optional[int] = None

    def _is_tree_based_model(self, model: BaseEstimator) -> bool:
        """Whether this estimator is a tree or an ensemble of trees, judged by its class name."""
        model_name = model.__class__.__name__.lower()
        model_module = model.__class__.__module__.lower()
        tree_markers = ("tree", "forest", "boost", "xgb", "lightgbm", "catboost")
        return any(marker in model_name or marker in model_module for marker in tree_markers)

    def _build_preprocessor(self, model: BaseEstimator, categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
        """Build the input-preparation step: fill gaps, scale numbers, encode categories."""
        is_tree_model = self._is_tree_based_model(model)

        # add_indicator adds a measured-or-filled flag next to each column that had gaps, so the
        # model can tell a filled-in value from a measured one - as the deep-learning side does.
        # Trees are not scaled; they split on values and do not care about their range.
        numeric_steps: List[Tuple[str, BaseEstimator]] = [
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ]
        if not is_tree_model:
            numeric_steps.append(('scaler', RobustScaler()))

        use_ordinal = is_tree_model and str(self.tree_categorical_encoding).lower() == "ordinal"
        if use_ordinal:
            categorical_encoder: BaseEstimator = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        elif is_tree_model and self.tree_onehot_max_categories:
            categorical_encoder = OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                max_categories=int(self.tree_onehot_max_categories),
            )
        else:
            categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        transformers: List[Tuple[str, BaseEstimator, List[str]]] = []
        if numeric_cols:
            transformers.append(('num', Pipeline(numeric_steps), numeric_cols))
        if categorical_cols:
            transformers.append((
                'cat',
                Pipeline([
                    # No flag here: a missing category already gets a code, or a column, of its
                    # own, so the flag would say the same thing twice.
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', categorical_encoder),
                ]),
                categorical_cols,
            ))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def build(
        self,
        model,
        is_log_target: bool = False,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> Pipeline:
        """Return the estimator wrapped in its input-preparation steps.

        Parameters
        ----------
        model : estimator
            The scikit-learn estimator to wrap.
        is_log_target : bool, default False
            Train on 10·ln(1 + *y*) and convert the predictions back, for the targets listed in
            ``COLUMNS_TO_TRANSFORM``.
        categorical_cols : list of str, optional
            The category columns.
        numeric_cols : list of str, optional
            The numeric columns. Anything in neither list is dropped.

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        categorical_cols = categorical_cols or []
        numeric_cols = numeric_cols or []
        preprocessor = self._build_preprocessor(model, categorical_cols, numeric_cols)

        from sklearn.compose import TransformedTargetRegressor

        steps: List[Tuple[str, BaseEstimator]] = [('preprocessor', preprocessor)]
        if is_log_target:
            steps.append((
                'model',
                TransformedTargetRegressor(
                    regressor=model,
                    func=LogTransformer().transform,
                    inverse_func=LogTransformer().inverse_transform,
                    check_inverse=False,
                ),
            ))
        else:
            steps.append(('model', model))
        return Pipeline(steps)
