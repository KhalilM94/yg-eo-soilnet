"""Stage 1 of A1: predict CaCO3 and CEC for every point of one split, without leaking a lab value.

Stage 2 reads these predictions as inputs for SOM, so a prediction made by a model that had seen
that point's own CaCO3 would hand stage 2 information a new point never has, and flatter every
score built on it. The rule is that no point is ever predicted by a model fitted on its lab value:

* **test points** (and validation points, when the plan has any) are predicted by one model fitted
  on the training points;
* **training points** are predicted *out of fold*: the training points are cut into folds, and each
  fold is predicted by a model fitted on the other folds (:func:`sklearn.model_selection.cross_val_predict`);
* a training point whose own value was never measured contributed no label to any fit, so it is
  predicted by the model fitted on all training points, like a test point.

The folds follow the split. On a random split they are random (``KFold``). On a spatial or regional
split they keep each block or region whole (``GroupKFold`` on the plan's clusters), so stage 1 is
no more optimistic about the training points than the split itself is about the test points. The
fold of every training point is returned with the predictions, so a run can be audited afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from yg_eo_soilnet.a1.config import stage1_prediction_column
from yg_eo_soilnet.a1.features import A1Table
from yg_eo_soilnet.a1.pipelines import build_stage_pipeline
from yg_eo_soilnet.a1.settings import A1Settings
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL, SplitPlan

#: Plan kinds whose folds keep the plan's clusters (blocks, regions) whole.
GROUPED_KINDS = ("spatial_kfold", "logo", "mixed")
#: Plan kinds whose folds are random: the baseline, and a control, whose training points all sit
#: in one region and so cannot be grouped by region.
UNGROUPED_KINDS = ("random", "control")

KFOLD = "kfold"
GROUP_KFOLD = "groupkfold"
#: ``predicted_by`` for a point predicted by the model fitted on every training point.
FULL_TRAIN = "full_train"
#: ``predicted_by`` for a point that could not be predicted without leaking: too few training points.
NOT_PREDICTED = "none"
#: ``folds`` for a point outside the training set.
NOT_IN_A_FOLD = -1


@dataclass(frozen=True)
class AuxiliaryPredictions:
    """Stage 1's predictions for one split, and how each was made.

    Attributes
    ----------
    predictions : pandas.DataFrame
        Indexed by point id, one ``a1_pred_<property>`` column per auxiliary property, a row for
        every point of the plan.
    folds : pandas.Series
        Each point's inner fold: 0, 1, ... for a training point, -1 for validation and test.
    predicted_by : pandas.DataFrame
        Per property, which fit predicted each point: ``oof_fold_<k>`` (the model fitted without
        fold *k*), ``full_train`` or ``none``.
    split : pandas.Series
        Each point's split.
    cv_kind : str
        ``kfold`` or ``groupkfold``.
    n_folds : int
        How many folds were used; 0 when the training set was too small to cut.
    """

    predictions: pd.DataFrame
    folds: pd.Series
    predicted_by: pd.DataFrame
    split: pd.Series
    cv_kind: str
    n_folds: int

    def to_frame(self) -> pd.DataFrame:
        """Everything in one table, one row per point, ready to save beside a run."""
        audit = self.predicted_by.add_prefix("predicted_by_")
        return pd.concat(
            [self.split.rename("split"), self.folds.rename("inner_fold"), self.predictions, audit], axis=1
        ).rename_axis("point_id")


def build_stage1_pipeline(
    settings: A1Settings,
    config,
    feature_columns,
    categorical_columns=(),
    *,
    target: Optional[str] = None,
):
    """The stage-1 estimator, wrapped in the repository's input preparation.

    Built by :func:`~yg_eo_soilnet.a1.pipelines.build_stage_pipeline` from ``stage1``: the
    model-list entry it names, with ``import_path``, ``init_args`` and ``params`` laid over it.
    Stage 1 is not tuned, so every setting must have a single value, which is applied.

    Parameters
    ----------
    settings : A1Settings
    config : Config
        The main configuration: the model list and the scikit-learn settings.
    feature_columns : sequence of str
        The inputs.
    categorical_columns : sequence of str
        The inputs that are classes.
    target : str, optional
        The property predicted, for the log transform.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        If the entry is not in the model list, or its grid holds more than one value for a setting.
    """
    pipeline, grid = build_stage_pipeline(
        settings, config, feature_columns, categorical_columns, stage=settings.stage1, target=target
    )
    several = sorted(name for name, values in grid.items() if len(values) > 1)
    if several:
        raise ValueError(
            f"The {settings.stage1.estimator} entry lists several values for {several}. Stage 1 fits one model "
            "per fold and does not search among settings; keep one value for each."
        )
    return pipeline


def inner_folds(
    plan: SplitPlan,
    settings: A1Settings,
    *,
    kind: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.Series, str]:
    """Cut a plan's training points into the folds stage 1 predicts them through.

    Parameters
    ----------
    plan : SplitPlan
    settings : A1Settings
        ``inner_cv_folds`` and the seed.
    kind : str, optional
        The plan's kind (``index.csv``); read from ``plan.strategy`` unless given. A plan of no
        known kind is grouped when it has clusters.
    logger : logging.Logger, optional

    Returns
    -------
    folds : pandas.Series
        Each training point's fold, 0, 1, ..., in the plan's order; empty when the training set is
        too small to cut in two.
    cv_kind : str
        ``kfold`` or ``groupkfold``.
    """
    logger = logger or logging.getLogger(__name__)
    kind = kind or plan.strategy
    train_ids = plan.point_ids_for(TRAIN)
    grouped = kind in GROUPED_KINDS or (kind not in UNGROUPED_KINDS and plan.clusters is not None)

    if grouped:
        groups = plan.clusters.reindex(train_ids) if plan.clusters is not None else pd.Series(np.nan, index=train_ids)
        # A training point in no block or region is a group of its own: it can go in any fold.
        missing = groups.isna().to_numpy()
        groups = groups.astype(float).to_numpy()
        groups[missing] = -(np.arange(missing.sum()) + 1)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            logger.warning(
                f"The training points of this {kind} plan form {n_groups} group(s); stage-1 folds cannot keep "
                "groups whole, so they are random instead."
            )
            grouped = False
    cv_kind = GROUP_KFOLD if grouped else KFOLD

    available = n_groups if grouped else len(train_ids)
    n_folds = min(settings.inner_cv_folds, available)
    if n_folds < settings.inner_cv_folds:
        logger.warning(
            f"Only {available} {'group(s)' if grouped else 'training point(s)'} for inner_cv_folds="
            f"{settings.inner_cv_folds}; using {max(n_folds, 0)} fold(s)."
        )
    if n_folds < 2:
        return pd.Series(dtype=int, index=pd.Index([], name="point_id")), cv_kind

    placeholder = np.zeros((len(train_ids), 1))
    if grouped:
        splits = GroupKFold(n_splits=n_folds).split(placeholder, groups=groups)
    else:
        splits = KFold(n_splits=n_folds, shuffle=True, random_state=settings.seed).split(placeholder)
    folds = np.full(len(train_ids), NOT_IN_A_FOLD, dtype=int)
    for fold, (_, held) in enumerate(splits):
        folds[held] = fold
    return pd.Series(folds, index=pd.Index(train_ids, name="point_id")), cv_kind


def predict_auxiliary(
    table: A1Table,
    plan: SplitPlan,
    settings: A1Settings,
    config,
    *,
    kind: Optional[str] = None,
    estimator_factory: Optional[Callable[[str], object]] = None,
    logger: Optional[logging.Logger] = None,
) -> AuxiliaryPredictions:
    """Predict every auxiliary property for every point of one split, each from a fit that never saw it.

    Parameters
    ----------
    table : A1Table
        The inputs and lab values (:func:`~yg_eo_soilnet.a1.features.build_table`).
    plan : SplitPlan
        The split, for example one written by ``experiments/a1/make_plans.py``.
    settings : A1Settings
    config : Config
        The main configuration.
    kind : str, optional
        The plan's kind; read from the plan unless given (a plan read from a file does not carry
        it, so pass the ``kind`` of ``index.csv``).
    estimator_factory : callable, optional
        ``factory(property) -> estimator``; :func:`build_stage1_pipeline` unless given. A test passes
        an estimator that records what it was fitted on.
    logger : logging.Logger, optional

    Returns
    -------
    AuxiliaryPredictions

    Raises
    ------
    KeyError
        If the plan names points the table does not have.
    """
    logger = logger or logging.getLogger(__name__)
    ids = plan.assignments.index
    unknown = ids.difference(table.frame.index)
    if len(unknown):
        raise KeyError(f"{len(unknown)} point(s) of the plan are not in the A1 table, e.g. {list(unknown[:3])}")

    features = list(table.feature_columns)
    X = table.frame.loc[ids, features]
    X = X.astype({column: "float64" for column in X.select_dtypes(include=["int64", "int32"]).columns})
    split = plan.assignments.rename("split")
    train_ids = plan.point_ids_for(TRAIN)
    held_out_ids = ids[split.isin([VAL, TEST]).to_numpy()]

    if estimator_factory is None:

        def estimator_factory(target):
            """The real stage-1 pipeline for one property."""
            return build_stage1_pipeline(settings, config, features, table.categorical_columns, target=target)

    folds, cv_kind = inner_folds(plan, settings, kind=kind, logger=logger)
    n_folds = int(folds.nunique()) if len(folds) else 0
    all_folds = pd.Series(NOT_IN_A_FOLD, index=ids, dtype=int)
    all_folds.loc[folds.index] = folds

    predictions = pd.DataFrame(index=ids)
    predicted_by = pd.DataFrame(index=ids)
    for target in settings.auxiliary.targets:
        column = stage1_prediction_column(target)
        values = pd.Series(np.nan, index=ids, dtype=float)
        source = pd.Series(NOT_PREDICTED, index=ids, dtype=object)
        measured = table.frame.loc[train_ids, target].notna()
        measured_ids = train_ids[measured.to_numpy()]
        unmeasured_ids = train_ids[~measured.to_numpy()]

        if len(measured_ids) == 0:
            logger.warning(f"No training point has a measured {target}; its stage-1 predictions are left empty.")
            predictions[column], predicted_by[target] = values, source
            continue

        out_of_fold = _out_of_fold(
            estimator_factory(target), X.loc[measured_ids], table.frame.loc[measured_ids, target], folds
        )
        if out_of_fold is None:
            logger.warning(
                f"{len(measured_ids)} training point(s) with a measured {target} cannot be cut into folds; "
                "they are left without a stage-1 prediction rather than predicted by a model that saw them."
            )
        else:
            values.loc[measured_ids] = out_of_fold
            source.loc[measured_ids] = [f"oof_fold_{fold}" for fold in folds.loc[measured_ids]]

        # One model on every measured training point, for the points none of those fits saw.
        to_predict = held_out_ids.append(unmeasured_ids)
        if len(to_predict):
            final = clone(estimator_factory(target))
            final.fit(X.loc[measured_ids], table.frame.loc[measured_ids, target])
            values.loc[to_predict] = np.asarray(final.predict(X.loc[to_predict]), dtype=float).reshape(-1)
            source.loc[to_predict] = FULL_TRAIN
        predictions[column], predicted_by[target] = values, source

    logger.info(
        f"Stage 1 ({settings.stage1.estimator}, {cv_kind}, {n_folds} fold(s)): {len(train_ids)} training point(s) "
        f"predicted out of fold, {len(held_out_ids)} validation/test point(s) by the full training fit."
    )
    return AuxiliaryPredictions(
        predictions=predictions,
        folds=all_folds,
        predicted_by=predicted_by,
        split=split,
        cv_kind=cv_kind,
        n_folds=n_folds,
    )


def _out_of_fold(estimator, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> Optional[np.ndarray]:
    """Out-of-fold predictions for the rows of ``X``, through the given folds; None when impossible.

    The folds are handed to ``cross_val_predict`` as explicit index pairs, so the fit that predicted
    each point is exactly the one the returned fold assignment names.
    """
    if folds.empty:
        return None
    fold_of = folds.reindex(X.index).to_numpy()
    pairs = []
    for fold in np.unique(fold_of):
        held = np.flatnonzero(fold_of == fold)
        kept = np.flatnonzero(fold_of != fold)
        if len(kept) == 0:
            return None
        pairs.append((kept, held))
    if len(pairs) < 2:
        return None
    return np.asarray(cross_val_predict(clone(estimator), X, y, cv=pairs, n_jobs=1), dtype=float).reshape(-1)
