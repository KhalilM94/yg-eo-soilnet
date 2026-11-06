"""Stage 2 of A1: the soil organic matter models compared on each split.

Every model is fitted on the training points of one split plan, its settings tuned by inner
cross-validation over the same folds stage 1 used (grouped by the plan's blocks or regions when it
has them), and scored on the test points - and on the validation points, when the plans have them
(by default they do not: nothing is fitted or tuned on them, so they would only shrink training):

``global``
    Every covariate except the stage-1 predictions.
``global_aux``
    The same, plus the predicted CaCO3 and CEC.
``per_class_measured``
    One ``global_aux`` model per FAO carbonate class, each point placed in its class by its
    **measured** CaCO3 - for the points it predicts too. A new point has no measured CaCO3, so this
    is a diagnostic upper bound on what routing by carbonate could give, never a usable model.
``per_class_predicted``
    The same, with each point placed by its **predicted** CaCO3: the usable version.
``local``
    Within one region only: out-of-fold predictions inside the held-out region of a ``logo`` plan,
    with no data from elsewhere - what a survey of that region alone would give. Built like the
    others so the transfer loss is fair (:func:`local_folds`, :func:`run_local`): outer folds of whole
    spatial blocks, stage 1 refitted inside the region for each fold, stage 2 tuned on the same grid.

A carbonate class with fewer than ``min_cell_size`` training points gets no model of its own; its
points - and points with no class - are predicted by ``global_aux``, and flagged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from yg_eo_soilnet.a1.auxiliary import AuxiliaryPredictions
from yg_eo_soilnet.a1.config import stage1_prediction_column
from yg_eo_soilnet.a1.features import A1Table
from yg_eo_soilnet.a1.pipelines import build_stage_pipeline
from yg_eo_soilnet.a1.settings import A1Settings
from yg_eo_soilnet.a1.strata import carbonate_class
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL, SplitPlan

GLOBAL = "global"
GLOBAL_AUX = "global_aux"
PER_CLASS_MEASURED = "per_class_measured"
PER_CLASS_PREDICTED = "per_class_predicted"
LOCAL = "local"
#: The models every plan gets.
MODELS = (GLOBAL, GLOBAL_AUX, PER_CLASS_MEASURED, PER_CLASS_PREDICTED)

#: ``model_role`` of an ordinary model.
ROLE_MODEL = "model"
#: ``model_role`` of a model that reads something a new point would not have.
ROLE_DIAGNOSTIC = "diagnostic_upper_bound"
#: The models that are diagnostics, not usable predictors.
DIAGNOSTIC_MODELS = (PER_CLASS_MEASURED,)


@dataclass
class Stage2Context:
    """Everything the stage-2 models of one plan read.

    Attributes
    ----------
    X : pandas.DataFrame
        The inputs of every plan point: the table's covariates and the stage-1 predictions.
    y : pandas.Series
        The measured target.
    split : pandas.Series
        Each point's split.
    folds : pandas.Series
        The inner fold of each training point (from stage 1), for tuning.
    class_measured, class_predicted : pandas.Series
        Each point's carbonate class from its measured and from its predicted CaCO3.
    categorical : tuple of str
        The inputs that are classes.
    settings, config
        The A1 settings and the main configuration.
    cache : dict
        Model results already computed for this plan, so ``global_aux`` is fitted once even when the
        per-class models fall back to it.
    """

    X: pd.DataFrame
    y: pd.Series
    split: pd.Series
    folds: pd.Series
    class_measured: pd.Series
    class_predicted: pd.Series
    categorical: tuple
    settings: A1Settings
    config: object
    cache: dict = field(default_factory=dict)

    @property
    def train_ids(self) -> pd.Index:
        """The training points with a measured target: what every model is fitted on."""
        train = self.split.index[self.split.to_numpy() == TRAIN]
        return train[self.y.reindex(train).notna().to_numpy()]

    @property
    def held_out_ids(self) -> pd.Index:
        """The held-out points - test, and validation when the plan has any: what every model is scored on."""
        return self.split.index[self.split.isin([VAL, TEST]).to_numpy()]


@dataclass(frozen=True)
class ModelResult:
    """One model's predictions for the held-out points of one plan.

    Attributes
    ----------
    predicted : pandas.Series
        Indexed by point id: every held-out point (test, and validation if any).
    fallback : pandas.Series
        True where the point was predicted by ``global_aux`` because its own class had no model.
    tuned_params : dict
        The settings chosen; per class for the per-class models.
    role : str
        ``model``, or ``diagnostic_upper_bound`` for a model reading what a new point lacks.
    local_fold, local_block : pandas.Series or None
        For the ``local`` model, each point's outer fold and spatial block.
    fitted : sklearn.pipeline.Pipeline or None
        The fitted pipeline, for a model fitted once on every training point (``global``,
        ``global_aux``, a covariate group); None for the per-class and local models. The
        dissimilarity index reads its feature importances.
    """

    predicted: pd.Series
    fallback: pd.Series
    tuned_params: dict
    role: str = ROLE_MODEL
    fitted: object = field(default=None, compare=False, repr=False)
    local_fold: Optional[pd.Series] = field(default=None, compare=False, repr=False)
    local_block: Optional[pd.Series] = field(default=None, compare=False, repr=False)

    @property
    def tuned_params_json(self) -> str:
        """The chosen settings as JSON, for the predictions table."""
        return json.dumps(self.tuned_params, sort_keys=True, default=str)


def build_context(
    table: A1Table, plan: SplitPlan, auxiliary: AuxiliaryPredictions, settings: A1Settings, config
) -> Stage2Context:
    """Assemble the inputs, target, folds and classes of one plan.

    Returns
    -------
    Stage2Context
    """
    ids = plan.assignments.index
    X = pd.concat([table.frame.loc[ids, list(table.feature_columns)], auxiliary.predictions.reindex(ids)], axis=1)
    X = X.astype({column: "float64" for column in X.select_dtypes(include=["int64", "int32"]).columns})
    carbonate = settings.carbonate
    predicted_caco3 = auxiliary.predictions.reindex(ids).get(stage1_prediction_column(carbonate.column))
    if predicted_caco3 is None:
        predicted_caco3 = pd.Series(np.nan, index=ids)
    return Stage2Context(
        X=X,
        y=table.frame.loc[ids, settings.target],
        split=plan.assignments.rename("split"),
        folds=auxiliary.folds[auxiliary.folds >= 0],
        class_measured=carbonate_class(table.frame.loc[ids, carbonate.column], carbonate.edges, carbonate.labels),
        class_predicted=carbonate_class(predicted_caco3, carbonate.edges, carbonate.labels),
        categorical=tuple(table.categorical_columns),
        settings=settings,
        config=config,
    )


def fit_tuned(pipeline, grid: Mapping[str, list], X: pd.DataFrame, y: pd.Series, folds: pd.Series):
    """Fit ``pipeline`` on these rows, its settings chosen by cross-validation over ``folds``.

    The search uses only the rows given - the training points - through the folds stage 1 used, so
    a block or region is never split between a search fold's training and validation parts.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        With the first grid point already set.
    grid : mapping
        Each setting and the values to try.
    X, y : pandas.DataFrame, pandas.Series
        The rows to fit on, indexed by point id.
    folds : pandas.Series
        A fold per point id; points of ``X`` not in it are never held out.

    Returns
    -------
    model : estimator
        Fitted on every row of ``X``.
    params : dict
        The settings chosen, and ``tuned``: False when there was nothing to choose or too few folds
        to choose with, in which case the first grid point was used.
    """
    combinations = int(np.prod([len(values) for values in grid.values()])) if grid else 1
    pairs = _fold_pairs(folds.reindex(X.index))
    if combinations > 1 and len(pairs) >= 2:
        search = GridSearchCV(
            clone(pipeline),
            param_grid=dict(grid),
            cv=pairs,
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=1,
        )
        search.fit(X, y)
        return search.best_estimator_, {**search.best_params_, "tuned": True}
    model = clone(pipeline).fit(X, y)
    return model, {**{name: values[0] for name, values in grid.items()}, "tuned": False}


def run_model(
    context: Stage2Context,
    model: str,
    *,
    global_columns: Sequence[str],
    auxiliary_columns: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> ModelResult:
    """Fit one of :data:`MODELS` on the plan's training points and predict its held-out points.

    Parameters
    ----------
    context : Stage2Context
    model : str
        One of :data:`MODELS`.
    global_columns : sequence of str
        The inputs of ``global``.
    auxiliary_columns : sequence of str
        The stage-1 prediction columns ``global_aux`` adds.
    logger : logging.Logger, optional

    Returns
    -------
    ModelResult
    """
    logger = logger or logging.getLogger(__name__)
    if model in context.cache:
        return context.cache[model]
    aux_features = list(global_columns) + [column for column in auxiliary_columns if column not in global_columns]

    if model == GLOBAL:
        result = _single_model(context, list(global_columns))
    elif model == GLOBAL_AUX:
        result = _single_model(context, aux_features)
    elif model in (PER_CLASS_MEASURED, PER_CLASS_PREDICTED):
        classes = context.class_measured if model == PER_CLASS_MEASURED else context.class_predicted
        fallback = run_model(
            context, GLOBAL_AUX, global_columns=global_columns, auxiliary_columns=auxiliary_columns, logger=logger
        )
        result = _per_class(
            context,
            aux_features,
            classes,
            fallback,
            role=ROLE_DIAGNOSTIC if model in DIAGNOSTIC_MODELS else ROLE_MODEL,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown stage-2 model {model!r}; expected one of {list(MODELS)}")
    context.cache[model] = result
    return result


def run_group(context: Stage2Context, columns: Sequence[str]) -> ModelResult:
    """``global_aux`` restricted to one covariate group's columns.

    Returns
    -------
    ModelResult
    """
    return _single_model(context, list(columns))


#: The fewest spatial blocks a region needs for a ``local`` score: with 3 or more, every outer
#: training fold holds at least 2 blocks, so stage 1 can be cross-fitted, and stage 2 tuned, by block.
MIN_LOCAL_BLOCKS = 3
#: Where :func:`local_folds` keeps its result in a plan's :class:`Stage2Context`, for every group.
LOCAL_CACHE_KEY = "local_folds"


@dataclass
class LocalFolds:
    """The outer folds of a ``local`` score inside one region, each with its own stage 1.

    Attributes
    ----------
    folds : list of (int, Stage2Context)
        Per outer fold, the context of a sub-plan of the region: the other folds train, this fold
        tests, the stage-1 predictions are refitted on those training points alone.
    block : pandas.Series
        Each measured point of the region's spatial block.
    outer_fold : pandas.Series
        Each measured point's outer fold.
    reason : str or None
        Why there is no ``local`` score, when there is none; ``folds`` is then empty.
    """

    folds: list = field(default_factory=list)
    block: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    outer_fold: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reason: Optional[str] = None


def local_folds(
    context: Stage2Context,
    region_ids: pd.Index,
    table: A1Table,
    *,
    estimator_factory=None,
    logger: Optional[logging.Logger] = None,
) -> LocalFolds:
    """Cut one held-out region into spatial-block folds and give each its own, local, stage 1.

    The ``local`` model is the reference a transferred model's loss is measured against, so it is
    built as the transferred one is, but from the region's own points alone:

    1. the region's points with a measured target are placed in square blocks of
       ``bootstrap.block_size_km`` (:func:`~yg_eo_soilnet.a1.evaluation.spatial_blocks`), and the
       blocks packed into ``min(inner_cv_folds, blocks)`` outer folds - whole blocks, so a point is
       never predicted from its neighbours;
    2. for each outer fold, a sub-plan of the region (the other folds train, this one tests) goes
       through :func:`~yg_eo_soilnet.a1.auxiliary.predict_auxiliary`: stage 1 is fitted on that
       fold's training points only, its training points cross-fitted through folds of whole blocks.
       The logo plan's own stage 1, fitted on the other regions, is not used.

    The result is kept in ``context.cache`` and reused by every covariate group, so each region's
    stage 1 is fitted once per fold.

    Parameters
    ----------
    context : Stage2Context
        The ``logo`` plan's context: its settings, configuration and measured target.
    region_ids : pandas.Index
        The held-out region's points - the plan's test points.
    table : A1Table
        The A1 points table, for the coordinates and the stage-1 inputs.
    estimator_factory : callable, optional
        Passed to :func:`~yg_eo_soilnet.a1.auxiliary.predict_auxiliary` (tests use it to watch what
        each stage-1 fit sees).
    logger : logging.Logger, optional

    Returns
    -------
    LocalFolds
        With ``reason`` set and no folds when the region holds fewer than ``min_cell_size``
        measured points or fewer than :data:`MIN_LOCAL_BLOCKS` blocks.
    """
    from sklearn.model_selection import GroupKFold

    from yg_eo_soilnet.a1.auxiliary import predict_auxiliary
    from yg_eo_soilnet.a1.evaluation import spatial_blocks
    from yg_eo_soilnet.a1.plans import SPATIAL_KFOLD

    if LOCAL_CACHE_KEY in context.cache:
        return context.cache[LOCAL_CACHE_KEY]
    logger = logger or logging.getLogger(__name__)
    settings = context.settings
    block_km = settings.bootstrap.block_size_km
    measured = region_ids[context.y.reindex(region_ids).notna().to_numpy()]

    if len(measured) < settings.min_cell_size:
        local = LocalFolds(
            reason=f"the region holds {len(measured)} measured point(s), fewer than min_cell_size="
            f"{settings.min_cell_size}"
        )
    else:
        coordinates = table.coordinates.reindex(measured).rename_axis("point_id").reset_index()
        block = pd.Series(spatial_blocks(coordinates, block_km).to_numpy(), index=measured, name="block")
        n_blocks = int(block.nunique())
        if n_blocks < MIN_LOCAL_BLOCKS:
            local = LocalFolds(
                block=block,
                reason=f"the region's points fall in {n_blocks} spatial block(s) of {block_km:g} km; at least "
                f"{MIN_LOCAL_BLOCKS} are needed to fit and tune it without a block on both sides",
            )
        else:
            n_folds = min(settings.inner_cv_folds, n_blocks)
            outer = pd.Series(-1, index=measured, name="outer_fold")
            splits = GroupKFold(n_splits=n_folds).split(np.zeros((len(measured), 1)), groups=block.to_numpy())
            for fold, (_, test_positions) in enumerate(splits):
                outer.iloc[test_positions] = fold
            folds = []
            for fold in range(n_folds):
                assignments = pd.Series(
                    np.where(outer.to_numpy() == fold, TEST, TRAIN), index=pd.Index(measured, name="point_id")
                )
                subplan = SplitPlan(
                    assignments=assignments,
                    strategy=SPATIAL_KFOLD,
                    test_size=round(float((outer == fold).mean()), 6),
                    val_size=0.0,
                    seed=settings.seed,
                    population_policy="intersect",
                    eligibility={"sklearn": frozenset(measured)},
                    clusters=block.astype(int),
                )
                auxiliary = predict_auxiliary(
                    table,
                    subplan,
                    settings,
                    context.config,
                    kind=SPATIAL_KFOLD,
                    estimator_factory=estimator_factory,
                    logger=logger,
                )
                folds.append((fold, build_context(table, subplan, auxiliary, settings, context.config)))
            local = LocalFolds(folds=folds, block=block, outer_fold=outer)
    context.cache[LOCAL_CACHE_KEY] = local
    return local


def run_local(context: Stage2Context, columns: Sequence[str], local: LocalFolds) -> Optional[ModelResult]:
    """Out-of-fold predictions inside one region, from that region's own points alone, fitted fairly.

    Each outer fold of :func:`local_folds` gets the same model ``global_aux`` gets on a split - the
    same inputs and the same settings grid, tuned by :func:`fit_tuned` through folds of whole blocks
    of the fold's training points (the ones its stage 1 was cross-fitted through) - and predicts
    the fold's points.

    Parameters
    ----------
    context : Stage2Context
        The ``logo`` plan's context (unused but for symmetry with the other models).
    columns : sequence of str
        The inputs.
    local : LocalFolds

    Returns
    -------
    ModelResult or None
        Every measured point of the region once, with ``local_fold`` and ``local_block``; None when
        ``local.reason`` says there is no local score.
    """
    if local.reason is not None:
        return None
    predicted, params = [], {}
    for fold, sub_context in local.folds:
        outcome = _single_model(sub_context, list(columns))
        predicted.append(outcome.predicted)
        params[f"fold_{fold}"] = outcome.tuned_params
    predicted = pd.concat(predicted).reindex(local.outer_fold.index)
    return ModelResult(
        predicted=predicted.astype(float),
        fallback=pd.Series(False, index=predicted.index),
        tuned_params={**params, "n_folds": len(local.folds)},
        local_fold=local.outer_fold.astype(float),
        local_block=local.block.astype(float),
    )


# --- the pieces ----------------------------------------------------------------------------


def _single_model(context: Stage2Context, columns: list[str]) -> ModelResult:
    """One model on these columns, fitted on every training point."""
    settings = context.settings
    pipeline, grid = build_stage_pipeline(
        settings, context.config, columns, context.categorical, stage=settings.stage2, target=settings.target
    )
    train, held = context.train_ids, context.held_out_ids
    fitted, params = fit_tuned(pipeline, grid, context.X.loc[train, columns], context.y.loc[train], context.folds)
    predicted = pd.Series(np.asarray(fitted.predict(context.X.loc[held, columns]), dtype=float), index=held)
    return ModelResult(predicted=predicted, fallback=pd.Series(False, index=held), tuned_params=params, fitted=fitted)


def _per_class(
    context: Stage2Context,
    columns: list[str],
    classes: pd.Series,
    fallback: ModelResult,
    *,
    role: str,
    logger: logging.Logger,
) -> ModelResult:
    """One model per carbonate class, each point routed by ``classes``; thin classes fall back."""
    settings = context.settings
    train, held = context.train_ids, context.held_out_ids
    predicted = fallback.predicted.copy()
    used_fallback = pd.Series(True, index=held)
    params: dict = {}

    for label in classes.cat.categories:
        class_train = train[(classes.reindex(train) == label).to_numpy()]
        class_held = held[(classes.reindex(held) == label).to_numpy()]
        if len(class_train) < settings.min_cell_size:
            params[str(label)] = {"fallback": True, "n_train": int(len(class_train))}
            if len(class_held):
                logger.info(
                    f"Carbonate class {label}: {len(class_train)} training point(s), under min_cell_size="
                    f"{settings.min_cell_size}; its {len(class_held)} held-out point(s) use {GLOBAL_AUX}."
                )
            continue
        pipeline, grid = build_stage_pipeline(
            settings, context.config, columns, context.categorical, stage=settings.stage2, target=settings.target
        )
        fitted, chosen = fit_tuned(
            pipeline, grid, context.X.loc[class_train, columns], context.y.loc[class_train], context.folds
        )
        params[str(label)] = {**chosen, "n_train": int(len(class_train))}
        if len(class_held):
            predicted.loc[class_held] = np.asarray(fitted.predict(context.X.loc[class_held, columns]), dtype=float)
            used_fallback.loc[class_held] = False
    return ModelResult(predicted=predicted, fallback=used_fallback, tuned_params=params, role=role)


def _fold_pairs(fold_of: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(fit positions, held positions)`` for each fold that leaves something on both sides."""
    values = fold_of.to_numpy()
    known = ~pd.isna(values)
    pairs = []
    for fold in pd.unique(values[known]):
        held = np.flatnonzero(known & (values == fold))
        kept = np.flatnonzero(~(known & (values == fold)))
        if len(held) and len(kept):
            pairs.append((kept, held))
    return pairs
