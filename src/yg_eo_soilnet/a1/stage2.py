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
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from yg_eo_soilnet.a1.arms import fit_kwargs
from yg_eo_soilnet.a1.auxiliary import AuxiliaryPredictions
from yg_eo_soilnet.a1.config import stage1_prediction_column
from yg_eo_soilnet.a1.features import A1Table
from yg_eo_soilnet.a1.logs import format_duration
from yg_eo_soilnet.a1.pipelines import build_stage_pipeline, predict_on_cpu
from yg_eo_soilnet.a1.settings import A1Settings
from yg_eo_soilnet.a1.strata import carbonate_class
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL, SplitPlan

GLOBAL = "global"
GLOBAL_AUX = "global_aux"
PER_CLASS_MEASURED = "per_class_measured"
PER_CLASS_PREDICTED = "per_class_predicted"
LOCAL = "local"
#: The local model on Global's inputs - every covariate, no predicted CaCO3/CEC - fitted inside a
#: held-out basin as :data:`LOCAL` is (same outer folds), so Global's transfer loss compares like with like.
LOCAL_GLOBAL = "local_global"
#: The local models, each the in-basin counterpart of one transferred model.
LOCAL_MODELS = (LOCAL, LOCAL_GLOBAL)
#: The local model with the same inputs as each transferred model.
LOCAL_OF = {GLOBAL: LOCAL_GLOBAL, GLOBAL_AUX: LOCAL}
#: The quantile model of the prediction intervals, fitted only when ``stage2.models`` lists it
#: (:mod:`yg_eo_soilnet.a1.uncertainty`).
GLOBAL_Q = "global_q"
#: The models every plan gets, unless ``stage2.models`` lists fewer.
MODELS = (GLOBAL, GLOBAL_AUX, PER_CLASS_MEASURED, PER_CLASS_PREDICTED)
#: The models that read stage 1's predicted CaCO3 and CEC (the per-class models through their
#: ``global_aux`` inputs and fallback); a run without any of them skips stage 1.
STAGE1_MODELS = (GLOBAL_AUX, PER_CLASS_MEASURED, PER_CLASS_PREDICTED)


def needs_stage1(models) -> bool:
    """Whether any of these SOM models reads stage 1's predictions."""
    return any(model in STAGE1_MODELS for model in models)


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
    global_settings : dict or None
        Global's chosen settings read back from its predictions file, for ``global_q`` on a resumed
        run where Global itself is not fitted again.
    quantile_device : str or None
        Where ``global_q`` fits (``cuda``, or ``cpu`` when the quantile objective fell back there);
        the stage-2 device when None.
    sample_weight : pandas.Series or None
        The weight of each training point in every fit (a ``mixed_weighted`` plan,
        :mod:`yg_eo_soilnet.a1.arms`); None fits every point with the same weight.
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
    sample_weight: Optional[pd.Series] = None
    quantile_device: Optional[str] = None
    global_settings: Optional[dict] = None

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
    extra : pandas.DataFrame or None
        For ``global_q``: the quantile, interval and ``quantiles_sorted`` columns of each held-out
        point, written beside the usual ones.
    calibration : dict or None
        For ``global_q``: the conformal correction ``Q`` of each level and what it rests on.
    """

    predicted: pd.Series
    fallback: pd.Series
    tuned_params: dict
    role: str = ROLE_MODEL
    fitted: object = field(default=None, compare=False, repr=False)
    local_fold: Optional[pd.Series] = field(default=None, compare=False, repr=False)
    extra: Optional[pd.DataFrame] = field(default=None, compare=False, repr=False)
    calibration: Optional[dict] = field(default=None, compare=False, repr=False)
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


def fit_tuned(
    pipeline,
    grid: Mapping[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    sample_weight: Optional[pd.Series] = None,
):
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
    sample_weight : pandas.Series, optional
        A weight per point id, given to every fit: the final one and each fit of the search. The
        search still scores each validation fold by its unweighted RMSE.

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
            scoring=_neg_rmse_on_cpu,
            refit=True,
            n_jobs=1,
        )
        # The weights are cut with the rows for each fit; the scorer never sees them.
        search.fit(X, y, **fit_kwargs(sample_weight, X.index))
        return search.best_estimator_, {**search.best_params_, "tuned": True}
    model = clone(pipeline).fit(X, y, **fit_kwargs(sample_weight, X.index))
    return model, {**{name: values[0] for name, values in grid.items()}, "tuned": False}


def _neg_rmse_on_cpu(estimator, X, y) -> float:
    """The search's score, ``neg_root_mean_squared_error``, predicted on the processor (:func:`to_cpu`)."""
    from sklearn.metrics import root_mean_squared_error

    return -float(root_mean_squared_error(y, predict_on_cpu(estimator, X)))


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
    elif model == GLOBAL_Q:
        result = _quantile_model(context, list(global_columns))
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
    with_stage1 : bool
        Whether each fold's context holds its own stage-1 predictions (``local`` needs them;
        ``local_global`` does not, and its folds skip stage 1).
    """

    folds: list = field(default_factory=list)
    block: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    outer_fold: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reason: Optional[str] = None
    with_stage1: bool = True


def local_folds(
    context: Stage2Context,
    region_ids: pd.Index,
    table: A1Table,
    *,
    estimator_factory=None,
    with_stage1: bool = True,
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
    stage 1 is fitted once per fold. With ``with_stage1=False`` (``local_global``, which reads no
    stage-1 prediction) step 2 only cuts each fold's tuning folds, the folds stage 1 would have
    used; the outer folds are the same, being cut before. A cached result without stage 1 is
    rebuilt when a later call needs it.

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
    with_stage1 : bool, default True
        Fit each fold's own stage 1; False skips it.
    logger : logging.Logger, optional

    Returns
    -------
    LocalFolds
        With ``reason`` set and no folds when the region holds fewer than ``min_cell_size``
        measured points or fewer than :data:`MIN_LOCAL_BLOCKS` blocks.
    """
    from sklearn.model_selection import GroupKFold

    from yg_eo_soilnet.a1.auxiliary import folds_only, predict_auxiliary
    from yg_eo_soilnet.a1.evaluation import spatial_blocks
    from yg_eo_soilnet.a1.plans import SPATIAL_KFOLD

    cached = context.cache.get(LOCAL_CACHE_KEY)
    if cached is not None and (cached.with_stage1 or not with_stage1):
        return cached
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
            # Whole blocks per fold; dealt at random for every seed but the first, as stage 1's folds are.
            cutter = (
                GroupKFold(n_splits=n_folds, shuffle=True, random_state=settings.model_seed)
                if settings.shuffle_groups
                else GroupKFold(n_splits=n_folds)
            )
            splits = cutter.split(np.zeros((len(measured), 1)), groups=block.to_numpy())
            for fold, (_, test_positions) in enumerate(splits):
                outer.iloc[test_positions] = fold
            folds = []
            region_name = (
                str(table.frame[table.region_column].reindex(measured).dropna().iloc[0])
                if table.region_column and table.frame[table.region_column].reindex(measured).notna().any()
                else "region"
            )
            for fold in range(n_folds):
                started = time.monotonic()
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
                auxiliary = (
                    predict_auxiliary(
                        table,
                        subplan,
                        settings,
                        context.config,
                        kind=SPATIAL_KFOLD,
                        estimator_factory=estimator_factory,
                        logger=logger,
                    )
                    if with_stage1
                    else folds_only(subplan, settings, kind=SPATIAL_KFOLD, logger=logger)
                )
                folds.append((fold, build_context(table, subplan, auxiliary, settings, context.config)))
                n_test = int((outer == fold).sum())
                logger.info(
                    f"  local {region_name} fold {fold + 1}/{n_folds}: {len(measured) - n_test:,} train / {n_test:,} "
                    f"test ({block[outer == fold].nunique()} of {n_blocks} blocks), stage 1 "
                    + (format_duration(time.monotonic() - started) if with_stage1 else "skipped (not read)")
                )
            local = LocalFolds(folds=folds, block=block, outer_fold=outer, with_stage1=with_stage1)
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
    fitted, params = fit_tuned(
        pipeline, grid, context.X.loc[train, columns], context.y.loc[train], context.folds, **_weights(context)
    )
    predicted = pd.Series(predict_on_cpu(fitted, context.X.loc[held, columns]), index=held)
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
            pipeline,
            grid,
            context.X.loc[class_train, columns],
            context.y.loc[class_train],
            context.folds,
            **_weights(context),
        )
        params[str(label)] = {**chosen, "n_train": int(len(class_train))}
        if len(class_held):
            predicted.loc[class_held] = predict_on_cpu(fitted, context.X.loc[class_held, columns])
            used_fallback.loc[class_held] = False
    return ModelResult(predicted=predicted, fallback=used_fallback, tuned_params=params, role=role)


def quantile_stage(settings: A1Settings, device: Optional[str] = None):
    """Stage 2's estimator turned into the one quantile model: ``reg:quantileerror`` at every quantile.

    One tree per quantile (``multi_strategy: one_output_per_tree``), whatever the model list's XGBoost
    entry sets: its ``multi_output_tree`` (one tree for both stage-1 targets) would make every
    quantile share each tree's splits.
    """
    from dataclasses import replace

    init_args = {
        **dict(settings.stage2.init_args),
        "objective": "reg:quantileerror",
        "quantile_alpha": [float(q) for q in settings.uncertainty.quantiles],
        "multi_strategy": "one_output_per_tree",
    }
    if device is not None:
        init_args["device"] = device
    return replace(settings.stage2, init_args=init_args)


def predict_quantiles(fitted, X: pd.DataFrame) -> np.ndarray:
    """A quantile model's predictions on the processor: one row per point, one column per quantile."""
    from yg_eo_soilnet.a1.pipelines import to_cpu

    return np.asarray(to_cpu(fitted).predict(X), dtype=float).reshape(len(X), -1)


def out_of_fold_quantiles(
    pipeline, params: Mapping, X: pd.DataFrame, y: pd.Series, folds: pd.Series, sample_weight=None
) -> Optional[pd.DataFrame]:
    """Each training point's quantiles from a model fitted on the other folds, with the tuned settings.

    The folds are the plan's own (random folds, blocks or basins, as for stage 1), so a point is
    never predicted by a model that saw its fold. None with fewer than two usable folds.
    """
    pairs = _fold_pairs(folds.reindex(X.index))
    if len(pairs) < 2:
        return None
    chosen = {name: value for name, value in params.items() if name != "tuned"}
    values = np.full((len(X), 0), np.nan)
    for fit_positions, held_positions in pairs:
        model = clone(pipeline).set_params(**chosen)
        model.fit(X.iloc[fit_positions], y.iloc[fit_positions], **fit_kwargs(sample_weight, X.index[fit_positions]))
        predicted = predict_quantiles(model, X.iloc[held_positions])
        if values.shape[1] == 0:
            values = np.full((len(X), predicted.shape[1]), np.nan)
        values[held_positions] = predicted
    frame = pd.DataFrame(values, index=X.index)
    return frame.dropna()


def global_settings(context: Stage2Context) -> dict:
    """The settings Global's search chose in this plan, seed and arm, for ``global_q`` to reuse.

    From Global's result when it was fitted in this process (it always comes first), otherwise from
    its predictions file (:attr:`Stage2Context.global_settings`, set by the runner on a resumed run).

    Raises
    ------
    RuntimeError
        If neither is there: ``global_q`` has no settings of its own.
    """
    if GLOBAL in context.cache:
        chosen = context.cache[GLOBAL].tuned_params
    elif context.global_settings is not None:
        chosen = context.global_settings
    else:
        raise RuntimeError(f"{GLOBAL_Q} reuses the settings of {GLOBAL}, which was not fitted in this plan.")
    return {name: value for name, value in chosen.items() if name != "tuned"}


def _quantile_model(context: Stage2Context, columns: list[str]) -> ModelResult:
    """``global_q``: every quantile in one model, on Global's tuned settings, and its intervals conformalized.

    ``global_q`` is not searched: it takes the settings Global's search chose in the same plan, seed
    and arm (:func:`global_settings`) - the hyperparameters of the tuned mean model. Each
    point's quantiles are sorted (``quantiles_sorted`` marks those that crossed). With
    ``uncertainty.conformal``, each central interval is corrected by ``Q`` from the training points'
    out-of-fold quantiles (:func:`~yg_eo_soilnet.a1.uncertainty.calibrate`). A log-transformed SOM is
    fitted in log space and each quantile back-transformed, which is exact: the transform keeps
    their order. The median (0.5) is the ``predicted`` column.
    """
    from dataclasses import replace

    from yg_eo_soilnet.a1 import uncertainty as uq

    settings = context.settings
    options = settings.uncertainty
    quantiles = list(options.quantiles)
    device = context.quantile_device
    built_with = replace(settings, device=device) if device is not None else settings
    pipeline, _ = build_stage_pipeline(
        built_with,
        context.config,
        columns,
        context.categorical,
        stage=quantile_stage(settings, device),
        target=settings.target,
    )
    train, held = context.train_ids, context.held_out_ids
    X_train, y_train = context.X.loc[train, columns], context.y.loc[train]
    chosen = global_settings(context)
    fitted = clone(pipeline).set_params(**chosen).fit(X_train, y_train, **fit_kwargs(context.sample_weight, train))
    params = {**chosen, "tuned": False, "settings_from": GLOBAL}
    values, crossed = uq.sort_quantiles(predict_quantiles(fitted, context.X.loc[held, columns]))

    calibrations, info = (
        None,
        {
            "quantiles": quantiles,
            "levels": list(options.levels),
            "conformal": bool(options.conformal),
            "crossing_share_held_out": float(crossed.mean()) if len(crossed) else 0.0,
            "device": device,
            "settings_from": GLOBAL,
            "settings": chosen,
        },
    )
    if options.conformal:
        oof = out_of_fold_quantiles(pipeline, chosen, X_train, y_train, context.folds, context.sample_weight)
        if oof is None or oof.empty:
            info["note"] = "fewer than two folds: no calibration point, raw intervals only"
        else:
            calibration_values, calibration_crossed = uq.sort_quantiles(oof.to_numpy())
            y_calibration = context.y.loc[oof.index].to_numpy(dtype=float)
            calibrations = {}
            for level in options.levels:
                low_q, high_q = uq.level_bounds(level)
                calibrations[level] = uq.calibrate(
                    y_calibration,
                    calibration_values[:, uq.quantile_index(quantiles, low_q)],
                    calibration_values[:, uq.quantile_index(quantiles, high_q)],
                    level,
                )
            info["n_calibration"] = int(len(oof))
            info["crossing_share_calibration"] = float(calibration_crossed.mean())
            info["levels_q"] = {
                uq.level_tag(level): {"q": c.q, "raw_coverage_calibration": c.raw_coverage}
                for level, c in calibrations.items()
            }
    extra = pd.DataFrame(uq.interval_columns(quantiles, values, options.levels, calibrations), index=held)
    extra[uq.SORTED_COLUMN] = crossed
    median = pd.Series(values[:, uq.quantile_index(quantiles, 0.5)], index=held)
    return ModelResult(
        predicted=median,
        fallback=pd.Series(False, index=held),
        tuned_params=params,
        fitted=fitted,
        extra=extra,
        calibration=info,
    )


def _weights(context: Stage2Context) -> dict:
    """``{"sample_weight": ...}`` when the plan weighs its points; ``{}`` otherwise."""
    return {} if context.sample_weight is None else {"sample_weight": context.sample_weight}


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
