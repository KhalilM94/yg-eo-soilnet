"""The ``local`` model: built inside the held-out region as fairly as the transferred one.

Outer folds of whole spatial blocks, a stage 1 refitted inside the region for each fold, and stage 2
tuned on the same grid through block-grouped folds. Spies record what every fit saw and predicted,
in both stages, so the tests can check that no point is predicted by a fit that saw it or its block.
"""

from dataclasses import replace
from itertools import count

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from yg_eo_soilnet.a1 import stage2
from yg_eo_soilnet.a1.auxiliary import predict_auxiliary
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.plans import LOGO, logo_plans, plan_population
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.a1.stage2 import (
    LOCAL_CACHE_KEY,
    MIN_LOCAL_BLOCKS,
    build_context,
    local_folds,
    run_local,
)
from yg_eo_soilnet.datamodules.splitting import TEST

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    # Two settings to choose between, so the local model has something to tune.
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}, "params": {"model__max_depth": [2, 3]}},
    "inner_cv_folds": 3,
}

STAGE1_LOG: dict = {"fits": {}, "predictions": []}
_fit_numbers = count()


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Records the ids each stage-1 fit saw and predicted; predicts the row mean."""

    def __init__(self, target=None):
        self.target = target

    def fit(self, X, y):
        self.fit_number_ = next(_fit_numbers)
        STAGE1_LOG["fits"][self.fit_number_] = frozenset(X.index)
        return self

    def predict(self, X):
        STAGE1_LOG["predictions"].append((self.fit_number_, frozenset(X.index)))
        return X.select_dtypes("number").mean(axis=1).fillna(0.0).to_numpy()


class Watched:
    """A fitted stage-2 model that records the points it was fitted on and the points it predicts."""

    def __init__(self, model, train_ids, log):
        self.model, self.train_ids, self.log = model, train_ids, log

    def predict(self, X):
        self.log.append((self.train_ids, frozenset(X.index)))
        return self.model.predict(X)

    def __getattr__(self, name):
        return getattr(self.model, name)


@pytest.fixture(scope="module")
def setup(a1_settings_in, tmp_path_factory):
    """The demo, with its first held-out region's logo plan and context."""
    folder = tmp_path_factory.mktemp("local")
    settings = load_settings(str(a1_settings_in(folder)(as_file=True, **FAST)))
    config = settings.load_config()
    table = build_table(settings, config)
    population = plan_population(settings, config, points=table)
    built = logo_plans(population, settings)[0]
    auxiliary = predict_auxiliary(table, built.plan, settings, config, kind=LOGO)
    region = built.plan.point_ids_for(TEST)
    return {
        "settings": settings,
        "config": config,
        "table": table,
        "built": built,
        "auxiliary": auxiliary,
        "region": region,
    }


def _context(setup, **settings_changes):
    settings = replace(setup["settings"], **settings_changes)
    return build_context(setup["table"], setup["built"].plan, setup["auxiliary"], settings, setup["config"])


@pytest.fixture(scope="module")
def spied(setup):
    """local_folds with stage 1 spied on, and the combined-group local model with stage 2 spied on."""
    STAGE1_LOG["fits"].clear()
    STAGE1_LOG["predictions"].clear()
    context = _context(setup)
    local = local_folds(context, setup["region"], setup["table"], estimator_factory=lambda target: SpyRegressor(target))

    stage2_log = []
    original = stage2.fit_tuned

    def watching(pipeline, grid, X, y, folds):
        model, params = original(pipeline, grid, X, y, folds)
        return Watched(model, frozenset(X.index), stage2_log), params

    stage2.fit_tuned = watching
    try:
        columns = [column for column in context.X.columns]
        outcome = run_local(context, columns, local)
    finally:
        stage2.fit_tuned = original
    return {"context": context, "local": local, "outcome": outcome, "stage2_log": stage2_log}


def _blocks_of(ids, block):
    return set(block.reindex(list(ids)).dropna())


# --- no fit sees the point it predicts, nor its block -------------------------------------------


def test_no_stage1_fit_predicts_a_point_it_saw_or_a_point_of_a_block_it_saw(spied):
    block = spied["local"].block
    assert STAGE1_LOG["predictions"], "the spy saw no stage-1 prediction"
    for fit_number, predicted in STAGE1_LOG["predictions"]:
        seen = STAGE1_LOG["fits"][fit_number]
        assert not seen & predicted
        assert not _blocks_of(seen, block) & _blocks_of(predicted, block)


def test_no_stage2_fit_predicts_a_point_it_saw_or_a_point_of_a_block_it_saw(spied):
    block = spied["local"].block
    log = spied["stage2_log"]
    assert len(log) == len(spied["local"].folds)  # one tuned fit per outer fold
    for seen, predicted in log:
        assert not seen & predicted
        assert not _blocks_of(seen, block) & _blocks_of(predicted, block)


def test_stage1_is_fitted_on_the_region_alone(spied, setup):
    region = set(setup["region"])
    for seen in STAGE1_LOG["fits"].values():
        assert seen <= region  # never the logo plan's training points from the other regions


# --- tuned like the transferred model --------------------------------------------------------------


def test_every_fold_of_the_local_model_is_tuned(spied):
    params = spied["outcome"].tuned_params
    folds = [key for key in params if key.startswith("fold_")]
    assert folds and all(params[key]["tuned"] is True for key in folds)
    assert all(params[key]["model__max_depth"] in (2, 3) for key in folds)


def test_the_search_folds_keep_each_block_whole(spied):
    block = spied["local"].block
    for _, sub_context in spied["local"].folds:
        search = sub_context.folds
        assert search.nunique() >= 2
        per_block = pd.Series(search.to_numpy(), index=search.index).groupby(block.reindex(search.index)).nunique()
        assert (per_block == 1).all()


# --- what is saved -------------------------------------------------------------------------------


def test_each_measured_point_is_predicted_once_with_its_fold_and_block(spied, setup):
    outcome, local = spied["outcome"], spied["local"]
    measured = setup["region"][spied["context"].y.reindex(setup["region"]).notna().to_numpy()]
    assert set(outcome.predicted.index) == set(measured) and outcome.predicted.notna().all()
    assert outcome.local_fold.equals(local.outer_fold.astype(float))
    assert outcome.local_block.equals(local.block.astype(float))
    # A block is never split between outer folds.
    assert (local.outer_fold.groupby(local.block).nunique() == 1).all()
    assert local.outer_fold.nunique() == min(setup["settings"].inner_cv_folds, local.block.nunique())


def test_stage1_runs_once_per_fold_whatever_the_number_of_groups(spied):
    fits_before = len(STAGE1_LOG["fits"])
    again = local_folds(spied["context"], pd.Index([]), None)
    assert again is spied["local"] and spied["context"].cache[LOCAL_CACHE_KEY] is again
    run_local(spied["context"], ["elevation", "slope"], again)
    assert len(STAGE1_LOG["fits"]) == fits_before


# --- regions too small to score fairly ---------------------------------------------------------


def _region_in_blocks(setup, block, n_blocks):
    """Points of the region whose grid - laid over them alone, as local_folds lays it - has n_blocks squares."""
    from itertools import combinations

    from yg_eo_soilnet.a1.evaluation import spatial_blocks

    coordinates = setup["table"].coordinates
    for chosen in combinations(block.value_counts().index[:12], n_blocks):
        region = block.index[block.isin(chosen).to_numpy()]
        frame = coordinates.reindex(region).rename_axis("point_id").reset_index()
        if spatial_blocks(frame, 10).nunique() == n_blocks:
            return region
    pytest.fail(f"no {n_blocks} demo blocks re-grid into {n_blocks} squares")


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_a_region_in_too_few_blocks_is_skipped_with_the_reason(setup, spied, n_blocks):
    region = _region_in_blocks(setup, spied["local"].block, n_blocks)
    context = _context(setup, min_cell_size=1)
    local = local_folds(context, region, setup["table"])
    assert local.folds == [] and run_local(context, ["elevation"], local) is None
    assert f"fall in {n_blocks} spatial block(s) of 10 km" in local.reason
    assert f"at least {MIN_LOCAL_BLOCKS}" in local.reason


def test_a_region_under_min_cell_size_is_skipped_with_the_reason(setup):
    context = _context(setup, min_cell_size=10_000)
    local = local_folds(context, setup["region"], setup["table"])
    assert local.folds == [] and "fewer than min_cell_size=10000" in local.reason


def test_the_blocks_are_squares_of_the_bootstrap_block_size(setup, spied):
    from yg_eo_soilnet.a1.settings import BootstrapSettings

    coarse = _context(setup, bootstrap=BootstrapSettings(block_size_km=50))
    local = local_folds(coarse, setup["region"], setup["table"])
    assert local.block.nunique() < spied["local"].block.nunique()
    assert np.isfinite(local.block).all()
