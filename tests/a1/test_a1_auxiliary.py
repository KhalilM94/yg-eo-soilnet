"""Stage 1 of A1 must never predict a point with a model that saw that point's lab value.

The check is direct: a spy estimator stands in for the model and records, for every fit, the point
ids it was fitted on, and for every prediction, which fit made it. Two rules are then asserted on
every kind of plan:

1. no validation or test id is ever in a fit;
2. no training point is predicted by a fit that contained it.
"""

import itertools
import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from yg_eo_soilnet.a1.auxiliary import (
    FULL_TRAIN,
    GROUP_KFOLD,
    KFOLD,
    NOT_IN_A_FOLD,
    NOT_PREDICTED,
    build_stage1_pipeline,
    inner_folds,
    predict_auxiliary,
)
from yg_eo_soilnet.a1.config import stage1_prediction_column
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.plans import (
    CONTROL,
    LOGO,
    MIXED,
    RANDOM,
    SPATIAL_KFOLD,
    logo_plans,
    mixed_plans,
    plan_population,
    random_plan,
    spatial_kfold_plans,
)
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL, SplitPlan

# --- the spy ----------------------------------------------------------------------------------

#: Every spy's record, by log key. Module level, because clone() copies an estimator's parameters
#: and a list held as one would be copied with it, losing what the copies recorded.
SPY_LOG: dict = {}
_fit_numbers = itertools.count()


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Records the ids each fit saw and the ids each fit predicted; predicts the row mean."""

    def __init__(self, log_key="default", target=None):
        self.log_key = log_key
        self.target = target

    def fit(self, X, y):
        self.fit_number_ = next(_fit_numbers)
        self.fit_ids_ = frozenset(X.index)
        SPY_LOG.setdefault(self.log_key, {"fits": {}, "predictions": []})
        SPY_LOG[self.log_key]["fits"][self.fit_number_] = (self.target, self.fit_ids_)
        return self

    def predict(self, X):
        SPY_LOG[self.log_key]["predictions"].append((self.fit_number_, self.target, frozenset(X.index)))
        return X.select_dtypes("number").mean(axis=1).fillna(0.0).to_numpy()


def _spy(key):
    return lambda target: SpyRegressor(log_key=key, target=target)


# --- the demo plans ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table(a1_demo_settings):
    return build_table(a1_demo_settings)


@pytest.fixture(scope="module")
def config(a1_demo_settings):
    return a1_demo_settings.load_config()


@pytest.fixture(scope="module")
def demo_plans(a1_demo_settings, table):
    population = plan_population(a1_demo_settings, points=table)
    mixed = mixed_plans(population, a1_demo_settings)
    return {
        RANDOM: random_plan(population, a1_demo_settings),
        SPATIAL_KFOLD: spatial_kfold_plans(population, a1_demo_settings)[0],
        LOGO: logo_plans(population, a1_demo_settings)[0],
        MIXED: next(built for built in mixed if built.kind == MIXED and built.fraction == 0.5),
        CONTROL: next(built for built in mixed if built.kind == CONTROL and built.fraction == 0.5),
    }


def _run_with_spy(table, built, settings, config, key):
    SPY_LOG.pop(key, None)
    result = predict_auxiliary(table, built.plan, settings, config, kind=built.kind, estimator_factory=_spy(key))
    return result, SPY_LOG[key]


ALL_KINDS = [RANDOM, SPATIAL_KFOLD, LOGO, MIXED, CONTROL]


# --- the acceptance test: no leakage on any kind of plan ------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_no_validation_or_test_point_is_ever_in_a_fit(kind, demo_plans, table, a1_demo_settings, config):
    built = demo_plans[kind]
    _, log = _run_with_spy(table, built, a1_demo_settings, config, f"held-{kind}")
    held_out = set(built.plan.point_ids_for(TEST)) | set(built.plan.point_ids_for(VAL))

    assert log["fits"], "nothing was fitted"
    for target, fitted in log["fits"].values():
        assert not fitted & held_out, f"a {target} fit saw validation or test points"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_no_training_point_is_predicted_by_a_fit_that_contained_it(kind, demo_plans, table, a1_demo_settings, config):
    built = demo_plans[kind]
    _, log = _run_with_spy(table, built, a1_demo_settings, config, f"oof-{kind}")
    train = set(built.plan.point_ids_for(TRAIN))

    for fit_number, target, predicted in log["predictions"]:
        fitted = log["fits"][fit_number][1]
        assert not (predicted & fitted), f"a {target} fit predicted points it was fitted on"
        # ...and the training points it predicted are exactly the ones it left out.
        assert predicted & train <= train - fitted


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_point_is_predicted_exactly_once_per_property(kind, demo_plans, table, a1_demo_settings, config):
    built = demo_plans[kind]
    result, log = _run_with_spy(table, built, a1_demo_settings, config, f"once-{kind}")
    plan_ids = set(built.plan.assignments.index)

    for target in a1_demo_settings.auxiliary.targets:
        predicted = [ids for _, fit_target, ids in log["predictions"] if fit_target == target]
        flat = [point for ids in predicted for point in ids]
        assert len(flat) == len(set(flat))
        assert set(flat) == plan_ids
        assert result.predictions[stage1_prediction_column(target)].notna().all()


# --- the folds ----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [SPATIAL_KFOLD, LOGO, MIXED])
def test_grouped_plans_keep_every_block_or_region_in_one_fold(kind, demo_plans, a1_demo_settings):
    built = demo_plans[kind]
    folds, cv_kind = inner_folds(built.plan, a1_demo_settings, kind=built.kind)
    clusters = built.plan.clusters.reindex(folds.index)

    assert cv_kind == GROUP_KFOLD
    assert (pd.DataFrame({"cluster": clusters, "fold": folds}).groupby("cluster")["fold"].nunique() == 1).all()


@pytest.mark.parametrize("kind", [RANDOM, CONTROL])
def test_random_and_control_plans_get_balanced_random_folds(kind, demo_plans, a1_demo_settings):
    built = demo_plans[kind]
    folds, cv_kind = inner_folds(built.plan, a1_demo_settings, kind=built.kind)
    sizes = folds.value_counts()

    assert cv_kind == KFOLD
    assert len(sizes) == a1_demo_settings.inner_cv_folds
    assert sizes.max() - sizes.min() <= 1
    # Shuffled: the folds are not consecutive runs of the plan's order.
    assert not folds.is_monotonic_increasing


def test_the_fold_assignment_is_the_one_the_fits_used(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[LOGO]
    result, log = _run_with_spy(table, built, a1_demo_settings, config, "audit")
    train = built.plan.point_ids_for(TRAIN)

    for fit_number, target, predicted in log["predictions"]:
        fitted = log["fits"][fit_number][1]
        if fitted == frozenset(train):
            continue  # the full training fit
        predicted_folds = set(result.folds.loc[list(predicted)])
        assert len(predicted_folds) == 1
        fold = predicted_folds.pop()
        assert set(result.folds.loc[list(fitted)]) == set(range(result.n_folds)) - {fold}
        assert set(result.predicted_by.loc[list(predicted), target]) == {f"oof_fold_{fold}"}


def test_validation_and_test_points_come_from_the_full_training_fit(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[MIXED]
    result, _ = _run_with_spy(table, built, a1_demo_settings, config, "full")
    held = built.plan.assignments.isin([VAL, TEST])

    for target in a1_demo_settings.auxiliary.targets:
        assert set(result.predicted_by.loc[held.to_numpy(), target]) == {FULL_TRAIN}
    assert set(result.folds[held.to_numpy()]) == {NOT_IN_A_FOLD}


def test_with_fewer_groups_than_folds_the_folds_are_reduced(demo_plans, a1_demo_settings, caplog):
    built = demo_plans[LOGO]  # 4 regions: one test, the other three to train on (no validation region)
    with caplog.at_level(logging.WARNING):
        folds, _ = inner_folds(built.plan, a1_demo_settings, kind=LOGO)
    assert folds.nunique() == 3
    assert "using 3 fold(s)" in caplog.text


def test_a_plan_read_from_a_file_is_grouped_by_its_clusters(demo_plans, a1_demo_settings):
    reread = SplitPlan.from_frame(demo_plans[SPATIAL_KFOLD].plan.to_frame())
    _, cv_kind = inner_folds(reread, a1_demo_settings)
    assert cv_kind == GROUP_KFOLD


# --- gaps and small training sets ------------------------------------------------------------------


def test_a_training_point_without_a_measured_value_is_never_fitted_on_but_still_predicted(
    demo_plans, table, a1_demo_settings, config
):
    built = demo_plans[RANDOM]
    gappy = replace(table, frame=table.frame.copy())
    unmeasured = list(built.plan.point_ids_for(TRAIN)[:7])
    gappy.frame.loc[unmeasured, "caco3_pct_total"] = np.nan

    result, log = _run_with_spy(gappy, built, a1_demo_settings, config, "gaps")

    for target, fitted in log["fits"].values():
        if target == "caco3_pct_total":
            assert not fitted & set(unmeasured)
    assert set(result.predicted_by.loc[unmeasured, "caco3_pct_total"]) == {FULL_TRAIN}
    assert result.predictions.loc[unmeasured, "a1_pred_caco3_pct_total"].notna().all()


def test_a_control_with_one_moved_point_is_left_unpredicted_rather_than_leaked(
    demo_plans, table, a1_demo_settings, config, caplog
):
    built = demo_plans[CONTROL]
    assignments = built.plan.assignments.copy()
    moved = assignments.index[assignments == TRAIN]
    assignments.loc[moved[1:]] = TEST
    tiny = replace(built, plan=replace(built.plan, assignments=assignments))

    with caplog.at_level(logging.WARNING):
        result, log = _run_with_spy(table, tiny, a1_demo_settings, config, "tiny")

    assert set(result.predicted_by.loc[[moved[0]], "caco3_pct_total"]) == {NOT_PREDICTED}
    assert result.predictions.loc[moved[0]].isna().all()
    assert "cannot be cut into folds" in caplog.text
    # The test points are still predicted, by the one-point fit, which never saw them.
    for target, fitted in log["fits"].values():
        assert fitted == {moved[0]}


def test_a_plan_with_points_the_table_lacks_is_refused(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[RANDOM]
    stray = pd.concat([built.plan.assignments, pd.Series([TEST], index=["nowhere"])])
    with pytest.raises(KeyError, match="not in the A1 table"):
        predict_auxiliary(
            table, replace(built.plan, assignments=stray), a1_demo_settings, config, estimator_factory=_spy("x")
        )


def test_the_audit_table_has_a_row_per_point(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[SPATIAL_KFOLD]
    result, _ = _run_with_spy(table, built, a1_demo_settings, config, "frame")
    frame = result.to_frame()

    assert len(frame) == len(built.plan.assignments)
    assert {"split", "inner_fold", "a1_pred_caco3_pct_total", "predicted_by_c_e_c_meq_100g"} <= set(frame.columns)


# --- the real pipeline -----------------------------------------------------------------------------


def _with_stage1(settings, **stage):
    return replace(settings, stage1=replace(settings.stage1, **stage))


def test_the_xgboost_pipeline_uses_the_repositorys_preparation(a1_demo_settings, config, table):
    pipeline = build_stage1_pipeline(
        _with_stage1(a1_demo_settings, estimator="XGBoost"), config, table.feature_columns, table.categorical_columns
    )
    assert isinstance(pipeline, Pipeline)
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    assert type(model).__name__ == "XGBRegressor"
    numeric = dict((name, steps) for name, steps, _ in preprocessor.transformers)["num"]
    assert "scaler" not in numeric.named_steps  # trees are not scaled
    assert numeric.named_steps["imputer"].add_indicator
    # The entry's single-valued grid is applied.
    assert model.get_params()["n_estimators"] == 100
    assert model.get_params()["max_depth"] == 6


def test_the_tabicl_pipeline_is_built_from_its_entry(a1_demo_settings, config, table):
    pipeline = build_stage1_pipeline(a1_demo_settings, config, table.feature_columns, table.categorical_columns)
    model = pipeline.named_steps["model"]
    assert type(model).__name__ == "TabICLRegressor"
    assert model.get_params()["offload_mode"] == "disk"  # from the model list's init_args


def test_import_path_and_init_args_in_the_settings_override_the_entry(a1_demo_settings, config, table):
    settings = _with_stage1(
        a1_demo_settings, estimator="XGBoost", import_path="xgboost.XGBRegressor", init_args={"max_leaves": 7}
    )
    model = build_stage1_pipeline(settings, config, table.feature_columns).named_steps["model"]
    assert model.get_params()["max_leaves"] == 7
    assert model.get_params()["tree_method"] == "hist"  # the entry's other init_args are kept


def test_a_grid_with_several_values_is_refused(a1_demo_settings, config, table):
    registry = dict(config.MODEL_REGISTRY)
    registry["XGBoost"] = {**registry["XGBoost"], "params": {"model__max_depth": [4, 6]}}
    widened = type("Config", (), {"MODEL_REGISTRY": registry})()
    with pytest.raises(ValueError, match="several values"):
        build_stage1_pipeline(_with_stage1(a1_demo_settings, estimator="XGBoost"), widened, table.feature_columns)


def test_real_xgboost_predictions_are_finite_and_follow_the_measurements(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[LOGO]
    result = predict_auxiliary(
        table, built.plan, _with_stage1(a1_demo_settings, estimator="XGBoost"), config, kind=LOGO
    )
    test = built.plan.point_ids_for(TEST)

    assert np.isfinite(result.predictions.to_numpy()).all()
    for target in a1_demo_settings.auxiliary.targets:
        rho = result.predictions.loc[test, stage1_prediction_column(target)].corr(
            table.frame.loc[test, target], method="spearman"
        )
        assert rho > 0.4


@pytest.mark.slow
def test_real_tabicl_predicts_without_leaking(demo_plans, table, a1_demo_settings, config):
    built = demo_plans[RANDOM]
    result = predict_auxiliary(table, built.plan, a1_demo_settings, config, kind=RANDOM)
    assert np.isfinite(result.predictions.to_numpy()).all()
    assert result.n_folds == a1_demo_settings.inner_cv_folds
