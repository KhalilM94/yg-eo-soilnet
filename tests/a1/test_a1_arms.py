"""The two extra arms of the mixed experiment: weighted moved points, and an in-basin indicator.

An arm must change only how a mixed plan is trained - never its points - and the plain ``mixed``
and ``control`` plans must not move when the arms are switched on, or the arms could not be
compared with an earlier plain run plan by plan.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from experiments.a1.check_plans import main as check_plans_main
from yg_eo_soilnet.a1 import runner as runner_module
from yg_eo_soilnet.a1 import stage2 as stage2_module
from yg_eo_soilnet.a1.arms import (
    INDICATOR_COLUMN,
    held_out_region,
    indicator,
    sample_weights,
)
from yg_eo_soilnet.a1.auxiliary import predict_auxiliary
from yg_eo_soilnet.a1.config import COMBINED_GROUP, load_a1_config
from yg_eo_soilnet.a1.evaluation import load_predictions, mixed_differences, pooled_rows
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.plans import (
    ARM_KINDS,
    CONTROL,
    MIXED,
    MIXED_INDICATOR,
    MIXED_WEIGHTED,
    PlanError,
    build_plans,
    compare_plan_folders,
    mixed_plans,
    plain_mixed_id,
    plan_population,
    validate_arm,
    write_plans,
)
from yg_eo_soilnet.a1.runner import run_plans
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.a1.stage2 import fit_tuned
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"

BOTH_ARMS = ["weighted", "indicator"]


def _with_arms(settings, arms=BOTH_ARMS, local_weight="balanced"):
    mixed = replace(settings.plans.mixed, arms=tuple(arms), local_weight=local_weight)
    return replace(settings, plans=replace(settings.plans, mixed=mixed))


@pytest.fixture(scope="module")
def population(a1_demo_settings):
    return plan_population(a1_demo_settings)


@pytest.fixture(scope="module")
def arm_plans(population, a1_demo_settings):
    built = mixed_plans(population, _with_arms(a1_demo_settings))
    return {plan.plan_id: plan for plan in built}


# --- the settings -----------------------------------------------------------------------------


def test_the_arms_are_off_and_balanced_by_default(a1_settings):
    mixed = a1_settings().plans.mixed
    assert mixed.arms == () and mixed.local_weight == "balanced"


@pytest.mark.parametrize(
    "values, message",
    [
        ({"arms": ["weighted", "boosted"]}, "unknown arm"),
        ({"arms": ["weighted", "weighted"]}, "repeats an arm"),
        ({"local_weight": "double"}, "local_weight"),
        ({"local_weight": 0}, "positive"),
        ({"local_weight": True}, "local_weight"),
    ],
)
def test_bad_arm_settings_are_refused(a1_settings, values, message):
    raw = {"plans": {"mixed": {"fractions": [0.5], "n_repeats": 1, **values}}}
    with pytest.raises(ValueError, match=message):
        a1_settings(**raw)


def test_a_number_sets_the_local_weight(a1_settings):
    settings = a1_settings(plans={"mixed": {"fractions": [0.5], "n_repeats": 1, "local_weight": 4}})
    assert settings.plans.mixed.local_weight == 4.0


@pytest.mark.parametrize("name, held_out", [("a1_mixed_arms.yml", 2), ("a1_mixed_contrast.yml", 3)])
def test_the_arms_configurations_load(name, held_out):
    settings = load_settings(str(A1_CONFIGS / name))
    assert settings.plans.mixed.arms == ("weighted", "indicator")
    assert settings.plans.mixed.local_weight == "balanced"
    assert settings.stage2.models == ("global",)
    assert len(settings.plans.logo.held_out) == held_out


def test_the_arms_configuration_draws_like_the_plain_mixed_run():
    """Same seed, regions, shares and repeats as a1_mixed.yml (when it is present): the same plans."""
    plain_path = A1_CONFIGS / "a1_mixed.yml"
    if not plain_path.exists():
        pytest.skip("configs/a1/a1_mixed.yml is not in this checkout")
    arms, plain = load_settings(str(A1_CONFIGS / "a1_mixed_arms.yml")), load_settings(str(plain_path))
    assert arms.seed == plain.seed
    assert arms.plans.logo.held_out == plain.plans.logo.held_out
    assert arms.plans.mixed.fractions == plain.plans.mixed.fractions
    assert arms.plans.mixed.n_repeats == plain.plans.mixed.n_repeats


# --- the plans -----------------------------------------------------------------------------------


def test_every_mixed_plan_gets_one_plan_per_arm_named_after_it(arm_plans):
    plain = [plan_id for plan_id, built in arm_plans.items() if built.kind == MIXED]
    for plan_id in plain:
        rest = plan_id[len("mixed_") :]
        assert arm_plans[f"mixed_w_{rest}"].kind == MIXED_WEIGHTED
        assert arm_plans[f"mixed_i_{rest}"].kind == MIXED_INDICATOR
    assert sum(built.kind in ARM_KINDS for built in arm_plans.values()) == 2 * len(plain)
    assert plain_mixed_id("mixed_i_r01_x_p10_s0") == "mixed_r01_x_p10_s0"


def test_an_arm_shares_the_moved_and_test_points_of_its_plain_plan(arm_plans):
    for plan_id, built in arm_plans.items():
        if built.kind not in ARM_KINDS:
            continue
        plain = arm_plans[plain_mixed_id(plan_id)]
        assert built.moved == plain.moved
        assert set(built.plan.point_ids_for(TEST)) == set(plain.plan.point_ids_for(TEST))
        assert set(built.plan.point_ids_for(TRAIN)) == set(plain.plan.point_ids_for(TRAIN))
        assert built.plan.strategy == built.kind


def test_an_arm_with_other_points_is_refused(arm_plans):
    plan_id = next(plan_id for plan_id, built in arm_plans.items() if built.kind == MIXED_WEIGHTED)
    arm, plain = arm_plans[plan_id], arm_plans[plain_mixed_id(plan_id)]
    assignments = arm.plan.assignments.copy()
    test_point = assignments.index[assignments.to_numpy() == TEST][0]
    assignments.loc[test_point] = TRAIN
    broken = replace(arm, plan=replace(arm.plan, assignments=assignments))
    with pytest.raises(PlanError, match="not exactly those"):
        validate_arm(plain, broken)


def test_switching_the_arms_on_leaves_the_plain_plans_unchanged(population, a1_demo_settings):
    without, _ = build_plans(population, a1_demo_settings, kinds=["logo", "mixed"])
    with_arms, _ = build_plans(population, _with_arms(a1_demo_settings), kinds=["logo", "mixed"])
    arms = {built.plan_id: built for built in with_arms}
    assert {built.kind for built in without} == {"logo", MIXED, CONTROL}
    for built in without:
        assert arms[built.plan_id].plan.assignments.equals(built.plan.assignments)
        assert arms[built.plan_id].plan.to_frame().equals(built.plan.to_frame())


@pytest.fixture(scope="module")
def two_folders(population, a1_demo_settings, tmp_path_factory):
    folder = tmp_path_factory.mktemp("plan_folders")
    write_plans(a1_demo_settings, str(folder / "plain"), kinds=["logo", "mixed"], population=population)
    write_plans(_with_arms(a1_demo_settings), str(folder / "arms"), kinds=["logo", "mixed"], population=population)
    return folder / "plain", folder / "arms"


def test_check_plans_passes_between_a_configuration_with_and_without_arms(two_folders, capsys):
    plain, arms = two_folders
    table, only_plain, only_arms = compare_plan_folders(plain, arms)
    assert table["same_file"].all() and table["same_test_points"].all()
    assert not only_plain and len(only_arms) == len(table) - (table["kind"] == "logo").sum()
    assert {plan_id[:8] for plan_id in only_arms} == {"mixed_w_", "mixed_i_"}

    assert check_plans_main([str(plain), str(arms)]) == 0
    assert f"{len(table)} plan(s) in both folders: {len(table)} the same, 0 different" in capsys.readouterr().out


def test_check_plans_fails_on_a_plan_with_other_test_points(two_folders, tmp_path, capsys):
    import shutil

    plain, arms = two_folders
    changed = tmp_path / "changed"
    shutil.copytree(plain, changed)
    plan_id = "control_" + sorted(path.stem for path in plain.glob("mixed_*.parquet"))[0][len("mixed_") :]
    frame = pd.read_parquet(changed / f"{plan_id}.parquet")
    frame.loc[frame.index[frame["split"] == TEST][0], "split"] = TRAIN
    frame.to_parquet(changed / f"{plan_id}.parquet", index=False)

    table, _, _ = compare_plan_folders(plain, changed)
    row = table.set_index("plan_id").loc[plan_id]
    assert not row["same_file"] and not row["same_test_points"] and row["n_test_b"] == row["n_test_a"] - 1
    assert check_plans_main([str(arms), str(changed)]) == 1
    assert f"DIFFERENT: {plan_id}" in capsys.readouterr().out


def test_check_plans_fails_when_the_folders_share_no_plan(two_folders, tmp_path, a1_demo_settings, population):
    other = tmp_path / "random_only"
    write_plans(a1_demo_settings, str(other), kinds=["random"], population=population)
    assert check_plans_main([str(two_folders[0]), str(other)]) == 1


# --- the weights ---------------------------------------------------------------------------------


def _weighted(arm_plans):
    return next(built for built in arm_plans.values() if built.kind == MIXED_WEIGHTED and built.fraction == 0.2)


def test_balanced_weights_give_the_moved_points_the_other_regions_total_weight(arm_plans, a1_demo_settings):
    built = _weighted(arm_plans)
    train = built.plan.point_ids_for(TRAIN)
    weights, w = sample_weights(built.plan, _with_arms(a1_demo_settings), train)
    moved = weights.index.isin(list(built.moved))
    assert moved.sum() == len(built.moved) and (~moved).sum() == len(train) - len(built.moved)
    assert w == pytest.approx((~moved).sum() / moved.sum())
    assert weights[moved].sum() == pytest.approx(weights[~moved].sum())
    assert (weights[~moved] == 1.0).all()


def test_a_number_weighs_each_moved_point_by_it(arm_plans, a1_demo_settings):
    built = _weighted(arm_plans)
    train = built.plan.point_ids_for(TRAIN)
    weights, w = sample_weights(built.plan, _with_arms(a1_demo_settings, local_weight=3.0), train)
    moved = weights.index.isin(list(built.moved))
    assert w == 3.0 and (weights[moved] == 3.0).all() and weights[moved].sum() == 3.0 * len(built.moved)


def test_the_held_out_region_is_the_one_the_test_points_are_in(arm_plans):
    built = _weighted(arm_plans)
    assert held_out_region(built.plan) == built.held_out_cluster


class _Recorder(BaseEstimator, RegressorMixin):
    """Predicts the (weighted) mean of its training targets and records every fit's weights."""

    fits: list = []

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y, sample_weight=None):
        type(self).fits.append((list(X.index), None if sample_weight is None else np.asarray(sample_weight)))
        self.mean_ = float(np.average(np.asarray(y), weights=sample_weight)) + self.shift
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_the_weights_reach_every_fit_of_the_search_and_the_final_fit(monkeypatch):
    rng = np.random.default_rng(0)
    ids = pd.Index([f"p{i}" for i in range(40)])
    X = pd.DataFrame({"x": rng.normal(size=40)}, index=ids)
    y = pd.Series(rng.normal(10, 3, 40), index=ids)
    folds = pd.Series(np.arange(40) % 2, index=ids)
    weights = pd.Series(np.where(np.arange(40) < 10, 5.0, 1.0), index=ids)
    searches = []

    class Recording(stage2_module.GridSearchCV):
        def fit(self, *args, **kwargs):
            searches.append(self)
            return super().fit(*args, **kwargs)

    monkeypatch.setattr(stage2_module, "GridSearchCV", Recording)
    _Recorder.fits = []
    pipeline = Pipeline([("model", _Recorder())])
    fit_tuned(pipeline, {"model__shift": [0.0, 1.0]}, X, y, folds, sample_weight=weights)

    assert len(_Recorder.fits) == 2 * 2 + 1  # two settings x two folds, then the refit
    for rows, given in _Recorder.fits:
        assert given is not None and np.array_equal(given, weights.loc[rows].to_numpy())

    # The search scored each validation fold by its plain RMSE, unweighted.
    search = searches[0]
    for fold in (0, 1):
        held, kept = ids[folds == fold], ids[folds != fold]
        mean = np.average(y.loc[kept], weights=weights.loc[kept])
        expected = -np.sqrt(np.mean((y.loc[held] - mean) ** 2))
        assert search.cv_results_[f"split{fold}_test_score"][0] == pytest.approx(expected)


def test_without_weights_no_fit_is_weighted():
    ids = pd.Index([f"p{i}" for i in range(20)])
    X = pd.DataFrame({"x": np.arange(20.0)}, index=ids)
    _Recorder.fits = []
    fit_tuned(
        Pipeline([("model", _Recorder())]),
        {"model__shift": [0.0, 1.0]},
        X,
        X["x"],
        pd.Series(np.arange(20) % 2, index=ids),
    )
    assert all(given is None for _, given in _Recorder.fits)


def test_stage_1_is_weighted_the_same_way(arm_plans, a1_demo_settings):
    settings = _with_arms(a1_demo_settings)
    config = settings.load_config()
    table = build_table(settings, config)
    built = _weighted(arm_plans)
    weights, w = sample_weights(built.plan, settings, built.plan.point_ids_for(TRAIN))
    _Recorder.fits = []
    predict_auxiliary(
        table,
        built.plan,
        settings,
        config,
        kind=MIXED_WEIGHTED,
        estimator_factory=lambda target: Pipeline([("model", _Recorder())]),
        sample_weight=weights,
    )
    assert _Recorder.fits
    for rows, given in _Recorder.fits:
        expected = np.where(pd.Index(rows).isin(list(built.moved)), w, 1.0)
        assert np.array_equal(given, expected)


# --- the indicator --------------------------------------------------------------------------------


def test_the_indicator_is_one_exactly_on_the_held_out_regions_points(arm_plans, population):
    built = next(built for built in arm_plans.values() if built.kind == MIXED_INDICATOR)
    values = indicator(built.plan)
    region = population.region_ids.reindex(values.index) == built.held_out_cluster
    assert values.name == INDICATOR_COLUMN
    assert set(values.unique()) == {0.0, 1.0}
    assert (values[region.to_numpy()] == 1.0).all() and (values[~region.to_numpy()] == 0.0).all()
    # Moved and test points alike.
    assert (values.loc[list(built.moved)] == 1.0).all() and (values.loc[built.plan.point_ids_for(TEST)] == 1.0).all()


# --- a run of the arms ---------------------------------------------------------------------------

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}, "params": {"model__max_depth": [2, 3]}},
    "inner_cv_folds": 2,
}


@pytest.fixture(scope="module")
def arms_run(a1_settings_in, tmp_path_factory, mlflow_store):
    """A fast run of one share and repeat of one region: mixed, control and both arms, with the groups."""
    folder = tmp_path_factory.mktemp("a1_arms_run")
    seen: list = []
    original = stage2_module.fit_tuned

    def recording(pipeline, grid, X, y, folds, sample_weight=None):
        seen.append((list(X.columns), sample_weight))
        return original(pipeline, grid, X, y, folds, sample_weight=sample_weight)

    with mlflow_store(folder / "mlruns") as uri, pytest.MonkeyPatch.context() as patch:
        patch.setattr(stage2_module, "fit_tuned", recording)
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            as_file=True,
            output_dir=str(folder / "out"),
            plans={
                "logo": {"min_points": 100, "held_out": ["North-East"], "validation_region": False},
                "mixed": {"fractions": [0.3], "n_repeats": 1, "arms": BOTH_ARMS},
            },
            **FAST,
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=["mixed"])
        calls = {}

        def per_kind(row, seed, shared, *, logger):
            seen.clear()
            outcome = original_run(row, seed, shared, logger=logger)
            calls[row.kind] = list(seen)
            return outcome

        original_run = runner_module._run_one_plan
        patch.setattr(runner_module, "_run_one_plan", per_kind)
        result = run_plans(a1, groups=True, kinds=[MIXED, CONTROL, *ARM_KINDS])
        yield {"a1": a1, "result": result, "calls": calls}


def test_each_arm_is_run_and_its_rows_say_which(arms_run):
    frame = load_predictions(arms_run["result"].run_dir)
    assert set(frame["kind"]) == {MIXED, CONTROL, MIXED_WEIGHTED, MIXED_INDICATOR}
    test = frame[frame["split"] == TEST]
    points = {kind: set(part["point_id"]) for kind, part in test.groupby("kind")}
    assert points[MIXED] == points[MIXED_WEIGHTED] == points[MIXED_INDICATOR] == points[CONTROL]


def test_the_weight_is_recorded_for_the_weighted_arm_only(arms_run):
    frame = load_predictions(arms_run["result"].run_dir)
    weighted = frame[frame["kind"] == MIXED_WEIGHTED]["local_weight"]
    assert weighted.notna().all() and weighted.nunique() == 1 and weighted.iloc[0] > 1.0
    assert frame[frame["kind"] != MIXED_WEIGHTED]["local_weight"].isna().all()


def test_only_the_weighted_arms_fits_are_weighted(arms_run):
    for kind, calls in arms_run["calls"].items():
        weighted = [weights is not None for _, weights in calls]
        assert all(weighted) if kind == MIXED_WEIGHTED else not any(weighted), kind


def test_the_indicator_is_read_by_the_combined_models_of_its_arm_only(arms_run):
    groups = arms_run["a1"].resolve_groups(build_table(arms_run["a1"].settings, arms_run["a1"].settings.load_config()))
    group_columns = {tuple(sorted(columns)) for columns in groups.groups.values()}
    for kind, calls in arms_run["calls"].items():
        for columns, _ in calls:
            reads = INDICATOR_COLUMN in columns
            if kind == MIXED_INDICATOR and tuple(sorted(columns)) not in group_columns:
                assert reads, "a model on every covariate of the indicator arm must read it"
            else:
                assert not reads, f"{kind} read the indicator with {len(columns)} input(s)"
    indicator_calls = arms_run["calls"][MIXED_INDICATOR]
    assert any(tuple(sorted(columns)) in group_columns for columns, _ in indicator_calls), "no group fit was checked"


def test_the_arms_are_never_pooled_and_each_is_compared_with_control(arms_run):
    frame = load_predictions(arms_run["result"].run_dir)
    assert pooled_rows(frame).empty
    differences = mixed_differences(frame, n=50)
    assert set(differences["kind"]) == {f"{kind}_vs_control" for kind in (MIXED, *ARM_KINDS)}
    assert set(differences["a"]) == {MIXED, *ARM_KINDS} and set(differences["b"]) == {CONTROL}


def test_holm_is_applied_within_each_region_model_score_and_comparison(monkeypatch):
    """Two shares, two comparisons: Holm runs over the shares of one comparison, not across comparisons."""
    from yg_eo_soilnet.a1 import evaluation

    p_values = {(MIXED, 0.1): 0.01, (MIXED, 0.2): 0.04, (MIXED_WEIGHTED, 0.1): 0.03, (MIXED_WEIGHTED, 0.2): 0.02}
    rows = [
        {
            "held_out": "A",
            "fraction": fraction,
            "model": "global",
            "a": kind,
            "b": CONTROL,
            "metric": "rmse",
            "estimate": 0.1,
            "low": 0.0,
            "high": 0.2,
            "p_value": p,
            "n": 50,
            "n_blocks": 9,
            "n_valid_draws": 50,
            "n_unpaired": 0,
            "small": False,
        }
        for (kind, fraction), p in p_values.items()
    ]

    class Result:
        differences = pd.DataFrame(rows)

    monkeypatch.setattr(evaluation, "block_bootstrap", lambda *args, **kwargs: Result())
    frame = pd.DataFrame(
        {
            "split": TEST,
            "kind": [MIXED, MIXED_WEIGHTED, CONTROL],
            "covariate_group": COMBINED_GROUP,
            "fraction": 0.1,
            "held_out": "A",
            "repeat": 0.0,
        }
    )
    out = evaluation.mixed_differences(frame).set_index(["a", "fraction"])["p_holm"]
    # Per comparison, the smaller p doubled, the larger kept (Holm over two tests).
    assert out[(MIXED, 0.1)] == pytest.approx(0.02) and out[(MIXED, 0.2)] == pytest.approx(0.04)
    assert out[(MIXED_WEIGHTED, 0.2)] == pytest.approx(0.04) and out[(MIXED_WEIGHTED, 0.1)] == pytest.approx(0.04)
