"""Leave one region out with the training set cut to the same size for every held-out region.

Holding out a large region also takes much of the training data away, so a matched plan must keep
the test set of the plain plan, learn from about ``plans.logo.training_size`` points drawn as whole
blocks, never from the held-out region, and draw the same blocks whenever it is rebuilt - while the
plain ``logo`` plans stay exactly as they were when the setting is off.
"""

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from experiments.a1.check_plans import main as check_plans_main
from experiments.a1.evaluate import main as evaluate_main
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.evaluation import load_predictions
from yg_eo_soilnet.a1.plans import (
    INDEX_COLUMNS,
    INDEX_FILE,
    LOGO,
    MATCHED_INDEX_COLUMNS,
    logo_plans,
    plan_population,
    training_blocks,
    write_plans,
)
from yg_eo_soilnet.a1.runner import LOCAL, run_plans
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"

#: About half the training points the demo leaves when one of its regions is held out.
SIZE = 150
REPEATS = 3


def _matched(settings, training_size=SIZE, n_repeats=REPEATS):
    logo = replace(settings.plans.logo, training_size=training_size, n_repeats=n_repeats)
    return replace(settings, plans=replace(settings.plans, logo=logo))


@pytest.fixture(scope="module")
def population(a1_demo_settings):
    return plan_population(a1_demo_settings)


@pytest.fixture(scope="module")
def matched(population, a1_demo_settings):
    return logo_plans(population, _matched(a1_demo_settings))


@pytest.fixture(scope="module")
def plain(population, a1_demo_settings):
    return {built.held_out: built for built in logo_plans(population, a1_demo_settings)}


# --- the settings -----------------------------------------------------------------------------


def test_the_training_set_is_not_cut_by_default(a1_settings):
    logo = a1_settings().plans.logo
    assert logo.training_size is None and logo.n_repeats == 1


@pytest.mark.parametrize(
    "values, message",
    [
        ({"training_size": 0}, "at least 1"),
        ({"training_size": 12.5}, "whole number"),
        ({"training_size": True}, "whole number"),
        ({"training_size": 100, "n_repeats": 0}, "at least 1"),
        ({"n_repeats": 3}, "not set"),
    ],
)
def test_bad_matched_settings_are_refused(a1_settings, values, message):
    with pytest.raises(ValueError, match=message):
        a1_settings(plans={"logo": {"min_points": 100, **values}})


def test_the_matched_configuration_loads():
    settings = load_settings(str(A1_CONFIGS / "a1_logo_matched.yml"))
    logo = settings.plans.logo
    assert (logo.training_size, logo.n_repeats) == (3600, 5)
    assert logo.held_out == ("Oum Er Rbia", "Sebou", "Tensift", "Loukkous", "Bouregreg")
    assert settings.output_dir == "results/a1_logo_matched"
    assert settings.bootstrap.block_size_km == 10.0


def test_the_mixed_configuration_loads():
    settings = load_settings(str(A1_CONFIGS / "a1_mixed.yml"))
    assert settings.output_dir == "results/a1_mixed"
    assert settings.plans.logo.training_size is None


# --- the plans --------------------------------------------------------------------------------


def test_each_held_out_region_gets_n_repeats_plans_named_by_size_region_and_repeat(matched, plain):
    ids = [built.plan_id for built in matched]
    assert len(ids) == len(plain) * REPEATS
    for built in matched:
        assert built.kind == LOGO
        tag = plain[built.held_out].plan_id[len("logo_") :]
        assert built.plan_id == f"logo_m{SIZE}_{tag}_s{built.repeat}"
    assert sorted({built.repeat for built in matched}) == list(range(REPEATS))


def test_every_training_set_is_within_one_block_of_the_size_asked(matched, population, a1_demo_settings):
    blocks = training_blocks(population, a1_demo_settings)
    for built in matched:
        train = built.plan.point_ids_for(TRAIN)
        largest = blocks.reindex(train).value_counts().max()
        assert abs(len(train) - SIZE) <= largest, built.plan_id
        assert built.training_size == SIZE


def test_only_whole_blocks_are_drawn(matched, plain, population, a1_demo_settings):
    blocks = training_blocks(population, a1_demo_settings)
    for built in matched:
        available = plain[built.held_out].plan.point_ids_for(TRAIN)
        drawn = built.plan.point_ids_for(TRAIN)
        drawn_blocks = set(blocks.reindex(drawn))
        assert set(available[blocks.reindex(available).isin(drawn_blocks).to_numpy()]) == set(drawn), built.plan_id


def test_the_held_out_region_is_never_trained_on_and_is_the_whole_test_set(matched, plain, population):
    for built in matched:
        region = set(population.region_points(built.held_out_cluster))
        assert not region & set(built.plan.point_ids_for(TRAIN)), built.plan_id
        assert set(built.plan.point_ids_for(TEST)) == region
        assert set(built.plan.point_ids_for(TEST)) == set(plain[built.held_out].plan.point_ids_for(TEST))


def test_the_points_not_drawn_are_left_out_not_tested(matched, plain):
    for built in matched:
        available = set(plain[built.held_out].plan.point_ids_for(TRAIN))
        drawn = set(built.plan.point_ids_for(TRAIN))
        assert drawn <= available
        assert set(built.plan.assignments.index) == drawn | set(built.plan.point_ids_for(TEST))
        assert built.n_left_out == len(available - drawn) > 0


def test_draws_differ_between_repeats(matched):
    by_region: dict = {}
    for built in matched:
        by_region.setdefault(built.held_out, []).append(frozenset(built.plan.point_ids_for(TRAIN)))
    for region, draws in by_region.items():
        assert len(set(draws)) == REPEATS, region


def test_draws_are_reproducible(matched, population, a1_demo_settings):
    again = logo_plans(population, _matched(a1_demo_settings))
    for first, second in zip(matched, again):
        assert first.plan_id == second.plan_id
        assert first.plan.assignments.sort_index().equals(second.plan.assignments.sort_index())


def test_a_region_with_too_few_training_points_keeps_them_all_and_says_so(population, plain, a1_demo_settings, caplog):
    with caplog.at_level(logging.WARNING, logger="yg_eo_soilnet.a1.plans"):
        built = logo_plans(population, _matched(a1_demo_settings, training_size=100_000, n_repeats=2))
    for plan in built:
        assert set(plan.plan.point_ids_for(TRAIN)) == set(plain[plan.held_out].plan.point_ids_for(TRAIN))
        assert plan.n_left_out == 0
    assert "use them all" in caplog.text


def test_each_plans_size_is_logged(population, a1_demo_settings, caplog):
    with caplog.at_level(logging.INFO, logger="yg_eo_soilnet.a1.plans"):
        built = logo_plans(population, _matched(a1_demo_settings))
    for plan in built:
        assert f"{plan.plan_id}: {len(plan.plan.point_ids_for(TRAIN))} training point(s) (asked {SIZE})" in caplog.text


# --- the folder ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def folders(population, a1_demo_settings, tmp_path_factory):
    root = tmp_path_factory.mktemp("logo_matched_plans")
    write_plans(a1_demo_settings, str(root / "plain"), kinds=[LOGO], population=population)
    write_plans(_matched(a1_demo_settings), str(root / "matched"), kinds=[LOGO], population=population)
    write_plans(a1_demo_settings, str(root / "plain_again"), kinds=[LOGO], population=population)
    return root


def test_the_index_records_the_size_asked_drawn_and_left_out(folders, matched):
    index = pd.read_csv(folders / "matched" / INDEX_FILE).set_index("plan_id")
    assert list(index.columns) == [*INDEX_COLUMNS[1:], *MATCHED_INDEX_COLUMNS]
    for built in matched:
        row = index.loc[built.plan_id]
        assert row["training_size"] == SIZE
        assert row["n_train"] == len(built.plan.point_ids_for(TRAIN))
        assert row["n_left_out"] == built.n_left_out
        assert row["repeat"] == built.repeat


def test_without_training_size_the_plans_and_index_are_unchanged(folders, capsys):
    index = pd.read_csv(folders / "plain" / INDEX_FILE)
    assert list(index.columns) == list(INDEX_COLUMNS)
    assert (folders / "plain" / INDEX_FILE).read_bytes() == (folders / "plain_again" / INDEX_FILE).read_bytes()
    assert check_plans_main([str(folders / "plain"), str(folders / "plain_again")]) == 0
    assert all(not path.name.startswith("logo_m") for path in (folders / "plain").glob("*.parquet"))


# --- a run --------------------------------------------------------------------------------------


FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}, "params": {"model__max_depth": [2]}},
    "inner_cv_folds": 2,
}


@pytest.fixture(scope="module")
def matched_run(a1_settings_in, tmp_path_factory, mlflow_store):
    """A fast run of two draws for one region, then its evaluation."""
    folder = tmp_path_factory.mktemp("a1_logo_matched_run")
    with mlflow_store(folder / "mlruns") as uri:
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            as_file=True,
            output_dir=str(folder / "out"),
            plans={"logo": {"min_points": 100, "held_out": ["North-East"], "training_size": SIZE, "n_repeats": 2}},
            **FAST,
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[LOGO])
        result = run_plans(a1, kinds=[LOGO], tracking=False)
        tables = evaluate_main(["--config", str(path), "--n-boot", "10", "--no-mlflow"])
        yield {"a1": a1, "result": result, "tables": tables}


def test_a_matched_run_scores_each_draw_on_the_whole_region(matched_run):
    frame = load_predictions(matched_run["result"].run_dir)
    test = frame[frame["split"] == TEST]
    ids = sorted(set(test["plan_id"]))
    assert len(ids) == 2 and all(plan_id.startswith(f"logo_m{SIZE}_") and "north_east" in plan_id for plan_id in ids)
    assert [plan_id[-3:] for plan_id in ids] == ["_s0", "_s1"]
    points = [set(part["point_id"]) for _, part in test.groupby("plan_id")]
    assert points[0] == points[1]
    assert set(test["repeat"].dropna().astype(int)) == {0, 1}


def test_the_local_model_is_the_same_for_every_draw(matched_run):
    frame = load_predictions(matched_run["result"].run_dir)
    local = frame[(frame["model"] == LOCAL) & (frame["split"] == TEST)]
    assert not local.empty
    by_plan = [part.set_index("point_id")["predicted"].sort_index() for _, part in local.groupby("plan_id")]
    pd.testing.assert_series_equal(by_plan[0], by_plan[1])


def test_a_matched_run_can_be_evaluated(matched_run):
    assert matched_run["tables"], "evaluate.py wrote no table"
