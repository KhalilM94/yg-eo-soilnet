"""The split plans the A1 experiments are scored on: each kind, the checks, and the files.

A plan that put a point in two splits, divided a block, or let the held-out region leak into
validation would make every score computed on it wrong without any error. Each of those is a
hand-broken plan here that the checks must refuse.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.a1.make_plans import main as make_plans_main
from yg_eo_soilnet.a1.plans import (
    CONTROL,
    INDEX_COLUMNS,
    INDEX_FILE,
    LOGO,
    MIXED,
    RANDOM,
    SPATIAL_KFOLD,
    PlanError,
    build_plans,
    logo_plans,
    mixed_plans,
    plan_population,
    random_plan,
    spatial_kfold_plans,
    validate_pair,
    validate_plan,
    validation_region,
    whole_blocks_for,
    write_plans,
)
from yg_eo_soilnet.a1.settings import settings_from_mapping
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL, SplitPlan

N_FRACTIONS, N_REPEATS = 9, 5


@pytest.fixture
def settings(a1_settings):
    return a1_settings()


@pytest.fixture(scope="module")
def population(a1_demo_settings):
    """The demo points, read once: 4 regions, all large enough to hold out."""
    return plan_population(a1_demo_settings)


@pytest.fixture(scope="module")
def plans(population, a1_demo_settings):
    """Every kind of plan over the demo, built once for the module."""
    return {
        RANDOM: [random_plan(population, a1_demo_settings)],
        SPATIAL_KFOLD: spatial_kfold_plans(population, a1_demo_settings),
        LOGO: logo_plans(population, a1_demo_settings),
        MIXED: mixed_plans(population, a1_demo_settings),
    }


def _synthetic_population(settings, region_sizes, *, seed=0):
    """A population with regions of the given sizes, built through plan_population."""
    rng = np.random.default_rng(seed)
    names = [name for name, size in region_sizes.items() for _ in range(size)]
    ids = pd.Index([f"p{i:04d}" for i in range(len(names))], name="uuid")
    frame = pd.DataFrame(
        {
            settings.target: rng.uniform(5, 50, len(ids)),
            "lat": rng.uniform(30, 34, len(ids)),
            "lon": rng.uniform(-9, -5, len(ids)),
            "a1_region": names,
        },
        index=ids,
    )
    points = SimpleNamespace(frame=frame, lat_column="lat", lon_column="lon", region_column="a1_region")
    return plan_population(settings, points=points)


# --- random ---------------------------------------------------------------------------------


def test_the_random_plan_is_70_15_15(plans, population):
    counts = plans[RANDOM][0].plan.counts()
    total = len(population.ids)
    assert counts["test"] == pytest.approx(0.15 * total, abs=2)
    assert counts["val"] == pytest.approx(0.15 * total, abs=2)
    assert sum(counts.values()) == total


def test_the_random_plan_is_the_same_for_the_same_seed_and_different_for_another(population, a1_settings):
    first = random_plan(population, a1_settings())
    again = random_plan(population, a1_settings())
    other = random_plan(population, a1_settings(seeds=[7]))
    assert first.plan.assignments.equals(again.plan.assignments)
    assert not first.plan.assignments.equals(other.plan.assignments)


def test_the_random_plan_is_saved_without_a_cluster_column(plans):
    assert list(plans[RANDOM][0].plan.to_frame().columns) == ["point_id", "split", "eligible_sklearn"]


# --- spatial k-fold -----------------------------------------------------------------------------


def test_every_point_is_tested_exactly_once_over_the_folds(plans, population):
    folds = plans[SPATIAL_KFOLD]
    assert [built.plan_id for built in folds] == [f"kfold_f{i}" for i in range(5)]
    tested = pd.concat([pd.Series(built.plan.point_ids_for(TEST)) for built in folds])
    assert not tested.duplicated().any()
    assert set(tested) == set(population.ids)


def test_no_block_is_divided_and_validation_is_about_15_percent(plans, population):
    for built in plans[SPATIAL_KFOLD]:
        frame = built.plan.to_frame()
        assert (frame.groupby("cluster")["split"].nunique() == 1).all()
        assert built.plan.counts()["val"] == pytest.approx(0.15 * len(population.ids), rel=0.2)


def test_the_folds_have_fifty_blocks_between_them(plans):
    assert plans[SPATIAL_KFOLD][0].plan.clusters.nunique() == 50


def test_a_grid_makes_the_blocks_instead_when_asked(population, a1_settings):
    settings = a1_settings(plans={"spatial_kfold": {"method": "grid", "cell_size_m": 50_000, "n_folds": 5}})
    folds = spatial_kfold_plans(population, settings)
    assert len(folds) == 5
    for built in folds:
        validate_plan(built)


def test_too_few_points_for_the_blocks_is_reported(a1_settings):
    settings = a1_settings()
    small = _synthetic_population(settings, {"A": 20})
    with pytest.raises(PlanError, match="at least n_clusters=50"):
        spatial_kfold_plans(small, settings)


def test_whole_blocks_are_drawn_until_they_hold_the_target():
    sizes = [5, 50, 5, 5, 30, 5, 5, 40, 5, 50]
    blocks = pd.Series(np.repeat(np.arange(10), sizes), index=[f"p{i}" for i in range(sum(sizes))])

    for seed in range(5):
        chosen = whole_blocks_for(blocks, 60, seed=seed)
        picked = set(blocks[chosen])
        # Every block drawn is taken whole...
        assert len(chosen) == sum(sizes[block] for block in picked)
        # ...the total lands near the target, and a block is always left for training.
        assert abs(len(chosen) - 60) <= 25
        assert len(picked) < len(sizes)


def test_no_block_is_drawn_for_a_target_of_zero_or_from_a_single_block():
    blocks = pd.Series([1, 1, 2], index=["a", "b", "c"])
    assert whole_blocks_for(blocks, 0, seed=0).empty
    assert whole_blocks_for(blocks[blocks == 1], 2, seed=0).empty


# --- leave one region out -------------------------------------------------------------------


def test_each_region_is_the_test_set_once_and_wholly(plans, population):
    logo = plans[LOGO]
    assert len(logo) == 4
    for built in logo:
        region = set(population.region_points(built.held_out_cluster))
        assert set(built.plan.point_ids_for(TEST)) == region
        assert built.held_out == population.region_names[built.held_out_cluster]


def test_validation_is_another_whole_region_and_the_choice_is_reproducible(plans, population, settings):
    for built in plans[LOGO]:
        val = set(built.plan.point_ids_for(VAL))
        validation = validation_region(population, built.held_out_cluster, settings.seed)
        assert validation != built.held_out_cluster
        assert val == set(population.region_points(validation))
        assert validation == validation_region(population, built.held_out_cluster, settings.seed)


def test_with_only_two_regions_there_is_no_validation_set(a1_settings):
    settings = a1_settings()
    population = _synthetic_population(settings, {"A": 40, "B": 40})
    for built in logo_plans(population, settings):
        assert built.plan.counts()["val"] == 0
        validate_plan(built)


def test_a_small_region_is_never_held_out_nor_used_for_validation(a1_settings, caplog):
    settings = a1_settings()
    with caplog.at_level("WARNING"):
        population = _synthetic_population(settings, {"A": 60, "B": 60, "C": 60, "tiny": 10})

    assert population.too_small == {"tiny": 10}
    assert "tiny (10)" in caplog.text
    tiny = set(population.region_points(population.region_ids.max()))
    logo = logo_plans(population, settings)
    assert len(logo) == 3
    for built in logo:
        assert tiny <= set(built.plan.point_ids_for(TRAIN))


def test_region_plans_name_the_region_in_plain_letters(plans):
    assert {built.plan_id for built in plans[LOGO]} == {
        "logo_r01_north_east",
        "logo_r02_north_west",
        "logo_r03_south_east",
        "logo_r04_south_west",
    }


# --- mixed transfer and its controls ------------------------------------------------------------


def test_every_region_share_and_repeat_has_a_mixed_plan_and_a_control(plans):
    kinds = pd.Series([built.kind for built in plans[MIXED]]).value_counts()
    assert kinds.to_dict() == {MIXED: 4 * N_FRACTIONS * N_REPEATS, CONTROL: 4 * N_FRACTIONS * N_REPEATS}


def _pairs(plans):
    by_id = {built.plan_id: built for built in plans[MIXED]}
    return [
        (by_id["mixed_" + plan_id[len("control_") :]], by_id[plan_id])
        for plan_id in by_id
        if plan_id.startswith("control_")
    ]


def test_the_share_moved_into_training_is_round_p_times_n(plans, population):
    for mixed, _ in _pairs(plans)[:40]:
        n = len(population.region_points(mixed.held_out_cluster))
        assert len(mixed.moved) == round(mixed.fraction * n)


def test_the_control_trains_on_only_the_moved_points_and_tests_on_the_same_points(plans):
    for mixed, control in _pairs(plans):
        assert set(control.plan.point_ids_for(TRAIN)) == set(mixed.moved)
        assert set(control.plan.point_ids_for(TEST)) == set(mixed.plan.point_ids_for(TEST))
        assert control.plan.counts()["val"] == 0


def test_a_mixed_plan_keeps_the_logo_assignment_outside_the_held_out_region(plans, population):
    logo = {built.held_out_cluster: built for built in plans[LOGO]}
    for mixed, _ in _pairs(plans)[:20]:
        outside = population.ids.difference(population.region_points(mixed.held_out_cluster))
        expected = logo[mixed.held_out_cluster].plan.assignments.reindex(outside)
        assert mixed.plan.assignments.reindex(outside).equals(expected)


def test_a_larger_share_holds_every_point_a_smaller_one_moved(plans):
    by_draw = {}
    for mixed, _ in _pairs(plans):
        by_draw.setdefault((mixed.held_out_cluster, mixed.repeat), []).append(mixed)
    for draws in by_draw.values():
        draws.sort(key=lambda built: built.fraction)
        for smaller, larger in zip(draws, draws[1:]):
            assert smaller.moved < larger.moved


def test_repeats_draw_differently_and_the_same_repeat_draws_the_same(population, settings, plans):
    first = {built.plan_id: built.moved for built in plans[MIXED]}
    again = {built.plan_id: built.moved for built in mixed_plans(population, settings)}
    assert first == again
    assert first["mixed_r01_north_east_p50_s0"] != first["mixed_r01_north_east_p50_s1"]


# --- the checks --------------------------------------------------------------------------------


def _broken(built, assignments=None, **changes):
    plan = built.plan if assignments is None else replace(built.plan, assignments=assignments)
    return replace(built, plan=plan, **changes)


def test_a_point_in_two_splits_is_refused(plans):
    built = plans[LOGO][0]
    twice = pd.concat([built.plan.assignments, pd.Series([VAL], index=[built.plan.assignments.index[0]])])
    with pytest.raises(PlanError, match="more than one split"):
        validate_plan(_broken(built, twice))


def test_an_empty_test_set_is_refused(plans):
    built = plans[RANDOM][0]
    no_test = built.plan.assignments.replace({TEST: TRAIN})
    with pytest.raises(PlanError, match="test set is empty"):
        validate_plan(_broken(built, no_test))


def test_a_divided_block_is_refused_in_a_kfold_plan(plans):
    built = plans[SPATIAL_KFOLD][0]
    assignments = built.plan.assignments.copy()
    block = built.plan.clusters.reindex(assignments.index)
    one_test_block = block[assignments == TEST].iloc[0]
    first_of_it = assignments.index[(block == one_test_block).to_numpy()][0]
    assignments.loc[first_of_it] = TRAIN
    with pytest.raises(PlanError, match=f"cluster {one_test_block} is divided"):
        validate_plan(_broken(built, assignments))


def test_a_divided_region_is_refused_in_a_logo_plan(plans):
    built = plans[LOGO][0]
    assignments = built.plan.assignments.copy()
    assignments.loc[built.plan.point_ids_for(TEST)[0]] = TRAIN
    with pytest.raises(PlanError, match="is divided"):
        validate_plan(_broken(built, assignments))


def test_the_held_out_region_leaking_into_validation_is_refused(plans):
    mixed, _ = _pairs(plans)[0]
    assignments = mixed.plan.assignments.copy()
    assignments.loc[next(iter(mixed.moved))] = VAL
    with pytest.raises(PlanError, match="only be divided between train and test"):
        validate_plan(_broken(mixed, assignments))


def test_a_mixed_plan_whose_training_is_not_the_moved_points_is_refused(plans):
    mixed, _ = _pairs(plans)[0]
    with pytest.raises(PlanError, match="not the points moved into training"):
        validate_plan(_broken(mixed, moved=frozenset(list(mixed.moved)[1:])))


def test_a_control_whose_training_is_not_the_mixed_plans_moved_points_is_refused(plans):
    mixed, control = _pairs(plans)[0]
    assignments = control.plan.assignments.copy()
    assignments.loc[next(iter(mixed.moved))] = TEST
    with pytest.raises(PlanError, match="not the points"):
        validate_pair(mixed, _broken(control, assignments))


def test_a_control_with_points_from_elsewhere_is_refused(plans, population):
    mixed, control = _pairs(plans)[0]
    elsewhere = population.ids.difference(population.region_points(mixed.held_out_cluster))[0]
    assignments = pd.concat([control.plan.assignments, pd.Series([TRAIN], index=[elsewhere])])
    with pytest.raises(PlanError):
        validate_plan(_broken(control, assignments))


def test_every_plan_passes_the_checks(plans):
    for group in plans.values():
        for built in group:
            validate_plan(built)


# --- writing them -----------------------------------------------------------------------------


@pytest.fixture
def written(settings, population, tmp_path):
    return write_plans(settings, str(tmp_path / "plans"), population=population)


def test_every_file_loads_as_a_split_plan_matching_the_index(written):
    index = pd.read_csv(written.out_dir / INDEX_FILE)
    assert list(index.columns) == list(INDEX_COLUMNS)
    assert len(index) == 1 + 5 + 4 + 2 * 4 * N_FRACTIONS * N_REPEATS
    for row in index.itertuples():
        plan = SplitPlan.from_frame(pd.read_parquet(written.out_dir / f"{row.plan_id}.parquet"))
        assert plan.counts() == {"train": row.n_train, "val": row.n_val, "test": row.n_test}


def test_the_saved_files_hold_exactly_the_split_plan_columns(written):
    for plan_id, expected in [
        ("random", ["point_id", "split", "eligible_sklearn"]),
        ("kfold_f0", ["point_id", "split", "cluster", "eligible_sklearn"]),
        ("logo_r01_north_east", ["point_id", "split", "cluster", "eligible_sklearn"]),
        ("control_r01_north_east_p10_s0", ["point_id", "split", "cluster", "eligible_sklearn"]),
    ]:
        assert list(pd.read_parquet(written.out_dir / f"{plan_id}.parquet").columns) == expected


def test_the_index_says_what_each_plan_is(written):
    index = written.index.set_index("plan_id")
    assert index.loc["kfold_f3", "held_out"] == "3"
    assert index.loc["logo_r02_north_west", "held_out"] == "North-West"
    row = index.loc["mixed_r02_north_west_p30_s4"]
    assert (row["kind"], row["fraction"], row["repeat"]) == (MIXED, 0.3, 4)
    assert pd.isna(index.loc["random", "fraction"])


def test_old_plans_are_cleared_before_writing(settings, population, tmp_path):
    out = tmp_path / "plans"
    out.mkdir()
    (out / "logo_r99_gone.parquet").write_text("stale")
    write_plans(settings, str(out), kinds=[RANDOM], population=population)
    assert sorted(path.name for path in out.iterdir()) == [INDEX_FILE, "random.parquet"]


def test_nothing_is_written_when_a_plan_is_broken(settings, population, tmp_path, monkeypatch):
    from yg_eo_soilnet.a1 import plans as plans_module

    def broken_logo(population, settings):
        built = logo_plans(population, settings)[0]
        return [_broken(built, built.plan.assignments.replace({TEST: TRAIN}))]

    monkeypatch.setattr(plans_module, "logo_plans", broken_logo)
    with pytest.raises(PlanError):
        write_plans(settings, str(tmp_path / "plans"), kinds=[LOGO], population=population)
    assert not (tmp_path / "plans").exists()


def test_without_regions_the_region_plans_are_skipped_with_the_reason(a1_settings, tmp_path):
    result = write_plans(a1_settings(regions={"source": None}), str(tmp_path / "plans"))
    assert set(result.skipped) == {LOGO, MIXED}
    assert set(result.index["kind"]) == {RANDOM, SPATIAL_KFOLD}


def test_an_unknown_kind_is_refused(population, settings):
    with pytest.raises(ValueError, match="Unknown plan kind"):
        build_plans(population, settings, kinds=["leave_one_out"])


def test_the_command_line_writes_the_kinds_asked_for(a1_settings, tmp_path, capsys):
    path = a1_settings(as_file=True)
    result = make_plans_main(["--config", str(path), "--out", str(tmp_path / "cli"), "--kinds", "random,logo"])

    assert set(result.index["kind"]) == {RANDOM, LOGO}
    assert (tmp_path / "cli" / INDEX_FILE).exists()
    assert "logo: 4" in capsys.readouterr().out


# --- settings -------------------------------------------------------------------------------------


@pytest.fixture
def main_config(tmp_path):
    path = tmp_path / "main.yml"
    path.write_text("common: {}\n")
    return path


@pytest.mark.parametrize(
    "plans_section, message",
    [
        ({"random": {"test_size": 0.6, "val_size": 0.5}}, "leave a training set"),
        ({"random": {"test_size": 0}}, "strictly between 0 and 1"),
        ({"spatial_kfold": {"method": "voronoi"}}, "method must be one of"),
        ({"spatial_kfold": {"n_folds": 1}}, "n_folds must be at least 2"),
        ({"spatial_kfold": {"n_clusters": 3, "n_folds": 5}}, "n_clusters"),
        ({"spatial_kfold": {"method": "grid", "cell_size_m": 0}}, "cell_size_m"),
        ({"mixed": {"fractions": [0.0, 0.5]}}, "fractions must lie strictly"),
        ({"mixed": {"fractions": [0.5, 0.5]}}, "repeats a value"),
        ({"mixed": {"n_repeats": 0}}, "n_repeats"),
        ({"logo": {}}, r"Unknown setting.*plans"),
        ({"mixed": {"repeats": 3}}, r"Unknown setting.*plans.mixed"),
    ],
)
def test_plan_settings_that_cannot_make_a_sound_plan_are_refused(main_config, plans_section, message):
    with pytest.raises(ValueError, match=message):
        settings_from_mapping({"main_config": str(main_config), "plans": plans_section})


def test_the_default_plan_settings_are_the_ones_agreed(main_config):
    plans = settings_from_mapping({"main_config": str(main_config)}).plans
    assert (plans.random.test_size, plans.random.val_size) == (0.15, 0.15)
    assert (plans.spatial_kfold.method, plans.spatial_kfold.n_clusters, plans.spatial_kfold.n_folds) == (
        "kmeans",
        50,
        5,
    )
    assert plans.mixed.fractions == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert plans.mixed.n_repeats == 5
