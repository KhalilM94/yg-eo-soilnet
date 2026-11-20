"""Mixed plans drawn by whole blocks, point against block draws in compare.py, figures C8 and C9.

A block draw must move exactly the share asked for, by whole 10 km blocks and part of one more,
leave the rest of that block out of the plan, and so leave no test point beside a moved one; its
shares must nest as the point draws' do, and the weighted arm must weigh the points actually moved.
compare.py must keep the two draw types apart and report how many test points share a block with
a moved point. The bias-correction scatter must show what an offset and a recalibration change.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import robustness
from yg_eo_soilnet.a1.arms import sample_weights
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.figures import core as figure_core
from yg_eo_soilnet.a1.plans import (
    BLOCK_DRAW_INDEX_COLUMNS,
    CONTROL,
    LOGO,
    MIXED,
    MIXED_WEIGHTED,
    block_order,
    mixed_plans,
    plan_population,
    shared_block_share,
    training_blocks,
    write_plans,
)
from yg_eo_soilnet.a1.runner import run_plans
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN


def _blocks_settings(settings, *, draw="blocks", fractions=(0.1, 0.3), repeats=2):
    mixed = replace(settings.plans.mixed, fractions=tuple(fractions), n_repeats=repeats, arms=("weighted",), draw=draw)
    return replace(settings, plans=replace(settings.plans, mixed=mixed))


@pytest.fixture(scope="module")
def population(a1_demo_settings):
    return plan_population(a1_demo_settings)


@pytest.fixture(scope="module")
def block_plans(population, a1_demo_settings):
    settings = _blocks_settings(a1_demo_settings)
    return settings, {plan.plan_id: plan for plan in mixed_plans(population, settings)}


def test_a_block_draw_moves_exactly_the_share_by_whole_blocks_and_leaves_the_rest_of_the_last_one_out(
    block_plans, population
):
    settings, plans = block_plans
    blocks = training_blocks(population, settings)
    mixed = [plan for plan in plans.values() if plan.kind == MIXED]
    assert mixed
    for built in mixed:
        region = population.region_points(built.held_out_cluster)
        moved = set(built.moved)
        assert len(moved) == min(max(int(round(built.fraction * len(region))), 1), len(region) - 1)
        assert built.draw == "blocks" and built.n_blocks_moved == blocks.reindex(list(moved)).nunique()
        # Every block but one is moved whole; the cut block's other points are in neither split.
        counts = blocks.reindex(list(moved)).value_counts()
        whole = blocks.reindex(region).value_counts()
        partial = [b for b, n in counts.items() if n < whole[b]]
        assert len(partial) <= 1
        assigned = set(built.plan.assignments.index)
        left_out = set(region) - assigned
        assert len(left_out) == built.n_left_out
        if partial:
            assert left_out == set(region[(blocks.reindex(region) == partial[0]).to_numpy()]) - moved
        test = built.plan.point_ids_for(TEST)
        assert shared_block_share(test, moved, blocks) == 0.0
        assert set(test) == set(region) - moved - left_out


def test_the_shares_of_one_repeat_nest_and_every_kind_has_the_same_points(block_plans):
    _, plans = block_plans
    for repeat in (0, 1):
        small = next(p for p in plans.values() if p.kind == MIXED and p.repeat == repeat and p.fraction == 0.1)
        large = next(p for p in plans.values() if p.kind == MIXED and p.repeat == repeat and p.fraction == 0.3)
        assert set(small.moved) <= set(large.moved)
    for plan_id, built in plans.items():
        if built.kind == MIXED:
            control = plans[plan_id.replace("mixed_", "control_", 1)]
            weighted = plans[plan_id.replace("mixed_", "mixed_w_", 1)]
            assert set(control.plan.point_ids_for(TEST)) == set(built.plan.point_ids_for(TEST))
            assert set(control.plan.point_ids_for(TRAIN)) == set(built.moved)
            assert weighted.plan.assignments.sort_index().equals(built.plan.assignments.sort_index())


def test_the_weighted_arm_weighs_the_points_actually_moved(block_plans):
    settings, plans = block_plans
    weighted = next(p for p in plans.values() if p.kind == MIXED_WEIGHTED)
    train = weighted.plan.point_ids_for(TRAIN)
    weights, w = sample_weights(weighted.plan, settings, train)
    n_moved = len(weighted.moved)
    assert w == pytest.approx((len(train) - n_moved) / n_moved)
    assert (weights[list(weighted.moved)] == w).all()


def test_point_draws_are_unchanged_and_block_draws_go_into_the_index(population, a1_demo_settings, tmp_path):
    points = mixed_plans(population, _blocks_settings(a1_demo_settings, draw="points"))
    default = mixed_plans(population, _blocks_settings(a1_demo_settings, draw="points"))
    assert all(a.plan.assignments.equals(b.plan.assignments) for a, b in zip(points, default))
    assert all(p.draw == "points" and p.n_left_out is None for p in points)
    plain = write_plans(_blocks_settings(a1_demo_settings, draw="points"), str(tmp_path / "points"), kinds=[MIXED])
    assert not set(BLOCK_DRAW_INDEX_COLUMNS) & set(pd.read_csv(plain.out_dir / "index.csv").columns)
    blocked = write_plans(_blocks_settings(a1_demo_settings), str(tmp_path / "blocks"), kinds=[MIXED])
    index = pd.read_csv(blocked.out_dir / "index.csv")
    assert set(BLOCK_DRAW_INDEX_COLUMNS) <= set(index.columns)
    assert set(index.loc[index["kind"] == MIXED, "draw"]) == {"blocks"}


def test_an_unknown_draw_is_refused(a1_settings):
    with pytest.raises(ValueError, match="plans.mixed.draw"):
        a1_settings(plans={"mixed": {"fractions": [0.5], "n_repeats": 1, "draw": "squares"}})


def test_block_order_moves_blocks_one_after_another():
    ids = pd.Index([f"p{i}" for i in range(9)])
    blocks = pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3], index=ids)
    order, block_of = block_order(ids, blocks, np.random.default_rng(0))
    labels = block_of.reindex(order).to_numpy()
    changes = int((labels[1:] != labels[:-1]).sum())
    assert changes == 2 and sorted(order) == sorted(ids)  # each block's points together


# --- compare.py: point against block draws --------------------------------------------------------------

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    "stage2": {
        "estimator": "XGBoost",
        "init_args": {"n_estimators": 20},
        "params": {"model__max_depth": [3]},
        "models": ["global"],
    },
    "inner_cv_folds": 2,
}


@pytest.fixture(scope="module")
def draw_runs(a1_settings_in, tmp_path_factory, mlflow_store):
    folder = tmp_path_factory.mktemp("a1_draws")
    logo = {"min_points": 100, "held_out": ["North-East"], "validation_region": False}
    run_dirs = {}
    with mlflow_store(folder / "mlruns"):
        for draw in ("points", "blocks"):
            sub = folder / f"a1_{draw}"
            sub.mkdir()
            path = a1_settings_in(sub)(
                as_file=True,
                output_dir=str(sub),
                plans={
                    "logo": logo,
                    "mixed": {"fractions": [0.2, 0.4], "n_repeats": 2, "arms": ["weighted"], "draw": draw},
                },
                **FAST,
            )
            a1 = load_a1_config(str(path))
            write_plans(a1.settings, str(sub / "plans"), kinds=[LOGO, MIXED])
            run_dirs[draw] = run_plans(a1, kinds=[LOGO, MIXED, CONTROL, MIXED_WEIGHTED], tracking=False).run_dir
    yield {"folder": folder, **run_dirs}


def test_compare_keeps_the_draw_types_apart_and_reports_block_sharing(draw_runs, tmp_path):
    from yg_eo_soilnet.a1.compare import compare_runs

    result = compare_runs([draw_runs["points"], draw_runs["blocks"]], tmp_path, n_boot=30, tracking=False)
    sharing = result.tables["block_sharing"]
    assert set(sharing["draw"]) == {"points", "blocks"}
    assert (sharing.loc[sharing["draw"] == "blocks", "share_test_sharing_block"] == 0).all()
    assert (sharing["mean_blocks_moved"] > 0).all()
    curves = result.tables["draw_type_curves"]
    assert set(curves["draw"]) == {"points", "blocks"} and "transferred" in set(curves["kind"])
    assert set(result.tables["arms_vs_transferred"]["run"]) == {"a1_points", "a1_blocks"}
    assert not figure_core.is_message_figure(result.figures["figC8_draw_types"])
    caption = result.captions["figC8_draw_types"]
    assert "Mean blocks moved" in caption and "Broeg et al. 2023" in caption


def test_without_both_draw_types_figure_c8_is_a_note(draw_runs, tmp_path):
    from yg_eo_soilnet.a1.compare import compare_runs

    result = compare_runs([draw_runs["blocks"]], tmp_path, n_boot=20, tracking=False)
    assert figure_core.is_message_figure(result.figures["figC8_draw_types"])


# --- figure C9 and the calibration ----------------------------------------------------------------------


def _sample(n=60, seed=0):
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(10, 40, n)
    observed = 5 + 0.6 * predicted + rng.normal(0, 2, n)
    return pd.DataFrame({"predicted": predicted, "observed": observed, "class": "<2"})


def test_the_offset_leaves_the_slope_and_the_linear_correction_gives_slope_one_on_its_samples():
    sample, test = _sample(seed=1), _sample(seed=2)
    before = robustness.calibration(test["observed"], test["predicted"])
    after = robustness.calibration(test["observed"], robustness.offset_correction(sample, test))
    assert after["slope"] == pytest.approx(before["slope"])  # only the intercept moves
    assert after["intercept"] != pytest.approx(before["intercept"])
    own = robustness.calibration(sample["observed"], robustness.linear_correction(sample, sample))
    assert own["slope"] == pytest.approx(1.0) and own["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert own["bias"] == pytest.approx(0.0, abs=1e-9)


def test_the_median_draw_and_the_calibration_table():
    draws = pd.DataFrame({"basin": "A", "k": 100, "draw": range(5), "rmse_offset": [3.0, 1.0, 5.0, 2.0, 4.0]})
    assert robustness.median_draw(draws, "A", 100) == 0  # RMSE 3.0, the middle of five
    assert robustness.median_draw(draws, "A", 400) is None
    rows = []
    for draw in range(4):
        row = {"basin": "A", "k": 100, "draw": draw}
        for method in robustness.CALIBRATED:
            row.update(
                {
                    f"slope_{method}": 1.0 + draw,
                    f"intercept_{method}": 0.0,
                    f"bias_{method}": 0.5,
                    f"rmse_{method}": 2.0,
                }
            )
        rows.append(row)
    table = robustness.calibration_summary(pd.DataFrame(rows), local_model="local_global")
    assert set(table["method"]) == set(robustness.CALIBRATED)
    one = table[table["method"] == "offset"].iloc[0]
    assert one["slope"] == pytest.approx(2.5) and one["slope_q05"] < one["slope"] < one["slope_q95"]


def test_figure_c9_renders_points_and_a_density():
    from yg_eo_soilnet.a1.figures.robustness import figure_bias_correction_scatter

    rng = np.random.default_rng(3)
    parts = []
    for basin, n in (("Oum Er Rbia", 1500), ("Bouregreg", 200)):
        observed = rng.gamma(4, 6, n)
        frame = pd.DataFrame(
            {
                "basin": basin,
                "point_id": [f"{basin}{i}" for i in range(n)],
                "class": rng.choice(["<2", ">=25"], n),
                "observed": observed,
            }
        )
        for method in robustness.CALIBRATED:
            frame[method] = observed * 0.5 + 12 + rng.normal(0, 3, n)
        parts.append(frame)
    figure, caption = figure_bias_correction_scatter(
        pd.concat(parts),
        k=100,
        local_model="local_global",
        labels=["<2", ">=25"],
        chosen={"Oum Er Rbia": 7, "Bouregreg": 3},
    )
    assert not figure_core.is_message_figure(figure)
    assert "median over the 100 draws" in caption and "changes only the intercept" in caption
    assert "Not shown" in caption and "Sebou" in caption


# --- the legacy soil-map sensitivity in fig05 and fig10 --------------------------------------------------------


def _inputs(tmp_path, basins):
    return SimpleNamespace(
        settings=SimpleNamespace(output_dir=str(tmp_path / "a1_xgb")),
        held_out_names=list(basins),
        predictions=pd.DataFrame({"x": [1]}),
    )


def test_the_legacy_soil_sentence_comes_from_e5_when_it_covers_the_runs_basins(tmp_path):
    assert figure_core.legacy_soil_sensitivity(_inputs(tmp_path, ["A"])) == ""  # no E5 yet
    rows = [
        {
            "design": design,
            "variant": "all_minus_legacy_soil",
            "metric": "rmse",
            "estimate": est,
            "low": est - 0.1,
            "high": est + 0.1,
            "p_holm": p,
            "n_seeds": 1,
        }
        for design, est, p in (
            ("random", -0.02, 1.0),
            ("10 km", 0.09, 0.32),
            ("logo pooled", -0.07, 1.0),
            ("logo A", 0.1, 1.0),
            ("logo B", 0.0, 1.0),
        )
    ]
    (tmp_path / "a1_e5").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "a1_e5" / "ablation.csv", index=False)
    sentence = figure_core.legacy_soil_sensitivity(_inputs(tmp_path, ["A", "B"]))
    assert (
        "Sensitivity (E5)" in sentence and "+0.09 [−0.01, 0.19]" in sentence and "p_Holm 1.00, 0.32, 1.00" in sentence
    )
    assert "source product not documented" in sentence
    assert figure_core.legacy_soil_sensitivity(_inputs(tmp_path, ["North-East"])) == ""  # another dataset


def test_the_block_configuration_loads():
    from pathlib import Path

    from yg_eo_soilnet.a1.settings import load_settings

    settings = load_settings(str(Path(__file__).resolve().parents[2] / "configs" / "a1" / "a1_mixed_blocks.yml"))
    mixed = settings.plans.mixed
    assert mixed.draw == "blocks" and mixed.fractions == (0.05, 0.1, 0.2, 0.3) and mixed.n_repeats == 5
    assert mixed.arms == ("weighted",) and settings.stage2.models == ("global",) and settings.seeds == (42,)
    assert settings.plans.logo.held_out == ("Oum Er Rbia", "Sebou") and settings.output_dir.endswith("a1_mixed_blocks")


def test_the_c9_line_stays_where_the_points_are_and_the_caption_explains_the_slope():
    from yg_eo_soilnet.a1.figures.robustness import SLOPE_SENTENCE, fit_segment

    xs, ys = fit_segment([24.0, 26.0, 30.0], {"slope": 5.6, "intercept": -127.0}, (5.0, 60.0))
    assert xs.min() >= 24.0 and xs.max() <= 30.0  # within the points' x-range
    assert ys.min() >= 5.0 - 1e-9 and ys.max() <= 60.0 + 1e-9  # within the panel's height
    assert fit_segment([1.0, 2.0], {"slope": float("nan"), "intercept": 0.0}, (0.0, 10.0)) is None
    assert "< 1 means the predictions vary more than the measured SOM" in SLOPE_SENTENCE
    assert "unstable (see bias_correction_calibration.csv" in SLOPE_SENTENCE
