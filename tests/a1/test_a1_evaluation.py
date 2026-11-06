"""The A1 evaluation: grouped scores, block-bootstrap intervals, Holm, and the transfer table.

Everything runs on small synthetic prediction tables whose answers are known in advance: a perfect
model, a constant offset, two identical models, a local model better by a set amount.
"""

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1.evaluation import (
    SCORES,
    _draws,
    _scores,
    block_bootstrap,
    grouped_metrics,
    holm,
    load_predictions,
    spatial_blocks,
    transfer_table,
)
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM
from yg_eo_soilnet.a1.runner import PREDICTION_COLUMNS
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL
from yg_eo_soilnet.datamodules.splitting import TEST, VAL
from yg_eo_soilnet.metrics import regression_metrics


def _points(n=240, seed=0):
    """``n`` points scattered over about 60 x 60 km of central Morocco, with a measured SOM value."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "point_id": [f"p{i:04d}" for i in range(n)],
            "lat": 32.0 + rng.uniform(0, 0.55, n),
            "lon": -7.0 + rng.uniform(0, 0.65, n),
            "observed": rng.gamma(4.0, 5.0, n),
        }
    )


def _rows(points, predicted, *, model="m", plan_id="random", kind=RANDOM, split=TEST, **extra):
    """Prediction rows in the runner's layout, for one model on one plan."""
    frame = pd.DataFrame(
        {
            "plan_id": plan_id,
            "kind": kind,
            "held_out": None,
            "fraction": np.nan,
            "repeat": np.nan,
            "point_id": points["point_id"].to_numpy(),
            "split": split,
            "region": "R",
            "lat": points["lat"].to_numpy(),
            "lon": points["lon"].to_numpy(),
            "carbonate_class_measured": "<2",
            "carbonate_class_predicted": "<2",
            "model": model,
            "covariate_group": "combined",
            "observed": points["observed"].to_numpy(),
            "predicted": np.asarray(predicted, dtype=float),
            "fallback": False,
            "model_role": "model",
            "tuned_params": "{}",
        },
        columns=list(PREDICTION_COLUMNS),
    )
    for column, value in extra.items():
        frame[column] = value
    return frame.astype({"fraction": float, "repeat": float})


def _noisy(points, sd, seed):
    return points["observed"].to_numpy() + np.random.default_rng(seed).normal(0.0, sd, len(points))


# --- grouped scores -----------------------------------------------------------------------------


def test_a_perfect_model_scores_zero_error_and_r2_of_one():
    points = _points()
    table = grouped_metrics(_rows(points, points["observed"]), "model")
    row = table.iloc[0]
    assert row["rmse"] == 0 and row["mae"] == 0 and row["bias"] == 0 and row["r2"] == 1
    assert np.isnan(row["rpd"]) and np.isnan(row["rpiq"])  # undefined with no error at all


def test_a_constant_offset_shows_as_bias_rmse_and_mae():
    points = _points()
    row = grouped_metrics(_rows(points, points["observed"] + 2.0), "model").iloc[0]
    assert row["bias"] == pytest.approx(2.0) and row["rmse"] == pytest.approx(2.0) and row["mae"] == pytest.approx(2.0)


def test_grouped_scores_are_the_ones_regression_metrics_gives():
    points = _points(4)
    points["observed"] = [10.0, 20.0, 30.0, 40.0]
    row = grouped_metrics(_rows(points, [12, 18, 33, 39]), "model", min_n=1).iloc[0]
    expected = regression_metrics([10, 20, 30, 40], [12, 18, 33, 39])
    for score in SCORES:
        assert row[score] == pytest.approx(expected[f"{score}_test"])


def test_each_group_reports_its_size_and_whether_it_is_too_small():
    points = _points(100)
    frame = pd.concat(
        [
            _rows(points.iloc[:70], _noisy(points.iloc[:70], 3, 1), model="a"),
            _rows(points.iloc[70:], _noisy(points.iloc[70:], 3, 2), model="b"),
            _rows(points.iloc[:50], _noisy(points.iloc[:50], 3, 3), model="a", split=VAL),
        ],
        ignore_index=True,
    )
    frame.loc[frame.index[:5], "observed"] = np.nan  # no measurement: not scored
    table = grouped_metrics(frame, ["model", "split"]).set_index(["model", "split"])
    assert table.loc[("a", TEST), "n"] == 65 and not table.loc[("a", TEST), "small"]
    assert table.loc[("b", TEST), "n"] == 30 and not table.loc[("b", TEST), "small"]
    assert table.loc[("a", VAL), "n"] == 50
    assert grouped_metrics(frame, ["model", "split"], min_n=50).set_index(["model", "split"]).loc[("b", TEST), "small"]


def test_the_fast_scorer_agrees_with_regression_metrics():
    rng = np.random.default_rng(7)
    for trial in range(300):
        n = int(rng.integers(2, 60))
        observed = np.round(rng.gamma(3.0, 6.0, n), 0 if trial % 3 == 0 else 3)  # some ties
        if trial % 25 == 0:
            observed[:] = observed[0]  # every measurement the same: R² undefined
        predicted = observed + rng.normal(0, rng.uniform(0.1, 8), n) if trial % 40 else observed.copy()
        expected = regression_metrics(observed, predicted)
        fast = _scores(observed, predicted)
        for position, score in enumerate(SCORES):
            name = f"{score}_test"
            if name in expected:
                assert fast[position] == pytest.approx(expected[name], rel=1e-9, abs=1e-12), (trial, score)
            else:
                assert np.isnan(fast[position]), (trial, score)


# --- the block bootstrap ----------------------------------------------------------------------


def test_identical_models_differ_by_exactly_zero_with_a_p_value_of_one():
    points = _points()
    predicted = _noisy(points, 4, 1)
    frame = pd.concat([_rows(points, predicted, model="a"), _rows(points, predicted, model="b")])
    result = block_bootstrap(frame, ["model"], n=200, compare=[("a", "b")])
    for row in result.differences.dropna(subset=["low"]).itertuples():
        assert row.low <= 0 <= row.high
        assert row.low == 0 and row.high == 0 and row.p_value == 1
    assert set(result.differences.dropna(subset=["low"])["metric"]) == set(SCORES)


def test_a_perfect_model_has_an_rmse_interval_of_zero():
    points = _points()
    result = block_bootstrap(_rows(points, points["observed"]), "model", n=200)
    rmse = result.intervals.set_index("metric").loc["rmse"]
    assert rmse["estimate"] == 0 and rmse["low"] == 0 and rmse["high"] == 0
    assert rmse["n_blocks"] > 10


def test_a_clearly_better_model_has_a_difference_interval_away_from_zero():
    points = _points()
    frame = pd.concat(
        [_rows(points, _noisy(points, 1, 1), model="good"), _rows(points, _noisy(points, 8, 2), model="poor")]
    )
    result = block_bootstrap(frame, ["model"], n=300, compare=[("good", "poor")])
    rmse = result.differences.set_index("metric").loc["rmse"]
    assert rmse["high"] < 0 and rmse["estimate"] < 0
    assert rmse["p_value"] < 0.01 and rmse["p_holm"] < 0.05
    assert rmse["n"] == len(points) and rmse["n_unpaired"] == 0


def test_the_intervals_bracket_the_estimate_and_come_out_the_same_for_the_same_seed():
    points = _points()
    frame = _rows(points, _noisy(points, 4, 1))
    first = block_bootstrap(frame, "model", n=200, seed=3).intervals
    again = block_bootstrap(frame, "model", n=200, seed=3).intervals
    other = block_bootstrap(frame, "model", n=200, seed=4).intervals
    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first["low"], other["low"], equal_nan=True)
    rmse = first.set_index("metric").loc["rmse"]
    assert rmse["low"] < rmse["estimate"] < rmse["high"]


def test_a_group_draws_the_same_whatever_other_groups_are_there():
    points = _points()
    alone = _rows(points, _noisy(points, 4, 1), model="a")
    together = pd.concat([alone, _rows(points, _noisy(points, 2, 2), model="b")])
    first = block_bootstrap(alone, "model", n=100).intervals
    second = block_bootstrap(together, "model", n=100).intervals
    pd.testing.assert_frame_equal(first, second[second["model"] == "a"].reset_index(drop=True))


def test_whole_blocks_are_drawn_together():
    blocks = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    for rows in _draws(blocks, 50, np.random.default_rng(0)):
        counts = np.bincount(rows, minlength=len(blocks))
        for block in np.unique(blocks):
            members = counts[blocks == block]
            assert (members == members[0]).all()  # every point of a block, as often as the block
        assert sum(counts[blocks == block][0] for block in np.unique(blocks)) == 4  # four blocks drawn


def test_the_blocks_are_10_km_squares_and_a_point_keeps_its_block_everywhere():
    points = _points()
    frame = pd.concat(
        [_rows(points, points["observed"], model="a"), _rows(points.iloc[::-1], points["observed"], model="b")]
    )
    blocks = spatial_blocks(frame, block_km=10)
    per_point = pd.Series(blocks.to_numpy()).groupby(frame["point_id"].to_numpy()).nunique()
    assert (per_point == 1).all()
    assert 25 <= blocks.nunique() <= 49  # about 60 x 60 km cut into 10 km squares
    assert spatial_blocks(frame, block_km=20).nunique() < blocks.nunique()


def test_a_group_with_a_single_block_has_no_interval():
    points = _points(40)
    points["lat"], points["lon"] = 32.0 + np.linspace(0, 0.01, 40), -7.0  # all within 1 km
    far = _points(40, seed=1).assign(point_id=lambda frame: "far" + frame["point_id"])
    frame = pd.concat([_rows(points, _noisy(points, 3, 1), model="near"), _rows(far, _noisy(far, 3, 2), model="far")])
    intervals = block_bootstrap(frame, "model", n=50).intervals
    near = intervals[intervals["model"] == "near"]
    assert (near["n_blocks"] == 1).all() and near["low"].isna().all() and near["high"].isna().all()
    assert intervals.loc[intervals["model"] == "far", "low"].notna().any()


def test_a_point_with_two_positions_is_refused():
    points = _points(20)
    moved = points.assign(lat=points["lat"] + 0.1)
    with pytest.raises(ValueError, match="two positions"):
        spatial_blocks(pd.concat([_rows(points, points["observed"]), _rows(moved, moved["observed"], model="b")]))


def test_rows_without_a_partner_are_counted_not_paired():
    points = _points()
    frame = pd.concat(
        [
            _rows(points, _noisy(points, 3, 1), model="a"),
            _rows(points.iloc[:200], _noisy(points.iloc[:200], 3, 2), model="b"),
        ]
    )
    rmse = block_bootstrap(frame, ["model"], n=50, compare=[("a", "b")]).differences.set_index("metric").loc["rmse"]
    assert rmse["n"] == 200 and rmse["n_unpaired"] == 40


def test_an_ambiguous_pairing_is_refused():
    points = _points()
    frame = pd.concat(
        [
            _rows(points, _noisy(points, 3, 1), model="a"),
            _rows(points, _noisy(points, 3, 2), model="a", covariate_group="terrain"),
            _rows(points, _noisy(points, 3, 3), model="b"),
        ]
    )
    with pytest.raises(ValueError, match="covariate_group"):
        block_bootstrap(frame, ["model"], n=10, compare=[("a", "b")])
    result = block_bootstrap(frame, ["model", "covariate_group"], n=10, compare=[("a", "b")])
    assert set(result.differences["covariate_group"]) == {"combined"}  # only there do both models exist


# --- Holm -----------------------------------------------------------------------------------------


def test_holm_matches_a_worked_example():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    for size in (1, 2, 5, 20, 100):
        pvalues = rng.uniform(0, 0.2, size) ** 2
        expected = multitest.multipletests(pvalues, method="holm")[1]
        assert holm(pvalues) == pytest.approx(expected)


def test_holm_keeps_missing_values_out_of_the_family_and_a_series_its_index():
    pvalues = pd.Series([0.01, np.nan, 0.02, 0.9], index=list("wxyz"), name="p")
    adjusted = holm(pvalues)
    assert list(adjusted.index) == list("wxyz") and adjusted.name == "p"
    assert np.isnan(adjusted["x"])
    assert adjusted["w"] == pytest.approx(0.03) and adjusted["y"] == pytest.approx(0.04)  # m = 3, not 4
    assert (adjusted.dropna() <= 1).all() and (adjusted.dropna() >= pvalues.dropna()).all()


def test_holm_refuses_values_that_are_not_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm([0.2, 1.5])


# --- transfer -------------------------------------------------------------------------------------


def _logo_rows(points, region, *, local_sd, transferred_sd, groups=("combined",)):
    plan = f"logo_{region}"
    rows = []
    for number, group in enumerate(groups):
        common = dict(plan_id=plan, kind=LOGO, held_out=region, covariate_group=group)
        rows.append(_rows(points, points["observed"] + transferred_sd, model=GLOBAL_AUX, **common))
        if local_sd is not None:
            rows.append(_rows(points, points["observed"] + local_sd, model=LOCAL, **common))
        rows.append(_rows(points, _noisy(points, 9, number), model=GLOBAL, **common))  # not compared
    return rows


def test_a_logo_drop_is_the_known_gap_between_transferred_and_local():
    points = _points(60)
    frame = pd.concat(
        [
            *_logo_rows(points.iloc[:30], "A", local_sd=1.0, transferred_sd=4.0, groups=("combined", "terrain")),
            *_logo_rows(points.iloc[30:], "B", local_sd=2.0, transferred_sd=2.5),
            _rows(points, points["observed"], model=GLOBAL_AUX, plan_id="logo_A", kind=LOGO, held_out="A", split=VAL),
        ]
    )
    table = transfer_table(frame).set_index(["region", "covariate_group"])
    assert len(table) == 3
    a = table.loc[("A", "combined")]
    assert a["rmse_local"] == pytest.approx(1.0) and a["rmse_transferred"] == pytest.approx(4.0)
    assert a["rmse_drop"] == pytest.approx(3.0) and a["rmse_drop_pct"] == pytest.approx(300.0)
    assert a["mae_drop"] == pytest.approx(3.0) and a["r2_drop"] > 0 and a["rpd_drop"] > 0
    assert a["bias_local"] == pytest.approx(1.0) and a["n"] == 30 and a["n_repeats"] == 1
    assert a["local_source"] == "local model" and a["has_local"] and np.isnan(a["fraction"])
    assert table.loc[("B", "combined"), "rmse_drop"] == pytest.approx(0.5)


def test_a_region_without_a_local_score_keeps_its_row_with_nan_locals():
    points = _points(30)
    table = transfer_table(pd.concat(_logo_rows(points, "A", local_sd=None, transferred_sd=3.0)))
    row = table.iloc[0]
    assert not row["has_local"] and np.isnan(row["rmse_local"]) and np.isnan(row["rmse_drop"])
    assert row["rmse_transferred"] == pytest.approx(3.0) and row["n"] == 30


def test_mixed_plans_are_compared_with_their_own_control_and_averaged_over_repeats():
    points = _points(40)
    rows = []
    for repeat, (offset_mixed, offset_control) in enumerate([(1.0, 3.0), (2.0, 6.0)]):
        test_points = points.iloc[repeat * 10 : repeat * 10 + 30]  # each repeat its own test points
        for kind, offset in ((MIXED, offset_mixed), (CONTROL, offset_control)):
            rows.append(
                _rows(
                    test_points,
                    test_points["observed"] + offset,
                    model=GLOBAL_AUX,
                    plan_id=f"{kind}_A_p30_s{repeat}",
                    kind=kind,
                    held_out="A",
                    fraction=0.3,
                    repeat=float(repeat),
                )
            )
    # Another share of the same region: a control that must not be matched with the 0.3 plans.
    rows.append(
        _rows(
            points,
            points["observed"] + 50.0,
            model=GLOBAL_AUX,
            plan_id="control_A_p50_s0",
            kind=CONTROL,
            held_out="A",
            fraction=0.5,
            repeat=0.0,
        )
    )
    rows.append(_rows(points, points["observed"], model=GLOBAL_AUX, kind=RANDOM))  # ignored

    table = transfer_table(pd.concat(rows))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["kind"] == MIXED and row["fraction"] == pytest.approx(0.3) and row["n_repeats"] == 2
    assert row["local_source"] == "control plan" and row["has_local"]
    assert row["rmse_transferred"] == pytest.approx(1.5) and row["rmse_local"] == pytest.approx(4.5)
    assert row["rmse_drop"] == pytest.approx(-3.0)  # negative: the other regions help
    assert row["rmse_drop_sd"] == pytest.approx(np.std([-2.0, -4.0], ddof=1))


# --- reading a run folder -------------------------------------------------------------------------


def test_load_predictions_reads_every_file_and_filters(tmp_path):
    points = _points(30)
    for plan_id, model in (("random", GLOBAL), ("random", GLOBAL_AUX), ("logo_A", GLOBAL)):
        folder = tmp_path / "predictions" / plan_id
        folder.mkdir(parents=True, exist_ok=True)
        kind = RANDOM if plan_id == "random" else LOGO
        _rows(points, points["observed"], model=model, plan_id=plan_id, kind=kind).to_parquet(
            folder / f"{model}__combined.parquet"
        )
    (tmp_path / "predictions" / "random" / "x.parquet.partial").write_text("half")

    assert len(load_predictions(tmp_path)) == 90
    assert set(load_predictions(tmp_path, kinds=[LOGO])["plan_id"]) == {"logo_A"}
    assert set(load_predictions(tmp_path, models=[GLOBAL_AUX])["model"]) == {GLOBAL_AUX}


def test_load_predictions_refuses_a_file_in_the_older_format(tmp_path):
    folder = tmp_path / "predictions" / "random"
    folder.mkdir(parents=True)
    points = _points(10)
    _rows(points, points["observed"]).drop(columns=["lat", "lon"]).to_parquet(folder / "global__combined.parquet")
    with pytest.raises(ValueError, match="--restart"):
        load_predictions(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "nothing")


def test_spatial_scores_of_different_block_sizes_are_never_pooled():
    points = _points(120)
    rows = []
    for size, sd in ((10_000.0, 2.0), (25_000.0, 6.0)):
        for model, extra in (("a", 0.0), ("b", 1.0)):
            rows.append(
                _rows(
                    points,
                    _noisy(points, sd + extra, int(size)),
                    model=model,
                    plan_id=f"kfold_{size:g}",
                    kind="spatial_kfold",
                    block_size_m=size,
                )
            )
    rows.append(_rows(points, _noisy(points, 3, 1), model="a", block_size_m=np.nan))  # random: no block size
    rows.append(_rows(points, _noisy(points, 4, 2), model="b", block_size_m=np.nan))
    frame = pd.concat(rows, ignore_index=True)

    table = grouped_metrics(frame, ["kind", "block_size_m", "model"])
    kfold = table[table["kind"] == "spatial_kfold"]
    assert len(kfold) == 4 and (kfold["n"] == len(points)).all()  # each size scores each point once
    by_size = kfold.pivot(index="model", columns="block_size_m", values="rmse")
    assert (by_size[25_000.0] > by_size[10_000.0]).all()  # the noisier size keeps its own, larger, RMSE

    result = block_bootstrap(frame, ["kind", "block_size_m", "model"], n=30, compare=[("a", "b")])
    differences = result.differences[result.differences["metric"] == "rmse"]
    assert len(differences) == 3  # random, and each block size on its own
    assert (differences["n"] == len(points)).all() and (differences["n_unpaired"] == 0).all()
