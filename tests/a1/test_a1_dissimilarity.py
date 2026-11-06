"""The dissimilarity index, area of applicability, MESS and the per-region summary.

Synthetic inputs throughout, with answers known in advance: new points on top of the training
points (DI 0, all inside), new points far away (DI large, all outside), and small hand-worked cases.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from yg_eo_soilnet.a1.dissimilarity import (
    _threshold,
    dissimilarity_index,
    importance_weights,
    mess,
    plan_dissimilarity,
    region_summary,
)
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, VAL


def _train(n=200, columns=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, columns)) * np.arange(1, columns + 1),
        columns=[f"x{i}" for i in range(columns)],
        index=[f"t{i:03d}" for i in range(n)],
    )


# --- acceptance ---------------------------------------------------------------------------------


def test_new_points_on_top_of_the_training_points_have_di_zero_and_are_all_inside():
    train = _train()
    result = dissimilarity_index(train, train.copy())
    assert (result.di_new == 0).all()
    assert result.inside_new.all()
    assert result.threshold > 0
    assert (mess(train, train)["mess"] >= 0).all()


def test_new_points_shifted_far_away_have_a_large_di_and_are_all_outside():
    train = _train()
    shifted = train + 50 * train.std()
    result = dissimilarity_index(train, shifted)
    assert (result.di_new > 10).all()
    assert (result.di_new > 10 * result.threshold).all()
    assert not result.inside_new.any()
    assert (mess(train, shifted)["mess"] < -100).all()


# --- the index itself ---------------------------------------------------------------------------


def test_the_index_matches_a_hand_worked_one_dimensional_case():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    result = dissimilarity_index(train, pd.DataFrame({"x": [1.0, 10.0, 1.5]}))
    # Pairwise distances 1, 2, 3, 1, 2, 1: mean 10/6, in standardised units, which cancel in the ratio.
    assert result.d_mean * train["x"].std() == pytest.approx(10 / 6)
    assert result.di_new.tolist() == pytest.approx([0.0, 7 / (10 / 6), 0.5 / (10 / 6)])
    assert result.di_train.tolist() == pytest.approx([0.6] * 4)  # each one unit from its neighbour


def test_the_index_does_not_depend_on_the_units_of_the_inputs():
    train = _train()
    new = _train(50, seed=1)
    scale = pd.Series([1000.0, 0.001, 1.0, 7.0, 1.0, 3.0], index=train.columns)
    first = dissimilarity_index(train, new)
    second = dissimilarity_index(train * scale, new * scale)
    assert np.allclose(first.di_new, second.di_new) and first.threshold == pytest.approx(second.threshold)


def test_a_zero_weight_removes_an_input_and_only_the_weights_proportions_matter():
    train = _train()
    new = _train(50, seed=1)
    weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], index=train.columns)
    base = dissimilarity_index(train, new, weights)
    moved = new.assign(x5=new["x5"] + 1000.0)
    assert np.allclose(dissimilarity_index(train, moved, weights).di_new, base.di_new)
    assert np.allclose(dissimilarity_index(train, new, weights * 2).di_new, base.di_new)
    assert not np.allclose(dissimilarity_index(train, moved).di_new, dissimilarity_index(train, new).di_new)


def test_training_points_are_compared_with_points_outside_their_own_fold():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    folds = pd.Series([0, 0, 1, 1], index=train.index)
    by_fold = dissimilarity_index(train, train, train_folds=folds).di_train
    assert by_fold.tolist() == pytest.approx([1.2, 0.6, 0.6, 1.2])  # 2, 1, 1, 2 units, over 10/6
    assert (by_fold >= dissimilarity_index(train, train).di_train).all()


def test_a_fold_of_near_duplicates_is_far_only_when_the_folds_are_respected():
    base = _train(60)
    copies = base.iloc[:20] + 1e-6
    copies.index = [f"c{i}" for i in range(20)]
    train = pd.concat([base, copies])
    folds = pd.Series(np.r_[np.arange(60) % 3, np.zeros(20)], index=train.index)
    folds.iloc[:20] = 0  # each original in the same fold as its copy
    leave_one_out = dissimilarity_index(train, train).di_train
    by_fold = dissimilarity_index(train, train, train_folds=folds).di_train
    assert leave_one_out.iloc[:20].max() < 1e-5  # its copy is right there
    assert by_fold.iloc[:20].min() > 0.05  # but the copy is in its own fold


def test_the_threshold_is_the_upper_whisker_of_the_training_indices():
    values = np.array([0.1, 0.2, 0.3, 0.4, 10.0])
    assert _threshold(values) == pytest.approx(0.4)  # Q75 + 1.5 IQR = 0.7: 10 is an outlier
    assert _threshold(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.3)

    train = _train()
    lone = pd.DataFrame([train.max() * 3], columns=train.columns, index=["far"])
    result = dissimilarity_index(pd.concat([train, lone]), train)
    assert result.di_train["far"] > result.threshold and not result.inside_train["far"]


def test_constant_inputs_are_dropped_gaps_filled_and_categories_refused():
    train = _train().assign(flat=5.0)
    new = _train(30, seed=1).assign(flat=9.0)
    result = dissimilarity_index(train, new)
    assert result.dropped == ["flat"] and "flat" not in result.columns
    assert result.weights.sum() == pytest.approx(1.0)

    gappy = new.copy()
    gappy.iloc[0, 0] = np.nan
    filled = new.copy()
    filled.iloc[0, 0] = train["x0"].median()
    assert np.allclose(dissimilarity_index(train, gappy).di_new, dissimilarity_index(train, filled).di_new)

    with pytest.raises(ValueError, match="non-numeric"):
        dissimilarity_index(train.assign(soil="A"), new.assign(soil="B"))
    with pytest.raises(ValueError, match="negative"):
        dissimilarity_index(train, new, pd.Series(-1.0, index=train.columns))
    with pytest.raises(ValueError, match="Every weight is 0"):
        dissimilarity_index(train, new, pd.Series(0.0, index=train.columns))


# --- MESS ------------------------------------------------------------------------------------------


def test_mess_follows_each_branch_of_its_formula():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    new = pd.DataFrame({"a": [-3.0, 0.0, 0.5, 2.5, 4.0, np.nan]})
    values = mess(train, new)["mess"].tolist()
    # f = 0 below: 100 (p - min) / range; f = 25: 2f; f = 75: 2 (100 - f); f = 100: 100 (max - p) / range.
    assert values[:5] == pytest.approx([-100.0, 0.0, 50.0, 50.0, -100 / 3])
    assert np.isnan(values[5])


def test_mess_names_the_input_that_lies_outside():
    train = _train()
    new = _train(20, seed=1)
    new["x3"] = train["x3"].max() + 100
    out = mess(train, new)
    assert (out["mess_variable"] == "x3").all() and (out["mess"] < 0).all()


# --- feature importances ----------------------------------------------------------------------


def _fitted(model, X, y, categorical=()):
    numeric = [column for column in X.columns if column not in categorical]
    return PipelineBuilder().build(model, categorical_cols=list(categorical), numeric_cols=numeric).fit(X, y)


def test_importances_come_back_per_input_and_count_the_gap_flag_for_its_column():
    rng = np.random.default_rng(0)
    X = _train(300).assign(soil=rng.choice(["A", "B"], 300))
    X.loc[X.index[::4], "x0"] = np.nan
    y = X["x0"].fillna(0) * 3 + X["x1"] + X["x0"].isna() * 5
    pipeline = _fitted(XGBRegressor(n_estimators=30, random_state=0), X, y, categorical=["soil"])
    numeric = [column for column in X.columns if column != "soil"]
    weights = importance_weights(pipeline, numeric)
    assert list(weights.index) == numeric and weights.sum() == pytest.approx(1.0)

    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    raw = pd.Series(pipeline.named_steps["model"].feature_importances_, index=names)
    numeric_total = raw[[name for name in names if name.startswith("num__")]].sum()
    expected = (raw["num__x0"] + raw["num__missingindicator_x0"]) / numeric_total
    assert weights["x0"] == pytest.approx(expected) and raw["num__missingindicator_x0"] > 0
    assert weights["x0"] > weights["x5"]


def test_a_model_without_importances_gives_no_weights():
    X = _train(50)
    assert importance_weights(_fitted(LinearRegression(), X, X["x0"]), list(X.columns)) is None


# --- one plan and the region summary ------------------------------------------------------------


def test_a_plans_table_holds_every_point_with_its_training_or_held_out_index():
    X = pd.concat([_train(100), _train(40, seed=1) + 20]).assign(soil="A")
    X.index = [f"p{i}" for i in range(len(X))]
    split = pd.Series([TRAIN] * 90 + [VAL] * 10 + [TEST] * 40, index=X.index)
    frame = plan_dissimilarity(
        X, split, train_ids=X.index[:90], categorical=["soil"], folds=pd.Series(np.arange(90) % 3, index=X.index[:90])
    )
    assert list(frame["point_id"]) == list(X.index)
    assert (frame.loc[frame["split"] == TRAIN, "mess"] >= 0).all()
    assert not frame.loc[frame["split"] == TEST, "inside"].any()  # 20 units away
    assert json.loads(frame["left_out"].iloc[0]) == {"categorical": ["soil"], "constant": []}
    assert (frame["n_columns"] == 6).all()


def _summary_frame(region_classes, train_classes, *, region_som, region_cec, train_som, train_cec, inside):
    def part(split, classes, som, cec):
        return pd.DataFrame(
            {
                "plan_id": "logo_A",
                "held_out": "A",
                "point_id": [f"{split}{i}" for i in range(len(classes))],
                "split": split,
                "di": np.linspace(0.1, 1.0, len(classes)),
                "inside": True,
                "mess": 10.0,
                "mess_variable": "x0",
                "carbonate_class": classes,
                "som": som,
                "cec": cec,
                "threshold": 0.5,
                "d_mean": 1.0,
                "n_columns": 3,
                "left_out": "{}",
            }
        )

    region = part(TEST, region_classes, region_som, region_cec)
    region["inside"] = inside
    return pd.concat([part(TRAIN, train_classes, train_som, train_cec), region], ignore_index=True)


def test_the_region_summary_reports_class_mix_correlations_and_share_outside():
    labels = ["<2", "2-10", "10-25", ">=25"]
    som = np.arange(10.0)
    frame = _summary_frame(
        ["<2"] * 5 + ["2-10"] * 5,
        ["<2"] * 10 + [">=25"] * 10,
        region_som=som,
        region_cec=-som,  # the opposite relation in the region
        train_som=np.arange(20.0),
        train_cec=np.arange(20.0) ** 2,
        inside=[True] * 7 + [False] * 3,
    )
    row = region_summary(frame, labels=labels).iloc[0]
    assert row["n_test"] == 10 and row["share_outside"] == pytest.approx(0.3)
    assert row["share_<2_region"] == pytest.approx(0.5) and row["share_>=25_train"] == pytest.approx(0.5)
    assert row["carbonate_tvd"] == pytest.approx(0.5)  # half the region's mix is a class training has, half not
    assert row["spearman_som_cec_train"] == pytest.approx(1.0)
    assert row["spearman_som_cec_region"] == pytest.approx(-1.0)
    assert row["spearman_som_cec_diff"] == pytest.approx(-2.0)


def test_the_same_class_mix_has_no_distance_and_too_few_pairs_no_correlation():
    frame = _summary_frame(
        ["<2", "2-10"],
        ["<2", "2-10", "<2", "2-10"],
        region_som=[1.0, 2.0],
        region_cec=[1.0, 2.0],
        train_som=[1.0, 2.0, 3.0, 4.0],
        train_cec=[1.0, 2.0, 3.0, np.nan],
        inside=True,
    )
    row = region_summary(frame).iloc[0]
    assert row["carbonate_tvd"] == pytest.approx(0.0) and row["share_outside"] == 0
    assert np.isnan(row["spearman_som_cec_region"])  # two pairs only
    assert row["spearman_som_cec_train"] == pytest.approx(1.0)  # three complete pairs
