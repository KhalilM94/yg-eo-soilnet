"""Prediction intervals for SOM: the quantile model global_q, conformalized (CQR), and its scores.

An interval is only worth reporting if its quantiles are in order, its calibration never uses a
point the model saw, its coverage is what the scores say, and a run without it is exactly the run
it was before.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from experiments.a1.evaluate import main as evaluate_main
from experiments.a1.make_figures import main as figures_main
from yg_eo_soilnet.a1 import devices as devices_module
from yg_eo_soilnet.a1 import pipelines as pipelines_module
from yg_eo_soilnet.a1 import uncertainty as uq
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.evaluation import load_predictions
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM, SPATIAL_KFOLD, write_plans
from yg_eo_soilnet.a1.runner import (
    CALIBRATION_DIR,
    DESIGN_DISSIMILARITY_DIR,
    DISSIMILARITY_SUMMARY_FILE,
    MANIFEST_FILE,
    run_plans,
)
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.a1.stage2 import out_of_fold_quantiles, predict_quantiles
from yg_eo_soilnet.datamodules.splitting import TEST

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
LEVELS = [0.5, 0.8, 0.9]

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    "inner_cv_folds": 3,
}


def _stage2(models):
    return {
        "estimator": "XGBoost",
        "init_args": {"n_estimators": 30},
        "params": {"model__max_depth": [2, 3]},
        "models": list(models),
    }


def _plans(cell_sizes=(10000, 25000)):
    return {
        "spatial_kfold": {"method": "grid", "cell_size_m": 10000, "cell_sizes_m": list(cell_sizes), "n_folds": 3},
        "logo": {"min_points": 100, "held_out": ["North-East"], "validation_region": False},
        "mixed": {"fractions": [0.2, 0.5], "n_repeats": 2},
    }


def _run(builder, folder: Path, *, models, uncertainty=True, kinds=(RANDOM, SPATIAL_KFOLD, LOGO), main=None):
    extra = {"uncertainty": {"quantiles": QUANTILES, "levels": LEVELS, "conformal": True}} if uncertainty else {}
    path = builder(folder)(
        main_overrides=main,
        as_file=True,
        output_dir=str(folder / "out"),
        plans=_plans(),
        stage2=_stage2(models),
        **FAST,
        **extra,
    )
    a1 = load_a1_config(str(path))
    plan_kinds = [kind for kind in (RANDOM, SPATIAL_KFOLD, LOGO, MIXED) if kind in kinds]
    write_plans(a1.settings, str(folder / "out" / "plans"), kinds=plan_kinds)
    result = run_plans(a1, kinds=list(kinds), tracking=False)
    return result.run_dir, path


@pytest.fixture(scope="module")
def builder(a1_settings_in):
    def build(folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        return a1_settings_in(folder)

    return build


@pytest.fixture(scope="module")
def runs(builder, tmp_path_factory):
    """Demo runs: global alone (with and without an uncertainty block), and global with global_q."""
    root = tmp_path_factory.mktemp("a1_uncertainty")
    plain, _ = _run(builder, root / "plain", models=["global"], uncertainty=False)
    plain_block, _ = _run(builder, root / "plain_block", models=["global"], uncertainty=True)
    with_q, with_q_path = _run(
        builder, root / "with_q", models=["global", "global_q"], kinds=(RANDOM, SPATIAL_KFOLD, LOGO, MIXED, CONTROL)
    )
    return {"plain": plain, "plain_block": plain_block, "with_q": with_q, "with_q_path": with_q_path}


@pytest.fixture(scope="module")
def tables(runs):
    return evaluate_main(["--config", str(runs["with_q_path"]), "--n-boot", "40", "--no-mlflow"])


def _quantile_rows(run_dir):
    frame = load_predictions(run_dir)
    return frame[frame["model"] == uq.GLOBAL_Q]


# --- settings ----------------------------------------------------------------------------------------


def test_the_quantile_model_is_off_by_default_and_the_intervals_take_the_paper_settings(a1_settings):
    settings = a1_settings()
    assert "global_q" not in settings.som_models
    assert settings.uncertainty.quantiles == tuple(QUANTILES)
    assert settings.uncertainty.levels == tuple(LEVELS) and settings.uncertainty.conformal is True


@pytest.mark.parametrize(
    "values, message",
    [
        ({"quantiles": [0.1, 0.9]}, "include 0.5"),
        ({"quantiles": [0.1, 0.5, 0.9], "levels": [0.9]}, "needs the quantiles"),
        ({"quantiles": [0.5, 0.25, 0.75]}, "increasing"),
        ({"quantiles": [0.0, 0.5]}, "strictly between"),
        ({"conformal": "yes"}, "true or false"),
    ],
)
def test_bad_uncertainty_settings_are_refused(a1_settings, values, message):
    with pytest.raises(ValueError, match=message):
        a1_settings(uncertainty=values)


def test_global_q_needs_xgboost_in_stage_2(a1_settings):
    with pytest.raises(ValueError, match="XGBoost"):
        a1_settings(stage2={"estimator": "RandomForest", "models": ["global", "global_q"]})


def test_the_round_3_configurations_list_global_q():
    assert "global_q" in load_settings(str(A1_CONFIGS / "a1_xgb.yml")).som_models
    for name in ("a1_mixed_arms.yml", "a1_mixed_contrast.yml"):
        assert load_settings(str(A1_CONFIGS / name)).som_models == ("global", "global_q")


# --- quantiles and conformal calibration --------------------------------------------------------


def test_quantiles_are_non_decreasing_after_sorting(runs):
    rows = _quantile_rows(runs["with_q"])
    values = rows[[uq.quantile_column(q) for q in QUANTILES]].to_numpy(float)
    assert (np.diff(values, axis=1) >= 0).all()
    assert np.allclose(rows["predicted"], rows["som_q50"])
    raw = np.array([[3.0, 1.0, 2.0]])
    sorted_values, crossed = uq.sort_quantiles(raw)
    assert sorted_values.tolist() == [[1.0, 2.0, 3.0]] and crossed.tolist() == [True]


def test_the_crossing_share_is_recorded(runs):
    rows = _quantile_rows(runs["with_q"])
    record = json.loads((runs["with_q"] / CALIBRATION_DIR / "random" / "seed_42.json").read_text())
    held = rows[rows["plan_id"] == "random"]
    assert record["crossing_share_held_out"] == pytest.approx(held[uq.SORTED_COLUMN].astype(bool).mean())
    assert set(record["levels_q"]) == {"50", "80", "90"}


@pytest.mark.parametrize("level", LEVELS)
def test_cqr_reaches_nominal_coverage_on_exchangeable_data(level):
    rng = np.random.default_rng(0)
    n_calibration, n_test = 2000, 40000
    x = rng.uniform(0, 10, n_calibration + n_test)
    y = x + rng.normal(scale=1.0 + 0.2 * x)
    # Deliberately poor raw quantiles: centred, but far too narrow.
    lower, upper = x - 0.3, x + 0.3
    calibration = uq.calibrate(y[:n_calibration], lower[:n_calibration], upper[:n_calibration], level)
    covered = (y[n_calibration:] >= lower[n_calibration:] - calibration.q) & (
        y[n_calibration:] <= upper[n_calibration:] + calibration.q
    )
    error = 3 * np.sqrt(level * (1 - level) / n_calibration)  # the calibration set's sampling error
    assert abs(covered.mean() - level) < error
    assert calibration.n_calibration == n_calibration and calibration.raw_coverage < level


def test_the_conformal_quantile_is_the_finite_sample_rank():
    scores = np.arange(1.0, 100.0)  # 99 scores: y = 0 lies below every interval, E_i = q_lo = score
    calibration = uq.calibrate(np.zeros(99), scores, scores + 1, 0.9)
    assert calibration.q == float(np.sort(scores)[int(np.ceil(100 * 0.9)) - 1])


class _FoldSpy(BaseEstimator, RegressorMixin):
    """Predicts 1 for a point it was fitted on, 0 otherwise, in every quantile column; remembers its rows."""

    fits: list = []

    def fit(self, X, y, **_):
        self.seen_ = frozenset(X.index)
        _FoldSpy.fits.append(self.seen_)
        return self

    def predict(self, X):
        flags = np.array([float(point in self.seen_) for point in X.index])
        return np.column_stack([flags] * 3)


def test_calibration_uses_only_out_of_fold_predictions():
    rng = np.random.default_rng(1)
    ids = pd.Index([f"p{i}" for i in range(120)], name="point_id")
    blocks = pd.Series(rng.integers(0, 12, len(ids)), index=ids)
    folds = blocks % 4  # whole blocks to a fold, as the plans' grouped folds are
    X = pd.DataFrame({"x": rng.normal(size=len(ids))}, index=ids)
    y = pd.Series(rng.normal(size=len(ids)), index=ids)
    _FoldSpy.fits = []
    oof = out_of_fold_quantiles(_FoldSpy(), {"tuned": False}, X, y, folds)
    assert len(_FoldSpy.fits) == 4
    assert (oof.to_numpy() == 0.0).all(), "a point was calibrated by a model fitted on it"
    for seen in _FoldSpy.fits:
        held = ids.difference(pd.Index(seen))
        assert not set(blocks[held]) & set(blocks[list(seen)]), "a block was on both sides of a fold"


def test_too_few_folds_give_no_calibration():
    ids = pd.Index(["a", "b", "c"])
    X, y = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=ids), pd.Series([1.0, 2.0, 3.0], index=ids)
    assert out_of_fold_quantiles(_FoldSpy(), {}, X, y, pd.Series(0, index=ids)) is None


def test_a_log_transformed_target_back_transforms_each_quantile_exactly(builder, tmp_path):
    from yg_eo_soilnet.a1.pipelines import final_estimator
    from yg_eo_soilnet.utils import LogTransformer

    run_dir, path = _run(
        builder,
        tmp_path / "log",
        models=["global", "global_q"],
        kinds=(RANDOM,),
        main={"COLUMNS_TO_TRANSFORM": ["organic_matter_g_kg"]},
    )
    rows = _quantile_rows(run_dir)
    values = rows[[uq.quantile_column(q) for q in QUANTILES]].to_numpy(float)
    assert (np.diff(values, axis=1) >= 0).all() and (values > -1).all()
    # The pipeline: quantiles predicted in log space, each one undone by the inverse transform.
    from yg_eo_soilnet.a1.features import build_table
    from yg_eo_soilnet.a1.pipelines import build_stage_pipeline
    from yg_eo_soilnet.a1.stage2 import quantile_stage

    a1 = load_a1_config(str(path))
    settings, config = a1.settings, a1.settings.load_config()
    table = build_table(settings, config)
    columns = list(table.feature_columns)[:5]
    pipeline, _ = build_stage_pipeline(
        settings, config, columns, stage=quantile_stage(settings), target=settings.target
    )
    X, y = table.frame[columns].iloc[:200], table.frame[settings.target].iloc[:200]
    pipeline.fit(X, y)
    log_space = final_estimator(pipeline).predict(pipeline[:-1].transform(X))
    np.testing.assert_allclose(predict_quantiles(pipeline, X), LogTransformer().inverse_transform(log_space), rtol=1e-6)


# --- what a run writes ---------------------------------------------------------------------------


def test_the_quantile_model_writes_its_columns_and_q_per_plan_and_seed(runs):
    rows = _quantile_rows(runs["with_q"])
    for level in LEVELS:
        for method in (uq.RAW, uq.CQR):
            low, high = uq.bound_columns(level, method)
            assert rows[low].notna().all() and (rows[high] >= rows[low]).all()
    for plan_id in rows["plan_id"].unique():
        path = runs["with_q"] / CALIBRATION_DIR / plan_id / "seed_42.json"
        record = json.loads(path.read_text())
        part = rows[rows["plan_id"] == plan_id]
        q80 = record["levels_q"]["80"]["q"]
        np.testing.assert_allclose(part["som_lo80_cqr"], part["som_lo80"] - q80)
        np.testing.assert_allclose(part["som_hi80_cqr"], part["som_hi80"] + q80)
    other = load_predictions(runs["with_q"])
    assert other.loc[other["model"] != uq.GLOBAL_Q, "som_q50"].isna().all()


def test_dissimilarity_is_computed_for_random_and_the_10_km_folds_only(runs):
    designs = sorted(path.stem for path in (runs["with_q"] / DESIGN_DISSIMILARITY_DIR).glob("*.parquet"))
    assert "random" in designs and any(name.startswith("kfold_10km_") for name in designs)
    assert not any(name.startswith("kfold_25km_") for name in designs)
    summary = pd.read_csv(runs["with_q"] / DISSIMILARITY_SUMMARY_FILE)
    assert set(summary["held_out"]) == {"North-East"}
    assert not (runs["plain"] / DESIGN_DISSIMILARITY_DIR).exists()


def test_a_run_without_global_q_writes_exactly_what_it_wrote_before(runs):
    def files(run_dir):
        return {
            str(path.relative_to(run_dir)): path.read_bytes()
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.suffix == ".parquet"
        }

    plain, with_block = files(runs["plain"]), files(runs["plain_block"])
    assert plain.keys() == with_block.keys()
    assert all(plain[name] == with_block[name] for name in plain)
    assert not (runs["plain_block"] / CALIBRATION_DIR).exists()
    assert not (runs["plain_block"] / DESIGN_DISSIMILARITY_DIR).exists()
    manifest = json.loads((runs["plain_block"] / MANIFEST_FILE).read_text())
    assert "global_q" not in manifest["devices"]
    # Adding global_q leaves every other model's predictions as they were.
    with_q = files(runs["with_q"])
    for name, content in plain.items():
        if name.startswith("predictions/") and name in with_q:
            assert with_q[name] == content, name


# --- the graphics card ------------------------------------------------------------------------------


def test_a_quantile_objective_that_falls_back_moves_global_q_alone_to_the_processor(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(pipelines_module, "stage_device", lambda settings, config, stage: "cuda")
    monkeypatch.setattr(devices_module, "check_cuda", lambda device="cuda": None)
    monkeypatch.setattr(devices_module, "check_cuda_quantiles", lambda device, quantiles: "XGBoost fell back")
    run_dir, _ = _run(builder, tmp_path / "fallback", models=["global", "global_q"], kinds=(RANDOM,))
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text())
    assert manifest["devices"]["global_q"] == "cpu" and manifest["devices"]["stage2"] == "cuda"
    frame = load_predictions(run_dir)
    assert set(frame.loc[frame["model"] == uq.GLOBAL_Q, "device_stage2"]) == {"cpu"}
    assert set(frame.loc[frame["model"] == "global", "device_stage2"]) == {"cuda"}
    record = json.loads((run_dir / CALIBRATION_DIR / "random" / "seed_42.json").read_text())
    assert record["device"] == "cpu"


def _has_gpu() -> bool:
    try:
        import xgboost

        return bool(xgboost.build_info().get("USE_CUDA")) and devices_module.check_cuda_quantiles("cuda") is None
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="no usable graphics card")
def test_the_quantile_objective_runs_on_the_card():
    assert devices_module.check_cuda_quantiles("cuda", QUANTILES) is None


# --- evaluation ------------------------------------------------------------------------------------


def test_picp_piw_and_qcp_equal_a_hand_computation_for_one_plan(runs, tables):
    scores = tables["uncertainty_scores.csv"]
    rows = _quantile_rows(runs["with_q"])
    test = rows[(rows["plan_id"] == "random") & (rows["split"] == TEST)]
    y = test["observed"].to_numpy(float)
    mine = scores[(scores["scope"] == "random") & (scores["class"] == "all")]

    def value(metric, method=uq.RAW, level=np.nan, quantile=np.nan):
        chosen = mine[(mine["metric"] == metric) & (mine["method"] == method)]
        if not np.isnan(level):
            chosen = chosen[np.isclose(chosen["level"], level)]
        if not np.isnan(quantile):
            chosen = chosen[np.isclose(chosen["quantile"], quantile)]
        assert len(chosen) == 1
        return chosen.iloc[0]

    for level in LEVELS:
        for method in (uq.RAW, uq.CQR):
            low, high = uq.bound_columns(level, method)
            inside = (y >= test[low]) & (y <= test[high])
            row = value("picp", method, level)
            assert row["value"] == pytest.approx(inside.mean())
            assert row["low"] <= row["value"] <= row["high"]
            assert value("piw_mean", method, level)["value"] == pytest.approx((test[high] - test[low]).mean())
            assert value("piw_median", method, level)["value"] == pytest.approx((test[high] - test[low]).median())
    for q in QUANTILES:
        assert value("qcp", quantile=q)["value"] == pytest.approx((y <= test[uq.quantile_column(q)]).mean())
    values = test[[uq.quantile_column(q) for q in QUANTILES]].to_numpy(float)
    gap = y[:, None] - values
    tau = np.array(QUANTILES)[None, :]
    assert value("pinball")["value"] == pytest.approx(np.maximum(tau * gap, (tau - 1) * gap).mean())
    assert value("rmse_median")["value"] == pytest.approx(np.sqrt(((test["predicted"] - y) ** 2).mean()))


def test_the_scores_follow_the_pooling_rules(tables):
    scores = tables["uncertainty_scores.csv"]
    scopes = set(scores["scope"])
    assert {"random", "spatial_kfold_10000m", "spatial_kfold_25000m", "logo_North-East"} <= scopes
    assert {"mixed_North-East_f0.2", "control_North-East_f0.5"} <= scopes
    mixed = scores[(scores["scope"] == "mixed_North-East_f0.2") & (scores["class"] == "all")]
    assert (mixed["n"] > 0).all()
    classes = scores[scores["class"] != "all"]
    assert (classes["n"] >= 30).all()


def test_global_q_is_left_out_of_every_accuracy_table(tables):
    for name in ("scores.csv", "scores_by_class.csv", "intervals.csv", "differences.csv", "scores_mixed.csv"):
        table = tables[name]
        for column in ("model", "a", "b"):
            if column in table:
                assert "global_q" not in set(table[column]), name


def test_the_coverage_headlines_are_logged(runs, tables):
    summary = tables["seed_summary.csv"]
    names = set(summary["name"])
    assert "picp80__random__global_q" in names and "piw80__random__global_q" in names
    assert "picp80_raw__random__global_q" in names
    # The interval bounds go to MLflow with the headline metrics (seed_summary.csv keeps values only).
    headline = {row["name"]: row["value"] for row in uq.headline_rows(tables["uncertainty_scores.csv"])}
    for bound in ("lo", "hi"):
        assert f"picp80__random__global_q__{bound}" in headline and f"piw80__random__global_q__{bound}" in headline
    assert headline["picp80__random__global_q__lo"] <= headline["picp80__random__global_q"]


def test_a_run_without_global_q_has_no_uncertainty_table(runs):
    tables = evaluate_main(
        ["--config", str(Path(runs["plain"]).parents[2] / "a1.yml"), "--n-boot", "20", "--no-mlflow"]
    )
    assert "uncertainty_scores.csv" not in tables
    assert not (runs["plain"] / "evaluation" / "uncertainty_scores.csv").exists()


# --- figures ---------------------------------------------------------------------------------------


def test_the_uncertainty_figures_render(runs, tables, tmp_path):
    out = tmp_path / "figures"
    figures_main(
        [
            "--config",
            str(runs["with_q_path"]),
            "--only",
            "figU1_bands,figU2_reliability,figU3_width,figU4_mixed_coverage",
            "--no-mlflow",
            "--no-relief",
            "--n-boot",
            "20",
            "--out",
            str(out),
        ]
    )
    for name in ("figU1_bands", "figU2_reliability", "figU3_width", "figU4_mixed_coverage"):
        assert (out / f"{name}.png").exists(), name
        assert (out / "data" / f"{name}.csv").exists(), name
    captions = (out / "captions.md").read_text()
    assert "average over the repeats' test points" in captions
    data = pd.read_csv(out / "data" / "figU3_width.csv")
    assert data["di"].notna().any()


def test_no_uncertainty_figure_without_global_q(runs, tmp_path):
    out = tmp_path / "figures"
    figures_main(
        [
            "--config",
            str(Path(runs["plain"]).parents[2] / "a1.yml"),
            "--only",
            "figU1_bands,fig05_spatial",
            "--no-mlflow",
            "--no-relief",
            "--n-boot",
            "20",
            "--out",
            str(out),
        ]
    )
    assert not list(out.glob("figU*"))
    assert (out / "fig05_spatial.png").exists()
