"""Seeds, the SOM models to fit, plan selection, parallel workers and the processor of each stage.

A fast configuration - XGBoost in both stages with 30 trees, two inner folds - runs on the demo's
random plan and one leave-one-region-out plan, with two seeds.
"""

import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import runner
from yg_eo_soilnet.a1.config import COMBINED_GROUP, load_a1_config
from yg_eo_soilnet.a1.devices import (
    DeviceMismatch,
    DeviceUnavailable,
    check_cuda,
    check_same_devices,
    is_cuda,
)
from yg_eo_soilnet.a1.evaluation import load_predictions, same_sign_counts, seed_order
from yg_eo_soilnet.a1.layout import find_stage1, prediction_dir, prediction_paths, stage1_path
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, write_plans
from yg_eo_soilnet.a1.runner import MANIFEST_FILE, run_plans, select_plans
from yg_eo_soilnet.a1.settings import SOM_MODELS, dev_mode_settings, load_settings
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL, MODELS, needs_stage1

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 30}},
    "stage2": {
        "estimator": "XGBoost",
        "init_args": {"n_estimators": 30},
        # Two grid points: the search, and so the seed's folds, matter for every model.
        "params": {"model__max_depth": [2, 4]},
    },
    "inner_cv_folds": 3,
    "seeds": [42, 7],
}
#: The plans the module's run covers: the random split and one held-out region.
PLANS = ["random", "logo_r04_south_west"]


def _hashes(folder: Path) -> dict:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.glob("*.parquet"))}


@pytest.fixture(scope="module")
def two_seeds(a1_settings_in, tmp_path_factory):
    """One fast run of two plans and two seeds, without MLflow."""
    folder = tmp_path_factory.mktemp("a1_seeds")
    path = a1_settings_in(folder)(as_file=True, output_dir=str(folder / "out"), **FAST)
    a1 = load_a1_config(str(path))
    write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[RANDOM, LOGO])
    result = run_plans(a1, only=PLANS, tracking=False)
    return {"a1": a1, "result": result, "path": path, "folder": folder}


# --- seeds ---------------------------------------------------------------------------------------


def test_adding_seeds_changes_no_plan_and_no_plan_hash(a1_settings, tmp_path):
    one = a1_settings(output_dir=str(tmp_path / "one"), seeds=[42])
    three = a1_settings(output_dir=str(tmp_path / "three"), seeds=[42, 7, 2026])
    write_plans(one, str(tmp_path / "one"), kinds=[RANDOM, LOGO, "spatial_kfold"])
    write_plans(three, str(tmp_path / "three"), kinds=[RANDOM, LOGO, "spatial_kfold"])
    assert _hashes(tmp_path / "one") and _hashes(tmp_path / "one") == _hashes(tmp_path / "three")
    assert (tmp_path / "one" / "index.csv").read_text() == (tmp_path / "three" / "index.csv").read_text()


def test_two_seeds_give_two_prediction_sets_for_the_same_plan(two_seeds):
    run_dir = two_seeds["result"].run_dir
    for seed in (42, 7):
        for plan_id in PLANS:
            files = {path.name for path in prediction_dir(run_dir, plan_id, seed).glob("*.parquet")}
            assert {f"{model}__{COMBINED_GROUP}.parquet" for model in MODELS} <= files
            assert stage1_path(run_dir, plan_id, seed).exists()
    frame = load_predictions(run_dir)
    assert set(frame["seed"]) == {42, 7}
    assert seed_order(frame, 7) == [7, 42] and seed_order(frame, 42) == [42, 7]
    # The same test points under both seeds...
    for (plan_id, model), part in frame.groupby(["plan_id", "model"]):
        by_seed = part.groupby("seed")["point_id"].apply(frozenset)
        assert by_seed[42] == by_seed[7], (plan_id, model)
    # ...but other fits: the stage-1 folds differ, so the predicted CaCO3 and CEC do.
    aux = frame[(frame["model"] == GLOBAL_AUX) & (frame["plan_id"] == "random")].pivot(
        index="point_id", columns="seed", values="predicted"
    )
    assert not np.allclose(aux[42], aux[7])


def test_further_seeds_deal_whole_blocks_into_other_folds(a1_settings, tmp_path):
    from yg_eo_soilnet.a1.auxiliary import inner_folds
    from yg_eo_soilnet.datamodules.splitting import SplitPlan

    settings = a1_settings(seeds=[42, 7])
    write_plans(settings, str(tmp_path), kinds=["spatial_kfold"])
    path = sorted(tmp_path.glob("kfold_*_f0.parquet"))[0]
    plan = SplitPlan.from_frame(pd.read_parquet(path), strategy="spatial_kfold")
    folds = {seed: inner_folds(plan, settings.for_seed(seed), kind="spatial_kfold")[0] for seed in (42, 7)}
    unseeded, _ = inner_folds(plan, settings, kind="spatial_kfold")
    pd.testing.assert_series_equal(folds[42], unseeded)  # the first seed: the folds of every earlier run
    blocks = plan.clusters.reindex(folds[42].index)
    partitions = {}
    for seed, fold in folds.items():
        assert (fold.groupby(blocks).nunique() == 1).all()  # every block whole, in one fold
        partitions[seed] = {frozenset(blocks[fold == k]) for k in fold.unique()}
    assert partitions[42] != partitions[7]


def test_the_manifest_records_the_seeds_done_per_plan_and_the_devices(two_seeds):
    manifest = json.loads((two_seeds["result"].run_dir / MANIFEST_FILE).read_text())
    assert manifest["format"] == 5
    assert manifest["seeds"] == [42, 7]
    assert manifest["seeds_done"] == {plan_id: [7, 42] for plan_id in PLANS}
    assert manifest["devices"] == {"stage1": "cpu", "stage2": "cpu"}
    assert manifest["models"] == list(SOM_MODELS)


def test_every_row_records_its_seed_and_devices(two_seeds):
    path = prediction_dir(two_seeds["result"].run_dir, "random", 7) / f"{GLOBAL}__{COMBINED_GROUP}.parquet"
    frame = pd.read_parquet(path)
    assert set(frame["seed"]) == {7}
    assert set(frame["device_stage1"]) == {"cpu"} and set(frame["device_stage2"]) == {"cpu"}


def test_a_run_resumes_per_plan_and_seed(two_seeds, monkeypatch):
    run_dir = two_seeds["result"].run_dir
    target = prediction_dir(run_dir, "random", 7) / f"{GLOBAL}__{COMBINED_GROUP}.parquet"
    before = pd.read_parquet(target)
    target.unlink()
    again = run_plans(two_seeds["a1"], only=PLANS, tracking=False)
    assert again.computed == [str(target)]
    pd.testing.assert_frame_equal(pd.read_parquet(target), before)
    only_42 = run_plans(two_seeds["a1"], only=PLANS, seeds=[42], tracking=False)
    assert only_42.computed == []
    assert all("seed_42" in path or "dissimilarity" in path for path in only_42.skipped)
    assert not any("seed_7" in path for path in only_42.skipped)


def test_adding_a_seed_to_the_settings_adds_to_the_same_run_folder(a1_settings_in, tmp_path):
    builder = a1_settings_in(tmp_path)
    path = builder(as_file=True, output_dir=str(tmp_path / "out"), **{**FAST, "seeds": [42]})
    a1 = load_a1_config(str(path))
    write_plans(a1.settings, str(tmp_path / "out" / "plans"), kinds=[RANDOM])
    first = run_plans(a1, kinds=[RANDOM], tracking=False)
    path = builder(as_file=True, output_dir=str(tmp_path / "out"), **FAST)  # seeds [42, 7]
    second = run_plans(load_a1_config(str(path)), kinds=[RANDOM], tracking=False)
    assert first.computed and all("seed_42" in name for name in second.skipped)
    assert second.computed and all("seed_7" in name for name in second.computed)


def test_a_seed_the_settings_do_not_list_is_refused(two_seeds):
    with pytest.raises(ValueError, match="not among the settings' seeds"):
        run_plans(two_seeds["a1"], seeds=[3], tracking=False)
    with pytest.raises(ValueError, match="not among the settings' seeds"):
        two_seeds["a1"].settings.for_seed(3)


def test_the_first_seed_keeps_the_estimators_seed_and_further_seeds_reseed_them(two_seeds):
    from yg_eo_soilnet.a1.pipelines import build_stage_pipeline, final_estimator

    settings = two_seeds["a1"].settings
    config = settings.load_config()
    states = {}
    for seed in (42, 7):
        pipeline, _ = build_stage_pipeline(settings.for_seed(seed), config, ["x"], stage=settings.stage2)
        states[seed] = final_estimator(pipeline).get_params()["random_state"]
    assert states == {42: 42, 7: 7}
    assert not settings.for_seed(42).shuffle_groups and settings.for_seed(7).shuffle_groups


# --- evaluation over seeds ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(two_seeds):
    from experiments.a1 import evaluate

    run_dir = two_seeds["result"].run_dir
    return evaluate._evaluate(run_dir, 10.0, 0, 30, None, first_seed=42)


def test_every_headline_table_has_a_seed_column_the_first_seed_first(evaluated):
    for name in ("scores.csv", "scores_by_class.csv", "intervals.csv", "differences.csv", "transfer.csv"):
        table = evaluated[name]
        assert list(table.columns)[0] == "seed", name
        assert set(table["seed"]) == {42, 7}, name
        assert table["seed"].iloc[0] == 42, name


def test_seed_summary_has_one_row_per_metric_and_model(evaluated):
    summary = evaluated["seed_summary.csv"]
    assert not summary.empty and not summary["name"].duplicated().any()
    assert {"mean", "sd", "min", "max", "first", "first_seed", "n_seeds"} <= set(summary.columns)
    random_rmse = summary[(summary["metric"] == "rmse") & (summary["scope"] == "random")]
    assert set(random_rmse["model"]) == set(MODELS)
    assert (random_rmse["n_seeds"] == 2).all() and (random_rmse["first_seed"] == 42).all()
    row = random_rmse.iloc[0]
    assert row["min"] <= row["mean"] <= row["max"] and row["sd"] >= 0
    scores = evaluated["scores.csv"]
    per_seed = scores[
        (scores["kind"] == RANDOM) & (scores["model"] == row["model"]) & (scores["covariate_group"] == COMBINED_GROUP)
    ]
    assert row["mean"] == pytest.approx(per_seed["rmse"].mean())
    assert row["first"] == pytest.approx(per_seed.loc[per_seed["seed"] == 42, "rmse"].iloc[0])


def test_each_model_difference_counts_the_seeds_that_keep_its_sign(evaluated):
    differences = evaluated["differences.csv"]
    assert differences["n_seeds_same_sign"].between(1, 2).all()  # the first seed always agrees with itself
    assert (differences["n_seeds"] == 2).all()


def test_same_sign_counts_compares_each_seed_with_the_first():
    rows = pd.DataFrame(
        {
            "seed": [42, 42, 7, 7, 9, 9],
            "a": ["x", "y"] * 3,
            "b": ["g"] * 6,
            "metric": ["rmse"] * 6,
            "estimate": [-1.0, 0.5, -0.2, -0.3, -2.0, 0.1],
        }
    )
    counted = same_sign_counts(rows).set_index(["seed", "a"])
    assert counted.loc[(42, "x"), "n_seeds_same_sign"] == 3
    assert counted.loc[(42, "y"), "n_seeds_same_sign"] == 2


def test_the_figures_show_the_first_seed_only(two_seeds):
    from yg_eo_soilnet.a1.figures.core import FigureInputs

    inputs = FigureInputs(
        a1=two_seeds["a1"],
        run_dir=two_seeds["result"].run_dir,
        plans_dir=two_seeds["result"].run_dir.parents[1] / "plans",
    )
    assert set(inputs.predictions["seed"]) == {42}
    assert inputs.fit_seed == 42


def test_a_folder_of_the_older_layout_reads_as_its_first_seed(tmp_path, two_seeds):
    source = prediction_dir(two_seeds["result"].run_dir, "random", 42) / f"{GLOBAL}__{COMBINED_GROUP}.parquet"
    legacy = tmp_path / "predictions" / "random"
    legacy.mkdir(parents=True)
    pd.read_parquet(source).drop(columns=["seed", "device_stage1", "device_stage2"]).to_parquet(legacy / source.name)
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "random.parquet").write_bytes(b"")
    frame = load_predictions(tmp_path, first_seed=42)
    assert set(frame["seed"]) == {42} and set(frame["device_stage2"]) == {"unknown"}
    assert find_stage1(tmp_path, "random", 42) == tmp_path / "stage1" / "random.parquet"
    assert prediction_paths(tmp_path) == [legacy / source.name]


# --- the SOM models to fit --------------------------------------------------------------------------


def test_stage2_models_must_be_known_and_only_stage2_lists_them(a1_settings):
    assert a1_settings().som_models == SOM_MODELS
    chosen = a1_settings(stage2={"estimator": "XGBoost", "models": ["per_class_predicted", "global"]})
    assert chosen.som_models == (GLOBAL, "per_class_predicted")  # in the usual order
    assert dev_mode_settings(chosen).som_models == chosen.som_models
    with pytest.raises(ValueError, match="stage2.models"):
        a1_settings(stage2={"estimator": "XGBoost", "models": ["global", "lasso"]})
    with pytest.raises(ValueError, match="only stage2"):
        a1_settings(stage1={"estimator": "XGBoost", "models": ["global"]})
    assert not needs_stage1([GLOBAL]) and needs_stage1([GLOBAL, "per_class_measured"])


def test_global_alone_runs_no_stage_1(a1_settings_in, tmp_path, monkeypatch):
    path = a1_settings_in(tmp_path)(
        as_file=True,
        output_dir=str(tmp_path / "out"),
        **{**FAST, "seeds": [42], "stage2": {**FAST["stage2"], "models": [GLOBAL]}},
    )
    a1 = load_a1_config(str(path))
    write_plans(a1.settings, str(tmp_path / "out" / "plans"), kinds=[RANDOM, LOGO])
    monkeypatch.setattr(
        runner, "predict_auxiliary", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stage 1 ran"))
    )
    result = run_plans(a1, only=PLANS, groups=True, tracking=False)
    run_dir = result.run_dir
    assert not (run_dir / "stage1").exists()
    frame = load_predictions(run_dir)
    assert set(frame["model"]) == {GLOBAL} and set(frame["covariate_group"]) == {COMBINED_GROUP}
    assert LOCAL not in set(frame["model"])
    assert set(frame["device_stage1"]) == {"none"}
    assert (run_dir / "dissimilarity" / "logo_r04_south_west.parquet").exists()
    assert any("--groups" in note for note in result.notes)

    from experiments.a1 import evaluate

    tables = evaluate._evaluate(run_dir, 10.0, 0, 20, None, first_seed=42)
    assert set(tables["scores.csv"]["model"]) == {GLOBAL}


# --- choosing plans -------------------------------------------------------------------------------


def test_only_selects_the_plans_matching_a_pattern():
    index = pd.DataFrame(
        {"plan_id": ["random", "kfold_10km_f0", "kfold_10km_f1", "kfold_25km_f0", "logo_r01_a", "mixed_r01_a_p05_s0"]}
    )
    chosen = select_plans(index, ["random", "kfold_10km_*", "logo_*"])["plan_id"].tolist()
    assert chosen == ["random", "kfold_10km_f0", "kfold_10km_f1", "logo_r01_a"]
    with pytest.raises(ValueError, match="matches none"):
        select_plans(index, ["nothing_*"])


def test_only_limits_what_a_run_fits(two_seeds, monkeypatch):
    seen = []

    def fake(row, seed, *args, **kwargs):
        seen.append((row.plan_id, seed))
        return runner.JobOutcome(plan_id=row.plan_id, seed=seed)

    monkeypatch.setattr(runner, "_run_one_plan", fake)
    run_plans(two_seeds["a1"], only=["logo_r04_*"], seeds=[7], tracking=False)
    assert seen == [("logo_r04_south_west", 7)]


# --- workers ----------------------------------------------------------------------------------------


def test_workers_give_the_same_predictions_as_one_process(two_seeds, tmp_path):
    a1 = two_seeds["a1"]
    parallel = run_plans(a1, only=PLANS, workers=2, tracking=False, run_name="parallel")
    assert len(parallel.computed) == len(two_seeds["result"].computed)
    key = ["plan_id", "seed", "model", "covariate_group", "point_id"]
    one = load_predictions(two_seeds["result"].run_dir).sort_values(key).reset_index(drop=True)
    two = load_predictions(parallel.run_dir).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(one, two)
    manifest = json.loads((parallel.run_dir / MANIFEST_FILE).read_text())
    assert manifest["seeds_done"] == {plan_id: [7, 42] for plan_id in PLANS}


# --- devices ----------------------------------------------------------------------------------------


def test_the_device_override_reaches_the_estimator_and_the_hash(two_seeds):
    from dataclasses import replace

    from yg_eo_soilnet.a1.pipelines import stage_device

    settings = two_seeds["a1"].settings
    config = settings.load_config()
    assert stage_device(settings, config, settings.stage2) == "cpu"
    assert stage_device(replace(settings, device="cuda"), config, settings.stage2) == "cuda"
    with pytest.raises(ValueError, match="device"):
        replace(settings, device="tpu")
    assert runner._config_hash(two_seeds["a1"], config, {"device": "cuda"}) != runner._config_hash(
        two_seeds["a1"], config, None
    )


def test_runs_fitted_on_different_processors_are_not_compared_unless_allowed():
    same = {"a": {"stage1": "cpu", "stage2": "cuda"}, "b": {"stage1": "cpu", "stage2": "cuda"}}
    check_same_devices(same)
    mixed = {"a": {"stage1": "cpu", "stage2": "cuda"}, "b": {"stage1": "cpu", "stage2": "cpu"}}
    with pytest.raises(DeviceMismatch, match="stage2.*--allow-mixed-devices"):
        check_same_devices(mixed)
    check_same_devices(mixed, allow_mixed=True)
    assert is_cuda("cuda:0") and is_cuda("gpu") and not is_cuda("cpu") and not is_cuda(None)


def test_check_cuda_fails_on_an_xgboost_without_cuda(monkeypatch):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})
    with pytest.raises(DeviceUnavailable, match="built without CUDA"):
        check_cuda()


def test_check_cuda_fails_when_xgboost_falls_back_to_the_processor(monkeypatch):
    import xgboost

    def fit(self, X, y, **kwargs):
        warnings.warn(
            "WARNING: context.cc:210: Device is changed from GPU to CPU as we couldn't find any available GPU"
        )
        return self

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setattr(xgboost.XGBRegressor, "fit", fit)
    with pytest.raises(DeviceUnavailable, match="fell back to the processor"):
        check_cuda()


def test_a_cuda_stage_is_checked_before_anything_is_fitted(two_seeds, monkeypatch):
    calls = []

    def refuse(device="cuda"):
        calls.append(device)
        raise DeviceUnavailable("no card")

    monkeypatch.setattr("yg_eo_soilnet.a1.devices.check_cuda", refuse)
    monkeypatch.setattr(runner, "build_table", lambda *a, **k: pytest.fail("the data were read"))
    with pytest.raises(DeviceUnavailable):
        run_plans(two_seeds["a1"], device="cuda", tracking=False, run_name="cuda")
    assert calls == ["cuda"]


def test_to_cpu_moves_only_a_fitted_model_on_the_card():
    from yg_eo_soilnet.a1.pipelines import final_estimator, to_cpu

    model = SimpleNamespace(
        get_params=lambda deep=False: {"device": "cuda"},
        get_booster=lambda: None,
        set_params=lambda **kw: moved.update(kw),
    )
    moved: dict = {}
    to_cpu(model)
    assert moved == {"device": "cpu"}
    assert final_estimator(model) is model


def _has_gpu() -> bool:
    try:
        check_cuda()
        return True
    except (DeviceUnavailable, ImportError):
        return False


@pytest.mark.skipif(not _has_gpu(), reason="no usable graphics card")
def test_on_the_card_models_fit_there_and_predict_on_the_processor_without_a_warning():
    import xgboost

    from yg_eo_soilnet.a1.pipelines import predict_on_cpu

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.normal(size=(200, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a device fallback or a mismatched-devices copy fails the test
        model = xgboost.XGBRegressor(
            n_estimators=5, tree_method="hist", device="cuda", multi_strategy="multi_output_tree"
        ).fit(X, y)
        predicted = predict_on_cpu(model, X)
    assert predicted.shape == (400,) and model.get_params()["device"] == "cpu"


def test_the_xgboost_configuration_loads():
    a1 = load_a1_config(str(A1_CONFIGS / "a1_xgb.yml"))
    settings = a1.settings
    assert settings.seeds == (42, 7, 2026)
    assert settings.stage1.estimator == "XGBoost" and settings.stage1.init_args["device"] == "cpu"
    assert settings.stage2.init_args["device"] == "cuda"
    assert settings.som_models == SOM_MODELS
    assert settings.output_dir == "results/a1_xgb"
    assert load_settings(str(A1_CONFIGS / "a1_xgb.yml")).seed == 42


def test_each_worker_gets_its_share_of_the_physical_cores(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 12 if logical else 6)
    assert runner.physical_cores() == 6
    assert [runner.threads_per_worker(n) for n in (1, 2, 4, 8)] == [6, 3, 1, 1]
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: None)  # cannot tell: the logical count
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 4)
    assert runner.threads_per_worker(1) == 4


def test_the_faster_mixed_configuration_bins_inputs_into_64_values_in_both_stages():
    from yg_eo_soilnet.a1.pipelines import build_stage_pipeline, final_estimator

    settings = load_settings(str(A1_CONFIGS / "a1_mixed_fast.yml"))
    assert settings.output_dir == "results/a1_mixed_fast"
    assert settings.stage1.estimator == settings.stage2.estimator == "XGBoost"
    config = settings.load_config()
    for stage in (settings.stage1, settings.stage2):
        pipeline, _ = build_stage_pipeline(settings, config, ["x"], stage=stage)
        assert final_estimator(pipeline).get_params()["max_bin"] == 64
    assert settings.stage2.params["model__max_depth"] == [3, 4, 5, 6, 7]  # the grid is unchanged
