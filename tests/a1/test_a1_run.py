"""The A1 runner: the stage-2 models, the per-point outputs, resuming, and MLflow.

A fast configuration runs here - XGBoost in both stages with 30 trees, one grid point, two inner
folds - on random and leave-one-region-out plans of the demo data. The full configuration, with its
four-point grid, runs once in the slow acceptance test.
"""

import json
import time

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from experiments.a1.run_a1 import main as run_a1_main
from yg_eo_soilnet.a1 import stage2 as stage2_module
from yg_eo_soilnet.a1.auxiliary import predict_auxiliary
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP, COMBINED_GROUP, load_a1_config
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, plan_population, random_plan, write_plans
from yg_eo_soilnet.a1.runner import (
    MANIFEST_FILE,
    METRICS_FILE,
    PREDICTION_COLUMNS,
    RunMismatch,
    collect_metrics,
    run_plans,
)
from yg_eo_soilnet.a1.stage2 import (
    GLOBAL,
    GLOBAL_AUX,
    LOCAL,
    MODELS,
    PER_CLASS_MEASURED,
    PER_CLASS_PREDICTED,
    ROLE_DIAGNOSTIC,
    ROLE_MODEL,
    _fold_pairs,
    build_context,
    fit_tuned,
    run_model,
)
from yg_eo_soilnet.datamodules.splitting import TEST, VAL

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 30}},
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 30}, "params": {"model__max_depth": [3]}},
    "inner_cv_folds": 2,
}


def _fast_settings_file(builder, folder, uri, **extra):
    return builder(
        main_overrides={"MLFLOW_TRACKING_URI": uri}, as_file=True, output_dir=str(folder), **{**FAST, **extra}
    )


@pytest.fixture(scope="module")
def demo_run(a1_settings_in, tmp_path_factory, mlflow_store):
    """One fast run over the demo's random and logo plans, with the covariate groups."""
    folder = tmp_path_factory.mktemp("a1_run")
    with mlflow_store(folder / "mlruns") as uri:
        path = _fast_settings_file(a1_settings_in(folder), folder / "out", uri)
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[RANDOM, LOGO])
        result = run_plans(a1, groups=True)
        yield {"a1": a1, "result": result, "uri": uri, "path": path, "folder": folder}


def _predictions(demo_run, plan_id=None):
    root = demo_run["result"].run_dir / "predictions"
    paths = sorted((root / plan_id).glob("*.parquet")) if plan_id else sorted(root.glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _plan(demo_run, plan_id):
    from yg_eo_soilnet.datamodules.splitting import SplitPlan

    folder = demo_run["result"].run_dir.parents[1] / "plans"
    return SplitPlan.from_frame(pd.read_parquet(folder / f"{plan_id}.parquet"))


# --- what is written ----------------------------------------------------------------------------


def test_every_plan_gets_every_model(demo_run):
    frame = _predictions(demo_run)
    combined = frame[frame["covariate_group"] == COMBINED_GROUP]
    assert set(combined["plan_id"]) == {
        "random",
        *(
            f"logo_r0{i}_{name}"
            for i, name in [(1, "north_east"), (2, "north_west"), (3, "south_east"), (4, "south_west")]
        ),
    }
    for _, models in combined.groupby("plan_id")["model"]:
        assert set(models) >= set(MODELS)


def test_the_rows_are_exactly_the_plans_validation_and_test_points(demo_run):
    for plan_id in ("random", "logo_r02_north_west"):
        plan = _plan(demo_run, plan_id)
        held = set(plan.point_ids_for(VAL)) | set(plan.point_ids_for(TEST))
        frame = _predictions(demo_run, plan_id)
        for (model, group), rows in frame.groupby(["model", "covariate_group"]):
            if model == LOCAL:
                continue
            assert set(rows["point_id"]) == held, (model, group)
            assert not rows["point_id"].duplicated().any()
            assert (rows["split"].map({VAL: True, TEST: True})).all()


def test_the_predictions_files_hold_exactly_the_agreed_columns(demo_run):
    assert list(_predictions(demo_run, "random").columns) == list(PREDICTION_COLUMNS)


def test_the_measured_class_model_is_labelled_a_diagnostic_upper_bound(demo_run):
    frame = _predictions(demo_run)
    roles = frame.groupby("model")["model_role"].unique()
    assert list(roles[PER_CLASS_MEASURED]) == [ROLE_DIAGNOSTIC]
    for model in (GLOBAL, GLOBAL_AUX, PER_CLASS_PREDICTED):
        assert list(roles[model]) == [ROLE_MODEL]


def test_the_classes_recorded_are_the_measured_and_the_predicted_ones(demo_run, a1_demo_settings):
    from yg_eo_soilnet.a1.strata import carbonate_class

    frame = _predictions(demo_run, "random")
    table = build_table(a1_demo_settings)
    measured = carbonate_class(table.frame["caco3_pct_total"]).reindex(frame["point_id"]).astype(object)
    assert (frame["carbonate_class_measured"].to_numpy() == measured.to_numpy()).all()
    assert frame["carbonate_class_predicted"].notna().all()
    assert (frame["carbonate_class_predicted"] != frame["carbonate_class_measured"]).any()


def test_the_region_of_each_point_is_recorded(demo_run):
    frame = _predictions(demo_run, "logo_r01_north_east")
    assert set(frame.loc[frame["split"] == TEST, "region"]) == {"North-East"}


def test_the_chosen_settings_are_recorded_with_every_row(demo_run):
    frame = _predictions(demo_run, "random")
    settings = json.loads(frame.loc[frame["model"] == GLOBAL, "tuned_params"].iloc[0])
    assert settings["model__max_depth"] == 3
    assert settings["tuned"] is False  # one grid point: nothing to choose


def test_the_stage_1_audit_is_saved_for_every_plan(demo_run):
    stage1 = sorted(path.stem for path in (demo_run["result"].run_dir / "stage1").glob("*.parquet"))
    assert len(stage1) == 5
    audit = pd.read_parquet(demo_run["result"].run_dir / "stage1" / "random.parquet")
    assert {"split", "inner_fold", "a1_pred_caco3_pct_total", "predicted_by_caco3_pct_total"} <= set(audit.columns)


# --- covariate groups and local scores -----------------------------------------------------------


def test_with_groups_each_covariate_group_gets_its_own_global_aux_model(demo_run):
    frame = _predictions(demo_run, "random")
    groups = set(frame.loc[frame["model"] == GLOBAL_AUX, "covariate_group"])
    assert groups == {COMBINED_GROUP, *demo_run["a1"].groups.names}


def test_logo_plans_get_local_scores_inside_the_held_out_region_only(demo_run):
    frame = _predictions(demo_run, "logo_r03_south_east")
    local = frame[frame["model"] == LOCAL]
    plan = _plan(demo_run, "logo_r03_south_east")
    assert set(local["point_id"]) == set(plan.point_ids_for(TEST))
    assert set(local["covariate_group"]) == {COMBINED_GROUP, *demo_run["a1"].groups.names}
    assert LOCAL not in set(_predictions(demo_run, "random")["model"])


def test_a_region_too_small_for_a_local_score_is_skipped_and_said(a1_settings_in, tmp_path, mlflow_store):
    with mlflow_store(tmp_path / "mlruns") as uri:
        path = _fast_settings_file(a1_settings_in(tmp_path), tmp_path / "out", uri)
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(tmp_path / "out" / "plans"), kinds=[LOGO])
        # A threshold above every region's size, set after the plans exist so they are still made.
        raw = __import__("yaml").safe_load(path.read_text())
        raw["min_cell_size"] = 10_000
        path.write_text(__import__("yaml").safe_dump(raw, sort_keys=False, allow_unicode=True))
        result = run_plans(load_a1_config(str(path)), groups=True, tracking=False, run_name="small")

    assert any("no local" in note for note in result.notes)
    assert not any(LOCAL in path for path in result.computed)


# --- the models themselves ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def context_parts(demo_run):
    a1 = demo_run["a1"]
    settings = a1.settings
    config = settings.load_config()
    table = build_table(settings, config)
    population = plan_population(settings, config, points=table)
    built = random_plan(population, settings)
    auxiliary = predict_auxiliary(table, built.plan, settings, config, kind=RANDOM)
    resolved = a1.resolve_groups(table)
    aux_columns = list(resolved.groups[AUXILIARY_GROUP])
    global_columns = [column for column in resolved.combined if column not in aux_columns]
    return table, built.plan, auxiliary, settings, config, global_columns, aux_columns


def _context(context_parts, **settings_changes):
    from dataclasses import replace

    table, plan, auxiliary, settings, config, _, _ = context_parts
    return build_context(table, plan, auxiliary, replace(settings, **settings_changes), config)


def test_global_never_reads_the_stage_1_predictions_and_global_aux_does(context_parts, monkeypatch):
    seen = {}
    original = stage2_module.fit_tuned

    def recording(pipeline, grid, X, y, folds):
        seen.setdefault("columns", []).append(list(X.columns))
        return original(pipeline, grid, X, y, folds)

    monkeypatch.setattr(stage2_module, "fit_tuned", recording)
    _, _, _, _, _, global_columns, aux_columns = context_parts
    for model in (GLOBAL, GLOBAL_AUX):
        seen.clear()
        run_model(_context(context_parts), model, global_columns=global_columns, auxiliary_columns=aux_columns)
        fitted = seen["columns"][0]
        reads_aux = any(column.startswith("a1_pred_") for column in fitted)
        assert reads_aux == (model == GLOBAL_AUX)


@pytest.mark.parametrize(
    "model, attribute", [(PER_CLASS_MEASURED, "class_measured"), (PER_CLASS_PREDICTED, "class_predicted")]
)
def test_each_class_model_is_fitted_on_its_classs_training_points(context_parts, model, attribute):
    _, _, _, _, _, global_columns, aux_columns = context_parts
    context = _context(context_parts, min_cell_size=20)
    result = run_model(context, model, global_columns=global_columns, auxiliary_columns=aux_columns)
    classes = getattr(context, attribute)
    train = context.train_ids

    for label, params in result.tuned_params.items():
        assert params["n_train"] == int((classes.reindex(train) == label).sum())


def test_a_thin_class_falls_back_to_global_aux_and_is_flagged(context_parts):
    _, _, _, _, _, global_columns, aux_columns = context_parts
    context = _context(context_parts, min_cell_size=150)
    fallback_source = run_model(context, GLOBAL_AUX, global_columns=global_columns, auxiliary_columns=aux_columns)
    result = run_model(context, PER_CLASS_MEASURED, global_columns=global_columns, auxiliary_columns=aux_columns)

    thin = [label for label, params in result.tuned_params.items() if params.get("fallback")]
    assert thin, "the threshold should leave at least one class without a model"
    held = context.held_out_ids
    in_thin = context.class_measured.reindex(held).isin(thin).to_numpy()
    assert result.fallback[in_thin].all()
    assert not result.fallback[~in_thin].any()
    assert np.allclose(result.predicted[in_thin], fallback_source.predicted[in_thin])


def test_a_held_out_point_without_a_class_falls_back(context_parts):
    _, _, _, _, _, global_columns, aux_columns = context_parts
    context = _context(context_parts)
    point = context.held_out_ids[0]
    context.class_measured = context.class_measured.copy()
    context.class_measured.loc[point] = np.nan
    result = run_model(context, PER_CLASS_MEASURED, global_columns=global_columns, auxiliary_columns=aux_columns)
    assert result.fallback.loc[point]


def test_tuning_searches_the_grid_through_the_given_folds(context_parts):
    from yg_eo_soilnet.a1.pipelines import build_stage_pipeline

    table, _, _, settings, config, global_columns, _ = context_parts
    context = _context(context_parts)
    train = context.train_ids
    pipeline, _ = build_stage_pipeline(
        settings, config, global_columns, table.categorical_columns, stage=settings.stage2
    )
    grid = {"model__max_depth": [2, 4]}

    model, params = fit_tuned(pipeline, grid, context.X.loc[train, global_columns], context.y.loc[train], context.folds)
    assert params["tuned"] is True and params["model__max_depth"] in (2, 4)

    one_fold = pd.Series(0, index=train)
    _, params = fit_tuned(pipeline, grid, context.X.loc[train, global_columns], context.y.loc[train], one_fold)
    assert params == {"model__max_depth": 2, "tuned": False}


def test_the_search_folds_hold_each_fold_out_once():
    folds = pd.Series([0, 0, 1, 1, 2, np.nan], index=list("abcdef"))
    pairs = _fold_pairs(folds)
    held = sorted(int(position) for _, held in pairs for position in held)
    assert held == [0, 1, 2, 3, 4]  # the point with no fold is never held out...
    assert all(5 in kept for kept, _ in pairs)  # ...but is always fitted on


# --- resuming ----------------------------------------------------------------------------------


def test_a_second_run_computes_nothing(demo_run, monkeypatch):
    calls = []
    monkeypatch.setattr("yg_eo_soilnet.a1.runner.predict_auxiliary", lambda *a, **k: calls.append(1))
    again = run_plans(demo_run["a1"], groups=True, tracking=False)
    assert again.computed == []
    assert len(again.skipped) == len(demo_run["result"].computed)
    assert calls == []


def test_deleting_one_output_recomputes_only_that_one(demo_run):
    target = demo_run["result"].run_dir / "predictions" / "random" / f"{GLOBAL}__{COMBINED_GROUP}.parquet"
    before = pd.read_parquet(target)
    target.unlink()
    again = run_plans(demo_run["a1"], groups=True, tracking=False)
    assert again.computed == [str(target)]
    pd.testing.assert_frame_equal(pd.read_parquet(target), before)


def test_a_half_written_file_is_not_taken_as_done(demo_run):
    target = demo_run["result"].run_dir / "predictions" / "random" / f"{GLOBAL_AUX}__{COMBINED_GROUP}.parquet"
    target.unlink()
    target.with_suffix(".parquet.partial").write_text("half")
    again = run_plans(demo_run["a1"], groups=True, tracking=False)
    assert str(target) in again.computed
    assert not target.with_suffix(".parquet.partial").exists()


def test_a_run_folder_made_from_other_settings_is_refused_then_cleared_on_restart(demo_run, tmp_path):
    a1 = demo_run["a1"]
    path = demo_run["path"]
    original = path.read_text()
    try:
        path.write_text(original.replace("inner_cv_folds: 2", "inner_cv_folds: 3"))
        changed = load_a1_config(str(path))
        with pytest.raises(RunMismatch, match="the settings differ"):
            run_plans(changed, kinds=[RANDOM], tracking=False)
        cleared = run_plans(changed, kinds=[RANDOM], tracking=False, restart=True, run_name="default")
        assert len(cleared.computed) == len(MODELS)
    finally:
        path.write_text(original)
        # Leave the module's run as the other tests expect it.
        run_plans(a1, groups=True, tracking=False, restart=True)


# --- MLflow --------------------------------------------------------------------------------------


def test_the_run_is_recorded_in_mlflow_with_the_commit_and_every_plans_scores(demo_run):
    client = MlflowClient(tracking_uri=demo_run["uri"])
    experiment = client.get_experiment_by_name(demo_run["a1"].settings.mlflow_experiment)
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]

    assert len(parents) == 1
    assert parents[0].data.params["a1_git_commit"] == demo_run["a1"].git_commit
    assert {run.data.tags["plan_id"] for run in children} == {
        "random",
        "logo_r01_north_east",
        "logo_r02_north_west",
        "logo_r03_south_east",
        "logo_r04_south_west",
    }
    for run in children:
        assert run.data.tags["kind"] in (RANDOM, LOGO)
        assert "rmse_test__global__combined" in run.data.metrics
        assert "rmse_val__per_class_predicted__combined" in run.data.metrics
    artifacts = {item.path for item in client.list_artifacts(parents[0].info.run_id)}
    assert {METRICS_FILE, MANIFEST_FILE} <= artifacts


def test_metrics_cover_every_file_and_split(demo_run):
    metrics = collect_metrics(demo_run["result"].run_dir)
    files = list((demo_run["result"].run_dir / "predictions").glob("*/*.parquet"))
    assert len(metrics) >= len(files)
    assert set(metrics["split"]) == {VAL, TEST}
    assert (demo_run["result"].run_dir / METRICS_FILE).exists()


def test_a_missing_plan_index_says_how_to_make_one(a1_settings_in, tmp_path, mlflow_store):
    with mlflow_store(tmp_path / "mlruns") as uri:
        path = _fast_settings_file(a1_settings_in(tmp_path), tmp_path / "empty", uri)
        with pytest.raises(FileNotFoundError, match="make_plans.py"):
            run_plans(load_a1_config(str(path)), tracking=False)


# --- the demo's own finding ------------------------------------------------------------------------


def _pooled_test_rmse(frame):
    test = frame[(frame["split"] == TEST) & (frame["covariate_group"] == COMBINED_GROUP)]
    squared = (test["observed"] - test["predicted"]) ** 2
    return np.sqrt(squared.groupby(test["model"]).mean())


def test_routing_by_measured_carbonate_beats_the_global_model_on_the_demo(demo_run):
    rmse = _pooled_test_rmse(_predictions(demo_run))
    assert rmse[PER_CLASS_MEASURED] < rmse[GLOBAL]


@pytest.mark.slow
def test_the_full_demo_run_from_the_command_line(a1_settings_in, tmp_path, mlflow_store):
    """Acceptance: the shipped grid, XGBoost as stage 1, random and logo plans, in a few minutes."""
    with mlflow_store(tmp_path / "mlruns") as uri:
        builder = a1_settings_in(tmp_path)
        path = builder(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            as_file=True,
            output_dir=str(tmp_path / "out"),
            stage1={"estimator": "XGBoost"},
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(tmp_path / "out" / "plans"), kinds=[RANDOM, LOGO])
        started = time.monotonic()
        result = run_a1_main(["--config", str(path), "--kinds", "random,logo"])
        elapsed = time.monotonic() - started
        mlflow.set_tracking_uri(uri)

    frame = pd.concat([pd.read_parquet(path) for path in (result.run_dir / "predictions").glob("*/*.parquet")])
    rmse = _pooled_test_rmse(frame)
    print(f"full demo run: {elapsed:.0f} s; pooled test RMSE {rmse.round(2).to_dict()}")
    assert elapsed < 600
    assert rmse[PER_CLASS_MEASURED] < rmse[GLOBAL]
    assert len(result.computed) == 5 * len(MODELS)


def test_stage_1_read_back_from_its_file_gives_the_same_predictions(demo_run):
    from yg_eo_soilnet.a1.runner import _auxiliary_from_frame

    a1 = demo_run["a1"]
    saved = pd.read_parquet(demo_run["result"].run_dir / "stage1" / "random.parquet")
    assert "point_id" in saved.columns
    reread = _auxiliary_from_frame(saved, a1.settings)
    plan = _plan(demo_run, "random")
    assert reread.predictions.index.equals(plan.assignments.index)
    assert reread.predictions.notna().all().all()
