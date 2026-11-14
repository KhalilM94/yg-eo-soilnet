"""Comparisons across run folders (compare.py) and matched leave-one-basin-out in evaluate.py.

A comparison is only honest on the same test points and the same processor, so pairing must refuse
anything else; its paired differences, Holm families and summaries must be what they say. The
demo runs below are small and fast: a main run with the local model (named ``a1_xgb``, as the real
one), a mixed run with both arms, and matched leave-one-basin-out runs at two training sizes.
"""

import hashlib
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from experiments.a1.compare import main as compare_main
from experiments.a1.evaluate import main as evaluate_main
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.compare import (
    CompareError,
    compare_runs,
    read_run,
)
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.devices import DeviceMismatch
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, MIXED_INDICATOR, MIXED_WEIGHTED, write_plans
from yg_eo_soilnet.a1.runner import MANIFEST_FILE, run_plans
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, SplitPlan

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}},
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 20}, "params": {"model__max_depth": [2]}},
    "inner_cv_folds": 2,
}
N_BOOT = 40


def _run(builder, folder: Path, name: str, *, plans: dict, models, kinds, uri=None, tracking=False):
    """Plans and a fast run in ``folder/name`` (its output folder); returns the run folder."""
    stage2 = {**FAST["stage2"], "models": list(models)}
    path = builder(folder / name)(
        main_overrides={"MLFLOW_TRACKING_URI": uri} if uri else None,
        as_file=True,
        output_dir=str(folder / name),
        plans=plans,
        **{**FAST, "stage2": stage2},
    )
    a1 = load_a1_config(str(path))
    write_plans(a1.settings, str(folder / name / "plans"), kinds=[LOGO, *([MIXED] if MIXED in kinds else [])])
    result = run_plans(a1, kinds=kinds, tracking=tracking)
    return result.run_dir, path


@pytest.fixture(scope="module")
def runs(a1_settings_in, tmp_path_factory, mlflow_store):
    folder = tmp_path_factory.mktemp("a1_compare")

    def builder(sub):
        sub.mkdir(parents=True, exist_ok=True)
        return a1_settings_in(sub)

    logo = {"min_points": 100, "held_out": ["North-East", "South-West"], "validation_region": False}
    with mlflow_store(folder / "mlruns") as uri:
        main, main_config = _run(
            builder,
            folder,
            "a1_xgb",
            plans={"logo": logo},
            models=["global", "global_aux"],
            kinds=[LOGO],
            uri=uri,
            tracking=True,
        )
    arms, _ = _run(
        builder,
        folder,
        "a1_arms",
        plans={
            "logo": {**logo, "held_out": ["North-East"]},
            "mixed": {"fractions": [0.3, 0.6], "n_repeats": 2, "arms": ["weighted", "indicator"]},
        },
        models=["global"],
        kinds=[LOGO, MIXED, CONTROL, MIXED_WEIGHTED, MIXED_INDICATOR],
    )
    matched = {}
    for size in (150, 100):
        matched[size], matched_config = _run(
            builder,
            folder,
            f"a1_matched_{size}",
            plans={"logo": {**logo, "training_size": size, "n_repeats": 2}},
            # The 150-point run with the local model too, for the transfer rows of evaluate.py.
            models=["global", "global_aux"] if size == 150 else ["global"],
            kinds=[LOGO],
        )
        if size == 150:
            matched_150_config = matched_config
    return {
        "folder": folder,
        "uri": uri,
        "main": main,
        "main_config": main_config,
        "arms": arms,
        "matched": matched,
        "matched_config": matched_150_config,
    }


def _copy_run(run_dir: Path, name: str) -> Path:
    """A copy of a run's output folder under another name; returns the copy's run folder."""
    source = run_dir.parent.parent
    target = source.parent / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target / "runs" / run_dir.name


def _set_devices(run_dir: Path, stage2: str) -> None:
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text())
    manifest["devices"] = {"stage1": "cpu", "stage2": stage2}
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest))


@pytest.fixture(scope="module")
def mixed_comparison(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("compare_mixed")
    return compare_runs([runs["main"], runs["arms"]], out, n_boot=N_BOOT, tracking=False)


@pytest.fixture(scope="module")
def logo_comparison(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("compare_logo")
    return compare_runs([runs["main"], runs["matched"][150], runs["matched"][100]], out, n_boot=N_BOOT, tracking=False)


# --- pairing ---------------------------------------------------------------------------------------


def test_pairing_refuses_plans_with_other_test_points(runs):
    copy = _copy_run(runs["arms"], "a1_arms_other_points")
    plans_dir = copy.parent.parent / "plans"
    index = pd.read_csv(plans_dir / "index.csv")
    plan_id = index.loc[index["kind"] == MIXED, "plan_id"].iloc[0]
    path = plans_dir / f"{plan_id}.parquet"
    plan = SplitPlan.from_frame(pd.read_parquet(path))
    assignments = plan.assignments.copy()
    assignments.loc[assignments.index[assignments.to_numpy() == TEST][0]] = TRAIN
    changed = SplitPlan(
        assignments=assignments,
        strategy=plan.strategy,
        test_size=plan.test_size,
        val_size=plan.val_size,
        seed=plan.seed,
        population_policy=plan.population_policy,
        eligibility=plan.eligibility,
        clusters=plan.clusters,
    )
    changed.to_frame().to_parquet(path, index=False)
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["plans"][plan_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(CompareError, match="test different points"):
        compare_runs([runs["arms"], copy], copy.parent / "out", n_boot=N_BOOT, tracking=False)


def test_a_plan_file_changed_since_the_run_is_refused(runs):
    copy = _copy_run(runs["arms"], "a1_arms_changed_plan")
    plans_dir = copy.parent.parent / "plans"
    plan_id = pd.read_csv(plans_dir / "index.csv")["plan_id"].iloc[0]
    path = plans_dir / f"{plan_id}.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[::-1].to_parquet(path, index=False)
    with pytest.raises(CompareError, match="hash changed"):
        read_run(copy).test_points(plan_id)


def test_runs_on_different_devices_are_refused_unless_allowed(runs, tmp_path):
    gpu = _copy_run(runs["arms"], "a1_arms_gpu")
    _set_devices(gpu, "cuda")
    cpu = _copy_run(runs["arms"], "a1_arms_cpu")
    _set_devices(cpu, "cpu")
    with pytest.raises(DeviceMismatch):
        compare_runs([cpu, gpu], tmp_path / "refused", n_boot=N_BOOT, tracking=False)
    result = compare_runs([cpu, gpu], tmp_path / "allowed", allow_mixed_devices=True, n_boot=N_BOOT, tracking=False)
    for name in ("mixed_arms", "mixed_arms_paired", "logo_matched"):
        assert "device_stage2" in result.tables[name].columns or "matched_device_stage2" in result.tables[name].columns
    assert set(result.tables["mixed_arms"]["device_stage2"]) == {"cpu", "cuda"}
    assert {"a_device_stage2", "b_device_stage2"} <= set(result.tables["cross_run"].columns)


def test_the_same_plans_in_two_runs_agree_exactly_when_the_fits_are_the_same(runs, tmp_path):
    copy = _copy_run(runs["arms"], "a1_arms_again")
    result = compare_runs([runs["arms"], copy], tmp_path, kinds=[MIXED, CONTROL], n_boot=N_BOOT, tracking=False)
    crossed = result.tables["cross_run"]
    assert set(crossed["kind"]) == {MIXED, CONTROL}
    assert np.allclose(crossed["rmse_difference"], 0) and np.allclose(crossed["pearson_r"], 1)
    summary = result.tables["cross_run_summary"]
    assert set(summary["kind"]) == {MIXED, CONTROL}
    assert (summary["max_abs_rmse_difference"] == 0).all() and (summary["mean_abs_prediction_difference"] == 0).all()
    arms = result.tables["mixed_arms"]
    first = arms[arms["run"] == "a1_arms"]
    expected = (first["rmse_mixed"] - first["rmse_control"]).mean()
    assert np.allclose(summary["mean_mixed_minus_control"], expected)
    assert (summary["effect_run"] == "a1_arms").all()


# --- the mixed arms ---------------------------------------------------------------------------------


def test_every_kind_is_scored_on_the_shared_test_points(mixed_comparison, runs):
    arms = mixed_comparison.tables["mixed_arms"]
    assert len(arms) == 2 * 2  # two shares, two repeats
    for column in ("rmse_logo", "rmse_control", "rmse_mixed", "rmse_mixed_weighted", "rmse_mixed_indicator"):
        assert arms[column].notna().all(), column
    assert arms["rmse_local"].notna().all() and (arms["local_run"] == "a1_xgb").all()
    assert (arms["logo_run"] == "a1_arms").all()
    predictions = evaluation.load_predictions(runs["arms"])
    row = arms.iloc[0]
    control = predictions[
        (predictions["kind"] == CONTROL)
        & (predictions["split"] == TEST)
        & np.isclose(predictions["fraction"], row["fraction"])
        & (predictions["repeat"] == row["repeat"])
    ]
    assert row["n_test"] == control["point_id"].nunique()
    assert (
        row["n_moved"]
        == pd.read_csv(runs["arms"].parent.parent / "plans" / "index.csv")
        .set_index("plan_id")
        .loc[control["plan_id"].iloc[0], "n_train"]
    )


def test_paired_differences_equal_those_recomputed_by_hand(mixed_comparison, runs):
    paired = mixed_comparison.tables["mixed_arms_paired"]
    predictions = evaluation.load_predictions(runs["arms"])
    test = predictions[(predictions["split"] == TEST) & (predictions["model"] == "global")]
    fraction = 0.3

    def rmse(kind, repeat):
        part = test[(test["kind"] == kind) & np.isclose(test["fraction"], fraction) & (test["repeat"] == repeat)]
        return float(np.sqrt(((part["predicted"] - part["observed"]) ** 2).mean()))

    for arm in (MIXED, MIXED_WEIGHTED, MIXED_INDICATOR):
        gaps = [rmse(arm, repeat) - rmse(CONTROL, repeat) for repeat in (0, 1)]
        row = paired[(paired["a"] == arm) & (paired["b"] == CONTROL) & np.isclose(paired["fraction"], fraction)]
        assert len(row) == 1
        assert row["mean_difference"].iloc[0] == pytest.approx(np.mean(gaps))
        assert row["n_repeats"].iloc[0] == 2
        assert row["n_repeats_better"].iloc[0] == sum(gap < 0 for gap in gaps)
    for arm in (MIXED_WEIGHTED, MIXED_INDICATOR):
        gaps = [rmse(arm, repeat) - rmse(MIXED, repeat) for repeat in (0, 1)]
        row = paired[(paired["a"] == arm) & (paired["b"] == MIXED) & np.isclose(paired["fraction"], fraction)]
        assert row["mean_difference"].iloc[0] == pytest.approx(np.mean(gaps))


def test_holm_is_applied_across_the_shares_within_basin_and_comparison(mixed_comparison):
    paired = mixed_comparison.tables["mixed_arms_paired"]
    assert len(paired) == 5 * 2  # five comparisons, two shares
    for _, family in paired.groupby(["run", "basin", "a", "b"]):
        assert len(family) == 2
        np.testing.assert_allclose(family["p_holm"], evaluation.holm(family["p_value"].to_numpy()))
    everything = evaluation.holm(paired["p_value"].to_numpy())
    assert not np.allclose(everything, paired["p_holm"]) or paired["p_value"].isna().all()


def test_the_main_runs_transfer_losses_classify_the_basins(mixed_comparison):
    status = mixed_comparison.tables["transfer_status"]
    assert set(status["basin"]) == {"North-East", "South-West"}
    np.testing.assert_allclose(status["p_holm"], evaluation.holm(status["p_value"].to_numpy()))
    expected = (status["difference"] > 0) & (status["low"] > 0) & (status["p_holm"] < 0.05)
    assert (status["transfer_fails"] == expected).all()
    caption = mixed_comparison.captions["figC1_mixed_arms"]
    assert "transfer fails" in caption and "Holm" in caption


def test_without_the_main_run_the_titles_give_the_basin_only(runs, tmp_path):
    result = compare_runs([runs["arms"]], tmp_path, n_boot=N_BOOT, tracking=False)
    assert result.tables["transfer_status"].empty
    assert "titles give the basin only" in result.captions["figC1_mixed_arms"]
    titles = [axis.get_title() for axis in result.figures["figC1_mixed_arms"].axes if axis.get_title()]
    assert titles == ["North-East"]


def test_the_figures_and_tables_are_written(mixed_comparison):
    out = mixed_comparison.out_dir
    for name in ("figC1_mixed_arms", "figC2_logo_matched"):
        assert (out / f"{name}.png").exists() and (out / f"{name}.pdf").exists()
    for name in ("mixed_arms_paired", "logo_matched", "transfer_status", "cross_run_summary"):
        assert (out / f"{name}.csv").exists() and (out / f"{name}.tex").exists()
    assert (out / "mixed_arms.csv").exists()
    captions = (out / "captions.md").read_text()
    assert "## figC1_mixed_arms" in captions and "## figC2_logo_matched" in captions


# --- matched leave one basin out ------------------------------------------------------------------


def test_matched_draws_are_summarised_against_the_unmatched_plan(logo_comparison, runs):
    matched = logo_comparison.tables["logo_matched"]
    draws = logo_comparison.tables["logo_matched_draws"]
    assert set(matched["training_size"]) == {150, 100}
    assert set(matched["basin"]) == {"North-East", "South-West"}
    index = pd.read_csv(runs["main"].parent.parent / "plans" / "index.csv").set_index("held_out")
    for row in matched.itertuples():
        part = draws[(draws["basin"] == row.basin) & (draws["training_size"] == row.training_size)]
        assert row.n_draws == len(part) == 2
        assert row.matched_rmse_mean == pytest.approx(part["rmse"].mean())
        assert row.matched_rmse_min == pytest.approx(part["rmse"].min())
        assert row.rmse_difference == pytest.approx(row.matched_rmse_mean - row.unmatched_rmse)
        assert row.unmatched_training_size == index.loc[row.basin, "n_train"]
        assert row.unmatched_run == "a1_xgb"


def test_the_size_curve_ends_at_the_unmatched_plan(logo_comparison):
    figure = logo_comparison.figures["figC2_logo_matched"]
    unmatched = logo_comparison.tables["logo_matched"].drop_duplicates("basin").set_index("basin")
    for axis in figure.axes:
        basin = axis.get_title()
        if basin not in unmatched.index:
            continue
        curve = [line for line in axis.get_lines() if len(line.get_xdata()) >= 2][0]
        assert list(curve.get_xdata())[:2] == [100.0, 150.0]
        assert curve.get_xdata()[-1] == unmatched.loc[basin, "unmatched_training_size"]
        assert curve.get_ydata()[-1] == pytest.approx(unmatched.loc[basin, "unmatched_rmse"])


def test_matched_draws_of_other_test_points_are_refused(runs, tmp_path):
    copy = _copy_run(runs["matched"][150], "a1_matched_other_points")
    plans_dir = copy.parent.parent / "plans"
    plan_id = pd.read_csv(plans_dir / "index.csv")["plan_id"].iloc[0]
    path = plans_dir / f"{plan_id}.parquet"
    frame = pd.read_parquet(path)
    test_rows = frame.index[frame["split"] == TEST] if "split" in frame else frame.index[:1]
    frame = frame.drop(test_rows[:1])
    frame.to_parquet(path, index=False)
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["plans"][plan_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(CompareError, match="test different points"):
        compare_runs([runs["main"], copy], tmp_path, n_boot=N_BOOT, tracking=False)


def test_compare_records_an_mlflow_run_in_the_first_runs_experiment(runs, tmp_path):
    mlflow.set_tracking_uri(runs["uri"])
    result = compare_main(
        ["--runs", str(runs["main"]), str(runs["arms"]), "--out", str(tmp_path / "out"), "--n-boot", "20"]
    )
    assert result.mlflow_run_id
    run = mlflow.tracking.MlflowClient().get_run(result.mlflow_run_id)
    manifest = json.loads((runs["main"] / MANIFEST_FILE).read_text())
    assert run.info.experiment_id == manifest["mlflow"]["experiment_id"]
    assert run.info.run_name == "compare"


# --- evaluate.py on matched draws -----------------------------------------------------------------


@pytest.fixture(scope="module")
def matched_tables(runs):
    return evaluate_main(["--config", str(runs["matched_config"]), "--n-boot", "10", "--no-mlflow"])


def test_matched_draws_are_scored_per_basin_and_summarised_over_draws(matched_tables, runs):
    table = matched_tables["scores_logo_matched.csv"]
    combined = table[(table["covariate_group"] == "combined") & (table["model"] == "global")]
    assert set(combined["region"]) == {"North-East", "South-West"}
    assert (combined["n_repeats"] == 2).all() and (combined["training_size"] == 150).all()
    predictions = evaluation.load_predictions(runs["matched"][150])
    test = predictions[
        (predictions["split"] == TEST) & (predictions["model"] == "global") & (predictions["held_out"] == "North-East")
    ]
    per_draw = test.groupby("plan_id").apply(
        lambda part: float(np.sqrt(((part["predicted"] - part["observed"]) ** 2).mean())), include_groups=False
    )
    row = combined[combined["region"] == "North-East"].iloc[0]
    assert row["rmse"] == pytest.approx(per_draw.mean())
    assert row["rmse_min"] == pytest.approx(per_draw.min()) and row["rmse_max"] == pytest.approx(per_draw.max())


def test_matched_draws_are_left_out_of_every_pooled_table(matched_tables):
    for name in ("scores.csv", "scores_by_class.csv", "intervals.csv", "intervals_by_region.csv"):
        table = matched_tables[name]
        assert table.empty or LOGO not in set(table.get("kind", pd.Series(dtype=object))), name
    transfer = matched_tables["transfer.csv"]
    assert set(transfer["kind"]) == {"logo_matched"}
    assert (transfer["n_repeats"] == 2).all()


def test_matched_headlines_are_the_mean_over_draws(runs, matched_tables):
    headline = pd.read_csv(runs["matched"][150] / "evaluation" / "seed_summary.csv")
    names = set(headline["name"])
    assert any(name.startswith("rmse__logo_m150_") for name in names)
    assert not any(name.startswith("rmse__logo_north") or name.startswith("rmse__logo_mean") for name in names)


def test_the_second_matched_configuration_loads():
    settings = load_settings(str(A1_CONFIGS / "a1_logo_matched_2000.yml"))
    assert settings.plans.logo.training_size == 2000 and settings.plans.logo.n_repeats == 5
    assert settings.output_dir == "results/a1_logo_matched_2000"
    first = load_settings(str(A1_CONFIGS / "a1_logo_matched.yml"))
    assert settings.plans.logo.held_out == first.plans.logo.held_out
