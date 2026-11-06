"""One MLflow run per A1 run folder: run_a1, evaluate and make_figures all write to the same parent.

The demo pipeline runs once for the module, in a file-based MLflow store, cut down to dev mode:
random plans, spatial folds of 10 km and 25 km, the two held-out demo regions, and the covariate
groups (so the local model, and the transfer drop, exist). Then evaluate.py and make_figures.py.
"""

import json
import re

import matplotlib.pyplot as plt
import pytest
from mlflow.tracking import MlflowClient

from experiments.a1.evaluate import REGION_NAMES_FILE, TABLES
from experiments.a1.evaluate import main as evaluate_main
from experiments.a1.make_figures import main as figures_main
from yg_eo_soilnet import map_style
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.figures import FIGURE_NAMES
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD, write_plans
from yg_eo_soilnet.a1.runner import MANIFEST_FILE, RunMismatch, run_plans
from yg_eo_soilnet.a1.tracking import A1TrackingError

PLANS = {
    "spatial_kfold": {"method": "grid", "cell_size_m": 10000, "cell_sizes_m": [10000, 25000], "n_folds": 2},
    "logo": {"min_points": 100, "held_out": ["North-East", "South-West"]},
}


def _offline(monkeypatch):
    def refuse(url, path):
        raise OSError("offline")

    monkeypatch.setattr(map_style, "cached_download", refuse)


@pytest.fixture(scope="module")
def pipeline(a1_settings_in, tmp_path_factory, mlflow_store):
    """The demo pipeline, run once, with MLflow on."""
    folder = tmp_path_factory.mktemp("a1_mlflow")
    with mlflow_store(folder / "mlruns") as uri, pytest.MonkeyPatch.context() as monkeypatch:
        _offline(monkeypatch)
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri}, as_file=True, output_dir=str(folder / "out"), plans=PLANS
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[RANDOM, SPATIAL_KFOLD, LOGO])
        result = run_plans(a1, groups=True, dev_mode=True)
        evaluate_main(["--config", str(path), "--n-boot", "20"])
        figures_main(["--config", str(path), "--n-boot", "20", "--no-relief"])
        plt.close("all")
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(a1.settings.mlflow_experiment)
        yield {
            "a1": a1,
            "path": path,
            "result": result,
            "uri": uri,
            "store": folder / "mlruns",
            "client": client,
            "experiment": experiment,
        }


@pytest.fixture
def in_store(pipeline, mlflow_store):
    """Point a test back at the module's MLflow store (each test otherwise gets a fresh one)."""
    with mlflow_store(pipeline["store"]):
        yield


def _runs(pipeline):
    runs = pipeline["client"].search_runs([pipeline["experiment"].experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
    return parents, children


def _manifest(pipeline, run_name="default"):
    run_dir = pipeline["result"].run_dir.parent / run_name
    return json.loads((run_dir / MANIFEST_FILE).read_text())


def _artifacts(pipeline, run_id, folder):
    return sorted(item.path.split("/", 1)[1] for item in pipeline["client"].list_artifacts(run_id, folder))


# --- one parent run, recorded in the manifest ----------------------------------------------------


def test_the_manifest_records_the_single_parent_run(pipeline):
    parents, children = _runs(pipeline)
    recorded = _manifest(pipeline)["mlflow"]
    assert len(parents) == 1
    assert recorded["run_id"] == parents[0].info.run_id
    assert recorded["experiment_id"] == pipeline["experiment"].experiment_id
    assert {run.data.tags["mlflow.parentRunId"] for run in children} == {recorded["run_id"]}


def test_the_parent_says_which_models_are_diagnostic(pipeline):
    tags = _runs(pipeline)[0][0].data.tags
    roles = json.loads(tags["model_role"])
    assert roles["per_class_measured"] == "diagnostic_upper_bound"
    assert {role for model, role in roles.items() if model != "per_class_measured"} == {"model"}
    assert tags["a1_diagnostic_models"] == "per_class_measured"


def test_a_resumed_run_reuses_the_same_parent(pipeline, in_store):
    parents_before, children_before = _runs(pipeline)
    again = run_plans(pipeline["a1"], groups=True, dev_mode=True)
    parents, children = _runs(pipeline)
    assert again.computed == []
    assert [run.info.run_id for run in parents] == [run.info.run_id for run in parents_before]
    assert len(children) == len(children_before)


# --- what evaluate.py and make_figures.py add ------------------------------------------------------


def test_the_parent_holds_every_evaluation_table(pipeline):
    run_id = _manifest(pipeline)["mlflow"]["run_id"]
    assert _artifacts(pipeline, run_id, "evaluation") == sorted([*TABLES, REGION_NAMES_FILE])


def test_the_parent_holds_the_eight_figures_as_png_and_pdf(pipeline):
    run_id = _manifest(pipeline)["mlflow"]["run_id"]
    expected = sorted(f"{name}{suffix}" for name in FIGURE_NAMES for suffix in (".png", ".pdf"))
    assert _artifacts(pipeline, run_id, "figures") == expected


def test_the_headline_metrics_are_named_as_documented(pipeline):
    metrics = _runs(pipeline)[0][0].data.metrics
    for name in (
        "rmse__random__global",
        "mae__random__per_class_measured",
        "rpiq__random__global_aux",
        "rmse__random__global_aux__terrain",
        "rmse__spatial_kfold_10000m__global_aux",
        "rmse__spatial_kfold_25000m__global_aux",
        "r2__logo_north_east__local",
        "bias__logo_south_west__global",
        "rmse__logo_mean__global",
        "rmse__random__global__lo",
        "rmse__random__global__hi",
        "r2__spatial_kfold_10000m__global__lo",
        "rmse__logo_north_east__global__hi",
        "rmse_drop__logo_north_east__combined",
        "rmse_drop__logo_mean__combined",
    ):
        assert name in metrics, name
    assert all(re.fullmatch(r"[0-9a-z_]+", name) for name in metrics)
    assert metrics["rmse__random__global__lo"] <= metrics["rmse__random__global"] <= metrics["rmse__random__global__hi"]


def test_the_logo_mean_is_the_mean_of_the_regions(pipeline):
    metrics = _runs(pipeline)[0][0].data.metrics
    regions = [metrics["rmse__logo_north_east__global"], metrics["rmse__logo_south_west__global"]]
    assert metrics["rmse__logo_mean__global"] == pytest.approx(sum(regions) / 2)


def test_region_names_are_mapped_back_to_their_real_names(pipeline):
    tags = _runs(pipeline)[0][0].data.tags
    assert tags["region__north_east"] == "North-East" and tags["region__south_west"] == "South-West"


def test_no_mlflow_logs_nothing(pipeline, in_store, tmp_path, monkeypatch):
    _offline(monkeypatch)
    client = pipeline["client"]
    run_id = _manifest(pipeline)["mlflow"]["run_id"]

    def snapshot():
        run = client.get_run(run_id)
        return run.data.metrics, run.data.tags, _artifacts(pipeline, run_id, "evaluation"), len(_runs(pipeline)[1])

    before = snapshot()
    evaluate_main(["--config", str(pipeline["path"]), "--n-boot", "10", "--no-mlflow", "--seed", "7"])
    figures_main(
        ["--config", str(pipeline["path"]), "--n-boot", "10", "--no-relief", "--no-mlflow", "--out", str(tmp_path)]
    )
    plt.close("all")
    assert snapshot() == before


# --- when the MLflow run is not there ------------------------------------------------------------


def test_a_folder_run_without_mlflow_is_refused_with_the_way_out(pipeline, in_store):
    run_plans(pipeline["a1"], kinds=[RANDOM], dev_mode=True, tracking=False, run_name="untracked")
    assert "mlflow" not in _manifest(pipeline, "untracked")
    for main, extra in ((evaluate_main, []), (figures_main, ["--no-relief"])):
        with pytest.raises(A1TrackingError, match="--no-mlflow"):
            main(["--config", str(pipeline["path"]), "--run-name", "untracked", *extra])


def test_a_deleted_parent_run_is_refused_by_every_step(pipeline, in_store):
    run_plans(pipeline["a1"], kinds=[RANDOM], dev_mode=True, run_name="doomed")
    run_id = _manifest(pipeline, "doomed")["mlflow"]["run_id"]
    pipeline["client"].delete_run(run_id)
    for main, extra in ((evaluate_main, []), (figures_main, ["--no-relief"])):
        with pytest.raises(A1TrackingError, match="no longer exists"):
            main(["--config", str(pipeline["path"]), "--run-name", "doomed", *extra])
    with pytest.raises(RunMismatch, match="no longer exists"):
        run_plans(pipeline["a1"], kinds=[RANDOM], dev_mode=True, run_name="doomed")
