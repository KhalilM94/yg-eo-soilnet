"""MLflow for A1: where runs are recorded, and logging finished run folders afterwards.

The tracking URI must come from one place in a fixed order (a folder's recorded store, the
environment, the configuration, the SQLite default) and never follow the migrated mlruns/ file
store; stored paths must be relative to the repository. A run folder synced into a fresh store
must give the runs, metrics, parameters and artifacts the live run logged - from another root too -
and syncing twice must duplicate nothing.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from mlflow import MlflowClient

from yg_eo_soilnet.a1 import tracking
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, write_plans
from yg_eo_soilnet.a1.runner import run_plans
from yg_eo_soilnet.a1.sync import FOLDER_TAG, headline_from_files, sync_run_folder

FAST = {
    "stage2": {
        "estimator": "XGBoost",
        "init_args": {"n_estimators": 20},
        "params": {"model__max_depth": [3]},
        "models": ["global"],
    },
    "inner_cv_folds": 2,
}
REPOSITORY = Path(__file__).resolve().parents[2]


# --- where A1 records ------------------------------------------------------------------------------------


def test_the_tracking_uri_order(tmp_path, monkeypatch):
    db = tmp_path / "a.db"
    sqlite3.connect(db).close()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = type("C", (), {"MLFLOW_TRACKING_URI": "sqlite:///from_config.db"})()
    # A recorded store that is there wins: a parent run lives where it was recorded.
    assert tracking.a1_tracking_uri(config, recorded=f"sqlite:///{db}") == f"sqlite:///{db}"
    # The migrated mlruns/ (or a link to it, as the run worktree has) is never followed.
    old = (REPOSITORY / "mlruns").as_uri()
    assert tracking.a1_tracking_uri(config, recorded=old) == "sqlite:///from_config.db"
    # A per-job store of another machine is not there: the next rule applies.
    assert tracking.a1_tracking_uri(None, recorded="sqlite:////cluster/job/mlflow.db") == (
        tracking.a1_default_tracking_uri()
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://server:5000")
    assert tracking.a1_tracking_uri(config) == "http://server:5000"  # the environment over the file
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert tracking.a1_tracking_uri(None) == f"sqlite:///{REPOSITORY / 'mlflow.db'}"


def test_stored_paths_are_relative_to_the_repository_and_old_ones_still_read(tmp_path):
    inside = REPOSITORY / "configs" / "a1" / "a1_xgb.yml"
    assert tracking.repository_path(inside) == "configs/a1/a1_xgb.yml"
    assert tracking.resolve_repository_path("configs/a1/a1_xgb.yml") == inside
    # An absolute path of another checkout, as older manifests hold: rebased on this repository.
    old = "/home/someone/elsewhere/yg-eo-soilnet-run/configs/a1/a1_xgb.yml"
    assert tracking.resolve_repository_path(old) == inside
    assert tracking.stored_tracking_uri(f"sqlite:///{REPOSITORY / 'mlflow.db'}") == "sqlite:///mlflow.db"
    assert tracking.stored_tracking_uri(f"sqlite:///{tmp_path / 'x.db'}") == f"sqlite:///{tmp_path / 'x.db'}"
    assert (
        tracking.a1_tracking_uri(None, recorded="sqlite:///mlflow.db").endswith(str(REPOSITORY / "mlflow.db"))
        or not (REPOSITORY / "mlflow.db").exists()
    )


def test_new_experiments_keep_artifacts_beside_the_database_and_creation_is_race_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'store.db'}")
    tracking.configure_a1_tracking(None, "Paper_A1_Test")
    found = MlflowClient().get_experiment_by_name("Paper_A1_Test")
    assert found.artifact_location == (tmp_path / "mlruns" / "paper_a1_test").as_uri()
    # A second process finding it missing and creating it too gets the same experiment, no error.
    assert tracking.ensure_experiment("Paper_A1_Test") == found.experiment_id
    with sqlite3.connect(tmp_path / "store.db") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


# --- syncing a finished run folder -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def live(a1_settings_in, tmp_path_factory, mlflow_store):
    """A small demo run (random and leave-one-region-out plans), evaluated, logged live to its own store."""
    from experiments.a1.evaluate import main as evaluate_main

    folder = tmp_path_factory.mktemp("a1_sync")
    with mlflow_store(folder / "live_store") as uri:
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri}, as_file=True, output_dir=str(folder / "out"), **FAST
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[RANDOM, LOGO])
        run_plans(a1, groups=False)
        evaluate_main(["--config", str(path), "--n-boot", "20"])
        run_dir = folder / "out" / "runs" / "default"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        yield {"uri": uri, "run_dir": run_dir, "run_id": manifest["mlflow"]["run_id"], "folder": folder}


def _artifacts(client, run_id, path=None):
    found = set()
    for item in client.list_artifacts(run_id, path):
        found |= _artifacts(client, run_id, item.path) if item.is_dir else {item.path}
    return found


def _snapshot(uri, run_id):
    client = MlflowClient(uri)
    parent = client.get_run(run_id)
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'", max_results=10000
    )
    return {
        "children": sorted(child.info.run_name for child in children),
        "child_metrics": {child.info.run_name: sorted(child.data.metrics) for child in children},
        "parent_metrics": parent.data.metrics,
        "parent_params": sorted(parent.data.params),
        "artifacts": _artifacts(client, run_id),
    }


def _sync_into(monkeypatch, db, run_dir, **kwargs):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    return sync_run_folder(run_dir, **kwargs)


def test_a_synced_folder_matches_the_live_run(live, tmp_path, monkeypatch):
    before = (live["run_dir"] / "manifest.json").read_text()
    result = _sync_into(monkeypatch, tmp_path / "fresh.db", live["run_dir"], write_manifest=False)
    assert result.created_parent and result.children_created > 0
    assert (live["run_dir"] / "manifest.json").read_text() == before  # the folder's record untouched
    synced, original = (
        _snapshot(f"sqlite:///{tmp_path / 'fresh.db'}", result.run_id),
        _snapshot(live["uri"], live["run_id"]),
    )
    assert synced["children"] == original["children"]
    assert synced["child_metrics"] == original["child_metrics"]
    assert synced["parent_params"] == original["parent_params"]
    assert synced["artifacts"] == original["artifacts"]
    # The headline metrics rebuilt from the files are the ones evaluate.py logged, to the digit.
    assert set(synced["parent_metrics"]) == set(original["parent_metrics"])
    for name, value in original["parent_metrics"].items():
        assert synced["parent_metrics"][name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


def test_a_folder_copied_under_another_root_syncs_the_same(live, tmp_path, monkeypatch):
    copy = tmp_path / "another" / "root" / "out"
    shutil.copytree(live["folder"] / "out", copy)
    run_dir = copy / "runs" / "default"
    result = _sync_into(monkeypatch, tmp_path / "copy.db", run_dir)
    synced = _snapshot(f"sqlite:///{tmp_path / 'copy.db'}", result.run_id)
    original = _snapshot(live["uri"], live["run_id"])
    assert synced["children"] == original["children"]
    assert synced["child_metrics"] == original["child_metrics"]
    assert set(synced["parent_metrics"]) == set(original["parent_metrics"])
    assert synced["artifacts"] == original["artifacts"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mlflow"]["run_id"] == result.run_id  # the copy now records its new parent
    tags = MlflowClient(f"sqlite:///{tmp_path / 'copy.db'}").get_run(result.run_id).data.tags
    assert tags[FOLDER_TAG] == str(run_dir) and tags["a1_synced"] == "true"
    assert tags["host_name"] == manifest.get("host", "unknown")


def test_syncing_twice_duplicates_nothing(live, tmp_path, monkeypatch):
    copy = tmp_path / "out"
    shutil.copytree(live["folder"] / "out", copy)
    run_dir = copy / "runs" / "default"
    first = _sync_into(monkeypatch, tmp_path / "twice.db", run_dir)
    again = _sync_into(monkeypatch, tmp_path / "twice.db", run_dir)
    assert again.run_id == first.run_id and not again.created_parent
    assert again.children_created == 0 and again.children_kept == first.children_created
    assert again.metrics_logged == 0
    client = MlflowClient(f"sqlite:///{tmp_path / 'twice.db'}")
    experiment = client.get_run(first.run_id).info.experiment_id
    assert len(client.search_runs([experiment], max_results=10000)) == 1 + first.children_created
    # Matched by its tags too, when the manifest no longer records the run.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest.pop("mlflow")
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    third = _sync_into(monkeypatch, tmp_path / "twice.db", run_dir)
    assert third.run_id == first.run_id and third.children_created == 0


def test_headline_rebuilt_without_evaluation_is_empty(tmp_path):
    assert headline_from_files(tmp_path, 42) == {}


@pytest.mark.slow
@pytest.mark.skipif(
    not (REPOSITORY / "results" / "a1_e5" / "runs" / "default" / "manifest.json").exists()
    or not (REPOSITORY / "mlflow.db").exists(),
    reason="needs results/a1_e5 and the migrated mlflow.db",
)
def test_e5_synced_into_a_scratch_store_matches_the_migrated_one(tmp_path, monkeypatch):
    run_dir = REPOSITORY / "results" / "a1_e5" / "runs" / "default"
    recorded = json.loads((run_dir / "manifest.json").read_text())["mlflow"]["run_id"]
    result = _sync_into(monkeypatch, tmp_path / "e5.db", run_dir, write_manifest=False)
    synced = _snapshot(f"sqlite:///{tmp_path / 'e5.db'}", result.run_id)
    migrated = _snapshot(f"sqlite:///{REPOSITORY / 'mlflow.db'}", recorded)
    # The live E5 parent holds each child up to three times (every resumed part logged all its plan's
    # variants again); the synced one holds each once.
    assert synced["children"] == sorted(set(migrated["children"]))
    assert synced["child_metrics"] == migrated["child_metrics"]
    assert set(synced["parent_params"]) >= set(migrated["parent_params"]) - {"a1_seeds"}


def test_a_resumed_run_stores_its_settings_relative_and_records_the_host(live, monkeypatch):
    import socket

    run_dir = live["run_dir"]
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest.get("host") == socket.gethostname()  # recorded when the folder was made
    # The run worktree's absolute path, as older manifests hold it, is stored relative.
    assert tracking.repository_path("/home/x/yg-eo-soilnet-run/configs/a1/a1_xgb.yml") == "configs/a1/a1_xgb.yml"
