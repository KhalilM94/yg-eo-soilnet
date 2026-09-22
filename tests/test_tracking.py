"""Tracking: where runs are recorded, how a run opens and closes, and how models are registered
and promoted.

Where runs are recorded, and under which experiment.

The defect this addresses: the original experiment was created in a different checkout, so its
`artifact_location` - an absolute path fixed at creation - pointed at that checkout forever after.
Metadata landed in this repo's `mlruns/` while artifacts went to the old one, which is why the
recent model directories here contain only `meta.yaml`, `metrics/`, `params/` and `tags/` and no
`artifacts/` at all. A new experiment created under the intended tracking root gets a correct
`artifact_location` from MLflow by itself.
"""

import datetime
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as loggers_module
import yg_eo_soilnet.tracking as tracking_module
from yg_eo_soilnet import tracking
from yg_eo_soilnet.artifacts import ArtifactLayout
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.tracking import (
    CHAMPION_METRIC,
    DEFAULT_EXPERIMENT_NAME,
    HOST_NAME_TAG,
    HOST_PID_TAG,
    close_stale_runs,
    configure_tracking,
    default_tracking_uri,
    promote_if_better,
    repair_corrupt_runs,
    resolve_tracking_uri,
    run_owner_tags,
    start_child_run,
    tracking_settings,
)


@pytest.fixture(autouse=True)
def clean_tracking_env(monkeypatch):
    """Start every test from an unset tracking environment.

    `mlflow.set_tracking_uri()` writes MLFLOW_TRACKING_URI into os.environ, so any earlier test in
    the session that configured tracking for real would otherwise leak its value into the env-first
    precedence being tested here.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)


def test_default_tracking_uri_is_anchored_to_the_repo_not_the_cwd() -> None:
    """A run launched from elsewhere must not quietly start a second, empty mlruns/ beside itself."""
    uri = default_tracking_uri()

    assert uri.startswith("file://")
    assert uri.endswith("/mlruns")
    assert "yg-eo-soilnet" in uri


def test_configured_uri_wins_over_the_default() -> None:
    config = SimpleNamespace(MLFLOW_TRACKING_URI="sqlite:///mlflow.db")
    assert resolve_tracking_uri(config) == "sqlite:///mlflow.db"


def test_blank_or_missing_uri_falls_back_to_the_repo_default() -> None:
    assert resolve_tracking_uri(SimpleNamespace(MLFLOW_TRACKING_URI="")) == default_tracking_uri()
    assert resolve_tracking_uri(SimpleNamespace(MLFLOW_TRACKING_URI="   ")) == default_tracking_uri()
    assert resolve_tracking_uri(SimpleNamespace()) == default_tracking_uri()
    assert resolve_tracking_uri(None) == default_tracking_uri()


# --- settings reader --------------------------------------------------------


def test_settings_read_the_common_block(tmp_path) -> None:
    path = tmp_path / "main_config.yml"
    path.write_text(
        "common:\n  MLFLOW_EXPERIMENT_NAME: 'From_Yaml'\n  MLFLOW_TRACKING_URI: 'file:///tmp/x'\n",
        encoding="utf-8",
    )

    settings = tracking_settings(path)

    assert settings.MLFLOW_EXPERIMENT_NAME == "From_Yaml"
    assert settings.MLFLOW_TRACKING_URI == "file:///tmp/x"


def test_env_overrides_the_yaml(tmp_path, monkeypatch) -> None:
    path = tmp_path / "main_config.yml"
    path.write_text("common:\n  MLFLOW_EXPERIMENT_NAME: 'From_Yaml'\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "From_Env")

    assert tracking_settings(path).MLFLOW_EXPERIMENT_NAME == "From_Env"


def test_settings_do_not_need_a_valid_config_tree(tmp_path) -> None:
    """Tracking is configured first, so it must not depend on the data spec or the registries.

    Building a full Config here would mean a typo in an unrelated file decides that runs land
    nowhere - the failure mode that made this a separate reader.
    """
    path = tmp_path / "main_config.yml"
    path.write_text(
        "common:\n  MLFLOW_EXPERIMENT_NAME: 'Fine'\n  DATA_SPEC_PATH: 'does_not_exist.yml'\n",
        encoding="utf-8",
    )

    assert tracking_settings(path).MLFLOW_EXPERIMENT_NAME == "Fine"


def test_settings_survive_an_unreadable_or_broken_config(tmp_path) -> None:
    broken = tmp_path / "broken.yml"
    broken.write_text("common: [this is not a mapping\n", encoding="utf-8")

    assert tracking_settings(broken).MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME
    assert tracking_settings(tmp_path / "missing.yml").MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME
    assert tracking_settings(None).MLFLOW_EXPERIMENT_NAME == DEFAULT_EXPERIMENT_NAME


# --- configure_tracking -----------------------------------------------------


def test_configure_sets_the_uri_before_the_experiment(monkeypatch) -> None:
    """Order matters: an experiment created against the wrong root bakes in the wrong
    artifact_location, which is the whole defect."""
    calls: list[str] = []

    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: calls.append("uri"))
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: calls.append("experiment"))

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI="", MLFLOW_EXPERIMENT_NAME="X"))

    assert calls == ["uri", "experiment"]


def test_configure_opts_into_the_file_store(monkeypatch) -> None:
    """MLflow 3.14 put the filesystem backend in maintenance mode and raises without this.

    Only the `pixi run mlflow` task used to set it, so whether `python main.py` could write to
    mlruns/ at all depended on how the process happened to be launched.
    """
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", MagicMock())

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI=""))

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_configure_leaves_a_database_backend_alone(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", MagicMock())

    configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI="sqlite:///mlflow.db"))

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_configure_recreates_a_deleted_experiment(monkeypatch) -> None:
    import mlflow

    attempts = {"set": 0}

    def flaky_set_experiment(name):
        attempts["set"] += 1
        if attempts["set"] == 1:
            raise mlflow.exceptions.MlflowException("deleted")

    created = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", flaky_set_experiment)
    monkeypatch.setattr(tracking.mlflow, "create_experiment", created)

    name = configure_tracking(SimpleNamespace(MLFLOW_EXPERIMENT_NAME="Revived"))

    created.assert_called_once_with("Revived")
    assert name == "Revived"


def test_explicit_experiment_name_wins(monkeypatch) -> None:
    """The HPO path passes its own name while reusing the same tracking root."""
    seen: list[str] = []
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_experiment", lambda name: seen.append(name))

    configure_tracking(SimpleNamespace(MLFLOW_EXPERIMENT_NAME="Training"), experiment_name="Soil_HPO")

    assert seen == ["Soil_HPO"]


def test_the_default_experiment_is_not_the_one_with_the_stale_artifact_location() -> None:
    """A clean experiment is the fix; reusing the old name would inherit its broken path."""
    assert DEFAULT_EXPERIMENT_NAME != "Soil_Model_Training_Experiment"


# --- the run lifecycle ------------------------------------------------------------------------
# Runs that outlive their process, and inference paid for twice.
#
# A multi-target run once produced a model for only its first target and looked like a bug in the
# target grouping. It was not: the kernel's OOM killer had taken the process partway through. The
# evidence was hard to reach because the run recorded nothing about what it had planned to fit, and
# what it left behind was still marked RUNNING - which reads as "in progress", or as a model that was
# successfully made.


def test_a_run_records_which_process_wrote_it() -> None:
    tags = run_owner_tags()
    assert tags[HOST_PID_TAG] == str(os.getpid())
    assert tags[HOST_NAME_TAG]


# --- the stale sweep -------------------------------------------------------


def _dead_pid() -> int:
    """A pid that certainly does not exist: fork a child and reap it."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _fake_client(runs, terminated):
    return SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="1"),
        search_runs=lambda experiment_ids, filter_string, max_results: runs,
        set_terminated=lambda run_id, status: terminated.append((run_id, status)),
    )


def _run(run_id, tags):
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))


def test_a_run_abandoned_by_a_dead_process_is_marked_killed(monkeypatch) -> None:
    import socket

    host = socket.gethostname()
    terminated: list = []
    runs = [_run("abandoned", {HOST_NAME_TAG: host, HOST_PID_TAG: str(_dead_pid())})]
    monkeypatch.setattr(
        loggers_module.mlflow.tracking, "MlflowClient", lambda: _fake_client(runs, terminated)
    )

    assert close_stale_runs("exp") == ["abandoned"]
    assert terminated == [("abandoned", "KILLED")]


def test_the_sweep_never_touches_a_run_that_could_still_be_writing(monkeypatch) -> None:
    """The failure this guards against is worse than the one it fixes.

    Sweeping on "status is RUNNING" alone would let one training process terminate a second one
    running beside it. Only a run from THIS host whose pid is genuinely gone is closed.
    """
    import socket

    host = socket.gethostname()
    terminated: list = []
    runs = [
        # Alive: this very process.
        _run("mine", {HOST_NAME_TAG: host, HOST_PID_TAG: str(os.getpid())}),
        # Another machine's run, whose pid means nothing here.
        _run("elsewhere", {HOST_NAME_TAG: "some-other-box", HOST_PID_TAG: str(_dead_pid())}),
        # Written before runs carried ownership tags: left alone rather than guessed at.
        _run("untagged", {}),
    ]
    monkeypatch.setattr(
        loggers_module.mlflow.tracking, "MlflowClient", lambda: _fake_client(runs, terminated)
    )

    assert close_stale_runs("exp") == []
    assert terminated == []


def test_a_sweep_that_cannot_reach_the_store_does_not_stop_the_run(monkeypatch) -> None:
    def explode():
        raise RuntimeError("tracking server is down")

    monkeypatch.setattr(loggers_module.mlflow.tracking, "MlflowClient", explode)
    logger = MagicMock()

    assert close_stale_runs("exp", logger=logger) == []
    logger.warning.assert_called_once()


# --- corrupt run repair ----------------------------------------------------
# The sweep above cannot run at all while a run in the same experiment has a truncated meta.yaml.
# MLflow rewrites that file in place to END a run, so a process death inside that window leaves
# zero bytes - and `_read_persisted_run_info_dict` then calls `.copy()` on the None its parser
# returns. `_list_run_infos` catches only `MissingConfigException`, so it escapes every read of the
# experiment: here it surfaced at the END of a successful run, in the parent summary.


def _store_with_a_run(tmp_path, run_id="run-abc", run_name="Run_20260906_155825", tags=None):
    """A file-backed store holding one finished run, written the way MLflow writes one."""
    mlflow.set_tracking_uri(tmp_path.as_uri())
    experiment_dir = tmp_path / "1"
    run_dir = experiment_dir / run_id
    (run_dir / "tags").mkdir(parents=True)
    (run_dir / "artifacts").mkdir()

    (experiment_dir / "meta.yaml").write_text(
        "artifact_location: " + experiment_dir.as_uri() + "\n"
        "creation_time: 1787051008023\n"
        "experiment_id: '1'\n"
        "last_update_time: 1787051008023\n"
        "lifecycle_stage: active\n"
        "name: exp\n"
    )
    (run_dir / "meta.yaml").write_text(
        "artifact_uri: " + (run_dir / "artifacts").as_uri() + "\n"
        "end_time: 1788273472365\n"
        "entry_point_name: ''\n"
        "experiment_id: '1'\n"
        "lifecycle_stage: active\n"
        f"run_id: {run_id}\n"
        f"run_name: {run_name}\n"
        "source_name: ''\n"
        "source_type: 4\n"
        "source_version: ''\n"
        "start_time: 1788273469596\n"
        "status: 3\n"
        "tags: []\n"
        "user_id: kmisbah\n"
    )
    for tag, value in {"mlflow.runName": run_name, "mlflow.user": "kmisbah", **(tags or {})}.items():
        (run_dir / "tags" / tag).write_text(value)
    return run_dir


def test_a_run_whose_meta_was_truncated_is_rebuilt_as_killed(tmp_path) -> None:
    """The regression test: one empty meta.yaml used to make every search_runs raise."""
    run_dir = _store_with_a_run(tmp_path)
    (run_dir / "meta.yaml").write_text("")

    client = mlflow.tracking.MlflowClient()
    with pytest.raises(AttributeError):
        client.search_runs(["1"], max_results=10)

    assert repair_corrupt_runs("exp") == [run_dir.name]

    runs = client.search_runs(["1"], max_results=10)
    assert [run.info.run_id for run in runs] == [run_dir.name]
    assert runs[0].info.status == "KILLED"
    # Recovered from the sidecar files a truncation never touched, not invented.
    assert runs[0].info.run_name == "Run_20260906_155825"
    assert runs[0].data.tags["mlflow.user"] == "kmisbah"
    # The start is carried by the run name, to the second: 2026-09-06 15:58:25 local.
    assert runs[0].info.start_time == int(
        datetime.datetime(2026, 9, 6, 15, 58, 25).timestamp() * 1000
    )
    assert runs[0].info.end_time >= runs[0].info.start_time


def test_a_repair_leaves_a_healthy_run_byte_for_byte_alone(tmp_path) -> None:
    run_dir = _store_with_a_run(tmp_path)
    before = (run_dir / "meta.yaml").read_bytes()

    assert repair_corrupt_runs("exp") == []
    assert (run_dir / "meta.yaml").read_bytes() == before


def test_a_repair_never_touches_a_run_a_live_process_could_still_be_writing(tmp_path) -> None:
    """Same rule as the sweep: an empty file may simply be mid-write on this host."""
    import socket

    run_dir = _store_with_a_run(
        tmp_path, tags={HOST_NAME_TAG: socket.gethostname(), HOST_PID_TAG: str(os.getpid())}
    )
    (run_dir / "meta.yaml").write_text("")
    logger = MagicMock()

    assert repair_corrupt_runs("exp", logger=logger) == []
    assert (run_dir / "meta.yaml").read_text() == ""
    logger.warning.assert_called_once()


def test_a_repair_reclaims_a_run_whose_process_is_gone(tmp_path) -> None:
    import socket

    run_dir = _store_with_a_run(
        tmp_path, tags={HOST_NAME_TAG: socket.gethostname(), HOST_PID_TAG: str(_dead_pid())}
    )
    (run_dir / "meta.yaml").write_text("")

    assert repair_corrupt_runs("exp") == [run_dir.name]


def test_a_run_directory_with_no_meta_at_all_is_left_to_mlflow(tmp_path) -> None:
    """A MISSING meta.yaml already has a handler upstream; only an empty one does not."""
    run_dir = _store_with_a_run(tmp_path)
    (run_dir / "meta.yaml").unlink()

    assert repair_corrupt_runs("exp") == []
    assert not (run_dir / "meta.yaml").exists()


def test_a_repair_is_a_no_op_against_a_real_tracking_backend(tmp_path, monkeypatch) -> None:
    """A database or HTTP store has no meta.yaml to truncate, and no filesystem to walk."""
    _store_with_a_run(tmp_path)
    monkeypatch.setattr(tracking_module.mlflow, "get_tracking_uri", lambda: "postgresql://host/db")
    client = MagicMock()
    monkeypatch.setattr(tracking_module.mlflow.tracking, "MlflowClient", client)

    assert repair_corrupt_runs("exp") == []
    client.assert_not_called()


# --- nesting ---------------------------------------------------------------


def test_a_child_run_nests_under_a_parent_but_stands_alone_without_one() -> None:
    """`nested=True` is an ERROR when nothing is active.

    Hardcoding it tied the trainers to being called from inside main.py's parent run, but they are
    also driven directly - by tests, and by anyone fitting a single model.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()

    with start_child_run("standalone") as run:
        assert mlflow.active_run().info.run_id == run.info.run_id
        standalone_id = run.info.run_id
    # Read back from the store: the object start_run returns is a snapshot taken before the tags
    # were written.
    standalone_tags = client.get_run(standalone_id).data.tags
    assert standalone_tags[HOST_PID_TAG] == str(os.getpid())
    assert "mlflow.parentRunId" not in standalone_tags

    with mlflow.start_run(run_name="parent") as parent:
        with start_child_run("child") as child:
            child_id = child.info.run_id
    assert client.get_run(child_id).data.tags["mlflow.parentRunId"] == parent.info.run_id


def test_a_child_that_fails_to_tag_does_not_strand_its_parent(monkeypatch) -> None:
    """start_run pushes onto the active-run stack; the caller's `with` is what pops it.

    If set_tags raises in between, the ActiveRun never reaches a `with` and nothing pops it. That
    is worse than one lost child: mlflow.end_run() pops the TOP of the stack rather than a named
    run, so the parent's own `with` would close the orphan and leave the PARENT at RUNNING forever
    - which is exactly the "parent run never ends" symptom, reachable with no OOM involved.
    """
    import mlflow

    with mlflow.start_run(run_name="parent") as parent:
        monkeypatch.setattr(
            mlflow, "set_tags", MagicMock(side_effect=RuntimeError("tracking store down"))
        )
        with pytest.raises(RuntimeError):
            start_child_run("doomed")
        monkeypatch.undo()

        # The parent, not the orphan, is what is active again.
        assert mlflow.active_run().info.run_id == parent.info.run_id

    assert mlflow.active_run() is None
    assert mlflow.tracking.MlflowClient().get_run(parent.info.run_id).info.status == "FINISHED"


# --- inference paid for once -----------------------------------------------


class _CountingEstimator:
    """Counts prediction passes, by row, the way an in-context model would charge for them."""

    def __init__(self):
        self.rows_predicted = 0
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        self.rows_predicted += len(X)
        return np.arange(len(X), dtype=float)


def test_mlflow_evaluate_scores_the_predictions_already_computed(monkeypatch) -> None:
    """Static-dataset evaluation: no model, so no reload and no second inference pass."""
    captured = {}
    monkeypatch.setattr(
        loggers_module.mlflow.models, "evaluate", lambda **kwargs: captured.update(kwargs)
    )

    frame = pd.DataFrame({"target_a": [1.0, 2.0, 3.0], "prediction": [1.1, 1.9, 3.2]})
    ChildRunLogger()._evaluate_sklearn_target(frame, "target_a")

    assert captured["predictions"] == "prediction"
    assert captured["targets"] == "target_a"
    # The absence is the point: passing a model URI made MLflow reload the model and predict the
    # test set all over again, duplicating what the evaluation frame had just done.
    assert "model" not in captured
    assert captured["data"] is frame


def test_evaluate_is_skipped_when_the_frame_has_nothing_to_score(monkeypatch) -> None:
    called = MagicMock()
    monkeypatch.setattr(loggers_module.mlflow.models, "evaluate", called)

    logger = ChildRunLogger()
    logger._evaluate_sklearn_target(None, "target_a")
    logger._evaluate_sklearn_target(pd.DataFrame({"prediction": [1.0]}), "target_a")

    called.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_the_train_fit_diagnostic_is_switchable(enabled, monkeypatch) -> None:
    """r2_train_fit costs a full pass over the TRAINING split for one number.

    Free for a tree, minutes per target for an in-context model - which is the difference between
    a run that finishes and one the OOM killer gets to first.
    """
    # Serialising a locally-defined estimator is not what this test is about, and skops refuses it.
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    estimator = _CountingEstimator()
    X_train = pd.DataFrame({"feature": np.arange(50, dtype=float)})
    X_test = pd.DataFrame({"feature": np.arange(10, dtype=float)})
    y_train = pd.Series(np.arange(50, dtype=float), name="target_a")
    y_test = pd.Series(np.arange(10, dtype=float), name="target_a")

    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_log_metric_dict", "_promote_champion",
                 "_log_plots", "_log_shap_slice", "_write_split_summary", "_write_json_artifact",
                 "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    # The explanation is built once, on the model run, and sliced per target. (None, {}) is "nothing
    # to explain"; a bare MagicMock would fail the tuple unpack at the call site.
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logged: dict = {}
    logger._log_metric_dict = lambda metrics: logged.update(metrics)

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=False,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=enabled,
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert ("r2_train_fit" in logged) is enabled
    # 10 test rows either way; the 50 training rows are what the switch buys back.
    assert estimator.rows_predicted == (60 if enabled else 10)


def test_the_train_fit_diagnostic_can_be_declined_per_model(monkeypatch) -> None:
    """One expensive estimator should not force the diagnostic off for the cheap ones.

    The cost is a property of the model, not of a row budget, so it can only be declined by name -
    the same reasoning EXPLAIN_SKIP_MODELS is built on.
    """
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    estimator = _CountingEstimator()
    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_promote_champion", "_log_plots",
                 "_log_shap_slice", "_write_split_summary", "_write_json_artifact",
                 "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logged: dict = {}
    logger._log_metric_dict = lambda metrics: logged.update(metrics)

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=False,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=True,
                LOG_TRAIN_FIT_METRIC_SKIP_MODELS=["Toy"],
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=estimator,
            X_train=pd.DataFrame({"feature": np.arange(50, dtype=float)}),
            y_train=pd.Series(np.arange(50, dtype=float), name="target_a"),
            X_test=pd.DataFrame({"feature": np.arange(10, dtype=float)}),
            y_test=pd.Series(np.arange(10, dtype=float), name="target_a"),
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert "r2_train_fit" not in logged
    # The 50 training rows were never asked for; only the one test pass happened.
    assert estimator.rows_predicted == 10


# --- registration follows the metrics --------------------------------------


def test_a_model_enters_the_registry_only_after_its_metrics_exist(monkeypatch) -> None:
    """Registering at log_model time meant a run killed partway through still minted a version.

    Four such versions of organic_matter_g_kg_TabICL accumulated that way, each READY, each backed
    by a run stuck at RUNNING with no rmse_test - and promote_if_better would have refused every
    one of them. Registration is now a separate step that the metrics happen before.
    """
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    order: list[str] = []
    monkeypatch.setattr(
        loggers_module.mlflow,
        "register_model",
        lambda uri, name: order.append("register") or SimpleNamespace(version=3),
    )

    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_log_plots", "_log_shap_slice",
                 "_write_split_summary", "_write_json_artifact", "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logger._log_metric_dict = lambda metrics: order.append("metrics")
    promote = MagicMock(return_value={})
    logger._promote_champion = promote

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=True,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=False,
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=_CountingEstimator(),
            X_train=pd.DataFrame({"feature": np.arange(50, dtype=float)}),
            y_train=pd.Series(np.arange(50, dtype=float), name="target_a"),
            X_test=pd.DataFrame({"feature": np.arange(10, dtype=float)}),
            y_test=pd.Series(np.arange(10, dtype=float), name="target_a"),
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert order.index("metrics") < order.index("register")
    # The version reaches promotion, which is the only consumer that decides on it.
    assert promote.call_args.args[-1] == 3


def test_a_registry_outage_does_not_discard_a_finished_run(monkeypatch) -> None:
    """By this point the model is logged and servable; only its registry entry is missing.

    Throwing away a completed fit over that is the worse trade, so it is recorded on the run and
    escalated only under FAIL_ON_MODEL_ERROR.
    """
    monkeypatch.setattr(
        loggers_module.mlflow,
        "register_model",
        MagicMock(side_effect=RuntimeError("registry unreachable")),
    )
    logger = ChildRunLogger()
    config = SimpleNamespace(MLFLOW_REGISTER_MODELS=True, FAIL_ON_MODEL_ERROR=False)
    model_info = SimpleNamespace(model_uri="models:/toy/1")

    import mlflow

    with mlflow.start_run(run_name="probe") as run:
        assert logger._register_sklearn_model(config, model_info, "target_a_Toy") is None

    tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
    assert tags["model_registered"] == "false"
    assert "registry unreachable" in tags["model_registration_error"]

    config.FAIL_ON_MODEL_ERROR = True
    with mlflow.start_run(run_name="strict"):
        with pytest.raises(RuntimeError):
            logger._register_sklearn_model(config, model_info, "target_a_Toy")


# --- immutable params ------------------------------------------------------
# MLflow params cannot change value once written. The parent run is written to twice - once at the
# start of training and once in the summary at the end - so a key written in both places with two
# different renderings does not fail fast. It fails AFTER every model has been fitted, logged and
# registered, and takes the summary, the leaderboard and the run's FINISHED status with it.


def _params_of(run_id):
    import mlflow

    return mlflow.tracking.MlflowClient().get_run(run_id).data.params


def test_log_params_once_writes_new_keys_and_tolerates_an_identical_relog() -> None:
    import mlflow

    from yg_eo_soilnet.tracking import log_params_once

    with mlflow.start_run() as run:
        log_params_once({"a": "1", "b": 2})
        log_params_once({"a": "1", "c": "3"})  # 'a' unchanged: allowed, and must not raise
        params = _params_of(run.info.run_id)

    assert params == {"a": "1", "b": "2", "c": "3"}


def test_log_params_once_keeps_the_first_value_and_warns_on_a_clash() -> None:
    import mlflow

    from yg_eo_soilnet.tracking import log_params_once

    logger = MagicMock()
    with mlflow.start_run() as run:
        log_params_once({"TARGET_COLUMNS": "clay_pct,sand_pct"})
        # The exact shape of the bug: same key, different rendering of the same information.
        log_params_once({"TARGET_COLUMNS": ["clay_pct", "sand_pct"], "other": "kept"}, logger=logger)
        params = _params_of(run.info.run_id)

    assert params["TARGET_COLUMNS"] == "clay_pct,sand_pct"
    # The rest of the batch still lands - the file store applies params one at a time and would
    # otherwise have written some and dropped others.
    assert params["other"] == "kept"
    assert "TARGET_COLUMNS" in logger.warning.call_args[0][0]


def test_the_target_plan_and_the_parent_summary_can_share_one_run() -> None:
    """The regression itself, and the test whose absence let it ship.

    `_log_target_plan` writes TARGET_COLUMNS at the start of training and `log_parent_summary`
    used to write it again at the end in a different format. Nothing exercised both against one
    run, so a green suite said nothing about it.
    """
    import mlflow

    import main as main_module

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["clay_pct", "sand_pct", "total_silt_pct"], MULTI_TARGET_MODE="per_target"
    )
    trainer.logger = MagicMock()

    with mlflow.start_run() as run:
        trainer._log_target_plan({("clay_pct",): {}, ("sand_pct",): {}}, {})
        # The payload log_parent_summary writes at the end of a run.
        loggers_module.log_params_once({
            "RANDOM_SEED": 42,
            "COLUMNS_TO_TRANSFORM": [],
            "SPLIT_TEST_SIZE": 0.2,
        })
        params = _params_of(run.info.run_id)

    assert params["TARGET_COLUMNS"] == "clay_pct,sand_pct,total_silt_pct"
    assert params["MULTI_TARGET_MODE"] == "per_target"
    assert params["sklearn_target_groups"] == "clay_pct | sand_pct"
    assert params["RANDOM_SEED"] == "42"


def test_the_parent_summary_no_longer_writes_target_columns(monkeypatch) -> None:
    """One writer, and it is the early one. Pins the invariant rather than the symptom."""
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    payloads: list[dict] = []
    monkeypatch.setattr(loggers_module, "log_params_once", lambda params, **kw: payloads.append(params))
    monkeypatch.setattr(loggers_module.mlflow, "set_tags", MagicMock())
    parent = ParentRunLogger()
    monkeypatch.setattr(parent, "_collect_leaderboard", lambda run_id: pd.DataFrame())

    config = SimpleNamespace(
        DATA_FOLDER="d", DATA_FILE="f.csv", RANDOM_SEED=42,
        TARGET_COLUMNS=["clay_pct"], COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False, CLUSTERING_STRATEGY={}, SPLIT_STRATEGY="kfold",
    )
    try:
        parent.log_parent_summary("run-1", SimpleNamespace(config=config))
    except Exception:
        # Whatever the leaderboard half does is not this test's business; the params are.
        pass

    assert payloads, "log_parent_summary should log params through log_params_once"
    assert all("TARGET_COLUMNS" not in payload for payload in payloads)


# --- registration and champion promotion ------------------------------------------------------
# Registration and champion promotion.
#
# The registry was empty: models were logged against runs but never registered, so there were no
# versions, no aliases and no `models:/Name@champion` URIs. Registration is now on by default and the
# champion alias moves only on a measured improvement.
#
# The promotion rules are where this can go quietly wrong - a rule that promotes on a missing metric,
# or churns the alias on a tie, is not visible in any artifact - so each one has its own test.


class _Client:
    """A registry stub: an aliased version, and the metric on the run behind it."""

    def __init__(self, alias_version=None, incumbent_metric=None, run_missing=False):
        self._alias_version = alias_version
        self._incumbent_metric = incumbent_metric
        self._run_missing = run_missing
        self.alias_calls: list[tuple] = []

    def get_model_version_by_alias(self, name, alias):
        if self._alias_version is None:
            raise RuntimeError(f"no version aliased {alias}")
        return SimpleNamespace(version=self._alias_version, run_id="run-incumbent")

    def get_run(self, run_id):
        if self._run_missing:
            raise RuntimeError("run deleted")
        metrics = {} if self._incumbent_metric is None else {CHAMPION_METRIC: self._incumbent_metric}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, version))


def test_the_first_version_becomes_champion() -> None:
    client = _Client(alias_version=None)

    decision = promote_if_better("om_soil_cnn", "1", 4.2, client=client)

    assert decision["promoted"] is True
    assert client.alias_calls == [("om_soil_cnn", "champion", "1")]


def test_a_better_score_promotes() -> None:
    client = _Client(alias_version="1", incumbent_metric=5.30)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is True
    assert decision["candidate"] == 4.81
    assert decision["incumbent"] == 5.30
    assert client.alias_calls == [("om_soil_cnn", "champion", "2")]


def test_a_worse_score_keeps_the_incumbent() -> None:
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", 5.62, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []
    assert "does not beat" in decision["reason"]


def test_an_equal_score_keeps_the_incumbent() -> None:
    """Re-running the same config must not churn the alias."""
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []


def test_a_version_without_the_metric_is_never_promoted() -> None:
    """A degenerate fit produces no metrics; shipping it because it has no worse score is the
    failure this guards against."""
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", None, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []
    assert CHAMPION_METRIC in decision["reason"]


def test_a_non_finite_score_is_never_promoted() -> None:
    client = _Client(alias_version="1", incumbent_metric=4.81)

    assert promote_if_better("om_soil_cnn", "2", float("nan"), client=client)["promoted"] is False
    assert client.alias_calls == []


def test_an_unreadable_incumbent_is_replaced_and_said_so() -> None:
    """A candidate we can score beats one we cannot - but the reason has to be recorded."""
    client = _Client(alias_version="1", run_missing=True)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is True
    assert "no readable" in decision["reason"]
    assert client.alias_calls == [("om_soil_cnn", "champion", "2")]


def test_an_incumbent_whose_run_lacks_the_metric_is_replaced() -> None:
    client = _Client(alias_version="1", incumbent_metric=None)

    assert promote_if_better("om_soil_cnn", "2", 4.81, client=client)["promoted"] is True


def test_an_unregistered_model_is_not_promoted() -> None:
    client = _Client()

    decision = promote_if_better("om_soil_cnn", None, 4.81, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []


def test_the_decision_records_both_scores_for_audit() -> None:
    client = _Client(alias_version="3", incumbent_metric=5.0)

    decision = promote_if_better("om_soil_cnn", "4", 4.0, client=client)

    assert decision["metric"] == CHAMPION_METRIC
    assert decision["candidate"] == 4.0
    assert decision["incumbent"] == 5.0
    assert decision["candidate_version"] == "4"
    assert decision["incumbent_version"] == "3"


def test_a_higher_is_better_metric_flips_the_comparison() -> None:
    """The direction comes from yg_eo_soilnet.metrics, not from a second copy of the rule here."""
    client = _Client(alias_version="1", incumbent_metric=0.30)

    decision = promote_if_better("om_soil_cnn", "2", 0.55, client=client, metric_name="r2_test")

    assert decision["promoted"] is True


# --- the logger's use of it -------------------------------------------------


def test_the_logger_records_the_decision_without_raising() -> None:
    """A registry that will not take the alias must not lose a finished training run."""
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    decision = ChildRunLogger()._promote_champion("om", "soil_cnn", {CHAMPION_METRIC: 4.0}, None)

    assert decision["promoted"] is False
    assert "not registered" in decision["reason"]


def test_the_registered_name_matches_the_logged_model_name() -> None:
    """Both families register under the same convention, so a target's versions accumulate in one
    place regardless of which family produced them."""
    assert ArtifactLayout.logged_model_name("organic_matter_g_kg", "soil_cnn") == (
        "organic_matter_g_kg_soil_cnn"
    )


# --- registration wiring ----------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_the_lightning_path_honours_the_registration_switch(monkeypatch, tmp_path, enabled) -> None:
    import mlflow.pyfunc
    import mlflow.pytorch

    import yg_eo_soilnet.logger.mlflow_loggers as loggers

    captured: dict = {}
    monkeypatch.setattr(mlflow.pytorch, "save_model", lambda *a, **k: None)
    monkeypatch.setattr(loggers.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(
        mlflow.pyfunc,
        "log_model",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(registered_model_version="1"),
    )

    checkpoint = tmp_path / "a.ckpt"
    checkpoint.write_bytes(b"x")

    loggers.ChildRunLogger()._log_lightning_serialized_model(
        model=SimpleNamespace(),
        model_name="soil_cnn",
        target="om",
        bundle=None,
        best_model_path=str(checkpoint),
        config=SimpleNamespace(MLFLOW_REGISTER_MODELS=enabled),
    )

    expected = "om_soil_cnn" if enabled else None
    assert captured["registered_model_name"] == expected


# --- a failed model log cannot look like a success ---------------------------
# This is what let the auxiliary-column regression run unnoticed: the model failed to log, nothing
# was registered, and the run still finished looking healthy because the only evidence was inside
# meta/run_summary.json.


def _run_lightning_child(monkeypatch, *, log_model_raises: bool):
    import mlflow.pyfunc
    import mlflow.pytorch
    import pandas as pd

    import yg_eo_soilnet.logger.mlflow_loggers as loggers

    tags: dict = {}
    monkeypatch.setattr(loggers.mlflow, "set_tags", lambda values: tags.update(values))
    monkeypatch.setattr(loggers.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(loggers.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(loggers.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pytorch, "save_model", lambda *a, **k: None)

    def log_model(**kwargs):
        if log_model_raises:
            raise ValueError("Batch carries 0 lab column(s) but this model resolved 24")
        return SimpleNamespace(registered_model_version="1")

    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)

    summaries: list[dict] = []
    logger = loggers.ChildRunLogger()
    monkeypatch.setattr(logger, "_write_json_artifact", lambda payload, *a, **k: summaries.append(payload))
    monkeypatch.setattr(logger, "_promote_champion", lambda *a, **k: {"promoted": False})

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"om": [1.0, 2.0, 3.0], "prediction": [1.1, 2.1, 2.9]}),
        validation_metrics={},
        test_metrics={},
        model=SimpleNamespace(),
    )
    return tags, summaries[-1]


def test_a_failed_model_log_is_tagged_on_the_run(monkeypatch) -> None:
    """Visible in the MLflow run list, not only inside an artifact nobody opens."""
    tags, summary = _run_lightning_child(monkeypatch, log_model_raises=True)

    assert tags["model_logged"] == "false"
    assert "Batch carries 0 lab column" in tags["model_logging_error"]
    assert summary["serialized_model_logged"] is False


def test_a_failed_model_log_does_not_kill_the_run(monkeypatch) -> None:
    """A fitted model should not be lost to a packaging problem."""
    _tags, summary = _run_lightning_child(monkeypatch, log_model_raises=True)

    assert summary["metrics"]["rmse_test"] > 0  # the run still recorded its results


def test_a_successful_model_log_is_tagged_too(monkeypatch) -> None:
    """The absence of a tag is not evidence, so the success case is tagged as well."""
    tags, summary = _run_lightning_child(monkeypatch, log_model_raises=False)

    assert tags["model_logged"] == "true"
    assert "model_logging_error" not in tags
    assert summary["serialized_model_logged"] is True


def test_a_failed_model_log_warns(monkeypatch, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        _run_lightning_child(monkeypatch, log_model_raises=True)

    assert any("NO servable model" in record.getMessage() for record in caplog.records)
    # The actual cause must be in the line, not just "something failed".
    assert any("Batch carries 0 lab column" in record.getMessage() for record in caplog.records)
