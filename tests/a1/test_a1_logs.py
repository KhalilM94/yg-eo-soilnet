"""What the A1 scripts report while they run: console lines, log files, --quiet, and the log in MLflow.

The static demo runs once through run_a1.py (dev mode, random and leave-one-region-out plans, with
the covariate groups so the local model runs too), recorded in a file-based MLflow store.
"""

import json
import logging
import re

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from experiments.a1.make_plans import main as make_plans_main
from experiments.a1.run_a1 import main as run_a1_main
from yg_eo_soilnet.a1.logs import setup_logging
from yg_eo_soilnet.a1.runner import LOG_FILE, MANIFEST_FILE

ARGS = ["--kinds", "random,logo", "--groups", "--dev-mode"]


@pytest.fixture(scope="module")
def run(a1_settings_in, tmp_path_factory, mlflow_store):
    """One dev-mode run of the static demo through run_a1.py, with MLflow."""
    folder = tmp_path_factory.mktemp("a1_logs")
    store = folder / "mlruns"
    with mlflow_store(store) as uri:
        path = a1_settings_in(folder)(
            {"MLFLOW_TRACKING_URI": uri}, static=True, as_file=True, output_dir=str(folder / "out")
        )
        make_plans_main(["--config", str(path), "--kinds", "random,logo", "--quiet"])
        result = run_a1_main(["--config", str(path), *ARGS])
        log = (result.run_dir / LOG_FILE).read_text()
        yield {"path": path, "result": result, "log": log, "folder": folder, "store": store, "uri": uri}
        root = logging.getLogger()  # close the run's log files before the module's folder goes
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler) and str(folder) in handler.baseFilename:
                root.removeHandler(handler)
                handler.close()


@pytest.fixture
def in_store(run, mlflow_store):
    with mlflow_store(run["store"]):
        yield


def _index(run):
    return pd.read_csv(run["folder"] / "out" / "plans" / "index.csv")


# --- the log file ---------------------------------------------------------------------------------


def test_every_plan_has_a_start_and_an_end_line(run):
    index, log = _index(run), run["log"]
    total = len(index)
    for position, row in enumerate(index.itertuples(), 1):
        assert re.search(rf"\[{position}/{total}\] {row.plan_id} \({row.kind}\): [\d,]+ train / [\d,]+ test", log)
        assert re.search(rf"\[{position}/{total}\] {row.plan_id} (done in|already done)", log)


def test_the_log_reports_stage1_each_model_the_local_folds_and_the_time_left(run):
    log = run["log"]
    assert re.search(r"stage 1: [\d.]+ s; out-of-fold R2 caco3_pct_total -?[\d.]+, c_e_c_meq_100g -?[\d.]+", log)
    assert re.search(r"global_aux / combined: [\d.]+ s, test RMSE [\d.]+", log)
    assert re.search(r"local (North-East|South-West) fold 1/\d+: [\d,]+ train / [\d,]+ test", log)
    assert "left (" in log and "that was the last plan" in log
    assert re.search(r"^\S+ \S+ INFO ", log, flags=re.M)  # the timestamped format


def test_a_second_run_appends_and_says_every_plan_is_already_done(run, in_store):
    run_a1_main(["--config", str(run["path"]), *ARGS])
    log = (run["result"].run_dir / LOG_FILE).read_text()
    assert log.startswith(run["log"])  # the first run's lines are kept
    second = log[len(run["log"]) :]
    for row in _index(run).itertuples():
        assert re.search(rf"\] {row.plan_id} already done", second)
    assert "already done, test RMSE" in second


def test_quiet_keeps_the_summary_on_the_console_and_everything_in_the_file(run, in_store, capsys):
    before = (run["result"].run_dir / LOG_FILE).read_text()
    run_a1_main(["--config", str(run["path"]), *ARGS, "--quiet"])
    captured = capsys.readouterr()
    assert "file(s) written" in captured.out
    assert not re.search(r"\[\d+/\d+\]", captured.err + captured.out)
    assert " INFO " not in captured.err
    after = (run["result"].run_dir / LOG_FILE).read_text()
    assert re.search(r"\[1/\d+\] random", after[len(before) :])


# --- MLflow, failures and restarts ---------------------------------------------------------------


def _parent(run, run_dir):
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text())
    return MlflowClient(tracking_uri=run["uri"]).get_run(manifest["mlflow"]["run_id"])


def test_the_log_is_added_to_the_mlflow_parent_run(run):
    parent = _parent(run, run["result"].run_dir)
    names = [item.path for item in MlflowClient(tracking_uri=run["uri"]).list_artifacts(parent.info.run_id)]
    assert LOG_FILE in names and parent.info.status == "FINISHED"


def test_a_failed_run_still_uploads_its_log_and_is_marked_failed(run, in_store, monkeypatch):
    from yg_eo_soilnet.a1 import runner

    def broken(*args, **kwargs):
        raise RuntimeError("the model fell over")

    monkeypatch.setattr(runner, "_run_one_plan", broken)
    with pytest.raises(RuntimeError, match="fell over"):
        run_a1_main(["--config", str(run["path"]), *ARGS, "--run-name", "broken"])
    run_dir = run["result"].run_dir.parent / "broken"
    assert "failed after" in (run_dir / LOG_FILE).read_text()
    parent = _parent(run, run_dir)
    names = [item.path for item in MlflowClient(tracking_uri=run["uri"]).list_artifacts(parent.info.run_id)]
    assert parent.info.status == "FAILED" and LOG_FILE in names


def test_restart_keeps_the_earlier_log(run, in_store):
    run_a1_main(["--config", str(run["path"]), "--kinds", "random", "--dev-mode", "--run-name", "again", "--no-mlflow"])
    run_dir = run["result"].run_dir.parent / "again"
    first = (run_dir / LOG_FILE).read_text()
    manifest = json.loads((run_dir / MANIFEST_FILE).read_text())
    manifest["config_hash"] = "something else"  # as if the settings had changed
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest))
    run_a1_main(
        [
            "--config",
            str(run["path"]),
            "--kinds",
            "random",
            "--dev-mode",
            "--run-name",
            "again",
            "--no-mlflow",
            "--restart",
        ]
    )
    log = (run_dir / LOG_FILE).read_text()
    assert log.startswith(first) and "Clearing" in log[len(first) :]
    assert (run_dir / "predictions" / "random").is_dir()


# --- the other scripts ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script, extra, log_path",
    [
        ("preflight", [], "preflight/preflight.log"),
        ("make_plans", ["--kinds", "random,logo"], "plans/make_plans.log"),
        ("evaluate", ["--no-mlflow", "--n-boot", "5"], "runs/default/evaluation/evaluate.log"),
        ("make_figures", ["--no-mlflow", "--n-boot", "5", "--no-relief"], "runs/default/figures/make_figures.log"),
    ],
)
def test_each_script_writes_its_own_log_and_quiet_shows_no_progress(
    run, in_store, capsys, monkeypatch, script, extra, log_path
):
    import importlib

    from yg_eo_soilnet import map_style

    monkeypatch.setattr(map_style, "cached_download", lambda url, path: (_ for _ in ()).throw(OSError("offline")))
    main = importlib.import_module(f"experiments.a1.{script}").main
    main(["--config", str(run["path"]), *extra, "--quiet"])
    captured = capsys.readouterr()
    assert " INFO " not in captured.err
    log = (run["folder"] / "out" / log_path).read_text()
    assert "$ " in log and " INFO " in log


def test_logging_shows_even_after_mlflow_switched_it_off(tmp_path):
    import mlflow.pytorch._lightning_autolog  # noqa: F401 - calls logging.basicConfig(level=ERROR) on import

    logging.basicConfig(level=logging.ERROR)  # in case it was imported already
    path = setup_logging(tmp_path / "x.log")
    logging.getLogger("yg_eo_soilnet.a1.test").info("a progress line")
    assert "a progress line" in path.read_text()
