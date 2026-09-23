"""The MLflow loggers: what a child run writes, what the parent collects back, the one metric
contract both families share, and the training log file."""

import gc
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as mlflow_loggers_module
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger, ParentRunLogger
from yg_eo_soilnet.logger.training_logger import TrainingLogger
from yg_eo_soilnet.metrics import METRIC_STEMS


def _checkpoint(tmp_path) -> Path:
    """A checkpoint file that actually exists.

    The logger copies the best checkpoint to a stable `best.ckpt` name rather than uploading the
    path it was handed, so these tests need a real file on disk.
    """
    path = tmp_path / "epoch=7-step=42.ckpt"
    path.write_bytes(b"fake-checkpoint")
    return path


def test_log_lightning_child_run_logs_metrics_artifacts_and_tags(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    fake_model = SimpleNamespace()

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "prediction": [1.3, 1.8, 3.4, 3.6, 5.5, 5.7],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        model=fake_model,
    )

    # Three calls: the run identity, the checkpoint's original filename, and the model-logging
    # outcome, which is tagged so a run that saved no model cannot look like one that did.
    assert set_tags.call_count == 3
    log_params.assert_any_call({
        "LIGHTNING_BATCH_SIZE": 8,
        "LIGHTNING_VAL_SIZE": 0.2,
        "LIGHTNING_MAX_EPOCHS": 10,
        "LIGHTNING_ACCELERATOR": "cpu",
        "LIGHTNING_DEVICES": 1,
        "LIGHTNING_PRECISION": "32-true",
    })
    log_metric.assert_any_call("val_loss", 0.5)
    log_metric.assert_any_call("test_loss", 0.4)
    # Logged under a STABLE leaf name, not Lightning's epoch=NN-step=MMM, so two runs of the
    # same model produce the same artifact path and MLflow can compare them.
    checkpoint_calls = [
        call for call in log_artifact.call_args_list
        if call.kwargs.get("artifact_path") == "checkpoints"
    ]
    assert len(checkpoint_calls) == 1
    assert Path(checkpoint_calls[0].args[0]).name == "best.ckpt"
    # The original name survives as a tag rather than in the path.
    assert any(
        c.args and c.args[0].get("checkpoint_filename") == "epoch=7-step=42.ckpt"
        for c in set_tags.call_args_list
    )
    assert any(call.kwargs.get("artifact_path") == "eval_results" for call in log_artifact.call_args_list)
    assert log_model.called


def test_log_lightning_child_run_logs_split_summary_metrics(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    bundle = SimpleNamespace(
        datamodule=SimpleNamespace(
            y_train_frame_=pd.DataFrame({"target_a": [1.0, 2.0, 3.0]}),
            y_val_frame_=pd.DataFrame({"target_a": [4.0, 5.0]}),
            y_test_frame_=pd.DataFrame({"target_a": [6.0, 7.0]}),
        ),
        trainer_kwargs={},
        registry_entry={"modeltype": "dl"},
    )

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [6.0, 7.0],
            "prediction": [5.5, 7.5],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        bundle=bundle,
        model=SimpleNamespace(),
    )

    assert any(call.args[0] == "train_mean" for call in log_metric.call_args_list)
    assert any(call.args[0] == "test_mean" for call in log_metric.call_args_list)
    assert any(call.args[0] == "eval_mae" for call in log_metric.call_args_list)
    assert any(call.kwargs.get("artifact_path") == "eval_results" for call in log_artifact.call_args_list)


def test_log_lightning_child_run_logs_architecture_dimensions(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import (
        SoilCNNLightningModule,
    )

    # No modalities: the CNN's static branch alone, so the head reads the 4-wide static encoding.
    model = SoilCNNLightningModule(
        static_dim=3,
        target_dim=1,
        target_names=["target_a"],
        static_hidden_dims=[4],
        head_hidden_dims=[5],
        learning_rate=0.0015,
    )

    bundle = SimpleNamespace(
        datamodule=SimpleNamespace(static_dim=3, target_dim=1, modality_dims={"s2": 2}),
        trainer_kwargs={"max_epochs": 100, "accelerator": "cuda", "devices": 1},
        registry_entry={"modeltype": "dl"},
    )

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "prediction": [1.3, 1.8, 3.4, 3.6, 5.5, 5.7],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        bundle=bundle,
        model=model,
    )

    architecture_calls = [call for call in log_params.call_args_list if any(key.startswith("architecture.") for key in call.args[0])]
    assert architecture_calls, "Expected architecture dimensions to be logged"
    assert architecture_calls[0].args[0] == {
        "architecture.static_dim": 3,
        "architecture.target_dim": 1,
        "architecture.static_hidden_dims": [4],
        "architecture.learning_rate": 0.0015,
        "architecture.temporal_enabled": False,
        "architecture.output_head_in_features": 4,
        "architecture.output_head_out_features": 1,
        "architecture.datamodule.static_dim": 3,
        "architecture.datamodule.target_dim": 1,
        "architecture.datamodule.modality_dim.s2": 2,
        "architecture.training.max_epochs": 100,
        "architecture.training.accelerator": "cuda",
        "architecture.training.devices": 1,
    }


def test_log_lightning_child_run_skips_pred_obs_for_multi_output(monkeypatch) -> None:
    logger = ChildRunLogger()

    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())

    logger.log_lightning_child_run(
        config=SimpleNamespace(),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame(
            {
                "target_a": [1.0, 2.0],
                "prediction_0": [0.9, 2.1],
                "prediction_1": [1.1, 1.9],
            }
        ),
        model=SimpleNamespace(),
    )

    eval_plot_calls = [
        call for call in log_artifact.call_args_list if call.kwargs.get("artifact_path") == "plots"
    ]
    assert eval_plot_calls == []


def _patch_child_runs(monkeypatch, children, download_artifacts):
    """Point ParentRunLogger at fake child runs, given as (run_id, tags) pairs."""
    runs = [
        SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))
        for run_id, tags in children
    ]

    class FakeMlflowClient:
        def get_run(self, parent_run_id):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="experiment-1"))

        def search_runs(self, experiment_ids, filter_string):
            return runs

    monkeypatch.setattr(mlflow_loggers_module.mlflow.tracking, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(mlflow_loggers_module.mlflow.artifacts, "download_artifacts", download_artifacts)


def _nothing_downloads(**kwargs):
    raise FileNotFoundError("missing")


LABELLED_CHILD = ("child-run-1", {"target": "target_a", "model_name": "toy_lightning"})


def test_collect_eval_dfs_uses_optional_fallback_path(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "eval_results_target_a_toy_lightning.csv"
    pd.DataFrame({"target_a": [1.0], "prediction": [1.2]}).to_csv(csv_path, index=False)

    def fake_download_artifacts(*, run_id, artifact_path):
        if artifact_path == "eval_results/eval_results_target_a_toy_lightning.csv":
            raise FileNotFoundError("missing nested path")
        if artifact_path == "eval_results_target_a_toy_lightning.csv":
            return str(csv_path)
        raise AssertionError(f"Unexpected artifact path: {artifact_path}")

    _patch_child_runs(monkeypatch, [LABELLED_CHILD], fake_download_artifacts)

    eval_dfs = ParentRunLogger()._collect_eval_dfs("parent-run-1")

    assert len(eval_dfs) == 1
    assert eval_dfs[0]["target_name"].iloc[0] == "target_a"
    assert eval_dfs[0]["model_name"].iloc[0] == "toy_lightning"


def test_collect_eval_dfs_warns_when_optional_eval_csv_missing(monkeypatch, capsys) -> None:
    _patch_child_runs(monkeypatch, [LABELLED_CHILD], _nothing_downloads)

    eval_dfs = ParentRunLogger()._collect_eval_dfs("parent-run-1")
    captured = capsys.readouterr()

    assert eval_dfs == []
    assert "Optional eval CSV missing" in captured.out


def test_collect_eval_dfs_ignores_unlabeled_runs(monkeypatch) -> None:
    requested = []

    def download(*, run_id, artifact_path):
        # Recorded rather than asserted here: the collector swallows download errors, so an
        # assertion raised in this function would never reach the test.
        requested.append(run_id)
        raise FileNotFoundError("missing")

    _patch_child_runs(monkeypatch, [("wrapper-run", {}), LABELLED_CHILD], download)

    assert ParentRunLogger()._collect_eval_dfs("parent-run-1") == []
    # Reaching for the wrapper's artifacts at all is the bug; the missing CSV alone would hide it.
    assert "wrapper-run" not in requested
    assert "child-run-1" in requested


def _fake_pred_obs_panel(eval_df, *, target_name=None):
    """A real but empty Figure, so log_figure can save and close it like the genuine one."""
    import matplotlib.pyplot as plt

    figure, _axis = plt.subplots()
    return figure


def test_log_pred_obs_artifact_separates_targets_by_directory(monkeypatch, tmp_path) -> None:
    """Two targets in one run must not overwrite each other - but they separate by DIRECTORY now.

    They used to separate by filename (`..._target_a_soil_cnn.png`), which kept them apart within
    a run at the cost of giving every run a different artifact path, so MLflow's compare view found
    nothing in common. Nesting under `plots/<target>/` keeps both properties.
    """
    logger = ChildRunLogger()

    logged: list[tuple[str, str]] = []

    def fake_log_artifact(path, artifact_path=None):
        logged.append((path, artifact_path))

    monkeypatch.setattr(mlflow_loggers_module, "pred_obs_panel", _fake_pred_obs_panel)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", fake_log_artifact)

    eval_a = pd.DataFrame({"target_a": [1.0, 2.0], "prediction": [1.1, 1.9]})
    eval_b = pd.DataFrame({"target_b": [3.0, 4.0], "prediction": [3.1, 3.9]})

    assert logger._log_pred_obs_artifact(
        eval_a, target="target_a", model_name="soil_cnn", artifact_path="plots/target_a"
    )
    assert logger._log_pred_obs_artifact(
        eval_b, target="target_b", model_name="soil_cnn", artifact_path="plots/target_b"
    )

    assert len(logged) == 2
    # Same stable leaf, different directory: distinct destinations, comparable across runs.
    assert [Path(path).name for path, _ in logged] == ["pred_obs.png", "pred_obs.png"]
    assert [artifact_path for _, artifact_path in logged] == ["plots/target_a", "plots/target_b"]


def test_single_target_pred_obs_lands_on_the_flat_comparable_path(monkeypatch) -> None:
    logger = ChildRunLogger()
    logged: list[tuple[str, str]] = []

    monkeypatch.setattr(mlflow_loggers_module, "pred_obs_panel", _fake_pred_obs_panel)
    monkeypatch.setattr(
        mlflow_loggers_module.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: logged.append((path, artifact_path)),
    )

    frame = pd.DataFrame({"target_a": [1.0, 2.0], "prediction": [1.1, 1.9]})
    assert logger._log_pred_obs_artifact(frame, target="target_a", model_name="soil_cnn")

    assert logged == [(logged[0][0], "plots")]
    assert Path(logged[0][0]).name == "pred_obs.png"


def test_log_lightning_child_run_logs_pred_obs_for_single_target_frame(monkeypatch) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())

    logger.log_lightning_child_run(
        config=SimpleNamespace(),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame(
            {
                "feature_1": [1.0, 2.0],
                "target_a": [1.0, 2.0],
                "prediction": [1.1, 1.9],
            }
        ),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        model=SimpleNamespace(),
    )

    assert any(call.kwargs.get("artifact_path") == "plots" for call in log_artifact.call_args_list)

# --- architecture introspection over real modules ---------------------------
# The fakes above use SimpleNamespace(in_features=...), which hides the case that actually
# matters: an nn.Sequential head, whose dims used to log as None.


def test_architecture_params_report_dims_for_a_sequential_head() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = SimpleNamespace(
        static_encoder=nn.Sequential(nn.Linear(11, 24), nn.LayerNorm(24), nn.ReLU()),
        output_head=nn.Sequential(nn.Linear(152, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
    )

    params = ChildRunLogger()._collect_lightning_architecture_params(model)

    assert params["architecture.static_encoder_in_features"] == 11
    assert params["architecture.static_encoder_out_features"] == 24
    # first Linear in, last Linear out - the whole block, not just one layer
    assert params["architecture.output_head_in_features"] == 152
    assert params["architecture.output_head_out_features"] == 1


def test_architecture_params_still_handle_a_plain_linear_head() -> None:
    import pytest

    torch = pytest.importorskip("torch")

    model = SimpleNamespace(output_head=torch.nn.Linear(152, 1))

    params = ChildRunLogger()._collect_lightning_architecture_params(model)

    assert params["architecture.output_head_in_features"] == 152
    assert params["architecture.output_head_out_features"] == 1


# --- one metric contract for both families ----------------------------------------------------
# The two training families must report the same numbers under the same names.
#
# This is the regression test for the bug that started all of it. `ChildRunLogger.log_child_run`
# logged a POSITIVE cross-validated RMSE as `mean_test_score`; `log_lightning_child_run` logged
# `-test_loss`, a NEGATIVE number, under that same name; and `log_parent_summary` plotted both on one
# axis. Three units, two signs, one column.
#
# The cross-family equality test below is the one that matters: give both loggers the same
# observations and predictions and every unified metric has to come out identical.


OBSERVED = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
PREDICTED = np.array([1.3, 1.8, 3.4, 3.7, 5.2, 6.1])


@pytest.fixture
def captured_metrics(monkeypatch) -> dict:
    """Collects every mlflow.log_metric call into a dict."""
    recorded: dict = {}

    def record(key, value, **kwargs):
        recorded[key] = value

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", record)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", MagicMock())
    return recorded


def _run_lightning(logger, captured_metrics) -> dict:
    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="organic_matter_pct",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"organic_matter_pct": OBSERVED, "prediction": PREDICTED}),
        validation_metrics={"val_loss": 0.51},
        test_metrics={"test_loss": 0.42, "test_r2": 0.66},
        model=SimpleNamespace(),
    )
    return dict(captured_metrics)


def test_lightning_no_longer_logs_the_negated_score(captured_metrics) -> None:
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert "mean_test_score" not in metrics
    assert "mean_train_score" not in metrics


def test_lightning_rmse_is_positive_and_matches_the_eval_frame(captured_metrics) -> None:
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert metrics["rmse_test"] > 0
    assert np.isclose(metrics["rmse_test"], np.sqrt(np.mean((PREDICTED - OBSERVED) ** 2)))


def test_the_training_space_metrics_keep_their_names_and_values(captured_metrics) -> None:
    """val_loss/test_loss/test_r2 are wired into EarlyStopping, ModelCheckpoint and the HPO
    objective whitelist. Renaming them would invalidate every exported tuned config."""
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert metrics["val_loss"] == 0.51
    assert metrics["test_loss"] == 0.42
    assert metrics["test_r2"] == 0.66


def test_test_loss_and_rmse_test_are_both_present_and_different(captured_metrics) -> None:
    """They measure different things in different spaces; keeping both is deliberate."""
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert "test_loss" in metrics and "rmse_test" in metrics
    assert metrics["test_loss"] != metrics["rmse_test"]


def test_both_families_report_the_same_numbers_for_the_same_predictions(monkeypatch) -> None:
    """The core regression test.

    A sklearn run and a Lightning run scored on identical observations and predictions must agree
    on every unified metric. Before yg_eo_soilnet.metrics they disagreed on the sign.
    """
    from yg_eo_soilnet.metrics import regression_metrics

    lightning_recorded: dict = {}
    monkeypatch.setattr(
        mlflow_loggers_module.mlflow, "log_metric", lambda key, value, **kw: lightning_recorded.update({key: value})
    )
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", MagicMock())

    ChildRunLogger().log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="organic_matter_pct",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"organic_matter_pct": OBSERVED, "prediction": PREDICTED}),
        validation_metrics={},
        test_metrics={},
        model=SimpleNamespace(),
    )

    # The sklearn branch computes the same set from the same helper on the same frame, so comparing
    # against the helper is comparing against what log_child_run logs.
    expected = regression_metrics(OBSERVED, PREDICTED)

    for stem in METRIC_STEMS:
        name = f"{stem}_test"
        if name not in expected:
            continue
        assert np.isclose(lightning_recorded[name], expected[name]), f"{name} disagrees"


# --- the parent leaderboard -------------------------------------------------


def test_leaderboard_backfills_legacy_runs_so_old_mlruns_still_plot() -> None:
    """A pre-unification sklearn row: positive RMSE under the retired name."""
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "XGBoost", "mean_test_score": 22.76, "r2_score": 0.41}
    )

    assert row["rmse_test"] == 22.76
    assert row["r2_test"] == 0.41
    assert row["legacy_metrics"] is True


def test_leaderboard_backfill_makes_a_negative_legacy_lightning_row_comparable() -> None:
    """A pre-unification Lightning row: -test_loss, hence negative. abs() puts it back on the
    same side of zero as the sklearn rows; it is a display fix, not a unit conversion, which is
    why the row is tagged."""
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "soil_cnn", "mean_test_score": -0.7284, "r2_score": 0.30}
    )

    assert row["rmse_test"] == pytest.approx(0.7284)
    assert row["legacy_metrics"] is True


def test_leaderboard_leaves_current_runs_untouched() -> None:
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "XGBoost", "rmse_test": 1.5, "r2_test": 0.8}
    )

    assert row["rmse_test"] == 1.5
    assert "legacy_metrics" not in row


def test_leaderboard_tolerates_a_row_with_neither_name() -> None:
    row = ParentRunLogger._backfill_legacy_metrics({"target": "om", "model": "XGBoost"})

    assert "rmse_test" not in row
    assert "legacy_metrics" not in row


def test_parent_summary_plots_the_unified_axes(monkeypatch) -> None:
    captured: dict = {}

    def fake_scatter(frame, metric_x, metric_y, **kwargs):
        captured["metric_x"] = metric_x
        captured["metric_y"] = metric_y
        import matplotlib.pyplot as plt

        return plt.figure()

    monkeypatch.setattr(mlflow_loggers_module, "plot_leaderboard_scatter", fake_scatter)
    monkeypatch.setattr(mlflow_loggers_module, "create_parent_pred_obs", lambda frames: None)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())

    parent = ParentRunLogger()
    monkeypatch.setattr(parent, "_collect_leaderboard", lambda run_id: pd.DataFrame())
    monkeypatch.setattr(parent, "_collect_eval_dfs", lambda run_id: [])

    config = SimpleNamespace(
        DATA_FILE="d.csv",
        ENABLE_CLUSTERING=False,
        CLUSTERING_STRATEGY={},
        SPLIT_STRATEGY="kfold",
        DATA_FOLDER="/data",
        RANDOM_SEED=42,
        TARGET_COLUMNS=["om"],
        COLUMNS_TO_TRANSFORM=[],
    )
    parent.log_parent_summary("run-1", SimpleNamespace(config=config))

    assert captured["metric_x"] == "rmse_test"
    assert captured["metric_y"] == "r2_test"


# --- the split summary, which BOTH families now write ---------------------------------------


@pytest.fixture
def captured_artifacts(monkeypatch, captured_metrics) -> dict:
    """Collects the JSON artifacts ChildRunLogger writes, keyed by filename."""
    written: dict = {}

    def record(self, payload, filename, artifact_path=None):
        written[filename] = payload

    monkeypatch.setattr(ChildRunLogger, "_write_json_artifact", record)
    return written


def test_the_split_summary_accepts_the_sklearn_single_target_series(captured_artifacts) -> None:
    """The regression test for `'Series' object has no attribute 'columns'`.

    ModelTrainer fits one target at a time and hands over `data['y_train'][target]` - a Series, not
    a frame. Assuming the Lightning shape here made every sklearn model fail at the end of an
    otherwise successful fit, reported as "Training failed" by the trainer's own except block.
    """
    ChildRunLogger()._write_split_summary(
        {
            "train": pd.Series(OBSERVED, name="caco3_pct_total"),
            "test": pd.Series(PREDICTED, name="caco3_pct_total"),
        },
        None,
        "caco3_pct_total",
    )

    summary = captured_artifacts["split_summary.json"]
    assert summary["train"]["target_column"] == "caco3_pct_total"
    assert summary["train"]["count"] == len(OBSERVED)
    assert summary["test"]["count"] == len(PREDICTED)


def test_the_split_summary_picks_this_run_target_out_of_a_multi_target_frame(
    captured_artifacts,
) -> None:
    """The Lightning shape. Taking column zero would describe some other target under this name."""
    ChildRunLogger()._write_split_summary(
        {"train": pd.DataFrame({"clay_pct": PREDICTED, "caco3_pct_total": OBSERVED})},
        None,
        "caco3_pct_total",
    )

    summary = captured_artifacts["split_summary.json"]
    assert summary["train"]["target_column"] == "caco3_pct_total"
    assert summary["train"]["mean"] == pytest.approx(float(OBSERVED.mean()))


def test_both_families_summarize_the_same_numbers_identically(captured_artifacts) -> None:
    """The invariant the artifact exists for: same data in, same summary out, either shape."""
    ChildRunLogger()._write_split_summary(
        {"train": pd.Series(OBSERVED, name="caco3_pct_total")}, None, "caco3_pct_total"
    )
    sklearn_summary = dict(captured_artifacts["split_summary.json"])

    ChildRunLogger()._write_split_summary(
        {"train": pd.DataFrame({"caco3_pct_total": OBSERVED})}, None, "caco3_pct_total"
    )
    lightning_summary = dict(captured_artifacts["split_summary.json"])

    assert sklearn_summary == lightning_summary


def test_an_unnamed_series_still_summarizes(captured_artifacts) -> None:
    """`to_frame()` needs a name; a nameless Series must not become a KeyError."""
    ChildRunLogger()._write_split_summary(
        {"train": pd.Series(OBSERVED)}, None, "caco3_pct_total"
    )

    assert captured_artifacts["split_summary.json"]["train"]["count"] == len(OBSERVED)


# --- the training log file --------------------------------------------------------------------
# TrainingLogger owns its temp directory.
#
# log_dir used to default to `os.path.join(tempfile.TemporaryDirectory().name, "logs")`, which was
# evaluated once at import (shared across every logger) and kept only `.name`, so the discarded
# handle's finalizer deleted the directory while file handlers still pointed at it.


def test_each_logger_gets_its_own_directory() -> None:
    first = TrainingLogger(name="logger-a", log_filename="a")
    second = TrainingLogger(name="logger-b", log_filename="b")

    assert first.log_dir != second.log_dir


def test_log_file_survives_garbage_collection() -> None:
    """The old default let a finalizer delete the directory out from under the handlers."""
    logger = TrainingLogger(name="logger-gc", log_filename="gc")
    logger.get_logger().info("written before collection")

    gc.collect()

    assert logger.log_file is not None
    assert os.path.exists(logger.log_file), "log file was removed by a finalizer"


def test_default_log_dir_has_no_finalizer_that_could_delete_it() -> None:
    """TemporaryDirectory would warn and delete on GC; a log directory must simply persist."""
    logger = TrainingLogger(name="logger-handle", log_filename="handle")

    assert logger._owns_log_dir is True
    assert "yg_eo_soilnet_logs_" in logger.log_dir


def test_explicit_log_dir_is_honoured_without_creating_a_temp_dir(tmp_path) -> None:
    logger = TrainingLogger(name="logger-explicit", log_dir=str(tmp_path), log_filename="explicit")

    assert logger._owns_log_dir is False
    assert logger.log_dir == str(tmp_path)
    assert os.path.exists(logger.log_file)


def test_file_logging_can_be_disabled() -> None:
    logger = TrainingLogger(name="logger-off", log_filename="off", enable_file_logging=False)

    assert logger.log_file is None
