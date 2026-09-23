from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import main as main_module


def test_parse_args_supports_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "custom.yml"])

    args = main_module.parse_args()

    assert args.config_path == "custom.yml"
    assert args.dev_mode is False


def test_parse_args_reads_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--dev-mode"])

    assert main_module.parse_args().dev_mode is True


def _dev_mode_config():
    """A stand-in Config holding the parts apply_dev_mode changes."""
    return SimpleNamespace(
        LIGHTNING_MODEL_REGISTRY={
            "soil_cnn": {
                "enabled": True,
                "trainer_args": {"max_epochs": 500, "precision": "32-true"},
                "callbacks": {"early_stopping": {"patience": 100}, "checkpoint": {"save_top_k": 1}},
            }
        },
        MODEL_REGISTRY={"Ridge": {"params": {"model__alpha": [0.1, 1, 10, 100], "model__fit_intercept": True}}},
        EXPLAIN_ENABLED=True,
        UNCERTAINTY_ENABLED=True,
        MLFLOW_REGISTER_MODELS=True,
        MLFLOW_EXPERIMENT_EXPORT_ENABLED=True,
    )


def test_dev_mode_trains_a_single_pass() -> None:
    config = _dev_mode_config()

    main_module.apply_dev_mode(config, MagicMock())

    trainer_args = config.LIGHTNING_MODEL_REGISTRY["soil_cnn"]["trainer_args"]
    assert trainer_args["max_epochs"] == 1
    assert trainer_args["limit_train_batches"] == 2
    assert trainer_args["accelerator"] == "cpu"
    # Settings it says nothing about are left alone.
    assert trainer_args["precision"] == "32-true"


def test_dev_mode_keeps_saving_the_model_but_drops_stopping_early() -> None:
    """Writing the model out is one of the steps most worth checking, so it stays on."""
    config = _dev_mode_config()

    main_module.apply_dev_mode(config, MagicMock())

    callbacks = config.LIGHTNING_MODEL_REGISTRY["soil_cnn"]["callbacks"]
    assert "early_stopping" not in callbacks
    assert callbacks["checkpoint"] == {"save_top_k": 1}


def test_dev_mode_tries_one_setting_per_classic_model() -> None:
    config = _dev_mode_config()

    main_module.apply_dev_mode(config, MagicMock())

    params = config.MODEL_REGISTRY["Ridge"]["params"]
    assert params["model__alpha"] == [0.1]
    # A setting that is not a list of values to try is left as it is.
    assert params["model__fit_intercept"] is True


def test_dev_mode_switches_off_the_slow_extras() -> None:
    config = _dev_mode_config()

    main_module.apply_dev_mode(config, MagicMock())

    assert config.EXPLAIN_ENABLED is False
    assert config.UNCERTAINTY_ENABLED is False
    # A model from a one-pass run must never be offered as something to serve.
    assert config.MLFLOW_REGISTER_MODELS is False
    assert config.MLFLOW_EXPERIMENT_EXPORT_ENABLED is False


def test_dev_mode_says_plainly_that_the_run_means_nothing() -> None:
    logger = MagicMock()

    main_module.apply_dev_mode(_dev_mode_config(), logger)

    assert logger.warning.call_count == 1
    assert "mean nothing" in logger.warning.call_args.args[0]


def test_dev_mode_copes_with_a_configuration_that_has_no_models() -> None:
    config = SimpleNamespace(LIGHTNING_MODEL_REGISTRY=None, MODEL_REGISTRY={})

    main_module.apply_dev_mode(config, MagicMock())

    assert config.EXPLAIN_ENABLED is False


def _fake_plan():
    """A stand-in SplitPlan: main() only reads describe() and counts() off it."""
    return SimpleNamespace(
        describe=MagicMock(return_value={"split_strategy": "random"}),
        counts=MagicMock(return_value={"train": 2, "val": 1, "test": 1}),
    )


def _stub_trainer(**overrides) -> SimpleNamespace:
    """A stand-in SoilModelTraining: main() loads, preprocesses, splits and trains through it."""
    trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(return_value="raw"),
            preprocess=MagicMock(return_value="processed"),
            split=MagicMock(return_value={"X_train": "x"}),
        ),
        # main() decides the split once, before either family sees the data, and logs its
        # provenance - so a stub trainer has to offer the provider.
        split_plan_provider=SimpleNamespace(plan=MagicMock(return_value=_fake_plan())),
        logger=SimpleNamespace(info=MagicMock(), error=MagicMock()),
        logger_wrapper=SimpleNamespace(log_file=None),
        train_models=MagicMock(),
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


class _FakeRun:
    def __init__(self, info):
        self.info = info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_main(
    monkeypatch,
    trainer,
    parent_logger=None,
    *,
    run_info=None,
    argv=("main.py", "--config-path", "config.yml"),
) -> dict:
    """Stub everything main() reaches outside itself; return the MLflow mocks by name."""
    mocks = {
        name: MagicMock()
        for name in (
            "enable_system_metrics_logging",
            "set_experiment",
            "create_experiment",
            "end_run",
            "log_param",
            # log_params too, not just log_param: an unstubbed one auto-starts a REAL run against
            # the default tracking store and never ends it, which surfaces three test files later
            # as "Run with UUID ... is already active".
            "log_params",
            "log_artifact",
            "set_tag",
        )
    }
    mocks["active_run"] = MagicMock(return_value=None)
    mocks["start_run"] = MagicMock(return_value=_FakeRun(run_info or SimpleNamespace(run_id="run-123")))
    for name, mock in mocks.items():
        monkeypatch.setattr(main_module.mlflow, name, mock)
    monkeypatch.setattr(
        main_module.datetime,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: "20260703_120000")),
    )
    monkeypatch.setattr(main_module, "SoilModelTraining", MagicMock(return_value=trainer))
    monkeypatch.setattr(
        main_module,
        "ParentRunLogger",
        MagicMock(return_value=parent_logger or SimpleNamespace(log_parent_summary=MagicMock())),
    )
    monkeypatch.setattr("sys.argv", list(argv))
    return mocks


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _stub_trainer()
    parent_logger = SimpleNamespace(log_parent_summary=MagicMock())
    _stub_main(monkeypatch, trainer, parent_logger)

    main_module.main()

    trainer.scikit_datamodule.load_frame.assert_called_once()
    trainer.scikit_datamodule.preprocess.assert_called_once_with("raw")
    # The shared plan is handed to the sklearn family rather than each family splitting for itself.
    trainer.scikit_datamodule.split.assert_called_once_with("processed", trainer.split_plan_provider.plan.return_value)
    trainer.train_models.assert_called_once()
    parent_logger.log_parent_summary.assert_called_once_with("run-123", trainer)


def test_main_survives_a_failing_parent_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary that cannot be built must not discard the training that already succeeded.

    Every model is trained and logged by its own child run before this point; the summary only
    reads them back. Letting it re-raise skipped the log artifact and the run-folder export and
    left the parent marked FAILED - which is how one truncated meta.yaml elsewhere in the store
    turned a finished run into a crash.
    """
    trainer = _stub_trainer(logger_wrapper=SimpleNamespace(log_file="train.log"))
    parent_logger = SimpleNamespace(log_parent_summary=MagicMock(side_effect=RuntimeError("leaderboard unreadable")))
    mocks = _stub_main(monkeypatch, trainer, parent_logger)

    main_module.main()

    trainer.train_models.assert_called_once()
    # The run still finishes: the log artifact goes up, and nothing propagates out of main().
    mocks["log_artifact"].assert_called_once_with("train.log")
    # Not a silent swallow - the run records why its summary is missing.
    assert mocks["set_tag"].call_args.args[0] == "parent_summary_error"
    assert "leaderboard unreadable" in mocks["set_tag"].call_args.args[1]
    trainer.logger.error.assert_called_once()


def test_main_logs_and_reraises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _stub_trainer(
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(side_effect=RuntimeError("load failed")),
            preprocess=MagicMock(),
            split=MagicMock(),
        )
    )
    _stub_main(monkeypatch, trainer, argv=("main.py",))

    with pytest.raises(RuntimeError, match="load failed"):
        main_module.main()

    trainer.logger.error.assert_called_once()


def test_main_skips_artifact_upload_when_logger_file_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _stub_main(monkeypatch, _stub_trainer())

    main_module.main()

    mocks["log_artifact"].assert_not_called()


def test_main_exports_mlflow_experiment_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    trainer = _stub_trainer(
        config=SimpleNamespace(
            config_path="config.yml",
            registry_path="registry.yml",
            MLFLOW_EXPERIMENT_EXPORT_ENABLED=True,
            MLFLOW_EXPERIMENT_EXPORT_PATH=str(tmp_path / "exports"),
        )
    )
    experiment_id = "12345"
    run_id = "67890"
    run_name = "Run_20260703_120000"
    tracking_root = tmp_path / "mlruns"
    run_dir = tracking_root / experiment_id / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.yaml").write_text(f"name: {run_name}\n")
    (run_dir / "dummy.txt").write_text("content\n")

    _stub_main(monkeypatch, trainer, run_info=SimpleNamespace(run_id=run_id, experiment_id=experiment_id))
    monkeypatch.setattr(main_module.mlflow, "get_tracking_uri", MagicMock(return_value=f"file://{tracking_root}"))
    monkeypatch.setattr(
        main_module.mlflow,
        "get_experiment",
        MagicMock(return_value=SimpleNamespace(name="Soil Model Training Experiment")),
    )
    copied = {}

    def fake_copytree(src, dst):
        copied["src"] = src
        copied["dst"] = dst
        return dst

    monkeypatch.setattr(main_module.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(main_module.shutil, "rmtree", MagicMock())

    main_module.main()

    assert copied["src"] == run_dir
    assert copied["dst"] == tmp_path / "exports" / "Soil_Model_Training_Experiment" / run_name


def test_train_models_dispatches_sklearn_and_lightning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())
    fake_lightning_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a"],
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"sklearn_model": {"enabled": True}},
        LIGHTNING_MODEL_REGISTRY={"lightning_model": {"enabled": True}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}}
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
    )
    trainer.lightning_trainer = fake_lightning_trainer

    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    data = {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
        "y_train": pd.DataFrame({"target_a": [3.0, 4.0]}),
        "X_test": pd.DataFrame({"feature": [5.0]}),
        "y_test": pd.DataFrame({"target_a": [6.0]}),
    }

    trainer.train_models(data)

    fake_sklearn_trainer.train.assert_called_once()
    fake_lightning_trainer.train.assert_called_once()


def _multi_target_trainer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    sklearn_spec: dict | None = None,
    lightning_spec: dict | None = None,
):
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())
    fake_lightning_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE=mode,
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"sklearn_model": {"enabled": True, **(sklearn_spec or {})}},
        LIGHTNING_MODEL_REGISTRY={"lightning_model": {"enabled": True, **(lightning_spec or {})}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}}
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
    )
    trainer.lightning_trainer = fake_lightning_trainer

    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    data = {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
        "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
        "X_test": pd.DataFrame({"feature": [5.0]}),
        "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
    }
    trainer.train_models(data)
    return trainer, fake_sklearn_trainer, fake_lightning_trainer


def test_joint_mode_fits_one_lightning_model_over_every_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """MULTI_TARGET_MODE: joint gives Lightning one run with a target_dim-wide head."""
    _, _, lightning_trainer = _multi_target_trainer(monkeypatch, mode="joint")

    lightning_trainer.train.assert_called_once()
    assert lightning_trainer.train.call_args.kwargs["target"] == "target_a__target_b"


def test_per_target_mode_fits_one_lightning_model_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """The switch the family never had: Lightning was joint whatever the config said."""
    _, _, lightning_trainer = _multi_target_trainer(monkeypatch, mode="per_target")

    assert lightning_trainer.train.call_count == 2
    assert [call.kwargs["target"] for call in lightning_trainer.train.call_args_list] == ["target_a", "target_b"]


def test_sklearn_joins_targets_only_when_the_estimator_declares_it_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most scikit estimators cannot fit a 2-D y, so an undeclared entry stays per-target."""
    _, undeclared, _ = _multi_target_trainer(monkeypatch, mode="joint")
    assert undeclared.train.call_count == 2
    assert [call.kwargs["target"] for call in undeclared.train.call_args_list] == ["target_a", "target_b"]

    _, native, _ = _multi_target_trainer(monkeypatch, mode="joint", sklearn_spec={"multi_target": "native"})
    native.train.assert_called_once()
    assert native.train.call_args.kwargs["target"] == "target_a__target_b"
    assert native.train.call_args.kwargs["targets"] == ["target_a", "target_b"]


def test_a_registry_entry_overrides_the_global_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, lightning_trainer = _multi_target_trainer(
        monkeypatch, mode="joint", lightning_spec={"multi_target": "per_target"}
    )

    assert lightning_trainer.train.call_count == 2
    assert [call.kwargs["target"] for call in lightning_trainer.train.call_args_list] == ["target_a", "target_b"]


def test_sklearn_entries_that_agree_on_a_grouping_are_trained_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET is a check ACROSS the models tried for a target.

    Splitting the entries into one call each would turn "every model failed" into "any model
    failed", so entries resolving to the same groups share a call.
    """
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={
            "Ridge": {"enabled": True, "multi_target": "native"},
            "PLSRegression": {"enabled": True, "multi_target": "native"},
            # Cannot take a 2-D y, so it falls back to one model per target on its own.
            "GradientBoosting": {"enabled": True},
        },
        LIGHTNING_MODEL_REGISTRY={},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={
                name: {"model": object(), "params": {}, "modeltype": "ml"}
                for name in ("Ridge", "PLSRegression", "GradientBoosting")
            }
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    calls = {
        call.kwargs["target"]: sorted(call.kwargs["model_pipelines"])
        for call in fake_sklearn_trainer.train.call_args_list
    }
    assert calls == {
        # The two native estimators share the joint group, in ONE call.
        "target_a__target_b": ["PLSRegression", "Ridge"],
        # The one that cannot fit a 2-D y gets its own per-target calls.
        "target_a": ["GradientBoosting"],
        "target_b": ["GradientBoosting"],
    }


@pytest.mark.parametrize(
    "mode,expected_sklearn",
    [
        ("joint", "target_a__target_b"),
        ("per_target", "target_a | target_b"),
    ],
)
def test_train_models_records_what_it_decided_to_fit(monkeypatch: pytest.MonkeyPatch, mode, expected_sklearn) -> None:
    """The run logs how it split the data but used to say nothing about what it planned to fit.

    That gap is why an OOM-killed multi-target run was mistaken for a bug in the grouping: the
    mode and the resolved groups were only recoverable by reading child run names afterwards.
    """
    logged: dict = {}
    monkeypatch.setattr(main_module.mlflow, "log_params", lambda params: logged.update(params))

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE=mode,
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"Ridge": {"enabled": True, "multi_target": "native"}},
        LIGHTNING_MODEL_REGISTRY={"soil_cnn": {"enabled": True}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"Ridge": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=SimpleNamespace(train=MagicMock())))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    assert logged["MULTI_TARGET_MODE"] == mode
    assert logged["TARGET_COLUMNS"] == "target_a,target_b"
    assert logged["sklearn_target_groups"] == expected_sklearn
    assert logged["lightning_target_groups"] == expected_sklearn


def test_the_target_plan_shows_a_fallback_as_a_mode_group_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """joint mode beside per-target groups IS the fallback, and is otherwise invisible afterwards."""
    logged: dict = {}
    monkeypatch.setattr(main_module.mlflow, "log_params", lambda params: logged.update(params))

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        # No multi_target declaration, so it cannot take a 2-D y.
        MODEL_REGISTRY={"TabICL": {"enabled": True}},
        LIGHTNING_MODEL_REGISTRY={},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"TabICL": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=SimpleNamespace(train=MagicMock())))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    assert logged["MULTI_TARGET_MODE"] == "joint"
    assert logged["sklearn_target_groups"] == "target_a | target_b"
