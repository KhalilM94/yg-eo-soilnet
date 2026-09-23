"""Uncertainty through both trainers.

The sklearn ensemble runs end to end - real estimators, the real trainer, the real logger - in a few
shared scenarios, which also carry the per-point prediction export. The Lightning ensemble runs
against a fake Trainer.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.trainers.lightning_trainer as lightning_trainer_module
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningModelBundle
from yg_eo_soilnet.trainers import ModelTrainer
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer
from yg_eo_soilnet.uncertainty import uncertainty_enabled_for
from yg_eo_soilnet.uncertainty.predictors import EnsembleRegressor


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        CATEGORICAL_FEATURES=[],
        CLUSTERING_STRATEGY={"enabled": False, "params": {}},
        ENABLE_CLUSTERING=False,
        MIN_FEATURE_COUNT=2,
        LOG_TRAIN_FIT_METRIC=False,
        MLFLOW_REGISTER_MODELS=False,
        EXPLAIN_ENABLED=False,
        FAIL_ON_MODEL_ERROR=True,
        UNCERTAINTY_ENABLED=True,
        UNCERTAINTY_N_MEMBERS=4,
        UNCERTAINTY_SEED_STRIDE=1000,
        UNCERTAINTY_BOOTSTRAP="auto",
        UNCERTAINTY_ALPHA=0.05,
        UNCERTAINTY_CALIBRATION_METHOD="split_conformal",
        UNCERTAINTY_CALIBRATION_SOURCE="val",
        UNCERTAINTY_MODELS=[],
        UNCERTAINTY_SKIP_MODELS=[],
        POINT_ID_COLUMN="uuid",
        EXPORT_POINT_PREDICTIONS=False,
        EXPORT_POINT_PREDICTIONS_MODELS=[],
        EXPORT_POINT_PREDICTIONS_SKIP_MODELS=[],
        EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _split(n_rows: int = 200, targets=("target_a",), seed: int = 0) -> dict:
    """A split dict shaped like SklearnDataSplitter's, including the audit-only val keys."""
    rng = np.random.default_rng(seed)
    features = pd.DataFrame({"f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows)})
    signal = 3.0 * features["f1"] - 2.0 * features["f2"]
    y = pd.DataFrame({name: signal + rng.normal(scale=1.0, size=n_rows) for name in targets})

    train = slice(0, 120)
    val = slice(120, 160)
    test = slice(160, n_rows)
    return {
        # X_train is the FIT POOL: train u val, matching the real splitter's contract.
        "X_train": features.iloc[0:160].reset_index(drop=True),
        "y_train": y.iloc[0:160].reset_index(drop=True),
        "X_train_only": features.iloc[train].reset_index(drop=True),
        "y_train_only": y.iloc[train].reset_index(drop=True),
        "X_val": features.iloc[val].reset_index(drop=True),
        "y_val": y.iloc[val].reset_index(drop=True),
        "X_test": features.iloc[test].reset_index(drop=True),
        "y_test": y.iloc[test].reset_index(drop=True),
        # The whole featurized population plus its ids, the way SklearnDataSplitter returns them:
        # `point_ids` is a Series indexed like X, which is what lets the export reindex rather than
        # pair by position.
        "X_all": features,
        "point_ids": pd.Series([f"p{index}" for index in range(n_rows)], index=features.index, name="point_id"),
    }


def _trainer(config) -> ModelTrainer:
    import logging

    logger = logging.getLogger("uncertainty-sklearn-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return ModelTrainer(config=config, seed=42, n_splits=3, logger=logger)


def _pipelines(name="Ridge"):
    from sklearn.linear_model import Ridge

    return {name: {"model": Ridge(), "params": {"model__alpha": [1.0]}, "modeltype": "ml"}}


def _run_and_collect(config, data, pipelines, target="target_a", targets=None):
    """Train inside a parent run and return (child runs by name, the MlflowClient)."""
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name="parent") as parent:
        _trainer(config).train(target=target, data=data, model_pipelines=pipelines, targets=targets)
        parent_id = parent.info.run_id

    experiment_id = client.get_run(parent_id).info.experiment_id

    def descendants(run_id):
        children = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        )
        collected = list(children)
        for child in children:
            collected.extend(descendants(child.info.run_id))
        return collected

    return parent_id, descendants(parent_id), client


def _export_config(**overrides):
    return _config(EXPORT_POINT_PREDICTIONS=True, **overrides)


def _download(client, run, path, tmp_path):
    return pd.read_csv(client.download_artifacts(run.info.run_id, path, str(tmp_path)))


def _export_parent(parent_id, config):
    """Run the parent-level combine step, which main.py does inside the parent run.

    Reopening the run matters: log_table writes to whichever run is ACTIVE, so calling this outside
    one would file the parent's summary against nothing.
    """
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    with mlflow.start_run(run_id=parent_id):
        ParentRunLogger()._log_point_prediction_export(parent_id, config)


@dataclass
class Scenario:
    """One training run inside a parent, shared read-only by every test that inspects it."""

    parent_id: str
    runs: list
    client: mlflow.tracking.MlflowClient
    uri: str

    def model_run(self, target: str = "target_a"):
        """The model's own run, not one of its ensemble members."""
        return next(
            run
            for run in self.runs
            if run.data.tags.get("target") == target
            and run.data.tags.get("model_name") == "Ridge"
            and run.data.tags.get("run_kind") != "ensemble_member"
        )


def _scenario(mlflow_store, tmp_path_factory, config, data, **train_kwargs) -> Scenario:
    """Train once in a store of its own, then run the parent-level export step as main.py does."""
    with mlflow_store(tmp_path_factory.mktemp("mlruns")) as uri:
        parent_id, runs, client = _run_and_collect(config, data, _pipelines(), **train_kwargs)
        _export_parent(parent_id, config)
    return Scenario(parent_id, runs, client, uri)


@pytest.fixture(scope="module")
def trained(mlflow_store, tmp_path_factory) -> Scenario:
    """The default: four members, split-conformal on the val rows, no point export."""
    return _scenario(mlflow_store, tmp_path_factory, _config(), _split(400))


@pytest.fixture(scope="module")
def exported(mlflow_store, tmp_path_factory) -> Scenario:
    return _scenario(mlflow_store, tmp_path_factory, _export_config(), _split(200))


@pytest.fixture(scope="module")
def joint_exported(mlflow_store, tmp_path_factory) -> Scenario:
    return _scenario(
        mlflow_store,
        tmp_path_factory,
        _export_config(),
        _split(targets=("target_a", "target_b")),
        target="target_a__target_b",
        targets=["target_a", "target_b"],
    )


# --- the switch ------------------------------------------------------------


def test_the_master_switch_turns_the_whole_thing_off():
    assert uncertainty_enabled_for(_config(UNCERTAINTY_ENABLED=False), "Ridge") is False


def test_a_skipped_model_gets_no_ensemble():
    config = _config(UNCERTAINTY_SKIP_MODELS=["TabICL"])
    assert uncertainty_enabled_for(config, "TabICL") is False
    assert uncertainty_enabled_for(config, "Ridge") is True


def test_naming_a_model_explicitly_beats_the_skip_list():
    config = _config(UNCERTAINTY_MODELS=["TabICL"], UNCERTAINTY_SKIP_MODELS=["TabICL"])
    assert uncertainty_enabled_for(config, "TabICL") is True
    # An allowlist is exclusive: everything not named is off.
    assert uncertainty_enabled_for(config, "Ridge") is False


# --- the fit pool ----------------------------------------------------------


def test_val_calibration_moves_the_fit_pool_off_the_val_rows():
    data = _split()
    fit_pool, calibration = _trainer(_config())._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 120
    assert calibration is not None
    assert len(calibration["X"]) == 40


def test_uncertainty_disabled_keeps_the_full_train_plus_val_fit_pool():
    data = _split()
    fit_pool, calibration = _trainer(_config(UNCERTAINTY_ENABLED=False))._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


def test_cv_oof_calibration_keeps_every_row_in_the_fit_pool():
    data = _split()
    fit_pool, calibration = _trainer(_config(UNCERTAINTY_CALIBRATION_SOURCE="cv_oof"))._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


@pytest.mark.parametrize("method", ["gaussian", "sigma", "none"])
def test_a_method_needing_no_calibration_keeps_the_whole_fit_pool(method):
    """Reserving val for a band that is pure arithmetic would cost ~15% of the rows for nothing."""
    fit_pool, calibration = _trainer(_config(UNCERTAINTY_INTERVAL_METHOD=method))._resolve_fit_pool(_split())
    assert len(fit_pool["X"]) == 160
    assert calibration is None


def test_a_split_without_val_rows_falls_back_and_says_so():
    data = _split()
    data["X_val"] = data["X_val"].iloc[0:0]
    trainer = _trainer(_config())
    fit_pool, calibration = trainer._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


# --- training --------------------------------------------------------------


@pytest.mark.slow
def test_every_member_gets_its_own_tagged_child_run(trained):
    members = [run for run in trained.runs if run.data.tags.get("run_kind") == "ensemble_member"]
    assert len(members) == 4
    assert {run.data.tags["ensemble_member"] for run in members} == {"0", "1", "2", "3"}
    # Seeds are strided, not consecutive.
    assert {int(run.data.tags["ensemble_seed"]) for run in members} == {42, 1042, 2042, 3042}


@pytest.mark.slow
def test_the_eval_frame_carries_the_estimate_and_its_uncertainty(trained, tmp_path):
    frame = _download(trained.client, trained.model_run(), "eval_results/eval_results.csv", tmp_path)

    for column in (
        "prediction",
        "prediction_std",
        "prediction_epistemic_std",
        "prediction_aleatoric_std",
        "prediction_lower",
        "prediction_upper",
    ):
        assert column in frame.columns

    # A sklearn ensemble has no aleatoric component: a point predictor cannot report one.
    assert (frame["prediction_aleatoric_std"] == 0.0).all()
    assert (frame["prediction_std"] > 0.0).all()
    assert (frame["prediction_lower"] < frame["prediction"]).all()
    assert (frame["prediction_upper"] > frame["prediction"]).all()


@pytest.mark.slow
def test_a_deterministic_estimator_still_produces_a_non_degenerate_ensemble(trained):
    # The bug the bootstrap exists to prevent: Ridge ignores its seed, so without resampling every
    # member is identical, sigma is exactly 0, and the run claims perfect confidence.
    assert trained.model_run().data.metrics["mean_sigma_test"] > 0.0


@pytest.mark.slow
def test_the_interval_covers_close_to_the_nominal_level(trained):
    # Conformal on 40 calibration rows is noisy, hence the wide band; the point is that the
    # interval is somewhere near its claim rather than covering 20% or 100%.
    assert trained.model_run().data.metrics["picp_test"] == pytest.approx(0.95, abs=0.12)


@pytest.mark.slow
def test_the_uncertainty_metrics_are_logged_beside_the_point_metrics(trained):
    model_run = trained.model_run()
    for metric in ("rmse_test", "r2_test", "picp_test", "mpiw_test", "interval_score_test"):
        assert metric in model_run.data.metrics


@pytest.mark.slow
def test_the_fit_pool_is_recorded_so_a_smaller_rmse_is_not_read_as_a_regression(trained):
    model_run = trained.model_run()
    assert model_run.data.params["uncertainty_fit_pool"] == "train_only"
    assert model_run.data.params["uncertainty_n_train_rows"] == "120"
    assert model_run.data.params["uncertainty_n_members"] == "4"
    assert "conformal_q" in model_run.data.params


@pytest.mark.slow
def test_uncertainty_disabled_writes_no_extra_columns(tmp_path):
    _parent, runs, client = _run_and_collect(_config(UNCERTAINTY_ENABLED=False), _split(), _pipelines())
    model_run = next(run for run in runs if run.data.tags.get("model_name") == "Ridge")
    local = client.download_artifacts(model_run.info.run_id, "eval_results/eval_results.csv", str(tmp_path))
    frame = pd.read_csv(local)

    assert "prediction" in frame.columns
    assert "prediction_std" not in frame.columns
    assert not any(run.data.tags.get("run_kind") == "ensemble_member" for run in runs)


@pytest.mark.slow
def test_a_joint_group_suffixes_the_uncertainty_columns_per_target(joint_exported, tmp_path):
    frame = _download(
        joint_exported.client,
        joint_exported.model_run("target_a__target_b"),
        "eval_results/eval_results.csv",
        tmp_path,
    )

    for target_name in ("target_a", "target_b"):
        assert f"prediction_{target_name}" in frame.columns
        assert f"prediction_std_{target_name}" in frame.columns
        assert f"prediction_lower_{target_name}" in frame.columns
    assert "prediction_std" not in frame.columns


# --- the leaderboard -------------------------------------------------------


@pytest.mark.slow
def test_members_do_not_replace_their_model_on_the_leaderboard(trained):
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    mlflow.set_tracking_uri(trained.uri)
    leaderboard = ParentRunLogger()._collect_leaderboard(trained.parent_id)

    # Exactly one row: the model. Without the run_kind filter this is four member rows and the
    # model itself vanishes, because the collector prefers a run's grandchildren to the run.
    assert len(leaderboard) == 1
    assert leaderboard.iloc[0]["model"] == "Ridge"
    assert leaderboard.iloc[0]["target"] == "target_a"
    assert "rmse_test" in leaderboard.columns


@pytest.mark.slow
def test_the_parent_eval_frames_exclude_the_members(trained):
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    mlflow.set_tracking_uri(trained.uri)
    frames = ParentRunLogger()._collect_eval_dfs(trained.parent_id)
    assert len(frames) == 1


# --- the fitted ensemble ---------------------------------------------------


def test_predict_returns_the_mean_in_the_shape_a_single_member_returns():
    from sklearn.linear_model import Ridge

    X = pd.DataFrame({"f": np.arange(20.0)})
    y = pd.Series(np.arange(20.0) * 2.0)
    members = [Ridge().fit(X, y) for _ in range(3)]

    ensemble = EnsembleRegressor(members, ["target_a"])
    predictions = ensemble.predict(X)

    # 1-D, exactly like one member: the MLflow signature, the evaluator and champion promotion all
    # depend on this staying true.
    assert predictions.ndim == 1
    assert predictions.shape == (20,)
    assert np.allclose(predictions, members[0].predict(X))


def test_predict_frame_returns_the_wide_output_with_named_columns():
    from sklearn.linear_model import Ridge

    from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator

    X = pd.DataFrame({"f": np.arange(20.0)})
    y = pd.Series(np.arange(20.0) * 2.0)
    ensemble = EnsembleRegressor(
        [Ridge().fit(X, y) for _ in range(3)],
        ["target_a"],
        calibrators={"target_a": ConformalCalibrator(q=1.5, alpha=0.05, n_calib=100)},
    )

    frame = ensemble.predict_frame(X)
    assert list(frame.columns) == [
        "target_a_pred",
        "target_a_std",
        "target_a_epistemic_std",
        "target_a_aleatoric_std",
        "target_a_lower",
        "target_a_upper",
    ]


def test_an_ensemble_needs_at_least_one_member():
    with pytest.raises(ValueError, match="at least one fitted member"):
        EnsembleRegressor([], ["target_a"])


# --- the per-point prediction export ---------------------------------------


@pytest.mark.slow
def test_the_export_covers_every_point_not_just_the_test_split(exported, tmp_path):
    child = _download(exported.client, exported.model_run(), "predictions/point_predictions.csv", tmp_path)
    # 200 points in the population; the test split is only 40 of them.
    assert len(child) == 200
    assert list(child.columns) == ["uuid", "target_a"]
    assert child["uuid"].iloc[0] == "p0"


@pytest.mark.slow
def test_the_exported_test_rows_match_the_eval_frame_value_for_value(exported, tmp_path):
    """The check that the ids are RIGHT rather than merely present.

    The export predicts the whole population in one pass; eval_results.csv predicts the test split
    in another. Where they overlap they must agree exactly, or the two passes are not describing
    the same points.
    """
    model_run = exported.model_run()
    child = _download(exported.client, model_run, "predictions/point_predictions.csv", tmp_path / "a")
    evaluation = _download(exported.client, model_run, "eval_results/eval_results.csv", tmp_path / "b")

    # Test rows are the last 40 points, ids p160..p199, in order.
    exported_rows = child.set_index("uuid").loc[[f"p{i}" for i in range(160, 200)], "target_a"]
    assert np.allclose(exported_rows.to_numpy(), evaluation["prediction"].to_numpy())


@pytest.mark.slow
def test_the_parent_writes_both_the_wide_and_the_long_file(exported, tmp_path):
    parent_run = exported.client.get_run(exported.parent_id)

    wide = _download(exported.client, parent_run, "predictions/point_predictions_wide.csv", tmp_path / "w")
    long_frame = _download(exported.client, parent_run, "predictions/point_predictions_long.csv", tmp_path / "l")

    assert list(wide.columns) == ["uuid", "target_a__Ridge"]
    assert wide["uuid"].is_unique
    assert list(long_frame.columns) == ["uuid", "target", "model", "prediction"]
    assert set(long_frame["model"]) == {"Ridge"}


@pytest.mark.slow
def test_a_joint_group_gets_one_wide_column_per_target(joint_exported, tmp_path):
    wide = _download(
        joint_exported.client,
        joint_exported.client.get_run(joint_exported.parent_id),
        "predictions/point_predictions_wide.csv",
        tmp_path,
    )
    assert set(wide.columns) == {"uuid", "target_a__Ridge", "target_b__Ridge"}


@pytest.mark.slow
def test_the_ensemble_exports_its_mean_and_no_uncertainty_columns(exported, tmp_path):
    wide = _download(
        exported.client,
        exported.client.get_run(exported.parent_id),
        "predictions/point_predictions_wide.csv",
        tmp_path,
    )
    assert list(wide.columns) == ["uuid", "target_a__Ridge"]
    assert not [c for c in wide.columns if c.endswith(("_std", "_lower", "_upper"))]
    # One row per point, not one per ensemble member.
    assert len(wide) == 200


@pytest.mark.slow
def test_the_export_is_absent_when_the_switch_is_off(trained):
    client = trained.client

    assert [a.path for a in client.list_artifacts(trained.model_run().info.run_id, "predictions")] == []
    assert [a.path for a in client.list_artifacts(trained.parent_id, "predictions")] == []


@pytest.mark.slow
def test_a_skipped_model_exports_nothing():
    config = _export_config(EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["Ridge"])
    parent_id, runs, client = _run_and_collect(config, _split(), _pipelines())
    model_run = next(run for run in runs if run.data.tags.get("model_name") == "Ridge")
    assert [a.path for a in client.list_artifacts(model_run.info.run_id, "predictions")] == []


# --- the Lightning ensemble -------------------------------------------------------------------
# The Lightning ensemble path: member runs, aggregation, calibration on val, and the off-switch.


N_TEST = 12
N_VAL = 40


@dataclass
class FakeCheckpointCallback:
    best_model_path: str = "/tmp/best.ckpt"


class FakeDataModule:
    """The frames and loaders the ensemble path reads, and nothing else."""

    def __init__(self, target_names=("target_a",), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.target_names = list(target_names)
        self.X_test_frame_ = pd.DataFrame({"f": rng.normal(size=N_TEST)})
        self.y_test_frame_ = pd.DataFrame({name: rng.normal(size=N_TEST) for name in self.target_names})
        self.X_val_frame_ = pd.DataFrame({"f": rng.normal(size=N_VAL)})
        self.y_val_frame_ = pd.DataFrame({name: rng.normal(size=N_VAL) for name in self.target_names})

    def setup(self, stage=None):
        return None

    def val_dataloader(self):
        return "val-loader"


class FakeTrainer:
    """Returns a different constant per member, so the ensemble spread is predictable."""

    def __init__(self, offset: float, n_targets: int = 1):
        self.offset = offset
        self.n_targets = n_targets
        self.checkpoint_callback = FakeCheckpointCallback()
        self.callbacks = [self.checkpoint_callback]
        self.predicted_splits = []

    def fit(self, model, datamodule=None):
        return None

    def validate(self, model, datamodule=None, verbose=False, ckpt_path=None):
        return [{"val_loss": 0.4 + self.offset}]

    def test(self, model, datamodule=None, verbose=False, ckpt_path=None):
        return [{"test_loss": 0.3 + self.offset}]

    def predict(self, model, datamodule=None, dataloaders=None, ckpt_path=None):
        if dataloaders is not None:
            self.predicted_splits.append("val")
            rows = N_VAL
        else:
            self.predicted_splits.append("test")
            rows = N_TEST
        return [np.full((rows, self.n_targets), self.offset, dtype=np.float32)]


def _lightning_config(**overrides) -> SimpleNamespace:
    base = dict(
        RANDOM_SEED=42,
        UNCERTAINTY_ENABLED=True,
        UNCERTAINTY_N_MEMBERS=3,
        UNCERTAINTY_SEED_STRIDE=1000,
        UNCERTAINTY_ALPHA=0.05,
        UNCERTAINTY_INTERVAL_METHOD="conformal",
        UNCERTAINTY_INTERVAL_K=1.0,
        # Config keeps both names in step; the stub does too, so a test that sets one does not
        # silently exercise the other's default.
        UNCERTAINTY_CALIBRATION_METHOD="split_conformal",
        UNCERTAINTY_CALIBRATION_SOURCE="val",
        UNCERTAINTY_MODELS=[],
        UNCERTAINTY_SKIP_MODELS=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _bundle(datamodule) -> LightningModelBundle:
    return LightningModelBundle(
        name="soil_cnn",
        target="target_a",
        model=SimpleNamespace(),
        datamodule=datamodule,
        trainer_kwargs={"max_epochs": 1},
        callback_specs={},
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )


def _run(monkeypatch, config, target_names=("target_a",)):
    """Train an ensemble with fake Lightning trainers; return (logged kwargs, run names)."""
    datamodule = FakeDataModule(target_names)
    bundle = _bundle(datamodule)

    built = []

    def bundle_builder(seed):
        built.append(seed)
        return {"soil_cnn": _bundle(datamodule)}

    fake_trainers = [FakeTrainer(offset=float(index + 1), n_targets=len(target_names)) for index in range(10)]
    handed_out = []

    # _log_checkpoint is part of the real ChildRunLogger's interface and _fit_member calls it for
    # every member, so the double carries it too rather than being guarded against in production.
    fake_mlflow_logger = SimpleNamespace(log_lightning_child_run=MagicMock(), _log_checkpoint=MagicMock())
    trainer = LightningTrainer(config=config, logger=MagicMock(), mlflow_logger=fake_mlflow_logger)

    def build_trainer(_bundle):
        fake = fake_trainers[len(handed_out)]
        handed_out.append(fake)
        return fake

    monkeypatch.setattr(trainer, "_build_trainer", build_trainer)

    run_names = []

    def fake_start_child_run(run_name, tags=None):
        run_names.append((run_name, dict(tags or {})))
        from contextlib import nullcontext

        return nullcontext()

    monkeypatch.setattr(lightning_trainer_module, "start_child_run", fake_start_child_run)
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_params", MagicMock())

    trainer.train(
        target="__".join(target_names),
        data={},
        model_bundles={"soil_cnn": bundle},
        bundle_builder=bundle_builder,
    )

    return (fake_mlflow_logger.log_lightning_child_run.call_args, run_names, built, handed_out, fake_mlflow_logger)


# --- the run tree ----------------------------------------------------------


def test_each_member_gets_its_own_child_run_tagged_as_a_member(monkeypatch):
    _call, run_names, _built, _trainers, _lg = _run(monkeypatch, _lightning_config())

    members = [(name, tags) for name, tags in run_names if tags.get("run_kind") == "ensemble_member"]
    assert len(members) == 3
    assert [name for name, _ in members] == [
        "target_a_soil_cnn_member0",
        "target_a_soil_cnn_member1",
        "target_a_soil_cnn_member2",
    ]
    # And one model run wrapping them.
    assert ("target_a_soil_cnn", {}) in run_names


def test_the_factory_is_asked_for_a_fresh_bundle_at_each_strided_seed(monkeypatch):
    _call, _run_names, built, _trainers, _lg = _run(monkeypatch, _lightning_config())
    # Rebuilt per member, because weights are constructed by the factory at seeding time.
    assert built == [42, 1042, 2042]


def test_uncertainty_disabled_takes_the_single_fit_path(monkeypatch):
    _call, run_names, built, _trainers, _lg = _run(monkeypatch, _lightning_config(UNCERTAINTY_ENABLED=False))
    assert built == []
    assert not any(tags.get("run_kind") == "ensemble_member" for _name, tags in run_names)


def test_a_skipped_entry_takes_the_single_fit_path(monkeypatch):
    _call, run_names, built, _trainers, _lg = _run(monkeypatch, _lightning_config(UNCERTAINTY_SKIP_MODELS=["soil_cnn"]))
    assert built == []
    assert not any(tags.get("run_kind") == "ensemble_member" for _name, tags in run_names)


# --- aggregation -----------------------------------------------------------


def test_the_logged_frame_carries_the_ensemble_mean_not_one_members_predictions(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _lightning_config())
    frame = call.kwargs["evaluation_df"]

    # Members predict the constants 1, 2 and 3, so the ensemble mean is 2 and the population
    # standard deviation is sqrt(2/3).
    assert np.allclose(frame["prediction"], 2.0)
    assert np.allclose(frame["prediction_epistemic_std"], np.sqrt(2.0 / 3.0))


def test_a_point_predicting_ensemble_reports_no_aleatoric_component(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _lightning_config())
    frame = call.kwargs["evaluation_df"]
    assert np.allclose(frame["prediction_aleatoric_std"], 0.0)
    assert np.allclose(frame["prediction_std"], frame["prediction_epistemic_std"])


def test_member_metrics_are_averaged_so_they_describe_the_ensemble(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _lightning_config())
    # Offsets 1, 2, 3 on a base of 0.3 -> mean 2.3.
    assert call.kwargs["test_metrics"]["test_loss"] == pytest.approx(2.3)
    assert call.kwargs["validation_metrics"]["val_loss"] == pytest.approx(2.4)


# --- calibration -----------------------------------------------------------


def test_calibration_uses_the_val_split_and_reaches_the_logger(monkeypatch):
    call, _run_names, _built, trainers, _lg = _run(monkeypatch, _lightning_config())

    calibrators = call.kwargs["calibrators"]
    assert set(calibrators) == {"target_a"}
    assert calibrators["target_a"].n_calib == N_VAL

    # Both splits are predicted for every member: test for the eval frame, val to calibrate.
    for fake in trainers[:3]:
        assert fake.predicted_splits.count("val") == 1
        assert fake.predicted_splits.count("test") >= 1


def test_the_interval_columns_are_written_from_the_calibrator(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _lightning_config())
    frame = call.kwargs["evaluation_df"]

    assert "prediction_lower" in frame.columns
    assert "prediction_upper" in frame.columns
    assert (frame["prediction_lower"] < frame["prediction"]).all()
    assert (frame["prediction_upper"] > frame["prediction"]).all()


def test_calibration_off_still_reports_sigma_but_writes_no_interval(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(
        monkeypatch,
        _lightning_config(UNCERTAINTY_INTERVAL_METHOD="none", UNCERTAINTY_CALIBRATION_METHOD="none"),
    )
    frame = call.kwargs["evaluation_df"]

    assert call.kwargs["calibrators"] == {}
    assert "prediction_std" in frame.columns
    assert "prediction_lower" not in frame.columns


def test_a_joint_group_suffixes_every_uncertainty_column(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(
        monkeypatch, _lightning_config(), target_names=("target_a", "target_b")
    )
    frame = call.kwargs["evaluation_df"]

    for target_name in ("target_a", "target_b"):
        assert f"prediction_{target_name}" in frame.columns
        assert f"prediction_std_{target_name}" in frame.columns
        assert f"prediction_lower_{target_name}" in frame.columns
    assert "prediction_std" not in frame.columns


def test_a_missing_validation_split_skips_calibration_rather_than_pairing_wrong_rows(monkeypatch):
    """A calibrator fitted on misaligned rows would be silently, confidently wrong."""
    datamodule = FakeDataModule()
    datamodule.y_val_frame_ = pd.DataFrame({"target_a": []})

    trainer = LightningTrainer(config=_lightning_config(), logger=MagicMock(), mlflow_logger=SimpleNamespace())
    members = [
        {"calibration_predictions": np.zeros((N_VAL, 1))},
    ]
    assert trainer._calibrate_ensemble(members, datamodule, ["target_a"]) == {}
    assert trainer.logger.warning.called


def test_every_member_logs_its_own_checkpoint(monkeypatch):
    """Without this the run keeps one checkpoint for the whole ensemble - the reference member's -
    and the other n-1 exist only as local files that nothing records, which is what made an
    ensemble unrecoverable from MLflow alone."""
    _call, _run_names, _built, _trainers, logger = _run(monkeypatch, _lightning_config())

    assert logger._log_checkpoint.call_count == 3  # one per member
    assert all(call.args[0] == "/tmp/best.ckpt" for call in logger._log_checkpoint.call_args_list)
