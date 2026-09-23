"""How explanations are gated and logged: the off-switch, the allow/deny lists, the budget skip,
and one explanation per joint model, sliced per target.

EXPLAIN_ENABLED has to be a real off-switch, not a plot suppressor.

shap 0.48 pulls in numba and is slow to import, and the suite runs with
``filterwarnings = ["error"]``, so a run that asked for no explanations must not import it at all.
The `"shap" not in sys.modules` assertion here is what fails if anyone hoists the import to module
scope in mlflow_loggers or explain/__init__.
"""

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as loggers_module
from tests.support.fakes import RecordingRuns
from yg_eo_soilnet.explain.sklearn_explainer import ExplainBudgetExceeded
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger


@pytest.fixture
def child_logger() -> ChildRunLogger:
    return ChildRunLogger()


def test_disabled_switch_returns_without_logging_anything(child_logger, monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", log_artifact)

    summary = child_logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        backend="lightning",
        payload={},
    )

    assert summary == {"enabled": False, "reason": "EXPLAIN_ENABLED is false"}
    log_artifact.assert_not_called()


def test_disabled_switch_does_not_import_shap(child_logger, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "shap", raising=False)

    child_logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        backend="lightning",
        payload={"model": object(), "bundle": object(), "target": "om"},
    )

    assert "shap" not in sys.modules, (
        "shap was imported on a run with EXPLAIN_ENABLED false; keep the import inside the "
        "explainer functions and the guard ahead of it"
    )


def test_explain_models_allowlist_skips_models_it_does_not_name(child_logger, monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", log_artifact)

    summary = child_logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_MODELS=["XGBoost"]),
        target="om",
        model_name="TabICL",
        backend="sklearn",
        payload={},
    )

    assert summary["enabled"] is False
    assert "TabICL" in summary["reason"]
    log_artifact.assert_not_called()


def test_a_named_model_is_not_skipped(child_logger) -> None:
    """The allowlist must not skip the model it names; the explainer then fails on the empty
    payload, which is recorded rather than raised."""
    summary = child_logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_MODELS=["XGBoost"]),
        target="om",
        model_name="XGBoost",
        backend="sklearn",
        payload={},
    )

    assert summary["enabled"] is True
    assert summary["logged"] is False


def test_explainer_failure_is_recorded_not_raised(child_logger) -> None:
    """Losing a finished training run because an explainer choked is a bad trade."""
    summary = child_logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True),
        target="om",
        model_name="soil_cnn",
        backend="not_a_backend",
        payload={},
    )

    assert summary["enabled"] is True
    assert summary["logged"] is False
    assert "ValueError" in summary["error"]


def test_explain_fail_on_error_re_raises(child_logger) -> None:
    with pytest.raises(ValueError, match="Unknown explain backend"):
        child_logger._log_shap_artifacts(
            config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_FAIL_ON_ERROR=True),
            target="om",
            model_name="soil_cnn",
            backend="not_a_backend",
            payload={},
        )


def test_switch_defaults_to_on_when_the_config_predates_the_key(child_logger) -> None:
    """A SimpleNamespace config with no EXPLAIN_* attributes at all must not crash."""
    summary = child_logger._log_shap_artifacts(
        config=SimpleNamespace(),
        target="om",
        model_name="soil_cnn",
        backend="not_a_backend",
        payload={},
    )

    assert summary["enabled"] is True


# The switch's config plumbing is covered in tests/test_config.py, beside the base_config_paths
# fixture that builds a complete config tree.


# --- the budget skip and the TabICL denylist --------------------------------------------------
# The explain gate: a budget skip is a decision, and the TabICL denylist.


def test_the_logger_reports_a_budget_skip_as_a_decision_not_a_crash() -> None:
    """A skipped explanation must not read like a failed run, and must not be escalated."""
    summary = ChildRunLogger()._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_FAIL_ON_ERROR=True),
        target="om",
        model_name="TabICL",
        backend="sklearn",
        payload={"fitted_estimator": _Exploder(), "X_train": None, "X_test": None, "target": "om"},
    )

    assert summary["skipped"] is True
    assert summary["logged"] is False
    assert "error" not in summary
    assert "EXPLAIN_MAX_EVALS" in summary["reason"]


class _Exploder:
    """Raises the budget error the moment the explainer touches it."""

    @property
    def named_steps(self):
        raise ExplainBudgetExceeded(
            "model-agnostic SHAP would need about 1,080,000 model evaluations, over the "
            "EXPLAIN_MAX_EVALS budget of 200,000."
        )


# --- the TabICL denylist ----------------------------------------------------


def _skip_summary(config, model_name: str) -> dict:
    """Run the guard alone; the payload is never reached when a model is skipped."""
    return ChildRunLogger()._log_shap_artifacts(
        config=config,
        target="om",
        model_name=model_name,
        backend="sklearn",
        payload={},
    )


def test_tabicl_is_skipped_by_name_with_a_reason() -> None:
    summary = _skip_summary(SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"]), "TabICL")

    assert summary["skipped"] is True
    assert summary["logged"] is False
    assert "EXPLAIN_SKIP_MODELS" in summary["reason"]
    # It reads as a decision, not a crash.
    assert "error" not in summary


@pytest.mark.parametrize("model_name", ["XGBoost", "GradientBoosting", "Ridge", "PLSRegression"])
def test_every_other_model_is_still_explained(model_name: str) -> None:
    """'Explainability for all models except TabICL' stated directly.

    These get past the guard and fail later on the empty payload, which is what distinguishes
    "was not skipped" from "was skipped".
    """
    summary = _skip_summary(
        SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"]), model_name
    )

    assert summary.get("skipped") is not True
    assert summary["enabled"] is True


def test_the_allowlist_overrides_the_denylist() -> None:
    """Naming a model explicitly is a deliberate request and must win.

    Otherwise the two settings contradict each other and EXPLAIN_MODELS silently does nothing -
    the user asks for the expensive explanation and gets neither a plot nor a reason.
    """
    summary = _skip_summary(
        SimpleNamespace(
            EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"], EXPLAIN_MODELS=["TabICL"]
        ),
        "TabICL",
    )

    assert summary.get("skipped") is not True
    assert summary["enabled"] is True


def test_the_off_switch_still_wins_over_everything() -> None:
    summary = _skip_summary(
        SimpleNamespace(EXPLAIN_ENABLED=False, EXPLAIN_MODELS=["TabICL"], EXPLAIN_SKIP_MODELS=[]),
        "TabICL",
    )

    assert summary["enabled"] is False


def test_a_config_without_the_key_skips_nothing() -> None:
    """A SimpleNamespace config predating EXPLAIN_SKIP_MODELS must not start skipping models."""
    summary = _skip_summary(SimpleNamespace(EXPLAIN_ENABLED=True), "TabICL")

    assert summary.get("skipped") is not True


def test_the_shipped_default_excludes_tabicl_and_nothing_else(tmp_path) -> None:
    """No per-run configuration should be needed for the behaviour the user asked for."""
    import yaml

    document = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "configs" / "main_config.yml").read_text()
    )

    assert document["common"]["EXPLAIN_SKIP_MODELS"] == ["TabICL"]
    assert document["common"]["EXPLAIN_MODELS"] == []


# --- explained once per joint model -----------------------------------------------------------
# A joint model is explained ONCE, and each target run gets only its own slice.
#
# An explanation is a property of the fitted MODEL, not of a target, but it used to be built inside
# the per-target fan-out - so a joint fit paid for the same answer once per target. On the Lightning
# side that was worse than wasteful: the explainer returns every output whatever `target` says, so
# each of the N child runs received all N results, nested them under explain/<target>/, and reported
# in its own run summary that it had explained the other targets too. Run
# 45c78e72b5c94aedaf6e771b7c0a893f shows it: three children of one soil_cnn model run, nine artifact
# sets, and three copies of explain/clay_pct/shap_values.parquet that disagree with each other by up
# to 2.8% of the mean |SHAP| because GradientExplainer is stochastic.
#
# These tests pin the two halves of the fix: built once, logged one slice per run.


def _quiet_logger(monkeypatch):
    """Silence everything that would touch a tracking server or the filesystem.

    Deliberately does NOT patch the explain seam - that is what these tests are about.
    """
    logger = ChildRunLogger()
    for name in ("set_tags", "log_params", "log_metric", "log_artifact", "log_metrics"):
        monkeypatch.setattr(loggers_module.mlflow, name, MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    for name in ("_log_plots", "_write_json_artifact", "_log_table_artifact", "_write_split_summary",
                 "_log_split_summary", "_promote_champion", "_log_cv_results", "_log_checkpoint",
                 "_tag_model_logging", "_log_pred_obs_artifact", "_evaluate_sklearn_target",
                 "_log_uncertainty_artifacts", "_log_train_fit_metric"):
        monkeypatch.setattr(logger, name, MagicMock())
    return logger


def _spy_on_explain(monkeypatch, target_names):
    """Replace the two explain entry points. Returns (build, log).

    Both are imported inside the method that calls them, so the import resolves the module attribute
    at call time and patching the module is enough - no shap install required.
    """
    build = MagicMock(
        return_value=[SimpleNamespace(target_name=name) for name in target_names]
    )
    log = MagicMock(return_value={"targets": [], "artifacts": []})
    monkeypatch.setattr("yg_eo_soilnet.explain.build_shap_results", build)
    monkeypatch.setattr("yg_eo_soilnet.explain.log_shap_artifacts", log)
    return build, log


TARGETS = ["clay_pct", "ph_water", "c_e_c_meq_100g"]


def _joint_eval_frame(targets=TARGETS):
    frame = {"target_names": ["__".join(targets)] * 4}
    for index, name in enumerate(targets):
        frame[name] = [1.0 + index, 2.0 + index, 3.0 + index, 4.0 + index]
        frame[f"prediction_{name}"] = [1.1 + index, 1.9 + index, 3.2 + index, 3.8 + index]
    return pd.DataFrame(frame)


# --- Lightning ---------------------------------------------------------------


def test_a_joint_lightning_fit_is_explained_once_and_each_child_logs_only_its_own_target(
    monkeypatch,
) -> None:
    logger = _quiet_logger(monkeypatch)
    monkeypatch.setattr(loggers_module.mlflow, "start_run", RecordingRuns())
    monkeypatch.setattr(logger, "_log_lightning_serialized_model", MagicMock(return_value=True))
    build, log = _spy_on_explain(monkeypatch, TARGETS)

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=True),
        target="__".join(TARGETS),
        model_name="soil_cnn",
        evaluation_df=_joint_eval_frame(),
        validation_metrics={"val_loss": 0.25},
        test_metrics={"test_loss": 0.5},
        model=SimpleNamespace(),
    )

    # ONE explainer pass for the model. This used to be three.
    assert build.call_count == 1
    # ...and the group label reaches the explainer, not a single target's name.
    assert build.call_args.kwargs["target"] == "__".join(TARGETS)

    # One slice written per target run, each holding exactly one result - so the artifacts land at
    # the flat explain/ path rather than nesting every target inside every child.
    assert log.call_count == 3
    assert [len(call.args[0]) for call in log.call_args_list] == [1, 1, 1]
    assert [call.args[0][0].target_name for call in log.call_args_list] == TARGETS


def test_a_single_target_lightning_fit_still_explains_once_into_its_own_run(monkeypatch) -> None:
    """The group of one has no fan-out, so nothing about it should change."""
    logger = _quiet_logger(monkeypatch)
    monkeypatch.setattr(loggers_module.mlflow, "start_run", RecordingRuns())
    monkeypatch.setattr(logger, "_log_lightning_serialized_model", MagicMock(return_value=True))
    build, log = _spy_on_explain(monkeypatch, ["clay_pct"])

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=True),
        target="clay_pct",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame(
            {"clay_pct": [1.0, 2.0, 3.0, 4.0], "prediction": [1.1, 1.9, 3.2, 3.8]}
        ),
        validation_metrics={"val_loss": 0.25},
        test_metrics={"test_loss": 0.5},
        model=SimpleNamespace(),
    )

    assert build.call_count == 1
    assert log.call_count == 1
    assert [result.target_name for result in log.call_args.args[0]] == ["clay_pct"]


# --- sklearn -----------------------------------------------------------------


class _JointEstimator:
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs

    def predict(self, X):
        base = np.arange(len(X), dtype=float)
        return np.column_stack([base + index for index in range(self.n_outputs)])


def test_a_joint_sklearn_fit_is_explained_once_and_each_child_logs_only_its_own_target(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    logger = _quiet_logger(monkeypatch)
    monkeypatch.setattr(loggers_module.mlflow, "start_run", RecordingRuns())
    monkeypatch.setattr(logger, "_log_metric_dict", MagicMock())
    monkeypatch.setattr(logger, "_register_sklearn_model", MagicMock(return_value=None))
    build, log = _spy_on_explain(monkeypatch, TARGETS)

    rows = 10
    X = pd.DataFrame({"a": np.arange(rows, dtype=float), "b": np.arange(rows, dtype=float)})
    y = pd.DataFrame({name: np.arange(rows, dtype=float) for name in TARGETS})

    logger.log_child_run(
        config=SimpleNamespace(
            ENABLE_CLUSTERING=False,
            CLUSTERING_STRATEGY={},
            MLFLOW_REGISTER_MODELS=False,
            EXPLAIN_ENABLED=True,
            LOG_TRAIN_FIT_METRIC=False,
        ),
        search=SimpleNamespace(best_params_={}, best_index_=0),
        cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
        best_model=_JointEstimator(len(TARGETS)),
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        target="__".join(TARGETS),
        targets=list(TARGETS),
        param_names=[],
        model_name="XGBoost",
        plot_functions={},
    )

    assert build.call_count == 1
    assert build.call_args.kwargs["target"] == "__".join(TARGETS)
    assert build.call_args.kwargs["target_names"] == TARGETS

    assert log.call_count == 3
    assert [len(call.args[0]) for call in log.call_args_list] == [1, 1, 1]
    assert [call.args[0][0].target_name for call in log.call_args_list] == TARGETS


# --- routing an output to its target -----------------------------------------


def test_the_slice_for_a_target_is_found_by_name() -> None:
    results = [SimpleNamespace(target_name=name) for name in TARGETS]
    picked = ChildRunLogger._shap_result_for(results, "ph_water", TARGETS)
    assert picked.target_name == "ph_water"


def test_the_slice_falls_back_to_position_when_the_explainer_used_synthetic_names() -> None:
    """A model carrying no target_names gets `<group>_0`, `<group>_1`, ... - a name miss by design.

    The position in the group is then the only thing tying an output to a target.
    """
    results = [SimpleNamespace(target_name=f"group_{index}") for index in range(len(TARGETS))]
    picked = ChildRunLogger._shap_result_for(results, "c_e_c_meq_100g", TARGETS)
    assert picked.target_name == "group_2"


def test_a_target_with_no_matching_output_is_reported_rather_than_mislabelled() -> None:
    """Better an honest gap than another run's attributions logged under this target's name."""
    logger = ChildRunLogger()
    results = [SimpleNamespace(target_name=name) for name in ("clay_pct", "ph_water")]

    summary = logger._log_shap_slice(
        results,
        {"enabled": True},
        target="sand_pct",
        target_names=TARGETS,
        config=SimpleNamespace(),
        is_model_run=False,
    )

    assert summary["logged"] is False
    assert "sand_pct" in summary["reason"]


# --- why there is nothing to explain, said once ------------------------------


def test_a_child_run_points_at_the_model_run_rather_than_repeating_the_reason() -> None:
    logger = ChildRunLogger()
    skipped = {"enabled": True, "logged": False, "skipped": True, "reason": "TabICL is expensive"}

    child = logger._log_shap_slice(
        None, skipped, target="ph_water", target_names=TARGETS,
        config=SimpleNamespace(), is_model_run=False,
    )
    assert child == {"enabled": True, "logged": False, "scope": "model_run"}

    # A group of one has no level to climb to, so it keeps the full reason inline as it always has.
    alone = logger._log_shap_slice(
        None, skipped, target="clay_pct", target_names=["clay_pct"],
        config=SimpleNamespace(), is_model_run=True,
    )
    assert alone == skipped


@pytest.mark.parametrize(
    "summary, expected",
    [
        ({"enabled": False, "reason": "EXPLAIN_ENABLED is false"}, "disabled"),
        ({"enabled": True, "logged": False, "skipped": True, "reason": "budget"}, "skipped"),
        ({"enabled": True, "logged": False, "error": "ValueError: boom"}, "error"),
    ],
)
def test_the_reason_a_joint_fit_has_no_explanation_is_recorded_once_on_the_model_run(
    monkeypatch, summary, expected
) -> None:
    logger = ChildRunLogger()
    write = MagicMock()
    tags = MagicMock()
    monkeypatch.setattr(logger, "_write_json_artifact", write)
    monkeypatch.setattr(loggers_module.mlflow, "set_tags", tags)

    logger._record_model_run_explain(summary, results=None, target_runs=[("a", {}), ("b", {})])

    write.assert_called_once()
    assert write.call_args.args[0] == summary
    assert write.call_args.args[1] == "explain_summary.json"
    assert tags.call_args.args[0] == {"explain_status": expected}


def test_a_single_target_run_records_no_separate_explain_summary(monkeypatch) -> None:
    """Its own run summary already carries the reason; a second file would only duplicate it."""
    logger = ChildRunLogger()
    write = MagicMock()
    monkeypatch.setattr(logger, "_write_json_artifact", write)
    monkeypatch.setattr(loggers_module.mlflow, "set_tags", MagicMock())

    logger._record_model_run_explain(
        {"enabled": False, "reason": "EXPLAIN_ENABLED is false"},
        results=None,
        target_runs=[("clay_pct", {})],
    )

    write.assert_not_called()
