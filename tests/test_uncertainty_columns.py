"""The uncertainty column contract: how the columns are named, written, and read back.

The logger's collectors are tested here too, because the bugs they had came from the naming:
`prediction_*` also matches the sigma and interval columns.
"""

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.plot_utils import _resolve_prediction_column
from yg_eo_soilnet.uncertainty import attach_uncertainty_columns
from yg_eo_soilnet.uncertainty.columns import column_name, interval_columns, is_prediction_column, sigma_column
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.ensemble import aggregate

from tests.support.builders import eval_frame


def test_a_single_target_run_writes_unsuffixed_columns():
    assert column_name("prediction_std", "clay_pct", multi_target=False) == "prediction_std"


def test_a_multi_target_run_suffixes_every_column_with_its_target():
    assert column_name("prediction_std", "clay_pct", multi_target=True) == "prediction_std_clay_pct"


def test_an_uncertainty_column_is_not_mistaken_for_a_prediction_column():
    assert is_prediction_column("prediction") is True
    assert is_prediction_column("prediction_clay_pct") is True
    assert is_prediction_column("prediction_std") is False
    assert is_prediction_column("prediction_std_clay_pct") is False
    assert is_prediction_column("prediction_lower_clay_pct") is False
    assert is_prediction_column("prediction_epistemic_std_clay_pct") is False
    assert is_prediction_column("clay_pct") is False


def test_the_prediction_column_fallback_never_selects_a_standard_deviation():
    # The specific trap: the positional fallback in _resolve_prediction_column takes the first
    # column starting with "prediction_", and on a joint uncertainty frame that can be a sigma.
    # Plotting it would put standard deviations on the predicted axis and look entirely plausible.
    frame = pd.DataFrame(
        {
            "prediction_epistemic_std_clay_pct": [9.0],
            "prediction_std_clay_pct": [9.0],
            "prediction_clay_pct": [1.0],
        }
    )
    assert _resolve_prediction_column(frame, target_name=None, target_index=0) == "prediction_clay_pct"


def test_interval_columns_prefers_the_suffixed_pair_on_a_joint_frame():
    frame = pd.DataFrame(
        {
            "prediction_lower_clay_pct": [1.0],
            "prediction_upper_clay_pct": [2.0],
            "prediction_lower_sand_pct": [3.0],
            "prediction_upper_sand_pct": [4.0],
        }
    )
    lower, upper = interval_columns(frame, "sand_pct")
    assert lower.iloc[0] == 3.0 and upper.iloc[0] == 4.0


def test_interval_columns_returns_none_for_a_frame_from_a_run_without_uncertainty():
    assert interval_columns(eval_frame(with_uncertainty=False)) is None
    assert sigma_column(eval_frame(with_uncertainty=False)) is None


def test_a_half_written_interval_is_treated_as_absent_rather_than_half_used():
    frame = eval_frame()
    frame = frame.drop(columns=["prediction_upper"])
    assert interval_columns(frame) is None


def test_attach_writes_unsuffixed_columns_for_one_target():
    frame = pd.DataFrame({"target": np.zeros(5), "prediction": np.zeros(5)})
    prediction = aggregate([np.zeros(5), np.ones(5)])
    calibrator = ConformalCalibrator(q=2.0, alpha=0.05, n_calib=100)

    attach_uncertainty_columns(frame, prediction, ["clay_pct"], {"clay_pct": calibrator})

    assert "prediction_std" in frame.columns
    assert "prediction_std_clay_pct" not in frame.columns
    # Members 0 and 1: ensemble mean 0.5, epistemic std 0.5, no aleatoric part, so the interval is
    # 0.5 -+ q * 0.5 with q = 2.
    assert frame["prediction_epistemic_std"].iloc[0] == pytest.approx(0.5)
    assert frame["prediction_aleatoric_std"].iloc[0] == pytest.approx(0.0)
    assert frame["prediction_lower"].iloc[0] == pytest.approx(-0.5)
    assert frame["prediction_upper"].iloc[0] == pytest.approx(1.5)


def test_attach_suffixes_every_column_for_a_joint_group():
    frame = pd.DataFrame({"a": np.zeros(5), "b": np.zeros(5)})
    prediction = aggregate([np.zeros((5, 2)), np.ones((5, 2))])

    attach_uncertainty_columns(frame, prediction, ["a", "b"])

    for stem in ("prediction_std", "prediction_epistemic_std", "prediction_aleatoric_std"):
        assert f"{stem}_a" in frame.columns
        assert f"{stem}_b" in frame.columns
        assert stem not in frame.columns


def test_attach_writes_no_interval_when_there_is_no_calibrator():
    frame = pd.DataFrame({"target": np.zeros(5)})
    attach_uncertainty_columns(frame, aggregate([np.zeros(5), np.ones(5)]), ["clay_pct"])
    assert "prediction_std" in frame.columns
    assert "prediction_lower" not in frame.columns


def test_a_single_target_frame_with_uncertainty_still_reads_as_single_target():
    """The uncertainty columns must not make a one-target frame look like a joint one.

    _iter_target_eval_frames decided that by scanning for `prediction_*`, which the sigma and
    interval columns now also match. A single-target uncertainty frame then found no
    prediction_<target>, yielded nothing, and the run logged neither rmse_test nor picp_test - with
    no error anywhere, because an empty metric dict is indistinguishable from a metric-free run.
    """
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = eval_frame()
    frame["clay_pct"] = frame["target"]

    frames = list(
        ChildRunLogger()._iter_target_eval_frames(frame, target="clay_pct", model_name="Ridge")
    )
    assert len(frames) == 1
    _yielded, target_name, prediction_column = frames[0]
    assert target_name == "clay_pct"
    assert prediction_column == "prediction"


def test_the_parent_collector_never_plots_a_standard_deviation_as_a_prediction(tmp_path):
    """`_collect_eval_dfs` had the same `prediction_*` bug as `_iter_target_eval_frames`.

    A single-target uncertainty frame took the multi-target branch, and the fallback there picked
    the first `prediction_*` column - which is `prediction_std`, since it sorts right after
    `prediction`. That value was then assigned to `prediction`, so the parent's pred_error_plot.png
    plotted standard deviations on the predicted axis and looked entirely plausible.
    """
    import mlflow

    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    frame = eval_frame(40)
    frame["clay_pct"] = frame["target"]
    # Sigma is deliberately far from the prediction, so picking the wrong column is unmissable.
    frame["prediction_std"] = 999.0
    frame["target_names"] = "clay_pct"

    with mlflow.start_run() as parent:
        parent_id = parent.info.run_id
        with mlflow.start_run(nested=True) as child:
            mlflow.set_tags({"target": "clay_pct", "model_name": "Ridge"})
            path = tmp_path / "eval_results.csv"
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="eval_results")
            child.info.run_id

    collected = ParentRunLogger()._collect_eval_dfs(parent_id)
    assert len(collected) == 1
    assert not (collected[0]["prediction"] == 999.0).any()
    assert np.allclose(collected[0]["prediction"], frame["prediction"])


def test_a_joint_frame_with_uncertainty_still_fans_out_per_target():
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = pd.DataFrame(
        {
            "clay_pct": [1.0, 2.0],
            "sand_pct": [3.0, 4.0],
            "prediction_clay_pct": [1.1, 2.1],
            "prediction_sand_pct": [3.1, 4.1],
            "prediction_std_clay_pct": [0.5, 0.5],
            "prediction_std_sand_pct": [0.5, 0.5],
            "target_names": ["clay_pct__sand_pct"] * 2,
        }
    )
    frames = list(
        ChildRunLogger()._iter_target_eval_frames(
            frame, target="clay_pct__sand_pct", model_name="Ridge"
        )
    )
    assert [name for _f, name, _c in frames] == ["clay_pct", "sand_pct"]
    assert [col for _f, _n, col in frames] == ["prediction_clay_pct", "prediction_sand_pct"]
