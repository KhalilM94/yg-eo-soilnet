"""Regenerating a finished run's figures from the artifacts it already holds.

The unit under test is the SELECTION logic - which run gets which figure, and which are skipped -
rather than the drawing, which tests/test_plots.py covers. That is where the bugs are:
a run's eval CSV holds the whole joint frame, so the obvious reading of it replots six targets into
a run that owns one.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.replotting import (
    _cv_plot_name,
    _interval_estimator,
    _own_target_frame,
    _planned,
    _wanted,
    regenerate_child_figures,
)


def _run(run_id="abc123", params=None, **tags):
    """A stand-in for an mlflow Run, carrying only what the replot reads off it."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, run_name=tags.get("target", "run")),
        data=SimpleNamespace(tags=tags, params=dict(params or {})),
    )


def _joint_eval_frame():
    """What a per-target child of a JOINT fit actually holds.

    Every target's observations and predictions, a `target_names` column naming all of them, and a
    bare `prediction` column aliasing this run's own target. That last column is the trap: the frame
    looks single-target to a column check and multi-target to a name check.
    """
    return pd.DataFrame(
        {
            "clay_pct": [10.0, 20.0, 30.0],
            "sand_pct": [70.0, 60.0, 50.0],
            "prediction_clay_pct": [11.0, 19.0, 31.0],
            "prediction_sand_pct": [69.0, 61.0, 49.0],
            "prediction": [69.0, 61.0, 49.0],
            "target_names": ["clay_pct__sand_pct"] * 3,
        }
    )


# --- which frame a run owns -------------------------------------------------


def test_a_child_replots_only_the_target_its_own_tag_names():
    """The bug this exists for: replotting every target in the CSV into one run.

    A run scopes exactly one target - _log_per_target_runs opens a child per target as soon as a
    group has more than one - so fanning out here would write six pictures into a run that owns
    one, five of them belonging to sibling runs.
    """
    frame = _own_target_frame(ChildRunLogger(), _joint_eval_frame(), "sand_pct", "soil_cnn")
    assert frame is not None
    assert list(frame["prediction"]) == [69.0, 61.0, 49.0]

    other = _own_target_frame(ChildRunLogger(), _joint_eval_frame(), "clay_pct", "soil_cnn")
    assert list(other["prediction"]) == [11.0, 19.0, 31.0]


def test_a_legacy_frame_with_no_target_name_is_still_this_runs_frame():
    # Older runs wrote target and prediction with no target_names column at all. There is exactly
    # one frame in that case and it belongs to the run holding it.
    legacy = pd.DataFrame({"clay_pct": [1.0, 2.0], "prediction": [1.1, 2.1]})
    assert _own_target_frame(ChildRunLogger(), legacy, "clay_pct", "Ridge") is not None


def test_a_frame_that_does_not_hold_this_target_yields_nothing():
    frame = pd.DataFrame(
        {
            "clay_pct": [1.0],
            "sand_pct": [2.0],
            "prediction_clay_pct": [1.1],
            "prediction_sand_pct": [2.1],
            "target_names": ["clay_pct__sand_pct"],
        }
    )
    assert _own_target_frame(ChildRunLogger(), frame, "ph_water", "Ridge") is None


# --- which runs are skipped -------------------------------------------------


def test_a_run_with_no_eval_csv_is_skipped_rather_than_raised_on(monkeypatch):
    """220 of 813 runs in the live experiment have none - members, and runs that failed early.

    One of those must not stop the rest of a tree from being refreshed.
    """
    monkeypatch.setattr("yg_eo_soilnet.replotting._read_first", lambda *a, **k: None)
    outcome = regenerate_child_figures(_run(target="clay_pct", model_name="Ridge"))
    assert outcome["skipped"] == "no eval_results.csv"
    assert outcome["written"] == []


def test_a_run_missing_its_tags_is_skipped():
    assert "no target/model_name tag" in regenerate_child_figures(_run())["skipped"]


def test_a_joint_model_run_defers_to_its_per_target_children():
    """It holds the fitted model and the joint frame, but draws no pred-vs-obs of its own."""
    outcome = regenerate_child_figures(_run(target="clay_pct__sand_pct", model_name="soil_cnn"))
    assert "per-target children" in outcome["skipped"]


# --- which figure a CV sweep gets -------------------------------------------


@pytest.mark.parametrize(
    "param_columns, expected",
    [
        ([], None),
        (["param_model__n_components"], "cv_val_curve"),
        (["param_model__max_depth", "param_model__n_estimators"], "cv_parallel_coordinates"),
    ],
)
def test_the_cv_figure_matches_what_training_would_have_drawn(param_columns, expected):
    """Including the no-sweep case, which draws NOTHING.

    A model fitted at fixed hyper-parameters has no sweep; drawing parallel coordinates over zero
    axes produced a degenerate one-axis plot and a matplotlib warning about identical axis limits.
    """
    frame = pd.DataFrame({column: [1] for column in param_columns} or {"mean_test_score": [1.0]})
    assert _cv_plot_name(frame) == expected


# --- the interval label -----------------------------------------------------


def test_the_bar_label_is_recovered_from_the_params_that_recorded_it():
    """ "bar: ±1σ" against "bar: conformal 95%" is what makes the PICP beside it interpretable."""
    sigma_run = _run(
        target="clay_pct",
        model_name="soil_cnn",
        params={"interval_method_clay_pct": "sigma", "interval_k_clay_pct": "1.0"},
    )
    assert _interval_estimator(sigma_run, "clay_pct").k == 1.0

    gaussian_run = _run(
        target="clay_pct",
        model_name="soil_cnn",
        params={
            "interval_method_clay_pct": "gaussian",
            "interval_nominal_coverage_clay_pct": "0.95",
        },
    )
    assert _interval_estimator(gaussian_run, "clay_pct").nominal_coverage == pytest.approx(0.95)


def test_a_run_that_recorded_no_interval_gets_no_label():
    assert _interval_estimator(_run(target="clay_pct", model_name="soil_cnn"), "clay_pct") is None


# --- what a dry run promises ------------------------------------------------


def test_the_planned_paths_are_flat_because_that_is_where_the_files_are():
    """Both training paths call plots_path() with no target, and every run in mlruns/ is flat.

    A replot that nested would write a second file beside the stale one it was meant to replace.
    """
    planned = _planned(has_sigma=True, cv_results=None, only=None)
    assert planned == [
        "plots/pred_obs.png",
        "uncertainty/reliability.png",
        "uncertainty/sigma_vs_error.png",
    ]


def test_only_narrows_what_gets_drawn():
    assert _planned(has_sigma=True, cv_results=None, only=["pred_obs"]) == ["plots/pred_obs.png"]
    assert _wanted(None, "pred_obs") is True
    assert _wanted(["uncertainty"], "pred_obs") is False


def test_a_run_without_sigma_plans_no_uncertainty_figures():
    assert _planned(has_sigma=False, cv_results=None, only=None) == ["plots/pred_obs.png"]


# --- the --since filter ----------------------------------------------------
# A run's start time is milliseconds since the epoch. Passing the date through as a quoted string
# made every `--experiment ... --since` run end in an MLflow parse error.


def test_since_is_converted_to_the_timestamp_mlflow_expects():
    import datetime

    import replot

    expected = int(datetime.datetime(2026, 9, 1).timestamp() * 1000)
    assert replot._since_filter("2026-09-01") == f"attributes.start_time >= {expected}"
    # Unquoted: a quoted value is what MLflow refuses for a numeric attribute.
    assert "'" not in replot._since_filter("2026-09-01")


def test_no_since_means_no_filter():
    import replot

    assert replot._since_filter(None) == ""
    assert replot._since_filter("") == ""


def test_a_malformed_since_is_reported_before_mlflow_sees_it():
    import replot

    with pytest.raises(SystemExit, match="YYYY-MM-DD"):
        replot._since_filter("01/09/2026")
