"""Give every prediction a "plus or minus", and a range it should fall in.

Switched on with ``uncertainty.enabled``. Each model is then trained several times from different
seeds - an :term:`ensemble` - and what the members disagree about becomes the uncertainty. A model
with a :term:`variance head` also predicts how noisy each point is, and the two are kept apart:
disagreement between members shrinks as more data is collected (:term:`epistemic uncertainty`),
noise in the measurement does not (:term:`aleatoric uncertainty`).

The spread is then turned into a :term:`prediction interval` - the range a measurement should fall
in, so often. ``conformal`` calibration measures how far the predictions actually fall from the
measurements on held-back points and scales the intervals to match, which is what makes the promised
coverage mean something.

The same for both model families, so an interval means the same thing whichever model produced it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from yg_eo_soilnet.artifacts import ArtifactLayout, log_figure, log_json
from yg_eo_soilnet.uncertainty.columns import (
    ALEATORIC_STD,
    EPISTEMIC_STD,
    LOWER,
    STD,
    UNCERTAINTY_STEMS,
    UPPER,
    column_name,
    interval_columns,
    is_prediction_column,
    sigma_column,
)
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator, fit_conformal
from yg_eo_soilnet.uncertainty.ensemble import (
    BOOTSTRAP_AUTO,
    EnsemblePrediction,
    aggregate,
    bootstrap_indices,
    member_seeds,
    should_bootstrap,
)
from yg_eo_soilnet.uncertainty.metrics import uncertainty_metrics

__all__ = [
    "ALEATORIC_STD",
    "BOOTSTRAP_AUTO",
    "ConformalCalibrator",
    "EPISTEMIC_STD",
    "EnsemblePrediction",
    "LOWER",
    "STD",
    "UNCERTAINTY_STEMS",
    "UPPER",
    "aggregate",
    "attach_uncertainty_columns",
    "bootstrap_indices",
    "column_name",
    "fit_calibrators",
    "fit_conformal",
    "interval_columns",
    "is_prediction_column",
    "log_uncertainty_artifacts",
    "member_seeds",
    "should_bootstrap",
    "sigma_column",
    "uncertainty_enabled_for",
    "uncertainty_metrics",
]


def uncertainty_enabled_for(config: Any, model_name: str) -> bool:
    """Whether this model should be trained as an :term:`ensemble`.

    ``uncertainty.models`` names the models to do it for; ``uncertainty.exclude_models`` names ones to
    leave out. Naming a model explicitly beats a blanket exclusion, since the cost is per model.
    """
    if not bool(getattr(config, "UNCERTAINTY_ENABLED", False)):
        return False

    allowed = [str(name) for name in (getattr(config, "UNCERTAINTY_MODELS", None) or [])]
    if allowed:
        return str(model_name) in allowed

    skipped = [str(name) for name in (getattr(config, "UNCERTAINTY_SKIP_MODELS", None) or [])]
    return str(model_name) not in skipped


def fit_calibrators(
    prediction: EnsemblePrediction,
    y_calib: Any,
    target_names: Sequence[str],
    *,
    alpha: float = 0.05,
    logger: Any = None,
) -> dict[str, ConformalCalibrator]:
    """Fit one :term:`conformal` calibrator per target, on the held-back points.

    One per target, never pooled: the amount by which a pH interval has to be widened is not the amount
    a g/kg interval needs.

    Parameters
    ----------
    prediction : EnsemblePrediction
        The ensemble's predictions for the held-back points.
    y_calib : pandas.DataFrame
        Their measured values.
    target_names : sequence of str
        The targets, in the order the predictions hold them.
    alpha : float, default 0.05
        The share of points allowed to fall outside the interval; 0.05 promises 95% coverage.
    logger : logging.Logger, optional
        Where a target that could not be calibrated is reported.

    Returns
    -------
    dict of str to :class:`~yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator`
    """
    calibrators: dict[str, ConformalCalibrator] = {}
    for index, target_name in enumerate(target_names):
        observed = (
            y_calib[target_name] if isinstance(y_calib, pd.DataFrame) and target_name in y_calib.columns else y_calib
        )
        calibrators[target_name] = fit_conformal(
            observed,
            prediction.mean[:, index],
            prediction.total_std[:, index],
            alpha=alpha,
            logger=logger,
        )
    return calibrators


def attach_uncertainty_columns(
    frame: pd.DataFrame,
    prediction: EnsemblePrediction,
    target_names: Sequence[str],
    calibrators: Optional[Mapping[str, ConformalCalibrator]] = None,
) -> pd.DataFrame:
    """Add the spread and interval columns to a results table, in place.

    Writes ``prediction_std``, ``prediction_lower`` and ``prediction_upper`` - suffixed with the target
    name when there are several. The predictions already in the table are left alone.
    """
    multi_target = len(target_names) > 1
    calibrators = calibrators or {}

    for index, target_name in enumerate(target_names):

        def name(stem: str) -> str:
            """This target's column name for one kind of uncertainty value."""
            return column_name(stem, target_name, multi_target=multi_target)

        total_std = prediction.total_std[:, index]
        frame[name(STD)] = total_std
        frame[name(EPISTEMIC_STD)] = prediction.epistemic_std[:, index]
        frame[name(ALEATORIC_STD)] = prediction.aleatoric_std[:, index]

        calibrator = calibrators.get(target_name)
        if calibrator is not None:
            lower, upper = calibrator.intervals(prediction.mean[:, index], total_std)
            frame[name(LOWER)] = lower
            frame[name(UPPER)] = upper

    return frame


def log_uncertainty_artifacts(
    frame: pd.DataFrame,
    target_name: str,
    *,
    calibrator: Optional[ConformalCalibrator] = None,
    artifact_path: Optional[str] = None,
) -> dict:
    """Write one target's uncertainty figures and summary, and say what was written.

    Returns
    -------
    dict
        What went into the run summary; empty when the table carries no uncertainty.
    """
    from yg_eo_soilnet.uncertainty.plots import reliability_curve, sigma_vs_error

    sigma = sigma_column(frame, target_name)
    if sigma is None or target_name not in frame.columns or "prediction" not in frame.columns:
        return {}

    observed = frame[target_name]
    predicted = frame["prediction"]
    destination = artifact_path or ArtifactLayout.uncertainty_path()

    written: dict = {"artifacts": []}
    figures = [
        (
            reliability_curve(observed, predicted, sigma, calibrator=calibrator, target_name=target_name),
            ArtifactLayout.RELIABILITY_FILE,
        ),
        (
            sigma_vs_error(observed, predicted, sigma, target_name=target_name),
            ArtifactLayout.SIGMA_ERROR_FILE,
        ),
    ]
    for figure, filename in figures:
        log_figure(figure, filename, destination)
        written["artifacts"].append(f"{destination}/{filename}")

    summary = {
        "target": target_name,
        "n_rows": int(len(frame)),
        "mean_sigma": float(sigma.mean()),
        **(calibrator.to_dict() if calibrator is not None else {}),
    }
    log_json(summary, ArtifactLayout.UNCERTAINTY_SUMMARY_FILE, destination)
    written["artifacts"].append(f"{destination}/{ArtifactLayout.UNCERTAINTY_SUMMARY_FILE}")
    written.update(summary)

    return written
