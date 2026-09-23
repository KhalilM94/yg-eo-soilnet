"""Scores that say whether a :term:`prediction interval` is honest.

The companion to :mod:`yg_eo_soilnet.metrics`, and bound by the same rules: one target at a time,
nothing negated, and every name recorded so a reader never has to guess what a number means.

An interval can be wrong in two different ways, and one score cannot catch both. A band spanning the
whole range of the data contains every measurement and says nothing; a band of zero width says a
great deal and contains nothing. So coverage is reported beside width, and beside whether the
model's uncertainty is actually larger where its errors are larger.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

# The name table lives in yg_eo_soilnet.metrics, which is the single registry of what a metric name
# means and which way is better. Imported rather than redeclared so the two cannot drift, and in
# this direction so the dependency stays acyclic.
from yg_eo_soilnet.metrics import (  # noqa: F401  (re-exported for callers of this module)
    UNCERTAINTY_METRIC_DIRECTION,
    UNCERTAINTY_METRIC_STEMS,
    _reject_multi_column,
)

# Bins used by the expected normalized calibration error. Ten is the usual choice and is about as
# fine as a ~900-row test split supports: at 20 bins each holds ~45 points and the per-bin RMSE is
# noise.
ENCE_BINS = 10


def uncertainty_metrics(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    lower: Optional[Any] = None,
    upper: Optional[Any] = None,
    *,
    alpha: float = 0.05,
    split: str = "test",
    suffix: str = "",
) -> dict[str, float]:
    """Score one target's intervals, in the target's own units.

    Parameters
    ----------
    y_true : array-like
        The measured values.
    y_pred : array-like
        The predictions, in the same order.
    sigma : array-like
        The predicted spread, in the same order.
    lower, upper : array-like, optional
        The interval bounds. Without them the interval scores are left out.
    split : str, default "test"
        Added to every score name.
    nominal_coverage : float, optional
        The share the interval claims to contain, which coverage is compared against.

    Returns
    -------
    dict of str to float
        ``picp`` the share of measurements inside the interval, ``coverage_error`` how far that is from
        what was promised, ``mpiw`` the average width, and the rest.
    """
    observed, predicted, sigma_values, lower_values, upper_values = _finite_rows(y_true, y_pred, sigma, lower, upper)

    count = int(observed.shape[0])
    if count < 2:
        return {}

    def key(stem: str) -> str:
        """One score name with the split appended, as it is recorded."""
        return f"{stem}_{split}{suffix}"

    residuals = observed - predicted
    absolute_residuals = np.abs(residuals)
    metrics: dict[str, float] = {key("mean_sigma"): float(np.mean(sigma_values))}

    if lower_values is not None and upper_values is not None:
        covered = (observed >= lower_values) & (observed <= upper_values)
        picp = float(np.mean(covered))
        widths = upper_values - lower_values
        target_range = float(np.max(observed) - np.min(observed))

        metrics[key("picp")] = picp
        # Signed, like `bias` in the point metrics: the magnitude says how far off the coverage is,
        # the sign says whether the model is over- or under-confident. Those call for opposite fixes.
        metrics[key("coverage_error")] = picp - (1.0 - alpha)
        metrics[key("mpiw")] = float(np.mean(widths))
        if target_range > 1e-12:
            metrics[key("nmpiw")] = float(np.mean(widths) / target_range)
        metrics[key("interval_score")] = _interval_score(observed, lower_values, upper_values, alpha)

    # Distributional scores need a non-degenerate sigma; a deterministic ensemble has none and gets
    # the interval metrics only. Omitted individually rather than reported as inf, the same policy
    # regression_metrics applies to r2 on a constant target.
    if np.any(sigma_values > 0.0):
        safe_sigma = np.maximum(sigma_values, np.finfo(float).tiny)
        metrics[key("crps")] = _gaussian_crps(residuals, safe_sigma)
        metrics[key("nll")] = _gaussian_nll(residuals, safe_sigma)
        ence = _ence(absolute_residuals, sigma_values)
        if ence is not None:
            metrics[key("ence")] = ence
        correlation = _sigma_error_correlation(absolute_residuals, sigma_values)
        if correlation is not None:
            metrics[key("sigma_error_corr")] = correlation

    return metrics


def _interval_score(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Width, plus a penalty for every measurement that falls outside, in proportion to how far.

    The one number that ranks intervals fairly: coverage alone rewards a band spanning everything, and
    width alone rewards a band of nothing.
    """
    widths = upper - lower
    below = np.maximum(lower - observed, 0.0)
    above = np.maximum(observed - upper, 0.0)
    return float(np.mean(widths + (2.0 / alpha) * (below + above)))


def _gaussian_crps(residuals: np.ndarray, sigma: np.ndarray) -> float:
    """How well the whole predicted distribution matches the single measured value.

    Unlike the likelihood score below, one badly missed point cannot dominate it.
    """
    standardized = residuals / sigma
    return float(
        np.mean(
            sigma
            * (
                standardized * (2.0 * stats.norm.cdf(standardized) - 1.0)
                + 2.0 * stats.norm.pdf(standardized)
                - 1.0 / np.sqrt(np.pi)
            )
        )
    )


def _gaussian_nll(residuals: np.ndarray, sigma: np.ndarray) -> float:
    """How likely the measurements are under the predicted distribution; lower is better.

    It can legitimately be negative when the intervals are genuinely tight.
    """
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + (residuals**2) / (2.0 * sigma**2)))


def _ence(absolute_residuals: np.ndarray, sigma: np.ndarray) -> Optional[float]:
    """Does the predicted spread match the real error *at each level* of spread?

    Coverage is one number for the whole test set, and a model can hit it while being over-confident on
    its easy points and over-cautious on its hard ones, the two cancelling out. This compares them
    level by level instead.
    """
    count = int(sigma.shape[0])
    if count < ENCE_BINS * 2:
        return None

    order = np.argsort(sigma)
    bins = np.array_split(order, ENCE_BINS)
    errors = []
    for indices in bins:
        if indices.size == 0:
            continue
        bin_rmse = float(np.sqrt(np.mean(absolute_residuals[indices] ** 2)))
        bin_sigma = float(np.mean(sigma[indices]))
        if bin_sigma <= 0.0:
            continue
        errors.append(abs(bin_sigma - bin_rmse) / bin_sigma)

    return float(np.mean(errors)) if errors else None


def _sigma_error_correlation(absolute_residuals: np.ndarray, sigma: np.ndarray) -> Optional[float]:
    """Is the model actually less accurate where it says it is less certain?

    An interval can be perfectly calibrated on average and still rank its points the wrong way round.
    """
    if np.ptp(sigma) <= 0.0 or np.ptp(absolute_residuals) <= 0.0:
        return None
    correlation = stats.spearmanr(sigma, absolute_residuals).statistic
    return None if not np.isfinite(correlation) else float(correlation)


def _finite_rows(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    lower: Optional[Any],
    upper: Optional[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Every input as an array, keeping only the points that are usable in all of them."""
    for name, values in (("y_true", y_true), ("y_pred", y_pred), ("sigma", sigma)):
        _reject_multi_column(name, values)

    columns = {
        "observed": _to_float(y_true),
        "predicted": _to_float(y_pred),
        "sigma": _to_float(sigma),
    }
    has_interval = lower is not None and upper is not None
    if has_interval:
        _reject_multi_column("lower", lower)
        _reject_multi_column("upper", upper)
        columns["lower"] = _to_float(lower)
        columns["upper"] = _to_float(upper)

    lengths = {name: values.shape[0] for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"uncertainty_metrics inputs have mismatched lengths: {lengths}")

    keep = np.ones(next(iter(lengths.values())), dtype=bool)
    for values in columns.values():
        keep &= np.isfinite(values)
    # A negative sigma is not a small sigma, it is a bug upstream - most likely a variance handed
    # over where a standard deviation was expected. Dropping those rows rather than taking their
    # absolute value keeps the mistake visible in the row count.
    keep &= columns["sigma"] >= 0.0

    return (
        columns["observed"][keep],
        columns["predicted"][keep],
        columns["sigma"][keep],
        columns["lower"][keep] if has_interval else None,
        columns["upper"][keep] if has_interval else None,
    )


def _to_float(values: Any) -> np.ndarray:
    """One value as a plain float, or NaN when it cannot be one."""
    return pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce").to_numpy(dtype=float)
