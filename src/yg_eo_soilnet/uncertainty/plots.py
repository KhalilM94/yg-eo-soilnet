"""The two figures that say whether the error bars on a run's plots are honest.

The bars themselves show what a model claims. These check the claim: one asks whether the promised
coverage actually holds across the whole range of promises, the other whether the model's stated
uncertainty is the size of its real error.

As everywhere else here, each function builds a figure and returns it; the caller saves it.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from yg_eo_soilnet.plot_style import (
    FERTIMAP_AREA_FILL,
    FIG_WIDTH_COLUMN,
    INK_2,
    PROJECT_COLORS,
    panel_subtitle,
    square_panel,
    styled,
)

# Nominal levels swept by the reliability curve. Dense enough to show the shape, coarse enough that
# each point is a distinct empirical fraction on an ~900-row test split.
NOMINAL_LEVELS = np.linspace(0.05, 0.95, 19)


@styled
def reliability_curve(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    *,
    calibrator: Any = None,
    target_name: str = "",
    axis: Optional[plt.Axes] = None,
):
    """Promised coverage against actual coverage, swept across levels.

    A perfectly calibrated set of intervals sits on the diagonal: whatever share you promise, that share
    of measurements falls inside. Below it the intervals are too narrow, above it too wide.

    Parameters
    ----------
    y_true, y_pred, sigma : array-like
        The measurements, predictions and predicted spreads.
    calibrator : ConformalCalibrator, optional
        With one, each level is built the way the run's own intervals were, so the curve grades the
        real procedure.
    target_name : str, optional
        Named in the title.

    Returns
    -------
    matplotlib.figure.Figure
        """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    sigma_values = np.asarray(sigma, dtype=float).reshape(-1)

    figure = None
    if axis is None:
        figure, axis = plt.subplots(
            figsize=(FIG_WIDTH_COLUMN, FIG_WIDTH_COLUMN), layout="constrained"
        )
    # Both axes are coverages on the same 0-1 scale, so this panel is read against its diagonal the
    # same way the pred-vs-obs scatter is.
    square_panel(axis)

    empirical = [
        _empirical_coverage(observed, predicted, sigma_values, level, calibrator)
        for level in NOMINAL_LEVELS
    ]

    axis.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8, label="perfect", zorder=2)
    axis.plot(
        NOMINAL_LEVELS,
        empirical,
        color=PROJECT_COLORS["Al Moutmir"],
        marker="o",
        markersize=3.5,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.6,
        label="observed",
        zorder=3,
    )
    # Shading the gap makes the DIRECTION of the miscalibration readable at a glance, which is the
    # thing that determines what to do about it: below the diagonal is over-confident (intervals too
    # narrow), above is over-cautious (too wide, and the bars are not saying much).
    axis.fill_between(
        NOMINAL_LEVELS, NOMINAL_LEVELS, empirical, color=FERTIMAP_AREA_FILL, alpha=0.5, lw=0, zorder=1
    )

    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    panel_subtitle(axis, f"{target_name} reliability" if target_name else "reliability")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left", fontsize=7.5)

    return figure


@styled
def sigma_vs_error(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    *,
    n_bins: int = 10,
    target_name: str = "",
    axis: Optional[plt.Axes] = None,
):
    """The predicted spread against the error actually made, in bands of similar spread.

    Points on the diagonal mean the model's stated uncertainty is the size of its real error. Below it
    the model is over-confident, above it over-cautious. A flat line - the same error whatever the
    model claimed - means the uncertainty carries no information at all.

    Returns
    -------
    matplotlib.figure.Figure
        """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    sigma_values = np.asarray(sigma, dtype=float).reshape(-1)
    absolute_residuals = np.abs(observed - predicted)

    figure = None
    if axis is None:
        figure, axis = plt.subplots(
            figsize=(FIG_WIDTH_COLUMN, FIG_WIDTH_COLUMN), layout="constrained"
        )
    # Both axes are in the target's units, and the whole reading is "how far off the diagonal".
    square_panel(axis)

    bin_sigmas, bin_rmses = _bin_by_sigma(absolute_residuals, sigma_values, n_bins)

    if bin_sigmas.size:
        limit = float(max(bin_sigmas.max(), bin_rmses.max())) * 1.05
        axis.plot([0, limit], [0, limit], linestyle="--", color=INK_2, linewidth=0.8, zorder=2)
        axis.scatter(
            bin_sigmas,
            bin_rmses,
            s=28,
            color=PROJECT_COLORS["Al Moutmir"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)

    axis.set_xlabel("Predicted σ (bin mean)")
    axis.set_ylabel("Observed RMSE (bin)")
    panel_subtitle(
        axis, f"{target_name} σ vs realised error" if target_name else "σ vs realised error"
    )

    return figure


def _empirical_coverage(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
    level: float,
    calibrator: Any,
) -> float:
    """The share of measurements that fall inside the interval built at one promised level."""
    if calibrator is None:
        # No calibrator: read the level off the Gaussian the raw sigma implies.
        from scipy import stats

        half_width = stats.norm.ppf(0.5 + level / 2.0) * sigma
        lower, upper = predicted - half_width, predicted + half_width
    else:
        lower, upper = calibrator.intervals(predicted, sigma)
        if not np.isclose(calibrator.alpha, 1.0 - level):
            # Rescale the fitted q to this level rather than refitting: the calibrator was fitted on
            # the calibration split, which is not available here. The ratio of Gaussian z-scores is
            # the same approximation the alternative would make and keeps the curve monotone.
            from scipy import stats

            scale = stats.norm.ppf(0.5 + level / 2.0) / stats.norm.ppf(
                0.5 + (1.0 - calibrator.alpha) / 2.0
            )
            half_width = (upper - lower) / 2.0 * scale
            lower, upper = predicted - half_width, predicted + half_width

    return float(np.mean((observed >= lower) & (observed <= upper)))


def _bin_by_sigma(
    absolute_residuals: np.ndarray,
    sigma: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group the points into equal-sized bands by predicted spread, and score each.

    Equal-sized rather than equal-width bands: predicted spreads are usually bunched at the low end, so
    equal-width bands would leave most of them holding one or two points.
        """
    finite = np.isfinite(absolute_residuals) & np.isfinite(sigma)
    absolute_residuals, sigma = absolute_residuals[finite], sigma[finite]
    if sigma.size < n_bins * 2:
        return np.array([]), np.array([])

    order = np.argsort(sigma)
    bin_sigmas, bin_rmses = [], []
    for indices in np.array_split(order, n_bins):
        if indices.size == 0:
            continue
        bin_sigmas.append(float(np.mean(sigma[indices])))
        bin_rmses.append(float(np.sqrt(np.mean(absolute_residuals[indices] ** 2))))

    return np.array(bin_sigmas), np.array(bin_rmses)
