"""Turn a model's spread into an interval that really does contain what it claims.

A raw ensemble spread is not a :term:`prediction interval`. Members trained on the same data agree
with each other far more than they agree with reality, so "the average, plus or minus twice the
spread" typically contains far fewer than the 95% of measurements it claims: the spread measures
disagreement, not error.

:term:`Conformal <conformal>` calibration fixes that by measurement. The model predicts points it
was not trained on, and the intervals are scaled by whatever factor it takes for the promised share
of those measurements to fall inside. The guarantee holds as long as new points resemble the
calibration points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Below this the interval is treated as coming from a model with no usable sigma and the calibrator
# falls back to a constant width. Not zero: a float sigma that underflows to 1e-300 would otherwise
# produce a conformity score of 1e300 and a q that swallows the whole target range.
SIGMA_FLOOR = 1e-12

# The guarantee is asymptotic in n_calib, and the finite-sample correction below cannot produce a
# valid quantile at all when n_calib < 1/alpha - 1 (at alpha=0.05, fewer than 19 points). Warn well
# above that, because coverage at n=25 is still extremely noisy run to run.
MIN_RELIABLE_CALIBRATION_ROWS = 50


@dataclass(frozen=True)
class ConformalCalibrator:
    """The factor that makes an interval cover, and how it was arrived at.

    Attributes
    ----------
    q : float
        What the spread is multiplied by.
    alpha : float
        The share of points allowed to fall outside; 0.05 promises 95% coverage.
    n_calibration : int
        How many held-back points it was fitted on.
    target : str
        Which target it belongs to.
        """

    q: float
    alpha: float
    n_calib: int
    # False when the calibration sigma was degenerate and q is an ABSOLUTE residual quantile rather
    # than a multiplier on sigma. The interval is then a constant width, which is honest for a
    # deterministic ensemble and would be silently wrong if read as `q * sigma`.
    normalized: bool = True

    @property
    def nominal_coverage(self) -> float:
        """The share of measurements this interval claims to contain.

        Compared against the share that actually fall inside, which is what ``coverage_error`` reports.
                """
        return 1.0 - float(self.alpha)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper bounds, in the units the calibrator was fitted on."""
        mean_array = np.asarray(mean, dtype=float)
        if self.normalized:
            half_width = self.q * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        else:
            half_width = np.full_like(mean_array, self.q)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        """The calibrator as plain values, to record with the run."""
        return {
            "conformal_q": float(self.q),
            "conformal_alpha": float(self.alpha),
            "conformal_n_calib": int(self.n_calib),
            "conformal_normalized": bool(self.normalized),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> Optional["ConformalCalibrator"]:
        """Rebuild a calibrator from what :meth:`to_dict` wrote, or None if it is not there.

        Everything it holds is a plain number and all of it is written into
        ``uncertainty/uncertainty_summary.json``, so a finished run carries enough to rebuild it exactly.
                """
        if not isinstance(payload, dict) or "conformal_q" not in payload:
            return None
        try:
            return cls(
                q=float(payload["conformal_q"]),
                alpha=float(payload["conformal_alpha"]),
                n_calib=int(payload["conformal_n_calib"]),
                normalized=bool(payload.get("conformal_normalized", True)),
            )
        except (KeyError, TypeError, ValueError):
            return None


def fit_conformal(
    y_calib: Any,
    mean_calib: Any,
    sigma_calib: Optional[Any] = None,
    *,
    alpha: float = 0.05,
    logger: Any = None,
) -> ConformalCalibrator:
    """Fit the factor for one target, from how far its predictions actually fell.

    Parameters
    ----------
    y_true : array-like
        The measured values of the held-back points.
    y_pred : array-like
        The predictions for them.
    sigma : array-like
        The predicted spread for them.
    alpha : float, default 0.05
        The share allowed to fall outside.
    target : str, optional
        The target's name.

    Returns
    -------
    ConformalCalibrator or None
        None when there are too few usable points to fit one.
        """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    y_array = np.asarray(y_calib, dtype=float).reshape(-1)
    mean_array = np.asarray(mean_calib, dtype=float).reshape(-1)
    if y_array.shape != mean_array.shape:
        raise ValueError(
            f"y_calib has {y_array.shape[0]} rows and mean_calib has {mean_array.shape[0]}; "
            "conformal calibration pairs them row for row."
        )

    if sigma_calib is None:
        sigma_array = np.zeros_like(y_array)
    else:
        sigma_array = np.asarray(sigma_calib, dtype=float).reshape(-1)
        if sigma_array.shape != y_array.shape:
            raise ValueError(
                f"sigma_calib has {sigma_array.shape[0]} rows but y_calib has {y_array.shape[0]}."
            )

    finite = np.isfinite(y_array) & np.isfinite(mean_array) & np.isfinite(sigma_array)
    y_array, mean_array, sigma_array = y_array[finite], mean_array[finite], sigma_array[finite]

    n_calib = int(y_array.shape[0])
    if n_calib < 2:
        raise ValueError(
            f"Conformal calibration needs at least 2 finite calibration rows; got {n_calib}. "
            "Check that the calibration split is non-empty and that this target is measured on it."
        )
    if n_calib < MIN_RELIABLE_CALIBRATION_ROWS and logger is not None:
        logger.warning(
            f"Conformal calibration has only {n_calib} rows (alpha={alpha}). The coverage guarantee "
            f"holds in expectation but is very noisy below ~{MIN_RELIABLE_CALIBRATION_ROWS} rows; "
            "treat picp_test from this run as indicative rather than as a measurement."
        )

    absolute_residuals = np.abs(y_array - mean_array)

    # A sigma that is degenerate on the calibration set cannot normalize anything, and dividing by
    # it would turn the scores into noise amplified by 1/eps. Fall back to an ABSOLUTE residual
    # quantile: a constant-width band, which is exactly the right answer for a deterministic
    # ensemble and is flagged so no reader mistakes q for a multiplier.
    normalized = bool(np.any(sigma_array > SIGMA_FLOOR))
    if normalized:
        scores = absolute_residuals / np.maximum(sigma_array, SIGMA_FLOOR)
    else:
        scores = absolute_residuals

    return ConformalCalibrator(
        q=_conformal_quantile(scores, alpha),
        alpha=float(alpha),
        n_calib=n_calib,
        normalized=normalized,
    )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The quantile of the calibration errors the guarantee rests on.

    Slightly beyond the plain quantile, because the promise is about where a *new* point falls among
    the calibration points, not among themselves.
        """
    n_scores = int(scores.shape[0])
    rank = math.ceil((n_scores + 1) * (1.0 - alpha))
    if rank > n_scores:
        # Too few calibration points to place the bound at all: at alpha=0.05 this is n < 19. The
        # widest honest answer available is the largest score observed.
        return float(np.max(scores))
    # Indexed directly into the sorted scores rather than going through np.quantile. Even with
    # method="higher", np.quantile maps a probability onto the position (n-1)*p and then rounds,
    # which is NOT the rank-th order statistic: at n=99 and alpha=0.05 the rank is 95 and
    # np.quantile(scores, 95/99, method="higher") returns the 96th. One rank too wide is harmless
    # for coverage and wrong for the quantity being reported, and the discrepancy grows with alpha.
    return float(np.sort(scores)[rank - 1])
