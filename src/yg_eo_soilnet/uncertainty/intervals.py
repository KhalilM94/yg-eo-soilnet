"""Turn a predicted spread into a lower and an upper bound.

Three ways, set by ``uncertainty.interval.method``, and they do not mean the same thing:

``conformal``
    The spread times a factor measured on held-back points, so the promised share of measurements
    really does fall inside. The only one whose coverage is checked rather than assumed, and the
    only one that means the same thing across both model families. The default.
``gaussian``
    The spread times the factor a bell curve implies for the promised coverage. Right only if the
    errors really are bell-shaped and the spread is their true size.
``sigma``
    The spread times a number you choose. It makes no coverage claim at all: a one-sigma band claims
    about 68%, not 95%, and the scores grade it against that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

CONFORMAL = "conformal"
GAUSSIAN = "gaussian"
SIGMA = "sigma"
NONE = "none"
INTERVAL_METHODS = (CONFORMAL, GAUSSIAN, SIGMA, NONE)

# The methods that need rows the model never saw. Only conformal does, and that single fact decides
# whether the sklearn family has to give up its val split - see ModelTrainer._resolve_fit_pool.
METHODS_NEEDING_CALIBRATION = (CONFORMAL,)


def needs_calibration_set(method: str) -> bool:
    """Whether this method has to be fitted on held-back points before it can be used."""
    return normalize_method(method) in METHODS_NEEDING_CALIBRATION


def normalize_method(method: Any) -> str:
    """Resolve a configured method name, including the older ``split_conformal`` spelling."""
    name = str(method or CONFORMAL).strip().lower()
    if name == "split_conformal":
        return CONFORMAL
    if name not in INTERVAL_METHODS:
        raise ValueError(
            f"Unknown uncertainty.interval.method {method!r}; expected one of "
            f"{', '.join(INTERVAL_METHODS)}."
        )
    return name


@dataclass(frozen=True)
class GaussianInterval:
    """The spread times what a bell curve implies for the promised coverage.

    Calibrated by nothing: right only if the errors really are bell-shaped and the spread is their true
    size.

    Attributes
    ----------
    alpha : float
        The share of points allowed to fall outside.
    target : str
        Which target it belongs to.
        """

    alpha: float = 0.05

    @property
    def z(self) -> float:
        """How many standard deviations wide the band is, for the promised coverage."""
        return float(stats.norm.ppf(1.0 - float(self.alpha) / 2.0))

    @property
    def nominal_coverage(self) -> float:
        """The share of measurements this interval claims to contain."""
        return 1.0 - float(self.alpha)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper bounds for these predictions."""
        mean_array = np.asarray(mean, dtype=float)
        half_width = self.z * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        """The interval settings as plain values, to record with the run."""
        return {
            "interval_method": GAUSSIAN,
            "interval_alpha": float(self.alpha),
            "interval_z": self.z,
        }


@dataclass(frozen=True)
class SigmaInterval:
    """The spread times a number you choose, with no coverage promise attached.

    ``nominal_coverage`` is what that number would mean if the errors were bell-shaped - about 68% at
    one, 95% at two - because the scores need something to compare the real coverage against.

    Attributes
    ----------
    k : float
        What the spread is multiplied by.
    target : str
        Which target it belongs to.
        """

    k: float = 1.0

    @property
    def nominal_coverage(self) -> float:
        """The share this band would contain if the errors were bell-shaped."""
        return float(2.0 * stats.norm.cdf(float(self.k)) - 1.0)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper bounds for these predictions."""
        mean_array = np.asarray(mean, dtype=float)
        half_width = float(self.k) * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        """The interval settings as plain values, to record with the run."""
        return {
            "interval_method": SIGMA,
            "interval_k": float(self.k),
            "interval_nominal_coverage": self.nominal_coverage,
        }


def effective_alpha(method: Any, alpha: float = 0.05, k: float = 1.0) -> float:
    """The share of points the configured interval really expects to fall outside.

    ``alpha`` for the calibrated and bell-curve methods; for a plain spread band, whatever that
    multiple implies. The scores use this, so a one-sigma band is graded against the 68% it claims
    rather than against 95%.
        """
    return 1.0 - SigmaInterval(k).nominal_coverage if normalize_method(method) == SIGMA else float(alpha)


def describe(estimator: Any) -> str:
    """A short label for a figure, so it says which claim its bars are making."""
    if estimator is None:
        return ""
    if isinstance(estimator, SigmaInterval):
        return f"±{float(estimator.k):g}σ"
    if isinstance(estimator, GaussianInterval):
        return f"gaussian {estimator.nominal_coverage:.0%}"
    coverage = getattr(estimator, "nominal_coverage", None)
    return f"conformal {coverage:.0%}" if coverage is not None else "interval"


def build_interval_estimators(
    method: Any,
    target_names: Sequence[str],
    *,
    alpha: float = 0.05,
    k: float = 1.0,
    prediction: Any = None,
    y_calib: Any = None,
    logger: Any = None,
) -> Mapping[str, Any]:
    """Build one interval estimator per target, for the configured method.

    ``prediction`` and ``y_calib`` are read by the calibrated method only; the others need no data.

    Returns
    -------
    dict of str to object
        """
    resolved = normalize_method(method)
    if resolved == NONE:
        return {}
    if resolved == GAUSSIAN:
        return {str(name): GaussianInterval(alpha=float(alpha)) for name in target_names}
    if resolved == SIGMA:
        return {str(name): SigmaInterval(k=float(k)) for name in target_names}

    from yg_eo_soilnet.uncertainty import fit_calibrators

    if prediction is None or y_calib is None:
        raise ValueError(
            "The conformal interval needs held-out predictions and observations to fit against. "
            "Either supply a calibration set or choose uncertainty.interval.method: gaussian|sigma, "
            "which need none."
        )
    return fit_calibrators(prediction, y_calib, target_names, alpha=float(alpha), logger=logger)


def estimator_from_config(config: Any, target_names: Sequence[str], **kwargs) -> Mapping[str, Any]:
    """Like :func:`build_interval_estimators`, with the settings read from the run's configuration."""
    return build_interval_estimators(
        getattr(config, "UNCERTAINTY_INTERVAL_METHOD", CONFORMAL),
        target_names,
        alpha=float(getattr(config, "UNCERTAINTY_ALPHA", 0.05)),
        k=float(getattr(config, "UNCERTAINTY_INTERVAL_K", 1.0)),
        **kwargs,
    )
