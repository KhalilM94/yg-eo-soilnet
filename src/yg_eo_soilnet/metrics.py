"""The accuracy scores every model reports, and which scale each score is on.

Every trained model - scikit-learn or deep learning - is scored by :func:`regression_metrics` on the
same test points, in the target's own units (g/kg, %, pH), so ``rmse_test`` means the same thing on
every row of the :term:`leaderboard`.

``rmse``
    Root mean squared error, in the target's units. Lower is better.
``mae``
    Mean absolute error, in the target's units. Lower is better.
``bias``
    Mean of (prediction - measurement); positive means the model over-predicts. Best near 0.
``r2``
    Coefficient of determination: 1 is perfect, 0 is no better than predicting the mean, and a
    negative value is worse than that.
``rpd``
    Standard deviation of the measurements divided by the RMSE (ratio of performance to
    deviation). Higher is better.
``rpiq``
    Interquartile range of the measurements divided by the RMSE; less sensitive to extreme values
    than ``rpd``. Higher is better.
``n``
    Number of points scored.

No score is negated to mean "higher is better"; :data:`METRIC_DIRECTION` records which way each one
is better.

Two scales
----------
``soil_cnn`` also logs its own training metrics - ``train_loss``, ``val_loss``, ``test_loss``,
``val_r2``, ... - on its training scale (log-transformed, then standardized targets), not in the
target's units, so they cannot be compared with ``rmse_test``. :data:`METRIC_SPACE` records the
scale of every metric name, and each run stores it alongside its scores.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from yg_eo_soilnet.utils import rpd_score, rpiq_score

#: For each score name, which way is better: "lower", "higher", or "zero" (best near zero, like
#: bias). ``None`` means it is not a score to rank by (a count, or a value that should sit near a
#: target rather than be maximized).
METRIC_DIRECTION: dict[str, str | None] = {
    "rmse": "lower",
    "mae": "lower",
    "r2": "higher",
    "rpd": "higher",
    "rpiq": "higher",
    "bias": "zero",
    "n": None,
    # The uncertainty scores are added below, so this stays the one table to look in.
}

#: The scores :func:`regression_metrics` computes, in the order they are recorded.
METRIC_STEMS: tuple[str, ...] = ("rmse", "mae", "r2", "rpd", "rpiq", "bias", "n")

#: Which way is better for the uncertainty scores computed by :mod:`yg_eo_soilnet.uncertainty.metrics`.
#: ``picp`` (how often the true value falls inside the predicted range) has no direction: it should
#: be close to the promised coverage (for example 95%), not as high as possible - a range covering
#: every possible value would reach 100% and say nothing. ``coverage_error`` is the rankable form.
UNCERTAINTY_METRIC_DIRECTION: dict[str, str | None] = {
    "picp": None,
    "coverage_error": "zero",
    "mpiw": "lower",
    "nmpiw": "lower",
    "interval_score": "lower",
    "crps": "lower",
    "nll": "lower",
    "ence": "lower",
    "sigma_error_corr": "higher",
    "mean_sigma": None,
}

UNCERTAINTY_METRIC_STEMS: tuple[str, ...] = tuple(UNCERTAINTY_METRIC_DIRECTION)

METRIC_DIRECTION.update(UNCERTAINTY_METRIC_DIRECTION)

#: The two scales a score can be on. See "Two scales" at the top of this module.
ORIGINAL_UNITS = "original_units"
STANDARDIZED_LOG1P = "standardized_log1p"

#: For each score name, the scale it is on: ``"original_units"`` or ``"standardized_log1p"``.
METRIC_SPACE: dict[str, str] = {
    # computed by this module, from the predictions converted back to the target's units
    **{f"{stem}_test": ORIGINAL_UNITS for stem in METRIC_STEMS},
    # computed by yg_eo_soilnet.uncertainty.metrics from the same predictions, so a mean_sigma can
    # be compared directly with the rmse_test of the same target
    **{f"{stem}_test": ORIGINAL_UNITS for stem in UNCERTAINTY_METRIC_STEMS},
    "conformal_q": ORIGINAL_UNITS,
    "rmse_cv_mean": ORIGINAL_UNITS,
    "rmse_cv_std": ORIGINAL_UNITS,
    "rmse_cv_train_mean": ORIGINAL_UNITS,
    "r2_train_fit": ORIGINAL_UNITS,
    # recorded by the deep-learning models during training, on their training scale
    "train_loss": STANDARDIZED_LOG1P,
    "val_loss": STANDARDIZED_LOG1P,
    "test_loss": STANDARDIZED_LOG1P,
    "train_r2": STANDARDIZED_LOG1P,
    "val_r2": STANDARDIZED_LOG1P,
    "test_r2": STANDARDIZED_LOG1P,
    "train_pred_std_ratio": STANDARDIZED_LOG1P,
    "val_pred_std_ratio": STANDARDIZED_LOG1P,
    "test_pred_std_ratio": STANDARDIZED_LOG1P,
}

#: Score names used by older runs. Nothing records them any more; they are listed so older runs
#: can still be read.
LEGACY_METRIC_NAMES: tuple[str, ...] = ("mean_test_score", "mean_train_score", "r2_test_legacy")


def _reject_multi_column(name: str, values: Any) -> None:
    """Raise ``ValueError`` if ``values`` holds more than one target column."""
    array = np.asarray(values)
    if array.ndim > 1 and array.shape[-1] > 1:
        raise ValueError(
            f"{name} has {array.shape[-1]} columns; regression_metrics scores ONE target at a time. "
            "Pooling several would mix their units into a single meaningless number. Call it once "
            "per target with suffix=f'_{target_name}'."
        )


def _finite_pairs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return both inputs as float arrays, keeping only positions where both have a number.

    A missing value in either one drops that position from both, so measurements and predictions
    stay paired. Several target columns at once are refused: scores are per target.
    """
    _reject_multi_column("y_true", y_true)
    _reject_multi_column("y_pred", y_pred)
    true_values = pd.to_numeric(pd.Series(np.asarray(y_true).reshape(-1)), errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(pd.Series(np.asarray(y_pred).reshape(-1)), errors="coerce").to_numpy(dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            f"y_true has {true_values.shape[0]} value(s) but y_pred has {predicted_values.shape[0]}"
        )

    keep = np.isfinite(true_values) & np.isfinite(predicted_values)
    return true_values[keep], predicted_values[keep]


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    split: str = "test",
    suffix: str = "",
) -> dict[str, float]:
    """Score predictions against lab measurements for one target.

    Computes every score listed at the top of this module. Points where the measurement or the
    prediction is missing are left out.

    Parameters
    ----------
    y_true : array-like
        Measured values for one target, in its own units.
    y_pred : array-like
        Predicted values, in the same order and units.
    split : str, default "test"
        Added to every score name: ``"test"`` gives ``rmse_test``, ``mae_test``, ...
    suffix : str, default ""
        Added after that, to tell targets apart when one model predicts several:
        ``suffix="_clay_pct"`` gives ``rmse_test_clay_pct``.

    Returns
    -------
    dict of str to float
        One entry per score. Empty if fewer than two points can be scored. A score that cannot be
        computed is left out rather than recorded as NaN: ``r2`` when every measurement is the same,
        ``rpd`` and ``rpiq`` when the predictions are perfect.

    Raises
    ------
    ValueError
        If the two inputs have different lengths, or either holds more than one target column.

    Examples
    --------
    >>> scores = regression_metrics([10, 20, 30, 40], [12, 18, 33, 39])
    >>> {name: round(value, 3) for name, value in scores.items()}
    {'rmse_test': 2.121, 'mae_test': 2.0, 'bias_test': 0.5, 'n_test': 4.0, 'r2_test': 0.964, 'rpd_test': 6.086, 'rpiq_test': 7.071}

    One target of several, named with ``suffix``:

    >>> sorted(regression_metrics([7.1, 7.9], [7.0, 8.2], suffix="_ph_water"))[:3]
    ['bias_test_ph_water', 'mae_test_ph_water', 'n_test_ph_water']

    Too few points to score:

    >>> regression_metrics([5.0], [4.0])
    {}
    """
    true_values, predicted_values = _finite_pairs(y_true, y_pred)

    count = int(true_values.shape[0])
    if count < 2:
        return {}

    def key(stem: str) -> str:
        return f"{stem}_{split}{suffix}"

    rmse = float(root_mean_squared_error(true_values, predicted_values))
    metrics: dict[str, float] = {
        key("rmse"): rmse,
        key("mae"): float(mean_absolute_error(true_values, predicted_values)),
        # Kept with its sign: positive means the model predicts too high on average.
        key("bias"): float(np.mean(predicted_values - true_values)),
        key("n"): float(count),
    }

    target_variance = float(np.var(true_values))
    if target_variance > 1e-12:
        metrics[key("r2")] = float(r2_score(true_values, predicted_values))

    if rmse > 0.0:
        # Order matters: both take (predictions, measurements) and measure the spread of the
        # measurements, the second argument.
        metrics[key("rpd")] = float(rpd_score(predicted_values, true_values))
        metrics[key("rpiq")] = float(rpiq_score(predicted_values, true_values))

    return metrics


def cv_rmse_from_search(cv_results: Any, best_index: int) -> dict[str, float]:
    """Read the :term:`cross-validation` RMSE of the best settings out of a scikit-learn search.

    scikit-learn's search reports errors as negative numbers (so that "higher is better"). This
    turns them back into ordinary positive RMSE values, in the target's units.

    Parameters
    ----------
    cv_results : dict or pandas.DataFrame
        The ``cv_results_`` of a finished ``GridSearchCV`` (or the same as a table).
    best_index : int
        Which row of it holds the chosen settings.

    Returns
    -------
    dict of str to float
        ``rmse_cv_mean`` (average RMSE over the folds), ``rmse_cv_std`` (how much it varied between
        folds) and, if recorded, ``rmse_cv_train_mean`` (the RMSE on the points each fold trained
        on). A value that is missing is left out.

    Examples
    --------
    >>> cv_rmse_from_search({"mean_test_score": [-3.2, -2.9], "std_test_score": [0.4, 0.3]}, best_index=1)
    {'rmse_cv_mean': 2.9, 'rmse_cv_std': 0.3}
    """
    if isinstance(cv_results, pd.DataFrame):
        columns: Mapping[str, Any] = {name: cv_results[name].to_numpy() for name in cv_results.columns}
    else:
        columns = cv_results

    def value_at(column_name: str) -> float | None:
        column = columns.get(column_name)
        if column is None:
            return None
        try:
            scalar = float(np.asarray(column)[best_index])
        except (IndexError, TypeError, ValueError):
            return None
        return scalar if np.isfinite(scalar) else None

    metrics: dict[str, float] = {}

    mean_test_score = value_at("mean_test_score")
    if mean_test_score is not None:
        metrics["rmse_cv_mean"] = -mean_test_score

    std_test_score = value_at("std_test_score")
    if std_test_score is not None:
        metrics["rmse_cv_std"] = std_test_score

    mean_train_score = value_at("mean_train_score")
    if mean_train_score is not None:
        metrics["rmse_cv_train_mean"] = -mean_train_score

    return metrics


def metric_space_for(metric_names: Any) -> dict[str, str]:
    """Say which scale each of the given score names is on.

    Parameters
    ----------
    metric_names : iterable of str
        Score names, as recorded on a run. Per-target names such as ``rmse_test_clay_pct`` are
        understood.

    Returns
    -------
    dict of str to str
        ``"original_units"``, ``"standardized_log1p"``, or ``"unknown"`` for a name not listed in
        :data:`METRIC_SPACE` - shown, rather than left out, so a new score nobody registered is
        noticed.

    Examples
    --------
    >>> metric_space_for(["rmse_test", "val_loss", "rmse_test_clay_pct", "my_new_metric"])
    {'rmse_test': 'original_units', 'val_loss': 'standardized_log1p', 'rmse_test_clay_pct': 'original_units', 'my_new_metric': 'unknown'}
    """
    return {str(name): _space_of(str(name)) for name in metric_names}


def _space_of(name: str) -> str:
    """Return the scale of one score name; ``rmse_test_clay_pct`` is looked up as ``rmse_test``."""
    known = METRIC_SPACE.get(name)
    if known is not None:
        return known
    # Longest names first, so the most specific match wins.
    for stem in sorted(METRIC_SPACE, key=len, reverse=True):
        if name.startswith(f"{stem}_"):
            return METRIC_SPACE[stem]
    return "unknown"
