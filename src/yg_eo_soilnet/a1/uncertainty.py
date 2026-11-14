"""Prediction intervals for SOM: quantile regression, conformalized (CQR), and how well they cover.

The ``global_q`` model (:mod:`yg_eo_soilnet.a1.stage2`) predicts several quantiles of SOM at once
(XGBoost, ``objective: reg:quantileerror``). Its central intervals are then *conformalized*
(Romano et al., 2019; Kakhani et al., 2024, eqs. 6-7): for an interval ``(q_lo, q_hi)`` of nominal
coverage ``1 - alpha``, each calibration point gets the conformity score

    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)),

positive when the point falls outside, negative inside; ``Q`` is the ``ceil((n + 1)(1 - alpha))``-th
smallest score (:func:`yg_eo_soilnet.uncertainty.conformal._conformal_quantile`), and the conformal
interval is ``[q_lo - Q, q_hi + Q]``: widened when the raw interval covered too little, narrowed
when it covered too much.

The calibration points are the training points' out-of-fold predictions, from the plan's own
folds (random folds, blocks or basins), so no training point is held back and none is calibrated
by a model that saw it. The guarantee of conformal prediction needs calibration and test points
to be exchangeable; between blocks or basins they are not, so coverage there is measured on the
test points, not promised.

This module holds the arithmetic - quantile names, sorting, the conformal step, the scores - and
:func:`uncertainty_scores`, the ``uncertainty_scores.csv`` of ``evaluate.py``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

#: The quantile model's name in ``stage2.models`` and the predictions.
GLOBAL_Q = "global_q"
#: A quantile column of the predictions: ``som_q05`` ... ``som_q95``.
QUANTILE_COLUMN = re.compile(r"^som_q(?P<q>\d{2})$")
#: The per-point flag: True where the quantiles had to be sorted (they crossed).
SORTED_COLUMN = "quantiles_sorted"
RAW = "raw"
CQR = "cqr"


def quantile_column(quantile: float) -> str:
    """``som_q05`` for 0.05.

    Examples
    --------
    >>> quantile_column(0.05), quantile_column(0.5), quantile_column(0.95)
    ('som_q05', 'som_q50', 'som_q95')
    """
    return f"som_q{int(round(quantile * 100)):02d}"


def level_tag(level: float) -> str:
    """``80`` for a level of 0.8."""
    return f"{int(round(level * 100))}"


def bound_columns(level: float, method: str = RAW) -> tuple[str, str]:
    """The two bound columns of a central interval: ``som_lo80`` / ``som_hi80``, ``som_lo80_cqr`` / ...

    Examples
    --------
    >>> bound_columns(0.8), bound_columns(0.9, "cqr")
    (('som_lo80', 'som_hi80'), ('som_lo90_cqr', 'som_hi90_cqr'))
    """
    suffix = "" if method == RAW else f"_{method}"
    tag = level_tag(level)
    return f"som_lo{tag}{suffix}", f"som_hi{tag}{suffix}"


def level_bounds(level: float) -> tuple[float, float]:
    """The quantiles bounding a central interval: ``((1 - level) / 2, (1 + level) / 2)``."""
    return round((1.0 - level) / 2.0, 6), round((1.0 + level) / 2.0, 6)


def quantile_index(quantiles: Sequence[float], wanted: float) -> int:
    """The position of ``wanted`` among ``quantiles``.

    Raises
    ------
    ValueError
        If it is not one of them.
    """
    for position, quantile in enumerate(quantiles):
        if math.isclose(float(quantile), float(wanted), abs_tol=1e-9):
            return position
    raise ValueError(f"The quantile {wanted} is not among {list(quantiles)}.")


def sort_quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's quantiles in increasing order, and which rows needed it (their quantiles crossed).

    Examples
    --------
    >>> values, crossed = sort_quantiles(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))
    >>> values.tolist(), crossed.tolist()
    ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [False, True])
    """
    values = np.asarray(values, dtype=float)
    crossed = (np.diff(values, axis=1) < 0).any(axis=1)
    return np.sort(values, axis=1), crossed


def pinball_loss(y: np.ndarray, predictions: np.ndarray, quantiles: Sequence[float]) -> float:
    """The mean pinball (quantile) loss over the points and the quantiles.

    For one quantile ``tau``: ``tau * (y - q)`` when ``y`` lies above ``q``, ``(1 - tau) * (q - y)``
    below. Lower is better; it rewards each quantile for sitting where that share of points lies
    below it.

    Examples
    --------
    >>> pinball_loss(np.array([1.0, 3.0]), np.array([[2.0], [2.0]]), [0.5])
    0.5
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    predictions = np.asarray(predictions, dtype=float).reshape(len(y), -1)
    tau = np.asarray(quantiles, dtype=float).reshape(1, -1)
    gap = y - predictions
    return float(np.mean(np.maximum(tau * gap, (tau - 1.0) * gap)))


@dataclass(frozen=True)
class Calibration:
    """The conformal correction of one central interval.

    Attributes
    ----------
    level : float
        Its nominal coverage, ``1 - alpha``.
    q : float
        ``Q``: added to the upper bound, taken from the lower (negative narrows the interval).
    n_calibration : int
        The calibration points it rests on.
    raw_coverage : float
        The share of calibration points the raw interval covered.
    """

    level: float
    q: float
    n_calibration: int
    raw_coverage: float


def conformity_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """``E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`` (Kakhani et al., 2024, eq. 6)."""
    y = np.asarray(y, dtype=float)
    return np.maximum(np.asarray(lower, dtype=float) - y, y - np.asarray(upper, dtype=float))


def calibrate(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float) -> Calibration:
    """``Q`` of one central interval from calibration points (Kakhani et al., 2024, eq. 7).

    ``Q`` is the ``ceil((n + 1) level)``-th smallest conformity score (the finite-sample rank of
    :func:`yg_eo_soilnet.uncertainty.conformal._conformal_quantile`); with too few points for that
    rank, the largest score.

    Raises
    ------
    ValueError
        Without a calibration point.
    """
    from yg_eo_soilnet.uncertainty.conformal import _conformal_quantile

    scores = conformity_scores(y, lower, upper)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("No calibration point to conformalize the interval with.")
    scores = scores[finite]
    return Calibration(
        level=float(level),
        q=_conformal_quantile(scores, 1.0 - float(level)),
        n_calibration=int(scores.size),
        raw_coverage=float(np.mean(scores <= 0)),
    )


def interval_columns(
    quantiles: Sequence[float],
    values: np.ndarray,
    levels: Sequence[float],
    calibrations: Optional[dict] = None,
) -> dict:
    """Every quantile, raw bound and (with ``calibrations``) conformal bound column of the predictions.

    Parameters
    ----------
    quantiles : sequence of float
    values : numpy.ndarray
        One row per point, one sorted column per quantile.
    levels : sequence of float
    calibrations : dict of float to Calibration, optional
        Per level; without them only the raw bounds are written.
    """
    columns = {quantile_column(q): values[:, position] for position, q in enumerate(quantiles)}
    for level in levels:
        low_q, high_q = level_bounds(level)
        low, high = values[:, quantile_index(quantiles, low_q)], values[:, quantile_index(quantiles, high_q)]
        low_name, high_name = bound_columns(level)
        columns[low_name], columns[high_name] = low, high
        if calibrations is not None and level in calibrations:
            q = calibrations[level].q
            low_name, high_name = bound_columns(level, CQR)
            columns[low_name], columns[high_name] = low - q, high + q
    return columns


# --- scores ---------------------------------------------------------------------------------------


def quantiles_of(frame: pd.DataFrame) -> list[float]:
    """The quantiles a predictions table holds, from its ``som_qNN`` columns, in order."""
    found = []
    for column in frame.columns:
        match = QUANTILE_COLUMN.match(str(column))
        if match and frame[column].notna().any():
            found.append(int(match.group("q")) / 100.0)
    return sorted(found)


def levels_of(frame: pd.DataFrame) -> list[float]:
    """The interval levels a predictions table holds, from its ``som_loNN`` columns."""
    found = []
    for column in frame.columns:
        match = re.match(r"^som_lo(?P<level>\d{2})$", str(column))
        if match and frame[column].notna().any():
            found.append(int(match.group("level")) / 100.0)
    return sorted(found)


def methods_of(frame: pd.DataFrame, level: float) -> list[str]:
    """``raw``, and ``cqr`` when the conformal bounds of ``level`` are there."""
    low, _ = bound_columns(level, CQR)
    return [RAW, CQR] if low in frame.columns and frame[low].notna().any() else [RAW]


def point_scores(part: pd.DataFrame, quantiles: Sequence[float], levels: Sequence[float]) -> list[dict]:
    """Every uncertainty score of some test points: rows ``method, level, quantile, metric, value``.

    PICP (the share of measured SOM inside the interval) and the mean and median width (PIW) per
    level and method; QCP (the share of measured SOM below it) per quantile; the mean pinball loss;
    RMSE, MAE and bias of the median (``predicted``).
    """
    y = part["observed"].to_numpy(dtype=float)
    rows = []
    for level in levels:
        for method in methods_of(part, level):
            low, high = bound_columns(level, method)
            lower, upper = part[low].to_numpy(dtype=float), part[high].to_numpy(dtype=float)
            inside = (y >= lower) & (y <= upper)
            width = upper - lower
            for metric, value in (
                ("picp", float(np.mean(inside))),
                ("piw_mean", float(np.mean(width))),
                ("piw_median", float(np.median(width))),
            ):
                rows.append({"method": method, "level": level, "quantile": np.nan, "metric": metric, "value": value})
    values = part[[quantile_column(q) for q in quantiles]].to_numpy(dtype=float)
    for position, q in enumerate(quantiles):
        rows.append(
            {
                "method": RAW,
                "level": np.nan,
                "quantile": q,
                "metric": "qcp",
                "value": float(np.mean(y <= values[:, position])),
            }
        )
    median = part["predicted"].to_numpy(dtype=float)
    error = median - y
    for metric, value in (
        ("pinball", pinball_loss(y, values, quantiles)),
        ("rmse_median", float(np.sqrt(np.mean(error**2)))),
        ("mae_median", float(np.mean(np.abs(error)))),
        ("bias_median", float(np.mean(error))),
    ):
        rows.append({"method": RAW, "level": np.nan, "quantile": np.nan, "metric": metric, "value": value})
    return rows


#: The scores that get a block-bootstrap interval.
BOOTSTRAPPED = ("picp", "piw_mean")


def _bootstrap(part: pd.DataFrame, blocks: np.ndarray, levels, n: int, seed: int, key: tuple) -> dict:
    """95 % block-bootstrap intervals of PICP and mean width per level and method: ``{(method, level, metric): (lo, hi)}``."""
    from yg_eo_soilnet.a1 import evaluation

    y = part["observed"].to_numpy(dtype=float)
    bounds = {}
    for level in levels:
        for method in methods_of(part, level):
            low, high = bound_columns(level, method)
            lower, upper = part[low].to_numpy(dtype=float), part[high].to_numpy(dtype=float)
            inside = ((y >= lower) & (y <= upper)).astype(float)
            width = upper - lower
            draws = np.array(
                [
                    (inside[rows].mean(), width[rows].mean())
                    for rows in evaluation._draws(blocks, n, evaluation._generator(seed, (*key, method, level)))
                ]
            )
            n_blocks = len(np.unique(blocks))
            for position, metric in enumerate(BOOTSTRAPPED):
                if n_blocks < 2 or not len(draws):
                    bounds[(method, level, metric)] = (np.nan, np.nan)
                else:
                    bounds[(method, level, metric)] = tuple(np.percentile(draws[:, position], [2.5, 97.5]))
    return bounds


def _scopes(test: pd.DataFrame):
    """The groups ``uncertainty_scores.csv`` scores, with the pooling rules of the other tables.

    Random pooled; spatial folds pooled per block size; plain ``logo`` per basin; the mixed kinds
    (``mixed``, its arms, ``control``) per basin and share, the repeats' test points pooled; matched
    ``logo`` draws per draw.
    """
    from yg_eo_soilnet.a1.plans import DIVIDED_KINDS, LOGO, RANDOM, SPATIAL_KFOLD, is_matched

    matched = is_matched(test["plan_id"])
    common = {"kind": None, "block_size_m": np.nan, "region": "", "fraction": np.nan, "plan_id": ""}
    for kind, part in test[test["kind"] == RANDOM].groupby("kind"):
        yield "random", {**common, "kind": kind}, part
    for size, part in test[test["kind"] == SPATIAL_KFOLD].groupby("block_size_m", dropna=False):
        scope = f"{SPATIAL_KFOLD}_{float(size):g}m" if pd.notna(size) else SPATIAL_KFOLD
        yield scope, {**common, "kind": SPATIAL_KFOLD, "block_size_m": size}, part
    plain = test[(test["kind"] == LOGO) & ~matched]
    for basin, part in plain.groupby("held_out"):
        yield f"logo_{basin}", {**common, "kind": LOGO, "region": basin}, part
    divided = test[test["kind"].isin(DIVIDED_KINDS)]
    for (kind, basin, fraction), part in divided.groupby(["kind", "held_out", "fraction"]):
        scope = f"{kind}_{basin}_f{float(fraction):g}"
        yield scope, {**common, "kind": kind, "region": basin, "fraction": float(fraction)}, part
    for plan_id, part in test[(test["kind"] == LOGO) & matched].groupby("plan_id"):
        yield plan_id, {**common, "kind": "logo_matched", "region": part["held_out"].iloc[0], "plan_id": plan_id}, part


def uncertainty_scores(
    frame: pd.DataFrame,
    *,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
    min_cell_size: int = 30,
    class_column: str = "carbonate_class_measured",
) -> pd.DataFrame:
    """``uncertainty_scores.csv``: every uncertainty score of ``global_q``, per scope and carbonate class.

    Parameters
    ----------
    frame : pandas.DataFrame
        Prediction rows (:func:`~yg_eo_soilnet.a1.evaluation.load_predictions`); only the test rows of
        ``global_q`` are read.
    block_km, n, seed
        The block bootstrap of PICP and mean width.
    min_cell_size : int
        A carbonate class gets its rows only with at least this many test points in the scope.

    Returns
    -------
    pandas.DataFrame
        ``scope, kind, block_size_m, region, fraction, plan_id, class`` (``all`` or a measured class),
        ``method`` (``raw`` / ``cqr``), ``level, quantile, metric, value, low, high, n, n_blocks``.
    """
    from yg_eo_soilnet.a1 import evaluation
    from yg_eo_soilnet.datamodules.splitting import TEST

    columns = [
        "scope",
        "kind",
        "block_size_m",
        "region",
        "fraction",
        "plan_id",
        "class",
        "method",
        "level",
        "quantile",
        "metric",
        "value",
        "low",
        "high",
        "n",
        "n_blocks",
    ]
    test = frame[(frame["model"] == GLOBAL_Q) & (frame["split"] == TEST)] if "model" in frame else frame.iloc[0:0]
    if test.empty:
        return pd.DataFrame(columns=columns)
    test = test.assign(_block=evaluation.spatial_blocks(test, block_km).to_numpy())
    quantiles, levels = quantiles_of(test), levels_of(test)
    rows = []
    for scope, keys, part in _scopes(test):
        groups = [("all", part)]
        if class_column in part:
            for label, chosen in part.groupby(class_column, observed=True):
                if len(chosen) >= min_cell_size:
                    groups.append((str(label), chosen))
        for label, chosen in groups:
            blocks = chosen["_block"].to_numpy()
            bounds = _bootstrap(chosen, blocks, levels, n, seed, (scope, label))
            for row in point_scores(chosen, quantiles, levels):
                low, high = bounds.get((row["method"], row["level"], row["metric"]), (np.nan, np.nan))
                rows.append(
                    {
                        "scope": scope,
                        **keys,
                        "class": label,
                        **row,
                        "low": low,
                        "high": high,
                        "n": len(chosen),
                        "n_blocks": len(np.unique(blocks)),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def headline_rows(scores: pd.DataFrame) -> list[dict]:
    """``picp<NN>__<scope>__global_q`` and ``piw<NN>__<scope>__global_q`` with ``__lo``/``__hi``, for MLflow.

    Every point of a scope (class ``all``); the conformal interval when the run has it, the raw one
    otherwise; the raw interval's PICP also as ``picp<NN>_raw``. Rows in the shape of
    :func:`~yg_eo_soilnet.a1.evaluation.headline_table`.
    """
    from yg_eo_soilnet.a1.tracking import metric_name

    rows = []
    if scores.empty:
        return rows
    overall = scores[(scores["class"] == "all") & scores["metric"].isin(BOOTSTRAPPED)]
    for (scope, level), part in overall.groupby(["scope", "level"]):
        chosen = CQR if CQR in set(part["method"]) else RAW
        tag = level_tag(level)
        for metric, short in (("picp", f"picp{tag}"), ("piw_mean", f"piw{tag}")):
            row = part[(part["method"] == chosen) & (part["metric"] == metric)]
            if row.empty:
                continue
            row = row.iloc[0]
            for bound, value in (("", row["value"]), ("lo", row["low"]), ("hi", row["high"])):
                if pd.notna(value):
                    rows.append(
                        _headline(metric_name(short, scope, GLOBAL_Q, None, bound or None), short, scope, bound, value)
                    )
        raw = part[(part["method"] == RAW) & (part["metric"] == "picp")]
        if chosen == CQR and not raw.empty:
            short = f"picp{tag}_raw"
            rows.append(
                _headline(metric_name(short, scope, GLOBAL_Q, None, None), short, scope, "", raw.iloc[0]["value"])
            )
    return rows


def _headline(name, metric, scope, bound, value) -> dict:
    return {
        "name": name,
        "metric": metric,
        "scope": scope,
        "model": GLOBAL_Q,
        "covariate_group": "",
        "bound": bound,
        "value": float(value),
    }
