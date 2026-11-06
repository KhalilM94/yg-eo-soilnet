"""Scores, intervals and transfer losses of the A1 experiments, from the per-point prediction files.

Everything here reads only what ``run_a1.py`` wrote - one row per plan, model, covariate group and
held-out point (:data:`~yg_eo_soilnet.a1.runner.PREDICTION_COLUMNS`) - so a table or a figure can be
rebuilt without the data, the settings or a model:

:func:`grouped_metrics`
    RMSE, MAE, R², bias, RPD and RPIQ for every group of rows (a model, a region, a carbonate
    class, ...), with the number of points behind each score and a flag on groups too small to
    trust.
:func:`block_bootstrap`
    How far each score could move by chance: 95 % intervals from resampling whole spatial blocks,
    and the same for the difference between two models scored on the same points, with a p-value.
:func:`holm`
    The Holm correction of a family of p-values, for when several comparisons are read together.
:func:`transfer_table`
    For ``logo`` and ``mixed`` plans, how much worse a model trained elsewhere does in a region than
    one trained there, per region and covariate group.

Why whole blocks? Neighbouring soil samples resemble each other, so resampling single points would
treat a cluster of near-copies as independent evidence and give intervals that are too narrow.
Resampling 10 km squares keeps each cluster together: the interval reflects how many separate
places were measured, not how many samples.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM, SPATIAL_KFOLD, slug
from yg_eo_soilnet.a1.runner import PREDICTION_COLUMNS
from yg_eo_soilnet.a1.stage2 import GLOBAL_AUX, LOCAL
from yg_eo_soilnet.datamodules.splitting import TEST
from yg_eo_soilnet.metrics import regression_metrics
from yg_eo_soilnet.utils import assign_grid_ids

#: The scores reported for every group, in order.
SCORES = ("rmse", "mae", "r2", "bias", "rpd", "rpiq")
#: Groups with fewer scored points than this are flagged ``small``: their scores rest on too few points.
MIN_GROUP_SIZE = 30
#: What makes two rows the same prediction target: one point, held out once, in one plan.
PAIR_KEYS = ("plan_id", "point_id", "split")
#: Scores for which a larger value is better; for the others (RMSE, MAE) smaller is better.
HIGHER_IS_BETTER = ("r2", "rpd", "rpiq")


@dataclass
class BootstrapResult:
    """What :func:`block_bootstrap` returns.

    Attributes
    ----------
    intervals : pandas.DataFrame
        One row per group and score: the ``by`` columns, ``metric, estimate, low, high, n,
        n_blocks, n_valid_draws, small``.
    differences : pandas.DataFrame
        One row per group, compared pair and score: the ``by`` columns other than the compared one,
        ``a, b, metric, estimate, low, high, p_value, p_holm, n, n_blocks, n_valid_draws,
        n_unpaired, small``. ``estimate`` is score(a) - score(b). Empty when nothing was compared.
    """

    intervals: pd.DataFrame
    differences: pd.DataFrame


# --- reading -------------------------------------------------------------------------------------


def load_predictions(
    run_dir,
    *,
    kinds: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    covariate_groups: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Every prediction file of a run folder, in one table.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        The run folder, ``<output_dir>/runs/<run name>/``.
    kinds, models, covariate_groups : iterable of str, optional
        Keep only these plan kinds, models or covariate groups.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`~yg_eo_soilnet.a1.runner.PREDICTION_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If the folder holds no prediction file.
    ValueError
        If a file lacks some of those columns: it was written by an older ``run_a1.py``, and the
        run must be redone with ``--restart``.
    """
    paths = sorted((Path(run_dir) / "predictions").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction files under {Path(run_dir) / 'predictions'}; run run_a1.py first.")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} has no {missing}: it was written by an older run_a1.py. Redo the run with --restart."
            )
        # One type per column in every file, so files of different kinds stack cleanly.
        frame = frame.astype(
            {
                "fraction": float,
                "repeat": float,
                "block_size_m": float,
                "local_fold": float,
                "local_block": float,
                "held_out": object,
            }
        )
        for column, wanted in (("kind", kinds), ("model", models), ("covariate_group", covariate_groups)):
            if wanted is not None:
                frame = frame[frame[column].isin(list(wanted))]
        frames.append(frame[list(PREDICTION_COLUMNS)])
    return pd.concat(frames, ignore_index=True)


# --- scores ----------------------------------------------------------------------------------------


def _point_scores(observed, predicted) -> dict[str, float]:
    """The six scores of one set of predictions, from ``regression_metrics``; NaN where undefined."""
    raw = regression_metrics(observed, predicted, split="x")
    return {score: raw.get(f"{score}_x", np.nan) for score in SCORES}


def _scored(frame: pd.DataFrame, columns=("observed", "predicted")) -> pd.DataFrame:
    """The rows that can be scored: a measurement and every prediction present."""
    keep = np.ones(len(frame), dtype=bool)
    for column in columns:
        keep &= np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    return frame[keep]


def _groups(frame: pd.DataFrame, by: Sequence[str]):
    """``(key tuple, rows)`` per group; a single group with key ``()`` when ``by`` is empty."""
    if not by:
        yield (), frame
        return
    for key, part in frame.groupby(list(by), sort=True, dropna=False, observed=True):
        yield (key if isinstance(key, tuple) else (key,)), part


def grouped_metrics(df: pd.DataFrame, by, *, min_n: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """RMSE, MAE, R², bias, RPD and RPIQ for every group of rows.

    Each group is scored with :func:`~yg_eo_soilnet.metrics.regression_metrics`, so the numbers are
    the ones MLflow and ``metrics.csv`` record. Rows without a measurement are left out.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows (:func:`load_predictions`), already narrowed to what should be pooled - one
        split, say.
    by : str or list of str
        The columns that define a group, e.g. ``["kind", "model", "carbonate_class_measured"]``.
    min_n : int, default :data:`MIN_GROUP_SIZE`
        Groups with fewer scored points are marked ``small``.

    Returns
    -------
    pandas.DataFrame
        One row per group: the ``by`` columns, ``n`` (points scored), ``n_fallback`` (of them,
        predicted by ``global_aux`` because their class had no model), the six scores and
        ``small``. A score that cannot be computed is NaN: R² when every measurement is the same,
        RPD and RPIQ when the predictions are perfect.

    Examples
    --------
    >>> rows = pd.DataFrame({
    ...     "model": ["a"] * 4 + ["b"] * 4,
    ...     "observed": [10.0, 20.0, 30.0, 40.0] * 2,
    ...     "predicted": [12.0, 18.0, 33.0, 39.0, 10.0, 20.0, 30.0, 40.0],
    ...     "fallback": False,
    ... })
    >>> table = grouped_metrics(rows, "model", min_n=3)
    >>> table[["model", "n", "rmse", "bias", "r2", "small"]].round(3).to_dict("records")
    [{'model': 'a', 'n': 4, 'rmse': 2.121, 'bias': 0.5, 'r2': 0.964, 'small': False}, {'model': 'b', 'n': 4, 'rmse': 0.0, 'bias': 0.0, 'r2': 1.0, 'small': False}]
    """
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, part in _groups(df, by):
        scored = _scored(part)
        rows.append(
            {
                **dict(zip(by, key)),
                "n": len(scored),
                "n_fallback": int(scored["fallback"].astype(bool).sum()) if "fallback" in scored else 0,
                **_point_scores(scored["observed"], scored["predicted"]),
                "small": len(scored) < min_n,
            }
        )
    return pd.DataFrame(rows, columns=[*by, "n", "n_fallback", *SCORES, "small"])


def _scores(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """The six scores of :data:`SCORES`, computed as ``regression_metrics`` computes them, quickly.

    ``regression_metrics`` takes a few milliseconds per call - fine once, too slow a thousand times
    per group inside the bootstrap. This repeats its formulas on plain arrays (a test holds the two
    equal): RMSE; MAE; bias = mean(predicted - observed); R² = 1 - SS_res / SS_tot, undefined when
    every measurement is the same; RPD = sd(observed, ddof=1) / RMSE and RPIQ = IQR(observed) /
    RMSE, both undefined when RMSE is 0. Fewer than two points: all undefined.
    """
    result = np.full(len(SCORES), np.nan)
    if observed.size < 2:
        return result
    error = predicted - observed
    squared = float(np.mean(error**2))
    rmse = float(np.sqrt(squared))
    result[0] = rmse
    result[1] = float(np.mean(np.abs(error)))
    result[3] = float(np.mean(error))
    variance = float(np.var(observed))
    if variance > 1e-12:
        result[2] = 1.0 - squared / variance
    if rmse > 0.0:
        result[4] = float(np.std(observed, ddof=1)) / rmse
        q75, q25 = np.percentile(observed, [75, 25])
        result[5] = float(q75 - q25) / rmse
    return result


# --- the bootstrap ---------------------------------------------------------------------------------


def spatial_blocks(df: pd.DataFrame, block_km: float = 10.0) -> pd.Series:
    """The spatial block of every row: a square of ``block_km`` on a side, laid over all the points.

    The squares come from :func:`~yg_eo_soilnet.utils.assign_grid_ids`, computed once over every
    distinct point of ``df``, so a point keeps its block in every plan, model and group. When all
    the points fit inside one square, they are all one block.

    Parameters
    ----------
    df : pandas.DataFrame
        With ``point_id``, ``lat`` and ``lon``.
    block_km : float, default 10

    Returns
    -------
    pandas.Series of int
        Indexed like ``df``.

    Raises
    ------
    ValueError
        If a point has no position, or two different ones.
    """
    positions = df[["point_id", "lat", "lon"]].drop_duplicates()
    if positions["point_id"].duplicated().any():
        moved = positions.loc[positions["point_id"].duplicated(), "point_id"].unique()[:3].tolist()
        raise ValueError(f"Some points have two positions, e.g. {moved}; point ids must name one place each.")
    points = positions.reset_index(drop=True)
    if points[["lat", "lon"]].isna().any().any():
        raise ValueError("Some points have no lat/lon; the spatial blocks need every point's position.")
    try:
        grid_id, _ = assign_grid_ids(points, cell_size_m=block_km * 1000.0)
        block_of = pd.Series(np.asarray(grid_id, dtype=int), index=points["point_id"].to_numpy())
    except ValueError as error:
        if "exceeds dataset extent" not in str(error):
            raise
        block_of = pd.Series(0, index=points["point_id"].to_numpy())
    return pd.Series(block_of.reindex(df["point_id"]).to_numpy(), index=df.index, name="block")


def _generator(seed: int, key: tuple) -> np.random.Generator:
    """A random generator for one group, the same whatever other groups there are."""
    return np.random.default_rng([int(seed), zlib.crc32("|".join(map(str, key)).encode())])


def _draws(blocks: np.ndarray, n: int, rng: np.random.Generator):
    """Yield ``n`` resamples, as row positions: whole blocks drawn with replacement, as many as there are."""
    labels, codes = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(codes == code) for code in range(len(labels))]
    for _ in range(n):
        chosen = rng.integers(0, len(labels), size=len(labels))
        yield np.concatenate([members[code] for code in chosen])


def _interval(values: np.ndarray, level: float, n_blocks: int) -> tuple[float, float, int]:
    """Percentile interval of the draws where the score is defined; NaN with fewer than two blocks."""
    finite = values[np.isfinite(values)]
    if n_blocks < 2 or finite.size == 0:
        return np.nan, np.nan, int(finite.size)
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return float(low), float(high), int(finite.size)


def _p_value(differences: np.ndarray) -> float:
    """Two-sided bootstrap p-value of "no difference": twice the smaller share of draws on either side of 0."""
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return np.nan
    return float(min(1.0, 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))))


def block_bootstrap(
    df: pd.DataFrame,
    by,
    *,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
    compare: Optional[Sequence[tuple[str, str]]] = None,
    compare_on: str = "model",
    level: float = 0.95,
    min_n: int = MIN_GROUP_SIZE,
) -> BootstrapResult:
    """Intervals for every score of every group, from resampling whole spatial blocks.

    Within each group, the group's blocks (:func:`spatial_blocks`) are drawn with replacement, as
    many as it has, and every row of a drawn block comes along each time it is drawn; the scores are
    computed on that resample, ``n`` times. The interval is the middle ``level`` of those scores.

    With ``compare``, two models (or covariate groups, with ``compare_on="covariate_group"``) are
    also compared on the points both predicted: each resample scores both on the same rows, so
    their shared luck cancels and only the difference between them varies. The p-value tests "no
    difference" two-sided, and ``p_holm`` corrects it for every comparison of this call taken
    together (one family per score, :func:`holm`).

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows (:func:`load_predictions`), narrowed to what should be pooled.
    by : str or list of str
        The columns that define a group.
    block_km : float, default 10
        The side of a block.
    n : int, default 1000
        Number of resamples.
    seed : int, default 42
        Each group draws from its own generator, seeded by ``seed`` and the group's name, so a
        group's interval does not change when other groups are added.
    compare : list of (str, str), optional
        Pairs ``(a, b)`` of values of ``compare_on``; the difference reported is score(a) -
        score(b).
    compare_on : str, default "model"
    level : float, default 0.95
    min_n : int, default :data:`MIN_GROUP_SIZE`
        Groups with fewer scored points are marked ``small``.

    Returns
    -------
    BootstrapResult
        Intervals with fewer than two blocks behind them are NaN: one block resampled is always
        itself.

    Raises
    ------
    ValueError
        If a compared value predicts a point more than once within a group - the pairing would be
        ambiguous; add the column that tells the rows apart (often ``covariate_group``) to ``by``.
    """
    by = [by] if isinstance(by, str) else list(by)
    blocks = spatial_blocks(df, block_km)
    scored = _scored(df.assign(block=blocks.to_numpy()))

    interval_rows = []
    for key, part in _groups(scored, by):
        observed = part["observed"].to_numpy(dtype=float)
        predicted = part["predicted"].to_numpy(dtype=float)
        part_blocks = part["block"].to_numpy()
        n_blocks = len(np.unique(part_blocks))
        draws = np.array(
            [_scores(observed[rows], predicted[rows]) for rows in _draws(part_blocks, n, _generator(seed, key))]
        )
        estimate = _point_scores(observed, predicted)
        for position, score in enumerate(SCORES):
            low, high, valid = _interval(draws[:, position], level, n_blocks)
            interval_rows.append(
                {
                    **dict(zip(by, key)),
                    "metric": score,
                    "estimate": estimate[score],
                    "low": low,
                    "high": high,
                    "n": len(part),
                    "n_blocks": n_blocks,
                    "n_valid_draws": valid,
                    "small": len(part) < min_n,
                }
            )
    intervals = pd.DataFrame(
        interval_rows, columns=[*by, "metric", "estimate", "low", "high", "n", "n_blocks", "n_valid_draws", "small"]
    )

    rest = [column for column in by if column != compare_on]
    difference_columns = [
        *rest,
        "a",
        "b",
        "metric",
        "estimate",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n",
        "n_blocks",
        "n_valid_draws",
        "n_unpaired",
        "small",
    ]
    difference_rows = []
    for a, b in compare or ():
        paired, unpaired = _pair(df.assign(block=blocks.to_numpy()), compare_on, a, b, rest)
        for key, part in _groups(paired, rest):
            observed = part["observed"].to_numpy(dtype=float)
            predicted_a = part["predicted_a"].to_numpy(dtype=float)
            predicted_b = part["predicted_b"].to_numpy(dtype=float)
            part_blocks = part["block"].to_numpy()
            n_blocks = len(np.unique(part_blocks))
            draws = np.array(
                [
                    _scores(observed[rows], predicted_a[rows]) - _scores(observed[rows], predicted_b[rows])
                    for rows in _draws(part_blocks, n, _generator(seed, key))
                ]
            )
            estimate_a = _point_scores(observed, predicted_a)
            estimate_b = _point_scores(observed, predicted_b)
            for position, score in enumerate(SCORES):
                low, high, valid = _interval(draws[:, position], level, n_blocks)
                difference_rows.append(
                    {
                        **dict(zip(rest, key)),
                        "a": a,
                        "b": b,
                        "metric": score,
                        "estimate": estimate_a[score] - estimate_b[score],
                        "low": low,
                        "high": high,
                        "p_value": _p_value(draws[:, position]) if n_blocks >= 2 else np.nan,
                        "n": len(part),
                        "n_blocks": n_blocks,
                        "n_valid_draws": valid,
                        "n_unpaired": unpaired.get(key, 0),
                        "small": len(part) < min_n,
                    }
                )
    differences = pd.DataFrame(difference_rows, columns=difference_columns)
    if not differences.empty:
        differences["p_holm"] = differences.groupby("metric", sort=False)["p_value"].transform(holm)
    return BootstrapResult(intervals=intervals, differences=differences)


def _pair(frame: pd.DataFrame, compare_on: str, a: str, b: str, rest: list[str]):
    """The rows of ``a`` and ``b`` side by side, one row per point both predicted; and the unpaired counts.

    Returns the paired, scorable rows (``observed, predicted_a, predicted_b, block`` and the
    keys), and, per group of ``rest``, how many rows of either side had no partner.
    """
    keys = [*PAIR_KEYS, *[column for column in rest if column not in PAIR_KEYS]]
    side_a = frame[frame[compare_on] == a]
    side_b = frame[frame[compare_on] == b]
    for name, side in ((a, side_a), (b, side_b)):
        if side.duplicated(keys).any():
            raise ValueError(
                f"{compare_on}={name!r} predicts some points more than once within a group of {rest}; add the "
                "column that tells those rows apart (often covariate_group) to `by`, or filter the rows first."
            )
    merged = side_a[[*keys, "observed", "predicted", "block"]].merge(
        side_b[[*keys, "predicted"]], on=keys, how="outer", suffixes=("_a", "_b"), indicator=True
    )
    lonely = merged[merged["_merge"] != "both"]
    unpaired = {key: len(part) for key, part in _groups(lonely, rest)} if not lonely.empty else {}
    paired = _scored(
        merged[merged["_merge"] == "both"].drop(columns="_merge"), ("observed", "predicted_a", "predicted_b")
    )
    return paired, unpaired


# --- many tests at once ----------------------------------------------------------------------------


def holm(pvalues):
    """Holm's correction of a family of p-values, so that reading them together stays honest.

    Testing many differences at the 5 % level finds some by chance alone. Holm's step-down method
    raises each p-value just enough that the chance of any false finding in the family stays at
    5 %: the smallest of ``m`` is multiplied by ``m``, the next by ``m - 1``, and so on, never
    decreasing along that order, and never above 1. Which tests form a family is the reader's
    choice - all comparisons in one table, typically.

    Parameters
    ----------
    pvalues : array-like or pandas.Series
        One p-value per test. Missing values stay missing and do not count as tests.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The corrected p-values, in the same order (and with the same index, for a Series).

    Raises
    ------
    ValueError
        If a p-value lies outside [0, 1].

    Examples
    --------
    >>> holm([0.01, 0.04, 0.03, 0.005]).round(3).tolist()
    [0.03, 0.06, 0.06, 0.02]
    """
    values = np.asarray(pvalues, dtype=float).reshape(-1)
    present = np.flatnonzero(~np.isnan(values))
    if ((values[present] < 0.0) | (values[present] > 1.0)).any():
        raise ValueError("p-values must lie between 0 and 1.")
    adjusted = np.full(values.shape, np.nan)
    m = present.size
    if m:
        order = present[np.argsort(values[present], kind="stable")]
        stepped = values[order] * (m - np.arange(m))
        adjusted[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    if isinstance(pvalues, pd.Series):
        return pd.Series(adjusted, index=pvalues.index, name=pvalues.name)
    return adjusted


# --- transfer --------------------------------------------------------------------------------------


def _drops(local: dict, transferred: dict) -> dict:
    """How much worse the transferred model is, per score: positive means worse."""
    drops = {}
    for score in SCORES:
        if score == "bias":
            continue
        gap = transferred[score] - local[score]
        drops[f"{score}_drop"] = -gap if score in HIGHER_IS_BETTER else gap
    drops["rmse_drop_pct"] = (
        100.0 * drops["rmse_drop"] / local["rmse"] if np.isfinite(local["rmse"]) and local["rmse"] > 0 else np.nan
    )
    return drops


def _transfer_row(transferred: pd.DataFrame, local: Optional[pd.DataFrame]) -> dict:
    """Both models scored on the points both predicted (or the transferred one alone, without a local)."""
    transferred = _scored(transferred)
    if local is None or local.empty:
        scores_transferred = _point_scores(transferred["observed"], transferred["predicted"])
        scores_local = dict.fromkeys(SCORES, np.nan)
        n, has_local = len(transferred), False
    else:
        paired = transferred[["point_id", "observed", "predicted"]].merge(
            _scored(local)[["point_id", "predicted"]], on="point_id", suffixes=("", "_local")
        )
        scores_transferred = _point_scores(paired["observed"], paired["predicted"])
        scores_local = _point_scores(paired["observed"], paired["predicted_local"])
        n, has_local = len(paired), True
    return {
        "n": n,
        "has_local": has_local,
        **{f"{score}_local": scores_local[score] for score in SCORES},
        **{f"{score}_transferred": scores_transferred[score] for score in SCORES},
        **_drops(scores_local, scores_transferred),
    }


def transfer_table(df: pd.DataFrame, *, transferred_model: str = GLOBAL_AUX) -> pd.DataFrame:
    """Local against transferred scores, per held-out region and covariate group, on test points.

    ``logo`` plans
        *Transferred* is ``transferred_model`` trained on the other regions; *local* is the
        ``local`` model, cross-validated inside the region on its own points. Both are scored on
        the points both predicted. The ``local`` model is only fitted by ``run_a1.py --groups``;
        without it the local columns are NaN.
    ``mixed`` plans
        *Transferred* is ``transferred_model`` trained on the other regions plus a share of this
        one; *local* is the matching ``control`` plan - the same share, the same repeat, trained on
        the moved points alone - scored on the same test points. Each share's repeats are averaged.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows (:func:`load_predictions`); rows of other kinds and of validation are
        ignored.
    transferred_model : str, default "global_aux"
        The model compared with the local one. ``local`` uses the same inputs as ``global_aux``.

    Returns
    -------
    pandas.DataFrame
        One row per region, covariate group and (for ``mixed``) share: ``kind, region,
        covariate_group, fraction, n_repeats, n, local_source, has_local``, then
        ``<score>_local`` and ``<score>_transferred`` for each score, then the drops -
        ``rmse_drop`` and ``mae_drop`` (transferred - local), ``r2_drop``, ``rpd_drop`` and
        ``rpiq_drop`` (local - transferred), so that a positive drop always means transfer does
        worse - ``rmse_drop_pct`` (the RMSE drop as a percentage of the local RMSE) and
        ``rmse_drop_sd`` (its spread over the repeats; NaN for ``logo``). A region without a local
        score keeps its row, with NaN local columns and ``has_local`` False.
    """
    test = df[(df["split"] == TEST) & (df["model"].isin([transferred_model, LOCAL]))]
    rows = []

    logo = test[test["kind"] == LOGO]
    for (_, group), part in logo.groupby(["plan_id", "covariate_group"], sort=True):
        transferred = part[part["model"] == transferred_model]
        if transferred.empty:
            continue
        rows.append(
            {
                "kind": LOGO,
                "region": transferred["held_out"].iloc[0],
                "covariate_group": group,
                "fraction": np.nan,
                "local_source": "local model",
                **_transfer_row(transferred, part[part["model"] == LOCAL]),
            }
        )

    shared = test[test["model"] == transferred_model].assign(fraction_key=lambda frame: frame["fraction"].round(6))
    match = ["held_out", "fraction_key", "repeat", "covariate_group"]
    controls = {key: part for key, part in shared[shared["kind"] == CONTROL].groupby(match, sort=False)}
    mixed_rows = []
    for key, transferred in shared[shared["kind"] == MIXED].groupby(match, sort=True):
        region, fraction, _, group = key
        mixed_rows.append(
            {
                "kind": MIXED,
                "region": region,
                "covariate_group": group,
                "fraction": float(fraction),
                "local_source": "control plan",
                **_transfer_row(transferred, controls.get(key)),
            }
        )
    if mixed_rows:
        per_repeat = pd.DataFrame(mixed_rows)
        keys = ["kind", "region", "covariate_group", "fraction", "local_source"]
        averaged = per_repeat.groupby(keys, sort=True).mean(numeric_only=True)
        averaged["has_local"] = per_repeat.groupby(keys, sort=True)["has_local"].all()
        averaged["n_repeats"] = per_repeat.groupby(keys, sort=True).size()
        averaged["rmse_drop_sd"] = per_repeat.groupby(keys, sort=True)["rmse_drop"].std(ddof=1)
        rows += averaged.reset_index().to_dict("records")

    score_columns = [f"{score}_{side}" for score in SCORES for side in ("local", "transferred")]
    drop_columns = [f"{score}_drop" for score in SCORES if score != "bias"] + ["rmse_drop_pct", "rmse_drop_sd"]
    columns = ["kind", "region", "covariate_group", "fraction", "n_repeats", "n", "local_source", "has_local"]
    table = pd.DataFrame(rows, columns=[*columns, *score_columns, *drop_columns])
    table.loc[table["kind"] == LOGO, "n_repeats"] = 1
    if not table.empty:
        table["n_repeats"] = table["n_repeats"].astype(int)
        table["has_local"] = table["has_local"].astype(bool)
    return table


# --- headline metrics, for MLflow ---------------------------------------------------------------

#: The scores logged as headline metrics.
HEADLINE_SCORES = ("rmse", "mae", "r2", "bias", "rpiq")
#: The scores whose bootstrap interval bounds are logged too (``__lo`` / ``__hi``).
HEADLINE_INTERVALS = ("rmse", "r2")
#: The covariate group whose name is left out of a headline metric's name: every covariate.
DEFAULT_GROUP = "combined"


def _pooled_scope(kind: str, block_size_m) -> str:
    """``random``, or ``spatial_kfold_<size>m`` for one block size (``spatial_kfold`` without one)."""
    if kind == SPATIAL_KFOLD and pd.notna(block_size_m):
        return f"{SPATIAL_KFOLD}_{float(block_size_m):g}m"
    return kind


def headline_metrics(
    frame: pd.DataFrame,
    *,
    intervals: Optional[pd.DataFrame] = None,
    region_intervals: Optional[pd.DataFrame] = None,
    transfer: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """The scores of a run worth a glance in MLflow, named ``<score>__<plans>__<model>[__<group>]``.

    On the test points only, for RMSE, MAE, R², bias and RPIQ:

    - ``random``: all its test points;
    - ``spatial_kfold_<size>m``: the test points of every fold of one block size, pooled - one value
      per size, never pooled across sizes;
    - ``logo_<region>``: one held-out region; and ``logo_mean``, the mean of the regions' scores.

    ``__<group>`` is added for a model fitted on one covariate group only, and left out for every
    covariate (``combined``). Also: the transfer drop per region and group
    (``rmse_drop__logo_<region>__<group>`` and ``rmse_drop__logo_mean__<group>``), and the
    bootstrap interval bounds of RMSE and R² (``...__lo`` / ``...__hi``) wherever ``intervals``
    (pooled: random and each block size) or ``region_intervals`` (each region) hold them. Names are
    MLflow-safe: lower-case letters, digits and underscores (:func:`~yg_eo_soilnet.a1.tracking.metric_name`).

    Parameters
    ----------
    frame : pandas.DataFrame
        Prediction rows (:func:`load_predictions`).
    intervals : pandas.DataFrame, optional
        :func:`block_bootstrap` intervals by ``kind, block_size_m, model``.
    region_intervals : pandas.DataFrame, optional
        Its intervals by ``kind, held_out, model``, on the ``logo`` rows.
    transfer : pandas.DataFrame, optional
        :func:`transfer_table`.

    Returns
    -------
    dict of str to float
        Undefined scores are left out.
    """
    from yg_eo_soilnet.a1.tracking import metric_name

    def group_part(group):
        return None if group == DEFAULT_GROUP else group

    metrics: dict[str, float] = {}

    def add(name: str, value) -> None:
        if value is not None and np.isfinite(value):
            metrics[name] = float(value)

    test = frame[frame["split"] == TEST]
    pooled = test[test["kind"].isin([RANDOM, SPATIAL_KFOLD])]
    if not pooled.empty:
        for row in grouped_metrics(pooled, ["kind", "block_size_m", "model", "covariate_group"], min_n=0).itertuples():
            scope = _pooled_scope(row.kind, row.block_size_m)
            for score in HEADLINE_SCORES:
                add(metric_name(score, scope, row.model, group_part(row.covariate_group)), getattr(row, score))

    logo = test[test["kind"] == LOGO]
    if not logo.empty:
        regions = grouped_metrics(logo, ["held_out", "model", "covariate_group"], min_n=0)
        for row in regions.itertuples():
            for score in HEADLINE_SCORES:
                add(
                    metric_name(score, f"logo_{row.held_out}", row.model, group_part(row.covariate_group)),
                    getattr(row, score),
                )
        means = regions.groupby(["model", "covariate_group"], sort=True)[list(HEADLINE_SCORES)].mean()
        for (model, group), row in means.iterrows():
            for score in HEADLINE_SCORES:
                add(metric_name(score, "logo_mean", model, group_part(group)), row[score])

    if transfer is not None and not transfer.empty:
        drops = transfer[(transfer["kind"] == LOGO) & transfer["has_local"]]
        for row in drops.itertuples():
            add(metric_name("rmse_drop", f"logo_{row.region}", row.covariate_group), row.rmse_drop)
        for group, values in drops.groupby("covariate_group")["rmse_drop"]:
            add(metric_name("rmse_drop", "logo_mean", group), values.mean())

    for table, scope_of in (
        (intervals, lambda row: _pooled_scope(row["kind"], row.get("block_size_m"))),
        (region_intervals, lambda row: f"logo_{row['held_out']}"),
    ):
        if table is None or table.empty:
            continue
        for _, row in table[table["metric"].isin(HEADLINE_INTERVALS)].iterrows():
            if row["kind"] not in (RANDOM, SPATIAL_KFOLD, LOGO) or (row["kind"] == LOGO and "held_out" not in row):
                continue
            for bound in ("low", "high"):
                suffix = "lo" if bound == "low" else "hi"
                add(metric_name(row["metric"], scope_of(row), row["model"], suffix), row[bound])
    return metrics


def region_names(frame: pd.DataFrame) -> dict[str, str]:
    """Each held-out region's name as it appears in metric names, mapped to its real name.

    Examples
    --------
    >>> region_names(pd.DataFrame({"kind": ["logo", "logo"], "held_out": ["Oum Er Rbia", "Guir - Ziz - Rhris"]}))
    {'guir_ziz_rhris': 'Guir - Ziz - Rhris', 'oum_er_rbia': 'Oum Er Rbia'}
    """
    names = frame.loc[frame["kind"] == LOGO, "held_out"].dropna().unique()
    return {slug(name): str(name) for name in sorted(names, key=slug)}
