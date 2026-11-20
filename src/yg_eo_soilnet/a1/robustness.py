"""Robustness checks of the transfer mechanism found in E1, from what the run saved; no model is fitted.

Leaving one basin out, the Global model fails to transfer to Oum Er Rbia and Sebou and transfers to
Tensift, Loukkous and Bouregreg; within measured carbonate classes, every significant basin-vs-rest
change in how SOM relates to a covariate lies in the two failing basins. Three checks:

:func:`subsample_relations` (the power check)
    Are the within-class differences found only in the large basins because only they have the
    points to show them? Each basin's class is cut down to the same number of points, by whole
    blocks, and the test repeated; a basin that keeps its differences at a size other basins
    have is genuinely different.
:func:`cell_table` and :func:`cell_correlations`
    Do the transfer errors follow the differences, cell by cell (basin x measured carbonate
    class)? The bias against the shift in SOM level, the loss against the change in relations and
    against the input dissimilarity - 20 cells from 5 basins, exploratory.
:func:`correction_draws`
    How much of the loss is a shift that a few local samples would correct, without refitting:
    an offset, a linear recalibration and an offset per class, learnt from k basin points.

Every check names the local model it compares with (``local_model``): the local model on Global's
inputs (``local_global``) when the run has it, otherwise the one with predicted CaCO3/CEC
(``local``), with a warning (:func:`choose_local_model`).
"""

from __future__ import annotations

import logging
import zlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, LOCAL_GLOBAL

logger = logging.getLogger(__name__)

#: The local models a check can compare with, the preferred one first.
LOCAL_MODEL_CHOICES = (LOCAL_GLOBAL, LOCAL)
#: How each local model is named in a caption.
LOCAL_MODEL_TEXT = {
    LOCAL_GLOBAL: "local model on Global's inputs",
    LOCAL: "local model with predicted CaCO3/CEC",
}
#: Significance level of a Holm-corrected p-value.
ALPHA = 0.05
#: The sizes of the power check: about Tensift's and Sebou's smallest class, and about Bouregreg's.
SUBSAMPLE_SIZES = (240, 55)
#: The numbers of local samples of the correction check.
K_VALUES = (10, 25, 50, 100, 200, 400)
#: A k is used only when at least this many of the basin's points remain to test on.
MIN_TEST_POINTS = 100
#: The linear correction needs at least this many samples.
MIN_LINEAR = 25
#: A class needs this many samples for an offset of its own; fewer, and the overall offset is used.
MIN_CLASS_SAMPLES = 5
#: The methods of the correction check, in the order the tables and figure show them.
METHODS = ("uncorrected", "offset", "linear", "per_class_offset")
#: The columns of every draw of the power check (:func:`subsample_relations`).
SUBSAMPLE_COLUMNS = (
    "n_sub",
    "draw",
    "basin",
    "class",
    "variable",
    "difference",
    "p_value",
    "p_holm",
    "significant",
    "p_method",
)
#: The pairs correlated across the cells: (y, x), with what each tests.
CELL_PAIRS = (
    ("bias_transferred", "level_shift"),
    ("loss", "relationship_change"),
    ("loss", "median_di"),
    ("debiased_minus_local", "relationship_change"),
)


def _key(text: str) -> int:
    """A stable integer for a name (a basin), for seeding."""
    return zlib.crc32(str(text).encode())


# --- the local model -------------------------------------------------------------------------------


def choose_local_model(present: Sequence[str], wanted: str = LOCAL_GLOBAL, *, log=None) -> str:
    """The local model a check compares with: ``wanted`` when the run has it, else ``local``, with a warning.

    Raises
    ------
    ValueError
        When the run has neither (or ``wanted`` is not a local model).
    """
    log = log or logger
    if wanted not in LOCAL_MODEL_CHOICES:
        raise ValueError(f"--local-model must be one of {LOCAL_MODEL_CHOICES}, not {wanted!r}.")
    present = set(present)
    if wanted in present:
        return wanted
    if wanted == LOCAL_GLOBAL and LOCAL in present:
        log.warning(
            f"No {LOCAL_GLOBAL} predictions in this run (the local model on Global's inputs); using {LOCAL}, the "
            "local model with predicted CaCO3/CEC. Fit local_global (run_a1.py resumes) and run this again."
        )
        return LOCAL
    raise ValueError(f"The run has no {wanted} predictions" + (f" and no {LOCAL} either." if wanted != LOCAL else "."))


def local_text(model: str) -> str:
    """How a caption names the local model a check used."""
    return LOCAL_MODEL_TEXT.get(model, model)


# --- sampling whole blocks -------------------------------------------------------------------------


def block_sample(ids: pd.Index, blocks: pd.Series, n: int, rng: np.random.Generator) -> tuple[pd.Index, pd.Index]:
    """Exactly ``n`` of ``ids``, taken by whole spatial blocks, and the unused rest of the last block.

    The blocks are taken in random order until ``n`` is reached or passed; random points of the
    last block are then dropped to make exactly ``n``. Those dropped points share a block with the
    sample, so they are returned apart (``leftover``) and are neither sampled nor tested.

    Parameters
    ----------
    ids : pandas.Index
        The points to sample from.
    blocks : pandas.Series
        Each point's block (indexed by point id; covers ``ids``).
    n : int
    rng : numpy.random.Generator

    Returns
    -------
    chosen, leftover : pandas.Index

    Raises
    ------
    ValueError
        If ``ids`` holds fewer than ``n`` points.

    Examples
    --------
    >>> ids = pd.Index(list("abcdef"))
    >>> blocks = pd.Series([1, 1, 2, 2, 3, 3], index=ids)
    >>> chosen, leftover = block_sample(ids, blocks, 3, np.random.default_rng(0))
    >>> len(chosen), len(leftover)
    (3, 1)
    """
    ids = pd.Index(sorted(ids))
    if len(ids) < n:
        raise ValueError(f"{len(ids)} point(s) to sample from, fewer than {n}.")
    of = blocks.reindex(ids).to_numpy()
    labels = np.unique(of)
    chosen: list = []
    leftover: list = []
    for label in rng.permutation(labels):
        members = list(ids[of == label])
        missing = n - len(chosen)
        if len(members) >= missing:
            kept = set(rng.choice(len(members), size=missing, replace=False).tolist())
            chosen += [member for position, member in enumerate(members) if position in kept]
            leftover = [member for position, member in enumerate(members) if position not in kept]
            break
        chosen += members
    return pd.Index(chosen), pd.Index(leftover)


# --- 1. the power check ------------------------------------------------------------------------------


def subsample_relations(
    data,
    relations: pd.DataFrame,
    *,
    n_sub: int,
    n_draws: int = 100,
    n_boot: int = 300,
    seed: int = 42,
    min_cell_size: int = 30,
    p_method: str = "normal",
) -> pd.DataFrame:
    """Every draw of the power check at one size: per basin, class and variable, the difference and its p_Holm.

    For each basin, every measured class holding at least ``n_sub`` basin points (and
    ``min_cell_size`` in the complement) is cut to ``n_sub`` points by whole blocks
    (:func:`block_sample`; the rest of the partly used block is not used); the complement is kept
    whole. SOM's Spearman rho with every variable is recomputed on the subsample, and the
    difference from the complement tested from ``n_boot`` block resamples of each side. The
    p-values are Holm-corrected within the basin over the family of :mod:`.basin_relations`: the
    within-class tests of the draw, plus the basin's all-point exploratory tests, whose p-values are
    kept from ``relations`` - so the family has the size it had, and only the subsample varies.
    ``p_method`` is how each p-value is computed
    (:func:`~yg_eo_soilnet.a1.basin_relations.difference_p_value`); ``relations`` should have been
    computed the same way (a warning says so otherwise).

    Parameters
    ----------
    data : yg_eo_soilnet.a1.basin_relations.BasinData
    relations : pandas.DataFrame
        ``basin_relations.csv``.
    n_sub, n_draws, n_boot, seed, min_cell_size : int
        Each draw is seeded by ``[seed, draw, basin]``.
    p_method : {"normal", "draws"}

    Returns
    -------
    pandas.DataFrame
        ``n_sub, draw, basin, class, variable, difference, p_value, p_holm, significant, p_method``;
        a basin with no class large enough has no rows.
    """
    from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES, EXPLORATORY, P_METHODS, _side, difference_p_value

    if p_method not in P_METHODS:
        raise ValueError(f"p_method {p_method!r}: expected one of {P_METHODS}")
    if "p_method" in relations and set(relations["p_method"].dropna()) - {p_method}:
        logger.warning(
            f"basin_relations.csv was computed with p_method {sorted(set(relations['p_method'].dropna()))}, the "
            f"power check with {p_method!r}: the family mixes the two."
        )

    rows = []
    for basin, ids in data.basins.items():
        complement = data.frame.index.difference(ids)
        cells = []
        for label in data.labels:
            in_class = data.frame.index[(data.classes.reindex(data.frame.index) == label).to_numpy()]
            basin_ids, rest_ids = ids.intersection(in_class), complement.intersection(in_class)
            if len(basin_ids) >= max(n_sub, min_cell_size) and len(rest_ids) >= min_cell_size:
                rho_c, draws_c = _side(data, rest_ids, n_boot, seed, (basin, label, "complement"))
                cells.append((label, basin_ids, rho_c, draws_c))
        if not cells:
            continue
        fixed = relations[
            (relations["basin"] == basin) & (relations["class"] == ALL_CLASSES) & (relations["family"] == EXPLORATORY)
        ]["p_value"].to_numpy(float)
        for draw in range(n_draws):
            rng = np.random.default_rng([int(seed), draw, _key(basin)])
            found = []
            for label, basin_ids, rho_c, draws_c in cells:
                chosen, _ = block_sample(basin_ids, data.blocks, n_sub, rng)
                rho_b, draws_b = _side(data, chosen, n_boot, seed, (basin, label, "basin", n_sub, draw))
                for position, variable in enumerate(data.pairs):
                    found.append(
                        {
                            "n_sub": n_sub,
                            "draw": draw,
                            "basin": basin,
                            "class": label,
                            "variable": variable,
                            "difference": rho_b[position] - rho_c[position],
                            "p_value": difference_p_value(
                                rho_b[position], rho_c[position], draws_b[:, position], draws_c[:, position], p_method
                            ),
                            "p_method": p_method,
                        }
                    )
            p_values = np.array([row["p_value"] for row in found] + list(fixed), dtype=float)
            adjusted = np.asarray(evaluation.holm(p_values), dtype=float)[: len(found)]
            for row, p_holm in zip(found, adjusted):
                row["p_holm"] = p_holm
                row["significant"] = bool(np.isfinite(p_holm) and p_holm < ALPHA)
            rows += found
    return pd.DataFrame(rows, columns=list(SUBSAMPLE_COLUMNS))


def subsample_summary(draws: pd.DataFrame, basins: Sequence[str], sizes: Sequence[int]) -> pd.DataFrame:
    """``basin_relations_subsample.csv``: the power check summarised.

    One ``level = basin`` row per basin and size - ``testable`` (whether any class was large
    enough), the classes tested, the mean number of significant within-class differences per draw,
    its 5-95 % range over the draws, and the share of draws with at least one - then one ``level =
    cell`` row per basin, size, class and variable: the share of draws in which that cell is
    significant, and its mean difference.
    """
    rows = []
    for n_sub in sizes:
        for basin in basins:
            part = draws[(draws["n_sub"] == n_sub) & (draws["basin"] == basin)]
            if part.empty:
                rows.append({"level": "basin", "basin": basin, "n_sub": n_sub, "testable": False, "n_classes": 0})
                continue
            counts = part.groupby("draw")["significant"].sum()
            rows.append(
                {
                    "level": "basin",
                    "basin": basin,
                    "n_sub": n_sub,
                    "testable": True,
                    "n_classes": int(part["class"].nunique()),
                    "n_draws": int(counts.size),
                    "mean_significant": float(counts.mean()),
                    "q05_significant": float(np.percentile(counts, 5)),
                    "q95_significant": float(np.percentile(counts, 95)),
                    "share_any": float((counts >= 1).mean()),
                }
            )
            for (label, variable), cell in part.groupby(["class", "variable"], sort=False):
                rows.append(
                    {
                        "level": "cell",
                        "basin": basin,
                        "n_sub": n_sub,
                        "testable": True,
                        "class": label,
                        "variable": variable,
                        "n_draws": int(cell["draw"].nunique()),
                        "share_significant": float(cell["significant"].mean()),
                        "mean_difference": float(cell["difference"].mean()),
                    }
                )
    columns = [
        "level",
        "basin",
        "n_sub",
        "testable",
        "n_classes",
        "class",
        "variable",
        "n_draws",
        "mean_significant",
        "q05_significant",
        "q95_significant",
        "share_any",
        "share_significant",
        "mean_difference",
    ]
    return pd.DataFrame(rows, columns=columns)


# --- predictions of the plain leave-one-basin-out plans ---------------------------------------------


def logo_test_rows(frame: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """The scored test rows of the plain ``logo`` plans, every input, of these models."""
    from yg_eo_soilnet.a1.config import COMBINED_GROUP
    from yg_eo_soilnet.a1.plans import LOGO, is_matched
    from yg_eo_soilnet.datamodules.splitting import TEST

    keep = (
        (frame["kind"] == LOGO)
        & ~is_matched(frame["plan_id"])
        & (frame["split"] == TEST)
        & (frame["covariate_group"] == COMBINED_GROUP)
        & frame["model"].isin(list(models))
    )
    rows = frame.loc[keep]
    return evaluation._scored(rows).reset_index(drop=True)


def paired(rows: pd.DataFrame, local_model: str, *, classed_only: bool = True) -> pd.DataFrame:
    """Global and the local model side by side, one row per seed, basin and point both predicted.

    Columns: ``seed, basin, plan_id, point_id, lat, lon, class, observed, predicted, local``. Points
    without a measured carbonate class belong to no cell: left out, or with ``classed_only=False``
    kept with an empty ``class``.
    """
    keys = ["seed", "plan_id", "point_id"]
    transferred = rows[rows["model"] == GLOBAL]
    local = rows[rows["model"] == local_model][[*keys, "predicted"]].rename(columns={"predicted": "local"})
    both = transferred.merge(local, on=keys, how="inner")
    if classed_only:
        both = both[both["carbonate_class_measured"].notna()]
    both = both.rename(columns={"held_out": "basin", "carbonate_class_measured": "class"})
    both["basin"] = both["basin"].astype(str)
    both["class"] = both["class"].astype(object).where(both["class"].notna(), "").astype(str)
    return both[["seed", "basin", "plan_id", "point_id", "lat", "lon", "class", "observed", "predicted", "local"]]


def _rmse(error) -> float:
    error = np.asarray(error, dtype=float)
    return float(np.sqrt(np.mean(error**2))) if error.size else np.nan


# --- 2. the cells ----------------------------------------------------------------------------------


def cell_errors(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per seed, basin and measured class: Global's bias, RMSE and debiased RMSE, the local RMSE, the loss.

    Both models are scored on the same points (``pairs``, :func:`paired`). ``loss`` is transferred
    minus local RMSE; ``debiased_minus_local`` is the loss once Global's mean offset in the cell is
    removed, sqrt(RMSE² - bias²) - local RMSE.
    """
    rows = []
    for (seed, basin, label), part in pairs.groupby(["seed", "basin", "class"], sort=True):
        error = (part["predicted"] - part["observed"]).to_numpy(float)
        local_error = (part["local"] - part["observed"]).to_numpy(float)
        rmse_t, rmse_l = _rmse(error), _rmse(local_error)
        bias = float(error.mean())
        debiased = float(np.std(error))
        rows.append(
            {
                "seed": int(seed),
                "basin": basin,
                "class": label,
                "n": len(part),
                "bias_transferred": bias,
                "rmse_transferred": rmse_t,
                "rmse_local": rmse_l,
                "loss": rmse_t - rmse_l,
                "rmse_debiased_transferred": debiased,
                "debiased_minus_local": debiased - rmse_l,
            }
        )
    return pd.DataFrame(rows)


def level_shifts(data) -> pd.DataFrame:
    """Per basin and measured class: the cell's median SOM minus that of the same class in every other point."""
    rows = []
    classes = data.classes.reindex(data.frame.index)
    som = data.frame[data.target]
    for basin, ids in data.basins.items():
        in_basin = data.frame.index.isin(ids)
        for label in data.labels:
            in_class = (classes == label).to_numpy()
            inside, outside = som[in_basin & in_class], som[~in_basin & in_class]
            rows.append(
                {
                    "basin": basin,
                    "class": label,
                    "som_median_basin": float(inside.median()) if len(inside) else np.nan,
                    "som_median_complement": float(outside.median()) if len(outside) else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table["level_shift"] = table["som_median_basin"] - table["som_median_complement"]
    return table


def som_profiles(data) -> pd.DataFrame:
    """Per basin, side (basin, complement) and measured class: n, median SOM and its quartiles (figure C5)."""
    rows = []
    classes = data.classes.reindex(data.frame.index)
    som = data.frame[data.target]
    for basin, ids in data.basins.items():
        in_basin = data.frame.index.isin(ids)
        for side, mask in (("basin", in_basin), ("complement", ~in_basin)):
            for label in data.labels:
                values = som[mask & (classes == label).to_numpy()]
                rows.append(
                    {
                        "basin": basin,
                        "side": side,
                        "class": label,
                        "n": len(values),
                        "som_median": float(values.median()) if len(values) else np.nan,
                        "som_q25": float(values.quantile(0.25)) if len(values) else np.nan,
                        "som_q75": float(values.quantile(0.75)) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def relationship_changes(relations: pd.DataFrame, primary: Sequence[str]) -> pd.DataFrame:
    """Per basin and measured class, from ``basin_relations.csv``: how SOM's relations differ from the rest.

    ``relationship_change`` is the mean |rho difference| over the covariates (CaCO3 and CEC left
    out), ``n_significant`` how many of those have p_Holm < 0.05, and
    ``n_significant_with_caco3_cec`` the same count with SOM-CaCO3 and SOM-CEC included.
    """
    from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES

    within = relations[relations["class"] != ALL_CLASSES]
    rows = []
    for (basin, label), part in within.groupby(["basin", "class"], sort=False):
        covariates = part[~part["variable"].isin(list(primary))]
        rows.append(
            {
                "basin": basin,
                "class": label,
                "n_covariates": len(covariates),
                "relationship_change": float(covariates["difference"].abs().mean()),
                "n_significant": int((covariates["p_holm"] < ALPHA).sum()),
                "n_significant_with_caco3_cec": int((part["p_holm"] < ALPHA).sum()),
            }
        )
    return pd.DataFrame(rows)


def cell_dissimilarity(dissimilarity: pd.DataFrame) -> pd.DataFrame:
    """Per held-out basin and measured class, of its test points: median dissimilarity index, share outside the AoA."""
    from yg_eo_soilnet.datamodules.splitting import TEST

    test = dissimilarity[(dissimilarity["split"] == TEST) & dissimilarity["carbonate_class"].notna()]
    grouped = test.groupby([test["held_out"].astype(str), test["carbonate_class"].astype(str)], sort=True)
    table = pd.DataFrame(
        {"median_di": grouped["di"].median(), "share_outside": 1.0 - grouped["inside"].mean()}
    ).reset_index()
    return table.rename(columns={"held_out": "basin", "carbonate_class": "class"})


def cell_table(
    pairs: pd.DataFrame,
    shifts: pd.DataFrame,
    changes: pd.DataFrame,
    dissimilar: pd.DataFrame,
    *,
    local_model: str,
    first_seed: int,
) -> pd.DataFrame:
    """``basin_cells.csv``: one row per seed (and ``mean`` over the seeds), basin and measured class.

    The errors of :func:`cell_errors`, the level shift (:func:`level_shifts`), the relationship
    change (:func:`relationship_changes`) and the input dissimilarity (:func:`cell_dissimilarity`);
    ``local_model`` names the local model compared with. The ``seed`` column is text: each seed,
    then ``mean``; the first seed's rows come first.
    """
    errors = cell_errors(pairs)
    keys = ["basin", "class"]
    if errors.empty:
        return pd.DataFrame()
    numeric = [column for column in errors.columns if column not in ("seed", "basin", "class")]
    mean = errors.groupby(keys, sort=True)[numeric].mean().reset_index().assign(seed="mean")
    seeds = sorted(errors["seed"].unique(), key=lambda value: (value != first_seed, value))
    per_seed = [errors[errors["seed"] == value].assign(seed=str(value)) for value in seeds]
    table = pd.concat([*per_seed, mean], ignore_index=True)
    for extra in (shifts, changes, dissimilar):
        table = table.merge(extra, on=keys, how="left")
    table.insert(0, "local_model", local_model)
    return table


def _spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def cell_correlations(
    cells: pd.DataFrame, *, n_draws: int = 1000, seed: int = 42, pairs=CELL_PAIRS, min_basins: int = 3
) -> pd.DataFrame:
    """``basin_cells_correlations.csv``: Spearman rho across the cells, with a bootstrap over basins.

    Each draw takes as many basins as there are, with replacement, and every cell of each (a basin
    drawn twice counts twice). A draw with fewer than ``min_basins`` distinct basins, or a constant
    variable, gives no value and is skipped; ``share_skipped`` reports how many. With 20 cells from
    5 basins this is exploratory.
    """
    rows = []
    for scope, part in cells.groupby("seed", sort=False):
        basins = sorted(part["basin"].unique())
        by_basin = {basin: part[part["basin"] == basin] for basin in basins}
        rng = np.random.default_rng([int(seed), _key(str(scope))])
        picks = [rng.integers(0, len(basins), size=len(basins)) for _ in range(n_draws)]
        for y, x in pairs:
            rho = _spearman(part[x], part[y])
            values = []
            for pick in picks:
                if len(set(pick.tolist())) < min_basins:
                    values.append(np.nan)
                    continue
                drawn = pd.concat([by_basin[basins[position]] for position in pick])
                values.append(_spearman(drawn[x], drawn[y]))
            values = np.array(values, dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "local_model": part["local_model"].iloc[0] if "local_model" in part else "",
                    "seed": scope,
                    "y": y,
                    "x": x,
                    "rho": rho,
                    "low": float(np.percentile(finite, 2.5)) if finite.size else np.nan,
                    "high": float(np.percentile(finite, 97.5)) if finite.size else np.nan,
                    "n_cells": int(part[[x, y]].dropna().shape[0]),
                    "n_basins": len(basins),
                    "n_draws": n_draws,
                    "share_skipped": float(1.0 - finite.size / n_draws) if n_draws else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --- 3. correcting with k local samples ------------------------------------------------------------


def offset_correction(sample: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The test predictions minus the mean residual (predicted - measured) of the samples."""
    offset = float((sample["predicted"] - sample["observed"]).mean())
    return test["predicted"].to_numpy(float) - offset


def linear_fit(sample: pd.DataFrame) -> tuple[float, float]:
    """``a, b`` of measured = a + b * predicted, least squares on the samples; NaN when the predictions are constant."""
    x, y = sample["predicted"].to_numpy(float), sample["observed"].to_numpy(float)
    if x.size < 2 or np.ptp(x) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def linear_correction(sample: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The test predictions recalibrated by :func:`linear_fit` of the samples."""
    a, b = linear_fit(sample)
    return a + b * test["predicted"].to_numpy(float)


def class_offset_correction(sample: pd.DataFrame, test: pd.DataFrame, *, min_samples: int = MIN_CLASS_SAMPLES):
    """One mean residual per measured class (a diagnostic: it needs each test point's measured class).

    A class with fewer than ``min_samples`` samples takes the overall offset, as does a point
    without a measured class (``class`` empty).
    """
    residual = sample["predicted"] - sample["observed"]
    overall = float(residual.mean())
    counts = sample.groupby("class")["observed"].size()
    offsets = residual.groupby(sample["class"]).mean()
    per_point = (
        test["class"]
        .map(lambda label: offsets[label] if label and counts.get(label, 0) >= min_samples else overall)
        .to_numpy(float)
    )
    return test["predicted"].to_numpy(float) - per_point


def one_draw(part: pd.DataFrame, blocks: pd.Series, *, basin: str, k: int, draw: int, seed: int = 42):
    """One draw of the correction check: its k samples, and its test points with every correction.

    ``part`` is one basin's :func:`paired` rows indexed by point id. Seeded by
    ``[seed, draw, basin, k]``, so a draw can be rebuilt exactly (figure C9 does).

    Returns
    -------
    test : pandas.DataFrame
        The test points: ``point_id, class, observed``, a column per method of :data:`METHODS`
        (the corrected predictions) and ``local``.
    info : dict
        ``n_test, n_set_aside, offset, a, b``.
    """
    rng = np.random.default_rng([int(seed), draw, _key(basin), int(k)])
    chosen, leftover = block_sample(part.index, blocks, k, rng)
    sample = part.loc[chosen]
    test = part.loc[part.index.difference(chosen).difference(leftover)]
    a, b = linear_fit(sample) if k >= MIN_LINEAR else (np.nan, np.nan)
    frame = pd.DataFrame(
        {
            "point_id": test.index,
            "class": test["class"].to_numpy(),
            "observed": test["observed"].to_numpy(float),
            "uncorrected": test["predicted"].to_numpy(float),
            "offset": offset_correction(sample, test),
            "linear": a + b * test["predicted"].to_numpy(float),
            "per_class_offset": class_offset_correction(sample, test),
            "local": test["local"].to_numpy(float),
        }
    )
    info = {
        "n_test": len(test),
        "n_set_aside": len(leftover),
        "offset": float((sample["predicted"] - sample["observed"]).mean()),
        "a": a,
        "b": b,
    }
    return frame, info


def calibration(observed, predicted) -> dict:
    """``slope, intercept`` of observed ~ a + b * predicted (least squares), ``bias`` (predicted - observed), ``rmse``.

    >>> calibration([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    {'slope': 1.0, 'intercept': -1.0, 'bias': 1.0, 'rmse': 1.0}
    """
    x, y = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.ptp(x) == 0:
        slope = intercept = np.nan
    else:
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    error = x - y
    return {
        "slope": slope,
        "intercept": intercept,
        "bias": float(error.mean()) if error.size else np.nan,
        "rmse": _rmse(error),
    }


#: The predictions figure C9 and the calibration table read: the corrections and the local model.
CALIBRATED = (*METHODS, "local")


def correction_draws(
    pairs: pd.DataFrame,
    blocks: pd.Series,
    *,
    ks: Sequence[int] = K_VALUES,
    n_draws: int = 100,
    seed: int = 42,
    min_test: int = MIN_TEST_POINTS,
) -> pd.DataFrame:
    """Every draw of the correction check: per basin and k, the RMSE of each correction and of the local model.

    For each held-out basin (one seed's ``pairs``, :func:`paired`) and each k leaving at least
    ``min_test`` points, ``n_draws`` times (:func:`one_draw`): k basin points are drawn by whole
    blocks (:func:`block_sample`) as local samples; the rest of the partly used block is set aside;
    every other basin point is a test point. On those same test points: the uncorrected Global
    predictions, the :func:`offset_correction`, the :func:`linear_correction` (k >= 25), the
    :func:`class_offset_correction`, and the local model.

    Returns
    -------
    pandas.DataFrame
        ``basin, k, draw, n_test, n_set_aside``, then for each of :data:`CALIBRATED`
        ``rmse_<m>, bias_<m>, slope_<m>, intercept_<m>`` (observed on predicted), then ``offset, a, b``.
    """
    rows = []
    for basin, part in pairs.groupby("basin", sort=True):
        part = part.set_index("point_id", drop=False)
        for k in ks:
            if len(part) - k < min_test:
                continue
            for draw in range(n_draws):
                test, info = one_draw(part, blocks, basin=basin, k=k, draw=draw, seed=seed)
                row = {
                    "basin": basin,
                    "k": int(k),
                    "draw": draw,
                    "n_test": info["n_test"],
                    "n_set_aside": info["n_set_aside"],
                }
                for method in CALIBRATED:
                    scores = calibration(test["observed"], test[method])
                    for name, value in scores.items():
                        row[f"{name}_{method}"] = value
                rows.append({**row, "offset": info["offset"], "a": info["a"], "b": info["b"]})
    return pd.DataFrame(rows)


def calibration_summary(draws: pd.DataFrame, *, local_model: str) -> pd.DataFrame:
    """``bias_correction_calibration.csv``: per basin, k and method (and the local model), the slope and
    intercept of observed on predicted, the bias and the RMSE - each the mean and 5-95 % over the draws,
    on each draw's own test points."""
    rows = []
    for (basin, k), part in draws.groupby(["basin", "k"], sort=True):
        for method in CALIBRATED:
            row = {"local_model": local_model, "basin": basin, "k": int(k), "method": method, "n_draws": len(part)}
            for name in ("slope", "intercept", "bias", "rmse"):
                values = part[f"{name}_{method}"].to_numpy(float)
                finite = values[np.isfinite(values)]
                row[name] = float(finite.mean()) if finite.size else np.nan
                row[f"{name}_q05"] = float(np.percentile(finite, 5)) if finite.size else np.nan
                row[f"{name}_q95"] = float(np.percentile(finite, 95)) if finite.size else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def median_draw(draws: pd.DataFrame, basin: str, k: int, method: str = "offset") -> Optional[int]:
    """The draw whose RMSE of ``method`` is the median over the draws of one basin and k (the lower
    middle one for an even count): figure C9 shows that one draw. None when there is none."""
    part = draws[(draws["basin"] == basin) & (draws["k"] == k)].dropna(subset=[f"rmse_{method}"])
    if part.empty:
        return None
    ordered = part.sort_values([f"rmse_{method}", "draw"])
    return int(ordered["draw"].iloc[(len(ordered) - 1) // 2])


def correction_summary(draws: pd.DataFrame, pairs: pd.DataFrame, *, local_model: str) -> pd.DataFrame:
    """``bias_correction.csv``: per basin and k, each RMSE's mean and 5-95 % range over the draws.

    Beside them, the transferred RMSE on every point of the basin (no correction, no sample set
    aside) and ``local_model``.
    """
    rows = []
    whole = {basin: _rmse(part["predicted"] - part["observed"]) for basin, part in pairs.groupby("basin", sort=True)}
    for (basin, k), part in draws.groupby(["basin", "k"], sort=True):
        row = {
            "local_model": local_model,
            "basin": basin,
            "k": int(k),
            "n_draws": len(part),
            "n_test_mean": float(part["n_test"].mean()),
            "rmse_transferred_all": whole.get(basin, np.nan),
        }
        for method in (*METHODS, "local"):
            values = part[f"rmse_{method}"].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"rmse_{method}"] = float(finite.mean()) if finite.size else np.nan
            row[f"rmse_{method}_q05"] = float(np.percentile(finite, 5)) if finite.size else np.nan
            row[f"rmse_{method}_q95"] = float(np.percentile(finite, 95)) if finite.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def e2_curves(arms: pd.DataFrame, basins: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The E2 retraining curves (``mixed_arms`` rows): per basin and share, mean moved points and RMSE.

    ``basin, fraction, n_moved, rmse_control, rmse_mixed`` (means over the repeats). These test the
    basin's remaining points of each mixed plan - not the test points of the correction draws.
    """
    columns = ["basin", "fraction", "n_moved", "rmse_control", "rmse_mixed", "n_repeats"]
    if arms is None or arms.empty:
        return pd.DataFrame(columns=columns)
    part = arms if basins is None else arms[arms["basin"].isin(list(basins))]
    grouped = part.groupby(["basin", "fraction"], sort=True)
    table = grouped[["n_moved", "rmse_control", "rmse_mixed"]].mean()
    table["n_repeats"] = grouped.size()
    return table.reset_index()[columns]


def read_e2_curves(run_dir, basins: Sequence[str]) -> pd.DataFrame:
    """:func:`e2_curves` of an E2 run folder (``results/a1_mixed_arms/runs/default``); empty when it is not there."""
    from pathlib import Path

    from yg_eo_soilnet.a1.compare import mixed_arms, read_run
    from yg_eo_soilnet.a1.plans import CONTROL, MIXED

    if run_dir is None or not (Path(run_dir) / "manifest.json").exists():
        return e2_curves(pd.DataFrame())
    run = read_run(run_dir, model=GLOBAL, kinds=[MIXED, CONTROL, "logo"])
    return e2_curves(mixed_arms([run]), basins)


# --- which basins fail -----------------------------------------------------------------------------


def failing_basins(run_dir, local_model: str, *, block_km: float = 10.0, n: int = 1000, seed: int = 42) -> list:
    """The basins whose transfer fails (:func:`~yg_eo_soilnet.a1.compare.transfer_status`), against ``local_model``."""
    from yg_eo_soilnet.a1.compare import read_run, transfer_status
    from yg_eo_soilnet.a1.plans import LOGO

    run = read_run(run_dir, model=GLOBAL, kinds=[LOGO], local=local_model)
    if local_model not in set(run.predictions["model"]):
        return []
    status = transfer_status(run, model=GLOBAL, block_km=block_km, n=n, seed=seed, local=local_model)
    return list(status.loc[status["transfer_fails"], "basin"].astype(str))


# --- files ---------------------------------------------------------------------------------------------


def models_in_run(run_dir) -> set:
    """The models a run folder has prediction files of (``predictions/<plan>/seed_<s>/<model>__<group>.parquet``)."""
    from pathlib import Path

    return {path.name.split("__")[0] for path in Path(run_dir).glob("predictions/*/seed_*/*__*.parquet")}


def save_figures(figures: dict, out) -> list:
    """Save each figure (PNG, PDF), its tables under ``data/`` and its caption in ``captions.md``.

    ``figures`` maps a figure name to ``(figure, caption, {table name: frame})``. Returns the paths.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    from yg_eo_soilnet.a1.figures.core import save_figure

    out = Path(out)
    paths = []
    for name, (figure, _, tables) in figures.items():
        paths += save_figure(figure, name, out)
        plt.close(figure)
        for table_name, frame in tables.items():
            path = out / "data" / f"{table_name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            paths.append(path)
    write_captions({name: caption for name, (_, caption, _) in figures.items()}, out / "captions.md")
    return paths


# --- MLflow ------------------------------------------------------------------------------------------


def log_robustness(run_dir, out, *, step: str) -> Optional[str]:
    """Add ``out`` to MLflow as a child of the newest ``compare`` run of the run's experiment.

    Without a ``compare`` run the child gets a new parent run ``robustness``. Returns the child
    run's id; None when the run folder records no MLflow experiment.
    """
    import json
    from pathlib import Path

    import mlflow

    manifest_path = Path(run_dir) / "manifest.json"
    entry = (json.loads(manifest_path.read_text()) if manifest_path.exists() else {}).get("mlflow") or {}
    if not entry.get("experiment_id"):
        logger.warning(f"{run_dir} records no MLflow experiment; {out} is not added to MLflow.")
        return None
    if entry.get("tracking_uri"):
        mlflow.set_tracking_uri(entry["tracking_uri"])
    found = mlflow.search_runs(
        experiment_ids=[entry["experiment_id"]],
        filter_string="tags.mlflow.runName = 'compare'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(found):
        parent = str(found["run_id"].iloc[0])
    else:
        with mlflow.start_run(experiment_id=entry["experiment_id"], run_name="robustness") as run:
            parent = run.info.run_id
    tags = {"mlflow.parentRunId": parent, "a1_step": "robustness"}
    with mlflow.start_run(experiment_id=entry["experiment_id"], run_name=step, tags=tags) as run:
        mlflow.log_artifacts(str(out), f"robustness/{step}")
        return run.info.run_id


def write_captions(captions: dict, path) -> None:
    """Add or replace these figures' sections of a ``captions.md`` shared by several scripts."""
    from pathlib import Path

    path = Path(path)
    sections: dict = {}
    if path.exists():
        name = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                name = line[3:].strip()
                sections[name] = []
            elif name is not None:
                sections[name].append(line)
        sections = {name: "\n".join(lines).strip() for name, lines in sections.items()}
    sections.update({name: caption.strip() for name, caption in captions.items()})
    lines = ["# Draft figure captions", "", "Robustness checks of the transfer mechanism; edit before use.", ""]
    for name in sorted(sections):
        lines += [f"## {name}", "", sections[name], ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
