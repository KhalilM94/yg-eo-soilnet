"""The points absent from the static-covariate export: are they predicted worse, and do E1's findings rest on them?

511 of the 5,806 points are not in ``Morocco_Static_Covariates_V_310826.csv``, the Earth Engine
export of the static covariates. They lack all 33 of its columns (bare-soil Sentinel-2, terrain,
aridity, the soil-map layers and texture class), which the models fill with the training median
(numbers, flagged by a measured-or-filled column) or the commonest class. No model is fitted here;
everything is read from E1's saved predictions and points:

:func:`missing_static_table` (``missing_static.csv``)
    Per design (random, 10 km folds, each held-out basin), model (Global, ``local_global``) and
    basin: RMSE, bias and n of the absent points and of the others, and the difference (absent -
    others) with a 10 km block-bootstrap interval and p-value. Descriptive: no multiple-testing
    correction.
:func:`transfer_without` (``missing_static_transfer.csv``)
    The transfer loss (Global - ``local_global`` RMSE on the same points) per held-out basin, with
    every point and without the absent ones, as :func:`~yg_eo_soilnet.a1.compare.transfer_status`
    computes it (paired block bootstrap, Holm across the basins).
:func:`relations_without` and :func:`within_class_counts` (``missing_static_relations*.csv``)
    The basin-vs-rest Spearman differences of :mod:`yg_eo_soilnet.a1.basin_relations` with the
    absent points removed everywhere (basin and complement), same 15 covariates, and the number of
    Holm-significant within-class differences before and after.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL_GLOBAL

logger = logging.getLogger(__name__)

#: The static-covariate export of the real data, in the data folder.
STATIC_EXPORT = "Morocco_Static_Covariates_V_310826.csv"
#: The designs, as named in ``missing_static.csv``.
DESIGN_RANDOM = "random"
DESIGN_10KM = "10 km folds"
HELD_OUT = "held out: "
#: The basin row pooling every basin of a design.
ALL_BASINS = "all"
LEVEL = 0.95

COLUMNS = [
    "design",
    "model",
    "basin",
    "n_absent",
    "n_other",
    "n_blocks",
    "rmse_absent",
    "rmse_other",
    "rmse_difference",
    "rmse_low",
    "rmse_high",
    "rmse_p_value",
    "bias_absent",
    "bias_other",
    "bias_difference",
    "bias_low",
    "bias_high",
    "bias_p_value",
]


def export_ids(path, id_column: str = "uuid") -> pd.Index:
    """The point ids of the static-covariate export, as text."""
    ids = pd.read_csv(path, usecols=[id_column])[id_column].dropna().astype(str)
    return pd.Index(ids.unique())


def absent_points(point_ids: Iterable, exported: pd.Index) -> pd.Index:
    """The point ids not in the export."""
    ids = pd.Index(pd.Series(list(point_ids), dtype=str).unique())
    return ids.difference(exported)


def scored_test_rows(frame: pd.DataFrame, models: Sequence[str], first_seed: int) -> pd.DataFrame:
    """The scored test rows of the first seed, every input, of these models; matched plans left out."""
    from yg_eo_soilnet.a1.config import COMBINED_GROUP
    from yg_eo_soilnet.a1.plans import is_matched
    from yg_eo_soilnet.datamodules.splitting import TEST

    seed = evaluation.seed_order(frame, first_seed)[0]
    keep = (
        (frame["seed"] == seed)
        & (frame["split"] == TEST)
        & (frame["covariate_group"] == COMBINED_GROUP)
        & frame["model"].isin(list(models))
        & ~is_matched(frame["plan_id"])
    )
    return evaluation._scored(frame.loc[keep]).reset_index(drop=True)


def designs(rows: pd.DataFrame):
    """``(design, rows)``: the random split, the 10 km folds pooled, and each held-out basin."""
    from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD

    yield DESIGN_RANDOM, rows[rows["kind"] == RANDOM]
    tenkm = (rows["kind"] == SPATIAL_KFOLD) & np.isclose(pd.to_numeric(rows["block_size_m"]), 10000.0)
    yield DESIGN_10KM, rows[tenkm]
    logo = rows[rows["kind"] == LOGO]
    for basin, part in logo.groupby("held_out", sort=True):
        yield f"{HELD_OUT}{basin}", part


def _scores(error: np.ndarray, absent: np.ndarray) -> tuple:
    """RMSE and bias of the absent points and of the others (NaN for an empty side)."""

    def rmse(e):
        return float(np.sqrt(np.mean(e**2))) if e.size else np.nan

    def bias(e):
        return float(np.mean(e)) if e.size else np.nan

    a, o = error[absent], error[~absent]
    return rmse(a), rmse(o), bias(a), bias(o)


def _difference(error, absent, blocks, *, n: int, rng) -> dict:
    """Absent - others for RMSE and bias, with a block-bootstrap interval and p-value."""
    rmse_a, rmse_o, bias_a, bias_o = _scores(error, absent)
    result = {
        "rmse_absent": rmse_a,
        "rmse_other": rmse_o,
        "rmse_difference": rmse_a - rmse_o,
        "bias_absent": bias_a,
        "bias_other": bias_o,
        "bias_difference": bias_a - bias_o,
    }
    n_blocks = len(np.unique(blocks))
    if not absent.any() or absent.all() or n == 0 or n_blocks < 2:
        result.update({f"{s}_{k}": np.nan for s in ("rmse", "bias") for k in ("low", "high", "p_value")})
        return result
    draws = np.full((n, 2), np.nan)
    for i, rows in enumerate(evaluation._draws(blocks, n, rng)):
        r_a, r_o, b_a, b_o = _scores(error[rows], absent[rows])
        draws[i] = (r_a - r_o, b_a - b_o)
    tail = 100.0 * (1.0 - LEVEL) / 2.0
    for column, score in enumerate(("rmse", "bias")):
        values = draws[:, column]
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, [tail, 100.0 - tail]) if finite.size else (np.nan, np.nan)
        result.update(
            {f"{score}_low": float(low), f"{score}_high": float(high), f"{score}_p_value": evaluation._p_value(values)}
        )
    return result


def missing_static_table(
    frame: pd.DataFrame,
    absent: pd.Index,
    *,
    first_seed: int,
    models: Sequence[str] = (GLOBAL, LOCAL_GLOBAL),
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """``missing_static.csv``: per design, model and basin, the absent points against the others.

    Parameters
    ----------
    frame : pandas.DataFrame
        The run's predictions (:func:`~yg_eo_soilnet.a1.evaluation.load_predictions`).
    absent : pandas.Index
        The point ids absent from the static export.
    first_seed : int
        The seed read (the others are not).
    models : sequence of str
        ``local_global`` exists for the held-out basins only.
    block_km, n, seed
        The bootstrap: blocks of ``block_km``, ``n`` resamples, seeded by ``seed`` and the row's key.

    Returns
    -------
    pandas.DataFrame
        :data:`COLUMNS`; a basin row per region of the design's test points, and ``all`` pooling
        them when there are several.
    """
    rows = scored_test_rows(frame, models, first_seed)
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    rows = rows.assign(block=evaluation.spatial_blocks(rows, block_km).to_numpy())
    absent_set = set(map(str, absent))
    out = []
    for design, part in designs(rows):
        for model in models:
            fitted = part[part["model"] == model]
            if fitted.empty:
                continue
            regions = fitted["region"].astype(str)
            cells = [(ALL_BASINS, fitted)] if regions.nunique() > 1 else []
            cells += [(basin, cell) for basin, cell in fitted.groupby(regions, sort=True)]
            for basin, cell in cells:
                flag = cell["point_id"].astype(str).isin(absent_set).to_numpy()
                error = (cell["predicted"] - cell["observed"]).to_numpy(dtype=float)
                blocks = cell["block"].to_numpy()
                rng = evaluation._generator(seed, (design, model, basin))
                out.append(
                    {
                        "design": design,
                        "model": model,
                        "basin": basin,
                        "n_absent": int(flag.sum()),
                        "n_other": int((~flag).sum()),
                        "n_blocks": len(np.unique(blocks)),
                        **_difference(error, flag, blocks, n=n, rng=rng),
                    }
                )
    return pd.DataFrame(out, columns=COLUMNS)


def transfer_without(run, absent: pd.Index, *, block_km: float = 10.0, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """``missing_static_transfer.csv``: the transfer status with every point and without the absent ones.

    ``run`` is a :class:`~yg_eo_soilnet.a1.compare.RunFolder` read with ``local_global``; ``points``
    is ``all`` or ``without absent``. Each is Holm-corrected across the basins on its own.
    """
    from yg_eo_soilnet.a1.compare import transfer_status

    import copy

    reduced = copy.copy(run)
    reduced.predictions = run.predictions[~run.predictions["point_id"].astype(str).isin(set(map(str, absent)))]
    tables = []
    for points, folder in (("all", run), ("without absent", reduced)):
        status = transfer_status(folder, model=GLOBAL, block_km=block_km, n=n, seed=seed, local=LOCAL_GLOBAL)
        tables.append(status.assign(points=points))
    table = pd.concat(tables, ignore_index=True)
    return table[["points", *[c for c in table.columns if c != "points"]]]


def relations_without(
    data,
    absent: pd.Index,
    *,
    basins: Optional[Sequence[str]] = None,
    min_cell_size: int = 30,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """:func:`~yg_eo_soilnet.a1.basin_relations.basin_relations` with the absent points removed everywhere.

    ``data`` is the :class:`~yg_eo_soilnet.a1.basin_relations.BasinData` of the full run, so the
    15 covariates are those chosen with every point; ``basins`` limits the basins recomputed (each
    basin's Holm family is its own, so the others are not needed).
    """
    from yg_eo_soilnet.a1.basin_relations import basin_relations

    keep = data.frame.index.difference(pd.Index(list(map(str, absent))))
    chosen = basins if basins is not None else list(data.basins)
    reduced = replace(
        data,
        frame=data.frame.loc[keep],
        classes=data.classes.reindex(keep),
        blocks=data.blocks.reindex(keep),
        basins={basin: data.basins[basin].intersection(keep) for basin in chosen},
    )
    return basin_relations(reduced, min_cell_size=min_cell_size, n=n, seed=seed)


def within_class_counts(before: pd.DataFrame, after: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """``missing_static_relations_counts.csv``: per basin and class, the Holm-significant differences before and after.

    Within each measured class (not ``all``), exploratory family: how many of the covariates (and
    of CaCO3/CEC) differ from the rest at p_Holm < ``alpha``, the class's n in the basin, and which
    variables, before (every point) and after (absent points removed).
    """
    from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES

    rows = []
    basins = sorted(set(after["basin"]))
    for basin in basins:
        for points, table in (("all", before), ("without absent", after)):
            part = table[(table["basin"] == basin) & (table["class"] != ALL_CLASSES)]
            for label, cell in part.groupby("class", sort=False):
                hits = cell[cell["p_holm"] < alpha]
                rows.append(
                    {
                        "basin": basin,
                        "class": label,
                        "points": points,
                        "n_basin": int(cell["n_basin"].iloc[0]),
                        "n_significant": len(hits),
                        "variables": ";".join(map(str, hits["variable"])),
                    }
                )
    return pd.DataFrame(rows, columns=["basin", "class", "points", "n_basin", "n_significant", "variables"])


def static_export_path(settings, config, name: Optional[str] = None) -> Path:
    """The static-covariate export: ``name`` as given, else :data:`STATIC_EXPORT` in the data folder."""
    if name and Path(name).is_absolute():
        return Path(name)
    root = Path(getattr(config, "DATA_FOLDER", None) or settings.data.get("root", "."))
    return root / (name or STATIC_EXPORT)
