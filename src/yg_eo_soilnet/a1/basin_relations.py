"""How SOM relates to carbonate, CEC and the covariates in each held-out basin and in the rest of the data.

No model is fitted. A model trained on the other basins can only transfer to a basin where SOM
relates to its inputs the way it does there; this module measures, for each held-out basin and its
*complement* (every other point: the training data of the basin's ``logo`` plan), the Spearman
correlation of SOM with:

- CaCO3 and CEC (the two stage-1 targets) - the **primary** family, 2 tests per basin;
- the 15 covariates of figure 3 (one percentile per variable, chosen by the figure's own rule over
  all the points, :func:`~yg_eo_soilnet.a1.figures.correlations.choose_rows`), and every pair again
  within each measured carbonate class that holds at least ``min_cell_size`` points on both sides -
  the **exploratory** family.

Each correlation gets a 95 % interval from resampling whole spatial blocks (``bootstrap.block_size_km``),
the basin's and the complement's blocks each on their own; the basin - complement difference takes
its interval and two-sided p-value from the same draws, and the p-values are Holm-corrected within
each basin and family (the ``family`` column).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.settings import A1Settings

logger = logging.getLogger(__name__)

#: The class label of a row over every point of a side, whatever its carbonate class.
ALL_CLASSES = "all"
PRIMARY = "primary"
EXPLORATORY = "exploratory"
LEVEL = 0.95


@dataclass
class BasinData:
    """What the relations are computed from.

    Attributes
    ----------
    frame : pandas.DataFrame
        One row per point with a measured SOM, indexed by point id.
    target : str
    pairs : list of str
        The columns SOM is correlated with: CaCO3 and CEC first, then the covariates.
    primary : list of str
        The columns of the primary family.
    classes : pandas.Series
        Each point's measured carbonate class.
    labels : list of str
        The classes, in order.
    blocks : pandas.Series
        Each point's spatial block.
    basins : dict of str to pandas.Index
        Each held-out basin's points.
    """

    frame: pd.DataFrame
    target: str
    pairs: list
    primary: list
    classes: pd.Series
    labels: list
    blocks: pd.Series
    basins: dict


def load_basin_data(settings: A1Settings, config=None, *, n_covariates: int = 15) -> BasinData:
    """Read the points, their basins, classes and blocks, and pick the covariates of figure 3."""
    from yg_eo_soilnet.a1.features import build_table
    from yg_eo_soilnet.a1.figures.core import _numeric_covariates, spearman_by_class
    from yg_eo_soilnet.a1.figures.correlations import choose_rows
    from yg_eo_soilnet.a1.figures.labels import source_of
    from yg_eo_soilnet.a1.plans import plan_population
    from yg_eo_soilnet.a1.strata import carbonate_class

    config = config if config is not None else settings.load_config()
    table = build_table(settings, config)
    population = plan_population(settings, config, points=table)
    frame = table.frame.loc[table.frame[settings.target].notna()]
    carbonate = settings.carbonate
    classes = carbonate_class(frame[carbonate.column], carbonate.edges, carbonate.labels)
    coordinates = table.coordinates.reindex(frame.index).rename_axis("point_id").reset_index()
    blocks = pd.Series(
        evaluation.spatial_blocks(coordinates, settings.bootstrap.block_size_km).to_numpy(), index=frame.index
    )

    primary = [column for column in settings.auxiliary.targets if column in frame.columns]
    covariates = [column for column in _numeric_covariates(table) if source_of(column) is not None]
    covariates = [column for column in covariates if column not in primary]
    chosen: list = []
    if covariates:
        correlations = spearman_by_class(
            frame, settings.target, covariates, classes, labels=list(carbonate.labels), n=0, seed=settings.seed
        )
        chosen = list(choose_rows(correlations, max_rows=n_covariates)["column"])
    basins = {
        population.region_names[region]: population.region_points(region).intersection(frame.index)
        for region in population.held_out
    }
    return BasinData(
        frame=frame,
        target=settings.target,
        pairs=[*primary, *chosen],
        primary=primary,
        classes=classes,
        labels=list(carbonate.labels),
        blocks=blocks,
        basins=basins,
    )


def _side(data: BasinData, ids: pd.Index, n: int, seed: int, key: tuple):
    """Spearman rho of SOM with every pair on these points, and ``n`` block-bootstrap draws of it."""
    from yg_eo_soilnet.a1.figures.core import _spearman_columns

    part = data.frame.loc[ids]
    y = part[data.target].to_numpy(dtype=float)
    X = part[data.pairs].to_numpy(dtype=float)
    rho = _spearman_columns(y, X)
    if n == 0 or len(part) < 3:
        return rho, np.full((max(n, 1), len(data.pairs)), np.nan)
    block_ids = data.blocks.reindex(part.index).to_numpy()
    draws = np.array(
        [
            _spearman_columns(y[rows], X[rows])
            for rows in evaluation._draws(block_ids, n, evaluation._generator(seed, key))
        ]
    )
    return rho, draws


def _bounds(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tail = 100.0 * (1.0 - LEVEL) / 2.0
    low, high = [], []
    for column in draws.T:
        finite = column[np.isfinite(column)]
        low.append(np.percentile(finite, tail) if finite.size else np.nan)
        high.append(np.percentile(finite, 100.0 - tail) if finite.size else np.nan)
    return np.array(low), np.array(high)


def basin_relations(data: BasinData, *, min_cell_size: int = 30, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """``basin_relations.csv``: per basin, class and pair, rho in the basin and its complement, and their difference.

    Returns
    -------
    pandas.DataFrame
        ``basin, class`` (``all`` or a measured class), ``variable, family, rho_basin, low_basin,
        high_basin, n_basin, rho_complement, low_complement, high_complement, n_complement,
        difference, low_difference, high_difference, p_value, p_holm``.
    """
    rows = []
    for basin, ids in data.basins.items():
        complement = data.frame.index.difference(ids)
        scopes = [(ALL_CLASSES, ids, complement)]
        for label in data.labels:
            in_class = data.classes.reindex(data.frame.index) == label
            basin_ids = ids.intersection(data.frame.index[in_class.to_numpy()])
            rest_ids = complement.intersection(data.frame.index[in_class.to_numpy()])
            if len(basin_ids) >= min_cell_size and len(rest_ids) >= min_cell_size:
                scopes.append((label, basin_ids, rest_ids))
        for label, basin_ids, rest_ids in scopes:
            rho_b, draws_b = _side(data, basin_ids, n, seed, (basin, label, "basin"))
            rho_c, draws_c = _side(data, rest_ids, n, seed, (basin, label, "complement"))
            low_b, high_b = _bounds(draws_b)
            low_c, high_c = _bounds(draws_c)
            gaps = draws_b - draws_c
            low_d, high_d = _bounds(gaps)
            for position, variable in enumerate(data.pairs):
                primary = label == ALL_CLASSES and variable in data.primary
                rows.append(
                    {
                        "basin": basin,
                        "class": label,
                        "variable": variable,
                        "family": PRIMARY if primary else EXPLORATORY,
                        "rho_basin": rho_b[position],
                        "low_basin": low_b[position],
                        "high_basin": high_b[position],
                        "n_basin": len(basin_ids),
                        "rho_complement": rho_c[position],
                        "low_complement": low_c[position],
                        "high_complement": high_c[position],
                        "n_complement": len(rest_ids),
                        "difference": rho_b[position] - rho_c[position],
                        "low_difference": low_d[position],
                        "high_difference": high_d[position],
                        "p_value": evaluation._p_value(gaps[:, position]) if n else np.nan,
                    }
                )
    columns = [
        "basin",
        "class",
        "variable",
        "family",
        "rho_basin",
        "low_basin",
        "high_basin",
        "n_basin",
        "rho_complement",
        "low_complement",
        "high_complement",
        "n_complement",
        "difference",
        "low_difference",
        "high_difference",
        "p_value",
        "p_holm",
    ]
    table = pd.DataFrame(rows, columns=[column for column in columns if column != "p_holm"])
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["p_holm"] = table.groupby(["basin", "family"], sort=False)["p_value"].transform(evaluation.holm)
    return table[columns]


def som_by_class(data: BasinData, carbonate_column: str) -> pd.DataFrame:
    """``basin_som_by_class.csv``: per basin and measured class, n, SOM median and IQR, CaCO3 median."""
    rows = []
    for basin, ids in data.basins.items():
        part = data.frame.loc[ids]
        classes = data.classes.reindex(part.index)
        for label in data.labels:
            chosen = part[(classes == label).to_numpy()]
            som = chosen[data.target]
            rows.append(
                {
                    "basin": basin,
                    "class": label,
                    "n": len(chosen),
                    "som_median": float(som.median()) if len(som) else np.nan,
                    "som_q25": float(som.quantile(0.25)) if len(som) else np.nan,
                    "som_q75": float(som.quantile(0.75)) if len(som) else np.nan,
                    "som_iqr": float(som.quantile(0.75) - som.quantile(0.25)) if len(som) else np.nan,
                    "caco3_median": float(chosen[carbonate_column].median()) if len(chosen) else np.nan,
                }
            )
    return pd.DataFrame(
        rows, columns=["basin", "class", "n", "som_median", "som_q25", "som_q75", "som_iqr", "caco3_median"]
    )


def run_basin_relations(
    settings: A1Settings, out_dir, *, n: int = 1000, config=None, n_covariates: int = 15
) -> tuple[dict, dict]:
    """Compute both tables, draw figure C3, and write them all into ``out_dir``.

    Returns
    -------
    tables : dict of str to pandas.DataFrame
    captions : dict of str to str
    """
    from pathlib import Path

    from yg_eo_soilnet.a1.figures.basin_relations import figure_basin_relations
    from yg_eo_soilnet.a1.figures.core import save_figure, write_table

    data = load_basin_data(settings, config, n_covariates=n_covariates)
    relations = basin_relations(data, min_cell_size=settings.min_cell_size, n=n, seed=settings.seed)
    by_class = som_by_class(data, settings.carbonate.column)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    relations.to_csv(out / "basin_relations.csv", index=False)
    write_table(by_class, out / "basin_som_by_class", float_format="%.2f")
    figure, caption = figure_basin_relations(relations, block_km=settings.bootstrap.block_size_km, n_boot=n)
    save_figure(figure, "figC3_basin_relations", out)
    captions = {"figC3_basin_relations": caption}
    from yg_eo_soilnet.a1.compare import write_compare_captions

    write_compare_captions(captions, out / "captions.md")
    return {"basin_relations": relations, "basin_som_by_class": by_class}, captions


def primary_pairs(relations: pd.DataFrame) -> pd.DataFrame:
    """The primary rows (SOM with CaCO3 and with CEC, every class), one per basin and variable."""
    return relations[relations["family"] == PRIMARY]


def variables_in_order(relations: pd.DataFrame, primary: Optional[Sequence[str]] = None) -> list:
    """The variables of the table, the primary ones first."""
    variables = list(dict.fromkeys(relations["variable"]))
    first = [variable for variable in variables if primary is not None and variable in primary]
    if primary is None:
        first = list(dict.fromkeys(relations.loc[relations["family"] == PRIMARY, "variable"]))
    return first + [variable for variable in variables if variable not in first]
