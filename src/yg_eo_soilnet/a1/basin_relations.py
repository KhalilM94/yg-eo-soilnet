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
its interval from the same draws. Its two-sided p-value (:func:`difference_p_value`, the
``p_method`` column) is by default from a normal approximation on Fisher's z (``normal``); the
older rule, the share of draws on either side of zero (``draws``), is kept for comparison: it
cannot go below 2/n, which with 1,000 draws (0.002) is above Holm's first threshold for an 83-test
family (0.05/83 = 0.0006). The p-values are Holm-corrected within each basin and family (the
``family`` column).
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
#: How a difference's p-value is computed (:func:`difference_p_value`): the default first.
P_METHODS = ("normal", "draws")
#: Fisher's z is capped this close to |rho| = 1, so a perfect correlation gives a large, finite z.
RHO_LIMIT = 1.0 - 1e-9


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


def fisher_z(rho):
    """Fisher's z = atanh(rho), with rho capped just inside (-1, 1)."""
    return np.arctanh(np.clip(np.asarray(rho, dtype=float), -RHO_LIMIT, RHO_LIMIT))


def difference_p_value(rho_basin: float, rho_complement: float, draws_basin, draws_complement, method: str = "normal"):
    """Two-sided p-value of "no difference" between the basin's and the complement's rho.

    ``normal``
        z = Δ / SE, with Δ = atanh(rho_basin) - atanh(rho_complement) and SE the standard deviation
        of the same difference over the bootstrap draws; p = 2 (1 - Φ(|z|)). Not bounded by the
        number of draws.
    ``draws``
        Twice the smaller share of draws of the difference on either side of 0
        (:func:`~yg_eo_soilnet.a1.evaluation._p_value`): never below 2/n with n draws.

    >>> rng = np.random.default_rng(0)
    >>> b, c = 0.6 + 0.05 * rng.standard_normal(300), 0.1 + 0.05 * rng.standard_normal(300)
    >>> difference_p_value(0.6, 0.1, b, c, "normal") < 1e-6, difference_p_value(0.6, 0.1, b, c, "draws")
    (True, 0.0)
    """
    draws_basin = np.asarray(draws_basin, dtype=float)
    draws_complement = np.asarray(draws_complement, dtype=float)
    if method == "draws":
        return evaluation._p_value(draws_basin - draws_complement)
    if method != "normal":
        raise ValueError(f"p_method {method!r}: expected one of {P_METHODS}")
    from scipy.stats import norm

    gaps = fisher_z(draws_basin) - fisher_z(draws_complement)
    gaps = gaps[np.isfinite(gaps)]
    estimate = float(fisher_z(rho_basin) - fisher_z(rho_complement))
    if gaps.size < 2 or not np.isfinite(estimate):
        return np.nan
    se = float(np.std(gaps, ddof=1))
    if se == 0.0:
        return 1.0 if estimate == 0.0 else 0.0
    return float(2.0 * norm.sf(abs(estimate) / se))


def p_method_text(table: pd.DataFrame) -> str:
    """How the table's p-values were computed, for a caption (empty for a table without ``p_method``)."""
    methods = set(table["p_method"].dropna()) if "p_method" in table else set()
    if methods == {"normal"}:
        return "p-values from a normal approximation on Fisher's z, with the bootstrap standard error"
    if methods == {"draws"}:
        return "p-values from the share of draws on either side of 0 (resolution-limited: never below 2/n)"
    return ""


def basin_relations(
    data: BasinData, *, min_cell_size: int = 30, n: int = 1000, seed: int = 42, p_method: str = "normal"
) -> pd.DataFrame:
    """``basin_relations.csv``: per basin, class and pair, rho in the basin and its complement, and their difference.

    Returns
    -------
    pandas.DataFrame
        ``basin, class`` (``all`` or a measured class), ``variable, family, rho_basin, low_basin,
        high_basin, n_basin, rho_complement, low_complement, high_complement, n_complement,
        difference, low_difference, high_difference, p_value, p_holm, p_method``; the p-value by
        ``p_method`` (:func:`difference_p_value`).
    """
    if p_method not in P_METHODS:
        raise ValueError(f"p_method {p_method!r}: expected one of {P_METHODS}")
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
                        "p_value": difference_p_value(
                            rho_b[position], rho_c[position], draws_b[:, position], draws_c[:, position], p_method
                        )
                        if n
                        else np.nan,
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
        "p_method",
    ]
    table = pd.DataFrame(rows, columns=[column for column in columns if column not in ("p_holm", "p_method")])
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["p_holm"] = table.groupby(["basin", "family"], sort=False)["p_value"].transform(evaluation.holm)
    table["p_method"] = p_method
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
    settings: A1Settings, out_dir, *, n: int = 1000, config=None, n_covariates: int = 15, p_method: str = "normal"
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
    relations = basin_relations(data, min_cell_size=settings.min_cell_size, n=n, seed=settings.seed, p_method=p_method)
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


def method_comparison(tables: dict, *, alpha: float = 0.05) -> pd.DataFrame:
    """``p_method_comparison.csv``: the same tests under several p-value runs, side by side.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        A label (``normal, 1,000``) to a ``basin_relations.csv`` table; the first is the reference.
    alpha : float
        The Holm level.

    Returns
    -------
    pandas.DataFrame
        One row per basin, class and variable: ``difference`` (from the first table), then
        ``p_value``, ``p_holm`` and ``significant`` per label (``p_holm [label]``), and
        ``differs``: the Holm decision is not the same under every label.
    """
    keys = ["basin", "class", "variable", "family"]
    labels = list(tables)
    merged = None
    for label in labels:
        part = tables[label][[*keys, "difference", "p_value", "p_holm"]].copy()
        part[f"significant [{label}]"] = part["p_holm"] < alpha
        part = part.rename(columns={"p_value": f"p_value [{label}]", "p_holm": f"p_holm [{label}]"})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part.drop(columns="difference"), on=keys, how="outer")
    flags = merged[[f"significant [{label}]" for label in labels]].fillna(False).astype(bool)
    merged["differs"] = flags.nunique(axis=1) > 1
    return merged


def significant_counts(comparison: pd.DataFrame, *, within_class: bool = True) -> pd.DataFrame:
    """Per basin, the number of Holm-significant tests under each label of :func:`method_comparison`."""
    rows = comparison[comparison["class"] != ALL_CLASSES] if within_class else comparison
    flags = [column for column in rows.columns if column.startswith("significant [")]
    counts = rows.groupby("basin")[flags].sum().astype(int)
    counts.columns = [column.removeprefix("significant [").removesuffix("]") for column in counts.columns]
    counts["n_differ"] = rows.groupby("basin")["differs"].sum().astype(int)
    return counts.reset_index()
