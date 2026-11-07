"""The pre-flight diagnostics: does the data hold up the A1 experiments, before any is run?

Four questions, each answered by a table written as a CSV and summarised in ``preflight.md``:

1. **Completeness.** How many points have SOM, CaCO3 and CEC all measured, and how empty is every
   column the experiments read.
2. **Regions x carbonate.** How many points fall in each region and FAO carbonate class. A cell
   with fewer than ``min_cell_size`` points is flagged: a per-class model, or a per-class
   score, cannot be trusted on so few points.
3. **Regions x sampling year.** Whether some regions were only sampled in some years, which would
   tangle "a different region" with "a different year".
4. **Earlier predictions.** When the data already carries a model's predictions of CaCO3 and CEC
   (``preflight.prediction_columns``) and a split is given, how well they score on the training,
   validation and test points separately. Much better on the training points than on the test
   points is the mark of a model that was fitted on those training points' lab values - a leak, if
   those predictions are then used as inputs for SOM and scored on the same split.

Nothing here trains a model or touches MLflow; it only reads.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1.features import A1Table, check_inputs, load_points
from yg_eo_soilnet.a1.settings import PERCENTILES_STATIC, A1Settings
from yg_eo_soilnet.a1.strata import carbonate_class
from yg_eo_soilnet.datamodules.splitting import SPLIT_NAMES, TEST, TRAIN, SplitPlan
from yg_eo_soilnet.metrics import regression_metrics

#: The row and column holding the totals of a count table.
TOTAL = "Total"
#: Where a point without a region is counted.
NO_REGION = "(no region)"
#: Where a point without a measured CaCO3 is counted.
NO_CACO3 = "(no CaCO3)"
#: Where a point without a sampling year is counted.
NO_YEAR = "(no year)"
#: The completeness row counting the points with every property measured.
ALL_MEASURED = "all measured"

COMPLETENESS_FILE = "completeness.csv"
MISSINGNESS_FILE = "missingness.csv"
REGION_BY_CARBONATE_FILE = "region_by_carbonate.csv"
SMALL_CELLS_FILE = "region_by_carbonate_small_cells.csv"
REGION_BY_YEAR_FILE = "region_by_year.csv"
PREDICTIONS_FILE = "auxiliary_predictions.csv"
INPUTS_FILE = "inputs.csv"
#: What the inputs table says of an input no covariate group claims.
NO_GROUP = "(none)"
REPORT_FILE = "preflight.md"


@dataclass
class PreflightResult:
    """What a pre-flight run found and wrote.

    Attributes
    ----------
    out_dir : pathlib.Path
        The folder everything was written to.
    unassigned : list of str
        The inputs no covariate group claims: no model reads them.
    files : list of pathlib.Path
        The files written, the report last.
    small_cells : pandas.DataFrame
        The region x carbonate-class cells under the threshold: ``region``, ``carbonate_class``,
        ``n``. Empty when none are, or when there are no regions.
    overfit : list of str
        The properties whose earlier predictions were flagged.
    skipped : dict of str to str
        Each section that could not run, and why.
    tables : dict of str to pandas.DataFrame
        Every table, by the name of the file it was written to.
    """

    out_dir: Path
    unassigned: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    small_cells: pd.DataFrame = field(default_factory=pd.DataFrame)
    overfit: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


# --- the four tables ------------------------------------------------------------------


def completeness(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """How many points have each of these columns measured, and how many have all of them.

    Returns
    -------
    pandas.DataFrame
        Indexed by column name, then :data:`ALL_MEASURED`: ``n_measured``, ``n_points`` and
        ``share_measured``.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"som": [1, 2, None, 4], "caco3": [1, None, None, 4]})
    >>> completeness(frame, ["som", "caco3"])["n_measured"].to_dict()
    {'som': 3, 'caco3': 2, 'all measured': 2}
    """
    measured = frame[columns].notna()
    counts = pd.concat([measured.sum(), pd.Series({ALL_MEASURED: int(measured.all(axis=1).sum())})])
    table = pd.DataFrame({"n_measured": counts.astype(int), "n_points": len(frame)})
    table["share_measured"] = (table["n_measured"] / max(len(frame), 1)).round(4)
    table.index.name = "column"
    return table


def missingness(frame: pd.DataFrame, kinds: Mapping[str, str]) -> pd.DataFrame:
    """How empty each column is, emptiest first.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per point.
    kinds : mapping of str to str
        Each column to report, and what kind of column it is (``static covariate``, ``lab value``...).

    Returns
    -------
    pandas.DataFrame
        ``column``, ``kind``, ``n_missing``, ``share_missing``.
    """
    rows = [
        {"column": column, "kind": kind, "n_missing": int(frame[column].isna().sum())}
        for column, kind in kinds.items()
        if column in frame.columns
    ]
    table = pd.DataFrame(rows, columns=["column", "kind", "n_missing"])
    table["share_missing"] = (table["n_missing"] / max(len(frame), 1)).round(4)
    return table.sort_values(["n_missing", "column"], ascending=[False, True], kind="stable").reset_index(drop=True)


def count_table(
    rows: pd.Series,
    columns: pd.Series,
    *,
    row_missing: str,
    column_missing: str,
    column_order: Optional[list] = None,
) -> pd.DataFrame:
    """Count the points in each row x column cell, with totals, keeping every point somewhere.

    A point missing its row or its column value is counted under ``row_missing`` or
    ``column_missing`` rather than dropped, so the grand total is always the number of points.

    Parameters
    ----------
    rows, columns : pandas.Series
        One value per point, aligned.
    row_missing, column_missing : str
        Where a point with no value is counted; added only when some point needs it.
    column_order : list, optional
        The column order; the sorted values otherwise. A categorical's own order is used when it
        has one.

    Returns
    -------
    pandas.DataFrame
        Integer counts, with a :data:`TOTAL` row and column.

    Examples
    --------
    >>> import pandas as pd
    >>> regions = pd.Series(["A", "A", "B", None])
    >>> years = pd.Series([2019, 2020, 2020, 2020])
    >>> count_table(regions, years, row_missing="?", column_missing="-").to_dict(orient="index")
    {'A': {2019: 1, 2020: 1, 'Total': 2}, 'B': {2019: 0, 2020: 1, 'Total': 1}, '?': {2019: 0, 2020: 1, 'Total': 1}, 'Total': {2019: 1, 2020: 3, 'Total': 4}}
    """
    if column_order is None:
        if isinstance(columns.dtype, pd.CategoricalDtype):
            column_order = list(columns.cat.categories)
        else:
            column_order = sorted(columns.dropna().unique().tolist())
    row_values = rows.astype(object).where(rows.notna(), row_missing)
    column_values = columns.astype(object).where(columns.notna(), column_missing)

    row_order = sorted(value for value in row_values.unique() if value != row_missing)
    if (row_values == row_missing).any():
        row_order.append(row_missing)
    if (column_values == column_missing).any():
        column_order = [*column_order, column_missing]

    counts = pd.crosstab(row_values.to_numpy(), column_values.to_numpy())
    counts = counts.reindex(index=row_order, columns=column_order, fill_value=0).astype(int)
    counts[TOTAL] = counts.sum(axis=1)
    counts.loc[TOTAL] = counts.sum(axis=0)
    counts.index.name = rows.name
    counts.columns.name = columns.name
    return counts


def small_cells(table: pd.DataFrame, threshold: int, classes: list) -> pd.DataFrame:
    """The region x class cells holding fewer than ``threshold`` points.

    Only real regions and real classes are looked at: not the totals, not the points without a
    region, not the points without a measured CaCO3.

    Returns
    -------
    pandas.DataFrame
        ``region``, ``carbonate_class``, ``n``, fewest first.
    """
    regions = [region for region in table.index if region not in (TOTAL, NO_REGION)]
    cells = table.loc[regions, [label for label in classes if label in table.columns]]
    long = cells.stack().rename("n").reset_index()
    long.columns = ["region", "carbonate_class", "n"]
    return long[long["n"] < threshold].sort_values(["n", "region"], kind="stable").reset_index(drop=True)


def prediction_scores(
    frame: pd.DataFrame,
    prediction_columns: Mapping[str, str],
    splits: pd.Series,
    *,
    overfit_r2_gap: float,
) -> pd.DataFrame:
    """Score each earlier prediction column against its measurement, on each split separately.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per point, holding the measurements and the predictions.
    prediction_columns : mapping of str to str
        Lab column to the column predicting it.
    splits : pandas.Series
        Each point's split, aligned with ``frame``; a point in none is not scored.
    overfit_r2_gap : float
        How far the training R² may beat the test R² before the column is flagged.

    Returns
    -------
    pandas.DataFrame
        One row per property and split: ``property``, ``prediction_column``, ``split``, ``n``,
        ``r2``, ``rmse``, and per property ``r2_gap`` (training minus test) and ``flag_overfit``.
    """
    rows = []
    for target, column in prediction_columns.items():
        for split in SPLIT_NAMES:
            chosen = splits.eq(split).to_numpy()
            scores = regression_metrics(frame.loc[chosen, target], frame.loc[chosen, column], split=split)
            rows.append(
                {
                    "property": target,
                    "prediction_column": column,
                    "split": split,
                    "n": int(scores.get(f"n_{split}", 0)),
                    "r2": scores.get(f"r2_{split}", np.nan),
                    "rmse": scores.get(f"rmse_{split}", np.nan),
                }
            )
    table = pd.DataFrame(rows, columns=["property", "prediction_column", "split", "n", "r2", "rmse"])
    r2 = table.pivot(index="property", columns="split", values="r2")
    gap = (r2.get(TRAIN) - r2.get(TEST)) if {TRAIN, TEST} <= set(r2.columns) else pd.Series(dtype=float)
    table["r2_gap"] = table["property"].map(gap)
    # A gap that could not be computed - a split with too few points - is not a flag.
    table["flag_overfit"] = table["r2_gap"].gt(overfit_r2_gap)
    return table


# --- running it -------------------------------------------------------------------------


def run_preflight(
    settings: A1Settings,
    *,
    split_plan_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    min_cell_points: Optional[int] = None,
    overfit_r2_gap: Optional[float] = None,
    config=None,
    groups=None,
    logger: Optional[logging.Logger] = None,
) -> PreflightResult:
    """Run the diagnostics and write their tables and the report.

    First, the inputs are checked: no lab value, and no column ``exclude_inputs`` names, may be one
    (:func:`~yg_eo_soilnet.a1.features.check_inputs`); the run stops if one is. Every input is then
    matched to its covariate group, and the ones no group claims - which no model would read - are
    listed.

    Parameters
    ----------
    settings : A1Settings
        The A1 settings; the thresholds come from its ``preflight`` section unless given here.
    split_plan_path : str, optional
        A saved split (``data_splits/split_assignments.parquet``). Without one the earlier
        predictions are not scored.
    out_dir : str, optional
        Where to write; ``<output_dir>/preflight`` unless given.
    min_cell_points : int, optional
        Overrides ``min_cell_size``.
    overfit_r2_gap : float, optional
        Overrides ``preflight.overfit_r2_gap``.
    config : Config, optional
        The main configuration; read from the settings unless given.
    groups : CovariateGroups, optional
        The covariate groups; read from ``covariate_groups`` in the settings unless given.
    logger : logging.Logger, optional
        Where progress is reported.

    Returns
    -------
    PreflightResult

    Raises
    ------
    ValueError
        If a lab value or an excluded column is an input.
    """
    logger = logger or logging.getLogger(__name__)
    config = config if config is not None else settings.load_config()
    threshold = int(min_cell_points if min_cell_points is not None else settings.min_cell_size)
    gap = float(overfit_r2_gap if overfit_r2_gap is not None else settings.preflight.overfit_r2_gap)
    out = Path(out_dir or os.path.join(settings.output_dir, "preflight"))
    out.mkdir(parents=True, exist_ok=True)

    points = load_points(settings, config, logger)
    check_inputs(points, settings, config)
    frame = points.frame
    result = PreflightResult(out_dir=out)
    sections: list[str] = []

    # 0. The inputs, and the covariate group of each.
    result.tables[INPUTS_FILE], result.unassigned = _inputs_table(points, settings, groups, logger)
    sections.append(_inputs_section(result, settings))

    # 1. Completeness and missingness.
    properties = list(dict.fromkeys([settings.target, *settings.auxiliary.targets]))
    result.tables[COMPLETENESS_FILE] = completeness(frame, properties)
    result.tables[MISSINGNESS_FILE] = missingness(frame, _column_kinds(points))
    sections.append(_completeness_section(result.tables[COMPLETENESS_FILE], result.tables[MISSINGNESS_FILE]))

    # 2. Regions x carbonate class.
    classes = carbonate_class(frame[settings.carbonate.column], settings.carbonate.edges, settings.carbonate.labels)
    classes.name = "carbonate_class"
    if points.region_column is None:
        result.skipped["region_by_carbonate"] = "regions.source is not set in the A1 settings."
    else:
        table = count_table(
            frame[points.region_column].rename("region"), classes, row_missing=NO_REGION, column_missing=NO_CACO3
        )
        result.tables[REGION_BY_CARBONATE_FILE] = table
        result.small_cells = small_cells(table, threshold, list(settings.carbonate.labels))
        result.tables[SMALL_CELLS_FILE] = result.small_cells
    sections.append(_carbonate_section(result, threshold, list(settings.carbonate.labels)))

    # 3. Regions x sampling year.
    if points.region_column is None:
        result.skipped["region_by_year"] = "regions.source is not set in the A1 settings."
    elif points.sample_year_column is None:
        result.skipped["region_by_year"] = "sample_year.column is not set in the A1 settings."
    else:
        result.tables[REGION_BY_YEAR_FILE] = count_table(
            frame[points.region_column].rename("region"),
            frame[points.sample_year_column].rename("sample_year"),
            row_missing=NO_REGION,
            column_missing=NO_YEAR,
        )
    sections.append(_year_section(result))

    # 4. Earlier predictions, per split.
    coverage_note = ""
    wanted = settings.preflight.prediction_columns
    if not points.prediction_columns:
        listed = ", ".join(f"`{column}`" for column in wanted.values()) or "none listed"
        result.skipped["auxiliary_predictions"] = f"none of the prediction columns ({listed}) is in the data."
    elif not split_plan_path:
        result.skipped["auxiliary_predictions"] = "no split plan was given (--split-plan)."
    else:
        plan = SplitPlan.from_frame(pd.read_parquet(split_plan_path))
        splits = plan.labels_for(frame.index)
        splits.index = frame.index
        uncovered = int(splits.isna().sum())
        extra = len(plan.assignments.index.difference(frame.index))
        coverage_note = (
            f"Split plan `{split_plan_path}`: {plan.counts()}. {uncovered} point(s) of the data are not in it "
            f"and are not scored; {extra} of its point(s) are not in the data."
        )
        table = prediction_scores(frame, points.prediction_columns, splits, overfit_r2_gap=gap)
        result.tables[PREDICTIONS_FILE] = table
        result.overfit = sorted(table.loc[table["flag_overfit"], "property"].unique().tolist())
    missing_columns = sorted(set(wanted) - set(points.prediction_columns))
    sections.append(
        _predictions_section(result, gap, coverage_note, {target: wanted[target] for target in missing_columns})
    )

    for name, table in result.tables.items():
        path = out / name
        table.to_csv(path, index=name not in (MISSINGNESS_FILE, SMALL_CELLS_FILE, PREDICTIONS_FILE, INPUTS_FILE))
        result.files.append(path)

    header = _header(settings, config, frame, threshold, gap, split_plan_path)
    report = out / REPORT_FILE
    report.write_text("\n\n".join([header, *sections]) + "\n")
    result.files.append(report)
    logger.info(f"Pre-flight report written to {report}")
    return result


def _inputs_table(points: A1Table, settings: A1Settings, groups, logger) -> tuple[pd.DataFrame, list[str]]:
    """One row per input - its kind and covariate group - and the inputs no group claims."""
    from yg_eo_soilnet.a1.config import load_covariate_groups

    if groups is None:
        groups = load_covariate_groups(settings.covariate_groups_path) if settings.covariate_groups_path else None
    inputs = list(points.feature_columns)
    group_of: dict[str, str] = {}
    if groups is not None:
        resolved = groups.resolve(inputs, settings.auxiliary.targets, strict=False, logger=logger)
        group_of = {column: name for name, columns in resolved.groups.items() for column in columns}
    percentiles = set(points.percentile_columns)
    table = pd.DataFrame(
        {
            "input": inputs,
            "kind": ["time-series percentile" if column in percentiles else "static covariate" for column in inputs],
            "group": [group_of.get(column, NO_GROUP) for column in inputs],
        }
    )
    return table, [column for column in inputs if column not in group_of]


def _inputs_section(result: PreflightResult, settings: A1Settings) -> str:
    """The report's section on the inputs and their covariate groups."""
    table = result.tables[INPUTS_FILE]
    counts = table["group"].value_counts()
    lines = [
        "## 0. Inputs and covariate groups",
        "",
        f"{len(table)} input(s); no lab value and no `exclude_inputs` column "
        f"({', '.join(f'`{prefix}`' for prefix in settings.exclude_inputs) or 'none'}) is among them.",
        "",
        "| Group | Inputs |",
        "|---|---|",
        *(f"| {name} | {count} |" for name, count in counts.items()),
        "",
    ]
    if result.unassigned:
        lines.append(
            f"**{len(result.unassigned)} input(s) are in no covariate group**, so no model reads them: "
            + ", ".join(f"`{column}`" for column in result.unassigned)
            + ". Add them to a group in the covariate-groups file, or leave them out on purpose."
        )
    else:
        lines.append("Every input is in a covariate group.")
    if settings.percentiles.source != PERCENTILES_STATIC:
        lines.append(
            "\nThe time-series percentiles are computed later, by `run_a1.py`, so they are not listed here; "
            "the covariate groups are checked against them then."
        )
    return "\n".join(lines)


def _column_kinds(points: A1Table) -> dict[str, str]:
    """Every column the experiments read, and what kind of column it is."""
    kinds = {column: "static covariate" for column in points.static_columns}
    kinds.update({column: "time-series percentile" for column in points.percentile_columns})
    kinds.update({column: "lab value" for column in points.label_columns})
    kinds.update({column: "earlier prediction" for column in points.prediction_columns.values()})
    if points.sample_year_column:
        kinds[points.sample_year_column] = "sampling year"
    if points.region_column:
        kinds[points.region_column] = "region"
    return kinds


# --- the report ---------------------------------------------------------------------------


def _header(settings, config, frame, threshold, gap, split_plan_path) -> str:
    """The report's title and what it was run on."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# A1 pre-flight diagnostics",
        "",
        f"- Generated: {stamp}",
        f"- A1 settings: `{settings.source_path or '(built in code)'}`",
        f"- Main configuration: `{settings.main_config_path}`",
        f"- Data folder: `{getattr(config, 'DATA_FOLDER', '?')}`",
        f"- Points: {len(frame):,}",
        f"- Thresholds: a region x carbonate cell is flagged under {threshold} points; an earlier "
        f"prediction is flagged when its training R² beats its test R² by more than {gap:g}.",
        f"- Split plan: `{split_plan_path}`" if split_plan_path else "- Split plan: none given",
    ]
    return "\n".join(lines)


def _completeness_section(table: pd.DataFrame, missing: pd.DataFrame) -> str:
    """Section 1 of the report."""
    complete = table.loc[ALL_MEASURED]
    lines = [
        "## 1. Completeness",
        "",
        f"**{int(complete['n_measured']):,} of {int(complete['n_points']):,} points "
        f"({complete['share_measured']:.1%}) have every property measured.**",
        "",
        _markdown(table.assign(share_measured=table["share_measured"].map("{:.1%}".format))),
        "",
        "Missing values per column, emptiest first (columns with none missing are left out here; "
        f"all are in `{MISSINGNESS_FILE}`):",
        "",
    ]
    gappy = missing[missing["n_missing"] > 0]
    if gappy.empty:
        lines.append("No column has a missing value.")
    else:
        shown = gappy.assign(share_missing=gappy["share_missing"].map("{:.1%}".format))
        lines.append(_markdown(shown, index=False))
    return "\n".join(lines)


def _carbonate_section(result: PreflightResult, threshold: int, classes: list) -> str:
    """Section 2 of the report."""
    lines = ["## 2. Regions x FAO carbonate class", ""]
    if REGION_BY_CARBONATE_FILE not in result.tables:
        return "\n".join([*lines, f"Skipped: {result.skipped['region_by_carbonate']}"])
    table = result.tables[REGION_BY_CARBONATE_FILE]
    flagged = {(row.region, row.carbonate_class) for row in result.small_cells.itertuples()}

    def mark(region, label, value):
        """A count, marked when its cell is under the threshold."""
        return f"{value} (!)" if (region, label) in flagged else str(value)

    lines += [
        f"Points per region and class of measured CaCO3. Cells marked (!) hold fewer than {threshold} points.",
        "",
        _markdown(table, mark=mark),
        "",
    ]
    if result.small_cells.empty:
        lines.append(f"Every region x class cell holds at least {threshold} points.")
    else:
        lines.append(f"**{len(result.small_cells)} cell(s) under {threshold} points:**")
        lines += [f"- {row.region}, {row.carbonate_class}: {row.n}" for row in result.small_cells.itertuples()]
    return "\n".join(lines)


def _year_section(result: PreflightResult) -> str:
    """Section 3 of the report."""
    lines = ["## 3. Regions x sampling year", ""]
    if REGION_BY_YEAR_FILE not in result.tables:
        return "\n".join([*lines, f"Skipped: {result.skipped['region_by_year']}"])
    return "\n".join([*lines, _markdown(result.tables[REGION_BY_YEAR_FILE])])


def _predictions_section(result: PreflightResult, gap: float, coverage_note: str, missing: dict) -> str:
    """Section 4 of the report."""
    lines = ["## 4. Earlier predictions of the auxiliary properties", ""]
    if missing:
        listed = ", ".join(f"`{column}` ({target})" for target, column in missing.items())
        lines += [f"Not in the data: {listed}.", ""]
    if PREDICTIONS_FILE not in result.tables:
        return "\n".join([*lines, f"Skipped: {result.skipped['auxiliary_predictions']}"])
    table = result.tables[PREDICTIONS_FILE]
    shown = table.assign(
        r2=table["r2"].map(_number), rmse=table["rmse"].map(_number), r2_gap=table["r2_gap"].map(_number)
    )
    lines += [coverage_note, "", _markdown(shown, index=False), ""]
    if result.overfit:
        lines.append(
            f"**Flagged: {', '.join(result.overfit)}.** The training R² beats the test R² by more than "
            f"{gap:g}. That is what predictions look like when the model that made them was fitted on "
            "these training points' lab values. Used as inputs for SOM on this split, they would carry "
            "information the test points cannot have; A1 predicts these properties itself, fold by fold, "
            "for that reason."
        )
    else:
        lines.append(f"No property is flagged: the training R² is within {gap:g} of the test R² for each.")
    return "\n".join(lines)


def _number(value) -> str:
    """A score to three decimals, or a dash when there is none."""
    return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.3f}"


def _markdown(table: pd.DataFrame, *, index: bool = True, mark: Optional[Callable] = None) -> str:
    """A table as GitHub markdown; ``mark(row, column, value)`` may rewrite a cell."""
    columns = [str(column) for column in table.columns]
    header = ([str(table.index.name or "")] if index else []) + columns
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row_label, row in table.iterrows():
        cells = []
        for column, value in row.items():
            text = mark(row_label, column, value) if mark is not None else str(value)
            cells.append(text)
        lines.append("| " + " | ".join(([str(row_label)] if index else []) + cells) + " |")
    return "\n".join(lines)
