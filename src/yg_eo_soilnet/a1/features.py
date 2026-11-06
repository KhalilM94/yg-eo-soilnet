"""The inputs of the A1 models: the static covariates, plus percentiles of each point's time series.

The deep-learning model reads a point's time series reading by reading. The A1 models are
scikit-learn models, which need one row per point, so each time-series variable is summarised by a
few percentiles of everything recorded at that point: the 10th percentile of NDVI is how green the
point gets in its leanest months, the 90th how green at its peak.

A percentile does not care when or how often a point was observed, which is the point: the readings
of most points are scattered over the years with gaps, and any summary that assumed one reading
per month would misread them. A point with no time series at all keeps its row, with empty
percentiles and ``ts_n_obs`` of 0; the models fill the gaps the way they fill any other.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from yg_eo_soilnet.a1.settings import REGIONS_FROM_COLUMN, REGIONS_FROM_POLYGONS, A1Settings

#: The prefix of every time-series summary column, so they can never collide with a static one.
PERCENTILE_PREFIX = "ts_"
#: How many dated readings a point has, over every variable.
N_OBS_COLUMN = "ts_n_obs"
#: The table's region column, whichever source the regions came from. Prefixed so it cannot be
#: mistaken for, or overwrite, a column of the data.
REGION_COLUMN = "a1_region"
#: The table's sampling-year column, an integer year.
SAMPLE_YEAR_COLUMN = "a1_sample_year"


def percentile_column(variable: str, quantile: float) -> str:
    """The name of one percentile column.

    Characters that are awkward in a column name - the ``/`` of a band ratio, say - become ``_``.

    Examples
    --------
    >>> percentile_column("S2_NDVI", 10)
    'ts_S2_NDVI_p10'
    >>> percentile_column("S1_asc_VH/VV", 50)
    'ts_S1_asc_VH_VV_p50'
    >>> percentile_column("CLIM_ETa", 2.5)
    'ts_CLIM_ETa_p2.5'
    """
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(variable)).strip("_")
    return f"{PERCENTILE_PREFIX}{safe}_p{float(quantile):g}"


def timeseries_percentiles(
    timeseries: pd.DataFrame,
    *,
    point_id_column: str,
    quantiles: Sequence[float],
    time_column: Optional[str] = None,
    variables: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Summarise each point's time series by percentiles of every variable over its whole record.

    Missing readings are left out of each variable's percentiles rather than counted, so a point
    observed through clouds on half its dates is summarised from the half it has.

    Parameters
    ----------
    timeseries : pandas.DataFrame
        One row per point per date.
    point_id_column : str
        The point id column.
    quantiles : sequence of float
        The percentiles to take, between 0 and 100.
    time_column : str, optional
        The date column, which is never summarised.
    variables : iterable of str, optional
        The columns to summarise; every numeric column apart from the id and the date unless given.
    exclude : iterable of str
        Columns never summarised.

    Returns
    -------
    pandas.DataFrame
        One row per point, indexed by point id: one column per variable and percentile (named by
        :func:`percentile_column`) and :data:`N_OBS_COLUMN`.

    Raises
    ------
    KeyError
        If the id column, or a variable asked for, is not in the table.
    ValueError
        If there is nothing to summarise, or two variables' names become the same column name.

    Examples
    --------
    >>> import pandas as pd
    >>> readings = pd.DataFrame({"uuid": ["a", "a", "a", "b"], "ndvi": [0.1, 0.3, 0.5, 0.2]})
    >>> summary = timeseries_percentiles(readings, point_id_column="uuid", quantiles=[50])
    >>> summary.to_dict(orient="index")
    {'a': {'ts_ndvi_p50': 0.3, 'ts_n_obs': 3}, 'b': {'ts_ndvi_p50': 0.2, 'ts_n_obs': 1}}
    """
    if point_id_column not in timeseries.columns:
        raise KeyError(f"The time series has no point id column {point_id_column!r}")

    excluded = {point_id_column, *(exclude or ())}
    if time_column:
        excluded.add(time_column)
    if variables is None:
        chosen = [
            column
            for column in timeseries.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(timeseries[column])
        ]
    else:
        chosen = [column for column in variables if column not in excluded]
        missing = [column for column in chosen if column not in timeseries.columns]
        if missing:
            raise KeyError(f"Time-series variable(s) {missing} are not in the time series")
    if not chosen:
        raise ValueError("No time-series variable left to summarise; check timeseries_percentiles.variables/exclude.")

    names = {(variable, q): percentile_column(variable, q) for variable in chosen for q in quantiles}
    clashing = pd.Series(list(names.values())).duplicated()
    if clashing.any():
        repeated = sorted(set(pd.Series(list(names.values()))[clashing]))
        raise ValueError(
            f"Two time-series variables become the same column name {repeated}; rename one of them, "
            "or leave one out with timeseries_percentiles.exclude."
        )

    grouped = timeseries.groupby(point_id_column, sort=False)
    # One pass for every variable and percentile: rows are (point, quantile), unstacked to columns.
    fractions = [float(q) / 100.0 for q in quantiles]
    summary = grouped[chosen].quantile(fractions).unstack(level=-1)
    by_fraction = dict(zip(fractions, quantiles))
    summary.columns = [names[(variable, by_fraction[fraction])] for variable, fraction in summary.columns]
    # In a fixed order - variable by variable, percentiles rising - whatever order pandas returns.
    summary = summary[[names[(variable, q)] for variable in chosen for q in quantiles]]
    summary[N_OBS_COLUMN] = grouped.size().reindex(summary.index).astype(int)
    summary.index.name = point_id_column
    return summary


@dataclass(frozen=True)
class A1Table:
    """One row per point: everything the A1 experiments read, and which column is which.

    Attributes
    ----------
    frame : pandas.DataFrame
        Indexed by point id. Holds the inputs, the lab columns A1 uses, the coordinates, and - when
        they are configured and found - the region (:data:`REGION_COLUMN`), the sampling year
        (:data:`SAMPLE_YEAR_COLUMN`) and earlier models' predictions.
    static_columns : tuple of str
        The static covariates, chosen by the same rules as ``main.py``.
    percentile_columns : tuple of str
        The time-series percentiles, and :data:`N_OBS_COLUMN`; empty from :func:`load_points`.
    categorical_columns : tuple of str
        The static covariates that are classes rather than numbers.
    label_columns : tuple of str
        The lab columns A1 uses: SOM, the properties predicted as extra inputs, and CaCO3.
    lat_column, lon_column : str
        The coordinate columns.
    region_column : str or None
        :data:`REGION_COLUMN` when regions are configured, otherwise None.
    sample_year_column : str or None
        :data:`SAMPLE_YEAR_COLUMN` when the sampling year is configured, otherwise None.
    prediction_columns : dict of str to str
        Lab column to the column holding an earlier model's prediction of it, for the ones found in
        the data. Never inputs.
    """

    frame: pd.DataFrame
    static_columns: tuple[str, ...]
    percentile_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    lat_column: str
    lon_column: str
    region_column: Optional[str] = None
    sample_year_column: Optional[str] = None
    prediction_columns: Mapping[str, str] = field(default_factory=dict)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """The static covariates and the percentiles together: the inputs of every SOM model."""
        return self.static_columns + self.percentile_columns

    @property
    def coordinates(self) -> pd.DataFrame:
        """Each point's position, as ``lat``/``lon`` columns indexed by point id."""
        return pd.DataFrame(
            {"lat": self.frame[self.lat_column], "lon": self.frame[self.lon_column]}, index=self.frame.index
        )


def load_points(settings: A1Settings, config=None, logger: Optional[logging.Logger] = None) -> A1Table:
    """Read every point's static covariates, lab values, region and year - everything but the time series.

    What the diagnostics need, and the first half of :func:`build_table`. The static covariates are
    exactly what ``main.py`` would use as inputs: ids, coordinates and every lab column are set aside
    by :meth:`DataManager.filter_schema <yg_eo_soilnet.data_manager.DataManager.filter_schema>`. So
    are the region, the sampling-date column and earlier models' predictions, which say where, when
    and what someone else predicted rather than what the soil is like.

    Parameters
    ----------
    settings : A1Settings
        The A1 settings.
    config : Config, optional
        The main configuration; read from ``settings.main_config_path`` unless given.
    logger : logging.Logger, optional
        Where the counts are reported.

    Returns
    -------
    A1Table
        With no percentile columns.

    Raises
    ------
    ValueError
        If a point id repeats, or a lab column A1 needs, the region column or the sampling-year
        column is not in the data.
    """
    logger = logger or logging.getLogger(__name__)
    config = config if config is not None else settings.load_config()
    return _load_points(settings, config, logger)[0]


def _load_points(settings: A1Settings, config, logger: logging.Logger):
    """:func:`load_points`, also handing back the loaded dataset so its time series can be read next."""
    from yg_eo_soilnet.data_manager import DataManager

    manager = DataManager(config, logger)
    dataset = manager.load_dataset()
    tabular = dataset.tabular
    point_col = dataset.point_id_column

    duplicated = tabular[point_col][tabular[point_col].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"{len(duplicated)} point id(s) repeat in the data, e.g. {list(duplicated[:5])}")
    tabular = tabular.set_index(point_col, drop=False)

    needed_labels = list(dict.fromkeys([settings.target, *settings.auxiliary.targets, settings.carbonate.column]))
    missing_labels = [column for column in needed_labels if column not in tabular.columns]
    if missing_labels:
        raise ValueError(
            f"Lab column(s) {missing_labels} are not in the data. A1 needs them joined from the targets "
            "file: list them in LABEL_COLUMNS (or TARGET_COLUMNS) and set CARRY_LABEL_COLUMNS: true in "
            "the data spec of the main configuration."
        )

    # Read before the inputs are chosen, so that each of these source columns can be left out of them.
    not_inputs: set[str] = set()
    extra: dict[str, pd.Series] = {}
    regions = settings.regions
    if regions.source == REGIONS_FROM_COLUMN:
        extra[REGION_COLUMN] = _column_from_files(manager, tabular, regions.column, point_col, "regions.column")
        not_inputs.add(regions.column)
    elif regions.source == REGIONS_FROM_POLYGONS:
        from yg_eo_soilnet.a1.regions import regions_from_polygons, resolve_polygons_path

        extra[REGION_COLUMN] = regions_from_polygons(
            tabular,
            resolve_polygons_path(regions.polygons_path, getattr(config, "DATA_FOLDER", None)),
            regions.name_field,
            lat_column=dataset.lat_column,
            lon_column=dataset.lon_column,
            logger=logger,
        )

    year_settings = settings.sample_year
    if year_settings.column:
        if year_settings.file:
            raw_years = _column_from_table_file(
                manager, year_settings.file, year_settings.column, point_col, tabular, config, logger
            )
        else:
            raw_years = _column_from_files(manager, tabular, year_settings.column, point_col, "sample_year.column")
            not_inputs.add(year_settings.column)
        extra[SAMPLE_YEAR_COLUMN] = sample_years(raw_years, logger=logger)

    prediction_columns = {}
    for target, column in settings.preflight.prediction_columns.items():
        not_inputs.add(column)
        values = _column_from_files(manager, tabular, column, point_col, "preflight.prediction_columns", required=False)
        if values is None:
            logger.info(f"No column {column!r} (a prediction of {target}) in the data.")
            continue
        extra[column] = pd.to_numeric(values, errors="coerce")
        prediction_columns[target] = column

    static = manager.filter_schema(tabular, target_columns=[settings.target])
    static = static.drop(columns=sorted(not_inputs & set(static.columns)))
    categorical = [column for column in getattr(config, "CATEGORICAL_FEATURES", []) or [] if column in static]

    lat_column, lon_column = dataset.lat_column, dataset.lon_column
    coordinates = [column for column in (lat_column, lon_column) if column in tabular.columns]
    frame = pd.concat([static, tabular[needed_labels + coordinates], pd.DataFrame(extra, index=tabular.index)], axis=1)
    frame.index.name = point_col

    logger.info(
        f"A1 points: {len(frame)}, {static.shape[1]} static covariate(s); "
        f"{int(frame[settings.target].notna().sum())} with a measured {settings.target}."
    )
    table = A1Table(
        frame=frame,
        static_columns=tuple(static.columns),
        percentile_columns=(),
        categorical_columns=tuple(categorical),
        label_columns=tuple(needed_labels),
        lat_column=lat_column,
        lon_column=lon_column,
        region_column=REGION_COLUMN if REGION_COLUMN in extra else None,
        sample_year_column=SAMPLE_YEAR_COLUMN if SAMPLE_YEAR_COLUMN in extra else None,
        prediction_columns=prediction_columns,
    )
    return table, dataset


def build_table(settings: A1Settings, config=None, logger: Optional[logging.Logger] = None) -> A1Table:
    """Read the data named by the main configuration and assemble the full :class:`A1Table`.

    :func:`load_points`, with the time-series percentiles joined on.

    Parameters
    ----------
    settings : A1Settings
        The A1 settings.
    config : Config, optional
        The main configuration; read from ``settings.main_config_path`` unless given.
    logger : logging.Logger, optional
        Where the counts are reported.

    Returns
    -------
    A1Table

    Raises
    ------
    ValueError
        If there is no time series to summarise, or :func:`load_points` refuses the data.
    """
    logger = logger or logging.getLogger(__name__)
    config = config if config is not None else settings.load_config()
    points, dataset = _load_points(settings, config, logger)

    timeseries = dataset.timeseries
    if timeseries is None:
        raise ValueError(
            "The main configuration has no time series switched on, and A1 summarises it. Set "
            "temporal.enabled: true and data.timeseries in the main configuration."
        )
    percentiles = timeseries_percentiles(
        timeseries,
        point_id_column=dataset.point_id_column,
        quantiles=settings.percentiles.quantiles,
        time_column=getattr(config, "TIME_COLUMN", None),
        variables=settings.percentiles.variables,
        exclude=settings.percentiles.exclude,
    )
    percentiles = percentiles.reindex(points.frame.index)
    without = int(percentiles[N_OBS_COLUMN].isna().sum())
    percentiles[N_OBS_COLUMN] = percentiles[N_OBS_COLUMN].fillna(0).astype(int)
    if without:
        logger.warning(
            f"{without} of {len(percentiles)} point(s) have no time series; their percentiles are left "
            "empty for the models to fill."
        )

    static = points.frame[list(points.static_columns)]
    rest = points.frame.drop(columns=list(points.static_columns))
    frame = pd.concat([static, percentiles, rest], axis=1)
    logger.info(f"A1 table: {percentiles.shape[1]} time-series summary column(s) added.")
    return replace(points, frame=frame, percentile_columns=tuple(percentiles.columns))


def sample_years(values: pd.Series, logger: Optional[logging.Logger] = None) -> pd.Series:
    """Turn a column of sampling dates, or of years, into whole years.

    A numeric column is taken to hold years already. Anything else is read as dates - in any of the
    usual layouts, ``2019/12/31`` or ``2019-12-31`` - and a value that cannot be read is left
    missing and counted.

    Examples
    --------
    >>> import pandas as pd
    >>> sample_years(pd.Series(["2019/12/31", "2018-03-02", None])).tolist()
    [2019, 2018, <NA>]
    >>> sample_years(pd.Series([2017, 2020.0])).tolist()
    [2017, 2020]
    """
    logger = logger or logging.getLogger(__name__)
    if pd.api.types.is_numeric_dtype(values):
        return values.round().astype("Int64")
    dates = pd.to_datetime(values, format="mixed", errors="coerce")
    unreadable = int((dates.isna() & values.notna()).sum())
    if unreadable:
        logger.warning(f"{unreadable} sampling date(s) could not be read and are left without a year.")
    return dates.dt.year.astype("Int64")


def _column_from_files(manager, tabular: pd.DataFrame, column: str, point_col: str, setting: str, *, required=True):
    """A column of the joined data, or else of the targets file, aligned on the points.

    The data loader joins only the targets, the lab columns and the coordinates from the targets
    file, so any other column kept there - a region, a date - would otherwise never arrive.

    Returns
    -------
    pandas.Series or None
        None when neither file has the column and ``required`` is false.

    Raises
    ------
    ValueError
        If neither file has the column and it is required.
    """
    if column in tabular.columns:
        return tabular[column]
    targets = manager._load_targets()
    if column in targets.columns and point_col in targets.columns:
        return targets.drop_duplicates(subset=[point_col]).set_index(point_col)[column].reindex(tabular.index)
    if not required:
        return None
    raise ValueError(f"{setting} {column!r} is in neither the static file nor the targets file.")


def _column_from_table_file(manager, path: str, column: str, point_col: str, tabular, config, logger) -> pd.Series:
    """A column of a separate table, joined to the points by id.

    A point listed several times - one row per sampled depth, say - keeps its first value, and the
    points whose rows disagree are counted.

    Raises
    ------
    ValueError
        If the table lacks the point id or the column.
    """
    data_folder = getattr(config, "DATA_FOLDER", None)
    resolved = path if os.path.isabs(path) or not data_folder else os.path.join(data_folder, path)
    table = manager._read_tabular_file(os.path.abspath(resolved), missing_label="sample_year.file")
    missing = [name for name in (point_col, column) if name not in table.columns]
    if missing:
        raise ValueError(f"sample_year.file {path!r} has no column(s) {missing}")
    rows = table[[point_col, column]].dropna(subset=[point_col])
    disagreeing = int((rows.groupby(point_col)[column].nunique(dropna=True) > 1).sum())
    if disagreeing:
        logger.warning(
            f"{disagreeing} point(s) have more than one {column!r} in {os.path.basename(path)}; each keeps its first."
        )
    return rows.drop_duplicates(subset=[point_col]).set_index(point_col)[column].reindex(tabular.index)
