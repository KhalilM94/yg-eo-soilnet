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
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

from yg_eo_soilnet.a1.settings import A1Settings

#: The prefix of every time-series summary column, so they can never collide with a static one.
PERCENTILE_PREFIX = "ts_"
#: How many dated readings a point has, over every variable.
N_OBS_COLUMN = "ts_n_obs"


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
        Indexed by point id. Holds the inputs, the lab columns A1 uses, the coordinates and, when
        regions come from a column, that column.
    static_columns : tuple of str
        The static covariates, chosen by the same rules as ``main.py``.
    percentile_columns : tuple of str
        The time-series percentiles, and :data:`N_OBS_COLUMN`.
    categorical_columns : tuple of str
        The static covariates that are classes rather than numbers.
    label_columns : tuple of str
        The lab columns A1 uses: SOM, the properties predicted as extra inputs, and CaCO3.
    lat_column, lon_column : str
        The coordinate columns.
    region_column : str or None
        The region column, when regions come from the data.
    """

    frame: pd.DataFrame
    static_columns: tuple[str, ...]
    percentile_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    lat_column: str
    lon_column: str
    region_column: Optional[str] = None

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


def build_table(settings: A1Settings, config=None, logger: Optional[logging.Logger] = None) -> A1Table:
    """Read the data named by the main configuration and assemble the :class:`A1Table`.

    The static covariates are exactly what ``main.py`` would use as inputs: ids, coordinates and
    every lab column are set aside by
    :meth:`DataManager.filter_schema <yg_eo_soilnet.data_manager.DataManager.filter_schema>`. The
    region column is set aside too, since it names where a point is rather than describing it.

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
        If there is no time series to summarise, a point id repeats, or a lab column A1 needs is
        not in the data.
    """
    from yg_eo_soilnet.data_manager import DataManager

    logger = logger or logging.getLogger(__name__)
    config = config if config is not None else settings.load_config()
    manager = DataManager(config, logger)
    dataset = manager.load_dataset()
    tabular = dataset.tabular
    point_col = dataset.point_id_column

    duplicated = tabular[point_col][tabular[point_col].duplicated()].unique()
    if len(duplicated):
        raise ValueError(f"{len(duplicated)} point id(s) repeat in the data, e.g. {list(duplicated[:5])}")
    tabular = tabular.set_index(point_col, drop=False)

    needed_labels = list(dict.fromkeys([settings.som_target, *settings.auxiliary.targets, settings.carbonate.column]))
    missing_labels = [column for column in needed_labels if column not in tabular.columns]
    if missing_labels:
        raise ValueError(
            f"Lab column(s) {missing_labels} are not in the data. A1 needs them joined from the targets "
            "file: list them in LABEL_COLUMNS (or TARGET_COLUMNS) and set CARRY_LABEL_COLUMNS: true in "
            "the data spec of the main configuration."
        )

    region_column = settings.regions.column
    if region_column and region_column not in tabular.columns:
        tabular = tabular.assign(**{region_column: _region_from_targets(manager, region_column, point_col, tabular)})

    static = manager.filter_schema(tabular, target_columns=[settings.som_target])
    static = static.drop(columns=[column for column in (region_column,) if column], errors="ignore")
    categorical = [column for column in getattr(config, "CATEGORICAL_FEATURES", []) or [] if column in static]

    timeseries = dataset.timeseries
    if timeseries is None:
        raise ValueError(
            "The main configuration has no time series switched on, and A1 summarises it. Set "
            "temporal.enabled: true and data.timeseries in the main configuration."
        )
    percentiles = timeseries_percentiles(
        timeseries,
        point_id_column=point_col,
        quantiles=settings.percentiles.quantiles,
        time_column=getattr(config, "TIME_COLUMN", None),
        variables=settings.percentiles.variables,
        exclude=settings.percentiles.exclude,
    )
    percentiles = percentiles.reindex(tabular.index)
    without = int(percentiles[N_OBS_COLUMN].isna().sum())
    percentiles[N_OBS_COLUMN] = percentiles[N_OBS_COLUMN].fillna(0).astype(int)
    if without:
        logger.warning(
            f"{without} of {len(tabular)} point(s) have no time series; their percentiles are left "
            "empty for the models to fill."
        )

    lat_column, lon_column = dataset.lat_column, dataset.lon_column
    extra = [column for column in (lat_column, lon_column, region_column) if column and column in tabular.columns]
    frame = pd.concat([static, percentiles, tabular[needed_labels + extra]], axis=1)
    frame.index.name = point_col

    logger.info(
        f"A1 table: {len(frame)} point(s), {static.shape[1]} static covariate(s), "
        f"{percentiles.shape[1]} time-series summary column(s); "
        f"{int(frame[settings.som_target].notna().sum())} with a measured {settings.som_target}."
    )
    return A1Table(
        frame=frame,
        static_columns=tuple(static.columns),
        percentile_columns=tuple(percentiles.columns),
        categorical_columns=tuple(categorical),
        label_columns=tuple(needed_labels),
        lat_column=lat_column,
        lon_column=lon_column,
        region_column=region_column or None,
    )


def _region_from_targets(manager, column: str, point_col: str, tabular: pd.DataFrame) -> pd.Series:
    """The region column, read from the targets file when the static file does not hold it.

    The data loader joins only the targets, the lab columns and the coordinates from the targets
    file, so a region column kept there would otherwise never arrive.

    Raises
    ------
    ValueError
        If neither file holds the column.
    """
    targets = manager._load_targets()
    if column not in targets.columns or point_col not in targets.columns:
        raise ValueError(f"regions.column {column!r} is in neither the static file nor the targets file.")
    regions = targets.drop_duplicates(subset=[point_col]).set_index(point_col)[column]
    return regions.reindex(tabular.index)
