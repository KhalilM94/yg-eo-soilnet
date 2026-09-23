"""The container for the loaded data: one table per point, plus the time series on demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


@dataclass
class SoilDataset:
    """The loaded data, the same whatever way it was stored on disk.

    Created by :meth:`DataManager.load_dataset <yg_eo_soilnet.data_manager.DataManager.load_dataset>`.

    Attributes
    ----------
    tabular : pandas.DataFrame
        One row per point: the static covariates with the targets (and, if carried, the other lab
        columns and the coordinates) already joined on.
    point_id_column, lat_column, lon_column : str
        Names of the id and coordinate columns.
    target_columns : list of str
        The targets being predicted.
    temporal_enabled : bool
        Whether a time series is configured.
    timeseries : pandas.DataFrame or None
        One row per point per date. Read from disk the first time it is used, so a run with only
        scikit-learn models never reads it.
    """

    tabular: pd.DataFrame
    point_id_column: str
    lat_column: str
    lon_column: str
    target_columns: list[str]
    temporal_enabled: bool
    _load_timeseries: Callable[[], Optional[pd.DataFrame]]

    _timeseries: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _timeseries_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def timeseries(self) -> Optional[pd.DataFrame]:
        """The time-series table, read on first use and kept in memory afterwards."""
        if not self._timeseries_loaded:
            self._timeseries = self._load_timeseries()
            self._timeseries_loaded = True
        return self._timeseries

    @property
    def has_timeseries(self) -> bool:
        """Whether a time series is configured, without reading it."""
        return self.temporal_enabled

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        loaded = "loaded" if self._timeseries_loaded else "not loaded"
        return (
            f"SoilDataset(tabular={self.tabular.shape}, targets={self.target_columns}, "
            f"temporal_enabled={self.temporal_enabled}, timeseries={loaded})"
        )
