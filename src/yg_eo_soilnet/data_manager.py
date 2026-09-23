"""Read the configured data files and decide which of their columns are model inputs."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

import pandas as pd

from yg_eo_soilnet.dataset import SoilDataset


class DataManager:
    """Read the data files and pick out the columns models may use as inputs.

    There are three data sources, set under ``common.data`` in ``main_config.yml``: the *static*
    covariates, the *targets* (lab measurements, with the coordinates) and the *time series*. Each
    may be a single CSV or Parquet file, a folder of them, or a list of files in a JSON manifest
    (``data_index_manifest``).

    :meth:`load_dataset` reads them into a :class:`~yg_eo_soilnet.dataset.SoilDataset`, joining the
    targets onto the static covariates; the time series is read only when a model asks for it.
    :meth:`filter_schema` then drops everything that is not an input - ids, coordinates, lab
    measurements and the excluded columns. Preparing those inputs for each model family happens
    later, in :mod:`yg_eo_soilnet.datamodules`.

    Files are read once and kept in memory; :meth:`reload` forgets them.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where progress messages go.

    Examples
    --------
    >>> import logging
    >>> from config import Config
    >>> manager = DataManager(Config("examples/demo_config/main_config.yml"),
    ...                       logging.getLogger("demo"))
    >>> dataset = manager.load_dataset()        # doctest: +SKIP
    >>> dataset.tabular.shape                   # doctest: +SKIP
    (300, 18)
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._cache: dict[str, pd.DataFrame] = {}

    # --- schema filtering -------------------------------------------------

    def coordinate_columns(self) -> tuple[str, str]:
        """Return the latitude and longitude column names, ``"lat"`` and ``"lon"`` unless set.

        Returns
        -------
        tuple of (str, str)
        """
        return (
            getattr(self.config, "LAT_COLUMN", "lat"),
            getattr(self.config, "LON_COLUMN", "lon"),
        )

    def context_feature_columns(self) -> list[str]:
        """Return the spatial-context covariates, or nothing when the group is switched off.

        ``CONTEXT_FEATURES`` in ``data_spec.yml`` groups ordinary numeric covariates so they can be
        switched off together (``USE_CONTEXT_FEATURES: false``), to measure how much they help, and
        reported together in the SHAP figures. While switched on they are ordinary inputs; switched
        off they join :meth:`excluded_columns`.

        Returns
        -------
        list of str
        """
        if not getattr(self.config, "USE_CONTEXT_FEATURES", True):
            return []
        return [str(column) for column in (getattr(self.config, "CONTEXT_FEATURES", []) or [])]

    def metadata_columns(self, target_columns: Optional[list[str]] = None, *, include_targets: bool = True) -> set[str]:
        """Return the columns that identify or measure a point rather than describe it.

        Never model inputs: the point id, the coordinates, ``geometry``, the targets and every
        :term:`lab column`. The coordinates stay here even with ``USE_HARMONIC_COORDS``, which lets
        `soil_cnn` read them through its own location branch: they are still not ordinary inputs.

        Parameters
        ----------
        target_columns : list of str, optional
            The targets of this model; ``TARGET_COLUMNS`` unless given.
        include_targets : bool, default True
            Whether the targets and lab columns are included.

        Returns
        -------
        set of str
        """
        lat_column, lon_column = self.coordinate_columns()
        columns = {
            getattr(self.config, "POINT_ID_COLUMN", "point_id"),
            lat_column,
            lon_column,
            "geometry",
        }
        if include_targets:
            active = target_columns if target_columns is not None else list(getattr(self.config, "TARGET_COLUMNS", []))
            # Every lab column, not only the targets being predicted: taking a target out of
            # TARGET_COLUMNS must not turn it into an input for the others.
            columns.update(active)
            columns.update(getattr(self.config, "LABEL_COLUMNS", []) or [])
        return columns

    def excluded_columns(self, columns: Iterable[str]) -> set[str]:
        """Return the columns the configuration excludes from the inputs.

        ``ELIMINATED_FEATURES``, ``EXCLUDE_CATEGORICAL``, the ignored bands (see
        :meth:`hyperspectral_drop_columns`), and the spatial-context group when it is switched off.

        Parameters
        ----------
        columns : iterable of str
            The columns present in the data, needed to match band-name prefixes.

        Returns
        -------
        set of str
        """
        declared_context = {str(column) for column in (getattr(self.config, "CONTEXT_FEATURES", []) or [])}
        return (
            set(getattr(self.config, "ELIMINATED_FEATURES", []) or [])
            | set(getattr(self.config, "EXCLUDE_CATEGORICAL", []) or [])
            | (declared_context - set(self.context_feature_columns()))
            | self.hyperspectral_drop_columns(columns)
        )

    def filter_schema(self, frame: pd.DataFrame, target_columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Return a copy of ``frame`` holding only the columns models may use as inputs.

        Drops :meth:`metadata_columns` and :meth:`excluded_columns`.

        Parameters
        ----------
        frame : pandas.DataFrame
            One row per point.
        target_columns : list of str, optional
            The targets of this model; ``TARGET_COLUMNS`` unless given.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> import logging, pandas as pd
        >>> from types import SimpleNamespace
        >>> config = SimpleNamespace(POINT_ID_COLUMN="uuid", TARGET_COLUMNS=["clay_pct"],
        ...                          LABEL_COLUMNS=["clay_pct", "ph_water"])
        >>> manager = DataManager(config, logging.getLogger("demo"))
        >>> frame = pd.DataFrame(
        ...     columns=["uuid", "lat", "lon", "elevation", "slope", "clay_pct", "ph_water"])
        >>> list(manager.filter_schema(frame).columns)
        ['elevation', 'slope']
        """
        drop_columns = self.metadata_columns(target_columns) | self.excluded_columns(frame.columns)
        columns_to_drop = sorted(column for column in frame.columns if column in drop_columns)
        if not columns_to_drop:
            return frame.copy()
        return frame.drop(columns=columns_to_drop, errors="ignore")

    def hyperspectral_drop_columns(self, columns: Iterable[str]) -> set[str]:
        """Return the spectral band columns to drop.

        With ``existing_hs_features`` both ``enabled`` and ``ignore``, that is its listed
        ``band_names`` plus every column starting with its ``prefix`` (one string, not a list).
        Columns named in the older ``IGNORE_BANDS`` setting are dropped too.

        Parameters
        ----------
        columns : iterable of str
            The columns present in the data.

        Returns
        -------
        set of str
        """
        drop_columns: set[str] = set()
        existing_hs = self.normalize_mapping(getattr(self.config, "EXISTING_HS_FEATURES", {}))
        if existing_hs.get("enabled", False) and existing_hs.get("ignore", False):
            band_names = existing_hs.get("band_names", [])
            if isinstance(band_names, (list, tuple, set)):
                drop_columns.update({str(name) for name in band_names if name})

            prefix = existing_hs.get("prefix", "")
            if prefix:
                drop_columns.update({str(column) for column in columns if str(column).startswith(str(prefix))})

        legacy_ignore_bands = getattr(self.config, "IGNORE_BANDS", [])
        if isinstance(legacy_ignore_bands, (list, tuple, set)):
            drop_columns.update({str(column) for column in columns if column in legacy_ignore_bands})

        return drop_columns

    def temporal_config(self) -> dict[str, Any]:
        """Return the ``temporal:`` settings as a dict (empty when there are none)."""
        return self.normalize_mapping(getattr(self.config, "TEMPORAL_FEATURES", {}) or {})

    @staticmethod
    def normalize_mapping(value: Any) -> dict:
        """Return a setting as a dict, whether it arrived as a dict or as JSON text.

        Anything else, unreadable JSON included, gives an empty dict.

        Examples
        --------
        >>> DataManager.normalize_mapping('{"enabled": true}')
        {'enabled': True}
        >>> DataManager.normalize_mapping(None)
        {}
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # --- path resolution and file reading ---------------------------------

    @staticmethod
    def _is_tabular_file(file_name: str) -> bool:
        """Whether a file name ends in .csv, .parquet or .pq."""
        lower_name = file_name.lower()
        return lower_name.endswith((".csv", ".parquet", ".pq"))

    def _resolve_data_path(self, path_value: Optional[str]) -> Optional[str]:
        """Return an absolute path; a relative one is read inside the data folder."""
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value
        # Config hands over absolute paths already, so resolving one twice does no harm.
        return os.path.abspath(os.path.join(self.config.DATA_FOLDER, path_value))

    def _read_tabular_file(self, file_path: str, missing_label: str = "Tabular file") -> pd.DataFrame:
        """Read one CSV or Parquet file."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{missing_label} not found: {file_path}")
        lower_path = file_path.lower()
        if lower_path.endswith((".parquet", ".pq")):
            return pd.read_parquet(file_path)
        if lower_path.endswith(".csv"):
            return pd.read_csv(file_path)
        raise ValueError(f"Unsupported tabular file format: {file_path}")

    def _load_data_index_manifest(self) -> Optional[dict[str, Any]]:
        """Read the JSON manifest listing the data files, if one is configured."""
        manifest_path = getattr(self.config, "DATA_INDEX_MANIFEST_PATH", None)
        if not manifest_path:
            return None
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Data index manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Data index manifest must contain a JSON object: {manifest_path}")
        return manifest

    def _discover_tabular_files_one_level(self, folder_path: str) -> list[str]:
        """List the data files in a folder and its immediate subfolders, sorted by path."""
        discovered: list[str] = []
        if not os.path.isdir(folder_path):
            return discovered

        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and self._is_tabular_file(entry.name):
                    discovered.append(entry.path)
                elif entry.is_dir():
                    with os.scandir(entry.path) as nested_entries:
                        for nested_entry in nested_entries:
                            if nested_entry.is_file() and self._is_tabular_file(nested_entry.name):
                                discovered.append(nested_entry.path)

        return sorted(discovered)

    def _normalize_manifest_paths(self, paths: Any) -> list[str]:
        """Return a manifest entry - one path or a list of them - as a list."""
        if paths is None:
            return []
        if isinstance(paths, str):
            return [paths]
        if isinstance(paths, (list, tuple)):
            return [str(path) for path in paths if path]
        return []

    def _load_tabular_paths(self, paths: list[str]) -> pd.DataFrame:
        """Read several files and stack their rows into one table."""
        if not paths:
            raise FileNotFoundError("No tabular files were discovered")
        frames = [self._read_tabular_file(path) for path in paths]
        return pd.concat(frames, ignore_index=True, sort=False)

    def _load_folder_or_file(self, source_path: Optional[str], manifest_paths: Optional[list[str]] = None) -> pd.DataFrame:
        """Read a source given as a list of files from the manifest, one file, or a folder."""
        if manifest_paths:
            absolute_paths = [self._resolve_data_path(path) for path in manifest_paths]
            return self._load_tabular_paths([path for path in absolute_paths if path])

        resolved_path = self._resolve_data_path(source_path)
        if not resolved_path:
            raise FileNotFoundError("Data source path is not configured")

        if os.path.isfile(resolved_path):
            return self._read_tabular_file(resolved_path)

        if not os.path.isdir(resolved_path):
            raise FileNotFoundError(f"Data source folder not found: {resolved_path}")

        discovered_paths = self._discover_tabular_files_one_level(resolved_path)
        return self._load_tabular_paths(discovered_paths)

    def _load_manifest_source_paths(self, manifest: dict[str, Any], manifest_key: str) -> list[str]:
        """Return the manifest's file list for one source, matching the key in any letter case."""
        if not manifest:
            return []
        for key in (manifest_key, manifest_key.lower(), manifest_key.upper()):
            if key in manifest:
                return self._normalize_manifest_paths(manifest[key])
        return []

    def _load_source(self, source_path: Optional[str], manifest_key: str, *, label: str) -> pd.DataFrame:
        """Read one source - static covariates or targets - and keep it in memory.

        The manifest is consulted first, so its file list is used whether the configured path is a
        file or a folder.
        """
        cache_key = f"{manifest_key}:{source_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        manifest = self._load_data_index_manifest()
        manifest_paths = self._load_manifest_source_paths(manifest or {}, manifest_key)
        if not manifest_paths and not source_path:
            raise FileNotFoundError(f"{label} source is not configured")

        self.logger.info(f"Loading {label} from {'manifest' if manifest_paths else source_path}")
        frame = self._load_folder_or_file(source_path, manifest_paths=manifest_paths)
        self._cache[cache_key] = frame
        return frame

    # --- loaders ----------------------------------------------------------

    def load_dataset(self) -> SoilDataset:
        """Read the data and return it as a :class:`~yg_eo_soilnet.dataset.SoilDataset`.

        The static covariates and the targets may share one file or sit in two; see
        :meth:`load_tabular_data`. The time series is read the first time a model asks for it.

        Returns
        -------
        SoilDataset

        Examples
        --------
        >>> dataset = manager.load_dataset()                 # doctest: +SKIP
        >>> len(dataset.tabular), dataset.target_columns     # doctest: +SKIP
        (300, ['organic_matter_g_kg', 'clay_pct', 'ph_water'])
        """
        lat_column, lon_column = self.coordinate_columns()
        return SoilDataset(
            tabular=self.load_tabular_data(),
            point_id_column=getattr(self.config, "POINT_ID_COLUMN", "point_id"),
            lat_column=lat_column,
            lon_column=lon_column,
            target_columns=list(getattr(self.config, "TARGET_COLUMNS", [])),
            temporal_enabled=self._temporal_enabled(),
            _load_timeseries=self.load_timeseries_data,
        )

    def reload(self) -> None:
        """Forget the files read so far, so the next load reads them from disk again."""
        self._cache.clear()

    def _static_source(self) -> Optional[str]:
        """The configured path to the static covariates, trying the older setting names too."""
        return (
            getattr(self.config, "STATIC_SOURCE", None)
            or getattr(self.config, "STATIC_FEATURES_FOLDER", None)
            or getattr(self.config, "STATIC_CSV_PATH", None)
            or self._resolve_data_path(getattr(self.config, "DATA_FILE", "data.csv"))
        )

    def _targets_source(self) -> Optional[str]:
        """The configured path to the targets, trying the older setting names too."""
        # With no targets setting this falls back to the static path, which is only read when the
        # static file turned out not to hold the targets.
        return (
            getattr(self.config, "TARGETS_SOURCE", None)
            or getattr(self.config, "TARGETS_FOLDER", None)
            or getattr(self.config, "TARGETS_CSV_PATH", None)
            or self._resolve_data_path(
                getattr(self.config, "TARGETS_FILE", getattr(self.config, "DATA_FILE", "data.csv"))
            )
        )

    def _timeseries_source(self) -> Optional[str]:
        """The configured path to the time series, from the ``temporal:`` settings or the older names."""
        temporal_config = self.temporal_config()
        return (
            temporal_config.get("timeseries_folder")
            or temporal_config.get("timeseries_file")
            or temporal_config.get("timeseries_csv_path")
            or getattr(self.config, "TIMESERIES_SOURCE", None)
            or getattr(self.config, "TIMESERIES_FOLDER", None)
            or getattr(self.config, "TIMESERIES_CSV_PATH", None)
        )

    def _temporal_enabled(self) -> bool:
        """Whether the time series is switched on *and* a source is configured.

        With ``temporal.enabled: true`` but no source, it is off and nothing says so: the
        deep-learning model then sees the static covariates only.
        """
        temporal_config = self.temporal_config()
        enabled = bool(
            temporal_config.get("enabled", getattr(self.config, "TEMPORAL_FEATURES_ENABLED", False))
        )
        return enabled and bool(self._timeseries_source())

    def _load_static(self) -> pd.DataFrame:
        """Read the static covariates; they may hold the targets as well."""
        return self._load_source(self._static_source(), "static_features", label="static data")

    def _load_targets(self) -> pd.DataFrame:
        """Read the targets file."""
        return self._load_source(self._targets_source(), "targets", label="targets")

    def load_tabular_data(self) -> pd.DataFrame:
        """Return one row per point: the static covariates with the targets joined on.

        If the static file already holds every target it is returned as it is. Otherwise the targets
        file is read and joined on the point id, keeping only points present in both. The
        coordinates come from the targets file when the static file lacks them, and with
        ``CARRY_LABEL_COLUMNS: true`` so do the other :term:`lab columns <lab column>`, for models
        using an :term:`auxiliary lab input`; joining them does not make them ordinary inputs.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If no targets are configured.
        KeyError
            If the point id column is missing from either file, or a target from the targets file.
        """
        static_df = self._load_static()
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))
        if not target_columns:
            raise ValueError("TARGET_COLUMNS must be configured for tabular preprocessing")

        if all(column in static_df.columns for column in target_columns):
            return static_df

        targets_df = self._load_targets()
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        if point_col not in static_df.columns:
            raise KeyError(f"Point id column '{point_col}' not found in static features file")
        if point_col not in targets_df.columns:
            raise KeyError(f"Point id column '{point_col}' not found in targets file")

        missing_targets = [column for column in target_columns if column not in targets_df.columns]
        if missing_targets:
            raise KeyError(f"Target column(s) missing from targets file: {', '.join(missing_targets)}")

        # The coordinates are often filed with the targets and not with the covariates.
        coordinate_columns = [
            column
            for column in self.coordinate_columns()
            if column not in static_df.columns and column in targets_df.columns
        ]
        if coordinate_columns:
            self.logger.info(f"Joining coordinate column(s) {coordinate_columns} from the targets source")

        # The other lab measurements, carried only when asked for. They stay out of the inputs
        # (metadata_columns drops them); only a model naming one as an auxiliary input reads it.
        label_columns = []
        if getattr(self.config, "CARRY_LABEL_COLUMNS", False):
            label_columns = [
                column
                for column in (getattr(self.config, "LABEL_COLUMNS", []) or [])
                if column in targets_df.columns and column not in static_df.columns
            ]
            if label_columns:
                self.logger.info(
                    f"Joining {len(label_columns)} measured label column(s) from the targets source "
                    "as auxiliary inputs; they remain excluded from the feature set"
                )

        # dict.fromkeys drops duplicates: every target is also a lab column, and selecting a column
        # twice would break the merge.
        join_columns = list(dict.fromkeys([point_col, *target_columns, *label_columns, *coordinate_columns]))
        dedup_targets = targets_df[join_columns].drop_duplicates(subset=[point_col])
        return static_df.merge(dedup_targets, on=point_col, how="inner")

    def load_timeseries_data(self) -> Optional[pd.DataFrame]:
        """Read the time series - one row per point per date - or None when it is switched off.

        The source may be one file, or a folder holding one file or subfolder per data source; those
        tables are merged on the point id and the date. The result is kept in memory.

        Returns
        -------
        pandas.DataFrame or None
        """
        if not self._temporal_enabled():
            return None

        source = self._timeseries_source()
        resolved_path = self._resolve_data_path(source)
        if not resolved_path:
            return None
        if os.path.isdir(resolved_path):
            return self._load_timeseries_folder(resolved_path)

        cache_key = f"timeseries:{resolved_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"Loading time-series from file: {resolved_path}")
        frame = self._read_tabular_file(resolved_path, missing_label="Time-series CSV file")
        self._cache[cache_key] = frame
        return frame

    def _load_timeseries_folder(self, timeseries_root: str) -> pd.DataFrame:
        """Read a folder of time-series files, merging the data sources on point id and date."""
        resolved_root = self._resolve_data_path(timeseries_root)
        if not resolved_root or not os.path.isdir(resolved_root):
            raise FileNotFoundError(f"Time-series folder not found: {resolved_root or timeseries_root}")

        cache_key = f"timeseries:{resolved_root}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        manifest = self._load_data_index_manifest()
        manifest_timeseries = (manifest or {}).get("timeseries", {}) if manifest else {}

        modality_frames: list[pd.DataFrame] = []
        if isinstance(manifest_timeseries, dict) and manifest_timeseries:
            for modality_name, modality_paths in manifest_timeseries.items():
                resolved_paths = [self._resolve_data_path(path) for path in self._normalize_manifest_paths(modality_paths)]
                frame_paths = [path for path in resolved_paths if path]
                if not frame_paths:
                    continue
                modality_frames.append(self._load_tabular_paths(frame_paths))
        else:
            with os.scandir(resolved_root) as modality_entries:
                for modality_entry in sorted(modality_entries, key=lambda entry: entry.name):
                    if modality_entry.is_dir():
                        shard_paths = self._discover_tabular_files_one_level(modality_entry.path)
                        if shard_paths:
                            modality_frames.append(self._load_tabular_paths(shard_paths))
                    elif modality_entry.is_file() and self._is_tabular_file(modality_entry.name):
                        modality_frames.append(self._read_tabular_file(modality_entry.path))

        if not modality_frames:
            raise FileNotFoundError(f"No time-series files were discovered under {resolved_root}")

        combined_frame = modality_frames[0]
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        time_col = self.temporal_config().get("time_column", getattr(self.config, "TIME_COLUMN", "date"))
        for frame in modality_frames[1:]:
            combined_frame = combined_frame.merge(frame, on=[point_col, time_col], how="outer")
        self._cache[cache_key] = combined_frame
        return combined_frame
