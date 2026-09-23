"""Build the bundle the deep-learning model reads: covariates, targets and dated readings."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.categorical import (
    resolve_categorical_columns,
    split_feature_blocks,
)
from yg_eo_soilnet.datamodules.frame_cleaning import (
    assert_columns_are_dense_enough,
    build_finite_row_mask,
    drop_non_finite_rows,
    sanitize_numeric_columns,
)
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

#: Days in an average year, used to place a reading inside its year.
DAYS_PER_YEAR = 365.25

# Prefix marking the measured-or-filled flags while they travel beside the readings, so every
# filter and sort applies to both.
_VALIDITY_PREFIX = "__valid__"


def to_decimal_year(dates: pd.Series) -> np.ndarray:
    """Turn dates into :term:`decimal years <decimal year>`: 2 July 2021 becomes 2021.498.

    One continuous axis for daily, monthly or irregular readings alike, so nothing downstream has to
    know how often a sensor reports. Kept at full precision, because what the model reads is the gap
    between one reading and the next.

    Parameters
    ----------
    dates : pandas.Series
        Dates, or anything pandas can read as dates. An unreadable one gives NaN.

    Returns
    -------
    numpy.ndarray of float

    Examples
    --------
    >>> import pandas as pd
    >>> to_decimal_year(pd.Series(["2021-01-01", "2021-07-02", "2022-01-01"])).round(3)
    array([2021.   , 2021.498, 2022.   ])
    """
    dates = pd.to_datetime(dates, errors="coerce")
    year = dates.dt.year.to_numpy(dtype=np.float64)
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=np.float64)
    return year + (day_of_year - 1.0) / DAYS_PER_YEAR


class SoilSequenceBuilder:
    """Build the :class:`~yg_eo_soilnet.datamodules.sequence.sequence_bundle.SoilSequenceBundle`.

    Reads the data through the same :class:`~yg_eo_soilnet.data_manager.DataManager` the
    scikit-learn family uses, so both train on the same covariates, then attaches each point's dated
    readings. It does not split the points and does not use PyTorch: the datamodule does both.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where the summary of what was built goes.
    data_manager : DataManager
        Reads the data files.

    Examples
    --------
    >>> bundle = SoilSequenceBuilder(config, logger, data_manager).build()   # doctest: +SKIP
    >>> bundle.num_points, sorted(bundle.sequences)                          # doctest: +SKIP
    (300, ['clim', 's2'])
    """

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager

    def clean_static_frame(self, static_df: pd.DataFrame, dataset) -> tuple[pd.DataFrame, Any]:
        """Keep the usable columns and rows of the static table.

        A point is dropped only for a missing id, a missing target or - when the model reads
        coordinates - a missing coordinate. A missing covariate is not a reason to drop a soil
        sample: it is filled in later and flagged. A covariate too empty to fill in honestly stops
        the run instead (see
        :func:`~yg_eo_soilnet.datamodules.frame_cleaning.assert_columns_are_dense_enough`).

        :meth:`usable_point_ids` calls this too, so the split is told exactly which points this
        family will keep.

        Parameters
        ----------
        static_df : pandas.DataFrame
            One row per point, covariates and targets together.
        dataset : SoilDataset
            The loaded data, for the id and target column names.

        Returns
        -------
        cleaned : pandas.DataFrame
            The rows that survived.
        blocks : FeatureBlocks
            Which columns are numeric and which are categories.

        Raises
        ------
        KeyError
            If a target column is missing from the data.
        """
        point_col = dataset.point_id_column
        target_columns = list(dataset.target_columns)
        missing_targets = [column for column in target_columns if column not in static_df.columns]
        if missing_targets:
            raise KeyError(f"Missing target columns in static CSV: {', '.join(missing_targets)}")

        feature_frame = self.data_manager.filter_schema(static_df, target_columns)
        blocks = resolve_categorical_columns(
            self.config, static_df, feature_frame.columns, logger=self.logger
        )
        self._assert_context_features_present(static_df, blocks.continuous_columns)
        feature_columns = list(blocks.continuous_columns)
        assert_columns_are_dense_enough(
            static_df,
            feature_columns,
            max_missing_ratio=float(getattr(self.config, "MAX_MISSING_COLUMN_RATIO", 0.2)),
            label="the static sequence source",
            logger=self.logger,
            allow=getattr(self.config, "ALLOW_SPARSE_COLUMNS", ()) or (),
            fail=bool(getattr(self.config, "FAIL_ON_SPARSE_COLUMNS", True)),
        )
        # The coordinate rule lives here, with the other row rules, so usable_point_ids reports it
        # to the split: a point the split assigns but the builder then drops would shrink the run.
        coordinate_columns = [
            column for column in self._coordinate_columns() if column in static_df.columns
        ]
        if coordinate_columns:
            # Counted separately from the drop below, which also removes points with no lab
            # measurement: those would go whatever the coordinate setting said.
            missing_coords = int(
                (~build_finite_row_mask(static_df, numeric_columns=coordinate_columns)).sum()
            )
            if missing_coords:
                self.logger.info(
                    f"USE_HARMONIC_COORDS is on: {missing_coords} of {len(static_df)} point(s) have "
                    f"a missing or non-finite {' / '.join(coordinate_columns)} and are dropped. "
                    "Turning the flag off restores them."
                )

        cleaned = drop_non_finite_rows(
            static_df,
            logger=self.logger,
            label="static sequence source",
            required_columns=[point_col, *target_columns, *coordinate_columns],
            numeric_columns=[*target_columns, *coordinate_columns],
        )
        return cleaned, blocks

    def usable_point_ids(self, static_df: Optional[pd.DataFrame] = None) -> pd.Index:
        """The points this family can use, after its own cleaning.

        The split asks every family the same question; see
        :class:`~yg_eo_soilnet.datamodules.split_plan_provider.SplitPlanProvider`.

        Parameters
        ----------
        static_df : pandas.DataFrame, optional
            Use this table instead of reading the data files.

        Returns
        -------
        pandas.Index
        """
        dataset = self.data_manager.load_dataset()
        frame = dataset.tabular if static_df is None else static_df
        cleaned, _ = self.clean_static_frame(frame, dataset)
        point_col = dataset.point_id_column
        if point_col not in cleaned.columns:
            return pd.Index(range(len(cleaned)))
        return pd.Index(cleaned[point_col].to_numpy())

    def build(self, sequence_data_args: Optional[Mapping[str, Any]] = None) -> SoilSequenceBundle:
        """Read the data and assemble the bundle, one entry per usable point.

        Parameters
        ----------
        sequence_data_args : mapping, optional
            Extra build settings; unused at present.

        Returns
        -------
        SoilSequenceBundle
            Checked for consistency before it is returned.
        """
        sequence_data_args = dict(sequence_data_args or {})
        dataset = self.data_manager.load_dataset()
        point_col = dataset.point_id_column
        target_columns = list(dataset.target_columns)

        static_df, blocks = self.clean_static_frame(dataset.tabular, dataset)

        static_features, static_categoricals = split_feature_blocks(static_df, blocks)
        static_validity, static_validity_names = self._static_validity(
            static_df, list(blocks.continuous_columns)
        )
        targets = static_df[target_columns].to_numpy(dtype=np.float32)
        label_features, label_feature_names = self._extract_label_features(static_df)
        coords, coord_names = self._extract_coordinates(static_df)
        # In the order the covariate columns actually sit in, not the order the configuration lists
        # them: these names say which column is which.
        context_columns = set(self.data_manager.context_feature_columns())
        context_feature_names = [
            column for column in blocks.continuous_columns if column in context_columns
        ]
        point_ids = (
            static_df[point_col].tolist() if point_col in static_df.columns else list(range(len(static_df)))
        )

        sequences: dict[str, list[np.ndarray]] = {}
        sequence_times: dict[str, list[np.ndarray]] = {}
        sequence_validity: dict[str, list[np.ndarray]] = {}
        modality_columns: dict[str, list[str]] = {}

        timeseries_df = dataset.timeseries
        if timeseries_df is not None and not timeseries_df.empty:
            sequences, sequence_times, sequence_validity, modality_columns = self._build_sequences(
                timeseries_df, point_ids=point_ids, point_col=point_col
            )

        bundle = SoilSequenceBundle(
            point_ids=point_ids,
            static_features=static_features,
            static_feature_names=list(blocks.continuous_columns),
            static_validity=static_validity,
            static_validity_names=static_validity_names,
            static_categoricals=static_categoricals,
            categorical_feature_names=list(blocks.categorical_columns),
            context_feature_names=context_feature_names,
            coords=coords,
            coord_names=coord_names,
            targets=targets,
            target_names=target_columns,
            label_features=label_features,
            label_feature_names=label_feature_names,
            sequences=sequences,
            sequence_times=sequence_times,
            sequence_validity=sequence_validity,
            modality_columns=modality_columns,
            temporal_enabled=bool(getattr(self.config, "TEMPORAL_FEATURES_ENABLED", False) and sequences),
        )
        bundle.validate()
        self._log_summary(bundle)
        return bundle

    # --- spatial context group --------------------------------------------

    def _assert_context_features_present(
        self, static_df: pd.DataFrame, continuous_columns: list[str]
    ) -> None:
        """Refuse a ``CONTEXT_FEATURES`` column the data does not carry or that something removed.

        Only checked while the group is switched on: switched off, the columns are meant to be gone.

        Raises
        ------
        KeyError
            If a declared column is not in the data.
        ValueError
            If it is in the data but something else - ``IGNORED_COLUMNS``, ``LABEL_COLUMNS``,
            ``CATEGORICAL_FEATURES`` - keeps it out of the inputs.
        """
        selected = self.data_manager.context_feature_columns()
        if not selected:
            return

        absent = [column for column in selected if column not in static_df.columns]
        if absent:
            raise KeyError(
                f"CONTEXT_FEATURES names column(s) the static source does not carry: "
                f"{', '.join(sorted(absent))}. Add them to the static CSV, remove them from the "
                "list, or set USE_CONTEXT_FEATURES: false to run without the group."
            )

        # Present, spelled right, but removed by another setting - a different fix from the case
        # above, so a different message.
        withheld = [
            column
            for column in selected
            if column in static_df.columns and column not in set(continuous_columns)
        ]
        if withheld:
            raise ValueError(
                f"CONTEXT_FEATURES names column(s) that are present but not continuous features: "
                f"{', '.join(sorted(withheld))}. Something else is removing them - check "
                "IGNORED_COLUMNS/ELIMINATED_FEATURES, LABEL_COLUMNS and CATEGORICAL_FEATURES."
            )

    # --- covariate gaps ---------------------------------------------------

    def _static_validity(
        self, static_df: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Build the :term:`validity flags <validity flag>` for the covariates that have gaps.

        A column with no gaps gets no flag: it would be an always-true channel saying nothing. The
        scikit-learn pipeline flags the same set.

        Returns
        -------
        validity : numpy.ndarray of bool
        names : list of str
            The columns the flags belong to.
        """
        rows = len(static_df)
        gappy = [
            column
            for column in feature_columns
            if column in static_df.columns
            and not bool(build_finite_row_mask(static_df, numeric_columns=[column]).all())
        ]
        if not gappy:
            return np.empty((rows, 0), dtype=bool), []

        validity = np.column_stack(
            [build_finite_row_mask(static_df, numeric_columns=[column]).to_numpy(dtype=bool) for column in gappy]
        )
        self.logger.info(
            f"Carrying measured-vs-filled flags for {len(gappy)} continuous covariate(s) with gaps: "
            + ", ".join(gappy)
        )
        return validity, gappy

    # --- measured lab values ----------------------------------------------

    def _extract_label_features(self, static_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """The lab measurements, carried beside the inputs rather than among them.

        Only with ``CARRY_LABEL_COLUMNS: true``; otherwise nothing is carried and every batch is
        exactly as it would be without the setting. Carrying them does not make them inputs: a model
        reads one only by naming it in ``auxiliary_label_columns``, an
        :term:`auxiliary lab input`.

        Missing values are kept as NaN rather than costing the point its row - lab coverage varies a
        lot between columns, and choosing an auxiliary input must not silently delete a third of the
        data. They are filled in later, from the training points, and flagged.

        Returns
        -------
        values : numpy.ndarray of shape (n_points, n_labels)
        names : list of str
        """
        if not getattr(self.config, "CARRY_LABEL_COLUMNS", False):
            return np.empty((len(static_df), 0), dtype=np.float32), []

        label_columns = [
            str(column)
            for column in (getattr(self.config, "LABEL_COLUMNS", []) or [])
            if str(column) in static_df.columns
        ]
        label_columns = list(dict.fromkeys(label_columns))
        if not label_columns:
            return np.empty((len(static_df), 0), dtype=np.float32), []

        values = np.column_stack(
            [pd.to_numeric(static_df[column], errors="coerce").to_numpy(dtype=np.float64) for column in label_columns]
        )
        # Infinities count as missing: no lab value is infinite, and one would spoil the column's
        # statistics.
        values[~np.isfinite(values)] = np.nan
        return values.astype(np.float32), label_columns

    # --- coordinates ------------------------------------------------------

    def _coordinate_columns(self) -> list[str]:
        """The coordinate column names when the model reads coordinates, otherwise nothing.

        The one place ``USE_HARMONIC_COORDS`` is read, so the rule that drops a point without
        coordinates and the code that reads them cannot disagree.
        """
        if not getattr(self.config, "USE_HARMONIC_COORDS", False):
            return []
        return list(self.data_manager.coordinate_columns())

    def _extract_coordinates(self, static_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """The coordinates, carried as their own block rather than as ordinary covariates.

        Only with ``USE_HARMONIC_COORDS``; otherwise nothing is carried. One part of the model reads
        them, against the area the training points cover, and no scikit-learn model ever sees them.

        Returns
        -------
        values : numpy.ndarray of shape (n_points, 2)
        names : list of str

        Raises
        ------
        KeyError
            If the coordinate columns are not in the data.
        """
        coordinate_columns = self._coordinate_columns()
        if not coordinate_columns:
            return np.empty((len(static_df), 0), dtype=np.float64), []

        missing = [column for column in coordinate_columns if column not in static_df.columns]
        if missing:
            raise KeyError(
                f"USE_HARMONIC_COORDS is on but the static frame has no {', '.join(missing)} "
                f"column(s). Coordinates often live in the TARGETS source rather than the covariates "
                f"one; DataManager joins them across automatically, so check that "
                f"LAT_COLUMN/LON_COLUMN name columns that exist in one of the two. "
                f"Available: {', '.join(map(str, static_df.columns))}"
            )

        values = np.column_stack(
            [
                pd.to_numeric(static_df[column], errors="coerce").to_numpy(dtype=np.float64)
                for column in coordinate_columns
            ]
        )
        return values, coordinate_columns

    # --- temporal assembly ------------------------------------------------

    def _modality_entries(self, timeseries_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
        """Which time-series columns belong to which :term:`data source`.

        From ``temporal.modality_prefix_map`` - each source is a column-name prefix such as ``S2_`` -
        or from an explicit list of columns per source.

        Returns
        -------
        list of tuple
            ``(source name, its columns)``.
        """
        temporal_config = self.data_manager.temporal_config()
        prefix_map = self.data_manager.normalize_mapping(
            temporal_config.get("modality_prefix_map", getattr(self.config, "MODALITY_PREFIX_MAP", {}))
        )
        if prefix_map:
            return [
                (
                    str(name).lower(),
                    [column for column in timeseries_df.columns if column.startswith(str(prefix))],
                )
                for name, prefix in prefix_map.items()
            ]

        explicit_columns = self.data_manager.normalize_mapping(temporal_config.get("modality_columns", {}))
        if not explicit_columns:
            explicit_columns = {
                "s1": list(getattr(self.config, "S1_COLUMNS", []) or []),
                "s2": list(getattr(self.config, "S2_COLUMNS", []) or []),
                "modis": list(getattr(self.config, "MODIS_COLUMNS", []) or []),
            }

        entries: list[tuple[str, list[str]]] = []
        for name, columns in explicit_columns.items():
            matched = [column for column in columns if column in timeseries_df.columns]
            if matched:
                entries.append((str(name).lower(), matched))
        return entries

    def _build_sequences(
        self,
        timeseries_df: pd.DataFrame,
        *,
        point_ids: list[Any],
        point_col: str,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[str]],
    ]:
        """Turn the time-series table into one array of readings per point per data source.

        Readings are sorted by date, repeated dates are averaged, and rows whose date cannot be read
        or whose point did not survive cleaning are dropped.

        Returns
        -------
        tuple of dict
            The readings, their dates, their measured-or-filled flags, and each source's columns.

        Raises
        ------
        KeyError
            If the time series has no point id or date column.
        """
        temporal_config = self.data_manager.temporal_config()
        time_col = temporal_config.get("time_column", getattr(self.config, "TIME_COLUMN", "date"))

        if point_col not in timeseries_df.columns or time_col not in timeseries_df.columns:
            raise KeyError(f"Time-series source must contain '{point_col}' and '{time_col}' columns")

        modality_entries = [(name, columns) for name, columns in self._modality_entries(timeseries_df) if columns]
        if not modality_entries:
            self.logger.warning("No temporal modality columns matched the time-series source")
            return {}, {}, {}, {}

        all_modality_columns = [column for _, columns in modality_entries for column in columns]
        timeseries_df, validity_df = sanitize_numeric_columns(
            timeseries_df, all_modality_columns, logger=self.logger, return_validity=True
        )

        unique_columns = list(dict.fromkeys(all_modality_columns))
        working = timeseries_df[[point_col, time_col, *unique_columns]].copy()
        # The flags travel as ordinary columns, so every filter and sort below moves them with the
        # readings they belong to.
        for column in unique_columns:
            working[_VALIDITY_PREFIX + column] = validity_df[column].to_numpy(dtype=np.float32)
        working = working[working[point_col].notna()]

        # Dates first: a reading whose date cannot be read has no place on the time axis.
        decimal_year = to_decimal_year(working[time_col])
        unparsed = int(np.isnan(decimal_year).sum())
        if unparsed:
            self.logger.warning(
                f"Dropped {unparsed} time-series row(s) whose '{time_col}' could not be parsed as a date"
            )
        working = working.loc[np.isfinite(decimal_year)].copy()
        working["__decimal_year__"] = decimal_year[np.isfinite(decimal_year)]

        # Keep the points that survived cleaning, then average repeated dates: two readings on one
        # date would leave a zero gap, which the bundle refuses.
        known_points = set(point_ids)
        before = len(working)
        working = working[working[point_col].isin(known_points)]
        if len(working) < before:
            self.logger.warning(
                f"Dropped {before - len(working)} time-series row(s) that did not match a surviving point id"
            )

        grouped_keys = [point_col, "__decimal_year__"]
        duplicates = int(working.duplicated(subset=grouped_keys).sum())
        if duplicates:
            self.logger.warning(
                f"Averaged {duplicates} duplicate (point, date) time-series row(s) so timestamps stay strictly ascending"
            )
            working = working.groupby(grouped_keys, as_index=False, sort=True).mean(numeric_only=True)

        working = working.sort_values(grouped_keys, kind="stable")

        sequences: dict[str, list[np.ndarray]] = {}
        sequence_times: dict[str, list[np.ndarray]] = {}
        sequence_validity: dict[str, list[np.ndarray]] = {}
        modality_columns: dict[str, list[str]] = {}

        # Each point's rows, found once and reused for every data source. The table is already
        # sorted by point and date, so these are in date order.
        row_groups: dict[Any, np.ndarray] = {
            point_id: np.asarray(rows, dtype=np.int64)
            for point_id, rows in working.groupby(point_col, sort=False).indices.items()
        }
        times_all = working["__decimal_year__"].to_numpy(dtype=np.float64)

        for modality_name, columns in modality_entries:
            values_all = working[columns].to_numpy(dtype=np.float32)
            # Averaging repeated dates averages the flags too: a reading counts as measured only
            # if every row behind it was.
            validity_all = working[[_VALIDITY_PREFIX + column for column in columns]].to_numpy() >= 1.0

            per_point_values: list[np.ndarray] = []
            per_point_times: list[np.ndarray] = []
            per_point_validity: list[np.ndarray] = []
            for point_id in point_ids:
                rows = row_groups.get(point_id)
                if rows is None or rows.size == 0:
                    per_point_values.append(np.empty((0, len(columns)), dtype=np.float32))
                    per_point_times.append(np.empty((0,), dtype=np.float64))
                    per_point_validity.append(np.empty((0, len(columns)), dtype=bool))
                    continue
                per_point_values.append(values_all[rows])
                per_point_times.append(times_all[rows])
                per_point_validity.append(validity_all[rows])

            sequences[modality_name] = per_point_values
            sequence_times[modality_name] = per_point_times
            sequence_validity[modality_name] = per_point_validity
            modality_columns[modality_name] = list(columns)

        return sequences, sequence_times, sequence_validity, modality_columns

    def _log_summary(self, bundle: SoilSequenceBundle) -> None:
        """Report what was built: points, covariates, lab columns and readings per data source."""
        self.logger.info(
            f"Built sequence bundle over {bundle.num_points} point(s) with "
            f"{bundle.static_features.shape[1] if bundle.static_features.size else 0} static feature(s)"
        )
        if bundle.label_dim:
            # With the missing share, which is what decides whether a column is worth using as an
            # auxiliary input: a mostly missing one reaches the model mostly filled in.
            sparse = sorted(
                (
                    (name, bundle.label_missing_fraction(name))
                    for name in bundle.label_feature_names
                    if bundle.label_missing_fraction(name) > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            self.logger.info(
                f"  {bundle.label_dim} measured lab column(s) available as auxiliary inputs"
                + (
                    f"; most incomplete: {', '.join(f'{name} ({share:.1%} missing)' for name, share in sparse[:3])}"
                    if sparse
                    else "; all complete"
                )
            )
        for modality_name in sorted(bundle.sequences):
            counts = bundle.observation_counts(modality_name)
            if counts.size == 0:
                continue
            empty = int((counts == 0).sum())
            imputed = bundle.imputed_fraction(modality_name)
            self.logger.info(
                f"  modality '{modality_name}': {len(bundle.modality_columns[modality_name])} channel(s), "
                f"observations per point min={int(counts.min())} median={int(np.median(counts))} "
                f"max={int(counts.max())}"
                + (f", {empty} point(s) with no observations" if empty else "")
                # A filled-in reading is a repair, not a measurement: worth seeing per source.
                + (f", {imputed:.1%} of cells median-filled" if imputed > 0 else "")
            )
