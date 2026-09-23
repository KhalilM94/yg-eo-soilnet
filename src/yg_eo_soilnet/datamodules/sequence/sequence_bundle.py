"""Everything the deep-learning model reads about every point, in one object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class SoilSequenceBundle:
    """Everything the deep-learning model reads about every point: the :term:`bundle`.

    Built by :class:`~yg_eo_soilnet.datamodules.sequence.sequence_builder.SoilSequenceBuilder` and
    read by :class:`~yg_eo_soilnet.datamodules.sequence.sequence_datamodule.SoilSequenceDataModule`.

    Each point carries only the readings it actually has, each with its own date in
    :term:`decimal years <decimal year>`: there is no shared time axis and no filled-in gap, so a
    point with 40 readings sits beside one with 90. Nothing records *which* years the data came
    from, which is what lets a model trained on one period read another.

    Attributes
    ----------
    point_ids : list
        The points, in the order every array below follows.
    static_features : numpy.ndarray of shape (n_points, n_static)
        The numeric covariates. Gaps are kept as NaN: they are filled with the training points'
        median later, and the split does not exist yet at build time.
    static_feature_names : list of str
        Their column names.
    static_validity : numpy.ndarray of bool
        True where the covariate was measured rather than filled in. Only columns that have gaps
        appear; empty means nothing was filled.
    static_validity_names : list of str
        The columns ``static_validity`` covers.
    static_categoricals : numpy.ndarray of shape (n_points, n_categorical)
        Category covariates as raw labels. They are numbered later, from the training points only.
    categorical_feature_names : list of str
        Their column names.
    context_feature_names : list of str
        Which of ``static_feature_names`` are spatial-context covariates. A list of names, not
        values: the columns stay in ``static_features`` and are read like any other.
    coords : numpy.ndarray of shape (n_points, 2)
        Latitude and longitude, only with ``USE_HARMONIC_COORDS``; otherwise no columns. Kept at
        full precision, since the model works with differences between nearby coordinates.
    coord_names : list of str
        Their column names.
    targets : numpy.ndarray of shape (n_points, n_targets)
        The lab measurements being predicted.
    target_names : list of str
        Their column names.
    label_features : numpy.ndarray of shape (n_points, n_labels)
        Every lab measurement the data carries, for models using an :term:`auxiliary lab input`. The
        bundle is built once and reused, so it holds them all and each model picks by name. Gaps
        stay NaN, as for the covariates.
    label_feature_names : list of str
        Their column names.
    sequences : dict of str to list of numpy.ndarray
        Per :term:`data source`, one ``(n_readings, n_channels)`` array per point.
    sequence_times : dict of str to list of numpy.ndarray
        Per data source, the :term:`decimal year` of each of those readings.
    sequence_validity : dict of str to list of numpy.ndarray
        Per data source, True where the reading was measured rather than filled in. An empty dict
        means nothing was filled.
    modality_columns : dict of str to list of str
        Per data source, the columns its channels correspond to.
    temporal_enabled : bool
        Whether the time series is being used at all.
    """

    point_ids: list[Any] = field(default_factory=list)
    # Gaps are kept as NaN, here and in label_features: the fill value is the training points'
    # median, and the split does not exist yet at build time.
    static_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    static_feature_names: list[str] = field(default_factory=list)
    # Only columns that really have gaps get a flag: an always-true one doubles the width and says
    # nothing. The scikit-learn pipeline flags the same set.
    static_validity: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=bool))
    static_validity_names: list[str] = field(default_factory=list)
    # Raw labels, numbered later by the datamodule from the training points alone.
    static_categoricals: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=object))
    categorical_feature_names: list[str] = field(default_factory=list)
    # Names only: the values stay in static_features and are read like any other covariate.
    # Grouping them buys the on/off switch and a block in the SHAP figures, nothing more.
    context_feature_names: list[str] = field(default_factory=list)
    # Full precision, because what the model reads is the difference between nearby coordinates.
    # A point without coordinates is dropped rather than filled: the median position is a place no
    # sample occupies.
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    coord_names: list[str] = field(default_factory=list)
    targets: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    target_names: list[str] = field(default_factory=list)
    # Every lab column the data carries, not only what one model asked for: the bundle is built
    # once and reused, so each model picks the columns it wants by name.
    label_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    label_feature_names: list[str] = field(default_factory=list)
    # data source -> one array of readings per point, in point_ids order
    sequences: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # data source -> the decimal year of each of those readings
    sequence_times: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # data source -> True where a reading was measured rather than filled in. An empty dict means
    # nothing was filled.
    sequence_validity: dict[str, list[np.ndarray]] = field(default_factory=dict)
    modality_columns: dict[str, list[str]] = field(default_factory=dict)
    temporal_enabled: bool = False

    # --- Also usable like a dict, for callers written before it was a class ---

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        """The named field, or ``default`` when there is no such field."""
        return getattr(self, key, default)

    def keys(self):
        """The field names."""
        return tuple(self.__dataclass_fields__)

    @classmethod
    def from_mapping(cls, value: "SoilSequenceBundle | Mapping[str, Any]") -> "SoilSequenceBundle":
        """Build a bundle from a dict of fields; a bundle is returned unchanged."""
        if isinstance(value, cls):
            return value
        known = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in dict(value).items() if key in known})

    @property
    def num_points(self) -> int:
        """How many points the bundle holds."""
        return len(self.point_ids)

    @property
    def modality_dims(self) -> dict[str, int]:
        """How many channels each :term:`data source` has, from its column list.

        Read from the names rather than from a point's readings, which a point may not have.
        """
        return {name: len(columns) for name, columns in self.modality_columns.items()}

    @property
    def coord_dim(self) -> int:
        """How many coordinate columns the bundle carries: 2, or 0 when they are not used."""
        return len(self.coord_names)

    @property
    def label_dim(self) -> int:
        """How many lab columns the bundle carries."""
        return len(self.label_feature_names)

    def label_missing_fraction(self, column: str) -> float:
        """Share of points whose value for this lab column is missing.

        Parameters
        ----------
        column : str
            One of ``label_feature_names``.

        Returns
        -------
        float
            Between 0 and 1.

        Raises
        ------
        KeyError
            If the bundle does not carry that column.
        """
        if column not in self.label_feature_names:
            raise KeyError(f"Unknown label column {column!r}; available: {self.label_feature_names}")
        values = np.asarray(self.label_features, dtype=np.float64)
        if values.size == 0:
            return 0.0
        column_values = values[:, self.label_feature_names.index(column)]
        return float((~np.isfinite(column_values)).mean())

    def observation_counts(self, modality: str) -> np.ndarray:
        """How many readings each point has for one :term:`data source`."""
        return np.asarray([len(values) for values in self.sequences.get(modality, [])], dtype=np.int64)

    def validity_for(self, modality: str, index: int) -> np.ndarray:
        """Which of one point's readings were measured rather than filled in.

        Parameters
        ----------
        modality : str
            The :term:`data source`.
        index : int
            The point's position in ``point_ids``.

        Returns
        -------
        numpy.ndarray of bool
            All true when nothing was filled in.
        """
        per_point = (self.sequence_validity or {}).get(modality)
        if per_point is not None and index < len(per_point):
            return np.asarray(per_point[index], dtype=bool)
        return np.ones(np.asarray(self.sequences[modality][index]).shape, dtype=bool)

    def imputed_fraction(self, modality: str) -> float:
        """Share of one :term:`data source`\'s readings that were filled in rather than measured."""
        per_point = (self.sequence_validity or {}).get(modality)
        if not per_point:
            return 0.0
        total = sum(array.size for array in per_point)
        if total == 0:
            return 0.0
        return float(sum(int((~np.asarray(array, dtype=bool)).sum()) for array in per_point) / total)

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """Check the bundle holds together, and say which point is wrong when it does not.

        Every array must cover every point, every set of readings must match its dates and its
        channel count, and the dates must be sorted with no repeats.

        Raises
        ------
        ValueError
            Naming the point id, and the column where it is known.
        """
        # The covariates are not checked for gaps: a gap is allowed and is filled in once the
        # split is known. The targets are, since a point with no measurement teaches nothing.
        self._validate_numeric_array("targets", self.targets, self.target_names)

        num_points = len(self.point_ids)
        if self.static_features.size and self.static_features.shape[0] != num_points:
            raise ValueError(
                f"static_features has {self.static_features.shape[0]} row(s) but there are {num_points} point(s)"
            )

        validity = np.asarray(self.static_validity)
        if validity.ndim == 2 and validity.shape[1]:
            if validity.shape[0] != num_points:
                raise ValueError(
                    f"static_validity has {validity.shape[0]} row(s) but there are {num_points} point(s)"
                )
            if validity.shape[1] != len(self.static_validity_names):
                raise ValueError(
                    f"static_validity has {validity.shape[1]} column(s) but "
                    f"{len(self.static_validity_names)} validity feature name(s)"
                )
            unknown = [name for name in self.static_validity_names if name not in self.static_feature_names]
            if unknown:
                raise ValueError(
                    f"static_validity_names must be a subset of static_feature_names; unknown: {unknown}"
                )

        # Categories are still raw labels, and a missing one gets the reserved code later, so only
        # the shapes matter here.
        categoricals = np.asarray(self.static_categoricals)
        if categoricals.ndim == 2 and categoricals.shape[1]:
            if categoricals.shape[0] != num_points:
                raise ValueError(
                    f"static_categoricals has {categoricals.shape[0]} row(s) but there are "
                    f"{num_points} point(s)"
                )
            if categoricals.shape[1] != len(self.categorical_feature_names):
                raise ValueError(
                    f"static_categoricals has {categoricals.shape[1]} column(s) but "
                    f"{len(self.categorical_feature_names)} categorical feature name(s)"
                )
        if self.targets.size and self.targets.shape[0] != num_points:
            raise ValueError(f"targets has {self.targets.shape[0]} row(s) but there are {num_points} point(s)")

        # Coordinates are checked, unlike the blocks above: nothing fills them in later, so a gap
        # here would reach the model.
        coords = np.asarray(self.coords)
        if coords.ndim == 2 and coords.shape[1]:
            if coords.shape[0] != num_points:
                raise ValueError(
                    f"coords has {coords.shape[0]} row(s) but there are {num_points} point(s)"
                )
            if coords.shape[1] != len(self.coord_names):
                raise ValueError(
                    f"coords has {coords.shape[1]} column(s) but {len(self.coord_names)} "
                    f"coordinate name(s)"
                )
            self._validate_numeric_array("coords", coords, list(self.coord_names))

        unknown_context = [
            name for name in self.context_feature_names if name not in self.static_feature_names
        ]
        if unknown_context:
            raise ValueError(
                f"context_feature_names must be a subset of static_feature_names; unknown: "
                f"{unknown_context}"
            )

        # A missing lab value is allowed and is filled in later, so only the shapes matter.
        labels = np.asarray(self.label_features)
        if labels.ndim == 2 and labels.shape[1]:
            if labels.shape[0] != num_points:
                raise ValueError(
                    f"label_features has {labels.shape[0]} row(s) but there are {num_points} point(s)"
                )
            if labels.shape[1] != len(self.label_feature_names):
                raise ValueError(
                    f"label_features has {labels.shape[1]} column(s) but "
                    f"{len(self.label_feature_names)} label feature name(s)"
                )

        for modality, per_point_values in (self.sequences or {}).items():
            per_point_times = (self.sequence_times or {}).get(modality)
            if per_point_times is None:
                raise ValueError(f"Modality '{modality}' has sequences but no sequence_times")
            if len(per_point_values) != num_points or len(per_point_times) != num_points:
                raise ValueError(
                    f"Modality '{modality}' covers {len(per_point_values)} point(s) and "
                    f"{len(per_point_times)} time array(s) but there are {num_points} point(s)"
                )

            per_point_validity = (self.sequence_validity or {}).get(modality)
            if per_point_validity is not None and len(per_point_validity) != num_points:
                raise ValueError(
                    f"Modality '{modality}' has {len(per_point_validity)} validity array(s) "
                    f"but there are {num_points} point(s)"
                )

            expected_channels = len(self.modality_columns.get(modality, []))
            for index, (values, times) in enumerate(zip(per_point_values, per_point_times)):
                point_label = self.point_ids[index] if index < len(self.point_ids) else index
                self._validate_point_sequence(modality, point_label, values, times, expected_channels)
                if per_point_validity is not None:
                    validity = np.asarray(per_point_validity[index])
                    if validity.shape != np.asarray(values).shape:
                        raise ValueError(
                            f"sequence_validity['{modality}'] at point {point_label} has shape "
                            f"{validity.shape}, expected {np.asarray(values).shape} to match the readings"
                        )

    def _validate_point_sequence(
        self,
        modality: str,
        point_label: Any,
        values: np.ndarray,
        times: np.ndarray,
        expected_channels: int,
    ) -> None:
        """Check one point's readings: shape, one date each, channel count, and sorted dates."""
        values = np.asarray(values)
        times = np.asarray(times)

        if values.ndim != 2:
            raise ValueError(
                f"sequences['{modality}'] at point {point_label} must be 2-D (observations, channels), "
                f"got {values.ndim}-D"
            )
        if len(times) != values.shape[0]:
            raise ValueError(
                f"Modality '{modality}' at point {point_label} has {values.shape[0]} observation(s) "
                f"but {len(times)} timestamp(s)"
            )
        if expected_channels and values.shape[1] != expected_channels:
            raise ValueError(
                f"Modality '{modality}' at point {point_label} has {values.shape[1]} channel(s), "
                f"expected {expected_channels}"
            )
        if values.size and not np.isfinite(values).all():
            row, column = (int(index) for index in np.argwhere(~np.isfinite(values))[0])
            columns = self.modality_columns.get(modality, [])
            column_label = columns[column] if column < len(columns) else column
            raise ValueError(
                f"Non-finite value in sequences['{modality}'] at point {point_label}, "
                f"observation {row}, column '{column_label}'"
            )
        if times.size:
            if not np.isfinite(times).all():
                raise ValueError(f"Non-finite timestamp in sequence_times['{modality}'] at point {point_label}")
            # Strictly ascending: the model reads the gap between one reading and the next, and an
            # unsorted or repeated date would make that gap zero or negative.
            if times.size > 1 and not bool(np.all(np.diff(times) > 0)):
                raise ValueError(
                    f"sequence_times['{modality}'] at point {point_label} is not strictly ascending; "
                    "observations must be sorted by date with no duplicate timestamps"
                )

    def _validate_numeric_array(self, name: str, array: Any, column_labels: list[str]) -> None:
        """Raise on the first missing or infinite value, naming the point and column."""
        values = np.asarray(array)
        if values.size == 0 or not np.issubdtype(values.dtype, np.number):
            return
        finite_mask = np.isfinite(values)
        if bool(finite_mask.all()):
            return

        first_bad = np.argwhere(~finite_mask)[0]
        row_index = int(first_bad[0])
        point_label = self.point_ids[row_index] if row_index < len(self.point_ids) else row_index
        if len(first_bad) == 2:
            column_index = int(first_bad[1])
            column_label = column_labels[column_index] if column_index < len(column_labels) else column_index
            raise ValueError(f"Non-finite value in '{name}' at point {point_label}, column '{column_label}'")
        raise ValueError(f"Non-finite value in '{name}' at point {point_label}")
