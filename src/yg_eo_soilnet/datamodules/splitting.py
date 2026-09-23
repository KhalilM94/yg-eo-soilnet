"""Assign every point to the training, validation or test set - once, for every model.

The split is decided before any model is trained and shared by all of them, so every model is scored
on the same test points and the scores can be compared. It is keyed on the point id rather than on
row numbers: the model families keep different rows (the deep-learning model needs a time series),
so only an id means the same thing to both.

``test_size`` and ``val_size`` are fractions of *all* the points, not of what the previous holdout
left over: 0.15 and 0.15 leave 70% for training. The result is a :class:`SplitPlan` - a table of
point id to split, saved with the run - rather than copies of the data, so a finished run can be
matched back to the file it was trained on.

Two strategies: ``random``, and ``spatial_group``, which groups points by location and holds out
whole groups, so a test point is not the near neighbour of a training point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: The points a model learns from.
TRAIN = "train"
#: The points watched during training, to decide when to stop.
VAL = "val"
#: The held-back points every model is scored on.
TEST = "test"
#: The three split names, in order.
SPLIT_NAMES = (TRAIN, VAL, TEST)

#: ``split.strategy: random`` - points are held out one by one.
RANDOM = "random"
#: ``split.strategy: spatial_group`` - whole groups of nearby points are held out.
SPATIAL_GROUP = "spatial_group"
#: The values ``split.strategy`` accepts.
STRATEGIES = (RANDOM, SPATIAL_GROUP)

#: ``split.population_policy: intersect`` - every family uses only the points all of them can use,
#: so their test sets are identical.
INTERSECT = "intersect"
#: ``split.population_policy: assign_all`` - each family uses every point it can.
ASSIGN_ALL = "assign_all"
#: The values ``split.population_policy`` accepts.
POPULATION_POLICIES = (INTERSECT, ASSIGN_ALL)


@dataclass(frozen=True)
class SplitPlan:
    """Which split each point belongs to, and the settings that produced it.

    A point missing from ``assignments`` is in no split and no model sees it: that is how
    ``population_policy: intersect`` leaves out the points only one model family could use.

    Attributes
    ----------
    assignments : pandas.Series
        One entry per point, indexed by point id, holding ``"train"``, ``"val"`` or ``"test"``.
    strategy : str
        ``"random"`` or ``"spatial_group"``.
    test_size, val_size : float
        The fractions asked for, of all the points.
    seed : int
        The random seed the split was drawn with.
    population_policy : str
        ``"intersect"`` or ``"assign_all"``.
    eligibility : mapping of str to frozenset
        Per model family, the ids of the points that family can use.
    clusters : pandas.Series or None
        For a spatial split, the group each point was put in.

    Examples
    --------
    >>> import pandas as pd
    >>> plan = SplitPlan(
    ...     assignments=pd.Series(["train", "val", "test"], index=["p1", "p2", "p3"]),
    ...     strategy="random", test_size=0.15, val_size=0.15, seed=42,
    ...     population_policy="intersect")
    >>> plan.counts()
    {'train': 1, 'val': 1, 'test': 1}
    >>> list(plan.point_ids_for("train"))
    ['p1']
    >>> plan.indices_for(["p3", "p1"], "test")   # "p3" sits first in this caller's own order
    array([0])
    """

    assignments: pd.Series
    strategy: str
    test_size: float
    val_size: float
    seed: int
    population_policy: str
    eligibility: Mapping[str, frozenset] = field(default_factory=dict)
    clusters: Optional[pd.Series] = None

    def point_ids_for(self, split: str) -> pd.Index:
        """The ids of the points in one split.

        Parameters
        ----------
        split : {"train", "val", "test"}

        Returns
        -------
        pandas.Index
        """
        _validate_split_name(split)
        return self.assignments.index[self.assignments.to_numpy() == split]

    def labels_for(self, point_ids: Sequence) -> pd.Series:
        """The split of each of these ids, in the order given; an unassigned id gives NaN."""
        return self.assignments.reindex(pd.Index(_as_index(point_ids)))

    def indices_for(self, point_ids: Sequence, split: str) -> np.ndarray:
        """Where the points of one split sit in ``point_ids``.

        ``point_ids`` is the caller's own ordering - the ids of its own rows - so each model family
        reads the shared plan against its own arrays.

        Parameters
        ----------
        point_ids : sequence
            Point ids in the caller's order.
        split : {"train", "val", "test"}

        Returns
        -------
        numpy.ndarray of int
            Positions in ``point_ids``.
        """
        _validate_split_name(split)
        labels = self.labels_for(point_ids).to_numpy()
        return np.flatnonzero(labels == split).astype(np.int64)

    def split_indices(self, point_ids: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the training, validation and test points sit in ``point_ids``, in one pass.

        Returns
        -------
        tuple of numpy.ndarray
            Positions of the training, validation and test points.
        """
        labels = self.labels_for(point_ids).to_numpy()
        return tuple(np.flatnonzero(labels == name).astype(np.int64) for name in SPLIT_NAMES)

    def counts(self) -> dict[str, int]:
        """How many points are in each split."""
        return {name: int((self.assignments.to_numpy() == name).sum()) for name in SPLIT_NAMES}

    def to_frame(self) -> pd.DataFrame:
        """The saved form: one row per point, with its id, split, group and per-family flags."""
        frame = pd.DataFrame(
            {
                "point_id": self.assignments.index.to_numpy(),
                "split": self.assignments.to_numpy(),
            }
        )
        if self.clusters is not None:
            frame["cluster"] = self.clusters.reindex(self.assignments.index).to_numpy()
        for family, ids in sorted(self.eligibility.items()):
            frame[f"eligible_{family}"] = frame["point_id"].isin(ids).to_numpy()
        return frame

    def describe(self) -> dict[str, Any]:
        """The plan as flat name/value pairs, ready to be recorded with the run.

        Sizes, counts and shares per split, the strategy, the seed, and how many points each model
        family can use.

        Returns
        -------
        dict of str to object
        """
        counts = self.counts()
        total = sum(counts.values()) or 1
        described: dict[str, Any] = {
            "split_strategy": self.strategy,
            "split_test_size": self.test_size,
            "split_val_size": self.val_size,
            "split_seed": self.seed,
            "split_population_policy": self.population_policy,
            "split_n_total": sum(counts.values()),
        }
        for name in SPLIT_NAMES:
            described[f"split_n_{name}"] = counts[name]
            described[f"split_fraction_{name}"] = round(counts[name] / total, 6)
        for family, ids in sorted(self.eligibility.items()):
            described[f"split_n_eligible_{family}"] = len(ids)
            described[f"split_n_excluded_{family}"] = len(
                set(self.assignments.index) - set(ids)
            )
        return described

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **provenance: Any) -> "SplitPlan":
        """Rebuild a plan from a saved table, so a run can reuse an earlier split.

        Set ``split.plan_path`` to the saved file to reuse it.

        Parameters
        ----------
        frame : pandas.DataFrame
            What :meth:`to_frame` wrote: columns ``point_id`` and ``split``, optionally ``cluster``
            and ``eligible_<family>``.
        **provenance
            Settings to record alongside; unknown ones are filled with placeholders.

        Returns
        -------
        SplitPlan

        Raises
        ------
        KeyError
            If the required columns are missing.
        ValueError
            If a row carries an unknown split name.
        """
        missing = {"point_id", "split"} - set(frame.columns)
        if missing:
            raise KeyError(f"A split plan frame needs columns {sorted(missing)}; got {list(frame.columns)}")
        assignments = pd.Series(
            frame["split"].to_numpy(), index=pd.Index(frame["point_id"].to_numpy(), name="point_id")
        )
        unknown = sorted(set(assignments.unique()) - set(SPLIT_NAMES))
        if unknown:
            raise ValueError(f"Split plan carries unknown split name(s) {unknown}; expected {list(SPLIT_NAMES)}")
        clusters = None
        if "cluster" in frame.columns:
            clusters = pd.Series(frame["cluster"].to_numpy(), index=assignments.index)
        eligibility = {
            column[len("eligible_") :]: frozenset(
                frame.loc[frame[column].astype(bool), "point_id"].to_numpy()
            )
            for column in frame.columns
            if column.startswith("eligible_")
        }
        provenance.setdefault("strategy", "loaded")
        provenance.setdefault("test_size", float("nan"))
        provenance.setdefault("val_size", float("nan"))
        provenance.setdefault("seed", -1)
        provenance.setdefault("population_policy", "loaded")
        return cls(
            assignments=assignments,
            eligibility=eligibility,
            clusters=clusters,
            **provenance,
        )


class UnifiedSplitter:
    """Turn a list of point ids into a :class:`SplitPlan`.

    Reads the ``split:`` settings: ``strategy``, ``test_size``, ``val_size``, ``seed`` and
    ``population_policy``. It is handed ids and coordinates and hands back assignments, which is what
    lets both model families share one split.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where the split sizes are reported.

    Raises
    ------
    ValueError
        If a setting holds an unknown value, or the two holdouts leave no training points.

    Examples
    --------
    >>> import logging
    >>> from types import SimpleNamespace
    >>> config = SimpleNamespace(SPLIT_HOLDOUT_STRATEGY="random", SPLIT_TEST_SIZE=0.2,
    ...                          SPLIT_VAL_SIZE=0.2, SPLIT_SEED=42)
    >>> splitter = UnifiedSplitter(config, logging.getLogger("demo"))
    >>> plan = splitter.build_plan([f"p{number}" for number in range(10)])
    >>> plan.counts()
    {'train': 6, 'val': 2, 'test': 2}
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.strategy = str(getattr(config, "SPLIT_HOLDOUT_STRATEGY", RANDOM)).lower()
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"split.strategy must be one of {list(STRATEGIES)}; got {self.strategy!r}"
            )
        self.test_size = _validated_fraction(getattr(config, "SPLIT_TEST_SIZE", 0.2), "split.test_size")
        self.val_size = _validated_fraction(getattr(config, "SPLIT_VAL_SIZE", 0.16), "split.val_size")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"split.test_size + split.val_size must leave a training set; got "
                f"{self.test_size} + {self.val_size} = {self.test_size + self.val_size}"
            )
        self.seed = int(getattr(config, "SPLIT_SEED", getattr(config, "RANDOM_SEED", 42)))
        self.population_policy = str(
            getattr(config, "SPLIT_POPULATION_POLICY", INTERSECT)
        ).lower()
        if self.population_policy not in POPULATION_POLICIES:
            raise ValueError(
                f"split.population_policy must be one of {list(POPULATION_POLICIES)}; "
                f"got {self.population_policy!r}"
            )
        self.cluster_strategy_ = None

    def build_plan(
        self,
        point_ids: Sequence,
        *,
        coordinates: Optional[pd.DataFrame] = None,
        eligibility: Optional[Mapping[str, Iterable]] = None,
    ) -> SplitPlan:
        """Assign every id to the training, validation or test set.

        Parameters
        ----------
        point_ids : sequence
            The points to split. Ids must be unique.
        coordinates : pandas.DataFrame, optional
            Indexed by point id, holding the configured latitude and longitude columns. Needed for
            ``spatial_group``; ignored otherwise.
        eligibility : mapping of str to iterable, optional
            Per model family, the ids that family can use. Recorded in the plan.

        Returns
        -------
        SplitPlan

        Raises
        ------
        ValueError
            If an id repeats, there are no points, or a spatial split was asked for without
            coordinates.
        """
        ids = _as_index(point_ids)
        duplicated = ids[ids.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"{len(duplicated)} duplicate point id(s) in the split population, e.g. "
                f"{list(duplicated[:5])}. A split is keyed on point id, so ids must be unique."
            )
        if len(ids) == 0:
            raise ValueError("Cannot build a split plan over an empty population.")

        eligibility = {
            family: frozenset(pd.Index(values)) for family, values in (eligibility or {}).items()
        }

        if self.strategy == SPATIAL_GROUP:
            assignments, clusters = self._spatial_group_assignments(ids, coordinates)
        else:
            assignments, clusters = self._random_assignments(ids), None

        plan = SplitPlan(
            assignments=assignments,
            strategy=self.strategy,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
            population_policy=self.population_policy,
            eligibility=eligibility,
            clusters=clusters,
        )
        counts = plan.counts()
        self.logger.info(
            f"Split plan ({self.strategy}, seed={self.seed}, policy={self.population_policy}): "
            f"train={counts[TRAIN]} | val={counts[VAL]} | test={counts[TEST]}"
        )
        return plan

    # --- strategies -----------------------------------------------------------------

    def _random_assignments(self, ids: pd.Index) -> pd.Series:
        """Hold out the test points at random, then the validation points from what is left."""
        test_ids, remainder = _carve(np.asarray(ids), self.test_size, self.seed)
        val_ids, train_ids = _carve(remainder, _remainder_fraction(self.val_size, self.test_size), self.seed)
        return _assignments_from_ids(ids, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    def _spatial_group_assignments(
        self, ids: pd.Index, coordinates: Optional[pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        """Group the points by location, then hold out whole groups, never part of one."""
        from sklearn.model_selection import GroupShuffleSplit

        if coordinates is None:
            raise ValueError("split.strategy 'spatial_group' needs coordinates; none were provided.")

        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")
        frame = self._clustering_frame(ids, coordinates, lat_col=lat_col, lon_col=lon_col)

        strategy = self._load_cluster_strategy()
        clustered = strategy.cluster(frame)
        cluster_series = pd.Series(
            clustered["cluster"].to_numpy(), index=pd.Index(clustered["point_id"].to_numpy(), name="point_id")
        )
        self.logger.info(f"Cluster value counts:\n{cluster_series.value_counts().sort_index().to_string()}")

        # A point without coordinates belongs to no group, so it cannot be held out honestly.
        # Training is the safe place for it; dropping it would shrink the dataset silently.
        unclustered = ids.difference(cluster_series.index)
        if len(unclustered):
            self.logger.warning(
                f"{len(unclustered)} point(s) could not be spatially clustered (missing coordinates) "
                f"and are assigned to the train split rather than held out."
            )

        clustered_ids = pd.Index(cluster_series.index)
        groups = cluster_series.to_numpy()
        placeholder = np.zeros((len(clustered_ids), 1))

        train_val_pos, test_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed).split(
                placeholder, groups=groups
            )
        )
        test_ids = clustered_ids[test_pos]

        val_fraction = _remainder_fraction(self.val_size, self.test_size)
        remaining_groups = len(np.unique(groups[train_val_pos]))
        if val_fraction <= 0.0 or len(train_val_pos) <= 1 or remaining_groups < 2:
            # One remaining group cannot be split without emptying training, and splitting inside a
            # group would defeat the grouping. Skip the validation holdout rather than fail.
            if remaining_groups < 2 and val_fraction > 0.0:
                self.logger.warning(
                    f"Only {remaining_groups} cluster(s) remain after the test holdout, so no "
                    f"grouped validation split is possible; val is empty and train keeps them all. "
                    f"Raise the cluster count or lower split.test_size to get a validation set."
                )
            val_ids = pd.Index([], dtype=clustered_ids.dtype)
            train_ids = clustered_ids[train_val_pos]
        else:
            inner_train_pos, inner_val_pos = next(
                GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=self.seed).split(
                    placeholder[train_val_pos], groups=groups[train_val_pos]
                )
            )
            val_ids = clustered_ids[train_val_pos[inner_val_pos]]
            train_ids = clustered_ids[train_val_pos[inner_train_pos]]

        train_ids = train_ids.append(unclustered)
        self._plot_split(frame, clustered, clustered_ids, test_ids)
        assignments = _assignments_from_ids(ids, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        return assignments, cluster_series

    def _clustering_frame(
        self, ids: pd.Index, coordinates: pd.DataFrame, *, lat_col: str, lon_col: str
    ) -> pd.DataFrame:
        """Build the table the clustering strategy reads: one row per point, with coordinates."""
        missing = [column for column in (lat_col, lon_col) if column not in coordinates.columns]
        if missing:
            raise KeyError(
                f"split.strategy 'spatial_group' needs coordinate column(s) {missing} on the "
                f"population frame; got {list(coordinates.columns)}"
            )
        aligned = coordinates.reindex(ids)
        # Numbered rows, for the strategies' plotting helpers, plus `lat`/`lon` copies of the
        # coordinates, whatever the configured column names are.
        frame = pd.DataFrame(
            {
                "point_id": ids.to_numpy(),
                lat_col: aligned[lat_col].to_numpy(),
                lon_col: aligned[lon_col].to_numpy(),
            }
        )
        frame["lat"] = frame[lat_col].to_numpy()
        frame["lon"] = frame[lon_col].to_numpy()
        return frame

    def _load_cluster_strategy(self):
        """Build the clustering strategy named by ``split.group.class_path``."""
        from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
        from yg_eo_soilnet.models import ModelConfigFactory  # local: avoids a circular import

        spec = dict(getattr(self.config, "SPLIT_GROUP_STRATEGY", {}) or {})
        if not spec.get("class_path"):
            raise ValueError(
                "split.strategy is 'spatial_group' but split.group.class_path is not set; "
                "name a BaseSpatialClusterStrategy, e.g. "
                "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy"
            )
        spec.setdefault("enabled", True)
        self.logger.info(f"Clustering the split population with {spec['class_path'].rsplit('.', 1)[-1]}...")
        strategy = ModelConfigFactory(spec, self.seed).load_splitter_from_config()
        if not isinstance(strategy, BaseSpatialClusterStrategy):
            # Without this check the run carries on with empty splits and fails much later.
            raise TypeError(
                "split.group did not resolve to a BaseSpatialClusterStrategy (got "
                f"{type(strategy).__name__}). Check class_path and 'enabled' in {spec!r}."
            )
        self.cluster_strategy_ = strategy
        return strategy

    def _plot_split(self, frame, clustered, clustered_ids, test_ids) -> None:
        """Draw the map of the spatial split. A drawing failure must not lose the split itself."""
        if self.cluster_strategy_ is None:
            return
        try:
            position = pd.Series(np.arange(len(clustered_ids)), index=clustered_ids)
            test_pos = position.reindex(pd.Index(test_ids)).dropna().to_numpy(dtype=int)
            train_pos = np.setdiff1d(np.arange(len(clustered_ids)), test_pos)
            self.cluster_strategy_.plot_train_test(
                clustered.reset_index(drop=True),
                train_pos,
                test_pos,
                title="Spatial Group Train/Test Split",
                filename="grid_split.png",
            )
        except Exception as error:  # noqa: BLE001 - a picture is never worth failing a run over
            self.logger.warning(f"Could not plot the spatial split map: {error}")


# --- helpers ------------------------------------------------------------------------


def _validate_split_name(split: str) -> None:
    """Raise unless this is one of the three split names."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_NAMES)}")


def _validated_fraction(value: Any, label: str) -> float:
    """Return a holdout fraction, refusing anything outside 0 (included) to 1 (excluded)."""
    fraction = float(value)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"{label} must be in [0, 1); got {fraction}")
    return fraction


def _as_index(point_ids: Sequence) -> pd.Index:
    """Return point ids of any sequence type as a pandas Index."""
    if isinstance(point_ids, pd.Index):
        return point_ids
    if isinstance(point_ids, pd.Series):
        return pd.Index(point_ids.to_numpy())
    return pd.Index(np.asarray(point_ids))


def _remainder_fraction(val_size: float, test_size: float) -> float:
    """Convert a fraction of all the points into a fraction of what the test holdout left."""
    remaining = 1.0 - test_size
    if remaining <= 0.0:
        return 0.0
    return min(val_size / remaining, 0.99)


def _carve(values: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``fraction`` of ``values`` at random, returning ``(held_out, remainder)``.

    A fraction of 0 means no holdout, which ``train_test_split`` refuses outright, so it is handled
    here - that is what keeps ``test_size: 0`` and ``val_size: 0`` usable.
    """
    values = np.asarray(values)
    if fraction <= 0.0 or values.size <= 1:
        return values[:0], values
    from sklearn.model_selection import train_test_split

    remainder, held_out = train_test_split(
        values, test_size=fraction, random_state=seed, shuffle=True
    )
    return np.asarray(held_out), np.asarray(remainder)


def _assignments_from_ids(ids: pd.Index, *, train_ids, val_ids, test_ids) -> pd.Series:
    """Build the assignment series; any id not named as validation or test is training."""
    assignments = pd.Series(TRAIN, index=pd.Index(ids, name="point_id"), dtype=object)
    assignments.loc[pd.Index(train_ids)] = TRAIN
    assignments.loc[pd.Index(val_ids)] = VAL
    assignments.loc[pd.Index(test_ids)] = TEST
    return assignments
