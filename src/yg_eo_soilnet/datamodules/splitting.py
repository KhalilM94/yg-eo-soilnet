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
#: ``split.strategy: stratified`` - every split gets the same share of each band of a column.
STRATIFIED = "stratified"
#: The values ``split.strategy`` accepts.
STRATEGIES = (RANDOM, SPATIAL_GROUP, STRATIFIED)

#: ``split.strategy_applies_to: test_and_val`` - the strategy chooses the test set too. The
#: default, and the honest one: with ``spatial_group`` it is what keeps a test point from being the
#: near neighbour of a training point.
TEST_AND_VAL = "test_and_val"
#: ``split.strategy_applies_to: train_val`` - the test set is drawn at random and the strategy only
#: separates training from validation. The test set is then the same whichever strategy is chosen,
#: so two runs' scores can be compared; the price is that the score is flattering for clustered
#: data, which is the very thing ``spatial_group`` exists to expose.
TRAIN_VAL = "train_val"
#: The values ``split.strategy_applies_to`` accepts.
STRATEGY_SCOPES = (TEST_AND_VAL, TRAIN_VAL)

#: The band a point with no value for the stratification column is put in. It keeps its share of
#: every split rather than being dropped or pushed into training.
MISSING_STRATUM = "missing"
#: Where bands too small to split are pooled together.
OTHER_STRATUM = "other"

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
    strata : pandas.Series or None
        For a stratified split, the band each point was put in.
    strategy_applies_to : str
        Whether the strategy chose the test set as well (``test_and_val``) or only separated
        training from validation while the test set was drawn at random (``train_val``).
    coordinates : pandas.DataFrame or None
        Where each point is, as ``lat``/``lon`` columns indexed by point id. Carried so a finished
        run records the position of every point, not just of the ones a model was tested on - which
        is what lets the residual map draw the training points underneath the test ones.

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
    coordinates: Optional[pd.DataFrame] = None
    strata: Optional[pd.Series] = None
    strategy_applies_to: str = TEST_AND_VAL

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
        """The saved form: one row per point, with its id, split, position, group and family flags."""
        frame = pd.DataFrame(
            {
                "point_id": self.assignments.index.to_numpy(),
                "split": self.assignments.to_numpy(),
            }
        )
        if self.coordinates is not None:
            # Always written as lat/lon whatever LAT_COLUMN/LON_COLUMN are called in the source
            # data, so a reader of the saved plan needs to know nothing about that configuration.
            # Taken by name when the frame already uses those names, and otherwise by position -
            # which is latitude first, the order the frame is built in.
            placed = self.coordinates.reindex(self.assignments.index)
            latitudes = placed["lat"] if "lat" in placed.columns else placed.iloc[:, 0]
            longitudes = placed["lon"] if "lon" in placed.columns else placed.iloc[:, 1]
            frame["lat"] = np.asarray(latitudes)
            frame["lon"] = np.asarray(longitudes)
        if self.clusters is not None:
            frame["cluster"] = self.clusters.reindex(self.assignments.index).to_numpy()
        if self.strata is not None:
            frame["stratum"] = self.strata.reindex(self.assignments.index).astype(object).to_numpy()
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
            "split_strategy_applies_to": self.strategy_applies_to,
            # The one that stops two runs' rmse_test being compared when they should not be: a
            # random test set and a spatially held-out one are not the same measurement.
            "split_test_drawn_by": RANDOM if self.strategy_applies_to == TRAIN_VAL else self.strategy,
            "split_n_total": sum(counts.values()),
        }
        if self.strata is not None:
            described["split_n_strata"] = int(self.strata.nunique(dropna=False))
        for name in SPLIT_NAMES:
            described[f"split_n_{name}"] = counts[name]
            described[f"split_fraction_{name}"] = round(counts[name] / total, 6)
        for family, ids in sorted(self.eligibility.items()):
            described[f"split_n_eligible_{family}"] = len(ids)
            described[f"split_n_excluded_{family}"] = len(set(self.assignments.index) - set(ids))
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
        strata = None
        if "stratum" in frame.columns:
            strata = pd.Series(frame["stratum"].to_numpy(), index=assignments.index)
        coordinates = None
        if {"lat", "lon"}.issubset(frame.columns):
            coordinates = pd.DataFrame(
                {"lat": frame["lat"].to_numpy(), "lon": frame["lon"].to_numpy()}, index=assignments.index
            )
        eligibility = {
            column[len("eligible_") :]: frozenset(frame.loc[frame[column].astype(bool), "point_id"].to_numpy())
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
            coordinates=coordinates,
            strata=strata,
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
            raise ValueError(f"split.strategy must be one of {list(STRATEGIES)}; got {self.strategy!r}")
        self.test_size = _validated_fraction(getattr(config, "SPLIT_TEST_SIZE", 0.2), "split.test_size")
        self.val_size = _validated_fraction(getattr(config, "SPLIT_VAL_SIZE", 0.16), "split.val_size")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"split.test_size + split.val_size must leave a training set; got "
                f"{self.test_size} + {self.val_size} = {self.test_size + self.val_size}"
            )
        self.seed = int(getattr(config, "SPLIT_SEED", getattr(config, "RANDOM_SEED", 42)))
        self.strategy_applies_to = str(getattr(config, "SPLIT_STRATEGY_APPLIES_TO", TEST_AND_VAL)).lower()
        if self.strategy_applies_to not in STRATEGY_SCOPES:
            raise ValueError(
                f"split.strategy_applies_to must be one of {list(STRATEGY_SCOPES)}; got {self.strategy_applies_to!r}"
            )
        self.population_policy = str(getattr(config, "SPLIT_POPULATION_POLICY", INTERSECT)).lower()
        if self.population_policy not in POPULATION_POLICIES:
            raise ValueError(
                f"split.population_policy must be one of {list(POPULATION_POLICIES)}; got {self.population_policy!r}"
            )
        self.cluster_strategy_ = None
        # The clustered frame the spatial strategy produced, kept for the split map.
        self.clustered_frame_ = None

    def build_plan(
        self,
        point_ids: Sequence,
        *,
        coordinates: Optional[pd.DataFrame] = None,
        strata: Optional[pd.Series] = None,
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

        eligibility = {family: frozenset(pd.Index(values)) for family, values in (eligibility or {}).items()}

        clusters = self._clusters_for(ids, coordinates) if self.strategy == SPATIAL_GROUP else None
        strata = self._strata_for(ids, strata) if self.strategy == STRATIFIED else None

        def by_strategy(subset: pd.Index, fraction: float) -> tuple[pd.Index, pd.Index]:
            """Hold out a fraction of `subset` the way this run's strategy says to."""
            if self.strategy == SPATIAL_GROUP:
                return self._grouped_carve(subset, fraction, clusters)
            if self.strategy == STRATIFIED:
                return self._stratified_carve(subset, fraction, strata)
            return self._random_carve(subset, fraction)

        # The test set first, either the strategy's way or at random; then the validation set out of
        # what is left, always the strategy's way. Drawing the test set at random is what makes it
        # the same set whichever strategy is chosen, so two runs' scores can be compared.
        test_carve = self._random_carve if self.strategy_applies_to == TRAIN_VAL else by_strategy
        test_ids, remainder = test_carve(ids, self.test_size)
        val_ids, train_ids = by_strategy(remainder, _remainder_fraction(self.val_size, self.test_size))
        assignments = _assignments_from_ids(ids, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

        if self.strategy == SPATIAL_GROUP and clusters is not None:
            self._plot_split(ids, clusters, test_ids)
        if self.strategy == STRATIFIED and strata is not None:
            self._plot_stratification(assignments, strata)

        plan = SplitPlan(
            assignments=assignments,
            strategy=self.strategy,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
            population_policy=self.population_policy,
            eligibility=eligibility,
            clusters=clusters,
            strata=strata,
            strategy_applies_to=self.strategy_applies_to,
            # Reindexed onto the population: `coordinates` covers every point in the file, while
            # the plan covers only the ones being split.
            coordinates=None if coordinates is None else coordinates.reindex(ids),
        )
        counts = plan.counts()
        self.logger.info(
            f"Split plan ({self.strategy}, applied to {self.strategy_applies_to}, seed={self.seed}, "
            f"policy={self.population_policy}): "
            f"train={counts[TRAIN]} | val={counts[VAL]} | test={counts[TEST]}"
        )
        return plan

    # --- strategies -----------------------------------------------------------------
    # Each strategy is one "carve": hold out this fraction of these ids, and give back what is
    # left. build_plan calls a carve twice - once for the test set and once for the validation set
    # - which is what lets `strategy_applies_to: train_val` draw the test set at random while the
    # strategy still shapes the train/validation division.

    def _random_carve(self, ids: pd.Index, fraction: float) -> tuple[pd.Index, pd.Index]:
        """Hold out a fraction of the ids, one point at a time."""
        held_out, remainder = _carve(np.asarray(ids), fraction, self.seed)
        return pd.Index(held_out), pd.Index(remainder)

    def _grouped_carve(self, ids: pd.Index, fraction: float, clusters: pd.Series) -> tuple[pd.Index, pd.Index]:
        """Hold out whole clusters, never part of one.

        A point with no cluster - because it had no coordinates - is never held out: it cannot be
        held out honestly, so it stays in the remainder and ends up in training.
        """
        from sklearn.model_selection import GroupShuffleSplit

        clustered = ids.intersection(clusters.index)
        unclustered = ids.difference(clusters.index)
        groups = clusters.reindex(clustered).to_numpy()
        distinct = len(np.unique(groups)) if len(clustered) else 0

        if fraction <= 0.0 or len(clustered) <= 1 or distinct < 2:
            if distinct < 2 and fraction > 0.0:
                # Splitting inside the one remaining cluster would defeat the grouping, and holding
                # it out whole would empty what it was carved from.
                self.logger.warning(
                    f"Only {distinct} cluster(s) available for a holdout of {fraction:.2f}, so none "
                    f"was taken. Raise the cluster count or lower split.test_size to get one."
                )
            return pd.Index([], dtype=ids.dtype), ids

        keep_pos, held_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=self.seed).split(
                np.zeros((len(clustered), 1)), groups=groups
            )
        )
        return clustered[held_pos], clustered[keep_pos].append(unclustered)

    def _stratified_carve(self, ids: pd.Index, fraction: float, strata: pd.Series) -> tuple[pd.Index, pd.Index]:
        """Hold out the same share of every band, so the two sides have the same spread."""
        from sklearn.model_selection import train_test_split

        if fraction <= 0.0 or len(ids) <= 1:
            return pd.Index([], dtype=ids.dtype), ids

        labels = self._poolable_strata(strata.reindex(ids), fraction)
        if labels is None:
            return self._random_carve(ids, fraction)

        remainder, held_out = train_test_split(
            np.asarray(ids), test_size=fraction, random_state=self.seed, shuffle=True, stratify=labels.to_numpy()
        )
        return pd.Index(held_out), pd.Index(remainder)

    def _poolable_strata(self, labels: pd.Series, fraction: float) -> Optional[pd.Series]:
        """The bands, with any too small to split merged into one; None when none of them can be.

        ``train_test_split`` refuses outright if a band holds fewer members than the split needs,
        and a band of a sparse column hits that easily. Merging keeps the run alive and balances
        what can be balanced, rather than failing or silently dropping points.
        """
        labels = labels.astype(object).where(labels.notna(), MISSING_STRATUM)
        # A band has to leave at least one point on each side of the carve.
        smallest = max(2, int(np.ceil(1 / fraction)))
        counts = labels.value_counts()
        thin = counts.index[counts < smallest]
        if len(thin):
            self.logger.info(
                f"{len(thin)} stratum/strata hold fewer than {smallest} point(s) and were merged "
                f"into {OTHER_STRATUM!r} for this holdout ({int(counts.loc[thin].sum())} point(s))."
            )
            labels = labels.where(~labels.isin(set(thin)), OTHER_STRATUM)

        counts = labels.value_counts()
        if len(counts) < 2 or counts.min() < smallest:
            # Everything ended up in one band, so there is nothing left to balance.
            self.logger.warning(
                "No usable strata remain for this holdout, so it was drawn at random instead. "
                "Lower split.stratify_bins, or stratify on a column with more points per band."
            )
            return None
        return labels

    def _clusters_for(self, ids: pd.Index, coordinates: Optional[pd.DataFrame]) -> pd.Series:
        """Group the points by location, once, for both carves to hold out whole groups of."""
        if coordinates is None:
            raise ValueError("split.strategy 'spatial_group' needs coordinates; none were provided.")

        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")
        frame = self._clustering_frame(ids, coordinates, lat_col=lat_col, lon_col=lon_col)

        clustered = self._load_cluster_strategy().cluster(frame)
        self.clustered_frame_ = clustered
        cluster_series = pd.Series(
            clustered["cluster"].to_numpy(), index=pd.Index(clustered["point_id"].to_numpy(), name="point_id")
        )
        self.logger.info(f"Cluster value counts:\n{cluster_series.value_counts().sort_index().to_string()}")

        # A point without coordinates belongs to no group, so it cannot be held out honestly.
        # Training is the safe place for it; dropping it would shrink the dataset silently.
        unclustered = len(ids.difference(cluster_series.index))
        if unclustered:
            self.logger.warning(
                f"{unclustered} point(s) could not be spatially clustered (missing coordinates) "
                f"and are assigned to the train split rather than held out."
            )
        return cluster_series

    def _strata_for(self, ids: pd.Index, strata: Optional[pd.Series]) -> pd.Series:
        """The band each point belongs to, on the population being split.

        Raises
        ------
        ValueError
            If no strata were supplied, which means ``split.stratify_on`` was not set. Said here
            rather than quietly falling back to a random split, which would look like it worked.
        """
        if strata is None:
            raise ValueError(
                "split.strategy 'stratified' needs a column to stratify on. Set split.stratify_on "
                "to the name of the column whose spread the splits should match."
            )
        return strata.reindex(ids)

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

    def _plot_stratification(self, assignments: pd.Series, strata: pd.Series) -> None:
        """Draw the stratification check. A drawing failure must not lose the split itself."""
        try:
            from yg_eo_soilnet.artifacts import ArtifactLayout, log_figure
            from yg_eo_soilnet.plot_utils import stratification_panel

            frame = pd.DataFrame(
                {"split": assignments.to_numpy(), "value": strata.reindex(assignments.index).to_numpy()}
            )
            log_figure(
                stratification_panel(frame, column=getattr(self.config, "SPLIT_STRATIFY_ON", None)),
                "stratification.png",
                ArtifactLayout.SPLITS_PLOTS,
            )
        except Exception as error:  # noqa: BLE001 - a picture is never worth failing a run over
            self.logger.warning(f"Could not plot the stratification check: {error}")

    def _plot_split(self, ids: pd.Index, clusters: pd.Series, test_ids: pd.Index) -> None:
        """Draw the map of the spatial split. A drawing failure must not lose the split itself."""
        if self.cluster_strategy_ is None or self.clustered_frame_ is None:
            return
        try:
            clustered_ids = ids.intersection(clusters.index)
            position = pd.Series(np.arange(len(clustered_ids)), index=clustered_ids)
            test_pos = position.reindex(pd.Index(test_ids)).dropna().to_numpy(dtype=int)
            train_pos = np.setdiff1d(np.arange(len(clustered_ids)), test_pos)
            self.cluster_strategy_.plot_train_test(
                self.clustered_frame_.reset_index(drop=True),
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

    remainder, held_out = train_test_split(values, test_size=fraction, random_state=seed, shuffle=True)
    return np.asarray(held_out), np.asarray(remainder)


def _assignments_from_ids(ids: pd.Index, *, train_ids, val_ids, test_ids) -> pd.Series:
    """Build the assignment series; any id not named as validation or test is training."""
    assignments = pd.Series(TRAIN, index=pd.Index(ids, name="point_id"), dtype=object)
    assignments.loc[pd.Index(train_ids)] = TRAIN
    assignments.loc[pd.Index(val_ids)] = VAL
    assignments.loc[pd.Index(test_ids)] = TEST
    return assignments
