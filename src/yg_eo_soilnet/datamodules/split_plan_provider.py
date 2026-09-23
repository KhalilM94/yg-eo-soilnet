"""Build the one split a run uses, before any model touches the data.

The model families do not agree on which points they can use: the scikit-learn models keep every
row, while the deep-learning model needs a usable time series. This module asks each family which
points it can use, decides who is in the split according to ``split.population_policy``, and hands
back a single :class:`~yg_eo_soilnet.datamodules.splitting.SplitPlan`.

With ``intersect`` (the default) only the points every family can use are split, so their test sets
are identical. With ``assign_all`` every point is assigned and each family uses what it can.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from yg_eo_soilnet.datamodules.splitting import (
    ASSIGN_ALL,
    INTERSECT,
    SplitPlan,
    UnifiedSplitter,
)

#: The scikit-learn model family.
SKLEARN = "sklearn"
#: The deep-learning family, which reads the time series.
SEQUENCE = "sequence"
#: The families a split can be built for.
KNOWN_FAMILIES = (SKLEARN, SEQUENCE)


class SplitPlanProvider:
    """The run's split: built on first use, then handed to every model.

    Parameters
    ----------
    config : Config
        The run configuration; reads the ``split:`` settings.
    logger : logging.Logger
        Where the population report goes.
    data_manager : DataManager
        Reads the data the split is made over.
    families : iterable of str, optional
        Which model families will run, ``"sklearn"`` and/or ``"sequence"``. Left out, it is read
        from the models switched on, so a scikit-learn-only run does not prepare the time series
        just to find out which points have one.

    Raises
    ------
    ValueError
        If a family name is unknown.

    Examples
    --------
    >>> provider = SplitPlanProvider(config, logger, data_manager)   # doctest: +SKIP
    >>> provider.plan().counts()                                     # doctest: +SKIP
    {'train': 210, 'val': 45, 'test': 45}
    """

    def __init__(self, config, logger, data_manager, families: Optional[Iterable[str]] = None):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.families = tuple(families) if families is not None else self._infer_families(config)
        unknown = [family for family in self.families if family not in KNOWN_FAMILIES]
        if unknown:
            raise ValueError(f"Unknown training family/families {unknown}; expected {list(KNOWN_FAMILIES)}")
        self._plan: Optional[SplitPlan] = None

    def plan(self) -> SplitPlan:
        """The run's split, built on the first call and reused afterwards.

        Returns
        -------
        SplitPlan
        """
        if self._plan is None:
            self._plan = self._build()
        return self._plan

    # --- internals ------------------------------------------------------------------

    def _build(self) -> SplitPlan:
        """Read the data, work out who is in the split, and make or load the plan."""
        dataset = self.data_manager.load_dataset()
        tabular = dataset.tabular
        point_col = dataset.point_id_column
        if point_col in tabular.columns:
            all_ids = pd.Index(tabular[point_col].to_numpy(), name="point_id")
        else:
            self.logger.warning(
                f"No point id column {point_col!r} in the source frame; falling back to row position "
                f"as the split key. Every family derives it from the same frame in the same order, "
                f"so they still agree, but the plan cannot be re-keyed to a different export."
            )
            all_ids = pd.Index(range(len(tabular)), name="point_id")

        eligibility = self._eligibility(tabular)
        population = self._population(all_ids, eligibility)

        splitter = UnifiedSplitter(self.config, self.logger)
        self._log_population(all_ids, population, eligibility, splitter.population_policy)

        plan_path = getattr(self.config, "SPLIT_PLAN_PATH", None)
        if plan_path:
            return self._load_plan(plan_path, population, eligibility, splitter)

        coordinates = self._coordinates(tabular, all_ids)
        return splitter.build_plan(population, coordinates=coordinates, eligibility=eligibility)

    def _eligibility(self, tabular: pd.DataFrame) -> dict[str, frozenset]:
        """The point ids each model family can actually use, after its own cleaning."""
        builders = {
            SKLEARN: self._sklearn_usable,
            SEQUENCE: self._sequence_usable,
        }
        eligibility: dict[str, frozenset] = {}
        for family in self.families:
            eligibility[family] = frozenset(builders[family](tabular))
        return eligibility

    def _sklearn_usable(self, tabular: pd.DataFrame) -> pd.Index:
        """The points the scikit-learn models can use."""
        from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor

        return TabularPreprocessor(self.config, self.logger, self.data_manager).usable_point_ids(tabular)

    def _sequence_usable(self, tabular: pd.DataFrame) -> pd.Index:
        """The points the deep-learning model can use: those with a usable time series."""
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        return SoilSequenceBuilder(self.config, self.logger, self.data_manager).usable_point_ids(tabular)

    def _population(self, all_ids: pd.Index, eligibility: Mapping[str, frozenset]) -> pd.Index:
        """The points to split, under the configured population policy."""
        policy = str(getattr(self.config, "SPLIT_POPULATION_POLICY", INTERSECT)).lower()
        if policy not in (INTERSECT, ASSIGN_ALL):
            raise ValueError(
                f"split.population_policy must be one of {[INTERSECT, ASSIGN_ALL]}; got {policy!r}"
            )

        if policy == ASSIGN_ALL or not eligibility:
            # Every point is assigned and each family takes what it can use, so their test sets
            # overlap without being identical.
            population = all_ids
        else:
            common: Optional[set] = None
            for ids in eligibility.values():
                common = set(ids) if common is None else (common & set(ids))
            # Keep the file's row order, so the same data always gives the same plan.
            population = all_ids[all_ids.isin(common or set())]

        self._guard_against_collapse(all_ids, population, eligibility, policy)
        return population

    def _guard_against_collapse(
        self,
        all_ids: pd.Index,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        policy: str,
    ) -> None:
        """Stop the run when most of the dataset has fallen out of the split.

        Two different things can cause it, and the message says which:

        * one family's own cleaning dropped most of the points, usually because a covariate is
          nearly empty - a data problem, and it happens under either policy;
        * the families each keep plenty of points but agree on few, so ``intersect`` left little.

        The limit is ``split.min_population_ratio`` (half the dataset unless set); 0 switches the
        check off.
        """
        minimum_ratio = float(getattr(self.config, "SPLIT_MIN_POPULATION_RATIO", 0.5))
        if minimum_ratio <= 0.0 or not len(all_ids):
            return

        total = len(all_ids)
        starved = {
            family: ids
            for family, ids in eligibility.items()
            if len(ids) / total < minimum_ratio
        }
        if starved:
            listed = "\n".join(
                f"  {family}: {len(ids)} of {total} usable ({len(ids) / total:.1%})"
                for family, ids in sorted(starved.items(), key=lambda item: len(item[1]))
            )
            raise ValueError(
                f"{len(starved)} training family/families can use less than "
                f"split.min_population_ratio={minimum_ratio:.0%} of the dataset after their own "
                f"cleaning:\n{listed}\n"
                f"This is a data problem, not a split problem, and it applies whichever "
                f"population_policy is set. The builders log which columns cost the most rows (look "
                f"for 'Worst columns by rows lost'); dropping a near-empty covariate via "
                f"IGNORED_COLUMNS/ELIMINATED_FEATURES in data_spec.yml usually restores it. Lower "
                f"split.min_population_ratio to proceed anyway."
            )

        ratio = len(population) / total
        if ratio >= minimum_ratio:
            return

        breakdown = ", ".join(
            f"{family}={len(ids)}"
            for family, ids in sorted(eligibility.items(), key=lambda item: len(item[1]))
        )
        raise ValueError(
            f"split.population_policy={policy!r} left {len(population)} of {total} point(s) "
            f"({ratio:.1%}), below split.min_population_ratio={minimum_ratio}. Every family is "
            f"individually fine ({breakdown}), so it is the OVERLAP between them that is small - "
            f"they disagree about which rows are usable rather than any one of them being starved.\n"
            f"Set split.population_policy='assign_all' to let each family use what it has (test "
            f"sets then share membership without being identical), or lower "
            f"split.min_population_ratio."
        )

    def _coordinates(self, tabular: pd.DataFrame, all_ids: pd.Index) -> Optional[pd.DataFrame]:
        """The coordinates indexed by point id, or None when the data has none."""
        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")
        if lat_col not in tabular.columns or lon_col not in tabular.columns:
            return None
        return pd.DataFrame(
            {
                lat_col: tabular[lat_col].to_numpy(),
                lon_col: tabular[lon_col].to_numpy(),
            },
            index=all_ids,
        )

    def _load_plan(
        self,
        plan_path: str,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        splitter: UnifiedSplitter,
    ) -> SplitPlan:
        """Load a saved plan, so a new run keeps an earlier run's test set.

        Raises
        ------
        ValueError
            If the saved plan does not cover every point now in the population.
        """
        frame = pd.read_parquet(plan_path)
        plan = SplitPlan.from_frame(
            frame,
            strategy=splitter.strategy,
            test_size=splitter.test_size,
            val_size=splitter.val_size,
            seed=splitter.seed,
            population_policy=splitter.population_policy,
        )
        plan = SplitPlan(
            assignments=plan.assignments,
            strategy=plan.strategy,
            test_size=plan.test_size,
            val_size=plan.val_size,
            seed=plan.seed,
            population_policy=plan.population_policy,
            eligibility=dict(eligibility),
            clusters=plan.clusters,
        )
        unseen = population.difference(plan.assignments.index)
        if len(unseen):
            raise ValueError(
                f"split.plan_path {plan_path!r} does not cover {len(unseen)} point(s) in the current "
                f"population, e.g. {list(unseen[:5])}. Rebuild the plan or restore the matching data."
            )
        counts = plan.counts()
        self.logger.info(
            f"Loaded a frozen split plan from {plan_path}: "
            f"train={counts['train']} | val={counts['val']} | test={counts['test']}"
        )
        return plan

    def _log_population(
        self,
        all_ids: pd.Index,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        policy: str,
    ) -> None:
        """Report how many points each family can use, and how many the policy then excludes."""
        self.logger.info(
            f"Split population ({policy}): {len(population)} of {len(all_ids)} point(s) from the "
            f"source frame."
        )
        for family in sorted(eligibility):
            usable = eligibility[family]
            dropped_by_family = len(all_ids) - len(usable)
            lost_to_policy = len([pid for pid in usable if pid not in set(population)])
            self.logger.info(
                f"  {family}: {len(usable)} usable ({dropped_by_family} dropped by its own cleaning); "
                f"{lost_to_policy} further excluded by population_policy={policy}"
            )

    @staticmethod
    def _infer_families(config) -> tuple[str, ...]:
        """The families with at least one model switched on."""
        families: list[str] = []
        sklearn_registry = getattr(config, "MODEL_REGISTRY", {}) or {}
        if any(_enabled(entry) for entry in sklearn_registry.values()):
            families.append(SKLEARN)

        lightning_registry = getattr(config, "LIGHTNING_MODEL_REGISTRY", {}) or {}
        for name, entry in lightning_registry.items():
            if name == "defaults" or not _enabled(entry):
                continue
            kind = str((entry or {}).get("input_kind", SEQUENCE)).lower()
            if kind == SEQUENCE and kind not in families:
                families.append(kind)

        # A run with no model switched on still needs a population: the scikit-learn rule keeps
        # every row, so it is the neutral choice.
        return tuple(families) or (SKLEARN,)


def _enabled(entry: Any) -> bool:
    """Whether a model-list entry says ``enabled: true``."""
    return bool(isinstance(entry, Mapping) and entry.get("enabled", False))
