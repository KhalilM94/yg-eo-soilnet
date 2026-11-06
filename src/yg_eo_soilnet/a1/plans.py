"""Every split the A1 experiments are trained and scored on, built once, checked, and saved.

A run that drew its own split would be scored on points no other run used; saving the splits up
front, as :class:`~yg_eo_soilnet.datamodules.splitting.SplitPlan` files, makes every run read the
same fixed split and lets anyone check which points a score came from. Five kinds:

``random``
    The baseline: points held out one at a time, 85 % training / 15 % test.
``spatial_kfold``
    Nearby points grouped into blocks (k-means or a grid), the blocks packed into folds, each fold
    the test set once; every other block is training. Whole blocks, so a test point is never the
    near neighbour of a training point.
``logo``
    Leave one region out: each region in turn is the test set, every other region training. Is a
    model trained elsewhere any good here?
``mixed``
    A region held out as in ``logo``, but with a share of its points moved into training: how much
    does a little local data help a model trained elsewhere?
``control``
    The same moved points as the matching ``mixed`` plan, alone - only local data, nothing from
    elsewhere - scored on the same test points. The difference between the two is what the other
    regions add.

A region with fewer than ``plans.logo.min_points`` points (100) is never held out and never used
for validation: its score would rest on too few points. ``plans.logo.held_out`` narrows the regions
held out to a list of names. Every region not held out stays in training.

**No validation set, by default.** A1 chooses its settings by cross-validation inside the training
points (grouped by block or region on the spatial plans) and fits both stages on the training points
only, so validation points would never be fitted or tuned on: they would only take data away from
training. ``plans.random.val_size``, ``plans.spatial_kfold.val_size`` and
``plans.logo.validation_region`` (for ``logo`` and ``mixed``) still set one aside when asked, and
loading such settings logs a warning.

Every plan is checked before anything is written (:func:`validate_plan`): no point in two splits, a
test set, no block or region divided between splits - except the held-out region of ``mixed`` and
``control``, which is divided on purpose - and a round trip through the saved form.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1.settings import A1Settings
from yg_eo_soilnet.datamodules.splitting import SPLIT_NAMES, TEST, TRAIN, VAL, SplitPlan, UnifiedSplitter

RANDOM = "random"
SPATIAL_KFOLD = "spatial_kfold"
LOGO = "logo"
MIXED = "mixed"
CONTROL = "control"
#: The kinds that can be asked for; asking for ``mixed`` writes the ``control`` plans with it.
KINDS = (RANDOM, SPATIAL_KFOLD, LOGO, MIXED)
#: The kinds whose held-out region is divided between training and test on purpose.
DIVIDED_KINDS = (MIXED, CONTROL)

#: The columns of ``index.csv``, in order.
INDEX_COLUMNS = (
    "plan_id",
    "kind",
    "held_out",
    "fraction",
    "repeat",
    "block_size_m",
    "min_points",
    "held_out_list",
    "n_train",
    "n_val",
    "n_test",
)
#: What ``held_out_list`` says when no list was given and every large enough region is held out.
ALL_ELIGIBLE = "all eligible"
INDEX_FILE = "index.csv"
#: The model family recorded in every plan's ``eligible_*`` column: A1 trains scikit-learn models.
FAMILY = "sklearn"


class PlanError(ValueError):
    """A plan that breaks one of the rules every plan must keep."""


@dataclass(frozen=True)
class BuiltPlan:
    """One plan and what it is.

    Attributes
    ----------
    plan_id : str
        Its name, which is also its file name.
    kind : str
        One of :data:`KINDS`, or ``control``.
    plan : SplitPlan
        The split.
    held_out : str or None
        The held-out region's name, or the fold number; None for ``random``.
    fraction : float or None
        For ``mixed`` and ``control``, the share of the held-out region moved into training.
    repeat : int or None
        For ``mixed`` and ``control``, which draw of that share.
    held_out_cluster : int or None
        The held-out region's id, for ``logo``, ``mixed`` and ``control``.
    moved : frozenset
        For ``mixed`` and ``control``, the held-out region's points moved into training.
    block_size_m : float or None
        For ``spatial_kfold`` grid plans, the side of a block in metres.
    min_points : int or None
        For the region plans, the fewest points a region needed to be held out.
    held_out_list : str or None
        For the region plans, the regions asked to be held out (``;``-separated), or
        :data:`ALL_ELIGIBLE`.
    """

    plan_id: str
    kind: str
    plan: SplitPlan
    held_out: Optional[str] = None
    fraction: Optional[float] = None
    repeat: Optional[int] = None
    held_out_cluster: Optional[int] = None
    moved: frozenset = field(default_factory=frozenset)
    block_size_m: Optional[float] = None
    min_points: Optional[int] = None
    held_out_list: Optional[str] = None

    def index_row(self) -> dict:
        """The plan's row of ``index.csv``."""
        counts = self.plan.counts()
        return {
            "plan_id": self.plan_id,
            "kind": self.kind,
            "held_out": self.held_out if self.held_out is not None else "",
            "fraction": self.fraction if self.fraction is not None else np.nan,
            "repeat": self.repeat if self.repeat is not None else np.nan,
            "block_size_m": self.block_size_m if self.block_size_m is not None else np.nan,
            "min_points": self.min_points if self.min_points is not None else np.nan,
            "held_out_list": self.held_out_list if self.held_out_list is not None else "",
            "n_train": counts[TRAIN],
            "n_val": counts[VAL],
            "n_test": counts[TEST],
        }


@dataclass(frozen=True)
class PlanPopulation:
    """The points the plans are made over, and their regions.

    Attributes
    ----------
    ids : pandas.Index
        Every point with a measured target, in the data's order.
    coordinates : pandas.DataFrame
        ``lat``/``lon`` per point.
    region_ids : pandas.Series or None
        Each point's region number (1, 2, ... over the sorted names), missing for a point in no
        region; None when the settings name no regions.
    region_names : dict of int to str
        Region number to name.
    eligible : tuple of int
        The regions with at least ``plans.logo.min_points`` points: the ones that may be held out
        or used for validation.
    held_out : tuple of int
        The regions held out in turn by the ``logo`` and ``mixed`` plans: those named in
        ``plans.logo.held_out``, or every eligible region.
    too_small : dict of str to int
        The regions with fewer than ``min_points`` points, with their counts. They stay in training.
    """

    ids: pd.Index
    coordinates: pd.DataFrame
    region_ids: Optional[pd.Series] = None
    region_names: dict = field(default_factory=dict)
    eligible: tuple = ()
    held_out: tuple = ()
    too_small: dict = field(default_factory=dict)

    def region_points(self, region: int) -> pd.Index:
        """The points of one region, in the data's order."""
        return self.region_ids.index[self.region_ids.to_numpy() == region]


def plan_population(
    settings: A1Settings, config=None, logger: Optional[logging.Logger] = None, *, points=None
) -> PlanPopulation:
    """Read the points to make plans over: every point with a measured target.

    Parameters
    ----------
    settings : A1Settings
        The A1 settings.
    config : Config, optional
        The main configuration; read from the settings unless given.
    logger : logging.Logger, optional
        Where the regions too small to hold out are reported.
    points : A1Table, optional
        Already-loaded points (:func:`~yg_eo_soilnet.a1.features.load_points`), to avoid reading the
        data twice.

    Returns
    -------
    PlanPopulation
    """
    from yg_eo_soilnet.a1.features import load_points
    from yg_eo_soilnet.clustering_utils import ColumnGroupStrategy

    logger = logger or logging.getLogger(__name__)
    points = points if points is not None else load_points(settings, config, logger)
    frame = points.frame[points.frame[settings.target].notna()]
    ids = pd.Index(frame.index, name="point_id")
    coordinates = pd.DataFrame(
        {"lat": frame[points.lat_column].to_numpy(), "lon": frame[points.lon_column].to_numpy()}, index=ids
    )
    if points.region_column is None:
        return PlanPopulation(ids=ids, coordinates=coordinates)

    # Numbered by the same rule the split's ColumnGroupStrategy uses, so the ids mean the same thing.
    strategy = ColumnGroupStrategy(column="region")
    labelled = pd.DataFrame({"point_id": ids, "region": frame[points.region_column].to_numpy()})
    clustered = strategy.cluster(labelled)
    region_ids = pd.Series(clustered["cluster"].to_numpy(), index=pd.Index(clustered["point_id"])).reindex(ids)
    counts = region_ids.value_counts()
    minimum = settings.plans.logo.min_points
    names = dict(strategy.cluster_names_)
    eligible = tuple(sorted(int(region) for region, n in counts.items() if n >= minimum))
    too_small = {names[int(region)]: int(n) for region, n in counts.items() if n < minimum}
    held_out = _held_out_regions(settings, names, counts, eligible)
    if too_small:
        listed = ", ".join(f"{name} ({n})" for name, n in sorted(too_small.items(), key=lambda item: item[1]))
        logger.warning(
            f"{len(too_small)} region(s) hold fewer than plans.logo.min_points={minimum} points and are "
            f"never held out or used for validation (they stay in training): {listed}"
        )
    return PlanPopulation(
        ids=ids,
        coordinates=coordinates,
        region_ids=region_ids.astype("Int64"),
        region_names=names,
        eligible=eligible,
        held_out=held_out,
        too_small=too_small,
    )


def _held_out_regions(settings: A1Settings, names: dict, counts: pd.Series, eligible: tuple) -> tuple:
    """The regions to hold out: the ones ``plans.logo.held_out`` names, or every eligible one.

    A name that is not a region of the data, or a listed region with fewer than ``min_points``
    points, stops the plans: a list written by hand should not be silently shortened.
    """
    asked = settings.plans.logo.held_out
    if asked is None:
        return eligible
    by_name = {name: region for region, name in names.items()}
    unknown = [name for name in asked if name not in by_name]
    if unknown:
        raise PlanError(
            f"plans.logo.held_out names {unknown}, which are not regions of the data. The regions are: "
            f"{sorted(by_name)}."
        )
    minimum = settings.plans.logo.min_points
    small = {name: int(counts.get(by_name[name], 0)) for name in asked if by_name[name] not in eligible}
    if small:
        raise PlanError(
            f"plans.logo.held_out names {small} (points per region), fewer than plans.logo.min_points={minimum}: "
            "their scores would rest on too few points. Lower min_points or remove them from the list."
        )
    return tuple(sorted(by_name[name] for name in asked))


# --- the builders ---------------------------------------------------------------------------


def random_plan(population: PlanPopulation, settings: A1Settings) -> BuiltPlan:
    """The random baseline: points held out one at a time.

    Returns
    -------
    BuiltPlan
        ``plan_id`` ``random``, with no clusters: there are no groups to keep whole.
    """
    sizes = settings.plans.random
    config = SimpleNamespace(
        SPLIT_HOLDOUT_STRATEGY=RANDOM,
        SPLIT_TEST_SIZE=sizes.test_size,
        SPLIT_VAL_SIZE=sizes.val_size,
        SPLIT_SEED=settings.seed,
    )
    plan = UnifiedSplitter(config, logging.getLogger(__name__)).build_plan(
        population.ids, eligibility={FAMILY: population.ids}
    )
    return BuiltPlan(plan_id=RANDOM, kind=RANDOM, plan=plan)


def spatial_kfold_plans(population: PlanPopulation, settings: A1Settings) -> list[BuiltPlan]:
    """One plan per fold of blocks: the fold is the test set, every other block training.

    With ``plans.spatial_kfold.val_size`` above 0, whole blocks of the other folds holding about
    that share of the points are set aside for validation instead. Points without coordinates are in
    no block and always in training.

    Returns
    -------
    list of BuiltPlan
        For each block size (:meth:`~yg_eo_soilnet.a1.settings.KFoldPlanSettings.block_sizes_m`),
        ``kfold_10km_f0``, ``kfold_10km_f1``, ... - ``kfold_k50_f0``, ... for k-means blocks - with
        the block of each point as its cluster.

    Raises
    ------
    PlanError
        If there are too few points with coordinates to make the blocks.
    """
    from sklearn.model_selection import GroupKFold

    from yg_eo_soilnet.clustering_utils import KMeansClusterStrategy, SpatialGridClusterStrategy

    options = settings.plans.spatial_kfold
    placed = population.coordinates.dropna()
    if options.method == "kmeans" and len(placed) < options.n_clusters:
        raise PlanError(
            f"spatial_kfold needs at least n_clusters={options.n_clusters} points with coordinates; "
            f"there are {len(placed)}."
        )
    frame = pd.DataFrame({"point_id": placed.index, "lat": placed["lat"].to_numpy(), "lon": placed["lon"].to_numpy()})
    unplaced = population.ids.difference(placed.index)
    plans = []
    for size in options.block_sizes_m():
        if size is None:
            strategy = KMeansClusterStrategy(n_clusters=options.n_clusters, random_state=settings.seed)
            tag = f"k{options.n_clusters}"
        else:
            strategy = SpatialGridClusterStrategy(cell_size_m=int(size), random_state=settings.seed)
            tag = block_size_tag(size)
        clustered = strategy.cluster(frame)
        blocks = pd.Series(clustered["cluster"].astype(int).to_numpy(), index=pd.Index(clustered["point_id"]))
        if blocks.nunique() < options.n_folds:
            raise PlanError(
                f"spatial_kfold ({tag} blocks) made {blocks.nunique()} block(s), fewer than "
                f"n_folds={options.n_folds}; use more clusters or a smaller grid cell."
            )
        folds = GroupKFold(n_splits=options.n_folds).split(np.zeros((len(blocks), 1)), groups=blocks.to_numpy())
        for fold, (rest_positions, test_positions) in enumerate(folds):
            test_ids = blocks.index[test_positions]
            rest = blocks.index[rest_positions]
            val_ids = whole_blocks_for(
                blocks.reindex(rest), round(options.val_size * len(population.ids)), seed=settings.seed + fold
            )
            assignments = _assignments(population.ids, test=test_ids, val=val_ids)
            plan = _plan(assignments, SPATIAL_KFOLD, settings, clusters=blocks, population=population)
            plans.append(
                BuiltPlan(
                    plan_id=f"kfold_{tag}_f{fold}",
                    kind=SPATIAL_KFOLD,
                    plan=plan,
                    held_out=str(fold),
                    block_size_m=size,
                )
            )
    if len(unplaced):
        logging.getLogger(__name__).info(
            f"{len(unplaced)} point(s) have no coordinates and are in training in every fold."
        )
    return plans


def block_size_tag(size_m: float) -> str:
    """A block size as it appears in a plan id: kilometres, without trailing zeros.

    Examples
    --------
    >>> block_size_tag(10_000), block_size_tag(2_500)
    ('10km', '2.5km')
    """
    return f"{size_m / 1000:g}km"


def whole_blocks_for(blocks: pd.Series, n_points: int, *, seed: int) -> pd.Index:
    """Whole blocks, drawn in a seeded random order, until they hold about ``n_points`` points.

    Drawing a share of the blocks would miss the target whenever blocks differ in size, which
    k-means blocks over unevenly sampled ground always do. Blocks are added one at a time, and the
    last one is kept only if that lands closer to the target than stopping short. At least one block
    is always left over, so the training set is never empty.

    Parameters
    ----------
    blocks : pandas.Series
        Each point's block, indexed by point id.
    n_points : int
        How many points to aim for.
    seed : int
        Fixes the order the blocks are drawn in.

    Returns
    -------
    pandas.Index
        The chosen points; empty for a target of 0 or a single block.

    Examples
    --------
    Four blocks of two points; asking for four points takes two whole blocks, whichever they are:

    >>> blocks = pd.Series([1, 1, 2, 2, 3, 3, 4, 4], index=list("abcdefgh"))
    >>> chosen = whole_blocks_for(blocks, 4, seed=0)
    >>> len(chosen), blocks[chosen].nunique()
    (4, 2)
    """
    sizes = blocks.value_counts()
    if n_points <= 0 or len(sizes) < 2:
        return pd.Index([])
    order = sizes.index.to_numpy()[np.random.default_rng(seed).permutation(len(sizes))]
    chosen, total = [], 0
    for block in order[:-1]:
        size = int(sizes[block])
        if total and abs(total + size - n_points) >= abs(total - n_points):
            break
        chosen.append(block)
        total += size
        if total >= n_points:
            break
    return blocks.index[blocks.isin(chosen).to_numpy()]


def logo_plans(population: PlanPopulation, settings: A1Settings) -> list[BuiltPlan]:
    """Leave one region out: each eligible region the test set once.

    Every other region is training. With ``plans.logo.validation_region``, one other eligible region
    is set aside for validation instead (:func:`validation_region`).

    Returns
    -------
    list of BuiltPlan
        ``logo_r<id>_<name>``, with each point's region as its cluster.
    """
    _require_regions(population, LOGO)
    plans = []
    for region in population.held_out:
        test_ids = population.region_points(region)
        val_ids = _validation_points(population, region, settings)
        assignments = _assignments(population.ids, test=test_ids, val=val_ids)
        plan = _plan(assignments, LOGO, settings, clusters=population.region_ids, population=population)
        plans.append(
            BuiltPlan(
                plan_id=f"logo_{_region_tag(population, region)}",
                kind=LOGO,
                plan=plan,
                held_out=population.region_names[region],
                held_out_cluster=region,
                **_region_record(settings),
            )
        )
    return plans


def _region_record(settings: A1Settings) -> dict:
    """What every region plan records about how its regions were chosen."""
    logo = settings.plans.logo
    listed = ";".join(logo.held_out) if logo.held_out is not None else ALL_ELIGIBLE
    return {"min_points": logo.min_points, "held_out_list": listed}


def validation_region(population: PlanPopulation, held_out: int, seed: int) -> Optional[int]:
    """The region set aside for validation when ``held_out`` is the test set and validation is asked for.

    Chosen with the run's seed and the held-out region's number, so the same data always gives the
    same choice; None with fewer than three eligible regions, when there is none to spare.
    """
    others = [region for region in population.eligible if region != held_out]
    if len(population.eligible) < 3:
        return None
    return int(np.random.default_rng([seed, held_out]).choice(others))


def _validation_points(population: PlanPopulation, held_out: int, settings: A1Settings) -> pd.Index:
    """The validation points of a region plan: none, unless ``plans.logo.validation_region`` asks."""
    if not settings.plans.logo.validation_region:
        return pd.Index([])
    validation = validation_region(population, held_out, settings.seed)
    return population.region_points(validation) if validation is not None else pd.Index([])


def mixed_plans(population: PlanPopulation, settings: A1Settings) -> list[BuiltPlan]:
    """Each held-out region with a share of its points moved into training, and the matching controls.

    For each repeat the region's points are shuffled once and the first round(p x n) moved, so a
    larger share holds every point a smaller one did: the curve over the shares is one experiment,
    not nine independent ones. At least one point is moved and at least one left to test.

    Returns
    -------
    list of BuiltPlan
        For every region, share and repeat, a ``mixed`` plan then its ``control``.
    """
    _require_regions(population, MIXED)
    options = settings.plans.mixed
    plans = []
    for region in population.held_out:
        region_ids = population.region_points(region)
        val_ids = _validation_points(population, region, settings)
        tag = _region_tag(population, region)
        name = population.region_names[region]
        for repeat in range(options.n_repeats):
            order = region_ids[np.random.default_rng([settings.seed, repeat, region]).permutation(len(region_ids))]
            for fraction in options.fractions:
                n_moved = min(max(int(round(fraction * len(region_ids))), 1), len(region_ids) - 1)
                moved, test_ids = order[:n_moved], order[n_moved:]
                suffix = f"{tag}_p{int(round(fraction * 100)):02d}_s{repeat}"
                common = dict(
                    held_out=name,
                    fraction=float(fraction),
                    repeat=repeat,
                    held_out_cluster=region,
                    moved=frozenset(moved),
                    **_region_record(settings),
                )

                assignments = _assignments(population.ids, test=test_ids, val=val_ids)
                mixed = _plan(assignments, MIXED, settings, clusters=population.region_ids, population=population)
                plans.append(BuiltPlan(plan_id=f"mixed_{suffix}", kind=MIXED, plan=mixed, **common))

                # Only the region's own points: the moved ones to learn from, the rest to test on.
                control_assignments = pd.Series(TEST, index=pd.Index(region_ids, name="point_id"), dtype=object)
                control_assignments.loc[moved] = TRAIN
                control = _plan(
                    control_assignments, CONTROL, settings, clusters=population.region_ids, population=population
                )
                plans.append(BuiltPlan(plan_id=f"control_{suffix}", kind=CONTROL, plan=control, **common))
    return plans


# --- the checks ---------------------------------------------------------------------------


def validate_plan(built: BuiltPlan) -> None:
    """Check one plan keeps the rules every plan must keep.

    Raises
    ------
    PlanError
        Naming the plan and the rule: a point in two splits, an unknown split name, an empty test
        set, a block or region divided between splits (apart from the held-out region of ``mixed``
        and ``control``), a held-out region that does not match the points moved out of it, or a
        plan that does not survive being saved and read back.
    """
    plan, name = built.plan, built.plan_id
    assignments = plan.assignments
    repeated = assignments.index[assignments.index.duplicated()].unique()
    if len(repeated):
        raise PlanError(f"{name}: {len(repeated)} point(s) are in more than one split, e.g. {list(repeated[:3])}")
    unknown = sorted(set(assignments.unique()) - set(SPLIT_NAMES))
    if unknown:
        raise PlanError(f"{name}: unknown split name(s) {unknown}")
    if not (assignments == TEST).any():
        raise PlanError(f"{name}: the test set is empty")

    if plan.clusters is not None:
        clusters = plan.clusters.reindex(assignments.index)
        splits_per_cluster = (
            pd.DataFrame({"cluster": clusters.to_numpy(), "split": assignments.to_numpy()})
            .dropna(subset=["cluster"])
            .groupby("cluster")["split"]
            .agg(lambda splits: frozenset(splits))
        )
        for cluster, splits in splits_per_cluster.items():
            if len(splits) == 1:
                continue
            divided_on_purpose = built.kind in DIVIDED_KINDS and cluster == built.held_out_cluster
            if not divided_on_purpose:
                raise PlanError(f"{name}: cluster {cluster} is divided between {sorted(splits)}")
            if splits != frozenset({TRAIN, TEST}):
                raise PlanError(
                    f"{name}: the held-out region {built.held_out!r} may only be divided between train and "
                    f"test; it is in {sorted(splits)}"
                )

    if built.kind in DIVIDED_KINDS:
        _validate_divided(built)
    if built.kind == LOGO:
        region = plan.clusters.reindex(assignments.index)
        test = set(assignments.index[assignments.to_numpy() == TEST])
        if test != set(assignments.index[(region == built.held_out_cluster).to_numpy()]):
            raise PlanError(f"{name}: the test set is not exactly the held-out region {built.held_out!r}")

    _validate_round_trip(built)


def validate_pair(mixed: BuiltPlan, control: BuiltPlan) -> None:
    """Check a ``control`` plan matches its ``mixed`` plan: the same moved points, the same test points.

    Raises
    ------
    PlanError
        If they differ.
    """
    region = mixed.plan.clusters.reindex(mixed.plan.assignments.index) == mixed.held_out_cluster
    mixed_moved = set(mixed.plan.point_ids_for(TRAIN)) & set(mixed.plan.assignments.index[region.to_numpy()])
    if set(control.plan.point_ids_for(TRAIN)) != mixed_moved:
        raise PlanError(f"{control.plan_id}: its training points are not the points {mixed.plan_id} moved")
    if set(control.plan.point_ids_for(TEST)) != set(mixed.plan.point_ids_for(TEST)):
        raise PlanError(f"{control.plan_id}: its test points differ from {mixed.plan_id}'s")


def _validate_divided(built: BuiltPlan) -> None:
    """The held-out region of a mixed or control plan: the moved points train, every other one tests."""
    plan, name = built.plan, built.plan_id
    assignments = plan.assignments
    in_region = (plan.clusters.reindex(assignments.index) == built.held_out_cluster).to_numpy()
    region_train = set(assignments.index[in_region & (assignments.to_numpy() == TRAIN)])
    if region_train != set(built.moved):
        raise PlanError(f"{name}: the held-out region's training points are not the points moved into training")
    test = set(assignments.index[assignments.to_numpy() == TEST])
    if test != set(assignments.index[in_region]) - set(built.moved):
        raise PlanError(f"{name}: the test set is not the rest of the held-out region")
    if built.kind == CONTROL:
        if (assignments == VAL).any():
            raise PlanError(f"{name}: a control plan has no validation set")
        if not in_region.all():
            raise PlanError(f"{name}: a control plan holds only the held-out region's points")


def _validate_round_trip(built: BuiltPlan) -> None:
    """The plan must come back unchanged from its saved form."""
    reread = SplitPlan.from_frame(built.plan.to_frame())
    if not reread.assignments.equals(built.plan.assignments.rename_axis("point_id")):
        raise PlanError(f"{built.plan_id}: the saved plan does not read back with the same assignments")
    if built.plan.clusters is not None:
        original = built.plan.clusters.reindex(built.plan.assignments.index).astype(float).to_numpy()
        back = reread.clusters.astype(float).to_numpy()
        if not np.array_equal(original, back, equal_nan=True):
            raise PlanError(f"{built.plan_id}: the saved plan does not read back with the same clusters")


# --- writing them ------------------------------------------------------------------------------


@dataclass
class PlanSet:
    """What :func:`write_plans` wrote.

    Attributes
    ----------
    out_dir : pathlib.Path
    index : pandas.DataFrame
        The rows of ``index.csv``.
    skipped : dict of str to str
        Each kind that could not be made, and why.
    too_small : dict of str to int
        The regions with fewer than ``plans.logo.min_points`` points, with their counts.
    held_out : list of str
        The regions the ``logo`` and ``mixed`` plans hold out.
    in_training : list of str
        The other regions: always in training.
    """

    out_dir: Path
    index: pd.DataFrame
    skipped: dict = field(default_factory=dict)
    too_small: dict = field(default_factory=dict)
    held_out: list = field(default_factory=list)
    in_training: list = field(default_factory=list)


def build_plans(
    population: PlanPopulation, settings: A1Settings, kinds: Iterable[str] = KINDS
) -> tuple[list[BuiltPlan], dict]:
    """Build and check every plan of the kinds asked for.

    Returns
    -------
    plans : list of BuiltPlan
    skipped : dict of str to str
        Each kind that could not be made, and why.

    Raises
    ------
    ValueError
        If a kind is unknown.
    PlanError
        If a plan breaks a rule.
    """
    kinds = list(kinds)
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"Unknown plan kind(s) {unknown}; expected some of {list(KINDS)}")
    builders = {RANDOM: lambda: [random_plan(population, settings)], SPATIAL_KFOLD: spatial_kfold_plans}
    builders.update({LOGO: logo_plans, MIXED: mixed_plans})

    plans, skipped = [], {}
    for kind in KINDS:
        if kind not in kinds:
            continue
        if kind in (LOGO, MIXED) and population.region_ids is None:
            skipped[kind] = "regions.source is not set in the A1 settings."
            continue
        if kind in (LOGO, MIXED) and not population.held_out:
            skipped[kind] = f"no region holds plans.logo.min_points={settings.plans.logo.min_points} points."
            continue
        if kind == SPATIAL_KFOLD and population.coordinates.dropna().empty:
            skipped[kind] = "no point has coordinates."
            continue
        built = builders[kind]() if kind == RANDOM else builders[kind](population, settings)
        plans.extend(built)

    for built in plans:
        validate_plan(built)
    by_id = {built.plan_id: built for built in plans}
    if len(by_id) != len(plans):
        raise PlanError("Two plans share an id; region names may collide once made into file names.")
    for built in plans:
        if built.kind == CONTROL:
            validate_pair(by_id["mixed_" + built.plan_id[len("control_") :]], built)
    return plans, skipped


def write_plans(
    settings: A1Settings,
    out_dir: Optional[str] = None,
    *,
    kinds: Iterable[str] = KINDS,
    config=None,
    logger: Optional[logging.Logger] = None,
    population: Optional[PlanPopulation] = None,
) -> PlanSet:
    """Build, check and save every plan, with an index of them.

    Nothing is written until every plan has passed :func:`validate_plan`. The folder's earlier plans
    are removed first, so a plan from an earlier run can never sit beside an index that does not list
    it.

    Parameters
    ----------
    settings : A1Settings
    out_dir : str, optional
        Where to write; ``<output_dir>/plans`` unless given.
    kinds : iterable of str
        Which kinds to make (:data:`KINDS`); ``mixed`` brings its ``control`` plans with it.
    config : Config, optional
        The main configuration; read from the settings unless given.
    logger : logging.Logger, optional
    population : PlanPopulation, optional
        Already-read points, to avoid reading the data again.

    Returns
    -------
    PlanSet
    """
    logger = logger or logging.getLogger(__name__)
    out = Path(out_dir or Path(settings.output_dir) / "plans")
    population = population if population is not None else plan_population(settings, config, logger)
    plans, skipped = build_plans(population, settings, kinds)

    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("*.parquet"), out / INDEX_FILE]:
        stale.unlink(missing_ok=True)
    for built in plans:
        built.plan.to_frame().to_parquet(out / f"{built.plan_id}.parquet", index=False)
    index = pd.DataFrame([built.index_row() for built in plans], columns=list(INDEX_COLUMNS))
    index["repeat"] = index["repeat"].astype("Int64")
    index.to_csv(out / INDEX_FILE, index=False)

    for kind, reason in skipped.items():
        logger.warning(f"No {kind} plans: {reason}")
    logger.info(f"Wrote {len(plans)} plan(s) to {out}: {index['kind'].value_counts().to_dict()}")
    names = population.region_names
    return PlanSet(
        out_dir=out,
        index=index,
        skipped=skipped,
        too_small=dict(population.too_small),
        held_out=[names[region] for region in population.held_out],
        in_training=sorted(names[region] for region in names if region not in population.held_out),
    )


# --- helpers ------------------------------------------------------------------------------


def _assignments(ids: pd.Index, *, test: pd.Index, val: pd.Index) -> pd.Series:
    """Training for every point, then the validation and test points marked."""
    assignments = pd.Series(TRAIN, index=pd.Index(ids, name="point_id"), dtype=object)
    assignments.loc[pd.Index(val)] = VAL
    assignments.loc[pd.Index(test)] = TEST
    return assignments


def _plan(assignments: pd.Series, kind: str, settings: A1Settings, *, clusters, population) -> SplitPlan:
    """A SplitPlan of these assignments, recording the blocks or regions as its clusters."""
    counts = assignments.value_counts()
    total = max(len(assignments), 1)
    return SplitPlan(
        assignments=assignments,
        strategy=kind,
        test_size=round(float(counts.get(TEST, 0)) / total, 6),
        val_size=round(float(counts.get(VAL, 0)) / total, 6),
        seed=settings.seed,
        population_policy="intersect",
        eligibility={FAMILY: frozenset(population.ids)},
        clusters=None if clusters is None else clusters.reindex(assignments.index).dropna().astype(int),
    )


def _require_regions(population: PlanPopulation, kind: str) -> None:
    """Refuse to make region plans without regions."""
    if population.region_ids is None:
        raise PlanError(f"{kind} plans need regions; set regions.source in the A1 settings.")


def _region_tag(population: PlanPopulation, region: int) -> str:
    """``r07_cotiers_mediterranes``: the region's number, then its name in plain lower-case letters.

    Examples
    --------
    >>> population = PlanPopulation(ids=pd.Index([]), coordinates=pd.DataFrame(), region_names={7: "Côtiers Méditérranés"})
    >>> _region_tag(population, 7)
    'r07_cotiers_mediterranes'
    """
    return f"r{region:02d}_{slug(population.region_names[region])}"


def slug(name) -> str:
    """A name in plain lower-case ASCII letters, digits and underscores: safe in file and metric names.

    Examples
    --------
    >>> slug("Côtiers Méditérranés"), slug("Guir - Ziz - Rhris"), slug("Oum Er Rbia")
    ('cotiers_mediterranes', 'guir_ziz_rhris', 'oum_er_rbia')
    """
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-z]+", "_", plain.lower()).strip("_")
