"""The sections of the Paper A1 settings file, read and checked one by one.

The file (``configs/a1/a1.yml``) names a main configuration (``main_config``) - the same kind of file
``main.py`` reads - and takes the data, the column lists and the split from it. What it adds is
what A1 needs on top: the target, the two model stages, the time-series percentiles, the carbonate
classes, the regions, the covariate groups and where the results go.

Everything is checked when the file is read, so a typo stops the run before any model is trained
rather than after an hour of it. :func:`yg_eo_soilnet.a1.config.load_a1_config` is the usual way
in: it reads these settings and the covariate groups together, and checks them against the main
configuration and the data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import yaml

# Re-exported: the classes are defined in strata, beside the function that cuts them.
from yg_eo_soilnet.a1.strata import FAO_CARBONATE_EDGES, FAO_CARBONATE_LABELS

__all__ = [
    "FAO_CARBONATE_EDGES",
    "FAO_CARBONATE_LABELS",
    "STAGE1_ESTIMATORS",
    "A1Settings",
    "AuxiliarySettings",
    "BootstrapSettings",
    "CarbonateSettings",
    "KFoldPlanSettings",
    "MixedPlanSettings",
    "PercentileSettings",
    "PlanSettings",
    "PreflightSettings",
    "RandomPlanSettings",
    "RegionSettings",
    "SampleYearSettings",
    "StageSettings",
    "load_settings",
    "settings_from_mapping",
]

#: ``regions.source: column`` - each point's region is read from a column of the data.
REGIONS_FROM_COLUMN = "column"
#: ``regions.source: polygons`` - each point's region is the outline in a polygon file it falls in.
REGIONS_FROM_POLYGONS = "polygons"
#: The values ``regions.source`` accepts, besides null (no regions).
REGION_SOURCES = (REGIONS_FROM_COLUMN, REGIONS_FROM_POLYGONS)

#: The estimators stage 1 may use to predict the auxiliary properties.
STAGE1_ESTIMATORS = ("TabICL", "XGBoost")

#: ``plans.spatial_kfold.method``: blocks from k-means on the coordinates, or from a regular grid.
KFOLD_METHODS = ("kmeans", "grid")

#: The keys of the ``data:`` block, which must agree with the main configuration's.
DATA_KEYS = ("root", "static", "targets", "timeseries")


@dataclass(frozen=True)
class PercentileSettings:
    """Which summaries of the time series become inputs.

    Attributes
    ----------
    quantiles : tuple of float
        The percentiles to take of each variable over a point's whole record, between 0 and 100.
    variables : tuple of str or None
        The time-series columns to summarise; None takes every numeric column.
    exclude : tuple of str
        Columns never summarised - acquisition geometry such as sun or incidence angles, which
        describe the satellite rather than the soil.
    """

    quantiles: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)
    variables: Optional[tuple[str, ...]] = None
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuxiliarySettings:
    """The soil properties stage 1 predicts, for stage 2 to use as inputs.

    Attributes
    ----------
    targets : tuple of str
        The lab columns to predict, by default total CaCO3 and CEC.
    """

    targets: tuple[str, ...] = ("caco3_pct_total", "c_e_c_meq_100g")


@dataclass(frozen=True)
class StageSettings:
    """One model stage.

    Attributes
    ----------
    estimator : str
        The scikit-learn model-list entry (``configs/sklearn/model_registry.yml``) it trains. The
        entry does not need ``enabled: true`` there to be used here.
    import_path : str or None
        Replaces the entry's ``import_path`` (the estimator class), when set.
    init_args : dict
        Laid over the entry's ``init_args``: a key here replaces the entry's, the others are kept.
    params : dict or None
        Replaces the entry's ``params`` grid when set: each setting (``model__max_depth``) with the
        list of values to try. Stage 2 is tuned over it by inner cross-validation; stage 1 is not
        tuned and takes one value per setting.
    """

    estimator: str
    import_path: Optional[str] = None
    init_args: Mapping[str, Any] = field(default_factory=dict)
    params: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class BootstrapSettings:
    """How the confidence intervals of the scores are drawn.

    Attributes
    ----------
    block_size_km : float
        The side of the square blocks points are resampled in. Points closer than this are alike,
        so resampling them one at a time would make the intervals too narrow.
    """

    block_size_km: float = 10.0


@dataclass(frozen=True)
class CarbonateSettings:
    """How points are grouped by how calcareous they are.

    Attributes
    ----------
    column : str
        The CaCO3 column, in %.
    edges : tuple of float
        The class boundaries. A value on a boundary belongs to the class above it.
    labels : tuple of str
        One name per class, one more than there are edges.
    """

    column: str = "caco3_pct_total"
    edges: tuple[float, ...] = FAO_CARBONATE_EDGES
    labels: tuple[str, ...] = FAO_CARBONATE_LABELS


@dataclass(frozen=True)
class RegionSettings:
    """Where each point's region comes from: a column in the data, or a file of region outlines.

    Attributes
    ----------
    source : {"column", "polygons"} or None
        Which of the two it is; None means the data has no regions, and the experiments that need
        them say so rather than run.
    column : str or None
        With ``source: column``, the column holding each point's region. It may sit in the static
        file or in the targets file.
    polygons_path : str or None
        With ``source: polygons``, a polygon file (GeoJSON, GeoPackage, shapefile) read with
        geopandas. A relative path is taken from the data folder, like the data files themselves.
    name_field : str or None
        With ``source: polygons``, the polygon file's field holding each region's name.
    """

    source: Optional[str] = None
    column: Optional[str] = None
    polygons_path: Optional[str] = None
    name_field: Optional[str] = None


@dataclass(frozen=True)
class SampleYearSettings:
    """Where each point's sampling year comes from.

    Attributes
    ----------
    column : str or None
        A column holding the sampling date or year. A date has its year taken; a number is used as
        it is. None means the year is unknown, and the diagnostics that need it are skipped.
    file : str or None
        A separate table holding that column with the point id, when neither data file has it. A
        relative path is taken from the data folder. A point listed more than once keeps its first
        year.
    """

    column: Optional[str] = None
    file: Optional[str] = None


@dataclass(frozen=True)
class RandomPlanSettings:
    """The random baseline split: points held out one at a time.

    Attributes
    ----------
    test_size, val_size : float
        Shares of all the points.
    """

    test_size: float = 0.15
    val_size: float = 0.15


@dataclass(frozen=True)
class KFoldPlanSettings:
    """The spatial k-fold splits: blocks of nearby points, packed into folds, each fold tested once.

    Attributes
    ----------
    method : {"kmeans", "grid"}
        How the blocks are made: k-means on the coordinates, or the squares of a regular grid.
    n_clusters : int
        With ``kmeans``, how many blocks.
    cell_size_m : float
        With ``grid``, the side of a square, in metres.
    n_folds : int
        How many folds the blocks are packed into; one plan per fold.
    val_size : float
        The share of all points held out for validation, as whole blocks of the other folds.
    """

    method: str = "kmeans"
    n_clusters: int = 50
    cell_size_m: float = 10_000.0
    n_folds: int = 5
    val_size: float = 0.15


@dataclass(frozen=True)
class MixedPlanSettings:
    """The mixed-transfer splits: a share of the held-out region moved into training.

    Attributes
    ----------
    fractions : tuple of float
        The shares of the region's points moved into training, strictly between 0 and 1.
    n_repeats : int
        How many different draws of each share; repeat ``r`` is drawn with seed ``seed + r``.
    """

    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    n_repeats: int = 5


@dataclass(frozen=True)
class PlanSettings:
    """The split plans ``experiments/a1/make_plans.py`` writes; see :mod:`yg_eo_soilnet.a1.plans`."""

    random: RandomPlanSettings = field(default_factory=RandomPlanSettings)
    spatial_kfold: KFoldPlanSettings = field(default_factory=KFoldPlanSettings)
    mixed: MixedPlanSettings = field(default_factory=MixedPlanSettings)


def _default_prediction_columns() -> dict[str, str]:
    """``<target>__TabICL`` for each default auxiliary target."""
    return {target: f"{target}__TabICL" for target in AuxiliarySettings().targets}


@dataclass(frozen=True)
class PreflightSettings:
    """The thresholds of the pre-flight diagnostics (``experiments/a1/preflight.py``).

    The smallest acceptable cell is :attr:`A1Settings.min_cell_size`, shared with the experiments.

    Attributes
    ----------
    overfit_r2_gap : float
        An existing prediction column whose R² on the training points beats its R² on the test
        points by more than this is flagged, as the mark of a model that saw those training points.
    prediction_columns : dict of str to str
        Lab column to the column holding an earlier model's prediction of it.
    """

    overfit_r2_gap: float = 0.1
    prediction_columns: dict[str, str] = field(default_factory=_default_prediction_columns)


@dataclass(frozen=True)
class A1Settings:
    """Everything an A1 experiment reads, besides the main configuration and the covariate groups.

    Attributes
    ----------
    main_config_path : str
        The main configuration file: data paths, column lists, the split.
    target : str
        The lab column being predicted.
    stage1, stage2 : StageSettings
        The estimator predicting the auxiliary properties, and the one predicting the target.
    inner_cv_folds : int
        How many folds the training points are split into inside each stage: for stage 1, so each
        point's prediction comes from a model that never saw it; for both, to choose settings.
    seeds : tuple of int
        The random seeds. The first one seeds the splits and folds; each further one repeats the
        experiment, so the spread between repeats can be reported.
    output_dir : str
        Where the tables and figures are written as plain files, one folder per experiment.
    mlflow_experiment : str
        The MLflow experiment every A1 run goes into.
    min_cell_size : int
        The fewest points a group - a region, a carbonate class, a region x class cell - may hold
        and still get its own model or its own score.
    covariate_groups_path : str or None
        The covariate-groups file, already resolved to a full path.
    data : dict of str to str
        The ``data:`` block, as written; checked against the main configuration's.
    percentiles, auxiliary, bootstrap, carbonate, regions, sample_year, preflight, plans
        The settings of each part; see their own classes.
    source_path : str or None
        The file these settings were read from.
    """

    main_config_path: str
    target: str = "organic_matter_g_kg"
    stage1: StageSettings = field(default_factory=lambda: StageSettings("TabICL"))
    stage2: StageSettings = field(default_factory=lambda: StageSettings("XGBoost"))
    inner_cv_folds: int = 5
    seeds: tuple[int, ...] = (42,)
    output_dir: str = "results/a1"
    mlflow_experiment: str = "Paper_A1"
    min_cell_size: int = 30
    covariate_groups_path: Optional[str] = None
    data: Mapping[str, str] = field(default_factory=dict)
    percentiles: PercentileSettings = field(default_factory=PercentileSettings)
    auxiliary: AuxiliarySettings = field(default_factory=AuxiliarySettings)
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)
    carbonate: CarbonateSettings = field(default_factory=CarbonateSettings)
    regions: RegionSettings = field(default_factory=RegionSettings)
    sample_year: SampleYearSettings = field(default_factory=SampleYearSettings)
    preflight: PreflightSettings = field(default_factory=PreflightSettings)
    plans: PlanSettings = field(default_factory=PlanSettings)
    source_path: Optional[str] = None

    def __post_init__(self):
        _validate(self)

    @property
    def seed(self) -> int:
        """The seed of the splits and folds: the first of :attr:`seeds`."""
        return self.seeds[0]

    def load_config(self):
        """Read the main configuration these settings point to.

        Returns
        -------
        Config
        """
        from config import Config  # local: the project's Config sits at the repository root

        return Config(config_path=self.main_config_path)


def load_settings(path: str) -> A1Settings:
    """Read and check an A1 settings file.

    Parameters
    ----------
    path : str
        The YAML file, such as ``configs/a1/a1_demo.yml``. A relative ``main_config`` or
        ``covariate_groups`` inside it is taken from the file's own folder.

    Returns
    -------
    A1Settings

    Raises
    ------
    FileNotFoundError
        If the file, or the main configuration it names, does not exist.
    ValueError
        If a setting is missing, unknown or out of range.

    Examples
    --------
    >>> settings = load_settings("configs/a1/a1_demo.yml")
    >>> settings.target, settings.auxiliary.targets
    ('organic_matter_g_kg', ('caco3_pct_total', 'c_e_c_meq_100g'))
    >>> settings.stage1.estimator, settings.stage2.estimator, settings.inner_cv_folds
    ('TabICL', 'XGBoost', 5)
    >>> settings.carbonate.labels
    ('<2', '2-10', '10-25', '>=25')
    >>> settings.regions.source, settings.regions.column
    ('column', 'region')
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"A1 settings file not found: {path}")
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must hold a mapping of settings; got {type(raw).__name__}")
    return settings_from_mapping(raw, base_dir=os.path.dirname(os.path.abspath(path)), source_path=path)


def settings_from_mapping(raw: Mapping[str, Any], *, base_dir: str = ".", source_path: Optional[str] = None):
    """Build :class:`A1Settings` from an already-read mapping.

    Parameters
    ----------
    raw : mapping
        The settings, laid out as in the YAML file.
    base_dir : str, default "."
        The folder a relative ``main_config`` or ``covariate_groups`` is taken from.
    source_path : str, optional
        Recorded on the result, for the run's provenance.

    Returns
    -------
    A1Settings

    Raises
    ------
    FileNotFoundError
        If the main configuration or the covariate-groups file does not exist.
    ValueError
        If a setting is missing, unknown or out of range.
    """
    known = {
        "main_config",
        "data",
        "target",
        "stage1",
        "stage2",
        "inner_cv_folds",
        "seeds",
        "output_dir",
        "mlflow_experiment",
        "min_cell_size",
        "covariate_groups",
        "timeseries_percentiles",
        "auxiliary",
        "bootstrap",
        "carbonate",
        "regions",
        "sample_year",
        "preflight",
        "plans",
    }
    _refuse_unknown(raw, known, "the A1 settings")
    main_config = raw.get("main_config")
    if not main_config:
        raise ValueError("The A1 settings need main_config: the main configuration naming the data files.")
    main_config_path = _existing_path(main_config, base_dir, "main_config")
    groups = raw.get("covariate_groups")
    groups_path = _existing_path(groups, base_dir, "covariate_groups") if groups else None

    data = raw.get("data") or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"data must be a mapping of {list(DATA_KEYS)}; got {data!r}")
    _refuse_unknown(data, set(DATA_KEYS), "data")

    # Only what the file sets is passed on; the dataclass's own defaults apply to everything else.
    kwargs: dict[str, Any] = {
        "main_config_path": main_config_path,
        "covariate_groups_path": groups_path,
        "data": {key: str(value) for key, value in data.items() if value is not None},
        "percentiles": _section(raw, "timeseries_percentiles", PercentileSettings, _percentile_kwargs),
        "auxiliary": _section(raw, "auxiliary", AuxiliarySettings, _auxiliary_kwargs),
        "bootstrap": _section(raw, "bootstrap", BootstrapSettings, _bootstrap_kwargs),
        "carbonate": _section(raw, "carbonate", CarbonateSettings, _carbonate_kwargs),
        "regions": _section(raw, "regions", RegionSettings, _region_kwargs),
        "sample_year": _section(raw, "sample_year", SampleYearSettings, _optional_strings),
        "preflight": _section(raw, "preflight", PreflightSettings, _preflight_kwargs),
        "plans": _plan_settings(raw.get("plans") or {}),
        "source_path": source_path,
    }
    for name in ("stage1", "stage2"):
        if name in raw:
            kwargs[name] = _section(raw, name, StageSettings, _stage_kwargs(name))
    for name, cast in (("target", str), ("output_dir", str), ("mlflow_experiment", str)):
        if raw.get(name) is not None:
            kwargs[name] = cast(raw[name])
    for name in ("inner_cv_folds", "min_cell_size"):
        if raw.get(name) is not None:
            kwargs[name] = int(raw[name])
    if "seeds" in raw:
        kwargs["seeds"] = tuple(int(seed) for seed in _strings(raw["seeds"], "seeds"))
    return A1Settings(**kwargs)


# --- reading each section -----------------------------------------------------------


def _existing_path(path: str, base_dir: str, setting: str) -> str:
    """A path named in the settings, taken from ``base_dir`` when relative, which must exist."""
    resolved = os.path.normpath(path if os.path.isabs(path) else os.path.join(base_dir, path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{setting} {path!r} not found (looked for {resolved})")
    return resolved


def _section(raw: Mapping, name: str, cls, to_kwargs, *, prefix: str = ""):
    """One section of the file as its settings class; a missing section takes every default."""
    values = raw.get(name) or {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{prefix}{name} must be a mapping of settings; got {type(values).__name__}")
    _refuse_unknown(values, set(cls.__dataclass_fields__), f"{prefix}{name}")
    return cls(**to_kwargs(values))


def _stage_kwargs(name: str):
    """The reader of one stage section, which must name its estimator."""

    def read(values: Mapping) -> dict:
        """The stage's settings, converted to their types."""
        if not values.get("estimator"):
            raise ValueError(f"{name}.estimator must name a scikit-learn model-list entry.")
        kwargs: dict[str, Any] = {"estimator": str(values["estimator"])}
        if values.get("import_path"):
            kwargs["import_path"] = str(values["import_path"])
        init_args = values.get("init_args") or {}
        if not isinstance(init_args, Mapping):
            raise ValueError(f"{name}.init_args must be a mapping of argument to value; got {init_args!r}")
        kwargs["init_args"] = dict(init_args)
        if values.get("params") is not None:
            params = values["params"]
            if not isinstance(params, Mapping):
                raise ValueError(f"{name}.params must map each setting to a list of values; got {params!r}")
            kwargs["params"] = {
                str(key): (list(value) if isinstance(value, (list, tuple)) else [value])
                for key, value in params.items()
            }
        return kwargs

    return read


def _percentile_kwargs(values: Mapping) -> dict:
    """The percentile section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "quantiles" in values:
        kwargs["quantiles"] = tuple(float(q) for q in values["quantiles"])
    if values.get("variables") is not None:
        kwargs["variables"] = _strings(values["variables"], "timeseries_percentiles.variables")
    if "exclude" in values:
        kwargs["exclude"] = _strings(values["exclude"] or (), "timeseries_percentiles.exclude")
    return kwargs


def _auxiliary_kwargs(values: Mapping) -> dict:
    """The auxiliary section's settings, converted to their types."""
    return {"targets": _strings(values["targets"], "auxiliary.targets")} if "targets" in values else {}


def _bootstrap_kwargs(values: Mapping) -> dict:
    """The bootstrap section's settings, converted to their types."""
    return {"block_size_km": float(values["block_size_km"])} if "block_size_km" in values else {}


def _carbonate_kwargs(values: Mapping) -> dict:
    """The carbonate section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "column" in values:
        kwargs["column"] = str(values["column"])
    if "edges" in values:
        kwargs["edges"] = tuple(float(edge) for edge in values["edges"])
    if "labels" in values:
        kwargs["labels"] = _strings(values["labels"], "carbonate.labels")
    return kwargs


def _region_kwargs(values: Mapping) -> dict:
    """The region section's settings, converted to their types."""
    return _optional_strings(values)


def _optional_strings(values: Mapping) -> dict:
    """Every setting of a section that is set, as a string."""
    return {key: str(value) for key, value in values.items() if value is not None}


def _plan_settings(values: Mapping) -> PlanSettings:
    """The ``plans:`` section, one sub-section per kind of plan."""
    if not isinstance(values, Mapping):
        raise ValueError(f"plans must be a mapping of settings; got {type(values).__name__}")
    _refuse_unknown(values, set(PlanSettings.__dataclass_fields__), "plans")
    return PlanSettings(
        random=_section(values, "random", RandomPlanSettings, _typed(RandomPlanSettings), prefix="plans."),
        spatial_kfold=_section(values, "spatial_kfold", KFoldPlanSettings, _typed(KFoldPlanSettings), prefix="plans."),
        mixed=_section(values, "mixed", MixedPlanSettings, _mixed_kwargs, prefix="plans."),
    )


def _typed(cls):
    """A reader converting each setting of a section to the type of its default."""
    defaults = {name: spec.default for name, spec in cls.__dataclass_fields__.items()}

    def read(values: Mapping) -> dict:
        """The section's settings, converted to their types."""
        return {key: type(defaults[key])(value) for key, value in values.items() if value is not None}

    return read


def _mixed_kwargs(values: Mapping) -> dict:
    """The mixed-plan section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "fractions" in values:
        kwargs["fractions"] = tuple(float(p) for p in _strings(values["fractions"], "plans.mixed.fractions"))
    if "n_repeats" in values:
        kwargs["n_repeats"] = int(values["n_repeats"])
    return kwargs


def _preflight_kwargs(values: Mapping) -> dict:
    """The pre-flight section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "overfit_r2_gap" in values:
        kwargs["overfit_r2_gap"] = float(values["overfit_r2_gap"])
    if "prediction_columns" in values:
        columns = values["prediction_columns"] or {}
        if not isinstance(columns, Mapping):
            raise ValueError(
                f"preflight.prediction_columns must map each lab column to its prediction column; got {columns!r}"
            )
        kwargs["prediction_columns"] = {str(target): str(column) for target, column in columns.items()}
    return kwargs


# --- checks ---------------------------------------------------------------------------


def _validate(settings: A1Settings) -> None:
    """Refuse settings that would fail later, or worse, run and mean something else."""
    if settings.stage1.estimator not in STAGE1_ESTIMATORS:
        raise ValueError(
            f"stage1.estimator must be one of {list(STAGE1_ESTIMATORS)}; got {settings.stage1.estimator!r}"
        )
    stage1_grid = settings.stage1.params or {}
    several = sorted(key for key, values in stage1_grid.items() if len(values) > 1)
    if several:
        raise ValueError(
            f"stage1.params lists several values for {several}; stage 1 fits one model per fold and is "
            "not tuned. Keep one value for each, or tune stage 2 instead."
        )
    if settings.inner_cv_folds < 2:
        raise ValueError(f"inner_cv_folds must be at least 2; got {settings.inner_cv_folds}")
    if not settings.seeds:
        raise ValueError("seeds must list at least one random seed.")
    if len(set(settings.seeds)) != len(settings.seeds):
        raise ValueError(f"seeds repeats a value, which would repeat a run rather than vary it: {list(settings.seeds)}")
    if settings.min_cell_size < 1:
        raise ValueError(f"min_cell_size must be at least 1; got {settings.min_cell_size}")
    if not settings.bootstrap.block_size_km > 0:
        raise ValueError(f"bootstrap.block_size_km must be greater than 0; got {settings.bootstrap.block_size_km}")

    quantiles = settings.percentiles.quantiles
    if not quantiles:
        raise ValueError("timeseries_percentiles.quantiles must list at least one percentile.")
    if any(not 0.0 < q < 100.0 for q in quantiles):
        raise ValueError(f"timeseries_percentiles.quantiles must lie strictly between 0 and 100; got {list(quantiles)}")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError(f"timeseries_percentiles.quantiles repeats a value: {list(quantiles)}")

    if settings.target in settings.auxiliary.targets:
        # Predicting SOM in stage 1 and feeding the prediction to stage 2 as an input for SOM would
        # be the model being handed its own answer.
        raise ValueError(f"auxiliary.targets must not include the target {settings.target!r}.")

    carbonate = settings.carbonate
    edges = list(carbonate.edges)
    if not edges or any(later <= earlier for earlier, later in zip(edges, edges[1:])):
        raise ValueError(f"carbonate.edges must be one or more values in increasing order; got {edges}")
    if len(carbonate.labels) != len(edges) + 1:
        raise ValueError(
            f"carbonate.labels needs one name per class - {len(edges) + 1} for {len(edges)} edge(s); "
            f"got {len(carbonate.labels)}"
        )

    _validate_regions(settings.regions)

    if settings.sample_year.file and not settings.sample_year.column:
        raise ValueError("sample_year.file needs sample_year.column, the column in it holding the date or year.")
    if settings.preflight.overfit_r2_gap < 0:
        raise ValueError(f"preflight.overfit_r2_gap must not be negative; got {settings.preflight.overfit_r2_gap}")

    _validate_plans(settings.plans)


def _validate_plans(plans: PlanSettings) -> None:
    """Refuse split-plan settings that could not produce a sound plan."""
    random = plans.random
    for name, value in (("test_size", random.test_size), ("val_size", random.val_size)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"plans.random.{name} must lie strictly between 0 and 1; got {value}")
    if random.test_size + random.val_size >= 1.0:
        raise ValueError(
            f"plans.random.test_size + val_size must leave a training set; got {random.test_size} + {random.val_size}"
        )

    kfold = plans.spatial_kfold
    if kfold.method not in KFOLD_METHODS:
        raise ValueError(f"plans.spatial_kfold.method must be one of {list(KFOLD_METHODS)}; got {kfold.method!r}")
    if kfold.n_folds < 2:
        raise ValueError(f"plans.spatial_kfold.n_folds must be at least 2; got {kfold.n_folds}")
    if kfold.method == "kmeans" and kfold.n_clusters < kfold.n_folds:
        raise ValueError(
            f"plans.spatial_kfold.n_clusters ({kfold.n_clusters}) must be at least n_folds ({kfold.n_folds}): "
            "every fold needs a block of its own."
        )
    if kfold.method == "grid" and not kfold.cell_size_m > 0:
        raise ValueError(f"plans.spatial_kfold.cell_size_m must be greater than 0; got {kfold.cell_size_m}")
    if not 0.0 <= kfold.val_size < 1.0 - 1.0 / kfold.n_folds:
        raise ValueError(
            f"plans.spatial_kfold.val_size must be at least 0 and leave a training set beside one test fold; "
            f"got {kfold.val_size}"
        )

    mixed = plans.mixed
    if not mixed.fractions or any(not 0.0 < p < 1.0 for p in mixed.fractions):
        raise ValueError(f"plans.mixed.fractions must lie strictly between 0 and 1; got {list(mixed.fractions)}")
    if len(set(mixed.fractions)) != len(mixed.fractions):
        raise ValueError(f"plans.mixed.fractions repeats a value: {list(mixed.fractions)}")
    if mixed.n_repeats < 1:
        raise ValueError(f"plans.mixed.n_repeats must be at least 1; got {mixed.n_repeats}")


def _validate_regions(regions: RegionSettings) -> None:
    """Refuse a region source that is half set up, or set up two ways at once."""
    polygon_keys = {"polygons_path": regions.polygons_path, "name_field": regions.name_field}
    if regions.source is None:
        stray = [key for key, value in {"column": regions.column, **polygon_keys}.items() if value]
        if stray:
            raise ValueError(
                f"regions.{stray[0]} is set but regions.source is not. Set regions.source to "
                f"{REGIONS_FROM_COLUMN!r} or {REGIONS_FROM_POLYGONS!r} to say which one to use."
            )
        return
    if regions.source not in REGION_SOURCES:
        raise ValueError(f"regions.source must be one of {list(REGION_SOURCES)} or null; got {regions.source!r}")

    if regions.source == REGIONS_FROM_COLUMN:
        if not regions.column:
            raise ValueError("regions.source is 'column', so regions.column must name the column.")
        stray = [key for key, value in polygon_keys.items() if value]
        if stray:
            raise ValueError(f"regions.{stray[0]} belongs to source 'polygons', but regions.source is 'column'.")
        return

    missing = [key for key, value in polygon_keys.items() if not value]
    if missing:
        raise ValueError(
            f"regions.source is 'polygons', so regions.{missing[0]} must be set: the polygon file, and the "
            "field in it holding each region's name."
        )
    if regions.column:
        raise ValueError("regions.column belongs to source 'column', but regions.source is 'polygons'.")


def _refuse_unknown(values: Mapping, known: set, where: str) -> None:
    """Raise on a key nobody reads, which is nearly always a typo of one somebody does."""
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown setting(s) {unknown} in {where}; expected some of {sorted(known)}")


def _strings(value: Any, name: str) -> tuple[str, ...]:
    """A list of names as a tuple of strings; a single string is refused rather than split."""
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise ValueError(f"{name} must be a list; got {value!r}")
    return tuple(str(item) for item in value)
