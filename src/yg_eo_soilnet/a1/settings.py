"""The settings of the Paper A1 experiments, read from one YAML file in ``configs/a1/``.

The file does not repeat where the data is or which columns are lab measurements: it names a main
configuration (``main_config``), the same kind of file ``main.py`` reads, and takes all of that from
it. What it adds is only what A1 needs on top - which target, which time-series percentiles, which
properties to predict as extra inputs, the carbonate classes, the regions and where the results go.

Everything is checked when the file is read, so a typo stops the run before any model is trained
rather than after an hour of it.
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
    "A1Settings",
    "AuxiliarySettings",
    "CarbonateSettings",
    "PercentileSettings",
    "PreflightSettings",
    "RegionSettings",
    "SampleYearSettings",
    "TrackingSettings",
    "load_settings",
    "settings_from_mapping",
]

#: ``regions.source: column`` - each point's region is read from a column of the data.
REGIONS_FROM_COLUMN = "column"
#: ``regions.source: polygons`` - each point's region is the outline in a polygon file it falls in.
REGIONS_FROM_POLYGONS = "polygons"
#: The values ``regions.source`` accepts, besides null (no regions).
REGION_SOURCES = (REGIONS_FROM_COLUMN, REGIONS_FROM_POLYGONS)


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
    """The soil properties predicted first and then used as inputs for SOM.

    Attributes
    ----------
    targets : tuple of str
        The lab columns to predict, by default total CaCO3 and CEC.
    model : str
        The scikit-learn model-list entry that predicts them (``TabICL`` by default).
    n_folds : int
        How many folds the training points are split into, so that each one's prediction comes from
        a model that never saw it.
    """

    targets: tuple[str, ...] = ("caco3_pct_total", "c_e_c_meq_100g")
    model: str = "TabICL"
    n_folds: int = 5


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
    min_points : int
        A class with fewer points than this gets no model of its own.
    """

    column: str = "caco3_pct_total"
    edges: tuple[float, ...] = FAO_CARBONATE_EDGES
    labels: tuple[str, ...] = FAO_CARBONATE_LABELS
    min_points: int = 30


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
    min_points : int
        A region with fewer points than this takes no part in the transfer experiments.
    """

    source: Optional[str] = None
    column: Optional[str] = None
    polygons_path: Optional[str] = None
    name_field: Optional[str] = None
    min_points: int = 30


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


def _default_prediction_columns() -> dict[str, str]:
    """``<target>__TabICL`` for each default auxiliary target."""
    return {target: f"{target}__TabICL" for target in AuxiliarySettings().targets}


@dataclass(frozen=True)
class PreflightSettings:
    """The thresholds of the pre-flight diagnostics (``experiments/a1/preflight.py``).

    Attributes
    ----------
    min_cell_points : int
        A region x carbonate-class cell with fewer points than this is flagged: too few to fit or
        score a model on.
    overfit_r2_gap : float
        An existing prediction column whose R² on the training points beats its R² on the test
        points by more than this is flagged, as the mark of a model that saw those training points.
    prediction_columns : dict of str to str
        Lab column to the column holding an earlier model's prediction of it.
    """

    min_cell_points: int = 30
    overfit_r2_gap: float = 0.1
    prediction_columns: dict[str, str] = field(default_factory=_default_prediction_columns)


@dataclass(frozen=True)
class TrackingSettings:
    """Where the results are recorded.

    Attributes
    ----------
    experiment_name : str
        The MLflow experiment every A1 run goes into.
    results_dir : str
        Where the tables and figures are also written as plain files, one folder per experiment.
    """

    experiment_name: str = "Paper_A1"
    results_dir: str = "results/a1"


@dataclass(frozen=True)
class A1Settings:
    """Everything an A1 experiment reads, besides the main configuration it points to.

    Attributes
    ----------
    main_config_path : str
        The main configuration file: data paths, column lists, the split.
    som_target : str
        The lab column being predicted.
    models : tuple of str
        The scikit-learn model-list entries trained for SOM.
    seed : int
        The random seed of everything A1 adds (folds, per-region splits).
    percentiles, auxiliary, carbonate, regions, sample_year, preflight, tracking
        The settings of each part; see their own classes.
    source_path : str or None
        The file these settings were read from.
    """

    main_config_path: str
    som_target: str = "organic_matter_g_kg"
    models: tuple[str, ...] = ("XGBoost", "TabICL")
    seed: int = 42
    percentiles: PercentileSettings = field(default_factory=PercentileSettings)
    auxiliary: AuxiliarySettings = field(default_factory=AuxiliarySettings)
    carbonate: CarbonateSettings = field(default_factory=CarbonateSettings)
    regions: RegionSettings = field(default_factory=RegionSettings)
    sample_year: SampleYearSettings = field(default_factory=SampleYearSettings)
    preflight: PreflightSettings = field(default_factory=PreflightSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    source_path: Optional[str] = None

    def __post_init__(self):
        _validate(self)

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
        The YAML file, such as ``configs/a1/a1_demo.yml``. A relative ``main_config`` inside it is
        taken from the file's own folder.

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
    >>> settings.som_target, settings.auxiliary.targets
    ('organic_matter_g_kg', ('caco3_pct_total', 'c_e_c_meq_100g'))
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
        The folder a relative ``main_config`` is taken from.
    source_path : str, optional
        Recorded on the result, for the run's provenance.

    Returns
    -------
    A1Settings

    Raises
    ------
    FileNotFoundError
        If the main configuration does not exist.
    ValueError
        If a setting is missing, unknown or out of range.
    """
    known = {
        "main_config",
        "som_target",
        "models",
        "seed",
        "timeseries_percentiles",
        "auxiliary",
        "carbonate",
        "regions",
        "sample_year",
        "preflight",
        "tracking",
    }
    _refuse_unknown(raw, known, "the A1 settings")
    main_config = raw.get("main_config")
    if not main_config:
        raise ValueError("The A1 settings need main_config: the main configuration naming the data files.")
    main_config_path = main_config if os.path.isabs(main_config) else os.path.join(base_dir, main_config)
    main_config_path = os.path.normpath(main_config_path)
    if not os.path.exists(main_config_path):
        raise FileNotFoundError(f"main_config {main_config!r} not found (looked for {main_config_path})")

    defaults = A1Settings.__dataclass_fields__
    return A1Settings(
        main_config_path=main_config_path,
        som_target=str(raw.get("som_target", defaults["som_target"].default)),
        models=_strings(raw.get("models", defaults["models"].default), "models"),
        seed=int(raw.get("seed", defaults["seed"].default)),
        percentiles=_section(raw, "timeseries_percentiles", PercentileSettings, _percentile_kwargs),
        auxiliary=_section(raw, "auxiliary", AuxiliarySettings, _auxiliary_kwargs),
        carbonate=_section(raw, "carbonate", CarbonateSettings, _carbonate_kwargs),
        regions=_section(raw, "regions", RegionSettings, _region_kwargs),
        sample_year=_section(raw, "sample_year", SampleYearSettings, _sample_year_kwargs),
        preflight=_section(raw, "preflight", PreflightSettings, _preflight_kwargs),
        tracking=_section(raw, "tracking", TrackingSettings, _tracking_kwargs),
        source_path=source_path,
    )


# --- reading each section -----------------------------------------------------------


def _section(raw: Mapping, name: str, cls, to_kwargs):
    """One section of the file as its settings class; a missing section takes every default."""
    values = raw.get(name) or {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of settings; got {type(values).__name__}")
    _refuse_unknown(values, set(cls.__dataclass_fields__), name)
    return cls(**to_kwargs(values))


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
    kwargs: dict[str, Any] = {}
    if "targets" in values:
        kwargs["targets"] = _strings(values["targets"], "auxiliary.targets")
    if "model" in values:
        kwargs["model"] = str(values["model"])
    if "n_folds" in values:
        kwargs["n_folds"] = int(values["n_folds"])
    return kwargs


def _carbonate_kwargs(values: Mapping) -> dict:
    """The carbonate section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "column" in values:
        kwargs["column"] = str(values["column"])
    if "edges" in values:
        kwargs["edges"] = tuple(float(edge) for edge in values["edges"])
    if "labels" in values:
        kwargs["labels"] = _strings(values["labels"], "carbonate.labels")
    if "min_points" in values:
        kwargs["min_points"] = int(values["min_points"])
    return kwargs


def _region_kwargs(values: Mapping) -> dict:
    """The region section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    for key in ("source", "column", "polygons_path", "name_field"):
        if values.get(key) is not None:
            kwargs[key] = str(values[key])
    if "min_points" in values:
        kwargs["min_points"] = int(values["min_points"])
    return kwargs


def _sample_year_kwargs(values: Mapping) -> dict:
    """The sample-year section's settings, converted to their types."""
    return {key: str(values[key]) for key in ("column", "file") if values.get(key) is not None}


def _preflight_kwargs(values: Mapping) -> dict:
    """The pre-flight section's settings, converted to their types."""
    kwargs: dict[str, Any] = {}
    if "min_cell_points" in values:
        kwargs["min_cell_points"] = int(values["min_cell_points"])
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


def _tracking_kwargs(values: Mapping) -> dict:
    """The tracking section's settings, converted to their types."""
    return {key: str(values[key]) for key in ("experiment_name", "results_dir") if values.get(key) is not None}


# --- checks ---------------------------------------------------------------------------


def _validate(settings: A1Settings) -> None:
    """Refuse settings that would fail later, or worse, run and mean something else."""
    if not settings.models:
        raise ValueError("models must name at least one scikit-learn model-list entry to train for SOM.")

    quantiles = settings.percentiles.quantiles
    if not quantiles:
        raise ValueError("timeseries_percentiles.quantiles must list at least one percentile.")
    if any(not 0.0 < q < 100.0 for q in quantiles):
        raise ValueError(f"timeseries_percentiles.quantiles must lie strictly between 0 and 100; got {list(quantiles)}")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError(f"timeseries_percentiles.quantiles repeats a value: {list(quantiles)}")

    auxiliary = settings.auxiliary
    if settings.som_target in auxiliary.targets:
        # Predicting SOM and then feeding the prediction back in as an input for SOM would be the
        # model being handed its own answer.
        raise ValueError(f"auxiliary.targets must not include the SOM target {settings.som_target!r}.")
    if auxiliary.n_folds < 2:
        raise ValueError(f"auxiliary.n_folds must be at least 2; got {auxiliary.n_folds}")

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

    preflight = settings.preflight
    if preflight.overfit_r2_gap < 0:
        raise ValueError(f"preflight.overfit_r2_gap must not be negative; got {preflight.overfit_r2_gap}")

    for name, value in (
        ("carbonate.min_points", carbonate.min_points),
        ("regions.min_points", settings.regions.min_points),
        ("preflight.min_cell_points", preflight.min_cell_points),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1; got {value}")


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
