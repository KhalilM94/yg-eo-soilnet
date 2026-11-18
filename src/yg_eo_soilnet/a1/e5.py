"""E5: what each input group adds, which time-series summary to use, and a hook for new covariates.

Global only (stage 2, the E1 grid, no stage 1), on E1's designs - the random split, the 10 km block
folds and the five held-out basins - and seeds. Each **variant** is a named list of input columns,
fitted by :func:`~yg_eo_soilnet.a1.runner.run_plans` exactly as Global is (same plans, folds, grid
and seeds), so two variants are compared point by point.

Part A, covariate ablation (:func:`part_a_variants`)
    ``all`` (Global's inputs), ``all_minus_<group>`` for each covariate group and each satellite
    sub-block (:data:`SUBBLOCKS`), ``only_satellite``, ``satellite_plus_terrain`` and
    ``all_plus_xy`` (the projected coordinates the blocks are cut on - an interpolation check,
    not a transfer model).
Part B, time-series representations (:func:`part_b_variants`)
    Every time-series percentile column (satellite and climate alike) replaced by another
    summary of each point's own monthly composites (:data:`REPRESENTATIONS`), with every static
    column kept; on ``all``, on ``all_minus_legacy_soil`` (so the comparison does not rest on the
    soil-map layers), and on ``only_satellite`` for R2 and R6.
Part C, external covariates (:func:`extra_covariates`)
    A CSV keyed by point id becomes a group: ``all_plus_<group>`` and ``only_<group>``.

The time series are **monthly composites** (2017-01 to 2025-12, about 100 per point). The static
percentiles (R0) are over every composite of that record, so they include years after sampling.
"""

from __future__ import annotations

import logging
import re
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.runner import Variant

logger = logging.getLogger(__name__)

#: The covariate groups of Part A (every group of Global's inputs; ``auxiliary`` is never one).
PART_A_GROUPS = ("satellite", "climate", "terrain", "management", "legacy_soil")
#: The satellite sub-blocks of Part A: name to the column prefixes it holds.
SUBBLOCKS = {
    "s2_barest": ("s2_",),
    "s2_ts": ("S2_",),
    "s1_ts": ("S1_VH_", "S1_VV_", "S1_VH/VV_"),
    "veg_ts": ("AG_",),
    "soilvar_ts": ("SOIL_",),
}
#: The time-series variable prefixes: a percentile column of one of these is a time-series summary.
TS_PREFIXES = ("S2_", "S1_", "AG_", "SOIL_", "CLIM_")
#: The satellite time-series prefixes (``only_satellite`` with a representation keeps these).
SATELLITE_TS_PREFIXES = ("S2_", "S1_", "AG_", "SOIL_")
#: A percentile column of the static file: ``<variable>_q10`` ... ``_q90`` and ``<variable>_iqr``.
PERCENTILE = re.compile(r"^(?P<variable>.+)_(?P<stat>q\d\d|iqr)$")
#: The time-series representations of Part B, with what each keeps.
REPRESENTATIONS = {
    "R0": "whole-record q10, q25, q50, q75, q90 and iqr (the static columns; the reference)",
    "R1": "whole-record q50 and iqr only",
    "R2": "median per calendar quarter (DJF, MAM, JJA, SON) over the record",
    "R3": "median per calendar quarter of the sampling year only",
    "R4": "harmonic fit: mean, amplitude and phase (sin, cos) of the annual term, semi-annual amplitude, residual SD",
    "R5": "PCA of the 12 monthly medians of S2 B11 and B12, fitted on each plan's training points",
    "R6": "R0, with the Sentinel-2 bands' percentiles over bare months only (NDVI < 0.25, NBR2 < 0.05)",
}
#: The quarters: name to calendar months.
QUARTERS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
#: The angular frequency of the annual cycle, per day.
OMEGA = 2.0 * np.pi / 365.25
#: The whole-record percentiles of R0 and R6.
QUANTILES = (10, 25, 50, 75, 90)
#: The projected coordinates of ``all_plus_xy``.
XY_COLUMNS = ("e5_x_m", "e5_y_m")
#: The column of R6 holding each point's number of bare months.
BARE_N = "s2_bare_n"
#: The Sentinel-2 reflectance bands of the time series.
S2_BANDS = ("S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7", "S2_B8", "S2_B8A", "S2_B11", "S2_B12")
#: The designs every comparison is reported on.
DESIGN_RANDOM, DESIGN_10KM, DESIGN_LOGO = "random", "10 km", "logo pooled"


# --- settings --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtraCovariates:
    """One external covariate file (Part C): a CSV keyed by the point id column."""

    path: Optional[str]
    group: str
    id_column: str = "uuid"


@dataclass(frozen=True)
class E5Settings:
    """The ``e5`` block of the settings file (:func:`e5_settings`)."""

    timeseries_file: Optional[str] = None
    timeseries_id_column: str = "uuid"
    timeseries_time_column: str = "observation_date"
    representations: tuple = ("R1", "R2", "R3", "R4", "R6")
    r5_enabled: bool = False
    r5_components: int = 5
    ndvi_max: float = 0.25
    nbr2_max: float = 0.05
    min_bare_months: int = 3
    quarter_min_readings: int = 2
    harmonic_min_readings: int = 12
    variants: Optional[tuple] = None
    extra_covariates: tuple = field(default_factory=tuple)


E5_KEYS = {
    "timeseries_file",
    "timeseries_id_column",
    "timeseries_time_column",
    "representations",
    "r5_pca",
    "bare_soil",
    "quarter_min_readings",
    "harmonic_min_readings",
    "variants",
    "extra_covariates",
}


def e5_settings(settings) -> E5Settings:
    """Read and check the ``e5`` block of A1 settings.

    Raises
    ------
    ValueError
        On an unknown key, an unknown representation, or a malformed extra-covariate entry.
    """
    raw = dict(getattr(settings, "e5", None) or {})
    unknown = sorted(set(raw) - E5_KEYS)
    if unknown:
        raise ValueError(f"Unknown e5 setting(s) {unknown}; expected some of {sorted(E5_KEYS)}")
    reps = tuple(raw.get("representations") or E5Settings.representations)
    bad = [rep for rep in reps if rep not in REPRESENTATIONS or rep == "R0"]
    if bad:
        raise ValueError(f"e5.representations: {bad} unknown (R0 is the reference); expected some of R1-R6")
    r5 = dict(raw.get("r5_pca") or {})
    bare = dict(raw.get("bare_soil") or {})
    extras = []
    for entry in raw.get("extra_covariates") or ():
        if not isinstance(entry, dict) or "group" not in entry:
            raise ValueError(f"e5.extra_covariates entries need group (and path, id_column); got {entry!r}")
        group = str(entry["group"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", group):
            raise ValueError(f"e5.extra_covariates group {group!r}: lower-case letters, digits and _ only")
        extras.append(ExtraCovariates(entry.get("path"), group, str(entry.get("id_column") or "uuid")))
    return E5Settings(
        timeseries_file=raw.get("timeseries_file"),
        timeseries_id_column=str(raw.get("timeseries_id_column") or "uuid"),
        timeseries_time_column=str(raw.get("timeseries_time_column") or "observation_date"),
        representations=reps,
        r5_enabled=bool(r5.get("enabled", False)),
        r5_components=int(r5.get("n_components", 5)),
        ndvi_max=float(bare.get("ndvi_max", 0.25)),
        nbr2_max=float(bare.get("nbr2_max", 0.05)),
        min_bare_months=int(bare.get("min_months", 3)),
        quarter_min_readings=int(raw.get("quarter_min_readings", 2)),
        harmonic_min_readings=int(raw.get("harmonic_min_readings", 12)),
        variants=tuple(raw["variants"]) if raw.get("variants") else None,
        extra_covariates=tuple(extras),
    )


# --- the time series ---------------------------------------------------------------------------------


def static_variable(column: str) -> str:
    """A time-series file column under the static file's name: ``S1_asc_VH`` -> ``S1_VH_asc``.

    >>> static_variable("S1_asc_VH/VV"), static_variable("S2_B4")
    ('S1_VH/VV_asc', 'S2_B4')
    """
    return re.sub(r"^S1_(asc|desc)_(.+)$", r"S1_\2_\1", column)


#: The indices computed from the Sentinel-2 bands of a composite: missing when its bands are.
S2_DERIVED = ("AG_NDVI", "SOIL_NBR2", "SOIL_SOCI")


def clean_composites(ts: pd.DataFrame, *, log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The composites with their faults made missing, and a report of what was changed, per column.

    - An infinite value (an index divided by a zero band) becomes NaN.
    - A composite with any Sentinel-2 band exactly 0 is a no-data composite: a reflectance of exactly
      0 is not physical (in this data: 130 composites, nearly all December-February, with B2, B3 or
      B4 at 0 and the other bands plausible). Every Sentinel-2 band and every index computed from
      them (:data:`S2_DERIVED`) becomes NaN for it - so the bare-month test of R6 never sees it.

    Returns
    -------
    cleaned : pandas.DataFrame
    report : pandas.DataFrame
        Per column: ``inf`` (set to NaN), ``zero_band_composites`` (set to NaN), ``nan_before``,
        ``nan_after``.
    """
    log = log or logger
    cleaned = ts.copy()
    numeric = [
        c for c in cleaned.columns if c not in ("point_id", "date") and pd.api.types.is_numeric_dtype(cleaned[c])
    ]
    nan_before = cleaned[numeric].isna().sum()
    infinite = np.isinf(cleaned[numeric]).sum()
    cleaned[numeric] = cleaned[numeric].replace([np.inf, -np.inf], np.nan)
    bands = [c for c in numeric if c in S2_BANDS]
    derived = [c for c in numeric if c in S2_DERIVED]
    zero = (cleaned[bands] == 0).any(axis=1) if bands else pd.Series(False, index=cleaned.index)
    masked = pd.Series(0, index=numeric)
    for column in bands + derived:
        masked[column] = int((zero & cleaned[column].notna()).sum())
        cleaned.loc[zero, column] = np.nan
    report = pd.DataFrame(
        {
            "inf": infinite,
            "zero_band_composites": masked,
            "nan_before": nan_before,
            "nan_after": cleaned[numeric].isna().sum(),
        }
    ).rename_axis("column")
    if int(infinite.sum()) or int(zero.sum()):
        points = cleaned.loc[zero, "point_id"].nunique() if "point_id" in cleaned else 0
        log.warning(
            f"E5 time series: {int(infinite.sum())} infinite value(s) set to NaN "
            f"({', '.join(f'{c}: {int(n)}' for c, n in infinite.items() if n)}); {int(zero.sum())} composite(s) of "
            f"{points} point(s) with a Sentinel-2 band at exactly 0 treated as missing for every band and "
            f"{', '.join(derived)}."
        )
    return cleaned, report


def load_timeseries(path, *, id_column: str = "uuid", time_column: str = "observation_date") -> pd.DataFrame:
    """The monthly composites, one row per point and date, columns under the static file's names.

    As read: :func:`clean_composites` makes their faults missing before any representation is built.

    Returns
    -------
    pandas.DataFrame
        ``point_id, date`` and one numeric column per variable.
    """
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.rename(columns={column: static_variable(column) for column in frame.columns})
    frame = frame.rename(columns={id_column: "point_id", time_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["point_id"] = frame["point_id"].astype(str)
    return frame.dropna(subset=["date"])


def ts_variables(columns: Sequence[str]) -> dict:
    """The time-series variables among these input columns: variable to its percentile columns.

    >>> ts_variables(["S2_B4_q10", "S2_B4_iqr", "elevation", "CLIM_ETa_q50"])
    {'S2_B4': ['S2_B4_q10', 'S2_B4_iqr'], 'CLIM_ETa': ['CLIM_ETa_q50']}
    """
    found: dict = {}
    for column in columns:
        match = PERCENTILE.match(column)
        if match and match.group("variable").startswith(TS_PREFIXES):
            found.setdefault(match.group("variable"), []).append(column)
    return found


def _column(rep: str, variable: str, stat: str) -> str:
    return f"{rep}__{variable}__{stat}"


def quarter_medians(
    ts: pd.DataFrame,
    variables: Sequence[str],
    *,
    rep: str = "R2",
    min_readings: int = 2,
    years: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Each point's median of each variable per calendar quarter (R2), or within one year per point (R3).

    Parameters
    ----------
    ts : pandas.DataFrame
        :func:`load_timeseries`.
    variables : sequence of str
    rep : str
        The column prefix (``R2``, ``R3``).
    min_readings : int
        A quarter with fewer readings of the variable gets NaN.
    years : pandas.Series, optional
        Point id to the calendar year to keep (R3); a point without one gets NaN everywhere.

    Returns
    -------
    pandas.DataFrame
        Indexed by point id; columns ``<rep>__<variable>__<quarter>``.
    """
    frame = ts[["point_id", "date", *variables]].copy()
    if years is not None:
        frame = frame[frame["date"].dt.year.to_numpy() == frame["point_id"].map(years).to_numpy()]
    month = frame["date"].dt.month
    frame["quarter"] = month.map({m: name for name, months in QUARTERS.items() for m in months})
    grouped = frame.groupby(["point_id", "quarter"])[list(variables)]
    medians, counts = grouped.median(), grouped.count()
    medians = medians.where(counts >= min_readings)
    wide = medians.unstack("quarter")
    wide.columns = [_column(rep, variable, quarter) for variable, quarter in wide.columns]
    order = [_column(rep, variable, quarter) for variable in variables for quarter in QUARTERS]
    points = ts["point_id"].unique()
    return wide.reindex(index=points, columns=order).rename_axis("point_id")


def harmonic_terms(days: np.ndarray, values: np.ndarray, *, min_readings: int = 12) -> np.ndarray:
    """The R4 summary of one series: ``a0, amplitude1, sin(phase1), cos(phase1), amplitude2, residual SD``.

    Least-squares fit of ``a0 + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) + b2 sin(2wt)``, ``w`` the
    annual frequency. ``a1 cos(wt) + b1 sin(wt) = A cos(wt - phi)`` with ``A = sqrt(a1² + b1²)``,
    ``cos phi = a1 / A``, ``sin phi = b1 / A``. NaN with fewer than ``min_readings`` readings.

    >>> t = np.arange(0, 3 * 365.25, 30.0)
    >>> np.round(harmonic_terms(t, 5 + 2 * np.cos(OMEGA * t - 0.5)), 6).tolist()[:4]
    [5.0, 2.0, 0.479426, 0.877583]
    """
    keep = np.isfinite(values) & np.isfinite(days)
    out = np.full(6, np.nan)
    if keep.sum() < max(min_readings, 6):
        return out
    t, y = days[keep], values[keep]
    design = np.column_stack(
        [np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t), np.cos(2 * OMEGA * t), np.sin(2 * OMEGA * t)]
    )
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    a0, a1, b1, a2, b2 = coefficients
    amplitude1 = float(np.hypot(a1, b1))
    residual = y - design @ coefficients
    out[:] = [
        a0,
        amplitude1,
        b1 / amplitude1 if amplitude1 > 0 else np.nan,
        a1 / amplitude1 if amplitude1 > 0 else np.nan,
        float(np.hypot(a2, b2)),
        float(np.sqrt(np.sum(residual**2) / max(1, len(y) - 5))),
    ]
    return out


HARMONIC_STATS = ("a0", "amp1", "phase1_sin", "phase1_cos", "amp2", "resid_sd")


def harmonic_features(ts: pd.DataFrame, variables: Sequence[str], *, min_readings: int = 12) -> pd.DataFrame:
    """R4 for every point and variable: :func:`harmonic_terms`, columns ``R4__<variable>__<stat>``."""
    epoch = pd.Timestamp("2017-01-01")
    rows = {}
    for point, part in ts.groupby("point_id", sort=False):
        days = (part["date"] - epoch).dt.days.to_numpy(dtype=float)
        values = []
        for variable in variables:
            values.extend(harmonic_terms(days, part[variable].to_numpy(dtype=float), min_readings=min_readings))
        rows[point] = values
    columns = [_column("R4", variable, stat) for variable in variables for stat in HARMONIC_STATS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).rename_axis("point_id")


def bare_indices(ts: pd.DataFrame) -> pd.DataFrame:
    """NDVI and NBR2 of each composite, from its bands: (B8 - B4)/(B8 + B4) and (B11 - B12)/(B11 + B12)."""
    ndvi = (ts["S2_B8"] - ts["S2_B4"]) / (ts["S2_B8"] + ts["S2_B4"])
    nbr2 = (ts["S2_B11"] - ts["S2_B12"]) / (ts["S2_B11"] + ts["S2_B12"])
    return pd.DataFrame({"ndvi": ndvi, "nbr2": nbr2}, index=ts.index)


def _percentiles(frame: pd.DataFrame, variables: Sequence[str], rep: str) -> pd.DataFrame:
    grouped = frame.groupby("point_id")[list(variables)]
    parts = {}
    for q in QUANTILES:
        values = grouped.quantile(q / 100.0)
        for variable in variables:
            parts[_column(rep, variable, f"q{q:02d}")] = values[variable]
    table = pd.DataFrame(parts)
    for variable in variables:
        table[_column(rep, variable, "iqr")] = (
            table[_column(rep, variable, "q75")] - table[_column(rep, variable, "q25")]
        )
    return table


def bare_soil_percentiles(
    ts: pd.DataFrame,
    variables: Sequence[str],
    *,
    ndvi_max: float = 0.25,
    nbr2_max: float = 0.05,
    min_months: int = 3,
) -> pd.DataFrame:
    """R6: whole-record percentiles of every variable, the Sentinel-2 bands over bare months only.

    A bare month is a composite with NDVI < ``ndvi_max`` and NBR2 < ``nbr2_max`` (:func:`bare_indices`)
    - a bare-month filter on monthly composites, a weaker version of the scene-level rule of
    Dvorakova et al. (2023). A point with fewer than ``min_months`` bare months gets NaN for the
    bands; :data:`BARE_N` counts its bare months.
    """
    bands = [variable for variable in variables if variable in S2_BANDS]
    others = [variable for variable in variables if variable not in S2_BANDS]
    table = _percentiles(ts, others, "R6") if others else pd.DataFrame(index=ts["point_id"].unique())
    indices = bare_indices(ts)
    bare = ts[(indices["ndvi"] < ndvi_max) & (indices["nbr2"] < nbr2_max)]
    counts = bare.groupby("point_id").size().reindex(ts["point_id"].unique()).fillna(0).astype(int)
    if bands:
        banded = _percentiles(bare, bands, "R6").reindex(counts.index)
        banded.loc[(counts < min_months).to_numpy()] = np.nan  # too few bare months: no bare-soil summary
        table = table.join(banded, how="outer")
    table[BARE_N] = counts.reindex(table.index).fillna(0).astype(int)
    order = [_column("R6", v, s) for v in variables for s in (*[f"q{q:02d}" for q in QUANTILES], "iqr")] + [BARE_N]
    return table.reindex(columns=order).rename_axis("point_id")


def monthly_medians(ts: pd.DataFrame, variables: Sequence[str] = ("S2_B11", "S2_B12")) -> pd.DataFrame:
    """R5's input: each point's median of each variable per calendar month, ``R5__<variable>__m01`` ..."""
    frame = ts[["point_id", "date", *variables]].assign(month=ts["date"].dt.month)
    wide = frame.groupby(["point_id", "month"])[list(variables)].median().unstack("month")
    wide.columns = [_column("R5", variable, f"m{month:02d}") for variable, month in wide.columns]
    return wide.rename_axis("point_id")


@dataclass(frozen=True)
class PlanPCA:
    """R5: a PCA of some columns fitted on the plan's training points alone, applied to every point.

    Standardised with the training points' means and SDs (their medians filling gaps) first.
    """

    source: tuple
    n_components: int = 5
    prefix: str = "R5__pc"

    def transform(self, context, table) -> pd.DataFrame:
        """The components of every point of the plan's context, fitted on its training points."""
        from sklearn.decomposition import PCA

        values = table.frame.reindex(context.X.index)[list(self.source)].astype(float)
        train = values.loc[context.train_ids]
        fill = train.median()
        train = train.fillna(fill)
        mean, sd = train.mean(), train.std(ddof=0).replace(0, 1.0)
        pca = PCA(n_components=min(self.n_components, train.shape[1], len(train)))
        pca.fit(((train - mean) / sd).to_numpy())
        scores = pca.transform(((values.fillna(fill) - mean) / sd).to_numpy())
        columns = [f"{self.prefix}{i + 1}" for i in range(scores.shape[1])]
        return pd.DataFrame(scores, index=values.index, columns=columns)


def representation(rep: str, ts: pd.DataFrame, variables: Sequence[str], options: E5Settings, *, years=None):
    """The point-level columns of one representation (R2, R3, R4, R6, and R5's monthly medians)."""
    if rep == "R2":
        return quarter_medians(ts, variables, rep="R2", min_readings=options.quarter_min_readings)
    if rep == "R3":
        if years is None:
            raise ValueError("R3 needs each point's sampling year: set sample_year in the settings.")
        return quarter_medians(ts, variables, rep="R3", min_readings=options.quarter_min_readings, years=years)
    if rep == "R4":
        return harmonic_features(ts, variables, min_readings=options.harmonic_min_readings)
    if rep == "R6":
        return bare_soil_percentiles(
            ts,
            variables,
            ndvi_max=options.ndvi_max,
            nbr2_max=options.nbr2_max,
            min_months=options.min_bare_months,
        )
    if rep == "R5":
        return monthly_medians(ts)
    raise ValueError(f"No point-level columns for {rep}")


#: Bumped whenever the way a representation is computed changes, so every older cache is rebuilt.
FEATURES_VERSION = "2: composites with a zero Sentinel-2 band and infinite values treated as missing"


def file_digest(path) -> str:
    """The SHA-256 of a file's bytes."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(rep: str, variables: Sequence[str], options: E5Settings, *, source: str, years=None) -> str:
    """What one cached representation was computed from: the rules' version, the settings it reads,
    the variables, the time-series file's content and (R3) the sampling years."""
    import hashlib
    import json

    payload = {
        "version": FEATURES_VERSION,
        "rep": rep,
        "variables": list(variables),
        "source": source,
        "quarter_min_readings": options.quarter_min_readings,
        "harmonic_min_readings": options.harmonic_min_readings,
        "bare": [options.ndvi_max, options.nbr2_max, options.min_bare_months],
        "years": None
        if years is None or rep != "R3"
        else hashlib.sha256(
            pd.util.hash_pandas_object(years.sort_index(), index=True).to_numpy().tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_representations(
    table, ts: pd.DataFrame, options: E5Settings, cache_dir, *, source_digest: str = "", log=None
) -> tuple[dict, dict]:
    """Every representation's columns, computed once and cached in ``cache_dir/<rep>.parquet``.

    Each cache has a sidecar ``<rep>.json`` with the :func:`cache_fingerprint` it was built for and
    the SHA-256 of the parquet file. A cache whose fingerprint differs (older rules, other settings,
    another time-series file) or whose file changed is rebuilt. R1 needs no new columns (a subset of
    the static percentiles); R5's PCA is plan-dependent and only its monthly medians are cached.

    Returns
    -------
    features : dict
        Representation to its point-level columns.
    hashes : dict
        Representation to the SHA-256 of its cached file: what the prediction files record.
    """
    import json

    log = log or logger
    variables = [variable for variable in ts_variables(table.feature_columns) if variable in ts.columns]
    absent = [variable for variable in ts_variables(table.feature_columns) if variable not in ts.columns]
    if absent:
        log.warning(
            f"E5: {absent} have static percentiles but are not in the time-series file; no representation of them."
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    years = None
    if table.sample_year_column:
        years = table.frame[table.sample_year_column]
        years.index = years.index.astype(str)
    wanted = [rep for rep in options.representations if rep != "R1"]
    if options.r5_enabled and "R5" not in wanted:
        wanted.append("R5")
    found, hashes = {}, {}
    for rep in wanted:
        if rep == "R3" and years is None:
            log.warning("E5 R3 skipped: no sampling year (set sample_year in the settings).")
            continue
        if rep == "R6" and not {"S2_B4", "S2_B8", "S2_B11", "S2_B12"} <= set(ts.columns):
            log.warning(
                "E5 R6 skipped: the time series lacks S2 B4, B8, B11 or B12, which the bare-month filter needs."
            )
            continue
        path, sidecar = cache_dir / f"{rep}.parquet", cache_dir / f"{rep}.json"
        fingerprint = cache_fingerprint(rep, variables, options, source=source_digest, years=years)
        if path.exists() and sidecar.exists():
            record = json.loads(sidecar.read_text())
            if record.get("fingerprint") == fingerprint and record.get("sha256") == file_digest(path):
                found[rep], hashes[rep] = pd.read_parquet(path), record["sha256"]
                log.info(f"E5 {rep}: read {path.name} ({found[rep].shape[1]} columns)")
                continue
            log.warning(f"E5 {rep}: {path.name} was built from other inputs or rules; rebuilt.")
        elif path.exists():
            log.warning(f"E5 {rep}: {path.name} has no record of what it was built from; rebuilt.")
        columns = representation(rep, ts, variables, options, years=years)
        columns.to_parquet(path)
        hashes[rep] = file_digest(path)
        sidecar.write_text(json.dumps({"fingerprint": fingerprint, "sha256": hashes[rep], "version": FEATURES_VERSION}))
        found[rep] = columns
        log.info(f"E5 {rep}: {columns.shape[1]} columns, {float(columns.isna().to_numpy().mean()):.1%} NaN, cached")
    return found, hashes


# --- coordinates and external covariates -------------------------------------------------------------------


def projected_xy(table) -> pd.DataFrame:
    """Each point's position in metres, in the UTM zone :func:`~yg_eo_soilnet.utils.assign_grid_ids` cuts blocks in."""
    from pyproj import Transformer

    coordinates = table.coordinates
    lon_mean, lat_mean = float(coordinates["lon"].mean()), float(coordinates["lat"].mean())
    zone = int((lon_mean + 180) // 6) + 1
    epsg = (32600 if lat_mean >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(coordinates["lon"].to_numpy(), coordinates["lat"].to_numpy())
    return pd.DataFrame({XY_COLUMNS[0]: x, XY_COLUMNS[1]: y}, index=coordinates.index)


#: Columns that would carry the lab measurements into the inputs: refused in an external file.
LAB_HINTS = ("organic_matter", "caco3", "c_e_c", "som", "ph_water", "clay_pct", "sand_pct")


def extra_covariates(spec: ExtraCovariates, table, *, lab_columns: Sequence[str] = ()) -> pd.DataFrame:
    """One external covariate file, checked and aligned on the points, columns ``<group>__<name>``.

    Raises
    ------
    ValueError
        If the id column is missing, an id repeats, a column is not numeric, a column is a lab
        measurement (by name, or one of ``lab_columns``), or a point of the table has no row.
    """
    frame = pd.read_csv(spec.path)
    if spec.id_column not in frame.columns:
        raise ValueError(f"{spec.path}: no id column {spec.id_column!r}")
    ids = frame[spec.id_column].astype(str)
    if ids.duplicated().any():
        raise ValueError(
            f"{spec.path}: {int(ids.duplicated().sum())} repeated id(s), e.g. {ids[ids.duplicated()].iloc[0]!r}"
        )
    values = frame.drop(columns=[spec.id_column]).set_axis(ids.to_numpy())
    lab = [c for c in values.columns if c in set(lab_columns) or c.lower().startswith(LAB_HINTS)]
    if lab:
        raise ValueError(f"{spec.path}: lab column(s) {lab} cannot be inputs")
    text = [c for c in values.columns if not pd.api.types.is_numeric_dtype(values[c])]
    if text:
        raise ValueError(f"{spec.path}: column(s) {text} are not numeric")
    points = table.frame.index.astype(str)
    missing = points.difference(values.index)
    if len(missing):
        raise ValueError(
            f"{spec.path}: {len(missing)} of {len(points)} point(s) have no row (e.g. {missing[0]!r}); every point needs one"
        )
    return values.reindex(points).set_axis(table.frame.index).add_prefix(f"{spec.group}__")


def extend_table(table, new: pd.DataFrame):
    """The table with these columns added as static inputs (aligned on its points by id)."""
    if new.empty or not len(new.columns):
        return table
    added = new.copy()
    added.index = added.index.astype(str)
    aligned = added.reindex(table.frame.index.astype(str)).set_axis(table.frame.index)
    clash = [c for c in aligned.columns if c in table.frame.columns]
    if clash:
        raise ValueError(f"E5 columns {clash[:3]} already exist in the table")
    numeric = aligned.select_dtypes("number")
    infinite = np.isinf(numeric).sum()
    if int(infinite.sum()):
        raise ValueError(
            f"E5 columns hold infinite values, which no model can read: {infinite[infinite > 0].to_dict()}. "
            "Make them missing (NaN) where they are built."
        )
    frame = pd.concat([table.frame, aligned], axis=1)
    return replace(table, frame=frame, static_columns=tuple(table.static_columns) + tuple(aligned.columns))


# --- the variants --------------------------------------------------------------------------------------------


def global_inputs(resolved) -> list:
    """Global's inputs as E1 fits them: every group but ``auxiliary``, in the groups' order."""
    from yg_eo_soilnet.a1.config import AUXILIARY_GROUP

    auxiliary = set(resolved.groups.get(AUXILIARY_GROUP, ()))
    return [column for column in resolved.combined if column not in auxiliary]


def subblock_columns(satellite: Sequence[str], *, log=None) -> dict:
    """The satellite columns split into :data:`SUBBLOCKS`; a column in none of them is named in a warning.

    Such a column stays in every ``all_minus_<sub-block>`` variant (it is removed only with the whole group).
    """
    blocks = {name: [c for c in satellite if c.startswith(prefixes)] for name, prefixes in SUBBLOCKS.items()}
    left = [c for c in satellite if not any(c in cols for cols in blocks.values())]
    if left:
        (log or logger).warning(f"Satellite column(s) in no sub-block of {list(SUBBLOCKS)}: {left}")
    return blocks


def _without(columns, removed) -> tuple:
    removed = set(removed)
    return tuple(c for c in columns if c not in removed)


def part_a_variants(resolved) -> dict:
    """Part A: ``all``, ``all_minus_<group>`` and ``all_minus_<sub-block>``, ``only_satellite``,
    ``satellite_plus_terrain`` and ``all_plus_xy``."""
    base = global_inputs(resolved)
    groups = {name: list(resolved.groups[name]) for name in PART_A_GROUPS if name in resolved.groups}
    variants = {"all": Variant(tuple(base))}
    for name, columns in groups.items():
        variants[f"all_minus_{name}"] = Variant(_without(base, columns))
    for name, columns in subblock_columns(groups.get("satellite", [])).items():
        variants[f"all_minus_{name}"] = Variant(_without(base, columns))
    variants["only_satellite"] = Variant(tuple(groups.get("satellite", ())))
    variants["satellite_plus_terrain"] = Variant(tuple(groups.get("satellite", ())) + tuple(groups.get("terrain", ())))
    variants["all_plus_xy"] = Variant(tuple(base) + XY_COLUMNS)
    return variants


def representation_columns(rep: str, table_columns: Sequence[str], percentile_columns: Sequence[str]) -> list:
    """The columns that stand for the time series under one representation."""
    if rep == "R0":
        return list(percentile_columns)
    if rep == "R1":
        return [c for c in percentile_columns if c.endswith(("_q50", "_iqr"))]
    return [c for c in table_columns if c.startswith(f"{rep}__")] + (
        [BARE_N] if rep == "R6" and BARE_N in table_columns else []
    )


def _is_satellite_ts(column: str) -> bool:
    if column == BARE_N:
        return True
    variable = (
        column.split("__")[1]
        if "__" in column
        else (PERCENTILE.match(column).group("variable") if PERCENTILE.match(column) else column)
    )
    return variable.startswith(SATELLITE_TS_PREFIXES)


def part_b_variants(
    resolved, table_columns: Sequence[str], representations: Sequence[str], *, r5=None, hashes=None
) -> dict:
    """Part B: each representation on ``all``, on ``all_minus_legacy_soil``, and (R2, R6) on ``only_satellite``.

    The references - ``all``, ``all_minus_legacy_soil``, ``only_satellite`` (all R0) - are included.
    ``hashes`` (representation to the SHA-256 of its cached file, :func:`build_representations`)
    become each variant's ``inputs``: a prediction made from another version of the file is refitted.
    """
    hashes = hashes or {}
    base = global_inputs(resolved)
    percentiles = [c for c in base if c in set(sum(ts_variables(base).values(), []))]
    legacy = list(resolved.groups.get("legacy_soil", ()))
    satellite = list(resolved.groups.get("satellite", ()))
    statics = _without(base, percentiles)
    sat_statics = _without(satellite, percentiles)
    variants = {
        "all": Variant(tuple(base)),
        "all_minus_legacy_soil": Variant(_without(base, legacy)),
        "only_satellite": Variant(tuple(satellite)),
    }
    for rep in representations:
        pca = r5 if rep == "R5" else None
        columns = representation_columns(rep, table_columns, percentiles) if rep != "R5" else []
        if (rep != "R5" and not columns) or (rep == "R5" and pca is None):
            logger.warning(f"E5 {rep}: no columns in the table (skipped when its features were built); not fitted.")
            continue
        inputs = ((rep, hashes[rep]),) if rep in hashes else ()
        variants[f"ts_{rep}"] = Variant(statics + tuple(columns), pca=pca, inputs=inputs)
        variants[f"ts_{rep}_minus_legacy_soil"] = Variant(
            _without(statics, legacy) + tuple(columns), pca=pca, inputs=inputs
        )
        if rep in ("R2", "R6"):
            sat = [c for c in columns if _is_satellite_ts(c)]
            variants[f"only_satellite_{rep}"] = Variant(sat_statics + tuple(sat), inputs=inputs)
    return variants


def part_c_variants(resolved, extras: dict) -> dict:
    """Part C: ``all_plus_<group>`` and ``only_<group>`` for each external covariate group (and ``all``)."""
    base = global_inputs(resolved)
    variants = {"all": Variant(tuple(base))}
    for group, columns in extras.items():
        variants[f"all_plus_{group}"] = Variant(tuple(base) + tuple(columns))
        variants[f"only_{group}"] = Variant(tuple(columns))
    return variants


#: Each Part B variant's reference: on all, on all minus the soil-map layers, on satellite alone.
def part_b_reference(variant: str) -> Optional[str]:
    """The reference of a Part B variant (None for a reference itself).

    >>> part_b_reference("ts_R2"), part_b_reference("ts_R2_minus_legacy_soil"), part_b_reference("only_satellite_R6")
    ('all', 'all_minus_legacy_soil', 'only_satellite')
    """
    if variant.startswith("ts_") and variant.endswith("_minus_legacy_soil"):
        return "all_minus_legacy_soil"
    if variant.startswith("ts_"):
        return "all"
    if variant.startswith("only_satellite_"):
        return "only_satellite"
    return None


def part_of(name: str) -> Optional[str]:
    """Which part a variant belongs to (``A``, ``B``, ``C``); None for ``all``, the shared reference.

    >>> [part_of(n) for n in ("all", "all_minus_climate", "all_plus_xy", "ts_R4", "only_satellite_R2", "all_plus_embeddings", "only_embeddings")]
    [None, 'A', 'A', 'B', 'B', 'C', 'C']
    """
    if name == "all":
        return None
    if name.startswith("ts_") or (name.startswith("only_satellite_") and name != "only_satellite"):
        return "B"
    if name.startswith("all_minus_") or name in ("only_satellite", "satellite_plus_terrain", "all_plus_xy"):
        return "A"
    if name.startswith(("all_plus_", "only_")):
        return "C"
    return None


def comparisons(part: str, names: Sequence[str]) -> list:
    """The ``(variant, reference)`` pairs of a part, among the variants fitted.

    Part A and C against ``all``; Part B against its own reference (:func:`part_b_reference`).
    """
    present = set(names)
    if part == "B":
        pairs = [(name, part_b_reference(name)) for name in names if part_of(name) == "B"]
    else:
        pairs = [(name, "all") for name in names if part_of(name) == part]
    return [(a, b) for a, b in pairs if a in present and b in present]


# --- scores and paired comparisons ------------------------------------------------------------------------------


def design_frames(frame: pd.DataFrame):
    """``(design, rows)``: random; the 10 km folds pooled; the held-out basins pooled; each basin."""
    from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD

    test = frame[frame["split"] == "test"]
    yield DESIGN_RANDOM, test[test["kind"] == RANDOM]
    yield DESIGN_10KM, test[(test["kind"] == SPATIAL_KFOLD) & np.isclose(test["block_size_m"].astype(float), 10000.0)]
    logo = test[test["kind"] == LOGO]
    yield DESIGN_LOGO, logo
    for basin, part in logo.groupby("held_out", sort=True):
        yield f"logo {basin}", part


def _paired_estimates(rows: pd.DataFrame, a: str, b: str) -> dict:
    """score(a) - score(b) on the points both predicted (one seed), for RMSE, bias and R²."""
    keys = ["plan_id", "point_id"]
    left = rows[rows["covariate_group"] == a][[*keys, "observed", "predicted"]]
    right = rows[rows["covariate_group"] == b][[*keys, "predicted"]]
    both = left.merge(right, on=keys, suffixes=("_a", "_b"))
    if len(both) < 2:
        return dict.fromkeys(evaluation.STACKED_SCORES, np.nan)
    sa = evaluation._point_scores(both["observed"], both["predicted_a"])
    sb = evaluation._point_scores(both["observed"], both["predicted_b"])
    return {score: sa[score] - sb[score] for score in evaluation.STACKED_SCORES}


def variant_differences(
    frame: pd.DataFrame,
    pairs: Sequence[tuple],
    *,
    part: str,
    first_seed: int,
    n_features: Optional[dict] = None,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Each ``(variant, reference)`` pair on every design: ΔRMSE, Δbias, ΔR² (variant - reference).

    The first seed's predictions give the estimate, a 95 % paired 10 km block-bootstrap interval
    and a p-value; Holm corrects within each family (``part``, design, score). Every seed gives
    its own estimate: ``mean_over_seeds`` and ``n_seeds_same_sign`` (as the first seed's).
    """
    columns = [
        "part",
        "design",
        "variant",
        "reference",
        "metric",
        "estimate",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n",
        "n_blocks",
        "mean_over_seeds",
        "n_seeds_same_sign",
        "n_seeds",
        "n_features",
        "n_features_reference",
    ]
    if not pairs or frame.empty:
        return pd.DataFrame(columns=columns)
    seeds = evaluation.seed_order(frame, first_seed)
    rows = []
    for design, part_rows in design_frames(frame):
        if part_rows.empty:
            continue
        first = part_rows[part_rows["seed"] == seeds[0]]
        result = evaluation.block_bootstrap(
            first,
            ["covariate_group"],
            block_km=block_km,
            n=n,
            seed=seed,
            compare=list(pairs),
            compare_on="covariate_group",
        )
        found = result.differences[result.differences["metric"].isin(evaluation.STACKED_SCORES)]
        by_seed = {
            s: {(a, b): _paired_estimates(part_rows[part_rows["seed"] == s], a, b) for a, b in pairs} for s in seeds
        }
        for row in found.itertuples():
            values = np.array([by_seed[s][(row.a, row.b)][row.metric] for s in seeds], dtype=float)
            sign = np.sign(row.estimate)
            rows.append(
                {
                    "part": part,
                    "design": design,
                    "variant": row.a,
                    "reference": row.b,
                    "metric": row.metric,
                    "estimate": row.estimate,
                    "low": row.low,
                    "high": row.high,
                    "p_value": row.p_value,
                    "n": row.n,
                    "n_blocks": row.n_blocks,
                    "mean_over_seeds": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                    "n_seeds_same_sign": int(np.sum(np.sign(values) == sign)),
                    "n_seeds": len(seeds),
                    "n_features": (n_features or {}).get(row.a, np.nan),
                    "n_features_reference": (n_features or {}).get(row.b, np.nan),
                }
            )
    table = pd.DataFrame(rows, columns=[c for c in columns if c != "p_holm"])
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["p_holm"] = table.groupby(["part", "design", "metric"], sort=False)["p_value"].transform(evaluation.holm)
    return table[columns]


def variant_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE, R², bias, RPD, RPIQ per seed, design and variant (the E1 ``scores.csv`` schema, plus ``variant``)."""
    parts = []
    for design, rows in design_frames(frame):
        if rows.empty:
            continue
        table = evaluation.grouped_metrics(rows, ["seed", "covariate_group"])
        parts.append(table.rename(columns={"covariate_group": "variant"}).assign(design=design))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    return table[["design", "variant", *[c for c in table.columns if c not in ("design", "variant")]]]


# --- reports on the features (no fitting) ---------------------------------------------------------------------


def feature_report(features: dict, table, ts: pd.DataFrame, regions: Optional[pd.Series] = None) -> dict:
    """What Part B rests on: features and NaN share per representation; sampling year by basin;
    the bare-month filter's indices against the stored ones; bare months per basin."""
    reports = {}
    rows = [
        {"representation": rep, "n_features": frame.shape[1], "nan_share": float(frame.isna().to_numpy().mean())}
        for rep, frame in features.items()
    ]
    reports["representations"] = pd.DataFrame(rows)
    if regions is not None and table.sample_year_column:
        years = table.frame[table.sample_year_column]
        reports["sample_year_by_basin"] = pd.crosstab(regions.reindex(years.index), years, dropna=False)
    has_bands = {"S2_B4", "S2_B8", "S2_B11", "S2_B12"} <= set(ts.columns)
    indices = bare_indices(ts) if has_bands else None
    agreement = []
    for mine, stored in (("ndvi", "AG_NDVI"), ("nbr2", "SOIL_NBR2")):
        if has_bands and stored in ts:
            pair = pd.concat([indices[mine], ts[stored]], axis=1).dropna()
            agreement.append(
                {
                    "index": mine,
                    "stored": stored,
                    "n": len(pair),
                    "pearson_r": float(pair.iloc[:, 0].corr(pair.iloc[:, 1])),
                    "max_abs_difference": float((pair.iloc[:, 0] - pair.iloc[:, 1]).abs().max()),
                }
            )
    reports["bare_index_agreement"] = pd.DataFrame(agreement)
    if "R6" in features and regions is not None:
        counts = features["R6"][BARE_N]
        counts.index = counts.index.astype(str)
        basin = regions.copy()
        basin.index = basin.index.astype(str)
        joined = pd.DataFrame({"basin": basin.reindex(counts.index), "bare_n": counts})
        reports["bare_months_by_basin"] = joined.groupby("basin")["bare_n"].describe(percentiles=[0.05, 0.5, 0.95])
    return reports


def stable_key(text: str) -> int:
    """A stable integer for a name, for seeding."""
    return zlib.crc32(str(text).encode())
