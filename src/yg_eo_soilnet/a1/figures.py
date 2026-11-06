"""The Paper A1 figures, drawn from what a run saved.

Each figure has one function, ``figure_*``, which reads what it needs from a :class:`FigureInputs` -
the run's prediction files (step 7), the stage-1 audits, the dissimilarity tables (step 9) and,
for the figures about the data itself, the A1 points table - and returns a matplotlib figure. The
numbers behind each figure come from a separate function that returns a plain table
(:func:`class_counts`, :func:`spearman_by_class`, :func:`class_confusion`, :func:`class_scores`,
:func:`shap_by_class`), so they can be checked, and quoted in the text, without the drawing.

1. :func:`figure_points_map` - where the points are, by carbonate class, and how many of each class
   every region holds.
2. :func:`figure_correlations` - how SOM correlates with each covariate within each carbonate class,
   and the bare-soil spectrum of each class.
3. :func:`figure_stage1` - how well stage 1 predicts CaCO3 and CEC, and the carbonate class.
4. :func:`figure_class_scores` - RPIQ and bias per carbonate class for the four models, under
   random and spatial cross-validation.
5. :func:`figure_transfer` - leave-one-region-out against local scores, and the transfer drop per
   covariate group.
6. :func:`figure_mixed` - how the error falls as a share of the target region joins training.
7. :func:`figure_dissimilarity` - the transfer drop against how different the region is.
8. :func:`figure_shap` - what ``global_aux`` relies on, within each carbonate class.

:func:`render_all` draws all eight and saves each as PNG and PDF (:func:`save_figure`), by default in
the run folder's ``figures/``. A figure whose inputs the run does not hold - no mixed plans, say -
comes back as a short message saying what to run, rather than as an error.

Colours come from :mod:`yg_eo_soilnet.plot_style`: one per carbonate class (light for < 2 % CaCO3,
dark for >= 25 %) and one per model, the same in every figure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP, COMBINED_GROUP, A1Config, stage1_prediction_column
from yg_eo_soilnet.a1.plans import LOGO, MIXED, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, MODELS, PER_CLASS_MEASURED, PER_CLASS_PREDICTED
from yg_eo_soilnet.a1.strata import carbonate_class
from yg_eo_soilnet.datamodules.splitting import TEST
from yg_eo_soilnet.metrics import regression_metrics

logger = logging.getLogger(__name__)

#: The figures :func:`render_all` draws, in order, by the file name each is saved under.
FIGURE_NAMES = (
    "fig1_points_map",
    "fig2_correlations",
    "fig3_stage1",
    "fig4_class_scores",
    "fig5_transfer",
    "fig6_mixed",
    "fig7_dissimilarity",
    "fig8_shap",
)
#: How each model is named in a legend.
MODEL_LABELS = {
    GLOBAL: "Global",
    GLOBAL_AUX: "Global + predicted CaCO₃/CEC",
    PER_CLASS_MEASURED: "Per class, measured CaCO₃ (upper bound)",
    PER_CLASS_PREDICTED: "Per class, predicted CaCO₃",
}
#: Central wavelength (nm) of each Sentinel-2 band, for the bare-soil spectrum.
S2_WAVELENGTHS = {
    "b1": 443,
    "b2": 490,
    "b3": 560,
    "b4": 665,
    "b5": 705,
    "b6": 740,
    "b7": 783,
    "b8": 842,
    "b8a": 865,
    "b9": 945,
    "b11": 1610,
    "b12": 2190,
}
BAREST_BAND = re.compile(r"^s2_(b\d+a?)_barest$", re.IGNORECASE)


# --- what the figures read -----------------------------------------------------------------------


@dataclass
class FigureInputs:
    """A run's outputs, read when a figure first asks for them and kept for the next one.

    Attributes
    ----------
    a1 : A1Config
    run_dir : pathlib.Path
        ``<output_dir>/runs/<run name>/``.
    plans_dir : pathlib.Path
        Where the plans the run used are (for :func:`shap_by_class`, which refits one).
    regions_path : str, optional
        A polygon file of the regions, for their outlines on the map; the settings' own when the
        regions come from polygons. Without one, each region is outlined by the hull of its points.
    regions_name_field : str, optional
        The polygon file's column holding the region names.
    """

    a1: A1Config
    run_dir: Path
    plans_dir: Path
    regions_path: Optional[str] = None
    regions_name_field: Optional[str] = None
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_run(
        cls,
        a1: A1Config,
        run_name: str = "default",
        *,
        plans_dir: Optional[str] = None,
        regions_path: Optional[str] = None,
        regions_name_field: Optional[str] = None,
    ) -> "FigureInputs":
        """The outputs of the run ``run_name`` made from these settings."""
        output = Path(a1.settings.output_dir)
        return cls(
            a1=a1,
            run_dir=output / "runs" / run_name,
            plans_dir=Path(plans_dir) if plans_dir else output / "plans",
            regions_path=regions_path,
            regions_name_field=regions_name_field,
        )

    def _cached(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def settings(self):
        """The A1 settings."""
        return self.a1.settings

    @property
    def config(self):
        """The main configuration."""
        return self._cached("config", self.settings.load_config)

    @property
    def predictions(self) -> pd.DataFrame:
        """Every prediction file of the run (:func:`~yg_eo_soilnet.a1.evaluation.load_predictions`)."""
        return self._cached("predictions", lambda: evaluation.load_predictions(self.run_dir))

    @property
    def transfer(self) -> pd.DataFrame:
        """:func:`~yg_eo_soilnet.a1.evaluation.transfer_table` of the predictions."""
        return self._cached("transfer", lambda: evaluation.transfer_table(self.predictions))

    @property
    def dissimilarity_summary(self) -> pd.DataFrame:
        """The run's ``dissimilarity_summary.csv``; empty without ``logo`` plans."""
        from yg_eo_soilnet.a1.runner import DISSIMILARITY_SUMMARY_FILE

        path = self.run_dir / DISSIMILARITY_SUMMARY_FILE
        return self._cached("dissimilarity", lambda: pd.read_csv(path) if path.exists() else pd.DataFrame())

    def stage1(self, plan_id: str) -> pd.DataFrame:
        """The stage-1 audit of one plan: ``point_id, split, inner_fold``, the predictions."""
        path = self.run_dir / "stage1" / f"{plan_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No stage-1 audit for plan {plan_id!r} in {self.run_dir}.")
        return pd.read_parquet(path)

    @property
    def table(self):
        """The A1 points table (:func:`~yg_eo_soilnet.a1.features.build_table`)."""
        from yg_eo_soilnet.a1.features import build_table

        return self._cached("table", lambda: build_table(self.settings, self.config))

    @property
    def regions(self):
        """The region polygons in degrees, with a ``region`` column; None when there are none."""

        def read():
            regions = self.settings.regions
            path, name = self.regions_path, self.regions_name_field
            if path is None and regions.polygons_path:
                from yg_eo_soilnet.a1.regions import resolve_polygons_path

                path = resolve_polygons_path(regions.polygons_path, getattr(self.config, "DATA_FOLDER", None))
                name = name or regions.name_field
            if path is None:
                return None
            from yg_eo_soilnet.map_style import _ensure_proj_data

            _ensure_proj_data()
            import geopandas as gpd

            frame = gpd.read_file(path)
            frame = frame.to_crs("EPSG:4326") if frame.crs is not None else frame.set_crs("EPSG:4326")
            name = name or ("name" if "name" in frame.columns else frame.columns[0])
            return frame.rename(columns={name: "region"})[["region", "geometry"]]

        return self._cached("regions", read)

    # The lab values the figures name.
    @property
    def carbonate_column(self) -> str:
        """The measured CaCO3 column."""
        return self.settings.carbonate.column

    @property
    def cec_column(self) -> Optional[str]:
        """The measured CEC column: the stage-1 target that is not CaCO3."""
        others = [target for target in self.settings.auxiliary.targets if target != self.carbonate_column]
        return others[0] if others else None

    @property
    def class_labels(self) -> list:
        """The carbonate classes, lightest first."""
        return list(self.settings.carbonate.labels)

    def measured_classes(self) -> pd.Series:
        """Each point's carbonate class from its measured CaCO3."""
        carbonate = self.settings.carbonate
        return carbonate_class(self.table.frame[carbonate.column], carbonate.edges, carbonate.labels)


def save_figure(figure, name: str, folder) -> list[Path]:
    """Save ``figure`` as ``<name>.png`` and ``<name>.pdf`` in ``folder``; return the two paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.png", folder / f"{name}.pdf"]
    for path in paths:
        figure.savefig(path, dpi=ps.SAVE_DPI)
    return paths


def _missing(message: str):
    """A figure saying why there is nothing to draw, marked so :func:`is_message_figure` knows it."""
    figure = ps.message_figure(message)
    figure._a1_message = message
    return figure


def is_message_figure(figure) -> bool:
    """Whether ``figure`` is a message saying its inputs are missing, rather than a drawing."""
    return hasattr(figure, "_a1_message")


def _class_color(label) -> str:
    return ps.A1_CARBONATE_COLORS.get(str(label), ps.MUTED)


def _class_legend(labels, counts=None) -> list:
    handles = []
    for label in labels:
        text = f"{label} %" if counts is None else f"{label} % (n = {int(counts.get(label, 0)):,})"
        handles.append(Patch(facecolor=_class_color(label), edgecolor="none", label=text))
    return handles


# --- 1. the points ------------------------------------------------------------------------------


def class_counts(regions: pd.Series, classes: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    """Points per region and carbonate class: one row per region, most points first.

    Examples
    --------
    >>> counts = class_counts(pd.Series(["A", "A", "B"]), pd.Series(["<2", ">=25", "<2"]), ["<2", ">=25"])
    >>> counts.to_dict("index")
    {'A': {'<2': 1, '>=25': 1}, 'B': {'<2': 1, '>=25': 0}}
    """
    frame = pd.DataFrame({"region": regions.astype(object).to_numpy(), "class": classes.astype(object).to_numpy()})
    table = pd.crosstab(frame["region"], frame["class"]).reindex(columns=list(labels), fill_value=0)
    return (
        table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]
        .rename_axis(None)
        .rename_axis(None, axis=1)
    )


def figure_points_map(inputs: FigureInputs, *, relief: bool = True, cache_dir: Optional[str] = None, max_regions=12):
    """Figure 1: the points by carbonate class, the region outlines, and each region's class mix."""
    table = inputs.table
    points = pd.DataFrame(
        {
            "lat": table.frame[table.lat_column].to_numpy(dtype=float),
            "lon": table.frame[table.lon_column].to_numpy(dtype=float),
            "region": table.frame[table.region_column].to_numpy() if table.region_column else None,
            "class": inputs.measured_classes().astype(object).to_numpy(),
        }
    ).dropna(subset=["lat", "lon"])
    return draw_points_map(
        points, inputs.regions, labels=inputs.class_labels, relief=relief, cache_dir=cache_dir, max_regions=max_regions
    )


@ps.styled
def draw_points_map(
    points: pd.DataFrame,
    regions=None,
    *,
    labels: Sequence[str],
    relief: bool = True,
    cache_dir: Optional[str] = None,
    max_regions: int = 12,
):
    """Draw figure 1 from ``lat, lon, region, class`` columns and, optionally, region polygons."""
    from yg_eo_soilnet import map_style
    from yg_eo_soilnet.plot_utils import residual_bounds, residual_figsize

    if points.empty:
        return _missing("No points to map.")
    bounds = residual_bounds(points["lat"].to_numpy(), points["lon"].to_numpy())
    figure, axis = plt.subplots(figsize=residual_figsize(bounds), layout="constrained")
    cache_dir = cache_dir or map_style.DEFAULT_CACHE_DIR
    drawn = map_style.draw_basemap(axis, bounds, cache_dir=cache_dir, relief=relief)

    outlines = _region_outlines(points, regions)
    if outlines is not None and not outlines.empty:
        outlines.to_crs("EPSG:3857").boundary.plot(ax=axis, color=ps.INK_2, lw=0.6, zorder=3)

    present = points["class"].dropna()
    counts = present.value_counts()
    for label in labels:
        chosen = points[points["class"] == label]
        if chosen.empty:
            continue
        x, y = map_style.to_mercator(chosen["lon"].to_numpy(), chosen["lat"].to_numpy())
        axis.scatter(x, y, s=9, color=_class_color(label), edgecolor="white", lw=0.25, zorder=4, rasterized=True)
    unknown = points[points["class"].isna()]
    handles = _class_legend(labels, counts)
    if len(unknown):
        x, y = map_style.to_mercator(unknown["lon"].to_numpy(), unknown["lat"].to_numpy())
        axis.scatter(x, y, s=7, facecolors="none", edgecolors=ps.MUTED, lw=0.4, zorder=4, rasterized=True)
        handles.append(
            Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=ps.MUTED, label="No CaCO₃")
        )
    axis.legend(
        handles=handles,
        title="CaCO₃ class",
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor=ps.BASELINE,
        fontsize=7,
        title_fontsize=7.5,
    ).set_zorder(6)

    scale_bar_at = map_style.scale_bar_placement(bounds)
    map_style.finish_map(
        axis,
        bounds,
        scale_bar_at=scale_bar_at,
        north_arrow_at=map_style.north_arrow_placement(bounds, scale_bar_at),
        credit=map_style.CREDIT if (drawn["relief"] or drawn["borders"]) else None,
    )

    if points["region"].notna().any():
        _class_mix_inset(axis, class_counts(points["region"], points["class"], labels), labels, max_regions)
    return figure


def _region_outlines(points: pd.DataFrame, regions):
    """The region polygons, or the convex hull of each region's points when there are none."""
    import geopandas as gpd

    if regions is not None:
        return regions
    if points["region"].isna().all():
        return None
    frame = gpd.GeoDataFrame(
        points[["region"]], geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326"
    )
    hulls = frame.dropna(subset=["region"]).dissolve(by="region").convex_hull
    return gpd.GeoDataFrame({"region": hulls.index}, geometry=hulls.to_numpy(), crs="EPSG:4326")


def _class_mix_inset(axis, counts: pd.DataFrame, labels, max_regions: int) -> None:
    """Stacked bars of points per region and class, in the map's lower-left corner."""
    if len(counts) > max_regions:
        kept = counts.iloc[: max_regions - 1]
        counts = pd.concat([kept, counts.iloc[max_regions - 1 :].sum().to_frame("other").T])
    # Room on its left for the region names, so they stay inside the map, clear of the latitude ticks.
    inset = axis.inset_axes([0.16, 0.08, 0.26, 0.05 + 0.035 * len(counts)])
    inset.set_facecolor((1, 1, 1, 0.85))
    left = np.zeros(len(counts))
    positions = np.arange(len(counts))[::-1]
    for label in labels:
        values = counts[label].to_numpy(dtype=float)
        inset.barh(positions, values, left=left, color=_class_color(label), height=0.75, lw=0)
        left += values
    inset.set_yticks(positions, [str(name) for name in counts.index], fontsize=5.5)
    inset.tick_params(axis="x", labelsize=5.5, length=2, color=ps.BASELINE, labelcolor=ps.INK_2)
    inset.tick_params(axis="y", length=0, labelcolor=ps.INK_2)
    for name, spine in inset.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(ps.BASELINE)
    inset.grid(False)
    inset.set_xlabel("Points", fontsize=5.5, labelpad=1)
    inset.set_zorder(7)


# --- 2. correlations and spectra -----------------------------------------------------------------


def _pearson_columns(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``a`` with each column of ``B``; NaN where either is constant."""
    a = a - a.mean()
    B = B - B.mean(axis=0)
    denominator = np.sqrt((a**2).sum() * (B**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = (a @ B) / denominator
    correlation[denominator == 0] = np.nan
    return correlation


def _spearman_columns(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Spearman correlation of ``y`` with each column of ``X``, over the points both have.

    Columns without gaps are ranked together, which is what makes a thousand bootstrap draws
    affordable; a column with gaps is ranked, with ``y``, on its own complete pairs.
    """
    result = np.full(X.shape[1], np.nan)
    if y.size < 3:
        return result
    complete = ~np.isnan(X).any(axis=0)
    if complete.any():
        result[complete] = _pearson_columns(rankdata(y), rankdata(X[:, complete], axis=0))
    for column in np.flatnonzero(~complete):
        present = ~np.isnan(X[:, column])
        if present.sum() >= 3:
            result[column] = _pearson_columns(rankdata(y[present]), rankdata(X[present, column])[:, None])[0]
    return result


def spearman_by_class(
    frame: pd.DataFrame,
    target: str,
    covariates: Sequence[str],
    classes: pd.Series,
    *,
    labels: Sequence[str],
    blocks: Optional[pd.Series] = None,
    n: int = 1000,
    seed: int = 42,
    level: float = 0.95,
) -> pd.DataFrame:
    """Spearman correlation of ``target`` with each covariate, within each carbonate class.

    Whether a correlation is told apart from 0 is judged by resampling whole spatial blocks, as in
    :func:`~yg_eo_soilnet.a1.evaluation.block_bootstrap`: nearby samples are not independent
    evidence. The p-values are Holm-corrected over every covariate and class together.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per point, with ``target`` and the covariates.
    target : str
    covariates : sequence of str
        Numeric columns.
    classes : pandas.Series
        Each point's carbonate class, indexed like ``frame``.
    labels : sequence of str
        The classes, in order.
    blocks : pandas.Series, optional
        Each point's spatial block (:func:`~yg_eo_soilnet.a1.evaluation.spatial_blocks`); every
        point is its own block unless given.
    n : int, default 1000
        Bootstrap draws; 0 skips the test.
    seed : int, default 42
    level : float, default 0.95

    Returns
    -------
    pandas.DataFrame
        ``class, covariate, rho, low, high, p_value, p_holm, significant, n``.
    """
    covariates = list(covariates)
    rows = []
    for label in labels:
        chosen = (classes.reindex(frame.index) == label).to_numpy() & frame[target].notna().to_numpy()
        part = frame[chosen]
        y = part[target].to_numpy(dtype=float)
        X = part[covariates].to_numpy(dtype=float)
        rho = _spearman_columns(y, X)
        low = high = p_value = np.full(len(covariates), np.nan)
        if n and len(part) >= 3:
            block_ids = (
                blocks.reindex(part.index) if blocks is not None else pd.Series(np.arange(len(part)))
            ).to_numpy()
            generator = evaluation._generator(seed, (label,))
            draws = np.array(
                [_spearman_columns(y[rows_], X[rows_]) for rows_ in evaluation._draws(block_ids, n, generator)]
            )
            tail = 100.0 * (1.0 - level) / 2.0
            with np.errstate(all="ignore"):
                finite = np.isfinite(draws)
                low = np.array(
                    [
                        np.percentile(draws[finite[:, j], j], tail) if finite[:, j].any() else np.nan
                        for j in range(len(covariates))
                    ]
                )
                high = np.array(
                    [
                        np.percentile(draws[finite[:, j], j], 100 - tail) if finite[:, j].any() else np.nan
                        for j in range(len(covariates))
                    ]
                )
            p_value = np.array([evaluation._p_value(draws[:, j]) for j in range(len(covariates))])
        for position, covariate in enumerate(covariates):
            rows.append(
                {
                    "class": label,
                    "covariate": covariate,
                    "rho": rho[position],
                    "low": low[position],
                    "high": high[position],
                    "p_value": p_value[position],
                    "n": len(part),
                }
            )
    table = pd.DataFrame(rows, columns=["class", "covariate", "rho", "low", "high", "p_value", "n"])
    table["p_holm"] = evaluation.holm(table["p_value"])
    table["significant"] = table["p_holm"] < (1.0 - level)
    return table


def bare_soil_spectra(frame: pd.DataFrame, classes: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    """Median and quartiles of each bare-soil Sentinel-2 band (``s2_<band>_barest``), per class.

    Returns
    -------
    pandas.DataFrame
        ``class, band, wavelength_nm, median, q25, q75, n``, bands in wavelength order; empty when
        the data holds no such band.
    """
    bands = {}
    for column in frame.columns:
        match = BAREST_BAND.match(column)
        if match and match.group(1).lower() in S2_WAVELENGTHS:
            bands[column] = S2_WAVELENGTHS[match.group(1).lower()]
    rows = []
    for label in labels:
        part = frame[(classes.reindex(frame.index) == label).to_numpy()]
        for column, wavelength in sorted(bands.items(), key=lambda item: item[1]):
            values = part[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "class": label,
                    "band": column,
                    "wavelength_nm": wavelength,
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows, columns=["class", "band", "wavelength_nm", "median", "q25", "q75", "n"])


def _numeric_covariates(table) -> list:
    """The table's numeric covariates: what a correlation can be computed with."""
    frame = table.frame
    return [
        column
        for column in table.feature_columns
        if column not in set(table.categorical_columns) and pd.api.types.is_numeric_dtype(frame[column])
    ]


def figure_correlations(inputs: FigureInputs, *, n: int = 1000, n_covariates: int = 20, block_km: float = 10.0):
    """Figure 2: (a) Spearman ρ of SOM with each covariate per carbonate class; (b) bare-soil spectra."""
    table = inputs.table
    frame = table.frame
    classes = inputs.measured_classes()
    coordinates = table.coordinates.rename_axis("point_id").reset_index()
    blocks = pd.Series(evaluation.spatial_blocks(coordinates, block_km).to_numpy(), index=frame.index)
    correlations = spearman_by_class(
        frame,
        inputs.settings.target,
        _numeric_covariates(table),
        classes,
        labels=inputs.class_labels,
        blocks=blocks,
        n=n,
        seed=inputs.settings.seed,
    )
    spectra = bare_soil_spectra(frame, classes, inputs.class_labels)
    return draw_correlations(correlations, spectra, labels=inputs.class_labels, n_covariates=n_covariates)


@ps.styled
def draw_correlations(correlations: pd.DataFrame, spectra: pd.DataFrame, *, labels, n_covariates: int = 20):
    """Draw figure 2 from :func:`spearman_by_class` and :func:`bare_soil_spectra`."""
    if correlations.empty or correlations["rho"].isna().all():
        return _missing("No correlations: no carbonate class has three points with SOM.")
    strength = correlations.groupby("covariate")["rho"].apply(lambda values: values.abs().max())
    top = strength.sort_values(ascending=False).index[:n_covariates]
    grid = correlations.pivot(index="covariate", columns="class", values="rho").reindex(index=top, columns=labels)
    marks = correlations.pivot(index="covariate", columns="class", values="significant").reindex(
        index=top, columns=labels
    )

    height = max(3.0, 0.2 * len(top) + 1.2)
    figure, (heat, spectrum) = plt.subplots(
        1, 2, figsize=(ps.FIG_WIDTH_FULL, height), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]}
    )
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) if np.isfinite(grid.to_numpy()).any() else 1.0
    image = heat.imshow(grid.to_numpy(dtype=float), cmap=ps.diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    rows, columns = np.nonzero(marks.fillna(False).to_numpy(dtype=bool))
    heat.scatter(columns, rows, s=6, color=ps.INK, zorder=3, label="Significant (block bootstrap, Holm)")
    heat.set_xticks(range(len(labels)), [f"{label} %" for label in labels], fontsize=7)
    heat.set_yticks(range(len(top)), list(top), fontsize=6.5)
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)
    heat.grid(False)
    heat.set_xlabel("CaCO₃ class")
    colorbar = figure.colorbar(image, ax=heat, orientation="horizontal", shrink=0.8, pad=0.02, aspect=25)
    colorbar.set_label("Spearman ρ with SOM", fontsize=7.5)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7)
    heat.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06 - 0.5 / height), fontsize=6.5, handletextpad=0.2)
    ps.panel_letter(heat, "a")

    if spectra.empty:
        spectrum.text(0.5, 0.5, "No bare-soil bands (s2_<band>_barest)", ha="center", va="center", color=ps.MUTED)
        spectrum.set_axis_off()
    else:
        for label in labels:
            part = spectra[spectra["class"] == label]
            if part.empty:
                continue
            color = _class_color(label)
            spectrum.fill_between(part["wavelength_nm"], part["q25"], part["q75"], color=color, alpha=0.18, lw=0)
            spectrum.plot(part["wavelength_nm"], part["median"], color=color, lw=1.6, marker="o", ms=3)
        spectrum.set_xlabel("Wavelength (nm)")
        spectrum.set_ylabel("Bare-soil reflectance (median, IQR)")
        spectrum.legend(handles=_class_legend(labels), title="CaCO₃ class", fontsize=7, title_fontsize=7.5)
    ps.panel_letter(spectrum, "b")
    return figure


# --- 3. stage 1 --------------------------------------------------------------------------------


def class_confusion(measured: pd.Series, predicted: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    """Counts of each measured class (rows) predicted as each class (columns).

    Examples
    --------
    >>> counts = class_confusion(pd.Series(["<2", "<2", ">=25"]), pd.Series(["<2", ">=25", ">=25"]), ["<2", ">=25"])
    >>> counts.to_dict("index")   # one <2 point predicted right, one predicted >=25
    {'<2': {'<2': 1, '>=25': 1}, '>=25': {'<2': 0, '>=25': 1}}
    """
    frame = pd.DataFrame(
        {"measured": measured.astype(object).to_numpy(), "predicted": predicted.astype(object).to_numpy()}
    )
    counts = pd.crosstab(frame["measured"], frame["predicted"]).reindex(
        index=list(labels), columns=list(labels), fill_value=0
    )
    return counts.rename_axis(None).rename_axis(None, axis=1)


def stage1_points(inputs: FigureInputs, plan_id: str = RANDOM) -> pd.DataFrame:
    """Stage 1's out-of-fold predictions on one plan's training points, beside the measurements."""
    audit = inputs.stage1(plan_id).set_index("point_id")
    audit = audit[audit["inner_fold"] >= 0]
    frame = inputs.table.frame.reindex(audit.index)
    carbonate = inputs.settings.carbonate
    out = pd.DataFrame(index=audit.index)
    for target in inputs.settings.auxiliary.targets:
        out[f"observed_{target}"] = frame[target].to_numpy(dtype=float)
        out[f"predicted_{target}"] = audit[stage1_prediction_column(target)].to_numpy(dtype=float)
    out["class_measured"] = carbonate_class(frame[carbonate.column], carbonate.edges, carbonate.labels).to_numpy()
    out["class_predicted"] = carbonate_class(
        audit[stage1_prediction_column(carbonate.column)], carbonate.edges, carbonate.labels
    ).to_numpy()
    return out


def figure_stage1(inputs: FigureInputs, *, plan_id: str = RANDOM):
    """Figure 3: measured against out-of-fold predicted CaCO3 and CEC, and the class confusion matrix."""
    try:
        points = stage1_points(inputs, plan_id)
    except FileNotFoundError:
        return _missing(f"No stage-1 audit for plan {plan_id!r}: run run_a1.py with that plan first.")
    names = {inputs.carbonate_column: "CaCO₃ (%)"}
    if inputs.cec_column:
        names[inputs.cec_column] = "CEC (meq/100 g)"
    return draw_stage1(points, names, labels=inputs.class_labels)


@ps.styled
def draw_stage1(points: pd.DataFrame, names: dict, *, labels):
    """Draw figure 3 from :func:`stage1_points`; ``names`` maps each target to its axis label."""
    figure, axes = plt.subplots(1, len(names) + 1, figsize=(ps.FIG_WIDTH_FULL, 2.6), layout="constrained")
    for letter, axis, (target, name) in zip("abc", axes, names.items()):
        observed, predicted = points[f"observed_{target}"], points[f"predicted_{target}"]
        keep = observed.notna() & predicted.notna()
        colors = [_class_color(label) for label in points.loc[keep, "class_measured"].astype(object)]
        axis.scatter(observed[keep], predicted[keep], s=6, c=colors, lw=0, alpha=0.8, rasterized=True)
        low = float(min(observed[keep].min(), predicted[keep].min())) if keep.any() else 0.0
        high = float(max(observed[keep].max(), predicted[keep].max())) if keep.any() else 1.0
        axis.plot([low, high], [low, high], color=ps.INK_2, lw=0.8, ls="--")
        ps.square_panel(axis)
        axis.set_xlabel(f"Measured {name}")
        axis.set_ylabel(f"Predicted {name}\n(out of fold)")
        scores = regression_metrics(observed[keep], predicted[keep])
        if scores:
            ps.metric_box(
                axis,
                ps.minus(
                    f"n = {int(scores['n_test']):,}\nR² = {scores.get('r2_test', np.nan):.2f}\nRMSE = {scores['rmse_test']:.2f}"
                ),
            )
        ps.panel_letter(axis, letter)

    matrix_axis = axes[-1]
    counts = class_confusion(points["class_measured"], points["class_predicted"], labels)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    # From white, so an empty cell reads as empty rather than as a few points.
    matrix_axis.imshow(
        shares.to_numpy(dtype=float), cmap=ps.sequential_cmap(["#ffffff", *ps.CARBONATE_RAMP]), vmin=0, vmax=1
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            share = shares.iat[row, column]
            matrix_axis.text(
                column,
                row,
                f"{counts.iat[row, column]:,}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if np.isfinite(share) and share > 0.5 else ps.INK,
            )
    matrix_axis.set_xticks(range(len(labels)), labels, fontsize=6.5)
    matrix_axis.set_yticks(range(len(labels)), labels, fontsize=6.5)
    matrix_axis.set_xlabel("Predicted class, % CaCO₃")
    matrix_axis.set_ylabel("Measured class, % CaCO₃")
    matrix_axis.grid(False)
    matrix_axis.tick_params(length=0)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)
    total = int(counts.to_numpy().sum())
    agreement = np.trace(counts.to_numpy()) / total if total else np.nan
    ps.panel_subtitle(matrix_axis, f"{agreement:.0%} on the diagonal")
    ps.panel_letter(matrix_axis, "abc"[len(names)])
    return figure


# --- 4. scores per carbonate class --------------------------------------------------------------


def class_scores(
    predictions: pd.DataFrame,
    *,
    n: int = 1000,
    seed: int = 42,
    block_km: float = 10.0,
    block_size_m: Optional[float] = None,
) -> pd.DataFrame:
    """RPIQ and bias per kind, model and measured carbonate class, with block-bootstrap intervals.

    The test points of the ``random`` and ``spatial_kfold`` plans, the four stage-2 models on every
    covariate (``combined``). With ``block_size_m``, only the ``spatial_kfold`` plans of that block
    size are kept: plans of several sizes each test every point once, and must not be pooled.

    Returns
    -------
    pandas.DataFrame
        The :class:`~yg_eo_soilnet.a1.evaluation.BootstrapResult` intervals of ``rpiq`` and
        ``bias``: ``kind, model, carbonate_class_measured, metric, estimate, low, high, n, small, ...``.
    """
    rows = predictions[
        (predictions["split"] == TEST)
        & predictions["kind"].isin([RANDOM, SPATIAL_KFOLD])
        & (predictions["covariate_group"] == COMBINED_GROUP)
        & predictions["model"].isin(MODELS)
    ].dropna(subset=["carbonate_class_measured"])
    if block_size_m is not None:
        rows = rows[(rows["kind"] != SPATIAL_KFOLD) | (rows["block_size_m"] == float(block_size_m))]
    if rows.empty:
        return pd.DataFrame()
    result = evaluation.block_bootstrap(
        rows, ["kind", "model", "carbonate_class_measured"], block_km=block_km, n=n, seed=seed
    )
    return result.intervals[result.intervals["metric"].isin(["rpiq", "bias"])].reset_index(drop=True)


def figure_class_scores(inputs: FigureInputs, *, n: int = 1000):
    """Figure 4: RPIQ and bias per carbonate class for the four models, random and spatial CV."""
    kfold = inputs.settings.plans.spatial_kfold
    primary = kfold.cell_size_m if kfold.method == "grid" else None
    return draw_class_scores(
        class_scores(inputs.predictions, n=n, seed=inputs.settings.seed, block_size_m=primary),
        labels=inputs.class_labels,
    )


@ps.styled
def draw_class_scores(scores: pd.DataFrame, *, labels):
    """Draw figure 4 from :func:`class_scores`."""
    if scores.empty:
        return _missing("No class scores: run run_a1.py with --kinds random,spatial_kfold.")
    kinds = [kind for kind in (RANDOM, SPATIAL_KFOLD) if kind in set(scores["kind"])]
    figure, axes = plt.subplots(
        2, len(kinds), figsize=(ps.FIG_WIDTH_FULL, 4.2), layout="constrained", sharey="row", squeeze=False
    )
    titles = {RANDOM: "Random split", SPATIAL_KFOLD: "Spatial block cross-validation"}
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    for column, kind in enumerate(kinds):
        for row, metric in enumerate(("rpiq", "bias")):
            axis = axes[row, column]
            part = scores[(scores["kind"] == kind) & (scores["metric"] == metric)]
            for offset, model in zip(offsets, MODELS):
                chosen = part[part["model"] == model].set_index("carbonate_class_measured").reindex(labels)
                x = np.arange(len(labels)) + offset
                color = ps.A1_MODEL_COLORS[model]
                errors = np.vstack(
                    [
                        (chosen["estimate"] - chosen["low"]).clip(lower=0),
                        (chosen["high"] - chosen["estimate"]).clip(lower=0),
                    ]
                )
                axis.errorbar(x, chosen["estimate"], yerr=errors, fmt="none", ecolor=color, elinewidth=1, capsize=0)
                small = chosen["small"].fillna(True).astype(bool).to_numpy()
                axis.scatter(x[~small], chosen["estimate"][~small], s=16, color=color, zorder=3)
                axis.scatter(x[small], chosen["estimate"][small], s=16, facecolors="white", edgecolors=color, zorder=3)
            if metric == "bias":
                axis.axhline(0, color=ps.BASELINE, lw=0.8)
            axis.set_xticks(range(len(labels)), [f"{label} %" for label in labels], fontsize=7)
            if column == 0:
                axis.set_ylabel("RPIQ" if metric == "rpiq" else "Bias, g/kg")
            if row == 0:
                axis.set_title(titles[kind], fontsize=8.5)
            else:
                axis.set_xlabel("Measured CaCO₃ class")
            ps.panel_letter(axis, "abcd"[row * len(kinds) + column])
    handles = [
        Line2D([], [], ls="", marker="o", color=ps.A1_MODEL_COLORS[model], label=MODEL_LABELS[model])
        for model in MODELS
    ]
    handles.append(
        Line2D(
            [], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=ps.MUTED, label="Fewer than 30 points"
        )
    )
    figure.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    return figure


# --- 5-7. transfer ---------------------------------------------------------------------------------


def figure_transfer(inputs: FigureInputs):
    """Figure 5: (a) leave-one-region-out against local RMSE per region; (b) the drop per covariate group."""
    return draw_transfer(inputs.transfer)


@ps.styled
def draw_transfer(transfer: pd.DataFrame):
    """Draw figure 5 from :func:`~yg_eo_soilnet.a1.evaluation.transfer_table`."""
    logo = transfer[(transfer["kind"] == LOGO) & transfer["has_local"]] if not transfer.empty else transfer
    if logo.empty:
        return _missing(
            "No transfer scores: run run_a1.py with --kinds logo --groups (the local model needs --groups)."
        )
    figure, (scatter, drops) = plt.subplots(
        1, 2, figsize=(ps.FIG_WIDTH_FULL, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]}
    )
    combined = logo[logo["covariate_group"] == COMBINED_GROUP]
    scatter.scatter(
        combined["rmse_local"], combined["rmse_transferred"], s=22, color=ps.A1_MODEL_COLORS[GLOBAL_AUX], zorder=3
    )
    for row in combined.itertuples():
        scatter.annotate(
            str(row.region),
            (row.rmse_local, row.rmse_transferred),
            fontsize=6,
            color=ps.INK_2,
            xytext=(3, 2),
            textcoords="offset points",
        )
    values = pd.concat([combined["rmse_local"], combined["rmse_transferred"]]).dropna()
    if len(values):
        low, high = float(values.min()) * 0.95, float(values.max()) * 1.05
        scatter.plot([low, high], [low, high], color=ps.INK_2, lw=0.8, ls="--")
    ps.square_panel(scatter)
    scatter.set_xlabel("Local RMSE, g/kg (region's own points)")
    scatter.set_ylabel("Transferred RMSE, g/kg (other regions)")
    ps.panel_letter(scatter, "a")

    order = logo.groupby("covariate_group")["rmse_drop"].median().sort_values(ascending=False).index.tolist()
    for position, group in enumerate(order):
        part = logo[logo["covariate_group"] == group]["rmse_drop"].dropna()
        jitter = np.random.default_rng(position).uniform(-0.12, 0.12, len(part))
        color = ps.A1_MODEL_COLORS[GLOBAL_AUX] if group == COMBINED_GROUP else ps.MUTED
        drops.scatter(np.full(len(part), position) + jitter, part, s=12, color=color, alpha=0.8, zorder=3)
        if len(part):
            drops.hlines(part.median(), position - 0.28, position + 0.28, color=ps.INK, lw=1.4, zorder=4)
    drops.axhline(0, color=ps.BASELINE, lw=0.8)
    drops.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=7)
    drops.set_ylabel("Transfer drop: RMSE increase, g/kg")
    ps.panel_subtitle(drops, "one dot per region; bar = median")
    ps.panel_letter(drops, "b")
    return figure


def figure_mixed(inputs: FigureInputs):
    """Figure 6: RMSE against the share of target-region points in training, mixed vs control."""
    return draw_mixed(inputs.transfer)


@ps.styled
def draw_mixed(transfer: pd.DataFrame):
    """Draw figure 6 from :func:`~yg_eo_soilnet.a1.evaluation.transfer_table`."""
    mixed = transfer[transfer["kind"] == MIXED] if not transfer.empty else transfer
    if mixed.empty:
        return _missing("No mixed-training scores: run run_a1.py with --kinds mixed (it brings the control plans).")
    logo = transfer[transfer["kind"] == LOGO]
    groups = [COMBINED_GROUP] + sorted(set(mixed["covariate_group"]) - {COMBINED_GROUP})
    groups = [group for group in groups if group in set(mixed["covariate_group"])]
    columns = min(3, len(groups))
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(ps.FIG_WIDTH_FULL, 2.2 * rows + 0.4), layout="constrained", squeeze=False, sharey=True
    )
    mixed_color, control_color = ps.A1_MODEL_COLORS[GLOBAL_AUX], ps.A1_MODEL_COLORS["local"]
    for index, group in enumerate(groups):
        axis = axes.flat[index]
        part = mixed[mixed["covariate_group"] == group]
        anchor = logo[logo["covariate_group"] == group].set_index("region")["rmse_transferred"]
        curves = []
        for region, curve in part.groupby("region"):
            curve = curve.sort_values("fraction")
            x = curve["fraction"].to_numpy()
            transferred = curve["rmse_transferred"].to_numpy()
            if region in anchor.index:  # no share of the region at all: the leave-one-region-out score
                x = np.r_[0.0, x]
                transferred = np.r_[anchor[region], transferred]
            axis.plot(x, transferred, color=mixed_color, lw=0.6, alpha=0.35)
            axis.plot(curve["fraction"], curve["rmse_local"], color=control_color, lw=0.6, alpha=0.35)
            curves.append(pd.DataFrame({"fraction": x, "mixed": transferred}))
        means = pd.concat(curves).groupby("fraction")["mixed"].mean()
        axis.plot(means.index, means.to_numpy(), color=mixed_color, lw=2, marker="o", ms=3)
        control = part.groupby("fraction")["rmse_local"].mean()
        axis.plot(control.index, control.to_numpy(), color=control_color, lw=2, marker="o", ms=3)
        axis.set_title(group, fontsize=8)
        axis.set_xlabel("Share of the region in training")
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        if index % columns == 0:
            axis.set_ylabel("Test RMSE, g/kg")
        ps.panel_letter(axis, "abcdefghijkl"[index])
    for axis in list(axes.flat)[len(groups) :]:
        axis.set_axis_off()
    handles = [
        Line2D([], [], color=mixed_color, lw=2, label="Mixed: other regions + share of this one"),
        Line2D([], [], color=control_color, lw=2, label="Control: the share alone"),
        Line2D([], [], color=ps.MUTED, lw=0.6, label="One region (thin); mean (bold)"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    return figure


def dissimilarity_frame(transfer: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """The ``logo`` transfer drop of every region beside how different the region is."""
    if transfer.empty or summary.empty:
        return pd.DataFrame()
    logo = transfer[
        (transfer["kind"] == LOGO) & (transfer["covariate_group"] == COMBINED_GROUP) & transfer["has_local"]
    ]
    summary = summary.assign(region=summary["held_out"].astype(str))
    return logo.assign(region=logo["region"].astype(str)).merge(summary, on="region", how="inner")


def figure_dissimilarity(inputs: FigureInputs):
    """Figure 7: the transfer drop against the dissimilarity index and the SOM-CEC correlation change."""
    return draw_dissimilarity(dissimilarity_frame(inputs.transfer, inputs.dissimilarity_summary))


@ps.styled
def draw_dissimilarity(frame: pd.DataFrame):
    """Draw figure 7 from :func:`dissimilarity_frame`."""
    if frame.empty:
        return _missing("No dissimilarity: run run_a1.py with --kinds logo --groups.")
    panels = [("median_di", "Median dissimilarity index")]
    if "spearman_som_cec_diff" in frame:
        panels.append(("spearman_som_cec_diff", "Δ Spearman ρ(SOM, CEC), region − training"))
    figure, axes = plt.subplots(1, len(panels), figsize=(ps.FIG_WIDTH_FULL, 2.8), layout="constrained", squeeze=False)
    color_values = frame["carbonate_tvd"] if "carbonate_tvd" in frame else pd.Series(0.0, index=frame.index)
    points = None
    for letter, axis, (column, label) in zip("ab", axes.flat, panels):
        points = axis.scatter(
            frame[column],
            frame["rmse_drop"],
            c=color_values,
            cmap=ps.sequential_cmap(),
            vmin=0,
            vmax=max(0.01, float(color_values.max())),
            s=26,
            edgecolor=ps.INK_2,
            lw=0.3,
            zorder=3,
        )
        for row in frame.itertuples():
            axis.annotate(
                str(row.region),
                (getattr(row, column), row.rmse_drop),
                fontsize=6,
                color=ps.INK_2,
                xytext=(3, 2),
                textcoords="offset points",
            )
        axis.axhline(0, color=ps.BASELINE, lw=0.8)
        axis.set_xlabel(label)
        if letter == "a":
            axis.set_ylabel("Transfer drop: RMSE increase, g/kg")
        pairs = frame[[column, "rmse_drop"]].dropna()
        if len(pairs) >= 3:
            rho = pairs[column].corr(pairs["rmse_drop"], method="spearman")
            # In the title slot, not a box inside the axes: with few regions a box hides one of them.
            ps.panel_subtitle(axis, ps.minus(f"Spearman ρ = {rho:.2f}, n = {len(pairs)} regions"))
        ps.panel_letter(axis, letter)
    colorbar = figure.colorbar(points, ax=list(axes.flat), shrink=0.8, pad=0.02, aspect=25)
    colorbar.set_label("Carbonate-mix difference from training (TVD)", fontsize=7)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7)
    return figure


# --- 8. SHAP -----------------------------------------------------------------------------------------


def _input_column(name: str) -> str:
    """The input column a prepared feature came from: ``num__x`` and its gap flag are both ``x``."""
    _, _, column = str(name).partition("__")
    return column.removeprefix("missingindicator_") if column else str(name)


def _tree_shap(model, prepared) -> np.ndarray:
    """SHAP values of a fitted tree model, one row per point and one column per prepared input.

    An XGBoost model computes its own exact tree SHAP values (``pred_contribs``), which is used
    for it: the ``shap`` library cannot read the model files of recent XGBoost releases. Any other
    tree model goes through ``shap.TreeExplainer``.
    """
    if hasattr(model, "get_booster"):
        import xgboost

        contributions = model.get_booster().predict(
            xgboost.DMatrix(np.asarray(prepared, dtype=float)), pred_contribs=True
        )
        return np.asarray(contributions)[:, :-1]  # the last column is the model's base value
    import shap

    return np.asarray(shap.TreeExplainer(model).shap_values(prepared))


def shap_by_class(inputs: FigureInputs, *, plan_id: str = RANDOM) -> pd.DataFrame:
    """Mean |SHAP| of each input of ``global_aux``, over one plan's test points, per carbonate class.

    The model is fitted again - run_a1.py keeps predictions, not models - on the plan's training
    points, exactly as the run fitted it: the same inputs, the stage-1 predictions read back from
    the saved audit, the same tuning. SHAP values are computed on the prepared inputs
    (:func:`_tree_shap`) and added up per input column (a value and its gap flag count together). They are in
    the units the model was trained on - the target's, or its log scale when the target is
    log-transformed.

    Returns
    -------
    pandas.DataFrame
        ``class, feature, mean_abs_shap, n``.
    """
    from yg_eo_soilnet.a1.runner import _auxiliary_from_frame
    from yg_eo_soilnet.a1.stage2 import build_context, run_model
    from yg_eo_soilnet.datamodules.splitting import SplitPlan

    settings = inputs.settings
    index = pd.read_csv(inputs.plans_dir / "index.csv")
    kind = index.set_index("plan_id").loc[plan_id, "kind"]
    plan = SplitPlan.from_frame(pd.read_parquet(inputs.plans_dir / f"{plan_id}.parquet"), strategy=kind)
    auxiliary = _auxiliary_from_frame(inputs.stage1(plan_id), settings)
    table = inputs.table
    context = build_context(table, plan, auxiliary, settings, inputs.config)
    resolved = inputs.a1.resolve_groups(table)
    auxiliary_columns = list(resolved.groups.get(AUXILIARY_GROUP, ()))
    global_columns = [column for column in resolved.combined if column not in auxiliary_columns]
    outcome = run_model(context, GLOBAL_AUX, global_columns=global_columns, auxiliary_columns=auxiliary_columns)
    columns = global_columns + [column for column in auxiliary_columns if column not in global_columns]

    test = context.split.index[(context.split == TEST).to_numpy()]
    preprocessor = outcome.fitted.named_steps["preprocessor"]
    model = outcome.fitted.named_steps["model"]
    model = getattr(model, "regressor_", model)
    prepared = preprocessor.transform(context.X.loc[test, columns])
    values = _tree_shap(model, prepared)
    names = [_input_column(name) for name in preprocessor.get_feature_names_out()]
    per_input = pd.DataFrame(np.abs(values), columns=names, index=test).T.groupby(level=0).sum().T

    classes = context.class_measured.reindex(test)
    rows = []
    for label in inputs.class_labels:
        part = per_input[(classes == label).to_numpy()]
        if part.empty:
            continue
        for feature, value in part.mean().items():
            rows.append({"class": label, "feature": feature, "mean_abs_shap": float(value), "n": len(part)})
    return pd.DataFrame(rows, columns=["class", "feature", "mean_abs_shap", "n"])


def figure_shap(inputs: FigureInputs, *, plan_id: str = RANDOM, n_features: int = 10):
    """Figure 8: the inputs ``global_aux`` relies on most, within each carbonate class."""
    try:
        values = shap_by_class(inputs, plan_id=plan_id)
    except FileNotFoundError as error:
        return _missing(f"No SHAP values: {error}")
    return draw_shap(values, labels=inputs.class_labels, n_features=n_features)


@ps.styled
def draw_shap(values: pd.DataFrame, *, labels, n_features: int = 10):
    """Draw figure 8 from :func:`shap_by_class`."""
    present = [label for label in labels if label in set(values["class"])]
    if not present:
        return _missing("No SHAP values: no test point has a carbonate class.")
    # Two per row: the feature names are long, and four panels side by side leave them no room.
    columns = min(2, len(present))
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(ps.FIG_WIDTH_FULL, rows * (0.2 * n_features + 0.8)),
        layout="constrained",
        squeeze=False,
    )
    for index, (axis, label) in enumerate(zip(axes.flat, present)):
        part = values[values["class"] == label].nlargest(n_features, "mean_abs_shap")[::-1]
        axis.barh(range(len(part)), part["mean_abs_shap"], color=_class_color(label), height=0.7)
        # Styled first: restyle_axes resets the tick label size.
        ps.restyle_axes(axis, grid_axis="x")
        axis.set_yticks(range(len(part)), part["feature"])
        axis.tick_params(axis="y", labelsize=6.5)
        axis.tick_params(axis="x", labelsize=7)
        ps.panel_subtitle(axis, f"{label} % CaCO₃, n = {int(part['n'].iloc[0]):,}")
        axis.set_xlabel("Mean |SHAP|", fontsize=7.5)
        ps.panel_letter(axis, "abcd"[index])
    for axis in list(axes.flat)[len(present) :]:
        axis.set_axis_off()
    return figure


# --- all of them -------------------------------------------------------------------------------------


def render_all(
    inputs: FigureInputs,
    folder=None,
    *,
    relief: bool = True,
    cache_dir: Optional[str] = None,
    n_boot: int = 1000,
    shap_plan: str = RANDOM,
    close: bool = False,
) -> dict:
    """Draw the eight figures and save each as PNG and PDF.

    Parameters
    ----------
    inputs : FigureInputs
    folder : str or pathlib.Path, optional
        Where to save; the run folder's ``figures/`` unless given.
    relief : bool, default True
        Shaded relief under the map (downloaded once, then cached); False draws flat ground.
    cache_dir : str, optional
        Where the map's tiles and borders are cached.
    n_boot : int, default 1000
        Bootstrap draws for figures 2 and 4.
    shap_plan : str, default "random"
        The plan whose ``global_aux`` model figure 8 explains.
    close : bool, default False
        Close each figure once saved, to free memory when they are not going to be shown.

    Returns
    -------
    dict
        File name to figure, in :data:`FIGURE_NAMES` order.
    """
    folder = Path(folder) if folder else inputs.run_dir / "figures"
    builders = {
        "fig1_points_map": lambda: figure_points_map(inputs, relief=relief, cache_dir=cache_dir),
        "fig2_correlations": lambda: figure_correlations(inputs, n=n_boot),
        "fig3_stage1": lambda: figure_stage1(inputs, plan_id=shap_plan),
        "fig4_class_scores": lambda: figure_class_scores(inputs, n=n_boot),
        "fig5_transfer": lambda: figure_transfer(inputs),
        "fig6_mixed": lambda: figure_mixed(inputs),
        "fig7_dissimilarity": lambda: figure_dissimilarity(inputs),
        "fig8_shap": lambda: figure_shap(inputs, plan_id=shap_plan),
    }
    figures = {}
    for name in FIGURE_NAMES:
        figure = builders[name]()
        save_figure(figure, name, folder)
        if is_message_figure(figure):
            logger.warning(f"{name}: {figure._a1_message}")
        if close:
            plt.close(figure)
        figures[name] = figure
    return figures
