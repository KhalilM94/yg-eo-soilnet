"""What every A1 figure shares: the run's outputs (:class:`FigureInputs`), and the tables behind them.

:class:`FigureInputs` reads a run's prediction files, stage-1 audits, evaluation and dissimilarity
tables and the A1 points table, each when a figure first asks for it. The numbers behind the
figures come from functions that return plain tables (:func:`class_counts`,
:func:`spearman_by_class`, :func:`class_confusion`, :func:`shap_by_class`), so they can be checked,
and quoted in the text, without the drawing. Each figure's own module - :mod:`.design`,
:mod:`.maps`, :mod:`.correlations`, ... - draws it; :mod:`.render` draws them all.

A figure keeps the tables it was drawn from and its draft caption with it (:func:`attach`). A figure
whose inputs the run does not hold comes back as a short message saying what to run
(:func:`_missing`), rather than as an error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from yg_eo_soilnet import plot_style as ps
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP, COMBINED_GROUP, A1Config, stage1_prediction_column
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX
from yg_eo_soilnet.a1.strata import carbonate_class
from yg_eo_soilnet.datamodules.splitting import TEST

logger = logging.getLogger(__name__)

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
    n_boot : int, default 1000
        Bootstrap draws, for the intervals a figure computes itself (when ``evaluate.py`` has not
        written them, and for the per-class bias of figure 7).
    """

    a1: A1Config
    run_dir: Path
    plans_dir: Path
    regions_path: Optional[str] = None
    regions_name_field: Optional[str] = None
    n_boot: int = 1000
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
        n_boot: int = 1000,
    ) -> "FigureInputs":
        """The outputs of the run ``run_name`` made from these settings."""
        output = Path(a1.settings.output_dir)
        return cls(
            a1=a1,
            run_dir=output / "runs" / run_name,
            plans_dir=Path(plans_dir) if plans_dir else output / "plans",
            regions_path=regions_path,
            regions_name_field=regions_name_field,
            n_boot=n_boot,
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

    @property
    def index(self) -> pd.DataFrame:
        """The plan index (``index.csv``) of the plans the run used."""
        return self._cached(
            "index", lambda: pd.read_csv(self.plans_dir / "index.csv", keep_default_na=False, na_values=[""])
        )

    @property
    def block_km(self) -> float:
        """The side of a bootstrap block, km (``bootstrap.block_size_km``)."""
        return float(self.settings.bootstrap.block_size_km)

    @property
    def primary_block_size(self) -> Optional[float]:
        """The spatial block size the main figures use: ``cell_size_m`` when the run has it."""
        sizes = self.block_sizes
        wanted = float(self.settings.plans.spatial_kfold.cell_size_m)
        if wanted in sizes:
            return wanted
        return sizes[0] if sizes else None

    @property
    def block_sizes(self) -> list:
        """The spatial block sizes the run's predictions hold, smallest first."""
        rows = self.predictions[self.predictions["kind"] == SPATIAL_KFOLD]
        return sorted(float(size) for size in rows["block_size_m"].dropna().unique())

    @property
    def held_out_names(self) -> list:
        """The basins the run's leave-one-basin-out plans hold out."""
        return sorted(self.predictions.loc[self.predictions["kind"] == LOGO, "held_out"].dropna().unique())

    def _test_combined(self) -> pd.DataFrame:
        frame = self.predictions
        return frame[(frame["split"] == TEST) & (frame["covariate_group"] == COMBINED_GROUP)]

    @property
    def intervals(self) -> pd.DataFrame:
        """95 % block-bootstrap intervals per kind, block size and model (``evaluation/intervals.csv``).

        Read from the file ``evaluate.py`` wrote; computed the same way when it is not there.
        """

        def build():
            path = self.run_dir / "evaluation" / "intervals.csv"
            if path.exists():
                return pd.read_csv(path)
            return evaluation.block_bootstrap(
                self._test_combined(),
                ["kind", "block_size_m", "model"],
                block_km=self.block_km,
                n=self.n_boot,
                seed=self.settings.seed,
            ).intervals

        return self._cached("intervals", build)

    @property
    def intervals_by_region(self) -> pd.DataFrame:
        """The same per held-out basin (``evaluation/intervals_by_region.csv``)."""

        def build():
            path = self.run_dir / "evaluation" / "intervals_by_region.csv"
            if path.exists():
                return pd.read_csv(path)
            logo = self._test_combined()
            logo = logo[logo["kind"] == LOGO]
            if logo.empty:
                return pd.DataFrame(columns=["kind", "held_out", "model", "metric", "estimate", "low", "high"])
            return evaluation.block_bootstrap(
                logo, ["kind", "held_out", "model"], block_km=self.block_km, n=self.n_boot, seed=self.settings.seed
            ).intervals

        return self._cached("intervals_by_region", build)

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
        # Tight, so the caption line under the axes is kept.
        figure.savefig(path, dpi=ps.SAVE_DPI, bbox_inches="tight")
    return paths


def _missing(message: str):
    """A figure saying why there is nothing to draw, marked so :func:`is_message_figure` knows it."""
    figure = ps.message_figure(message)
    figure._a1_message = message
    return figure


def is_message_figure(figure) -> bool:
    """Whether ``figure`` is a message saying its inputs are missing, rather than a drawing."""
    return hasattr(figure, "_a1_message")


def attach(figure, data: dict, caption: str):
    """Keep with a figure the tables it was drawn from and its draft caption, for :func:`render_all`.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
    data : dict of str to pandas.DataFrame
        Each table plotted, by the name its CSV is written under (``figures/data/<name>.csv``).
    caption : str
        A draft caption stating the finding with its numbers (``figures/captions.md``).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.
    """
    figure._a1_data = {name: table for name, table in data.items() if table is not None}
    figure._a1_caption = caption
    return figure


def figure_data(figure) -> dict:
    """The tables a figure was drawn from (:func:`attach`); empty for a message figure."""
    return getattr(figure, "_a1_data", {})


def figure_caption(figure) -> str:
    """A figure's draft caption (:func:`attach`), or its message when its inputs are missing."""
    if is_message_figure(figure):
        return f"Not drawn: {figure._a1_message}"
    return getattr(figure, "_a1_caption", "")


def write_table(frame: pd.DataFrame, path_stem, *, float_format: str = "%.2f") -> list:
    """Write a table as CSV and as LaTeX (``<path_stem>.csv``, ``<path_stem>.tex``); return both paths."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    csv, tex = path_stem.with_suffix(".csv"), path_stem.with_suffix(".tex")
    frame.to_csv(csv, index=False)
    frame.to_latex(tex, index=False, float_format=lambda value: float_format % value, escape=True)
    return [csv, tex]


# --- the points ------------------------------------------------------------------------------


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


# --- correlations and spectra -----------------------------------------------------------------


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


# --- stage 1 --------------------------------------------------------------------------------


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


# --- transfer ---------------------------------------------------------------------------------


def dissimilarity_frame(transfer: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """The ``logo`` transfer drop of every region beside how different the region is."""
    if transfer.empty or summary.empty:
        return pd.DataFrame()
    logo = transfer[
        (transfer["kind"] == LOGO) & (transfer["covariate_group"] == COMBINED_GROUP) & transfer["has_local"]
    ]
    summary = summary.assign(region=summary["held_out"].astype(str))
    return logo.assign(region=logo["region"].astype(str)).merge(summary, on="region", how="inner")


# --- SHAP -----------------------------------------------------------------------------------------


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


def shap_by_class(inputs: FigureInputs, *, plan_id: str = RANDOM, model: str = GLOBAL_AUX) -> pd.DataFrame:
    """Mean |SHAP| of each input of ``model``, over one plan's test points, per carbonate class.

    The model is fitted again - run_a1.py keeps predictions, not models - on the plan's training
    points, exactly as the run fitted it: the same inputs, the stage-1 predictions read back from
    the saved audit, the same tuning. SHAP values are computed on the prepared inputs
    (:func:`_tree_shap`) and added up per input column (a value and its gap flag count together). They are in
    the units the model was trained on - the target's, or its log scale when the target is
    log-transformed. ``model`` is ``global_aux`` (every input and the stage-1 predictions) or
    ``global`` (every input but them).

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
    outcome = run_model(context, model, global_columns=global_columns, auxiliary_columns=auxiliary_columns)
    columns = (
        list(global_columns)
        if model == GLOBAL
        else global_columns + [column for column in auxiliary_columns if column not in global_columns]
    )

    test = context.split.index[(context.split == TEST).to_numpy()]
    preprocessor = outcome.fitted.named_steps["preprocessor"]
    estimator = outcome.fitted.named_steps["model"]
    estimator = getattr(estimator, "regressor_", estimator)
    prepared = preprocessor.transform(context.X.loc[test, columns])
    values = _tree_shap(estimator, prepared)
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
