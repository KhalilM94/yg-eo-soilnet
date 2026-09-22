"""Small helpers: the log transform for targets, a regular spatial grid, and the RPD/RPIQ scores."""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer
from sklearn.metrics import make_scorer, root_mean_squared_error
from mlflow.models import make_metric

class LogTransformer(BaseEstimator, TransformerMixin):
    """Convert target values to 10·ln(1 + y) and back.

    Used by the scikit-learn models for the targets listed in ``COLUMNS_TO_TRANSFORM``: it spreads
    out small values and pulls in large ones, which suits skewed soil properties. Predictions are
    converted back to the target's units with :meth:`inverse_transform`.

    Examples
    --------
    >>> import numpy as np
    >>> transformer = LogTransformer()
    >>> np.round(transformer.transform(np.array([0.0, 9.0, 99.0])), 3)
    array([ 0.   , 23.026, 46.052])
    >>> transformer.inverse_transform(transformer.transform(np.array([0.0, 9.0, 99.0])))
    array([ 0.,  9., 99.])
    """

    def transform(self, y):
        """Return 10·ln(1 + y)."""
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        """Return exp(y / 10) - 1, undoing :meth:`transform`."""
        return np.expm1(y / 10)

def _infer_utm_crs(lon_series, lat_series):
    """Return the UTM zone projection covering the average of the given coordinates."""
    lon_mean = lon_series.mean()
    lat_mean = lat_series.mean()
    zone = int((lon_mean + 180) // 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def assign_grid_ids(df, cell_size_m, lon_col='lon', lat_col='lat'):
    """Assign each point to a square cell of a regular grid laid over the data.

    The points are projected to the local UTM zone (units: metres) and a grid of ``cell_size_m``
    squares is laid over their bounding box. Used for the spatial split's grid strategy.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        The points, with longitude and latitude columns (degrees) or a geometry.
    cell_size_m : float
        Width of a grid cell in metres; at least 100 and no larger than the data's extent.
    lon_col, lat_col : str
        Names of the coordinate columns.

    Returns
    -------
    grid_id : pandas.Series of int
        The cell number of each point, in the same order.
    grid_gdf : geopandas.GeoDataFrame
        One square polygon per occupied cell (column ``Grid_ID``), in WGS84, for plotting.

    Raises
    ------
    ValueError
        If ``cell_size_m`` is missing, under 100 m, or larger than the data's extent.

    Examples
    --------
    >>> import pandas as pd
    >>> points = pd.DataFrame({"lat": [31.000, 31.001, 31.050], "lon": [-8.000, -8.001, -8.050]})
    >>> grid_id, grid = assign_grid_ids(points, cell_size_m=1000)
    >>> list(grid_id)   # the first two points, 150 m apart, share a cell
    [4, 4, 25]
    >>> len(grid)
    2
    """
    if isinstance(df, gpd.GeoDataFrame):
        gdf_wgs = df.copy()
        if gdf_wgs.crs is None:
            gdf_wgs.set_crs('EPSG:4326', inplace=True)
        elif gdf_wgs.crs.to_epsg() != 4326:
            gdf_wgs = gdf_wgs.to_crs(epsg=4326)
    else:
        gdf_wgs = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs='EPSG:4326'
        )

    lon_mean = float(gdf_wgs[lon_col].mean())
    lat_mean = float(gdf_wgs[lat_col].mean())
    utm_epsg = 32600 + int((lon_mean + 180) // 6) + 1 if lat_mean >= 0 else 32700 + int((lon_mean + 180) // 6) + 1
    utm_crs = CRS.from_epsg(utm_epsg)

    if gdf_wgs.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS to reproject")

    transformer = Transformer.from_crs(gdf_wgs.crs, utm_crs, always_xy=True)
    gdf_utm = gdf_wgs.copy()
    gdf_utm.geometry = gdf_utm.geometry.apply(
        lambda geom: shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
        if geom is not None
        else None
    )
    gdf_utm = gdf_utm.set_crs(utm_crs, allow_override=True)

    xmin, ymin, xmax, ymax = gdf_utm.total_bounds
    width, height = xmax - xmin, ymax - ymin

    if cell_size_m is None or cell_size_m <= 0:
        raise ValueError("cell_size_m must be a positive number.")
    if cell_size_m < 100:
        raise ValueError(f"cell_size_m={cell_size_m} too small (meters expected).")
    if cell_size_m > max(width, height):
        raise ValueError(f"cell_size_m={cell_size_m} exceeds dataset extent ({max(width, height):.1f}).")

    nx = max(1, int(np.ceil(width / cell_size_m)))
    ny = max(1, int(np.ceil(height / cell_size_m)))

    xs, ys = gdf_utm.geometry.x, gdf_utm.geometry.y
    col = np.clip(((xs - xmin) / cell_size_m).astype(int), 0, nx - 1)
    row = np.clip(((ys - ymin) / cell_size_m).astype(int), 0, ny - 1)
    grid_id = (row * nx + col).astype(int)

    # One polygon per occupied cell, for maps.
    occupied_cells = set(zip(row, col))
    grid_polys_utm, grid_ids = [], []
    for r in range(ny):
        y0, y1 = ymin + r * cell_size_m, min(ymin + (r + 1) * cell_size_m, ymax)
        for c in range(nx):
            if (r, c) not in occupied_cells:
                continue
            x0, x1 = xmin + c * cell_size_m, min(xmin + (c + 1) * cell_size_m, xmax)
            grid_polys_utm.append(box(x0, y0, x1, y1))
            grid_ids.append(r * nx + c)

    grid_gdf_utm = gpd.GeoDataFrame({"Grid_ID": grid_ids}, geometry=grid_polys_utm, crs=utm_crs)
    grid_gdf = grid_gdf_utm.to_crs(4326)

    return grid_id, grid_gdf

def rpd_score(predictions, targets):
    """Ratio of performance to deviation: standard deviation of the measurements / RMSE.

    Note the argument order: predictions first, measurements second.

    Examples
    --------
    >>> print(round(rpd_score([12, 18, 33, 39], [10, 20, 30, 40]), 3))
    6.086
    """
    std_dev = np.std(targets, ddof=1)
    rmse = root_mean_squared_error(targets, predictions)
    return std_dev / rmse

def rpiq_score(predictions, targets):
    """Ratio of performance to interquartile range: IQR of the measurements / RMSE.

    Note the argument order: predictions first, measurements second.

    Examples
    --------
    >>> print(round(rpiq_score([12, 18, 33, 39], [10, 20, 30, 40]), 3))
    7.071
    """
    iqr = np.percentile(targets, 75) - np.percentile(targets, 25)
    rmse = root_mean_squared_error(targets, predictions)
    return iqr / rmse

#: RPIQ as an MLflow evaluation metric, added to MLflow's own scores for scikit-learn models.
mlflow_rpiq_score = make_metric(eval_fn=rpiq_score, greater_is_better=True, name="rpiq_score")
