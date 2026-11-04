"""Which region each point is in, for the experiments that train in one region and test in another.

Regions come from one of two places, chosen by ``regions.source`` in the A1 settings: a column of
the data that already names each point's region, or a polygon file of region outlines, each point
then taking the name of the outline it falls in. This module does the second: it reads the file,
brings it to latitude/longitude, and matches the points to it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

#: The coordinate system the points are in: latitude/longitude in degrees, on WGS84.
POINTS_CRS = "EPSG:4326"


def resolve_polygons_path(path: str, data_folder: Optional[str]) -> str:
    """The full path of a polygon file; a relative one is taken from the data folder.

    Examples
    --------
    >>> resolve_polygons_path("regions.geojson", "/data/soil")
    '/data/soil/regions.geojson'
    >>> resolve_polygons_path("/elsewhere/regions.gpkg", "/data/soil")
    '/elsewhere/regions.gpkg'
    """
    if os.path.isabs(path) or not data_folder:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(data_folder, path))


def regions_from_polygons(
    coordinates: pd.DataFrame,
    polygons_path: str,
    name_field: str,
    *,
    lat_column: str = "lat",
    lon_column: str = "lon",
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """Name each point's region by the outline it falls in.

    The polygon file can be in any coordinate system: it is reprojected to latitude/longitude
    before the points are matched. A file that does not say which system it is in is taken to be in
    latitude/longitude already, and the run says so.

    A point on the border two regions share belongs to both, and so does one inside two outlines
    that overlap; it keeps the first, and the count is reported. A point inside no outline, or
    without coordinates, has no region.

    Parameters
    ----------
    coordinates : pandas.DataFrame
        One row per point, indexed by point id, with latitude and longitude in degrees.
    polygons_path : str
        The polygon file: anything geopandas reads (GeoJSON, GeoPackage, shapefile...).
    name_field : str
        The field of the file holding each region's name.
    lat_column, lon_column : str
        The coordinate columns of ``coordinates``.
    logger : logging.Logger, optional
        Where the counts are reported.

    Returns
    -------
    pandas.Series
        Each point's region name, indexed like ``coordinates``; missing for a point in no region.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the file has no field ``name_field``; the message lists the fields it does have.
    """
    from yg_eo_soilnet.map_style import _ensure_proj_data

    logger = logger or logging.getLogger(__name__)
    if not os.path.exists(polygons_path):
        raise FileNotFoundError(f"regions.polygons_path not found: {polygons_path}")

    # Without PROJ's database GDAL cannot read a file's coordinate system, which a process started
    # outside its environment's activation script is missing.
    _ensure_proj_data()
    import geopandas as gpd

    outlines = gpd.read_file(polygons_path)
    if name_field not in outlines.columns:
        fields = sorted(column for column in outlines.columns if column != outlines.geometry.name)
        raise KeyError(f"The polygon file has no field {name_field!r}; its fields are {fields}")
    if outlines.crs is None:
        logger.warning(
            f"{os.path.basename(polygons_path)} does not say which coordinate system it is in; it is "
            "taken to be latitude/longitude (EPSG:4326)."
        )
        outlines = outlines.set_crs(POINTS_CRS)
    else:
        outlines = outlines.to_crs(POINTS_CRS)
    outlines = outlines[[name_field, outlines.geometry.name]].rename(columns={name_field: "_region"})

    placed = coordinates[[lat_column, lon_column]].dropna()
    points = gpd.GeoDataFrame(
        index=placed.index,
        geometry=gpd.points_from_xy(placed[lon_column], placed[lat_column]),
        crs=POINTS_CRS,
    )
    # "intersects" rather than "within": a point exactly on a shared border is within neither of the
    # two outlines, and would otherwise lose its region for lying on a line.
    matched = gpd.sjoin(points, outlines, how="left", predicate="intersects")
    repeated = matched.index.duplicated(keep="first")
    if repeated.any():
        logger.warning(
            f"{int(pd.Index(matched.index[repeated]).nunique())} point(s) fall in more than one region "
            "(on a shared border, or where outlines overlap); each keeps the first."
        )
    regions = matched.loc[~repeated, "_region"]

    result = regions.reindex(coordinates.index)
    unplaced = int(result.isna().sum())
    if unplaced:
        logger.warning(
            f"{unplaced} of {len(result)} point(s) are in no region - outside every outline, or "
            "without coordinates - and take no part in the region experiments."
        )
    logger.info(f"Regions from {os.path.basename(polygons_path)}: {result.value_counts().to_dict()}")
    return result.rename(None)
