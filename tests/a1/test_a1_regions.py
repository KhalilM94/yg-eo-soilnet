"""Matching points to region outlines read from a polygon file.

The demo writes its regions both as a column and as outlines, so the outlines can be checked
against the column point by point.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from yg_eo_soilnet.a1.regions import regions_from_polygons, resolve_polygons_path


@pytest.fixture(scope="module")
def targets(a1_demo_dir):
    return pd.read_csv(a1_demo_dir / "targets.csv").set_index("uuid")


def _regions(coordinates, path, name_field="name", **kwargs):
    return regions_from_polygons(coordinates, str(path), name_field, **kwargs)


def test_every_point_takes_the_region_of_the_outline_it_falls_in(targets, a1_demo_dir):
    regions = _regions(targets[["lat", "lon"]], a1_demo_dir / "regions.geojson")

    pd.testing.assert_series_equal(regions, targets["region"].rename(None))


def test_an_outline_file_in_another_coordinate_system_gives_the_same_regions(targets, a1_demo_dir, tmp_path):
    # The same outlines in metres (Web Mercator), as a GeoPackage rather than GeoJSON.
    path = tmp_path / "regions_3857.gpkg"
    gpd.read_file(a1_demo_dir / "regions.geojson").to_crs("EPSG:3857").to_file(path)

    regions = _regions(targets[["lat", "lon"]], path)

    pd.testing.assert_series_equal(regions, targets["region"].rename(None))


def test_a_point_outside_every_outline_has_no_region(targets, a1_demo_dir, caplog):
    coordinates = targets[["lat", "lon"]].iloc[:3].copy()
    coordinates.iloc[0] = [40.0, 10.0]  # far outside the demo area

    with caplog.at_level(logging.WARNING):
        regions = _regions(coordinates, a1_demo_dir / "regions.geojson")

    assert pd.isna(regions.iloc[0])
    assert regions.iloc[1:].notna().all()
    assert "1 of 3 point(s) are in no region" in caplog.text


def test_a_point_without_coordinates_has_no_region(targets, a1_demo_dir):
    coordinates = targets[["lat", "lon"]].iloc[:2].copy()
    coordinates.iloc[0] = [np.nan, np.nan]

    regions = _regions(coordinates, a1_demo_dir / "regions.geojson")

    assert pd.isna(regions.iloc[0]) and pd.notna(regions.iloc[1])


def _two_squares(tmp_path, *, overlap, crs="EPSG:4326"):
    """Two squares side by side - touching, or overlapping by a quarter degree."""
    shift = 0.75 if overlap else 1.0
    outlines = gpd.GeoDataFrame(
        {"name": ["west", "east"]}, geometry=[box(0, 0, 1, 1), box(shift, 0, shift + 1, 1)], crs=crs
    )
    path = tmp_path / "squares.geojson"
    outlines.to_file(path)
    return path


def test_a_point_on_a_shared_border_keeps_one_region_rather_than_none(tmp_path, caplog):
    coordinates = pd.DataFrame({"lat": [0.5, 0.5], "lon": [1.0, 0.25]}, index=["border", "inside"])

    with caplog.at_level(logging.WARNING):
        regions = _regions(coordinates, _two_squares(tmp_path, overlap=False))

    assert regions["border"] in {"west", "east"}
    assert regions["inside"] == "west"
    assert "more than one region" in caplog.text


def test_a_point_where_outlines_overlap_keeps_the_first(tmp_path):
    coordinates = pd.DataFrame({"lat": [0.5], "lon": [0.9]}, index=["both"])
    regions = _regions(coordinates, _two_squares(tmp_path, overlap=True))
    assert regions["both"] == "west"


def test_an_outline_file_that_does_not_say_its_coordinate_system_is_taken_as_lat_lon(tmp_path, caplog):
    # A shapefile without its .prj companion carries no coordinate system at all. (A GeoJSON always
    # does: the format is latitude/longitude by definition.)
    path = tmp_path / "no_crs.shp"
    with pytest.warns(UserWarning, match="'crs' was not provided"):
        gpd.GeoDataFrame({"name": ["only"]}, geometry=[box(0, 0, 1, 1)], crs=None).to_file(path)
    assert not path.with_suffix(".prj").exists()
    coordinates = pd.DataFrame({"lat": [0.5], "lon": [0.5]}, index=["p"])

    with caplog.at_level(logging.WARNING):
        regions = _regions(coordinates, path)

    assert regions["p"] == "only"
    assert "does not say which coordinate system" in caplog.text


def test_a_missing_name_field_lists_the_fields_there_are(a1_demo_dir, targets):
    with pytest.raises(KeyError, match=r"no field 'region_name'.*\['name'\]"):
        _regions(targets[["lat", "lon"]], a1_demo_dir / "regions.geojson", name_field="region_name")


def test_a_missing_file_is_reported(targets, tmp_path):
    with pytest.raises(FileNotFoundError, match="polygons_path"):
        _regions(targets[["lat", "lon"]], tmp_path / "nowhere.geojson")


def test_other_coordinate_column_names_are_read(targets, a1_demo_dir):
    coordinates = targets[["lat", "lon"]].rename(columns={"lat": "y", "lon": "x"})
    regions = _regions(coordinates, a1_demo_dir / "regions.geojson", lat_column="y", lon_column="x")
    assert regions.notna().all()


def test_a_relative_polygon_path_is_taken_from_the_data_folder(tmp_path):
    assert resolve_polygons_path("r.geojson", str(tmp_path)) == str(tmp_path / "r.geojson")
    assert resolve_polygons_path(str(tmp_path / "x.gpkg"), "/elsewhere") == str(tmp_path / "x.gpkg")
