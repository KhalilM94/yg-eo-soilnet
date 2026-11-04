"""The made-up A1 dataset: it must give each A1 experiment something real to find.

If every point were slightly calcareous, or one region held all the points, the carbonate and
transfer experiments would run on the demo and prove nothing about the code.
"""

import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, shape

from experiments.a1.make_demo_data import (
    REGION_EFFECTS,
    make_a1_data,
    region_of,
    region_polygons,
)
from yg_eo_soilnet.a1.settings import FAO_CARBONATE_EDGES


@pytest.fixture(scope="module")
def demo(a1_demo_n_points):
    return make_a1_data(a1_demo_n_points, seed=0)


def test_the_same_seed_gives_the_same_data(demo, a1_demo_n_points):
    again = make_a1_data(a1_demo_n_points, seed=0)
    for first, second in zip(demo, again):
        pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_different_data(demo, a1_demo_n_points):
    other = make_a1_data(a1_demo_n_points, seed=1)
    assert not demo[1]["organic_matter_g_kg"].equals(other[1]["organic_matter_g_kg"])


def test_the_targets_file_holds_everything_a1_measures(demo):
    _, targets, _ = demo
    expected = {"uuid", "lat", "lon", "organic_matter_g_kg", "caco3_pct_total", "c_e_c_meq_100g", "region"}
    assert expected <= set(targets.columns)
    assert targets[list(expected)].notna().all().all()


def test_the_static_file_holds_no_coordinates_and_no_lab_values(demo):
    static, _, _ = demo
    assert not {"lat", "lon", "region", "caco3_pct_total", "organic_matter_g_kg"} & set(static.columns)


def test_every_fao_carbonate_class_is_well_populated(demo):
    _, targets, _ = demo
    classes = pd.cut(targets["caco3_pct_total"], [-np.inf, *FAO_CARBONATE_EDGES, np.inf], right=False)
    assert classes.value_counts().min() >= 30


def test_every_region_is_well_populated(demo):
    _, targets, _ = demo
    counts = targets["region"].value_counts()
    assert set(counts.index) == set(REGION_EFFECTS)
    assert counts.min() >= 30


def test_organic_matter_rises_with_carbonate_as_in_the_real_sample(demo):
    _, targets, _ = demo
    rho = targets["organic_matter_g_kg"].corr(targets["caco3_pct_total"], method="spearman")
    assert rho > 0.2


def test_carbonate_can_be_read_from_the_bare_soil_reflectance(demo):
    static, targets, _ = demo
    rho = static["s2_b2_barest"].corr(targets["caco3_pct_total"], method="spearman")
    assert rho > 0.2


def test_the_regions_differ_in_organic_matter(demo):
    _, targets, _ = demo
    means = targets.groupby("region")["organic_matter_g_kg"].mean()
    assert means["North-West"] > means["South-East"] + 3


def test_each_points_region_is_the_outline_it_falls_inside(demo):
    _, targets, _ = demo
    outlines = {feature["properties"]["name"]: shape(feature["geometry"]) for feature in region_polygons()["features"]}

    for row in targets.itertuples():
        assert outlines[row.region].contains(Point(row.lon, row.lat))


def test_the_outlines_do_not_overlap():
    outlines = [shape(feature["geometry"]) for feature in region_polygons()["features"]]
    for i, first in enumerate(outlines):
        for second in outlines[i + 1 :]:
            assert first.intersection(second).area == pytest.approx(0.0)


def test_a_point_on_the_dividing_line_goes_north_and_east():
    assert list(region_of(np.array([32.5]), np.array([-7.0]))) == ["North-East"]


def test_the_command_writes_four_files(a1_demo_dir):
    assert {path.name for path in a1_demo_dir.iterdir()} >= {
        "static.csv",
        "targets.csv",
        "timeseries.csv",
        "regions.geojson",
    }
    outlines = json.loads((a1_demo_dir / "regions.geojson").read_text())
    assert {feature["properties"]["name"] for feature in outlines["features"]} == set(REGION_EFFECTS)
