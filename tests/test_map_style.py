"""The relief basemap and its chrome, including what happens with no tiles and no network."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from yg_eo_soilnet import map_style


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No cache and no network: every download fails, as it would on an isolated machine."""

    def refuse(url, path):
        """Stand in for a download that cannot happen."""
        raise OSError(f"no network, and {path} is not cached")

    monkeypatch.setattr(map_style, "cached_download", refuse)
    return str(tmp_path / "empty_cache")


# --- the projection --------------------------------------------------------


def test_the_origin_projects_to_the_origin():
    x, y = map_style.to_mercator(0.0, 0.0)
    assert (float(x), float(y)) == (0.0, 0.0)


def test_longitude_is_linear_but_latitude_is_not():
    # Web Mercator stretches north-south increasingly towards the poles; east-west it does not.
    x_one, _ = map_style.to_mercator(10.0, 0.0)
    x_two, _ = map_style.to_mercator(20.0, 0.0)
    assert float(x_two) == pytest.approx(2 * float(x_one))

    _, y_low = map_style.to_mercator(0.0, 10.0)
    _, y_high = map_style.to_mercator(0.0, 20.0)
    assert float(y_high) > 2 * float(y_low)


def test_a_round_trip_through_the_projection_returns_the_original_degrees():
    lons = np.array([-10.0, -6.0, 0.0, 6.0])
    lats = np.array([28.0, 32.0, 36.0, 0.0])
    x, y = map_style.to_mercator(lons, lats)
    back_lon = np.degrees(x / map_style.R_EARTH)
    back_lat = np.degrees(np.arctan(np.sinh(y / map_style.R_EARTH)))
    assert back_lon == pytest.approx(lons)
    assert back_lat == pytest.approx(lats)


# --- shading ---------------------------------------------------------------


def test_flat_ground_shades_to_one():
    flat = np.zeros((8, 8))
    extent = (0.0, 8000.0, 0.0, 8000.0)
    assert map_style.hillshade(flat, extent) == pytest.approx(np.ones((8, 8)))


def test_a_slope_facing_the_light_is_brighter_than_one_facing_away():
    # Lit from the north-west. Rows run southwards, so ground rising towards the south-east has a
    # surface tilted back towards the light and comes out bright; the opposite slope is in shadow.
    rows, cols = np.mgrid[0:16, 0:16]
    extent = (0.0, 16000.0, 0.0, 16000.0)
    rising_south_east = (rows + cols).astype(float) * 10.0

    lit = map_style.hillshade(rising_south_east, extent)
    shadowed = map_style.hillshade(-rising_south_east, extent)

    assert lit.mean() > 1.0 > shadowed.mean()
    assert lit.shape == (16, 16)
    assert (lit >= 0).all() and (shadowed >= 0).all()


# --- ticks and the scale bar ----------------------------------------------


def test_degree_ticks_are_whole_degrees_inside_the_bounds():
    lons, lats = map_style.degree_ticks_for((-10.2, 30.1, -5.8, 34.4), max_ticks=6)
    assert lons == [-10.0, -9.0, -8.0, -7.0, -6.0]
    assert lats == [31.0, 32.0, 33.0, 34.0]


def test_degree_ticks_thin_out_rather_than_crowd_the_axis():
    lons, _ = map_style.degree_ticks_for((-30.0, 0.0, 30.0, 1.0), max_ticks=5)
    assert len(lons) <= 5


def test_the_scale_bar_is_shorter_than_the_map_and_sits_inside_it():
    bounds = (-9.5, 31.0, -4.5, 35.0)
    lon, lat, marks = map_style.scale_bar_placement(bounds)
    west, south, east, north = bounds
    span_degrees = marks[-1] / (111.32 * np.cos(np.radians(lat)))
    assert west < lon
    # The bar, and the "km" that hangs off its end, stay inside the right edge.
    assert lon + span_degrees < east
    assert south < lat < north


def test_a_small_map_gets_a_small_scale_bar():
    _, _, wide = map_style.scale_bar_placement((-10.0, 30.0, -2.0, 36.0))
    _, _, narrow = map_style.scale_bar_placement((-7.2, 32.0, -6.4, 32.6))
    assert narrow[-1] < wide[-1]


# --- drawing, offline ------------------------------------------------------


def test_the_basemap_still_draws_with_no_tiles_and_no_network(offline):
    figure, axis = plt.subplots()
    drawn = map_style.draw_basemap(axis, (-9.0, 31.0, -5.0, 34.0), cache_dir=offline)
    assert drawn == {"relief": False, "borders": False}
    # Flat ground stands in for the relief, so the points are not drawn onto white paper.
    assert axis.patches
    plt.close(figure)


def test_relief_false_fetches_no_elevation_tiles(monkeypatch, tmp_path):
    fetched = []

    def record(url, path):
        """Note what was asked for, then refuse it."""
        fetched.append(url)
        raise OSError("not cached")

    monkeypatch.setattr(map_style, "cached_download", record)
    figure, axis = plt.subplots()
    drawn = map_style.draw_basemap(axis, (-9.0, 31.0, -5.0, 34.0), cache_dir=str(tmp_path), relief=False)

    assert drawn["relief"] is False
    # The borders are a separate source and are still wanted; the 64 elevation tiles are not.
    assert not [url for url in fetched if "terrarium" in url]
    plt.close(figure)


def test_finish_map_zooms_to_the_bounds_and_squares_the_aspect():
    figure, axis = plt.subplots()
    bounds = (-9.0, 31.0, -5.0, 34.0)
    map_style.finish_map(axis, bounds, scale_bar_at=None, north_arrow_at=None)
    left, right = axis.get_xlim()
    bottom, top = axis.get_ylim()
    assert (left, right) == pytest.approx(
        (float(map_style.to_mercator(-9.0, 0)[0]), float(map_style.to_mercator(-5.0, 0)[0]))
    )
    assert (bottom, top) == pytest.approx(
        (float(map_style.to_mercator(0, 31.0)[1]), float(map_style.to_mercator(0, 34.0)[1]))
    )
    assert axis.get_aspect() == 1.0
    plt.close(figure)


def test_every_map_carries_the_credit_for_its_data_sources():
    figure, axis = plt.subplots()
    map_style.finish_map(axis, (-9.0, 31.0, -5.0, 34.0))
    assert any(map_style.CREDIT in text.get_text() for text in axis.texts)
    plt.close(figure)


# --- lining the chrome up --------------------------------------------------
# The scale bar is anchored in real coordinates and the north arrow in axes coordinates, so putting
# one above the other means converting between the two.


def test_a_coordinate_at_an_edge_of_the_map_is_at_an_edge_of_the_axes():
    bounds = (-10.0, 30.0, -2.0, 36.0)
    assert map_style.axes_fraction(bounds, lon=-10.0) == pytest.approx(0.0)
    assert map_style.axes_fraction(bounds, lon=-2.0) == pytest.approx(1.0)
    assert map_style.axes_fraction(bounds, lat=30.0) == pytest.approx(0.0)
    assert map_style.axes_fraction(bounds, lat=36.0) == pytest.approx(1.0)


def test_longitude_is_halfway_at_the_midpoint_but_latitude_is_not():
    bounds = (-10.0, 30.0, -2.0, 36.0)
    assert map_style.axes_fraction(bounds, lon=-6.0) == pytest.approx(0.5)
    # Mercator stretches the north, so the middle latitude sits below the middle of the axes.
    assert map_style.axes_fraction(bounds, lat=33.0) < 0.5


def test_asking_for_both_coordinates_gives_both_fractions():
    x, y = map_style.axes_fraction((-10.0, 30.0, -2.0, 36.0), lon=-6.0, lat=30.0)
    assert (x, y) == pytest.approx((0.5, 0.0))


def test_the_north_arrow_is_centred_over_the_scale_bar():
    # The collision this prevents is what drove the arrow into the opposite corner once before.
    bounds = (-10.24, 28.40, -1.84, 35.69)
    bar = map_style.scale_bar_placement(bounds)
    x, y, size = map_style.north_arrow_placement(bounds, bar)

    lon, lat, marks = bar
    left = map_style.axes_fraction(bounds, lon=lon)
    right = map_style.axes_fraction(bounds, lon=lon + marks[-1] / (111.32 * np.cos(np.radians(lat))))
    assert x == pytest.approx((left + right) / 2, abs=0.01)


def test_the_north_arrow_clears_the_top_of_the_scale_bar_and_stays_on_the_map():
    bounds = (-10.24, 28.40, -1.84, 35.69)
    bar = map_style.scale_bar_placement(bounds)
    x, y, size = map_style.north_arrow_placement(bounds, bar)

    assert y > map_style.axes_fraction(bounds, lat=bar[1])  # above the bar's foot
    assert 0.0 < x < 1.0
    assert 0.0 < y and y + size < 1.0  # the whole arrow, tip included, is inside


def test_the_placement_reproduces_the_paper_figures_own_numbers():
    # Figure 2 of the paper places its bar at 4.4 degrees west and its arrow at 0.755 across the
    # axes, both tuned by hand. Deriving the arrow from the bar has to land in the same place, or
    # the run's maps and the published ones stop looking like the same family of figure.
    bounds = (-10.8, 27.6, -1.0, 36.2)
    x, y, size = map_style.north_arrow_placement(bounds, (-4.4, 28.3, (0, 50, 100, 200)))
    assert (x, y) == pytest.approx((0.755, 0.125), abs=0.005)


# --- the locator inset -----------------------------------------------------


@pytest.fixture
def fake_countries(monkeypatch):
    """Two square countries instead of Natural Earth, so no test needs the real archive."""
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"ADM0_A3": ["MAR", "DZA"]},
        geometry=[box(-13.0, 21.0, -1.0, 36.0), box(-1.0, 21.0, 4.0, 36.0)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(map_style, "_all_countries", lambda cache_dir: frame)
    return frame


def test_the_locator_adds_a_panel_with_the_mapped_area_marked_on_it(fake_countries):
    figure, axis = plt.subplots()

    assert map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0)) is True

    # An inset belongs to its parent axes, not to the figure's own list.
    assert len(axis.child_axes) == 1
    inset = axis.child_axes[-1]
    assert inset.patches, "the mapped area should be drawn on the locator as a rectangle"
    # A locator carries no ticks: it says where, not how far.
    assert list(inset.get_xticks()) == [] and list(inset.get_yticks()) == []
    plt.close(figure)


def test_the_marked_area_matches_the_bounds_it_was_given(fake_countries):
    figure, axis = plt.subplots()
    bounds = (-10.0, 31.0, -5.0, 35.0)
    map_style.add_locator_inset(axis, bounds)

    rectangle = axis.child_axes[-1].patches[-1]
    left, bottom = map_style.to_mercator(bounds[0], bounds[1])
    right, top = map_style.to_mercator(bounds[2], bounds[3])
    assert rectangle.get_x() == pytest.approx(left)
    assert rectangle.get_width() == pytest.approx(right - left)
    assert rectangle.get_height() == pytest.approx(top - bottom)
    plt.close(figure)


def test_the_locator_goes_in_whichever_corner_it_is_asked_for(fake_countries):
    figure, axis = plt.subplots()
    map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0), loc="upper left")
    map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0), loc="lower right")

    left, right = (panel.get_position() for panel in axis.child_axes)
    assert left.x0 < 0.5 and left.y1 > 0.5
    assert right.x0 > 0.5 and right.y1 < 0.5
    plt.close(figure)


def test_an_unknown_corner_is_refused_rather_than_guessed(fake_countries):
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="loc must be one of"):
        map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0), loc="middle")
    plt.close(figure)


def test_the_locator_frames_the_whole_country_not_just_the_mapped_part(fake_countries):
    figure, axis = plt.subplots()
    map_style.add_locator_inset(axis, (-10.0, 34.0, -9.0, 35.0))

    inset = axis.child_axes[-1]
    # The country reaches 21 degrees north; a locator showing only the mapped sliver says nothing.
    assert inset.get_ylim()[0] < map_style.to_mercator(0, 22.0)[1]
    plt.close(figure)


def test_no_borders_means_no_locator_rather_than_no_map(offline, monkeypatch):
    def refuse(cache_dir):
        """Stand in for an archive that is neither cached nor reachable."""
        raise OSError("not cached")

    monkeypatch.setattr(map_style, "_all_countries", refuse)
    figure, axis = plt.subplots()

    assert map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0), cache_dir=offline) is False
    assert not axis.child_axes, "a missing locator must leave no empty panel behind"

    plt.close(figure)


def test_a_country_that_is_not_in_the_data_is_reported_rather_than_drawn_empty(fake_countries):
    figure, axis = plt.subplots()
    assert map_style.add_locator_inset(axis, (-10.0, 31.0, -5.0, 35.0), country="XXX") is False
    plt.close(figure)


def test_the_natural_earth_archive_is_parsed_once_however_many_maps_are_drawn(monkeypatch, tmp_path):
    # A map wants the borders twice - once for itself and once for its locator - and a run draws
    # one map per target, so an unmemoised read meant parsing the same 4.9 MB archive a dozen times.
    import geopandas as gpd
    from shapely.geometry import box

    parses = []

    def count(path):
        """Note each parse of the archive."""
        parses.append(path)
        return gpd.GeoDataFrame({"ADM0_A3": ["MAR"]}, geometry=[box(-13.0, 21.0, -1.0, 36.0)], crs="EPSG:4326")

    monkeypatch.setattr(map_style, "cached_download", lambda url, path: path)
    monkeypatch.setattr(gpd, "read_file", count)
    monkeypatch.setattr(map_style, "_COUNTRIES_CACHE", {})

    for _ in range(4):
        map_style._countries((-9.0, 31.0, -5.0, 34.0), str(tmp_path))

    assert len(parses) == 1


def test_a_map_with_no_coast_in_it_draws_rather_than_failing(fake_countries):
    # The sea is cut out of the view, so a wholly inland map leaves an empty shape. geopandas warns
    # when asked to plot one, and warnings are errors in this project - so this used to take the
    # whole run down for any holdout that happened to sit away from the coast.
    figure, axis = plt.subplots()
    inland = (-10.0, 25.0, -5.0, 30.0)  # well inside the fake country, no coastline

    drawn = map_style.draw_basemap(axis, inland, relief=False)

    assert drawn["borders"] is True
    plt.close(figure)
