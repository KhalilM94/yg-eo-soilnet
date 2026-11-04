"""The A1 inputs: time-series percentiles, and the table that joins them onto the static covariates.

Two things matter most. The percentiles must not depend on when or how often a point was observed,
since most points' readings are scattered over the years with gaps. And the static covariates must
be exactly what main.py would use - no lab value, coordinate or region slipping in as an input.
"""

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1.features import (
    N_OBS_COLUMN,
    A1Table,
    build_table,
    percentile_column,
    timeseries_percentiles,
)


def _readings(seed=0, n_points=5):
    """Ragged, unsorted readings: each point has its own number of dates, with gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for point in range(n_points):
        dates = rng.choice(pd.date_range("2018-01-01", "2024-12-01", freq="MS"), size=3 + point * 4, replace=False)
        for date in dates:
            rows.append(
                {
                    "uuid": f"p{point}",
                    "date": date,
                    "ndvi": rng.uniform(0.05, 0.8),
                    "lst": rng.normal(20, 6),
                    "sza": rng.uniform(20, 70),
                    "sensor": "S2",
                }
            )
    return pd.DataFrame(rows)


def _summarise(readings, **kwargs):
    kwargs.setdefault("quantiles", [10, 50, 90])
    return timeseries_percentiles(readings, point_id_column="uuid", time_column="date", **kwargs)


# --- timeseries_percentiles ---------------------------------------------------------


def test_each_percentile_is_the_percentile_of_that_points_own_readings():
    readings = _readings()
    summary = _summarise(readings)

    for point, group in readings.groupby("uuid"):
        for q in (10, 50, 90):
            expected = np.percentile(group["ndvi"], q)
            assert summary.loc[point, percentile_column("ndvi", q)] == pytest.approx(expected)


def test_the_order_of_the_readings_makes_no_difference():
    readings = _readings()
    shuffled = readings.sample(frac=1.0, random_state=3)

    pd.testing.assert_frame_equal(_summarise(readings).sort_index(), _summarise(shuffled).sort_index())


def test_a_missing_reading_is_left_out_rather_than_counted():
    readings = pd.DataFrame({"uuid": ["a"] * 4, "ndvi": [0.1, np.nan, 0.3, 0.5]})
    summary = timeseries_percentiles(readings, point_id_column="uuid", quantiles=[50])

    assert summary.loc["a", "ts_ndvi_p50"] == pytest.approx(0.3)


def test_ts_n_obs_counts_each_points_dated_readings():
    readings = _readings()
    summary = _summarise(readings)

    assert summary[N_OBS_COLUMN].to_dict() == readings.groupby("uuid").size().to_dict()


def test_the_date_and_text_columns_are_never_summarised():
    summary = _summarise(_readings())

    assert not any(column.startswith(("ts_date", "ts_sensor", "ts_uuid")) for column in summary.columns)


def test_excluded_variables_are_left_out():
    summary = _summarise(_readings(), exclude=["sza"])

    assert not any(column.startswith("ts_sza") for column in summary.columns)
    assert percentile_column("ndvi", 50) in summary.columns


def test_named_variables_are_the_only_ones_summarised():
    summary = _summarise(_readings(), variables=["lst"])

    assert list(summary.columns) == ["ts_lst_p10", "ts_lst_p50", "ts_lst_p90", N_OBS_COLUMN]


def test_columns_come_variable_by_variable_with_percentiles_rising():
    summary = _summarise(_readings(), variables=["ndvi", "lst"], quantiles=[90, 10])

    assert list(summary.columns) == ["ts_ndvi_p90", "ts_ndvi_p10", "ts_lst_p90", "ts_lst_p10", N_OBS_COLUMN]


def test_a_variable_that_is_not_there_is_named():
    with pytest.raises(KeyError, match="evi"):
        _summarise(_readings(), variables=["ndvi", "evi"])


def test_a_table_without_the_id_column_is_refused():
    with pytest.raises(KeyError, match="point id"):
        timeseries_percentiles(pd.DataFrame({"ndvi": [0.1]}), point_id_column="uuid", quantiles=[50])


def test_nothing_left_to_summarise_is_refused():
    with pytest.raises(ValueError, match="No time-series variable"):
        _summarise(_readings(), variables=["ndvi"], exclude=["ndvi"])


def test_two_variables_that_would_share_a_column_name_are_refused():
    readings = pd.DataFrame({"uuid": ["a", "a"], "VH/VV": [1.0, 2.0], "VH_VV": [3.0, 4.0]})
    with pytest.raises(ValueError, match="same column name"):
        timeseries_percentiles(readings, point_id_column="uuid", quantiles=[50])


def test_the_column_name_survives_a_band_ratio_and_a_fractional_percentile():
    assert percentile_column("S1_asc_VH/VV", 50) == "ts_S1_asc_VH_VV_p50"
    assert percentile_column("x", 2.5) == "ts_x_p2.5"


# --- build_table ----------------------------------------------------------------------


@pytest.fixture
def table(a1_settings) -> A1Table:
    return build_table(a1_settings())


def test_the_static_covariates_are_the_ones_main_py_would_use(table):
    static = set(table.static_columns)

    assert {"elevation", "aridity_index", "s2_b12_barest", "landform_class"} <= static
    # No lab value, coordinate, id or region is ever an input.
    for column in ("organic_matter_g_kg", "caco3_pct_total", "c_e_c_meq_100g", "clay_pct", "ph_water"):
        assert column not in static
    for column in ("uuid", "lat", "lon", "region"):
        assert column not in static


def test_the_inputs_are_the_static_covariates_then_the_percentiles(table):
    assert table.feature_columns == table.static_columns + table.percentile_columns
    assert N_OBS_COLUMN in table.percentile_columns
    assert "ts_S2_NDVI_p90" in table.percentile_columns
    assert not set(table.percentile_columns) & set(table.static_columns)


def test_the_table_carries_the_lab_columns_a1_uses(table):
    assert table.label_columns == ("organic_matter_g_kg", "caco3_pct_total", "c_e_c_meq_100g")
    assert table.frame[list(table.label_columns)].notna().all().all()


def test_the_table_has_one_row_per_point_indexed_by_its_id(table, a1_demo_dir):
    targets = pd.read_csv(a1_demo_dir / "targets.csv")

    assert table.frame.index.name == "uuid"
    assert table.frame.index.is_unique
    assert set(table.frame.index) == set(targets["uuid"])


def test_a_region_kept_in_the_targets_file_still_arrives(table, a1_demo_dir):
    targets = pd.read_csv(a1_demo_dir / "targets.csv").set_index("uuid")

    assert table.region_column == "region"
    pd.testing.assert_series_equal(
        table.frame["region"].sort_index(), targets["region"].sort_index(), check_names=False
    )


def test_the_category_columns_are_named(table):
    assert table.categorical_columns == ("landform_class",)


def test_the_coordinates_come_as_lat_and_lon(table):
    assert list(table.coordinates.columns) == ["lat", "lon"]
    assert table.coordinates.index.equals(table.frame.index)


def test_the_percentiles_in_the_table_match_the_time_series(table, a1_demo_dir):
    readings = pd.read_csv(a1_demo_dir / "timeseries.csv")
    point = table.frame.index[0]
    ndvi = readings.loc[readings["uuid"] == point, "S2_NDVI"]

    assert table.frame.loc[point, "ts_S2_NDVI_p50"] == pytest.approx(np.percentile(ndvi, 50))


def test_a_point_without_a_time_series_keeps_its_row_with_empty_percentiles(a1_settings, a1_demo_dir, tmp_path):
    # A copy of the dataset with one point's readings removed.
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("static.csv", "targets.csv"):
        (folder / name).write_text((a1_demo_dir / name).read_text())
    readings = pd.read_csv(a1_demo_dir / "timeseries.csv")
    lost = readings["uuid"].iloc[0]
    readings[readings["uuid"] != lost].to_csv(folder / "timeseries.csv", index=False)

    settings = a1_settings(
        main_overrides={
            "data": {
                "root": str(folder),
                "static": "static.csv",
                "targets": "targets.csv",
                "timeseries": "timeseries.csv",
            }
        }
    )
    table = build_table(settings)

    assert lost in table.frame.index
    assert table.frame.loc[lost, N_OBS_COLUMN] == 0
    assert table.frame.loc[lost, "ts_S2_NDVI_p50"] != table.frame.loc[lost, "ts_S2_NDVI_p50"]  # NaN


def test_without_a_time_series_a1_says_so_instead_of_running_without_it(a1_settings):
    temporal = {"enabled": False, "time_column": "observation_date", "modality_prefix_map": {"s2": "S2_"}}
    settings = a1_settings(main_overrides={"temporal": temporal})
    with pytest.raises(ValueError, match="temporal.enabled"):
        build_table(settings)


def test_a_lab_column_that_was_not_carried_over_names_the_settings_to_change(a1_settings):
    settings = a1_settings(auxiliary={"targets": ["caco3_pct_total", "cn_ratio"]})
    with pytest.raises(ValueError, match="CARRY_LABEL_COLUMNS"):
        build_table(settings)


def test_a_region_column_in_neither_file_is_reported(a1_settings):
    settings = a1_settings(regions={"column": "province"})
    with pytest.raises(ValueError, match="province"):
        build_table(settings)


def test_without_a_region_column_there_is_none_in_the_table(a1_settings):
    table = build_table(a1_settings(regions={"column": None}))

    assert table.region_column is None
    assert "region" not in table.frame.columns
