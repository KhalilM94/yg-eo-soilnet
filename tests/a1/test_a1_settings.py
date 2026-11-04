"""Reading and checking the A1 settings file.

A wrong setting should stop the run before any model is trained, and say which setting it was.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml

from yg_eo_soilnet.a1.settings import (
    FAO_CARBONATE_EDGES,
    FAO_CARBONATE_LABELS,
    A1Settings,
    load_settings,
    settings_from_mapping,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def main_config(tmp_path):
    """Any existing file will do: the settings only check that the main configuration exists."""
    path = tmp_path / "main.yml"
    path.write_text("common: {}\n")
    return path


def _settings(main_config, **raw):
    return settings_from_mapping({"main_config": str(main_config), **raw})


@pytest.mark.parametrize("name", ["a1_demo.yml", "a1.yml"])
def test_the_shipped_settings_files_load(name):
    settings = load_settings(str(PROJECT_ROOT / "configs" / "a1" / name))

    assert Path(settings.main_config_path).is_file()
    assert settings.som_target == "organic_matter_g_kg"
    assert settings.auxiliary.targets == ("caco3_pct_total", "c_e_c_meq_100g")
    assert settings.carbonate.edges == FAO_CARBONATE_EDGES
    assert settings.carbonate.labels == FAO_CARBONATE_LABELS


def test_the_real_settings_leave_acquisition_geometry_out_of_the_percentiles():
    settings = load_settings(str(PROJECT_ROOT / "configs" / "a1" / "a1.yml"))
    assert {"S2_SZA", "S1_asc_angle", "S1_desc_angle"} <= set(settings.percentiles.exclude)


def test_a_relative_main_config_is_taken_from_the_settings_files_own_folder(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "main.yml").write_text("common: {}\n")
    (folder / "a1.yml").write_text(yaml.safe_dump({"main_config": "main.yml"}))
    # Run from elsewhere, so a path read from the working directory would not be found.
    monkeypatch.chdir(tmp_path)

    settings = load_settings(str(folder / "a1.yml"))

    assert Path(settings.main_config_path) == folder / "main.yml"


def test_a_file_with_only_main_config_takes_every_default(main_config):
    settings = _settings(main_config)

    assert settings.percentiles.quantiles == (10.0, 25.0, 50.0, 75.0, 90.0)
    assert settings.auxiliary.model == "TabICL"
    assert settings.auxiliary.n_folds == 5
    assert settings.regions.source is None
    assert settings.sample_year.column is None
    assert settings.preflight.min_cell_points == 30
    assert settings.preflight.prediction_columns == {
        "caco3_pct_total": "caco3_pct_total__TabICL",
        "c_e_c_meq_100g": "c_e_c_meq_100g__TabICL",
    }
    assert settings.tracking.experiment_name == "Paper_A1"


def test_settings_cannot_be_changed_once_read(main_config):
    settings = _settings(main_config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.som_target = "clay_pct"  # type: ignore[misc]


def test_a_missing_settings_file_is_reported():
    with pytest.raises(FileNotFoundError, match="A1 settings file not found"):
        load_settings("no/such/a1.yml")


def test_a_main_config_that_does_not_exist_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="main_config"):
        settings_from_mapping({"main_config": "nowhere.yml"}, base_dir=str(tmp_path))


def test_main_config_is_required():
    with pytest.raises(ValueError, match="main_config"):
        settings_from_mapping({})


@pytest.mark.parametrize(
    "raw, where",
    [
        ({"som_targte": "x"}, "the A1 settings"),
        ({"carbonate": {"edge": [2, 10]}}, "carbonate"),
        ({"timeseries_percentiles": {"quantile": [50]}}, "timeseries_percentiles"),
    ],
)
def test_a_misspelt_setting_is_refused_rather_than_ignored(main_config, raw, where):
    with pytest.raises(ValueError, match=f"Unknown setting.*{where}"):
        _settings(main_config, **raw)


def test_feeding_som_back_in_as_its_own_input_is_refused(main_config):
    with pytest.raises(ValueError, match="must not include the SOM target"):
        _settings(main_config, auxiliary={"targets": ["organic_matter_g_kg", "caco3_pct_total"]})


@pytest.mark.parametrize("quantiles", [[0, 50], [50, 100], [-5], [50, 50], []])
def test_percentiles_must_be_distinct_and_strictly_inside_0_to_100(main_config, quantiles):
    with pytest.raises(ValueError, match="quantiles"):
        _settings(main_config, timeseries_percentiles={"quantiles": quantiles})


@pytest.mark.parametrize("edges", [[10, 2, 25], [2, 2, 25], []])
def test_carbonate_edges_must_rise(main_config, edges):
    with pytest.raises(ValueError, match="carbonate.edges"):
        _settings(main_config, carbonate={"edges": edges, "labels": ["a"] * (len(edges) + 1)})


def test_every_carbonate_class_needs_a_name(main_config):
    with pytest.raises(ValueError, match="one name per class"):
        _settings(main_config, carbonate={"edges": [2, 10, 25], "labels": ["low", "high"]})


def test_regions_from_a_column_need_only_its_name(main_config):
    regions = _settings(main_config, regions={"source": "column", "column": "province"}).regions
    assert (regions.source, regions.column) == ("column", "province")


def test_regions_from_polygons_need_the_file_and_the_name_field(main_config):
    regions = _settings(
        main_config, regions={"source": "polygons", "polygons_path": "r.geojson", "name_field": "name"}
    ).regions
    assert (regions.polygons_path, regions.name_field) == ("r.geojson", "name")


@pytest.mark.parametrize(
    "regions, message",
    [
        ({"source": "column"}, "must name the column"),
        ({"source": "polygons", "polygons_path": "r.geojson"}, "name_field"),
        ({"source": "polygons", "name_field": "name"}, "polygons_path"),
        ({"source": "column", "column": "region", "polygons_path": "r.geojson"}, "belongs to source 'polygons'"),
        (
            {"source": "polygons", "polygons_path": "r.geojson", "name_field": "name", "column": "region"},
            "belongs to source 'column'",
        ),
        ({"column": "region"}, "regions.source is not"),
        ({"source": "shapefile"}, "must be one of"),
    ],
)
def test_a_region_source_that_is_half_set_up_or_mixed_is_refused(main_config, regions, message):
    with pytest.raises(ValueError, match=message):
        _settings(main_config, regions=regions)


def test_a_sampling_year_file_needs_the_column_to_read(main_config):
    with pytest.raises(ValueError, match="sample_year.column"):
        _settings(main_config, sample_year={"file": "analyses.csv"})


def test_the_preflight_thresholds_are_checked(main_config):
    with pytest.raises(ValueError, match="overfit_r2_gap"):
        _settings(main_config, preflight={"overfit_r2_gap": -0.1})
    with pytest.raises(ValueError, match="min_cell_points"):
        _settings(main_config, preflight={"min_cell_points": 0})
    with pytest.raises(ValueError, match="prediction_columns"):
        _settings(main_config, preflight={"prediction_columns": ["caco3_pct_total__TabICL"]})


def test_the_real_settings_read_the_baseline_columns_as_the_earlier_predictions():
    settings = load_settings(str(PROJECT_ROOT / "configs" / "a1" / "a1.yml"))
    assert settings.preflight.prediction_columns == {
        "caco3_pct_total": "caco3_pct_total_baseline",
        "c_e_c_meq_100g": "c_e_c_meq_100g_baseline",
    }
    assert (settings.sample_year.column, settings.sample_year.file) == (
        "date",
        "Morocco_soil_analysis_filtered_uuid_roadsafe.csv",
    )


def test_fewer_than_two_folds_is_refused(main_config):
    with pytest.raises(ValueError, match="n_folds"):
        _settings(main_config, auxiliary={"n_folds": 1})


def test_at_least_one_som_model_is_needed(main_config):
    with pytest.raises(ValueError, match="models"):
        _settings(main_config, models=[])


def test_a_single_name_where_a_list_is_expected_is_refused_rather_than_split_into_letters(main_config):
    with pytest.raises(ValueError, match="must be a list"):
        _settings(main_config, models="XGBoost")


def test_the_main_configuration_is_read_through_the_projects_own_config(a1_settings):
    config = a1_settings().load_config()
    assert config.POINT_ID_COLUMN == "uuid"
    assert "caco3_pct_total" in config.LABEL_COLUMNS


def test_a1_settings_is_constructible_directly_with_its_defaults(main_config):
    settings = A1Settings(main_config_path=str(main_config))
    assert settings.models == ("XGBoost", "TabICL")
