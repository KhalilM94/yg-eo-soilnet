"""A1 on a static file whose time-series percentiles are already columns, with no time series.

The static demo (``configs/a1/a1_demo_static.yml``) is laid out like the real data: one static file
holding the covariates, the ``<variable>_qNN`` / ``_iqr`` percentiles (Sentinel-1 named polarisation
first, acquisition geometry included so it can be left out), management indicators, the lab values
and the coordinates.
"""

import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.features import (
    N_OBS_COLUMN,
    PRECOMPUTED_PERCENTILE,
    build_table,
    check_inputs,
    excluded_inputs,
    lab_columns,
)
from yg_eo_soilnet.a1.plans import RANDOM, write_plans
from yg_eo_soilnet.a1.preflight import INPUTS_FILE, NO_GROUP, run_preflight
from yg_eo_soilnet.a1.runner import run_plans
from yg_eo_soilnet.a1.settings import load_settings, settings_from_mapping
from yg_eo_soilnet.a1.stage2 import MODELS

A1_CONFIGS = "configs/a1"


@pytest.fixture(scope="module")
def static_demo(a1_settings_in, tmp_path_factory):
    """The static demo's settings, the table built from them, and its covariate groups."""
    folder = tmp_path_factory.mktemp("static_demo")
    path = a1_settings_in(folder)(static=True, as_file=True, output_dir=str(folder / "out"))
    a1 = load_a1_config(str(path))
    config = a1.settings.load_config()
    table = build_table(a1.settings, config)
    return {"a1": a1, "path": path, "config": config, "table": table, "folder": folder}


def _file(static_demo):
    return pd.read_csv(Path(static_demo["config"].DATA_FOLDER) / "static_with_percentiles.csv")


def _file_columns(static_demo):
    return _file(static_demo).columns


def _yaml(path):
    return yaml.safe_load(Path(path).read_text())


# --- no time series --------------------------------------------------------------------------------


def test_the_table_is_built_without_a_time_series(static_demo):
    table, settings = static_demo["table"], static_demo["a1"].settings
    assert settings.percentiles.source == "static" and "timeseries" not in settings.data
    assert N_OBS_COLUMN not in table.frame.columns
    assert table.percentile_columns and all(
        PRECOMPUTED_PERCENTILE.search(column) for column in table.percentile_columns
    )
    assert not any(PRECOMPUTED_PERCENTILE.search(column) for column in table.static_columns)
    assert len(table.frame) == len(_file(static_demo)) and table.frame[settings.target].notna().all()


def test_the_main_configuration_reads_no_time_series(static_demo):
    from yg_eo_soilnet.data_manager import DataManager

    import logging

    dataset = DataManager(static_demo["config"], logging.getLogger("test")).load_dataset()
    assert dataset.timeseries is None


# --- what is not an input --------------------------------------------------------------------------


def test_acquisition_geometry_is_in_the_file_but_never_an_input(static_demo, a1_settings_in, tmp_path, caplog):
    in_file = set(_file_columns(static_demo))
    inputs = set(static_demo["table"].feature_columns)
    for column in ("S2_SZA_q50", "S1_angle_asc_iqr"):
        assert column in in_file and column not in inputs
    assert not [column for column in inputs if column.startswith(("S2_SZA", "S1_angle"))]

    settings = load_settings(str(a1_settings_in(tmp_path)(static=True, as_file=True)))
    with caplog.at_level("INFO"):
        build_table(settings)
    assert "exclude_inputs removed 12 column(s)" in caplog.text  # 2 variables x 6 percentile columns


def test_no_lab_column_is_an_input(static_demo):
    inputs = set(static_demo["table"].feature_columns)
    labs = lab_columns(static_demo["a1"].settings, static_demo["config"])
    assert {"organic_matter_g_kg", "clay_pct", "caco3_pct_total", "c_e_c_meq_100g"} <= set(_file_columns(static_demo))
    assert {"organic_matter_g_kg", "clay_pct", "ph_water", "caco3_pct_total"} <= labs
    assert not inputs & labs
    assert not inputs & {"lat", "lon", "uuid", "region"}


def test_a_lab_column_or_an_excluded_column_slipped_back_in_is_refused(static_demo):
    table, settings, config = static_demo["table"], static_demo["a1"].settings, static_demo["config"]
    with pytest.raises(ValueError, match=r"lab column\(s\) are model inputs: \['clay_pct'\]"):
        check_inputs(replace(table, static_columns=(*table.static_columns, "clay_pct")), settings, config)
    with pytest.raises(ValueError, match="exclude_inputs prefix"):
        check_inputs(replace(table, percentile_columns=(*table.percentile_columns, "S2_SZA_q50")), settings, config)


def test_one_prefix_excludes_a_variable_whichever_way_its_percentiles_were_made():
    columns = ["S2_SZA_q50", "S1_angle_asc_iqr", "ts_S2_SZA_p50", "ts_S1_angle_asc_p10", "S2_B12_q50", "ts_S2_B4_p90"]
    assert excluded_inputs(columns, ["S2_SZA", "S1_angle"]) == columns[:4]
    assert excluded_inputs(columns, []) == []


# --- covariate groups -----------------------------------------------------------------------------


def test_every_sentinel1_percentile_is_a_satellite_input_and_management_has_its_own_group(static_demo):
    resolved = static_demo["a1"].resolve_groups(static_demo["table"])
    vh = [column for column in static_demo["table"].feature_columns if re.fullmatch(r"S1_VH_asc_q\d{2}", column)]
    assert len(vh) == 5 and set(vh) <= set(resolved.groups["satellite"])
    assert {"S1_VV_desc_q50", "S1_VH/VV_asc_iqr"} <= set(resolved.groups["satellite"])
    assert set(resolved.groups["management"]) == {"CUE_mean", "NDVI_CV", "Tillage_Freq_asc"}
    assert resolved.unassigned == ()


# --- the settings ---------------------------------------------------------------------------------


def test_static_without_any_qnn_column_is_refused(a1_settings_in, a1_demo_dir, tmp_path):
    data = {"root": str(a1_demo_dir), "static": "static.csv", "targets": "targets.csv", "timeseries": None}
    settings = load_settings(str(a1_settings_in(tmp_path)({"data": data}, static=True, as_file=True)))
    with pytest.raises(ValueError, match=r"source is static, but the static file has no <variable>_qNN column"):
        build_table(settings)


def test_an_unknown_percentile_source_is_refused(tmp_path):
    main = tmp_path / "main.yml"
    main.write_text("common: {}\n")
    with pytest.raises(ValueError, match=r"timeseries_percentiles.source must be one of \['compute', 'static'\]"):
        settings_from_mapping({"main_config": str(main), "timeseries_percentiles": {"source": "precomputed"}})


@pytest.mark.parametrize("name", ["a1.yml", "a1_demo_static.yml"])
def test_the_shipped_static_settings_load_without_a_time_series(name):
    settings = load_settings(f"{A1_CONFIGS}/{name}")
    assert settings.percentiles.source == "static"
    assert settings.exclude_inputs == ("S2_SZA", "S1_angle")
    assert "timeseries" not in settings.data and settings.sample_year.column is None
    main = _yaml(settings.main_config_path)["common"]
    assert main["temporal"]["enabled"] is False and main["data"]["timeseries"] is None


# --- preflight and a run ------------------------------------------------------------------------


def test_preflight_lists_every_input_with_its_group(static_demo, tmp_path):
    result = run_preflight(static_demo["a1"].settings, out_dir=str(tmp_path / "pre"))
    table = pd.read_csv(tmp_path / "pre" / INPUTS_FILE)
    assert len(table) == len(static_demo["table"].feature_columns) and result.unassigned == []
    assert set(table.loc[table["kind"] == "time-series percentile", "input"]) == set(
        static_demo["table"].percentile_columns
    )
    assert "Every input is in a covariate group" in (tmp_path / "pre" / "preflight.md").read_text()


def test_preflight_names_the_inputs_no_group_claims(static_demo, tmp_path):
    from yg_eo_soilnet.a1.config import load_covariate_groups

    groups_file = tmp_path / "groups.yml"
    raw = _yaml(f"{A1_CONFIGS}/covariate_groups_demo_static.yml")
    del raw["management"]
    groups_file.write_text(yaml.safe_dump(raw, sort_keys=False))
    result = run_preflight(
        static_demo["a1"].settings, out_dir=str(tmp_path / "pre"), groups=load_covariate_groups(str(groups_file))
    )
    assert sorted(result.unassigned) == ["CUE_mean", "NDVI_CV", "Tillage_Freq_asc"]
    table = pd.read_csv(tmp_path / "pre" / INPUTS_FILE).set_index("input")
    assert table.loc["CUE_mean", "group"] == NO_GROUP
    assert "3 input(s) are in no covariate group" in (tmp_path / "pre" / "preflight.md").read_text()


def test_preflight_fails_when_a_lab_column_would_be_an_input(a1_settings_in, tmp_path):
    spec = _yaml(f"{A1_CONFIGS}/demo_data_spec.yml")
    spec["LABEL_COLUMNS"] = [column for column in spec["LABEL_COLUMNS"] if column != "clay_pct"]
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    settings = load_settings(
        str(a1_settings_in(tmp_path)({"DATA_SPEC_PATH": str(spec_path)}, static=True, as_file=True))
    )
    with pytest.raises(ValueError, match=r"lab column\(s\) are model inputs: \['clay_pct'\]"):
        run_preflight(settings, out_dir=str(tmp_path / "pre"))


def test_a_quick_run_completes_on_the_static_demo(static_demo):
    a1, folder = static_demo["a1"], static_demo["folder"]
    write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[RANDOM])
    result = run_plans(a1, kinds=[RANDOM], dev_mode=True, tracking=False)
    assert len(result.computed) == len(MODELS)
    assert set(result.metrics["model"]) == set(MODELS)
