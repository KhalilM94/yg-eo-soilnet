"""Loading the whole A1 configuration: settings, covariate groups, the A1 data spec, the git commit.

A covariate group that names a column the data does not have, or that shares a column with another
group, would silently change which inputs a group-by-group comparison reads. Each of those stops the
load, naming the group and the column.
"""

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from yg_eo_soilnet.a1 import config as a1_config
from yg_eo_soilnet.a1.config import (
    COMBINED_GROUP,
    CovariateGroup,
    covariate_groups_from_mapping,
    git_state,
    load_a1_config,
    load_covariate_groups,
    stage1_prediction_column,
)
from yg_eo_soilnet.data_manager import DataManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
A1_CONFIGS = PROJECT_ROOT / "configs" / "a1"
LEGACY_SOIL = (
    "bd_median_0_30cm",
    "clay_median_0_30cm",
    "ph_median_0_30cm",
    "sand_median_0_30cm",
    "soc_median_0_30cm",
    "som_median_0_30cm",
    "swc_median_0_30cm",
    "texture_20cm",
)
AUXILIARY = ("caco3_pct_total", "c_e_c_meq_100g")


def _demo_groups():
    return yaml.safe_load((A1_CONFIGS / "covariate_groups_demo.yml").read_text())


def _head_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# --- the shipped files ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a1.yml", "a1_demo.yml"])
def test_the_shipped_configurations_load_and_record_the_commit(name):
    a1 = load_a1_config(str(A1_CONFIGS / name))

    assert a1.git_commit == _head_commit()
    assert isinstance(a1.git_dirty, bool)
    expected = ("satellite", "terrain", "climate", "legacy_soil", "auxiliary")
    if name == "a1.yml":  # the real data adds the land-management indicators
        expected = ("satellite", "terrain", "climate", "legacy_soil", "management", "auxiliary")
    assert a1.groups.names == expected
    assert a1.settings.stage1.estimator == "TabICL"
    assert a1.settings.stage2.estimator == "XGBoost"
    assert a1.settings.inner_cv_folds == 5
    assert a1.settings.min_cell_size == 30
    assert a1.settings.bootstrap.block_size_km == 10
    assert a1.resolved is None  # the data was not read


def test_the_real_configuration_reads_its_data_through_the_a1_main_config():
    a1 = load_a1_config(str(A1_CONFIGS / "a1.yml"))
    config = a1.settings.load_config()

    assert Path(a1.settings.main_config_path) == A1_CONFIGS / "main_config.yml"
    assert Path(config.data_spec_path).resolve() == A1_CONFIGS / "data_spec.yml"
    assert "clay_median_0_30cm" not in config.LABEL_COLUMNS
    assert config.MLFLOW_EXPERIMENT_NAME == "Paper_A1"


def test_the_real_legacy_soil_group_holds_the_soil_map_layers():
    groups = load_covariate_groups(str(A1_CONFIGS / "covariate_groups.yml"))
    legacy = next(group for group in groups.groups if group.name == "legacy_soil")
    # som_median_0_30cm is not in the real static file (the same SOC map, x 17.24, where it is).
    assert legacy.columns == tuple(column for column in LEGACY_SOIL if column != "som_median_0_30cm")


def test_the_real_groups_name_s1_by_polarisation_and_have_a_management_group():
    groups = {group.name: group for group in load_covariate_groups(str(A1_CONFIGS / "covariate_groups.yml")).groups}
    satellite = groups["satellite"]
    assert {"S1_VH_", "S1_VV_", "S1_VH/VV_"} <= set(satellite.prefixes)
    assert not {"S1_asc_", "S1_desc_"} & set(satellite.prefixes)
    assert "ts_n_obs" not in satellite.columns
    assert "som_median_0_30cm" not in groups["legacy_soil"].columns
    assert groups["management"].columns[:2] == ("CUE_mean", "CUE_cv") and len(groups["management"].columns) == 17


def test_loading_on_the_demo_data_puts_every_input_in_exactly_one_group(a1_settings):
    a1 = load_a1_config(str(a1_settings(as_file=True)), check_data=True)
    resolved = a1.resolved

    assert resolved.unassigned == ()
    combined = resolved.combined
    assert len(combined) == len(set(combined))
    assert resolved.columns(COMBINED_GROUP) == combined
    assert resolved.groups["legacy_soil"] == ("clay_median_0_30cm",)
    assert resolved.groups["auxiliary"] == tuple(stage1_prediction_column(target) for target in AUXILIARY)
    # The prefixes reach the time-series percentiles of the variables they name.
    assert "ts_S2_NDVI_p50" in resolved.groups["satellite"]
    assert "ts_CLIM_precip_mm_p90" in resolved.groups["climate"]


def test_describe_records_the_commit_and_the_group_sizes(a1_settings):
    a1 = load_a1_config(str(a1_settings(as_file=True)), check_data=True)
    described = a1.describe()

    assert described["a1_git_commit"] == _head_commit()
    assert described["a1_n_auxiliary"] == 2
    assert described["a1_n_combined"] == len(a1.resolved.combined)
    assert described["a1_stage1_estimator"] == "TabICL"


def test_resolving_against_a_table_matches_the_data_check(a1_settings):
    from yg_eo_soilnet.a1.features import build_table

    a1 = load_a1_config(str(a1_settings(as_file=True)), check_data=True)
    assert a1.resolve_groups(build_table(a1.settings)) == a1.resolved


# --- covariate-group validation --------------------------------------------------------------


def _load_with_groups(a1_settings, groups):
    return load_a1_config(str(a1_settings(as_file=True, groups=groups)), check_data=True)


def test_a_column_the_data_does_not_have_is_refused(a1_settings):
    groups = _demo_groups()
    groups["terrain"]["columns"].append("curvature")
    with pytest.raises(ValueError, match="'terrain': column 'curvature' is not one of the inputs"):
        _load_with_groups(a1_settings, groups)


def test_a_lab_column_listed_as_an_input_says_what_it_is(a1_settings):
    groups = _demo_groups()
    groups["legacy_soil"]["columns"].append("caco3_pct_total")
    with pytest.raises(ValueError, match="'caco3_pct_total' is a lab column, id or coordinate, not an input"):
        _load_with_groups(a1_settings, groups)


def test_a_prefix_matching_nothing_is_refused(a1_settings):
    groups = _demo_groups()
    groups["satellite"]["prefixes"].append("S1_asc_")
    with pytest.raises(ValueError, match="'satellite': prefix 'S1_asc_' matches no input"):
        _load_with_groups(a1_settings, groups)


def test_a_column_named_in_two_groups_is_refused(a1_settings):
    groups = _demo_groups()
    groups["climate"]["columns"].append("elevation")
    with pytest.raises(ValueError, match="'elevation' is in two covariate groups, 'terrain' and 'climate'"):
        _load_with_groups(a1_settings, groups)


def test_a_column_claimed_by_a_name_and_by_another_groups_prefix_is_refused(a1_settings):
    groups = _demo_groups()
    groups["legacy_soil"]["columns"].append("s2_b2_barest")  # also matched by satellite's s2_ prefix
    with pytest.raises(ValueError, match="'s2_b2_barest' is in two covariate groups"):
        _load_with_groups(a1_settings, groups)


@pytest.mark.parametrize("empty", [None, {}, {"columns": []}, {"columns": [], "prefixes": []}])
def test_an_empty_group_is_refused(empty):
    with pytest.raises(ValueError, match="'terrain' is empty"):
        covariate_groups_from_mapping({"terrain": empty})


def test_a_group_whose_columns_all_belong_to_an_earlier_group_is_refused_not_left_empty():
    groups = covariate_groups_from_mapping({"satellite": {"prefixes": ["S2_"]}, "more": {"prefixes": ["S2_B"]}})
    with pytest.raises(ValueError, match="in two covariate groups, 'satellite' and 'more'"):
        groups.resolve(["S2_B4", "elevation"], [])


def test_an_unknown_key_in_a_group_is_refused():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['column'\]"):
        covariate_groups_from_mapping({"terrain": {"column": ["slope"]}})


def test_combined_cannot_be_defined_by_hand():
    with pytest.raises(ValueError, match="'combined' is the union"):
        covariate_groups_from_mapping({COMBINED_GROUP: {"columns": ["slope"]}})


def test_a_column_listed_twice_in_one_group_is_refused():
    with pytest.raises(ValueError, match=r"lists \['slope'\] twice"):
        covariate_groups_from_mapping({"terrain": {"columns": ["slope", "slope"]}})


def test_a_file_with_no_groups_is_refused():
    with pytest.raises(ValueError, match="at least one group"):
        covariate_groups_from_mapping({})


def test_the_auxiliary_group_may_only_name_what_stage_1_predicts(a1_settings):
    groups = _demo_groups()
    groups["auxiliary"]["columns"].append("ph_water")
    with pytest.raises(ValueError, match=r"names \['ph_water'\], which stage 1 does not predict"):
        _load_with_groups(a1_settings, groups)


def test_the_auxiliary_group_takes_no_prefixes():
    groups = covariate_groups_from_mapping({"auxiliary": {"prefixes": ["caco3"]}})
    with pytest.raises(ValueError, match="takes no prefixes"):
        groups.resolve(["slope"], ["caco3_pct_total"])


def test_an_input_no_group_claims_is_warned_about_not_refused(a1_settings, caplog):
    groups = _demo_groups()
    groups["terrain"]["columns"].remove("twi")
    with caplog.at_level(logging.WARNING):
        a1 = _load_with_groups(a1_settings, groups)

    assert a1.resolved.unassigned == ("twi",)
    assert "twi" not in a1.resolved.combined
    assert "1 input(s) are in no covariate group" in caplog.text


def test_a_prefix_also_matches_a_percentile_of_a_variable_with_that_prefix():
    group = CovariateGroup("climate", prefixes=("CLIM_",))
    assert group.matches("CLIM_ETa")
    assert group.matches("ts_CLIM_ETa_p50")
    assert not group.matches("ts_S2_B4_p50")
    assert not group.matches("aridity_index")


def test_asking_for_a_group_that_does_not_exist_names_the_ones_that_do():
    resolved = covariate_groups_from_mapping({"terrain": {"columns": ["slope"]}}).resolve(["slope"], [])
    with pytest.raises(KeyError, match="'terrain', 'combined'"):
        resolved.columns("satellite")


# --- checks against the main configuration ------------------------------------------------------


def test_a_data_block_that_disagrees_with_the_main_config_is_refused(a1_settings):
    path = a1_settings(as_file=True, data={"root": "/somewhere/else", "static": "static.csv"})
    with pytest.raises(ValueError, match=r"data.root .* '/somewhere/else' but the main configuration"):
        load_a1_config(str(path))


@pytest.mark.parametrize("stage, missing", [("stage1", "TabICL"), ("stage2", "XGBoost")])
def test_an_estimator_missing_from_the_model_list_is_refused(a1_settings, stage, missing, monkeypatch):
    from config import Config

    original = Config.__init__

    def without_one_estimator(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.MODEL_REGISTRY = {name: entry for name, entry in self.MODEL_REGISTRY.items() if name != missing}

    monkeypatch.setattr(Config, "__init__", without_one_estimator)
    with pytest.raises(ValueError, match=rf"{stage}.estimator '{missing}' is not an entry"):
        load_a1_config(str(a1_settings(as_file=True)))


def test_an_xgboost_stage_1_loads(a1_settings):
    a1 = load_a1_config(str(a1_settings(as_file=True, stage1={"estimator": "XGBoost"})))
    assert a1.settings.stage1.estimator == "XGBoost"


def test_a_configuration_without_a_covariate_groups_file_is_refused(a1_settings):
    path = a1_settings(as_file=True)
    raw = yaml.safe_load(path.read_text())
    del raw["covariate_groups"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="covariate_groups"):
        load_a1_config(str(path))


# --- the git commit -----------------------------------------------------------------------------


def test_outside_a_git_checkout_the_commit_is_unknown_and_the_run_says_so(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        commit, dirty = git_state(str(tmp_path))
    assert (commit, dirty) == (None, None)
    assert "Could not read the git commit" in caplog.text


def test_without_git_installed_loading_still_works(a1_settings, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(a1_config.subprocess, "run", no_git)
    a1 = load_a1_config(str(a1_settings(as_file=True)))
    assert a1.git_commit is None
    assert a1.describe()["a1_git_commit"] == "unknown"


# --- the A1 data spec ---------------------------------------------------------------------------


def _spec(path):
    return yaml.safe_load(Path(path).read_text())


def test_the_a1_data_spec_makes_the_soil_map_layers_inputs():
    labels = set(_spec(A1_CONFIGS / "data_spec.yml")["LABEL_COLUMNS"])

    assert not labels & set(LEGACY_SOIL)
    # The lab values themselves, and the earlier predictions, are still never inputs.
    assert {"organic_matter_g_kg", *AUXILIARY, "caco3_pct_total_baseline", "c_e_c_meq_100g_baseline"} <= labels


def test_the_a1_data_spec_makes_landform_class_a_categorical_terrain_input():
    spec = _spec(A1_CONFIGS / "data_spec.yml")
    assert "landform_class" not in spec["LABEL_COLUMNS"]
    assert "landform_class" in spec["CATEGORICAL_FEATURES"]
    groups = yaml.safe_load((A1_CONFIGS / "covariate_groups.yml").read_text())
    assert "landform_class" in groups["terrain"]["columns"]


def test_the_a1_data_spec_differs_from_the_projects_only_in_those_layers():
    project, a1 = _spec(PROJECT_ROOT / "configs" / "data_spec.yml"), _spec(A1_CONFIGS / "data_spec.yml")

    moved = set(project["LABEL_COLUMNS"]) - set(a1["LABEL_COLUMNS"])
    assert moved == set(LEGACY_SOIL) | {"silt_median_0_30cm", "landform_class"}
    assert set(a1["LABEL_COLUMNS"]) <= set(project["LABEL_COLUMNS"])
    assert a1["CATEGORICAL_FEATURES"] == ["texture_20cm", "landform_class"]
    for key in set(project) - {"LABEL_COLUMNS", "CATEGORICAL_FEATURES"}:
        assert a1[key] == project[key], key


def test_the_data_manager_keeps_the_layers_under_the_a1_spec_and_drops_them_under_the_projects():
    frame = pd.DataFrame(columns=["uuid", "lat", "lon", "elevation", "organic_matter_g_kg", *LEGACY_SOIL])

    def inputs(spec_path):
        spec = _spec(spec_path)
        config = SimpleNamespace(
            POINT_ID_COLUMN="uuid",
            TARGET_COLUMNS=["organic_matter_g_kg"],
            LABEL_COLUMNS=spec["LABEL_COLUMNS"],
            ELIMINATED_FEATURES=spec["ELIMINATED_FEATURES"],
        )
        return set(DataManager(config, logging.getLogger("test")).filter_schema(frame).columns)

    assert inputs(A1_CONFIGS / "data_spec.yml") == {"elevation", *LEGACY_SOIL}
    assert inputs(PROJECT_ROOT / "configs" / "data_spec.yml") == {"elevation"}


def test_the_projects_own_data_spec_is_untouched():
    has_main = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "main"], cwd=PROJECT_ROOT, capture_output=True
    )
    if has_main.returncode != 0:
        pytest.skip("no local main branch to compare with (a shallow CI checkout)")
    changed = subprocess.run(
        ["git", "diff", "--name-only", "main", "--", "configs/data_spec.yml"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert changed == ""
