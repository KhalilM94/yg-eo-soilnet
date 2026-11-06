"""Holding out whole regions the data already names, through split.group.class_path.

The transfer experiments of Paper A1 train in some regions and test in others, so a region must
never be split between training and test. A spatial split with ColumnGroupStrategy is what keeps it
whole; the end-to-end test runs one on the A1 demo data and checks exactly that.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import Config
from yg_eo_soilnet.clustering_utils import (
    ColumnGroupStrategy,
    KMeansClusterStrategy,
    SpatialGridClusterStrategy,
)
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.datamodules.splitting import SPLIT_NAMES, UnifiedSplitter
from yg_eo_soilnet.models import ModelConfigFactory

CLASS_PATH = "yg_eo_soilnet.clustering_utils.ColumnGroupStrategy"


def _frame(regions):
    return pd.DataFrame(
        {
            "point_id": [f"p{i}" for i in range(len(regions))],
            "lat": np.linspace(30, 34, len(regions)),
            "lon": np.linspace(-9, -5, len(regions)),
            "region": regions,
        }
    )


# --- the strategy on its own ---------------------------------------------------------------


def test_each_name_gets_a_number_from_one_in_sorted_order():
    strategy = ColumnGroupStrategy()
    clustered = strategy.cluster(_frame(["Tensift", "Souss", "Tensift", "Bouregreg"]))

    assert clustered["cluster"].tolist() == [3, 2, 3, 1]
    assert clustered["cluster"].dtype.kind == "i"
    assert strategy.cluster_names_ == {1: "Bouregreg", 2: "Souss", 3: "Tensift"}
    assert strategy.cluster_ids_ == {"Bouregreg": 1, "Souss": 2, "Tensift": 3}


def test_the_same_names_in_another_order_get_the_same_numbers():
    first, second = ColumnGroupStrategy(), ColumnGroupStrategy()
    first.cluster(_frame(["b", "a", "c"]))
    second.cluster(_frame(["c", "c", "a", "b"]))
    assert first.cluster_ids_ == second.cluster_ids_


def test_a_point_without_a_region_is_left_out_and_the_rest_keep_every_column():
    clustered = ColumnGroupStrategy().cluster(_frame(["a", None, "b", np.nan]))

    assert clustered["point_id"].tolist() == ["p0", "p2"]
    assert {"point_id", "lat", "lon", "region", "cluster"} <= set(clustered.columns)


def test_numeric_group_codes_work_too():
    strategy = ColumnGroupStrategy(column="basin_code")
    frame = _frame([20, 3, 20]).rename(columns={"region": "basin_code"})

    clustered = strategy.cluster(frame)

    assert strategy.cluster_names_ == {1: 3, 2: 20}
    assert clustered["cluster"].tolist() == [2, 1, 2]


def test_a_column_that_is_not_there_names_the_setting_to_change():
    with pytest.raises(KeyError, match=r"'basin'.*split.group.params.column"):
        ColumnGroupStrategy(column="basin").cluster(_frame(["a", "b"]))


def test_a_column_with_no_value_at_all_is_refused():
    with pytest.raises(ValueError, match="No point has a value"):
        ColumnGroupStrategy().cluster(_frame([None, None]))


def test_the_strategy_asks_for_its_column_and_the_others_for_nothing():
    assert ColumnGroupStrategy.extra_columns({}) == ("region",)
    assert ColumnGroupStrategy.extra_columns({"column": "SousBassin"}) == ("SousBassin",)
    assert KMeansClusterStrategy.extra_columns({"n_clusters": 12}) == ()
    assert SpatialGridClusterStrategy.extra_columns({"cell_size_m": 5000}) == ()


def test_it_is_built_from_split_group_like_the_other_strategies():
    spec = {"class_path": CLASS_PATH, "params": {"column": "region"}}
    strategy = ModelConfigFactory(spec, 7).load_splitter_from_config()

    assert isinstance(strategy, ColumnGroupStrategy)
    assert strategy.column == "region"
    assert strategy.random_state == 7  # handed to every strategy; this one has no use for it


# --- through the splitter --------------------------------------------------------------------


def _splitter(column="region"):
    config = SimpleNamespace(
        SPLIT_HOLDOUT_STRATEGY="spatial_group",
        SPLIT_TEST_SIZE=0.25,
        SPLIT_VAL_SIZE=0.25,
        SPLIT_SEED=3,
        SPLIT_GROUP_STRATEGY={"class_path": CLASS_PATH, "params": {"column": column}},
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
    )
    return UnifiedSplitter(config, logging.getLogger("test"))


def _population(n=200, n_regions=8, missing=0, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "lat": rng.uniform(30, 34, n),
            "lon": rng.uniform(-9, -5, n),
            "region": rng.choice([f"R{i}" for i in range(n_regions)], size=n),
        },
        index=pd.Index([f"p{i:03d}" for i in range(n)], name="point_id"),
    )
    if missing:
        frame.iloc[:missing, frame.columns.get_loc("region")] = None
    return frame


def _splits_per_region(plan, regions):
    labels = plan.assignments.reindex(regions.index)
    return pd.DataFrame({"region": regions, "split": labels}).dropna().groupby("region")["split"].nunique()


def test_every_region_falls_wholly_in_one_split():
    population = _population()
    plan = _splitter().build_plan(population.index, coordinates=population)

    assert (_splits_per_region(plan, population["region"]) == 1).all()
    assert plan.counts()["test"] > 0 and plan.counts()["val"] > 0


def test_points_without_a_region_stay_in_training(caplog):
    population = _population(missing=15)
    with caplog.at_level(logging.WARNING):
        plan = _splitter().build_plan(population.index, coordinates=population)

    assert set(plan.assignments.iloc[:15]) == {"train"}
    assert "missing coordinates, or no group" in caplog.text


def test_the_plan_keeps_the_position_but_not_the_group_column():
    population = _population()
    plan = _splitter().build_plan(population.index, coordinates=population)

    assert list(plan.coordinates.columns) == ["lat", "lon"]
    assert list(plan.to_frame().columns[:4]) == ["point_id", "split", "lat", "lon"]
    assert "region" not in plan.to_frame().columns


def test_the_plan_records_each_points_group_number():
    population = _population()
    splitter = _splitter()
    plan = splitter.build_plan(population.index, coordinates=population)

    names = plan.clusters.map(splitter.cluster_strategy_.cluster_names_)
    pd.testing.assert_series_equal(names, population["region"].reindex(plan.clusters.index), check_names=False)


def test_a_kmeans_split_still_hands_its_strategy_only_the_coordinates():
    population = _population()[["lat", "lon"]]
    config = SimpleNamespace(
        SPLIT_HOLDOUT_STRATEGY="spatial_group",
        SPLIT_TEST_SIZE=0.2,
        SPLIT_VAL_SIZE=0.2,
        SPLIT_SEED=3,
        SPLIT_GROUP_STRATEGY={
            "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy",
            "params": {"n_clusters": 5},
        },
    )
    splitter = UnifiedSplitter(config, logging.getLogger("test"))
    splitter.build_plan(population.index, coordinates=population)

    assert set(splitter.clustered_frame_.columns) == {"point_id", "lat", "lon", "cluster"}


# --- end to end on the A1 demo data ------------------------------------------------------------


def _region_split_settings(a1_settings, column="region"):
    split = {
        "strategy": "spatial_group",
        "test_size": 0.15,
        "val_size": 0.15,
        "seed": 42,
        "population_policy": "intersect",
        "min_population_ratio": 0.5,
        "plan_path": None,
        "group": {"class_path": CLASS_PATH, "params": {"column": column}},
    }
    return a1_settings(main_overrides={"split": split})


def _demo_plan(settings):
    config = Config(config_path=settings.main_config_path)
    logger = logging.getLogger("test")
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=["sklearn"])
    return provider.plan()


def test_a_spatial_group_run_on_the_demo_holds_out_whole_regions(a1_settings, a1_demo_dir):
    plan = _demo_plan(_region_split_settings(a1_settings))
    regions = pd.read_csv(a1_demo_dir / "targets.csv").set_index("uuid")["region"]

    per_region = _splits_per_region(plan, regions)
    assert len(per_region) == 4
    assert (per_region == 1).all()
    # Test and validation are each made of whole regions, and training keeps the rest.
    region_split = pd.DataFrame({"region": regions, "split": plan.assignments.reindex(regions.index)})
    held = region_split.groupby("split")["region"].nunique()
    assert set(held.index) == set(SPLIT_NAMES)
    assert held["test"] >= 1 and held["val"] >= 1 and held["train"] >= 1


def test_the_demo_plan_is_saved_with_positions_and_group_numbers_only(a1_settings):
    frame = _demo_plan(_region_split_settings(a1_settings)).to_frame()

    assert {"point_id", "split", "lat", "lon", "cluster"} <= set(frame.columns)
    assert "region" not in frame.columns
    assert sorted(frame["cluster"].unique()) == [1, 2, 3, 4]


def test_grouping_by_a_column_the_data_does_not_have_is_reported(a1_settings):
    with pytest.raises(ValueError, match=r"groups points by 'basin'.*split.group.params.column"):
        _demo_plan(_region_split_settings(a1_settings, column="basin"))
