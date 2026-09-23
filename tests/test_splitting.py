"""The split, and the sklearn datamodule that consumes it.

The train/val/test split is decided once and shared by every training family.

Before this existed, sklearn carved 70/30 off the tabular frame while the Lightning datamodules
carved 64/16/20 off a bundle rebuilt from the same CSVs. The holdouts overlapped, so a Lightning
*test* point was usually an sklearn *training* point - and both families publish the result as
`rmse_test` on one leaderboard axis. `test_the_two_families_hold_out_the_same_points` is the test
that would have caught that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.datamodules.splitting import SplitPlan, UnifiedSplitter

SPLITTER_MODULE = "yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter"


def _ids(n: int) -> pd.Index:
    return pd.Index([f"p{i:04d}" for i in range(n)])


def _coordinates(ids: pd.Index, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"lat": rng.uniform(30.0, 36.0, len(ids)), "lon": rng.uniform(-9.0, -2.0, len(ids))},
        index=ids,
    )


@pytest.fixture
def split_config(toy_config):
    toy_config.SPLIT_HOLDOUT_STRATEGY = "random"
    toy_config.SPLIT_TEST_SIZE = 0.2
    toy_config.SPLIT_VAL_SIZE = 0.16
    toy_config.SPLIT_SEED = 42
    toy_config.SPLIT_POPULATION_POLICY = "intersect"
    return toy_config


# --- ratios and determinism ---------------------------------------------------------------


def test_sizes_are_fractions_of_the_whole_population_not_of_the_remainder(split_config, logger):
    """Asking for 0.2 must give 20%.

    The old code applied test_size then val_size sequentially, which is why the registry carried
    the note "without this, train was 0.7*0.7=0.49". Requesting a number and silently getting a
    smaller one is the footgun this convention removes.
    """
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(1000))

    counts = plan.counts()
    assert counts == {"train": 640, "val": 160, "test": 200}


def test_the_same_seed_and_population_give_the_same_assignment(split_config, logger):
    first = UnifiedSplitter(split_config, logger).build_plan(_ids(500))
    second = UnifiedSplitter(split_config, logger).build_plan(_ids(500))

    assert first.assignments.equals(second.assignments)


def test_a_different_seed_moves_the_test_set(split_config, logger):
    first = UnifiedSplitter(split_config, logger).build_plan(_ids(500))
    split_config.SPLIT_SEED = 7
    second = UnifiedSplitter(split_config, logger).build_plan(_ids(500))

    assert set(first.point_ids_for("test")) != set(second.point_ids_for("test"))


def test_the_three_splits_partition_the_population_without_overlap(split_config, logger):
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(300))

    train, val, test = (set(plan.point_ids_for(name)) for name in ("train", "val", "test"))
    assert train | val | test == set(_ids(300))
    assert not (train & val) and not (train & test) and not (val & test)


def test_a_zero_val_size_is_allowed_rather_than_raising(split_config, logger):
    split_config.SPLIT_VAL_SIZE = 0.0

    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(100))

    assert plan.counts()["val"] == 0
    assert plan.counts()["train"] == 80


def test_sizes_that_leave_no_training_set_are_refused(split_config, logger):
    split_config.SPLIT_TEST_SIZE = 0.6
    split_config.SPLIT_VAL_SIZE = 0.5

    with pytest.raises(ValueError, match="must leave a training set"):
        UnifiedSplitter(split_config, logger)


def test_duplicate_point_ids_are_refused(split_config, logger):
    with pytest.raises(ValueError, match="duplicate point id"):
        UnifiedSplitter(split_config, logger).build_plan(pd.Index(["a", "b", "a"]))


def test_an_unknown_strategy_is_refused(split_config, logger):
    split_config.SPLIT_HOLDOUT_STRATEGY = "kfold"

    with pytest.raises(ValueError, match="split.strategy must be one of"):
        UnifiedSplitter(split_config, logger)


# --- resolving the plan against a family's own ordering -----------------------------------


def test_indices_resolve_against_each_family_own_row_order(split_config, logger):
    """Point id, not row position, is what makes the plan shareable.

    The families hold different subsets in different orders, so a positional index into one
    family's arrays means nothing to the other.
    """
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(200))

    shuffled_subset = list(_ids(200)[::3][::-1])
    train_idx, val_idx, test_idx = plan.split_indices(shuffled_subset)

    resolved_test = {shuffled_subset[i] for i in test_idx}
    assert resolved_test == set(plan.point_ids_for("test")) & set(shuffled_subset)
    assert train_idx.size + val_idx.size + test_idx.size == len(shuffled_subset)


def test_points_outside_the_population_land_in_no_split(split_config, logger):
    """That is population_policy=intersect doing its job, not a bug."""
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(100))

    with_strangers = list(_ids(100)) + ["unknown-1", "unknown-2"]
    train_idx, val_idx, test_idx = plan.split_indices(with_strangers)

    assert train_idx.size + val_idx.size + test_idx.size == 100


# --- the spatial-group strategy -----------------------------------------------------------


def test_spatial_group_never_lets_a_cluster_straddle_two_splits(split_config, logger, monkeypatch):
    """Splitting inside a cluster would defeat the point of blocking on it."""
    monkeypatch.setattr("mlflow.log_artifact", lambda *args, **kwargs: None)
    split_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    split_config.SPLIT_GROUP_STRATEGY = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy",
        "params": {"n_clusters": 20},
    }
    ids = _ids(1000)

    plan = UnifiedSplitter(split_config, logger).build_plan(ids, coordinates=_coordinates(ids))

    frame = plan.to_frame()
    assert frame.groupby("cluster")["split"].nunique().max() == 1
    assert plan.counts()["test"] > 0


def test_spatial_group_without_coordinates_is_refused(split_config, logger):
    split_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    split_config.SPLIT_GROUP_STRATEGY = {"class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy"}

    with pytest.raises(ValueError, match="needs coordinates"):
        UnifiedSplitter(split_config, logger).build_plan(_ids(50))


# --- persistence --------------------------------------------------------------------------


def test_the_plan_round_trips_through_its_frame(split_config, logger):
    """The artifact has to be re-keyable: the old data_splits parquet carried no join key at all."""
    plan = UnifiedSplitter(split_config, logger).build_plan(
        _ids(120), eligibility={"sklearn": _ids(120), "sequence": _ids(120)[:100]}
    )

    restored = SplitPlan.from_frame(plan.to_frame())

    assert restored.assignments.equals(plan.assignments)
    assert restored.eligibility["sequence"] == frozenset(_ids(120)[:100])


def test_describe_reports_the_counts_and_the_per_family_exclusions(split_config, logger):
    plan = UnifiedSplitter(split_config, logger).build_plan(
        _ids(100), eligibility={"sklearn": _ids(100), "sequence": _ids(100)[:90]}
    )

    described = plan.describe()

    assert described["split_n_total"] == 100
    assert described["split_n_test"] == 20
    assert described["split_n_excluded_sequence"] == 10
    assert described["split_n_excluded_sklearn"] == 0


# --- population policy, over a real frame the families disagree about ----------------------


def _write_dataset(tmp_path: Path, n: int = 200, unusable: int = 20) -> Path:
    """A frame whose last `unusable` rows have no measured TARGET.

    The sequence builder drops those rows - there is nothing to learn from a point with no label -
    while the tabular preprocessor keeps them and filters per target later. That is the disagreement
    population_policy exists to reconcile, and since covariate gaps became fill-and-flag it is the
    only one left.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "point_id": [f"p{i:04d}" for i in range(n)],
            "lat": rng.uniform(30.0, 36.0, n),
            "lon": rng.uniform(-9.0, -2.0, n),
            "feature": rng.normal(size=n),
            "target_a": rng.normal(size=n),
            "target_b": rng.normal(size=n),
        }
    )
    frame.loc[frame.index[-unusable:], "target_a"] = np.nan
    path = tmp_path / "static.csv"
    frame.to_csv(path, index=False)
    return path


def _configure(toy_config, tmp_path: Path, policy: str, **dataset_kwargs):
    _write_dataset(tmp_path, **dataset_kwargs)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "static.csv"
    toy_config.CATEGORICAL_FEATURES = []
    toy_config.SPLIT_HOLDOUT_STRATEGY = "random"
    toy_config.SPLIT_TEST_SIZE = 0.2
    toy_config.SPLIT_VAL_SIZE = 0.16
    toy_config.SPLIT_SEED = 42
    toy_config.SPLIT_POPULATION_POLICY = policy
    return toy_config


def test_intersect_splits_only_the_points_every_family_can_use(toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, "intersect")
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    plan = provider.plan()

    # 200 rows, 20 of which the sequence family drops.
    assert len(plan.assignments) == 180
    assert plan.describe()["split_n_excluded_sequence"] == 0


def test_assign_all_labels_every_point_including_the_ones_only_sklearn_can_use(toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, "assign_all")
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    plan = provider.plan()

    assert len(plan.assignments) == 200
    # The delta is still recorded under either policy - the cost of unification stays visible.
    assert plan.describe()["split_n_excluded_sequence"] == 20


# --- the cross-family invariant -----------------------------------------------------------


def _sequence_test_ids(config, logger, plan):
    """The point ids the sequence datamodule would actually hold out."""
    from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

    bundle = SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()
    _, _, test_idx = plan.split_indices(list(bundle.point_ids))
    return {bundle.point_ids[i] for i in test_idx}


def _sklearn_test_ids(config, logger, plan, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("f"))
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)

    module = ScikitDataModule(config, logger, DataManager(config, logger))
    processed = module.preprocess(module.load_frame())
    split = module.split(processed, plan)
    point_ids = processed["point_ids"].to_numpy()
    return set(point_ids[split["X_test"].index])


def test_the_two_families_hold_out_the_same_points(toy_config, logger, tmp_path, monkeypatch):
    """The regression test for the original bug.

    Under `intersect` the test sets must be identical row for row, so `rmse_test` from XGBoost and
    `rmse_test` from soil_cnn are measured on the same data and belong on the same axis.
    """
    config = _configure(toy_config, tmp_path, "intersect")
    plan = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence")).plan()

    sklearn_test = _sklearn_test_ids(config, logger, plan, monkeypatch)
    sequence_test = _sequence_test_ids(config, logger, plan)

    assert sklearn_test == sequence_test
    assert len(sklearn_test) == plan.counts()["test"]


def test_under_assign_all_the_sklearn_test_set_is_a_superset(toy_config, logger, tmp_path, monkeypatch):
    """The documented trade-off: shared membership, but not the identical rows."""
    config = _configure(toy_config, tmp_path, "assign_all")
    plan = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence")).plan()

    sklearn_test = _sklearn_test_ids(config, logger, plan, monkeypatch)
    sequence_test = _sequence_test_ids(config, logger, plan)

    assert sequence_test < sklearn_test


def test_the_sklearn_fit_pool_never_touches_the_shared_test_set(toy_config, logger, tmp_path, monkeypatch):
    """X_train is train ∪ val, and neither may leak a test point into GridSearchCV."""
    config = _configure(toy_config, tmp_path, "intersect")
    plan = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence")).plan()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("f"))
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)
    module = ScikitDataModule(config, logger, DataManager(config, logger))
    processed = module.preprocess(module.load_frame())
    split = module.split(processed, plan)

    point_ids = processed["point_ids"].to_numpy()
    fit_pool = set(point_ids[split["X_train"].index])
    assert not fit_pool & set(plan.point_ids_for("test"))
    assert fit_pool == set(plan.point_ids_for("train")) | set(plan.point_ids_for("val"))


# --- the collapse guard -------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["intersect", "assign_all"])
def test_a_starved_family_stops_the_run_under_every_policy(toy_config, logger, tmp_path, policy):
    """A family that cannot use the data must not quietly train on the scraps.

    On the real dataset three near-empty covariates reduced the sequence family to 17 of 5761
    points and nothing said so. The check is per family and policy-independent precisely because
    `assign_all` and a single-family run have no intersection to notice it - those were the two
    blind spots that let the original case through.
    """
    config = _configure(toy_config, tmp_path, policy, n=200, unusable=180)
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    with pytest.raises(ValueError, match="can use less than split.min_population_ratio"):
        provider.plan()


def test_a_starved_family_stops_a_single_family_run(toy_config, logger, tmp_path):
    """No intersection exists here at all, which is exactly why the old guard missed this."""
    config = _configure(toy_config, tmp_path, "intersect", n=200, unusable=180)
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sequence",))

    with pytest.raises(ValueError, match="sequence: 20 of 200 usable"):
        provider.plan()


def test_a_small_overlap_between_healthy_families_names_the_overlap_not_a_family(toy_config, logger):
    """The other half of the guard: everyone is individually fine, they just disagree.

    `assign_all` is the documented way past this one - unlike a starved family, which no policy
    excuses.
    """
    provider = SplitPlanProvider.__new__(SplitPlanProvider)
    provider.config = toy_config
    provider.logger = logger
    all_ids = _ids(100)
    eligibility = {
        "sklearn": frozenset(all_ids[:60]),
        "sequence": frozenset(all_ids[40:]),
    }

    with pytest.raises(ValueError, match="it is the OVERLAP between them that is small"):
        provider._guard_against_collapse(all_ids, all_ids[40:60], eligibility, "intersect")


def test_the_guard_can_be_disabled(toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, "intersect", n=200, unusable=180)
    config.SPLIT_MIN_POPULATION_RATIO = 0.0

    plan = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence")).plan()

    assert len(plan.assignments) == 20


def test_a_mass_drop_names_the_columns_that_caused_it(logger, caplog):
    """ "the bundle has 17 points" is not actionable; "these columns are 99% empty" is."""
    from yg_eo_soilnet.datamodules.frame_cleaning import drop_non_finite_rows

    frame = pd.DataFrame(
        {
            "keep": np.arange(100.0),
            "mostly_empty": [1.0] + [np.nan] * 99,
            "fine": np.arange(100.0),
        }
    )

    with caplog.at_level("WARNING"):
        drop_non_finite_rows(
            frame, logger=logger, label="test source", numeric_columns=["keep", "mostly_empty", "fine"]
        )

    assert "Worst columns by rows lost" in caplog.text
    assert "mostly_empty (99)" in caplog.text


# --- the sklearn datamodule -------------------------------------------------------------------
# Preparation that used to live on DataManager now belongs to the scikit datamodule.


@pytest.fixture
def datamodule(toy_config, logger) -> ScikitDataModule:
    return ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))


def _stub_artifact_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("frame"))
    monkeypatch.setattr(
        pd.Series, "to_parquet", lambda self, path, index=False: Path(path).write_text("series"), raising=False
    )
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)


def test_preprocess_returns_expected_structure(datamodule, toy_dataframe, caplog) -> None:
    with caplog.at_level("INFO"):
        processed = datamodule.preprocess(toy_dataframe)

    assert list(processed["X"].columns) == ["cat", "feature"]
    assert list(processed["y"].columns) == ["target_a", "target_b"]
    assert list(processed["lat"].index) == [0, 1, 2, 3]
    assert "Categorical encoding will be applied" in caplog.text


def test_preprocess_excludes_metadata_columns_even_if_present(datamodule, toy_dataframe) -> None:
    data = toy_dataframe.copy()
    data["point_id"] = [1, 2, 3, 4]
    data["geometry"] = ["g1", "g2", "g3", "g4"]

    processed = datamodule.preprocess(data)

    assert not {"point_id", "lat", "lon", "geometry"} & set(processed["X"].columns)


def test_preprocess_warns_when_too_many_features_are_dropped(toy_config, logger, caplog) -> None:
    toy_config.MIN_FEATURE_COUNT = 3
    toy_config.MAX_FEATURE_DROP_RATIO_WARNING = 0.2
    toy_config.EXISTING_HS_FEATURES = {
        "enabled": True,
        "ignore": True,
        "prefix": "S2_",
        "band_count": 1,
        "band_names": [],
    }

    data = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [10.0, 11.0, 12.0],
            "cat": ["a", "b", "a"],
            "feature_keep": [100.0, 101.0, 102.0],
            "feature_drop": [None, None, None],
            "S2_B11": [0.1, 0.2, 0.3],
        }
    )

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    with caplog.at_level("WARNING"):
        processed = module.preprocess(data)

    assert "usable features remain" in caplog.text
    assert "Dropped feature ratio" in caplog.text
    assert list(processed["X"].columns) == ["cat", "feature_keep"]


def test_preprocess_drops_existing_hyperspectral_columns_by_prefix(toy_config, logger) -> None:
    toy_config.EXISTING_HS_FEATURES = {
        "enabled": True,
        "ignore": True,
        "prefix": "S2_",
        "band_count": 2,
        "band_names": [],
    }

    data = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [10.0, 11.0, 12.0],
            "cat": ["a", "b", "a"],
            "feature": [100.0, 101.0, 102.0],
            "S2_B11": [0.1, 0.2, 0.3],
            "S2_B12": [0.4, 0.5, 0.6],
        }
    )

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    processed = module.preprocess(data)

    assert list(processed["X"].columns) == ["cat", "feature"]


def test_split_selects_this_family_rows_from_the_shared_plan(
    monkeypatch: pytest.MonkeyPatch, datamodule, toy_dataframe, split_plan_for
) -> None:
    """The sklearn family no longer decides the split; it selects its rows out of the shared one."""
    processed = datamodule.preprocess(toy_dataframe)
    _stub_artifact_logging(monkeypatch)
    plan = split_plan_for(toy_dataframe)

    split = datamodule.split(processed, plan)

    point_ids = processed["point_ids"].to_numpy()
    expected_test = set(plan.point_ids_for("test"))
    assert set(point_ids[split["X_test"].index]) == expected_test
    # X_train is the FIT POOL: train ∪ val, because GridSearchCV k-folds inside it.
    assert split["X_train"].shape[0] == len(plan.point_ids_for("train")) + len(plan.point_ids_for("val"))
    assert split["X_train_only"].shape[0] == len(plan.point_ids_for("train"))
    assert split["X_val"].shape[0] == len(plan.point_ids_for("val"))
    assert "groups_train" not in split


def test_split_without_a_plan_refuses_rather_than_inventing_one(
    monkeypatch: pytest.MonkeyPatch, datamodule, toy_dataframe
) -> None:
    """A private split is exactly the bug the shared plan exists to prevent."""
    processed = datamodule.preprocess(toy_dataframe)
    _stub_artifact_logging(monkeypatch)

    with pytest.raises(ValueError, match="needs the run's split_plan"):
        datamodule.splitter.split_data(processed, split_plan=None)


def test_split_with_clustering(
    monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger, split_plan_for
) -> None:
    class FakeClusterStrategy(BaseSpatialClusterStrategy):
        def __init__(self) -> None:
            self.plot_calls = []

        def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
            clustered = df.copy()
            clustered["cluster"] = [1, 1, 2, 2]
            return clustered

        def plot_train_test(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            self.strategy = FakeClusterStrategy()

        def load_splitter_from_config(self):
            return self.strategy

    toy_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    toy_config.SPLIT_GROUP_STRATEGY = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy",
        "params": {"n_clusters": 2},
    }

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    processed = module.preprocess(toy_dataframe)
    processed = {key: value.copy() if hasattr(value, "copy") else value for key, value in processed.items()}

    monkeypatch.setattr("yg_eo_soilnet.models.ModelConfigFactory", FakeFactory)
    _stub_artifact_logging(monkeypatch)

    plan = split_plan_for(toy_dataframe)
    split = module.split(processed, plan)

    # The clusters the holdout was blocked on travel with the split, so GroupKFold inside the fit
    # pool respects the same spatial structure the holdout did.
    assert "groups_train" in split
    assert "groups_test" in split
    assert set(split["groups_train"].unique()).issubset({1, 2})
    # No cluster may straddle two splits - that is the whole point of a grouped holdout.
    frame = plan.to_frame()
    assert frame.groupby("cluster")["split"].nunique().max() == 1


def test_split_sanitizes_features_with_the_full_schema_filter(
    monkeypatch: pytest.MonkeyPatch, toy_config, logger
) -> None:
    """The splitter must receive DataManager.filter_schema, not a metadata-only variant."""
    toy_config.ELIMINATED_FEATURES = ["eliminated"]

    captured = {}

    def fake_split_data(processed_data, *, sanitize_features=None, model_config_factory=None, split_plan=None):
        captured["sanitize_features"] = sanitize_features
        return {}

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    monkeypatch.setattr(module.splitter, "split_data", fake_split_data)

    module.split(
        {"X": pd.DataFrame(), "y": pd.DataFrame(), "lat": pd.Series(), "lon": pd.Series()},
        object(),
    )

    frame = pd.DataFrame({"eliminated": [1], "keep_me": [2]})
    assert list(captured["sanitize_features"](frame).columns) == ["keep_me"]


def test_load_frame_joins_targets_when_static_frame_lacks_them(tmp_path: Path, toy_config, logger) -> None:
    pd.DataFrame({"point_id": [1], "lat": [0.0], "lon": [0.0], "feature": [10.0]}).to_csv(
        tmp_path / "static.csv", index=False
    )
    pd.DataFrame({"point_id": [1], "target_a": [1.0], "target_b": [2.0]}).to_csv(tmp_path / "targets.csv", index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    frame = module.load_frame()

    assert {"target_a", "target_b"} <= set(frame.columns)


def test_clustering_with_a_non_strategy_raises_instead_of_returning_empty_splits(
    monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger
) -> None:
    """Falling through used to yield empty frames and a KeyError('groups_train') much later.

    The guard moved with the strategy: grouping is now decided by the unified splitter, so this is
    where a mis-declared class_path has to be caught.
    """
    from yg_eo_soilnet.datamodules.splitting import UnifiedSplitter

    toy_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    toy_config.SPLIT_GROUP_STRATEGY = {"enabled": False, "class_path": "a.b.C", "params": {}}

    class NotAStrategy:
        def load_splitter_from_config(self):
            return None

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("yg_eo_soilnet.models.ModelConfigFactory", NotAStrategy)

    coordinates = pd.DataFrame(
        {"lat": toy_dataframe["lat"].to_numpy(), "lon": toy_dataframe["lon"].to_numpy()},
        index=pd.Index(range(len(toy_dataframe))),
    )
    with pytest.raises(TypeError, match="did not resolve to a BaseSpatialClusterStrategy"):
        UnifiedSplitter(toy_config, logger).build_plan(pd.Index(range(len(toy_dataframe))), coordinates=coordinates)
