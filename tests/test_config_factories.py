"""The two model factories: LightningConfigFactory and the sklearn ModelConfigFactory."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import RobustScaler

from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder
from yg_eo_soilnet.models import ModelConfigFactory
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import (
    LightningConfigFactory,
)


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_lightning_factory_rejects_non_dl_entries() -> None:
    factory = LightningConfigFactory(registry={}, config=SimpleNamespace())

    with pytest.raises(ValueError, match="must use modeltype 'dl'"):
        factory._validate_entry(
            "bad",
            {
                "enabled": True,
                "modeltype": "ml",
                "import_path": "fake.module.FakeModel",
                "datamodule_import_path": "fake.module.FakeDataModule",
            },
        )


def _factory(registry: dict) -> LightningConfigFactory:
    return LightningConfigFactory(registry, SimpleNamespace())


@pytest.mark.parametrize("input_kind", ["tabular", None])
def test_factory_rejects_an_input_kind_other_than_sequence(input_kind) -> None:
    """Sequence is the only datamodule; an entry asking for another, or for none, must fail loudly."""
    factory = _factory({})
    spec = {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": input_kind,
        "import_path": "fake.module.FakeModel",
        "datamodule_import_path": "fake.module.Whatever",
    }
    if input_kind is None:
        del spec["input_kind"]

    with pytest.raises(ValueError, match=f"Unsupported input_kind {input_kind!r}"):
        factory._build_datamodule(target="target_a", spec=spec, data={})
    # The predicate must agree with what gets built.
    assert _factory({"soil_cnn": spec}).has_sequence_input() is False


def test_has_sequence_input_finds_an_enabled_sequence_entry() -> None:
    factory = _factory({"soil_cnn": {"enabled": True, "input_kind": "sequence"}})

    assert factory.has_sequence_input() is True
    assert factory.sequence_spec()["input_kind"] == "sequence"


class FakeSequenceDataModule:
    """Mirrors the real sequence datamodule's contract."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sequence_bundle = kwargs["sequence_bundle"]
        self.static_dim = 3
        self.target_dim = 1
        self.modality_dims = {"s1": 4, "s2": 5}
        self.temporal_enabled = True
        self.target_mean_ = np.array([2.0])
        self.target_scale_ = np.array([0.5])
        self.target_covariance_ = np.array([[1.0, 0.4], [0.4, 1.0]])
        self.target_transform = "log1p"

    def setup(self, stage=None):
        self.did_setup = stage


def _sequence_spec(**overrides) -> dict:
    spec = {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.FakeModel",
        "datamodule_import_path": f"{__name__}.FakeSequenceDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto", "modality_dims": "auto"},
        "datamodule_init_args": {"batch_size": 8},
    }
    spec.update(overrides)
    return spec


def test_factory_builds_a_sequence_datamodule_from_a_supplied_bundle() -> None:
    factory = _factory({})
    datamodule = factory._build_datamodule(
        target="target_a", spec=_sequence_spec(), data={"sequence_bundle": {"marker": 1}}
    )

    assert isinstance(datamodule, FakeSequenceDataModule)
    assert datamodule.sequence_bundle == {"marker": 1}
    assert datamodule.kwargs["batch_size"] == 8
    assert datamodule.did_setup == "fit"


def test_factory_resolves_shapes_and_target_stats_from_the_datamodule() -> None:
    factory = _factory({})
    datamodule = factory._build_datamodule(target="target_a", spec=_sequence_spec(), data={"sequence_bundle": {}})
    model = factory._build_model(_sequence_spec(), datamodule)

    assert model.kwargs["static_dim"] == 3
    assert model.kwargs["modality_dims"] == {"s1": 4, "s2": 5}
    # Target stats still arrive as plain floats so the checkpoint stays weights_only-loadable.
    assert model.kwargs["target_mean"] == [2.0]
    assert model.kwargs["target_transform"] == "log1p"


def test_sequence_bundle_requires_a_data_manager_when_none_is_supplied() -> None:
    factory = _factory({})

    with pytest.raises(KeyError, match="data_manager"):
        factory._build_datamodule(target="target_a", spec=_sequence_spec(), data={})


def test_has_sequence_input_ignores_a_disabled_entry() -> None:
    factory = _factory({"soil_cnn": {"enabled": False, "input_kind": "sequence"}})
    assert factory.has_sequence_input() is False
    assert factory.sequence_spec() is None


class FakeGridModel:
    """Accepts grid_years; stands in for the calendar-grid CNN."""

    def __init__(self, static_dim=None, target_dim=None, modality_dims=None, grid_years=None, **rest):
        self.kwargs = {
            "static_dim": static_dim,
            "target_dim": target_dim,
            "modality_dims": modality_dims,
            "grid_years": grid_years,
            **rest,
        }


class FakeGridFreeModel:
    """Does NOT accept grid_years; stands in for a model that does not rasterise."""

    def __init__(
        self,
        static_dim=None,
        target_dim=None,
        modality_dims=None,
        temporal_enabled=None,
        target_mean=None,
        target_scale=None,
        target_transform=None,
    ):
        self.kwargs = {
            "static_dim": static_dim,
            "target_dim": target_dim,
            "modality_dims": modality_dims,
            "temporal_enabled": temporal_enabled,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "target_transform": target_transform,
        }


class FakeGridDataModule(FakeSequenceDataModule):
    """The shared sequence datamodule, which exposes grid_years for whoever wants it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_years = 9


def test_grid_years_is_injected_only_into_models_that_accept_it() -> None:
    """Two entries share one datamodule; what it can offer is not what each model wants.

    Regression test: the factory used to inject every datamodule attribute unconditionally, so a
    model that does not rasterise crashed with an unexpected 'grid_years' keyword.
    """
    factory = _factory({})
    datamodule = FakeGridDataModule(sequence_bundle={})

    grid_model = factory._build_model({"import_path": f"{__name__}.FakeGridModel", "init_args": {}}, datamodule)
    assert grid_model.kwargs["grid_years"] == 9

    grid_free_model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {}}, datamodule
    )
    assert "grid_years" not in grid_free_model.kwargs
    # The shapes it does accept must still arrive.
    assert grid_free_model.kwargs["static_dim"] == 3
    assert grid_free_model.kwargs["target_mean"] == [2.0]
    assert grid_free_model.kwargs["target_transform"] == "log1p"


def test_target_covariance_is_injected_only_into_models_that_accept_it() -> None:
    """The structure-aware losses need the training targets' correlation, and only the datamodule
    has seen the whole training split. It travels the same route as target_mean/target_scale, and
    like them must reach hyper_parameters as plain floats rather than a numpy array."""
    factory = _factory({})
    datamodule = FakeGridDataModule(sequence_bundle={})

    model = factory._build_model({"import_path": f"{__name__}.FakeModel", "init_args": {}}, datamodule)
    covariance = model.kwargs["target_covariance"]
    assert covariance == [[1.0, 0.4], [0.4, 1.0]]
    assert all(isinstance(value, float) for row in covariance for value in row)

    # A model whose signature does not declare it is left alone rather than failing on an
    # unexpected keyword.
    grid_free_model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {}}, datamodule
    )
    assert "target_covariance" not in grid_free_model.kwargs


def test_a_model_taking_kwargs_still_receives_everything() -> None:
    factory = _factory({})
    model = factory._build_model(
        {"import_path": f"{__name__}.FakeModel", "init_args": {}}, FakeGridDataModule(sequence_bundle={})
    )
    assert model.kwargs["grid_years"] == 9


# --- categorical / entity-embedding contract ------------------------------


class FakeCategoricalDataModule(FakeSequenceDataModule):
    """A sequence datamodule whose setup() has fitted a train-only vocabulary."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.categorical_cardinalities = [7, 13]
        self.categorical_vocabularies = [["cl", "lo"], ["peak", "valley"]]
        self.categorical_feature_names = ["texture_20cm", "landform_class"]


class FakeEmbeddingModel:
    """Declares the categorical contract; stands in for the real Lightning modules."""

    def __init__(
        self,
        static_dim=None,
        target_dim=None,
        categorical_cardinalities=None,
        categorical_vocabularies=None,
        categorical_feature_names=None,
    ):
        self.kwargs = {
            "static_dim": static_dim,
            "target_dim": target_dim,
            "categorical_cardinalities": categorical_cardinalities,
            "categorical_vocabularies": categorical_vocabularies,
            "categorical_feature_names": categorical_feature_names,
        }


def test_categorical_contract_is_injected_into_a_model_that_declares_it() -> None:
    factory = _factory({})
    datamodule = FakeCategoricalDataModule(sequence_bundle={})

    model = factory._build_model({"import_path": f"{__name__}.FakeEmbeddingModel", "init_args": {}}, datamodule)

    assert model.kwargs["categorical_cardinalities"] == [7, 13]
    assert model.kwargs["categorical_feature_names"] == ["texture_20cm", "landform_class"]
    # The vocabulary travels into hparams so the checkpoint carries its own label->index mapping.
    assert model.kwargs["categorical_vocabularies"] == [["cl", "lo"], ["peak", "valley"]]


def test_auto_categorical_placeholders_are_resolved_from_the_datamodule() -> None:
    factory = _factory({})
    datamodule = FakeCategoricalDataModule(sequence_bundle={})

    model = factory._build_model(
        {
            "import_path": f"{__name__}.FakeEmbeddingModel",
            "init_args": {"categorical_cardinalities": "auto", "categorical_vocabularies": "auto"},
        },
        datamodule,
    )

    assert model.kwargs["categorical_cardinalities"] == [7, 13]
    assert model.kwargs["categorical_vocabularies"] == [["cl", "lo"], ["peak", "valley"]]


def test_categorical_contract_is_not_injected_into_a_model_that_ignores_it() -> None:
    factory = _factory({})
    datamodule = FakeCategoricalDataModule(sequence_bundle={})

    model = factory._build_model({"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {}}, datamodule)

    assert "categorical_cardinalities" not in model.kwargs
    assert model.kwargs["static_dim"] == 3


def test_an_empty_categorical_list_is_not_offered_as_data() -> None:
    """A dataset with no categoricals must not hand [] over as though it were a real shape."""
    factory = _factory({})
    datamodule = FakeSequenceDataModule(sequence_bundle={})
    datamodule.categorical_cardinalities = []
    datamodule.categorical_vocabularies = []
    datamodule.categorical_feature_names = []

    model = factory._build_model({"import_path": f"{__name__}.FakeModel", "init_args": {}}, datamodule)

    assert "categorical_cardinalities" not in model.kwargs
    assert "categorical_vocabularies" not in model.kwargs


def test_an_unknown_key_written_in_init_args_still_fails_loudly() -> None:
    """Filtering must not swallow a typo the user actually wrote in the registry."""
    factory = _factory({})
    with pytest.raises(TypeError, match="nonsense_arg"):
        factory._build_model(
            {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {"nonsense_arg": 1}},
            FakeGridDataModule(sequence_bundle={}),
        )


# --- sklearn ModelConfigFactory ---------------------------------------------------------------


def test_dynamic_import_loads_known_class() -> None:
    loaded = ModelConfigFactory._dynamic_import("sklearn.linear_model.LinearRegression")

    assert loaded.__name__ == "LinearRegression"


def test_dynamic_import_raises_for_missing_module() -> None:
    with pytest.raises(ImportError, match="Failed to import module"):
        ModelConfigFactory._dynamic_import("does_not_exist.SomeClass")


def test_build_model_configs_expands_enabled_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_builder(num_features: int) -> dict[str, int]:
        return {"num_features": num_features}

    registry = {
        "enabled_model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "init_args": {"input_dim": 4, "alpha": 0.5},
            "params": {"grid": [1, 2]},
            "modeltype": "ml",
            "random_seed": 123,
        },
        "custom_builder_model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "custom_model_builder": "fake.module.fake_builder",
        },
        "disabled_model": {
            "enabled": False,
            "import_path": "fake.module.FakeModel",
        },
    }

    factory = ModelConfigFactory(registry=registry)

    def fake_dynamic_import(path: str):
        if path == "fake.module.FakeModel":
            return FakeModel
        if path == "fake.module.fake_builder":
            return fake_builder
        raise AssertionError(f"Unexpected import path: {path}")

    monkeypatch.setattr(factory, "_dynamic_import", fake_dynamic_import)

    configs = factory.build_model_configs(num_features=8)

    assert set(configs) == {"enabled_model", "custom_builder_model"}
    assert configs["enabled_model"]["model"].kwargs["input_dim"] == 8
    assert configs["enabled_model"]["model"].kwargs["alpha"] == 0.5
    assert configs["enabled_model"]["params"] == {"grid": [1, 2]}
    assert configs["enabled_model"]["random_seed"] == 123
    assert configs["custom_builder_model"]["model"].kwargs["build_fn"]() == {"num_features": 8}


def test_build_model_configs_passes_through_search_n_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries opt out of GridSearchCV fan-out; everything else keeps the -1 default."""

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    registry = {
        "capped": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "search_n_jobs": 1,
        },
        "default_parallelism": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
        },
    }

    factory = ModelConfigFactory(registry=registry)
    monkeypatch.setattr(factory, "_dynamic_import", lambda path: FakeModel)

    configs = factory.build_model_configs(num_features=8)

    assert configs["capped"]["search_n_jobs"] == 1
    assert configs["default_parallelism"]["search_n_jobs"] == -1


def test_estimators_inherit_the_run_seed_instead_of_a_hardcoded_one() -> None:
    """RANDOM_SEED must reach the estimator, not just the CV splitter.

    Entries used to carry `random_state: 42` in init_args, so changing the main seed moved the
    folds but left every model on 42.
    """
    registry = {
        "inherits": {"enabled": True, "import_path": "sklearn.linear_model.Ridge"},
        "per_entry_override": {
            "enabled": True,
            "import_path": "sklearn.linear_model.Ridge",
            "random_seed": 5,
        },
        "pinned_in_init_args": {
            "enabled": True,
            "import_path": "sklearn.linear_model.Ridge",
            "init_args": {"random_state": 99},
        },
        "has_no_seed": {"enabled": True, "import_path": "sklearn.cross_decomposition.PLSRegression"},
    }

    configs = ModelConfigFactory(registry=registry).build_model_configs(num_features=4, default_seed=7)

    assert configs["inherits"]["model"].get_params()["random_state"] == 7
    assert configs["per_entry_override"]["model"].get_params()["random_state"] == 5
    # An explicit init_args entry is still an explicit override.
    assert configs["pinned_in_init_args"]["model"].get_params()["random_state"] == 99
    # Estimators without a random_state are left alone rather than erroring.
    assert "random_state" not in configs["has_no_seed"]["model"].get_params()


def test_xgboost_inherits_the_run_seed_despite_kwargs_signature() -> None:
    """XGBRegressor keeps random_state in **kwargs, so signature inspection would miss it."""
    registry = {"XGBoost": {"enabled": True, "import_path": "xgboost.XGBRegressor"}}

    configs = ModelConfigFactory(registry=registry).build_model_configs(num_features=4, default_seed=7)

    assert configs["XGBoost"]["model"].get_params()["random_state"] == 7


def test_build_model_configs_does_not_mutate_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """main.py rebuilds configs once per target off the same registry dict."""

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    registry = {
        "model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "init_args": {"input_dim": 4},
        }
    }

    factory = ModelConfigFactory(registry=registry)
    monkeypatch.setattr(factory, "_dynamic_import", lambda path: FakeModel)

    factory.build_model_configs(num_features=8)

    assert registry["model"]["init_args"]["input_dim"] == 4


def test_load_splitter_from_config_returns_enabled_splitter() -> None:
    registry = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy",
        "params": {"n_clusters": 3},
    }

    splitter = ModelConfigFactory(registry=registry, random_state=11).load_splitter_from_config()

    assert isinstance(splitter, BaseSpatialClusterStrategy)
    assert splitter.random_state == 11
    assert splitter.n_clusters == 3


def test_load_splitter_from_config_returns_none_when_disabled() -> None:
    registry = {"enabled": False}

    assert ModelConfigFactory(registry=registry).load_splitter_from_config() is None


# --- TabICL registry entry --------------------------------------------------------------------
# TabICL is wired in through the sklearn registry with no factory or trainer special-casing.
#
# These tests read the shipped registry rather than a fixture so the entry and its expectations
# cannot drift apart. Everything here is offline: TabICLRegressor's constructor does not touch the
# network, the checkpoint is only fetched on the first fit(), which is why the one test that really
# fits is opt-in.
#
#     TABICL_INTEGRATION=1 pixi run -e dev pytest tests/test_config_factories.py -k tabicl
#
# If the download dies with "Network error: Request middleware error", the HuggingFace xet CDN is
# blocked; prefix HF_HUB_DISABLE_XET=1 to fall back to the plain HTTP transfer.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "sklearn" / "model_registry.yml"

RUN_INTEGRATION = bool(os.environ.get("TABICL_INTEGRATION"))
requires_checkpoint = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="downloads a checkpoint from the HuggingFace hub; set TABICL_INTEGRATION=1 to run",
)


def _tabicl_spec() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())["TabICL"]


def test_tabicl_entry_shape() -> None:
    spec = _tabicl_spec()

    assert spec["modeltype"] == "ml"
    assert spec["import_path"] == "tabicl.TabICLRegressor"
    # A full checkpoint per worker: -1 would load one for every fold and grid point at once.
    assert spec["search_n_jobs"] == 1
    # TabICL is designed to need no tuning and every grid point is a full inference pass.
    assert spec["params"] == {}


def test_factory_builds_tabicl_from_shipped_registry() -> None:
    spec = {**_tabicl_spec(), "enabled": True}

    # A seed that is not TabICL's own default (42), so inheritance is distinguishable from it.
    configs = ModelConfigFactory(registry={"TabICL": spec}).build_model_configs(num_features=10, default_seed=7)

    model = configs["TabICL"]["model"]
    assert type(model).__name__ == "TabICLRegressor"
    assert model.get_params()["n_estimators"] == spec["init_args"]["n_estimators"]
    # The entry does not pin a seed, so it takes the run's.
    assert "random_state" not in spec["init_args"]
    assert model.get_params()["random_state"] == 7
    assert configs["TabICL"]["search_n_jobs"] == 1
    assert configs["TabICL"]["params"] == {}


def test_tabicl_uses_the_non_tree_pipeline_branch() -> None:
    """TabICL is a neural model, so it must keep the RobustScaler the tree branch drops.

    _is_tree_based_model matches on substrings of the class and module name; nothing in
    'tabiclregressor' or 'tabicl._sklearn.regressor' hits a tree marker today, and this pins that.
    """
    from tabicl import TabICLRegressor

    builder = PipelineBuilder()
    model = TabICLRegressor()

    assert builder._is_tree_based_model(model) is False

    pipeline = builder.build(model, numeric_cols=["a", "b"], categorical_cols=[])
    transformers = pipeline.named_steps["preprocessor"].transformers
    numeric_branch = next(branch for name, branch, _ in transformers if name == "num")

    assert any(isinstance(step, RobustScaler) for _, step in numeric_branch.steps)


@requires_checkpoint
def test_tabicl_fits_and_predicts_through_the_pipeline() -> None:
    from sklearn.datasets import make_regression
    from tabicl import TabICLRegressor

    X, y = make_regression(n_samples=120, n_features=8, n_informative=5, noise=0.5, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)

    pipeline = PipelineBuilder().build(
        TabICLRegressor(n_estimators=2, device="cpu", random_state=42),
        numeric_cols=list(X.columns),
        categorical_cols=[],
    )
    predictions = pipeline.fit(X, y).predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    # A collapsed constant prediction would still pass the shape check above.
    assert np.std(predictions) > 0
