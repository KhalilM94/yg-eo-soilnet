from types import SimpleNamespace

import pytest
import numpy as np

from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_lightning_factory_rejects_non_dl_entries() -> None:
    factory = LightningConfigFactory(registry={}, config=SimpleNamespace())

    try:
        factory._validate_entry(
            "bad",
            {
                "enabled": True,
                "modeltype": "ml",
                "import_path": "fake.module.FakeModel",
                "datamodule_import_path": "fake.module.FakeDataModule",
            },
        )
    except ValueError as exc:
        assert "must use modeltype 'dl'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


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
    datamodule = factory._build_datamodule(
        target="target_a", spec=_sequence_spec(), data={"sequence_bundle": {}}
    )
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
    """Does NOT accept grid_years; stands in for the sequence encoders."""

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

    grid_model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridModel", "init_args": {}}, datamodule
    )
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

    model = factory._build_model(
        {"import_path": f"{__name__}.FakeModel", "init_args": {}}, datamodule
    )
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

    model = factory._build_model(
        {"import_path": f"{__name__}.FakeEmbeddingModel", "init_args": {}}, datamodule
    )

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

    model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {}}, datamodule
    )

    assert "categorical_cardinalities" not in model.kwargs
    assert model.kwargs["static_dim"] == 3


def test_an_empty_categorical_list_is_not_offered_as_data() -> None:
    """A dataset with no categoricals must not hand [] over as though it were a real shape."""
    factory = _factory({})
    datamodule = FakeSequenceDataModule(sequence_bundle={})
    datamodule.categorical_cardinalities = []
    datamodule.categorical_vocabularies = []
    datamodule.categorical_feature_names = []

    model = factory._build_model(
        {"import_path": f"{__name__}.FakeModel", "init_args": {}}, datamodule
    )

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
