"""soil_cnn's three architecture switches, and the legacy names that are now presets of them.

SoilCNNLightningModule used to be the base of a three-class chain: a residual subclass that rebuilt
its head, and an attention subclass that rebuilt its fusion and head again. The switches fold those
into one class. This file pins what the fold must not break: every combination builds only what it
names and explains itself exactly, each legacy name is the same network as the switches it stands
for, and checkpoints and pickles written before the switches existed still load and predict.

The residual arithmetic and the attention fusion themselves are pinned by test_residual_cnn.py and
test_residual_attention_cnn.py, which run through the legacy names.
"""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.models.lightningmodules.soil_residual_attention_cnn_lightning_module import (
    SoilResidualAttentionCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import AttentionFusion, ConcatGatedFusion

from tests.support.cnn import LABEL_MEAN, LABEL_NAMES, LABEL_SCALE, built_sequence_bundle, cnn_batch


# (fusion, auxiliary_enabled, residual_enabled) - every combination.
SWITCHES = list(itertools.product(["gated", "attention"], [False, True], [False, True]))

ATTENTION_KEYS = (
    "fusion",
    "attention_static_tokens",
    "attention_d_model",
    "attention_nhead",
    "attention_num_layers",
    "attention_ff_multiplier",
    "attention_dropout",
    "attention_readout",
)
RESIDUAL_KEYS = (
    "residual_enabled",
    "residual_base_columns",
    "residual_base_hidden_dims",
    "residual_base_dropout",
    "residual_base_validity_channels",
    "residual_base_max_missing",
    "auxiliary_label_mean",
    "auxiliary_label_scale",
)
# What each class's checkpoints did NOT carry before the switches existed.
PRE_SWITCH_MISSING = {
    SoilCNNLightningModule: ("auxiliary_enabled", *ATTENTION_KEYS, *RESIDUAL_KEYS),
    SoilResidualCNNLightningModule: ("auxiliary_enabled", "residual_enabled", *ATTENTION_KEYS),
    SoilResidualAttentionCNNLightningModule: ("auxiliary_enabled", "fusion", "residual_enabled"),
}
LEGACY_CLASSES = list(PRE_SWITCH_MISSING)


def _kwargs(**overrides):
    kwargs = dict(
        static_dim=5,
        target_dim=1,
        modality_dims={"m": 3},
        temporal_encoder="dilated_tempcnn",
        grid_years=3,
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_mean=LABEL_MEAN,
        auxiliary_label_scale=LABEL_SCALE,
        auxiliary_label_columns=["lab_a"],
        residual_base_columns={"target_a": "lab_b"},
        cnn_hidden_dims=[8],
        modality_embed_dim=8,
        static_hidden_dims=[8],
        head_hidden_dims=[8],
        attention_d_model=16,
        attention_nhead=4,
        attention_num_layers=1,
    )
    kwargs.update(overrides)
    return kwargs


def _build(cls=SoilCNNLightningModule, **overrides):
    torch.manual_seed(0)
    return cls(**_kwargs(**overrides)).eval()


def _switched(fusion, auxiliary, residual):
    return _build(fusion=fusion, auxiliary_enabled=auxiliary, residual_enabled=residual)


def _first_linear(block: nn.Module) -> nn.Linear:
    return next(child for child in block.modules() if isinstance(child, nn.Linear))


def _round_trip(module, cls, tmp_path: Path, hparams=None):
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams) if hparams is None else hparams,
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    torch.load(path, weights_only=True)  # the constraint that forbids numpy in hyper_parameters
    # Strict by default, so a key the switches add or drop fails here rather than loading silently.
    return cls.load_from_checkpoint(path, map_location="cpu").eval()


# --- every combination ---------------------------------------------------------------------------


@pytest.mark.parametrize("fusion, auxiliary, residual", SWITCHES)
def test_every_combination_builds_what_it_names_and_nothing_else(fusion, auxiliary, residual) -> None:
    module = _switched(fusion, auxiliary, residual)

    built, other = (AttentionFusion, ConcatGatedFusion) if fusion == "attention" else (ConcatGatedFusion, AttentionFusion)
    assert isinstance(module.fusion, built)
    # Built once: no fusion of the other kind survives anywhere in the tree.
    assert not any(isinstance(child, other) for child in module.modules())
    # fused vector + auxiliary block (value + flag) + base block (value + flag), each only when on
    expected = module.fusion.output_dim + (2 if auxiliary else 0) + (2 if residual else 0)
    assert _first_linear(module.output_head).in_features == expected
    assert module.has_auxiliary_labels is auxiliary
    assert ("residual_base_index" in dict(module.named_buffers())) is residual


@pytest.mark.parametrize("fusion, auxiliary, residual", SWITCHES)
def test_every_combination_explains_itself_exactly(fusion, auxiliary, residual) -> None:
    """forward_from_parts(explanation_parts(batch)) == forward(batch), whatever is switched on."""
    module = _switched(fusion, auxiliary, residual)
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        prediction = module(batch)
        parts, groups = module.explanation_parts(batch)
        assert torch.allclose(module.forward_from_parts(parts), prediction, atol=1e-6)

    assert prediction.shape == (4, 1)
    kinds = {group["kind"] for group in groups}
    assert ("auxiliary" in kinds) is auxiliary
    assert ("residual_base" in kinds) is residual


@pytest.mark.parametrize("fusion, auxiliary, residual", SWITCHES)
def test_every_combination_survives_a_weights_only_checkpoint_round_trip(
    tmp_path: Path, fusion, auxiliary, residual
) -> None:
    module = _switched(fusion, auxiliary, residual)
    restored = _round_trip(module, SoilCNNLightningModule, tmp_path)

    assert (restored.fusion_type, restored.auxiliary_enabled, restored.residual_enabled) == (
        fusion,
        auxiliary,
        residual,
    )
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        assert torch.allclose(restored(batch), module(batch), atol=1e-6)


def test_attention_without_a_residual_base_trains() -> None:
    """The combination no class could build before the switches."""
    module = _build(fusion="attention", residual_enabled=False).train()
    batch = cnn_batch(labels=3)

    nn.functional.mse_loss(module(batch), batch["y"]).backward()

    assert all(parameter.grad is not None for parameter in module.fusion.parameters())
    assert "residual_base_index" not in dict(module.named_buffers())


# --- the switches' own semantics -----------------------------------------------------------------


def test_residual_off_ignores_the_mapping_entirely() -> None:
    """Off means unread: a mapping naming a column the data lacks is not even validated."""
    module = _build(residual_enabled=False, residual_base_columns={"target_a": "not_a_column"})

    assert module.serving_label_columns == ["lab_a"]
    assert module.hparams["residual_base_columns"] == {"target_a": "not_a_column"}


def test_residual_on_with_an_empty_mapping_names_the_switch() -> None:
    with pytest.raises(ValueError, match="Set residual_enabled: false"):
        _build(residual_enabled=True, residual_base_columns={})


def test_auxiliary_off_ignores_the_columns_but_keeps_them_in_hparams() -> None:
    module = _build(auxiliary_enabled=False)

    assert not module.has_auxiliary_labels
    assert module.serving_label_columns == []
    # Kept, so switching back on needs no second edit.
    assert module.hparams["auxiliary_label_columns"] == ["lab_a"]


def test_the_base_and_auxiliary_overlap_is_checked_against_what_is_actually_read() -> None:
    with pytest.raises(ValueError, match="appear in both"):
        _build(residual_enabled=True, auxiliary_label_columns=["lab_b"])

    module = _build(residual_enabled=True, auxiliary_enabled=False, auxiliary_label_columns=["lab_b"])
    assert module.serving_label_columns == ["lab_b"]


def test_an_unknown_fusion_is_refused() -> None:
    with pytest.raises(ValueError, match="fusion must be one of"):
        _build(fusion="mlp")


def test_attention_settings_are_inert_under_gated_fusion() -> None:
    module = _build(fusion="gated", attention_static_tokens="bogus", attention_readout="bogus")

    assert isinstance(module.fusion, ConcatGatedFusion)


def test_per_feature_tokens_build_the_static_encoder_without_its_mlp() -> None:
    """Not built and then discarded, as the attention subclass used to - never built at all."""
    module = _build(fusion="attention", attention_static_tokens="per_feature")

    assert module.static_encoder.mlp is False
    assert not any(isinstance(child, nn.Linear) for child in module.static_encoder.modules())
    assert module.static_encoder.output_dim == 5


# --- the legacy names -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy_cls, switches",
    [
        (SoilResidualCNNLightningModule, dict(residual_enabled=True)),
        (SoilResidualAttentionCNNLightningModule, dict(residual_enabled=True, fusion="attention")),
    ],
)
def test_a_legacy_name_is_the_same_network_as_its_switches(legacy_cls, switches) -> None:
    legacy = _build(legacy_cls)
    unified = _build(SoilCNNLightningModule, **switches)

    legacy_state, unified_state = legacy.state_dict(), unified.state_dict()
    assert legacy_state.keys() == unified_state.keys()
    # Same construction order under the same seed, so the same weights, not just the same shapes.
    for key, value in legacy_state.items():
        assert torch.equal(value, unified_state[key]), key
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        assert torch.equal(legacy(batch), unified(batch))


@pytest.mark.parametrize("cls", LEGACY_CLASSES)
def test_a_checkpoint_written_before_the_switches_loads_into_its_own_class(tmp_path: Path, cls) -> None:
    """Old hyper_parameters lack the switches; the class's own defaults must rebuild the same model."""
    module = _build(cls)
    missing = PRE_SWITCH_MISSING[cls]
    hparams = {key: value for key, value in dict(module.hparams).items() if key not in missing}

    restored = _round_trip(module, cls, tmp_path, hparams=hparams)

    assert (restored.fusion_type, restored.residual_enabled) == (module.fusion_type, module.residual_enabled)
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        assert torch.allclose(restored(batch), module(batch), atol=1e-6)


@pytest.mark.parametrize("cls", LEGACY_CLASSES)
def test_a_model_pickled_before_the_switches_still_predicts(cls) -> None:
    """MLflow serves a pickle, which restores the instance __dict__ only.

    An object pickled before the switches existed has none of them in its __dict__, so every read
    falls through to the class attribute - which each legacy name sets to what it was built with.
    """
    module = _build(cls)
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        expected = module(batch)

    restored = pickle.loads(pickle.dumps(module))
    for name in ("fusion_type", "attention_static_tokens", "auxiliary_enabled", "residual_enabled"):
        restored.__dict__.pop(name, None)

    with torch.no_grad():
        assert torch.equal(restored(batch), expected)
    assert restored.serving_label_columns == module.serving_label_columns


def test_relog_resolves_a_folded_entry_to_its_legacy_class() -> None:
    """A run logged under a model_name the registry no longer has still finds its class."""
    from relog import infer_model_class_path, resolve_model_class

    config = SimpleNamespace(LIGHTNING_MODEL_REGISTRY={"soil_cnn": {"import_path": "unused.Path"}})

    assert resolve_model_class(infer_model_class_path(config, "soil_residual_cnn")) is SoilResidualCNNLightningModule
    assert (
        resolve_model_class(infer_model_class_path(config, "soil_residual_attention_cnn"))
        is SoilResidualAttentionCNNLightningModule
    )


# --- the shipped entry ---------------------------------------------------------------------------


def test_the_shipped_soil_cnn_entry_builds_when_no_lab_columns_are_carried(tmp_path: Path, logger) -> None:
    """The entry writes auxiliary_label_mean/scale as `auto` for the residual base. With no lab
    columns carried there is nothing to fill them from, and the sentinel must not reach the model
    as the string "auto" - a plain soil_cnn run has no reason to carry lab columns at all.
    """
    from config import load_lightning_registry
    from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
    from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory


    registry = load_lightning_registry("configs/lightning/models/defaults.yml")
    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)}, carry_labels=False)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    factory = LightningConfigFactory(registry, SimpleNamespace(TARGET_COLUMNS=["target_a"]))
    module = factory._build_model(registry["soil_cnn"], datamodule)

    assert not module.residual_enabled
    assert module.hparams["auxiliary_label_mean"] == []
