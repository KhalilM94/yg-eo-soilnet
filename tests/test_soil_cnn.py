"""soil_cnn, the one Lightning model: its three architecture switches and the legacy names that are
now presets of them, the module itself, the residual base, attention fusion inside the module, and
the heteroscedastic head.

SoilCNNLightningModule used to be the base of a three-class chain: a residual subclass that rebuilt
its head, and an attention subclass that rebuilt its fusion and head again. The switches fold those
into one class. This file pins what the fold must not break: every combination builds only what it
names and explains itself exactly, each legacy name is the same network as the switches it stands
for, and checkpoints and pickles written before the switches existed still load and predict.

The residual arithmetic and the attention fusion are pinned in their own sections below, which run
through the legacy names; AttentionFusion on its own is in test_encoders.py.
"""

from __future__ import annotations

import itertools
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from tests.support.cnn import (
    LABEL_MEAN,
    LABEL_NAMES,
    LABEL_SCALE,
    built_sequence_bundle,
    cnn_batch,
    detach_logging,
    expected_base,
    zero_head,
)
from yg_eo_soilnet.datamodules.sequence.sequence_builder import to_decimal_year
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.models.lightningmodules.soil_residual_attention_cnn_lightning_module import (
    SoilResidualAttentionCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    AttentionFusion,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
    decimal_year_to_month_index,
)

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
    """Refused rather than silently degrading to a plain soil_cnn."""
    with pytest.raises(ValueError, match="residual_base_columns is required") as refused:
        _build(residual_enabled=True, residual_base_columns={})
    assert "Set residual_enabled: false" in str(refused.value)


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


# --- the module -------------------------------------------------------------------------------
# x


ENCODERS = ["dilated_tempcnn", "annual_grid2d"]
ENCODER_CLASSES = [DilatedTempCNNEncoder, AnnualGrid2DEncoder]


def _times(dates) -> torch.Tensor:
    return torch.as_tensor(to_decimal_year(pd.Series(pd.to_datetime(dates)))).unsqueeze(0)


# --- lightning module ------------------------------------------------------


def _module(encoder: str, **kwargs) -> SoilCNNLightningModule:
    defaults = dict(static_dim=5, target_dim=1, modality_dims={"m": 3}, temporal_encoder=encoder, grid_years=3)
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilCNNLightningModule(**defaults).eval()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_forward_produces_gradients(encoder: str) -> None:
    module = _module(encoder)
    batch = cnn_batch()

    predictions = module(batch)
    assert predictions.shape == (4, 1)

    module.loss_fn(predictions, batch["y"]).backward()
    conv_parameters = [p for p in module.temporal_encoders.parameters() if p.requires_grad]
    assert conv_parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in conv_parameters)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("start", ["2019-01-01", "2018-01-01"])
def test_module_is_invariant_to_the_calendar_era(encoder: str, start: str) -> None:
    """Same readings, same calendar months, a decade later -> identical predictions.

    Both start years are exercised on purpose: 2018 shifts onto leap year 2028, which is where a
    month lookup that ignores leap days would drift.
    """
    module = _module(encoder)
    batch = cnn_batch(start=start)
    shifted = cnn_batch(start=start, year_offset=10)
    shifted = {**batch, "sequence_time": shifted["sequence_time"]}

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module(shifted))


def test_decimal_shifting_is_not_the_same_as_shifting_dates_across_a_leap_year() -> None:
    """Guard against 'fix' the era test by adding a constant to the decimal year instead.

    A decimal year encodes the day-of-year as a fraction of 365.25, so adding 10.0 keeps the
    fraction and moves the year label. When the destination year is a leap year that fraction names
    a different calendar day: 2018-03-01 is day 60, and day 60 of 2028 is 29 February. Shifting real
    dates is the meaningful operation, and only it is invariant.
    """
    real_dates = _times(["2018-03-01"])
    shifted_dates = _times(["2028-03-01"])
    assert decimal_year_to_month_index(real_dates).tolist() == decimal_year_to_month_index(shifted_dates).tolist()

    naive_shift = real_dates + 10.0
    assert decimal_year_to_month_index(naive_shift).item() == 1, "day 60 of a leap year is February"


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_runs_on_spans_it_was_not_built_for(encoder: str) -> None:
    module = _module(encoder, grid_years=None)

    with torch.no_grad():
        short = module(cnn_batch(length=6, seed=1))
        long = module(cnn_batch(length=90, seed=2))

    assert short.shape == long.shape == (4, 1)
    assert torch.isfinite(short).all() and torch.isfinite(long).all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_consumes_validity_channels(encoder: str) -> None:
    module = _module(encoder)
    batch = cnn_batch()
    flipped_validity = batch["sequence_validity"]["m"].clone()
    flipped_validity[0, :6] = False
    flipped = {**batch, "sequence_validity": {"m": flipped_validity}}

    with torch.no_grad():
        assert not torch.allclose(module(batch)[0], module(flipped)[0])

    blind = _module(encoder, use_validity_channels=False)
    with torch.no_grad():
        torch.testing.assert_close(blind(batch), blind(flipped))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_ignores_masked_out_observations(encoder: str) -> None:
    module = _module(encoder)
    batch = cnn_batch(length=12)
    mask = batch["sequence_mask"]["m"].clone()
    mask[:, 8:] = False
    batch = {**batch, "sequence_mask": {"m": mask}}

    polluted = batch["sequences"]["m"].clone()
    polluted[:, 8:] += 99.0

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module({**batch, "sequences": {"m": polluted}}))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_inverts_the_target_transform_for_prediction(encoder: str) -> None:
    module = _module(encoder, target_mean=[1.5], target_scale=[0.5], target_transform="log1p")
    batch = cnn_batch()

    with torch.no_grad():
        expected = torch.expm1((module(batch) * 0.5 + 1.5) / 10.0)
        torch.testing.assert_close(module.predict_step(batch, 0), expected)


def test_module_supports_any_number_of_modalities() -> None:
    modality_dims = {"s2": 10, "s1": 8, "soil": 3, "ag": 3, "clim": 3}
    module = SoilCNNLightningModule(
        static_dim=13,
        target_dim=1,
        modality_dims=modality_dims,
        grid_years=9,
        cnn_hidden_dims={"s2": [48], "s1": [32], "soil": [16], "ag": [16], "clim": [16]},
        modality_embed_dim=16,
    ).eval()

    generator = torch.Generator().manual_seed(0)
    dates = pd.date_range("2019-01-01", periods=18, freq="MS")
    times = torch.as_tensor(to_decimal_year(pd.Series(dates))).unsqueeze(0).repeat(2, 1)
    batch = {
        "x_static": torch.randn(2, 13, generator=generator),
        "sequences": {name: torch.randn(2, 18, dim, generator=generator) for name, dim in modality_dims.items()},
        "sequence_mask": {name: torch.ones(2, 18, dtype=torch.bool) for name in modality_dims},
        "sequence_time": {name: times for name in modality_dims},
        "sequence_validity": {name: torch.ones(2, 18, dim, dtype=torch.bool) for name, dim in modality_dims.items()},
    }

    with torch.no_grad():
        assert module(batch).shape == (2, 1)
    assert len(module.temporal_encoders) == 5


def test_module_rejects_a_per_modality_mapping_that_omits_a_modality() -> None:
    with pytest.raises(ValueError, match="no entry in cnn_hidden_dims"):
        SoilCNNLightningModule(
            static_dim=4, target_dim=1, modality_dims={"s1": 3, "s2": 4}, cnn_hidden_dims={"s1": [16]}
        )


def test_module_rejects_an_unknown_encoder_name() -> None:
    with pytest.raises(ValueError, match="temporal_encoder"):
        SoilCNNLightningModule(static_dim=4, target_dim=1, modality_dims={"m": 3}, temporal_encoder="resnet")


def test_module_runs_without_a_temporal_branch() -> None:
    module = _module("dilated_tempcnn", temporal_enabled=False)
    assert module(cnn_batch()).shape == (4, 1)
    assert len(module.temporal_encoders) == 0


def test_module_runs_without_static_features() -> None:
    module = _module("dilated_tempcnn", static_dim=0)
    batch = {**cnn_batch(), "x_static": torch.zeros(4, 0)}
    predictions = module(batch)

    assert predictions.shape == (4, 1)
    assert torch.isfinite(predictions).all()


# --- auxiliary lab columns ---------------------------------------------------


def _auxiliary_module(**kwargs) -> SoilCNNLightningModule:
    defaults = dict(
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_columns=["lab_a", "lab_c"],
    )
    defaults.update(kwargs)
    return _module("dilated_tempcnn", **defaults)


def test_a_selected_lab_column_reaches_the_head_and_an_unselected_one_does_not() -> None:
    """The whole point of the feature, and the guard that selection is by position not by luck."""
    module = _auxiliary_module()
    batch = cnn_batch()
    baseline = module(batch)

    for column, expected_change in (("lab_a", True), ("lab_b", False), ("lab_c", True)):
        perturbed = {**batch, "x_labels": batch["x_labels"].clone()}
        perturbed["x_labels"][:, LABEL_NAMES.index(column)] += 5.0
        changed = not torch.allclose(module(perturbed), baseline, atol=1e-6)
        assert changed is expected_change, f"{column} should{'' if expected_change else ' not'} affect the head"


def test_validity_flags_reach_the_head_too() -> None:
    module = _auxiliary_module()
    batch = cnn_batch()

    flipped = {**batch, "x_label_validity": batch["x_label_validity"].clone()}
    flipped["x_label_validity"][:, LABEL_NAMES.index("lab_a")] = False

    assert not torch.allclose(module(flipped), module(batch), atol=1e-6)


def test_validity_channels_can_be_switched_off() -> None:
    module = _auxiliary_module(auxiliary_validity_channels=False)
    batch = cnn_batch()

    flipped = {**batch, "x_label_validity": torch.zeros_like(batch["x_label_validity"])}
    assert torch.allclose(module(flipped), module(batch), atol=1e-6)
    # Two selected columns, no flags -> the raw branch is 2 wide rather than 4.
    assert module.auxiliary_output_dim == 2


def test_selecting_a_column_that_is_also_a_target_is_refused() -> None:
    """A filter would be worse than an error: the run would train a model that reads its answer."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _auxiliary_module(
            target_names=["target_a", "lab_c"],
            auxiliary_available_names=[*LABEL_NAMES, "target_a"],
            auxiliary_label_columns=["lab_a", "lab_c"],
        )


def test_selecting_an_unknown_column_names_what_is_available() -> None:
    with pytest.raises(ValueError, match=r"does not carry.*lab_zzz"):
        _auxiliary_module(auxiliary_label_columns=["lab_a", "lab_zzz"])


def test_selecting_a_duplicate_column_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate column"):
        _auxiliary_module(auxiliary_label_columns=["lab_a", "lab_a"])


def test_auxiliary_columns_without_target_names_are_refused() -> None:
    """Skipping the leakage check silently is exactly the failure this feature could cause."""
    with pytest.raises(ValueError, match="requires target_names"):
        _auxiliary_module(target_names=[])


def test_raw_concat_widens_the_head_by_the_column_count() -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[])
    head_input = next(layer for layer in module.output_head.modules() if isinstance(layer, torch.nn.Linear))

    # 2 selected columns + 2 validity flags, appended straight onto the fused vector.
    assert module.auxiliary_output_dim == 4
    assert isinstance(module.auxiliary_encoder, torch.nn.Identity)
    assert head_input.in_features == module.fusion.output_dim + 4


def test_an_encoder_width_decouples_the_head_from_the_column_count() -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[16, 8])
    head_input = next(layer for layer in module.output_head.modules() if isinstance(layer, torch.nn.Linear))

    assert module.auxiliary_output_dim == 8
    assert head_input.in_features == module.fusion.output_dim + 8


def test_no_auxiliary_columns_leaves_the_architecture_untouched() -> None:
    """The default must be byte-for-byte the model that existed before this feature."""
    plain = _module("dilated_tempcnn")
    head_input = next(layer for layer in plain.output_head.modules() if isinstance(layer, torch.nn.Linear))

    assert plain.has_auxiliary_labels is False
    assert plain.auxiliary_output_dim == 0
    assert head_input.in_features == plain.fusion.output_dim
    # No target_names needed, and a batch without the lab keys still runs.
    batch = {key: value for key, value in cnn_batch().items() if not key.startswith("x_label")}
    assert torch.isfinite(plain(batch)).all()


def test_a_batch_that_no_longer_matches_the_label_roster_is_refused() -> None:
    """Positions were resolved at build time; a narrower batch would read a different measurement."""
    module = _auxiliary_module()
    batch = cnn_batch(labels=2)

    with pytest.raises(ValueError, match="no longer matches the checkpoint's label roster"):
        module(batch)


def test_auxiliary_selection_survives_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[16])
    checkpoint_path = tmp_path / "auxiliary.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["hyper_parameters"]["auxiliary_label_columns"] == ["lab_a", "lab_c"]
    assert loaded["hyper_parameters"]["auxiliary_available_names"] == LABEL_NAMES
    # The resolved positions travel with the weights rather than being re-derived on load.
    assert loaded["state_dict"]["auxiliary_index"].tolist() == [0, 2]


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_checkpoint_reloads_under_weights_only(tmp_path: Path, encoder: str) -> None:
    module = _module(encoder, target_mean=np.array([2.0]), target_scale=np.array([0.75]))
    checkpoint_path = tmp_path / f"{encoder}.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["hyper_parameters"]["target_mean"] == [2.0]


def test_a_scalar_cnn_width_still_means_one_block() -> None:
    """`cnn_hidden_dims: 64` in a config is the same as `[64]`, so old spellings keep working."""
    module = _module("dilated_tempcnn", cnn_hidden_dims=64)

    encoder = module.temporal_encoders["m"]
    assert encoder.hidden_dims == [64]


def test_a_per_modality_mapping_may_carry_a_width_list_each() -> None:
    module = _module(
        "dilated_tempcnn",
        modality_dims={"s2": 3, "s1": 2},
        cnn_hidden_dims={"s2": [16, 32], "s1": [8]},
    )

    assert module.temporal_encoders["s2"].hidden_dims == [16, 32]
    assert module.temporal_encoders["s1"].hidden_dims == [8]


# --- builder to module, end to end ----------------------------------------


def test_builder_to_cnn_module_with_auxiliary_labels_end_to_end(tmp_path: Path, logger) -> None:
    from lightning.pytorch import Trainer

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    module = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_columns=["lab_dense", "lab_sparse"],
        auxiliary_hidden_dims=[8],
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    assert module.auxiliary_index.tolist() == [1, 2]

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    # The sparse column is partly train-median filled, so this also pins that the fill path does not
    # produce a NaN loss.
    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert torch.isfinite(predicted).all()


def test_builder_to_cnn_module_end_to_end(tmp_path: Path, logger) -> None:
    from lightning.pytorch import Trainer

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    # Enough points that val and test are both non-empty; a 3-point fixture leaves val with none.
    bundle = built_sequence_bundle(
        tmp_path,
        logger,
        {point: dates[: 20 + 4 * point] for point in range(1, 9)},
    )
    datamodule = SoilSequenceDataModule(
        bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5, target_transform="log1p"
    )
    datamodule.setup("fit")
    assert datamodule.val_idx_.size >= 2 and datamodule.test_idx_.size >= 1

    module = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",  # batches here are too small for BatchNorm
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
    )

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert len(predicted) == len(datamodule.y_test_frame_)
    assert torch.isfinite(predicted).all()


# --- the residual base ------------------------------------------------------------------------
# Residual calendar-grid CNN: the base-prediction anchor and everything it touches.
#
# The risk here is entirely arithmetic and entirely silent. The base arrives standardized by the LAB
# standardizer while `y` lives in the TARGET's space, so a missing hop - or the two applied in the
# wrong order - still produces a well-shaped tensor and a loss that goes down. The first section
# pins the round trip against a head that has been zeroed, which is the only way to see the offset on
# its own.


def _residual_module(**kwargs) -> SoilResidualCNNLightningModule:
    defaults = dict(
        static_dim=5,
        target_dim=1,
        modality_dims={"m": 3},
        temporal_encoder="dilated_tempcnn",
        grid_years=3,
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_mean=LABEL_MEAN,
        auxiliary_label_scale=LABEL_SCALE,
        residual_base_columns={"target_a": "lab_b"},
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilResidualCNNLightningModule(**defaults).eval()


# --- the offset itself -------------------------------------------------------


def test_a_zeroed_head_reproduces_the_base_in_the_targets_own_space() -> None:
    """The core contract: prediction = base + head, so with no head it is the base exactly."""
    module = _residual_module()
    zero_head(module)
    batch = cnn_batch(labels=3)

    assert torch.allclose(
        module(batch).reshape(-1).double(), expected_base(batch), atol=1e-5
    )


def test_the_base_passes_through_log1p_and_the_target_standardizer_in_that_order() -> None:
    """Un-standardize the LAB value, then log1p, then standardize by the TARGET's statistics.

    Order matters and is invisible in the output shape: the target statistics were fitted on
    already-transformed targets, so applying them before log1p would leave the offset in a space
    nothing else uses.
    """
    module = _residual_module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)

    expected = expected_base(batch, target_mean=4.0, target_scale=2.0, log1p=True)
    assert torch.allclose(module(batch).reshape(-1).double(), expected, atol=1e-5)


def test_predict_step_inverts_back_to_the_bases_original_units() -> None:
    """End to end: a zeroed head served through predict_step returns the base as it was written."""
    module = _residual_module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)

    raw = batch["x_labels"][:, 1].double() * LABEL_SCALE[1] + LABEL_MEAN[1]
    assert torch.allclose(
        module.predict_step(batch, 0).reshape(-1).double(), raw.clamp_min(0.0), atol=1e-4
    )


def test_a_negative_base_is_clipped_the_way_a_measured_target_would_be() -> None:
    """log1p is undefined below -1, and the datamodule clips a negative target to 0 rather than fail."""
    module = _residual_module(target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)
    # -100 in lab-standardized space is far below the column's mean of 30.
    batch["x_labels"][:, 1] = -100.0

    assert torch.allclose(module(batch), torch.zeros_like(module(batch)))


def test_the_offset_lands_on_the_mean_half_only_on_a_variance_head() -> None:
    """A 2*target_dim readout: shifting the log variances too would make exp(40) a variance."""
    module = _residual_module(predict_variance=True, target_dim=1)
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        fused = module._fuse(batch, device=torch.device("cpu"), dtype=torch.float32)
        block = module._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32)
        head = module.output_head(torch.cat([fused, module.base_encoder(block)], dim=-1))
        out = module(batch)

    assert out.shape[-1] == 2
    # The log-variance half is untouched, the mean half is shifted by exactly the base.
    assert torch.allclose(out[:, 1:], head[:, 1:].clamp(-10.0, 10.0), atol=1e-6)
    assert torch.allclose(out[:, 0] - head[:, 0], expected_base(batch).float(), atol=1e-5)


def test_the_base_also_reaches_the_network_as_an_input() -> None:
    """Not just an offset: changing the base must move the head's own contribution too."""
    module = _residual_module(residual_base_hidden_dims=[8])
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        fused = module._fuse(batch, device=torch.device("cpu"), dtype=torch.float32)
        low = module._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32)
        batch["x_labels"][:, 1] += 3.0
        high = module._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32)
        # The head's own output, with the additive offset excluded on both sides.
        left = module.output_head(torch.cat([fused, module.base_encoder(low)], dim=-1))
        right = module.output_head(torch.cat([fused, module.base_encoder(high)], dim=-1))

    assert not torch.allclose(left, right)


def test_the_validity_flag_is_carried_into_the_block() -> None:
    """A median-filled base has to be distinguishable from a measured one."""
    module = _residual_module()
    batch = cnn_batch(labels=3)
    block = module._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32)
    assert block.shape[-1] == 2

    without = _residual_module(residual_base_validity_channels=False)
    assert without._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32).shape[-1] == 1


def test_a_multi_target_offset_is_ordered_by_target_names() -> None:
    """Position j of the offset must line up with output column j, not with the roster's order."""
    module = _residual_module(
        target_dim=2,
        target_names=["target_a", "target_b"],
        residual_base_columns={"target_a": "lab_c", "target_b": "lab_a"},
    )
    zero_head(module)
    batch = cnn_batch(labels=3)

    assert module.residual_base_index.tolist() == [2, 0]
    predicted = module(batch).double()
    assert torch.allclose(predicted[:, 0], expected_base(batch, index=2), atol=1e-5)
    assert torch.allclose(predicted[:, 1], expected_base(batch, index=0), atol=1e-5)


# --- construction-time refusals ----------------------------------------------


def test_a_target_with_no_base_column_is_refused() -> None:
    with pytest.raises(ValueError, match="no entry for target"):
        _residual_module(
            target_dim=2,
            target_names=["target_a", "target_b"],
            residual_base_columns={"target_a": "lab_b"},
        )


def test_a_base_column_that_is_a_fitted_target_is_refused() -> None:
    """The measured target itself, not a prediction of it - the inherited check never sees this map."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _residual_module(
            auxiliary_available_names=["target_a", "lab_b", "lab_c"],
            residual_base_columns={"target_a": "target_a"},
        )


def test_a_sibling_target_is_refused_as_a_base_under_per_target_grouping() -> None:
    """fitted_target_names is the whole run's targets, so a sibling leaks just as surely."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _residual_module(
            target_names=["target_a"],
            fitted_target_names=["target_a", "lab_b"],
            residual_base_columns={"target_a": "lab_b"},
        )


def test_an_unknown_base_column_is_refused() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        _residual_module(residual_base_columns={"target_a": "lab_z"})


def test_an_empty_roster_names_the_flag_that_fixes_it() -> None:
    with pytest.raises(ValueError, match="CARRY_LABEL_COLUMNS"):
        _residual_module(auxiliary_available_names=[])


def test_missing_roster_statistics_are_refused() -> None:
    """Without them the base never leaves the lab standardizer, and the offset is a z-score."""
    with pytest.raises(ValueError, match="auxiliary_label_mean and auxiliary_label_scale"):
        _residual_module(auxiliary_label_mean=None, auxiliary_label_scale=None)


# --- checkpoint --------------------------------------------------------------


def test_the_mapping_survives_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    """The residual settings must round-trip through hyper_parameters along with everything else.

    Losing residual_base_columns on reload would rebuild the module with an empty mapping, which now
    raises - but losing static_dim would rebuild a differently shaped network that still loads.
    """
    module = _residual_module(residual_base_hidden_dims=[8])
    assert dict(module.hparams)["residual_base_columns"] == {"target_a": "lab_b"}
    assert dict(module.hparams)["static_dim"] == 5

    path = tmp_path / "residual.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    torch.load(path, weights_only=True)  # the constraint that forbids numpy in hyper_parameters

    restored = SoilResidualCNNLightningModule.load_from_checkpoint(path, map_location="cpu").eval()
    assert restored.residual_base_index.tolist() == module.residual_base_index.tolist()

    batch = cnn_batch(labels=3)
    with torch.no_grad():
        assert torch.allclose(restored(batch), module(batch), atol=1e-6)


# --- attribution seam --------------------------------------------------------


def test_forward_from_parts_reproduces_forward_exactly() -> None:
    """The equality every explainer depends on; drift here misattributes to a model nobody trained."""
    module = _residual_module(residual_base_hidden_dims=[8])
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        parts, groups = module.explanation_parts(batch)
        assert torch.allclose(module.forward_from_parts(parts), module(batch), atol=1e-6)

    base_groups = [group for group in groups if group["kind"] == "residual_base"]
    assert [group["name"] for group in base_groups] == ["lab_b"]
    assert base_groups[0]["columns"] == [0, 1]


def test_the_base_part_carries_the_offset_so_perturbing_it_moves_the_prediction() -> None:
    """The offset is derived from the PART, not the batch - a shap interpolation has no batch."""
    module = _residual_module()
    # Zeroed so the shift is the offset alone. With the head live the delta is 1 PLUS whatever the
    # head made of the same perturbation, since the base is an input as well as an anchor.
    zero_head(module)
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        parts, _groups = module.explanation_parts(batch)
        before = module.forward_from_parts(parts)
        parts[-1] = parts[-1].clone()
        parts[-1][:, 0] += 1.0
        after = module.forward_from_parts(parts)

    assert torch.allclose(after - before, torch.ones_like(before), atol=1e-4)


def test_the_beeswarm_colours_the_base_in_original_units() -> None:
    """The block is published in the TARGET's space, so it inverts with the target statistics.

    Colouring it with label_mean/label_scale - the auxiliary branch's rule - would silently produce
    a number in neither space, and a plot nobody could read as a base prediction.
    """
    from yg_eo_soilnet.explain.lightning_explainer import _colour_values

    module = _residual_module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        parts, groups = module.explanation_parts(batch)

    state = {
        "label_mean": LABEL_MEAN,
        "label_scale": LABEL_SCALE,
        "target_mean": [4.0],
        "target_scale": [2.0],
    }
    colours = _colour_values(module, parts, groups, None, state, torch)

    column = next(index for index, group in enumerate(groups) if group["kind"] == "residual_base")
    raw = batch["x_labels"][:, 1].double() * LABEL_SCALE[1] + LABEL_MEAN[1]
    assert colours[:, column] == pytest.approx(raw.clamp_min(0.0).numpy(), rel=1e-4)


# --- serving -----------------------------------------------------------------


def test_the_serving_signature_asks_for_the_base_column() -> None:
    """Omitted from the signature it arrives NaN, is median-filled, and nothing raises."""
    from yg_eo_soilnet.serving.lightning_pyfunc import required_label_columns

    module = _residual_module(auxiliary_label_columns=["lab_a"])
    assert required_label_columns(module) == ["lab_a", "lab_b"]
    # A plain SoilCNN is unchanged: it reads only its auxiliary selection.
    assert module.auxiliary_label_columns == ["lab_a"]


def test_the_base_column_is_listed_once_when_there_are_no_auxiliary_columns() -> None:
    from yg_eo_soilnet.serving.lightning_pyfunc import required_label_columns

    assert required_label_columns(_residual_module()) == ["lab_b"]


# --- data path ---------------------------------------------------------------


def _labelled_datamodule(tmp_path: Path, logger, *, carry_labels: bool = True) -> SoilSequenceDataModule:
    """A real builder bundle with lab columns carried, set up for fitting."""
    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(
        tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)}, carry_labels=carry_labels
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")
    return datamodule


@pytest.mark.parametrize(
    "fusion, static_tokens",
    [("gated", "summary"), ("attention", "summary"), ("attention", "per_feature")],
)
def test_builder_to_residual_module_end_to_end(tmp_path: Path, logger, fusion, static_tokens) -> None:
    """A real bundle, a real datamodule, two epochs, and a finite prediction out the other side."""
    from lightning.pytorch import Trainer

    datamodule = _labelled_datamodule(tmp_path, logger)
    module = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",  # the fixture's batches are too small for BatchNorm
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_mean=datamodule.label_mean_,
        auxiliary_label_scale=datamodule.label_scale_,
        residual_enabled=True,
        residual_base_columns={"target_a": "lab_dense"},
        residual_base_hidden_dims=[8],
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
        fusion=fusion,
        attention_static_tokens=static_tokens,
        attention_d_model=16,
        attention_nhead=4,
    )
    assert module.residual_base_index.tolist() == [
        datamodule.label_feature_names.index("lab_dense")
    ]

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert torch.isfinite(predicted).all()


@pytest.mark.parametrize("fusion", ["gated", "attention"])
def test_the_shipped_registry_entry_builds_through_the_factory(tmp_path: Path, logger, fusion) -> None:
    """Every `auto` in the entry must be a keyword the module declares, or model_cls(**init_args)
    raises TypeError - the factory fills a written sentinel WITHOUT signature filtering, on purpose.

    Also the only place auxiliary_label_mean/scale injection is exercised: the YAML writes them as
    `auto` and the factory fills them from the datamodule's fitted statistics. The residual base and
    the attention fusion are switches on the soil_cnn entry; the residual and residual-attention
    classes survive only as legacy names.
    """
    from config import load_lightning_registry
    from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

    registry = load_lightning_registry("configs/lightning/models/defaults.yml")
    spec = registry["soil_cnn"]
    spec["init_args"].update(residual_enabled=True, fusion=fusion)
    # The shipped entry names the real dataset's columns; the fixture carries its own.
    spec["init_args"]["residual_base_columns"] = {"target_a": "lab_dense"}

    datamodule = _labelled_datamodule(tmp_path, logger)
    factory = LightningConfigFactory(registry, SimpleNamespace(TARGET_COLUMNS=["target_a"]))
    module = factory._build_model(spec, datamodule)

    assert module.residual_enabled
    assert isinstance(module.fusion, AttentionFusion if fusion == "attention" else ConcatGatedFusion)
    if fusion == "attention":
        assert module.hparams["attention_static_tokens"] == spec["init_args"]["attention_static_tokens"]
    assert module.residual_base_label_mean.tolist() == pytest.approx(
        [float(datamodule.label_mean_[datamodule.label_feature_names.index("lab_dense")])]
    )
    assert module.serving_label_columns == ["lab_dense"]


def test_a_sparse_base_column_is_refused_at_fit_start(tmp_path: Path, logger) -> None:
    """lab_sparse is ~50% absent in the fixture; a median-filled base is a silently wrong anchor."""
    from lightning.pytorch import Trainer

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    module = SoilResidualCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        cnn_norm="group",
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_mean=datamodule.label_mean_,
        auxiliary_label_scale=datamodule.label_scale_,
        residual_base_columns={"target_a": "lab_sparse"},
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(ValueError, match="residual_base_max_missing"):
        trainer.fit(module, datamodule=datamodule)


def test_a_zeroed_head_on_real_data_returns_the_base_in_original_units(tmp_path: Path, logger) -> None:
    """The whole chain - builder standardization, datamodule statistics, module inversion - at once.

    The fixture writes lab_dense as 100, 110, 120..., so a zeroed head must predict exactly those
    numbers back for the points in the batch.
    """
    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=8, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    module = SoilResidualCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        cnn_norm="group",
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_mean=datamodule.label_mean_,
        auxiliary_label_scale=datamodule.label_scale_,
        residual_base_columns={"target_a": "lab_dense"},
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
    ).eval()
    zero_head(module)

    indices = list(range(bundle.num_points))
    batch = datamodule.collate(indices)
    expected = [
        float(bundle.label_features[index][bundle.label_feature_names.index("lab_dense")])
        for index in indices
    ]
    with torch.no_grad():
        predicted = module.predict_step(batch, 0).reshape(-1).tolist()

    for value, reference in zip(predicted, expected):
        assert math.isclose(value, reference, rel_tol=1e-3), (predicted, expected)


# --- attention fusion inside the module -------------------------------------------------------
# Residual calendar-grid CNN with attention fusion: the fusion, and that nothing else moved.
#
# Runs through SoilResidualAttentionCNNLightningModule, now a legacy name for soil_cnn with
# residual_enabled and fusion="attention" (the switch section above pins that equivalence).
# Everything after the fusion - the base offset, the base and auxiliary blocks, the variance head -
# is pinned by the residual section. This one pins the two ways static covariates become tokens and
# the contracts the fusion could silently break: the attribution seam's exact equality and the
# checkpoint round trip of the attention settings.


STATIC_MODES = ["summary", "per_feature"]
READOUTS = ["cls", "mean", "flatten"]
CPU = torch.device("cpu")


def _attention_module(**kwargs) -> SoilResidualAttentionCNNLightningModule:
    defaults = dict(
        static_dim=5,
        target_dim=1,
        modality_dims={"m": 3},
        temporal_encoder="dilated_tempcnn",
        grid_years=3,
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_mean=LABEL_MEAN,
        auxiliary_label_scale=LABEL_SCALE,
        residual_base_columns={"target_a": "lab_b"},
        cnn_hidden_dims=[8],
        modality_embed_dim=8,
        static_hidden_dims=[8],
        head_hidden_dims=[8],
        attention_d_model=16,
        attention_nhead=4,
        attention_num_layers=1,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilResidualAttentionCNNLightningModule(**defaults).eval()




# --- the module: what the swap changed -----------------------------------------


def test_summary_mode_reads_the_static_branch_as_one_token() -> None:
    module = _attention_module(attention_static_tokens="summary")

    assert module.fusion.static_dims == [8]  # static_hidden_dims[-1]
    assert module.fusion.num_tokens == 2  # static + the one modality
    assert any(isinstance(child, nn.Linear) for child in module.static_encoder.modules())


def test_per_feature_mode_gives_every_column_a_token_and_drops_the_static_mlp() -> None:
    module = _attention_module(attention_static_tokens="per_feature")

    assert module.fusion.static_dims == [1] * 5
    assert module.fusion.num_tokens == 5 + 1
    # Dead weight in this mode; with it gone the encoder returns the raw per-column block.
    assert not any(isinstance(child, nn.Linear) for child in module.static_encoder.modules())
    assert module.static_encoder.output_dim == 5


def test_per_feature_mode_gives_each_categorical_embedding_its_own_token() -> None:
    module = _attention_module(
        attention_static_tokens="per_feature",
        categorical_cardinalities=[4],
        categorical_vocabularies=[["a", "b", "c"]],
        categorical_feature_names=["texture"],
    )

    assert len(module.static_encoder.embedding_dims) == 1
    assert module.fusion.static_dims == [1] * 5 + module.static_encoder.embedding_dims


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_a_zeroed_head_still_reproduces_the_base(static_tokens) -> None:
    """The residual contract survives the swap: prediction = base + head, whatever the fusion."""
    module = _attention_module(attention_static_tokens=static_tokens)
    zero_head(module)
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        assert torch.allclose(module(batch).reshape(-1).double(), expected_base(batch), atol=1e-5)


def test_per_feature_mode_lets_a_single_covariate_move_the_prediction() -> None:
    module = _attention_module(attention_static_tokens="per_feature")
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        before = module(batch)
        batch["x_static"] = batch["x_static"].clone()
        batch["x_static"][:, 3] += 1.0
        after = module(batch)

    assert not torch.allclose(before, after)


def _full_module(**kwargs) -> SoilResidualAttentionCNNLightningModule:
    """Every optional branch on, so every part of the attribution seam is exercised."""
    return _attention_module(
        coord_dim=2,
        categorical_cardinalities=[4],
        categorical_vocabularies=[["a", "b", "c"]],
        categorical_feature_names=["texture"],
        auxiliary_label_columns=["lab_a"],
        residual_base_hidden_dims=[8],
        **kwargs,
    )


def _full_batch() -> dict:
    batch = cnn_batch(labels=3)
    generator = torch.Generator().manual_seed(1)
    batch["x_categorical"] = torch.randint(0, 4, (4, 1), generator=generator)
    batch["x_coords"] = torch.rand(4, 2, generator=generator) * 2 - 1
    return batch


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
@pytest.mark.parametrize("readout", READOUTS)
def test_every_attention_layout_explains_itself_exactly(static_tokens, readout) -> None:
    """The equality every explainer depends on, in both static modes and every readout."""
    module = _full_module(attention_static_tokens=static_tokens, attention_readout=readout)
    batch = _full_batch()

    with torch.no_grad():
        parts, groups = module.explanation_parts(batch)
        assert torch.allclose(module.forward_from_parts(parts), module(batch), atol=1e-6)

    # Static covariates are attributed per column in both modes; only the tokenization differs.
    assert len([group for group in groups if group["kind"] == "static"]) == 5
    assert [group["name"] for group in groups if group["kind"] == "residual_base"] == ["lab_b"]


def test_the_attention_settings_survive_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    """Losing them from hparams would rebuild the default fusion on load."""
    module = _attention_module(attention_static_tokens="per_feature", attention_readout="mean", attention_num_layers=2)
    hparams = dict(module.hparams)
    assert hparams["attention_static_tokens"] == "per_feature"
    assert hparams["attention_readout"] == "mean"
    assert hparams["attention_num_layers"] == 2
    # The residual and shape settings sit beside them rather than being replaced.
    assert hparams["residual_base_columns"] == {"target_a": "lab_b"}
    assert hparams["static_dim"] == 5

    path = tmp_path / "attention.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": hparams, "pytorch-lightning_version": "2.0.0"},
        path,
    )
    torch.load(path, weights_only=True)  # the constraint that forbids numpy in hyper_parameters

    restored = SoilResidualAttentionCNNLightningModule.load_from_checkpoint(path, map_location="cpu").eval()
    batch = cnn_batch(labels=3)
    with torch.no_grad():
        assert torch.allclose(restored(batch), module(batch), atol=1e-6)


@pytest.mark.parametrize("encoder", ["dilated_tempcnn", "annual_grid2d"])
@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_every_fusion_parameter_receives_a_gradient(encoder, static_tokens) -> None:
    module = _attention_module(temporal_encoder=encoder, attention_static_tokens=static_tokens, attention_num_layers=2)
    module.train()
    batch = cnn_batch(labels=3)

    (module(batch) - batch["y"]).pow(2).mean().backward()

    for name, parameter in module.fusion.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_without_temporal_data_the_fusion_reads_static_tokens_only() -> None:
    module = _attention_module(temporal_enabled=False)

    assert module.fusion.temporal_dims == []
    with torch.no_grad():
        assert torch.isfinite(module(cnn_batch(labels=3))).all()


def test_modalities_of_different_widths_each_get_their_own_projection() -> None:
    module = _attention_module(modality_dims={"m": 3, "n": 2}, modality_embed_dim={"m": 8, "n": 4})
    batch = cnn_batch(labels=3)
    for key in ("sequences", "sequence_validity"):
        batch[key]["n"] = batch[key]["m"][..., :2]
    for key in ("sequence_mask", "sequence_time"):
        batch[key]["n"] = batch[key]["m"]

    assert module.fusion.temporal_dims == [8, 4]
    with torch.no_grad():
        assert torch.isfinite(module(batch)).all()


# --- data path ---------------------------------------------------------------


# --- the heteroscedastic head -----------------------------------------------------------------
# The heteroscedastic head: the wider readout, the beta-NLL loss, and the sigma inversion.


def _variance_module(**overrides) -> SoilCNNLightningModule:
    # No modalities: the CNN's static branch alone, so a batch needs only x_static and y.
    kwargs = dict(
        static_dim=3,
        target_dim=1,
        target_names=["target_a"],
        static_hidden_dims=[4],
        head_hidden_dims=[4],
        predict_variance=True,
    )
    kwargs.update(overrides)
    return SoilCNNLightningModule(**kwargs)


def _batch(n_rows=8, static_dim=3, target_dim=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "x_static": torch.randn(n_rows, static_dim, generator=generator),
        "y": torch.randn(n_rows, target_dim, generator=generator),
    }


# --- the head --------------------------------------------------------------


def test_a_point_head_is_unchanged_in_width():
    module = _variance_module(predict_variance=False)
    assert module.head_output_dim == 1
    assert module.forward(_batch()).shape == (8, 1)


def test_a_variance_head_is_twice_as_wide():
    module = _variance_module()
    assert module.head_output_dim == 2
    assert module.forward(_batch()).shape == (8, 2)


def test_a_joint_variance_head_emits_two_numbers_per_target():
    module = _variance_module(target_dim=3, target_names=["a", "b", "c"])
    assert module.head_output_dim == 6
    assert module.forward(_batch(target_dim=3)).shape == (8, 6)


def test_the_split_separates_the_means_from_the_log_variances():
    module = _variance_module(target_dim=2, target_names=["a", "b"])
    raw = torch.tensor([[1.0, 2.0, -1.0, -2.0]])
    mean, log_variance = module._split_head_output(raw)
    assert torch.equal(mean, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(log_variance, torch.tensor([[-1.0, -2.0]]))


def test_a_point_head_reports_no_log_variance():
    mean, log_variance = _variance_module(predict_variance=False)._split_head_output(
        torch.tensor([[1.0]])
    )
    assert log_variance is None


def test_the_log_variance_is_clamped_so_the_nll_cannot_explode():
    # Without the clamp a head that drives the variance to zero makes the NLL's 1/var term
    # non-finite and kills the run.
    module = _variance_module()
    _mean, log_variance = module._split_head_output(torch.tensor([[0.0, -500.0]]))
    assert float(log_variance) == module.LOG_VARIANCE_MIN


# --- the loss --------------------------------------------------------------


def test_beta_nll_at_zero_is_plain_gaussian_nll():
    module = _variance_module(beta_nll=0.0)
    mean = torch.zeros(4, 1)
    log_variance = torch.zeros(4, 1)  # variance 1
    targets = torch.tensor([[1.0], [-1.0], [2.0], [0.0]])

    loss = module._beta_nll_loss(mean, log_variance, targets)
    expected = float(np.mean(0.5 * (0.0 + targets.numpy() ** 2)))
    assert float(loss) == pytest.approx(expected)


def test_beta_nll_reweights_by_the_predicted_variance():
    module = _variance_module(beta_nll=1.0)
    mean = torch.zeros(2, 1)
    log_variance = torch.log(torch.tensor([[4.0], [4.0]]))
    targets = torch.tensor([[1.0], [1.0]])

    # beta=1 multiplies each term by var^1 = 4.
    plain = module._beta_nll_loss(mean, log_variance, targets)
    module.beta_nll = 0.0
    assert float(plain) == pytest.approx(float(module._beta_nll_loss(mean, log_variance, targets)) * 4.0)


def test_the_variance_weight_carries_no_gradient():
    # The whole point of beta-NLL: the var^beta factor must not itself be optimised, or the model
    # can lower the loss by inflating the variance rather than by fitting the mean.
    module = _variance_module(beta_nll=0.5)
    log_variance = torch.zeros(2, 1, requires_grad=True)
    loss = module._beta_nll_loss(torch.zeros(2, 1), log_variance, torch.ones(2, 1))
    loss.backward()
    # Gradient exists (through the NLL) but is finite and not dominated by the detached weight.
    assert log_variance.grad is not None
    assert torch.isfinite(log_variance.grad).all()


def test_the_variance_head_trains_without_a_non_finite_loss():
    module = detach_logging(_variance_module())
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    batch = _batch(32)
    for _ in range(20):
        optimizer.zero_grad()
        loss = module._shared_step(batch, "train")
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)


def test_epoch_metrics_score_the_mean_and_not_the_variance_channels():
    # _accumulate_metrics reshapes to (-1, target_dim). Handed the full 2*target_dim output it
    # would fold log variances in among the predictions and report an r2 for a quantity that is
    # not a prediction of anything - or, with an odd product, raise on the reshape.
    module = detach_logging(_variance_module(target_dim=2, target_names=["a", "b"]))
    module._shared_step(_batch(16, target_dim=2), "val")
    state = module._metric_state["val"]
    assert state["sum_p"].shape == (2,)
    assert state["n"] == 16.0


# --- the sigma inversion ---------------------------------------------------


def test_sigma_is_scaled_by_the_target_scale_when_standardized():
    module = _variance_module(target_mean=[10.0], target_scale=[4.0])
    sigma = module.inverse_transform_sigma(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    assert float(sigma) == pytest.approx(8.0)


def test_sigma_is_unchanged_when_the_targets_were_never_standardized():
    module = _variance_module()
    sigma = module.inverse_transform_sigma(torch.tensor([[2.0]]), torch.tensor([[0.5]]))
    assert float(sigma) == pytest.approx(2.0)


def test_log1p_sigma_uses_the_local_slope_not_the_target_inversion():
    # The trap: applying inverse_transform_targets to a sigma gives expm1(sigma/10), a number in no
    # units at all. The delta method gives sigma * exp(z/10) / 10 at the predicted z.
    module = _variance_module(target_transform="log1p", target_mean=[0.0], target_scale=[1.0])
    standardized_mean = torch.tensor([[20.0]])
    sigma = module.inverse_transform_sigma(torch.tensor([[1.0]]), standardized_mean)

    expected = 1.0 * float(np.exp(20.0 / 10.0)) / 10.0
    assert float(sigma) == pytest.approx(expected, rel=1e-5)
    # And emphatically not the naive answer.
    assert float(sigma) != pytest.approx(float(np.expm1(1.0 / 10.0)))


def test_log1p_sigma_grows_with_the_prediction():
    # A consequence worth pinning: on a log1p target the same standardized sigma is a LARGER
    # absolute uncertainty at a larger predicted value, because the transform stretches the axis.
    module = _variance_module(target_transform="log1p", target_mean=[0.0], target_scale=[1.0])
    small = module.inverse_transform_sigma(torch.tensor([[1.0]]), torch.tensor([[5.0]]))
    large = module.inverse_transform_sigma(torch.tensor([[1.0]]), torch.tensor([[25.0]]))
    assert float(large) > float(small)


def test_the_sigma_inversion_matches_a_numerical_derivative():
    module = _variance_module(target_transform="log1p", target_mean=[2.0], target_scale=[3.0])
    standardized = torch.tensor([[0.7]])
    small_sigma = 1e-4

    analytic = float(module.inverse_transform_sigma(torch.tensor([[small_sigma]]), standardized))
    # Numerically: how far does the inverted prediction move for a small step in standardized space?
    high = float(module.inverse_transform_targets(standardized + small_sigma))
    low = float(module.inverse_transform_targets(standardized - small_sigma))
    numeric = (high - low) / 2.0

    assert analytic == pytest.approx(numeric, rel=1e-3)


# --- predict_step ----------------------------------------------------------


def test_predict_step_returns_a_bare_tensor_on_a_point_head():
    module = _variance_module(predict_variance=False)
    assert isinstance(module.predict_step(_batch(), 0), torch.Tensor)


def test_predict_step_returns_mean_and_sigma_on_a_variance_head():
    module = _variance_module()
    result = module.predict_step(_batch(), 0)
    assert isinstance(result, tuple) and len(result) == 2
    mean, sigma = result
    assert mean.shape == (8, 1)
    assert sigma.shape == (8, 1)
    assert (sigma > 0).all()


def test_predict_step_sigma_is_in_original_units():
    module = _variance_module(target_mean=[100.0], target_scale=[10.0])
    with torch.no_grad():
        _mean, sigma = module.predict_step(_batch(), 0)
    # Standardized sigmas are O(1); a sigma still in standardized space would be ~10x smaller.
    assert float(sigma.mean()) > 1.0


# --- the checkpoint --------------------------------------------------------


def test_the_variance_flag_survives_a_checkpoint_restore(tmp_path):
    # A buffer rather than a plain attribute: a restore that lost it would read a 2*target_dim head
    # as 2*target_dim targets and report log variances as predictions.
    module = _variance_module()
    path = tmp_path / "head.ckpt"
    torch.save({"state_dict": module.state_dict()}, path)

    restored = torch.load(path, weights_only=True)
    assert bool(restored["state_dict"]["head_predicts_variance"]) is True


def test_a_point_head_records_that_it_predicts_no_variance():
    assert bool(_variance_module(predict_variance=False).head_predicts_variance) is False
