"""Residual calendar-grid CNN: the base-prediction anchor and everything it touches.

The risk here is entirely arithmetic and entirely silent. The base arrives standardized by the LAB
standardizer while `y` lives in the TARGET's space, so a missing hop - or the two applied in the
wrong order - still produces a well-shaped tensor and a loss that goes down. The first section
pins the round trip against a head that has been zeroed, which is the only way to see the offset on
its own.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)

from tests.support.cnn import LABEL_MEAN, LABEL_NAMES, LABEL_SCALE, built_sequence_bundle, cnn_batch, expected_base, zero_head


def _module(**kwargs) -> SoilResidualCNNLightningModule:
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
    module = _module()
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
    module = _module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)

    expected = expected_base(batch, target_mean=4.0, target_scale=2.0, log1p=True)
    assert torch.allclose(module(batch).reshape(-1).double(), expected, atol=1e-5)


def test_predict_step_inverts_back_to_the_bases_original_units() -> None:
    """End to end: a zeroed head served through predict_step returns the base as it was written."""
    module = _module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)

    raw = batch["x_labels"][:, 1].double() * LABEL_SCALE[1] + LABEL_MEAN[1]
    assert torch.allclose(
        module.predict_step(batch, 0).reshape(-1).double(), raw.clamp_min(0.0), atol=1e-4
    )


def test_a_negative_base_is_clipped_the_way_a_measured_target_would_be() -> None:
    """log1p is undefined below -1, and the datamodule clips a negative target to 0 rather than fail."""
    module = _module(target_transform="log1p")
    zero_head(module)
    batch = cnn_batch(labels=3)
    # -100 in lab-standardized space is far below the column's mean of 30.
    batch["x_labels"][:, 1] = -100.0

    assert torch.allclose(module(batch), torch.zeros_like(module(batch)))


def test_the_offset_lands_on_the_mean_half_only_on_a_variance_head() -> None:
    """A 2*target_dim readout: shifting the log variances too would make exp(40) a variance."""
    module = _module(predict_variance=True, target_dim=1)
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
    module = _module(residual_base_hidden_dims=[8])
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
    module = _module()
    batch = cnn_batch(labels=3)
    block = module._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32)
    assert block.shape[-1] == 2

    without = _module(residual_base_validity_channels=False)
    assert without._residual_base(batch, device=torch.device("cpu"), dtype=torch.float32).shape[-1] == 1


def test_a_multi_target_offset_is_ordered_by_target_names() -> None:
    """Position j of the offset must line up with output column j, not with the roster's order."""
    module = _module(
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


def test_an_empty_mapping_is_refused_rather_than_degrading_to_soil_cnn() -> None:
    with pytest.raises(ValueError, match="residual_base_columns is required"):
        _module(residual_base_columns={})


def test_a_target_with_no_base_column_is_refused() -> None:
    with pytest.raises(ValueError, match="no entry for target"):
        _module(
            target_dim=2,
            target_names=["target_a", "target_b"],
            residual_base_columns={"target_a": "lab_b"},
        )


def test_a_base_column_that_is_a_fitted_target_is_refused() -> None:
    """The measured target itself, not a prediction of it - the inherited check never sees this map."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _module(
            auxiliary_available_names=["target_a", "lab_b", "lab_c"],
            residual_base_columns={"target_a": "target_a"},
        )


def test_a_sibling_target_is_refused_as_a_base_under_per_target_grouping() -> None:
    """fitted_target_names is the whole run's targets, so a sibling leaks just as surely."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _module(
            target_names=["target_a"],
            fitted_target_names=["target_a", "lab_b"],
            residual_base_columns={"target_a": "lab_b"},
        )


def test_an_unknown_base_column_is_refused() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        _module(residual_base_columns={"target_a": "lab_z"})


def test_a_column_in_both_roles_is_refused() -> None:
    with pytest.raises(ValueError, match="both residual_base_columns and auxiliary_label_columns"):
        _module(auxiliary_label_columns=["lab_b"], residual_base_columns={"target_a": "lab_b"})


def test_an_empty_roster_names_the_flag_that_fixes_it() -> None:
    with pytest.raises(ValueError, match="CARRY_LABEL_COLUMNS"):
        _module(auxiliary_available_names=[])


def test_missing_roster_statistics_are_refused() -> None:
    """Without them the base never leaves the lab standardizer, and the offset is a z-score."""
    with pytest.raises(ValueError, match="auxiliary_label_mean and auxiliary_label_scale"):
        _module(auxiliary_label_mean=None, auxiliary_label_scale=None)


# --- checkpoint --------------------------------------------------------------


def test_the_mapping_survives_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    """The residual settings must round-trip through hyper_parameters along with everything else.

    Losing residual_base_columns on reload would rebuild the module with an empty mapping, which now
    raises - but losing static_dim would rebuild a differently shaped network that still loads.
    """
    module = _module(residual_base_hidden_dims=[8])
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
    module = _module(residual_base_hidden_dims=[8])
    batch = cnn_batch(labels=3)

    with torch.no_grad():
        parts, groups = module.explanation_parts(batch)
        assert torch.allclose(module.forward_from_parts(parts), module(batch), atol=1e-6)

    base_groups = [group for group in groups if group["kind"] == "residual_base"]
    assert [group["name"] for group in base_groups] == ["lab_b"]
    assert base_groups[0]["columns"] == [0, 1]


def test_the_base_part_carries_the_offset_so_perturbing_it_moves_the_prediction() -> None:
    """The offset is derived from the PART, not the batch - a shap interpolation has no batch."""
    module = _module()
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

    module = _module(target_mean=[4.0], target_scale=[2.0], target_transform="log1p")
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

    module = _module(auxiliary_label_columns=["lab_a"])
    assert required_label_columns(module) == ["lab_a", "lab_b"]
    # A plain SoilCNN is unchanged: it reads only its auxiliary selection.
    assert module.auxiliary_label_columns == ["lab_a"]


def test_the_base_column_is_listed_once_when_there_are_no_auxiliary_columns() -> None:
    from yg_eo_soilnet.serving.lightning_pyfunc import required_label_columns

    assert required_label_columns(_module()) == ["lab_b"]


# --- data path ---------------------------------------------------------------


def test_builder_to_residual_module_end_to_end(tmp_path: Path, logger) -> None:
    """A real bundle, a real datamodule, two epochs, and a finite prediction out the other side."""
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
        temporal_encoder="annual_grid2d",
        cnn_norm="group",  # the fixture's batches are too small for BatchNorm
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_mean=datamodule.label_mean_,
        auxiliary_label_scale=datamodule.label_scale_,
        residual_base_columns={"target_a": "lab_dense"},
        residual_base_hidden_dims=[8],
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
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


def test_the_shipped_registry_entry_builds_through_the_factory(tmp_path: Path, logger) -> None:
    """Every `auto` in the entry must be a keyword the module declares, or model_cls(**init_args)
    raises TypeError - the factory fills a written sentinel WITHOUT signature filtering, on purpose.

    Also the only place auxiliary_label_mean/scale injection is exercised: the YAML writes them as
    `auto` and the factory fills them from the datamodule's fitted statistics. The residual base is a
    switch on the soil_cnn entry now; soil_residual_cnn survives only as a legacy class name.
    """
    from config import load_lightning_registry
    from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

    registry = load_lightning_registry("configs/lightning/models/defaults.yml")
    spec = registry["soil_cnn"]
    spec["init_args"]["residual_enabled"] = True
    # The shipped entry names the real dataset's columns; the fixture carries its own.
    spec["init_args"]["residual_base_columns"] = {"target_a": "lab_dense"}

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = built_sequence_bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    factory = LightningConfigFactory(registry, SimpleNamespace(TARGET_COLUMNS=["target_a"]))
    module = factory._build_model(spec, datamodule)

    assert module.residual_enabled
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
