"""Reloading a trained deep-learning model and predicting with it.

The gap this closes: the checkpoint always carried the weights and the TARGET inverse-transform
(both buffers), but the INPUT standardization and the categorical vocabulary were fitted on the
datamodule and thrown away with it. A restored model could therefore only be fed data some
datamodule had already scaled - which is to say, it could not be deployed the way a pickled sklearn
Pipeline can.

The load-bearing test here is the round-trip one: save, reload, predict from a raw bundle, and get
back what trainer.predict produced on the same points.
"""

import numpy as np
import pytest
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving import SoilSequencePredictor

from tests.support.builders import sequence_bundle, tiny_cnn

STATIC_NAMES = ["clay_pct", "ph"]
S2_BANDS = ["S2_B02", "S2_B08"]
N_POINTS = 24


@pytest.fixture
def cnn() -> tuple[SoilCNNLightningModule, SoilSequenceDataModule, SoilSequenceBundle]:
    bundle = sequence_bundle(
        n_points=N_POINTS, static=STATIC_NAMES, modalities={"s2": S2_BANDS}, observations=(3, 8)
    )
    model, datamodule = tiny_cnn(bundle)
    return model, datamodule, bundle


def test_preprocessing_state_is_plain_builtins(cnn) -> None:
    """A numpy array in hyper_parameters makes the checkpoint unloadable under weights_only=True."""
    _model, datamodule, _bundle_ = cnn
    state = datamodule.preprocessing_state()

    def assert_plain(value, path="state"):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), f"{path} has a non-string key {key!r}"
                assert_plain(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                assert_plain(item, f"{path}[{index}]")
        else:
            assert isinstance(value, (str, int, float, bool)), f"{path} is {type(value).__name__}"

    assert_plain(state)


def test_preprocessing_state_carries_the_input_scalers(cnn) -> None:
    _model, datamodule, _bundle_ = cnn
    state = datamodule.preprocessing_state()

    assert len(state["static_mean"]) == len(STATIC_NAMES)
    assert len(state["static_scale"]) == len(STATIC_NAMES)
    assert state["static_feature_names"] == STATIC_NAMES
    assert state["modality_column_names"] == {"s2": S2_BANDS}
    assert len(state["sequence_mean"]["s2"]) == len(S2_BANDS)
    assert state["categorical_vocabularies"] and state["categorical_feature_names"] == ["texture"]


def test_a_reloaded_checkpoint_predicts_the_same_values(cnn, tmp_path) -> None:
    """The round trip: save, reload with weights_only semantics, predict from a RAW bundle."""
    model, _datamodule, bundle = cnn

    before = SoilSequencePredictor(model).predict(bundle)

    checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
    model.on_save_checkpoint(checkpoint)

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, checkpoint_path)
    # weights_only=True is torch's default from 2.6 on, so anything the checkpoint carries has to
    # survive it. This is why preprocessing_state is plain builtins.
    payload = torch.load(checkpoint_path, weights_only=True)

    assert "preprocessing_state" in payload

    restored = SoilCNNLightningModule(**payload["hyper_parameters"])
    restored.load_state_dict(payload["state_dict"])
    restored.on_load_checkpoint(payload)
    restored.eval()

    after = SoilSequencePredictor(restored).predict(bundle)

    assert np.allclose(before, after)


def test_predictions_come_back_in_original_target_units(cnn) -> None:
    """predict_step inverts the standardization; forward alone would return standardized values."""
    model, _datamodule, bundle = cnn

    predictions = SoilSequencePredictor(model).predict(bundle)

    datamodule = SoilSequenceDataModule(sequence_bundle=bundle, batch_size=8)
    datamodule.apply_preprocessing_state(model.get_preprocessing_state())
    with torch.no_grad():
        standardized = model(datamodule.collate(np.arange(bundle.num_points))).numpy()

    assert predictions.shape == (bundle.num_points, 1)
    assert not np.allclose(predictions, standardized)


def test_scalers_are_not_refitted_on_the_incoming_points(cnn) -> None:
    """A serving batch is not a training split.

    Predicting for a subset must give each point the same answer it gets in the full batch. If the
    scaler were refitted per request, a point's prediction would depend on which other points
    happened to arrive with it.
    """
    model, _datamodule, bundle = cnn
    predictor = SoilSequencePredictor(model)

    full = predictor.predict(bundle)

    subset = SoilSequenceBundle.from_mapping(
        {
            **{field: getattr(bundle, field) for field in bundle.keys()},
            "point_ids": bundle.point_ids[:4],
            "static_features": bundle.static_features[:4],
            "static_categoricals": bundle.static_categoricals[:4],
            "targets": bundle.targets[:4],
            "sequences": {"s2": bundle.sequences["s2"][:4]},
            "sequence_times": {"s2": bundle.sequence_times["s2"][:4]},
        }
    )

    assert np.allclose(predictor.predict(subset), full[:4], atol=1e-5)


def test_a_single_point_is_scored_consistently(cnn) -> None:
    model, _datamodule, bundle = cnn
    predictor = SoilSequencePredictor(model)
    full = predictor.predict(bundle)

    single = SoilSequenceBundle.from_mapping(
        {
            **{field: getattr(bundle, field) for field in bundle.keys()},
            "point_ids": bundle.point_ids[:1],
            "static_features": bundle.static_features[:1],
            "static_categoricals": bundle.static_categoricals[:1],
            "targets": bundle.targets[:1],
            "sequences": {"s2": bundle.sequences["s2"][:1]},
            "sequence_times": {"s2": bundle.sequence_times["s2"][:1]},
        }
    )

    assert np.allclose(predictor.predict(single), full[:1], atol=1e-5)


def test_an_unseen_category_lands_on_the_reserved_index_rather_than_shifting_the_others(
    cnn,
) -> None:
    model, _datamodule, bundle = cnn
    predictor = SoilSequencePredictor(model)

    unseen = SoilSequenceBundle.from_mapping(
        {**{field: getattr(bundle, field) for field in bundle.keys()}}
    )
    unseen.static_categoricals = np.asarray([["volcanic"]] * bundle.num_points, dtype=object)

    predictions = predictor.predict(unseen)

    assert predictions.shape == (bundle.num_points, 1)
    assert np.isfinite(predictions).all()


def test_predict_frame_labels_columns_with_the_target_names(cnn) -> None:
    model, _datamodule, bundle = cnn

    frame = SoilSequencePredictor(model).predict_frame(bundle)

    assert list(frame.columns) == ["organic_matter_pct"]
    assert list(frame.index) == list(bundle.point_ids)


def test_a_checkpoint_without_the_state_is_refused_with_an_actionable_message() -> None:
    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)

    with pytest.raises(ValueError, match="no preprocessing state"):
        SoilSequencePredictor(model)


def test_an_empty_bundle_predicts_nothing_rather_than_raising(cnn) -> None:
    model, _datamodule, _bundle_ = cnn

    predictions = SoilSequencePredictor(model).predict(SoilSequenceBundle())

    assert predictions.shape[0] == 0
