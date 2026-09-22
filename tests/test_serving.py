"""Serving the sequence model: the MLflow pyfunc contract, and the predictor it wraps.

The pyfunc contract for the sequence models.

Why a pyfunc at all: MLflow 3 defaults `mlflow.pytorch.log_model` to `serialization_format="pt2"`,
which traces `model.forward` from an example input. These models consume a dict batch of ragged,
date-stamped sequences, so nothing can trace them - the run failed with "If serialization_format is
set to 'pt2', then input_example is required" and left the logged model in status FAILED. A pyfunc
sidesteps tracing and, unlike a bare checkpoint, carries the preprocessing needed to consume raw
data.

The load-bearing test is the round trip: build an example, predict through the wrapper, and get the
same numbers as SoilSequencePredictor on the same points.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tests.support.builders import sequence_bundle, tiny_cnn
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving import SoilSequencePredictor
from yg_eo_soilnet.serving.lightning_pyfunc import (
    SoilSequencePyfunc,
    build_input_example,
    bundle_from_frame,
    frame_from_bundle,
    time_column,
    values_column,
)

STATIC = ["clay_pct", "ph"]
BANDS = ["S2_B02", "S2_B08"]
N_POINTS = 16
# The full lab roster the bundle carries when CARRY_LABEL_COLUMNS is on. It includes the target,
# which is exactly why the serving contract must never ask for all of it.
LAB_ROSTER = ["organic_matter_pct", "ph_lab", "clay_lab", "sand_lab"]


@pytest.fixture
def cnn():
    bundle = sequence_bundle(n_points=N_POINTS, static=STATIC, modalities={"s2": BANDS})
    model, _datamodule = tiny_cnn(bundle)
    return model, bundle


def test_input_example_carries_the_documented_columns(cnn) -> None:
    model, bundle = cnn
    example = build_input_example(model, bundle, n_rows=3)

    assert len(example) == 3
    for name in ["point_id", *STATIC, "texture", time_column("s2"), values_column("s2")]:
        assert name in example.columns, name

    # Ragged sequences travel as nested lists: one time per observation, one value per band.
    times = example[time_column("s2")].iloc[0]
    values = example[values_column("s2")].iloc[0]
    assert len(times) == len(values)
    assert all(len(row) == len(BANDS) for row in values)


def test_the_wrapper_matches_the_predictor_on_the_same_points(cnn) -> None:
    """The round trip that proves the contract reconstructs the model's real inputs."""
    model, bundle = cnn

    example = build_input_example(model, bundle, n_rows=N_POINTS)
    through_pyfunc = SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float)
    through_predictor = SoilSequencePredictor(model).predict(bundle)

    assert np.allclose(through_pyfunc, through_predictor, atol=1e-5)


def test_predictions_are_labelled_with_the_target_names(cnn) -> None:
    model, bundle = cnn
    example = build_input_example(model, bundle, n_rows=2)

    assert list(SoilSequencePyfunc(model).predict(None, example).columns) == ["organic_matter_pct"]


def test_a_frame_round_trips_through_the_bundle(cnn) -> None:
    model, bundle = cnn
    state = model.get_preprocessing_state()

    rebuilt = bundle_from_frame(frame_from_bundle(bundle, state), state)

    assert rebuilt.num_points == bundle.num_points
    assert rebuilt.static_feature_names == bundle.static_feature_names
    assert rebuilt.modality_columns == bundle.modality_columns
    assert np.allclose(rebuilt.static_features, bundle.static_features, atol=1e-5)
    for original, restored in zip(bundle.sequences["s2"], rebuilt.sequences["s2"]):
        assert restored.shape == original.shape
        assert np.allclose(restored, original, atol=1e-5)


def test_a_missing_static_column_is_refused_by_name(cnn) -> None:
    model, bundle = cnn
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state).drop(columns=["ph"])

    with pytest.raises(KeyError, match="ph"):
        bundle_from_frame(example, state)


def test_a_missing_modality_column_is_refused_with_the_band_order(cnn) -> None:
    model, bundle = cnn
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state).drop(columns=[values_column("s2")])

    with pytest.raises(KeyError, match="s2__values"):
        bundle_from_frame(example, state)


def test_misaligned_times_and_values_are_refused(cnn) -> None:
    model, bundle = cnn
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state)
    example.at[0, time_column("s2")] = list(example[time_column("s2")].iloc[0])[:-1]

    with pytest.raises(ValueError, match="line up"):
        bundle_from_frame(example, state)


def test_a_point_with_no_observations_is_accepted(cnn) -> None:
    """A real request can carry a point that has never been imaged; it must not crash the batch."""
    model, bundle = cnn
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state, n_rows=4)
    example.at[0, time_column("s2")] = []
    example.at[0, values_column("s2")] = []

    predictions = SoilSequencePyfunc(model).predict(None, example)

    assert len(predictions) == 4
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()


def test_signature_inference_produces_array_types(cnn) -> None:
    """MLflow has to describe the nested columns, or a served request cannot be validated."""
    from mlflow.models import infer_signature

    model, bundle = cnn
    example = build_input_example(model, bundle, n_rows=3)
    predictions = SoilSequencePyfunc(model).predict(None, example)

    rendered = str(infer_signature(example, predictions).inputs)

    assert "Array(double)" in rendered
    assert "Array(Array(double))" in rendered


def test_a_model_without_preprocessing_state_cannot_build_an_example() -> None:
    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)

    with pytest.raises(ValueError, match="no preprocessing state"):
        build_input_example(model, SoilSequenceBundle(), n_rows=1)


def test_frame_from_bundle_honours_the_row_cap(cnn) -> None:
    model, bundle = cnn
    assert len(frame_from_bundle(bundle, model.get_preprocessing_state(), n_rows=2)) == 2
    # Asking for more rows than exist yields what there is, rather than raising.
    assert len(frame_from_bundle(bundle, model.get_preprocessing_state(), n_rows=999)) == N_POINTS


def test_extra_columns_in_a_request_are_ignored(cnn) -> None:
    """A caller sending a wider frame must not reshape the model's inputs."""
    model, bundle = cnn
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state, n_rows=3)
    example["an_unrelated_column"] = 1.0

    rebuilt = bundle_from_frame(example, state)

    assert rebuilt.static_feature_names == STATIC
    assert rebuilt.static_features.shape[1] == len(STATIC)


def test_predict_accepts_a_plain_dict_of_columns(cnn) -> None:
    """MLflow hands scoring payloads over as records; a DataFrame constructor must cover it."""
    model, bundle = cnn
    example = build_input_example(model, bundle, n_rows=2)

    from_records = SoilSequencePyfunc(model).predict(None, pd.DataFrame(example.to_dict("list")))

    assert np.allclose(
        from_records.to_numpy(dtype=float),
        SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float),
    )


# --- models that USE auxiliary lab values ------------------------------------
# The shape every test above misses: with auxiliary_label_columns=[], _select_auxiliary returns
# before the roster-width check, so the path that broke in production is never executed here.

AUXILIARY = ["ph_lab", "clay_lab"]          # a strict subset of LAB_ROSTER, like soil_cnn-58e689


@pytest.fixture
def cnn_with_auxiliary():
    """A model that reads a subset of the lab roster, as the tuned config does."""
    bundle = sequence_bundle(
        n_points=N_POINTS,
        static=STATIC,
        modalities={"s2": BANDS},
        categorical=False,
        lab_roster=LAB_ROSTER,
        observations=5,
    )
    model, _datamodule = tiny_cnn(bundle, auxiliary=AUXILIARY)
    return model, bundle


def test_the_example_asks_only_for_the_columns_the_model_reads(cnn_with_auxiliary) -> None:
    model, bundle = cnn_with_auxiliary
    columns = set(build_input_example(model, bundle, n_rows=3).columns)

    assert set(AUXILIARY) <= columns
    # The roster columns it does not read stay out of the contract, and so does the target.
    assert "sand_lab" not in columns
    assert "organic_matter_pct" not in columns


def test_the_rebuilt_bundle_keeps_the_full_roster_width(cnn_with_auxiliary) -> None:
    """THE regression. The model index_selects roster POSITIONS, so a narrower block is unusable.

    Before the fix this came back (3, 0), and the model rejected the batch with
    'Batch carries 0 lab column(s) ... against 6' - which is what stopped every model being logged
    and therefore registered.
    """
    model, bundle = cnn_with_auxiliary
    state = model.get_preprocessing_state()

    rebuilt = bundle_from_frame(build_input_example(model, bundle, n_rows=3), state)

    assert np.asarray(rebuilt.label_features).shape == (3, len(LAB_ROSTER))


def test_supplied_values_land_at_their_roster_positions(cnn_with_auxiliary) -> None:
    """Position, not order of appearance - a shifted column would feed the model the wrong
    measurement without raising anything."""
    model, bundle = cnn_with_auxiliary
    state = model.get_preprocessing_state()
    example = build_input_example(model, bundle, n_rows=3)
    example["clay_lab"] = [111.0, 222.0, 333.0]

    rebuilt = bundle_from_frame(example, state)
    column = np.asarray(rebuilt.label_features)[:, LAB_ROSTER.index("clay_lab")]

    assert np.allclose(column, [111.0, 222.0, 333.0])
    # A column the caller did not supply is absent, not zero: NaN is the bundle's own convention,
    # and the datamodule median-fills it and flags it as unmeasured.
    assert np.isnan(np.asarray(rebuilt.label_features)[:, LAB_ROSTER.index("sand_lab")]).all()


def test_a_model_with_auxiliary_columns_still_predicts(cnn_with_auxiliary) -> None:
    model, bundle = cnn_with_auxiliary
    example = build_input_example(model, bundle, n_rows=N_POINTS)

    through_pyfunc = SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float)
    through_predictor = SoilSequencePredictor(model).predict(bundle)

    assert np.allclose(through_pyfunc, through_predictor, atol=1e-5)


# --- models that USE harmonic coordinates ------------------------------------
# The same shape as the auxiliary section above, and it broke the same way. With coord_dim=0
# _select_coordinates returns before the width check, so no test above executes the path at all -
# and every model logged under USE_HARMONIC_COORDS failed with "Batch carries 0 coordinate
# column(s) but this model was built for 2", which stopped registration for four runs while each
# one still finished green.

COORD_NAMES = ["lat", "lon"]


@pytest.fixture
def cnn_with_coords():
    """A model with the harmonic coordinate branch, as USE_HARMONIC_COORDS produces."""
    bundle = sequence_bundle(
        n_points=N_POINTS, static=STATIC, modalities={"s2": BANDS}, coord_names=COORD_NAMES
    )
    model, datamodule = tiny_cnn(bundle)
    return model, bundle, datamodule


def test_a_coordinate_model_can_be_logged_at_all(cnn_with_coords) -> None:
    """The regression test proper.

    This is mlflow_loggers._log_lightning_serialized_model's own sequence - build the example, then
    predict through the wrapper to infer a signature. It raised "Batch carries 0 coordinate
    column(s) but this model was built for 2", model logging was skipped, and because registration
    is downstream of logging no version was ever created.
    """
    model, bundle, _datamodule = cnn_with_coords
    example = build_input_example(model, bundle, n_rows=3)

    predictions = SoilSequencePyfunc(model).predict(None, example)

    assert len(predictions) == 3
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()


def test_the_example_carries_the_coordinate_columns(cnn_with_coords) -> None:
    model, bundle, _datamodule = cnn_with_coords
    example = build_input_example(model, bundle, n_rows=3)

    for name in COORD_NAMES:
        assert name in example.columns, f"{name} missing from the serving contract"
    np.testing.assert_allclose(example["lat"].to_numpy(), bundle.coords[:3, 0])


def test_coordinates_survive_the_round_trip_without_losing_precision(cnn_with_coords) -> None:
    """float64 end to end. float32 resolves about a metre here, and the train-bbox normalization
    downstream subtracts two nearby numbers, so it would spend most of that."""
    model, bundle, _datamodule = cnn_with_coords
    state = model.get_preprocessing_state()

    rebuilt = bundle_from_frame(frame_from_bundle(bundle, state), state)

    assert rebuilt.coord_names == COORD_NAMES
    assert rebuilt.coords.dtype == np.float64
    np.testing.assert_array_equal(rebuilt.coords, bundle.coords)


def test_a_round_tripped_point_normalizes_exactly_as_it_did_in_training(
    cnn_with_coords,
) -> None:
    """The property that actually matters: the served point must land on the same spot of the
    train bounding box it occupied during training, not merely carry the same degrees."""
    model, bundle, datamodule = cnn_with_coords
    state = model.get_preprocessing_state()

    served = SoilSequenceDataModule(
        sequence_bundle=bundle_from_frame(frame_from_bundle(bundle, state), state), batch_size=4
    )
    served.apply_preprocessing_state(state)

    torch.testing.assert_close(
        served.collate(np.arange(4))["x_coords"], datamodule.collate(np.arange(4))["x_coords"]
    )


def test_a_missing_coordinate_column_is_refused_by_name(cnn_with_coords) -> None:
    """Required, not optional: there is no honest fill for a position, so it must fail loudly
    rather than reach the model as a zero-width block - which is what it used to do."""
    model, bundle, _datamodule = cnn_with_coords
    state = model.get_preprocessing_state()
    without_latitude = frame_from_bundle(bundle, state).drop(columns=["lat"])

    with pytest.raises(KeyError, match="lat"):
        bundle_from_frame(without_latitude, state)


def test_the_wrapper_matches_the_predictor_for_a_coordinate_model(cnn_with_coords) -> None:
    """The frame path and the bundle path must agree. Only the frame path lost the coordinates,
    so a disagreement here is exactly the bug returning."""
    model, bundle, _datamodule = cnn_with_coords
    example = build_input_example(model, bundle, n_rows=N_POINTS)

    through_pyfunc = SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float)
    through_predictor = SoilSequencePredictor(model).predict(bundle)

    assert np.allclose(through_pyfunc, through_predictor, atol=1e-5)


def test_a_model_without_coordinates_asks_for_none(cnn) -> None:
    """The backward-compatibility guarantee: a checkpoint trained before the branch, or with the
    flag off, has no coord_names in its state, so the contract is unchanged."""
    model, bundle = cnn
    state = model.get_preprocessing_state()
    frame = frame_from_bundle(bundle, state)

    assert not [name for name in frame.columns if name in COORD_NAMES]
    assert bundle_from_frame(frame, state).coord_dim == 0


# --- the predictor: reloading a checkpoint and predicting from raw data -----------------------
# Reloading a trained deep-learning model and predicting with it.
#
# The gap this closes: the checkpoint always carried the weights and the TARGET inverse-transform
# (both buffers), but the INPUT standardization and the categorical vocabulary were fitted on the
# datamodule and thrown away with it. A restored model could therefore only be fed data some
# datamodule had already scaled - which is to say, it could not be deployed the way a pickled sklearn
# Pipeline can.
#
# The load-bearing test here is the round-trip one: save, reload, predict from a raw bundle, and get
# back what trainer.predict produced on the same points.


@pytest.fixture
def cnn_with_datamodule() -> tuple[SoilCNNLightningModule, SoilSequenceDataModule, SoilSequenceBundle]:
    bundle = sequence_bundle(
        n_points=24, static=STATIC, modalities={"s2": BANDS}, observations=(3, 8)
    )
    model, datamodule = tiny_cnn(bundle)
    return model, datamodule, bundle


def test_preprocessing_state_is_plain_builtins(cnn_with_datamodule) -> None:
    """A numpy array in hyper_parameters makes the checkpoint unloadable under weights_only=True."""
    _model, datamodule, _bundle_ = cnn_with_datamodule
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


def test_preprocessing_state_carries_the_input_scalers(cnn_with_datamodule) -> None:
    _model, datamodule, _bundle_ = cnn_with_datamodule
    state = datamodule.preprocessing_state()

    assert len(state["static_mean"]) == len(STATIC)
    assert len(state["static_scale"]) == len(STATIC)
    assert state["static_feature_names"] == STATIC
    assert state["modality_column_names"] == {"s2": BANDS}
    assert len(state["sequence_mean"]["s2"]) == len(BANDS)
    assert state["categorical_vocabularies"] and state["categorical_feature_names"] == ["texture"]


def test_a_reloaded_checkpoint_predicts_the_same_values(cnn_with_datamodule, tmp_path) -> None:
    """The round trip: save, reload with weights_only semantics, predict from a RAW bundle."""
    model, _datamodule, bundle = cnn_with_datamodule

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


def test_predictions_come_back_in_original_target_units(cnn_with_datamodule) -> None:
    """predict_step inverts the standardization; forward alone would return standardized values."""
    model, _datamodule, bundle = cnn_with_datamodule

    predictions = SoilSequencePredictor(model).predict(bundle)

    datamodule = SoilSequenceDataModule(sequence_bundle=bundle, batch_size=8)
    datamodule.apply_preprocessing_state(model.get_preprocessing_state())
    with torch.no_grad():
        standardized = model(datamodule.collate(np.arange(bundle.num_points))).numpy()

    assert predictions.shape == (bundle.num_points, 1)
    assert not np.allclose(predictions, standardized)


def test_scalers_are_not_refitted_on_the_incoming_points(cnn_with_datamodule) -> None:
    """A serving batch is not a training split.

    Predicting for a subset must give each point the same answer it gets in the full batch. If the
    scaler were refitted per request, a point's prediction would depend on which other points
    happened to arrive with it.
    """
    model, _datamodule, bundle = cnn_with_datamodule
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


def test_a_single_point_is_scored_consistently(cnn_with_datamodule) -> None:
    model, _datamodule, bundle = cnn_with_datamodule
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
    cnn_with_datamodule,
) -> None:
    model, _datamodule, bundle = cnn_with_datamodule
    predictor = SoilSequencePredictor(model)

    unseen = SoilSequenceBundle.from_mapping(
        {**{field: getattr(bundle, field) for field in bundle.keys()}}
    )
    unseen.static_categoricals = np.asarray([["volcanic"]] * bundle.num_points, dtype=object)

    predictions = predictor.predict(unseen)

    assert predictions.shape == (bundle.num_points, 1)
    assert np.isfinite(predictions).all()


def test_predict_frame_labels_columns_with_the_target_names(cnn_with_datamodule) -> None:
    model, _datamodule, bundle = cnn_with_datamodule

    frame = SoilSequencePredictor(model).predict_frame(bundle)

    assert list(frame.columns) == ["organic_matter_pct"]
    assert list(frame.index) == list(bundle.point_ids)


def test_a_checkpoint_without_the_state_is_refused_with_an_actionable_message() -> None:
    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)

    with pytest.raises(ValueError, match="no preprocessing state"):
        SoilSequencePredictor(model)


def test_an_empty_bundle_predicts_nothing_rather_than_raising(cnn_with_datamodule) -> None:
    model, _datamodule, _bundle_ = cnn_with_datamodule

    predictions = SoilSequencePredictor(model).predict(SoilSequenceBundle())

    assert predictions.shape[0] == 0
