"""Unit tests for the fitted categorical vocabulary (the embedding blocks are in test_encoders).

Both components are deliberately free of any datamodule or bundle concept, so these tests build
their inputs inline rather than going through the pipeline; the end-to-end wiring is covered by
test_sequence_data.py.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.datamodules.categorical import (
    OOV_INDEX,
    CategoricalEncoder,
    resolve_categorical_columns,
    split_feature_blocks,
)
from yg_eo_soilnet.models.lightningmodules.tabular_encoders import (
    embedding_dim_for,
    resolve_embedding_dims,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "texture_20cm": ["lo", "cl", "lo", "salo"],
            "landform_class": ["valley", "peak_ridge", "valley", "upper_slope"],
            "slope": [1.0, 2.0, 3.0, 4.0],
            "twi": [0.5, 0.6, 0.7, 0.8],
        }
    )


# --- CategoricalEncoder ---------------------------------------------------


def test_fit_reserves_index_zero_and_codes_from_one():
    encoder = CategoricalEncoder().fit(_frame()[["texture_20cm"]])

    assert encoder.vocabularies == [["cl", "lo", "salo"]]
    assert encoder.cardinalities == [4]  # 3 categories + the reserved slot
    codes = encoder.transform(pd.DataFrame({"texture_20cm": ["cl", "lo", "salo"]}))
    assert codes.tolist() == [[1], [2], [3]]
    assert OOV_INDEX not in codes


def test_category_absent_from_fit_transforms_to_the_reserved_slot():
    """The defect this module exists to fix: a held-out category must not invent a new index."""
    encoder = CategoricalEncoder().fit(pd.DataFrame({"landform_class": ["valley", "peak_ridge"]}))

    codes = encoder.transform(pd.DataFrame({"landform_class": ["valley", "lower_slope_cool"]}))

    assert codes[:, 0].tolist() == [2, OOV_INDEX]
    assert encoder.cardinalities == [3]  # unchanged by the unseen category


@pytest.mark.parametrize("missing", [None, np.nan, float("nan"), "", "   ", pd.NA])
def test_missing_values_share_the_reserved_slot(missing):
    encoder = CategoricalEncoder().fit(pd.DataFrame({"texture_20cm": ["lo", "cl"]}))

    codes = encoder.transform(pd.DataFrame({"texture_20cm": [missing]}, dtype=object))

    assert codes.tolist() == [[OOV_INDEX]]


def test_vocabulary_does_not_depend_on_row_order():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = CategoricalEncoder().fit(frame[["texture_20cm", "landform_class"]])
    second = CategoricalEncoder().fit(shuffled[["texture_20cm", "landform_class"]])

    assert first.vocabularies == second.vocabularies
    probe = pd.DataFrame({"texture_20cm": ["salo"], "landform_class": ["valley"]})
    assert first.transform(probe).tolist() == second.transform(probe).tolist()


def test_transform_is_stable_across_calls():
    encoder = CategoricalEncoder().fit(_frame()[["texture_20cm"]])
    probe = _frame()[["texture_20cm"]]

    assert encoder.transform(probe).tolist() == encoder.transform(probe).tolist()


def test_from_vocabularies_reproduces_a_fitted_encoder():
    """The checkpoint-portability path: the mapping travels with the model, not with the data."""
    fitted = CategoricalEncoder().fit(_frame()[["texture_20cm", "landform_class"]])

    restored = CategoricalEncoder.from_vocabularies(fitted.feature_names, fitted.vocabularies)

    probe = pd.DataFrame(
        {"texture_20cm": ["lo", "unseen"], "landform_class": ["valley", "unseen"]}
    )
    assert restored.cardinalities == fitted.cardinalities
    assert restored.transform(probe).tolist() == fitted.transform(probe).tolist()


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        CategoricalEncoder().transform(pd.DataFrame({"texture_20cm": ["lo"]}))


def test_transform_rejects_a_different_column_count():
    encoder = CategoricalEncoder().fit(_frame()[["texture_20cm", "landform_class"]])

    with pytest.raises(ValueError, match="Expected 2 categorical column"):
        encoder.transform(pd.DataFrame({"texture_20cm": ["lo"]}))


def test_oov_fraction_reports_per_column_coverage():
    encoder = CategoricalEncoder().fit(pd.DataFrame({"texture_20cm": ["lo", "cl"]}))

    codes = encoder.transform(pd.DataFrame({"texture_20cm": ["lo", "unseen", "unseen", "unseen"]}))

    assert encoder.oov_fraction(codes) == {"texture_20cm": 0.75}


def test_numeric_columns_are_encoded_as_labels_when_declared():
    """A declared integer class id must be embedded, not read as a magnitude."""
    encoder = CategoricalEncoder().fit(pd.DataFrame({"class_id": [10, 3, 10]}))

    assert encoder.vocabularies == [["10", "3"]]
    assert encoder.transform(pd.DataFrame({"class_id": [3, 10, 99]})).tolist() == [[2], [1], [0]]


# --- column resolution ----------------------------------------------------


def _config(categorical, excluded=()):
    return SimpleNamespace(CATEGORICAL_FEATURES=list(categorical), EXCLUDE_CATEGORICAL=list(excluded))


def test_resolve_splits_declared_columns_from_the_rest(logger):
    frame = _frame()

    blocks = resolve_categorical_columns(
        _config(["texture_20cm", "landform_class"]), frame, frame.columns, logger=logger
    )

    assert blocks.categorical_columns == ["texture_20cm", "landform_class"]
    assert blocks.continuous_columns == ["slope", "twi"]


def test_resolve_raises_on_a_declared_column_absent_from_the_data(logger):
    frame = _frame()

    with pytest.raises(KeyError, match="SU_WRB1_PH"):
        resolve_categorical_columns(_config(["SU_WRB1_PH"]), frame, frame.columns, logger=logger)


def test_resolve_raises_on_an_undeclared_non_numeric_column(logger):
    frame = _frame()

    with pytest.raises(ValueError, match="landform_class"):
        resolve_categorical_columns(_config(["texture_20cm"]), frame, frame.columns, logger=logger)


def test_excluded_categorical_is_neither_required_nor_embedded(logger):
    frame = _frame().drop(columns=["landform_class"])

    blocks = resolve_categorical_columns(
        _config(["texture_20cm", "SU_WRB1_PH"], excluded=["SU_WRB1_PH"]),
        frame,
        frame.columns,
        logger=logger,
    )

    assert blocks.categorical_columns == ["texture_20cm"]


def test_split_feature_blocks_keeps_labels_unencoded(logger):
    frame = _frame()
    blocks = resolve_categorical_columns(
        _config(["texture_20cm", "landform_class"]), frame, frame.columns, logger=logger
    )

    continuous, categorical = split_feature_blocks(frame, blocks)

    assert continuous.shape == (4, 2) and continuous.dtype == np.float32
    assert categorical.shape == (4, 2) and categorical.dtype == object
    assert categorical[0].tolist() == ["lo", "valley"]


def test_split_feature_blocks_handles_no_categoricals(logger):
    frame = _frame().drop(columns=["texture_20cm", "landform_class"])
    blocks = resolve_categorical_columns(_config([]), frame, frame.columns, logger=logger)

    continuous, categorical = split_feature_blocks(frame, blocks)

    assert continuous.shape == (4, 2)
    assert categorical.shape == (4, 0)


# --- embedding heuristic --------------------------------------------------


@pytest.mark.parametrize(
    "cardinality,expected",
    [(2, 1), (7, 4), (13, 7), (100, 50), (1000, 50)],
)
def test_embedding_dim_heuristic(cardinality, expected):
    assert embedding_dim_for(cardinality) == expected


def test_embedding_dim_cap_is_configurable():
    assert embedding_dim_for(1000, max_dim=16) == 16


def test_resolve_embedding_dims_accepts_auto_scalar_and_sequence():
    assert resolve_embedding_dims([7, 13]) == [4, 7]
    assert resolve_embedding_dims([7, 13], "auto") == [4, 7]
    assert resolve_embedding_dims([7, 13], 8) == [8, 8]
    assert resolve_embedding_dims([7, 13], [3, 5]) == [3, 5]


def test_resolve_embedding_dims_mapping_must_name_every_feature():
    names = ["texture_20cm", "landform_class"]
    assert resolve_embedding_dims([7, 13], {"texture_20cm": 3, "landform_class": 5}, feature_names=names) == [3, 5]

    with pytest.raises(ValueError, match="landform_class"):
        resolve_embedding_dims([7, 13], {"texture_20cm": 3}, feature_names=names)


def test_resolve_embedding_dims_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="1 entry"):
        resolve_embedding_dims([7, 13], [4])
