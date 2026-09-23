"""Number the labels of a category column, so a deep-learning model can look them up.

A category such as a landform class reaches `soil_cnn` as an :term:`embedding`: one learned vector
per label, found by the label's number. The numbering here is learned from the training points only
and saved with the model, so a label keeps its number when the model is later applied to new points.
Code 0 is reserved for labels the training points never held and for missing values, so an unknown
label cannot stop a prediction.

The codes are labels, not quantities: nothing compares or adds them. The scikit-learn models encode
their categories inside their own pipeline instead. This module uses pandas and NumPy only.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd


#: The code for a label the training points never held, and for a missing value. They share one
#: slot, so a column has ``len(vocabulary) + 1`` codes in all.
OOV_INDEX = 0

#: How :data:`OOV_INDEX` is named in messages. Never a label itself.
OOV_TOKEN = "<OOV>"


class FeatureBlocks(NamedTuple):
    """Which of a table's input columns hold numbers and which hold categories.

    Attributes
    ----------
    continuous_columns : list of str
        The numeric inputs.
    categorical_columns : list of str
        The category inputs, which have to be numbered before a model can read them.
    """

    continuous_columns: list[str]
    categorical_columns: list[str]


def _normalize_label(value: Any) -> Optional[str]:
    """Return a cell as a label, or None when it is missing.

    A blank string counts as missing: a CSV that quotes its empty cells writes ``""``, which is not
    a category of its own.
    """
    if value is None:
        return None
    # pd.isna raises on a list-like cell; the guard keeps one odd cell from stopping the run.
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar cells are not expected
        pass
    text = str(value).strip()
    return text or None


def _as_object_matrix(values: Any, feature_names: Optional[Sequence[str]]) -> tuple[np.ndarray, list[str]]:
    """Return a table or 2-D array as an ``(n_rows, n_columns)`` object array plus its column names."""
    if isinstance(values, pd.DataFrame):
        names = list(values.columns) if feature_names is None else list(feature_names)
        return values.to_numpy(dtype=object), [str(name) for name in names]

    matrix = np.asarray(values, dtype=object)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"Categorical values must be 2-D (rows, features), got {matrix.ndim}-D")
    names = (
        [f"categorical_{index}" for index in range(matrix.shape[1])]
        if feature_names is None
        else [str(name) for name in feature_names]
    )
    return matrix, names


class CategoricalEncoder:
    """Number the labels of each category column, and apply that numbering to other rows.

    Follows the familiar ``fit`` / ``transform`` shape. Fit it on the training points only: their
    labels are numbered ``1, 2, 3, ...`` in alphabetical order, and anything else - an unseen label,
    a missing value - gets :data:`OOV_INDEX`. The numbering is saved in the model's checkpoint.

    Examples
    --------
    >>> import pandas as pd
    >>> training = pd.DataFrame({"landform_class": ["plateau", "valley", "plateau"]})
    >>> encoder = CategoricalEncoder().fit(training)
    >>> encoder.vocabularies
    [['plateau', 'valley']]
    >>> new_points = pd.DataFrame({"landform_class": ["valley", "dune", None]})
    >>> encoder.transform(new_points).ravel().tolist()   # "dune" was never seen; None is missing
    [2, 0, 0]
    >>> encoder.cardinalities                            # two labels plus the reserved code
    [3]
    """

    def __init__(self) -> None:
        self._feature_names: list[str] = []
        self._vocabularies: list[list[str]] = []
        self._lookups: list[dict[str, int]] = []
        self._is_fitted = False

    # --- construction ------------------------------------------------------

    @classmethod
    def from_vocabularies(
        cls, feature_names: Sequence[str], vocabularies: Sequence[Sequence[str]]
    ) -> "CategoricalEncoder":
        """Rebuild a fitted encoder from the numbering saved in a checkpoint.

        The numbering travels with the weights, which is what lets a saved model read data it has
        never seen.

        Parameters
        ----------
        feature_names : sequence of str
            The category columns, in order.
        vocabularies : sequence of sequence of str
            Each column's labels in code order; the first gets code 1.

        Returns
        -------
        CategoricalEncoder

        Raises
        ------
        ValueError
            If there are not as many vocabularies as column names.
        """
        feature_names = [str(name) for name in feature_names]
        vocabularies = [[str(category) for category in vocabulary] for vocabulary in vocabularies]
        if len(feature_names) != len(vocabularies):
            raise ValueError(f"Got {len(feature_names)} feature name(s) but {len(vocabularies)} vocabulary/ies")

        encoder = cls()
        encoder._feature_names = feature_names
        encoder._vocabularies = vocabularies
        encoder._lookups = [
            {category: index + 1 for index, category in enumerate(vocabulary)} for vocabulary in vocabularies
        ]
        encoder._is_fitted = True
        return encoder

    # --- fitted state ------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether a numbering has been fitted or loaded."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """The category columns, in order."""
        return list(self._feature_names)

    @property
    def vocabularies(self) -> list[list[str]]:
        """Each column's training labels, in code order (the first has code 1).

        Plain strings, so they can be saved in a checkpoint and read back.
        """
        return [list(vocabulary) for vocabulary in self._vocabularies]

    @property
    def cardinalities(self) -> list[int]:
        """How many codes each column has: its labels plus the reserved code."""
        return [len(vocabulary) + 1 for vocabulary in self._vocabularies]

    # --- fit / transform ---------------------------------------------------

    def fit(self, values: Any, feature_names: Optional[Sequence[str]] = None) -> "CategoricalEncoder":
        """Learn the numbering from these rows - the training points only.

        Parameters
        ----------
        values : pandas.DataFrame or array-like of shape (n_rows, n_columns)
            The raw labels.
        feature_names : sequence of str, optional
            The column names, needed when ``values`` is not a DataFrame.

        Returns
        -------
        CategoricalEncoder
            This encoder, fitted.
        """
        matrix, names = _as_object_matrix(values, feature_names)
        self._feature_names = names
        self._vocabularies = []
        self._lookups = []

        for column_index in range(matrix.shape[1]):
            observed = {
                label for label in (_normalize_label(cell) for cell in matrix[:, column_index]) if label is not None
            }
            # Sorted, so the numbering depends on which labels appeared and not on the order the
            # rows arrived in.
            vocabulary = sorted(observed)
            self._vocabularies.append(vocabulary)
            self._lookups.append({category: index + 1 for index, category in enumerate(vocabulary)})

        self._is_fitted = True
        return self

    def transform(self, values: Any) -> np.ndarray:
        """Return the code of every label as an ``(n_rows, n_columns)`` array of whole numbers.

        Unseen labels and missing values give :data:`OOV_INDEX`.

        Parameters
        ----------
        values : pandas.DataFrame or array-like
            Raw labels, in the columns the encoder was fitted on.

        Returns
        -------
        numpy.ndarray of int

        Raises
        ------
        RuntimeError
            If the encoder has not been fitted.
        ValueError
            If the number of columns is not the fitted one.
        """
        if not self._is_fitted:
            raise RuntimeError("CategoricalEncoder.transform called before fit")

        matrix, _ = _as_object_matrix(values, self._feature_names)
        if matrix.shape[1] != len(self._lookups):
            raise ValueError(
                f"Expected {len(self._lookups)} categorical column(s) "
                f"({', '.join(self._feature_names) or 'none'}), got {matrix.shape[1]}"
            )

        codes = np.zeros(matrix.shape, dtype=np.int64)
        for column_index, lookup in enumerate(self._lookups):
            for row_index, cell in enumerate(matrix[:, column_index]):
                label = _normalize_label(cell)
                # Unknown and missing both keep OOV_INDEX, which `codes` is already filled with.
                if label is not None:
                    codes[row_index, column_index] = lookup.get(label, OOV_INDEX)
        return codes

    def fit_transform(self, values: Any, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Learn the numbering from these rows and return their codes."""
        return self.fit(values, feature_names).transform(values)

    # --- reporting ---------------------------------------------------------

    def oov_fraction(self, codes: np.ndarray) -> dict[str, float]:
        """Share of rows falling on the reserved code, per column.

        Worth checking: a column mostly on the reserved code means the training points did not cover
        its labels, and the model learns almost nothing from it.

        Parameters
        ----------
        codes : numpy.ndarray
            What :meth:`transform` returned.

        Returns
        -------
        dict of str to float
            One share per column, between 0 and 1.
        """
        codes = np.asarray(codes)
        if codes.size == 0:
            return {name: 0.0 for name in self._feature_names}
        return {name: float((codes[:, index] == OOV_INDEX).mean()) for index, name in enumerate(self._feature_names)}

    def summary(self) -> str:
        """One line naming each category column and how many labels it has."""
        return ", ".join(
            f"{name} ({len(vocabulary)} categories + {OOV_TOKEN})"
            for name, vocabulary in zip(self._feature_names, self._vocabularies)
        )


def resolve_categorical_columns(
    config: Any,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    logger: Any,
) -> FeatureBlocks:
    """Split the input columns into the numeric ones and the category ones.

    ``CATEGORICAL_FEATURES`` in ``data_spec.yml`` decides, for both model families: the type of a
    column cannot say whether a whole number is a measurement or a class number.

    Parameters
    ----------
    config : Config
        The run configuration; reads ``CATEGORICAL_FEATURES`` and ``EXCLUDE_CATEGORICAL``.
    frame : pandas.DataFrame
        The data, read to check which columns exist and which hold numbers.
    feature_columns : iterable of str
        The columns models may use as inputs, from
        :meth:`DataManager.filter_schema <yg_eo_soilnet.data_manager.DataManager.filter_schema>`.
    logger : logging.Logger
        Where a note about declared columns the models will not see goes.

    Returns
    -------
    FeatureBlocks

    Raises
    ------
    KeyError
        If a declared category column is not in the data - usually a leftover from another dataset.
        Remove it, or list it under ``EXCLUDE_CATEGORICAL``.
    ValueError
        If an input column holds text but is not declared. Numbering it silently would invent an
        order between its labels.
    """
    declared = [str(column) for column in (getattr(config, "CATEGORICAL_FEATURES", None) or [])]
    excluded = {str(column) for column in (getattr(config, "EXCLUDE_CATEGORICAL", None) or [])}
    declared = [column for column in dict.fromkeys(declared) if column not in excluded]

    absent = [column for column in declared if column not in frame.columns]
    if absent:
        raise KeyError(
            f"CATEGORICAL_FEATURES declares column(s) absent from the data: {', '.join(absent)}. "
            "Remove them from the configuration, or add them to EXCLUDE_CATEGORICAL."
        )

    feature_columns = [column for column in feature_columns if column in frame.columns]
    declared_set = set(declared)

    categorical_columns = [column for column in feature_columns if column in declared_set]
    continuous_columns = [column for column in feature_columns if column not in declared_set]

    undeclared_non_numeric = [
        column for column in continuous_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if undeclared_non_numeric:
        raise ValueError(
            f"Non-numeric feature column(s) not declared in CATEGORICAL_FEATURES: "
            f"{', '.join(undeclared_non_numeric)}. Declare them to give them entity embeddings, or "
            "add them to ELIMINATED_FEATURES to drop them."
        )

    # Declared, but not an input: dropped as an id, a lab column or an eliminated feature. That is
    # allowed, unlike the two cases above, but the model will not see the column.
    withheld = [column for column in declared if column not in categorical_columns]
    if withheld:
        logger.info(f"Declared categorical feature(s) not in the model's feature set: {', '.join(withheld)}")

    return FeatureBlocks(continuous_columns=continuous_columns, categorical_columns=categorical_columns)


def split_feature_blocks(frame: pd.DataFrame, blocks: FeatureBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Take the two blocks out of the table: numbers as 32-bit floats, categories as raw labels.

    The labels stay unnumbered here: the numbering has to be learned from the training points, and
    the split does not exist yet at this point.

    Parameters
    ----------
    frame : pandas.DataFrame
        The data.
    blocks : FeatureBlocks
        Which columns go in which block.

    Returns
    -------
    continuous : numpy.ndarray of shape (n_rows, n_continuous)
    categorical : numpy.ndarray of shape (n_rows, n_categorical)
        Raw labels, as objects.
    """
    rows = len(frame)
    continuous = (
        frame[list(blocks.continuous_columns)].to_numpy(dtype=np.float32)
        if blocks.continuous_columns
        else np.empty((rows, 0), dtype=np.float32)
    )
    categorical = (
        frame[list(blocks.categorical_columns)].to_numpy(dtype=object)
        if blocks.categorical_columns
        else np.empty((rows, 0), dtype=object)
    )
    return continuous, categorical
