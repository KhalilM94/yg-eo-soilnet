"""Package a trained deep-learning model so MLflow can serve it raw data.

A saved model here accepts a table: one row per point, with the covariates as ordinary columns and
each :term:`data source`'s readings and their dates as nested columns. It rebuilds the internal form
from that, using the statistics stored in the :term:`checkpoint`, and returns predictions in the
target's own units.

Saved in MLflow's generic format rather than as a raw PyTorch model, which would have to be traced
from an example input - impossible for a model that reads a different number of readings per point.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

TIME_SUFFIX = "__time"
VALUES_SUFFIX = "__values"
POINT_ID_COLUMN = "point_id"

# The only modules inference touches. Shipping the whole package instead made MLflow infer
# geopandas, pyproj, matplotlib, seaborn and statsmodels as runtime dependencies of a CNN forward
# pass, because importing yg_eo_soilnet pulls all of them through its __init__.
SERVING_MODULES: tuple[str, ...] = (
    "serving",
    "models/__init__.py",
    "models/lightningmodules",
    "datamodules/__init__.py",
    "datamodules/sequence",
    "datamodules/categorical.py",
    "datamodules/frame_cleaning.py",
    # The sequence datamodule imports SplitPlan for its type hint and for the shared-split branch.
    # It is deliberately dependency-free (numpy + pandas only), so it costs serving nothing.
    "datamodules/splitting.py",
    # The sequence datamodule builds every DataLoader through build_loader. Torch only, which serving
    # already needs for the forward pass.
    "datamodules/loaders.py",
    # The sequence datamodule narrows its targets through select_target_columns. Pure numpy, like
    # splitting.py above, so it costs serving nothing.
    "targets.py",
    "utils.py",
)

# Every __init__.py in the staged copy is replaced with this. Mandatory, not cosmetic: every package
# __init__ in this project re-exports its subpackages, and those re-exports reach modules the
# serving subset deliberately leaves out - yg_eo_soilnet/__init__ pulls clustering_utils and
# plot_utils, datamodules/__init__ pulls the sklearn path. A staged copy carrying the real ones
# fails on import.
#
# Blanking ALL of them rather than curating a list is safe because nothing in the serving subset
# imports from a package namespace; every import is a fully-qualified submodule import. The
# subprocess load test is what keeps that true.
_MINIMAL_INIT = '''"""Inference-only subset of yg_eo_soilnet, staged by the model logger.

Deliberately empty. The real package __init__ files re-export the training surface, which would drag
the geospatial and plotting stacks into a serving environment that only runs a forward pass. Import
submodules directly.
"""
'''


def stage_serving_package(destination: str) -> str:
    """Copy the few modules a served model needs next to it, and say where they went."""
    import shutil
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1]
    staged_root = Path(destination) / source_root.name
    staged_root.mkdir(parents=True, exist_ok=True)

    for relative in SERVING_MODULES:
        source = source_root / relative
        target = staged_root / relative
        if not source.exists():
            raise FileNotFoundError(f"SERVING_MODULES names {relative!r}, which does not exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, target)

    staged_root.joinpath("__init__.py").write_text(_MINIMAL_INIT, encoding="utf-8")
    for init_file in staged_root.rglob("__init__.py"):
        init_file.write_text(_MINIMAL_INIT, encoding="utf-8")

    return str(staged_root)


def serving_requirements() -> list[str]:
    """The libraries a served model needs, listed rather than guessed.

    Guessing them from what was loaded while saving would list the whole training stack.
        """
    from importlib.metadata import PackageNotFoundError, version

    requirements = []
    for package in ("torch", "lightning", "numpy", "pandas"):
        try:
            requirements.append(f"{package}=={version(package)}")
        except PackageNotFoundError:  # pragma: no cover - all four are hard dependencies
            requirements.append(package)
    return requirements


def time_column(modality: str) -> str:
    """The column holding one data source's dates."""
    return f"{modality}{TIME_SUFFIX}"


def values_column(modality: str) -> str:
    """The column holding one data source's readings."""
    return f"{modality}{VALUES_SUFFIX}"


def _as_list_of_arrays(column: pd.Series, *, dtype, label: str) -> list[np.ndarray]:
    """One array per row, whatever shape the nested column arrived in.

    A request can arrive as JSON, as a saved file or as a hand-built table, and each delivers nested
    values slightly differently.
        """
    arrays: list[np.ndarray] = []
    for position, value in enumerate(column.tolist()):
        if value is None:
            arrays.append(np.empty((0,), dtype=dtype))
            continue
        try:
            arrays.append(np.asarray(list(value), dtype=dtype))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {position} is not a numeric sequence: {value!r}") from exc
    return arrays


def bundle_from_frame(frame: pd.DataFrame, state: Mapping[str, Any]):
    """Rebuild the model's internal form from a request table.

    Every column list comes from the model's own stored settings rather than from the request, so a
    request with extra columns, missing ones, or columns in another order still produces exactly the
    inputs the model was trained on.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per point.
    state : mapping
        The model's stored input statistics and column names.

    Returns
    -------
    SoilSequenceBundle
        """
    from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

    static_names = list(state.get("static_feature_names") or [])
    categorical_names = list(state.get("categorical_feature_names") or [])
    label_names = list(state.get("label_feature_names") or [])
    # Empty unless the model was trained with USE_HARMONIC_COORDS, so a checkpoint without the
    # coordinate branch - or one predating it - requires nothing new and rebuilds nothing.
    coord_names = list(state.get("coord_names") or [])
    modality_columns = {
        str(name): list(columns)
        for name, columns in (state.get("modality_column_names") or {}).items()
    }

    # Coordinates are REQUIRED, alongside the static and categorical blocks, rather than optional
    # like the lab columns. A model built with coord_dim=2 cannot predict without a position, and
    # there is no honest fill for one: the train-median trick that rescues a covariate would place
    # the sample at a location it does not occupy, and the encoder would read that as a confident
    # position rather than as an absence. The builder drops such a row for the same reason.
    missing = [
        name for name in static_names + categorical_names + coord_names if name not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"Input is missing {len(missing)} column(s) this model was trained on: "
            f"{', '.join(missing[:10])}"
        )

    n_rows = len(frame)
    point_ids = (
        frame[POINT_ID_COLUMN].tolist() if POINT_ID_COLUMN in frame.columns else list(range(n_rows))
    )

    static_features = (
        frame[static_names].to_numpy(dtype=np.float32)
        if static_names
        else np.empty((n_rows, 0), dtype=np.float32)
    )
    static_categoricals = (
        frame[categorical_names].astype(object).to_numpy()
        if categorical_names
        else np.empty((n_rows, 0), dtype=object)
    )
    # float64, not the float32 the static block uses: the bundle keeps coordinates in float64
    # because float32 resolves about a metre at this latitude, and the train-bbox normalization
    # downstream subtracts two nearby numbers and would spend most of it.
    coords = (
        frame[coord_names].to_numpy(dtype=np.float64)
        if coord_names
        else np.empty((n_rows, 0), dtype=np.float64)
    )
    # The lab block is rebuilt at FULL ROSTER WIDTH, with each supplied column at its own roster
    # position and the rest left NaN.
    #
    # Width, because the model index_selects roster positions: it reads only the few columns named
    # in auxiliary_label_columns, but it addresses them by where they sit in the roster it was built
    # against, and it refuses a batch of any other width. Handing it just the supplied columns
    # produced "Batch carries 0 lab column(s) ... against 24" and silently stopped every model being
    # logged, hence registered.
    #
    # Position rather than order of appearance, because a shifted column would feed the model a
    # different measurement under the right name without raising anything.
    #
    # NaN rather than zero, because NaN is the bundle's own convention for absent (see
    # SoilSequenceBundle.label_features) and SoilSequenceDataModule._standardize_labels median-fills
    # it from the training split and flags it in the validity channel - so a column the caller did
    # not supply arrives marked as unmeasured instead of as a real zero.
    if label_names:
        label_features = np.full((n_rows, len(label_names)), np.nan, dtype=np.float32)
        for index, name in enumerate(label_names):
            if name in frame.columns:
                label_features[:, index] = frame[name].to_numpy(dtype=np.float32)
    else:
        label_features = np.empty((n_rows, 0), dtype=np.float32)

    sequences: dict[str, list[np.ndarray]] = {}
    sequence_times: dict[str, list[np.ndarray]] = {}
    for modality, columns in modality_columns.items():
        time_name, values_name = time_column(modality), values_column(modality)
        for name in (time_name, values_name):
            if name not in frame.columns:
                raise KeyError(
                    f"Input is missing '{name}'. Modality '{modality}' needs '{time_name}' "
                    f"(decimal years) and '{values_name}' (one value per band, ordered "
                    f"{columns})."
                )

        times = _as_list_of_arrays(frame[time_name], dtype=np.float64, label=time_name)
        raw_values = _as_list_of_arrays(frame[values_name], dtype=np.float32, label=values_name)

        per_point = []
        for position, values in enumerate(raw_values):
            reshaped = values.reshape(-1, len(columns)) if values.size else np.empty(
                (0, len(columns)), dtype=np.float32
            )
            if reshaped.shape[0] != times[position].shape[0]:
                raise ValueError(
                    f"Row {position}: '{values_name}' has {reshaped.shape[0]} observation(s) but "
                    f"'{time_name}' has {times[position].shape[0]}; they must line up."
                )
            per_point.append(reshaped)

        sequences[modality] = per_point
        sequence_times[modality] = times

    return SoilSequenceBundle(
        point_ids=point_ids,
        static_features=static_features,
        static_feature_names=static_names,
        static_categoricals=static_categoricals,
        categorical_feature_names=categorical_names,
        coords=coords,
        coord_names=coord_names,
        targets=np.empty((n_rows, 0), dtype=np.float32),
        target_names=list(state.get("target_names") or []),
        label_features=label_features,
        label_feature_names=list(label_names),
        sequences=sequences,
        sequence_times=sequence_times,
        modality_columns=modality_columns,
        temporal_enabled=bool(modality_columns),
    )


def frame_from_bundle(
    bundle,
    state: Mapping[str, Any],
    n_rows: int | None = None,
    auxiliary_columns: Any = None,
) -> pd.DataFrame:
    """The reverse: a request table built from training data.

    Used to produce the example that ships with a saved model, which is what makes its declared inputs
    real rather than a description.
        """
    from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

    bundle = SoilSequenceBundle.from_mapping(bundle)
    count = bundle.num_points if n_rows is None else min(int(n_rows), bundle.num_points)

    data: dict[str, Any] = {POINT_ID_COLUMN: [str(value) for value in bundle.point_ids[:count]]}

    for index, name in enumerate(state.get("static_feature_names") or []):
        data[name] = np.asarray(bundle.static_features[:count, index], dtype=np.float64)

    categoricals = np.asarray(bundle.static_categoricals, dtype=object)
    for index, name in enumerate(state.get("categorical_feature_names") or []):
        data[name] = [str(value) for value in categoricals[:count, index]]

    # Coordinates, when the model was trained with the harmonic branch. Omitting them is what made
    # every logged model fail with "Batch carries 0 coordinate column(s) but this model was built
    # for 2" - the round trip produced a zero-width coords block, the datamodule collated an empty
    # x_coords, and the branch refused it. Nothing was logged, so nothing was registered, and the
    # run still finished green. Same failure the lab roster had; see bundle_from_frame.
    #
    # float64 to match the bundle's own dtype - see bundle_from_frame for why the precision matters.
    coords = np.asarray(bundle.coords)
    for index, name in enumerate(state.get("coord_names") or []):
        data[name] = np.asarray(coords[:count, index], dtype=np.float64)

    labels = np.asarray(bundle.label_features)
    wanted = {str(name) for name in (auxiliary_columns or [])}
    if labels.size and wanted:
        for index, name in enumerate(state.get("label_feature_names") or []):
            if name in wanted:
                data[name] = np.asarray(labels[:count, index], dtype=np.float64)

    for modality in (state.get("modality_column_names") or {}):
        data[time_column(modality)] = [
            np.asarray(values, dtype=np.float64).tolist()
            for values in bundle.sequence_times.get(modality, [])[:count]
        ]
        data[values_column(modality)] = [
            np.asarray(values, dtype=np.float64).tolist()
            for values in bundle.sequences.get(modality, [])[:count]
        ]

    return pd.DataFrame(data)


class SoilSequencePyfunc:
    """The prediction half of a saved model: a request table in, predictions out.

    Examples
    --------
    >>> import mlflow                                                    # doctest: +SKIP
    >>> model = mlflow.pyfunc.load_model("models:/clay_pct_soil_cnn@champion")   # doctest: +SKIP
    >>> model.predict(request_frame)                                     # doctest: +SKIP
        """

    def __init__(self, model=None):
        """Hold the stored input statistics the model was trained with."""
        self.model = model
        self._predictor = None

    def load_context(self, context) -> None:  # pragma: no cover - exercised via mlflow
        """Load the saved network when MLflow restores the model."""
        self._predictor = None

    def _ensure_predictor(self):
        """Build the predictor on first use."""
        if self._predictor is None:
            from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

            self._predictor = SoilSequencePredictor(self.model)
        return self._predictor

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Predict for the points in a request table.

        Parameters
        ----------
        context : object
            MLflow's loading context.
        model_input : pandas.DataFrame
            One row per point; see :func:`bundle_from_frame` for the columns.
        params : dict, optional
            ``{"uncertainty": True}`` adds a ``<target>_std`` column beside each prediction, when the model
            predicts a spread.

        Returns
        -------
        pandas.DataFrame
            One column per target, in the target's own units.
                """
        predictor = self._ensure_predictor()
        frame = pd.DataFrame(model_input)
        bundle = bundle_from_frame(frame, predictor.preprocessing_state)
        predictions, sigma = predictor.predict_with_uncertainty(bundle)

        names = list(predictor.preprocessing_state.get("target_names") or [])
        names = names[: predictions.shape[1]] or [
            f"prediction_{index}" for index in range(predictions.shape[1])
        ]
        output = pd.DataFrame(predictions, columns=names)

        if bool((params or {}).get("uncertainty", False)) and sigma is not None:
            for index, name in enumerate(names):
                output[f"{name}_std"] = sigma[:, index]

        return output


def example_from_state(
    state: Mapping[str, Any],
    static_frame: pd.DataFrame | None = None,
    n_rows: int = 3,
    sequence_length: int = 4,
    auxiliary_columns: Any = None,
) -> pd.DataFrame:
    """An example request table built from the model's stored settings alone.

    The version of :func:`build_input_example` that needs no data - a checkpoint carries every column
    name it expects.
        """
    static_names = list(state.get("static_feature_names") or [])
    static_mean = list(state.get("static_mean") or [])
    categorical_names = list(state.get("categorical_feature_names") or [])
    vocabularies = list(state.get("categorical_vocabularies") or [])
    label_names = list(state.get("label_feature_names") or [])
    label_mean = list(state.get("label_mean") or [])
    modality_columns = {
        str(name): list(columns)
        for name, columns in (state.get("modality_column_names") or {}).items()
    }
    sequence_mean = state.get("sequence_mean") or {}

    data: dict[str, Any] = {POINT_ID_COLUMN: [f"point_{index}" for index in range(n_rows)]}

    for index, name in enumerate(static_names):
        if static_frame is not None and name in static_frame.columns:
            column = pd.to_numeric(static_frame[name], errors="coerce").to_numpy(dtype=np.float64)
            values = np.resize(column[np.isfinite(column)], n_rows) if np.isfinite(column).any() else None
            if values is not None:
                data[name] = values
                continue
        fallback = float(static_mean[index]) if index < len(static_mean) else 0.0
        data[name] = np.full(n_rows, fallback, dtype=np.float64)

    for index, name in enumerate(categorical_names):
        # A real vocabulary entry, not a placeholder: an unknown label would land on the reserved
        # out-of-vocabulary index and quietly exercise a different embedding row than production.
        vocabulary = vocabularies[index] if index < len(vocabularies) else []
        data[name] = [str(vocabulary[0]) if vocabulary else "unknown"] * n_rows

    # Only the columns the model actually reads reach the signature; the rest of the roster is
    # reconstructed at its own position by bundle_from_frame. Passed in rather than read from the
    # state because the selection is a MODEL hyper-parameter - the state describes the data.
    for name in list(auxiliary_columns or []):
        index = label_names.index(name) if name in label_names else -1
        fallback = float(label_mean[index]) if 0 <= index < len(label_mean) else 0.0
        data[name] = np.full(n_rows, fallback, dtype=np.float64)

    for modality, columns in modality_columns.items():
        means = list(sequence_mean.get(modality) or [])
        per_band = [float(means[index]) if index < len(means) else 0.0 for index in range(len(columns))]
        data[time_column(modality)] = [
            [2020.0 + step / 4.0 for step in range(sequence_length)] for _ in range(n_rows)
        ]
        data[values_column(modality)] = [
            [list(per_band) for _ in range(sequence_length)] for _ in range(n_rows)
        ]

    return pd.DataFrame(data)


def required_label_columns(model) -> list[str]:
    """The lab columns a request has to supply for this model.

    Asked of the model itself, since a model may read a lab column as an
    :term:`auxiliary lab input`, as a :term:`residual base`, or not at all.
        """
    columns = getattr(model, "serving_label_columns", None)
    if columns is None:
        columns = getattr(model, "auxiliary_label_columns", None)
    return [str(column) for column in (columns or [])]


def build_input_example(model, bundle, n_rows: int = 3) -> pd.DataFrame:
    """A small, valid request table built from the model's own training data.

    It ships with the saved model, so anyone loading it can see exactly what a request looks like.
        """
    state = model.get_preprocessing_state() if hasattr(model, "get_preprocessing_state") else {}
    if not state:
        raise ValueError("The model carries no preprocessing state, so no input example can be built.")
    return frame_from_bundle(
        bundle,
        state,
        n_rows=n_rows,
        auxiliary_columns=required_label_columns(model),
    )
