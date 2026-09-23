"""Every model's prediction for every point, in one table.

A run already records what each model predicted for its own test points. This answers the other
question: given a point, what did each model say about it? Switched on with
``export_point_predictions.enabled``.

Two files on the :term:`main run`: ``point_predictions_wide.csv``, one row per point with a column
per target and model, which is what to join onto a map; and ``point_predictions_long.csv``, one row
per point per target per model, which is what to group and compare.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.artifacts import ArtifactLayout

# Separates the two halves of a wide column name. Two underscores rather than one because every
# realistic target and model name already contains single underscores; this at least makes the
# boundary visible to a human, even though it cannot be parsed reliably.
NAME_SEPARATOR = "__"

# Long-format column names.
TARGET_COLUMN = "target"
MODEL_COLUMN = "model"
PREDICTION_COLUMN = "prediction"


def export_enabled_for(config: Any, model_name: str) -> bool:
    """Whether this model should predict every point, not only the test points.

    ``export_point_predictions.models`` names the models to do it for and ``exclude_models`` names ones
    to leave out; naming a model explicitly wins.
    """
    if not bool(getattr(config, "EXPORT_POINT_PREDICTIONS", False)):
        return False

    allowed = [str(name) for name in (getattr(config, "EXPORT_POINT_PREDICTIONS_MODELS", None) or [])]
    if allowed:
        return str(model_name) in allowed

    skipped = [str(name) for name in (getattr(config, "EXPORT_POINT_PREDICTIONS_SKIP_MODELS", None) or [])]
    return str(model_name) not in skipped


def point_id_column(config: Any) -> str:
    """What the id column is called in the exported files."""
    return str(config.POINT_ID_COLUMN)


def point_prediction_frame(
    point_ids: Any,
    predictions: Any,
    target_names: Sequence[str],
    id_column: str = "point_id",
) -> pd.DataFrame:
    """One model's contribution: the point ids, and one column per target it predicts.

    ``point_ids`` must line up with ``predictions`` row for row.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    ids = list(point_ids)
    if len(ids) != values.shape[0]:
        raise ValueError(
            f"{len(ids)} point ids against {values.shape[0]} prediction rows. These are paired "
            "positionally, so a mismatch means the ids describe different points than the "
            "predictions do - refusing rather than writing a plausible, wrong file."
        )
    if values.shape[1] != len(target_names):
        raise ValueError(f"{values.shape[1]} prediction columns against {len(target_names)} target names.")

    frame = pd.DataFrame({id_column: ids})
    for index, target_name in enumerate(target_names):
        frame[str(target_name)] = values[:, index]
    return frame


def to_long(frames: Iterable[tuple[str, pd.DataFrame]], id_column: str = "point_id") -> pd.DataFrame:
    """Every model's predictions as one row per point per target per model."""
    melted = []
    for model_name, frame in frames:
        if frame is None or frame.empty or id_column not in frame.columns:
            continue
        target_columns = [column for column in frame.columns if column != id_column]
        if not target_columns:
            continue
        long_frame = frame.melt(
            id_vars=[id_column],
            value_vars=target_columns,
            var_name=TARGET_COLUMN,
            value_name=PREDICTION_COLUMN,
        )
        long_frame[MODEL_COLUMN] = str(model_name)
        melted.append(long_frame)

    if not melted:
        return pd.DataFrame(columns=[id_column, TARGET_COLUMN, MODEL_COLUMN, PREDICTION_COLUMN])

    combined = pd.concat(melted, ignore_index=True)
    return combined[[id_column, TARGET_COLUMN, MODEL_COLUMN, PREDICTION_COLUMN]]


def to_wide(long_frame: pd.DataFrame, id_column: str = "point_id") -> pd.DataFrame:
    """One row per point, one column per target and model.

    Built from the long form rather than from the models again, so the two files cannot disagree.
    """
    if long_frame.empty:
        return pd.DataFrame(columns=[id_column])

    frame = long_frame.copy()
    frame["_column"] = [
        wide_column_name(target, model) for target, model in zip(frame[TARGET_COLUMN], frame[MODEL_COLUMN])
    ]
    # `first` rather than the default mean: a duplicated (point, target, model) means the same model
    # reported twice for one point, which is a bug upstream, and silently averaging it away would
    # hide it. The count check below is what surfaces it.
    wide = frame.pivot_table(index=id_column, columns="_column", values=PREDICTION_COLUMN, aggfunc="first")
    wide.columns.name = None
    return wide.reset_index()


def wide_column_name(target: Any, model: Any) -> str:
    """The wide form's column name for one target and model.

    Examples
    --------
    >>> wide_column_name("clay_pct", "soil_cnn")
    'clay_pct__soil_cnn'
    """
    return f"{ArtifactLayout.safe(target)}{NAME_SEPARATOR}{ArtifactLayout.safe(model)}"


def combine(
    frames: Iterable[tuple[str, pd.DataFrame]],
    id_column: str = "point_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both forms, from every model's contribution to one run."""
    long_frame = to_long(frames, id_column=id_column)
    return to_wide(long_frame, id_column=id_column), long_frame


def duplicate_report(long_frame: pd.DataFrame, id_column: str = "point_id") -> Optional[str]:
    """A message when a point, target and model appear twice, or None when they do not.

    The wide form keeps only the first of a duplicate, so without this a model counted twice would
    quietly lose predictions.
    """
    if long_frame.empty:
        return None
    keys = [id_column, TARGET_COLUMN, MODEL_COLUMN]
    duplicated = int(long_frame.duplicated(subset=keys).sum())
    if duplicated == 0:
        return None
    return (
        f"{duplicated} duplicate (point, target, model) rows in the prediction export; the wide "
        "file keeps the first of each. This usually means one model was collected from two runs."
    )


def child_frame_from_predictor_output(
    frame: pd.DataFrame,
    id_column: str,
    target_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Turn what the deep-learning predictor returned into one model's contribution."""
    reset = frame.reset_index()
    reset = reset.rename(columns={reset.columns[0]: id_column})
    if target_names:
        keep = [id_column] + [str(name) for name in target_names if str(name) in reset.columns]
        reset = reset[keep]
    return reset


def summarize(wide: pd.DataFrame, long_frame: pd.DataFrame, id_column: str) -> Mapping[str, Any]:
    """A short description of what was exported, for the run summary."""
    return {
        "n_points": int(wide.shape[0]),
        "n_columns": int(max(wide.shape[1] - 1, 0)),
        "models": sorted(long_frame[MODEL_COLUMN].unique().tolist()) if not long_frame.empty else [],
        "targets": sorted(long_frame[TARGET_COLUMN].unique().tolist()) if not long_frame.empty else [],
        "id_column": id_column,
    }
