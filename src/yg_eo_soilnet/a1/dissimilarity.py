"""How different a held-out region is from the data the model was trained on.

A model trained in some regions is applied in another; its error there should grow with how unlike
the training data the new region is. This module measures that unlikeness three ways, per point
and per region:

:func:`dissimilarity_index`
    The dissimilarity index (DI) and area of applicability (AOA) of Meyer & Pebesma (2021): the
    distance from a new point to its nearest training point, in the space of the model's inputs -
    each standardised, and weighted by how much the model relies on it - divided by the mean
    distance between training points. A point whose DI exceeds what cross-validation saw among
    the training points themselves is *outside* the area of applicability: the model is
    extrapolating there.
:func:`mess`
    The multivariate environmental similarity surface (MESS) of Elith et al. (2010): for each input,
    how far inside (positive) or outside (negative) the training range a point lies, and the worst
    input. Unlike the DI it needs no model, and it names the input responsible.
:func:`region_summary`
    Per held-out region: the median DI, the share of points outside the AOA, the difference in
    carbonate-class mix from training, and how much the SOM-CEC and SOM-CaCO3 relations (Spearman)
    differ from training's - a model learns those relations where it is trained.

Categorical inputs are left out of the DI and MESS: both are built on distances and ranges, which
categories do not have. The columns left out are recorded with every result.

References
----------
Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability
of spatial prediction models. *Methods in Ecology and Evolution* 12, 1620-1633.

Elith, J., Kearney, M. & Phillips, S. (2010). The art of modelling range-shifting species.
*Methods in Ecology and Evolution* 1, 330-342.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors

from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN

#: The quartile multiplier of the AOA threshold: the upper whisker of the training DIs' box plot.
WHISKER = 1.5
#: Fewer pairs than this and a Spearman correlation is not reported.
MIN_CORRELATION_PAIRS = 3


@dataclass
class DissimilarityResult:
    """What :func:`dissimilarity_index` returns.

    Attributes
    ----------
    di_new : pandas.Series
        The DI of each new point, indexed like ``X_new``.
    di_train : pandas.Series
        The cross-validated DI of each training point: its distance to the nearest training point
        outside its own fold.
    threshold : float
        The largest training DI that is not an outlier (the box plot's upper whisker).
    inside_new, inside_train : pandas.Series of bool
        DI at or below the threshold: inside the area of applicability.
    d_mean : float
        The mean distance between two training points, the DI's unit.
    columns : list of str
        The inputs the distances were computed on.
    dropped : list of str
        Inputs left out because they are the same at every training point.
    weights : pandas.Series
        The weight of each input in ``columns``, summing to 1.
    """

    di_new: pd.Series
    di_train: pd.Series
    threshold: float
    inside_new: pd.Series
    inside_train: pd.Series
    d_mean: float
    columns: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    weights: pd.Series = field(default_factory=pd.Series)


def _numeric(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """Refuse a frame holding a non-numeric column: distances need numbers."""
    other = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    if other:
        raise ValueError(f"{name} has non-numeric column(s) {other}; leave categorical inputs out of the DI and MESS.")
    return X.astype(float)


def _threshold(di_train: np.ndarray) -> float:
    """The upper whisker of a box plot of the training DIs: the largest one within Q75 + 1.5 IQR."""
    finite = di_train[np.isfinite(di_train)]
    if finite.size == 0:
        return np.nan
    q25, q75 = np.percentile(finite, [25, 75])
    return float(finite[finite <= q75 + WHISKER * (q75 - q25)].max())


def _mean_pairwise_distance(Z: np.ndarray) -> float:
    """The mean distance over every pair of distinct rows, without holding the whole matrix."""
    n = Z.shape[0]
    total = sum(float(chunk.sum()) for chunk in pairwise_distances_chunked(Z, metric="euclidean"))
    return total / (n * (n - 1))


def dissimilarity_index(
    X_train: pd.DataFrame,
    X_new: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    *,
    train_folds: Optional[pd.Series] = None,
) -> DissimilarityResult:
    """The dissimilarity index of new points from the training points, and the area of applicability.

    Following Meyer & Pebesma (2021):

    1. gaps are filled with the training median (as the stage-2 models fill them) and each input is
       standardised with the training mean and standard deviation, so inputs in large units do not
       dominate;
    2. each input is multiplied by its weight - how much the model relies on it - so that a
       difference in an input the model ignores does not count;
    3. the DI of a new point is its distance to the nearest training point, divided by the mean
       distance between two training points: 0 on top of a training point, about 1 as far away as
       training points are from each other on average;
    4. each training point gets a DI too, to its nearest training point *outside its own
       cross-validation fold* - how far the model had to reach during cross-validation;
    5. the threshold is the largest of those training DIs that is not an outlier (the upper whisker
       of their box plot); a new point beyond it is outside the area of applicability.

    Parameters
    ----------
    X_train, X_new : pandas.DataFrame
        The numeric inputs of the training points and of the points to assess, same columns.
    weights : pandas.Series, optional
        A weight per column, e.g. the model's feature importances (:func:`importance_weights`);
        normalised to sum to 1. Every column weighs the same unless given.
    train_folds : pandas.Series, optional
        The cross-validation fold of each training point, indexed like ``X_train``. Without it, each
        training point is compared with every other one (leave one out). A point with no fold is
        a fold of its own.

    Returns
    -------
    DissimilarityResult

    Raises
    ------
    ValueError
        If a column is not numeric, the columns differ, there are fewer than two training points,
        a weight is negative or missing, or every weight is 0.

    Examples
    --------
    >>> train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    >>> result = dissimilarity_index(train, pd.DataFrame({"x": [1.0, 10.0]}))
    >>> result.di_new.round(3).tolist()   # on a training point; then 7 units beyond the last
    [0.0, 4.2]
    >>> result.inside_new.tolist()
    [True, False]
    """
    X_train = _numeric(X_train, "X_train")
    X_new = _numeric(X_new, "X_new")
    if list(X_train.columns) != list(X_new.columns):
        raise ValueError("X_train and X_new must have the same columns, in the same order.")
    if len(X_train) < 2:
        raise ValueError("The DI needs at least two training points.")

    medians = X_train.median()
    train = X_train.fillna(medians)
    new = X_new.fillna(medians)
    spread = train.std(ddof=1)
    dropped = [column for column in train.columns if not np.isfinite(spread[column]) or spread[column] == 0]
    columns = [column for column in train.columns if column not in dropped]
    if not columns:
        raise ValueError("Every input is the same at every training point; there is nothing to measure.")

    if weights is None:
        weight = pd.Series(1.0, index=columns)
    else:
        missing = [column for column in columns if column not in weights.index]
        if missing:
            raise ValueError(f"No weight for {missing}.")
        weight = weights.reindex(columns).astype(float)
        if (weight < 0).any():
            raise ValueError("Weights cannot be negative.")
    if weight.sum() <= 0:
        raise ValueError("Every weight is 0: no input counts.")
    weight = weight / weight.sum()

    mean, sd = train[columns].mean(), spread[columns]
    Z_train = ((train[columns] - mean) / sd * weight).to_numpy()
    Z_new = ((new[columns] - mean) / sd * weight).to_numpy()

    d_mean = _mean_pairwise_distance(Z_train)
    if d_mean == 0:
        raise ValueError("Every training point is identical in the weighted inputs.")
    nearest_new, _ = NearestNeighbors(n_neighbors=1).fit(Z_train).kneighbors(Z_new)
    di_new = nearest_new[:, 0] / d_mean
    di_train = _train_distances(Z_train, X_train.index, train_folds) / d_mean

    threshold = _threshold(di_train)
    return DissimilarityResult(
        di_new=pd.Series(di_new, index=X_new.index, name="di"),
        di_train=pd.Series(di_train, index=X_train.index, name="di"),
        threshold=threshold,
        inside_new=pd.Series(di_new <= threshold, index=X_new.index, name="inside"),
        inside_train=pd.Series(di_train <= threshold, index=X_train.index, name="inside"),
        d_mean=d_mean,
        columns=columns,
        dropped=dropped,
        weights=weight,
    )


def _train_distances(Z: np.ndarray, index: pd.Index, folds: Optional[pd.Series]) -> np.ndarray:
    """Each training point's distance to the nearest training point outside its own fold."""
    if folds is None:
        labels = np.arange(len(index))
    else:
        labels = folds.reindex(index).to_numpy(dtype=float)
        # A point without a fold is a fold of its own: compared with every other point.
        lonely = np.isnan(labels)
        labels[lonely] = labels[~lonely].max(initial=0) + 1 + np.arange(lonely.sum())
    distances = np.full(len(index), np.nan)
    unique = np.unique(labels)
    if unique.size == len(index):
        # Leave one out: the nearest other point is the second neighbour (the first is itself).
        found, _ = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
        return found[:, 1]
    for label in unique:
        inside = labels == label
        if inside.all():
            continue
        found, _ = NearestNeighbors(n_neighbors=1).fit(Z[~inside]).kneighbors(Z[inside])
        distances[inside] = found[:, 0]
    return distances


def mess(X_train: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    """The multivariate environmental similarity surface of each new point (Elith et al. 2010).

    For each input, with *f* the percentage of training points below the new point's value *p*,
    and *min* and *max* the training range:

    - *f* = 0 (below every training value): 100 (p - min) / (max - min), negative;
    - 0 < *f* <= 50: 2 *f*;
    - 50 < *f* < 100: 2 (100 - *f*);
    - *f* = 100 (above every training value): 100 (max - p) / (max - min), negative at the top.

    A point's MESS is its lowest value over the inputs. Up to 100, at the training median on every
    input; 0 at the edge of the training range; negative outside it, the more so the further out.

    Parameters
    ----------
    X_train, X_new : pandas.DataFrame
        Numeric inputs, same columns. Gaps are skipped, input by input; an input with a single
        training value has no range and is skipped.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``X_new``: ``mess``, and ``mess_variable``, the input with the lowest value
        (the "most dissimilar variable"). NaN and None for a point with no usable input.

    Examples
    --------
    >>> train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0]})
    >>> out = mess(train, pd.DataFrame({"a": [1.5, 6.0], "b": [25.0, 25.0]}))
    >>> out["mess"].tolist(), out["mess_variable"].tolist()
    ([100.0, -100.0], ['a', 'a'])
    """
    X_train = _numeric(X_train, "X_train")
    X_new = _numeric(X_new, "X_new")
    if list(X_train.columns) != list(X_new.columns):
        raise ValueError("X_train and X_new must have the same columns, in the same order.")

    values = np.full(X_new.shape, np.nan)
    for position, column in enumerate(X_train.columns):
        reference = np.sort(X_train[column].dropna().to_numpy())
        if reference.size < 2 or reference[0] == reference[-1]:
            continue
        low, high = reference[0], reference[-1]
        p = X_new[column].to_numpy(dtype=float)
        f = 100.0 * np.searchsorted(reference, p, side="left") / reference.size
        similarity = np.where(
            f == 0,
            100.0 * (p - low) / (high - low),
            np.where(f <= 50, 2.0 * f, np.where(f < 100, 2.0 * (100.0 - f), 100.0 * (high - p) / (high - low))),
        )
        similarity[np.isnan(p)] = np.nan
        values[:, position] = similarity

    usable = ~np.isnan(values).all(axis=1)
    worst = np.full(len(X_new), np.nan)
    variable = np.full(len(X_new), None, dtype=object)
    if usable.any():
        positions = np.nanargmin(values[usable], axis=1)
        worst[usable] = values[usable][np.arange(usable.sum()), positions]
        variable[usable] = np.asarray(X_train.columns, dtype=object)[positions]
    return pd.DataFrame({"mess": worst, "mess_variable": variable}, index=X_new.index)


def importance_weights(pipeline, columns: Sequence[str]) -> Optional[pd.Series]:
    """The fitted model's feature importances, one per input column, summing to 1.

    The repository's pipelines prepare their inputs before the model sees them (``PipelineBuilder``):
    a numeric input ``x`` becomes ``num__x``, and, when it had gaps, a measured-or-filled flag
    ``num__missingindicator_x``. Both count for ``x``. Categorical inputs (``cat__...``) are not
    returned: the DI leaves them out.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted, with a ``preprocessor`` step and a ``model`` step (or a ``TransformedTargetRegressor``
        around one).
    columns : sequence of str
        The numeric inputs to return weights for.

    Returns
    -------
    pandas.Series or None
        None when the model has no ``feature_importances_`` or they are all 0; the DI then weighs
        every input the same.
    """
    model = pipeline.named_steps["model"]
    model = getattr(model, "regressor_", model)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    totals = pd.Series(0.0, index=list(columns))
    for name, value in zip(names, np.asarray(importances, dtype=float)):
        block, _, column = str(name).partition("__")
        if block != "num":
            continue
        column = column.removeprefix("missingindicator_")
        if column in totals.index:
            totals[column] += value
    if totals.sum() <= 0:
        return None
    return totals / totals.sum()


# --- one plan, and the regions -----------------------------------------------------------------


def plan_dissimilarity(
    X: pd.DataFrame,
    split: pd.Series,
    *,
    train_ids: pd.Index,
    weights: Optional[pd.Series] = None,
    folds: Optional[pd.Series] = None,
    categorical: Sequence[str] = (),
    labels: Optional[pd.DataFrame] = None,
    carbonate_class: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """The DI, AOA and MESS of every point of one plan, against that plan's training points.

    Validation and test points are compared with the training points; each training point carries
    its cross-validated DI (against the training points outside its fold) and its MESS against the
    training points, which is never negative.

    Parameters
    ----------
    X : pandas.DataFrame
        The model's inputs for every point of the plan, indexed by point id.
    split : pandas.Series
        Each point's split.
    train_ids : pandas.Index
        The training points the model was fitted on.
    weights : pandas.Series, optional
        Per input (:func:`importance_weights`).
    folds : pandas.Series, optional
        The training points' cross-validation folds.
    categorical : sequence of str
        Inputs to leave out.
    labels : pandas.DataFrame, optional
        Lab values to carry along for :func:`region_summary`, e.g. columns ``som``, ``cec``,
        ``caco3``.
    carbonate_class : pandas.Series, optional
        Each point's measured carbonate class.

    Returns
    -------
    pandas.DataFrame
        One row per point: ``point_id, split, di, inside, mess, mess_variable``, then
        ``carbonate_class`` and the ``labels`` columns, then the plan's ``threshold``, ``d_mean``,
        ``n_columns`` and ``left_out`` (JSON: the categorical inputs and the constant ones).
    """
    numeric = [column for column in X.columns if column not in set(categorical)]
    non_numeric = [column for column in numeric if not pd.api.types.is_numeric_dtype(X[column])]
    numeric = [column for column in numeric if column not in non_numeric]
    held = split.index[(split != TRAIN).to_numpy()]
    X_train, X_held = X.loc[train_ids, numeric], X.loc[held, numeric]

    result = dissimilarity_index(X_train, X_held, weights, train_folds=folds)
    surface = mess(X_train, pd.concat([X_train, X_held]))
    ids = train_ids.append(held)
    frame = pd.DataFrame(
        {
            "point_id": ids.to_numpy(),
            "split": split.reindex(ids).to_numpy(),
            "di": np.concatenate([result.di_train.to_numpy(), result.di_new.to_numpy()]),
            "inside": np.concatenate([result.inside_train.to_numpy(), result.inside_new.to_numpy()]),
            "mess": surface["mess"].to_numpy(),
            "mess_variable": surface["mess_variable"].to_numpy(),
            "carbonate_class": (
                carbonate_class.reindex(ids).astype(object).to_numpy() if carbonate_class is not None else None
            ),
        }
    )
    if labels is not None:
        for column in labels.columns:
            frame[column] = labels[column].reindex(ids).to_numpy(dtype=float)
    left_out = {"categorical": sorted([*set(categorical) & set(X.columns), *non_numeric]), "constant": result.dropped}
    frame["threshold"] = result.threshold
    frame["d_mean"] = result.d_mean
    frame["n_columns"] = len(result.columns)
    frame["left_out"] = json.dumps(left_out)
    return frame


def _shares(classes: pd.Series, labels: Sequence[str]) -> pd.Series:
    """The share of each class among the points that have one."""
    present = classes.dropna()
    counts = present.value_counts().reindex(list(labels), fill_value=0)
    return counts / counts.sum() if counts.sum() else counts.astype(float) * np.nan


def _spearman(frame: pd.DataFrame, a: str, b: str) -> float:
    """Spearman correlation of two columns over the rows where both are present; NaN if too few."""
    pairs = frame[[a, b]].dropna()
    if len(pairs) < MIN_CORRELATION_PAIRS or pairs[a].nunique() < 2 or pairs[b].nunique() < 2:
        return np.nan
    return float(pairs[a].corr(pairs[b], method="spearman"))


def region_summary(
    frame: pd.DataFrame,
    *,
    labels: Optional[Sequence[str]] = None,
    target: str = "som",
    by: Sequence[str] = ("plan_id", "held_out"),
) -> pd.DataFrame:
    """How different each held-out region is from its plan's training points.

    Parameters
    ----------
    frame : pandas.DataFrame
        :func:`plan_dissimilarity` tables of one or more plans, with ``plan_id`` and ``held_out``
        columns. The region is the plan's test points.
    labels : sequence of str, optional
        The carbonate classes, in order; those present in the data unless given.
    target : str, default "som"
        The lab value whose correlation with each other lab value is compared: every column of
        ``frame`` named like ``cec`` or ``caco3`` that :func:`plan_dissimilarity` carried along.
    by : sequence of str
        The columns naming a plan.

    Returns
    -------
    pandas.DataFrame
        One row per plan:

        - ``n_test``, ``median_di``, ``share_outside`` (of the test points, outside the area of
          applicability), ``threshold``;
        - ``median_mess``, ``share_mess_negative`` and ``top_mess_variable`` (the input most often
          responsible where MESS is negative);
        - ``share_<class>_region`` and ``share_<class>_train`` per carbonate class, and
          ``carbonate_tvd``: half the sum of the absolute share differences, 0 for the same mix
          and 1 for no class in common;
        - per other lab value ``x``: ``spearman_<target>_x_region``, ``..._train`` and ``..._diff``
          (region - train).
    """
    by = list(by)
    known = {"point_id", "split", "di", "inside", "mess", "mess_variable", "carbonate_class", "threshold"}
    known |= {"d_mean", "n_columns", "left_out", target, *by}
    others = [column for column in frame.columns if column not in known]
    if labels is None:
        labels = sorted(frame["carbonate_class"].dropna().unique()) if "carbonate_class" in frame else []
    rows = []
    for key, plan in frame.groupby(by, sort=True, dropna=False):
        region = plan[plan["split"] == TEST]
        train = plan[plan["split"] == TRAIN]
        negative = region[region["mess"] < 0]
        row = {
            **dict(zip(by, key if isinstance(key, tuple) else (key,))),
            "n_test": len(region),
            "median_di": float(region["di"].median()),
            "share_outside": float((~region["inside"].astype(bool)).mean()) if len(region) else np.nan,
            "threshold": float(plan["threshold"].iloc[0]),
            "median_mess": float(region["mess"].median()),
            "share_mess_negative": float((region["mess"] < 0).mean()) if len(region) else np.nan,
            "top_mess_variable": negative["mess_variable"].mode().iloc[0] if len(negative) else None,
        }
        if labels:
            region_shares = _shares(region["carbonate_class"], labels)
            train_shares = _shares(train["carbonate_class"], labels)
            for label in labels:
                row[f"share_{label}_region"] = float(region_shares[label])
                row[f"share_{label}_train"] = float(train_shares[label])
            row["carbonate_tvd"] = float(0.5 * (region_shares - train_shares).abs().sum(min_count=1))
        if target in plan:
            for other in others:
                stem = f"spearman_{target}_{other}"
                row[f"{stem}_region"] = _spearman(region, target, other)
                row[f"{stem}_train"] = _spearman(train, target, other)
                row[f"{stem}_diff"] = row[f"{stem}_region"] - row[f"{stem}_train"]
        rows.append(row)
    return pd.DataFrame(rows)
