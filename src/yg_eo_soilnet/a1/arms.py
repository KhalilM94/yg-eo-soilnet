"""What the two extra arms of the mixed experiment change in the fits.

A ``mixed_weighted`` or ``mixed_indicator`` plan has exactly the points of its ``mixed`` plan
(:mod:`yg_eo_soilnet.a1.plans`); only the training differs:

``mixed_weighted``
    Each moved point - a training point of the held-out region - weighs ``w`` in every fit, every
    other training point 1 (:func:`sample_weights`). With ``plans.mixed.local_weight: balanced``,
    ``w`` = (training points from the other regions) / (moved points), counted over the training
    points with a measured target, which are the ones fitted: both sets then carry the same total
    weight. The weights reach the final fit, the settings search (each fit of the search is
    weighted the same way) and, when it runs, stage 1. The search still scores each validation fold
    by its plain, unweighted RMSE, so the settings are chosen for how well points are predicted,
    not for how heavily they were weighted.
``mixed_indicator``
    One more input, :data:`INDICATOR_COLUMN`: 1 for every point of the held-out region (moved and
    test), 0 for every other (:func:`indicator`). Only the models on every covariate (the
    ``combined`` set) read it; the per-group models and stage 1 do not.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from yg_eo_soilnet.a1.plans import MIXED_INDICATOR, MIXED_WEIGHTED
from yg_eo_soilnet.a1.settings import BALANCED, A1Settings
from yg_eo_soilnet.datamodules.splitting import TEST, SplitPlan

#: The input the ``mixed_indicator`` arm adds: 1 for the held-out region's points, 0 otherwise.
INDICATOR_COLUMN = "a1_in_target_basin"


def held_out_region(plan: SplitPlan) -> int:
    """The region a mixed plan holds out: the one its test points are in.

    Raises
    ------
    ValueError
        If the plan has no regions, or its test points lie in more than one.
    """
    if plan.clusters is None:
        raise ValueError("A mixed plan records its regions as clusters; this plan has none.")
    regions = plan.clusters.reindex(plan.point_ids_for(TEST)).dropna().unique()
    if len(regions) != 1:
        raise ValueError(f"The test points of a mixed plan lie in one region; these lie in {sorted(regions)}")
    return int(regions[0])


def in_held_out_region(plan: SplitPlan) -> pd.Series:
    """True for each point of the plan in its held-out region, indexed by point id."""
    region = held_out_region(plan)
    clusters = plan.clusters.reindex(plan.assignments.index)
    return (clusters == region).rename(None)


def indicator(plan: SplitPlan) -> pd.Series:
    """:data:`INDICATOR_COLUMN` for every point of the plan: 1.0 in the held-out region, 0.0 elsewhere."""
    return in_held_out_region(plan).astype(float).rename(INDICATOR_COLUMN)


def local_weight(settings: A1Settings, n_other: int, n_moved: int) -> float:
    """The weight of one moved point: ``plans.mixed.local_weight``, or ``n_other / n_moved`` when balanced.

    Examples
    --------
    >>> from yg_eo_soilnet.a1.settings import A1Settings
    >>> local_weight(A1Settings(main_config_path="main_config.yml"), n_other=3000, n_moved=150)
    20.0
    """
    configured = settings.plans.mixed.local_weight
    if configured == BALANCED:
        if n_moved == 0:
            raise ValueError("A weighted plan has no moved training point to weigh.")
        return float(n_other) / float(n_moved)
    return float(configured)


def sample_weights(plan: SplitPlan, settings: A1Settings, train_ids: pd.Index) -> tuple[pd.Series, float]:
    """The weight of each training point of a ``mixed_weighted`` plan, and the moved points' weight ``w``.

    Parameters
    ----------
    plan : SplitPlan
    settings : A1Settings
    train_ids : pandas.Index
        The training points fitted: those with a measured target.

    Returns
    -------
    weights : pandas.Series
        Indexed by ``train_ids``: ``w`` for a moved point, 1.0 for every other.
    w : float
    """
    moved = in_held_out_region(plan).reindex(train_ids).fillna(False).astype(bool).to_numpy()
    w = local_weight(settings, n_other=int((~moved).sum()), n_moved=int(moved.sum()))
    weights = pd.Series(1.0, index=train_ids, name="sample_weight")
    weights[moved] = w
    return weights, w


def weights_for(
    kind: str, plan: SplitPlan, settings: A1Settings, train_ids: pd.Index
) -> tuple[Optional[pd.Series], float]:
    """:func:`sample_weights` for a ``mixed_weighted`` plan; ``(None, nan)`` for any other kind."""
    if kind != MIXED_WEIGHTED:
        return None, float("nan")
    return sample_weights(plan, settings, train_ids)


def adds_indicator(kind: str) -> bool:
    """Whether plans of this kind read :data:`INDICATOR_COLUMN`."""
    return kind == MIXED_INDICATOR


def fit_kwargs(weights: Optional[pd.Series], index: pd.Index) -> dict:
    """``{"model__sample_weight": ...}`` for a pipeline fitted on the rows ``index``; ``{}`` without weights.

    The A1 pipelines name their estimator ``model``; a log-transformed target hands the weights on
    to the model it wraps.
    """
    if weights is None:
        return {}
    return {"model__sample_weight": weights.reindex(index).to_numpy(dtype=float)}
