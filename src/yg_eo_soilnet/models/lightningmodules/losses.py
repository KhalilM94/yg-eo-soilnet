"""What the deep-learning model is trained to minimize.

Three ordinary choices score each target on its own: ``mse`` (the default), and ``huber`` /
``smooth_l1``, which are less swayed by a single badly wrong point.

Three more are for a model predicting several targets at once, where the targets are not
independent - high organic matter goes with high cation exchange capacity, and some combinations do
not occur in soil at all. They add that structure to what the model is scored on:

``mahalanobis``
    Charges more for an error in a direction the training data never shows - say raising cation
    exchange capacity while lowering organic matter - than for an equally large error along a
    combination that does occur.
``correlation_penalty``
    An ordinary loss, plus a penalty when the predictions do not vary together the way the measured
    targets do.
``cosine``
    An ordinary loss, plus a penalty on getting the *ratios* between the targets wrong, whatever
    their size.

All of them see the targets in the units the model trains in - log-transformed, if that is switched
on, and standardized - not in the target's own units. Only the predictions the run reports are
converted back.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

#: The losses that score each target on its own. Also the values ``loss_base`` accepts.
BASE_LOSSES = {"mse", "l2", "huber", "smooth_l1", "smoothl1"}

#: The losses that read across targets, so they need at least two.
STRUCTURAL_LOSSES = {"mahalanobis", "correlation_penalty", "cosine"}

#: Every value ``loss_name`` accepts.
LOSS_NAMES = BASE_LOSSES | STRUCTURAL_LOSSES

#: The units the cosine loss may measure its angle in.
COSINE_SPACES = {"original", "standardized"}


def inverse_transform_targets(
    values: torch.Tensor,
    *,
    mean: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    standardized: bool,
    log1p: bool,
) -> torch.Tensor:
    """Convert values from the units the model trains in back to the target's own units.

    Undoes the standardization first and the log transform second, the reverse of the order they
    were applied in.

    Parameters
    ----------
    values : torch.Tensor
        Predictions or targets, in training units.
    mean, scale : torch.Tensor or None
        The standardization statistics, from the datamodule.
    standardized : bool
        Whether the values were standardized.
    log1p : bool
        Whether the log transform was applied.

    Returns
    -------
    torch.Tensor
        The values in the target's own units.
    """
    if standardized and mean is not None and scale is not None:
        values = values * scale.to(values.device) + mean.to(values.device)
    if log1p:
        # The same transform the scikit-learn side uses: 10 * ln(1 + y).
        values = torch.expm1(values / 10.0)
    return values


def build_base_loss(loss_name: str, huber_delta: float) -> nn.Module:
    """Build one of the per-target losses: ``mse``, ``huber`` or ``smooth_l1``.

    Parameters
    ----------
    loss_name : str
        The name from the configuration.
    huber_delta : float
        Where ``huber`` and ``smooth_l1`` switch from squared to absolute error - the size of error
        beyond which a point stops pulling harder.

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        If the name is not one of the three.
    """
    if loss_name in {"mse", "l2"}:
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if loss_name in {"smooth_l1", "smoothl1"}:
        return nn.SmoothL1Loss(beta=huber_delta)
    raise ValueError(f"Unknown loss_name '{loss_name}'; expected one of mse, huber, smooth_l1")


def _prepare_covariance(covariance: Any, target_dim: int, shrinkage: float) -> np.ndarray:
    """Check the targets' covariance and pull it slightly towards the identity matrix."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (target_dim, target_dim):
        raise ValueError(f"target_covariance must be a {target_dim}x{target_dim} matrix; got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("target_covariance contains non-finite values.")
    shrinkage = float(shrinkage)
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError(f"loss_shrinkage must be in [0, 1); got {shrinkage}.")
    # Pulled towards the identity before inverting. Two targets that track each other almost
    # exactly would otherwise put nearly the whole loss on one barely observable difference.
    matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(target_dim)
    # Made exactly symmetric: rounding can leave the two halves slightly different.
    return 0.5 * (matrix + matrix.T)


class MahalanobisLoss(nn.Module):
    """Charge more for an error in a direction the measured targets never take.

    ``loss_name: mahalanobis``. It weighs an error by how unusual its *direction* is, given how the
    training targets vary together, not only by its size. An error that raises one target while
    lowering another that normally rises with it costs more than one that moves both together.

    With targets that do not vary together at all, this is exactly mean squared error.

    Parameters
    ----------
    covariance : array-like of shape (n_targets, n_targets)
        How the training targets vary together, from the datamodule.
    target_dim : int
        How many targets.
    shrinkage : float, default 0.05
        How far the covariance is pulled towards treating the targets as independent, between 0 and
        1. Raise it if two targets track each other almost exactly.

    Raises
    ------
    ValueError
        If the covariance has the wrong shape, holds missing values, or is not a covariance.
    """

    def __init__(self, covariance: Any, *, target_dim: int, shrinkage: float = 0.05):
        super().__init__()
        self.target_dim = int(target_dim)
        matrix = _prepare_covariance(covariance, self.target_dim, shrinkage)

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        largest = float(eigenvalues.max())
        if largest <= 0.0:
            raise ValueError("target_covariance has no positive eigenvalue; it is not a covariance.")
        # A floor relative to the matrix itself: what counts as a flat direction depends on its
        # overall scale.
        floor = largest * 1e-6
        clipped = np.maximum(eigenvalues, floor)
        self.condition_number = float(clipped.max() / clipped.min())
        if not np.allclose(clipped, eigenvalues):
            logger.warning(
                "target_covariance is rank-deficient; %d of %d eigenvalues were clipped to %.3e. "
                "Consider raising loss_shrinkage.",
                int((clipped != eigenvalues).sum()),
                self.target_dim,
                floor,
            )
        logger.info(
            "MahalanobisLoss built over %d targets; condition number %.1f after shrinkage.",
            self.target_dim,
            self.condition_number,
        )
        whitening = (eigenvectors * (clipped**-0.5)) @ eigenvectors.T

        # Not saved with the weights: it follows from the covariance, which is saved, and the loss
        # is only used while training.
        self.register_buffer("whitening", torch.as_tensor(whitening, dtype=torch.float32), persistent=False)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Score one batch."""
        difference = predictions - targets
        # The matrix is symmetric, so the order of the multiplication makes no difference.
        whitened = difference @ self.whitening.to(difference.dtype)
        return (whitened**2).sum(dim=-1).mean() / self.target_dim


class CorrelationPenaltyLoss(nn.Module):
    """An ordinary loss, plus a penalty when the predictions do not vary together as the data does.

    ``loss_name: correlation_penalty``. A model trained on squared error alone often predicts targets
    that track each other more closely than the measurements do - they all follow the same strong
    covariate - or not at all, if each is fitted separately. Neither shows up in a per-target R²,
    and both show up here.

    The comparison is against the correlations measured on the training points, not on the current
    batch, which would be too small to measure them reliably. Batches smaller than ``min_batch``
    skip the penalty instead of using a poor estimate.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The per-target loss the penalty is added to.
    reference_correlation : array-like of shape (n_targets, n_targets)
        How the training targets vary together.
    target_dim : int
        How many targets.
    weight : float, default 0.1
        How heavily the penalty counts against the base loss.
    min_batch : int, default 16
        Below this many points, the penalty is skipped.

    Raises
    ------
    ValueError
        If the matrix has the wrong shape or holds missing values.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        reference_correlation: Any,
        *,
        target_dim: int,
        weight: float = 0.1,
        min_batch: int = 16,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.target_dim = int(target_dim)
        self.weight = float(weight)
        self.min_batch = max(2, int(min_batch))
        self.last_components: dict[str, float] = {}

        matrix = np.asarray(reference_correlation, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape != (self.target_dim, self.target_dim):
            raise ValueError(
                f"target_covariance must be a {self.target_dim}x{self.target_dim} matrix; got shape {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("target_covariance contains non-finite values.")
        # Already a correlation matrix as the datamodule fits it; normalizing again costs nothing
        # and makes this correct if it is ever handed one in the target's own units.
        deviation = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
        correlation = matrix / np.outer(deviation, deviation)
        self.register_buffer("reference", torch.as_tensor(correlation, dtype=torch.float32), persistent=False)
        # Each pair of targets once. A target against itself is always 1 on both sides.
        rows, columns = np.triu_indices(self.target_dim, k=1)
        self.register_buffer("pair_rows", torch.as_tensor(rows, dtype=torch.long), persistent=False)
        self.register_buffer("pair_columns", torch.as_tensor(columns, dtype=torch.long), persistent=False)

    @staticmethod
    def _batch_correlation(values: torch.Tensor) -> torch.Tensor:
        """How the targets vary together within one batch."""
        centered = values - values.mean(dim=0, keepdim=True)
        # The floor keeps a target the model currently predicts as a constant - which happens
        # early in training - from dividing by zero.
        deviation = centered.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        covariance = (centered.T @ centered) / centered.shape[0]
        correlation = covariance / torch.outer(deviation, deviation)
        # Rounding can push a correlation just past 1, which would reward overshooting.
        return correlation.clamp(-1.0, 1.0)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Score one batch; ``last_components`` then holds the two parts separately."""
        base = self.base_loss(predictions, targets)
        if predictions.shape[0] < self.min_batch:
            self.last_components = {"base": float(base.detach()), "penalty": 0.0}
            return base

        predicted = self._batch_correlation(predictions)
        reference = self.reference.to(predictions.dtype)
        difference = predicted[self.pair_rows, self.pair_columns] - reference[self.pair_rows, self.pair_columns]
        # Averaged over the pairs, so the weight means the same thing however many targets there
        # are.
        penalty = difference.pow(2).mean()
        self.last_components = {"base": float(base.detach()), "penalty": float(penalty.detach())}
        return base + self.weight * penalty


class CosineStructureLoss(nn.Module):
    """An ordinary loss, plus a penalty on getting the ratios between the targets wrong.

    ``loss_name: cosine``. It compares each point's predicted targets with its measured ones as a
    set of proportions, ignoring their overall size. Unlike the correlation penalty this is computed
    per point, so it works at any batch size.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The per-target loss the penalty is added to.
    space : {"original", "standardized"}, default "original"
        Which units the proportions are measured in. ``"original"`` converts back to the target's
        own units first, so the penalty is on the ratios soil chemistry actually constrains; because
        those values are all positive, the penalty is small and ``weight`` has to be larger than for
        the other losses. ``"standardized"`` stays in training units, asking instead whether the
        model gets the shape of a point's departure from average right.
    weight : float, default 0.1
        How heavily the penalty counts against the base loss.
    target_mean, target_scale : sequence of float, optional
        The standardization statistics, needed to convert back.
    target_transform : str, optional
        ``"log1p"`` when the targets were log-transformed.

    Raises
    ------
    ValueError
        If ``space`` is not one of the two.
    """

    #: A point whose measured targets are all near the average has no meaningful set of
    #: proportions, so it is left out of the penalty.
    NORM_FLOOR = 1e-3

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        space: str = "original",
        weight: float = 0.1,
        target_mean: Optional[Sequence[float]] = None,
        target_scale: Optional[Sequence[float]] = None,
        target_transform: Optional[str] = None,
    ):
        super().__init__()
        space = str(space).lower()
        if space not in COSINE_SPACES:
            raise ValueError(f"Unknown cosine_space {space!r}; expected one of: {', '.join(sorted(COSINE_SPACES))}.")
        self.base_loss = base_loss
        self.space = space
        self.weight = float(weight)
        self.last_components: dict[str, float] = {}

        self.standardized = target_mean is not None and target_scale is not None
        self.log1p = str(target_transform).lower() == "log1p"
        if self.space == "original" and not (self.standardized or self.log1p):
            # With no transform applied there is nothing to undo and the two settings agree.
            # Legal, but worth saying, since the configuration reads as though it chose something.
            logger.info("cosine_space='original' with untransformed targets: identical to 'standardized'.")
        # The loss keeps its own copy of these rather than reaching back into the model.
        self.register_buffer(
            "target_mean",
            torch.as_tensor(list(target_mean or []), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "target_scale",
            torch.as_tensor(list(target_scale or []), dtype=torch.float32),
            persistent=False,
        )

    def _to_scored_space(self, values: torch.Tensor) -> torch.Tensor:
        """Put the values in the units the penalty is measured in."""
        if self.space != "original":
            return values
        return inverse_transform_targets(
            values,
            mean=self.target_mean if self.standardized else None,
            scale=self.target_scale if self.standardized else None,
            standardized=self.standardized,
            log1p=self.log1p,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Score one batch; ``last_components`` then holds the two parts separately."""
        base = self.base_loss(predictions, targets)
        scored_predictions = self._to_scored_space(predictions)
        scored_targets = self._to_scored_space(targets)

        similarity = torch.nn.functional.cosine_similarity(scored_predictions, scored_targets, dim=-1, eps=1e-8)
        deviation = 1.0 - similarity
        # A point sitting at the average of every target has no proportions to reproduce, and
        # scoring it would add noise.
        measurable = scored_targets.norm(dim=-1) > self.NORM_FLOOR
        if bool(measurable.all()):
            penalty = deviation.mean()
        elif bool(measurable.any()):
            penalty = (deviation * measurable).sum() / measurable.sum()
        else:
            penalty = deviation.sum() * 0.0

        self.last_components = {"base": float(base.detach()), "penalty": float(penalty.detach())}
        return base + self.weight * penalty


def build_loss_fn(
    loss_name: str,
    *,
    huber_delta: float = 1.0,
    loss_base: str = "mse",
    loss_lambda: float = 0.1,
    loss_shrinkage: float = 0.05,
    loss_min_batch: int = 16,
    cosine_space: str = "original",
    target_dim: int = 1,
    target_covariance: Any = None,
    target_mean: Optional[Sequence[float]] = None,
    target_scale: Optional[Sequence[float]] = None,
    target_transform: Optional[str] = None,
) -> nn.Module:
    """Build the loss a model-list entry asks for.

    Everything is checked here, while the model is being built, rather than at the first training
    step: a run that fails after the data is ready has already cost minutes, and a loss that quietly
    falls back to a simpler one costs a whole experiment, because nothing looks wrong afterwards.

    Each loss reads only the settings that apply to it; the others are ignored:

    =========================  ================================================================
    ``loss_name``              settings it reads
    =========================  ================================================================
    ``mse``                    none
    ``huber``                  ``huber_delta``
    ``smooth_l1``              ``huber_delta``
    ``mahalanobis``            ``loss_shrinkage``, ``target_covariance``
    ``correlation_penalty``    ``loss_base``, ``loss_lambda``, ``loss_min_batch``,
                               ``target_covariance``
    ``cosine``                 ``loss_base``, ``loss_lambda``, ``cosine_space``, the target
                               statistics
    =========================  ================================================================

    Parameters
    ----------
    loss_name : str
        One of :data:`LOSS_NAMES`.
    huber_delta : float, default 1.0
        Where ``huber`` and ``smooth_l1`` switch from squared to absolute error.
    loss_base : str, default "mse"
        The per-target loss the structural ones add their penalty to.
    loss_lambda : float, default 0.1
        How heavily that penalty counts.
    loss_shrinkage : float, default 0.05
        For ``mahalanobis``: how far the targets' covariance is pulled towards treating them as
        independent.
    loss_min_batch : int, default 16
        For ``correlation_penalty``: the smallest batch the penalty is measured on.
    cosine_space : {"original", "standardized"}, default "original"
        For ``cosine``: which units the ratios are measured in.
    target_dim : int, default 1
        How many targets; the structural losses need at least two.
    target_covariance : array-like, optional
        How the training targets vary together. Supplied by the datamodule.
    target_mean, target_scale : sequence of float, optional
        The standardization statistics.
    target_transform : str, optional
        ``"log1p"`` when the targets were log-transformed.

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        If the name is unknown, a structural loss is asked for with one target (set
        ``MULTI_TARGET_MODE: joint``, or use a per-target loss), or the covariance it needs is
        missing.

    Examples
    --------
    >>> type(build_loss_fn("mse")).__name__
    'MSELoss'
    >>> type(build_loss_fn("huber", huber_delta=0.5)).__name__
    'HuberLoss'
    """
    loss_name = str(loss_name).lower()
    if loss_name in BASE_LOSSES:
        return build_base_loss(loss_name, huber_delta)
    if loss_name not in STRUCTURAL_LOSSES:
        raise ValueError(
            f"Unknown loss_name '{loss_name}'; expected one of {', '.join(sorted(BASE_LOSSES | STRUCTURAL_LOSSES))}"
        )

    if int(target_dim) < 2:
        raise ValueError(
            f"loss_name '{loss_name}' reads across targets and needs at least 2 of them, but this "
            f"model has {int(target_dim)}. Set MULTI_TARGET_MODE: joint in configs/data_spec.yml, or "
            f"use a point loss (mse, huber, smooth_l1)."
        )

    loss_base = str(loss_base).lower()
    if loss_base not in BASE_LOSSES:
        raise ValueError(f"Unknown loss_base '{loss_base}'; expected one of {', '.join(sorted(BASE_LOSSES))}")

    if loss_name == "cosine":
        return CosineStructureLoss(
            build_base_loss(loss_base, huber_delta),
            space=cosine_space,
            weight=loss_lambda,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=target_transform,
        )

    if target_covariance is None:
        raise ValueError(
            f"loss_name '{loss_name}' needs the training targets' covariance, but none was supplied. "
            f"LightningConfigFactory injects it from SoilSequenceDataModule.target_covariance_, "
            f"which _fit_normalization computes in setup(); a hand-built module must pass "
            f"target_covariance itself."
        )

    if loss_name == "mahalanobis":
        return MahalanobisLoss(target_covariance, target_dim=int(target_dim), shrinkage=loss_shrinkage)
    return CorrelationPenaltyLoss(
        build_base_loss(loss_base, huber_delta),
        target_covariance,
        target_dim=int(target_dim),
        weight=loss_lambda,
        min_batch=loss_min_batch,
    )
