"""How :term:`ensemble` members are seeded, and how their predictions are combined.

Plain arrays only, no scikit-learn and no PyTorch, so both families combine their members exactly
the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

# `auto` bootstraps only estimators with no random_state; see should_bootstrap.
BOOTSTRAP_AUTO = "auto"
BOOTSTRAP_ALWAYS = "always"
BOOTSTRAP_NEVER = "never"
BOOTSTRAP_MODES = (BOOTSTRAP_AUTO, BOOTSTRAP_ALWAYS, BOOTSTRAP_NEVER)


def member_seeds(base_seed: int, n_members: int, stride: int = 1000) -> list[int]:
    """The seed each ensemble member is trained at.

    Spaced well apart rather than consecutive, so a member's seed does not collide with the seed of the
    next model or of the split - two members trained at the same seed are the same model twice.

    Parameters
    ----------
    base_seed : int
        The run's seed.
    n_members : int
        How many members.
    stride : int, default 1000
        The gap between members' seeds.

    Returns
    -------
    list of int

    Examples
    --------
    >>> member_seeds(42, 3)
    [42, 1042, 2042]
    """
    if n_members < 1:
        raise ValueError(f"n_members must be at least 1; got {n_members}")
    return [int(base_seed) + index * int(stride) for index in range(int(n_members))]


def should_bootstrap(estimator: Any, mode: str = BOOTSTRAP_AUTO) -> bool:
    """Whether this model's members have to be trained on resampled rows to differ.

    ``auto`` resamples for every scikit-learn model: a model that ignores its seed - a plain linear
    regression does - would otherwise produce identical members and no uncertainty at all. Deep-learning
    models are not resampled: two networks started from different weights already differ.

    Parameters
    ----------
    model : estimator
        The model being trained.
    setting : {"auto", "always", "never"}, default "auto"

    Returns
    -------
    bool
    """
    normalized = str(mode).lower()
    if normalized not in BOOTSTRAP_MODES:
        raise ValueError(f"bootstrap must be one of {BOOTSTRAP_MODES}; got {mode!r}")
    if normalized == BOOTSTRAP_NEVER:
        return False
    return True


def bootstrap_indices(n_rows: int, seed: int) -> np.ndarray:
    """Which rows one member trains on: ``n_rows`` draws, with repeats allowed.

    Examples
    --------
    >>> len(bootstrap_indices(10, seed=42))
    10
    """
    generator = np.random.default_rng(int(seed))
    return generator.integers(0, int(n_rows), size=int(n_rows))


@dataclass(frozen=True)
class EnsemblePrediction:
    """What an ensemble predicted: the value, and the two kinds of uncertainty.

    Attributes
    ----------
    mean : numpy.ndarray of shape (n_points, n_targets)
        The members' average - the prediction.
    epistemic_std : numpy.ndarray
        How much the members disagree; see :term:`epistemic uncertainty`. More training data reduces it.
    aleatoric_std : numpy.ndarray
        Noise the members agree about; see :term:`aleatoric uncertainty`. Only a model with a
        :term:`variance head` predicts it, and more data does not reduce it.
    """

    mean: np.ndarray
    epistemic_std: np.ndarray
    aleatoric_std: np.ndarray

    @property
    def total_std(self) -> np.ndarray:
        """The overall spread, which is what an interval is built from.

        The two kinds are combined as variances - added under a square root - because adding the spreads
        themselves would overstate the width by up to 41%.
        """
        return np.sqrt(self.epistemic_std**2 + self.aleatoric_std**2)


def aggregate(
    member_predictions: Sequence[Any],
    member_sigmas: Optional[Sequence[Any]] = None,
) -> EnsemblePrediction:
    """Combine the members' predictions into an average and its two uncertainties.

    Parameters
    ----------
    member_predictions : sequence of array-like
        One entry per member.
    member_sigmas : sequence of array-like, optional
        Each member's own predicted spread, from a :term:`variance head`.

    Returns
    -------
    EnsemblePrediction
    """
    if not len(member_predictions):
        raise ValueError("aggregate needs at least one member prediction")

    stacked = np.stack([_as_2d(values) for values in member_predictions], axis=0)
    mean = stacked.mean(axis=0)
    epistemic_std = stacked.std(axis=0, ddof=0)

    if member_sigmas is None:
        aleatoric_std = np.zeros_like(mean)
    else:
        if len(member_sigmas) != len(member_predictions):
            raise ValueError(
                f"member_sigmas has {len(member_sigmas)} entries but there are "
                f"{len(member_predictions)} members; they must correspond one to one."
            )
        sigmas = np.stack([_as_2d(values) for values in member_sigmas], axis=0)
        if sigmas.shape != stacked.shape:
            raise ValueError(f"member_sigmas shape {sigmas.shape} does not match member predictions {stacked.shape}.")
        aleatoric_std = np.sqrt((sigmas**2).mean(axis=0))

    return EnsemblePrediction(
        mean=mean,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
    )


def _as_2d(values: Any) -> np.ndarray:
    """One member's predictions as a points-by-targets array, whatever shape arrived."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Member predictions must be 1-D or 2-D; got shape {array.shape}")
