"""Seed every random-number generator, so a run can be repeated exactly."""

from __future__ import annotations

import importlib
import random

import numpy as np

try:  # pragma: no cover - optional dependency, mirrors the rest of the package
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (and data-loading worker processes) from one number.

    Call it just before building a deep-learning model: the starting weights are drawn at that
    moment, so seeding first is what makes two runs - or a tuned config and the tuning trial it came
    from - start from the same weights.

    Parameters
    ----------
    seed : int
        The seed, usually ``RANDOM_SEED`` from the configuration.

    Examples
    --------
    >>> import numpy as np
    >>> seed_everything(42); first = np.random.rand()
    >>> seed_everything(42); second = np.random.rand()
    >>> first == second
    True
    """
    seed_value = int(seed)
    try:
        lightning = importlib.import_module("lightning.pytorch")
    except ImportError:  # pragma: no cover - exercised only when lightning is absent
        lightning = None
    lightning_seed = getattr(lightning, "seed_everything", None)
    if callable(lightning_seed):
        lightning_seed(seed_value, workers=True)
        return

    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch is not None:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
