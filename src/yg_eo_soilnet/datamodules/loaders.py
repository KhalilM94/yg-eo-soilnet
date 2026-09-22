"""Build DataLoaders whose batch order does not depend on how many processes load them.

A DataLoader left to itself shuffles using the same global random stream the model's dropout draws
from, and how many numbers it draws depends on ``num_workers`` and ``persistent_workers``. Those
settings then change the training itself, and a run no longer reproduces one made with the same seed
but different loading settings.

Every loader built here gets its own generators, seeded once from the global stream. The run seed
still decides the order - a different seed, or a different ensemble member, still shuffles
differently - while the loading settings change only how fast batches arrive.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


def _private_generator() -> torch.Generator:
    """A random generator seeded by a single draw from the global stream."""
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    return torch.Generator().manual_seed(seed)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    collate_fn: Callable[[Any], Any] | None = None,
) -> DataLoader:
    """Build a DataLoader with its own random generators.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        What to load from.
    batch_size : int
        Points per batch.
    shuffle : bool, default False
        Draw the points in a new order each epoch. Used for training, not for scoring.
    drop_last : bool, default False
        Throw away the last batch when it is smaller than the others.
    num_workers : int, default 0
        Background processes preparing batches; 0 prepares them in the main process.
    pin_memory : bool, default False
        Speeds up copying batches to a GPU.
    persistent_workers : bool, default False
        Keep the worker processes alive between epochs instead of starting them again.
    collate_fn : callable, optional
        How single points are combined into a batch.

    Returns
    -------
    torch.utils.data.DataLoader

    Notes
    -----
    The sampler and the loader get one generator each: they draw at different moments, and sharing
    one would let the loading settings shift the shuffling again.
    """
    # Drawn whether or not this loader shuffles, so the number of draws from the global stream
    # never depends on the settings either.
    sampler_generator = _private_generator()
    loader_generator = _private_generator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset, generator=sampler_generator) if shuffle else None,
        drop_last=bool(drop_last),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
