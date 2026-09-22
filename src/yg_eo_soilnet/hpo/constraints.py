"""Hooks for settings that depend on one another, which a YAML list cannot express.

Almost every setting is drawn on its own and belongs in the :term:`search space`. The exceptions are
settings constrained by each other - a width that has to divide by a head count, or a list of layer
widths whose length is itself being searched. Those are named hooks here, and a search space asks
for one by name.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping, MutableMapping

import optuna

ConstraintHook = Callable[[optuna.Trial, MutableMapping[str, Any]], None]

CONSTRAINTS: dict[str, ConstraintHook] = {}


def constraint(name: str) -> Callable[[ConstraintHook], ConstraintHook]:
    """Register a hook under a name a search space can ask for."""
    def register(hook: ConstraintHook) -> ConstraintHook:
        """Record the decorated function under that name."""
        if name in CONSTRAINTS:
            raise ValueError(f"A constraint hook named {name!r} is already registered.")
        CONSTRAINTS[name] = hook
        return hook

    return register


def split_derive_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    """Read a ``derive:`` entry: a bare name, or a name with options.

    The options form lets one hook serve several search spaces on their own terms.
        """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, Mapping) and len(entry) == 1:
        name, options = next(iter(entry.items()))
        if isinstance(name, str) and isinstance(options or {}, Mapping):
            return name, dict(options or {})
    raise ValueError(
        f"Malformed 'derive' entry {entry!r}; expected a hook name, or a single-key mapping of "
        f"a hook name to its options, e.g. {{head_hidden_dims_pyramid: {{floor: 16}}}}."
    )


def split_when(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a hook's options from its condition.

    The condition belongs to the search space, not to the hook: the hook runs only in trials where the
    named settings took those values.
        """
    options = dict(options)
    when = options.pop("when", None) or {}
    if not isinstance(when, Mapping):
        raise ValueError(f"A derive entry's 'when' must map parameters to values; got {when!r}.")
    return options, dict(when)


def _guarded(hook: ConstraintHook, when: Mapping[str, Any]) -> ConstraintHook:
    """Wrap a hook so it runs only in trials matching its condition."""
    def run(trial: optuna.Trial, chosen: MutableMapping[str, Any]) -> None:
        """Run the hook if this trial matches, otherwise do nothing."""
        if all(key in chosen and chosen[key] == expected for key, expected in when.items()):
            hook(trial, chosen)

    return run


def resolve_constraints(entries: list[Any]) -> list[ConstraintHook]:
    """Look up the hooks a search space names, failing now rather than mid-study.

    Returns
    -------
    list of callable
        Each takes the trial and what has been drawn so far, and writes its own settings in.
        """
    parsed = [split_derive_entry(entry) for entry in entries]
    unknown = [name for name, _ in parsed if name not in CONSTRAINTS]
    if unknown:
        available = ", ".join(sorted(CONSTRAINTS)) or "(none)"
        raise ValueError(f"Unknown constraint hook(s): {', '.join(unknown)}. Available: {available}.")
    hooks: list[ConstraintHook] = []
    for name, options in parsed:
        options, when = split_when(options)
        hook = partial(CONSTRAINTS[name], **options) if options else CONSTRAINTS[name]
        hooks.append(_guarded(hook, when) if when else hook)
    return hooks


@constraint("d_model_divisible_by_nhead")
def d_model_divisible_by_nhead(
    trial: optuna.Trial,
    chosen: MutableMapping[str, Any],
    *,
    d_model_key: str = "model.d_model",
    nhead_key: str = "model.nhead",
) -> None:
    """Round a width up until it divides by the number of attention heads.

    The model refuses a width that does not. Repairing the draw rather than rejecting the trial keeps
    every trial useful.
        """
    d_model = chosen.get(d_model_key)
    nhead = chosen.get(nhead_key)
    if d_model is None or nhead is None:
        return

    d_model, nhead = int(d_model), int(nhead)
    remainder = d_model % nhead
    if remainder:
        chosen[d_model_key] = d_model + (nhead - remainder)


DEFAULT_PYRAMID_WIDTHS = [32, 64, 128, 256]


@constraint("dims_pyramid")
def dims_pyramid(
    trial: optuna.Trial,
    chosen: MutableMapping[str, Any],
    *,
    key: str = "model.head_hidden_dims",
    min_depth: int = 1,
    max_depth: int = 3,
    widths: list[int] | None = None,
    floor: int = 8,
    taper: float = 0.5,
) -> None:
    """Draw a list of layer widths: so many layers, each a fraction of the one before.

    Every list-valued width in the models - the head, the covariate branch, the CNN - is searched this
    way, since the number of layers and their widths cannot be drawn independently.
        """
    label = key.rsplit(".", 1)[-1]
    depth = trial.suggest_int(f"{label}_depth", int(min_depth), int(max_depth))
    if depth <= 0:
        # No width to draw: a zero-layer stack is a bare projection, and suggesting one anyway
        # would leave the sampler a dimension that changes nothing about the trial.
        chosen[key] = []
        return

    base = trial.suggest_categorical(
        f"{label}_width", [int(w) for w in (widths or DEFAULT_PYRAMID_WIDTHS)]
    )
    chosen[key] = [max(int(floor), int(base * float(taper) ** step)) for step in range(depth)]
