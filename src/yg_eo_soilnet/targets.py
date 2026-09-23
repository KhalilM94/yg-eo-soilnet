"""Decide which targets each model predicts, and name the resulting groups.

With several targets, one model can predict them all at once or each target can get its own model.
``MULTI_TARGET_MODE`` in ``data_spec.yml`` chooses, for both model families, and a model-list entry
can override it. The answer is a list of :term:`target groups <target group>`, one model per group::

    joint       -> [[a, b, c]]        one model predicting three targets
    per_target  -> [[a], [b], [c]]    three models, one target each

A group is named by joining its target names with ``__`` (``clay_pct__ph_water``). That name labels
the MLflow runs and the saved files, so a target name may not itself contain ``__``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

#: One model predicts every target.
JOINT = "joint"
#: One model per target.
PER_TARGET = "per_target"
#: The values ``MULTI_TARGET_MODE`` accepts.
VALID_MODES = (JOINT, PER_TARGET)

#: A scikit-learn model list entry says ``multi_target: native`` when the estimator can predict
#: several targets at once. That is a capability, not a mode: such an entry still follows
#: ``MULTI_TARGET_MODE``.
NATIVE = "native"

#: Joins target names into a group name.
TARGET_NAME_SEPARATOR = "__"


def join_target_names(names: Sequence[str]) -> str:
    """Name a group of targets by joining their names with ``__``.

    A group of one is named after its target.

    Parameters
    ----------
    names : sequence of str
        The targets in the group.

    Returns
    -------
    str
        The group name, used to label runs and files.

    Raises
    ------
    ValueError
        If ``names`` is empty, or a group of several contains a name with ``__`` in it.

    Examples
    --------
    >>> join_target_names(["clay_pct", "ph_water"])
    'clay_pct__ph_water'
    >>> join_target_names(["clay_pct"])
    'clay_pct'
    """
    names = [str(name) for name in names]
    if not names:
        raise ValueError("Cannot build a run label from an empty target group.")
    if len(names) == 1:
        return names[0]
    # The separator is also how the name is read back, so a target name containing it would split
    # into pieces naming no column.
    offenders = [name for name in names if TARGET_NAME_SEPARATOR in name]
    if offenders:
        raise ValueError(
            f"Target names may not contain {TARGET_NAME_SEPARATOR!r}, which separates the names of "
            f"a joint run: {sorted(offenders)}. Rename the column or fit these targets separately."
        )
    return TARGET_NAME_SEPARATOR.join(names)


def split_target_names(encoded: Any) -> list[str]:
    """Split a group name back into its target names; the reverse of :func:`join_target_names`.

    Parameters
    ----------
    encoded : str or None
        A group name such as ``"clay_pct__ph_water"``.

    Returns
    -------
    list of str

    Examples
    --------
    >>> split_target_names("clay_pct__ph_water")
    ['clay_pct', 'ph_water']
    >>> split_target_names(None)
    []
    """
    if encoded is None:
        return []
    return [name for name in str(encoded).split(TARGET_NAME_SEPARATOR) if name]


def resolve_mode(config: Any, spec: Optional[Mapping[str, Any]] = None) -> str:
    """Return the grouping mode for one model: ``"joint"`` or ``"per_target"``.

    A model-list entry's own ``multi_target`` setting wins over the configuration's
    ``MULTI_TARGET_MODE`` (``"joint"`` unless set). ``multi_target: native`` is a capability, not a
    mode, so it leaves the configured mode in force.

    Parameters
    ----------
    config : Config
        The run configuration.
    spec : mapping, optional
        The model's entry in the model list.

    Returns
    -------
    str
        ``"joint"`` or ``"per_target"``.

    Raises
    ------
    ValueError
        If either setting holds an unknown value.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> config = SimpleNamespace(MULTI_TARGET_MODE="joint")
    >>> resolve_mode(config)
    'joint'
    >>> resolve_mode(config, {"multi_target": "per_target"})
    'per_target'
    >>> resolve_mode(config, {"multi_target": "native"})
    'joint'
    """
    if spec is not None:
        declared = spec.get("multi_target")
        if declared is not None and str(declared).lower() != NATIVE:
            mode = str(declared).lower()
            if mode not in VALID_MODES:
                raise ValueError(
                    f"multi_target must be one of {VALID_MODES} or {NATIVE!r}, got {declared!r}."
                )
            return mode

    mode = str(getattr(config, "MULTI_TARGET_MODE", JOINT) or JOINT).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"MULTI_TARGET_MODE must be one of {VALID_MODES}, got {mode!r}.")
    return mode


def supports_joint(spec: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether a scikit-learn model-list entry can predict several targets at once.

    Only if the entry says ``multi_target: native``. Several estimators predict one target at a time
    (gradient boosting, TabICL), so an entry has to declare the capability. Deep-learning models
    always have it and do not use this check.

    Parameters
    ----------
    spec : mapping, optional
        The model's entry in the model list.

    Returns
    -------
    bool

    Examples
    --------
    >>> supports_joint({"multi_target": "native"})
    True
    >>> supports_joint({"params": {"alpha": [1.0]}})
    False
    """
    return spec is not None and str(spec.get("multi_target", "")).lower() == NATIVE


def resolve_target_groups(
    config: Any,
    spec: Optional[Mapping[str, Any]] = None,
    *,
    require_joint_support: bool = False,
    logger: Any = None,
    entry_name: str = "",
) -> list[list[str]]:
    """List the target groups one model is trained on - one trained model per group.

    Parameters
    ----------
    config : Config
        The run configuration; reads ``TARGET_COLUMNS`` and ``MULTI_TARGET_MODE``.
    spec : mapping, optional
        The model's entry in the model list.
    require_joint_support : bool, default False
        Set for scikit-learn models. In joint mode, an entry that has not declared
        ``multi_target: native`` gets one model per target instead, with a warning: one estimator
        that cannot predict several targets must not stop the run.
    logger : logging.Logger, optional
        Where that warning goes.
    entry_name : str, optional
        The model's name, named in the warning.

    Returns
    -------
    list of list of str
        One list of target names per model. Empty if no targets are configured.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> config = SimpleNamespace(TARGET_COLUMNS=["clay_pct", "ph_water"], MULTI_TARGET_MODE="joint")
    >>> resolve_target_groups(config)
    [['clay_pct', 'ph_water']]
    >>> resolve_target_groups(config, {}, require_joint_support=True)   # not multi-target
    [['clay_pct'], ['ph_water']]
    """
    targets = [str(name) for name in (getattr(config, "TARGET_COLUMNS", []) or [])]
    if not targets:
        return []
    if len(targets) == 1:
        return [targets]

    mode = resolve_mode(config, spec)
    if mode == JOINT and require_joint_support and not supports_joint(spec):
        if logger is not None:
            label = entry_name or "this entry"
            logger.warning(
                f"MULTI_TARGET_MODE is 'joint' but {label} does not declare 'multi_target: native', "
                f"so it cannot fit a 2-D target. Falling back to one model per target."
            )
        mode = PER_TARGET

    return [list(targets)] if mode == JOINT else [[name] for name in targets]


def group_label(group: Iterable[str]) -> str:
    """Name a target group held as any iterable; see :func:`join_target_names`.

    Examples
    --------
    >>> group_label(("clay_pct", "ph_water"))
    'clay_pct__ph_water'
    """
    return join_target_names(list(group))


def select_target_columns(
    targets: Any,
    target_names: Sequence[str],
    active_targets: Optional[Sequence[str]],
) -> tuple[Any, list[str], Optional[list[int]]]:
    """Keep only the columns of the targets one model predicts.

    The deep-learning data is prepared once for every configured target and reused, so a model that
    predicts one target takes its column here instead of preparing the data again.

    Parameters
    ----------
    targets : array-like of shape (n_points, n_targets)
        Target values, one column per name in ``target_names``.
    target_names : sequence of str
        The names of those columns.
    active_targets : sequence of str or None
        The targets to keep, in the order wanted. None or empty keeps everything.

    Returns
    -------
    values : array-like
        The selected columns, or ``targets`` unchanged when nothing was removed.
    names : list of str
        Their names.
    indices : list of int or None
        Where the selected columns sat in ``targets``; None when nothing was removed.

    Raises
    ------
    ValueError
        If a wanted target is not among ``target_names``.

    Examples
    --------
    >>> import numpy as np
    >>> values, names, indices = select_target_columns(
    ...     np.array([[1, 2, 3], [4, 5, 6]]), ["a", "b", "c"], ["c", "a"])
    >>> values
    array([[3, 1],
           [6, 4]])
    >>> names, indices
    (['c', 'a'], [2, 0])
    """
    import numpy as np

    names = [str(name) for name in target_names]
    if not active_targets:
        return targets, names, None

    wanted = [str(name) for name in active_targets]
    if wanted == names:
        return targets, names, None

    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(
            f"active_targets names columns the bundle does not carry: {sorted(missing)}. "
            f"Available: {names}."
        )

    indices = [names.index(name) for name in wanted]
    array = np.asarray(targets)
    # An empty block has no columns to take: leave its (0, 0) shape alone.
    if array.ndim != 2 or array.shape[1] == 0:
        return targets, wanted, indices
    return array[:, indices], wanted, indices
