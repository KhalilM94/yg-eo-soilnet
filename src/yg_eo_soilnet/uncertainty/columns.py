"""The names of the uncertainty columns in a results table, and how to read them.

Both model families write the same names - ``prediction_std``, ``prediction_lower``,
``prediction_upper``, suffixed with the target when a run has several - so one reader handles
either. The figures and the scores all go through the helpers here rather than spelling the names
out again.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# The stems, in the order they are appended to a frame.
STD = "prediction_std"
EPISTEMIC_STD = "prediction_epistemic_std"
ALEATORIC_STD = "prediction_aleatoric_std"
LOWER = "prediction_lower"
UPPER = "prediction_upper"

UNCERTAINTY_STEMS: tuple[str, ...] = (STD, EPISTEMIC_STD, ALEATORIC_STD, LOWER, UPPER)

# Every prefix that starts with "prediction_" but is NOT a per-target prediction column.
#
# This exists because of a specific trap. `_resolve_prediction_column` in plot_utils falls back to
# "the first column starting with prediction_", and `prediction_std_clay_pct` sorts before
# `prediction_clay_pct` in some frames - so without this guard a plot can silently draw standard
# deviations on the predicted axis. Any new column added above must be listed here.
NON_PREDICTION_PREFIXES: tuple[str, ...] = tuple(f"{stem}_" for stem in UNCERTAINTY_STEMS)


def column_name(stem: str, target_name: Optional[str] = None, *, multi_target: bool = False) -> str:
    """The column name for one kind of uncertainty value.

    Parameters
    ----------
    stem : str
        ``"prediction_std"``, ``"prediction_lower"`` or ``"prediction_upper"``.
    target_name : str, optional
        The target, appended when the run has several.
    multi_target : bool, default False
        Whether this run predicts several targets.

    Returns
    -------
    str

    Examples
    --------
    >>> column_name("prediction_std")
    'prediction_std'
    >>> column_name("prediction_std", "clay_pct", multi_target=True)
    'prediction_std_clay_pct'
    """
    if not multi_target or target_name is None:
        return stem
    return f"{stem}_{target_name}"


def is_prediction_column(column_name_value: Any) -> bool:
    """Whether a column holds predictions rather than an uncertainty value.

    The test any reader scanning for prediction columns should use: without it, ``prediction_std`` and
    ``prediction_lower`` would be counted as targets.

    Examples
    --------
    >>> is_prediction_column("prediction_clay_pct"), is_prediction_column("prediction_std")
    (True, False)
    """
    name = str(column_name_value)
    if not name.startswith("prediction"):
        return False
    if name in UNCERTAINTY_STEMS:
        return False
    return not any(name.startswith(prefix) for prefix in NON_PREDICTION_PREFIXES)


def interval_columns(
    frame: pd.DataFrame,
    target_name: Optional[str] = None,
) -> Optional[tuple[pd.Series, pd.Series]]:
    """The lower and upper bounds for one target, or None when the table has none.

    None rather than an error: most runs have no uncertainty, and every caller would otherwise have to
    check first.
    """
    lower = _first_present(frame, LOWER, target_name)
    upper = _first_present(frame, UPPER, target_name)
    if lower is None or upper is None:
        return None
    return frame[lower], frame[upper]


def sigma_column(frame: pd.DataFrame, target_name: Optional[str] = None) -> Optional[pd.Series]:
    """The predicted spread for one target, or None when the table has none."""
    name = _first_present(frame, STD, target_name)
    return None if name is None else frame[name]


def _first_present(frame: pd.DataFrame, stem: str, target_name: Optional[str]) -> Optional[str]:
    """The suffixed column name if the table has it, else the plain one, else None.

    Suffixed first: a single target's table is a copy of the whole run's, so it carries every target's
    columns and the plain name may be missing.
    """
    if target_name:
        suffixed = f"{stem}_{target_name}"
        if suffixed in frame.columns:
            return suffixed
    if stem in frame.columns:
        return stem
    return None
