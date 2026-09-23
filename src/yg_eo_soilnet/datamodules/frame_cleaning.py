"""Repair and filter raw data tables: fill small gaps, refuse columns that are too empty.

Both model families clean their inputs through these helpers, so a column is judged the same way
whichever model reads it.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def build_finite_row_mask(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.Series:
    """Mark the rows whose values are all present and finite.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to check.
    required_columns : iterable of str, optional
        Columns that must not be blank, such as ids, dates and targets.
    numeric_columns : iterable of str, optional
        Columns that must hold a finite number; text that is not a number counts as missing.

    Returns
    -------
    pandas.Series of bool
        True for the rows to keep, in the table's own order.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"uuid": ["a", "b", None], "clay_pct": [22.0, float("nan"), 30.0]})
    >>> list(build_finite_row_mask(frame, required_columns=["uuid"], numeric_columns=["clay_pct"]))
    [True, False, False]
    """
    mask = pd.Series(True, index=frame.index)

    required_columns = [column for column in required_columns if column in frame.columns]
    if required_columns:
        mask &= frame[required_columns].notna().all(axis=1)

    for column in numeric_columns:
        if column not in frame.columns:
            continue
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        mask &= np.isfinite(numeric_values.to_numpy(dtype=np.float64, copy=False))

    return mask


class SparseColumnError(ValueError):
    """Raised when a covariate is missing on too many rows to be filled in honestly.

    Attributes
    ----------
    offenders : list of tuple
        ``(column, missing_rows, missing_ratio)`` for every column over the limit, worst first.
    """

    def __init__(self, message: str, offenders: "list[tuple[str, int, float]]"):
        super().__init__(message)
        self.offenders = offenders


def column_missing_ratios(frame: pd.DataFrame, columns: Iterable[str]) -> "dict[str, float]":
    """Measure the share of rows on which each column is missing or not a finite number.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to measure.
    columns : iterable of str
        The columns to measure; names not in the table are skipped.

    Returns
    -------
    dict of str to float
        One share per column, between 0 (never missing) and 1 (always missing).

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"clay_pct": [22.0, float("nan"), 30.0]})
    >>> round(column_missing_ratios(frame, ["clay_pct"])["clay_pct"], 2)
    0.33
    """
    if frame.empty:
        return {column: 0.0 for column in columns if column in frame.columns}

    ratios: "dict[str, float]" = {}
    for column in dict.fromkeys(columns):
        if column not in frame.columns:
            continue
        mask = build_finite_row_mask(frame, numeric_columns=[column])
        ratios[column] = float((~mask).sum()) / float(len(frame))
    return ratios


def assert_columns_are_dense_enough(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    max_missing_ratio: float,
    label: str,
    logger: Any,
    allow: Iterable[str] = (),
    fail: bool = True,
) -> "list[tuple[str, int, float]]":
    """Stop the run when a covariate is missing on too many rows to be filled in honestly.

    Gaps are filled with the column's median, which repairs a few rows but invents the column when
    most of it is blank. The limit is ``common.data_quality.max_missing_column_ratio`` (20% as
    shipped) and ``allow_sparse_columns`` exempts named columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to check.
    columns : iterable of str
        The covariates to check.
    max_missing_ratio : float
        The largest share of missing rows a covariate may have, between 0 and 1.
    label : str
        What this table is, named in the message.
    logger : logging.Logger
        Where the warning goes when ``fail`` is false.
    allow : iterable of str, optional
        Covariates exempted from the check.
    fail : bool, default True
        Raise on an offending column; false only warns and carries on.

    Returns
    -------
    list of tuple
        ``(column, missing_rows, missing_ratio)`` for every column over the limit, worst first.
        Empty when they all pass.

    Raises
    ------
    SparseColumnError
        If a covariate is over the limit and ``fail`` is true. The message names the columns and how
        to let them through.
    """
    allowed = set(allow or ())
    ratios = column_missing_ratios(frame, columns)
    offenders = [
        (column, int(round(ratio * len(frame))), ratio)
        for column, ratio in ratios.items()
        if ratio > max_missing_ratio and column not in allowed
    ]
    offenders.sort(key=lambda item: item[2], reverse=True)
    if not offenders:
        return []

    listed = "\n".join(
        f"  {column}: blank on {missing} of {len(frame)} rows ({ratio:.1%})" for column, missing, ratio in offenders
    )
    message = (
        f"{len(offenders)} covariate(s) in {label} are blank on more than "
        f"{max_missing_ratio:.1%} of rows, so imputing them would fabricate most of the column "
        f"rather than recover it:\n{listed}\n"
        f"Either drop them - add the names to IGNORED_COLUMNS/ELIMINATED_FEATURES in "
        f"data_spec.yml - or, if the imputed values are genuinely wanted, list them under "
        f"common.data_quality.allow_sparse_columns. Raising "
        f"common.data_quality.max_missing_column_ratio above {max(r for _, _, r in offenders):.2f} "
        f"would also let them through."
    )
    if fail:
        raise SparseColumnError(message, offenders)
    logger.warning(f"{message}\n(fail_on_sparse_columns is false, so the run continues.)")
    return offenders


def drop_non_finite_rows(
    frame: pd.DataFrame,
    *,
    logger: Any,
    label: str,
    required_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Remove the rows holding a missing or non-finite value, and report how many went.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to filter.
    logger : logging.Logger
        Where the count of dropped rows goes.
    label : str
        What this table is, named in the message.
    required_columns : iterable of str, optional
        Columns that must not be blank.
    numeric_columns : iterable of str, optional
        Columns that must hold a finite number.

    Returns
    -------
    pandas.DataFrame
        A copy holding the rows that passed.
    """
    if frame.empty:
        return frame.copy()

    mask = build_finite_row_mask(
        frame,
        required_columns=required_columns,
        numeric_columns=numeric_columns,
    )
    if bool(mask.all()):
        return frame.copy()

    dropped_count = int((~mask).sum())
    kept_frame = frame.loc[mask].copy()
    logger.warning(
        f"Dropped {dropped_count} row(s) with non-finite values from {label}; remaining rows: {len(kept_frame)}"
    )
    if dropped_count and dropped_count > len(frame) // 2:
        # A row goes if any one of its required values is missing, so a nearly empty covariate can
        # take most of the dataset with it. Name the worst offenders, which is what to fix.
        logger.warning(
            f"That is most of {label}. Worst columns by rows lost: "
            + ", ".join(
                f"{name} ({count})"
                for name, count in worst_non_finite_columns(
                    frame, required_columns=required_columns, numeric_columns=numeric_columns
                )
            )
            + ". Drop them via IGNORED_COLUMNS/ELIMINATED_FEATURES in data_spec.yml if they are "
            "not worth the rows they cost."
        )
    return kept_frame


def worst_non_finite_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
    limit: int = 5,
) -> list[tuple[str, int]]:
    """List the columns costing the most rows, worst first.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table being filtered.
    required_columns : iterable of str, optional
        Columns that must not be blank.
    numeric_columns : iterable of str, optional
        Columns that must hold a finite number.
    limit : int, default 5
        How many columns to name.

    Returns
    -------
    list of tuple
        ``(column, rows_lost)``, worst first.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"uuid": ["a", "b", None], "clay_pct": [22.0, float("nan"), 30.0]})
    >>> worst_non_finite_columns(frame, required_columns=["uuid"], numeric_columns=["clay_pct"])
    [('uuid', 1), ('clay_pct', 1)]
    """
    counts: list[tuple[str, int]] = []
    for column in dict.fromkeys([*required_columns, *numeric_columns]):
        if column not in frame.columns:
            continue
        column_mask = build_finite_row_mask(
            frame,
            required_columns=[column] if column in set(required_columns) else (),
            numeric_columns=[column] if column in set(numeric_columns) else (),
        )
        lost = int((~column_mask).sum())
        if lost:
            counts.append((column, lost))
    counts.sort(key=lambda item: item[1], reverse=True)
    return counts[:limit]


def sanitize_numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    logger: Any,
    return_validity: bool = False,
):
    """Make the named columns numeric and finite without dropping any rows.

    Repairs three things real files contain: numbers written with a decimal comma (``"0,05"``),
    infinities from ratio indices such as NDVI, and missing cells - filled with the column's median.

    Parameters
    ----------
    frame : pandas.DataFrame
        The table to repair.
    columns : iterable of str
        The columns to repair; names not in the table are skipped.
    logger : logging.Logger
        Where the counts of repaired values go.
    return_validity : bool, default False
        Also return the :term:`validity flags <validity flag>`.

    Returns
    -------
    sanitized : pandas.DataFrame
        A copy with those columns as 32-bit floats.
    validity : pandas.DataFrame of bool
        Only with ``return_validity``: True where the cell held a real value **before** the median
        fill, so a model can tell a measurement from a filled-in gap.

    Examples
    --------
    >>> import logging, pandas as pd
    >>> frame = pd.DataFrame({"S2_B4": ["0,05", "0,07"]})
    >>> repaired = sanitize_numeric_columns(frame, ["S2_B4"], logger=logging.getLogger("demo"))
    >>> [round(float(value), 3) for value in repaired["S2_B4"]]
    [0.05, 0.07]
    """
    columns = [column for column in dict.fromkeys(columns) if column in frame.columns]
    if not columns:
        return (frame, pd.DataFrame(index=frame.index)) if return_validity else frame

    sanitized = frame.copy()
    validity: dict[str, np.ndarray] = {}
    repaired_text: dict[str, int] = {}
    non_finite: dict[str, int] = {}

    for column in columns:
        series = sanitized[column]
        if series.dtype == object:
            as_text = series.astype("string")
            comma_count = int(as_text.str.contains(",", na=False).sum())
            if comma_count:
                repaired_text[column] = comma_count
                as_text = as_text.str.replace(",", ".", regex=False)
            series = pd.to_numeric(as_text, errors="coerce")
        else:
            series = pd.to_numeric(series, errors="coerce")

        series = series.replace([np.inf, -np.inf], np.nan)
        # Recorded before the fill below, after which a filled cell is indistinguishable.
        validity[column] = series.notna().to_numpy(dtype=bool)
        missing = int(series.isna().sum())
        if missing:
            non_finite[column] = missing
            median = series.median()
            series = series.fillna(0.0 if pd.isna(median) else median)
        sanitized[column] = series.astype(np.float32)

    if repaired_text:
        logger.warning(
            f"Repaired decimal-comma values in time-series column(s): "
            f"{', '.join(f'{name} ({count})' for name, count in repaired_text.items())}"
        )
    if non_finite:
        total = sum(non_finite.values())
        worst = sorted(non_finite.items(), key=lambda item: item[1], reverse=True)[:3]
        logger.warning(
            f"Median-filled {total} non-finite time-series cell(s) across {len(non_finite)} column(s); "
            f"worst: {', '.join(f'{name} ({count})' for name, count in worst)}"
        )

    if return_validity:
        return sanitized, pd.DataFrame(validity, index=frame.index)
    return sanitized
