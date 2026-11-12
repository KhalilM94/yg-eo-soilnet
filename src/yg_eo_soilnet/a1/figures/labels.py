"""Plain names for the A1 figures, read from ``configs/a1/labels.yml``.

Every figure labels its models, split kinds, covariate groups and variables through
:func:`pretty_name` (and :func:`pretty_split` for a split kind with its block size), so a reader
never meets a code name such as ``global_aux`` or ``S2_B12_q75``. The names live in a YAML file the
author edits; a name it does not cover is shown as it is and logged once.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

#: The labels file shipped with the repository.
LABELS_FILE = Path(__file__).resolve().parents[4] / "configs" / "a1" / "labels.yml"
#: A time-series percentile column: ``<variable>_q10`` ... ``_iqr``, or the computed ``ts_<variable>_p10``.
PERCENTILE = re.compile(r"^(?:ts_)?(?P<variable>.+?)_(?P<suffix>q\d{2}|iqr|p\d{1,3})$")
#: A bare-soil Sentinel-2 band: ``s2_b12_barest`` and its spread ``s2_b12_barest_sd``.
BAREST = re.compile(r"^s2_(?P<band>b\d+a?)_barest(?P<sd>_sd)?$", re.IGNORECASE)
#: A stage-1 prediction column.
PREDICTION = re.compile(r"^a1_pred_(?P<target>.+)$")

_warned: set = set()


@lru_cache(maxsize=4)
def load_labels(path: Optional[str] = None) -> dict:
    """The labels file, read once (``configs/a1/labels.yml`` unless another is given)."""
    with open(path or LABELS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _unknown(name: str) -> str:
    if name not in _warned:
        _warned.add(name)
        logger.info(f"No plain name for {name!r} in {LABELS_FILE.name}; shown as it is.")
    return name


def _suffix(suffix: str, labels: dict) -> Optional[str]:
    known = (labels.get("suffixes") or {}).get(suffix)
    if known:
        return known
    if suffix.startswith("p") and suffix[1:].isdigit():  # a computed percentile, ts_<variable>_p75
        number = int(suffix[1:])
        return "median" if number == 50 else f"{number}th pct"
    return None


def pretty_name(name, labels: Optional[dict] = None) -> str:
    """The plain name of a model, covariate group, carbonate class or input column.

    Examples
    --------
    >>> pretty_name("per_class_measured")
    'Per class, measured CaCO₃ (upper bound)'
    >>> pretty_name("S2_B12_q75"), pretty_name("s2_b12_barest"), pretty_name("a1_pred_c_e_c_meq_100g")
    ('S2 B12 (SWIR2), 75th pct', 'Bare-soil B12 (SWIR2)', 'Predicted CEC')
    >>> pretty_name("legacy_soil"), pretty_name("CLIM_ETa_iqr"), pretty_name("ts_S2_B4_p10")
    ('Legacy soil maps', 'Actual ET, IQR', 'S2 B4 (red), 10th pct')
    """
    labels = labels if labels is not None else load_labels()
    name = str(name)
    for section in ("models", "groups", "classes", "columns", "variables"):
        found = (labels.get(section) or {}).get(name)
        if found:
            return found
    kind = (labels.get("kinds") or {}).get(name)
    if kind and "{" not in kind:
        return kind
    prediction = PREDICTION.match(name)
    if prediction:
        return f"Predicted {pretty_name(prediction.group('target'), labels)}"
    barest = BAREST.match(name)
    if barest:
        band = (labels.get("bands") or {}).get(barest.group("band").lower())
        if band:
            return f"Bare-soil {band}" + (", spread" if barest.group("sd") else "")
    percentile = PERCENTILE.match(name)
    if percentile:
        variable = (labels.get("variables") or {}).get(percentile.group("variable"))
        suffix = _suffix(percentile.group("suffix"), labels)
        if variable and suffix:
            return f"{variable}, {suffix}"
    return _unknown(name)


def pretty_split(kind: str, block_size_m=None, labels: Optional[dict] = None) -> str:
    """The plain name of a split kind, with its block size for spatial blocks.

    Examples
    --------
    >>> pretty_split("spatial_kfold", 5000), pretty_split("logo"), pretty_split("random")
    ('5 km blocks', 'Leave one basin out', 'Random')
    """
    labels = labels if labels is not None else load_labels()
    kinds = labels.get("kinds") or {}
    if kind == "spatial_kfold":
        if block_size_m is None or pd.isna(block_size_m):
            return kinds.get("spatial_kfold_kmeans", "Spatial clusters")
        return kinds.get("spatial_kfold", "{km} km blocks").format(km=f"{float(block_size_m) / 1000:g}")
    return kinds.get(kind) or _unknown(kind)


def variable_of(column: str) -> str:
    """The variable a column measures: the part of a percentile column before its suffix.

    Examples
    --------
    >>> variable_of("S2_B12_q75"), variable_of("ts_CLIM_ETa_p50"), variable_of("elevation")
    ('S2_B12', 'CLIM_ETa', 'elevation')
    """
    match = PERCENTILE.match(str(column))
    return match.group("variable") if match else str(column)


def percentile_of(column: str) -> Optional[str]:
    """The percentile suffix of a column (``q50``, ``iqr``, ``p75``), or None."""
    match = PERCENTILE.match(str(column))
    return match.group("suffix") if match else None


def source_of(column: str, labels: Optional[dict] = None) -> Optional[str]:
    """The figure-3 source a column belongs to (``labels.yml`` ``sources``), or None.

    Examples
    --------
    >>> source_of("S2_B12_q50"), source_of("s2_b11_barest"), source_of("s2_b11_barest_sd"), source_of("S2_B4_q50")
    ('Sentinel-2 SWIR', 'Bare-soil SWIR', None, None)
    """
    labels = labels if labels is not None else load_labels()
    variable = variable_of(column)
    for source in labels.get("sources") or []:
        for prefix in source.get("prefixes") or []:
            if (prefix.endswith("_") and variable.startswith(prefix)) or variable == prefix:
                return source["name"]
    return None


def source_order(labels: Optional[dict] = None) -> list:
    """The figure-3 sources, in the file's order."""
    labels = labels if labels is not None else load_labels()
    return [source["name"] for source in labels.get("sources") or []]
