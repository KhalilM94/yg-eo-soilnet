"""The groups A1 breaks its results down by: the FAO carbonate classes.

How calcareous a soil is changes how organic matter shows in its reflectance - carbonate brightens
the soil and masks the darkening organic matter brings (see the paper figures, Figure 7) - so a
model's error is expected to depend on it. The classes are the FAO calcareousness classes, cut
exactly as in the paper figures (``notebooks/paper_figures.ipynb``, cell 20).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: The FAO class boundaries, in % total CaCO3. A value on a boundary belongs to the class above it.
FAO_CARBONATE_EDGES = (2.0, 10.0, 25.0)
#: Slightly, moderately, strongly and extremely calcareous.
FAO_CARBONATE_LABELS = ("<2", "2-10", "10-25", ">=25")


def carbonate_class(
    caco3: pd.Series,
    edges: Sequence[float] = FAO_CARBONATE_EDGES,
    labels: Sequence[str] = FAO_CARBONATE_LABELS,
) -> pd.Series:
    """Put each CaCO3 value in its FAO carbonate class.

    The classes are closed on the left and open on the right, so a value exactly on a boundary
    goes to the class above it: 2 % is moderately calcareous, 25 % extremely. A missing value has
    no class.

    Parameters
    ----------
    caco3 : pandas.Series
        Total CaCO3, in %.
    edges : sequence of float
        The class boundaries, rising.
    labels : sequence of str
        One name per class, one more than there are boundaries.

    Returns
    -------
    pandas.Series
        An ordered categorical with the same index, so classes sort from least to most calcareous
        and a missing value stays missing.

    Raises
    ------
    ValueError
        If the boundaries do not rise, or there is not one name per class.

    Examples
    --------
    >>> import pandas as pd
    >>> carbonate_class(pd.Series([0.5, 2.0, 9.99, 10.0, 25.0, None])).tolist()
    ['<2', '2-10', '2-10', '10-25', '>=25', nan]
    """
    edges = [float(edge) for edge in edges]
    labels = list(labels)
    if not edges or any(later <= earlier for earlier, later in zip(edges, edges[1:])):
        raise ValueError(f"Carbonate class edges must be one or more values in increasing order; got {edges}")
    if len(labels) != len(edges) + 1:
        raise ValueError(f"Carbonate classes need one name per class - {len(edges) + 1}; got {len(labels)}")

    values = pd.to_numeric(pd.Series(caco3), errors="coerce")
    classes = pd.cut(values, [-np.inf, *edges, np.inf], right=False, labels=labels)
    return classes.astype(pd.CategoricalDtype(categories=labels, ordered=True))
