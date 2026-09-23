"""What an explanation looks like once either family has finished with it.

Both explainers return a list of these, one per target, so the figures, the table and the summary
are written once rather than once per family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Which numeric space the SHAP values are contributions TO. This is not decoration: for a
# log-transformed target the sklearn pipeline wraps the estimator in a TransformedTargetRegressor
# and the explained estimator predicts 10*log1p(y), so a value of 0.4 means 0.4 in that space and
# not 0.4 percentage points of organic matter.
ORIGINAL_UNITS = "original_units"
LOG1P_X10 = "log1p_x10"
STANDARDIZED_LOG1P = "standardized_log1p"


@dataclass
class ShapResult:
    """One target's :term:`SHAP` contributions: one number per input per point.

    Attributes
    ----------
    target : str
        Which target this explains.
    values : numpy.ndarray of shape (n_points, n_features)
        Each input's contribution to each point's prediction. They add up to that prediction minus the
        average prediction.
    feature_names : list of str
        The inputs, in the order of the columns.
    blocks : list of str
        Which kind of input each column is - covariate, category, time series, and so on - so the
        contributions can be reported by group.
    base_value : float
        The average prediction the contributions are measured from.
    explainer : str
        Which method produced them.
    feature_values : numpy.ndarray, optional
        The inputs' own values, which colour the figures.
    """

    values: np.ndarray  # (n_samples, n_features)
    data: np.ndarray  # (n_samples, n_features), the beeswarm colour values; NaN where meaningless
    feature_names: list[str]
    target_name: str
    output_space: str
    # Which block each feature belongs to: "static", "categorical", "auxiliary", or a modality name.
    # Drives the rolled-up bar plot.
    blocks: list[str] = field(default_factory=list)
    base_value: float = 0.0
    # Which shap explainer produced these values. Recorded because the sklearn path silently falls
    # back from the exact TreeExplainer to a model-agnostic one when shap cannot parse the model -
    # an approximation the reader of a plot deserves to know about.
    explainer: str = "unknown"

    def __post_init__(self) -> None:
        """Check the names, blocks and values describe the same inputs."""
        self.values = np.asarray(self.values, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.values.shape != self.data.shape:
            raise ValueError(f"values {self.values.shape} and data {self.data.shape} must have the same shape")
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError(f"{self.values.shape[1]} value column(s) but {len(self.feature_names)} feature name(s)")
        if not self.blocks:
            self.blocks = ["all"] * len(self.feature_names)
        if len(self.blocks) != len(self.feature_names):
            raise ValueError(f"{len(self.blocks)} block label(s) but {len(self.feature_names)} feature name(s)")

    @property
    def n_samples(self) -> int:
        """How many points were explained."""
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        """How many inputs each point was explained by."""
        return int(self.values.shape[1])

    def mean_abs(self) -> np.ndarray:
        """How much each input matters on average - the height of each bar."""
        return np.abs(self.values).mean(axis=0)

    def ranking(self) -> list[int]:
        """The inputs ordered by how much they matter, most first."""
        return list(np.argsort(-self.mean_abs()))

    def block_mean_abs(self) -> dict[str, float]:
        """How much each *group* of inputs matters on average.

        The contributions are added up within a group for each point before their size is taken, so two
        inputs in one group that cancel each other out count as the small contribution they jointly make,
        not as two large ones.
        """
        totals: dict[str, float] = {}
        for block in dict.fromkeys(self.blocks):
            columns = [index for index, name in enumerate(self.blocks) if name == block]
            totals[block] = float(np.abs(self.values[:, columns].sum(axis=1)).mean())
        return totals

    def to_frame(self) -> pd.DataFrame:
        """The complete table of contributions, one row per point per input.

        Long rather than wide: with every band of every data source having its own row, there are hundreds
        of inputs.
        """
        n_samples, n_features = self.values.shape
        return pd.DataFrame(
            {
                "sample": np.repeat(np.arange(n_samples), n_features),
                "feature": np.tile(np.asarray(self.feature_names, dtype=object), n_samples),
                "block": np.tile(np.asarray(self.blocks, dtype=object), n_samples),
                "shap_value": self.values.reshape(-1),
                "feature_value": self.data.reshape(-1),
                "target": self.target_name,
                "output_space": self.output_space,
            }
        )

    def summary(self, top_n: int | None = None) -> dict:
        """The ranking and where it came from, for the run summary."""
        mean_abs = self.mean_abs()
        order = self.ranking()
        if top_n is not None:
            order = order[:top_n]
        return {
            "target": self.target_name,
            "output_space": self.output_space,
            "explainer": self.explainer,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "base_value": float(self.base_value),
            "blocks": self.block_mean_abs(),
            "ranking": [
                {
                    "feature": self.feature_names[index],
                    "block": self.blocks[index],
                    "mean_abs_shap": float(mean_abs[index]),
                }
                for index in order
            ],
        }
