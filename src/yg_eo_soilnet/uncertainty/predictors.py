"""The fitted :term:`ensemble`, presented as one model that behaves like a single one.

Its ``predict`` returns plain predictions, exactly as one member would, because everything
downstream - the saved model's signature, the scoring, the results table - reads it that way. The
uncertainty is asked for separately.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.ensemble import EnsemblePrediction, aggregate


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Several fitted models and their interval calibrators, behaving as one model.

    Already trained when it is built: the members are trained by the trainer, which is what lets each
    one have its own sub-run and its own seed. :meth:`fit` therefore does nothing.

    Parameters
    ----------
    members : sequence
        The fitted models.
    target_names : sequence of str
        The targets they predict.
    member_seeds : sequence of int, optional
        The seed each member was trained at.
    bootstrapped : bool, default False
        Whether the members were trained on resampled rows.
        """

    def __init__(
        self,
        members: Sequence[Any],
        target_names: Sequence[str],
        calibrators: Optional[Mapping[str, ConformalCalibrator]] = None,
        member_seeds: Optional[Sequence[int]] = None,
        bootstrapped: bool = False,
    ):
        """Hold the fitted members and record how they were made."""
        if not len(members):
            raise ValueError("An EnsembleRegressor needs at least one fitted member")
        self.members = list(members)
        self.target_names = [str(name) for name in target_names]
        self.calibrators = dict(calibrators or {})
        self.member_seeds = list(member_seeds or [])
        self.bootstrapped = bool(bootstrapped)

    # --- the single-model contract -----------------------------------------

    def fit(self, X, y=None):  # noqa: N803 - sklearn's argument name
        """Does nothing: the members arrive already trained. Present so this looks like any model."""
        return self

    def predict(self, X):  # noqa: N803 - sklearn's argument name
        """The ensemble's prediction - the members' average - shaped as one member's would be."""
        mean = self.predict_uncertainty(X).mean
        return mean.reshape(-1) if len(self.target_names) <= 1 else mean

    # --- the uncertainty API -----------------------------------------------

    def predict_members(self, X) -> list[np.ndarray]:  # noqa: N803
        """Each member's own prediction, in member order."""
        return [np.asarray(member.predict(X)) for member in self.members]

    def predict_uncertainty(self, X) -> EnsemblePrediction:  # noqa: N803
        """The average and how much the members disagree.

        No noise component: a scikit-learn model predicts a value, not a distribution, so the only
        uncertainty an ensemble of them can measure is their disagreement.

        Returns
        -------
        EnsemblePrediction
                """
        return aggregate(self.predict_members(X))

    def predict_frame(self, X) -> pd.DataFrame:  # noqa: N803
        """The wide form: prediction, spread and interval per target, as named columns.

        What serving returns when asked for uncertainty.
                """
        prediction = self.predict_uncertainty(X)
        frame = pd.DataFrame(index=getattr(X, "index", None))

        for index, target_name in enumerate(self.target_names):
            total_std = prediction.total_std[:, index]
            frame[f"{target_name}_pred"] = prediction.mean[:, index]
            frame[f"{target_name}_std"] = total_std
            frame[f"{target_name}_epistemic_std"] = prediction.epistemic_std[:, index]
            frame[f"{target_name}_aleatoric_std"] = prediction.aleatoric_std[:, index]

            calibrator = self.calibrators.get(target_name)
            if calibrator is not None:
                lower, upper = calibrator.intervals(prediction.mean[:, index], total_std)
                frame[f"{target_name}_lower"] = lower
                frame[f"{target_name}_upper"] = upper

        return frame

    # --- provenance ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """The ensemble's settings as plain values, to record with the run."""
        payload: dict[str, Any] = {
            "uncertainty_n_members": len(self.members),
            "uncertainty_bootstrapped": self.bootstrapped,
            "uncertainty_member_seeds": ",".join(str(seed) for seed in self.member_seeds),
        }
        for target_name, calibrator in self.calibrators.items():
            suffix = "" if len(self.target_names) <= 1 else f"_{target_name}"
            for key, value in calibrator.to_dict().items():
                payload[f"{key}{suffix}"] = value
        return payload
