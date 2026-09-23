"""Predict with a saved deep-learning model, using the statistics it was trained with."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule


def _as_array(tensor) -> np.ndarray:
    """Return a tensor or array as a plain NumPy array."""
    return np.asarray(tensor.detach().cpu(), dtype=np.float64)


class SoilSequencePredictor:
    """A trained model plus its training-time input preparation, ready to score new points.

    Parameters
    ----------
    model : SoilCNNLightningModule
        A model loaded from a :term:`checkpoint` or from MLflow.

    Examples
    --------
    >>> import mlflow                                                        # doctest: +SKIP
    >>> model = mlflow.pytorch.load_model(model_uri)                         # doctest: +SKIP
    >>> predictions = SoilSequencePredictor(model).predict(bundle)           # doctest: +SKIP
    >>> predictions.shape                                                    # doctest: +SKIP
    (300, 3)
        """

    def __init__(self, model, preprocessing_state: Mapping[str, Any] | None = None):
        """Hold the model and read the input statistics out of its checkpoint."""
        self.model = model
        state = preprocessing_state
        if state is None and hasattr(model, "get_preprocessing_state"):
            state = model.get_preprocessing_state()
        if not state:
            raise ValueError(
                "This checkpoint carries no preprocessing state, so it cannot standardize raw input. "
                "It predates attach_preprocessing_state; retrain, or pass preprocessing_state "
                "explicitly from the datamodule that trained it."
            )
        self.preprocessing_state = dict(state)

    def _datamodule(self, bundle: SoilSequenceBundle, **datamodule_kwargs) -> SoilSequenceDataModule:
        """A datamodule over these points, set up with the model's own statistics."""
        datamodule = SoilSequenceDataModule(sequence_bundle=bundle, **datamodule_kwargs)
        datamodule.apply_preprocessing_state(self.preprocessing_state)
        return datamodule

    @property
    def predicts_variance(self) -> bool:
        """Whether this model also predicts a spread per point; see :term:`variance head`.

        Read from the saved weights rather than from the settings, because the weights are what a restored
        model always carries.
                """
        return bool(getattr(self.model, "predict_variance", False)) or bool(
            getattr(self.model, "head_predicts_variance", False)
        )

    @torch.no_grad()
    def predict(
        self,
        bundle: "SoilSequenceBundle | Mapping[str, Any]",
        *,
        batch_size: int = 64,
        **datamodule_kwargs,
    ) -> np.ndarray:
        """Predict for every point, in the target's own units.

        Parameters
        ----------
        bundle : SoilSequenceBundle
            The points to predict for.

        Returns
        -------
        numpy.ndarray of shape (n_points, n_targets)
                """
        return self.predict_with_uncertainty(bundle, batch_size=batch_size, **datamodule_kwargs)[0]

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        bundle: "SoilSequenceBundle | Mapping[str, Any]",
        *,
        batch_size: int = 64,
        **datamodule_kwargs,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        """Predict, with the spread beside each value where the model predicts one.

        Returns
        -------
        predictions : numpy.ndarray
        sigma : numpy.ndarray or None
            None unless the model has a :term:`variance head`. Both in the target's own units.
                """
        bundle = SoilSequenceBundle.from_mapping(bundle)
        if bundle.num_points == 0:
            empty = np.empty((0, int(getattr(self.model, "target_dim", 1))), dtype=np.float64)
            return empty, (empty.copy() if self.predicts_variance else None)

        datamodule = self._datamodule(bundle, **datamodule_kwargs)

        was_training = self.model.training
        self.model.eval()
        try:
            outputs, sigmas = [], []
            for start in range(0, bundle.num_points, max(1, int(batch_size))):
                indices = np.arange(start, min(start + batch_size, bundle.num_points))
                batch = datamodule.collate(indices)
                # predict_step, not forward: forward stops in standardized log1p space and only
                # predict_step inverts it. Calling forward here would return predictions that look
                # plausible and are in the wrong units.
                step_output = self.model.predict_step(batch, 0)
                if isinstance(step_output, tuple):
                    outputs.append(_as_array(step_output[0]))
                    sigmas.append(_as_array(step_output[1]))
                else:
                    outputs.append(_as_array(step_output))
            predictions = np.concatenate(outputs, axis=0)
            sigma = np.concatenate(sigmas, axis=0) if len(sigmas) == len(outputs) and sigmas else None
        finally:
            if was_training:
                self.model.train()

        predictions = predictions.reshape(bundle.num_points, -1)
        if sigma is not None:
            sigma = sigma.reshape(bundle.num_points, -1)
        return predictions, sigma

    def predict_frame(self, bundle, **kwargs):
        """Like :meth:`predict`, as a table: one column per target, one row per point."""
        import pandas as pd

        bundle = SoilSequenceBundle.from_mapping(bundle)
        predictions = self.predict(bundle, **kwargs)
        names = list(self.preprocessing_state.get("target_names") or []) or [
            f"target_{index}" for index in range(predictions.shape[1])
        ]
        names = names[: predictions.shape[1]]
        return pd.DataFrame(predictions, columns=names, index=list(bundle.point_ids))
