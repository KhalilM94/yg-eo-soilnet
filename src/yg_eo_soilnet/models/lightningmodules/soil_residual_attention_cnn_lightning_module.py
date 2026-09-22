"""An older name for `soil_cnn` with a :term:`residual base` and attention :term:`fusion`.

Both were once classes of their own and are now switches. This name is kept so models saved before
the change can still be loaded; new configuration should use the ``soil_cnn`` entry with
``residual_enabled: true`` and ``fusion: attention``.
"""

from __future__ import annotations

from typing import Any

from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import STATIC_TOKEN_MODES
from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)

__all__ = ["STATIC_TOKEN_MODES", "SoilResidualAttentionCNNLightningModule"]


class SoilResidualAttentionCNNLightningModule(SoilResidualCNNLightningModule):
    """`soil_cnn` with a :term:`residual base` and attention :term:`fusion`."""

    # Read by a saved model of this class that predates the switch.
    fusion_type = "attention"

    def __init__(self, *, fusion: str = "attention", **kwargs: Any):
        super().__init__(fusion=fusion, **kwargs)
