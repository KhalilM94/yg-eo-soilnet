"""Legacy name: SoilCNNLightningModule with ``residual_enabled`` on and ``fusion="attention"``.

The attention fusion used to live in this subclass, which built its parent's gated fusion and head
and then replaced both. It is now a switch on SoilCNNLightningModule, and this name stays for
everything that still resolves it - see soil_residual_cnn_lightning_module for the list. New
configuration should use the ``soil_cnn`` entry with ``residual_enabled: true`` and
``fusion: attention``.
"""

from __future__ import annotations

from typing import Any

from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import STATIC_TOKEN_MODES
from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)

__all__ = ["STATIC_TOKEN_MODES", "SoilResidualAttentionCNNLightningModule"]


class SoilResidualAttentionCNNLightningModule(SoilResidualCNNLightningModule):
    """SoilResidualCNNLightningModule with attention fusion switched on by default."""

    # What a pickled model of this class reads - see the note on SoilCNNLightningModule's switches.
    fusion_type = "attention"

    def __init__(self, *, fusion: str = "attention", **kwargs: Any):
        super().__init__(fusion=fusion, **kwargs)
