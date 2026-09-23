"""An older name for `soil_cnn` with its :term:`residual base` switched on.

The residual base was once a class of its own; it is now a switch. This name is kept because saved
models, checkpoints and tuned configuration files written before the change still name it, and
loading one has to find the class it was saved as. New configuration should use the ``soil_cnn``
entry with ``residual_enabled: true``.
"""

from __future__ import annotations

from typing import Any

from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule


class SoilResidualCNNLightningModule(SoilCNNLightningModule):
    """`soil_cnn` with the :term:`residual base` switched on.

    See :class:`~yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module.SoilCNNLightningModule`
    for what the residual base does and when it is honest to use.
    """

    # Read by a saved model of this class that predates the switch.
    residual_enabled = True

    def __init__(self, *, residual_enabled: bool = True, **kwargs: Any):
        # Keyword-only, as this class always was: a saved model records its settings by name, so a
        # positional one would be missing when the checkpoint is loaded.
        super().__init__(residual_enabled=residual_enabled, **kwargs)
