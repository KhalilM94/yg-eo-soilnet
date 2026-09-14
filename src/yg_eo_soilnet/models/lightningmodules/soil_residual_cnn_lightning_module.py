"""Legacy name: SoilCNNLightningModule with ``residual_enabled`` on.

The residual base used to live in this subclass. It is now a switch on SoilCNNLightningModule, and
this name stays for everything that still resolves it: checkpoints whose hyper_parameters predate the
switch and so do not carry it, models logged to MLflow as pickles of this class, tuned registry files
exported before the switch, and ``relog.py``. New configuration should use the ``soil_cnn`` entry
with ``residual_enabled: true``.
"""

from __future__ import annotations

from typing import Any

from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule


class SoilResidualCNNLightningModule(SoilCNNLightningModule):
    """SoilCNNLightningModule with the residual base switched on by default.

    See SoilCNNLightningModule for the residual arithmetic, the leakage caveat and the validation.
    """

    # What a pickled model of this class reads - see the note on SoilCNNLightningModule's switches.
    residual_enabled = True

    def __init__(self, *, residual_enabled: bool = True, **kwargs: Any):
        # Keyword-only, as this class always was: Lightning's save_hyperparameters drops *args, and a
        # positional static_dim would then be missing when load_from_checkpoint re-calls __init__.
        super().__init__(residual_enabled=residual_enabled, **kwargs)
