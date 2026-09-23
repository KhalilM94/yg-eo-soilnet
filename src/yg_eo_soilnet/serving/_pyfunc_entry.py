"""The small file MLflow stores with a saved model and runs to rebuild it.

Stored as a file rather than as a live object, so loading the model does not depend on the exact
library versions the training machine happened to have.
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import mlflow.pytorch

from yg_eo_soilnet.serving.lightning_pyfunc import SoilSequencePyfunc

TORCH_MODEL_ARTIFACT = "torch_model"


class SoilSequenceEntry(SoilSequencePyfunc, mlflow.pyfunc.PythonModel):
    """Loads the saved network, then predicts through the raw-data wrapper."""

    def load_context(self, context) -> None:
        # The nested model carries its own class, so nothing here has to name the architecture.
        # It also restores `preprocessing_state` - the fitted scalers and the categorical
        # vocabulary - which is what lets the wrapper standardize raw input.
        """Load the network stored inside the saved model."""
        self.model = mlflow.pytorch.load_model(
            context.artifacts[TORCH_MODEL_ARTIFACT],
            map_location="cpu",
        )
        self.model.eval()
        self._predictor = None


mlflow.models.set_model(SoilSequenceEntry())
