import importlib

import pytest

# These imports are the guard: if a module moved, collection of this file fails. Asserting
# `X is not None` on them afterwards could never fail independently, so that test was removed.
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule  # noqa: F401
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import (  # noqa: F401
    CVSplitter,
    PipelineBuilder,
    TargetNanFilter,
)
from yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter import SklearnDataSplitter  # noqa: F401
from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor  # noqa: F401
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule  # noqa: F401


RETIRED_MODULE_PATHS = [
    "yg_eo_soilnet.datamodules.lightning_graph",
    "yg_eo_soilnet.datamodules.lightning_tabular",
    "yg_eo_soilnet.datamodules.lightning.lightning_tabular",
    "yg_eo_soilnet.pipelines.spatiotemporal_graph_builder",
    "yg_eo_soilnet.pipelines.tabular_preprocessor",
    "yg_eo_soilnet.trainer_utils",
    # The graph, sequence and tabular families, removed in favour of the one switchable soil_cnn.
    "yg_eo_soilnet.datamodules.lightning",
    "yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module",
    "yg_eo_soilnet.models.lightningmodules.soil_sequence_lightning_module",
    "yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module",
    "yg_eo_soilnet.models.lightningmodules.temporal_encoders",
]


@pytest.mark.parametrize("module_path", RETIRED_MODULE_PATHS)
def test_retired_module_paths_no_longer_import(module_path: str) -> None:
    """Guards against the duplicate copies silently reappearing."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)
