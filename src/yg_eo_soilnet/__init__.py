"""Predict lab-measured soil properties from satellite, climate and terrain data.

The pieces of a run, in the order it uses them:

:mod:`~yg_eo_soilnet.data_manager`
    Reads the three data files and decides which columns models may use.
:mod:`~yg_eo_soilnet.datamodules`
    Makes the one split every model shares, and prepares the inputs for each family.
:mod:`~yg_eo_soilnet.models`
    Builds the models the configuration asks for.
:mod:`~yg_eo_soilnet.trainers`
    Trains them.
:mod:`~yg_eo_soilnet.metrics` and :mod:`~yg_eo_soilnet.logger`
    Scores them and records everything in MLflow.

And the optional extras: :mod:`~yg_eo_soilnet.uncertainty`, :mod:`~yg_eo_soilnet.explain`,
:mod:`~yg_eo_soilnet.hpo` for tuning, and :mod:`~yg_eo_soilnet.serving` for predicting new points
with a saved model.
"""
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.dataset import SoilDataset
from yg_eo_soilnet.logger import ChildRunLogger, ParentRunLogger, TrainingLogger
from yg_eo_soilnet.models import ModelConfigFactory
from yg_eo_soilnet.clustering_utils import (
	BaseSpatialClusterStrategy,
	KMeansClusterStrategy,
	SpatialGridClusterStrategy,
)
from yg_eo_soilnet.datamodules.scikit import (
	CVSplitter,
	PipelineBuilder,
	ScikitDataModule,
	SklearnDataSplitter,
	TabularPreprocessor,
	TargetNanFilter,
)
from yg_eo_soilnet.trainers import ModelTrainer
from yg_eo_soilnet.utils import LogTransformer
from yg_eo_soilnet.artifacts import ArtifactLayout
from yg_eo_soilnet.metrics import cv_rmse_from_search, regression_metrics
from yg_eo_soilnet.serving import SoilSequencePredictor

# Deliberately NOT re-exported: yg_eo_soilnet.explain. Importing it here would defeat the
# EXPLAIN_ENABLED off-switch, which depends on nothing reaching the explain package until the
# switch has been checked.

__all__ = [
	"DataManager",
	"SoilDataset",
	"LogTransformer",
	"ArtifactLayout",
	"regression_metrics",
	"cv_rmse_from_search",
	"SoilSequencePredictor",
	"CVSplitter",
	"PipelineBuilder",
	"TargetNanFilter",
	"ScikitDataModule",
	"SklearnDataSplitter",
	"TabularPreprocessor",
	"ModelConfigFactory",
	"BaseSpatialClusterStrategy",
	"KMeansClusterStrategy",
	"SpatialGridClusterStrategy",
	"TrainingLogger",
	"ChildRunLogger",
	"ParentRunLogger",
	"ModelTrainer",
]
