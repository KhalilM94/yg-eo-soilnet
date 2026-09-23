"""Train the models of each family and record the results."""

from yg_eo_soilnet.trainers.sklearn_trainer import ModelTrainer
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer

__all__ = ["ModelTrainer", "LightningTrainer"]
