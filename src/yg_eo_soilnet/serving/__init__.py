"""Use a saved deep-learning model to predict new points, outside a training run.

A saved scikit-learn model carries its own input preparation, so it can be handed raw data as it
is. A deep-learning model cannot: the standardization statistics, the fill values and the category
numbering are learned from the training points and live on the datamodule. They are written into the
:term:`checkpoint`, and this package is what puts them back - which is what makes a saved model
usable on data it has never seen.
"""

from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

__all__ = ["SoilSequencePredictor"]
