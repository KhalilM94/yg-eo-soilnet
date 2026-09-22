Building the models
===================

The scikit-learn estimators, and the deep-learning model with every part it is built from.

From a model list to a model
----------------------------

.. automodule:: yg_eo_soilnet.models.config_fatories.model_config_factory

.. automodule:: yg_eo_soilnet.models.config_fatories.lightning_config_factory

The deep-learning model
-----------------------

.. automodule:: yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module

.. automodule:: yg_eo_soilnet.models.lightningmodules._regression_base

Its parts
---------

.. automodule:: yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders

.. automodule:: yg_eo_soilnet.models.lightningmodules.tabular_encoders

.. automodule:: yg_eo_soilnet.models.lightningmodules.spatial_encoders

.. automodule:: yg_eo_soilnet.models.lightningmodules.mlp

What it is trained to minimize
------------------------------

.. automodule:: yg_eo_soilnet.models.lightningmodules.losses

Older names
-----------

Kept so that models saved before these became switches can still be loaded.

.. automodule:: yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module

.. automodule:: yg_eo_soilnet.models.lightningmodules.soil_residual_attention_cnn_lightning_module
