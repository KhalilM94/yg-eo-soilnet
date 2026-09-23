"""The three data-preparation steps of the scikit-learn family, in one place."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter import SklearnDataSplitter
from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.datamodules.splitting import SplitPlan


class ScikitDataModule:
    """Prepare the data for the scikit-learn models, from raw table to split covariates.

    :meth:`prepare` does the three steps in order: read the table, turn it into covariates and
    targets, then take this family's rows out of the run's shared split.

    Parameters
    ----------
    config : Config
        The run configuration.
    logger : logging.Logger
        Where progress messages go.
    data_manager : DataManager
        Reads the data files.
    split_plan_provider : SplitPlanProvider, optional
        The run's shared split. Left out, one is built here.

    Examples
    --------
    >>> datamodule = ScikitDataModule(config, logger, data_manager)   # doctest: +SKIP
    >>> data = datamodule.prepare()                                   # doctest: +SKIP
    >>> data["X_train"].shape, data["X_test"].shape                   # doctest: +SKIP
    ((255, 12), (45, 12))
    """

    def __init__(self, config, logger, data_manager, split_plan_provider=None):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.preprocessor = TabularPreprocessor(config, logger, data_manager)
        self.splitter = SklearnDataSplitter(config, logger)
        # Passed in by the run, so both families read one plan rather than two that happen to
        # agree.
        self.split_plan_provider = split_plan_provider or SplitPlanProvider(config, logger, data_manager)

    def load_frame(self) -> pd.DataFrame:
        """Read the data: one row per point, covariates and targets together."""
        return self.data_manager.load_dataset().tabular

    def preprocess(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Split the table into covariates, targets, coordinates and point ids."""
        return self.preprocessor.preprocess_data(data)

    def split(self, processed_data: Dict[str, Any], split_plan: SplitPlan | None = None) -> Dict[str, Any]:
        """Take this family's training, validation and test rows out of the shared split.

        Parameters
        ----------
        processed_data : dict
            What :meth:`preprocess` returned.
        split_plan : SplitPlan, optional
            The split to follow; the run's shared one unless given.

        Returns
        -------
        dict
            The covariates and targets per split, plus what the models need to rebuild them.
        """
        from yg_eo_soilnet.models import ModelConfigFactory  # local import avoids a circular dependency

        return self.splitter.split_data(
            processed_data,
            sanitize_features=self.data_manager.filter_schema,
            model_config_factory=ModelConfigFactory,
            split_plan=split_plan if split_plan is not None else self.split_plan(),
        )

    def split_plan(self) -> SplitPlan:
        """The run's shared split, built on first use."""
        return self.split_plan_provider.plan()

    def prepare(self, data: pd.DataFrame | None = None) -> Dict[str, Any]:
        """Read, prepare and split the data in one call.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Use this table instead of reading the data files.

        Returns
        -------
        dict
            The split covariates and targets, ready for training.
        """
        frame = self.load_frame() if data is None else data
        return self.split(self.preprocess(frame))
