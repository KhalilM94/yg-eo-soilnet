"""Build the scikit-learn estimators named in the model list."""

from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfigFactory:
    """Build the scikit-learn estimators a model list asks for.

    Each entry names a class by its import path, so a new model is added by editing YAML rather than
    code. The class is imported, built with the entry's ``init_args``, and given the run's random
    seed unless the entry sets its own.

    Attributes
    ----------
    registry : dict
        The model list: model name to its settings. This class is also used with a single
        clustering-strategy entry, which :meth:`load_splitter_from_config` reads.
    random_state : int, default 42
        The run's random seed.

    Examples
    --------
    >>> registry = {"Ridge": {"enabled": True, "import_path": "sklearn.linear_model.Ridge",
    ...                       "params": {"alpha": [0.1, 1.0]}}}
    >>> configs = ModelConfigFactory(registry).build_model_configs(num_features=12)
    >>> type(configs["Ridge"]["model"]).__name__, configs["Ridge"]["params"]
    ('Ridge', {'alpha': [0.1, 1.0]})
    """

    registry: dict
    random_state: int = 42

    @staticmethod
    def _dynamic_import(import_path):
        """Import a class from its full path, such as ``sklearn.linear_model.Ridge``."""
        module_path, class_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_path}': {e}")
        return getattr(module, class_name)

    @staticmethod
    def _seed_estimator(model, seed: int) -> None:
        """Give the run's seed to any estimator that takes one.

        So that changing ``RANDOM_SEED`` changes the whole run. An entry names a seed only to depart
        from it.
        """
        try:
            exposes_seed = "random_state" in model.get_params(deep=False)
        except (AttributeError, TypeError):
            return  # not a sklearn-style estimator, nothing to seed
        if exposes_seed:
            model.set_params(random_state=int(seed))

    def build_model_configs(self, num_features, default_seed: int | None = None):
        """Build every model switched on in the list.

        Parameters
        ----------
        num_features : int
            How many covariates the models will see; passed to entries that need it.
        default_seed : int, optional
            The run's seed, used unless an entry sets its own.

        Returns
        -------
        dict
            Model name to ``{"model", "params", "modeltype", "random_seed", "search_n_jobs"}``,
            where ``params`` is the grid of hyperparameters to search.
        """
        model_configs = {}

        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue
            try:
                ModelClass = self._dynamic_import(spec["import_path"])
            except Exception as e:
                print(f"[Warning] Failed to import {name}: {e}")
                continue

            # Copied: this runs once per target, and the settings below would otherwise be
            # written back into the shared model list.
            init_args = dict(spec.get("init_args", {}))
            custom_model_builder = spec.get("custom_model_builder", None)
            model_seed = spec.get("random_seed", default_seed)

            # Models built by a function of their own rather than by their constructor.
            if custom_model_builder:
                builder_func = self._dynamic_import(custom_model_builder)
                model_instance = ModelClass(build_fn=lambda: builder_func(num_features))
            else:
                if "input_dim" in init_args:
                    init_args["input_dim"] = num_features
                model_instance = ModelClass(**init_args)
                # A seed in the entry wins; otherwise the estimator takes the run's seed.
                if "random_state" not in init_args and model_seed is not None:
                    self._seed_estimator(model_instance, model_seed)

            model_configs[name] = {
                "model": model_instance,
                "params": spec.get("params", {}),
                "modeltype": spec.get("modeltype", "ml"),
                "random_seed": model_seed,
                # How many searches run at once. -1 uses every processor, which suits cheap
                # models; one that loads a large file per process (TabICL) has to limit it.
                "search_n_jobs": spec.get("search_n_jobs", -1),
            }

        return model_configs

    def load_splitter_from_config(self) -> Optional[BaseSpatialClusterStrategy]:
        """Build the spatial clustering strategy named by ``split.group.class_path``.

        Returns
        -------
        BaseSpatialClusterStrategy or None
            None when the entry says ``enabled: false``.
        """
        if self.registry.get("enabled", True) is True:
            class_path = self.registry["class_path"]
            params = self.registry.get("params", {})
            params.setdefault("random_state", self.random_state)

            SplitterClass = self._dynamic_import(class_path)
            return SplitterClass(**params)
        else:
            print("[Warning] Splitter configuration not found or disabled in the registry.")
            return None
