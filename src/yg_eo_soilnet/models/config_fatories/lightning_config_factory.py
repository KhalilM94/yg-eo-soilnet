"""Build a deep-learning model and its data from one model-list entry."""

from __future__ import annotations

import importlib
import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from yg_eo_soilnet.seeding import seed_everything
from yg_eo_soilnet.targets import split_target_names


@dataclass
class LightningModelBundle:
    """One deep-learning model with everything needed to train it - a model :term:`bundle`.

    Attributes
    ----------
    name : str
        The model's name in the model list.
    target : str
        The :term:`target group` it predicts.
    model : lightning.pytorch.LightningModule
        The built model.
    datamodule : lightning.pytorch.LightningDataModule
        Its data, already prepared.
    trainer_kwargs : dict
        Training settings: epochs, processor, precision.
    callback_specs : dict
        Early-stopping and checkpoint settings.
    registry_entry : dict
        The model-list entry it was built from, copied.
    """

    name: str
    target: str
    model: Any
    datamodule: Any
    trainer_kwargs: dict[str, Any]
    callback_specs: dict[str, Any]
    registry_entry: dict[str, Any]


class LightningConfigFactory:
    """Build the deep-learning models a model list asks for, ready to train.

    For each entry switched on it prepares the data, seeds the random generators, builds the model,
    and fills in everything the model needs but the configuration cannot know: the number of
    covariates, the data sources and their widths, the category numbering, the target statistics.
    Those come from the datamodule, which has seen the training points.

    Parameters
    ----------
    registry : mapping
        The deep-learning model list: model name to its settings.
    config : Config
        The run configuration.
    logger : logging.Logger, optional
        Where progress messages go.
    data_manager : DataManager, optional
        Used to prepare the data when the caller supplies none.
    datamodule_cache : mutable mapping, optional
        Lets several models share one prepared datamodule. Used by the hyperparameter search, where
        preparing the data again for every trial would dominate the cost.

    Examples
    --------
    >>> factory = LightningConfigFactory(config.LIGHTNING_MODEL_REGISTRY, config,   # doctest: +SKIP
    ...                                  logger, data_manager)
    >>> bundles = factory.build_lightning_configs("clay_pct", data)                 # doctest: +SKIP
    >>> sorted(bundles)                                                             # doctest: +SKIP
    ['soil_cnn']
    """

    def __init__(
        self,
        registry: Mapping[str, dict],
        config: Any,
        logger: Any = None,
        data_manager: Any = None,
        datamodule_cache: MutableMapping[Any, Any] | None = None,
    ):
        self.registry = registry
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        # Reuse of prepared datamodules, keyed by what built them. Off unless asked for. Safe: a
        # datamodule holds no model state, and preparing it twice gives the same result.
        self.datamodule_cache = datamodule_cache
        # Built only if the caller passes no split of its own.
        self._split_plan_provider = None

    @staticmethod
    def _dynamic_import(import_path: str):
        """Import a class from its full path, such as the one an entry's ``import_path`` names."""
        module_path, attr_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc
        return getattr(module, attr_name)

    def build_lightning_configs(
        self,
        target: str,
        data: Mapping[str, Any],
        seed: int | None = None,
        entries: Any = None,
    ) -> dict[str, LightningModelBundle]:
        """Build one :class:`LightningModelBundle` per model switched on.

        Parameters
        ----------
        target : str
            The :term:`target group` these models predict.
        data : mapping
            The prepared data and the run's shared split.
        seed : int, optional
            Applied just before each model is built, which is when its starting weights are drawn -
            the only moment at which seeding reaches them. Per model, so a second model does not
            inherit whatever the first consumed.
        entries : iterable of str, optional
            Build only these models.

        Returns
        -------
        dict of str to LightningModelBundle

        Raises
        ------
        ValueError
            If an entry is missing a required setting or is not a deep-learning entry.
        """
        wanted = None if entries is None else {str(name) for name in entries}
        bundles: dict[str, LightningModelBundle] = {}
        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue
            if wanted is not None and name not in wanted:
                continue

            self._validate_entry(name, spec)
            datamodule = self._build_datamodule(target=target, spec=spec, data=data)
            if seed is not None:
                seed_everything(int(spec.get("random_seed", seed)))
            model = self._build_model(spec, datamodule)
            bundles[name] = LightningModelBundle(
                name=name,
                target=target,
                model=model,
                datamodule=datamodule,
                trainer_kwargs=self._build_trainer_kwargs(spec),
                callback_specs=self._build_callback_specs(spec),
                registry_entry=deepcopy(spec),
            )

        return bundles

    @staticmethod
    def _input_kind(spec: Mapping[str, Any]) -> str | None:
        """What kind of data an entry needs. Only ``"sequence"`` can be built."""
        return spec.get("input_kind")

    def sequence_spec(self) -> dict[str, Any] | None:
        """The first model switched on that needs the time series, or None."""
        for spec in self.registry.values():
            if spec.get("enabled", False) and self._input_kind(spec) == "sequence":
                return spec
        return None

    def has_sequence_input(self) -> bool:
        """Whether any model switched on needs the time series."""
        return self.sequence_spec() is not None

    def _validate_entry(self, name: str, spec: Mapping[str, Any]) -> None:
        """Check an entry has the settings every deep-learning model needs."""
        required_keys = {"enabled", "modeltype", "import_path", "datamodule_import_path"}
        missing = sorted(required_keys - set(spec))
        if missing:
            raise ValueError(f"Lightning registry entry '{name}' is missing required keys: {', '.join(missing)}")
        if spec.get("modeltype") != "dl":
            raise ValueError(f"Lightning registry entry '{name}' must use modeltype 'dl'.")

    def _build_datamodule(self, target: str, spec: Mapping[str, Any], data: Mapping[str, Any]) -> Any:
        """Prepare this model's data, reusing an already prepared one where that is allowed."""
        input_kind = self._input_kind(spec)
        if input_kind != "sequence":
            raise ValueError(
                f"Unsupported input_kind {input_kind!r} in the Lightning registry; expected 'sequence'"
            )

        datamodule_cls = self._dynamic_import(spec["datamodule_import_path"])
        fallback_seed = int(
            spec.get(
                "random_seed",
                getattr(self.config, "RANDOM_SEED", 42),
            )
        )

        datamodule_kwargs = deepcopy(spec.get("datamodule_init_args", {}))
        datamodule_kwargs.setdefault("batch_size", getattr(self.config, "LIGHTNING_BATCH_SIZE", 32))
        # Only used if no shared split reaches the datamodule below.
        datamodule_kwargs.setdefault("val_size", getattr(self.config, "SPLIT_VAL_SIZE", 0.2))
        datamodule_kwargs.setdefault("test_size", getattr(self.config, "SPLIT_TEST_SIZE", 0.2))
        datamodule_kwargs.setdefault("num_workers", getattr(self.config, "LIGHTNING_NUM_WORKERS", 0))
        datamodule_kwargs.setdefault("pin_memory", getattr(self.config, "LIGHTNING_PIN_MEMORY", False))
        datamodule_kwargs.setdefault(
            "persistent_workers", getattr(self.config, "LIGHTNING_PERSISTENT_WORKERS", False)
        )
        datamodule_kwargs.setdefault("seed", fallback_seed)

        # The targets this model predicts, read back from the group name. The data is prepared once
        # for every target, so a single-target model narrows it rather than preparing it again.
        active_targets = split_target_names(target)
        if active_targets and self._accepts_kwarg(datamodule_cls, "active_targets"):
            datamodule_kwargs["active_targets"] = active_targets

        # Built before the data itself is attached, which is identified by object rather than by
        # value. The targets are part of the key, so two target groups cannot share a slot.
        cache_key = (spec["datamodule_import_path"], repr(sorted(datamodule_kwargs.items())))

        payload = data.get("sequence_bundle")
        if payload is None:
            payload = self._build_sequence_bundle(spec)
        datamodule_kwargs["sequence_bundle"] = payload

        split_plan = self._resolve_split_plan(data)
        if split_plan is not None and self._accepts_kwarg(datamodule_cls, "split_plan"):
            datamodule_kwargs["split_plan"] = split_plan

        if self.datamodule_cache is not None:
            # The split is part of the key too: the same model under a different split is a
            # different datamodule.
            cache_key = (*cache_key, id(payload), id(split_plan))
            cached = self.datamodule_cache.get(cache_key)
            if cached is not None:
                return cached

        datamodule = datamodule_cls(**datamodule_kwargs)
        datamodule.setup("fit")
        if self.datamodule_cache is not None:
            self.datamodule_cache[cache_key] = datamodule
        return datamodule

    def _build_sequence_bundle(self, spec: Mapping[str, Any]) -> Any:
        """Prepare the data from scratch, when the caller supplied none.

        Raises
        ------
        KeyError
            If there is neither prepared data nor a data manager to prepare it with.
        """
        if self.data_manager is None:
            raise KeyError(
                "Sequence lightning registry entries require either a 'sequence_bundle' payload "
                "or a data_manager on LightningConfigFactory"
            )
        # Imported here, so building a factory does not pay for it.
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        builder = SoilSequenceBuilder(self.config, self.logger, self.data_manager)
        return builder.build(sequence_data_args=dict(spec.get("sequence_data_args", {}) or {}))

    def _resolve_split_plan(self, data: Mapping[str, Any]):
        """The run's shared split, from the data handed in or, failing that, built here.

        Without it this family would split for itself, and its test points would overlap the ones
        the scikit-learn models trained on.
        """
        plan = data.get("split_plan")
        if plan is not None:
            return plan
        if self.data_manager is None:
            return None
        from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider

        if getattr(self, "_split_plan_provider", None) is None:
            self._split_plan_provider = SplitPlanProvider(self.config, self.logger, self.data_manager)
        return self._split_plan_provider.plan()

    @classmethod
    def _accepts_kwarg(cls, target_cls: Any, name: str) -> bool:
        """Whether a class's constructor takes a setting of this name."""
        accepted = cls._accepted_init_args(target_cls)
        return accepted is None or name in accepted

    @staticmethod
    def _accepted_init_args(model_cls: Any) -> set[str] | None:
        """The settings a class's constructor accepts, or None when it accepts anything."""
        try:
            parameters = inspect.signature(model_cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins and C extensions
            return None
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return None
        return {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def _build_model(self, spec: Mapping[str, Any], datamodule: Any):
        """Build the model, filling in the shapes and statistics from the prepared data."""
        model_cls = self._dynamic_import(spec["import_path"])
        init_args = deepcopy(spec.get("init_args", {}))
        # One datamodule can serve several models, so what it offers is not what every model
        # wants: anything offered here is filtered to what the model actually accepts. Whatever the
        # model list names explicitly is passed through, so a typo there fails loudly.
        accepted = self._accepted_init_args(model_cls)

        def offer(key: str, value: Any) -> None:
            """Pass a setting to the model, but only if its constructor takes one by that name."""
            if accepted is None or key in accepted:
                init_args[key] = value

        shape_args = {
            "static_dim": getattr(datamodule, "static_dim", None),
            "target_dim": getattr(datamodule, "target_dim", None),
            "modality_dims": getattr(datamodule, "modality_dims", None),
            # How many years the calendar grid spans, read from the data.
            "grid_years": getattr(datamodule, "grid_years", None),
            # 2 when the data carries coordinates, 0 otherwise.
            "coord_dim": getattr(datamodule, "coord_dim", None),
            # The category numbering, learned from the training points. It travels into the model
            # so a saved model carries its own numbering.
            "categorical_cardinalities": getattr(datamodule, "categorical_cardinalities", None),
            "categorical_vocabularies": getattr(datamodule, "categorical_vocabularies", None),
            "categorical_feature_names": getattr(datamodule, "categorical_feature_names", None),
            # The lab columns available as auxiliary inputs, and the targets they must not be.
            "auxiliary_available_names": getattr(datamodule, "label_feature_names", None),
            "target_names": getattr(datamodule, "target_names", None),
            # Every target the run fits, not only this model's: with one model per target,
            # checking against the narrower list would let another target in as an input.
            "fitted_target_names": list(getattr(self.config, "TARGET_COLUMNS", []) or []),
        }
        # An empty list means "this dataset has none", which is not worth offering as a shape.
        unset = (None, 0, "auto", {}, [])
        for key, value in shape_args.items():
            if key in init_args and init_args[key] in (None, 0, "auto", {}):
                init_args[key] = value
            elif key not in init_args and value not in unset:
                offer(key, value)

        if "temporal_enabled" in init_args and init_args["temporal_enabled"] in (None, "auto"):
            init_args["temporal_enabled"] = getattr(datamodule, "temporal_enabled", False)
        elif "temporal_enabled" not in init_args:
            offer("temporal_enabled", getattr(datamodule, "temporal_enabled", False))
        if "output_dim" in init_args and init_args["output_dim"] in (None, 0, "auto"):
            init_args["output_dim"] = getattr(datamodule, "target_dim", 1)
        # The target statistics, so the model can convert its predictions back. Plain numbers,
        # because they are saved in the checkpoint.
        for key, attribute in (("target_mean", "target_mean_"), ("target_scale", "target_scale_")):
            value = getattr(datamodule, attribute, None)
            if value is not None and init_args.get(key) in (None, "auto"):
                offer(key, [float(item) for item in np.asarray(value).ravel()])
        # The lab-value statistics, which only a model with a residual base needs: they convert
        # the base back into the target's own units.
        for key, attribute in (
            ("auxiliary_label_mean", "label_mean_"),
            ("auxiliary_label_scale", "label_scale_"),
        ):
            value = getattr(datamodule, attribute, None)
            if value is not None and init_args.get(key) in (None, "auto"):
                offer(key, [float(item) for item in np.asarray(value).ravel()])
            elif init_args.get(key) == "auto":
                # No lab columns carried, so there is nothing to fill in.
                init_args[key] = None
        # The datamodule decides the target transform; the model is told so it can undo it.
        if init_args.get("target_transform") in (None, "auto"):
            offer("target_transform", getattr(datamodule, "target_transform", None))
        # How the training targets vary together, for the losses that read across targets. Only
        # the datamodule has seen every training point.
        covariance = getattr(datamodule, "target_covariance_", None)
        if covariance is not None and init_args.get("target_covariance") in (None, "auto"):
            offer(
                "target_covariance",
                [[float(value) for value in row] for row in np.asarray(covariance)],
            )

        # A variance head, when the run asks for one. Offered rather than forced, so a model
        # without that setting is left alone; an entry naming it wins, so one model can opt out.
        if "predict_variance" not in init_args and self._heteroscedastic_enabled():
            offer("predict_variance", True)
            offer("beta_nll", float(getattr(self.config, "UNCERTAINTY_BETA_NLL", 0.5)))

        return model_cls(**init_args)

    def _heteroscedastic_enabled(self) -> bool:
        """Whether the run asks for a :term:`variance head`: uncertainty on and heteroscedastic set."""
        return bool(getattr(self.config, "UNCERTAINTY_ENABLED", False)) and bool(
            getattr(self.config, "UNCERTAINTY_HETEROSCEDASTIC", False)
        )

    def _build_trainer_kwargs(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """The training settings: epochs, processor, precision, and how often to report."""
        trainer_kwargs = deepcopy(spec.get("trainer_args", {}))
        trainer_kwargs.setdefault("max_epochs", getattr(self.config, "LIGHTNING_MAX_EPOCHS", 50))
        trainer_kwargs.setdefault("accelerator", getattr(self.config, "LIGHTNING_ACCELERATOR", "auto"))
        trainer_kwargs.setdefault("devices", getattr(self.config, "LIGHTNING_DEVICES", "auto"))
        trainer_kwargs.setdefault("precision", getattr(self.config, "LIGHTNING_PRECISION", "32-true"))
        trainer_kwargs.setdefault(
            "accumulate_grad_batches", getattr(self.config, "LIGHTNING_ACCUMULATE_GRAD_BATCHES", 1)
        )
        trainer_kwargs.setdefault("gradient_clip_val", getattr(self.config, "LIGHTNING_GRADIENT_CLIP_VAL", 0.0))
        trainer_kwargs.setdefault("log_every_n_steps", getattr(self.config, "LIGHTNING_LOG_EVERY_N_STEPS", 1))
        trainer_kwargs.setdefault("enable_checkpointing", True)
        trainer_kwargs.setdefault("deterministic", True)
        return trainer_kwargs

    def _build_callback_specs(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """The early-stopping and checkpoint settings, with an entry's own values merged in."""
        callbacks = {
            "early_stopping": {
                "monitor": getattr(self.config, "LIGHTNING_EARLY_STOPPING_MONITOR", "val_loss"),
                "mode": getattr(self.config, "LIGHTNING_EARLY_STOPPING_MODE", "min"),
                "patience": getattr(self.config, "LIGHTNING_EARLY_STOPPING_PATIENCE", 5),
            },
            "checkpoint": {
                "monitor": getattr(self.config, "LIGHTNING_CHECKPOINT_MONITOR", "val_loss"),
                "mode": getattr(self.config, "LIGHTNING_CHECKPOINT_MODE", "min"),
                "save_top_k": getattr(self.config, "LIGHTNING_SAVE_TOP_K", 1),
            },
        }

        # Merged setting by setting: an entry naming only `patience` must keep the configured
        # monitor and mode rather than losing them.
        for group, overrides in deepcopy(spec.get("callbacks") or {}).items():
            if isinstance(overrides, Mapping) and isinstance(callbacks.get(group), MutableMapping):
                callbacks[group].update(overrides)
            else:
                callbacks[group] = overrides

        return callbacks