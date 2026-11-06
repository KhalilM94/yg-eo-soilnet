"""Build either A1 stage's estimator the way ``main.py`` builds a scikit-learn model.

A stage names an entry of the scikit-learn model list (``configs/sklearn/model_registry.yml``);
``import_path``, ``init_args`` and ``params`` in the A1 settings can be laid over it. The estimator
is built by :class:`~yg_eo_soilnet.models.config_fatories.model_config_factory.ModelConfigFactory`
and wrapped by :class:`~yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils.PipelineBuilder`, so
its inputs are prepared exactly as the main pipeline prepares them: gaps filled with the median and
flagged, numbers scaled for non-tree models, categories encoded, and the target log-transformed
when it is listed in ``COLUMNS_TO_TRANSFORM``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from yg_eo_soilnet.a1.settings import A1Settings, StageSettings


def build_stage_pipeline(
    settings: A1Settings,
    config,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    *,
    stage: StageSettings,
    target: Optional[str] = None,
):
    """One stage's estimator in its input preparation, and the grid of settings to tune it over.

    Parameters
    ----------
    settings : A1Settings
    config : Config
        The main configuration: the model list and the scikit-learn settings.
    feature_columns : sequence of str
        The inputs.
    categorical_columns : sequence of str
        The inputs that are classes; any not among ``feature_columns`` is ignored.
    stage : StageSettings
        ``settings.stage1`` or ``settings.stage2``.
    target : str, optional
        The property predicted, for the log transform.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        With the first value of every grid setting already applied, so it can be fitted untuned.
    grid : dict of str to list
        Each setting, named as the pipeline names it, with the values to try.

    Raises
    ------
    ValueError
        If the entry is not in the model list, or cannot be built.
    """
    from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder
    from yg_eo_soilnet.models import ModelConfigFactory

    registry = getattr(config, "MODEL_REGISTRY", {}) or {}
    if stage.estimator not in registry:
        raise ValueError(
            f"{stage.estimator!r} is not an entry of the scikit-learn model list; it has {sorted(registry)}"
        )
    spec = dict(registry[stage.estimator])
    spec["enabled"] = True
    if stage.import_path:
        spec["import_path"] = stage.import_path
    spec["init_args"] = {**(spec.get("init_args") or {}), **dict(stage.init_args)}

    built = ModelConfigFactory(registry={stage.estimator: spec}).build_model_configs(
        num_features=len(feature_columns), default_seed=settings.seed
    )
    if stage.estimator not in built:
        raise ValueError(f"Could not build {stage.estimator} from {spec.get('import_path')!r}; see the warning above.")
    model = built[stage.estimator]["model"]

    features = list(feature_columns)
    categorical = [column for column in categorical_columns if column in features]
    numeric = [column for column in features if column not in categorical]
    is_log_target = target is not None and target in (getattr(config, "COLUMNS_TO_TRANSFORM", []) or [])
    builder = PipelineBuilder(
        seed=settings.seed,
        tree_categorical_encoding=getattr(config, "TREE_CATEGORICAL_ENCODING", "ordinal"),
        tree_onehot_max_categories=getattr(config, "TREE_ONEHOT_MAX_CATEGORIES", None),
    )
    pipeline = builder.build(model, is_log_target, categorical_cols=categorical, numeric_cols=numeric)

    raw_grid = stage.params if stage.params is not None else (spec.get("params") or {})
    grid = {
        name: (list(values) if isinstance(values, (list, tuple)) else [values]) for name, values in raw_grid.items()
    }
    if is_log_target:
        # A logged target wraps the model one layer deeper, so the setting names follow.
        grid = {name.replace("model__", "model__regressor__"): values for name, values in grid.items()}
    if grid:
        pipeline.set_params(**{name: values[0] for name, values in grid.items()})
    return pipeline, grid
