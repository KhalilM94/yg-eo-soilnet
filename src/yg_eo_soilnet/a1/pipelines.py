"""Build either A1 stage's estimator the way ``main.py`` builds a scikit-learn model.

A stage names an entry of the scikit-learn model list (``configs/sklearn/model_registry.yml``);
``import_path``, ``init_args`` and ``params`` in the A1 settings can be laid over it. The estimator
is built by :class:`~yg_eo_soilnet.models.config_fatories.model_config_factory.ModelConfigFactory`
and wrapped by :class:`~yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils.PipelineBuilder`, so
its inputs are prepared exactly as the main pipeline prepares them: gaps filled with the median and
flagged, numbers scaled for non-tree models, categories encoded, and the target log-transformed
when it is listed in ``COLUMNS_TO_TRANSFORM``.

The estimator is seeded with the settings' model seed
(:attr:`~yg_eo_soilnet.a1.settings.A1Settings.model_seed`), and takes the run's processor
(``run_a1.py --device``) and thread count (``--workers``) when it has those settings. A model
fitted on the graphics card is moved to the processor before it predicts (:func:`to_cpu`): the
data are on the processor, and XGBoost would otherwise copy them over, with a warning, on every
call.

E5's ``native_categorical`` variants hand the class inputs to XGBoost as classes instead
(:class:`NativeCategories`): no one-hot columns and no filling, so a tree can split any group of
classes from the rest and learns where a point without a class goes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from yg_eo_soilnet.a1.settings import A1Settings, StageSettings


def build_stage_pipeline(
    settings: A1Settings,
    config,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    *,
    stage: StageSettings,
    target: Optional[str] = None,
    native_categorical: bool = False,
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
    native_categorical : bool, default False
        Give the class inputs to the model as classes (:class:`NativeCategories`), gaps left
        missing, with XGBoost's ``enable_categorical``. Only for XGBoost; the numeric inputs are
        prepared as usual.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        With the first value of every grid setting already applied, so it can be fitted untuned.
    grid : dict of str to list
        Each setting, named as the pipeline names it, with the values to try.

    Raises
    ------
    ValueError
        If the entry is not in the model list, or cannot be built, or ``native_categorical`` is
        asked of a model other than XGBoost.
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
        num_features=len(feature_columns), default_seed=settings.model_seed
    )
    if stage.estimator not in built:
        raise ValueError(f"Could not build {stage.estimator} from {spec.get('import_path')!r}; see the warning above.")
    model = built[stage.estimator]["model"]
    exposed = model.get_params(deep=False)
    if "random_state" in exposed and settings.model_seed != settings.seed:
        # A further seed reseeds every estimator; the first keeps what the model list gives (the
        # seed, unless an entry pins its own random_state).
        model.set_params(random_state=int(settings.model_seed))
    if settings.device is not None and "device" in exposed:
        model.set_params(device=settings.device)
    if settings.n_jobs is not None and "n_jobs" in exposed:
        model.set_params(n_jobs=int(settings.n_jobs))

    features = list(feature_columns)
    categorical = [column for column in categorical_columns if column in features]
    numeric = [column for column in features if column not in categorical]
    is_log_target = target is not None and target in (getattr(config, "COLUMNS_TO_TRANSFORM", []) or [])
    builder = PipelineBuilder(
        seed=settings.model_seed,
        tree_categorical_encoding=getattr(config, "TREE_CATEGORICAL_ENCODING", "ordinal"),
        tree_onehot_max_categories=getattr(config, "TREE_ONEHOT_MAX_CATEGORIES", None),
    )
    if native_categorical and categorical:
        if not type(model).__module__.startswith("xgboost"):
            raise ValueError(
                f"native_categorical needs XGBoost, which reads classes itself; {stage.estimator} is "
                f"{type(model).__name__}."
            )
        model.set_params(enable_categorical=True)
        pipeline = builder.build(model, is_log_target, categorical_cols=[], numeric_cols=numeric)
        pipeline.steps[0] = ("preprocessor", native_preprocessor(pipeline.steps[0][1], categorical))
    else:
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


class NativeCategories(TransformerMixin, BaseEstimator):
    """Class inputs as pandas classes, for a model that reads them itself (XGBoost's ``enable_categorical``).

    The classes are learned on the training points and fixed: a class first seen at prediction
    becomes missing, like a gap. Gaps are not filled - the model decides where a point without a
    class goes.

    >>> step = NativeCategories().fit(pd.DataFrame({"texture": ["lo", "cl", None]}))
    >>> out = step.transform(pd.DataFrame({"texture": ["cl", "sa", None]}))
    >>> list(out["texture"].cat.categories), out["texture"].isna().tolist()
    (['cl', 'lo'], [False, True, True])
    """

    def fit(self, X, y=None):
        """Learn each column's classes (sorted, gaps left out)."""
        frame = pd.DataFrame(X)
        self.categories_ = {column: sorted(set(frame[column].dropna().map(_as_text))) for column in frame.columns}
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        """Each column as a pandas class column with the learned classes."""
        frame = pd.DataFrame(X)
        return pd.DataFrame(
            {
                column: pd.Categorical(frame[column].map(_as_text), categories=classes)
                for column, classes in self.categories_.items()
            },
            index=frame.index,
        )

    def get_feature_names_out(self, input_features=None):
        """The column names, unchanged."""
        return np.asarray(list(self.categories_), dtype=object)


def _as_text(value):
    """A class as text (so 3, 3.0 - a code in a column with gaps - and "3" are one class); a gap stays a gap."""
    if pd.isna(value):
        return value
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def native_preprocessor(numeric_preprocessor, categorical: Sequence[str]):
    """The builder's numeric preparation with the class columns added as :class:`NativeCategories`.

    Returns a :class:`~sklearn.compose.ColumnTransformer` giving a pandas table, so the class
    columns reach the model with their class type.
    """
    from sklearn.compose import ColumnTransformer

    transformers = [*numeric_preprocessor.transformers, ("cat", NativeCategories(), list(categorical))]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False).set_output(
        transform="pandas"
    )


def final_estimator(fitted):
    """The model inside a fitted pipeline: past the input preparation and the target transform."""
    estimator = fitted.named_steps["model"] if hasattr(fitted, "named_steps") else fitted
    # A log-transformed target wraps the model: regressor_ once fitted, regressor before.
    for name in ("regressor_", "regressor"):
        inner = getattr(estimator, name, None)
        if inner is not None:
            return inner
    return estimator


def to_cpu(fitted):
    """Move a fitted model to the processor, where the data are, before it predicts; returns it.

    Only a model with a ``device`` setting set to something else (XGBoost on ``cuda``) is changed.
    The trees are the same; only where the prediction runs changes.
    """
    estimator = final_estimator(fitted)
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    if params.get("device") not in (None, "cpu") and hasattr(estimator, "get_booster"):
        estimator.set_params(device="cpu")
    return fitted


def predict_on_cpu(fitted, X) -> np.ndarray:
    """:func:`to_cpu`, then the model's predictions for ``X`` as a flat float array."""
    return np.asarray(to_cpu(fitted).predict(X), dtype=float).reshape(-1)


def stage_device(settings: A1Settings, config, stage: StageSettings) -> str:
    """The processor a stage fits on: ``cpu``, ``cuda`` (or ``cuda:<n>``), or ``auto``.

    Read from the estimator as it is built (the settings' ``--device`` override, the stage's
    ``init_args``, the model list's), so it is what the fits use. An estimator without a ``device``
    setting runs on the processor; one left to choose for itself (TabICL's default) is ``auto``.
    """
    pipeline, _ = build_stage_pipeline(settings, config, ["x"], stage=stage)
    estimator = final_estimator(pipeline)
    params = estimator.get_params(deep=False)
    if "device" not in params:
        return "cpu"
    device = params["device"]
    if device is None:
        # XGBoost left unset stays on the processor; others (TabICL) pick for themselves.
        return "cpu" if type(estimator).__module__.startswith("xgboost") else "auto"
    return str(device)
