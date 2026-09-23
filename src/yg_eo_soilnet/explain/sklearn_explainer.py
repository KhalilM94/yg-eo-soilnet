"""Explaining a scikit-learn model.

The estimator is explained rather than the whole pipeline, so the contributions are per prepared
input - a one-hot column, not the raw category. The column names come from the pipeline itself, so
each contribution is labelled with the input it belongs to.

Tree and linear models have exact, fast explanations and are used where possible. Anything else
falls back to a method that repeatedly re-predicts with inputs hidden, which costs roughly the
number of points times the number of inputs; ``explain.max_evaluations`` caps that, and a model over
the cap is skipped with the reason recorded rather than left to run for hours.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yg_eo_soilnet.explain.result import LOG1P_X10, ORIGINAL_UNITS, ShapResult

_TREE_MARKERS = ("tree", "forest", "boost", "xgb", "lightgbm", "catboost")
_LINEAR_MARKERS = ("linear", "ridge", "lasso", "elasticnet", "ols", "bayesianridge")

# Ceiling on model evaluations for the model-agnostic path, in units of "rows scored".
# The number that motivated it: TabICL matches no marker, so it fell to KernelExplainer, whose
# default is 2*n_features + 2048 coalitions PER ROW. At 59 features and 500 explained rows that is
# ~1.08M forward passes through a transformer - the run never finished.
DEFAULT_MAX_EVALS = 200_000

# Coalitions per row for the agnostic explainer. Permutation SHAP needs 2*n_features + 1 to be
# exact, so this is expressed relative to the feature count and only clamped by the budget.
_AGNOSTIC_EVALS_PER_ROW = lambda n_features: 2 * n_features + 1  # noqa: E731


class ExplainBudgetExceeded(RuntimeError):
    """Raised when explaining a model would cost more predictions than allowed.

    Raised rather than quietly doing something cheaper, so the estimate and the limit are recorded and
    the skip is visible in the run.
    """


def _looks_like(estimator: Any, markers: tuple[str, ...]) -> bool:
    """Whether an estimator is of a given kind, judged by its class name.

    By name rather than by type, so XGBoost and CatBoost are recognised without importing them.
    """
    name = estimator.__class__.__name__.lower()
    module = estimator.__class__.__module__.lower()
    return any(marker in name or marker in module for marker in markers)


def _densify(matrix: Any) -> np.ndarray:
    """Return the prepared inputs as an ordinary array, whatever the pipeline produced."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _subsample(matrix: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Take at most this many points, at random, to explain."""
    if limit <= 0 or matrix.shape[0] <= limit:
        return matrix
    generator = np.random.default_rng(seed)
    picked = generator.choice(matrix.shape[0], size=limit, replace=False)
    return matrix[np.sort(picked)]


def _unwrap(fitted_estimator: Any) -> tuple[Any, Any, str]:
    """Take a pipeline apart into its input preparation, its estimator, and its output units."""
    from sklearn.compose import TransformedTargetRegressor

    named_steps = getattr(fitted_estimator, "named_steps", None)
    if named_steps is None or "model" not in named_steps:
        raise TypeError(
            "Expected a fitted Pipeline with a 'model' step as built by PipelineBuilder.build, got "
            f"{type(fitted_estimator).__name__}"
        )

    preprocessor = named_steps.get("preprocessor")
    estimator = named_steps["model"]
    output_space = ORIGINAL_UNITS

    if isinstance(estimator, TransformedTargetRegressor):
        # regressor_ (fitted) rather than regressor (the unfitted template).
        estimator = estimator.regressor_
        output_space = LOG1P_X10

    return preprocessor, estimator, output_space


def _as_values(raw: Any) -> np.ndarray:
    """The contributions as an array, keeping the per-target axis when the model has one.

    Kept, because a model predicting several targets has one set of contributions per target.
    """
    return np.asarray(getattr(raw, "values", raw), dtype=np.float64)


def _output_slice(values: np.ndarray, index: int) -> np.ndarray:
    """One target's contributions."""
    return values[..., index] if values.ndim == 3 else values


def _base_value_at(base_value: Any, index: int) -> float:
    """The average prediction for one target."""
    array = np.asarray(base_value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array[index]) if index < array.size else float(array[0])


def _agnostic(shap, estimator, background_matrix, explain_matrix, max_evals):
    """The general explanation method: re-predict with inputs hidden, many times over.

    Used for anything that is neither a tree nor a linear model. Refuses to start when the cost would
    exceed ``explain.max_evaluations``.
    """
    n_rows, n_features = explain_matrix.shape
    evals_per_row = _AGNOSTIC_EVALS_PER_ROW(n_features)
    estimated = n_rows * evals_per_row

    if estimated > max_evals:
        raise ExplainBudgetExceeded(
            f"model-agnostic SHAP would need about {estimated:,} model evaluations "
            f"({n_rows} rows x {evals_per_row} per row at {n_features} features), over the "
            f"EXPLAIN_MAX_EVALS budget of {max_evals:,}. Lower EXPLAIN_MAX_SAMPLES, or raise "
            f"EXPLAIN_MAX_EVALS if you want to pay for it."
        )

    # PermutationExplainer explicitly, not shap.Explainer's auto-selection. Auto-selection picks the
    # Exact explainer when the feature count is small, and Exact needs 2**n_features evaluations per
    # row - so the estimate above would be wrong by orders of magnitude in exactly the case it looks
    # safest, and shap would then refuse the budget we handed it. A budget is only meaningful
    # against a known cost model.
    explainer = shap.PermutationExplainer(estimator.predict, background_matrix)
    # silent=True, or shap writes one progress line per explained row to stderr - several hundred
    # per model, which buries the training log. It is a __call__ argument, not a constructor one.
    values = explainer(explain_matrix, max_evals=evals_per_row, silent=True)
    return _as_values(values), explainer


def _explain(shap, *, estimator, explain_matrix, background_matrix, max_evals):
    """Explain the estimator with the cheapest method that actually works.

    The exact methods are tried first and can fail on a model they nominally support, so the slower
    general method is the fallback.

    Returns
    -------
    values : numpy.ndarray
    base_value : float or numpy.ndarray
    explainer_name : str
    """
    fast = None
    if _looks_like(estimator, _TREE_MARKERS):
        fast = ("TreeExplainer", lambda: shap.TreeExplainer(estimator))
    elif _looks_like(estimator, _LINEAR_MARKERS):
        fast = ("LinearExplainer", lambda: shap.LinearExplainer(estimator, background_matrix))

    if fast is not None:
        name, build = fast
        try:
            explainer = build()
            values = _as_values(explainer.shap_values(explain_matrix))
            return values, _base_value(explainer), name
        except ExplainBudgetExceeded:
            raise
        except Exception:
            pass  # fall through; the reason is recorded via the explainer name that ends up used

    values, explainer = _agnostic(shap, estimator, background_matrix, explain_matrix, max_evals)
    return values, _base_value(explainer), type(explainer).__name__


def _blocks_from_feature_names(feature_names: list[str]) -> list[str]:
    """Group the prepared inputs by whether they came from a number or a category."""
    labels = {"num": "continuous", "cat": "categorical"}
    blocks = []
    for name in feature_names:
        prefix = name.split("__", 1)[0] if "__" in name else ""
        blocks.append(labels.get(prefix, "features"))
    return blocks


def _base_value(explainer):
    """The average prediction the contributions are measured from, per target."""
    expected = getattr(explainer, "expected_value", None)
    if expected is None:
        return 0.0
    array = np.asarray(expected, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return array if array.size > 1 else float(array[0])


def sklearn_shap_results(
    *,
    config,
    fitted_estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target: str,
    target_names: Any = None,
) -> list[ShapResult]:
    """Explain a fitted scikit-learn model, for every target it predicts.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.
    X : pandas.DataFrame
        The points to explain, before preparation.
    target : str
        The :term:`target group`, used to name outputs when there is nothing better.
    config : Config, optional
        Read for ``explain.max_samples`` and ``explain.max_evaluations``.

    Returns
    -------
    list of ShapResult
        One per target.

    Raises
    ------
    ExplainBudgetExceeded
        If the model would cost more predictions than allowed.
    """
    import shap

    preprocessor, estimator, output_space = _unwrap(fitted_estimator)

    if preprocessor is None:
        feature_names = list(X_test.columns)
        explain_matrix = _densify(X_test.to_numpy())
        background_matrix = _densify(X_train.to_numpy())
    else:
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        explain_matrix = _densify(preprocessor.transform(X_test))
        background_matrix = _densify(preprocessor.transform(X_train))

    seed = int(getattr(config, "RANDOM_SEED", 42) or 42)
    explain_matrix = _subsample(explain_matrix, int(getattr(config, "EXPLAIN_MAX_SAMPLES", 500)), seed)
    background_matrix = _subsample(background_matrix, int(getattr(config, "EXPLAIN_BACKGROUND_SAMPLES", 100)), seed)

    if explain_matrix.size == 0 or not feature_names:
        return []

    values, base_value, explainer_name = _explain(
        shap,
        estimator=estimator,
        explain_matrix=explain_matrix,
        background_matrix=background_matrix,
        max_evals=int(getattr(config, "EXPLAIN_MAX_EVALS", DEFAULT_MAX_EVALS)),
    )

    # A joint fit has one output per target. `_as_values` keeps that axis - it used to be dropped
    # with `[..., 0]`, so every target's plots showed the FIRST target's attributions under its own
    # name. EVERY output is returned, in output order: the caller explains the joint model once, at
    # model-run scope, and routes each output to the run that holds that target's evaluation. This
    # used to slice down to the single output matching `target`, which paid for the full multi-output
    # explanation N times over to throw away N-1 of it each time.
    n_outputs = values.shape[2] if values.ndim == 3 else 1
    names = [str(name) for name in (target_names or [])]
    if len(names) != n_outputs:
        names = [str(target)] if n_outputs == 1 else [f"{target}_{index}" for index in range(n_outputs)]

    return [
        ShapResult(
            values=_output_slice(values, index),
            data=explain_matrix,
            feature_names=feature_names,
            blocks=_blocks_from_feature_names(feature_names),
            target_name=names[index],
            output_space=output_space,
            base_value=_base_value_at(base_value, index),
            explainer=explainer_name,
        )
        for index in range(n_outputs)
    ]
