"""Work out how much each input pushed each prediction up or down - :term:`SHAP`.

Switched on with ``explain.enabled``. For every point, each input gets a contribution, and those
contributions add up to the difference between that point's prediction and the average prediction.
So a figure can say "this point is predicted high mainly because of its July greenness and its
elevation", rather than only reporting which inputs matter on average.

Both model families are handled, and the results take the same shape either way, so the figures and
the tables are written once. Nothing here is imported unless explanations are switched on: the
library is slow to load.
"""

from __future__ import annotations

from yg_eo_soilnet.artifacts import ArtifactLayout, log_figure, log_json, log_parquet
from yg_eo_soilnet.explain.result import (
    LOG1P_X10,
    ORIGINAL_UNITS,
    STANDARDIZED_LOG1P,
    ShapResult,
)

__all__ = [
    "ShapResult",
    "ORIGINAL_UNITS",
    "LOG1P_X10",
    "STANDARDIZED_LOG1P",
    "build_shap_results",
    "log_shap_artifacts",
]


def build_shap_results(*, config, backend: str, **payload) -> list[ShapResult]:
    """Explain one trained model, for every target it predicts.

    Parameters
    ----------
    backend : {"sklearn", "lightning"}
        Which family the model belongs to.
    target : str
        The :term:`target group`, used to name outputs when the model carries no target names.
    **kwargs
        Passed to that family's explainer.

    Returns
    -------
    list of ShapResult
        One per target the model predicts, in order. A model predicting several is explained once.
        """
    if backend == "sklearn":
        from yg_eo_soilnet.explain.sklearn_explainer import sklearn_shap_results

        return sklearn_shap_results(config=config, **payload)

    if backend == "lightning":
        from yg_eo_soilnet.explain.lightning_explainer import lightning_shap_results

        return lightning_shap_results(config=config, **payload)

    raise ValueError(f"Unknown explain backend {backend!r}; expected 'sklearn' or 'lightning'")


def log_shap_artifacts(results: list[ShapResult], *, max_display: int = 25) -> dict:
    """Write one target's explanation into the current run, and say what was written.

    Per target: a figure showing each input's contribution for every point, a bar chart of what matters
    on average, a bar chart grouped by input type, and the complete table of contributions.

    Returns
    -------
    dict
        What went into the run summary.
        """
    from yg_eo_soilnet.explain.plots import shap_bar, shap_beeswarm, shap_block_bar

    written: dict = {"targets": [], "artifacts": []}

    # Filenames are stable so the same plot occupies the same path in every run and MLflow's compare
    # view can line them up. Only a run emitting SEVERAL targets nests them under explain/<target>/,
    # where the flat layout would have them overwrite each other.
    multi_target = len(results) > 1

    for result in results:
        artifact_path = ArtifactLayout.explain_path(result.target_name if multi_target else None)

        figures = [
            (shap_beeswarm(result, max_display=max_display), ArtifactLayout.SHAP_BEESWARM_FILE),
            (shap_bar(result, max_display=max_display), ArtifactLayout.SHAP_BAR_FILE),
            (shap_block_bar(result), ArtifactLayout.SHAP_BLOCK_BAR_FILE),
        ]
        for figure, filename in figures:
            log_figure(figure, filename, artifact_path)
            written["artifacts"].append(f"{artifact_path}/{filename}")

        log_parquet(result.to_frame(), ArtifactLayout.SHAP_VALUES_FILE, artifact_path)
        written["artifacts"].append(f"{artifact_path}/{ArtifactLayout.SHAP_VALUES_FILE}")

        log_json(result.summary(top_n=max_display), ArtifactLayout.SHAP_SUMMARY_FILE, artifact_path)
        written["artifacts"].append(f"{artifact_path}/{ArtifactLayout.SHAP_SUMMARY_FILE}")

        written["targets"].append(
            {
                "target": result.target_name,
                "output_space": result.output_space,
                "n_samples": result.n_samples,
                "n_features": result.n_features,
                "top_features": [
                    result.feature_names[index] for index in result.ranking()[:10]
                ],
                "blocks": result.block_mean_abs(),
            }
        )

    return written
