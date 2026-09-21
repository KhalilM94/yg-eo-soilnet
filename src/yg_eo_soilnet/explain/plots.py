"""SHAP figures, following the project's plotting convention.

As in plot_utils and hpo.plots: build a Figure, style it through plot_style, and RETURN it - callers
save and close (yg_eo_soilnet.artifacts.log_figure does both). When there is nothing to draw, hand
back a figure carrying a message rather than None, so a caller never has to branch on the return
value.

Two of the three figures here are drawn by shap itself, so the only lever on their style is the
rcParams in force while shap draws - hence the ``style_context()`` inside ``_capture`` rather than a
``@styled`` decorator that would have closed before shap ran.

``import shap`` stays inside the functions so the module is cheap to import with SHAP disabled.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from yg_eo_soilnet.explain.result import ShapResult
from yg_eo_soilnet.plot_style import (
    FIG_WIDTH_COLUMN,
    INK_2,
    PROJECT_COLORS,
    message_figure as _message_figure,
    panel_subtitle,
    style_context,
    styled,
)


def _explanation(result: ShapResult):
    import shap

    return shap.Explanation(
        values=result.values,
        data=result.data,
        feature_names=list(result.feature_names),
        base_values=np.full(result.n_samples, float(result.base_value)),
    )


def _capture(draw, title: str):
    """Run a shap plotting call that draws on the current axes and hand back its Figure.

    shap's plotting helpers draw into pyplot state and return None, which is the opposite of this
    project's convention, so the figure is created here and reclaimed with gcf() afterwards.

    The style context wraps the DRAW, not just the figure creation: shap sets its own colours on the
    artists it makes, and the fonts, spines and tick colours it does not set are read from rcParams
    at draw time.
    """
    with style_context():
        figure = plt.figure()
        try:
            draw()
        except Exception as exc:
            plt.close(figure)
            return _message_figure(f"{title} could not be drawn: {type(exc).__name__}: {exc}")
        drawn = plt.gcf()
        if drawn is not figure:
            plt.close(figure)
        if drawn.axes:
            panel_subtitle(drawn.axes[0], title)
        else:
            drawn.suptitle(title, fontsize=8, color=INK_2)
        drawn.tight_layout()
        return drawn


def shap_beeswarm(result: ShapResult, max_display: int = 25):
    """Per-feature beeswarm: one row per feature, one dot per sample, coloured by feature value."""
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.beeswarm(_explanation(result), max_display=max_display, show=False),
        f"SHAP beeswarm - {result.target_name} ({result.output_space})",
    )


def shap_bar(result: ShapResult, max_display: int = 25):
    """Per-feature mean |SHAP| bar. shap folds the tail into a 'sum of N other features' row."""
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.bar(_explanation(result), max_display=max_display, show=False),
        f"Mean |SHAP| - {result.target_name} ({result.output_space})",
    )


@styled
def shap_block_bar(result: ShapResult):
    """Rolled-up mean |SHAP| per block: static, categorical, auxiliary, and one bar per modality.

    The per-feature plots are capped at the top N, and with every band of every modality holding its
    own row the temporal branch can dominate in aggregate while no single band ranks highly enough to
    be displayed. This view is the antidote, and the direct quantitative answer to the question of
    what the time series actually buys over the static features.
    """
    blocks = result.block_mean_abs()
    if not blocks:
        return _message_figure("No SHAP blocks to plot")

    names = list(blocks)
    heights = [blocks[name] for name in names]
    order = np.argsort(heights)
    ordered_names = [names[index] for index in order]
    ordered_heights = [heights[index] for index in order]

    # Height grows with the bar count: a fixed chrome allowance plus a fixed slice per row, so ten
    # blocks and three blocks both get bars of the same thickness.
    figure, axes = plt.subplots(
        figsize=(FIG_WIDTH_COLUMN, 0.9 + 0.2 * len(names)), layout="constrained"
    )
    bars = axes.barh(ordered_names, ordered_heights, color=PROJECT_COLORS["Fertimap"])
    # Horizontal bars flip which grid does the work: the value runs along x, so the y grid marks
    # nothing and the y ticks are labels rather than measurements.
    axes.grid(False)
    axes.grid(True, axis="x")
    axes.set_axisbelow(True)
    axes.tick_params(axis="y", length=0)
    axes.bar_label(bars, labels=[f"{value:,.3g}" for value in ordered_heights],
                   padding=2, fontsize=7, color=INK_2)
    axes.set_xlabel(f"mean |sum of SHAP within block| ({result.output_space})")
    panel_subtitle(axes, f"contribution by block - {result.target_name}")
    return figure
