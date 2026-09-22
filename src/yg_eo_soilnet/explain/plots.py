"""The :term:`SHAP` figures.

As everywhere else here, each function builds a figure and returns it; the caller saves it. With
nothing to draw, a figure carrying a message comes back rather than nothing, so no caller has to
check.
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
    """Package the values in the form the SHAP plotting library expects."""
    import shap

    return shap.Explanation(
        values=result.values,
        data=result.data,
        feature_names=list(result.feature_names),
        base_values=np.full(result.n_samples, float(result.base_value)),
    )


def _capture(draw, title: str):
    """Run a SHAP plotting call and hand back the figure it drew.

    The library draws into the current figure and returns nothing, so one is made here first.
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
    """One row per input, one dot per point: how much that input moved that prediction.

    The dots are coloured by the input's own value, so a row reading "high values on the right" means
    more of that input predicts more of the target.

    Returns
    -------
    matplotlib.figure.Figure
        """
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.beeswarm(_explanation(result), max_display=max_display, show=False),
        f"SHAP beeswarm - {result.target_name} ({result.output_space})",
    )


def shap_bar(result: ShapResult, max_display: int = 25):
    """A bar per input: how much it matters on average. The rest are folded into one row."""
    import shap

    if result.n_samples == 0 or result.n_features == 0:
        return _message_figure("No SHAP values to plot")

    return _capture(
        lambda: shap.plots.bar(_explanation(result), max_display=max_display, show=False),
        f"Mean |SHAP| - {result.target_name} ({result.output_space})",
    )


@styled
def shap_block_bar(result: ShapResult):
    """A bar per *group* of inputs: covariates, categories, each data source, and so on.

    The per-input figures show only the top rows, and with every band of every data source having its
    own row the time series can fill them all. This is the view that says how much each kind of input
    contributes in total.
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
