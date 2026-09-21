"""Figures for a training run: predicted-vs-observed, the parent overlay, the leaderboard, CV sweeps.

Every function here builds a Figure, styles it through :mod:`yg_eo_soilnet.plot_style`, and RETURNS
it. The caller saves and closes - ``yg_eo_soilnet.artifacts.log_figure`` does both. That is now the
whole repo's contract without exception; ``create_pred_obs_plot`` used to save itself to satisfy
MLflow's custom-artifact hook, and that hook has been removed because the frame the evaluator handed
it never carried the uncertainty columns, so it only ever produced a worse duplicate of a plot the
logger was already writing.
"""

import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path
from sklearn.metrics import r2_score, root_mean_squared_error

from yg_eo_soilnet.plot_style import (
    BASELINE,
    CARBONATE_RAMP,
    FIG_WIDTH_COLUMN,
    FIG_WIDTH_FULL,
    FERTIMAP_TINT,
    INK,
    INK_2,
    MODEL_COLORS,
    PROJECT_COLORS,
    message_figure,
    metric_box,
    panel_letter,
    panel_subtitle,
    sequential_cmap,
    square_panel,
    styled,
)
from yg_eo_soilnet.uncertainty.columns import interval_columns, is_prediction_column, sigma_column
from yg_eo_soilnet.utils import rpiq_score


def _normalise_target_names(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return [value for value in values if value is not None and str(value) != ""]
    except TypeError:
        return [values]


def _resolve_prediction_column(df, target_name=None, target_index=None):
    candidates = []
    if target_name:
        candidates.extend(
            [
                f"prediction_{target_name}",
                f"prediction_{str(target_name).replace(' ', '_')}",
                f"pred_{target_name}",
            ]
        )
    if target_index is not None:
        candidates.append(f"prediction_{target_index}")
    candidates.append("prediction")

    for column_name in candidates:
        if column_name in df.columns:
            return column_name

    # is_prediction_column, not `startswith("prediction_")`. An eval frame from an uncertainty run
    # also carries prediction_std_<t>, prediction_lower_<t> and friends, and the positional fallback
    # below would happily return one of those - drawing standard deviations on the predicted axis,
    # with a plot that looks plausible and is wrong.
    prediction_columns = [
        column_name for column_name in df.columns if is_prediction_column(column_name)
    ]
    if target_index is not None and target_index < len(prediction_columns):
        return prediction_columns[target_index]
    if prediction_columns:
        return prediction_columns[0]
    return None


@styled
def _create_parent_pred_obs_multitarget(eval_dfs):
    if not eval_dfs:
        return None

    prepared_frames = []
    target_names = []

    for eval_df in eval_dfs:
        if eval_df is None or eval_df.empty:
            continue
        frame = eval_df.copy()
        frame_targets = _normalise_target_names(frame["target_name"].dropna().unique()) if "target_name" in frame.columns else []
        if not frame_targets:
            frame_targets = [None]
        frame["_resolved_target_name"] = frame["target_name"] if "target_name" in frame.columns else None
        prepared_frames.append((frame, frame_targets))
        for target_name in frame_targets:
            if target_name is not None and target_name not in target_names:
                target_names.append(target_name)

    if not prepared_frames:
        return None

    if not target_names:
        target_names = [None]

    n_targets = len(target_names)
    n_cols = min(2, n_targets) if n_targets > 1 else 1
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        # One panel is a single column wide; two abreast fill the page. Square-ish, because every
        # panel here is a pred-vs-obs scatter framed at a 1:1 aspect.
        figsize=(FIG_WIDTH_FULL if n_cols > 1 else FIG_WIDTH_COLUMN, 3.1 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    # One colour per MODEL, assigned once across the whole figure rather than per panel, so a model
    # keeps its colour from one target to the next and the shared legend below means the same thing
    # on every panel.
    model_colors: dict[str, str] = {}

    def color_for(model_name):
        key = str(model_name)
        if key not in model_colors:
            model_colors[key] = MODEL_COLORS[len(model_colors) % len(MODEL_COLORS)]
        return model_colors[key]

    for target_index, target_name in enumerate(target_names):
        axis = axes[target_index // n_cols][target_index % n_cols]
        square_panel(axis)
        panel_letter(axis, "abcdefghijklmnopqrstuvwxyz"[target_index % 26])
        panel_subtitle(axis, str(target_name) if target_name is not None else "predicted vs observed")
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")

        all_values = []
        for frame, frame_targets in prepared_frames:
            # Only frames that actually scored THIS target belong on this panel. A run whose targets
            # each got their own child - which is what happens whenever an estimator cannot fit a
            # 2-D y, so `MULTI_TARGET_MODE: joint` falls back to one model per target - produces one
            # frame per target, and every one of them carries a plain `prediction` column. Without
            # this check the panel for target A also picks up target B's frame, fails to find an
            # `A` column in it, and invents one from whatever column happens to come first: a
            # reflectance band on a numeric dataset, a categorical on this one. The former plotted
            # silently wrong for months; the latter is the isfinite TypeError that finally surfaced
            # it.
            #
            # Both halves of `covers` earn their place. `frame_targets != [None]` keeps a frame that
            # carries no target_name at all on its legacy path, and the column test means a frame
            # that genuinely holds this target's observations is never dropped over its label.
            covers = target_name in frame_targets or target_name in frame.columns
            if target_name is not None and frame_targets != [None] and not covers:
                continue

            resolved_target_name = target_name if target_name is not None else (frame_targets[0] if frame_targets else None)
            prediction_column = _resolve_prediction_column(frame, resolved_target_name, target_index)
            if prediction_column is None:
                continue

            observed_column = None
            for candidate in (
                resolved_target_name,
                "target",
                "y_true",
                "obs",
                "observation",
                "actual",
            ):
                if candidate is not None and candidate in frame.columns:
                    observed_column = candidate
                    break

            if observed_column is None:
                # Numeric by DTYPE, not merely by not being one of two known label columns - the
                # old test was a name check under a name that promised otherwise. Skipping when
                # nothing qualifies beats plotting an arbitrary column against the predictions.
                numeric_candidates = [
                    column_name
                    for column_name in frame.columns
                    if column_name not in {"model_name", "target_name", "target_names"}
                    and pd.api.types.is_numeric_dtype(frame[column_name])
                ]
                if not numeric_candidates:
                    continue
                observed_column = numeric_candidates[0]

            # Coerced rather than trusted, the same way metrics._finite_pairs does it: a column of
            # numbers-as-strings still plots, and anything genuinely non-numeric becomes NaN and is
            # masked out below instead of raising from inside a ufunc.
            x_values = pd.to_numeric(frame[observed_column], errors="coerce").to_numpy(dtype=float)
            y_values = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            if not valid_mask.any():
                continue

            model_name = frame["model_name"].iloc[0] if "model_name" in frame.columns and not frame["model_name"].empty else "model"
            series_color = color_for(model_name)
            # Bars before the points, under them, and in the series' own colour so two models
            # overlaid on one axis stay distinguishable. Thinner and fainter than on the per-run
            # plot: this axis carries every model at once, so several models' bars overlap here
            # where only one model's do there.
            interval = interval_columns(frame, resolved_target_name)
            if interval is not None:
                observed = x_values[valid_mask]
                predicted = y_values[valid_mask]
                lower = interval[0].to_numpy()[valid_mask]
                upper = interval[1].to_numpy()[valid_mask]
                # Thinned by the same rule the per-run panel uses, and for the same reason: a bar on
                # every one of ~750 points merges into a solid curtain that hides the scatter
                # underneath, which is the one thing the panel exists to show. Spread evenly across
                # the x range rather than sampled at random, so the drawn bars still span the axis.
                positions = _error_bar_positions(observed, cap=MAX_PARENT_ERROR_BARS)
                axis.errorbar(
                    observed[positions],
                    predicted[positions],
                    yerr=[
                        np.maximum(predicted - lower, 0.0)[positions],
                        np.maximum(upper - predicted, 0.0)[positions],
                    ],
                    fmt="none",
                    ecolor=series_color,
                    elinewidth=0.6,
                    alpha=0.2,
                    capsize=0,
                    zorder=1,
                )
            axis.scatter(
                x_values[valid_mask],
                y_values[valid_mask],
                s=5,
                lw=0,
                label=str(model_name),
                color=series_color,
                zorder=3,
                rasterized=True,
            )
            all_values.extend([x_values[valid_mask], y_values[valid_mask]])

        if all_values:
            combined = np.concatenate(all_values)
            # The same square framing the per-run panel gets, for the same reason: without one
            # shared range and a 1:1 aspect the identity line is not a 45 degree diagonal and the
            # cloud is stretched along whichever axis spans less.
            low, high = _square_limits(combined, combined)
            axis.plot(
                [low, high], [low, high], linestyle="--", color=INK_2, linewidth=0.8, zorder=5
            )
            _frame_square(axis, low, high)

    for axis in axes.flatten()[n_targets:]:
        axis.set_visible(False)

    # One legend for the whole figure, not one per panel. A model's colour is the same everywhere
    # here, so repeating the key on every panel spends space saying the same thing several times -
    # and an in-panel legend on a square scatter always covers data.
    if model_colors:
        fig.legend(
            handles=[
                Line2D([], [], ls="", marker="o", markersize=4, color=color, label=name)
                for name, color in model_colors.items()
            ],
            loc="outside lower center",
            ncol=min(len(model_colors), 4),
            fontsize=7.5,
        )
    return fig


@styled
def cv_val_curve(cv_results, scoring: str = "neg_root_mean_squared_error"):
    """
    Plot mean train/test CV scores with std bands, and highlight the best parameter.

    Args:
        cv_results (dict): The cv_results_ attribute from GridSearchCV
        param_name (str): The hyperparameter name (e.g. 'model__n_components')
        scoring (str): The scoring metric used in GridSearchCV.
                       If it's a "neg_*" metric, values will be flipped.

    Returns:
        matplotlib.figure.Figure: The figure object
    """

    param_key, = [str(col) for col in cv_results.columns if str(col).startswith("param_")]
    param_name = param_key.rsplit("__", 1)[-1]
    param_values = np.array(cv_results[param_key], dtype=object)

    # Handle categorical (non-numeric) params
    if not np.issubdtype(param_values.dtype, np.number):
        param_values = param_values.astype(str)

    # Flip scores if it's a neg_* metric
    def process_scores(scores):
        return -scores if scoring.startswith("neg_") else scores

    mean_train = process_scores(np.array(cv_results["mean_train_score"], dtype=float))
    std_train = np.array(cv_results["std_train_score"], dtype=float)

    mean_test = process_scores(np.array(cv_results["mean_test_score"], dtype=float))
    std_test = np.array(cv_results["std_test_score"], dtype=float)

    # Best param index
    best_idx = np.argmax(mean_test) if not scoring.startswith("neg_") else np.argmin(mean_test)
    best_param = param_values[best_idx]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 3.0), layout="constrained")

    # Train is context and test is the focus, so train takes the pale tint and test the full colour.
    for values, deviations, color, label in (
        (mean_train, std_train, FERTIMAP_TINT, "train"),
        (mean_test, std_test, PROJECT_COLORS["Al Moutmir"], "test"),
    ):
        ax.plot(
            param_values,
            values,
            color=color,
            marker="o",
            markersize=3.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.6,
            label=f"mean {label}",
            zorder=3,
        )
        if np.issubdtype(values.dtype, np.number):
            ax.fill_between(
                param_values,
                values - deviations,
                values + deviations,
                color=color,
                alpha=0.15,
                lw=0,
                zorder=1,
                label=f"±1 sd across folds ({label})",
            )

    # Where the search landed, marked on both curves.
    ax.scatter(
        [best_param, best_param],
        [mean_train[best_idx], mean_test[best_idx]],
        color=[FERTIMAP_TINT, PROJECT_COLORS["Al Moutmir"]],
        s=90,
        marker="*",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="best",
    )

    ylabel = scoring if not scoring.startswith("neg_") else scoring.replace("neg_", "")
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel.replace("_", " "))
    panel_subtitle(ax, f"best {param_name} = {best_param}")
    # Below the axes rather than inside it. Five entries at `loc="best"` covered the very part of
    # the sweep the plot is read for - where the train and test curves separate.
    fig.legend(loc="outside lower center", ncol=3, fontsize=7.5)
    return fig


@styled
def cv_parallel_coordinates(cv_results):
    # Copied before anything is written to it. The caller passes the same frame that is later saved
    # as cv/cv_results.csv, and taking the absolute value in place silently rewrote that file's
    # scores.
    cv_results = cv_results.copy()
    cv_results['mean_test_score'] = cv_results['mean_test_score'].abs()
    best_index = cv_results['mean_test_score'].idxmin()

    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    ynames = [col.rsplit('__', 1)[-1] for col in param_cols] + ['Mean Test Score']
    ys = cv_results[param_cols + ['mean_test_score']].values
    parallels = ys.shape[0]

    # Scaling
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    dys = np.where(dys == 0, 1, dys)
    ymins -= dys*0.02
    ymaxs += dys*0.02
    dys = ymaxs - ymins
    zs = np.zeros_like(ys)
    zs[:,0] = ys[:,0]
    zs[:,1:] = (ys[:,1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # Main axes
    fig, host = plt.subplots(figsize=(FIG_WIDTH_FULL, 3.6), layout="constrained")

    axes = [host] + [host.twinx() for _ in range(ys.shape[1]-1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_color(BASELINE)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i/(ys.shape[1]-1)))
            ax.tick_params(axis="y", length=3, color=BASELINE, labelsize=8)

    # Colormap
    norm = mcolors.Normalize(vmin=cv_results['mean_test_score'].min(), vmax=cv_results['mean_test_score'].max())
    cmap = sequential_cmap()
    # Draw other lines
    for j in range(parallels):
        verts = list(zip(np.linspace(0,len(ys[0])-1,len(ys[0])*3-2),
                             np.repeat(zs[j,:],3)[1:-1]))
        codes = [Path.MOVETO]+[Path.CURVE4]*(len(verts)-1)
        path = Path(verts, codes)
        if j!=best_index:
            patch = patches.PathPatch(path, facecolor='none', lw=0.5,
                                      edgecolor=cmap(norm(cv_results['mean_test_score'].iloc[j])))
        else:
            patch = patches.PathPatch(
                path, facecolor='none', lw=2.5, edgecolor=PROJECT_COLORS["Al Moutmir"], label='best trial'
            )
        host.add_patch(patch)
    legend_line = Line2D([0], [0], color=PROJECT_COLORS["Al Moutmir"], lw=2.5, label='best trial')
    host.legend(handles=[legend_line], loc="lower left", bbox_to_anchor=(1.0, -0.1), fontsize=7.5)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=host, anchor=(0.2, 0.5))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)

    best_params = cv_results.loc[best_index][param_cols].to_dict()
    best_params_str = ", ".join([f"{str(k).rsplit('__', 1)[-1]}={v}" for k, v in best_params.items()])

    host.set_xlim(0, ys.shape[1]-1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, rotation=45, ha='right', fontsize=8)
    host.tick_params(axis='x', which='major', pad=7)
    host.grid(True, which='major', axis='y')
    host.spines['top'].set_visible(True)
    host.spines['bottom'].set_visible(True)
    host.spines['top'].set_color(BASELINE)
    host.spines['bottom'].set_color(BASELINE)
    host.set_axisbelow(True)
    panel_subtitle(host, best_params_str)
    return fig


# Most bars a pred-vs-obs panel will draw. Past roughly this many, neighbouring bars are less than
# a pixel apart and merge into a solid grey curtain that hides the scatter underneath - which is
# worse than showing fewer bars, because it hides the very thing the panel is about. The metrics and
# the CSV always cover every point; only the picture is thinned.
MAX_ERROR_BARS = 200

# Fewer on the parent overlay. Its panels are half the width of a per-run panel and each one can
# carry several models' bars at once, so the density at which they stop being readable arrives
# sooner.
MAX_PARENT_ERROR_BARS = 120


def _error_bar_positions(x_values, cap=MAX_ERROR_BARS):
    """Row positions to draw bars for: every row, or an even spread across the x range.

    Evenly spaced through the x-SORTED order rather than a random sample, so the drawn bars span the
    whole range of the axis instead of clustering wherever the data is dense. Deterministic, so the
    same run always produces the same picture.
    """
    values = np.asarray(x_values, dtype=float)
    if values.size <= cap:
        return np.arange(values.size)
    order = np.argsort(values)
    return np.sort(order[np.linspace(0, values.size - 1, cap).astype(int)])


# How much of the interval extent the predicted-vs-observed panel is allowed to ignore, per end.
# The bars have to be reachable for their caps to be visible, but framing on their true extent is
# not an option: on a linear model the mean interval already exceeds the target's own range and the
# widest is several times it, so the scatter would collapse into a band. Clipping the outer 2% at
# each end puts most caps on-screen and leaves a handful running off the edge, which is the honest
# reading for a point the model is genuinely unsure about.
INTERVAL_CLIP_PERCENTILE = 2.0

# Hard ceiling on how far past the data the axis may stretch to accommodate bars, as a fraction of
# the data's own span, per end. The percentile clip alone is not enough: sigma has a long right
# tail, so on a real run the 98th percentile of the bar ends still sat about twice the target range
# away and squeezed the scatter into the middle fifth of the panel. This bounds the compression
# directly - the points always keep at least ~1/(1 + 2*0.35) of the axis.
MAX_INTERVAL_EXTENSION = 0.35

# The error bars and their caps.
#
# COLOURED, not grey, and that is the whole point. The bars are DATA; the grid behind them is
# chrome. In this palette chrome is the warm greys, so a grey bar is competing in the one register
# the reader has learned to ignore - and a pale grey vertical is not merely quiet, it is
# indistinguishable from a vertical gridline. A grey attempt measured out at luminance 0.866
# against a grid of 0.877: a difference of 0.011, which is no difference at all. Darkening the grey
# separates them tonally but still says "chrome" in a palette where every grey is chrome.
#
# Hue is what does the work here, so these sit on the project's accent instead, quietly: thin
# verticals at low alpha and caps only a little stronger. The points keep the slate sigma ramp, so
# bars and points stay distinct from each other as well as from the grid.
ERROR_BAR_LINE_COLOR = PROJECT_COLORS["Al Moutmir"]
ERROR_BAR_CAP_COLOR = PROJECT_COLORS["Al Moutmir"]
ERROR_BAR_LINE_ALPHA = 0.30
ERROR_BAR_CAP_ALPHA = 0.65
# The caps have to read as ends rather than as more line, which is a question of RELATIVE weight -
# a cap is found because it is heavier than the vertical it terminates, not because of any absolute
# width. Keep the cap wider than the line if these are ever retuned.
ERROR_BAR_LINE_WIDTH = 0.5
ERROR_BAR_CAP_WIDTH = 0.8
ERROR_BAR_CAP_SIZE = 2.0


def _square_limits(observed, predicted, interval=None, percentile=INTERVAL_CLIP_PERCENTILE, margin=0.05):
    """One ``(low, high)`` range for BOTH axes of a predicted-vs-observed panel.

    A pred-vs-obs scatter is only readable when the identity line is a true 45 degree diagonal, and
    that needs the two axes to share a range as well as an aspect - otherwise the cloud is stretched
    along whichever axis happens to span less.

    ``interval``, when given, widens the range toward the bar ends, but by a PERCENTILE rather than
    by their min and max. That distinction is the whole point of this function: a single very
    uncertain point has an interval several times the target's range, and letting it set the limits
    is exactly the blow-out that framing on the data alone was introduced to avoid.
    """
    candidates = [np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)]
    finite = np.concatenate([values[np.isfinite(values)] for values in candidates])
    if finite.size == 0:
        return 0.0, 1.0

    low, high = _extend_range(
        float(finite.min()), float(finite.max()), interval, percentile=percentile
    )
    pad = (high - low) * margin or 1.0
    return low - pad, high + pad


def _extend_range(low, high, interval, percentile=INTERVAL_CLIP_PERCENTILE):
    """Widen ``(low, high)`` toward the interval ends, under two independent limits.

    The percentile drops the few pathological bars. The extension ceiling handles the case the
    percentile cannot - a heavy-tailed sigma, where even the 98th percentile is far enough out to
    squash the data into the middle of the panel. Whichever binds first wins.
    """
    if interval is None:
        return low, high

    headroom = MAX_INTERVAL_EXTENSION * ((high - low) or 1.0)
    lower, upper = (np.asarray(bound, dtype=float) for bound in interval)
    if np.isfinite(lower).any():
        clipped = float(np.percentile(lower[np.isfinite(lower)], percentile))
        low = min(low, max(clipped, low - headroom))
    if np.isfinite(upper).any():
        clipped = float(np.percentile(upper[np.isfinite(upper)], 100.0 - percentile))
        high = max(high, min(clipped, high + headroom))
    return low, high


def _frame_square(ax, low, high):
    """Give an axes one shared range and a 1:1 aspect, so its diagonal is a real diagonal.

    ``adjustable="box"`` reshapes the axes box rather than the data limits, which is what keeps the
    range exactly as asked. Call this AFTER every plotting call on the panel, so nothing that
    autoscales on draw can overwrite the limits set here.
    """
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")


def _draw_error_bars(ax, x_values, y_values, lower, upper, positions):
    """Vertical prediction intervals with visible ends.

    The verticals and the caps are styled SEPARATELY, which a single ``alpha=`` on the errorbar call
    cannot do - it fades both by the same amount, and the setting that makes a few hundred
    overlapping verticals bearable is far too faint for the caps that mark where each interval
    actually stops. Faint accent lines, firmer accent caps.

    ``yerr`` takes the two half-widths rather than half of ``upper - lower``: a conformal interval is
    only symmetric when its calibrator is, and halving the width would bake in an assumption that
    need not hold.
    """
    _plotline, caplines, barlinecols = ax.errorbar(
        np.asarray(x_values)[positions],
        np.asarray(y_values)[positions],
        yerr=[
            np.maximum(np.asarray(y_values - lower)[positions], 0.0),
            np.maximum(np.asarray(upper - y_values)[positions], 0.0),
        ],
        fmt="none",
        ecolor=ERROR_BAR_LINE_COLOR,
        elinewidth=ERROR_BAR_LINE_WIDTH,
        capsize=ERROR_BAR_CAP_SIZE,
        zorder=1,
    )
    for bar in barlinecols:
        bar.set_alpha(ERROR_BAR_LINE_ALPHA)
    for cap in caplines:
        cap.set_alpha(ERROR_BAR_CAP_ALPHA)
        cap.set_markeredgewidth(ERROR_BAR_CAP_WIDTH)
        cap.set_color(ERROR_BAR_CAP_COLOR)
    return caplines, barlinecols


def _least_squares_line(x_values, y_values):
    """``(slope, intercept)`` of the OLS fit, or ``None`` when there is nothing to fit.

    Closed-form rather than ``np.polyfit``, which warns on a poorly-conditioned fit - and this suite
    turns warnings into errors, so a degenerate target would take the whole plot down.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return None
    variance = float(np.sum((x - x.mean()) ** 2))
    if variance <= 0.0:
        return None
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance)
    return slope, float(y.mean() - slope * x.mean())


@styled
def pred_obs_panel(eval_df, *, target_name=None):
    """One square panel: predictions against observations, with the 1:1 line and the metrics.

    This used to be three panels - the scatter, residuals against predicted, and a KDE of the same
    two variables. Both extras were dropped: the residual panel is the scatter rotated onto the
    identity line and says nothing the metric box does not, and the KDE redraws the first panel's
    data with the individual points - the thing a reader is looking for - smoothed away.

    The uncertainty columns are OPTIONAL. Most frames reaching this function come from runs with
    uncertainty disabled and must render exactly as they always have, so every addition below is
    guarded on the column being present rather than on a flag the caller would have to pass.

    Args:
        eval_df (DataFrame): must carry `target` and `prediction`; may carry `prediction_std`
            and the `prediction_lower`/`prediction_upper` pair, and may set `attrs["interval_label"]`.
        target_name (str, optional): what to caption the panel with; defaults to the target column's
            own name.

    Returns:
        matplotlib.figure.Figure: the caller saves and closes it.
    """
    y_test = eval_df["target"]
    y_pred = eval_df["prediction"]

    interval = interval_columns(eval_df)
    sigma = sigma_column(eval_df)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_COLUMN, FIG_WIDTH_COLUMN + 0.4), layout="constrained"
    )
    square_panel(ax)

    bar_positions = _error_bar_positions(y_test) if interval is not None else None
    if interval is not None:
        # Drawn FIRST and at zorder 1 so the bars sit under the points: bars painted on top hide the
        # very structure the plot exists to show.
        _draw_error_bars(ax, y_test, y_pred, interval[0], interval[1], bar_positions)

    if sigma is not None:
        # Colour by sigma. At this point count the bars overlap and stop being readable per point,
        # while the colour survives - so "where is this model uncertain?" stays answerable from the
        # picture rather than only from the CSV.
        scatter = ax.scatter(
            y_test, y_pred, c=sigma, cmap=sequential_cmap(), s=14,
            edgecolor="white", linewidth=0.3, zorder=4, rasterized=True,
        )
        # An INSET axes, not `fig.colorbar(..., ax=ax)`. The `ax=` form makes room for the colorbar
        # by shrinking the axes it is given, which fights the 1:1 aspect set below. inset_axes
        # positions in axes-fraction coordinates and leaves the box alone.
        colorbar = fig.colorbar(scatter, cax=ax.inset_axes([1.03, 0.0, 0.035, 1.0]))
        colorbar.set_label("predictive σ", fontsize=8, color=INK_2)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=7.5, labelcolor=INK_2)
    else:
        # Slate when there are bars to stay clear of, the accent when the points are the only thing
        # on the panel. The bars carry the accent colour, so orange points beside orange bars would
        # reintroduce - between data and data this time - exactly the ambiguity the accent was
        # chosen to remove. A frame with an interval but no sigma is unusual, since the interval is
        # built from sigma, but it costs one branch to not draw it wrong.
        ax.scatter(
            y_test, y_pred,
            color=CARBONATE_RAMP[-2] if interval is not None else PROJECT_COLORS["Al Moutmir"],
            s=14, edgecolor="white", linewidth=0.3, zorder=4, rasterized=True,
        )

    # One shared range for both axes, so the identity line below is a true 45 degree diagonal and
    # the cloud is not stretched along whichever axis spans less.
    low, high = _square_limits(y_test, y_pred, interval)

    # Identity line across the WHOLE panel rather than the observed range, so it runs corner to
    # corner instead of stopping short inside a wider frame.
    ax.plot([low, high], [low, high], linestyle="--", color=INK_2, lw=0.8, zorder=5)

    # The OLS fit, solid and darker against the dashed identity. The gap between the two is the
    # regression to the mean every soil model shows, and it only reads when both are on the panel -
    # so they are distinguished by weight and dash rather than by a second colour, which would
    # compete with the sigma ramp the points are coloured on.
    fit = _least_squares_line(y_test, y_pred)
    if fit is not None:
        slope, intercept = fit
        ax.plot(
            [low, high], [slope * low + intercept, slope * high + intercept],
            color=INK, lw=1.4, zorder=5,
        )

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    panel_subtitle(ax, str(target_name or y_test.name or ""))
    # After every plotting call, so nothing that autoscales on draw overwrites these limits.
    _frame_square(ax, low, high)

    rmse = root_mean_squared_error(y_test, y_pred)
    annotation = f"RMSE = {rmse:.2f}\nR² = {r2_score(y_test, y_pred):.2f}"
    if rmse > 0.0:
        # Skipped at rmse == 0 rather than divided anyway. RPIQ and RPD are ratios with rmse in the
        # denominator, so a model that fits its test split exactly - a degenerate estimator, or a
        # target that leaked into the features - makes them infinite. This is the same policy
        # regression_metrics applies; without it the annotation raises a divide-by-zero and takes
        # the whole plot, and the artifact logging around it, down with it.
        #
        # (predictions, targets), in that order: the IQR in the numerator is read off the
        # SECOND argument. Passing (y_test, y_pred) measures the spread of the predictions,
        # which under-reports RPIQ because predictions are systematically under-dispersed.
        annotation += f"\nRPIQ = {rpiq_score(y_pred, y_test):.2f}"
    if interval is not None:
        lower, upper = interval
        covered = float(np.mean((y_test >= lower) & (y_test <= upper)))
        # The coverage is annotated beside the bars ON PURPOSE. A prediction interval drawn without
        # the fraction it actually captured is a decoration; printed together, the picture states a
        # claim and the number checks it.
        # Computed over EVERY point even when only a subset is drawn, so the number never describes
        # a different population from the metric of the same name in the run.
        annotation += (
            f"\nPICP = {covered:.3f}"
            f"\nMPIW = {float(np.mean(upper - lower)):.2f}"
        )
        # What KIND of bar this is, when the caller said. The same picture means different things
        # under conformal, gaussian and sigma - a reader cannot tell them apart by looking, and the
        # PICP beside it is only interpretable once you know which claim is being made. Carried on
        # the frame's `attrs` so the caller need not widen this signature.
        label = eval_df.attrs.get("interval_label") if hasattr(eval_df, "attrs") else None
        if label:
            annotation += f"\nbar: {label}"
        if bar_positions is not None and len(bar_positions) < len(y_test):
            annotation += f"\nbars: {len(bar_positions)} of {len(y_test)}"
    metric_box(ax, annotation)

    # No reliability panel here on purpose. Grading the interval needs the conformal calibrator, and
    # this function is handed a frame and nothing else. A version that swept Gaussian z-multiples of
    # the RAW sigma instead produced a curve far below the diagonal sitting next to an annotation
    # reporting PICP=0.96, because those two grade different things. The reliability curve lives in
    # uncertainty/reliability.png, which is written by log_uncertainty_artifacts and does have the
    # calibrator.
    return fig


def create_parent_pred_obs(eval_dfs):
    return _create_parent_pred_obs_multitarget(eval_dfs)


# Metric column names as they should appear on an axis. Anything not listed falls through to a
# readable default, so a new metric never renders as a raw column name in SHOUTING CAPS.
_METRIC_LABELS = {
    "rmse_test": "RMSE (test)",
    "rmse_train": "RMSE (train)",
    "r2_test": "R² (test)",
    "r2_train": "R² (train)",
    "rpiq_test": "RPIQ (test)",
    "mae_test": "MAE (test)",
}


def _metric_label(metric: str) -> str:
    if metric in _METRIC_LABELS:
        return _METRIC_LABELS[metric]
    stem, _, split = str(metric).rpartition("_")
    if stem and split in {"test", "train", "val"}:
        return f"{stem.replace('_', ' ').upper()} ({split})"
    return str(metric).replace("_", " ")


@styled
def plot_leaderboard_scatter(leaderboard_df, metric_x="rmse_test", metric_y="r2_test",
                                        label_col="model", hue_col="target"):
    """
    Create a scatter subplot for each target showing model performance,
    with average RMSE and R² lines per target.
    """
    if leaderboard_df is None or leaderboard_df.empty:
        return message_figure("No leaderboard rows available")

    if metric_x not in leaderboard_df.columns or metric_y not in leaderboard_df.columns:
        return message_figure(
            f"Skipping leaderboard scatter: missing metric columns '{metric_x}' or '{metric_y}'",
            figsize=(FIG_WIDTH_FULL, 1.6),
        )

    # Fallback if expected hue column missing
    if hue_col not in leaderboard_df.columns:
        # Try common alternate names
        if 'target_name' in leaderboard_df.columns:
            hue_col = 'target_name'
        else:
            # Create a pseudo target column
            hue_col = '_target_tmp_'
            leaderboard_df = leaderboard_df.copy()
            leaderboard_df[hue_col] = 'All'

    if label_col not in leaderboard_df.columns:
        # Try alternative naming
        if 'model_name' in leaderboard_df.columns:
            label_col = 'model_name'
        else:
            label_col = '_model_tmp_'
            leaderboard_df = leaderboard_df.copy()
            leaderboard_df[label_col] = range(len(leaderboard_df))

    targets = leaderboard_df[hue_col].dropna().unique()
    n_targets = max(1, len(targets))
    # Columns from the number of TARGETS, which is what a panel shows. Deriving them from the number
    # of models built a grid with as many columns as the run had models and then deleted most of it.
    n_cols = min(2, n_targets)
    n_rows = int(np.ceil(n_targets / n_cols)) or 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH_FULL if n_cols > 1 else FIG_WIDTH_COLUMN, 2.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.flatten()

    for index, (ax, target) in enumerate(zip(axes, targets)):
        df_target = leaderboard_df[leaderboard_df[hue_col] == target]

        # Compute averages for this target
        avg_rmse = df_target[metric_x].mean()
        avg_r2 = df_target[metric_y].mean()

        ax.scatter(
            df_target[metric_x],
            df_target[metric_y],
            s=28,
            color=PROJECT_COLORS["Al Moutmir"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

        # Annotate points
        for _, row in df_target.iterrows():
            ax.text(row[metric_x], row[metric_y], f"  {row[label_col]}",
                    horizontalalignment='left', verticalalignment='center',
                    fontsize=7, color=INK_2)

        # Add average lines per target
        ax.axvline(avg_rmse, color=BASELINE, linestyle="--", lw=0.8, zorder=1, label="mean RMSE")
        ax.axhline(avg_r2, color=INK_2, linestyle="--", lw=0.8, zorder=1, label="mean R²")

        panel_letter(ax, "abcdefghijklmnopqrstuvwxyz"[index % 26])
        panel_subtitle(ax, str(target))
        ax.set_xlabel(_metric_label(metric_x))
        ax.set_ylabel(_metric_label(metric_y))
        # Room on the right for the model names, which are written to the RIGHT of their points and
        # otherwise run off the panel - matplotlib autoscales to the points, not to their labels.
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + (right - left) * 0.28)
        if index == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    # Remove empty subplots
    for j in range(len(targets), len(axes)):
        fig.delaxes(axes[j])

    return fig
