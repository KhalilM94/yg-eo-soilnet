"""The plots: the house style, pred-vs-obs panels, the parent overlay and the uncertainty figures."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.support.builders import eval_frame
from yg_eo_soilnet.plot_style import (
    BASELINE,
    GRID,
    INK,
    RC_PARAMS,
    message_figure,
    metric_box,
    panel_letter,
    panel_subtitle,
    square_panel,
    style_context,
    styled,
)
from yg_eo_soilnet.plot_utils import (
    create_parent_pred_obs,
    pred_obs_panel,
)
from yg_eo_soilnet.uncertainty.plots import reliability_curve, sigma_vs_error

# --- the plots -------------------------------------------------------------


def test_the_pred_obs_plot_still_renders_without_any_uncertainty_columns():
    # The majority case: every frame from a run with uncertainty disabled.

    figure = pred_obs_panel(eval_frame(with_uncertainty=False))
    assert figure is not None
    assert figure.axes[0].collections, "nothing was drawn on the panel"


def test_the_pred_obs_plot_renders_with_uncertainty_columns():
    figure = pred_obs_panel(eval_frame())
    # The colorbar is an INSET of the panel, not a second entry in figure.axes - that is what
    # keeps the panel's width and its 1:1 aspect intact.
    assert figure.axes[0].child_axes, "no sigma colorbar on a frame that carries sigma"


def test_the_pred_obs_plot_is_one_panel_with_no_residual_or_density_companion():
    """The residual scatter and the KDE were dropped; the square panel is the whole figure.

    Both extras restated the first panel: the residuals are that scatter rotated onto the identity
    line, and the KDE redrew the same two variables with the individual points - the thing a reader
    is looking for - smoothed away.
    """

    for frame in (eval_frame(with_uncertainty=False), eval_frame()):
        figure = pred_obs_panel(frame)
        assert len(figure.axes) == 1


def test_the_pred_obs_panel_is_captioned_with_the_target_it_was_given():
    figure = pred_obs_panel(eval_frame(with_uncertainty=False), target_name="clay_pct")
    assert figure.axes[0].get_title(loc="right") == "clay_pct"


def test_the_pred_obs_panel_shares_one_range_across_both_axes():
    """A pred-vs-obs scatter only reads correctly when its identity line is a true diagonal."""
    from yg_eo_soilnet.plot_utils import _square_limits

    low, high = _square_limits(np.array([0.0, 100.0]), np.array([40.0, 60.0]))
    assert low < 0.0 and high > 100.0


def test_the_square_range_reaches_the_bar_ends():
    from yg_eo_soilnet.plot_utils import _square_limits

    observed = np.array([0.0, 100.0])
    predicted = np.array([20.0, 80.0])
    interval = (np.array([-20.0, -10.0]), np.array([110.0, 120.0]))

    plain_low, plain_high = _square_limits(observed, predicted)
    low, high = _square_limits(observed, predicted, interval)
    assert low < plain_low and high > plain_high


def test_a_heavy_tailed_sigma_cannot_squash_the_scatter():
    """The percentile clip alone is not enough when the far tail is still very far out.

    Real case: with sigma right-skewed, the 98th percentile of the bar ends sat about twice the
    target range beyond the data and left the points in the middle fifth of the panel.
    """
    from yg_eo_soilnet.plot_utils import MAX_INTERVAL_EXTENSION, _square_limits

    observed = np.linspace(0.0, 100.0, 500)
    predicted = observed.copy()
    # Every bar end is far out, so no percentile can drop them - only the extension ceiling can.
    interval = (np.full(500, -500.0), np.full(500, 600.0))

    low, high = _square_limits(observed, predicted, interval)
    data_span = 100.0
    assert low >= -(MAX_INTERVAL_EXTENSION * data_span) * 1.1 - 10.0
    assert high <= 100.0 + (MAX_INTERVAL_EXTENSION * data_span) * 1.1 + 10.0
    # The points still occupy a usable share of the panel.
    assert data_span / (high - low) > 0.5


def test_one_enormous_bar_does_not_decide_the_whole_axis():
    # The 790-wide case. min/max framing would put the axis at roughly -400..400 and collapse the
    # scatter into a band; the percentile clip ignores the outer 2% at each end.
    from yg_eo_soilnet.plot_utils import _square_limits

    observed = np.linspace(0.0, 100.0, 200)
    predicted = observed.copy()
    lower = np.full(200, -10.0)
    upper = np.full(200, 110.0)
    lower[0], upper[0] = -400.0, 400.0

    low, high = _square_limits(observed, predicted, (lower, upper))
    assert low > -100.0 and high < 200.0


def test_the_square_range_survives_a_frame_with_no_interval():
    from yg_eo_soilnet.plot_utils import _square_limits

    low, high = _square_limits(np.array([1.0, 9.0]), np.array([2.0, 8.0]), None)
    assert low < 1.0 and high > 9.0


def test_frame_square_leaves_the_panel_at_a_one_to_one_aspect():
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.plot_utils import _frame_square

    figure, axis = plt.subplots()
    _frame_square(axis, -5.0, 105.0)
    assert axis.get_aspect() == 1.0
    assert axis.get_xlim() == axis.get_ylim() == (-5.0, 105.0)


def test_the_error_bars_have_ends_you_can_actually_see():
    """Caps and verticals are styled separately; one alpha on the container cannot do both."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.plot_utils import (
        ERROR_BAR_CAP_COLOR,
        ERROR_BAR_LINE_WIDTH,
        _draw_error_bars,
    )

    figure, axis = plt.subplots()
    y = np.array([1.0, 2.0, 3.0])
    caplines, barlinecols = _draw_error_bars(axis, np.arange(3.0), y, y - 1.0, y + 1.0, np.arange(3))
    assert caplines, "no caps drawn, so the interval ends are invisible"
    expected = mcolors.to_rgba(ERROR_BAR_CAP_COLOR)[:3]
    for cap in caplines:
        assert mcolors.to_rgba(cap.get_color())[:3] == expected
        # Heavier than the vertical it terminates. The absolute width is a free parameter; the
        # RATIO is what makes a cap read as an end rather than as more line.
        assert cap.get_markeredgewidth() > ERROR_BAR_LINE_WIDTH
    # The ends have to stand out from the verticals, not fade with them.
    assert caplines[0].get_alpha() > barlinecols[0].get_alpha()


def test_the_bars_are_not_drawn_in_a_colour_the_grid_also_uses():
    """The bars are data and the grid is chrome, so they must not share a register.

    A pale grey bar is not merely quiet against this palette's warm-grey grid - it is
    indistinguishable from a vertical gridline. Measured, the grey the bars once used sat at
    luminance 0.866 against a grid at 0.877.
    """
    import matplotlib.colors as mcolors

    from yg_eo_soilnet.plot_style import BASELINE, GRID
    from yg_eo_soilnet.plot_utils import ERROR_BAR_CAP_COLOR, ERROR_BAR_LINE_COLOR

    def saturation(color):
        red, green, blue = mcolors.to_rgb(color)
        return max(red, green, blue) - min(red, green, blue)

    chrome = max(saturation(GRID), saturation(BASELINE))
    for bar_color in (ERROR_BAR_LINE_COLOR, ERROR_BAR_CAP_COLOR):
        assert bar_color not in (GRID, BASELINE)
        # Carries real hue, which is what separates it from chrome at any lightness.
        assert saturation(bar_color) > chrome * 3


def test_the_bars_are_thinned_but_the_metrics_still_cover_every_point():
    from yg_eo_soilnet.plot_utils import MAX_ERROR_BARS, _error_bar_positions

    positions = _error_bar_positions(np.arange(1000.0))
    assert len(positions) == MAX_ERROR_BARS
    # Spread across the whole range rather than clustered where the data is dense.
    assert positions[0] < 10 and positions[-1] > 990
    # Deterministic: the same run always draws the same picture.
    assert np.array_equal(positions, _error_bar_positions(np.arange(1000.0)))


def test_a_small_split_keeps_a_bar_on_every_point():
    from yg_eo_soilnet.plot_utils import _error_bar_positions

    assert len(_error_bar_positions(np.arange(50.0))) == 50


def test_the_plot_opens_exactly_one_figure_for_its_caller_to_close():
    # Called once per target inside a training loop, so a leak here accumulates until matplotlib
    # warns - and with warnings as errors that fails a run several targets later. The panel returns
    # its figure now instead of saving it, so closing is log_figure's job; what this checks is that
    # nothing EXTRA is left behind.
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    for frame in (eval_frame(with_uncertainty=False), eval_frame()):
        figure = pred_obs_panel(frame)
        assert set(plt.get_fignums()) - before == {figure.number}
        plt.close(figure)
    assert set(plt.get_fignums()) == before


def test_the_plot_survives_a_model_that_fits_its_test_split_exactly():
    # RPIQ and RPD divide by rmse. A degenerate estimator makes that a divide-by-zero, which used to
    # take the plot - and the artifact logging around it - down with it.

    exact = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0], "prediction": [1.0, 2.0, 3.0, 4.0]})
    figure = pred_obs_panel(exact)
    assert figure is not None


def test_the_parent_overlay_renders_with_and_without_intervals():
    frame = eval_frame()
    frame["target_name"] = "clay_pct"
    frame["clay_pct"] = frame["target"]
    frame["prediction_clay_pct"] = frame["prediction"]
    frame["prediction_lower_clay_pct"] = frame["prediction_lower"]
    frame["prediction_upper_clay_pct"] = frame["prediction_upper"]
    frame["model_name"] = "Ridge"

    assert create_parent_pred_obs([frame]) is not None

    bare = eval_frame(with_uncertainty=False)
    bare["target_name"] = "clay_pct"
    bare["clay_pct"] = bare["target"]
    bare["prediction_clay_pct"] = bare["prediction"]
    bare["model_name"] = "Ridge"
    assert create_parent_pred_obs([bare]) is not None


def test_the_reliability_curve_returns_a_figure_the_caller_saves():
    frame = eval_frame(200)

    figure = reliability_curve(frame["target"], frame["prediction"], frame["prediction_std"], target_name="clay_pct")
    assert figure is not None
    assert figure.axes[0].get_xlabel() == "Nominal coverage"


def test_the_reliability_curve_draws_into_a_supplied_axis_and_returns_none():
    import matplotlib.pyplot as plt

    frame = eval_frame(200)
    figure, axis = plt.subplots()
    assert reliability_curve(frame["target"], frame["prediction"], frame["prediction_std"], axis=axis) is None


def test_sigma_vs_error_bins_by_equal_count_and_returns_a_figure():
    frame = eval_frame(200)

    figure = sigma_vs_error(frame["target"], frame["prediction"], frame["prediction_std"], target_name="clay_pct")
    assert figure is not None
    assert figure.axes[0].get_xlabel().startswith("Predicted")


def test_sigma_vs_error_degrades_gracefully_on_a_split_too_small_to_bin():
    figure = sigma_vs_error(np.zeros(5), np.zeros(5), np.ones(5))
    assert figure is not None


# --- the parent overlay: one frame per target, on its own panel ------------


def _target_frame(target, observed, predicted, first_column=None):
    """A per-target eval frame shaped like ParentRunLogger._collect_eval_dfs yields.

    `first_column` stands in for the feature that happens to come first in the frame - the column
    the old positional fallback would have seized on.
    """
    frame = pd.DataFrame()
    if first_column is not None:
        frame["landform_class"] = first_column
    frame[target] = observed
    frame["prediction"] = predicted
    frame["target_name"] = target
    frame["model_name"] = "TabICL"
    return frame


def test_a_categorical_first_column_no_longer_crashes_the_parent_overlay():
    """The reported failure: run 13a8925b, five per-target children, landform_class first.

    np.isfinite on 'upper_slope_flat' raised a TypeError from inside the ufunc and took the whole
    run down after every model had already trained.
    """
    frames = [
        _target_frame("clay_pct", [10.0, 20.0], [11.0, 21.0], ["upper_slope_flat", "lower_slope"]),
        _target_frame("ph_water", [7.0, 8.0], [7.1, 8.1], ["upper_slope_flat", "lower_slope"]),
    ]
    figure = create_parent_pred_obs(frames)
    assert figure is not None


def _panel_x(figure, title):
    # The target name lives in the RIGHT title slot: the house style reserves the left slot for the
    # bold panel letter, so a panel carries both without either crowding the other.
    import numpy as np

    for axis in figure.axes:
        if axis.get_title(loc="right") == title:
            chunks = [c.get_offsets()[:, 0] for c in axis.collections if len(c.get_offsets())]
            return np.concatenate(chunks) if chunks else np.array([])
    raise AssertionError(f"no panel titled {title!r}")


def test_a_frame_for_one_target_stays_off_another_targets_panel():
    """The root cause, asserted on the DRAWN data rather than on not raising.

    The two targets are given disjoint ranges, so a panel that borrowed the other frame's rows
    would show it immediately.
    """

    frames = [
        _target_frame("clay_pct", [10.0, 12.0], [10.5, 12.5]),
        _target_frame("ph_water", [900.0, 950.0], [905.0, 955.0]),
    ]
    figure = create_parent_pred_obs(frames)
    clay = _panel_x(figure, "clay_pct")
    ph = _panel_x(figure, "ph_water")
    assert len(clay) == 2 and clay.max() < 100
    assert len(ph) == 2 and ph.min() > 100


def test_the_silent_case_is_fixed_too_not_just_the_crash():
    """An all-numeric first column never raised - it plotted a reflectance band as Observed.

    Run 7d09124 did exactly this and produced a pred_error_plot nobody could tell was wrong, which
    is why the fix is a target check and not a try/except.
    """

    frames = [
        _target_frame("clay_pct", [10.0, 12.0], [10.5, 12.5], [0.031, 0.032]),
        _target_frame("ph_water", [7.0, 8.0], [7.1, 8.1], [0.041, 0.042]),
    ]
    figure = create_parent_pred_obs(frames)
    clay = _panel_x(figure, "clay_pct")
    # Two points, not four, and none of them a reflectance value near 0.03.
    assert len(clay) == 2
    assert clay.min() > 1.0


def test_a_frame_without_a_target_name_still_plots():
    """The legacy single-target path, which the target check must not break."""

    frame = pd.DataFrame({"target": [1.0, 2.0], "prediction": [1.1, 2.1], "model_name": "Ridge"})
    figure = create_parent_pred_obs([frame])
    assert figure is not None


def test_a_frame_with_no_numeric_column_is_skipped_rather_than_guessed_at():
    frame = pd.DataFrame({"landform_class": ["a", "b"], "prediction": [1.0, 2.0], "model_name": "Ridge"})
    figure = create_parent_pred_obs([frame])
    assert figure is not None  # renders an empty panel rather than inventing an x axis


def test_numbers_stored_as_strings_still_plot():
    """Coercion rather than trust: a numeric-looking object column is usable, not fatal."""

    frame = _target_frame("clay_pct", ["10.0", "12.0"], [10.5, 12.5])
    figure = create_parent_pred_obs([frame])
    assert len(_panel_x(figure, "clay_pct")) == 2


# --- the house style --------------------------------------------------------------------------
# The shared style, and the one promise it has to keep: it does not leak.


def test_the_style_is_a_context_not_a_global_mutation():
    """These modules get imported inside notebooks and inside a warnings-as-errors test suite.

    A module-level rcParams.update would silently restyle whatever else the process is drawing, so
    the style is applied through rc_context and every parameter must be back afterwards.
    """
    before = {key: plt.rcParams[key] for key in RC_PARAMS}

    @styled
    def draw():
        # Every parameter is in force INSIDE.
        assert plt.rcParams["axes.spines.left"] is False
        assert plt.rcParams["font.size"] == 9
        return plt.rcParams["grid.color"]

    assert matplotlib.colors.to_hex(draw()) == GRID
    assert {key: plt.rcParams[key] for key in RC_PARAMS} == before


def test_the_style_survives_an_exception_in_the_plotting_function():
    before = {key: plt.rcParams[key] for key in RC_PARAMS}

    @styled
    def explode():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        explode()
    assert {key: plt.rcParams[key] for key in RC_PARAMS} == before


def test_styled_keeps_the_wrapped_function_name():
    # _log_plots names the artifact after plot_func.__name__, so a decorator that replaced it would
    # rename every CV figure to "wrapper.png".
    @styled
    def cv_val_curve():
        return None

    assert cv_val_curve.__name__ == "cv_val_curve"


def test_a_square_panel_gets_its_left_edge_and_both_grids_back():
    """The documented exception to the y-only house grid.

    A pred-vs-obs panel is read against its 1:1 line, and reading a point against that line needs
    gridlines running both ways plus a left edge to anchor them.
    """
    with style_context():
        figure, axis = plt.subplots()
        assert axis.spines["left"].get_visible() is False  # the house default
        # The house grid is y-only, so no x gridline is drawn yet.
        assert not any(line.get_visible() for line in axis.xaxis.get_gridlines())

        square_panel(axis)

        assert axis.spines["left"].get_visible() is True
        assert matplotlib.colors.to_hex(axis.spines["left"].get_edgecolor()) == BASELINE
        assert all(line.get_visible() for line in axis.xaxis.get_gridlines())
        assert all(line.get_visible() for line in axis.yaxis.get_gridlines())


def test_the_letter_and_the_caption_occupy_different_title_slots():
    """Both have to fit on one line, which is why the letter is not glued into the caption string."""
    figure, axis = plt.subplots()
    panel_letter(axis, "a")
    panel_subtitle(axis, "clay_pct")
    assert axis.get_title(loc="left") == "a"
    assert axis.get_title(loc="right") == "clay_pct"
    # The letter is the loud one; the caption is secondary ink at a smaller size.
    letter, caption = axis._left_title, axis._right_title
    assert matplotlib.colors.to_hex(letter.get_color()) == INK
    assert letter.get_fontweight() == "bold"
    assert caption.get_fontsize() < letter.get_fontsize()


def test_the_metric_box_is_framed_and_inside_the_axes():
    figure, axis = plt.subplots()
    text = metric_box(axis, "RMSE = 1.00")
    assert text.get_transform() is axis.transAxes
    assert text.get_bbox_patch() is not None


def test_an_empty_figure_carries_a_message_rather_than_being_none():
    """The repo's empty-data convention, so no caller has to branch on the return value."""
    figure = message_figure("No leaderboard rows available")
    assert figure is not None
    assert figure.axes[0].texts[0].get_text() == "No leaderboard rows available"
    assert figure.axes[0].axison is False
