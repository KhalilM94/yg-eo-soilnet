"""The shared style, and the one promise it has to keep: it does not leak."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from yg_eo_soilnet.plot_style import (  # noqa: E402
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
        try:
            assert axis.spines["left"].get_visible() is False  # the house default
            # The house grid is y-only, so no x gridline is drawn yet.
            assert not any(line.get_visible() for line in axis.xaxis.get_gridlines())

            square_panel(axis)

            assert axis.spines["left"].get_visible() is True
            assert matplotlib.colors.to_hex(axis.spines["left"].get_edgecolor()) == BASELINE
            assert all(line.get_visible() for line in axis.xaxis.get_gridlines())
            assert all(line.get_visible() for line in axis.yaxis.get_gridlines())
        finally:
            plt.close(figure)


def test_the_letter_and_the_caption_occupy_different_title_slots():
    """Both have to fit on one line, which is why the letter is not glued into the caption string."""
    figure, axis = plt.subplots()
    try:
        panel_letter(axis, "a")
        panel_subtitle(axis, "clay_pct")
        assert axis.get_title(loc="left") == "a"
        assert axis.get_title(loc="right") == "clay_pct"
        # The letter is the loud one; the caption is secondary ink at a smaller size.
        letter, caption = axis._left_title, axis._right_title
        assert matplotlib.colors.to_hex(letter.get_color()) == INK
        assert letter.get_fontweight() == "bold"
        assert caption.get_fontsize() < letter.get_fontsize()
    finally:
        plt.close(figure)


def test_the_metric_box_is_framed_and_inside_the_axes():
    figure, axis = plt.subplots()
    try:
        text = metric_box(axis, "RMSE = 1.00")
        assert text.get_transform() is axis.transAxes
        assert text.get_bbox_patch() is not None
    finally:
        plt.close(figure)


def test_an_empty_figure_carries_a_message_rather_than_being_none():
    """The repo's empty-data convention, so no caller has to branch on the return value."""
    figure = message_figure("No leaderboard rows available")
    try:
        assert figure is not None
        assert figure.axes[0].texts[0].get_text() == "No leaderboard rows available"
        assert figure.axes[0].axison is False
    finally:
        plt.close(figure)
