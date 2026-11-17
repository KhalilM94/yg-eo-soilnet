"""One model pair everywhere: Global against the local model on its own inputs, and table S6.

The transfer rows every figure reads compare Global with ``local_global`` (no predicted CaCO3/CEC on
either side); the ``global_aux`` / ``local`` pair stays as extra rows. The debiased RMSE,
sqrt(RMSE² - bias²), gets a block-bootstrap interval like every other score, and table S6 sets it
beside the transferred RMSE, the bias and the local RMSE. Also the layout fixes of figures 5, 10,
U1 and U3.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import compare, evaluation
from yg_eo_soilnet.a1.evaluation import (
    INPUTS_EVERY,
    INPUTS_GROUP,
    INPUTS_WITH_PREDICTED,
    INTERVAL_SCORES,
    _scores,
    basin_error_table,
    block_bootstrap,
    main_transfer,
    transfer_table,
)
from yg_eo_soilnet.a1.figures import basins, uncertainty
from yg_eo_soilnet.a1.figures.core import figure_caption
from yg_eo_soilnet.a1.figures.style import WIDTH_2COL
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED
from yg_eo_soilnet.a1.runner import PREDICTION_COLUMNS
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL, LOCAL_GLOBAL, LOCAL_OF
from yg_eo_soilnet.datamodules.splitting import TEST


def _points(n=240, seed=0):
    """``n`` points over about 60 x 60 km, with a measured SOM value (as in test_a1_evaluation)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "point_id": [f"p{i:04d}" for i in range(n)],
            "lat": 32.0 + rng.uniform(0, 0.55, n),
            "lon": -7.0 + rng.uniform(0, 0.65, n),
            "observed": rng.gamma(4.0, 5.0, n),
        }
    )


def _rows(points, predicted, *, model, plan_id="random", kind="random", **extra):
    """Prediction rows in the runner's layout, for one model on one plan's test points."""
    frame = pd.DataFrame(
        {
            "plan_id": plan_id,
            "kind": kind,
            "held_out": None,
            "fraction": np.nan,
            "repeat": np.nan,
            "point_id": points["point_id"].to_numpy(),
            "split": TEST,
            "region": "R",
            "lat": points["lat"].to_numpy(),
            "lon": points["lon"].to_numpy(),
            "carbonate_class_measured": "<2",
            "carbonate_class_predicted": "<2",
            "model": model,
            "covariate_group": "combined",
            "observed": points["observed"].to_numpy(),
            "predicted": np.asarray(predicted, dtype=float),
            "fallback": False,
            "model_role": "model",
            "tuned_params": "{}",
            "seed": 42,
            "device_stage1": "cpu",
            "device_stage2": "cpu",
        },
        columns=list(PREDICTION_COLUMNS),
    )
    for column, value in extra.items():
        frame[column] = value
    return frame.astype({"fraction": float, "repeat": float})


def _logo(points, region, offsets: dict, groups=("combined",)):
    """Rows of one logo plan: each model predicts the measurement plus its own offset."""
    rows = []
    for group in groups:
        for model, offset in offsets.items():
            if group != "combined" and model in (GLOBAL, LOCAL_GLOBAL):
                continue  # only global_aux and local are fitted per group
            rows.append(
                _rows(
                    points,
                    points["observed"] + offset,
                    model=model,
                    plan_id=f"logo_{region}",
                    kind=LOGO,
                    held_out=region,
                    covariate_group=group,
                )
            )
    return rows


# --- the transfer table ---------------------------------------------------------------------------


def test_the_main_rows_compare_global_with_the_local_model_on_its_inputs():
    points = _points(40)
    offsets = {GLOBAL: 3.0, LOCAL_GLOBAL: 1.0, GLOBAL_AUX: 5.0, LOCAL: 2.0}
    table = transfer_table(pd.concat(_logo(points, "A", offsets, groups=("combined", "terrain"))))
    rows = table.set_index(["inputs", "covariate_group"])
    main = rows.loc[(INPUTS_EVERY, "combined")]
    assert (main["transferred_model"], main["local_model"]) == (GLOBAL, LOCAL_GLOBAL)
    assert main["rmse_transferred"] == pytest.approx(3.0) and main["rmse_local"] == pytest.approx(1.0)
    assert main["rmse_drop"] == pytest.approx(2.0)
    extra = rows.loc[(INPUTS_WITH_PREDICTED, "combined")]
    assert (extra["transferred_model"], extra["local_model"]) == (GLOBAL_AUX, LOCAL)
    assert extra["rmse_drop"] == pytest.approx(3.0)
    group = rows.loc[(INPUTS_GROUP, "terrain")]
    assert (group["transferred_model"], group["local_model"]) == (GLOBAL_AUX, LOCAL)
    assert len(table) == 3


def test_main_transfer_leaves_out_the_pair_with_predicted_caco3_and_cec():
    points = _points(30)
    offsets = {GLOBAL: 3.0, LOCAL_GLOBAL: 1.0, GLOBAL_AUX: 5.0, LOCAL: 2.0}
    main = main_transfer(transfer_table(pd.concat(_logo(points, "A", offsets, groups=("combined", "terrain")))))
    assert set(main["inputs"]) == {INPUTS_EVERY, INPUTS_GROUP}
    assert not main.duplicated(["kind", "region", "covariate_group"]).any()  # one row per region and group


def test_without_local_global_the_main_row_has_empty_local_columns_not_the_other_pair():
    points = _points(30)
    table = transfer_table(pd.concat(_logo(points, "A", {GLOBAL: 3.0, GLOBAL_AUX: 5.0, LOCAL: 2.0})))
    main = table[table["inputs"] == INPUTS_EVERY].iloc[0]
    assert not main["has_local"] and np.isnan(main["rmse_local"]) and np.isnan(main["rmse_drop"])
    assert main["rmse_transferred"] == pytest.approx(3.0)
    extra = table[table["inputs"] == INPUTS_WITH_PREDICTED].iloc[0]
    assert extra["has_local"] and extra["rmse_local"] == pytest.approx(2.0)


def test_mixed_main_rows_compare_global_with_the_control_plans_global():
    points = _points(30)
    rows = []
    for kind, offsets in ((MIXED, {GLOBAL: 1.0, GLOBAL_AUX: 2.0}), (CONTROL, {GLOBAL: 4.0, GLOBAL_AUX: 8.0})):
        for model, offset in offsets.items():
            rows.append(
                _rows(
                    points,
                    points["observed"] + offset,
                    model=model,
                    plan_id=f"{kind}_A_p30_s0",
                    kind=kind,
                    held_out="A",
                    fraction=0.3,
                    repeat=0.0,
                )
            )
    table = transfer_table(pd.concat(rows)).set_index("inputs")
    assert table.loc[INPUTS_EVERY, "rmse_drop"] == pytest.approx(-3.0)  # Global mixed 1 against Global control 4
    assert table.loc[INPUTS_WITH_PREDICTED, "rmse_drop"] == pytest.approx(-6.0)


def test_the_local_reference_of_each_transferred_model_reads_the_same_inputs():
    assert LOCAL_OF == {GLOBAL: LOCAL_GLOBAL, GLOBAL_AUX: LOCAL}
    assert compare.local_of(GLOBAL) == LOCAL_GLOBAL and compare.local_of(GLOBAL_AUX) == LOCAL


# --- the debiased RMSE ----------------------------------------------------------------------------


def test_the_debiased_rmse_is_the_rmse_with_the_bias_taken_out():
    observed = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([13.0, 21.0, 35.0, 43.0])  # errors 3, 1, 5, 3: bias 3
    scores = dict(zip(INTERVAL_SCORES, _scores(observed, predicted)))
    rmse, bias = np.sqrt(np.mean([9, 1, 25, 9])), 3.0
    assert scores["rmse_debiased"] == pytest.approx(np.sqrt(rmse**2 - bias**2))
    assert scores["rmse_debiased"] == pytest.approx(np.std([3, 1, 5, 3]))
    # A pure offset leaves nothing once debiased.
    assert dict(zip(INTERVAL_SCORES, _scores(observed, observed + 4.0)))["rmse_debiased"] == pytest.approx(0.0)


def test_the_bootstrap_gives_the_debiased_rmse_an_interval_around_its_estimate():
    points = _points(200)
    rng = np.random.default_rng(1)
    frame = _rows(points, points["observed"] + 2.0 + rng.normal(0, 3.0, len(points)), model=GLOBAL)
    intervals = block_bootstrap(frame, ["model"], n=200).intervals.set_index("metric")
    row = intervals.loc["rmse_debiased"]
    error = frame["predicted"] - frame["observed"]
    assert row["estimate"] == pytest.approx(float(np.std(error)))
    assert row["low"] <= row["estimate"] <= row["high"]
    rmse, bias = intervals.loc["rmse", "estimate"], intervals.loc["bias", "estimate"]
    assert row["estimate"] == pytest.approx(np.sqrt(rmse**2 - bias**2))


# --- table S6 ---------------------------------------------------------------------------------------


def _region_intervals(*, with_local=True, with_debiased=True, seeds=(42,)):
    rows = []
    for seed in seeds:
        for basin, base in (("A", 10.0), ("B", 8.0)):
            values = {
                (GLOBAL, "rmse"): base,
                (GLOBAL, "bias"): 2.0,
                (GLOBAL, "r2"): 0.1,
                (LOCAL_GLOBAL, "rmse"): base - 1.0,
                (LOCAL_GLOBAL, "r2"): 0.3,
            }
            if with_debiased:
                values[(GLOBAL, "rmse_debiased")] = np.sqrt(base**2 - 4.0)
            for (model, metric), value in values.items():
                if model == LOCAL_GLOBAL and not with_local:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "kind": LOGO,
                        "held_out": basin,
                        "model": model,
                        "metric": metric,
                        "estimate": value + seed / 1000,
                        "low": value - 0.5,
                        "high": value + 0.5,
                        "n": 100 if basin == "A" else 50,
                        "n_blocks": 9,
                    }
                )
    return pd.DataFrame(rows)


def test_table_s6_reads_each_score_and_its_interval_per_basin_and_seed():
    table = basin_error_table(_region_intervals(seeds=(42, 7)))
    assert len(table) == 4 and set(table["seed"]) == {42, 7}
    a = table[(table["seed"] == 42) & (table["basin"] == "A")].iloc[0]
    assert a["rmse_transferred"] == pytest.approx(10.042) and a["rmse_transferred_low"] == pytest.approx(9.5)
    assert a["bias_transferred"] == pytest.approx(2.042)
    assert a["rmse_debiased_transferred"] == pytest.approx(np.sqrt(96) + 0.042)
    assert a["rmse_local"] == pytest.approx(9.042) and a["rmse_local_high"] == pytest.approx(9.5)
    assert a["n"] == 100


def test_table_s6_leaves_what_the_run_lacks_empty():
    table = basin_error_table(_region_intervals(with_local=False, with_debiased=False))
    assert table["rmse_local"].isna().all() and table["rmse_debiased_transferred"].isna().all()
    assert table["rmse_transferred"].notna().all()
    assert basin_error_table(pd.DataFrame()).empty


def _inputs(region_intervals, *, transfer=None, summary=None, index=None, predictions=None):
    return SimpleNamespace(
        intervals_by_region=region_intervals,
        fit_seed=42,
        block_km=10.0,
        predictions=predictions if predictions is not None else pd.DataFrame({"seed": [42]}),
        transfer=transfer if transfer is not None else pd.DataFrame(),
        dissimilarity_summary=summary if summary is not None else pd.DataFrame(),
        index=index if index is not None else pd.DataFrame(),
    )


def test_figure_s6_shows_the_seed_the_figures_show_and_says_when_local_is_missing():
    inputs = _inputs(_region_intervals(seeds=(42, 7)))
    table = basins.basin_errors(inputs)
    assert list(table["basin"]) == ["A", "B"] and "seed" not in table  # largest basin first, one seed
    text = basins.printable_errors(table)
    assert text.iloc[0, 2] == "10.04\n[9.50, 10.50]" and text.iloc[0, 3].startswith("+2.04")
    caption = figure_caption(basins.figure_basin_errors(inputs))
    assert "below the local RMSE" in caption
    missing = figure_caption(basins.figure_basin_errors(_inputs(_region_intervals(with_local=False))))
    assert LOCAL_GLOBAL in missing and "not in this run" in missing


# --- figure 10 --------------------------------------------------------------------------------------


def test_figure_10_gets_the_local_r2_and_the_transferred_bias_with_their_intervals():
    summary = pd.DataFrame({"held_out": ["A", "B"], "n_test": [100, 50], "carbonate_tvd": [0.1, 0.2]})
    index = pd.DataFrame({"kind": [LOGO, LOGO], "held_out": ["A", "B"], "n_train": [900, 950], "n_test": [100, 50]})
    transfer = pd.DataFrame(
        {
            "kind": [LOGO, LOGO],
            "region": ["A", "B"],
            "covariate_group": ["combined", "combined"],
            "rmse_transferred": [10.0, 8.0],
            "rmse_local": [9.0, 7.0],
            "rmse_drop": [1.0, 1.0],
            "r2_transferred": [0.1, 0.2],
        }
    )
    inputs = _inputs(_region_intervals(), transfer=transfer, summary=summary, index=index)
    table = basins.basin_table(inputs)
    a = table.set_index("basin").loc["A"]
    assert a["r2_local"] == pytest.approx(0.342) and a["r2_local_low"] == pytest.approx(-0.2)
    assert a["bias_transferred"] == pytest.approx(2.042) and a["bias_transferred_high"] == pytest.approx(2.5)
    text = basins.printable(table)
    assert text.loc[0, basins.HEADINGS["r2_local"]] == "0.34\n[−0.20, 0.80]"
    assert text.loc[0, basins.HEADINGS["bias_transferred"]] == "+2.04\n[1.50, 2.50]"


def test_figure_10_fits_two_page_columns_with_the_longest_basin_name_whole():
    row = {
        basins.HEADINGS["basin"]: "Oum Er Rbia",
        basins.HEADINGS["n"]: "2,146",
        basins.HEADINGS["share_removed"]: "37.0",
        basins.HEADINGS["share_outside"]: "1.4",
        basins.HEADINGS["carbonate_tvd"]: "0.11",
        basins.HEADINGS["rho_caco3"]: "0.51 / 0.22",
        basins.HEADINGS["rho_cec"]: "0.33 / 0.42",
        basins.HEADINGS["rmse_transferred"]: "11.67",
        basins.HEADINGS["rmse_local"]: "9.49",
        basins.HEADINGS["rmse_drop"]: "+2.18",
        basins.HEADINGS["r2_transferred"]: "−0.23",
        basins.HEADINGS["r2_local"]: "0.34\n[−0.20, 0.80]",
        basins.HEADINGS["bias_transferred"]: "−3.18\n[−5.32, −1.16]",
    }
    text = pd.DataFrame([row])
    headings = [basins._wrapped_heading(heading) for heading in text.columns]
    widths = basins.column_widths(text, headings)
    assert sum(widths) <= WIDTH_2COL * 72  # never wider than the page, at 7 pt
    # Every column holds its widest line with room to spare: nothing touches a cell edge.
    for position, heading in enumerate(headings):
        widest = max(
            basins._text_width(heading, bold=False), basins._text_width(str(text.iloc[0, position]), bold=position == 0)
        )
        assert widths[position] >= widest + 2 * basins.CELL_PAD - 1e-9
    assert basins._wrapped_heading("Removed (%)") == "Removed\n(%)"
    assert basins._wrapped_heading("R², local [95 % CI]").endswith("\n[95 % CI]")


# --- figures U1 and U3 ------------------------------------------------------------------------------


def test_u1_draws_a_crowded_panel_with_smaller_fainter_points():
    small, crowded = uncertainty.marker_style(300), uncertainty.marker_style(5806)
    assert crowded[0] < small[0] and crowded[1] < small[1]


def test_u3_gives_every_held_out_basin_a_panel_titled_by_its_name():
    basins_ = ["Bouregreg", "Loukkous", "Oum Er Rbia", "Sebou", "Tensift"]
    names = ["Random", "10 km blocks", *[f"Leave one basin out: {basin}" for basin in basins_]]
    rows = uncertainty.width_rows(names)
    assert [label for label, _ in rows] == ["", "Leave one basin out"]
    assert [title for _, title in rows[1][1]] == basins_
    assert sum(len(panels) for _, panels in rows) == len(names)
    assert uncertainty.short_design("10 km blocks") == "10 km"


def _width_data(names, n=200, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for name in names:
        di = rng.gamma(2.0, 0.15, n)
        di[0] = 25.0  # one far outlier per panel
        frames.append(
            pd.DataFrame(
                {
                    "design": name,
                    "plan_id": name,
                    "point_id": [f"{name}_{i}" for i in range(n)],
                    "di": di,
                    "width": rng.normal(25, 4, n),
                    "inside": rng.random(n) < 0.8,
                    "carbonate_class_measured": "<2",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_u3_clips_each_panels_dissimilarity_axis_at_its_99th_percentile():
    import matplotlib.pyplot as plt

    names = ["Random", "10 km blocks", *[f"Leave one basin out: B{i}" for i in range(5)]]
    data = _width_data(names)
    limits = uncertainty.di_limits(data).set_index("design")
    for name in names:
        part = data[data["design"] == name]
        assert limits.loc[name, "di_limit"] == pytest.approx(np.percentile(part["di"], 99))
        assert limits.loc[name, "n_beyond"] == int((part["di"] > np.percentile(part["di"], 99)).sum()) >= 1
    by_class = pd.DataFrame(columns=["design", "class", "n", "picp", "width_mean"])
    figure = uncertainty.draw_width(data, by_class, labels=["<2"])
    shown = {axis.get_title(): axis for axis in figure.findobj(plt.Axes) if axis.get_visible() and axis.get_title()}
    assert {"Random", "10 km blocks", "B0", "B1", "B2", "B3", "B4"} <= set(shown)
    for title, axis in shown.items():
        name = title if title in limits.index else f"Leave one basin out: {title}"
        assert axis.get_xlim()[1] < 25.0  # the outlier lies beyond the axis
        assert axis.get_xlim()[1] == pytest.approx(
            limits.loc[name, "di_limit"]
            + 0.03 * (limits.loc[name, "di_limit"] - data.loc[data["design"] == name, "di"].min())
        )
    plt.close(figure)


def test_u3_rotates_the_short_names_under_the_bars():
    import matplotlib.pyplot as plt

    names = ["Random", "10 km blocks", "Leave one basin out: Oum Er Rbia"]
    data = _width_data(names, n=60)
    by_class = pd.DataFrame(
        [{"design": name, "class": "<2", "n": 60, "picp": 0.8, "width_mean": 25.0} for name in names]
    )
    figure = uncertainty.draw_width(data, by_class, labels=["<2"])
    bars = [axis for axis in figure.findobj(plt.Axes) if axis.get_ylabel().startswith("PICP")][0]
    labels = [label.get_text() for label in bars.get_xticklabels()]
    assert labels == ["Random", "10 km", "Oum Er Rbia"]
    assert all(label.get_rotation() == pytest.approx(30) for label in bars.get_xticklabels())
    plt.close(figure)


# --- figure 5 ---------------------------------------------------------------------------------------


def test_figure_5_keeps_the_bias_words_inside_their_panel_and_wraps_the_legend():
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.a1.figures import spatial
    from yg_eo_soilnet.a1.figures.style import MAIN_MODELS

    rows = []
    for x, design in enumerate(["Random", "10 km blocks", "Leave one basin out"]):
        for model in MAIN_MODELS:
            for metric, value in (("r2", 0.3), ("rmse", 9.0), ("rpiq", 1.5), ("bias", 0.2)):
                rows.append(
                    {
                        "x": x,
                        "design": design,
                        "model": model,
                        "metric": metric,
                        "estimate": value,
                        "low": value - 0.1,
                        "high": value + 0.1,
                    }
                )
    stage1 = pd.DataFrame({"x": [0, 1, 2], "target": "caco3", "r2": [0.4, 0.3, 0.1]})
    figure = spatial.draw_spatial(pd.DataFrame(rows), stage1, labels=["caco3"])
    figure.canvas.draw()
    axes = [axis for axis in figure.axes if axis.get_visible()]
    bias = [axis for axis in axes if axis.get_ylabel().startswith("Bias")][0]
    box = bias.get_window_extent()
    words = [text for text in bias.texts if "predicts" in text.get_text()]
    assert len(words) == 2
    for text in words:
        extent = text.get_window_extent()
        assert box.x0 <= extent.x0 and extent.x1 <= box.x1  # inside the panel, not in its right margin
    legend_texts = [
        text.get_text() for legend in figure.findobj(plt.matplotlib.legend.Legend) for text in legend.get_texts()
    ]
    assert any("\n" in text and "upper bound" in text for text in legend_texts)
    width = figure.get_window_extent().x1
    for legend in figure.findobj(plt.matplotlib.legend.Legend):
        assert legend.get_window_extent().x1 <= width  # no label runs off the figure
    plt.close(figure)


def test_the_evaluation_module_lists_the_debiased_rmse_after_the_six_scores():
    assert evaluation.INTERVAL_SCORES[:6] == evaluation.SCORES and evaluation.INTERVAL_SCORES[6] == "rmse_debiased"
