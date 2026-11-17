"""E2/E4 on stacked repeats: the arms against the transferred model, the matched sizes, figure 9.

Repeats are stacked - one row per point and repeat - and every paired comparison scores both sides
on the same rows of each block resample. The tests pin the stacking (the transferred model on each
repeat's own points; the full and local models repeated once per repeat), the Holm families (within
a basin), the local-model fallback and its label, and figure 9's parts.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import compare, evaluation
from yg_eo_soilnet.a1.figures import core as figure_core
from yg_eo_soilnet.a1.figures import transfer
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, MIXED_WEIGHTED
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL, LOCAL_GLOBAL
from yg_eo_soilnet.datamodules.splitting import TEST

N_BOOT = 60


def _points(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "point_id": [f"p{i:03d}" for i in range(n)],
            "lat": 32.0 + rng.uniform(0, 0.5, n),
            "lon": -7.0 + rng.uniform(0, 0.6, n),
            "observed": rng.gamma(4.0, 5.0, n),
        }
    )


def _rows(points, predicted, *, model, kind, plan_id, held_out="A", fraction=np.nan, repeat=np.nan):
    return pd.DataFrame(
        {
            "plan_id": plan_id,
            "kind": kind,
            "held_out": held_out,
            "fraction": fraction,
            "repeat": repeat,
            "point_id": points["point_id"].to_numpy(),
            "split": TEST,
            "lat": points["lat"].to_numpy(),
            "lon": points["lon"].to_numpy(),
            "model": model,
            "covariate_group": "combined",
            "observed": points["observed"].to_numpy(),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )


def _mixed_frame(points, *, kinds=(CONTROL, MIXED, MIXED_WEIGHTED), offsets=None, repeats=2, shares=(0.3, 0.6)):
    """Mixed kinds of basin A: within a share and repeat every kind tests the same points; the logo plan all."""
    offsets = offsets or {CONTROL: 3.0, MIXED: 1.0, MIXED_WEIGHTED: 1.0, "logo": 5.0}
    rows = [_rows(points, points["observed"] + offsets["logo"], model=GLOBAL, kind=LOGO, plan_id="logo_A")]
    for fraction in shares:
        for repeat in range(repeats):
            test = points.sample(frac=1 - fraction, random_state=int(fraction * 100) + repeat)
            for kind in kinds:
                rows.append(
                    _rows(
                        test,
                        test["observed"] + offsets[kind],
                        model=GLOBAL,
                        kind=kind,
                        plan_id=f"{kind}_A_p{int(fraction * 100)}_s{repeat}",
                        fraction=fraction,
                        repeat=float(repeat),
                    )
                )
    return pd.concat(rows, ignore_index=True)


# --- the mixed kinds against the transferred model ---------------------------------------------------


def test_the_transferred_model_is_scored_on_each_repeats_own_test_points():
    points = _points()
    frame = _mixed_frame(points)
    stacked = evaluation.mixed_stack(frame, model=GLOBAL)
    for (fraction, repeat), part in stacked.groupby(["fraction", "repeat"]):
        by_kind = {kind: set(rows["point_id"]) for kind, rows in part.groupby("kind")}
        assert by_kind[evaluation.TRANSFERRED] == by_kind[CONTROL] == by_kind[MIXED]
        assert not part.groupby("kind")["point_id"].apply(lambda ids: ids.duplicated().any()).any()
        assert (part["plan_id"] == f"A|{fraction}|{repeat}").all()
    transferred = stacked[stacked["kind"] == evaluation.TRANSFERRED]
    # One row per point and repeat: the logo plan's predictions repeated, once per repeat they test.
    assert len(transferred) == len(stacked[stacked["kind"] == CONTROL])
    np.testing.assert_allclose(transferred["predicted"] - transferred["observed"], 5.0)


def test_arms_against_transferred_give_the_known_gap_and_holm_within_the_basin():
    points = _points()
    frame = _mixed_frame(points)
    stacked = evaluation.mixed_stack(frame, model=GLOBAL)
    table = evaluation.arms_vs_transferred(stacked, n=N_BOOT, seed=3)
    bias = table[table["metric"] == "bias"].set_index(["a", "fraction"])
    assert bias.loc[(CONTROL, 0.3), "estimate"] == pytest.approx(3.0 - 5.0)
    assert bias.loc[(MIXED, 0.6), "estimate"] == pytest.approx(1.0 - 5.0)
    assert (table["b"] == evaluation.TRANSFERRED).all()
    for (_, metric), part in table.groupby(["held_out", "metric"]):
        assert len(part) == 3 * 2  # three kinds x two shares, one family
        np.testing.assert_allclose(part["p_holm"], evaluation.holm(part["p_value"].to_numpy()))


def test_figure_9_intervals_and_dots_come_from_the_same_stacked_rows():
    points = _points()
    stacked = evaluation.mixed_stack(_mixed_frame(points), model=GLOBAL)
    intervals = evaluation.mixed_intervals(stacked, n=N_BOOT, seed=1)
    repeats = evaluation.mixed_repeat_scores(stacked)
    one = intervals[(intervals["kind"] == MIXED) & (intervals["fraction"] == 0.3) & (intervals["metric"] == "bias")]
    assert one["estimate"].iloc[0] == pytest.approx(1.0)
    assert one["n"].iloc[0] == 2 * len(
        stacked[(stacked["kind"] == MIXED) & (stacked["repeat"] == 0) & (stacked["fraction"] == 0.3)]
    )
    dots = repeats[(repeats["kind"] == MIXED) & (repeats["fraction"] == 0.3)]
    assert len(dots) == 2 and np.allclose(dots["bias"], 1.0)


# --- matched sizes -------------------------------------------------------------------------------------


def _run(name, frame, n_train=None):
    return SimpleNamespace(name=name, predictions=frame, n_train=lambda plan_id: (n_train or {}).get(plan_id, np.nan))


def _matched_runs(points, *, local=LOCAL_GLOBAL):
    rows = []
    for size, offset in ((3600, 2.0), (2000, 3.0)):
        for repeat in range(3):
            rows.append(
                _rows(
                    points,
                    points["observed"] + offset,
                    model=GLOBAL,
                    kind=LOGO,
                    plan_id=f"logo_m{size}_r01_a_s{repeat}",
                    repeat=float(repeat),
                )
            )
    matched = pd.concat(rows, ignore_index=True)
    plain = [_rows(points, points["observed"] + 1.0, model=GLOBAL, kind=LOGO, plan_id="logo_r01_a")]
    if local:
        plain.append(_rows(points, points["observed"] + 0.5, model=local, kind=LOGO, plan_id="logo_r01_a"))
    reference = _run("a1_xgb", pd.concat(plain, ignore_index=True), {"logo_r01_a": 5000})
    return [_run("m", matched, {f"logo_m{s}_r01_a_s{r}": s - 3 for s in (3600, 2000) for r in range(3)}), reference]


def test_matched_draws_are_stacked_and_the_single_fits_repeated_per_repeat():
    points = _points(80)
    runs = _matched_runs(points)
    stack, local_model, n_train = compare.matched_stack(runs, runs[1])
    assert local_model == LOCAL_GLOBAL
    assert set(stack["model"]) == {"3600", "2000", compare.FULL, compare.LOCAL_LABEL}
    assert len(stack[stack["model"] == "3600"]) == 3 * len(points)  # one row per point and repeat
    assert len(stack[stack["model"] == compare.FULL]) == len(points)  # once here; repeated when paired
    assert n_train[("A", "3600")] == pytest.approx(3597) and n_train[("A", compare.FULL)] == 5000
    paired = compare.matched_paired(stack, local_model=local_model, n=N_BOOT)
    rows = paired[paired["metric"] == "bias"].set_index(["a", "b"])
    assert rows.loc[("3600", compare.FULL), "estimate"] == pytest.approx(1.0)
    assert rows.loc[("2000", "3600"), "estimate"] == pytest.approx(1.0)
    assert rows.loc[("2000", compare.LOCAL_LABEL), "estimate"] == pytest.approx(2.5)
    assert rows.loc[("3600", compare.FULL), "n"] == 3 * len(points)  # the full fit paired with every repeat
    for (_, metric), part in paired.groupby(["basin", "metric"]):
        assert len(part) == 4  # 3600-full, 2000-3600, 3600-local, 2000-local: one family per basin
        np.testing.assert_allclose(part["p_holm"], evaluation.holm(part["p_value"].to_numpy()))
    intervals = compare.matched_intervals(stack, n_train, local_model=local_model, n=N_BOOT)
    assert set(intervals["metric"]) == {"rmse", "bias", "r2"}
    assert intervals.set_index(["model", "metric"]).loc[(compare.FULL, "rmse"), "n"] == len(points)


def test_matched_tables_fall_back_on_local_and_say_so():
    points = _points(60)
    runs = _matched_runs(points, local=LOCAL)
    stack, local_model, _ = compare.matched_stack(runs, runs[1])
    assert local_model == LOCAL
    paired = compare.matched_paired(stack, local_model=local_model, n=20)
    assert set(paired["local_model"]) == {LOCAL}
    assert compare.matched_intervals(stack.iloc[0:0], {}).empty


# --- the transfer table's fallback --------------------------------------------------------------------


def test_transfer_main_rows_use_local_when_local_global_is_missing_and_label_it():
    points = _points(40)
    rows = [
        _rows(points, points["observed"] + 3.0, model=GLOBAL, kind=LOGO, plan_id="logo_a"),
        _rows(points, points["observed"] + 1.0, model=LOCAL, kind=LOGO, plan_id="logo_a"),
        _rows(points, points["observed"] + 4.0, model=GLOBAL_AUX, kind=LOGO, plan_id="logo_a"),
    ]
    table = evaluation.transfer_table(pd.concat(rows, ignore_index=True))
    main = table[table["inputs"] == evaluation.INPUTS_EVERY].iloc[0]
    assert main["local_model"] == LOCAL and main["rmse_local"] == pytest.approx(1.0) and main["has_local"]
    assert evaluation.fell_back(table)
    rows.append(_rows(points, points["observed"] + 2.0, model=LOCAL_GLOBAL, kind=LOGO, plan_id="logo_a"))
    table = evaluation.transfer_table(pd.concat(rows, ignore_index=True))
    main = table[table["inputs"] == evaluation.INPUTS_EVERY].iloc[0]
    assert main["local_model"] == LOCAL_GLOBAL and main["rmse_local"] == pytest.approx(2.0)
    assert not evaluation.fell_back(table)


def test_table_s6_falls_back_on_local_too():
    intervals = pd.DataFrame(
        [
            {
                "kind": LOGO,
                "held_out": "A",
                "model": model,
                "metric": "rmse",
                "estimate": value,
                "low": 0,
                "high": 1,
                "n": 5,
                "n_blocks": 3,
            }
            for model, value in ((GLOBAL, 10.0), (LOCAL, 8.0))
        ]
    )
    table = evaluation.basin_error_table(intervals)
    assert table["local_model"].iloc[0] == LOCAL and table["rmse_local"].iloc[0] == 8.0


def test_transfer_status_falls_back_on_local_and_names_it():
    points = _points(80)
    frame = pd.concat(
        [
            _rows(points, points["observed"] + 3.0, model=GLOBAL, kind=LOGO, plan_id="logo_a"),
            _rows(points, points["observed"] + 0.5, model=LOCAL, kind=LOGO, plan_id="logo_a"),
        ],
        ignore_index=True,
    )
    status = compare.transfer_status(_run("a1_xgb", frame), n=N_BOOT)
    assert status["local_model"].iloc[0] == LOCAL and status["difference"].iloc[0] > 0


# --- figure 9 -------------------------------------------------------------------------------------------


def _figure_inputs(frame):
    index = pd.DataFrame(
        [
            {
                "plan_id": plan_id,
                "kind": kind,
                "held_out": "A",
                "fraction": fraction,
                "repeat": repeat,
                "n_train": int(120 * fraction),
            }
            for (plan_id, kind, fraction, repeat), _ in frame[frame["kind"] != LOGO].groupby(
                ["plan_id", "kind", "fraction", "repeat"]
            )
        ]
    )
    return SimpleNamespace(
        predictions=frame,
        index=index,
        settings=SimpleNamespace(seed=42),
        block_km=10.0,
        n_boot=N_BOOT,
        transfer=pd.DataFrame(),
    )


def test_figure_9_draws_every_kind_with_bands_dots_and_rings():
    points = _points()
    offsets = {CONTROL: 3.0, MIXED: 0.0, MIXED_WEIGHTED: 2.9, "logo": 5.0}
    inputs = _figure_inputs(_mixed_frame(points, offsets=offsets))
    figure = transfer.figure_mixed(inputs)
    assert not figure_core.is_message_figure(figure)
    data = figure_core.figure_data(figure)
    assert {"fig09_mixed_intervals", "fig09_mixed_repeats", "fig09_mixed_vs_control"} <= set(data)
    intervals = data["fig09_mixed_intervals"]
    assert set(intervals["kind"]) == {CONTROL, MIXED, MIXED_WEIGHTED, evaluation.TRANSFERRED}
    assert sorted(intervals["moved"].unique()) == [36, 72]  # the control plans' n_train, mean over repeats
    caption = figure_core.figure_caption(figure)
    assert transfer.FIG9_NOTE in caption and "E4" not in caption
    # The RMSE axis of the basin: log x, one ring at least (mixed is far better than control).
    axes = [axis for axis in figure.axes if axis.get_ylabel().startswith("RMSE")]
    assert axes and axes[0].get_xscale() == "log"
    rings = [
        collection
        for axis in figure.axes
        for collection in axis.collections
        if len(collection.get_offsets()) and collection.get_facecolor().size == 0
    ]
    assert rings


def test_figure_9_with_control_and_mixed_only_and_with_the_e4_panel():
    points = _points()
    frame = _mixed_frame(points, kinds=(CONTROL, MIXED))
    inputs = _figure_inputs(frame)
    figure = transfer.figure_mixed(inputs)
    assert set(figure_core.figure_data(figure)["fig09_mixed_intervals"]["kind"]) == {
        CONTROL,
        MIXED,
        evaluation.TRANSFERRED,
    }
    stacked = evaluation.mixed_stack(frame, model=GLOBAL)
    intervals = transfer._with_moved(evaluation.mixed_intervals(stacked, n=20), transfer.moved_points(inputs))
    repeats = transfer._with_moved(evaluation.mixed_repeat_scores(stacked), transfer.moved_points(inputs))
    e4 = pd.DataFrame(
        {
            "basin": ["A", "A", "A"],
            "model": ["2000", "3600", "full"],
            "n_train": [2000, 3600, 5000],
            "metric": "rmse",
            "estimate": [12.0, 11.5, 11.4],
            "low": [11.0, 10.5, 10.4],
            "high": [13.0, 12.5, 12.4],
        }
    )
    e4_repeats = pd.DataFrame(
        {"basin": ["A"] * 2, "model": ["2000", "3600"], "repeat": [0, 0], "n_train": [2000, 3600], "rmse": [12.1, 11.6]}
    )
    drawn = transfer.draw_mixed(
        intervals, repeats, pd.DataFrame(columns=["region", "a", "metric", "p_holm", "fraction"]), e4, e4_repeats
    )
    titles = {axis.get_title() for axis in drawn.axes}
    assert "E4: training size" in titles
    e4_axis = next(axis for axis in drawn.axes if axis.get_title() == "E4: training size")
    assert not e4_axis.texts  # no labels at the line ends: the basins are in a legend
    legends = [legend for axis in drawn.axes for legend in [axis.get_legend(), *axis.artists] if legend is not None]
    names = {text.get_text() for legend in legends if hasattr(legend, "get_texts") for text in legend.get_texts()}
    assert "A" in names


def test_e4_points_are_offset_per_basin_and_coloured_as_in_figure_c7():
    from yg_eo_soilnet.a1.figures.style import basin_colors

    points = _points()
    frame = _mixed_frame(points, kinds=(CONTROL, MIXED))
    inputs = _figure_inputs(frame)
    stacked = evaluation.mixed_stack(frame, model=GLOBAL)
    intervals = transfer._with_moved(evaluation.mixed_intervals(stacked, n=20), transfer.moved_points(inputs))
    repeats = transfer._with_moved(evaluation.mixed_repeat_scores(stacked), transfer.moved_points(inputs))
    e4 = pd.DataFrame(
        [
            {
                "basin": basin,
                "model": label,
                "n_train": n,
                "metric": "rmse",
                "estimate": 11.0,
                "low": 10.0,
                "high": 12.0,
            }
            for basin in ("Sebou", "A")
            for label, n in (("2000", 2000), ("3600", 3600))
        ]
    )
    drawn = transfer.draw_mixed(
        intervals, repeats, pd.DataFrame(columns=["region", "a", "metric", "p_holm", "fraction"]), e4
    )
    e4_axis = next(axis for axis in drawn.axes if axis.get_title() == "E4: training size")
    xs = sorted({round(float(x), 3) for line in e4_axis.get_lines() for x in line.get_xdata()})
    assert 2000.0 not in xs and len(xs) == 4  # every basin shifted, none on the size itself
    colors = basin_colors(["Sebou", "A"])
    drawn_colors = {line.get_color() for line in e4_axis.get_lines()}
    assert set(colors.values()) <= drawn_colors


def test_figure_9_is_a_message_without_mixed_plans():
    points = _points(30)
    frame = _rows(points, points["observed"], model=GLOBAL, kind=LOGO, plan_id="logo_a")
    assert figure_core.is_message_figure(transfer.figure_mixed(_figure_inputs(frame)))


def test_e4_sizes_without_folders_is_empty(tmp_path):
    intervals, repeats = transfer.e4_sizes([tmp_path / "nothing"])
    assert intervals.empty and repeats.empty
