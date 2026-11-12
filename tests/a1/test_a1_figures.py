"""The A1 figures: names, palette, the numbers behind each figure, and that every figure draws.

The numbers are checked on small hand-made tables with known answers; each figure is drawn from
synthetic inputs, offline (no map tiles, no borders), and checked for the journal sizes (90 or
190 mm wide) and the 7 pt font floor. The slow acceptance test runs the demo end to end and renders
every figure from its outputs.
"""

import logging
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from yg_eo_soilnet import map_style, plot_style  # noqa: E402
from yg_eo_soilnet.a1 import figures  # noqa: E402
from yg_eo_soilnet.a1.config import stage1_prediction_column  # noqa: E402
from yg_eo_soilnet.a1.figures import (  # noqa: E402
    basins,
    class_bias,
    correlations,
    design,
    labels,
    maps,
    obs_pred,
    render,
    shap,
    spatial,
    stage1,
    style,
    transfer,
)
from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM, SPATIAL_KFOLD  # noqa: E402
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, LOCAL, MODELS  # noqa: E402
from yg_eo_soilnet.metrics import regression_metrics  # noqa: E402

LABELS = ["<2", "2-10", "10-25", ">=25"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def offline(monkeypatch):
    """No map downloads: the map falls back to flat ground with no borders."""

    def refuse(url, path):
        raise OSError(f"offline: {url}")

    monkeypatch.setattr(map_style, "cached_download", refuse)


def _check_print_size(figure):
    """A figure is 90 or 190 mm wide, and no text in it is smaller than 7 pt."""
    assert figure.get_figwidth() == pytest.approx(style.WIDTH_1COL) or figure.get_figwidth() == pytest.approx(
        style.WIDTH_2COL
    )
    assert min(style.figure_fonts(figure)) >= style.MIN_FONT


# --- plain names -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("global", "Global"),
        ("global_aux", "Global + predicted CaCO₃/CEC"),
        ("legacy_soil", "Legacy soil maps"),
        (">=25", "≥ 25 %"),
        ("S2_B12_q75", "S2 B12 (SWIR2), 75th pct"),
        ("S2_B11_q50", "S2 B11 (SWIR1), median"),
        ("AG_NPP_yearly_iqr", "NPP (yearly), IQR"),
        ("s2_b12_barest", "Bare-soil B12 (SWIR2)"),
        ("a1_pred_c_e_c_meq_100g", "Predicted CEC"),
    ],
)
def test_pretty_name_turns_code_names_into_plain_ones(name, expected):
    assert labels.pretty_name(name) == expected


def test_a_split_kind_is_named_with_its_block_size():
    assert labels.pretty_split(SPATIAL_KFOLD, 25_000.0) == "25 km blocks"
    assert labels.pretty_split(LOGO, None) == "Leave one basin out"
    assert labels.pretty_split(RANDOM, np.nan) == "Random"


def test_an_unknown_name_is_shown_as_it_is_and_logged_once(caplog):
    labels._warned.discard("zz_mystery_column")
    with caplog.at_level(logging.INFO, logger=labels.__name__):
        assert labels.pretty_name("zz_mystery_column") == "zz_mystery_column"
        assert labels.pretty_name("zz_mystery_column") == "zz_mystery_column"
    assert sum("zz_mystery_column" in record.getMessage() for record in caplog.records) == 1


def test_each_source_of_figure_3_is_found_by_prefix():
    assert labels.source_of("s2_b11_barest") == "Bare-soil SWIR"
    assert labels.source_of("S2_B12_q50") == "Sentinel-2 SWIR"
    assert labels.source_of("CLIM_ETa_q90") == "Climate"
    assert labels.source_of("S1_VV_asc_iqr") == "Radar"
    assert labels.source_of("S2_B4_q50") is None  # not one of the seven sources: left out


# --- the palette and the axes --------------------------------------------------------------------


def _luminance(color):
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_the_a1_carbonate_classes_run_sand_to_brown_and_the_old_ramp_is_untouched():
    colors = [plot_style.A1_CARBONATE_COLORS[label] for label in plot_style.A1_CARBONATE_CLASSES]
    assert list(plot_style.A1_CARBONATE_CLASSES) == LABELS
    assert all(_luminance(a) > _luminance(b) for a, b in zip(colors, colors[1:]))
    assert plot_style.CARBONATE_RAMP == ["#a7b3c0", "#7d8ea1", "#546a82", "#2c4561"]


def test_each_model_has_its_own_colour_and_marker_and_the_main_three_come_first():
    assert set(MODELS) | {LOCAL} <= set(plot_style.A1_MODEL_COLORS)
    assert len(set(plot_style.A1_MODEL_COLORS.values())) == len(plot_style.A1_MODEL_COLORS)
    assert len(set(style.MODEL_MARKERS.values())) == len(style.MODEL_MARKERS)
    assert style.MODEL_ORDER[:3] == style.MAIN_MODELS and GLOBAL_AUX not in style.MAIN_MODELS


def test_a_bias_axis_names_zero_and_both_sides():
    figure, axis = plt.subplots()
    style.bias_axis(axis)
    texts = [text.get_text() for text in axis.texts]
    assert " no bias" in texts and "↑ over-predicts" in texts and "↓ under-predicts" in texts
    assert "g kg⁻¹" in axis.get_ylabel()


def test_labels_are_placed_without_covering_each_other():
    figure, axis = plt.subplots(figsize=(style.WIDTH_1COL, 2.5), layout="constrained")
    xs, ys = [1.0, 1.05, 1.1, 3.0], [1.0, 1.02, 0.98, 2.0]
    axis.scatter(xs, ys)
    axis.set_xlim(0, 4)
    axis.set_ylim(0, 3)
    placed = style.place_labels(axis, xs, ys, ["North", "South", "East", "West"])
    renderer = figure.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in placed]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


# --- the numbers -------------------------------------------------------------------------------------


def test_class_counts_and_the_confusion_matrix_count_what_they_should():
    counts = figures.class_counts(pd.Series(["A", "B", "B", "B"]), pd.Series(["<2", "<2", ">=25", None]), LABELS)
    assert list(counts.index) == ["B", "A"]  # most points first
    assert counts.loc["B"].tolist() == [1, 0, 0, 1] and counts.loc["A"].tolist() == [1, 0, 0, 0]

    classes = pd.Series(LABELS * 3)
    perfect = figures.class_confusion(classes, classes, LABELS)
    assert (np.diag(perfect.to_numpy()) == 3).all() and perfect.to_numpy().sum() == 12


def test_the_fast_spearman_agrees_with_pandas_also_with_gaps():
    rng = np.random.default_rng(0)
    y = rng.normal(size=80)
    X = np.column_stack([y + rng.normal(size=80), np.round(rng.normal(size=80)), rng.normal(size=80)])
    X[::7, 2] = np.nan
    fast = figures._spearman_columns(y, X)
    frame = pd.DataFrame(X, columns=list("abc")).assign(y=y)
    expected = [frame["y"].corr(frame[column], method="spearman") for column in "abc"]
    assert fast == pytest.approx(expected)


def test_a_monotone_covariate_is_significant_and_noise_is_not():
    rng = np.random.default_rng(1)
    n = 240
    frame = pd.DataFrame({"som": rng.gamma(3, 5, n)})
    frame["monotone"] = np.log(frame["som"])
    frame["noise"] = rng.normal(size=n)
    classes = pd.Series(np.where(np.arange(n) % 2, "<2", ">=25"), index=frame.index)
    table = figures.spearman_by_class(frame, "som", ["monotone", "noise"], classes, labels=LABELS, n=200)
    table = table.set_index(["class", "covariate"])
    assert table.loc[("<2", "monotone"), "rho"] == pytest.approx(1.0)
    assert table.loc[("<2", "monotone"), "significant"] and table.loc[(">=25", "monotone"), "significant"]
    assert not table.loc[("<2", "noise"), "significant"] and not table.loc[(">=25", "noise"), "significant"]
    assert np.isnan(table.loc[("2-10", "monotone"), "rho"])  # no point in that class
    assert (table["p_holm"].dropna() >= table["p_value"].dropna()).all()


def test_the_bare_soil_spectrum_is_in_wavelength_order_per_class():
    frame = pd.DataFrame({"s2_b12_barest": [0.4, 0.5], "s2_b2_barest": [0.1, 0.2], "elevation": [1, 2]})
    spectra = figures.bare_soil_spectra(frame, pd.Series(["<2", "<2"]), LABELS)
    assert spectra["band"].tolist() == ["s2_b2_barest", "s2_b12_barest"]
    assert spectra["wavelength_nm"].tolist() == [490, 2190] and spectra["median"].tolist() == pytest.approx(
        [0.15, 0.45]
    )
    assert figures.bare_soil_spectra(frame[["elevation"]], pd.Series(["<2", "<2"]), LABELS).empty


def test_a_prepared_feature_maps_back_to_its_input_column():
    assert figures._input_column("num__clay") == "clay"
    assert figures._input_column("num__missingindicator_clay") == "clay"
    assert figures._input_column("cat__landform_class") == "landform_class"


def test_tree_shap_values_add_up_to_the_models_predictions():
    from xgboost import XGBRegressor

    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 4))
    y = 3 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, 200)
    model = XGBRegressor(n_estimators=40, max_depth=3, random_state=0).fit(X, y)
    values = figures._tree_shap(model, X)
    assert values.shape == (200, 4)
    base = model.get_booster().predict(__import__("xgboost").DMatrix(X), pred_contribs=True)[:, -1]
    assert values.sum(axis=1) + base == pytest.approx(model.predict(X), abs=1e-3)
    assert np.abs(values).mean(axis=0).argmax() == 0  # the strongest input leads


def _correlations():
    rows = []
    for column, rhos in {
        "S2_B12_q50": [0.30, 0.10, 0.0, 0.0],
        "S2_B12_q90": [0.35, 0.10, 0.0, 0.0],  # beats the median by less than 0.10: the median stays
        "CLIM_ETa_q50": [0.10, 0.0, 0.0, 0.0],
        "CLIM_ETa_q10": [0.40, 0.0, 0.0, 0.0],  # beats the median by 0.30: replaces it
        "elevation": [0.50, 0.2, 0.1, 0.0],
        "S2_B4_q50": [0.90, 0.0, 0.0, 0.0],  # not a figure-3 source
    }.items():
        rows += [
            {"covariate": column, "class": label, "rho": rho, "significant": rho > 0.2}
            for label, rho in zip(LABELS, rhos)
        ]
    return pd.DataFrame(rows)


def test_figure_3_keeps_the_median_unless_another_percentile_is_clearly_stronger():
    rows = correlations.choose_rows(_correlations())
    assert set(rows["column"]) == {"S2_B12_q50", "CLIM_ETa_q10", "elevation"}
    # Ordered by source, as labels.yml lists them.
    assert rows["source"].tolist() == ["Sentinel-2 SWIR", "Climate", "Terrain"]
    assert len(correlations.choose_rows(_correlations(), max_rows=2)) == 2


# --- figure 5: one x position per design present -----------------------------------------------


def _intervals(sizes=(5_000.0, 25_000.0)):
    designs = [(RANDOM, np.nan), *((SPATIAL_KFOLD, size) for size in sizes), (LOGO, np.nan)]
    rows = []
    for step, (kind, size) in enumerate(designs):
        for model in style.MAIN_MODELS:
            for metric in ("r2", "rmse", "rpiq", "bias"):
                value = {"r2": 0.5 - 0.1 * step, "rmse": 8 + step, "rpiq": 1.8 - 0.1 * step, "bias": 0.2 * step}[metric]
                rows.append(
                    {
                        "kind": kind,
                        "block_size_m": size,
                        "model": model,
                        "metric": metric,
                        "estimate": value,
                        "low": value - 0.3,
                        "high": value + 0.3,
                        "n": 100,
                    }
                )
    rows.append(
        {
            "kind": RANDOM,
            "block_size_m": np.nan,
            "model": GLOBAL_AUX,
            "metric": "rmse",
            "estimate": 9.0,
            "low": 8.0,
            "high": 10.0,
            "n": 100,
        }
    )  # not a main model: not drawn
    return pd.DataFrame(rows)


def test_figure_5_has_one_x_position_per_design_present_in_distance_order():
    scores = spatial.spatial_scores(_intervals())
    designs = scores.drop_duplicates("x").sort_values("x")["design"].tolist()
    assert designs == ["Random", "5 km blocks", "25 km blocks", "Leave one basin out"]  # no 10 km: not in the run
    assert set(scores["model"]) == set(style.MAIN_MODELS)
    figure = spatial.draw_spatial(scores, pd.DataFrame())
    assert [tick.get_text() for tick in figure.axes[3].get_xticklabels()] == designs
    _check_print_size(figure)


def test_figure_5_stage_1_r2_is_the_out_of_fold_r2_of_the_audits(tmp_path):
    rng = np.random.default_rng(5)
    n = 60
    points = [f"p{i}" for i in range(n)]
    measured = pd.DataFrame({"caco3": rng.gamma(2, 5, n), "cec": rng.gamma(4, 5, n)}, index=points)
    (tmp_path / "stage1").mkdir()
    index = pd.DataFrame(
        {
            "plan_id": ["random", "kfold_5km_f0", "kfold_5km_f1"],
            "kind": [RANDOM, SPATIAL_KFOLD, SPATIAL_KFOLD],
            "block_size_m": [np.nan, 5_000.0, 5_000.0],
        }
    )
    pooled = {"random": [], "kfold": []}
    for plan_id in index["plan_id"]:
        audit = pd.DataFrame({"point_id": points})
        for target in ("caco3", "cec"):
            audit[stage1_prediction_column(target)] = measured[target].to_numpy() + rng.normal(0, 3, n)
            audit[f"predicted_by_{target}"] = ["oof_fold_0"] * 40 + ["full_train"] * 20
        audit.to_parquet(tmp_path / "stage1" / f"{plan_id}.parquet")
        pooled["random" if plan_id == "random" else "kfold"].append(audit.iloc[:40])
    inputs = SimpleNamespace(
        index=index,
        run_dir=tmp_path,
        table=SimpleNamespace(frame=measured),
        settings=SimpleNamespace(auxiliary=SimpleNamespace(targets=["caco3", "cec"])),
    )
    table = spatial.stage1_out_of_fold(inputs).set_index(["kind", "target"])
    for kind, key in ((RANDOM, "random"), (SPATIAL_KFOLD, "kfold")):
        audit = pd.concat(pooled[key])
        expected = regression_metrics(
            measured.loc[audit["point_id"], "cec"].to_numpy(), audit[stage1_prediction_column("cec")].to_numpy()
        )["r2_test"]
        assert table.loc[(kind, "cec"), "r2"] == pytest.approx(expected)
    assert table.loc[(SPATIAL_KFOLD, "cec"), "n"] == 80  # both folds pooled, out-of-fold rows only


# --- figure 6: the numbers in the box --------------------------------------------------------------


def test_figure_6_stats_are_the_metrics_slope_and_spread_ratio_recomputed():
    rng = np.random.default_rng(6)
    observed = rng.gamma(4, 6, 300)
    predicted = 10 + 0.5 * observed + rng.normal(0, 3, 300)
    stats = obs_pred.obs_pred_stats(observed, predicted)
    expected = regression_metrics(observed, predicted)
    for key in ("r2", "rmse", "rpiq", "bias"):
        assert stats[key] == pytest.approx(expected[f"{key}_test"])
    assert stats["n"] == 300
    assert stats["slope"] == pytest.approx(np.polyfit(observed, predicted, 1)[0])
    assert stats["sd_ratio"] == pytest.approx(np.std(predicted, ddof=1) / np.std(observed, ddof=1))

    points = pd.DataFrame(
        {
            "model": GLOBAL,
            "kind": RANDOM,
            "block_size_m": np.nan,
            "design": "Random",
            "point_id": range(300),
            "observed": observed,
            "predicted": predicted,
            "carbonate_class_measured": rng.choice(LABELS, 300),
        }
    )
    figure = obs_pred.draw_obs_pred(points, labels=LABELS)
    box = next(text.get_text() for text in figure.axes[0].texts if text.get_text().startswith("n = "))
    assert f"slope = {stats['slope']:.2f}" in box and f"RMSE = {stats['rmse']:.2f}" in box
    assert figure.axes[0].get_xlim() == figure.axes[0].get_ylim() and figure.axes[0].get_xlim()[0] == 0
    _check_print_size(figure)


# --- figure 9: mixed training ------------------------------------------------------------------------


def _mixed_inputs(with_mixed=True):
    rng = np.random.default_rng(7)
    rows, index = [], []
    if with_mixed:
        for fraction, moved in ((0.2, 20), (0.5, 50)):
            for repeat in (0, 1):
                for kind in (MIXED, CONTROL):
                    plan_id = f"{kind}_north_p{int(fraction * 100)}_s{repeat}"
                    observed = rng.gamma(4, 5, 40)
                    rows.append(
                        pd.DataFrame(
                            {
                                "plan_id": plan_id,
                                "kind": kind,
                                "held_out": "North",
                                "fraction": fraction,
                                "repeat": repeat,
                                "split": "test",
                                "model": GLOBAL_AUX,
                                "covariate_group": "combined",
                                "observed": observed,
                                "predicted": observed + rng.normal(0, 2 if kind == MIXED else 4, 40),
                            }
                        )
                    )
                    index.append(
                        {
                            "plan_id": plan_id,
                            "kind": kind,
                            "held_out": "North",
                            "fraction": fraction,
                            "repeat": repeat,
                            "n_train": moved if kind == CONTROL else 500 + moved,
                        }
                    )
    else:
        rows.append(
            pd.DataFrame(
                {
                    "plan_id": "random",
                    "kind": RANDOM,
                    "held_out": np.nan,
                    "fraction": np.nan,
                    "repeat": np.nan,
                    "split": "test",
                    "model": GLOBAL_AUX,
                    "covariate_group": "combined",
                    "observed": [1.0, 2.0],
                    "predicted": [1.5, 2.5],
                }
            )
        )
    return SimpleNamespace(
        predictions=pd.concat(rows, ignore_index=True),
        index=pd.DataFrame(index, columns=["plan_id", "kind", "held_out", "fraction", "repeat", "n_train"]),
        transfer=pd.DataFrame(),
    )


def test_figure_9_is_a_message_without_mixed_plans():
    assert figures.is_message_figure(transfer.figure_mixed(_mixed_inputs(with_mixed=False)))


def test_figure_9_puts_the_number_of_moved_samples_on_x_with_the_range_over_repeats():
    curves = transfer.mixed_curves(_mixed_inputs())
    assert sorted(curves["moved"].unique()) == [20, 50]  # the control plans' n_train, not the share
    summary = transfer.mixed_summary(curves)
    row = summary[(summary["kind"] == MIXED) & (summary["moved"] == 20)].iloc[0]
    repeats = curves[(curves["kind"] == MIXED) & (curves["moved"] == 20)]["rmse"]
    assert (row["mean"], row["low"], row["high"], row["repeats"]) == pytest.approx(
        (repeats.mean(), repeats.min(), repeats.max(), 2)
    )
    figure = transfer.figure_mixed(_mixed_inputs())
    assert not figures.is_message_figure(figure) and "fig09_mixed_summary" in figures.figure_data(figure)
    _check_print_size(figure)


# --- every figure draws ----------------------------------------------------------------------


def _points(n=160, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lat": 32.0 + rng.uniform(0, 0.8, n),
            "lon": -7.5 + rng.uniform(0, 1.0, n),
            "region": rng.choice(["North", "South", "East"], n),
            "class": rng.choice([*LABELS, None], n),
        }
    )


def _prediction_rows(kind, plan_id, *, n=120, seed=0, block_size_m=np.nan):
    places = np.random.default_rng(0)  # a point is in the same place in every plan
    lat, lon = 32.0 + places.uniform(0, 0.6, n), -7.0 + places.uniform(0, 0.6, n)
    rng = np.random.default_rng(seed)
    observed = rng.gamma(4, 5, n)
    base = pd.DataFrame(
        {
            "plan_id": plan_id,
            "kind": kind,
            "block_size_m": block_size_m,
            "point_id": [f"p{i}" for i in range(n)],
            "split": "test",
            "lat": lat,
            "lon": lon,
            "carbonate_class_measured": rng.choice(LABELS, n),
            "covariate_group": "combined",
            "observed": observed,
            "fallback": False,
        }
    )
    return pd.concat(
        [base.assign(model=model, predicted=observed + rng.normal(0, 2 + i, n)) for i, model in enumerate(MODELS)],
        ignore_index=True,
    )


def _class_intervals():
    predictions = pd.concat(
        [_prediction_rows(RANDOM, "random"), _prediction_rows(SPATIAL_KFOLD, "kfold_f0", seed=1, block_size_m=10_000.0)]
    )
    inputs = SimpleNamespace(
        predictions=predictions,
        primary_block_size=10_000.0,
        block_km=10.0,
        n_boot=20,
        settings=SimpleNamespace(seed=0),
        _cache={},
    )
    inputs._cached = lambda key, build: inputs._cache.setdefault(key, build())
    return class_bias.class_intervals(inputs)


def _drawn_figures(offline_points):
    """Every main and supplementary figure drawn from synthetic tables (no run needed)."""
    rng = np.random.default_rng(9)
    design_rows = pd.DataFrame(
        [
            ("inputs", "Satellite, climate"),
            ("stage 1", "CaCO₃ and CEC"),
            ("stage 2", "Global"),
            ("stage 2", "Per class, measured CaCO₃"),
            ("stage 2", "Per class, predicted CaCO₃"),
            ("stage 2", "Global + predicted CaCO₃/CEC"),
            ("validation", "Random split"),
            ("validation", "Spatial blocks (5, 25 km)"),
            ("validation", "Leave one basin out (3 basins)"),
        ],
        columns=["step", "text"],
    )
    frame = pd.DataFrame(
        {"som": rng.gamma(3, 5, 160), "s2_b2_barest": rng.uniform(0, 1, 160), "s2_b11_barest": rng.uniform(0, 1, 160)}
    )
    classes = pd.Series(rng.choice(LABELS, 160))
    shown = (
        _correlations().assign(source=lambda part: part["covariate"].map(labels.source_of)).dropna(subset=["source"])
    )
    caco3 = rng.gamma(1.5, 8, 150)
    stage1_points = pd.DataFrame(
        {
            "observed_caco3": caco3,
            "predicted_caco3": caco3 + rng.normal(0, 3, 150),
            "observed_cec": rng.gamma(4, 5, 150),
            "predicted_cec": rng.gamma(4, 5, 150),
            "class_measured": rng.choice(LABELS, 150),
            "class_predicted": rng.choice(LABELS, 150),
        }
    )
    obs = pd.concat(
        [
            _prediction_rows(kind, kind, seed=seed).assign(design=name)
            for seed, (kind, name) in enumerate(
                [(RANDOM, "Random"), (SPATIAL_KFOLD, "10 km blocks"), (LOGO, "Leave one basin out")]
            )
        ]
    )
    obs = obs[obs["model"].isin(style.MAIN_MODELS)]
    intervals = _class_intervals()
    points = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "n": [300, 200, 100],
            "rmse_local": [8.0, 9.0, 10.0],
            "rmse_transferred": [9.0, 9.5, 9.0],
            "rmse_drop": [1.0, 0.5, -1.0],
            "rmse_drop_pct": [12.0, 5.0, -10.0],
            "share_removed": [0.5, 0.33, 0.17],
        }
    )
    grid = pd.DataFrame(
        rng.normal(0, 1, (3, 3)), index=["North", "South", "East"], columns=["combined", "terrain", "climate"]
    )
    summary = transfer.mixed_summary(transfer.mixed_curves(_mixed_inputs()))
    basin_table = pd.DataFrame(
        {
            "basin": ["North", "South"],
            "n": [300, 200],
            "share_removed": [0.5, 0.3],
            "share_outside": [0.01, 0.2],
            "carbonate_tvd": [0.1, 0.3],
            "rho_caco3_basin": [0.2, 0.3],
            "rho_caco3_train": [0.25, 0.2],
            "rho_cec_basin": [0.4, 0.1],
            "rho_cec_train": [0.35, 0.3],
            "rmse_drop": [1.2, -0.3],
        }
    )
    groups = pd.DataFrame(
        [
            {"class": label, "group": group, "mean_abs_shap": float(i + 1), "share": 0.25, "n": 40}
            for label in LABELS
            for i, group in enumerate(["satellite", "climate", "terrain", "legacy_soil"])
        ]
    )
    values = pd.DataFrame(
        [
            {"class": label, "feature": f"S2_B{i}_q50", "mean_abs_shap": float(i), "n": 40}
            for label in ("<2", ">=25")
            for i in range(2, 13)
        ]
    )
    dissimilar = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "median_di": [0.2, 0.5, 0.9],
            "carbonate_tvd": [0.1, 0.3, 0.6],
            "spearman_som_cec_diff": [0.0, -0.2, -0.5],
            "rmse_drop": [1.0, 0.5, -1.0],
        }
    )
    return {
        "fig01_design": design.draw_design(design_rows),
        "fig02_map": maps.draw_map(offline_points, labels=LABELS, held_out=["North"], relief=False),
        "fig03_correlations": correlations.draw_correlations(
            shown,
            correlations.choose_rows(_correlations()),
            figures.bare_soil_spectra(frame, classes, LABELS),
            labels=LABELS,
        ),
        "fig04_stage1": stage1.draw_stage1(
            stage1_points,
            {"caco3": ("CaCO₃", "%"), "cec": ("CEC", "meq 100 g⁻¹")},
            figures.class_confusion(stage1_points["class_measured"], stage1_points["class_predicted"], LABELS),
            labels=LABELS,
        ),
        "fig05_spatial": spatial.draw_spatial(spatial.spatial_scores(_intervals()), pd.DataFrame()),
        "fig06_obs_pred": obs_pred.draw_obs_pred(obs[obs["model"] == GLOBAL], labels=LABELS),
        "fig07_class_bias": class_bias.draw_class_bias(intervals[intervals["metric"] == "bias"], labels=LABELS),
        "fig08_transfer": transfer.draw_transfer(points, grid),
        "fig09_mixed": transfer.draw_mixed(summary, points),
        "fig10_basins": basins.draw_basins(basins.printable(basin_table)),
        "fig11_shap_groups": shap.draw_shap_groups(
            groups, labels=LABELS, order=["satellite", "climate", "terrain", "legacy_soil"]
        ),
        "figS2_obs_pred_models": obs_pred.draw_obs_pred(obs, labels=LABELS, by_model=True),
        "figS3_class_scores": class_bias.draw_class_scores(intervals, labels=LABELS),
        "figS4_shap_features": shap.draw_shap(values, labels=LABELS, n_features=5),
        "figS5_dissimilarity": basins.draw_dissimilarity(dissimilar),
    }


def test_every_figure_draws_at_print_size_with_no_text_under_7_pt(offline):
    drawn = _drawn_figures(_points())
    for name, figure in drawn.items():
        assert not figures.is_message_figure(figure), name
        _check_print_size(figure)


def test_the_map_has_a_bar_panel_beside_it_and_names_the_points_without_carbonate(offline):
    figure = maps.draw_map(_points(), labels=LABELS, held_out=["North"], relief=False)
    assert len(figure.axes) == 2  # the map, and the points per basin beside it - not an inset
    legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any(text.startswith("CaCO₃ not measured") for text in legend)


def test_the_confusion_matrix_is_in_row_percent_with_the_share_correct_in_the_title():
    measured = pd.Series(["<2"] * 3 + [">=25"] * 2)
    predicted = pd.Series(["<2", "<2", ">=25", ">=25", ">=25"])
    counts = figures.class_confusion(measured, predicted, LABELS)
    points = pd.DataFrame(
        {
            "observed_caco3": [1.0, 2, 3, 30, 40],
            "predicted_caco3": [1.0, 2, 30, 30, 40],
            "class_measured": measured,
            "class_predicted": predicted,
        }
    )
    figure = stage1.draw_stage1(points, {"caco3": ("CaCO₃", "%")}, counts, labels=LABELS)
    matrix = figure.axes[-1]
    assert "80%" in matrix.get_title(loc="right")
    assert {"67 %", "33 %", "100 %"} <= {text.get_text() for text in matrix.texts}


def test_missing_inputs_give_a_message_not_an_error():
    for figure in (
        transfer.draw_transfer(pd.DataFrame(), pd.DataFrame()),
        transfer.draw_mixed(pd.DataFrame(), pd.DataFrame()),
        basins.draw_dissimilarity(pd.DataFrame()),
        basins.draw_basins(pd.DataFrame()),
        class_bias.draw_class_scores(pd.DataFrame(), labels=LABELS),
        class_bias.draw_class_bias(pd.DataFrame(), labels=LABELS),
        obs_pred.draw_obs_pred(pd.DataFrame(columns=["design", "model"]), labels=LABELS),
        shap.draw_shap(pd.DataFrame(columns=["class"]), labels=LABELS),
        shap.draw_shap_groups(pd.DataFrame(), labels=LABELS, order=[]),
    ):
        assert figures.is_message_figure(figure)


def test_the_class_bias_intervals_keep_only_the_primary_block_size():
    predictions = pd.concat(
        [
            _prediction_rows(RANDOM, "random"),
            _prediction_rows(SPATIAL_KFOLD, "kfold_10km_f0", seed=1, block_size_m=10_000.0),
            _prediction_rows(SPATIAL_KFOLD, "kfold_25km_f0", seed=2, block_size_m=25_000.0),
        ]
    )
    inputs = SimpleNamespace(
        predictions=predictions,
        primary_block_size=10_000.0,
        block_km=10.0,
        n_boot=20,
        settings=SimpleNamespace(seed=0),
        _cache={},
    )
    inputs._cached = lambda key, build: inputs._cache.setdefault(key, build())
    table = class_bias.class_intervals(inputs)
    kfold = table[(table["kind"] == SPATIAL_KFOLD) & (table["metric"] == "rpiq")]
    assert set(kfold["design"]) == {"10 km blocks"}
    assert kfold["n"].sum() == 4 * 120  # four models, each scored on one size's 120 test points once
    rpiq = class_bias.rpiq_table(table, LABELS)
    assert list(rpiq.columns) == ["Design", "Model", *(f"{labels.pretty_name(label)} CaCO₃" for label in LABELS)]


# --- saving, data, captions, --only/--skip ------------------------------------------------------------


def test_a_figure_is_saved_as_png_and_pdf(tmp_path):
    figure = basins.draw_dissimilarity(
        pd.DataFrame({"region": ["N", "S", "E"], "median_di": [0.2, 0.5, 0.9], "rmse_drop": [1.0, 0.5, -1.0]})
    )
    paths = figures.save_figure(figure, "figS5_dissimilarity", tmp_path / "figures")
    assert [path.name for path in paths] == ["figS5_dissimilarity.png", "figS5_dissimilarity.pdf"]
    assert all(path.stat().st_size > 1000 for path in paths)


def test_only_and_skip_choose_figures_in_order_and_refuse_unknown_names():
    assert render.select_figures(only=["fig06_obs_pred", "fig05_spatial"]) == ["fig05_spatial", "fig06_obs_pred"]
    assert render.select_figures(skip=["fig02_map"]) == [name for name in render.FIGURE_NAMES if name != "fig02_map"]
    assert render.select_figures(only=["fig05_spatial", "fig06_obs_pred"], skip=["fig06_obs_pred"]) == ["fig05_spatial"]
    with pytest.raises(ValueError, match="fig5_transfer"):
        render.select_figures(only=["fig5_transfer"])


def test_render_all_writes_the_figure_its_data_and_its_caption(tmp_path):
    inputs = SimpleNamespace(
        run_dir=tmp_path,
        n_boot=10,
        block_sizes=[5_000.0, 25_000.0],
        held_out_names=["North", "South"],
        settings=SimpleNamespace(
            inner_cv_folds=5, plans=SimpleNamespace(spatial_kfold=SimpleNamespace(block_sizes_m=lambda: ()))
        ),
    )
    drawn = figures.render_all(inputs, tmp_path / "figures", only=["fig01_design"], close=True)
    assert list(drawn) == ["fig01_design"]
    folder = tmp_path / "figures"
    assert (folder / "fig01_design.png").exists() and (folder / "data" / "fig01_design.csv").exists()
    captions = render.read_captions(folder / "captions.md")
    assert list(captions) == ["fig01_design"] and "5, 25 km" in captions["fig01_design"]
    # A later --only run keeps the captions of the figures it did not draw.
    render.write_captions({"fig05_spatial": "Figure 5."}, folder / "captions.md")
    assert list(render.read_captions(folder / "captions.md")) == ["fig01_design", "fig05_spatial"]


def test_tables_for_the_paper_are_written_as_csv_and_latex(tmp_path):
    paths = figures.write_table(pd.DataFrame({"Basin": ["North"], "RMSE": [1.234]}), tmp_path / "table_basins")
    assert [path.suffix for path in paths] == [".csv", ".tex"]
    assert "1.23" in paths[1].read_text() and "\\begin{tabular}" in paths[1].read_text()


def test_the_make_figures_script_refuses_an_unknown_figure_name(capsys):
    from experiments.a1.make_figures import main

    with pytest.raises(SystemExit):
        main(["--only", "fig1_points_map"])
    assert "Unknown figure names" in capsys.readouterr().err


# --- acceptance ------------------------------------------------------------------------------------


@pytest.mark.slow
def test_every_figure_renders_from_a_demo_run(a1_settings_in, tmp_path, mlflow_store, offline):
    """Acceptance: a fast demo run of every plan kind, then every figure from its outputs."""
    from yg_eo_soilnet.a1.config import load_a1_config
    from yg_eo_soilnet.a1.plans import write_plans
    from yg_eo_soilnet.a1.runner import run_plans

    with mlflow_store(tmp_path / "mlruns") as uri:
        path = a1_settings_in(tmp_path)(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            as_file=True,
            output_dir=str(tmp_path / "out"),
            stage1={"estimator": "XGBoost", "init_args": {"n_estimators": 30}},
            stage2={"estimator": "XGBoost", "init_args": {"n_estimators": 30}, "params": {"model__max_depth": [3]}},
            inner_cv_folds=2,
            plans={"mixed": {"fractions": [0.3, 0.6], "n_repeats": 1}, "spatial_kfold": {"n_folds": 3}},
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(tmp_path / "out" / "plans"), kinds=[RANDOM, SPATIAL_KFOLD, LOGO, MIXED])
        run_plans(a1, groups=True, tracking=False)

    inputs = figures.FigureInputs.from_run(a1)
    drawn = figures.render_all(inputs, relief=False, n_boot=50, close=True)
    assert list(drawn) == list(figures.FIGURE_NAMES)
    messages = {name: figure._a1_message for name, figure in drawn.items() if figures.is_message_figure(figure)}
    assert not messages
    folder = inputs.run_dir / "figures"
    for name in figures.FIGURE_NAMES:
        for suffix in (".png", ".pdf"):
            assert (folder / f"{name}{suffix}").stat().st_size > 1000
    assert list(render.read_captions(folder / "captions.md")) == list(figures.FIGURE_NAMES)
    assert {"table_rpiq_by_class.tex", "table_basins.tex", "fig05_spatial_scores.csv"} <= {
        path.name for path in (folder / "data").iterdir()
    }
