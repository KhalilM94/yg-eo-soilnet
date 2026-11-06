"""The A1 figures: the palette, the numbers behind each figure, and that every figure draws.

The numbers are checked on small hand-made tables with known answers; each figure is drawn from
synthetic inputs, offline (no map tiles, no borders). The slow acceptance test runs the demo end to
end and renders all eight figures from its outputs.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from yg_eo_soilnet import map_style, plot_style  # noqa: E402
from yg_eo_soilnet.a1 import figures  # noqa: E402
from yg_eo_soilnet.a1.plans import LOGO, MIXED, RANDOM, SPATIAL_KFOLD  # noqa: E402
from yg_eo_soilnet.a1.stage2 import MODELS  # noqa: E402

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


# --- the palette ---------------------------------------------------------------------------------


def _luminance(color):
    red, green, blue = to_rgb(color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_the_carbonate_classes_run_light_to_dark_and_each_model_has_its_own_colour():
    colors = [plot_style.A1_CARBONATE_COLORS[label] for label in plot_style.A1_CARBONATE_CLASSES]
    assert list(plot_style.A1_CARBONATE_CLASSES) == LABELS
    assert all(_luminance(a) > _luminance(b) for a, b in zip(colors, colors[1:]))
    assert set(MODELS) <= set(plot_style.A1_MODEL_COLORS)
    assert len(set(plot_style.A1_MODEL_COLORS.values())) == len(plot_style.A1_MODEL_COLORS)


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


# --- every figure draws ----------------------------------------------------------------------


def _points(n=160, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lat": 32.0 + rng.uniform(0, 0.8, n),
            "lon": -7.5 + rng.uniform(0, 1.0, n),
            "region": rng.choice(["North", "South", "East"], n),
            "class": rng.choice(LABELS, n),
        }
    )


def test_the_map_draws_offline_with_hulls_for_outlines_and_a_class_inset(offline):
    figure = figures.draw_points_map(_points(), labels=LABELS, relief=False)
    assert not figures.is_message_figure(figure)
    assert len(figure.axes) == 1 and len(figure.axes[0].child_axes) == 1  # the map and its inset


def test_the_map_draws_given_region_polygons(offline):
    import geopandas as gpd
    from shapely.geometry import box

    regions = gpd.GeoDataFrame({"region": ["all"]}, geometry=[box(-7.6, 31.9, -6.4, 32.9)], crs="EPSG:4326")
    assert len(figures.draw_points_map(_points(), regions, labels=LABELS, relief=False).axes[0].child_axes) == 1


def test_the_correlation_figure_draws_with_and_without_bands():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {"som": rng.gamma(3, 5, 120), "s2_b2_barest": rng.uniform(0, 1, 120), "s2_b11_barest": rng.uniform(0, 1, 120)}
    )
    classes = pd.Series(rng.choice(LABELS, 120))
    correlations = figures.spearman_by_class(
        frame, "som", ["s2_b2_barest", "s2_b11_barest"], classes, labels=LABELS, n=20
    )
    spectra = figures.bare_soil_spectra(frame, classes, LABELS)
    assert len(figures.draw_correlations(correlations, spectra, labels=LABELS).axes) == 3  # two panels and a colour bar
    assert len(figures.draw_correlations(correlations, spectra.iloc[0:0], labels=LABELS).axes) == 3


def test_the_stage_1_figure_draws_both_targets_and_the_confusion_matrix():
    rng = np.random.default_rng(3)
    caco3 = rng.gamma(1.5, 8, 150)
    points = pd.DataFrame(
        {
            "observed_caco3": caco3,
            "predicted_caco3": caco3 + rng.normal(0, 3, 150),
            "observed_cec": rng.gamma(4, 5, 150),
            "predicted_cec": rng.gamma(4, 5, 150),
            "class_measured": rng.choice(LABELS, 150),
            "class_predicted": rng.choice(LABELS, 150),
        }
    )
    figure = figures.draw_stage1(points, {"caco3": "CaCO3 (%)", "cec": "CEC"}, labels=LABELS)
    assert len(figure.axes) == 3


def _prediction_rows(kind, plan_id, *, n=120, seed=0):
    places = np.random.default_rng(0)  # a point is in the same place in every plan
    lat, lon = 32.0 + places.uniform(0, 0.6, n), -7.0 + places.uniform(0, 0.6, n)
    rng = np.random.default_rng(seed)
    observed = rng.gamma(4, 5, n)
    base = pd.DataFrame(
        {
            "plan_id": plan_id,
            "kind": kind,
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


def test_the_class_score_figure_draws_random_and_spatial_side_by_side():
    predictions = pd.concat([_prediction_rows(RANDOM, "random"), _prediction_rows(SPATIAL_KFOLD, "kfold_f0", seed=1)])
    scores = figures.class_scores(predictions, n=30)
    assert set(scores["metric"]) == {"rpiq", "bias"} and set(scores["kind"]) == {RANDOM, SPATIAL_KFOLD}
    assert len(figures.draw_class_scores(scores, labels=LABELS).axes) == 4
    assert len(figures.draw_class_scores(scores[scores["kind"] == RANDOM], labels=LABELS).axes) == 2


def _transfer():
    rows = []
    for number, region in enumerate(["North", "South", "East"]):
        for group in ("combined", "terrain", "satellite"):
            local, transferred = 6.0 + number, 8.0 + 2 * number
            rows.append(
                dict(
                    kind=LOGO,
                    region=region,
                    covariate_group=group,
                    fraction=np.nan,
                    has_local=True,
                    rmse_local=local,
                    rmse_transferred=transferred,
                    rmse_drop=transferred - local,
                )
            )
            for fraction in (0.3, 0.6):
                rows.append(
                    dict(
                        kind=MIXED,
                        region=region,
                        covariate_group=group,
                        fraction=fraction,
                        has_local=True,
                        rmse_local=9.0 - 4 * fraction,
                        rmse_transferred=transferred - 3 * fraction,
                        rmse_drop=transferred - 3 * fraction - (9.0 - 4 * fraction),
                    )
                )
    return pd.DataFrame(rows)


def test_the_transfer_and_mixed_figures_draw_one_panel_per_group():
    transfer = _transfer()
    assert len(figures.draw_transfer(transfer).axes) == 2
    assert len(figures.draw_mixed(transfer).axes) == 3  # combined, satellite, terrain


def test_the_dissimilarity_figure_joins_drop_and_summary_by_region():
    summary = pd.DataFrame(
        {
            "plan_id": ["logo_n", "logo_s", "logo_e"],
            "held_out": ["North", "South", "East"],
            "median_di": [0.2, 0.5, 0.9],
            "carbonate_tvd": [0.1, 0.3, 0.6],
            "spearman_som_cec_diff": [0.0, -0.2, -0.5],
        }
    )
    joined = figures.dissimilarity_frame(_transfer(), summary)
    assert len(joined) == 3 and joined.set_index("region").loc["East", "median_di"] == 0.9
    assert len(figures.draw_dissimilarity(joined).axes) == 3  # two panels and a colour bar


def test_the_shap_figure_draws_one_panel_per_class_present():
    values = pd.DataFrame(
        [
            {"class": label, "feature": f"x{i}", "mean_abs_shap": float(i), "n": 40}
            for label in ("<2", ">=25")
            for i in range(12)
        ]
    )
    figure = figures.draw_shap(values, labels=LABELS, n_features=5)
    assert len(figure.axes) == 2
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["x7", "x8", "x9", "x10", "x11"]


def test_missing_inputs_give_a_message_not_an_error():
    empty = pd.DataFrame(columns=["kind", "covariate_group", "has_local"])
    for figure in (
        figures.draw_transfer(empty),
        figures.draw_mixed(empty),
        figures.draw_dissimilarity(pd.DataFrame()),
        figures.draw_class_scores(pd.DataFrame(), labels=LABELS),
        figures.draw_shap(pd.DataFrame(columns=["class"]), labels=LABELS),
    ):
        assert figures.is_message_figure(figure)


def test_a_figure_is_saved_as_png_and_pdf(tmp_path):
    paths = figures.save_figure(figures.draw_mixed(_transfer()), "fig6_mixed", tmp_path / "figures")
    assert [path.name for path in paths] == ["fig6_mixed.png", "fig6_mixed.pdf"]
    assert all(path.stat().st_size > 1000 for path in paths)


# --- acceptance ------------------------------------------------------------------------------------


@pytest.mark.slow
def test_every_figure_renders_from_a_demo_run(a1_settings_in, tmp_path, mlflow_store, offline):
    """Acceptance: a fast demo run of every plan kind, then all eight figures from its outputs."""
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
    for name in figures.FIGURE_NAMES:
        for suffix in (".png", ".pdf"):
            assert (inputs.run_dir / "figures" / f"{name}{suffix}").stat().st_size > 1000


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


def test_the_class_scores_keep_only_the_primary_block_size():
    predictions = pd.concat(
        [
            _prediction_rows(RANDOM, "random").assign(block_size_m=np.nan),
            _prediction_rows(SPATIAL_KFOLD, "kfold_10km_f0", seed=1).assign(block_size_m=10_000.0),
            _prediction_rows(SPATIAL_KFOLD, "kfold_25km_f0", seed=2).assign(block_size_m=25_000.0),
        ]
    )
    primary = figures.class_scores(predictions, n=20, block_size_m=10_000)
    pooled = figures.class_scores(predictions, n=20)
    kfold = primary[(primary["kind"] == SPATIAL_KFOLD) & (primary["metric"] == "rpiq")]
    assert kfold["n"].sum() == 4 * 120  # four models, each scored on one size's 120 test points once
    assert pooled[(pooled["kind"] == SPATIAL_KFOLD) & (pooled["metric"] == "rpiq")]["n"].sum() == 4 * 240
    assert set(primary.loc[primary["kind"] == RANDOM, "n"]) == set(pooled.loc[pooled["kind"] == RANDOM, "n"])
