"""E5's native-class variants: the class inputs given to XGBoost as classes, gaps left missing.

The native step must learn its classes on the training points and leave gaps (and classes first
seen at prediction) missing; the pipeline must hand XGBoost class columns with enable_categorical
on, and only XGBoost; the one-hot path every other run uses must not change; the two variants must
hold exactly the columns they claim; the variants fitted before this switch must keep their
fingerprints; and the comparisons and figures must carry the extra references.
"""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from yg_eo_soilnet.a1 import e5
from yg_eo_soilnet.a1.pipelines import NativeCategories, build_stage_pipeline, final_estimator
from yg_eo_soilnet.a1.runner import Variant, _check_variants
from yg_eo_soilnet.a1.settings import load_settings
from yg_eo_soilnet.a1.stage2 import fit_tuned

A1_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "a1"
MEDIANS = [
    "bd_median_0_30cm",
    "clay_median_0_30cm",
    "ph_median_0_30cm",
    "sand_median_0_30cm",
    "soc_median_0_30cm",
    "swc_median_0_30cm",
]


# --- the native step and the pipeline ---------------------------------------------------------------------


def test_the_native_step_learns_its_classes_and_leaves_gaps_and_new_classes_missing():
    step = NativeCategories().fit(pd.DataFrame({"texture": ["lo", "cl", None, "lo"], "code": [3, 1, 3, np.nan]}))
    assert step.categories_ == {"texture": ["cl", "lo"], "code": ["1", "3"]}
    out = step.transform(pd.DataFrame({"texture": ["cl", "sa", None], "code": [1, 3, 7]}, index=["a", "b", "c"]))
    assert list(out.index) == ["a", "b", "c"]
    assert all(isinstance(dtype, pd.CategoricalDtype) for dtype in out.dtypes)
    assert out["texture"].isna().tolist() == [False, True, True]  # an unseen class is a gap; a gap stays one
    assert out["code"].astype(object).tolist()[:2] == ["1", "3"] and pd.isna(out["code"].iloc[2])
    assert list(step.get_feature_names_out()) == ["texture", "code"]


@pytest.fixture(scope="module")
def xgb_settings():
    settings = load_settings(str(A1_CONFIGS / "a1_mixed_fast.yml"))
    return settings, settings.load_config()


def _points(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "texture_20cm": rng.choice(["cl", "lo", "sa", None], n),
            "landform_class": rng.choice(["valley", "ridge"], n),
        },
        index=[f"p{i}" for i in range(n)],
    )
    y = pd.Series(rng.gamma(2.0, 5.0, n) + 10.0 * (X["texture_20cm"] == "cl"), index=X.index)
    return X, y


def _build(xgb_settings, native):
    settings, config = xgb_settings
    pipeline, grid = build_stage_pipeline(
        settings,
        config,
        ["x", "texture_20cm", "landform_class"],
        ["texture_20cm", "landform_class"],
        stage=settings.stage2,
        target=settings.target,
        native_categorical=native,
    )
    final_estimator(pipeline).set_params(device="cpu")
    return pipeline, grid


def test_the_native_pipeline_gives_xgboost_class_columns_with_gaps_left_missing(xgb_settings):
    pipeline, grid = _build(xgb_settings, native=True)
    assert final_estimator(pipeline).get_params()["enable_categorical"] is True
    preprocessor = pipeline.named_steps["preprocessor"]
    _, step, columns = next(t for t in preprocessor.transformers if t[0] == "cat")
    assert isinstance(step, NativeCategories) and columns == ["texture_20cm", "landform_class"]
    X, y = _points()
    model, params = fit_tuned(pipeline, grid, X, y, pd.Series(np.arange(len(X)) % 3, index=X.index))
    prepared = model.named_steps["preprocessor"].transform(X.head(10))
    assert isinstance(prepared, pd.DataFrame)
    assert isinstance(prepared["texture_20cm"].dtype, pd.CategoricalDtype)
    assert prepared["texture_20cm"].isna().sum() == X.head(10)["texture_20cm"].isna().sum()  # not filled
    new = X.head(3).copy()
    new.loc["p0", "texture_20cm"] = "unseen"
    assert np.isfinite(model.predict(new)).all()


def test_the_one_hot_path_is_unchanged(xgb_settings):
    pipeline, _ = _build(xgb_settings, native=False)
    _, branch, columns = next(t for t in pipeline.named_steps["preprocessor"].transformers if t[0] == "cat")
    assert columns == ["texture_20cm", "landform_class"]
    assert isinstance(branch.named_steps["imputer"], SimpleImputer)
    assert branch.named_steps["imputer"].strategy == "most_frequent"
    assert isinstance(branch.named_steps["encoder"], OneHotEncoder)


def test_native_classes_are_refused_for_a_model_other_than_xgboost(xgb_settings):
    settings, config = xgb_settings
    stage = replace(settings.stage2, estimator="Ridge", init_args={}, params={})
    with pytest.raises(ValueError, match="native_categorical needs XGBoost"):
        build_stage_pipeline(settings, config, ["x", "t"], ["t"], stage=stage, native_categorical=True)
    # Without a class input there is nothing to read natively: built as usual.
    build_stage_pipeline(settings, config, ["x"], ["t"], stage=stage, native_categorical=True)


# --- the variants ----------------------------------------------------------------------------------------


def _resolved():
    groups = {
        "satellite": ["s2_b4_barest", "S2_B4_q50"],
        "climate": ["aridity_index"],
        "terrain": ["elevation", "slope", "landform_class"],
        "management": ["CUE_mean"],
        "legacy_soil": [*MEDIANS, "texture_20cm"],
        "auxiliary": ["a1_pred_caco3_pct_total"],
    }
    return SimpleNamespace(groups=groups, combined=tuple(c for cols in groups.values() for c in cols))


def test_the_two_native_variants_hold_exactly_their_columns():
    variants = e5.part_a_variants(_resolved())
    every = variants["all"].columns
    native = variants[e5.NATIVE_CAT]
    assert native.columns == every and native.native_categorical
    keep = variants[e5.KEEP_TEXTURE]
    assert keep.native_categorical
    assert set(keep.columns) == set(every) - set(MEDIANS)
    assert {"texture_20cm", "landform_class"} <= set(keep.columns)
    assert not [
        name for name, v in variants.items() if v.native_categorical and name not in (e5.NATIVE_CAT, e5.KEEP_TEXTURE)
    ]
    assert "texture_20cm" not in variants["all_minus_legacy_soil"].columns  # the old ablation still drops it


def test_earlier_variants_keep_their_fingerprint_and_the_encoding_changes_it(tmp_path):
    columns = ("a", "b")
    assert Variant(columns).fingerprint == hashlib.sha256("a\nb".encode()).hexdigest()
    assert Variant(columns, native_categorical=True).fingerprint != Variant(columns).fingerprint
    manifest = {}
    log = SimpleNamespace(info=lambda *a, **k: None)
    _check_variants(
        manifest,
        {"all": Variant(columns), e5.NATIVE_CAT: Variant(columns, native_categorical=True)},
        tmp_path,
        restart=False,
        logger=log,
    )
    assert manifest["variant_class_encoding"] == {"all": "onehot", e5.NATIVE_CAT: "native"}
    assert (tmp_path / "variants" / f"{e5.NATIVE_CAT}.txt").read_text() == "a\nb\n"  # columns only


def test_the_native_variants_are_compared_with_all_the_soil_map_ablation_and_each_other():
    names = ["all", "all_minus_legacy_soil", e5.NATIVE_CAT, e5.KEEP_TEXTURE, "all_minus_climate"]
    assert e5.part_of(e5.NATIVE_CAT) == e5.part_of(e5.KEEP_TEXTURE) == "A"
    pairs = set(e5.comparisons("A", names))
    assert {
        (e5.NATIVE_CAT, "all"),
        (e5.KEEP_TEXTURE, "all"),
        (e5.KEEP_TEXTURE, "all_minus_legacy_soil"),
        (e5.KEEP_TEXTURE, e5.NATIVE_CAT),
    } <= pairs
    # A reference that was not fitted is left out rather than compared with nothing.
    assert set(e5.comparisons("A", ["all", e5.KEEP_TEXTURE])) == {(e5.KEEP_TEXTURE, "all")}


# --- the figure ------------------------------------------------------------------------------------------


def _ablation_rows():
    rows = []
    for design in (e5.DESIGN_RANDOM, e5.DESIGN_10KM, e5.DESIGN_LOGO, "logo Sebou"):
        for variant, reference, estimate in [
            ("all_minus_climate", "all", 0.4),
            ("all_minus_legacy_soil", "all", 0.2),
            (e5.NATIVE_CAT, "all", 0.05),
            (e5.KEEP_TEXTURE, "all", 0.1),
            (e5.KEEP_TEXTURE, "all_minus_legacy_soil", -0.1),
            (e5.KEEP_TEXTURE, e5.NATIVE_CAT, 0.05),
        ]:
            rows.append(
                {
                    "part": "A",
                    "design": design,
                    "metric": "rmse",
                    "variant": variant,
                    "reference": reference,
                    "estimate": estimate,
                    "low": estimate - 0.1,
                    "high": estimate + 0.1,
                    "p_holm": 0.5,
                    "n_features": 10,
                }
            )
    return pd.DataFrame(rows)


def test_figures_e5a_and_e5b_draw_the_native_row_and_say_what_it_measures():
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.a1.figures import e5 as figures

    table = _ablation_rows()
    figure, caption = figures.figure_ablation(table)
    assert len(figure.axes) >= 2 * 3  # the groups row and the native row, three designs
    assert "native categorical handling, including missing values" in caption
    # The costliest removal is read against all only, never against another reference.
    assert f"removing {figures.variant_label('all_minus_climate')} costs most" in caption
    plt.close(figure)
    figure, caption = figures.figure_ablation_basins(table)
    assert "including missing values" in caption
    plt.close(figure)
    assert figures.variant_label(e5.KEEP_TEXTURE) == "all − soil-map medians, texture kept"
    # Without the native variants the caption stays as it was.
    _, caption = figures.figure_ablation(table[~table["variant"].isin((e5.NATIVE_CAT, e5.KEEP_TEXTURE))])
    assert "native" not in caption
