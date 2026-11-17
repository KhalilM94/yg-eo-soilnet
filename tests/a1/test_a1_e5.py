"""E5: input variants, time-series representations and the external-covariate hook.

The variants must be exactly the column lists they claim (no group leaking into its own ablation,
coordinates only where asked); each representation must give the known answer on a tiny synthetic
series; the plan-dependent PCA must see the training points alone; the manifest must refuse a
variant whose columns changed; an external CSV must be checked before it becomes inputs; and the
scripts must run end to end on the demo.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import e5, evaluation
from yg_eo_soilnet.a1.runner import RunMismatch, Variant, _check_variants

# --- variants -----------------------------------------------------------------------------------------


def _resolved():
    groups = {
        "satellite": [
            "s2_b4_barest",
            "s2_b4_barest_sd",
            "S2_B4_q50",
            "S2_B4_iqr",
            "S1_VH_asc_q50",
            "S1_VH/VV_asc_q50",
            "AG_NDVI_q50",
            "SOIL_NBR2_q50",
        ],
        "climate": ["aridity_index", "CLIM_ETa_q50", "CLIM_ETa_iqr"],
        "terrain": ["elevation", "slope"],
        "management": ["CUE_mean"],
        "legacy_soil": ["soc_median_0_30cm", "clay_median_0_30cm"],
        "auxiliary": ["a1_pred_caco3_pct_total"],
    }
    combined = tuple(c for cols in groups.values() for c in cols)
    return SimpleNamespace(groups=groups, combined=combined)


def test_no_group_leaks_into_its_own_ablation_and_xy_appears_only_where_asked():
    resolved = _resolved()
    variants = e5.part_a_variants(resolved)
    every = set(variants["all"].columns)
    assert "a1_pred_caco3_pct_total" not in every  # Global's inputs: no stage-1 prediction
    for group in e5.PART_A_GROUPS:
        columns = set(variants[f"all_minus_{group}"].columns)
        assert not columns & set(resolved.groups[group]), group
        assert columns | set(resolved.groups[group]) == every, group
    for block, prefixes in e5.SUBBLOCKS.items():
        columns = variants[f"all_minus_{block}"].columns
        assert not [c for c in columns if c.startswith(prefixes)], block
    for name, variant in variants.items():
        has_xy = bool(set(e5.XY_COLUMNS) & set(variant.columns))
        assert has_xy == (name == "all_plus_xy"), name
    assert set(variants["only_satellite"].columns) == set(resolved.groups["satellite"])
    assert set(variants["satellite_plus_terrain"].columns) == set(resolved.groups["satellite"]) | {"elevation", "slope"}


def test_the_satellite_sub_blocks_partition_the_satellite_group():
    blocks = e5.subblock_columns(_resolved().groups["satellite"])
    assert blocks["s2_barest"] == ["s2_b4_barest", "s2_b4_barest_sd"]
    assert blocks["s1_ts"] == ["S1_VH_asc_q50", "S1_VH/VV_asc_q50"]
    assert sum(len(cols) for cols in blocks.values()) == len(_resolved().groups["satellite"])


def test_part_b_replaces_every_percentile_column_and_keeps_every_static_one():
    resolved = _resolved()
    table_columns = [*resolved.combined, "R2__S2_B4__DJF", "R2__CLIM_ETa__JJA", "R6__S2_B4__q50", e5.BARE_N]
    variants = e5.part_b_variants(resolved, table_columns, ["R1", "R2", "R6"])
    statics = {
        "s2_b4_barest",
        "s2_b4_barest_sd",
        "aridity_index",
        "elevation",
        "slope",
        "CUE_mean",
        "soc_median_0_30cm",
        "clay_median_0_30cm",
    }
    r2 = set(variants["ts_R2"].columns)
    assert r2 == statics | {"R2__S2_B4__DJF", "R2__CLIM_ETa__JJA"}
    assert set(variants["ts_R1"].columns) == statics | {
        "S2_B4_q50",
        "S2_B4_iqr",
        "S1_VH_asc_q50",
        "S1_VH/VV_asc_q50",
        "AG_NDVI_q50",
        "SOIL_NBR2_q50",
        "CLIM_ETa_q50",
        "CLIM_ETa_iqr",
    }
    assert not set(variants["ts_R2_minus_legacy_soil"].columns) & {"soc_median_0_30cm", "clay_median_0_30cm"}
    assert set(variants["only_satellite_R2"].columns) == {"s2_b4_barest", "s2_b4_barest_sd", "R2__S2_B4__DJF"}
    assert set(variants["only_satellite_R6"].columns) == {
        "s2_b4_barest",
        "s2_b4_barest_sd",
        "R6__S2_B4__q50",
        e5.BARE_N,
    }
    assert {e5.part_b_reference(name) for name in variants if name.startswith(("ts_", "only_satellite_"))} == {
        "all",
        "all_minus_legacy_soil",
        "only_satellite",
    }


def test_the_comparisons_of_each_part():
    names = [
        "all",
        "all_minus_climate",
        "all_plus_xy",
        "only_satellite",
        "ts_R2",
        "ts_R2_minus_legacy_soil",
        "all_minus_legacy_soil",
        "only_satellite_R2",
        "all_plus_embeddings",
        "only_embeddings",
    ]
    assert ("all_minus_climate", "all") in e5.comparisons("A", names)
    assert ("ts_R2", "all") not in e5.comparisons("A", names)
    assert set(e5.comparisons("B", names)) == {
        ("ts_R2", "all"),
        ("ts_R2_minus_legacy_soil", "all_minus_legacy_soil"),
        ("only_satellite_R2", "only_satellite"),
    }
    assert set(e5.comparisons("C", names)) == {("all_plus_embeddings", "all"), ("only_embeddings", "all")}


def test_the_manifest_refuses_a_variant_whose_columns_changed(tmp_path):
    manifest = {}
    log = SimpleNamespace(info=lambda *a, **k: None)
    _check_variants(manifest, {"all": Variant(("a", "b"))}, tmp_path, restart=False, logger=log)
    assert (tmp_path / "variants" / "all.txt").read_text() == "a\nb\n"
    _check_variants(
        manifest, {"all": Variant(("a", "b")), "other": Variant(("a",))}, tmp_path, restart=False, logger=log
    )
    with pytest.raises(RunMismatch, match="other input columns"):
        _check_variants(manifest, {"all": Variant(("a", "c"))}, tmp_path, restart=False, logger=log)
    _check_variants(manifest, {"all": Variant(("a", "c"))}, tmp_path, restart=True, logger=log)  # restart allows it
    assert manifest["variants"]["all"] == Variant(("a", "c")).fingerprint


# --- representations -----------------------------------------------------------------------------------


def _series(values_by_month: dict, years=(2017, 2018), point="p1", variable="S2_B4"):
    rows = []
    for year in years:
        for month, value in values_by_month.items():
            rows.append({"point_id": point, "date": pd.Timestamp(year=year, month=month, day=1), variable: value})
    return pd.DataFrame(rows)


def test_quarter_medians_with_the_minimum_readings_rule():
    ts = _series({1: 1.0, 2: 3.0, 12: 2.0, 4: 10.0, 7: 5.0, 8: 7.0})
    table = e5.quarter_medians(ts, ["S2_B4"], min_readings=2)
    row = table.loc["p1"]
    assert row["R2__S2_B4__DJF"] == pytest.approx(2.0)  # median of 1, 3, 2 over two years
    assert row["R2__S2_B4__MAM"] == pytest.approx(10.0)  # April twice: 2 readings
    assert row["R2__S2_B4__JJA"] == pytest.approx(6.0)
    assert np.isnan(row["R2__S2_B4__SON"])  # no reading
    single = e5.quarter_medians(_series({4: 10.0}, years=(2017,)), ["S2_B4"], min_readings=2)
    assert np.isnan(single.loc["p1", "R2__S2_B4__MAM"])  # one reading: below the minimum


def test_r3_keeps_the_sampling_year_only():
    ts = pd.concat([_series({1: 1.0, 2: 1.0}, years=(2017,)), _series({1: 9.0, 2: 9.0}, years=(2019,))])
    ts = pd.concat([ts, _series({1: 5.0, 2: 5.0}, years=(2017,), point="p2")])
    years = pd.Series({"p1": 2019, "p2": 2011})
    table = e5.quarter_medians(ts, ["S2_B4"], rep="R3", years=years, min_readings=2)
    assert table.loc["p1", "R3__S2_B4__DJF"] == pytest.approx(9.0)
    assert table.loc["p2"].isna().all()  # sampled in 2011: no composite that year


def test_the_harmonic_fit_recovers_a_pure_sinusoid_and_needs_twelve_readings():
    days = np.arange(0, 4 * 365.25, 30.4)
    values = 3.0 + 2.0 * np.cos(e5.OMEGA * days - 1.0) + 0.5 * np.cos(2 * e5.OMEGA * days)
    a0, amp1, phase_sin, phase_cos, amp2, resid = e5.harmonic_terms(days, values)
    assert a0 == pytest.approx(3.0) and amp1 == pytest.approx(2.0) and amp2 == pytest.approx(0.5)
    assert phase_sin == pytest.approx(np.sin(1.0)) and phase_cos == pytest.approx(np.cos(1.0))
    assert resid == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(e5.harmonic_terms(days[:11], values[:11])).all()


def test_the_bare_month_filter_on_ndvi_and_nbr2():
    months = []
    for month in range(1, 13):
        bare = month <= 4  # 4 bare months: low NDVI, low NBR2
        months.append(
            {
                "point_id": "p1",
                "date": pd.Timestamp(2018, month, 1),
                "S2_B4": 0.20 if bare else 0.05,
                "S2_B8": 0.24 if bare else 0.40,  # NDVI 0.09 (bare) or 0.78
                "S2_B11": 0.30,
                "S2_B12": 0.29 if bare else 0.20,  # NBR2 0.017 (bare) or 0.2
                "CLIM_ETa": float(month),
            }
        )
    ts = pd.DataFrame(months)
    indices = e5.bare_indices(ts)
    assert indices["ndvi"].iloc[0] == pytest.approx((0.24 - 0.20) / 0.44)
    assert indices["nbr2"].iloc[0] == pytest.approx(0.01 / 0.59)
    table = e5.bare_soil_percentiles(ts, ["S2_B4", "CLIM_ETa"], min_months=3)
    assert table.loc["p1", e5.BARE_N] == 4
    assert table.loc["p1", "R6__S2_B4__q50"] == pytest.approx(0.20)  # the bare months only
    assert table.loc["p1", "R6__CLIM_ETa__q50"] == pytest.approx(6.5)  # every month for the non-band variables
    strict = e5.bare_soil_percentiles(ts, ["S2_B4"], min_months=5)
    assert np.isnan(strict.loc["p1", "R6__S2_B4__q50"]) and strict.loc["p1", e5.BARE_N] == 4


def test_r5_pca_is_fitted_on_the_training_points_only():
    rng = np.random.default_rng(0)
    ids = pd.Index([f"p{i}" for i in range(40)])
    frame = pd.DataFrame(rng.normal(size=(40, 4)), index=ids, columns=[f"R5__S2_B11__m{m:02d}" for m in range(1, 5)])
    table = SimpleNamespace(frame=frame)
    train = ids[:30]
    context = SimpleNamespace(X=pd.DataFrame(index=ids), train_ids=train)
    pca = e5.PlanPCA(tuple(frame.columns), n_components=2)
    scores = pca.transform(context, table)
    # Changing the test points' values changes their scores, never the fitted axes: training scores stay.
    moved = frame.copy()
    moved.loc[ids[30:]] += 100.0
    again = pca.transform(context, SimpleNamespace(frame=moved))
    pd.testing.assert_frame_equal(scores.loc[train], again.loc[train])
    assert not np.allclose(scores.loc[ids[30:]], again.loc[ids[30:]])
    assert list(scores.columns) == ["R5__pc1", "R5__pc2"]


def test_static_variable_names_and_settings_checks():
    assert e5.static_variable("S1_desc_VV") == "S1_VV_desc"
    with pytest.raises(ValueError, match="R0 is the reference"):
        e5.e5_settings(SimpleNamespace(e5={"representations": ["R0"]}))
    with pytest.raises(ValueError, match="Unknown e5"):
        e5.e5_settings(SimpleNamespace(e5={"nope": 1}))
    options = e5.e5_settings(SimpleNamespace(e5={"extra_covariates": [{"path": None, "group": "embeddings"}]}))
    assert options.extra_covariates[0].group == "embeddings" and options.representations == (
        "R1",
        "R2",
        "R3",
        "R4",
        "R6",
    )


# --- external covariates --------------------------------------------------------------------------------


def _table(ids):
    return SimpleNamespace(frame=pd.DataFrame(index=pd.Index(ids, name="uuid")))


def test_external_covariates_are_checked_before_they_become_inputs(tmp_path):
    table = _table(["a", "b", "c"])
    good = tmp_path / "good.csv"
    pd.DataFrame({"uuid": ["c", "a", "b", "z"], "emb_0": [3.0, 1.0, 2.0, 9.0], "emb_1": [0.3, 0.1, 0.2, 0.9]}).to_csv(
        good, index=False
    )
    spec = e5.ExtraCovariates(str(good), "embeddings")
    values = e5.extra_covariates(spec, table)
    assert list(values.columns) == ["embeddings__emb_0", "embeddings__emb_1"]
    assert values["embeddings__emb_0"].tolist() == [1.0, 2.0, 3.0]  # aligned on the table's points
    cases = {
        "missing": (pd.DataFrame({"uuid": ["a", "b"], "emb": [1.0, 2.0]}), "have no row"),
        "duplicate": (pd.DataFrame({"uuid": ["a", "a", "b", "c"], "emb": [1.0, 1.0, 2.0, 3.0]}), "repeated id"),
        "lab": (pd.DataFrame({"uuid": ["a", "b", "c"], "organic_matter_pct": [1.0, 2.0, 3.0]}), "lab column"),
        "text": (pd.DataFrame({"uuid": ["a", "b", "c"], "emb": ["x", "y", "z"]}), "not numeric"),
        "no_id": (pd.DataFrame({"id": ["a", "b", "c"], "emb": [1.0, 2.0, 3.0]}), "no id column"),
    }
    for name, (frame, message) in cases.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match=message):
            e5.extra_covariates(e5.ExtraCovariates(str(path), "embeddings"), table)
    variants = e5.part_c_variants(_resolved(), {"embeddings": list(values.columns)})
    assert set(variants) == {"all", "all_plus_embeddings", "only_embeddings"}
    assert set(variants["only_embeddings"].columns) == set(values.columns)


# --- paired comparisons ------------------------------------------------------------------------------------


def _predictions():
    rng = np.random.default_rng(1)
    n = 150
    lat = 32 + rng.uniform(0, 0.5, n)
    lon = -7 + rng.uniform(0, 0.5, n)
    observed = rng.gamma(4, 5, n)
    rows = []
    for seed in (42, 7):
        for variant, offset in (("all", 0.0), ("all_minus_climate", 2.0), ("all_minus_terrain", 0.0)):
            noise = rng.normal(0, 3, n) if variant != "all_minus_terrain" else 0
            base = observed + rng.normal(0, 3, n)
            rows.append(
                pd.DataFrame(
                    {
                        "plan_id": "random",
                        "kind": "random",
                        "held_out": None,
                        "block_size_m": np.nan,
                        "seed": seed,
                        "point_id": [f"p{i}" for i in range(n)],
                        "split": "test",
                        "lat": lat,
                        "lon": lon,
                        "covariate_group": variant,
                        "observed": observed,
                        "predicted": base + offset + (noise if variant == "all_minus_climate" else 0.0),
                        "fallback": False,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_variant_differences_holm_within_the_family_and_the_seed_summary():
    frame = _predictions()
    pairs = e5.comparisons("A", sorted(frame["covariate_group"].unique()))
    table = e5.variant_differences(
        frame, pairs, part="A", first_seed=42, n_features={"all": 10, "all_minus_climate": 8}, n=200
    )
    assert set(table["design"]) == {e5.DESIGN_RANDOM}
    bias = table[(table["metric"] == "bias") & (table["variant"] == "all_minus_climate")].iloc[0]
    assert bias["estimate"] == pytest.approx(2.0, abs=0.8)
    assert bias["n_seeds"] == 2 and bias["n_seeds_same_sign"] == 2 and bias["n_features"] == 8
    for _, family in table.groupby(["part", "design", "metric"]):
        np.testing.assert_allclose(family["p_holm"], evaluation.holm(family["p_value"].to_numpy()))
    scores = e5.variant_scores(frame)
    assert {"design", "variant", "seed", "rmse"} <= set(scores.columns) and len(scores) == 6


def test_the_figures_render_from_the_tables():
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.a1.figures import e5 as figures

    rows = []
    designs = [e5.DESIGN_RANDOM, e5.DESIGN_10KM, e5.DESIGN_LOGO, "logo Sebou", "logo Tensift"]
    for design in designs:
        for variant in ["all_minus_satellite", "all_minus_legacy_soil", "all_minus_s2_ts", "all_plus_xy"]:
            rows.append(
                {
                    "part": "A",
                    "design": design,
                    "variant": variant,
                    "reference": "all",
                    "metric": "rmse",
                    "estimate": 0.5,
                    "low": 0.1,
                    "high": 0.9,
                    "p_holm": 0.01,
                    "n_features": 150,
                }
            )
    ablation = pd.DataFrame(rows)
    reps = []
    for design in designs[:3]:
        for variant, reference in (
            ("ts_R2", "all"),
            ("ts_R2_minus_legacy_soil", "all_minus_legacy_soil"),
            ("only_satellite_R2", "only_satellite"),
        ):
            reps.append(
                {
                    "part": "B",
                    "design": design,
                    "variant": variant,
                    "reference": reference,
                    "metric": "rmse",
                    "estimate": -0.1,
                    "low": -0.3,
                    "high": 0.1,
                    "p_holm": 0.5,
                    "n_features": 150,
                }
            )
    for figure, _ in (
        figures.figure_ablation(ablation),
        figures.figure_ablation_basins(ablation),
        figures.figure_representations(pd.DataFrame(reps)),
    ):
        assert not hasattr(figure, "_a1_message")
        plt.close(figure)
    assert figures.variant_label("all_minus_satellite").startswith("all − ")


# --- end to end on the demo --------------------------------------------------------------------------------

FAST = {"inner_cv_folds": 2}
FAST_STAGE2 = {
    "estimator": "XGBoost",
    "init_args": {"n_estimators": 20},
    "params": {"model__max_depth": [3]},
    "models": ["global"],
}


@pytest.fixture(scope="module")
def e5_run(a1_settings_in, tmp_path_factory, mlflow_store):
    from experiments.a1.make_plans import main as make_plans_main
    from experiments.a1.run_e5 import main as run_e5_main

    folder = tmp_path_factory.mktemp("a1_e5")
    with mlflow_store(folder / "mlruns") as uri:
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            static=True,
            as_file=True,
            output_dir=str(folder / "out"),
            stage2=FAST_STAGE2,
            e5={"timeseries_file": "timeseries.csv", "representations": ["R2", "R4", "R6"]},
            **FAST,
        )
        make_plans_main(["--config", str(path), "--kinds", "random,logo"])
        part_a = run_e5_main(
            [
                "--config",
                str(path),
                "--part",
                "A",
                "--variants",
                "all,all_minus_terrain,all_plus_xy",
                "--only",
                "random,logo_*",
            ]
        )
        part_b = run_e5_main(["--config", str(path), "--part", "B", "--only", "random,logo_*"])
        yield {"path": path, "out": folder / "out", "uri": uri, "part_a": part_a, "part_b": part_b}


def test_run_e5_fits_each_variant_and_records_its_columns(e5_run):
    run_dir = e5_run["out"] / "runs" / "default"
    frame = evaluation.load_predictions(run_dir)
    variants = set(frame["covariate_group"])
    assert {"all", "all_minus_terrain", "all_plus_xy", "ts_R2", "ts_R4"} <= variants
    assert "ts_R6" not in variants  # the demo series has no B11/B12: R6 skipped, and said so
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for name in variants:
        listed = (run_dir / "variants" / f"{name}.txt").read_text().split()
        assert manifest["variants"][name] == Variant(tuple(listed)).fingerprint
    xy = (run_dir / "variants" / "all_plus_xy.txt").read_text().split()
    assert set(e5.XY_COLUMNS) <= set(xy) and not {"lat", "lon"} & set(xy)
    # Part B reused Part A's `all` rather than fitting it again.
    assert not any("global__all.parquet" in path for path in e5_run["part_b"].computed)
    assert (e5_run["out"] / "features" / "R2.parquet").exists()


def test_evaluate_e5_writes_both_tables_and_the_figures(e5_run):
    import mlflow

    from experiments.a1.evaluate_e5 import main as evaluate_main

    mlflow.set_tracking_uri(e5_run["uri"])  # the module's store (the per-test fixture undoes it)
    manifest = json.loads((e5_run["out"] / "runs" / "default" / "manifest.json").read_text())
    mlflow.set_experiment(experiment_id=manifest["mlflow"]["experiment_id"])
    tables = evaluate_main(["--config", str(e5_run["path"]), "--n-boot", "40"])
    runs = mlflow.search_runs(search_all_experiments=True)
    summary = runs[runs["tags.mlflow.runName"] == "e5_summary"]
    assert len(summary) == 1 and summary["tags.experiment"].iloc[0] == "E5"
    parents = runs[runs["tags.mlflow.runName"].str.startswith("e5_default_", na=False)]
    assert len(parents) == 1 and summary["tags.mlflow.parentRunId"].iloc[0] == parents["run_id"].iloc[0]
    per_variant = runs[runs["tags.variant"].notna()] if "tags.variant" in runs else runs.iloc[0:0]
    assert {"all", "all_plus_xy", "ts_R2"} <= set(per_variant["tags.variant"])
    out = e5_run["out"]
    for name in ("scores.csv", "ablation.csv", "ts_representation.csv"):
        assert (out / name).exists(), name
    for name in ("figE5a_ablation", "figE5b_ablation_basins", "figE5c_ts_representation"):
        assert (out / "figures" / f"{name}.png").exists(), name
    # Part A's two, and the two Part B fitted as its references (they are Part A variants too).
    assert set(tables["ablation"]["variant"]) == {
        "all_minus_terrain",
        "all_plus_xy",
        "all_minus_legacy_soil",
        "only_satellite",
    }
    assert set(tables["ts_representation"]["reference"]) <= {"all", "all_minus_legacy_soil", "only_satellite"}
    assert "variant" in tables["scores"].columns


def test_run_e5_refuses_a_changed_variant(e5_run, tmp_path):
    run_dir = e5_run["out"] / "runs" / "default"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    with pytest.raises(RunMismatch):
        _check_variants(
            dict(manifest), {"all": Variant(("elevation",))}, run_dir, restart=False, logger=SimpleNamespace(info=print)
        )
