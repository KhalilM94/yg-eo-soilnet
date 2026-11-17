"""The robustness checks of the transfer mechanism: the power check, the cells, the correction with k samples.

Each check reads only what a run saved. The tests pin what each promises: samples of exactly n by
whole blocks, reproducible, with the rest of the last block set aside; a cell's numbers as a hand
computation gives them; an offset learnt on every point leaving no mean residual; a linear
recalibration recovering a known line; every correction of one draw scored on the same points;
and the three scripts running end to end on the demo run, with or without the E2 run.
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.a1.basin_cells import main as basin_cells_main
from experiments.a1.basin_relations import main as basin_relations_main
from experiments.a1.bias_correction import main as bias_correction_main
from yg_eo_soilnet.a1 import robustness
from yg_eo_soilnet.a1.config import load_a1_config
from yg_eo_soilnet.a1.figures.core import figure_caption, is_message_figure
from yg_eo_soilnet.a1.figures.robustness import (
    figure_bias_correction,
    figure_cells,
    figure_som_profiles,
    figure_within_class,
)
from yg_eo_soilnet.a1.plans import LOGO, write_plans
from yg_eo_soilnet.a1.runner import run_plans
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, LOCAL_GLOBAL

FAST = {
    "stage1": {"estimator": "XGBoost", "init_args": {"n_estimators": 30}},
    "stage2": {"estimator": "XGBoost", "init_args": {"n_estimators": 30}, "params": {"model__max_depth": [3]}},
    "inner_cv_folds": 2,
}


# --- sampling whole blocks -------------------------------------------------------------------------


def _blocked(n_blocks=12, per_block=7):
    ids = pd.Index([f"p{block:02d}_{i}" for block in range(n_blocks) for i in range(per_block)])
    blocks = pd.Series([block for block in range(n_blocks) for _ in range(per_block)], index=ids)
    return ids, blocks


@pytest.mark.parametrize("n", [1, 7, 10, 30, 84])
def test_block_sample_returns_exactly_n_points_by_whole_blocks(n):
    ids, blocks = _blocked()
    chosen, leftover = robustness.block_sample(ids, blocks, n, np.random.default_rng(3))
    assert len(chosen) == n and not chosen.duplicated().any() and set(chosen) <= set(ids)
    used = blocks.reindex(chosen)
    counts = used.value_counts()
    # Every block but at most one is taken whole; the partly used one's rest is set aside.
    partial = counts[counts < 7]
    assert len(partial) <= 1
    if len(partial):
        block = partial.index[0]
        assert set(leftover) == set(ids[blocks.to_numpy() == block]) - set(chosen)
    else:
        assert leftover.empty
    assert not set(leftover) & set(chosen)


def test_block_sample_is_reproducible_and_refuses_too_few_points():
    ids, blocks = _blocked()
    first = robustness.block_sample(ids, blocks, 20, np.random.default_rng([42, 0, 5]))
    again = robustness.block_sample(ids[::-1], blocks, 20, np.random.default_rng([42, 0, 5]))
    assert list(first[0]) == list(again[0]) and list(first[1]) == list(again[1])  # order of ids does not matter
    other = robustness.block_sample(ids, blocks, 20, np.random.default_rng([42, 1, 5]))
    assert set(other[0]) != set(first[0])
    with pytest.raises(ValueError, match="fewer than"):
        robustness.block_sample(ids, blocks, len(ids) + 1, np.random.default_rng(0))


# --- the local model ---------------------------------------------------------------------------------


def test_the_local_model_falls_back_to_local_with_a_warning(caplog):
    assert robustness.choose_local_model({GLOBAL, LOCAL_GLOBAL, LOCAL}) == LOCAL_GLOBAL
    with caplog.at_level("WARNING"):
        assert robustness.choose_local_model({GLOBAL, LOCAL}) == LOCAL
    assert "No local_global predictions" in caplog.text
    assert robustness.choose_local_model({GLOBAL, LOCAL, LOCAL_GLOBAL}, LOCAL) == LOCAL
    with pytest.raises(ValueError, match="no local_global"):
        robustness.choose_local_model({GLOBAL})
    assert robustness.local_text(LOCAL) == "local model with predicted CaCO3/CEC"


# --- the cells ---------------------------------------------------------------------------------------


def _pairs():
    rng = np.random.default_rng(0)
    rows = []
    for seed in (42, 7):
        for basin in ("A", "B", "C"):
            for label in ("<2", ">=25"):
                for i in range(12):
                    observed = 20 + rng.normal(0, 5)
                    rows.append(
                        {
                            "seed": seed,
                            "basin": basin,
                            "plan_id": f"logo_{basin}",
                            "point_id": f"{basin}{label}{i}",
                            "lat": 32.0,
                            "lon": -7.0,
                            "class": label,
                            "observed": observed,
                            "predicted": observed + 2.0 + rng.normal(0, 3),
                            "local": observed + rng.normal(0, 2),
                        }
                    )
    return pd.DataFrame(rows)


def test_one_cell_matches_a_hand_computation():
    pairs = _pairs()
    shifts = pd.DataFrame({"basin": ["A"], "class": ["<2"], "level_shift": [1.5]})
    changes = pd.DataFrame(
        {
            "basin": ["A"],
            "class": ["<2"],
            "relationship_change": [0.2],
            "n_significant": [1],
            "n_significant_with_caco3_cec": [2],
        }
    )
    dissimilar = pd.DataFrame({"basin": ["A"], "class": ["<2"], "median_di": [0.4], "share_outside": [0.1]})
    cells = robustness.cell_table(pairs, shifts, changes, dissimilar, local_model=LOCAL, first_seed=42)
    assert list(dict.fromkeys(cells["seed"])) == ["42", "7", "mean"]
    row = cells[(cells["seed"] == "42") & (cells["basin"] == "A") & (cells["class"] == "<2")].iloc[0]
    part = pairs[(pairs["seed"] == 42) & (pairs["basin"] == "A") & (pairs["class"] == "<2")]
    error = part["predicted"] - part["observed"]
    rmse = np.sqrt(np.mean(error**2))
    local = np.sqrt(np.mean((part["local"] - part["observed"]) ** 2))
    assert row["n"] == 12 and row["local_model"] == LOCAL
    assert row["bias_transferred"] == pytest.approx(error.mean())
    assert row["rmse_transferred"] == pytest.approx(rmse) and row["rmse_local"] == pytest.approx(local)
    assert row["loss"] == pytest.approx(rmse - local)
    assert row["rmse_debiased_transferred"] == pytest.approx(np.sqrt(rmse**2 - error.mean() ** 2))
    assert row["level_shift"] == 1.5 and row["relationship_change"] == 0.2 and row["median_di"] == 0.4
    assert row["n_significant"] == 1 and row["n_significant_with_caco3_cec"] == 2
    mean = cells[(cells["seed"] == "mean") & (cells["basin"] == "A") & (cells["class"] == "<2")].iloc[0]
    by_seed = cells[(cells["seed"].isin(["42", "7"])) & (cells["basin"] == "A") & (cells["class"] == "<2")]
    assert mean["loss"] == pytest.approx(by_seed["loss"].mean())


def test_relationship_change_leaves_out_caco3_and_cec_but_counts_them_apart():
    relations = pd.DataFrame(
        {
            "basin": "A",
            "class": ["all", "<2", "<2", "<2"],
            "variable": ["x", "caco3", "x", "y"],
            "difference": [0.9, -0.8, 0.2, -0.4],
            "p_holm": [0.01, 0.01, 0.01, 0.5],
        }
    )
    row = robustness.relationship_changes(relations, ["caco3", "cec"]).iloc[0]
    assert row["relationship_change"] == pytest.approx(0.3) and row["n_covariates"] == 2
    assert row["n_significant"] == 1 and row["n_significant_with_caco3_cec"] == 2


def test_the_basin_bootstrap_skips_draws_of_fewer_than_three_basins_and_says_how_many():
    cells = pd.DataFrame(
        {
            "seed": "42",
            "local_model": LOCAL,
            "basin": np.repeat(list("ABCDE"), 4),
            "loss": np.arange(20.0),
            "relationship_change": np.arange(20.0) + np.tile([0.0, 0.5, -0.5, 0.2], 5),
            "bias_transferred": np.zeros(20),  # constant: no value at all
            "level_shift": np.arange(20.0),
            "median_di": np.arange(20.0),
            "debiased_minus_local": np.arange(20.0),
        }
    )
    table = robustness.cell_correlations(cells, n_draws=400, seed=1).set_index(["y", "x"])
    row = table.loc[("loss", "relationship_change")]
    # 5 basins drawn with replacement: fewer than 3 distinct in about 9.8 % of draws.
    assert row["share_skipped"] == pytest.approx(0.098, abs=0.04)
    assert row["low"] <= row["rho"] <= row["high"] and row["n_cells"] == 20
    constant = table.loc[("bias_transferred", "level_shift")]
    assert np.isnan(constant["rho"]) and constant["share_skipped"] == 1.0


# --- correcting with k samples -------------------------------------------------------------------------


def _basin(n=300, seed=0, a=4.0, b=0.6):
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(10, 40, n)
    return pd.DataFrame(
        {
            "basin": "A",
            "point_id": [f"p{i:03d}" for i in range(n)],
            "class": np.where(np.arange(n) % 2, "<2", ">=25"),
            "predicted": predicted,
            "observed": a + b * predicted,
            "local": a + b * predicted + rng.normal(0, 1, n),
        }
    )


def test_an_offset_learnt_on_every_point_leaves_no_mean_residual():
    frame = _basin()
    frame["observed"] += np.random.default_rng(1).normal(0, 3, len(frame))
    corrected = robustness.offset_correction(frame, frame)
    assert np.mean(corrected - frame["observed"]) == pytest.approx(0.0, abs=1e-10)


def test_the_linear_correction_recovers_a_known_line():
    frame = _basin(a=4.0, b=0.6)
    a, b = robustness.linear_fit(frame.iloc[:40])
    assert a == pytest.approx(4.0) and b == pytest.approx(0.6)
    corrected = robustness.linear_correction(frame.iloc[:40], frame.iloc[40:])
    np.testing.assert_allclose(corrected, frame["observed"].iloc[40:])
    assert np.isnan(robustness.linear_fit(frame.iloc[:5].assign(predicted=3.0))[0])


def test_the_class_offset_falls_back_to_the_overall_one_for_thin_or_missing_classes():
    sample = pd.DataFrame(
        {"class": ["<2"] * 5 + [">=25"] * 2, "predicted": [10.0] * 7, "observed": [8.0] * 5 + [4.0] * 2}
    )
    test = pd.DataFrame({"class": ["<2", ">=25", ""], "predicted": [10.0, 10.0, 10.0]})
    overall = float((sample["predicted"] - sample["observed"]).mean())
    corrected = robustness.class_offset_correction(sample, test)
    np.testing.assert_allclose(corrected, [8.0, 10.0 - overall, 10.0 - overall])


def test_every_correction_of_a_draw_is_scored_on_the_same_test_points(monkeypatch):
    frame = _basin(n=260)
    blocks = pd.Series(np.arange(260) // 6, index=frame["point_id"].to_numpy())
    seen = []
    original = robustness._rmse

    def recording(error):
        seen.append(len(np.asarray(error)))
        return original(error)

    monkeypatch.setattr(robustness, "_rmse", recording)
    draws = robustness.correction_draws(frame, blocks, ks=[25, 50], n_draws=3, seed=4)
    # Per draw: the four corrections and the local model, five scores, all on n_test points.
    assert len(seen) == 5 * len(draws)
    for position, row in enumerate(draws.itertuples()):
        assert set(seen[5 * position : 5 * position + 5]) == {row.n_test}
        assert row.n_test + row.k + row.n_set_aside == 260
    # A k leaving fewer than 100 test points is skipped.
    assert robustness.correction_draws(frame, blocks, ks=[200], n_draws=1).empty


def test_the_test_points_never_share_a_block_with_a_sample():
    frame = _basin(n=260)
    blocks = pd.Series(np.arange(260) // 6, index=frame["point_id"].to_numpy())
    part = frame.set_index("point_id", drop=False)
    for draw in range(5):
        rng = np.random.default_rng([4, draw, robustness._key("A"), 50])
        chosen, leftover = robustness.block_sample(part.index, blocks, 50, rng)
        test = part.index.difference(chosen).difference(leftover)
        assert not set(blocks.reindex(test)) & set(blocks.reindex(chosen))


def test_the_e2_curves_average_the_repeats_and_are_empty_without_e2():
    arms = pd.DataFrame(
        {
            "basin": ["Sebou"] * 4 + ["Tensift"],
            "fraction": [0.1, 0.1, 0.3, 0.3, 0.1],
            "n_moved": [40, 44, 120, 124, 30],
            "rmse_control": [12.0, 11.0, 10.0, 9.0, 8.0],
            "rmse_mixed": [11.0, 11.0, 10.0, 10.0, 8.0],
        }
    )
    curves = robustness.e2_curves(arms, ["Sebou"])
    assert list(curves["n_moved"]) == [42, 122] and list(curves["rmse_control"]) == [11.5, 9.5]
    assert (curves["n_repeats"] == 2).all()
    assert robustness.read_e2_curves(None, ["Sebou"]).empty


def test_captions_files_are_merged_not_overwritten(tmp_path):
    path = tmp_path / "captions.md"
    robustness.write_captions({"figC6": "six"}, path)
    robustness.write_captions({"figC5": "five", "figC7": "seven"}, path)
    robustness.write_captions({"figC5": "five again"}, path)
    text = path.read_text()
    assert "## figC6\n\nsix" in text and "five again" in text and "## figC7\n\nseven" in text
    assert "## figC5\n\nfive\n" not in text


# --- the scripts, end to end on the demo ------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo(a1_settings_in, tmp_path_factory, mlflow_store):
    """A fast demo run of the leave-one-region-out plans (Global, global_aux, both local models)."""
    folder = tmp_path_factory.mktemp("a1_robustness")
    with mlflow_store(folder / "mlruns") as uri:
        path = a1_settings_in(folder)(
            main_overrides={"MLFLOW_TRACKING_URI": uri},
            as_file=True,
            output_dir=str(folder / "out"),
            stage2={**FAST["stage2"], "models": ["global", "global_aux"]},
            **{key: value for key, value in FAST.items() if key != "stage2"},
        )
        a1 = load_a1_config(str(path))
        write_plans(a1.settings, str(folder / "out" / "plans"), kinds=[LOGO])
        run_plans(a1, groups=False)
        basin_relations_main(["--config", str(path), "--n-boot", "20"])
        yield {"path": path, "out": folder / "out", "uri": uri, "folder": folder}


def test_basin_cells_writes_its_tables_and_figures(demo):
    tables = basin_cells_main(["--config", str(demo["path"]), "--n-boot", "50", "--no-mlflow"])
    out = demo["out"] / "robustness"
    for name in ("basin_cells.csv", "basin_cells_correlations.csv", "captions.md"):
        assert (out / name).exists(), name
    for name in ("figC5_som_profiles", "figC7_cells"):
        assert (out / f"{name}.png").exists() and (out / "data" / f"{name}.csv").exists(), name
    cells = tables["basin_cells"]
    assert set(cells["local_model"]) == {LOCAL_GLOBAL}  # the demo run has it
    assert {"loss", "level_shift", "relationship_change", "median_di", "share_outside"} <= set(cells.columns)
    assert set(tables["basin_cells_correlations"]["local_model"]) == {LOCAL_GLOBAL}
    assert "local model on Global's inputs" in (out / "captions.md").read_text()


def test_basin_cells_can_be_asked_for_the_local_model_with_predicted_caco3_cec(demo, tmp_path):
    tables = basin_cells_main(
        [
            "--config",
            str(demo["path"]),
            "--n-boot",
            "20",
            "--no-mlflow",
            "--local-model",
            "local",
            "--out",
            str(tmp_path),
        ]
    )
    assert set(tables["basin_cells"]["local_model"]) == {LOCAL}
    assert "with predicted CaCO3/CEC" in (tmp_path / "captions.md").read_text()


def test_bias_correction_runs_without_the_e2_run_and_scores_one_set_of_points(demo, tmp_path):
    tables = bias_correction_main(
        [
            "--config",
            str(demo["path"]),
            "--k",
            "10,25",
            "--n-draws",
            "4",
            "--no-mlflow",
            "--mixed-run",
            str(tmp_path / "no_such_run"),
            "--out",
            str(tmp_path),
        ]
    )
    table = tables["bias_correction"]
    assert not table.empty and set(table["local_model"]) == {LOCAL_GLOBAL}
    assert tables["bias_correction_e2"].empty
    assert (tmp_path / "figC6_bias_correction.png").exists() and (tmp_path / "bias_correction.csv").exists()
    caption = (tmp_path / "captions.md").read_text()
    assert "E2" not in caption and "local model on Global's inputs" in caption


def test_the_power_check_writes_its_tables_and_figure_c4_and_logs_to_mlflow(demo):
    import mlflow

    tables = basin_relations_main(
        ["--config", str(demo["path"]), "--subsample", "--n-sub", "30,10000", "--n-draws", "2", "--n-boot", "20"]
    )
    summary, draws = tables["basin_relations_subsample"], tables["basin_relations_subsample_draws"]
    folder = demo["out"] / "basin_relations"
    assert (folder / "basin_relations_subsample.csv").exists() and (folder / "basin_relations.csv").exists()
    basins = summary[summary["level"] == "basin"]
    assert not basins[basins["n_sub"] == 10000]["testable"].any()  # no class that large: not testable
    tested = draws[draws["n_sub"] == 30]
    assert not tested.empty and set(tested["draw"]) == {0, 1}
    assert (demo["out"] / "robustness" / "figC4_within_class.png").exists()
    # A child run of a new "robustness" parent (the demo has no compare run).
    mlflow.set_tracking_uri(demo["uri"])
    manifest = json.loads((demo["out"] / "runs" / "default" / "manifest.json").read_text())
    runs = mlflow.search_runs(experiment_ids=[manifest["mlflow"]["experiment_id"]])
    names = set(runs["tags.mlflow.runName"])
    assert {"robustness", "power_check"} <= names
    child = runs[runs["tags.mlflow.runName"] == "power_check"].iloc[0]
    parent = runs[runs["tags.mlflow.runName"] == "robustness"].iloc[0]
    assert child["tags.mlflow.parentRunId"] == parent["run_id"]


def test_the_power_check_is_reproducible(demo):
    from yg_eo_soilnet.a1.basin_relations import load_basin_data
    from yg_eo_soilnet.a1.settings import load_settings

    settings = load_settings(str(demo["path"]))
    data = load_basin_data(settings)
    relations = pd.read_csv(demo["out"] / "basin_relations" / "basin_relations.csv")
    kwargs = dict(n_sub=30, n_draws=2, n_boot=20, seed=settings.seed, min_cell_size=settings.min_cell_size)
    first = robustness.subsample_relations(data, relations, **kwargs)
    again = robustness.subsample_relations(data, relations, **kwargs)
    pd.testing.assert_frame_equal(first, again)
    assert not first.empty
    # The family per basin and draw: the draw's within-class tests and the basin's all-point covariates.
    one = first[(first["basin"] == first["basin"].iloc[0]) & (first["draw"] == 0)]
    fixed = relations[
        (relations["basin"] == one["basin"].iloc[0])
        & (relations["class"] == "all")
        & (relations["family"] == "exploratory")
    ]
    from yg_eo_soilnet.a1 import evaluation

    expected = evaluation.holm(np.concatenate([one["p_value"].to_numpy(float), fixed["p_value"].to_numpy(float)]))
    np.testing.assert_allclose(one["p_holm"].to_numpy(float), np.asarray(expected)[: len(one)])


# --- the figures, from small tables ---------------------------------------------------------------------


def test_the_figures_render_from_small_tables():
    import matplotlib.pyplot as plt

    labels = ["<2", ">=25"]
    relations = pd.DataFrame(
        [
            {"basin": basin, "class": label, "variable": variable, "difference": d, "p_holm": p}
            for basin, d, p in (("A", 0.4, 0.01), ("B", -0.1, 0.6))
            for label in labels
            for variable in ("caco3", "S2_B11_q50", "elevation")
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "level": "basin",
                "basin": "A",
                "n_sub": 240,
                "testable": True,
                "share_any": 1.0,
                "mean_significant": 2.0,
                "q05_significant": 1.0,
                "q95_significant": 3.0,
            },
            {"level": "basin", "basin": "B", "n_sub": 240, "testable": False},
            {
                "level": "basin",
                "basin": "A",
                "n_sub": 55,
                "testable": True,
                "share_any": 0.4,
                "mean_significant": 0.5,
                "q05_significant": 0.0,
                "q95_significant": 2.0,
            },
            {
                "level": "basin",
                "basin": "B",
                "n_sub": 55,
                "testable": True,
                "share_any": 0.0,
                "mean_significant": 0.0,
                "q05_significant": 0.0,
                "q95_significant": 0.0,
            },
        ]
    )
    c4, caption = figure_within_class(relations, summary, primary=["caco3"], labels=labels, failing=["A"])
    assert not is_message_figure(c4) and "not testable" in caption
    texts = [text.get_text() for axis in c4.axes for text in axis.texts]
    assert any("not testable" in text for text in texts)
    profiles = pd.DataFrame(
        [
            {
                "basin": basin,
                "side": side,
                "class": label,
                "n": 40,
                "som_median": 20.0,
                "som_q25": 15.0,
                "som_q75": 25.0,
            }
            for basin in "AB"
            for side in ("basin", "complement")
            for label in labels
        ]
    )
    c5, _ = figure_som_profiles(profiles, labels=labels, failing=["A"])
    corrections = pd.DataFrame(
        [
            {
                "basin": basin,
                "k": k,
                **{
                    f"rmse_{m}{end}": 10.0 + i
                    for i, m in enumerate((*robustness.METHODS, "local"))
                    for end in ("", "_q05", "_q95")
                },
            }
            for basin in "AB"
            for k in (10, 25, 50)
        ]
    )
    e2 = pd.DataFrame({"basin": ["A"], "fraction": [0.1], "n_moved": [40], "rmse_control": [9.0], "rmse_mixed": [9.5]})
    c6, c6_caption = figure_bias_correction(corrections, local_model=LOCAL, failing=["A"], e2=e2)
    assert "with predicted CaCO3/CEC" in c6_caption and "test points of the correction draws" in c6_caption
    cells = pd.DataFrame(
        {
            "seed": "42",
            "basin": np.repeat(list("ABC"), 2),
            "class": labels * 3,
            "loss": [1.0, 2, 0, -1, 0.5, 0.2],
            "relationship_change": [0.1, 0.3, 0.1, 0.2, 0.15, 0.12],
            "bias_transferred": [1.0, -2, 0, 1, 2, -1],
            "level_shift": [-1.0, 2, 0, -1, -2, 1],
        }
    )
    rho = pd.DataFrame(
        [
            {
                "seed": "42",
                "y": "loss",
                "x": "relationship_change",
                "rho": 0.5,
                "low": 0.1,
                "high": 0.9,
                "share_skipped": 0.1,
            }
        ]
    )
    c7, c7_caption = figure_cells(cells, rho, labels=labels, local_model=LOCAL_GLOBAL, scope="42")
    assert "exploratory" in c7_caption.lower() and "local model on Global's inputs" in c7_caption
    for figure in (c4, c5, c6, c7):
        assert not is_message_figure(figure)
        plt.close(figure)
    assert "Not drawn" in figure_caption(
        figure_cells(cells.iloc[0:0], rho, labels=labels, local_model=LOCAL, scope="42")[0]
    )
