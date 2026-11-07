"""The pre-flight diagnostics, end to end on the demo data and table by table.

The demo's earlier predictions are deliberately leaky and some of its region x carbonate cells are
deliberately thin, so a pre-flight that found nothing on it would be broken.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.a1.preflight import main as preflight_main
from yg_eo_soilnet.a1.preflight import (
    ALL_MEASURED,
    COMPLETENESS_FILE,
    MISSINGNESS_FILE,
    NO_CACO3,
    NO_REGION,
    PREDICTIONS_FILE,
    REGION_BY_CARBONATE_FILE,
    REGION_BY_YEAR_FILE,
    INPUTS_FILE,
    REPORT_FILE,
    SMALL_CELLS_FILE,
    TOTAL,
    completeness,
    count_table,
    missingness,
    prediction_scores,
    run_preflight,
    small_cells,
)
from yg_eo_soilnet.a1.strata import FAO_CARBONATE_LABELS, carbonate_class


@pytest.fixture
def split_plan(a1_demo_dir):
    return str(a1_demo_dir / "split_assignments.parquet")


@pytest.fixture
def report(a1_settings, split_plan, tmp_path):
    """One full pre-flight run on the demo data, with its split."""
    return run_preflight(a1_settings(), split_plan_path=split_plan, out_dir=str(tmp_path / "preflight"))


# --- end to end ------------------------------------------------------------------------


def test_every_table_and_the_report_are_written(report):
    names = {path.name for path in report.files}
    assert names == {
        COMPLETENESS_FILE,
        MISSINGNESS_FILE,
        REGION_BY_CARBONATE_FILE,
        SMALL_CELLS_FILE,
        REGION_BY_YEAR_FILE,
        PREDICTIONS_FILE,
        INPUTS_FILE,
        REPORT_FILE,
    }
    assert all(path.exists() for path in report.files)
    assert report.files[-1].name == REPORT_FILE
    assert report.skipped == {}


def test_the_report_has_its_four_sections_and_says_what_it_was_run_on(report, split_plan):
    text = (report.out_dir / REPORT_FILE).read_text()
    for heading in ("## 1. Completeness", "## 2. Regions x FAO carbonate class", "## 3.", "## 4."):
        assert heading in text
    assert split_plan in text
    assert "flagged under 30 points" in text


def test_every_point_is_counted_once_in_the_region_by_carbonate_table(report, a1_demo_n_points):
    table = pd.read_csv(report.out_dir / REGION_BY_CARBONATE_FILE, index_col=0)

    assert table.loc[TOTAL, TOTAL] == a1_demo_n_points
    assert list(table.columns) == [*FAO_CARBONATE_LABELS, TOTAL]
    body = table.drop(index=TOTAL, columns=TOTAL)
    assert (body.sum(axis=1) == table[TOTAL].drop(TOTAL)).all()
    assert (body.sum(axis=0) == table.loc[TOTAL].drop(TOTAL)).all()


def test_the_counts_match_the_carbonate_classes_of_the_data(report, a1_demo_dir):
    targets = pd.read_csv(a1_demo_dir / "targets.csv")
    expected = pd.crosstab(targets["region"], carbonate_class(targets["caco3_pct_total"]))
    table = report.tables[REGION_BY_CARBONATE_FILE]

    assert (table.loc[expected.index, list(expected.columns)].to_numpy() == expected.to_numpy()).all()


def test_the_thin_cells_are_flagged_and_marked_in_the_report(report):
    cells = report.small_cells
    assert not cells.empty
    assert (cells["n"] < 30).all()

    text = (report.out_dir / REPORT_FILE).read_text()
    for row in cells.itertuples():
        assert f"- {row.region}, {row.carbonate_class}: {row.n}" in text
    table_rows = [line for line in text.splitlines() if line.startswith("| ")]
    assert sum(line.count(" (!)") for line in table_rows) == len(cells)


def test_no_cell_at_or_above_the_threshold_is_flagged(report):
    table = report.tables[REGION_BY_CARBONATE_FILE].drop(index=TOTAL, columns=TOTAL)
    thin = {(row.region, row.carbonate_class) for row in report.small_cells.itertuples()}
    for region in table.index:
        for label in table.columns:
            assert ((region, label) in thin) == (table.loc[region, label] < 30)


def test_a_higher_threshold_flags_more_cells(a1_settings, split_plan, tmp_path):
    result = run_preflight(a1_settings(), split_plan_path=split_plan, out_dir=str(tmp_path), min_cell_points=10_000)
    assert len(result.small_cells) == 4 * len(FAO_CARBONATE_LABELS)


def test_the_region_by_year_table_adds_up(report, a1_demo_dir, a1_demo_n_points):
    table = report.tables[REGION_BY_YEAR_FILE]
    targets = pd.read_csv(a1_demo_dir / "targets.csv")

    assert table.loc[TOTAL, TOTAL] == a1_demo_n_points
    assert sorted(column for column in table.columns if column != TOTAL) == sorted(targets["sample_year"].unique())
    assert table.loc[TOTAL].drop(TOTAL).to_dict() == targets["sample_year"].value_counts().to_dict()


def test_the_leaky_demo_predictions_are_flagged(report):
    assert report.overfit == ["c_e_c_meq_100g", "caco3_pct_total"]
    table = report.tables[PREDICTIONS_FILE]
    assert set(table["split"]) == {"train", "val", "test"}
    assert table["flag_overfit"].all()
    assert (table["r2_gap"] > 0.1).all()
    assert "**Flagged: c_e_c_meq_100g, caco3_pct_total.**" in (report.out_dir / REPORT_FILE).read_text()


def test_a_lenient_gap_flags_nothing(a1_settings, split_plan, tmp_path):
    result = run_preflight(a1_settings(), split_plan_path=split_plan, out_dir=str(tmp_path), overfit_r2_gap=0.9)
    assert result.overfit == []
    assert "No property is flagged" in (tmp_path / REPORT_FILE).read_text()


def test_without_a_split_plan_the_predictions_are_skipped_with_the_reason(a1_settings, tmp_path):
    result = run_preflight(a1_settings(), out_dir=str(tmp_path))

    assert "split plan" in result.skipped["auxiliary_predictions"]
    assert PREDICTIONS_FILE not in result.tables
    assert "Skipped: no split plan was given" in (tmp_path / REPORT_FILE).read_text()


def test_without_the_prediction_columns_the_check_is_skipped_and_says_which(a1_settings, split_plan, tmp_path):
    settings = a1_settings(preflight={"prediction_columns": {"caco3_pct_total": "caco3_pct_total__nowhere"}})
    result = run_preflight(settings, split_plan_path=split_plan, out_dir=str(tmp_path))

    assert "caco3_pct_total__nowhere" in result.skipped["auxiliary_predictions"]
    assert "`caco3_pct_total__nowhere` (caco3_pct_total)" in (tmp_path / REPORT_FILE).read_text()


def test_without_regions_both_region_tables_are_skipped(a1_settings, split_plan, tmp_path):
    result = run_preflight(a1_settings(regions={"source": None}), split_plan_path=split_plan, out_dir=str(tmp_path))

    assert set(result.skipped) == {"region_by_carbonate", "region_by_year"}
    assert REGION_BY_CARBONATE_FILE not in result.tables
    assert result.small_cells.empty
    # The other two sections still run.
    assert COMPLETENESS_FILE in result.tables and PREDICTIONS_FILE in result.tables


def test_without_a_sampling_year_only_the_year_table_is_skipped(a1_settings, tmp_path):
    result = run_preflight(a1_settings(sample_year={"column": None}), out_dir=str(tmp_path))

    assert "sample_year" in result.skipped["region_by_year"]
    assert REGION_BY_CARBONATE_FILE in result.tables


def test_points_the_split_does_not_cover_are_counted_not_scored(a1_settings, split_plan, tmp_path):
    plan = pd.read_parquet(split_plan)
    partial = tmp_path / "partial.parquet"
    plan.iloc[10:].to_parquet(partial, index=False)

    result = run_preflight(a1_settings(), split_plan_path=str(partial), out_dir=str(tmp_path / "out"))

    assert result.tables[PREDICTIONS_FILE].groupby("property")["n"].sum().eq(len(plan) - 10).all()
    assert "10 point(s) of the data are not in it" in (tmp_path / "out" / REPORT_FILE).read_text()


def test_the_command_line_writes_to_the_folder_it_is_given(a1_settings, split_plan, tmp_path):
    path = a1_settings(as_file=True)

    result = preflight_main(["--config", str(path), "--split-plan", split_plan, "--out", str(tmp_path / "cli")])

    assert (tmp_path / "cli" / REPORT_FILE).exists()
    assert result.overfit == ["c_e_c_meq_100g", "caco3_pct_total"]


# --- the tables, one by one --------------------------------------------------------------


def test_completeness_counts_each_column_and_the_points_with_all_of_them():
    frame = pd.DataFrame({"som": [1, 2, 3, 4], "caco3": [1, np.nan, 3, 4], "cec": [1, 2, np.nan, 4]})
    table = completeness(frame, ["som", "caco3", "cec"])

    assert table["n_measured"].to_dict() == {"som": 4, "caco3": 3, "cec": 3, ALL_MEASURED: 2}
    assert table.loc[ALL_MEASURED, "share_measured"] == 0.5


def test_missingness_lists_the_emptiest_column_first():
    frame = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 2, np.nan], "c": [1, 2, 3]})
    table = missingness(frame, {"a": "static covariate", "b": "lab value", "c": "lab value", "absent": "x"})

    assert table["column"].tolist() == ["a", "b", "c"]
    assert table["n_missing"].tolist() == [2, 1, 0]
    assert table["kind"].tolist() == ["static covariate", "lab value", "lab value"]


def test_a_point_without_a_region_or_a_class_is_still_counted():
    regions = pd.Series(["A", None, "B", "A"], name="region")
    classes = carbonate_class(pd.Series([1.0, 30.0, np.nan, 12.0]))
    table = count_table(regions, classes, row_missing=NO_REGION, column_missing=NO_CACO3)

    assert table.loc[TOTAL, TOTAL] == 4
    assert table.index.tolist() == ["A", "B", NO_REGION, TOTAL]
    assert table.columns.tolist() == [*FAO_CARBONATE_LABELS, NO_CACO3, TOTAL]
    assert table.loc["B", NO_CACO3] == 1
    assert table.loc[NO_REGION, ">=25"] == 1


def test_an_empty_class_still_gets_its_column():
    table = count_table(
        pd.Series(["A"], name="region"),
        carbonate_class(pd.Series([5.0])),
        row_missing=NO_REGION,
        column_missing=NO_CACO3,
    )
    assert table.columns.tolist() == [*FAO_CARBONATE_LABELS, TOTAL]
    assert table.loc["A", "<2"] == 0


def test_small_cells_ignore_the_totals_and_the_points_without_a_region_or_a_class():
    regions = pd.Series(["A"] * 40 + [None] * 2, name="region")
    classes = carbonate_class(pd.Series([5.0] * 39 + [np.nan] + [5.0] * 2))
    table = count_table(regions, classes, row_missing=NO_REGION, column_missing=NO_CACO3)

    cells = small_cells(table, 30, list(FAO_CARBONATE_LABELS))

    assert set(cells["region"]) == {"A"}
    assert set(cells["carbonate_class"]) == {"<2", "10-25", ">=25"}
    assert (cells["n"] == 0).all()


def _scored_frame(train_noise, test_noise, n=300, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 50, n)
    splits = pd.Series(np.where(np.arange(n) < 200, "train", np.where(np.arange(n) < 250, "val", "test")))
    noise = np.where(splits == "train", train_noise, test_noise)
    frame = pd.DataFrame({"caco3": truth, "caco3_pred": truth + rng.normal(0, 1, n) * noise})
    return frame, splits


def test_predictions_as_good_everywhere_are_not_flagged():
    frame, splits = _scored_frame(train_noise=5.0, test_noise=5.0)
    table = prediction_scores(frame, {"caco3": "caco3_pred"}, splits, overfit_r2_gap=0.1)

    assert not table["flag_overfit"].any()
    assert table.set_index("split").loc["test", "n"] == 50


def test_predictions_far_better_on_the_training_points_are_flagged():
    frame, splits = _scored_frame(train_noise=1.0, test_noise=10.0)
    table = prediction_scores(frame, {"caco3": "caco3_pred"}, splits, overfit_r2_gap=0.1)

    assert table["flag_overfit"].all()
    scores = table.set_index("split")
    assert scores.loc["train", "r2"] - scores.loc["test", "r2"] == pytest.approx(scores.loc["train", "r2_gap"])


def test_a_split_too_small_to_score_gives_no_flag_rather_than_a_false_one():
    frame, _ = _scored_frame(train_noise=1.0, test_noise=10.0, n=10)
    splits = pd.Series(["train"] * 9 + ["test"])
    table = prediction_scores(frame, {"caco3": "caco3_pred"}, splits, overfit_r2_gap=0.1)

    assert np.isnan(table.set_index("split").loc["test", "r2"])
    assert not table["flag_overfit"].any()
