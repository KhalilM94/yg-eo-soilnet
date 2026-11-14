"""How SOM relates to CaCO3, CEC and the covariates in each held-out basin against the rest of the data.

The relations must be measured on the right points (the basin, and every other point), with
intervals that make sense, and corrected within the two families the paper reads them in.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.a1.basin_relations import main as basin_relations_main
from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.basin_relations import (
    ALL_CLASSES,
    EXPLORATORY,
    PRIMARY,
    basin_relations,
    load_basin_data,
    som_by_class,
)
from yg_eo_soilnet.a1.figures.core import _spearman_columns

N_BOOT = 60


@pytest.fixture(scope="module")
def data(a1_demo_settings):
    return load_basin_data(a1_demo_settings)


@pytest.fixture(scope="module")
def relations(data, a1_demo_settings):
    return basin_relations(data, min_cell_size=a1_demo_settings.min_cell_size, n=N_BOOT, seed=7)


def test_caco3_and_cec_come_first_then_the_figure_3_covariates(data, a1_demo_settings):
    assert data.primary == list(a1_demo_settings.auxiliary.targets)
    assert data.pairs[:2] == data.primary
    assert 0 < len(data.pairs) - 2 <= 15
    assert set(data.basins) == set(a1_demo_settings.plans.logo.held_out)


def test_the_estimates_are_spearman_rho_on_the_basin_and_on_every_other_point(data, relations):
    basin, ids = next(iter(data.basins.items()))
    rest = data.frame.index.difference(ids)
    for side, points in (("basin", ids), ("complement", rest)):
        part = data.frame.loc[points]
        rho = _spearman_columns(part[data.target].to_numpy(float), part[data.pairs].to_numpy(float))
        rows = relations[(relations["basin"] == basin) & (relations["class"] == ALL_CLASSES)].set_index("variable")
        np.testing.assert_allclose(rows.loc[data.pairs, f"rho_{side}"].to_numpy(float), rho)
        assert (rows[f"n_{side}"] == len(points)).all()


def test_the_intervals_contain_the_point_estimate(relations):
    for side in ("basin", "complement"):
        rows = relations.dropna(subset=[f"rho_{side}", f"low_{side}", f"high_{side}"])
        assert not rows.empty
        assert (rows[f"low_{side}"] <= rows[f"rho_{side}"] + 1e-12).all(), side
        assert (rows[f"rho_{side}"] <= rows[f"high_{side}"] + 1e-12).all(), side
    rows = relations.dropna(subset=["difference", "low_difference", "high_difference"])
    assert (
        (rows["low_difference"] <= rows["difference"] + 1e-12) & (rows["difference"] <= rows["high_difference"] + 1e-12)
    ).all()


def test_within_class_rows_only_where_both_sides_hold_enough_points(relations, a1_demo_settings):
    within = relations[relations["class"] != ALL_CLASSES]
    assert (within["n_basin"] >= a1_demo_settings.min_cell_size).all()
    assert (within["n_complement"] >= a1_demo_settings.min_cell_size).all()


def test_two_holm_families_per_basin(relations, data):
    for basin, part in relations.groupby("basin"):
        primary = part[part["family"] == PRIMARY]
        assert sorted(primary["variable"]) == sorted(data.primary)
        assert (primary["class"] == ALL_CLASSES).all()
        assert set(part["family"]) == {PRIMARY, EXPLORATORY}
        for _, family in part.groupby("family"):
            np.testing.assert_allclose(family["p_holm"], evaluation.holm(family["p_value"].to_numpy()))


def test_som_by_class_counts_and_medians(data, a1_demo_settings):
    table = som_by_class(data, a1_demo_settings.carbonate.column)
    basin, ids = next(iter(data.basins.items()))
    part = data.frame.loc[ids]
    classes = data.classes.reindex(part.index)
    for label in data.labels:
        row = table[(table["basin"] == basin) & (table["class"] == label)].iloc[0]
        chosen = part[(classes == label).to_numpy()]
        assert row["n"] == len(chosen)
        if len(chosen):
            assert row["som_median"] == pytest.approx(chosen[data.target].median())
            assert row["som_iqr"] == pytest.approx(row["som_q75"] - row["som_q25"])


def test_the_script_writes_the_tables_and_figure(a1_settings, tmp_path):
    path = a1_settings(as_file=True, output_dir=str(tmp_path / "out"))
    tables = basin_relations_main(["--config", str(path), "--n-boot", "20"])
    out = tmp_path / "out" / "basin_relations"
    for name in ("basin_relations.csv", "basin_som_by_class.csv", "basin_som_by_class.tex", "captions.md"):
        assert (out / name).exists(), name
    assert (out / "figC3_basin_relations.png").exists() and (out / "figC3_basin_relations.pdf").exists()
    assert "family" in pd.read_csv(out / "basin_relations.csv").columns
    assert set(tables) == {"basin_relations", "basin_som_by_class"}
