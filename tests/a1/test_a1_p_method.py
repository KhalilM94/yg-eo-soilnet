"""How basin_relations computes a difference's p-value: Fisher's z over the bootstrap SE, or the share of draws.

With n draws, the share of draws on either side of zero cannot go below 2/n (or is exactly 0), so
under an 83-test Holm family (first threshold 0.05/83 = 0.0006) a test passes only when no draw
crosses zero. The normal approximation must not have that floor, must reproduce a hand-computed z,
must reach the power check, and the comparison table must flag every test whose decision differs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from yg_eo_soilnet.a1 import robustness
from yg_eo_soilnet.a1.basin_relations import (
    ALL_CLASSES,
    BasinData,
    basin_relations,
    difference_p_value,
    fisher_z,
    method_comparison,
    p_method_text,
    significant_counts,
)
from yg_eo_soilnet.a1.figures.robustness import offset_share
from yg_eo_soilnet.a1.figures.transfer import smallest_share_text


def test_the_normal_p_value_is_fishers_z_over_the_bootstrap_se():
    rng = np.random.default_rng(0)
    b = 0.5 + 0.1 * rng.standard_normal(500)
    c = 0.3 + 0.1 * rng.standard_normal(500)
    gaps = np.arctanh(b) - np.arctanh(c)
    z = (np.arctanh(0.5) - np.arctanh(0.3)) / np.std(gaps, ddof=1)
    assert difference_p_value(0.5, 0.3, b, c, "normal") == pytest.approx(2 * norm.sf(abs(z)))
    # The same difference the other way round: the same p.
    assert difference_p_value(0.3, 0.5, c, b, "normal") == pytest.approx(2 * norm.sf(abs(z)))


def test_the_draws_rule_is_unchanged_and_has_a_floor_the_normal_one_does_not():
    rng = np.random.default_rng(1)
    b = 0.6 + 0.04 * rng.standard_normal(1000)
    c = 0.1 + 0.04 * rng.standard_normal(1000)
    # Every draw on one side: the share is 0, a value the draws cannot tell from "smaller than 1/n".
    assert difference_p_value(0.6, 0.1, b, c, "draws") == 0.0
    assert difference_p_value(0.6, 0.1, b, c, "normal") < 0.05 / 83
    # One draw of 1,000 across zero: the smallest nonzero p of the draws, 2/n.
    b2 = b.copy()
    b2[0] = c[0] - 0.01
    assert difference_p_value(0.6, 0.1, b2, c, "draws") == pytest.approx(0.002)
    assert difference_p_value(0.6, 0.1, b2, c, "draws") > 0.05 / 83  # can never pass Holm's first step


def test_fisher_z_stays_finite_at_a_perfect_correlation_and_bad_methods_are_refused():
    assert np.isfinite(fisher_z([1.0, -1.0])).all()
    with pytest.raises(ValueError, match="p_method"):
        difference_p_value(0.1, 0.2, [0.1, 0.2], [0.2, 0.3], "exact")
    assert np.isnan(difference_p_value(0.1, 0.2, [0.1], [0.2], "normal"))  # one draw: no SE


def _data(n=480, seed=2):
    rng = np.random.default_rng(seed)
    ids = pd.Index([f"p{i:03d}" for i in range(n)])
    som = rng.gamma(4.0, 5.0, n)
    frame = pd.DataFrame(
        {"som": som, "x": som * np.where(np.arange(n) < n // 2, 0.8, 0.1) + rng.normal(0, 4, n)}, index=ids
    )
    return BasinData(
        frame=frame,
        target="som",
        pairs=["x"],
        primary=[],
        classes=pd.Series(np.where(np.arange(n) % 2 == 0, "<2", ">=25"), index=ids),
        labels=["<2", ">=25"],
        blocks=pd.Series(np.arange(n) // 3, index=ids),
        basins={"A": ids[: n // 2]},
    )


def test_basin_relations_records_its_method_and_the_captions_say_it():
    data = _data()
    normal = basin_relations(data, min_cell_size=30, n=100, seed=42)
    draws = basin_relations(data, min_cell_size=30, n=100, seed=42, p_method="draws")
    assert set(normal["p_method"]) == {"normal"} and set(draws["p_method"]) == {"draws"}
    # Same estimates and intervals: only the p-value changes.
    for column in ("difference", "low_difference", "high_difference"):
        assert np.allclose(normal[column], draws[column])
    assert "Fisher" in p_method_text(normal) and "resolution-limited" in p_method_text(draws)
    with pytest.raises(ValueError, match="p_method"):
        basin_relations(data, n=10, p_method="exact")


def test_the_power_check_uses_the_method_and_warns_on_a_mixed_family(caplog):
    data = _data()
    relations = basin_relations(data, min_cell_size=30, n=50, seed=42)
    draws = robustness.subsample_relations(data, relations, n_sub=60, n_draws=2, n_boot=50, p_method="normal")
    assert set(draws["p_method"]) == {"normal"} and len(draws)
    with caplog.at_level("WARNING"):
        robustness.subsample_relations(data, relations, n_sub=60, n_draws=1, n_boot=20, p_method="draws")
    assert "mixes the two" in caplog.text


def _table(p_holm, method):
    return pd.DataFrame(
        {
            "basin": ["A", "A", "A", "B"],
            "class": [ALL_CLASSES, "<2", ">=25", "<2"],
            "variable": ["x", "x", "y", "x"],
            "family": ["exploratory"] * 4,
            "difference": [0.3, 0.2, -0.4, 0.1],
            "p_value": [p / 10 for p in p_holm],
            "p_holm": p_holm,
            "p_method": method,
        }
    )


def test_the_comparison_flags_every_test_whose_holm_decision_differs():
    tables = {
        "normal, 1,000": _table([0.01, 0.20, 0.001, 0.9], "normal"),
        "draws, 1,000 (resolution-limited)": _table([0.0, 0.0, 0.13, 0.9], "draws"),
    }
    comparison = method_comparison(tables)
    flagged = comparison[comparison["differs"]][["basin", "class", "variable"]].values.tolist()
    assert flagged == [["A", "<2", "x"], ["A", ">=25", "y"]]
    counts = significant_counts(comparison).set_index("basin")
    assert counts.loc["A", "normal, 1,000"] == 1  # within-class only: the "all" row is not counted
    assert counts.loc["A", "draws, 1,000 (resolution-limited)"] == 1
    assert counts.loc["A", "n_differ"] == 2 and counts.loc["B", "n_differ"] == 0


def test_figure_9_quotes_pooled_difference_at_the_smallest_share():
    rings = pd.DataFrame(
        {
            "region": ["Oum Er Rbia"] * 3,
            "fraction": [0.05, 0.05, 0.1],
            "a": ["mixed", "mixed_weighted", "mixed_weighted"],
            "estimate": [-1.69, -1.95, -0.77],
            "low": [-2.39, -2.36, -1.11],
            "high": [-0.85, -1.49, -0.45],
            "n_repeats_better": [5, 5, 5],
            "n_repeats": [5, 5, 5],
        }
    )
    text = smallest_share_text(rings, "Oum Er Rbia")
    assert "weighted − control at 5 %: −1.95 [−2.36, −1.49], 5/5 repeats better" in text
    assert "mixed (pooled) − control at 5 %: −1.69" in text and "10 %" not in text
    assert smallest_share_text(rings, "Sebou") == ""


def test_figure_c6_gives_the_share_of_the_loss_an_offset_removes():
    assert offset_share({"rmse_uncorrected": 11.88, "rmse_offset": 12.00, "rmse_local": 9.88}) < 0  # worse
    assert offset_share({"rmse_uncorrected": 11.66, "rmse_offset": 11.03, "rmse_local": 9.50}) == pytest.approx(
        0.63 / 2.16
    )
    assert offset_share({"rmse_uncorrected": 7.96, "rmse_offset": 8.0, "rmse_local": 8.18}) is None  # no loss


def test_the_script_compares_tables_already_written_without_computing(tmp_path):
    from pathlib import Path

    from experiments.a1.basin_relations import main

    config = Path(__file__).resolve().parents[2] / "configs" / "a1" / "a1_demo.yml"
    normal, draws = tmp_path / "normal.csv", tmp_path / "draws.csv"
    _table([0.01, 0.20, 0.001, 0.9], "normal").to_csv(normal, index=False)
    _table([0.0, 0.0, 0.13, 0.9], "draws").to_csv(draws, index=False)
    out = tmp_path / "out"
    tables = main(
        ["--config", str(config), "--out", str(out), "--compare", f"normal={normal}", "--compare", f"draws={draws}"]
    )
    assert (out / "p_method_comparison.csv").exists() and (out / "p_method_counts.csv").exists()
    assert int(tables["p_method_comparison"]["differs"].sum()) == 2
    with pytest.raises(SystemExit, match="LABEL=PATH"):
        main(["--config", str(config), "--out", str(out), "--compare", str(normal)])
