"""The check of the points absent from the static-covariate export (no model fitted).

The absent points must be the ones missing from the export file; the error table must split each
design, model and basin into absent and other points with a block-bootstrap interval that finds a
real difference and no difference where there is none; the transfer loss must be recomputed on the
same points without them; and the relations must drop them everywhere while a covariate already
missing on them keeps its correlation.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.a1 import missing_static as ms
from yg_eo_soilnet.a1.basin_relations import ALL_CLASSES, BasinData, basin_relations
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL_GLOBAL

N_BOOT = 200


def _points(n=400, seed=0):
    rng = np.random.default_rng(seed)
    regions = np.where(np.arange(n) < n // 2, "North", "South")
    return pd.DataFrame(
        {
            "point_id": [f"p{i:03d}" for i in range(n)],
            "region": regions,
            # Spread over about 1 degree, so there are many 10 km blocks.
            "lat": 33.0 + rng.uniform(0, 1.0, n),
            "lon": -6.0 + rng.uniform(0, 1.0, n),
            "observed": rng.gamma(4.0, 5.0, n),
        }
    )


def _rows(points, predicted, *, model, kind, plan_id, held_out=None, block_size_m=np.nan, seed=42):
    return points.assign(
        predicted=predicted,
        model=model,
        kind=kind,
        plan_id=plan_id,
        held_out=held_out,
        block_size_m=block_size_m,
        seed=seed,
        split="test",
        covariate_group=COMBINED_GROUP,
    )


@pytest.fixture(scope="module")
def setting():
    points = _points()
    absent = pd.Index(points["point_id"].iloc[::5])  # every fifth point
    is_absent = points["point_id"].isin(absent).to_numpy()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1.0, len(points))
    # Absent points: three times the error everywhere except the random split.
    worse = points["observed"] + noise * np.where(is_absent, 3.0, 1.0)
    same = points["observed"] + noise
    north = points[points["region"] == "North"]
    frame = pd.concat(
        [
            _rows(points, same, model=GLOBAL, kind=RANDOM, plan_id="random"),
            _rows(points, worse, model=GLOBAL, kind=SPATIAL_KFOLD, plan_id="kfold_10km_f0", block_size_m=10000.0),
            _rows(points, worse + 5, model=GLOBAL, kind=SPATIAL_KFOLD, plan_id="kfold_50km_f0", block_size_m=50000.0),
            _rows(north, worse[north.index] + 2.0, model=GLOBAL, kind=LOGO, plan_id="logo_r01_north", held_out="North"),
            _rows(north, worse[north.index], model=LOCAL_GLOBAL, kind=LOGO, plan_id="logo_r01_north", held_out="North"),
            _rows(points, same + 9, model=GLOBAL, kind=RANDOM, plan_id="random", seed=7),  # not the first seed
        ],
        ignore_index=True,
    )
    return SimpleNamespace(points=points, absent=absent, frame=frame)


def test_absent_points_are_those_missing_from_the_export(tmp_path):
    path = tmp_path / "export.csv"
    pd.DataFrame({"uuid": ["a", "b", "c"], "elevation": [1, 2, 3]}).to_csv(path, index=False)
    exported = ms.export_ids(path)
    assert list(ms.absent_points(["a", "d", "c", "e", "d"], exported)) == ["d", "e"]


def test_the_error_table_splits_each_design_model_and_basin(setting):
    table = ms.missing_static_table(setting.frame, setting.absent, first_seed=42, n=N_BOOT)
    assert list(table.columns) == ms.COLUMNS
    designs = set(table["design"])
    assert designs == {ms.DESIGN_RANDOM, ms.DESIGN_10KM, f"{ms.HELD_OUT}North"}  # the 50 km folds are not read
    pooled = table[(table["design"] == ms.DESIGN_10KM) & (table["basin"] == ms.ALL_BASINS)].iloc[0]
    assert pooled["n_absent"] == 80 and pooled["n_other"] == 320
    assert pooled["rmse_difference"] > 0 and pooled["rmse_low"] > 0  # three times the error: found
    random = table[(table["design"] == ms.DESIGN_RANDOM) & (table["basin"] == ms.ALL_BASINS)].iloc[0]
    assert random["rmse_low"] < 0 < random["rmse_high"]  # the same error: not found
    assert random["n_absent"] + random["n_other"] == 400  # the other seed is not read
    held = table[table["design"] == f"{ms.HELD_OUT}North"]
    assert set(held["model"]) == {GLOBAL, LOCAL_GLOBAL}
    assert set(held["basin"]) == {"North"}  # one region: no pooled row
    glob = held[held["model"] == GLOBAL].iloc[0]
    assert glob["bias_absent"] == pytest.approx(glob["bias_other"] + glob["bias_difference"])


def test_a_cell_without_absent_points_has_no_interval(setting):
    frame = setting.frame[setting.frame["kind"] == RANDOM]
    table = ms.missing_static_table(frame, pd.Index([]), first_seed=42, n=N_BOOT)
    assert (table["n_absent"] == 0).all()
    assert table[["rmse_difference", "rmse_low", "rmse_p_value"]].isna().all().all()


def test_the_transfer_loss_is_recomputed_without_the_absent_points(setting):
    run = SimpleNamespace(name="a1_xgb", predictions=setting.frame[setting.frame["kind"] == LOGO].copy())
    table = ms.transfer_without(run, setting.absent, n=N_BOOT)
    assert list(table["points"]) == ["all", "without absent"]
    n_all, n_without = table["n"].tolist()
    assert n_all == 200 and n_without == 160
    # Global is the local model + 2 on every point: the loss stays, with or without them.
    assert (table["difference"] > 0).all()
    assert len(run.predictions) == 400  # the run read is left as it was


def _basin_data(n=600, seed=3):
    rng = np.random.default_rng(seed)
    ids = pd.Index([f"p{i:03d}" for i in range(n)])
    som = rng.gamma(4.0, 5.0, n)
    static = som * 0.5 + rng.normal(0, 3, n)
    series = som * 0.3 + rng.normal(0, 3, n)
    absent = ids[::5]  # both classes, as every fifth point alternates
    frame = pd.DataFrame({"som": som, "elevation": static, "S2_B4_q50": series}, index=ids)
    frame.loc[absent, "elevation"] = np.nan  # a static covariate: missing on the absent points
    classes = pd.Series(np.where(np.arange(n) % 2 == 0, "<2", ">=25"), index=ids)
    blocks = pd.Series(np.arange(n) // 4, index=ids)
    data = BasinData(
        frame=frame,
        target="som",
        pairs=["elevation", "S2_B4_q50"],
        primary=[],
        classes=classes,
        labels=["<2", ">=25"],
        blocks=blocks,
        basins={"Sebou": ids[: n // 2], "Tensift": ids[n // 2 :]},
    )
    return data, absent


def test_relations_drop_the_absent_points_everywhere_and_keep_a_static_correlation():
    data, absent = _basin_data()
    before = basin_relations(data, min_cell_size=30, n=50, seed=42)
    after = ms.relations_without(data, absent, basins=["Sebou"], min_cell_size=30, n=50, seed=42)
    assert set(after["basin"]) == {"Sebou"}
    cell = after[(after["class"] == ALL_CLASSES) & (after["variable"] == "elevation")].iloc[0]
    old = before[(before["basin"] == "Sebou") & (before["class"] == ALL_CLASSES) & (before["variable"] == "elevation")]
    assert cell["n_basin"] == 300 - len(absent.intersection(data.basins["Sebou"]))
    # The static covariate was already computed without the absent points: same correlations.
    assert cell["rho_basin"] == pytest.approx(old["rho_basin"].iloc[0])
    assert cell["rho_complement"] == pytest.approx(old["rho_complement"].iloc[0])
    counts = ms.within_class_counts(before, after)
    assert set(counts["points"]) == {"all", "without absent"}
    assert set(counts["class"]) == {"<2", ">=25"} and set(counts["basin"]) == {"Sebou"}
    first = counts[(counts["points"] == "without absent")]
    assert (first["n_basin"] < counts[counts["points"] == "all"]["n_basin"].to_numpy()).all()
