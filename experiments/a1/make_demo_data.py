"""Write a made-up dataset for trying the Paper A1 experiments without the real data.

It extends the project's demo dataset (``examples/make_demo_data.py``), whose static covariates and
monthly time series it reuses, with what A1 needs and the ordinary demo lacks:

- **total CaCO3** (``caco3_pct_total``, %), high where the climate is dry, spread over all four FAO
  carbonate classes. Carbonate brightens the bare-soil reflectance, so it can be predicted from
  the covariates.
- **CEC** (``c_e_c_meq_100g``), from clay and organic matter.
- **four regions** - the quarters of the area - as a ``region`` column and as outlines in
  ``regions.geojson`` (field ``name``), the two ways A1 can be told about regions.
- **organic matter** that depends on carbonate and differs from region to region, so carbonate
  stratification (b) and transfer between regions (c, d) have a pattern to find. As in the real
  soils, organic matter darkens the shortwave-infrared reflectance only of non-calcareous soils.
- **a sampling year** (``sample_year``, 2017-2020).
- **earlier predictions of CaCO3 and CEC** (``caco3_pct_total__TabICL``, ``c_e_c_meq_100g__TabICL``)
  that are deliberately **leaky**: much closer to the truth on the training points of the saved
  split than on the others, as if the model had been fitted on them. They are there so the
  pre-flight check that looks for exactly that has something to find.
- **the split** those predictions leak through, saved as ``split_assignments.parquet`` in the same
  form as a run's ``data_splits/split_assignments.parquet``.

The region is written to both ``static.csv`` and ``targets.csv``: A1 reads it from either, and a
``split.group`` strategy that groups by it (``ColumnGroupStrategy``) needs it in the static file.

Nothing in it is a real measurement. Five files are written to ``experiments/a1/demo_data/``:
``static.csv``, ``targets.csv``, ``timeseries.csv``, ``regions.geojson`` and
``split_assignments.parquet``. Run it from the repository root::

    pixi run -e core python experiments/a1/make_demo_data.py
    # or: python experiments/a1/make_demo_data.py --n-points 1200 --seed 0

The same seed always produces the same files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.make_demo_data import make_static, make_targets, make_timeseries  # noqa: E402
from yg_eo_soilnet.datamodules.splitting import TRAIN, UnifiedSplitter  # noqa: E402

OUTPUT_FOLDER = Path(__file__).resolve().parent / "demo_data"

#: Where the area is cut into its four regions, in degrees. The ordinary demo's points cover
#: latitude 30.5-34.5 and longitude -9 to -5.
REGION_SPLIT_LAT = 32.5
REGION_SPLIT_LON = -7.0
#: The extent the region outlines cover, a little wider than the points (west, south, east, north).
REGION_EXTENT = (-9.5, 30.0, -4.5, 35.0)

#: Each region's own organic matter: an offset (g/kg) and how strongly vegetation vigour drives it.
#: The regions differ, so a model learned in one does not transfer perfectly to another, and the
#: more different two regions are the more it loses.
REGION_EFFECTS = {
    "North-West": {"offset": 4.0, "vigour_slope": 40.0},
    "North-East": {"offset": 1.0, "vigour_slope": 34.0},
    "South-West": {"offset": -1.0, "vigour_slope": 30.0},
    "South-East": {"offset": -4.0, "vigour_slope": 24.0},
}

#: The bare-soil Sentinel-2 bands carbonate brightens, and the two shortwave-infrared ones organic
#: matter darkens.
BAREST_BANDS = ("s2_b2_barest", "s2_b4_barest", "s2_b8_barest", "s2_b11_barest", "s2_b12_barest")
SWIR_BANDS = ("s2_b11_barest", "s2_b12_barest")
#: Below this CaCO3 (%), organic matter shows in the reflectance; above it, carbonate masks it.
SOM_VISIBLE_BELOW_CACO3 = 10.0

#: The years the demo points were sampled in.
SAMPLE_YEARS = (2017, 2018, 2019, 2020)

#: The split the leaky predictions leak through: the sizes and seed of
#: configs/a1/demo_main_config.yml, so it is the split an A1 run on the demo would draw.
SPLIT_SETTINGS = SimpleNamespace(
    SPLIT_HOLDOUT_STRATEGY="random", SPLIT_TEST_SIZE=0.15, SPLIT_VAL_SIZE=0.15, SPLIT_SEED=42
)
#: The leaky predictions' error, as a share of each property's spread: small on the training points,
#: as from a model that saw them, and three times larger elsewhere.
LEAK_NOISE_TRAIN = 0.2
LEAK_NOISE_HELD_OUT = 0.6
#: The properties given leaky predictions, and the suffix of their prediction columns.
LEAKY_PROPERTIES = ("caco3_pct_total", "c_e_c_meq_100g")
PREDICTION_SUFFIX = "__TabICL"


def region_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The region each point falls in: the quarter of the area it sits in.

    Examples
    --------
    >>> region_of(np.array([33.0, 31.0]), np.array([-8.0, -6.0])).tolist()
    ['North-West', 'South-East']
    """
    north = np.asarray(lat) >= REGION_SPLIT_LAT
    west = np.asarray(lon) < REGION_SPLIT_LON
    return np.where(north, np.where(west, "North-West", "North-East"), np.where(west, "South-West", "South-East"))


def region_polygons() -> dict:
    """The four regions as a GeoJSON FeatureCollection, with each region's name in ``name``.

    Returns
    -------
    dict
        Ready to be written with :func:`json.dump`.
    """
    west, south, east, north = REGION_EXTENT
    boxes = {
        "North-West": (west, REGION_SPLIT_LAT, REGION_SPLIT_LON, north),
        "North-East": (REGION_SPLIT_LON, REGION_SPLIT_LAT, east, north),
        "South-West": (west, south, REGION_SPLIT_LON, REGION_SPLIT_LAT),
        "South-East": (REGION_SPLIT_LON, south, east, REGION_SPLIT_LAT),
    }
    features = []
    for name, (x0, y0, x1, y1) in boxes.items():
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        features.append(
            {"type": "Feature", "properties": {"name": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
        )
    return {"type": "FeatureCollection", "features": features}


def make_a1_data(n_points: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Invent the four tables of the A1 demo dataset.

    Parameters
    ----------
    n_points : int
        How many sample points to create.
    seed : int
        The random seed; the same seed gives the same tables.

    Returns
    -------
    static : pandas.DataFrame
        One row per point: ``uuid`` and the static covariates (no coordinates).
    targets : pandas.DataFrame
        One row per point: ``uuid``, ``lat``, ``lon``, the lab values, ``region``, ``sample_year`` and
        the leaky predictions.
    timeseries : pandas.DataFrame
        One row per point per observed month.
    split : pandas.DataFrame
        The split the predictions leak through, as :meth:`SplitPlan.to_frame
        <yg_eo_soilnet.datamodules.splitting.SplitPlan.to_frame>` writes it.
    """
    rng = np.random.default_rng(seed)
    static = make_static(n_points, rng)
    timeseries, vigour = make_timeseries(static, rng)
    # Clay and pH as the ordinary demo has them; organic matter is redone below.
    base_targets = make_targets(static, vigour, rng)

    aridity = static["aridity_index"].to_numpy()
    # Low aridity index means a dry climate here, and dry soils are the calcareous ones. The spread
    # is wide enough that every FAO class holds a good share of the points.
    caco3 = np.clip(np.exp(np.log(8.0) - 6.0 * (aridity - 0.3) + rng.normal(0, 1.0, n_points)), 0.1, 60.0)

    regions = region_of(static["lat"].to_numpy(), static["lon"].to_numpy())
    offset = np.array([REGION_EFFECTS[name]["offset"] for name in regions])
    slope = np.array([REGION_EFFECTS[name]["vigour_slope"] for name in regions])
    # Calcareous soils hold more organic matter, as in the real sample (Paper A1 Figure 6).
    organic_matter = np.clip(
        4 + slope * vigour + 0.4 * static["twi"].to_numpy() + 0.60 * caco3 + offset + rng.normal(0, 3, n_points),
        2,
        90,
    )
    clay = base_targets["clay_pct"].to_numpy()
    cec = np.clip(2 + 0.5 * clay + 0.15 * organic_matter + rng.normal(0, 1.5, n_points), 1, 60)

    # Carbonate brightens every band; organic matter darkens the SWIR, but only where carbonate
    # does not mask it.
    static = static.copy()
    for band in BAREST_BANDS:
        static[band] = static[band] + 0.0015 * caco3
    visible = caco3 < SOM_VISIBLE_BELOW_CACO3
    for band in SWIR_BANDS:
        static[band] = static[band] - 0.0012 * organic_matter * visible
    for band in BAREST_BANDS:
        static[band] = static[band].round(4)

    targets = pd.DataFrame(
        {
            "uuid": static["uuid"],
            "lat": static["lat"],
            "lon": static["lon"],
            "organic_matter_g_kg": organic_matter.round(2),
            "clay_pct": base_targets["clay_pct"],
            "ph_water": base_targets["ph_water"],
            "caco3_pct_total": caco3.round(2),
            "c_e_c_meq_100g": cec.round(2),
            "region": regions,
        }
    )
    # Drawn after everything above, so adding them did not change any earlier value.
    targets["sample_year"] = rng.choice(SAMPLE_YEARS, size=n_points)
    split = demo_split(targets)
    targets = add_leaky_predictions(targets, split, rng)
    return static.drop(columns=["lat", "lon"]), targets, timeseries, split


def demo_split(targets: pd.DataFrame) -> pd.DataFrame:
    """Split the demo points the way an A1 run on the demo would, and return the saved form.

    Returns
    -------
    pandas.DataFrame
        ``point_id``, ``split``, ``lat`` and ``lon``, one row per point.
    """
    splitter = UnifiedSplitter(SPLIT_SETTINGS, logging.getLogger(__name__))
    coordinates = targets.set_index("uuid")[["lat", "lon"]]
    plan = splitter.build_plan(pd.Index(targets["uuid"]), coordinates=coordinates)
    return plan.to_frame()


def add_leaky_predictions(targets: pd.DataFrame, split: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a prediction column per property in :data:`LEAKY_PROPERTIES`, too good on the training points.

    Returns
    -------
    pandas.DataFrame
        ``targets`` with ``<property>__TabICL`` columns added.
    """
    in_training = targets["uuid"].map(split.set_index("point_id")["split"]).eq(TRAIN).to_numpy()
    noise_share = np.where(in_training, LEAK_NOISE_TRAIN, LEAK_NOISE_HELD_OUT)
    targets = targets.copy()
    for column in LEAKY_PROPERTIES:
        truth = targets[column].to_numpy()
        predicted = truth + rng.normal(0, 1, len(truth)) * noise_share * truth.std()
        targets[f"{column}{PREDICTION_SUFFIX}"] = np.clip(predicted, 0, None).round(2)
    return targets


def main(argv: list[str] | None = None) -> Path:
    """Generate the A1 demo dataset and write its four files.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    pathlib.Path
        The folder the files were written to.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-points", type=int, default=1200, help="How many sample points to invent (default 1200).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    parser.add_argument("--out", type=Path, default=OUTPUT_FOLDER, help="Folder to write into.")
    args = parser.parse_args(argv)

    static, targets, timeseries, split = make_a1_data(args.n_points, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # The region goes in the static file too: the split reads its groups from there (a column kept
    # only in the targets file never reaches it), so a ColumnGroupStrategy split can see it.
    static.assign(region=targets["region"].to_numpy()).to_csv(args.out / "static.csv", index=False)
    targets.to_csv(args.out / "targets.csv", index=False)
    timeseries.to_csv(args.out / "timeseries.csv", index=False)
    with open(args.out / "regions.geojson", "w") as handle:
        json.dump(region_polygons(), handle)
    split.to_parquet(args.out / "split_assignments.parquet", index=False)
    print(
        f"Wrote {len(static)} points and {len(timeseries)} monthly observations to {args.out}/ "
        "(static.csv, targets.csv, timeseries.csv, regions.geojson, split_assignments.parquet)."
    )
    return args.out


if __name__ == "__main__":
    main()
