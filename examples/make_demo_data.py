"""Write a small, made-up dataset for trying the pipeline without real data.

The real data lives outside this repository, so this script invents a dataset with the same layout
as the real one. Nothing in it is a real measurement.

Three files are written to ``examples/demo_data/``:

``static.csv``
    One row per sample point: an id (``uuid``) and values that do not change over time - terrain
    (elevation, slope, aspect, wetness index), aridity, bare-soil Sentinel-2 reflectance, a value
    from an existing clay map, and a landform class (a category: plain, hill or valley).
``targets.csv``
    One row per sample point: the id, its latitude/longitude, and three "lab measurements" the
    models learn to predict - organic matter (g/kg), clay (%) and pH in water.
``timeseries.csv``
    One row per point per month, 2021 to 2023, with Sentinel-2 values (``S2_`` columns) and climate
    values (``CLIM_`` columns). About one month in six is missing at random, as with real
    cloud-covered images, so points have different numbers of observations.

The lab values are simple formulas of the other columns plus random noise, so a model has a real
pattern to find: organic matter follows how green the vegetation gets (a time-series signal), clay
follows the clay map and the bare-soil shortwave-infrared band, and pH falls as aridity rises.

Run it from the repository root::

    pixi run -e core demo-data
    # or: python examples/make_demo_data.py --n-points 300 --seed 0

The same seed always produces the same files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FOLDER = Path(__file__).resolve().parent / "demo_data"
# The 15th of every month from January 2021 to December 2023 (36 months).
MONTHS = pd.date_range("2021-01-01", "2023-12-01", freq="MS") + pd.Timedelta(days=14)


def make_static(n_points: int, rng: np.random.Generator) -> pd.DataFrame:
    """Invent the per-point values that do not change over time.

    Parameters
    ----------
    n_points : int
        How many sample points to create.
    rng : numpy.random.Generator
        Source of random numbers, so the result is repeatable.

    Returns
    -------
    pandas.DataFrame
        One row per point, with ``uuid``, ``lat``, ``lon`` and the static columns.
    """
    lat = rng.uniform(30.5, 34.5, n_points)
    lon = rng.uniform(-9.0, -5.0, n_points)
    # Drier towards the south, wetter towards the north-west coast.
    aridity = np.clip(0.08 * (lat - 30.0) + 0.03 * (-5.0 - lon) + rng.normal(0, 0.05, n_points), 0.05, 0.65)
    elevation = np.clip(200 + 350 * (lon + 9.0) + rng.normal(0, 150, n_points), 20, 2200)
    slope = np.clip(rng.gamma(2.0, 3.0, n_points), 0, 35)
    twi = np.clip(12 - 0.25 * slope + rng.normal(0, 1.5, n_points), 3, 16)
    clay_map = np.clip(rng.normal(28, 8, n_points), 8, 55)
    landform = np.where(slope > 12, "hill", np.where(twi > 10.5, "valley", "plain"))

    # Bare-soil reflectance: clay darkens the shortwave-infrared bands (b11, b12).
    brightness = rng.normal(0.25, 0.04, n_points)
    static = pd.DataFrame(
        {
            "uuid": [f"demo-{index:04d}" for index in range(n_points)],
            "lat": lat.round(5),
            "lon": lon.round(5),
            "elevation": elevation.round(1),
            "slope": slope.round(2),
            "aspect": rng.uniform(0, 360, n_points).round(1),
            "twi": twi.round(2),
            "aridity_index": aridity.round(3),
            "s2_b2_barest": (brightness * 0.55 + rng.normal(0, 0.01, n_points)).round(4),
            "s2_b4_barest": (brightness * 0.85 + rng.normal(0, 0.01, n_points)).round(4),
            "s2_b8_barest": (brightness * 1.10 + rng.normal(0, 0.01, n_points)).round(4),
            "s2_b11_barest": (brightness * 1.30 - 0.002 * clay_map + rng.normal(0, 0.01, n_points)).round(4),
            "s2_b12_barest": (brightness * 1.10 - 0.003 * clay_map + rng.normal(0, 0.01, n_points)).round(4),
            "clay_median_0_30cm": clay_map.round(1),
            "landform_class": landform,
        }
    )
    return static


def make_timeseries(static: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Invent monthly Sentinel-2 and climate observations for every point.

    Parameters
    ----------
    static : pandas.DataFrame
        The output of :func:`make_static`.
    rng : numpy.random.Generator
        Source of random numbers, so the result is repeatable.

    Returns
    -------
    timeseries : pandas.DataFrame
        One row per point and observed month, in long format.
    vigour : numpy.ndarray
        Each point's hidden "vegetation vigour" (0 to 1), which also drives organic matter.
    """
    n_points = len(static)
    aridity = static["aridity_index"].to_numpy()
    vigour = np.clip(0.2 + 0.9 * aridity + rng.normal(0, 0.1, n_points), 0.05, 1.0)

    rows = []
    for point in range(n_points):
        for date in MONTHS:
            if rng.random() < 0.17:  # a cloudy month: no observation at all
                continue
            month = date.month
            # Green-up peaks in March-April in Morocco, and is stronger at vigorous points.
            season = max(0.0, np.sin(2 * np.pi * (month - 11) / 12))
            ndvi = 0.12 + 0.55 * vigour[point] * season + rng.normal(0, 0.03)
            rows.append(
                {
                    "uuid": static["uuid"].iat[point],
                    "observation_date": date.strftime("%Y-%m-%d"),
                    "S2_B4": round(0.16 - 0.10 * ndvi + rng.normal(0, 0.01), 4),
                    "S2_B8": round(0.20 + 0.35 * ndvi + rng.normal(0, 0.01), 4),
                    "S2_NDVI": round(ndvi, 4),
                    "CLIM_precip_mm": round(
                        max(
                            0.0, 80 * aridity[point] * max(0.0, np.cos(2 * np.pi * (month - 1) / 12)) + rng.normal(0, 5)
                        ),
                        1,
                    ),
                    "CLIM_LST_celsius": round(
                        18
                        + 12 * np.sin(2 * np.pi * (month - 4) / 12)
                        - 0.004 * static["elevation"].iat[point]
                        + rng.normal(0, 1.5),
                        2,
                    ),
                }
            )
    return pd.DataFrame(rows), vigour


def make_targets(static: pd.DataFrame, vigour: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Invent the three lab measurements from the other columns, plus noise.

    Parameters
    ----------
    static : pandas.DataFrame
        The output of :func:`make_static`.
    vigour : numpy.ndarray
        The hidden vegetation vigour returned by :func:`make_timeseries`.
    rng : numpy.random.Generator
        Source of random numbers, so the result is repeatable.

    Returns
    -------
    pandas.DataFrame
        ``uuid``, ``lat``, ``lon``, ``organic_matter_g_kg``, ``clay_pct`` and ``ph_water``.
    """
    n_points = len(static)
    organic_matter = np.clip(4 + 38 * vigour + 0.4 * static["twi"] + rng.normal(0, 3, n_points), 2, 70)
    clay = np.clip(
        0.8 * static["clay_median_0_30cm"] - 120 * (static["s2_b12_barest"] - 0.2) + rng.normal(0, 3, n_points),
        3,
        70,
    )
    ph = np.clip(8.6 - 2.0 * static["aridity_index"] - 0.02 * organic_matter + rng.normal(0, 0.15, n_points), 5.5, 9.2)
    return pd.DataFrame(
        {
            "uuid": static["uuid"],
            "lat": static["lat"],
            "lon": static["lon"],
            "organic_matter_g_kg": organic_matter.round(2),
            "clay_pct": clay.round(1),
            "ph_water": ph.round(2),
        }
    )


def main(argv: list[str] | None = None) -> Path:
    """Generate the demo dataset and write the three CSV files.

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
    parser.add_argument("--n-points", type=int, default=300, help="How many sample points to invent (default 300).")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed; the same seed gives the same files (default 0)."
    )
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_FOLDER, help="Folder to write into (default examples/demo_data)."
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    static = make_static(args.n_points, rng)
    timeseries, vigour = make_timeseries(static, rng)
    targets = make_targets(static, vigour, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    # lat/lon travel with the targets file; the static file keeps only the id and the covariates.
    static.drop(columns=["lat", "lon"]).to_csv(args.out / "static.csv", index=False)
    targets.to_csv(args.out / "targets.csv", index=False)
    timeseries.to_csv(args.out / "timeseries.csv", index=False)
    print(
        f"Wrote {len(static)} points and {len(timeseries)} monthly observations to {args.out}/ "
        "(static.csv, targets.csv, timeseries.csv)."
    )
    return args.out


if __name__ == "__main__":
    main()
