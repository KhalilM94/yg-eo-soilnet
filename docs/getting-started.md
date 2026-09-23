# Using your own data

You have soil samples with lab measurements, and covariates for each of them. This page gets from
there to a first run of your own. It assumes you have been through the
[Quickstart](quickstart.md), so the project is installed and you have seen a run work.

## 1. Choose an environment

| Environment | When |
|---|---|
| `core` | Anywhere. Trains on the processor. |
| `core-gpu` | You have an NVIDIA card. Several times faster for the deep-learning model. |
| `dev` | You are changing the code: adds the tests and this documentation. Needs a card with CUDA 12. |
| `explore` | Jupyter, for looking at data. |

```bash
pixi install -e core
pixi shell -e core        # or prefix each command with: pixi run -e core
```

## 2. Lay out your data

Three files in one folder. CSV or Parquet; a folder of files works too, and is read as one table.

**Static covariates** - one row per sample point. Anything that does not change over time: terrain,
climate averages, bare-soil reflectance, existing soil maps, a landform class.

```text
uuid,elevation,slope,aspect,twi,aridity_index,s2_b4_barest,clay_median_0_30cm,landform_class
demo-0000,1555.9,2.25,60.8,11.36,0.251,0.1946,24.1,valley
```

**Targets** - the lab measurements, with each point's coordinates. They may instead be columns of
the static file, in which case leave `targets` blank.

```text
uuid,lat,lon,organic_matter_g_kg,clay_pct,ph_water
demo-0000,33.04785,-5.4128,26.64,23.5,7.62
```

**Time series** - one row per point per date. Column names start with a prefix saying which data
source they are: `S2_` for Sentinel-2, `CLIM_` for climate, and so on.

```text
uuid,observation_date,S2_B4,S2_B8,S2_NDVI,CLIM_precip_mm,CLIM_LST_celsius
demo-0000,2021-01-15,0.1066,0.346,0.3745,21.9,-0.92
```

What matters:

- **An id column in all three**, with the same values. It is what joins them, and what the split is
  keyed on. `uuid` unless you say otherwise.
- **Points may have different numbers of readings**, and gaps are expected. Nothing is padded or
  filled in on the time axis.
- **You do not need a time series at all.** With `temporal.enabled: false` the deep-learning model
  reads only the static covariates, and the scikit-learn models never read it anyway.

## 3. Point the configuration at it

In `configs/main_config.yml`:

```yaml
common:
  data:
    root: "/path/to/your/data"
    static: "your_static.csv"
    targets: "your_targets.csv"     # or null, if the static file holds them
    timeseries: "your_timeseries.csv"
  POINT_ID_COLUMN: "uuid"
  LAT_COLUMN: "lat"
  LON_COLUMN: "lon"
```

A relative `root` is taken from the folder you run the command in.

`root`, `static` and `POINT_ID_COLUMN` are required, as is `temporal.time_column` when you have a
time series: they name your data, so the run stops and says which is missing rather than guessing.

## 4. Say what your columns are

In `configs/data_spec.yml`:

```yaml
# What to predict.
TARGET_COLUMNS:
  - organic_matter_g_kg
  - clay_pct
  - ph_water

# Every lab measurement in your data, targets included. These are never model inputs, so a
# property you stop predicting cannot leak into predicting the others.
LABEL_COLUMNS:
  - organic_matter_g_kg
  - clay_pct
  - ph_water
  - cec_meq_100g

# Columns holding categories rather than numbers.
CATEGORICAL_FEATURES:
  - landform_class

# Ids, coordinates, geometry - anything that identifies a point rather than describing it.
IGNORED_COLUMNS:
  - uuid
  - lat
  - lon
```

:::{note}
A column cannot be in both `LABEL_COLUMNS` and `CATEGORICAL_FEATURES`: a lab column is removed from
the inputs before the category handling sees it, so no model would receive it. The run stops and
names the column, rather than dropping it quietly.
:::

Everything not set aside and not too empty becomes a model input. You do not list the inputs.

## 5. Choose how to split

`split:` in `main_config.yml`. The default holds out 15% for testing and 15% for validation, at
random, and every model uses that same split.

If your points are clustered - several samples from one field, or a dense transect - a random split
leaves near-copies of your test points in the training set, and every score comes out flattering.
Hold out whole areas instead:

```yaml
  split:
    strategy: spatial_group
    test_size: 0.15
    val_size: 0.15
    group:
      class_path: yg_eo_soilnet.clustering_utils.KMeansClusterStrategy
      params:
        n_clusters: 12
```

Run it both ways: the gap between the two is how much of your score was proximity.

## 6. Switch a model on

**Every model ships switched off.** Open `configs/sklearn/model_registry.yml` and set
`enabled: true` on one or two - `Ridge` is a good first choice, `XGBoost` a good second - and
`configs/lightning/models/soil_cnn.yml` for the deep-learning model.

On a machine without an NVIDIA card, also set `accelerator: cpu` in
`configs/lightning/models/defaults.yml`, and `device: "cpu"` for TabICL.

## 7. Run it

```bash
python main.py
pixi run -e core mlflow      # then open http://127.0.0.1:5000
```

[Reading the results](outputs.md) explains what you are looking at.

## What to check first

- **`n_test` on the leaderboard.** If it is much smaller than you expect, points are being dropped -
  the run's log says which columns cost them.
- **`r2_test` near zero or negative.** The model is no better than predicting the average. Usually
  too few points, or inputs that carry nothing about the target.
- **A suspiciously good score with a random split.** Try `spatial_group` before believing it.
- **`soil_cnn` no better than the linear model.** Check the time series is really being read: with
  `temporal.enabled: true` but no `timeseries` file configured, it is silently switched off, and the
  run's log says how many readings each point has.

## Next

- [Examples](examples.md) - one model per target, uncertainty, tuning, predicting new points.
- [Configuration](configuration/index.md) - every setting, file by file.
