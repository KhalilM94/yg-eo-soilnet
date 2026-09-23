# Quickstart

From a fresh copy of the project to a first comparison of two models, on a small made-up dataset
that ships with it. The first install downloads a few gigabytes; after that the example takes about
two minutes.

## 1. Install pixi

pixi installs Python and every library the project needs, pinned to the versions it was built with,
into a folder inside the project. Nothing is installed system-wide.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

([Other systems](https://pixi.sh/latest/#installation).)

## 2. Get the project

```bash
git clone https://github.com/KhalilM94/yg-eo-soilnet.git
cd yg-eo-soilnet
pixi install -e core
```

`core` runs on an ordinary processor. If you have an NVIDIA graphics card, `core-gpu` uses it, and
training is several times faster.

## 3. Make the example data

```bash
pixi run -e core demo-data
```

```text
Wrote 300 points and 8951 monthly observations to .../examples/demo_data/
```

Three files, which is the layout your own data needs too:

- `static.csv` - one row per sample point: terrain, aridity, bare-soil reflectance, a soil-map
  value, a landform class.
- `targets.csv` - the lab measurements to predict, with each point's latitude and longitude.
- `timeseries.csv` - one row per point per month: two Sentinel-2 bands, a vegetation index,
  rainfall and temperature. About one month in six is missing, as real data is.

It is invented, but not arbitrarily: organic matter is made to follow how green the vegetation gets
over the year, and clay and pH to follow the static covariates. So there is something real for the
models to find, and something for them to disagree about.

## 4. Train

```bash
pixi run -e core demo
```

This trains two models on the same data and scores both on the same 45 held-back points:

- **Ridge**, a regularized linear model, on the static covariates;
- **`soil_cnn`**, the project's deep-learning model, which also reads the monthly time series.

It takes about two minutes and ends with something like:

```text
2026-09-22 20:49:41 - AlMoutmir Soil Models Training - INFO - [sklearn group 1/1] organic_matter_g_kg__clay_pct__ph_water - done in 0.9min
2026-09-22 20:50:03 - AlMoutmir Soil Models Training - INFO - [lightning group 1/1] organic_matter_g_kg__clay_pct__ph_water - done in 0.4min
```

One model predicts all three targets at once, which is why the group's name has all three in it.

## 5. Look at the results

```bash
pixi run -e core mlflow
```

Open <http://127.0.0.1:5000>, choose the experiment **Soil_Demo** on the left, and click the run
named `Run_<date>_<time>`. Under **Artifacts**, open `leaderboard.csv`. Your numbers will be close
to these:

| target | model | rmse_test | r2_test |
|---|---|---|---|
| organic_matter_g_kg | soil_cnn | 3.41 | 0.69 |
| organic_matter_g_kg | Ridge | 4.83 | 0.37 |
| clay_pct | Ridge | 2.77 | 0.93 |
| clay_pct | soil_cnn | 4.08 | 0.84 |
| ph_water | Ridge | 0.19 | 0.75 |
| ph_water | soil_cnn | 0.21 | 0.70 |

`rmse_test` is the typical error in the target's own units - g/kg, %, pH units - so lower is better.
`r2_test` is the share of the variation explained, so higher is better.

`soil_cnn` wins on organic matter, which in this data follows the seasonal pattern only the time
series carries. Ridge wins on clay and pH, which follow the static covariates - and a linear model
on the right inputs is hard to beat. That is the comparison the project exists to make, and neither
answer is the general one.

## What just happened

1. **The settings were read.** `examples/demo_config/main_config.yml` said where the data was, which
   columns are targets, how to split, and which model lists to use.
2. **The data was loaded** and the columns sorted out: ids and coordinates set aside, lab values
   kept out of the inputs, small gaps filled in and flagged.
3. **The points were split once** - 70% training, 15% validation, 15% test - and *both* models used
   that same split. That is what makes the leaderboard's rows comparable.
4. **Ridge was trained**, trying each value of its one setting and keeping the best by
   cross-validation.
5. **`soil_cnn` was trained**, laying each point's monthly readings out as a year-by-month table and
   scanning it for seasonal patterns, then combining that with the static covariates.
6. **Both were scored** on the held-back points, and everything was recorded: the settings, the
   scores, the figures, the test predictions, the saved models, and the leaderboard.

[How it works](how-it-works.md) goes through those steps properly.

## Next

- **Use your own data:** [Getting started](getting-started.md).
- **Do something specific** - a spatial split, one model per target, uncertainty, tuning:
  [Examples](examples.md).
- **Understand the run you just made:** [Reading the results](outputs.md).
