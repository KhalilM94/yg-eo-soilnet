# Examples

Eight things people actually want to do, each with the commands and the settings to change. They
assume the [Quickstart](quickstart.md) works; to try one on the demo data, add
`--config-path examples/demo_config/main_config.yml` to each command and edit the files under
`examples/demo_config/` instead.

## 1. Compare two models on the same points

The default behaviour, and worth stating on its own: switch on as many models as you like and they
are all trained on the same split and scored on the same held-back points.

In `configs/sklearn/model_registry.yml`:

```yaml
Ridge:
  enabled: true      # was false
XGBoost:
  enabled: true      # was false
```

In `configs/lightning/models/soil_cnn.yml`, `enabled: true`.

```bash
python main.py
pixi run -e core mlflow
```

`leaderboard.csv` on the main run has one row per model per target, ranked. Rows are comparable
because the points behind them are the same ones.

## 2. One model per target instead of one for all

By default one model predicts every target at once. To give each target its own model, in
`configs/data_spec.yml`:

```yaml
MULTI_TARGET_MODE: per_target      # was joint
```

Worth doing when your targets have different coverage - a point measured for pH but not for
carbonate is dropped entirely by a joint model, and kept for pH by a per-target one - or when one
target wants the log transform and another does not.

The run then has one model per target, named `clay_pct_Ridge` rather than
`organic_matter_g_kg__clay_pct__ph_water_Ridge`, and takes correspondingly longer.

## 3. Hold out whole areas instead of scattered points

If your samples are clustered on the ground, a random split leaves near-copies of your test points
in the training set and every score comes out flattering. In `configs/main_config.yml`:

```yaml
  split:
    strategy: spatial_group     # was random
    test_size: 0.15
    val_size: 0.15
    group:
      class_path: yg_eo_soilnet.clustering_utils.KMeansClusterStrategy
      params:
        n_clusters: 12
```

Run it both ways and compare the leaderboards. The drop is the part of your score that was
proximity rather than skill. `SpatialGridClusterStrategy` with `cell_size_m` lays a grid over the
points instead of clustering them.

## 4. Add a scikit-learn model

Any regressor with the usual `fit` / `predict` can be added by editing YAML - no code. In
`configs/sklearn/model_registry.yml`:

```yaml
RandomForest:
  enabled: true
  modeltype: "ml"
  import_path: sklearn.ensemble.RandomForestRegressor
  multi_target: native          # this one can predict several targets at once
  params:                       # every combination is tried; the best is kept
    model__n_estimators: [200, 500]
    model__max_depth: [10, 20, null]
```

The `params` names are the pipeline's: the model step is called `model`, so a setting of the
estimator is `model__<name>`. Leave `multi_target` out for a model that predicts one target at a
time - it then gets one model per target automatically.

The random seed comes from `RANDOM_SEED`; you only set one in the entry to depart from it.

## 5. Tune the deep-learning model, then train the winner

```bash
# 50 attempts; stop with Ctrl-C whenever, the same command resumes.
python tune.py --entry soil_cnn --n-trials 50 --cache-datamodules

# Re-run the best 5 over 3 seeds each and export the best average, not the luckiest run.
python tune.py --entry soil_cnn --rerank-top 5

# Train the winner.
LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py
```

The exported file is an ordinary model list, so you can read it, edit it, and commit it. What is
searched, and between what limits, is in `configs/lightning/search_spaces/`.

On the demo data a short search is a good way to see the machinery work:

```bash
python tune.py --entry soil_cnn --n-trials 5 \
  --config-path examples/demo_config/main_config.yml
```

## 6. Get a "plus or minus" with every prediction

In `configs/main_config.yml`:

```yaml
  uncertainty:
    enabled: true
    n_members: 5          # each model is trained this many times; 3 while iterating, 10 for a report
    bootstrap: auto
    heteroscedastic: true # the deep-learning model also predicts how noisy each point is
    alpha: 0.05           # so the intervals promise to contain 95% of measurements
    interval:
      method: conformal   # measured against held-back points, not assumed
```

Training takes `n_members` times as long.

Afterwards, `eval_results/eval_results.csv` has `prediction_std`, `prediction_lower` and
`prediction_upper` beside each prediction, the predicted-against-measured figure has error bars,
and `uncertainty/` has two figures answering the question that matters: **are the intervals the
width they claim to be?** Read `picp_test` - the share of measurements that actually fell inside -
against the 0.95 they promised.

:::{note}
With `method: conformal` and `calibration.source: val`, the scikit-learn models train on about 15%
fewer points, because the validation points are held back to calibrate against. A slightly worse
`rmse_test` than an uncertainty-free run is expected, not a regression.
:::

## 7. Predict new points with a saved model

Every trained model is saved so it can be used later:

```python
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("models:/clay_pct_Ridge@champion")
new_points = pd.read_csv("new_static_covariates.csv")
predictions = model.predict(new_points)
```

`@champion` is the best version of *that* model for *that* target, by `rmse_test`. `models:/<name>/3`
takes version 3 instead.

A saved deep-learning model takes a table with the covariates as ordinary columns and each data
source's readings and dates as nested columns; the example saved with the model shows the exact
shape. It carries its own preparation - the scaling, the fill values, the category numbering - so
you hand it raw values, not prepared ones.

## 8. Get every point's predictions, not just the test points

Switch it on before the run, in `configs/main_config.yml`:

```yaml
  export_point_predictions:
    enabled: true
```

Or add them to a run that has already finished:

```bash
python export_predictions.py --parent-run-id <run id>
```

Either way the main run gets `predictions/point_predictions_wide.csv` - one row per point, one
column per target and model - which is what you join onto a map, and a long form, one row per point
per target per model, which is what you group and compare.

:::{warning}
These are predictions for points the models were *trained* on, as well as for the held-back ones.
They are for mapping, not for judging how good a model is: only the test points can tell you that.
:::
