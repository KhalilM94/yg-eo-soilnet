# How it works

One run of `main.py`, step by step, and what each step decides.

```text
 configs/*.yml ─────► 1. read the settings
                              │
 your three data files ─► 2. load the data, set aside what is not an input
                              │
                        3. split the points once: training / validation / test
                              │
                        4. decide which targets each model predicts
                              │
                ┌─────────────┴──────────────┐
      5a. scikit-learn models        5b. the deep-learning model
          (static covariates)            (covariates + time series)
                └─────────────┬──────────────┘
                              │
                        6. score every model on the same test points, and
                           record everything: scores, figures, saved models,
                           a leaderboard comparing them
```

## 1. Settings

Everything is in YAML: where the data is, which columns are what, how to split, which models to
train and with what settings. Nothing is hard-coded, and any setting can be overridden for one run
by an environment variable of the same name. See [Configuration](configuration/index.md).

## 2. Loading the data, and deciding what models may see

`DataManager` reads the static covariates and the targets - one file or two - and joins them on the
point id. The time series is read only if a model needs it.

Then the columns are sorted out, and this is where most of the honesty of the whole run lives:

- **Ids, coordinates and geometry are set aside.** They identify a point, they do not describe it.
- **Every lab measurement is set aside**, not only the ones being predicted. Otherwise dropping a
  property from `TARGET_COLUMNS` would quietly turn it into an input for the others, and the model
  would be told half the answer.
- **Small gaps are filled in** with the training points' median, and flagged, so the model can tell
  a filled value from a measured one. A covariate missing on more than 20% of points stops the run
  instead: filling in most of a column does not recover it, it invents it.
- **What is left is the inputs.** You never list them.

The time series keeps its irregularity: each point carries the readings it actually has, each with
its own date. Nothing is padded, and two points with different numbers of readings are both fine.

## 3. One split, shared by everything

Every point is assigned once to training, validation or test - before any model is built - and every
model uses that assignment. That is the single most important thing about the pipeline: it is what
makes the leaderboard's rows comparable, rather than six models each scored on its own favourite
points.

- **`random`** holds out points one at a time.
- **`spatial_group`** clusters the points by location and holds out whole clusters, so a test point
  is not the near-neighbour of a training point. Use it whenever your samples are clustered on the
  ground; the difference between the two is how much of your score was proximity.

The split is keyed on the point id, not on row numbers, because the two families keep different rows
- the deep-learning model needs a usable time series. `population_policy: intersect`, the default,
splits only the points every family can use, so their test sets are identical.

The fractions are of *all* the points: 0.15 and 0.15 leave 70% for training.

## 4. Target groups

With several targets you can train one model that predicts them all
(`MULTI_TARGET_MODE: joint`), or one model per target (`per_target`).

Joint is the default. One model, one set of settings, and the targets can inform each other; the
cost is that a point is only usable if every target was measured for it, and that one setting - the
log transform, say - applies to them all. Per-target keeps every point for whichever targets it has,
at the cost of a model each.

Not every scikit-learn model can predict several targets at once. One that cannot gets one model per
target automatically, with a warning.

## 5a. The scikit-learn models

Ridge, PLS regression, XGBoost, TabICL. They read the static covariates only.

Each model is wrapped in a pipeline - fill gaps, scale, encode categories, then the model - which is
saved as one object, so new points are prepared exactly like the training points were.

Their settings are chosen by cross-validation *inside* the training data: the training and
validation points together (the {term}`fit pool`) are split into 5 folds, every combination in the
model's grid is tried, and the best is kept and refitted on the whole pool. That is why these models
do not use the validation split the way the deep-learning model does.

## 5b. The deep-learning model

`soil_cnn` reads the covariates *and* the time series.

The trick is how the time series is presented. Each data source's readings are laid out on a
{term}`calendar grid`: one row per year, one column per calendar month, each reading in the cell for
its date, empty cells marked as empty. A small convolutional network then scans that grid the way
one would scan an image - so a step sideways is a step through the seasons, and a step downwards is
the same month a year later. Seasonality is something the network can see rather than something it
has to remember.

The rows are counted back from each point's own latest reading, and nothing records which years
they were, so a model trained on 2017-2025 data can read 2030-2035 data.

Three branches - the covariates, the time series, and optionally the point's location - are combined
by a {term}`fusion` step, and a small stack of layers predicts every target. Training stops when the
validation score stops improving, and the best epoch is kept.

The switches - what the fusion does, whether measured lab values are fed in directly, whether the
model predicts a correction to an existing prediction - are described in
{class}`~yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module.SoilCNNLightningModule`.

## 6. Scoring and recording

Every model is scored on the test points in the target's own units: RMSE, MAE, bias, R², RPD, RPIQ.
Both families are scored by the same code, so the numbers mean the same thing.

Everything goes into MLflow: the settings, the scores, a predicted-against-measured figure, the
table of test predictions, the saved model, and on the {term}`main run` a leaderboard comparing
every model on the same points. See [Reading the results](outputs.md).

## The two scales

Worth stating plainly, because it is the easiest thing to misread.

The scores named `rmse_test`, `r2_test` and so on are in the **target's own units** and are
comparable between models. `soil_cnn` also reports `train_loss`, `val_loss` and `test_loss`, which
are on **its own training scale** - log-transformed and standardized. Those say whether training
went well. They cannot be compared with `rmse_test`, nor with another model's loss.

## The optional extras

**Uncertainty** (`uncertainty.enabled`) trains each model several times from different seeds and
uses the disagreement between them as a "plus or minus". The intervals are then calibrated against
held-back points, so that a "95% interval" really does contain about 95% of measurements. See
{mod}`yg_eo_soilnet.uncertainty`.

**Explanations** (`explain.enabled`) work out how much each input pushed each prediction up or
down, so a figure can say *this* point is predicted high because of its July greenness and its
elevation. See {mod}`yg_eo_soilnet.explain`.

**Per-point predictions** (`export_point_predictions.enabled`) add a table of every model's
prediction for every point, not only the test points - which is what you join onto a map.

## Tuning

`tune.py` is a separate tool that searches for a deep-learning model's best settings: it trains it
over and over with different settings, keeps track of what works, and writes the winner out as a
file `main.py` can train. See [its page](cli/tune.md).
