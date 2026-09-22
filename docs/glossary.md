# Glossary

The terms this documentation uses, with what they mean **in this project**: where they are set,
their defaults, and the project-specific ones in full.

A few words have **two meanings** here. They are marked ⚠: *registry*, *bundle*, *residual*,
*calibration* and *grid*.

## Data

```{glossary}
point
sample point
  One soil sampling location: an id (`uuid`), latitude/longitude, lab measurements, static
  covariates and a satellite/climate history. Most tables have one row per point.

static covariate
static feature
  A per-point value that does not change over time - terrain, aridity, bare-soil Sentinel-2
  reflectance, a prior soil-map value. Read from the *static* data file.

time series
  Dated readings per point (for example monthly Sentinel-2 bands, 2017-2025), in the *time-series*
  file: one row per point per date. Points have different numbers of readings.

modality
data source
  One family of time-series columns sharing a name prefix - `S2_` (Sentinel-2), `S1_asc_` /
  `S1_desc_` (Sentinel-1), `CLIM_`, `AG_`, `SOIL_`. Mapped in `temporal.modality_prefix_map`.

feature
  A model input: the static covariates, plus the time series for the deep-learning model. Lab
  measurements, ids and coordinates are never features.

target
  A lab-measured property being predicted (`TARGET_COLUMNS` in `data_spec.yml`), e.g.
  `organic_matter_g_kg`, `clay_pct`, `ph_water`.

lab column
label column
  Any lab measurement, listed under `LABEL_COLUMNS` whether or not it is a target. Never used as an
  ordinary feature, so a property you stop predicting cannot leak into predicting the others.

auxiliary lab input
  A deliberate exception to that rule: a measured lab value that is genuinely available at
  prediction time (say pH), fed to `soil_cnn` as an extra input for a *different* target.
  `auxiliary_enabled` + `auxiliary_label_columns`; naming a current target is refused.

residual base
  ⚠ Not an ordinary residual (prediction error). A `soil_cnn` option (`residual_enabled`) where the
  network predicts a *correction* to an existing prediction column - typically an earlier model's
  exported prediction - and outputs base + correction. Only honest if the base was predicted
  without seeing the current test points.

categorical feature
  A label column such as a landform class, declared under `CATEGORICAL_FEATURES`. One-hot or
  ordinal encoded for scikit-learn models; an {term}`embedding` for `soil_cnn`.

vocabulary
  The category labels a model knows, learned from training points only; unseen labels map to a
  reserved "unknown" slot.

embedding
  The learned vector `soil_cnn` uses for each category label.

validity flag
  A 0/1 companion value recording whether a value was measured or filled in. Gaps in covariates
  are median-filled (from training points) and flagged; a covariate missing in more than 20% of
  points stops the run (`data_quality.max_missing_column_ratio`).

standardization
  Rescaling to mean 0 and standard deviation 1, with statistics from the training points only.

log1p
target transform
  `soil_cnn` trains on 10·ln(1 + *y*) rather than *y* (`target_transform: log1p`), then converts
  predictions back to the target's units.

original units
training scale
  Scores are on one of two scales. `rmse_test`, `r2_test`, ... are in the target's own units and
  comparable across models. `soil_cnn`'s `train_loss`, `val_loss`, `test_loss` and `val_r2` are on
  its log-transformed, standardized training scale and cannot be compared with them.

decimal year
  A date as one number: 15 July 2021 ≈ 2021.54. Time-series dates are converted to this form.

calendar grid
  ⚠ Not grid search or the spatial grid. How `soil_cnn` sees a data source's time series: a
  years × 12-months table per point, filled with its readings (empty cells masked), with years
  counted back from the point's latest reading. A small CNN then reads it like an image.
```

## Splitting and scoring

```{glossary}
split
  The one assignment of every point to the training, validation or test set, made before any model
  is trained and shared by all of them (`split:` in `main_config.yml`; 15% test, 15% validation by
  default). Saved on each run as `data_splits/split_assignments.parquet`.

fit pool
  What scikit-learn models are fitted on: training **plus** validation points. They choose their
  hyperparameters by {term}`cross-validation` inside it rather than on a separate validation set.

cross-validation
  Here: 5-fold, inside the fit pool, used by scikit-learn models to choose hyperparameters (their
  `params` grid). `SPLIT_STRATEGY: groupkfold` keeps spatial clusters within one fold.

spatial split
  `split.strategy: spatial_group`: points are clustered by location (k-means or a regular grid) and
  whole clusters are held out, so test points are not near-duplicates of training points.

population policy
  What to do when the model families can use different points (the deep-learning model needs a time
  series). `intersect`: everyone uses only points every family can use, so test sets are identical.
  `assign_all`: each family uses all it can.

leakage
  Information about the answer reaching training - a lab value that is really the target, or a test
  point used for scaling statistics - which inflates scores.

target group
joint
per_target
  The targets one model predicts. `MULTI_TARGET_MODE: joint` fits one model for all targets;
  `per_target` one model per target. A scikit-learn model fits jointly only if its entry declares
  `multi_target: native`. Group names join targets with `__`.

leaderboard
  The table ranking every model of a run by test scores: `leaderboard.csv` and
  `leaderboard_plots/leaderboard.png` on the main run. Scores are defined in
  {mod}`yg_eo_soilnet.metrics`.
```

## Models

```{glossary}
model family
  The two kinds of model trained side by side: the **scikit-learn family** (Ridge, PLS, XGBoost,
  TabICL; static covariates only) and the **deep-learning family** (`soil_cnn`; static covariates
  plus time series, built with PyTorch Lightning).

model registry
registry
  ⚠ Two things. The **model list**: the YAML files declaring which models exist, their
  hyperparameters and `enabled: true/false` (`configs/sklearn/model_registry.yml`,
  `configs/lightning/models/`). The **MLflow model registry**: named, versioned saved models (see
  {term}`registered model`).

pipeline
  A scikit-learn `Pipeline`: imputation → scaling → categorical encoding → estimator, saved as one
  object so new data is prepared exactly like the training data.

TabICL
  A pretrained "foundation model" for tables that predicts from the training rows directly instead
  of fitting weights. Memory-hungry; downloads a checkpoint on first use.

soil_cnn
  The project's deep-learning model. Branches summarise the static covariates, each data source's
  {term}`calendar grid` (through a small CNN) and optionally location; a {term}`fusion` step
  combines them and an MLP head predicts every target. Switches: `fusion`, `auxiliary_enabled`,
  `residual_enabled`.

fusion
  How `soil_cnn` combines its branch summaries. **gated**: each value is scaled by a learned 0-1
  weight. **attention**: a small transformer where each branch summary (a "token") is updated from
  the others - e.g. the Sentinel-2 summary re-read in light of the climate summary.

dilated_tempcnn
annual_grid2d
  The two CNNs that read a calendar grid. `dilated_tempcnn` scans along months with a spacing that
  links each month to the same month a year earlier; `annual_grid2d` scans months and years as a
  2-D image.

hidden dims
  Settings such as `head_hidden_dims: [64, 32]` list layer widths: two layers, 64 then 32 units.
  The length of the list is the number of layers.

early stopping
  Training stops once `val_loss` has not improved for `patience` epochs (100 in the shipped
  defaults), and the best checkpoint is kept.

checkpoint
  A saved `soil_cnn` (`.ckpt`): weights, hyperparameters, and the scaling statistics and category
  vocabularies needed to prepare new data - so it can be served on its own.

random seed
  `RANDOM_SEED` fixes the split, the initial weights and the batch order, so a run can be repeated.

loss function
  What `soil_cnn` minimizes: `mse` by default; `huber`/`smooth_l1` are less sensitive to outliers;
  `mahalanobis`, `correlation_penalty` and `cosine` (joint mode only) also penalize predictions whose
  targets do not co-vary the way the measured ones do.

bundle
  ⚠ Two things. The **sequence bundle** holds everything `soil_cnn` reads about every point. A
  **model bundle** is one deep-learning model with its datamodule and training settings, ready to
  train.
```

## Uncertainty

```{glossary}
ensemble
  With `uncertainty.enabled`, each model is trained `n_members` times (10 by default) from different
  seeds; the members' mean is the prediction and their spread feeds the uncertainty. Members appear
  as extra sub-runs tagged `run_kind=ensemble_member`.

bootstrap
  Resampling the training rows for each member, used for models that would otherwise give identical
  members (Ridge ignores its seed).

sigma
  A prediction's standard deviation, in the target's units (column `prediction_std`).

epistemic uncertainty
aleatoric uncertainty
  **Epistemic**: the model's own uncertainty, seen as disagreement between ensemble members; more
  data reduces it. **Aleatoric**: noise in the data itself (lab error, small-scale variability),
  predicted by a {term}`variance head`; more data does not reduce it.

variance head
heteroscedastic
  `uncertainty.heteroscedastic`: `soil_cnn` predicts a mean and a variance per target, so each
  prediction carries its own noise estimate. Trained with a beta-NLL loss.

prediction interval
  The `prediction_lower` / `prediction_upper` range that should contain the true value at the
  promised rate (for example 95%). Method set by `uncertainty.interval.method`: `sigma`
  (mean ± k·σ), `gaussian`, or `conformal`.

coverage
PICP
  The share of test points whose true value falls inside its interval. Compare it with the promised
  rate; `coverage_error` is the difference.

calibration
  ⚠ Also the name of the points reserved for it. Adjusting interval widths using errors observed on
  held-back points, so the promised coverage holds.

conformal
  A calibration method: scale the intervals so they contain the promised share of the validation
  points' true values. Its guarantee holds if new points resemble the validation points.
```

## Explanations and tuning

```{glossary}
SHAP
  Per-prediction attributions: one contribution per input, adding up to the prediction minus the
  average prediction. `EXPLAIN_ENABLED`; figures under `explain/` on each run.

tuning
  `tune.py`: a hyperparameter search for one deep-learning model with Optuna.

study
trial
  A **study** is one search campaign, stored in `optuna_studies/soilnet.db` and resumable; a
  **trial** is one training run with one combination of settings. The study minimizes `val_loss`
  by default.

search space
  The hyperparameters a study may vary and their ranges, in `configs/lightning/search_spaces/`.

fingerprint
  A short hash of a search space that ends the study's name (`soil_cnn-e6c9f8`). Editing the space
  changes it, which starts a new study instead of mixing incompatible trials.

rerank
  `tune.py --rerank-top K`: re-train the K best trials over several seeds and pick the best average,
  so a lucky trial is not chosen.

tuned config
  The model-list file `tune.py` writes for the winner in `configs/lightning/tuned/`, trainable with
  `LIGHTNING_MODEL_REGISTRY_PATH=<file> python main.py`.
```

## MLflow

```{glossary}
experiment
  A named group of MLflow runs: `Soil_Model_Training_v2` (training), `Soil_HPO_Experiment`
  (tuning), `Soil_Demo` (the demo).

main run
parent run
  One `main.py` execution (`Run_<date>_<time>`): holds the split, the leaderboard and the combined
  predictions.

sub-run
child run
  One per trained model inside the main run (named `<target group>_<model>`), plus one per target
  when a model predicts several.

artifact
  A file stored with a run: `eval_results/eval_results.csv` (test predictions), plots, the saved
  model, the log file. See {doc}`outputs`.

registered model
champion
  A saved model with a name (`<target>_<model>`) and versions in the MLflow model registry. The
  `champion` alias points at the version with the lowest `rmse_test` - among versions of that one
  model and target, not across models (the leaderboard does that).

pyfunc
  MLflow's generic saved-model format: `mlflow.pyfunc.load_model(uri).predict(table)`.
```

## Tools

```{glossary}
pixi
environment
  pixi installs pinned versions of Python and every library into a project-local environment:
  `core` (CPU), `core-gpu` (NVIDIA GPU), `dev` (adds tests and docs), `explore` (adds Jupyter).
```
