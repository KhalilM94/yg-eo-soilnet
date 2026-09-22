# Glossary

Plain definitions of the terms used in this documentation and in the code. When a page uses one of
these words, it links here.

A few words have **two meanings** in this project. They are marked with ⚠ below: *registry*,
*bundle*, *residual*, *calibration* and *grid*.

## Data

```{glossary}
point
sample point
  One place where soil was sampled and measured in a lab. It has an identifier (the `uuid`
  column), a latitude and longitude, lab measurements, fixed descriptions of the place, and a
  history of satellite and climate readings. Most tables in the project have one row per point.

static covariate
static feature
  A value describing a point that does not change over time: elevation, slope, aridity, a
  bare-soil satellite reflectance, a value read from an existing soil map, and so on. They come
  from the *static* data file.

time series
  Readings of the same thing at the same point on different dates - for example a Sentinel-2
  band every month from 2017 to 2025. They come from the *time-series* data file, which has one
  row per point per date. Points can have different numbers of readings (cloudy months are
  missing).

modality
data source
  One source of time-series readings with its own columns: Sentinel-2 optical bands, Sentinel-1
  radar, climate, soil moisture, and so on. In the data file, each source's columns share a name
  prefix (`S2_`, `S1_asc_`, `CLIM_`, ...); `temporal.modality_prefix_map` in the main config
  says which prefix belongs to which source.

feature
input
predictor
  Any column a model is given to make its prediction. Features are the static covariates (and,
  for the deep-learning model, the time series). Lab measurements, identifiers and coordinates are
  not features.

target
  A lab-measured soil property the models learn to predict, such as organic matter (g/kg), clay
  (%) or pH. They are listed under `TARGET_COLUMNS` in `data_spec.yml`.

lab column
label column
  Any column holding a lab measurement, listed under `LABEL_COLUMNS` in `data_spec.yml`,
  whether or not it is currently a target. Lab columns are never used as ordinary inputs, so a
  property you stop predicting cannot quietly start helping to predict the others.

auxiliary lab input
  A deliberate exception to the rule above: a measured lab value (say, pH) that is genuinely
  available when predicting, fed to the deep-learning model as an extra input when predicting a
  different property. Switched on with `auxiliary_enabled` and `auxiliary_label_columns`.

residual base
  ⚠ *Residual* also means an ordinary prediction error. Here it is an option of the
  deep-learning model: instead of predicting a property directly, the model predicts a
  **correction** to an existing prediction (the "base"), and the final prediction is base plus
  correction. Switched on with `residual_enabled`.

categorical feature
category
  A column holding labels rather than numbers - a landform class or a soil texture class, for
  example. Listed under `CATEGORICAL_FEATURES` in `data_spec.yml`.

vocabulary
  The list of category labels a model knows, learned from the training points only. A label
  first seen when predicting is treated as "unknown" instead of causing an error.

embedding
  A short list of numbers the deep-learning model learns for each category label, so it can use
  the category like a measurement. Categories that behave alike end up with similar numbers.

missing value
gap
  A cell with no value. A column missing in more than 20% of the points stops the run (the
  threshold is `data_quality.max_missing_column_ratio`); smaller gaps are filled with the
  middle value (median) of the training points, and a validity flag records that the value was
  filled in.

validity flag
  A yes/no value stored next to a filled-in value, saying whether it was really measured. It lets
  a model treat filled-in values with care.

standardization
  Rescaling a column so that, over the training points, its average is 0 and its standard
  deviation is 1. It puts every column on a comparable scale. The average and standard deviation
  are learned from the training points only.

log transform
log1p
target transform
  Before training, the deep-learning model converts each target value *y* to 10 × ln(1 + *y*),
  which spreads out the many small values and pulls in the few very large ones. Predictions are
  converted back to the target's units afterwards. Set with `target_transform: log1p`.

original units
training scale
  Scores are reported on one of two scales. In **original units** (g/kg, %, pH) they can be
  compared with each other and read directly. The **training scale** is the log-transformed and
  standardized scale the deep-learning model trains on; `train_loss`, `val_loss`, `test_loss`
  and `val_r2` are on it and cannot be compared with `rmse_test`.

decimal year
  A date written as a single number: 15 July 2021 is about 2021.54. The time series is converted
  to decimal years before it reaches the deep-learning model.

calendar grid
year-by-month grid
  ⚠ *Grid* also appears in *grid search* and in *spatial grid* (see {term}`spatial split`).
  Here it is how the deep-learning model sees a point's time series: a small table with one row
  per year and one column per calendar month, like a tiny image. Years are counted back from the
  point's most recent reading, and empty cells are marked as empty.
```

## Splitting the data and scoring models

```{glossary}
split
  Every point is assigned once, before any model is trained, to one of three sets: the
  **training set** (used to fit the models), the **validation set** (used during training to
  decide when to stop and which settings are best) and the **test set** (kept aside and used only
  to score the finished models). All models share the same split, so their scores are comparable.
  Set in the `split:` block of `main_config.yml`.

training set
  The points a model learns from. See {term}`split`.

validation set
  Points kept out of training and used to check progress during training - for example to stop
  when the model stops improving. See {term}`split`.

test set
  Points kept out of training entirely and used once, at the end, to score each model. Test
  scores (`rmse_test`, `r2_test`, ...) are the fair measure of how a model does on points it has
  never seen. See {term}`split`.

fit pool
  The points a scikit-learn model is fitted on: the training set **plus** the validation set
  (scikit-learn models choose their settings by {term}`cross-validation` instead of using a
  separate validation set).

cross-validation
fold
  A way to test settings without touching the test set. The fit pool is cut into *k* parts
  (folds; 5 here). The model is trained *k* times, each time leaving one part out and scoring the
  model on it. The average score over the folds is used to choose the settings.

spatial split
spatial_group
  A split that keeps nearby points together: points are grouped into clusters by location
  (k-means or a regular spatial grid), and whole clusters go to the test set. It stops a model
  from looking good just because a test point sits a few metres from a training point. Set
  `split.strategy: spatial_group`.

leakage
  Information about the answer reaching a model during training - for example a lab value that is
  really the target under another name, or a test point used to learn scaling statistics. It
  makes scores look better than they will be on new data. Much of the project's design (lab
  columns never used as inputs, statistics learned from training points only) exists to prevent it.

target group
joint
per_target
  The set of targets one model predicts. With `MULTI_TARGET_MODE: joint` one model predicts all
  targets at once; with `per_target` each target gets its own model. A group's name joins its
  targets with `__`, for example `organic_matter_g_kg__clay_pct__ph_water`.

RMSE
MAE
R²
RPD
RPIQ
bias
  The accuracy scores reported for every model. See {mod}`yg_eo_soilnet.metrics` for what
  each one means.

leaderboard
  A table (and plot) ranking every model trained in a run by its test scores, saved on the main
  MLflow run as `leaderboard.csv` and `leaderboard_plots/leaderboard.png`.

overfitting
  A model learning the particular points it was trained on - including their noise - instead of
  the general pattern, so it scores well on those points and badly on new ones. Dropout, early
  stopping and weight decay are ways to reduce it.
```

## Models

```{glossary}
model family
  The project trains two kinds of model side by side. The **scikit-learn family** holds classic
  statistical and machine-learning models (Ridge, PLS, XGBoost, TabICL) that read only the static
  covariates. The **deep-learning family** holds `soil_cnn`, a neural network that also reads
  the time series.

scikit-learn
  A widely used Python library of classic models. The project's scikit-learn models are listed in
  `configs/sklearn/model_registry.yml`.

estimator
  scikit-learn's word for a model object with `fit` and `predict` methods.

pipeline
  A chain of steps run as one model: for example "fill gaps → rescale → encode categories →
  Ridge". Each scikit-learn model is wrapped in one, so the preprocessing is saved with the model.

model registry
registry
  ⚠ Two different things share this name. The **model list** is a YAML file listing which models
  exist, their settings and whether they are switched on (`enabled: true`):
  `configs/sklearn/model_registry.yml` and `configs/lightning/models/`. The **MLflow model
  registry** is where saved models are given a name and version numbers (see
  {term}`registered model`).

hyperparameter
setting
  A setting chosen before training rather than learned from the data: the strength of Ridge's
  simplification, the number of layers of a network, the learning rate. The project tries several
  values and keeps the best (see {term}`cross-validation` and {term}`tuning`).

deep learning
neural network
  Models built from many layers of simple calculations whose numbers ("weights") are adjusted step
  by step to reduce the error. `soil_cnn` is the project's only deep-learning model.

soil_cnn
  The project's deep-learning model. It summarises the static covariates, turns each data source's
  time series into a {term}`calendar grid` and reads it with a small {term}`convolutional
  network`, combines the summaries, and predicts every target. Three switches change it: `fusion`,
  `auxiliary_enabled` and `residual_enabled`.

Lightning
PyTorch
  The Python libraries the deep-learning model is written with. *Lightning* runs the training
  loop; the configuration of deep-learning models lives under `configs/lightning/`.

convolutional network
CNN
  A kind of neural network that slides small filters over a grid - usually an image - to find
  patterns. Here the "image" is a point's {term}`calendar grid`, and the filters find seasonal
  patterns, such as a green-up every spring.

layer
width
  A network is a stack of layers; each layer turns a list of numbers into another list. Its width
  is how many numbers it outputs. Settings such as `head_hidden_dims: [64, 32]` list the widths:
  two layers, 64 then 32 wide.

encoder
branch
  The part of the deep-learning model that summarises one kind of input into a short list of
  numbers: one for the static covariates, one per time-series data source, one for location.

fusion
  How the deep-learning model combines its branches' summaries. **Gated** fusion weighs each number
  up or down; **attention** fusion lets each summary be re-read in the light of the others (for
  example, the Sentinel-2 summary in the light of the climate summary).

head
  The last layers of the deep-learning model, which turn the combined summary into one predicted
  value per target.

epoch
  One full pass of the deep-learning model over all training points.

batch
  A small group of points the model processes together in one training step (32 by default).

early stopping
  Stopping training once the score on the validation set has not improved for a set number of
  epochs (`patience`), and keeping the best version seen.

checkpoint
  A file saving a trained deep-learning model: its weights, its settings, and the scaling
  statistics and category lists it needs to prepare new data.

random seed
seed
  A number that fixes every random choice (the split, the starting weights, the order of
  batches), so a run can be repeated exactly. Set with `RANDOM_SEED`.

loss
loss function
  The error score the deep-learning model tries to reduce during training. `mse` (the average
  squared error) is the default.

learning rate
  How big a step the deep-learning model takes each time it adjusts its weights. Too big and
  training is unstable; too small and it is slow.

dropout
  During training only, randomly ignoring a fraction of the numbers passing through the network,
  so it cannot rely too much on any single one. It reduces {term}`overfitting`.

bundle
  ⚠ Used for two things. The **sequence bundle** holds everything the deep-learning model reads
  about every point (static values, categories, time series, dates, targets). A **model bundle**
  is one deep-learning model packed with its data and training settings, ready to train.
```

## Uncertainty

```{glossary}
uncertainty
  How far off a prediction might be. When switched on (`uncertainty.enabled`), every prediction
  comes with a {term}`sigma` and a {term}`prediction interval`.

ensemble
member
  The same model trained several times (10 by default), each time starting from a different random
  seed; each copy is a *member*. The average of their predictions is the final prediction, and
  how much they disagree shows how unsure the model is.

bootstrap
  Training each ensemble member on a slightly different sample of the training points (drawn at
  random, some points more than once). Used for models that would otherwise give the same answer
  every time.

sigma
predicted standard deviation
  The predicted "±" of a prediction, in the target's units: the model expects its error to be about
  this big.

epistemic uncertainty
aleatoric uncertainty
  Two reasons a prediction can be unsure. **Epistemic** uncertainty is the model's own lack of
  knowledge, seen as disagreement between ensemble members; more data reduces it. **Aleatoric**
  uncertainty is noise in the data itself (lab error, small-scale variation); more data does not
  reduce it.

variance head
heteroscedastic
  An option of the deep-learning model (`uncertainty.heteroscedastic`) where it predicts, for each
  point, both a value and how noisy that value is likely to be - so the uncertainty can differ
  from point to point.

prediction interval
  A range - a lower and an upper value - that should contain the true value most of the time
  (for example 95% of the time).

coverage
PICP
  How often the true value actually falls inside its prediction interval, measured on the test
  set. It should be close to the promised rate (for example 95%).

calibration
  ⚠ Also used for the points reserved for it. Here: correcting the size of the prediction
  intervals using the errors seen on held-back points, so the promised coverage is actually met.

conformal
  A calibration method: widen or narrow the intervals until they cover the promised share of the
  held-back points' true values.
```

## Explaining predictions

```{glossary}
SHAP
  A way of splitting each prediction into one contribution per input, so you can see which inputs
  pushed it up or down. The contributions add up to the prediction minus the average prediction.
  Switched on with `EXPLAIN_ENABLED`.

beeswarm plot
  The main SHAP figure: one row per input, one dot per point, placed by how much that input moved
  that point's prediction and coloured by the input's value.
```

## Tuning

```{glossary}
tuning
hyperparameter search
  Trying many combinations of a model's settings automatically and keeping the best. `tune.py`
  does this for the deep-learning model with {term}`Optuna`.

Optuna
  The Python library `tune.py` uses to choose which settings to try next.

study
  One tuning campaign: every setting combination tried for one model, stored in
  `optuna_studies/soilnet.db` so it can be continued later.

trial
  One attempt in a study: the model trained once with one combination of settings.

objective
  The score a study tries to improve, `val_loss` by default (lower is better).

search space
  The settings a study may change and the values allowed for each, defined in
  `configs/lightning/search_spaces/`.

pruning
  Stopping a trial early when it is clearly doing worse than earlier trials, to save time.

fingerprint
  A short code computed from a search space. A study's name ends with it (`soil_cnn-e6c9f8`), so
  editing the search space starts a new study instead of mixing incompatible trials.

rerank
  Re-running the best few trials with several seeds and choosing on their average score, so a trial
  that was merely lucky is not picked.

tuned config
  The file `tune.py` writes for the winning trial, in `configs/lightning/tuned/`. It is a complete
  model-list entry you can train directly.
```

## MLflow and saved models

```{glossary}
MLflow
  The tool that records every training run: its settings, scores, figures, tables and saved
  models. By default it stores them in the `mlruns/` folder; `pixi run mlflow` opens a browser
  view of them.

experiment
  A named group of runs in MLflow - for example `Soil_Model_Training_v2` for real training runs,
  `Soil_HPO_Experiment` for tuning and `Soil_Demo` for the demo.

run
main run
parent run
  One execution of `main.py` is one **main run** (named `Run_<date>_<time>`). It holds the split,
  the leaderboard and the combined predictions.

sub-run
child run
  A run inside the main run: one per trained model (and, when one model predicts several targets,
  one per target inside that).

artifact
  Any file saved with a run: plots, tables such as `eval_results/eval_results.csv`, the saved
  model, the log file.

registered model
champion
  A saved model given a name (such as `clay_pct_Ridge`) and a version number in the MLflow model
  registry. The *champion* label points at the version of that model with the lowest `rmse_test`
  so far. It compares versions of the same model on the same target only; the leaderboard compares
  different models.

pyfunc
  MLflow's standard saved-model format: load it with `mlflow.pyfunc.load_model(...)` and call
  `predict` on a table of new points.
```

## Tools

```{glossary}
pixi
environment
  pixi installs the exact versions of Python and every library the project needs into a
  project-local **environment**. `core` runs on a normal processor (CPU), `core-gpu` on an NVIDIA
  graphics card, `dev` adds the tools for tests and documentation, and `explore` adds Jupyter.

YAML
config file
  The plain-text format of the configuration files in `configs/`: `name: value` pairs, with
  indentation for nesting and `#` for comments.
```
