# Configuration

Everything a run does is set in YAML. Nothing is hard-coded, and no setting lives in two places.

## The files

One main file names the others, relative to its own folder:

```text
configs/
  main_config.yml           where the data is, how to split, MLflow, the optional extras
  data_spec.yml             which columns are targets, lab values, categories, ids
  sklearn/
    config.yml              options for the scikit-learn family
    model_registry.yml      the scikit-learn model list: what exists, and what is switched on
  lightning/
    config.yml              options for the deep-learning family
    models/
      defaults.yml          settings shared by every deep-learning model
      soil_cnn.yml          the deep-learning model list
    search_spaces/          what tune.py may vary, and between what limits
    tuned/                  what tune.py writes: model lists you can train directly
```

`--config-path` chooses the main file, and therefore the whole run:

```bash
python main.py                                                   # configs/main_config.yml
python main.py --config-path examples/demo_config/main_config.yml # the demo
```

Copying the whole `configs/` folder and pointing `--config-path` at the copy is a perfectly good way
to keep two setups side by side.

## Where a setting's value comes from

First match wins:

1. **An environment variable of the same name** - `RANDOM_SEED=7 python main.py`.
2. The `common:` block of the main file.
3. `data_spec.yml`, then the scikit-learn config, then the Lightning config.
4. The top level of the main file.
5. The default in the code.

So a setting can be overridden for a single run without editing anything:

```bash
MLFLOW_EXPERIMENT_NAME=Soil_Experiment_B RANDOM_SEED=7 python main.py
LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py
```

The settings that sit in a nested block have a flat name for this purpose:

| In the file | As an environment variable |
|---|---|
| `split.strategy` | `SPLIT_HOLDOUT_STRATEGY` |
| `split.test_size`, `split.val_size` | `SPLIT_TEST_SIZE`, `SPLIT_VAL_SIZE` |
| `split.seed` | `SPLIT_SEED` |
| `split.population_policy` | `SPLIT_POPULATION_POLICY` |
| `data_quality.max_missing_column_ratio` | `MAX_MISSING_COLUMN_RATIO` |
| `uncertainty.enabled` | `UNCERTAINTY_ENABLED` |
| `uncertainty.n_members` | `UNCERTAINTY_N_MEMBERS` |
| `export_point_predictions.enabled` | `EXPORT_POINT_PREDICTIONS_ENABLED` |
| `explain.enabled` | `EXPLAIN_ENABLED` |

## The settings you will actually touch

### Where the data is - `main_config.yml`

```yaml
common:
  data:
    root: "/path/to/your/data"      # relative paths are taken from where you run the command
    static: "static.csv"            # one row per point; a folder of files also works
    targets: "targets.csv"          # or null if the static file already holds them
    timeseries: "timeseries.csv"    # one row per point per date
  POINT_ID_COLUMN: "uuid"
  LAT_COLUMN: "lat"
  LON_COLUMN: "lon"
  RANDOM_SEED: 42                   # fixes the split, the weights and the batch order
```

### Which columns are what - `data_spec.yml`

| Setting | What it is |
|---|---|
| `TARGET_COLUMNS` | What to predict. |
| `LABEL_COLUMNS` | **Every** lab measurement, targets included. These are never inputs, so a property you stop predicting cannot leak into predicting the others. |
| `CATEGORICAL_FEATURES` | Columns holding categories rather than numbers. A column cannot also be in `LABEL_COLUMNS` - a lab column never reaches a model, so the run stops and names it. |
| `IGNORED_COLUMNS` | Ids, coordinates, geometry - anything identifying a point rather than describing it. |
| `MULTI_TARGET_MODE` | `joint` (one model predicts every target) or `per_target`. |
| `COLUMNS_TO_TRANSFORM` | Targets trained on a log scale, which suits skewed properties. |
| `CARRY_LABEL_COLUMNS` | Carry the other lab values alongside, so a model can use one as an input on purpose - see {term}`auxiliary lab input`. |
| `USE_HARMONIC_COORDS` | Let the deep-learning model read each point's location. |

Everything not set aside and not too empty becomes an input; you never list the inputs.

### How to split - `main_config.yml`

```yaml
  split:
    strategy: random          # or spatial_group, which holds out whole areas
    test_size: 0.15           # fractions of ALL the points: these two leave 70% to train on
    val_size: 0.15
    seed: 42
    population_policy: intersect   # every family uses the same points, so test sets match
    min_population_ratio: 0.5      # stop if most of the data falls out of the split
```

### What to train - the two model lists

Every model ships with `enabled: false`. Switch on what you want:

```yaml
# configs/sklearn/model_registry.yml
Ridge:
  enabled: true
  modeltype: "ml"
  import_path: sklearn.linear_model.Ridge
  multi_target: native            # this one can predict several targets at once
  params:                         # every combination is tried; the best is kept
    model__alpha: [0.01, 0.1, 1, 10, 100, 1000]
```

```yaml
# configs/lightning/models/soil_cnn.yml
soil_cnn:
  enabled: true
  init_args:
    fusion: gated                 # how the branches are combined
    temporal_encoder: dilated_tempcnn
    head_hidden_dims: [128, 64]   # two layers, 128 units then 64
    dropout: 0.1
    learning_rate: 0.001
```

Settings shared by every deep-learning model - the processor, the epochs, early stopping - are in
`configs/lightning/models/defaults.yml`. On a machine without an NVIDIA card, set
`accelerator: cpu` there.

The full meaning of every `soil_cnn` setting is in
{class}`~yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module.SoilCNNLightningModule`,
and the files themselves are commented throughout.

### Data quality - `main_config.yml`

```yaml
  data_quality:
    max_missing_column_ratio: 0.2   # a covariate blanker than this stops the run
    allow_sparse_columns: []        # except these
    fail_on_sparse_columns: true    # false warns instead
```

Small gaps are filled with the training points' median and flagged. A column that is mostly empty is
refused, because filling in most of a column invents it rather than recovering it.

### The optional extras - `main_config.yml`

```yaml
  uncertainty:
    enabled: false        # train each model several times; see Examples
  explain:
    enabled: false        # how much each input contributed
  export_point_predictions:
    enabled: false        # every point's prediction, not just the test points
```

### Where runs are recorded - `main_config.yml`

```yaml
  MLFLOW_TRACKING_URI: ""                          # empty means <repo>/mlruns
  MLFLOW_EXPERIMENT_NAME: "Soil_Model_Training_v2"
  MLFLOW_REGISTER_MODELS: true                     # false for throwaway runs
```

## Tuning - `configs/lightning/search_spaces/`

One file per model, declaring what `tune.py` may vary:

```yaml
soil_cnn:
  objective:
    metric: val_loss          # what to make better
    direction: minimize
  fixed:                      # pinned for every trial
    trainer.max_epochs: 100
    datamodule.num_workers: 4
  params:                     # searched, keyed by where the setting goes
    model.fusion:
      type: categorical
      choices: [gated, attention]
    model.learning_rate:
      type: float
      low: 0.00001
      high: 0.001
      log: true               # draw across orders of magnitude
    model.dropout:
      type: float
      low: 0.2
      high: 0.6
      step: 0.1
  derive:                     # settings that depend on each other; see hpo.constraints
    - dims_pyramid:
        key: model.head_hidden_dims
        min_depth: 0
        max_depth: 3
        widths: [32, 64, 128]
        taper: 0.5
```

Each key under `params` is where the setting goes in the model-list entry. A list of layer widths
cannot be drawn as one number, so those come from a named hook under `derive:`.

Editing the file changes the study's {term}`fingerprint`, so a new study starts rather than mixing
trials run under different rules.

## Run everything from the repository root

The default paths are relative to it, and so is a relative `data.root`.

## Reading the settings from Python

```python
from config import Config

config = Config("examples/demo_config/main_config.yml")
config.TARGET_COLUMNS
config.SPLIT_TEST_SIZE
sorted(config.MODEL_REGISTRY)
```

Every setting is an upper-case attribute; see {class}`config.Config`.
