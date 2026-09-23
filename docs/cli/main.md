# `main.py` - train the models

Trains every model switched on in the model lists, scores them all on the same held-back points,
and records everything in MLflow: the settings, the scores, the figures, the saved models and a
[leaderboard](../outputs.md) comparing them.

## Examples

```bash
# Train with the shipped configuration.
python main.py

# Train the demo dataset instead.
python main.py --config-path examples/demo_config/main_config.yml

# Train a tuned deep-learning configuration without editing any file.
LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py
```

## Options

| Option | Default | What it does |
|---|---|---|
| `--config-path` | `configs/main_config.yml` | The main configuration file. Every other file - the column definitions, the two model lists - is named inside it, so this one option chooses the whole run. |

Everything else is set in the configuration files; see the [configuration guide](../configuration/index.md).
Any setting can also be overridden for one run by an environment variable of the same name:

```bash
MLFLOW_EXPERIMENT_NAME=Soil_Experiment_B RANDOM_SEED=7 python main.py
```

## What it reads

- The main configuration file, and the files it names.
- The three data files, from `common.data.root`.
- The two model lists, to decide what to train.

## What it writes

One MLflow [main run](../outputs.md) named `Run_<date>_<time>`, holding a sub-run per model, and
inside them the scores, the figures, the test predictions and the saved models. Nothing is written
to a results folder - see [Reading the results](../outputs.md) for how to get at it all.

## What it prints

```text
2026-09-22 20:49:41 - AlMoutmir Soil Models Training - INFO - [sklearn group 1/1] organic_matter_g_kg__clay_pct__ph_water - done in 0.9min
2026-09-22 20:50:03 - AlMoutmir Soil Models Training - INFO - [lightning group 1/1] organic_matter_g_kg__clay_pct__ph_water - done in 0.4min
```

Two lines per target group per family - one when it starts, one when it finishes. `organic_matter_g_kg__clay_pct__ph_water` is a
{term}`target group`: three targets predicted by one model, because `MULTI_TARGET_MODE` is `joint`.

## Common problems

**"Data source folder not found"** - `common.data.root` does not point at your data. A relative path
is taken from the folder you ran the command in.

**The run finishes in seconds and the leaderboard is empty** - every model is switched off. As
shipped, every entry in both model lists has `enabled: false`; switch on the ones you want.

**"Target column(s) missing from targets file"** - a name in `TARGET_COLUMNS` is not a column in
your targets file. Names are case-sensitive.

**"covariate(s) ... are blank on more than 20% of rows"** - a covariate is too empty to fill in
honestly. Drop it (`IGNORED_COLUMNS`), exempt it (`common.data_quality.allow_sparse_columns`), or
raise the limit deliberately.

**It tries to use a graphics card you do not have** - the shipped model lists detect one
rather than assume it, so this means something names a card explicitly. Look for
`accelerator:` in `configs/lightning/models/defaults.yml` or the model's own file, and
`device:` for TabICL in the scikit-learn model list; `auto` and unset respectively let each
one choose.
