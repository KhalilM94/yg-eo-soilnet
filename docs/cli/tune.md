# `tune.py` - search for the best settings

Trains a deep-learning model over and over with different settings, keeps track of what works, and
writes the winner out as a model-list file that `main.py` can train directly.

One search campaign is a {term}`study`; one attempt inside it is a {term}`trial`. Which settings to
try, and between what limits, is declared in a {term}`search space` file - see
[Configuration](../configuration/index.md). Studies are stored in a small database file, so a search
can be stopped with Ctrl-C and picked up later: running the same command again continues it.

## Examples

```bash
# 50 trials on soil_cnn, then write the winner out.
python tune.py --entry soil_cnn --n-trials 50

# Continue that study for 50 more trials.
python tune.py --entry soil_cnn --n-trials 50

# Start again from scratch.
python tune.py --entry soil_cnn --n-trials 50 --reset

# Re-run the best 5 trials over 3 seeds each and export the best average.
python tune.py --entry soil_cnn --rerank-top 5

# Just export the winner of a finished study; trains nothing.
python tune.py --entry soil_cnn --export-only

# Then train the tuned configuration:
LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py
```

## Options

### Choosing what to tune

| Option | Default | What it does |
|---|---|---|
| `--entry` | **required** | The model to tune, as named in the deep-learning model list - `soil_cnn`. |
| `--config-path` | `configs/main_config.yml` | The main configuration file; the data and the split come from it. |
| `--target` | every target | Which {term}`target group` to tune. Needed only with `MULTI_TARGET_MODE: per_target`, where each target has its own model. |
| `--search-spaces` | `configs/lightning/search_spaces` | Where the search spaces are: a folder with one file per model, or a single file. |

### How much searching

| Option | Default | What it does |
|---|---|---|
| `--n-trials` | 50 | How many trials to run now. They are added to the study if it already exists. |
| `--timeout` | none | Stop starting new trials after this many seconds. The running trial is finished first. |
| `--max-epochs` | from the search space | Cap the epochs per trial. This changes the study's {term}`fingerprint`, so it starts a new study. |
| `--seed` | `RANDOM_SEED` | The random seed for the trials. |
| `--seed-repeats` | 1 | Train each trial this many times with different seeds and score the average. Slower, but a trial's score then depends less on luck. |

### The study

| Option | Default | What it does |
|---|---|---|
| `--study-name` | `<entry>-<fingerprint>` | The study's name, which is what resuming works on. The default ends with a digest of the search space, so editing the space starts a new study rather than mixing incomparable trials. |
| `--storage` | `sqlite:///optuna_studies/soilnet.db` | Where studies are stored. |
| `--reset` | off | Delete the study and all its trials before running. |

### Picking and exporting the winner

| Option | Default | What it does |
|---|---|---|
| `--rerank-top` | off | Re-train the K best trials with several seeds each and export the best *average*. The headline best trial is partly whichever one drew the luckiest seed; this is a far better guide to what the settings are really worth. |
| `--rerank-seeds` | 3 | Seeds per trial when re-ranking. The first is the trial's own, which doubles as a check that the run reproduces. |
| `--export-only` | off | Write the winner of an existing study and stop - no data loading, no training. |
| `--export-path` | `configs/lightning/tuned/<study name>_best.yml`, or `_reranked.yml` after re-ranking | Where to write the tuned configuration. |
| `--top-n` | 10 | How many of the best trials to list at the end. |

### Running and watching

| Option | Default | What it does |
|---|---|---|
| `--cache-datamodules` | off | Reuse the prepared data between trials whose data settings are the same. Saves a lot of time when only the model's settings are being searched. |
| `--fail-fast` | off | Stop with the error when a trial fails, instead of marking it failed and going on. |
| `--progress` | `auto` | How progress is shown: `bar`, `plain` (one line per trial), `none`, or `auto` - bars in a terminal, plain lines when the output is redirected. |
| `--verbose` | off | Show Lightning's own training output for every trial. |
| `--no-mlflow` | off | Do not record the study in MLflow; the trials are still saved in the study database. Note the data-preparation run is still recorded - see [Known issues](../known-issues.md). |

## What it writes

- The study database (`optuna_studies/soilnet.db` unless changed), which holds every trial.
- One MLflow run per study in the `Soil_HPO_Experiment` experiment, with `trials.csv` and two
  figures: the score of each trial over time, and which settings actually mattered.
- The tuned model-list file, which is what you train with afterwards.

## What it prints

```text
soil_cnn-e6c9f8 (resuming, 30 trials on record)
trial  31  val_loss=0.5412  best=0.5405 (t8) ok28 pruned2
```

The bar at the bottom shows the trials, and the inner bar the epochs of the current trial.

## Common problems

**Every trial fails the same way** - the search space asks for something the model refuses. Run one
trial with `--fail-fast --verbose` to see the error in full.

**The study will not resume** - the search space changed, so the name changed with it. `--study-name`
can name the old one explicitly, but its trials were run under different rules.

**A tuned configuration scores worse than its trial said** - that is the point of `--rerank-top`; see
above.
