# `replot.py` - redraw the figures

Redraws the figures of runs that have already finished, from what those runs recorded. Nothing is
retrained, no data is read and no model is loaded: every figure a run produces can be rebuilt from
the two tables it already wrote.

Use it after changing how a figure looks, or when a figure failed to draw during a run.

## Examples

```bash
# Redraw one whole training run: every model's figures, then the leaderboard.
python replot.py --parent-run-id 7c2f1a9b4d8e4f0a91b0c3d2e5f60718

# Just one model's run.
python replot.py --run-id 3e8d5b1c0a7f4e2d8c9b6a5f4e3d2c1b

# Every run in an experiment - check first.
python replot.py --experiment Soil_Model_Training_v2 --dry-run

# Only the predicted-against-measured figures.
python replot.py --parent-run-id <run id> --only pred_obs
```

## Options

Exactly one of `--run-id`, `--parent-run-id` or `--experiment` is required.

| Option | Default | What it does |
|---|---|---|
| `--run-id` | - | Redraw one run's figures. For a {term}`main run`, that means its two leaderboard figures. |
| `--parent-run-id` | - | Redraw a whole training run: every model's figures, then the leaderboard figures. |
| `--experiment` | - | Redraw every training run in this MLflow experiment, by name or id. |
| `--config-path` | `configs/main_config.yml` | The main configuration file, read only to find where the runs are recorded. |
| `--only` | all | Only redraw these kinds, comma-separated: `pred_obs`, `uncertainty`, `cv`, `leaderboard`. |
| `--since` | - | With `--experiment`, skip runs that started before this date (`YYYY-MM-DD`). **This option currently stops with an error** - see [Known issues](../known-issues.md). |
| `--dry-run` | off | List which runs would be redrawn and which files would be written, without writing. |

## What it writes

The same figures the run wrote in the first place, in the same places, replacing them:
`plots/pred_obs.png`, the uncertainty figures, the hyperparameter-search figures, and on a main run
the leaderboard table and its two figures.

## Common problems

**"no eval_results found"** - that run never wrote a results table, so there is nothing to draw
from. A run whose models all failed is the usual reason.

**The figures come back looking the same** - the run recorded what it drew, so a change to the
drawing code has to be in the working tree you run this from.
