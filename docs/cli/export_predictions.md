# `export_predictions.py` - every point's predictions

Adds a table of every model's prediction for *every* point to a run that has already finished.

A run normally records what each model predicted for its own test points. This answers the other
question - given a point, what did each model say about it? - which is what you need to map the
predictions or compare them with something else.

You do not need this tool if `export_point_predictions.enabled` was true when the run trained: the
run wrote those tables itself. It is for adding them afterwards.

## Examples

```bash
# Add the predictions to a finished run.
python export_predictions.py --parent-run-id 7c2f1a9b4d8e4f0a91b0c3d2e5f60718

# Check what it would do first.
python export_predictions.py --parent-run-id <run id> --dry-run

# Only one model.
python export_predictions.py --parent-run-id <run id> --models soil_cnn
```

The run id is on the run's page in MLflow, and in the `Run ID` column of the run list.

## Options

| Option | Default | What it does |
|---|---|---|
| `--parent-run-id` | **required** | The finished run to add the predictions to. |
| `--config-path` | `configs/main_config.yml` | The main configuration file. It must point at the data the run trained on: the points are rebuilt from it, and predictions for different points would be meaningless. |
| `--models` | every model the run trained | Only export these, comma-separated: `Ridge,soil_cnn`. |
| `--skip-models` | from the configuration | Models to leave out, comma-separated. |
| `--allow-population-drift` | off | Carry on even when the rebuilt data holds different points from the run - points added since, say - instead of stopping. |
| `--member-checkpoint-dir` | `lightning_logs` | Where to look for the checkpoints of a deep-learning {term}`ensemble`'s members that were not saved in MLflow. |
| `--no-member-recovery` | off | Do not look for those checkpoints; deep-learning ensembles are then skipped. |
| `--dry-run` | off | Rebuild the data, run the checks and list what would be exported, without writing anything. |

## What it writes

Two tables on the main run, under `predictions/`:

- `point_predictions_wide.csv` - one row per point, one column per target and model
  (`clay_pct__soil_cnn`). This is what you join onto a map.
- `point_predictions_long.csv` - one row per point per target per model. This is what you group and
  compare.

## Common problems

**"the rebuilt population differs from the run"** - the data has changed since the run trained.
Point `--config-path` at the data as it was, or accept it with `--allow-population-drift`.

**A deep-learning ensemble is skipped** - its members' checkpoints are not in MLflow and were not
found on disk. `--member-checkpoint-dir` can point at where they are.
