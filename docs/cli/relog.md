# `relog.py` - save a model from its checkpoint

Rebuilds a deep-learning model from the {term}`checkpoint` file a run wrote, saves it into that run
in MLflow's own format, and registers it - all without retraining.

Use it when a run trained a model successfully but failed to save it: the checkpoint is on disk,
but the run has no model you can load.

## Examples

```bash
python relog.py \
  --checkpoint lightning_logs/version_79/checkpoints/epoch=42-step=1290.ckpt \
  --run-id 3e8d5b1c0a7f4e2d8c9b6a5f4e3d2c1b

# Save it into the run but leave the registry alone.
python relog.py --checkpoint <file.ckpt> --run-id <run id> --no-register
```

The run id is on the run's page in MLflow. The checkpoint path is recorded on the run too, as the
`best_model_path` setting.

## Options

| Option | Default | What it does |
|---|---|---|
| `--checkpoint` | **required** | The `.ckpt` file to rebuild the model from. |
| `--run-id` | **required** | The run the model was trained in. The model is saved into that run, so it sits with the scores it earned. |
| `--model-class` | read from the checkpoint | The model's class, as an import path. Only needed when the checkpoint does not record it. |
| `--config-path` | `configs/main_config.yml` | The main configuration file, read for where runs are recorded and for the model list. |
| `--no-register` | off | Save the model in the run but do not add it to the {term}`model registry`. |
| `--allow-ensemble-member` | off | Accept a run that trained an {term}`ensemble`. Refused by default, because one member is not the ensemble and saving it under the ensemble's name would misrepresent what the run produced. |
| `--rows` | 3 | How many rows the example input saved with the model has. |

## What it writes

Into the run named by `--run-id`: the saved model in MLflow's generic format, with the example input
and the list of inputs it expects; and, unless `--no-register`, a new version in the registry under
`<target>_<model>`.

## Common problems

**"could not determine the model class"** - the checkpoint does not say what it is. Pass
`--model-class`.

**"this run trained an ensemble"** - see `--allow-ensemble-member` above. What you probably want is
the ensemble as a whole, which means re-running the training.

**The checkpoint will not load** - it was written by a different version of the code, with settings
the current model no longer takes. The error names the setting.
