# The command-line tools

Five tools, all run from the repository root inside a pixi environment - either
`pixi run -e core python <tool>.py ...`, or `pixi shell -e core` once and then `python <tool>.py`.

| Tool | What it is for |
|---|---|
| [`main.py`](main.md) | Train every model that is switched on, and record the results. |
| [`tune.py`](tune.md) | Search for a deep-learning model's best settings. |
| [`export_predictions.py`](export_predictions.md) | Add every point's predictions to a finished run. |
| [`replot.py`](replot.md) | Redraw the figures of finished runs. |
| [`relog.py`](relog.md) | Save a deep-learning model from its checkpoint, without retraining. |

Every tool takes `--config-path`, which says where the main configuration file is. It defaults to
`configs/main_config.yml`; the demo uses `examples/demo_config/main_config.yml`.

```{toctree}
:maxdepth: 1
:hidden:

main
tune
export_predictions
replot
relog
```

## Which one do I want?

- **I have data and want models** - `main.py`.
- **A model trains but is not good enough** - `tune.py`, then train the file it writes with
  `main.py`.
- **I want a prediction for every point, not just the test points** - switch
  `export_point_predictions` on before the run, or use `export_predictions.py` afterwards.
- **The figures look wrong, or I changed how they are drawn** - `replot.py`.
- **A run trained a model but failed to save it** - `relog.py`.
