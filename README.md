# yg-eo-soilnet

Predict lab-measured soil properties - organic matter, clay, sand, pH, carbonate, cation exchange
capacity - at soil sample points, from satellite time series (Sentinel-1 and Sentinel-2), climate,
terrain and existing soil maps.

The project trains several kinds of model on the same data, scores them all on the same held-back
points, and records everything - settings, scores, figures and the trained models - so runs can be
compared and repeated. It was built for the Al Moutmir soil dataset in Morocco, but works with any
dataset laid out the same way.

**Who this is for:** environmental scientists and analysts comfortable with Python and the basics
of machine learning. The [glossary](docs/glossary.md) defines the project's own terms.

---

## Quickstart

Five commands take you from a fresh copy of the project to a first comparison of two models, on a
small made-up dataset that ships with the project. The first install downloads a few gigabytes
(Python, PyTorch and the other libraries); after that, the example runs in about two minutes.

**1. Install pixi**, the tool that installs Python and every library the project needs
([other systems](https://pixi.sh/latest/#installation)):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**2. Get the project and install it.** The `core` environment runs on an ordinary processor; no
graphics card needed.

```bash
git clone https://github.com/KhalilM94/yg-eo-soilnet.git
cd yg-eo-soilnet
pixi install -e core
```

**3. Create the example data** - 300 invented sample points with three years of monthly satellite
and climate readings:

```bash
pixi run -e core demo-data
```

You should see: `Wrote 300 points and 8951 monthly observations to .../examples/demo_data/`.

**4. Train the models.** This trains two models on the example data - Ridge (a regularized linear
model on the static covariates) and `soil_cnn` (the project's convolutional network, which also reads
the monthly time series) - and scores both on the same 45 test points:

```bash
pixi run -e core demo
```

It takes about two minutes. The last lines look like
`[lightning group 1/1] organic_matter_g_kg__clay_pct__ph_water - done in 0.4min`.

**5. Look at the results** in MLflow, the tool that recorded the run:

```bash
pixi run -e core mlflow
```

Open <http://127.0.0.1:5000>, choose the experiment **Soil_Demo** on the left, and click the run
named `Run_<date>_<time>`. Under **Artifacts**, `leaderboard.csv` compares the two models on the
test points. Your numbers will be close to these:

| target | model | rmse_test | r2_test |
|---|---|---|---|
| organic_matter_g_kg | soil_cnn | 3.41 | 0.69 |
| organic_matter_g_kg | Ridge | 4.83 | 0.37 |
| clay_pct | Ridge | 2.77 | 0.93 |
| clay_pct | soil_cnn | 4.08 | 0.84 |
| ph_water | Ridge | 0.19 | 0.75 |
| ph_water | soil_cnn | 0.21 | 0.70 |

In the made-up data, organic matter follows how green the vegetation gets over the year - a signal
only the time series carries - so `soil_cnn` wins there, while Ridge wins for clay and pH, which
follow the static covariates. `rmse_test` is in each property's own units (g/kg, %, pH units).

**Next:** [use your own data](docs/getting-started.md), or read the
[step-by-step Quickstart](docs/quickstart.md) for what each step did.

---

## How it works

One run of `python main.py` goes through these steps:

```text
 configs/*.yml ─────► 1. read the settings
                              │
 your three data files ─► 2. load the data and keep the usable columns
 (static, targets,            │
  time series)          3. split the points once: training / validation / test
                              │
                        4. decide which targets each model predicts
                              │
                ┌─────────────┴──────────────┐
      5a. scikit-learn models        5b. deep-learning model (soil_cnn)
          (static covariates)            (static covariates + time series)
                └─────────────┬──────────────┘
                              │
                        6. score every model on the test points and record
                           everything in MLflow: scores, plots, saved models,
                           a leaderboard comparing them
```

1. **Settings.** Everything is controlled by YAML files in [`configs/`](configs/): where the data
   is, which columns are what, which models to train and how. Nothing is hard-coded.
2. **Data.** Three files: *static* covariates (one row per sample point: terrain, aridity,
   bare-soil reflectance, existing soil maps), *targets* (the lab measurements to predict, with
   each point's latitude and longitude) and a *time series* (one row per point per date: Sentinel-1,
   Sentinel-2, climate, vegetation and soil-moisture readings). Columns that are identifiers, lab
   measurements or too empty are set aside; small gaps are filled in and flagged.
3. **One shared split.** Before any model sees the data, every point is assigned once to the
   training, validation or test set - at random, or by keeping nearby points together. Every model
   uses the same split, so every model is scored on the same test points and the scores can be
   compared directly.
4. **Target groups.** With several targets, either one model predicts them all at once
   (`MULTI_TARGET_MODE: joint`) or each target gets its own model (`per_target`).
5. **Two model families**, trained side by side:
   - **scikit-learn models** - Ridge, PLS regression, XGBoost, TabICL - read the static covariates.
     Each tries every combination of its settings listed in the configuration and keeps the best,
     judged by cross-validation on the training points.
   - **`soil_cnn`**, a deep-learning model, also reads the time series. It lays each point's
     readings out as a small year-by-month table - like a tiny image - and scans it for seasonal
     patterns, then combines that with the static covariates.
6. **Results.** Each trained model is scored on the test points in the target's own units (RMSE,
   MAE, R², RPD, RPIQ, bias) and recorded in [MLflow](https://mlflow.org): its settings, scores,
   a predicted-versus-measured plot, the table of test predictions, and the saved model, ready to
   predict new points. The main run also gets a leaderboard comparing every model.

Three optional extras can be switched on in the main configuration: **uncertainty** (a "±" and a
range with every prediction, from training each model several times), **explanations** (how much
each input pushed each prediction up or down), and **exporting every point's prediction** to one
table.

A separate tool, `tune.py`, searches automatically for the best settings of the deep-learning
model and writes them to a file that `main.py` can train directly.

The [How it works](docs/how-it-works.md) page goes through each step in detail.

## The command-line tools

Run every command from the repository root, inside an environment (`pixi run -e core ...`, or
`pixi shell -e core` first).

| Tool | What it does | Example |
|---|---|---|
| [`main.py`](docs/cli/main.md) | Train every switched-on model and record the results. | `python main.py` |
| [`tune.py`](docs/cli/tune.md) | Search for the best settings of a deep-learning model. | `python tune.py --entry soil_cnn --n-trials 50` |
| [`export_predictions.py`](docs/cli/export_predictions.md) | Add a table of every point's predictions to a finished run. | `python export_predictions.py --parent-run-id <run id>` |
| [`replot.py`](docs/cli/replot.md) | Redraw the figures of finished runs. | `python replot.py --parent-run-id <run id>` |
| [`relog.py`](docs/cli/relog.md) | Save and register a deep-learning model from its checkpoint file, without retraining. | `python relog.py --checkpoint <file.ckpt> --run-id <run id>` |

Two things to know before running `main.py` on the real configuration:

- **Every model is switched off** in the configuration files as shipped (`enabled: false`). Switch
  on the ones you want in `configs/sklearn/model_registry.yml` and
  `configs/lightning/models/soil_cnn.yml`, or train a tuned configuration:
  `LIGHTNING_MODEL_REGISTRY_PATH=configs/lightning/tuned/soil_cnn-e6c9f8_best.yml python main.py`.
- **The data folder** is set by `common.data.root` in `configs/main_config.yml`. Point it at the
  folder holding your three files.

## Project layout

```text
main.py                  train every switched-on model (the main entry point)
tune.py                  search for the best deep-learning settings
export_predictions.py    add every point's predictions to a finished run
replot.py                redraw the figures of finished runs
relog.py                 save a deep-learning model from its checkpoint
config.py                reads the YAML configuration into one Config object

configs/                 all settings: data, columns, models, tuning (see docs/configuration/)
examples/                the made-up demo dataset generator and its configuration
docs/                    this documentation (build it with `pixi run -e dev docs`)
tests/                   automated tests (`pixi run -e dev test`)
notebooks/               exploratory analyses and the paper figures

src/yg_eo_soilnet/       the Python package
  data_manager.py          loads the three data files and picks the usable columns
  datamodules/             prepares data for each model family and makes the shared split
  models/                  builds the models; the deep-learning model itself
  trainers/                trains the scikit-learn and deep-learning models
  logger/  tracking.py     records everything in MLflow
  metrics.py               the accuracy scores
  uncertainty/             optional: prediction ranges ("±")
  explain/                 optional: how much each input contributed to each prediction
  hpo/                     the automatic settings search behind tune.py
  serving/                 loading a saved deep-learning model to predict new points
```

## Documentation

The full documentation is in [`docs/`](docs/) and builds into a website:

```bash
pixi run -e dev docs        # then open docs/_build/html/index.html
```

It covers the [Quickstart](docs/quickstart.md), [using your own data](docs/getting-started.md),
[how it works](docs/how-it-works.md), [worked examples](docs/examples.md), one page per
[command-line tool](docs/cli/index.md), every [configuration setting](docs/configuration/index.md),
[reading the results](docs/outputs.md), [known issues](docs/known-issues.md) and a
[glossary](docs/glossary.md).

## Development

```bash
pixi run -e dev test        # all tests
pixi run -e dev test-fast   # skip the slowest tests
pixi run -e dev lint        # style, and a docstring on every public function
pixi run -e dev doctest     # run the examples written in the docstrings
```

The `dev` environment needs a computer with an NVIDIA graphics card and a driver supporting
CUDA 12. See
[Contributing](docs/contributing.md) for how the code and documentation are written.

## License

MIT - see [LICENSE](LICENSE).
