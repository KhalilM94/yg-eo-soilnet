# Reading the results

Nothing is written to a results folder. Everything a run produces goes into MLflow, which you read
in a browser:

```bash
pixi run -e core mlflow       # then open http://127.0.0.1:5000
```

Pick the {term}`experiment` on the left - `Soil_Model_Training_v2` for a normal run, `Soil_Demo` for
the demo, `Soil_HPO_Experiment` for tuning - and click the run you want.

## The run tree

One `python main.py` produces one {term}`main run` with the models beneath it:

```text
Run_20260922_181108                                  the main run
├── leaderboard.csv                                  every model, compared
├── leaderboard_plots/                               the same, as figures
├── data_splits/                                     which point went where
├── predictions/                                     every point's predictions (optional)
├── Run_20260922_181108_training.log                 everything the run printed
│
├── organic_matter_g_kg__clay_pct__ph_water_Ridge    one model
│   ├── eval_results/eval_results.csv                its test predictions
│   ├── cv/cv_results.csv                            every setting it tried
│   ├── plots/                                       its figures
│   ├── meta/run_summary.json                        what it did
│   ├── predictions/point_predictions.csv            its prediction for every point
│   ├── organic_matter_g_kg_Ridge                    ─┐
│   ├── clay_pct_Ridge                                │ one sub-run per target,
│   └── ph_water_Ridge                               ─┘ each with that target's scores
│
└── organic_matter_g_kg__clay_pct__ph_water_soil_cnn
    ├── checkpoints/best.ckpt                        the best epoch's weights
    └── ...                                          the same as above
```

The long name is a {term}`target group`: `MULTI_TARGET_MODE: joint` means one model predicts all
three targets, so the model has one run and each target has a sub-run beneath it. With
`per_target`, each target gets its own model and there are no sub-runs.

With uncertainty on, each model also has one sub-run per {term}`ensemble` member, tagged
`run_kind=ensemble_member`. Those are members, not results - the leaderboard leaves them out.

## The leaderboard

`leaderboard.csv` on the main run, and the same as a figure under `leaderboard_plots/`. One row per
model per target:

| target | model | rmse_test | r2_test | n_test |
|---|---|---|---|---|
| organic_matter_g_kg | soil_cnn | 3.41 | 0.69 | 45 |
| organic_matter_g_kg | Ridge | 4.83 | 0.37 | 45 |
| clay_pct | Ridge | 2.77 | 0.93 | 45 |

Every row was scored on the same held-back points, so the rows are directly comparable. That is what
the shared split is for.

## The scores

All in the target's own units - g/kg, %, pH units - and all on the test points:

| Score | What it means | Better |
|---|---|---|
| `rmse_test` | Typical size of the error, in the target's units. Large errors count for more. | lower |
| `mae_test` | Average size of the error. Less swayed by a few bad points. | lower |
| `bias_test` | Average signed error: positive means the model predicts high overall. | nearer 0 |
| `r2_test` | Share of the variation the model explains. 1 is perfect; 0 is no better than always predicting the average; negative is worse than that. | higher |
| `rpd_test` | Spread of the measurements divided by the error. Above 2 is usually called useful, above 3 good. | higher |
| `rpiq_test` | The same idea, using the middle half of the measurements, so a few extremes do not flatter it. | higher |
| `n_test` | How many test points the score is from. A small number means a noisy score. | - |

A model predicting several targets also reports each target's score under its own name
(`rmse_test_clay_pct`) and the average across them under the plain name (`rmse_test`).

:::{warning}
`soil_cnn` also reports `train_loss`, `val_loss` and `test_loss`. Those are **not** in the target's
units: they are on the model's own {term}`training scale`, which is log-transformed and
standardized. They say whether training went well; they cannot be compared with `rmse_test` or with
another model's loss. See {term}`original units`.
:::

## The files in a run

| File | What is in it |
|---|---|
| `eval_results/eval_results.csv` | One row per test point: its covariates, the lab measurement, and the prediction. With uncertainty on, also `prediction_std` and the interval. |
| `eval_results/split_summary.json` | How many points in each split, and how the target is distributed in each. |
| `plots/pred_obs.png` | Predicted against measured, with the 1:1 line, a fitted line and the scores. |
| `cv/cv_results.csv` | Every combination of settings the search tried, and how each scored (scikit-learn only). |
| `checkpoints/best.ckpt` | The best epoch's weights (deep learning only). |
| `meta/run_summary.json` | What the model was, what it did, and anything that was skipped and why. |
| `explain/` | The SHAP figures and the full table of contributions, when explanations are on. |
| `uncertainty/` | Whether the intervals are the width they claim, when uncertainty is on. |
| `predictions/point_predictions.csv` | This model's prediction for every point, when the export is on. |
| `data_splits/split_assignments.parquet` | Which split each point went to. On the main run. |

## The saved models

Each trained model is saved so it can predict new points later, under the name
`<target group>_<model>`, and - with `MLFLOW_REGISTER_MODELS: true` - added to the
{term}`model registry` as a new version.

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/clay_pct_Ridge@champion")
predictions = model.predict(new_points)
```

:::{note}
{term}`champion` means "the best version of *this* model for *this* target", by `rmse_test`. It
does not mean the best model for that target - comparing models is the leaderboard's job.
:::

## Reading a run from Python

```python
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()
leaderboard = pd.read_csv(client.download_artifacts(parent_run_id, "leaderboard.csv"))
leaderboard.sort_values("rmse_test").head()
```
