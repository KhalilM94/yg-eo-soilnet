"""One MLflow run per A1 run folder: what ``run_a1.py`` opens, and the later steps reopen.

``run_a1.py`` records each run folder under one *parent* MLflow run - its settings, its git commit,
and one child run per plan - and writes that run's id into the folder's ``manifest.json``. A resumed
``run_a1.py`` reopens the same parent, and so do ``evaluate.py`` and ``make_figures.py``, which add
the evaluation tables, the headline metrics and the figures to it. Everything one run folder
produced is then found in one place in the MLflow browser.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import mlflow

from yg_eo_soilnet.a1.plans import slug

#: The manifest entry holding the parent run's ids.
MANIFEST_KEY = "mlflow"


class A1TrackingError(RuntimeError):
    """The run folder's MLflow run cannot be found or reopened."""


def recorded_run(run_dir) -> dict:
    """The parent run's ids recorded in a run folder's manifest: ``run_id``, ``experiment_id``.

    Raises
    ------
    A1TrackingError
        If the folder has no manifest, or the manifest no MLflow run.
    """
    from yg_eo_soilnet.a1.runner import MANIFEST_FILE

    path = Path(run_dir) / MANIFEST_FILE
    if not path.exists():
        raise A1TrackingError(f"No {MANIFEST_FILE} in {run_dir}: run run_a1.py first.")
    entry = json.loads(path.read_text()).get(MANIFEST_KEY) or {}
    if not entry.get("run_id"):
        raise A1TrackingError(
            f"{path} records no MLflow run: the run was made with --no-mlflow, or by an older run_a1.py. "
            "Pass --no-mlflow to skip MLflow, or run run_a1.py again (with MLflow) to record one."
        )
    return entry


def check_run(run_id: str, run_dir) -> None:
    """Refuse a parent run that MLflow no longer has, or that was deleted.

    Raises
    ------
    A1TrackingError
    """
    try:
        run = mlflow.tracking.MlflowClient().get_run(run_id)
    except Exception as error:  # MlflowException: the store does not know the id
        raise A1TrackingError(
            f"The MLflow run {run_id} recorded for {run_dir} no longer exists ({type(error).__name__}). "
            "Use --no-mlflow, or start the run folder again with run_a1.py --restart."
        ) from error
    if run.info.lifecycle_stage == "deleted":
        raise A1TrackingError(
            f"The MLflow run {run_id} recorded for {run_dir} no longer exists: it was deleted. "
            "Use --no-mlflow, or start the run folder again with run_a1.py --restart."
        )


@contextmanager
def reopen_parent(run_dir, config) -> Iterator[mlflow.ActiveRun]:
    """Reopen the parent MLflow run of a run folder, for the steps that come after ``run_a1.py``.

    The run's status is put back as it was when the block ends: reopening a finished run marks it
    running only while something is added to it.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        The run folder, ``<output_dir>/runs/<run name>/``.
    config : Config
        The main configuration: where MLflow stores its runs.

    Raises
    ------
    A1TrackingError
        If the folder records no run, or MLflow no longer has it.
    """
    from yg_eo_soilnet.tracking import configure_tracking_uri

    entry = recorded_run(run_dir)
    configure_tracking_uri(config)
    check_run(entry["run_id"], run_dir)
    with mlflow.start_run(run_id=entry["run_id"]) as run:
        yield run


def metric_name(*parts) -> str:
    """A metric name MLflow accepts and a reader can parse: each part slugged, joined by ``__``.

    Examples
    --------
    >>> metric_name("rmse", "logo_Oum Er Rbia", "global_aux")
    'rmse__logo_oum_er_rbia__global_aux'
    >>> metric_name("r2", "spatial_kfold_10000m", "local", "combined", "lo")
    'r2__spatial_kfold_10000m__local__combined__lo'
    """
    return "__".join(slug(part) for part in parts if part not in (None, ""))


def log_figures(figures: dict, folder) -> None:
    """Log each figure under ``figures/`` of the active run, with the tables and captions behind them.

    The PDF goes as a file and the PNG as a figure: ``mlflow.log_figure`` is what makes the PNG
    preview in the MLflow browser, and it is saved with the same resolution and trimming as the PNG
    on disk, so the two are the same image. The figures' tables (``figures/data/*.csv`` and the
    ``.tex`` tables) and ``captions.md`` are logged beside them.

    Parameters
    ----------
    figures : dict
        File name (``fig05_spatial``) to matplotlib figure, as :func:`~yg_eo_soilnet.a1.figures.render_all`
        returns them.
    folder : str or pathlib.Path
        Where :func:`~yg_eo_soilnet.a1.figures.render_all` saved them.
    """
    from yg_eo_soilnet.plot_style import SAVE_DPI

    folder = Path(folder)
    for name, figure in figures.items():
        pdf = folder / f"{name}.pdf"
        if pdf.exists():
            mlflow.log_artifact(str(pdf), "figures")
        mlflow.log_figure(figure, f"figures/{name}.png", save_kwargs={"dpi": SAVE_DPI, "bbox_inches": "tight"})
    data = folder / "data"
    for path in sorted(data.glob("*.csv")) + sorted(data.glob("*.tex")) if data.is_dir() else []:
        mlflow.log_artifact(str(path), "figures/data")
    captions = folder / "captions.md"
    if captions.exists():
        mlflow.log_artifact(str(captions), "figures")


def record_parent(run_dir, manifest: dict, run: mlflow.ActiveRun) -> None:
    """Write the parent run's ids into the run folder's manifest, straight away."""
    from yg_eo_soilnet.a1.runner import MANIFEST_FILE, _write_json

    manifest[MANIFEST_KEY] = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "tracking_uri": mlflow.get_tracking_uri(),
    }
    _write_json(Path(run_dir) / MANIFEST_FILE, manifest)


def recorded_run_id(manifest: dict) -> Optional[str]:
    """The parent run id a manifest holds, or None."""
    return (manifest.get(MANIFEST_KEY) or {}).get("run_id")
