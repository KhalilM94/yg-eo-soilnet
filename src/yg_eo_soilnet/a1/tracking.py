"""One MLflow run per A1 run folder: what ``run_a1.py`` opens, and the later steps reopen.

``run_a1.py`` records each run folder under one *parent* MLflow run - its settings, its git commit,
and one child run per plan - and writes that run's id into the folder's ``manifest.json``. A resumed
``run_a1.py`` reopens the same parent, and so do ``evaluate.py`` and ``make_figures.py``, which add
the evaluation tables, the headline metrics and the figures to it. Everything one run folder
produced is then found in one place in the MLflow browser.

**Where A1 records** (:func:`a1_tracking_uri`, the one place every A1 script asks): a run folder's
recorded store when it can be reopened here (not the migrated ``mlruns/``); else the environment
variable ``MLFLOW_TRACKING_URI``; else the main configuration's ``MLFLOW_TRACKING_URI``; else
``sqlite:///<repository>/mlflow.db`` (:data:`A1_DB_FILE`). The metadata (runs, parameters, metrics, tags) live in that database, the
artifacts in files beside it (``mlruns/<experiment>/``), never inside it. ``main.py`` keeps its own
``mlruns/`` default. On a cluster, point ``MLFLOW_TRACKING_URI`` elsewhere or run with
``--no-mlflow`` and ingest the folders afterwards (``experiments/a1/mlflow_sync.py``).
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
#: The repository root, found from this file's own location (not the working directory).
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
#: A1's default MLflow store: a SQLite database at the repository root.
A1_DB_FILE = "mlflow.db"
#: The folder beside a SQLite store where new experiments keep their artifacts.
ARTIFACT_FOLDER = "mlruns"
#: How long a SQLite connection waits for another writer's lock before failing, in seconds.
SQLITE_TIMEOUT_S = 60


def a1_default_tracking_uri() -> str:
    """``sqlite:///<repository root>/mlflow.db``."""
    return f"sqlite:///{REPOSITORY_ROOT / A1_DB_FILE}"


def repository_path(path) -> str:
    """A path as it is stored: relative to the repository root when inside it, else unchanged.

    >>> repository_path(REPOSITORY_ROOT / "configs" / "a1" / "a1_xgb.yml")
    'configs/a1/a1_xgb.yml'
    >>> repository_path("/elsewhere/a.yml")
    '/elsewhere/a.yml'
    >>> repository_path("/home/x/yg-eo-soilnet-run/configs/a1/a1_xgb.yml")
    'configs/a1/a1_xgb.yml'
    """
    if path in (None, ""):
        return path
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(path)
    try:
        return candidate.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        pass
    # Another checkout of this repository (the run worktree): relative from its configs/, results/
    # or experiments/ part, when this repository has the same file.
    parts = candidate.parts
    for anchor in ("configs", "results", "experiments"):
        if anchor in parts:
            relative = Path(*parts[parts.index(anchor) :])
            if (REPOSITORY_ROOT / relative).exists():
                return relative.as_posix()
    return str(path)


def resolve_repository_path(path) -> Optional[Path]:
    """A stored path read back: relative to the repository root, or an old absolute one.

    An absolute path of another machine or checkout that no longer exists is rebased on this
    repository from its ``configs/``, ``results/`` or ``experiments/`` part.
    """
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        return REPOSITORY_ROOT / candidate
    if candidate.exists():
        return candidate
    parts = candidate.parts
    for anchor in ("configs", "results", "experiments"):
        if anchor in parts:
            rebased = REPOSITORY_ROOT.joinpath(*parts[parts.index(anchor) :])
            if rebased.exists():
                return rebased
    return candidate


def _sqlite_path(uri: str) -> Optional[Path]:
    if not str(uri).startswith("sqlite:///"):
        return None
    raw = str(uri).removeprefix("sqlite:///").split("?", 1)[0]
    path = Path(raw)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def stored_tracking_uri(uri: str) -> str:
    """A tracking URI as a manifest stores it: a SQLite file inside the repository becomes relative."""
    path = _sqlite_path(uri)
    if path is None:
        return uri
    return f"sqlite:///{repository_path(path)}"


def _is_migrated_file_store(uri: str) -> bool:
    """Whether a recorded URI is the repository's old ``mlruns/`` file store (or a link to it)."""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(str(uri))
    if parsed.scheme not in ("", "file"):
        return False
    path = Path(unquote(parsed.path) if parsed.scheme == "file" else str(uri))
    old = REPOSITORY_ROOT / ARTIFACT_FOLDER
    try:
        if path.resolve() == old.resolve():
            return True
    except OSError:
        pass
    # Another checkout's link to it (the run worktree's mlruns -> this mlruns), seen from here.
    return path.name == ARTIFACT_FOLDER and path.parent.name.startswith(REPOSITORY_ROOT.name)


def _usable_record(uri: str) -> bool:
    """Whether a recorded store can be reopened here: not the migrated ``mlruns/``, and still there."""
    from urllib.parse import unquote, urlparse

    if not uri or _is_migrated_file_store(uri):
        return False
    path = _sqlite_path(uri)
    if path is not None:
        return path.exists()
    parsed = urlparse(str(uri))
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path) if parsed.scheme == "file" else str(uri)).is_dir()
    return True  # a tracking server


def a1_tracking_uri(config=None, recorded: Optional[str] = None) -> str:
    """Where A1 records: the recorded store, else the environment, else the configuration, else SQLite.

    1. ``recorded``, the ``tracking_uri`` a run folder's manifest holds, when it can be reopened
       here (:func:`_usable_record`): a parent run lives where it was recorded. A relative SQLite
       path is resolved against the repository root.
    2. The environment variable ``MLFLOW_TRACKING_URI`` (MLflow also sets it when a URI is chosen
       in this process).
    3. The main configuration's ``MLFLOW_TRACKING_URI``.
    4. ``sqlite:///<repository root>/mlflow.db``.
    """
    import os

    if recorded and _usable_record(recorded):
        path = _sqlite_path(recorded)
        return f"sqlite:///{path}" if path is not None else recorded
    from_env = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if from_env:
        return from_env
    configured = str(getattr(config, "MLFLOW_TRACKING_URI", "") or "").strip()
    if configured:
        return configured
    return a1_default_tracking_uri()


def _prepare_sqlite(uri: str) -> None:
    """Write-ahead logging on a SQLite store (readers no longer block the writer); once per file.

    Best effort: the switch needs the file to itself, and is skipped while another process has it
    open (the mode persists once any process has set it).
    """
    import sqlite3

    path = _sqlite_path(uri)
    if path is None or not path.exists():
        return
    try:
        with sqlite3.connect(str(path), timeout=SQLITE_TIMEOUT_S) as connection:
            if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Switching needs the file to itself; another process has it open. The mode persists once
        # set, and MLflow's own busy timeout covers the rest, so this is best effort.
        pass


def configure_a1_tracking(config=None, experiment: Optional[str] = None, *, recorded: Optional[str] = None) -> str:
    """Point MLflow at A1's store (:func:`a1_tracking_uri`), and at ``experiment`` when given.

    A new experiment on a SQLite store keeps its artifacts in ``mlruns/<experiment>/`` beside the
    database, as files. A file store still needs ``MLFLOW_ALLOW_FILE_STORE``, set here. MLflow
    itself waits up to 20 s on another writer's lock (``PRAGMA busy_timeout``); write-ahead logging
    is switched on so concurrent readers never block it.

    Returns
    -------
    str
        The tracking URI now in force.
    """
    import os
    from types import SimpleNamespace
    from urllib.parse import urlparse

    uri = a1_tracking_uri(config, recorded)
    if urlparse(uri).scheme in ("", "file"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    if _sqlite_path(uri) is not None:
        # A new file gets MLflow's tables first; then write-ahead logging is switched on.
        mlflow.tracking.MlflowClient().search_experiments(max_results=1)
        _prepare_sqlite(uri)
    if experiment:
        path = _sqlite_path(uri)
        if path is not None:
            ensure_experiment(experiment, (path.parent / ARTIFACT_FOLDER / slug(experiment)).as_uri())
        from yg_eo_soilnet.tracking import configure_tracking

        # The shared function sets (or restores) the experiment; the URI it resolves is ours.
        configure_tracking(SimpleNamespace(MLFLOW_TRACKING_URI=uri), experiment)
    return uri


def ensure_experiment(name: str, artifact_location: Optional[str] = None, *, attempts: int = 5) -> str:
    """The experiment's id, creating it if needed - safe when several processes start at once.

    Two runs starting together (two checkouts, or a script and a sync) can both find the experiment
    missing; the second creation then fails on the database's unique name. That loser reads the
    experiment the winner made instead of failing.
    """
    import time

    client = mlflow.tracking.MlflowClient()
    for attempt in range(attempts):
        found = client.get_experiment_by_name(name)
        if found is not None:
            return found.experiment_id
        try:
            return client.create_experiment(name, artifact_location=artifact_location)
        except Exception:  # created meanwhile by another process (IntegrityError), or a busy lock
            time.sleep(0.2 * (attempt + 1))
    found = client.get_experiment_by_name(name)
    if found is None:
        raise A1TrackingError(f"Could not create or find the MLflow experiment {name!r}.")
    return found.experiment_id


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
    entry = recorded_run(run_dir)
    configure_a1_tracking(config, recorded=entry.get("tracking_uri"))
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


def figure_files(folder) -> list:
    """``(file, artifact path)`` for every figure file :func:`log_figures` adds, from a figures folder.

    The PNG and PDF of each figure and ``captions.md`` under ``figures/``, the tables under
    ``figures/data/`` - the same paths a live run gets, for logging a folder afterwards.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [(path, "figures") for path in sorted(folder.glob("*.png")) + sorted(folder.glob("*.pdf"))]
    data = folder / "data"
    if data.is_dir():
        found += [(path, "figures/data") for path in sorted(data.glob("*.csv")) + sorted(data.glob("*.tex"))]
    if (folder / "captions.md").exists():
        found.append((folder / "captions.md", "figures"))
    return found


def record_parent(run_dir, manifest: dict, run: mlflow.ActiveRun) -> None:
    """Write the parent run's ids into the run folder's manifest, straight away."""
    from yg_eo_soilnet.a1.runner import MANIFEST_FILE, _write_json

    manifest[MANIFEST_KEY] = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "tracking_uri": stored_tracking_uri(mlflow.get_tracking_uri()),
    }
    _write_json(Path(run_dir) / MANIFEST_FILE, manifest)


def recorded_run_id(manifest: dict) -> Optional[str]:
    """The parent run id a manifest holds, or None."""
    return (manifest.get(MANIFEST_KEY) or {}).get("run_id")
