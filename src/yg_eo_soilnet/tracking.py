"""Where runs are recorded, how abandoned ones are tidied up, and which saved model is champion.

Every tool configures MLflow through this module, so runs always land in the same place: the
``mlruns/`` folder beside the project unless ``MLFLOW_TRACKING_URI`` says otherwise, under the
:term:`experiment` named by ``MLFLOW_EXPERIMENT_NAME``.

It also repairs what a killed process leaves behind. A run whose process is killed outright stays
marked as still running for ever, and if it died at the wrong moment its record can be left empty,
which makes MLflow unable to read the whole experiment. Both are cleaned up at the start of a run,
and only for runs whose process is genuinely gone: a run another process is still writing is never
touched.
"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import mlflow
import numpy as np
import yaml

#: The experiment runs land in unless ``MLFLOW_EXPERIMENT_NAME`` says otherwise.
DEFAULT_EXPERIMENT_NAME = "Soil_Model_Training_v2"

_TRACKING_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME")


def tracking_settings(config_path: str | os.PathLike | None) -> SimpleNamespace:
    """Read where runs should go, from a configuration file and the environment.

    Only those two settings, not the whole configuration: where runs go must not depend on every
    other file being valid, or a typo somewhere unrelated would decide that a run is recorded
    nowhere. The environment wins over the file.

    Parameters
    ----------
    config_path : str or path-like or None
        The main configuration file. An unreadable one is treated as empty.

    Returns
    -------
    types.SimpleNamespace
        With ``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME``.
    """
    values: dict[str, str] = {}

    if config_path:
        try:
            with open(config_path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            document = {}
        common = document.get("common") if isinstance(document, dict) else {}
        for key in _TRACKING_KEYS:
            for source in (common if isinstance(common, dict) else {}, document if isinstance(document, dict) else {}):
                if key in source:
                    values[key] = str(source[key] or "")
                    break

    # The environment wins, as it does everywhere else in the configuration.
    for key in _TRACKING_KEYS:
        override = os.environ.get(key)
        if override is not None:
            values[key] = override

    return SimpleNamespace(
        MLFLOW_TRACKING_URI=values.get("MLFLOW_TRACKING_URI", ""),
        MLFLOW_EXPERIMENT_NAME=values.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )


def default_tracking_uri() -> str:
    """The ``mlruns/`` folder beside the project, as a URI.

    Found from this file's own location, not the working directory, so a run started from another
    folder does not quietly begin a second, empty store beside itself.
    """
    return (Path(__file__).resolve().parents[2] / "mlruns").as_uri()


def resolve_tracking_uri(config=None) -> str:
    """Where runs are recorded: ``MLFLOW_TRACKING_URI`` if set, else :func:`default_tracking_uri`."""
    configured = str(getattr(config, "MLFLOW_TRACKING_URI", "") or "").strip()
    return configured or default_tracking_uri()


def resolve_local_tracking_root(tracking_uri: str) -> Path | None:
    """The folder a file-based tracking URI points at, or None for a server or database."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(tracking_uri)


def configure_tracking_uri(config=None) -> str:
    """Point MLflow at where runs are recorded, leaving the current experiment alone.

    Used by the tools that reopen an existing run - which belongs to its own experiment already.

    Returns
    -------
    str
        The tracking URI now in force.
    """
    tracking_uri = resolve_tracking_uri(config)

    if urlparse(tracking_uri).scheme in ("", "file"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def configure_tracking(config=None, experiment_name: str | None = None) -> str:
    """Point MLflow at where runs are recorded, and at the experiment to record them under.

    Must be called before anything opens a run, or that run lands in whatever experiment happened
    to be current.

    Parameters
    ----------
    config : Config, optional
        Read for ``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME``.
    experiment_name : str, optional
        Use this experiment instead of the configured one.

    Returns
    -------
    str
        The experiment name in force.
    """
    configure_tracking_uri(config)

    name = experiment_name or str(
        getattr(config, "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME) or DEFAULT_EXPERIMENT_NAME
    )
    try:
        mlflow.set_experiment(name)
    except mlflow.exceptions.MlflowException:  # type: ignore[attr-defined]
        # Create it, or bring back one that was deleted.
        mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    return name


# --- tidying up runs whose process died ----------------------------------------------------
# A run whose process is killed outright stays marked as running for ever, which reads as "still
# working" or, worse, as a model that was made successfully. That kind of death cannot be caught,
# so the next run sweeps up after it. Each run records who wrote it, which is how a later process
# can tell an abandoned run from one still being written.

#: Tag recording which machine wrote a run.
HOST_NAME_TAG = "host_name"
#: Tag recording which process wrote it.
HOST_PID_TAG = "host_pid"


def run_owner_tags() -> dict[str, str]:
    """This machine and process, tagged on every run so abandoned ones can be recognised."""
    import socket

    return {HOST_NAME_TAG: socket.gethostname(), HOST_PID_TAG: str(os.getpid())}


def _process_is_alive(pid: int) -> bool:
    """Whether a process with this id is still running on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Running, but someone else's. Not ours to tidy up either way.
        return True
    except (OverflowError, ValueError):
        return False
    return True


def close_stale_runs(experiment_name: str | None = None, logger: Any = None) -> list[str]:
    """Mark as killed the runs whose process is gone, so they stop reading as still running.

    Deliberately narrow: only a run still marked running, written by this machine, whose process no
    longer exists. Sweeping more broadly would let one run end another that is still going.

    Parameters
    ----------
    experiment_name : str, optional
        Which experiment to sweep; the configured one unless given.
    logger : logging.Logger, optional
        Where the report goes.

    Returns
    -------
    list of str
        The runs closed. Never raises: failing to tidy up is not a reason to refuse to train.
    """
    import socket

    name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    try:
        client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            return []
        running = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=1000,
        )
    except Exception as exc:  # pragma: no cover - tracking store unreachable
        # Never fatal: refusing to train because old runs could not be tidied is the worse trade.
        if logger is not None:
            logger.warning(f"Could not sweep stale runs: {type(exc).__name__}: {exc}")
        return []

    hostname = socket.gethostname()
    this_pid = os.getpid()
    closed: list[str] = []
    for run in running:
        tags = run.data.tags
        if tags.get(HOST_NAME_TAG) != hostname:
            continue
        raw_pid = tags.get(HOST_PID_TAG)
        if raw_pid is None:
            # Written before runs recorded who owned them. Left alone rather than guessed at.
            continue
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid == this_pid or _process_is_alive(pid):
            continue
        try:
            client.set_terminated(run.info.run_id, "KILLED")
            closed.append(run.info.run_id)
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning(f"Could not terminate stale run {run.info.run_id}: {exc}")

    if closed and logger is not None:
        logger.info(
            f"Marked {len(closed)} abandoned run(s) as KILLED; their process is gone. "
            "A run left RUNNING is usually one the OOM killer took."
        )
    return closed


# --- repairing runs whose record was left empty --------------------------------------------
# The sweep above has to be able to list the runs, and it cannot if a process died while rewriting
# a run's own record: that file is emptied before it is rewritten, so dying in between leaves
# nothing behind. One such file makes MLflow unable to read the whole experiment, which surfaces as
# a crash in whatever touches MLflow next - here, at the end of an otherwise successful run.

# Folders that sit beside the runs and are not runs.
_RESERVED_EXPERIMENT_FOLDERS = ("tags", "datasets", "traces", "models")

# The timestamp in a run name, as main.py writes it: Run_20260922_113000.
_RUN_NAME_TIMESTAMP = re.compile(r"\d{8}_\d{6}")


def _read_run_tag(run_dir: Path, tag: str) -> str | None:
    """Read one of a run's tags straight off disk: MLflow keeps one file per tag."""
    try:
        return (run_dir / "tags" / tag).read_text().strip()
    except OSError:
        return None


def _meta_is_corrupt(meta_path: Path) -> bool:
    """Whether a run's record exists but no longer describes a run.

    A missing file is not counted: MLflow skips those by itself. Only an empty or unreadable one is
    what breaks reading the experiment.
    """
    try:
        loaded = yaml.safe_load(meta_path.read_text())
    except (OSError, yaml.YAMLError):
        return True
    return not isinstance(loaded, dict) or not loaded.get("run_id")


def _owned_by_live_process(run_dir: Path) -> bool:
    """Whether the process that wrote this run is still going.

    Same rule as :func:`close_stale_runs`: a run another process is still writing is never touched.
    """
    import socket

    if _read_run_tag(run_dir, HOST_NAME_TAG) != socket.gethostname():
        # Another machine's run, so its process id means nothing here.
        return False
    raw_pid = _read_run_tag(run_dir, HOST_PID_TAG)
    if raw_pid is None:
        return False
    try:
        return _process_is_alive(int(raw_pid))
    except (TypeError, ValueError):
        return False


def _experiment_artifact_location(experiment_dir: Path) -> str:
    """Where this experiment stores its files, which is fixed when the experiment is created."""
    try:
        loaded = yaml.safe_load((experiment_dir / "meta.yaml").read_text())
    except (OSError, yaml.YAMLError):
        loaded = None
    location = loaded.get("artifact_location") if isinstance(loaded, dict) else None
    return str(location or experiment_dir.as_uri())


def _rebuild_meta(run_dir: Path, experiment_id: str, experiment_dir: Path) -> dict:
    """Rebuild a run's lost record from the files beside it that survived.

    Everything needed is recoverable: the run id is its folder name, and its tags and settings were
    written as separate files. Only the times are estimated - the run's name carries its start, and
    the emptied file's own timestamp is when the process died.
    """
    run_id = run_dir.name
    run_name = _read_run_tag(run_dir, "mlflow.runName") or run_id

    match = _RUN_NAME_TIMESTAMP.search(run_name)
    start_time = None
    if match:
        try:
            # The name was written in local time, which is what this reads it back as.
            start_time = int(datetime.datetime.strptime(match.group(), "%Y%m%d_%H%M%S").timestamp() * 1000)
        except ValueError:
            start_time = None
    if start_time is None:
        start_time = int(run_dir.stat().st_mtime * 1000)

    end_time = max(int((run_dir / "meta.yaml").stat().st_mtime * 1000), start_time)
    artifact_location = _experiment_artifact_location(experiment_dir).rstrip("/")

    return {
        "artifact_uri": f"{artifact_location}/{run_id}/artifacts",
        "end_time": end_time,
        "entry_point_name": "",
        "experiment_id": str(experiment_id),
        "lifecycle_stage": "active",
        "run_id": run_id,
        "run_name": run_name,
        "source_name": "",
        "source_type": 4,
        "source_version": "",
        "start_time": start_time,
        # Killed: the process died partway through ending the run, which is why the file was
        # emptied. The same status the sweep above gives an abandoned run.
        "status": 5,
        "tags": [],
        "user_id": _read_run_tag(run_dir, "mlflow.user") or "",
    }


def _write_meta_atomically(meta_path: Path, payload: dict) -> None:
    """Write to a temporary file and move it into place.

    The damage being repaired is an emptied file, so a repair that emptied the file first could
    leave things exactly as it found them.
    """
    tmp_path = meta_path.parent / f".{meta_path.name}.repair"
    with open(tmp_path, "w") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, meta_path)


def repair_corrupt_runs(experiment_name: str | None = None, logger: Any = None) -> list[str]:
    """Rebuild the records of runs whose process died partway through ending them.

    Reads the store as plain files, because the damage is exactly what stops MLflow reading it. A
    repaired run comes back marked killed, keeping the settings and files that were never lost.

    Parameters
    ----------
    experiment_name : str, optional
        Which experiment to check; the configured one unless given.
    logger : logging.Logger, optional
        Where the report goes.

    Returns
    -------
    list of str
        The runs repaired. Never raises.
    """
    name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    try:
        root = resolve_local_tracking_root(mlflow.get_tracking_uri())
        if root is None:
            # A server or database keeps no such files.
            return []
        client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
        # Reads the experiment's own record only, so a damaged run cannot break this lookup.
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            return []
        experiment_dir = root / str(experiment.experiment_id)
        run_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir())
    except Exception as exc:  # pragma: no cover - tracking store unreachable
        if logger is not None:
            logger.warning(f"Could not scan for corrupt runs: {type(exc).__name__}: {exc}")
        return []

    repaired: list[str] = []
    for run_dir in run_dirs:
        if run_dir.name in _RESERVED_EXPERIMENT_FOLDERS:
            continue
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists() or not _meta_is_corrupt(meta_path):
            continue
        if _owned_by_live_process(run_dir):
            if logger is not None:
                logger.warning(
                    f"Run {run_dir.name} has an unreadable meta.yaml but its process is still "
                    "alive; leaving it alone in case the file is mid-write."
                )
            continue
        try:
            _write_meta_atomically(meta_path, _rebuild_meta(run_dir, experiment.experiment_id, experiment_dir))
            repaired.append(run_dir.name)
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning(f"Could not repair run {run_dir.name}: {type(exc).__name__}: {exc}")

    if repaired and logger is not None:
        logger.info(
            f"Rebuilt meta.yaml for {len(repaired)} run(s) and marked them KILLED: "
            f"{', '.join(repaired)}. A truncated meta.yaml is written when a process dies partway "
            "through ending a run, and one of them makes every MLflow read of this experiment fail."
        )
    return repaired


def start_child_run(run_name: str, tags: dict | None = None):
    """Open a :term:`sub-run`, inside the current run when there is one.

    Every run opened here records who wrote it, which is what lets a later run tell an abandoned
    run from one still being written.

    Parameters
    ----------
    run_name : str
        The run's name, such as ``clay_pct_soil_cnn``.
    tags : dict, optional
        Extra tags to record on it.

    Returns
    -------
    mlflow.ActiveRun
        Use it as a context manager: ``with start_child_run(name): ...``.
    """
    run = mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
    try:
        mlflow.set_tags({**run_owner_tags(), **(tags or {})})
    except Exception:
        # The run is already open but never reaches the caller, so nothing would ever close it -
        # and the run above it would then be closed in its place, leaving the real one open for
        # ever. Close it here, then let the caller see the failure.
        mlflow.end_run("FAILED")
        raise
    return run


def log_params_once(params: Any, logger: Any = None) -> None:
    """Record settings on the current run, keeping any value already recorded under that name.

    A recorded setting cannot be changed: writing a different value for the same name is an error.
    That error would land at the very end of a run, after every model was trained, and would take
    the summary and the leaderboard with it - so the clash is reported and skipped instead.

    Parameters
    ----------
    params : mapping
        The settings to record.
    logger : logging.Logger, optional
        Where a clash is reported.
    """
    params = dict(params)
    if not params:
        return

    run = mlflow.active_run()
    existing: dict = {}
    if run is not None:
        try:
            existing = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.params  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - unreadable store; let log_params speak for itself
            existing = {}

    writable = {}
    for key, value in params.items():
        # Compared as text, which is how MLflow stores them.
        current = existing.get(str(key))
        if current is not None and current != str(value):
            if logger is not None:
                logger.warning(
                    f"Param {key!r} is already logged as {current!r} on this run; keeping that and "
                    f"not overwriting it with {str(value)!r}. Two places are writing the same key."
                )
            continue
        writable[key] = value

    if writable:
        mlflow.log_params(writable)


def install_run_signal_handlers(logger: Any = None) -> None:
    """Mark the open runs as killed when the process is interrupted, then exit as usual.

    Covers Ctrl-C and an ordinary ``kill``. Nothing can cover a forced kill, which is why
    :func:`close_stale_runs` exists as well.
    """
    import signal

    def handler(signum, frame):
        try:
            while mlflow.active_run() is not None:
                mlflow.end_run("KILLED")
        except Exception:  # pragma: no cover - best effort during teardown
            pass
        if logger is not None:
            logger.warning(f"Received signal {signum}; marked the active run(s) KILLED.")
        # Then behave as the process normally would, so the exit status still says what happened.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):  # pragma: no cover - not on the main thread
            pass


#: The alias marking the best saved version of a model; see :term:`champion`.
CHAMPION_ALIAS = "champion"
#: The score the champion is chosen by. It means the same thing for both model families and is in
#: the target's own units; :mod:`yg_eo_soilnet.metrics` says which direction is better.
CHAMPION_METRIC = "rmse_test"


def _version_metric(client, version, metric_name: str) -> float | None:
    """One score of a saved model version, read from the run that produced it."""
    run_id = getattr(version, "run_id", None)
    if not run_id:
        return None
    try:
        value = client.get_run(run_id).data.metrics.get(metric_name)
    except Exception:
        # The run behind an aliased version can be deleted while the version survives.
        return None
    return None if value is None else float(value)


def promote_if_better(
    name: str,
    version: Any,
    metric_value: float | None,
    *,
    client=None,
    alias: str = CHAMPION_ALIAS,
    metric_name: str = CHAMPION_METRIC,
) -> dict:
    """Make a newly saved model the :term:`champion`, but only if it beats the current one.

    **What "champion" means here:** the best version *of this model, for this target* - not the best
    model for the target. Choosing between `soil_cnn` and XGBoost is the :term:`leaderboard`\'s job.

    The rules: with no champion yet, or one whose score can no longer be read, the new version
    takes the alias. A new version with no score never does - a model that failed to produce one
    must not ship because it has "no worse" score. Equal scores keep the current champion, so
    re-running the same configuration does not shuffle the alias.

    Parameters
    ----------
    name : str
        The registered model's name, ``<target>_<model>``.
    version : str or int or None
        The newly saved version.
    metric_value : float or None
        Its score on the test points.
    client : mlflow.MlflowClient, optional
        A client to use instead of a new one.
    alias : str, default "champion"
        The alias to move.
    metric_name : str, default "rmse_test"
        Which score decides.

    Returns
    -------
    dict
        What was decided: both scores, the versions, and a reason in words, so a promotion is on
        the record rather than a surprise.
    """
    from yg_eo_soilnet.metrics import METRIC_DIRECTION

    if version is None:
        return {"promoted": False, "reason": "the model was not registered"}

    if metric_value is None or not np.isfinite(metric_value):
        return {
            "promoted": False,
            "reason": f"the new version has no usable {metric_name}",
            "candidate": None,
        }

    if client is None:
        client = mlflow.MlflowClient()

    try:
        incumbent = client.get_model_version_by_alias(name, alias)
    except Exception:
        incumbent = None

    decision: dict[str, Any] = {
        "alias": alias,
        "metric": metric_name,
        "candidate": float(metric_value),
        "candidate_version": str(version),
    }

    if incumbent is None:
        client.set_registered_model_alias(name, alias, version)
        return {**decision, "promoted": True, "reason": f"no version was aliased {alias} yet"}

    decision["incumbent_version"] = str(incumbent.version)
    incumbent_value = _version_metric(client, incumbent, metric_name)
    decision["incumbent"] = incumbent_value

    if incumbent_value is None:
        client.set_registered_model_alias(name, alias, version)
        return {
            **decision,
            "promoted": True,
            "reason": f"the {alias} version has no readable {metric_name}",
        }

    stem = metric_name.split("_")[0]
    higher_is_better = METRIC_DIRECTION.get(stem) == "higher"
    better = metric_value > incumbent_value if higher_is_better else metric_value < incumbent_value

    if not better:
        return {
            **decision,
            "promoted": False,
            "reason": f"{metric_name} {metric_value:.6g} does not beat {incumbent_value:.6g}",
        }

    client.set_registered_model_alias(name, alias, version)
    return {
        **decision,
        "promoted": True,
        "reason": f"{metric_name} {metric_value:.6g} beats {incumbent_value:.6g}",
    }
