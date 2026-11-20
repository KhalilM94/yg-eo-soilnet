"""Log finished A1 run folders to MLflow afterwards: ``results/`` is the record, MLflow the index.

A run made elsewhere - on a cluster with ``--no-mlflow``, or one whose logging failed - is logged
from its folder the way the scripts would have logged it (:func:`sync_run_folder`):

- a **parent run** with the settings' description as parameters (the run's own git commit, from the
  manifest), the run's tags, the settings file, ``metrics.csv``, ``manifest.json``,
  ``dissimilarity_summary.csv`` and the log as artifacts;
- one **child run** per plan and seed (E5: per plan, seed and variant) with every model's scores,
  computed from the saved predictions by the runner's own function;
- the **evaluation** tables (``evaluation/``), the headline metrics rebuilt from the first seed's
  predictions and those tables (:func:`headline_from_files`, no resampling), and the region tags;
- the **figures** (``figures/``: PDF, PNG, data and captions);
- for E5, the ``e5_summary`` child with the E5 tables and figures of the output folder.

It is idempotent. A parent is found by the run id the manifest already records in this store, or
by its tags - the run folder relative to the repository root (``a1_run_folder``) and a hash of the
plans it holds (``a1_plans_hash``) - and is updated: a child that exists is left as it is, a
parameter is only added, a metric only logged when missing or changed, an artifact overwritten in
place. The parent's id is then written into the manifest. Tags record the machine that ran the
folder (``host_name``, from the manifest; ``unknown`` for older ones) and the one that synced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: Tags matching a synced parent to its folder.
FOLDER_TAG = "a1_run_folder"
PLANS_TAG = "a1_plans_hash"
#: Tags saying a run came from a folder, and from where.
SYNCED_TAG = "a1_synced"
SYNC_HOST_TAG = "a1_sync_host"
#: The evaluation tables a run folder may hold (``evaluate.py``).
EVALUATION_FILES = (
    "scores.csv",
    "scores_by_class.csv",
    "intervals.csv",
    "intervals_by_region.csv",
    "differences.csv",
    "transfer.csv",
    "basin_errors.csv",
    "scores_mixed.csv",
    "scores_logo_matched.csv",
    "seed_summary.csv",
    "uncertainty_scores.csv",
    "region_names.json",
)
#: The E5 tables of the output folder (``evaluate_e5.py``).
E5_TABLES = ("scores.csv", "ablation.csv", "ts_representation.csv")


@dataclass
class SyncResult:
    """What one folder's sync did."""

    run_dir: Path
    run_id: str = ""
    created_parent: bool = False
    children_created: int = 0
    children_kept: int = 0
    metrics_logged: int = 0
    artifacts_logged: int = 0
    notes: list = field(default_factory=list)


def plans_hash(manifest: dict) -> str:
    """A hash of the plans a run folder holds (plan id and file hash), for matching its parent."""
    items = sorted((manifest.get("plans") or {}).items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def is_e5(manifest: dict) -> bool:
    """Whether a run folder holds E5 variants."""
    return bool(manifest.get("variants"))


def headline_from_files(run_dir: Path, first_seed: Optional[int]) -> dict:
    """The headline metrics ``evaluate.py`` logged, rebuilt from the predictions and its tables.

    The first seed's scores and the saved intervals, transfer, mixed and matched tables go through
    :func:`~yg_eo_soilnet.a1.evaluation.headline_table`; the quantile model's rows through
    :func:`~yg_eo_soilnet.a1.uncertainty.headline_rows`; with several seeds each metric's SD
    (``<name>_sd``) from ``seed_summary.csv``. Empty without ``evaluation/``.
    """
    from yg_eo_soilnet.a1 import evaluation
    from yg_eo_soilnet.a1.uncertainty import headline_rows

    folder = Path(run_dir) / "evaluation"
    if not (folder / "scores.csv").exists():
        return {}
    everything = evaluation.load_predictions(run_dir, first_seed=first_seed)
    if everything.empty:
        return {}
    seed = evaluation.seed_order(everything, first_seed)[0]
    rows = evaluation.accuracy_rows(everything[everything["seed"] == seed])
    frame = evaluation.pooled_rows(rows)

    def table(name: str) -> Optional[pd.DataFrame]:
        path = folder / name
        if not path.exists():
            return None
        found = pd.read_csv(path)
        if "seed" in found:
            found = found[found["seed"] == seed].drop(columns="seed")
        return found

    headline = evaluation.headline_table(
        frame,
        intervals=table("intervals.csv"),
        region_intervals=table("intervals_by_region.csv"),
        transfer=table("transfer.csv"),
        mixed=table("scores_mixed.csv"),
        matched=table("scores_logo_matched.csv"),
    )
    metrics = dict(zip(headline["name"], headline["value"]))
    uncertainty = table("uncertainty_scores.csv")
    if uncertainty is not None and not uncertainty.empty:
        metrics.update({row["name"]: row["value"] for row in headline_rows(uncertainty)})
    summary = table("seed_summary.csv")
    if summary is not None and "n_seeds" in summary and (summary["n_seeds"] > 1).any():
        spread = summary.dropna(subset=["sd"])
        metrics.update({f"{name}_sd": float(sd) for name, sd in zip(spread["name"], spread["sd"])})
    return {name: float(value) for name, value in metrics.items() if pd.notna(value)}


def _find_parent(client, experiment_id: str, folder: str, manifest: dict) -> Optional[str]:
    """The parent of this folder already in the store: by the manifest's run id, else by its tags."""
    recorded = (manifest.get("mlflow") or {}).get("run_id")
    if recorded:
        try:
            run = client.get_run(recorded)
            if run.info.experiment_id == experiment_id and run.info.lifecycle_stage == "active":
                return recorded
        except Exception:  # not in this store
            pass
    found = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{FOLDER_TAG} = '{folder}' and tags.{PLANS_TAG} = '{plans_hash(manifest)}'",
        max_results=1,
    )
    return found[0].info.run_id if found else None


def _children(client, experiment_id: str, parent_id: str) -> set:
    found = client.search_runs(
        [experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'", max_results=50000
    )
    return {run.info.run_name for run in found}


def _log_new_metrics(client, run_id: str, metrics: dict) -> int:
    """Log the metrics a run lacks or holds with another value; returns how many."""
    import mlflow

    have = client.get_run(run_id).data.metrics
    new = {name: value for name, value in metrics.items() if have.get(name) != value}
    if new:
        mlflow.log_metrics(new)
    return len(new)


def _log_new_params(client, run_id: str, params: dict) -> None:
    import mlflow

    have = client.get_run(run_id).data.params
    new = {name: value for name, value in params.items() if name not in have}
    if new:
        mlflow.log_params(new)


def _log_artifacts(files, artifact_path: Optional[str] = None) -> int:
    import mlflow

    count = 0
    for path in files:
        if path is not None and Path(path).is_file():
            mlflow.log_artifact(str(path), artifact_path)
            count += 1
    return count


def _run_kinds(rows: dict, manifest: dict) -> list:
    """The plan kinds the run fitted, in the plan index's order of kinds."""
    from yg_eo_soilnet.a1.plans import CONTROL, LOGO, MIXED, RANDOM, SPATIAL_KFOLD

    found = {rows[plan].kind for plan in manifest.get("plans") or {} if plan in rows}
    order = [RANDOM, SPATIAL_KFOLD, LOGO, MIXED, CONTROL]
    return [kind for kind in order if kind in found] + sorted(found - set(order))


def _fitted_groups(run_dir: Path, e5: bool) -> bool:
    """Whether the run fitted models on single covariate groups (``run_a1.py --groups``)."""
    from yg_eo_soilnet.a1.config import COMBINED_GROUP

    if e5:
        return False
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return False
    groups = set(pd.read_csv(metrics, usecols=["covariate_group"])["covariate_group"].dropna().astype(str))
    return bool(groups - {COMBINED_GROUP})


def _parent_params(a1, manifest: dict, run_dir: Path, rows: dict) -> dict:
    """The parent's parameters as ``run_a1.py`` logs them, with the run's own commit and paths.

    The settings get the manifest's overrides back (``dev_mode``); the kinds come from the plans the
    run holds and ``a1_groups`` from its predictions. Without readable settings, only what the
    manifest says.
    """
    from dataclasses import replace

    from yg_eo_soilnet.a1.runner import parent_params
    from yg_eo_soilnet.a1.settings import dev_mode_settings
    from yg_eo_soilnet.a1.tracking import repository_path

    overrides = manifest.get("overrides") or {}
    e5 = is_e5(manifest)
    params = {}
    if a1 is not None:
        if overrides.get("dev_mode"):
            a1 = replace(a1, settings=dev_mode_settings(a1.settings))
        params = parent_params(
            a1,
            kinds=_run_kinds(rows, manifest),
            groups=_fitted_groups(run_dir, e5),
            dev_mode=bool(overrides.get("dev_mode")),
            devices=manifest.get("devices"),
        )
    params["a1_git_commit"] = manifest.get("git_commit") or params.get("a1_git_commit", "unknown")
    params["a1_git_dirty"] = manifest.get("git_dirty", params.get("a1_git_dirty", ""))
    params["a1_config_path"] = repository_path(manifest.get("settings") or params.get("a1_config_path", ""))
    params.update({f"a1_device_{stage}": name for stage, name in (manifest.get("devices") or {}).items()})
    return params


def _plan_rows(run_dir: Path) -> dict:
    """Each plan's index row (``plan_id, kind, held_out``) from the plans the run read."""
    from yg_eo_soilnet.a1.plans import INDEX_FILE

    path = run_dir.parent.parent / "plans" / INDEX_FILE
    if not path.exists():
        return {}
    index = pd.read_csv(path)
    return {row.plan_id: row for row in index.itertuples(index=False)}


def _log_variant_children(run_dir: Path, plan_row, seed: int, existing: set, tags: dict) -> tuple[int, int]:
    """E5's children of one plan and seed, those not yet there; returns (created, kept)."""
    import mlflow

    from yg_eo_soilnet.a1.layout import prediction_dir
    from yg_eo_soilnet.metrics import regression_metrics
    from yg_eo_soilnet.tracking import start_child_run

    held_out = "" if pd.isna(plan_row.held_out) else str(plan_row.held_out)
    base = {"plan_id": plan_row.plan_id, "kind": plan_row.kind, "held_out": held_out, "seed": str(seed), **tags}
    created = kept = 0
    for path in sorted(prediction_dir(run_dir, plan_row.plan_id, seed).glob("*.parquet")):
        frame = pd.read_parquet(path)
        variant = str(frame["covariate_group"].iloc[0])
        name = f"{plan_row.plan_id}__seed_{seed}__{variant}"
        if name in existing:
            kept += 1
            continue
        with start_child_run(name, tags={**base, "variant": variant}):
            mlflow.log_params({"seed": int(seed), "variant": variant})
            metrics = {}
            for (split,), part in frame.groupby(["split"]):
                metrics.update(regression_metrics(part["observed"], part["predicted"], split=split))
            mlflow.log_metrics(metrics)
        created += 1
    return created, kept


def sync_run_folder(
    run_dir, *, config=None, experiment: Optional[str] = None, write_manifest: bool = True
) -> SyncResult:
    """Log one finished run folder (``<output_dir>/runs/<run name>``) to the current A1 store.

    Parameters
    ----------
    run_dir : str or pathlib.Path
    config : Config, optional
        The main configuration, for ``MLFLOW_TRACKING_URI`` (else the environment or the default).
    experiment : str, optional
        The experiment; else the settings' ``mlflow_experiment``. Needed when the settings file the
        manifest names cannot be read here.
    write_manifest : bool, default True
        Write the parent's id and store into the manifest; off to sync a copy into a scratch store
        without touching the folder's record.

    Returns
    -------
    SyncResult
    """
    import mlflow

    from yg_eo_soilnet.a1.config import load_a1_config
    from yg_eo_soilnet.a1.runner import (
        DISSIMILARITY_SUMMARY_FILE,
        LOG_FILE,
        MANIFEST_FILE,
        METRICS_FILE,
        _log_plan,
        _write_json,
    )
    from yg_eo_soilnet.a1.tracking import (
        MANIFEST_KEY,
        configure_a1_tracking,
        figure_files,
        repository_path,
        resolve_repository_path,
        stored_tracking_uri,
    )

    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"No {MANIFEST_FILE} in {run_dir}: is it a run folder (<output_dir>/runs/<name>)?")
    manifest = json.loads(manifest_path.read_text())
    result = SyncResult(run_dir=run_dir)

    settings_path = resolve_repository_path(manifest.get("settings"))
    a1 = None
    if settings_path is not None and settings_path.exists():
        try:
            a1 = load_a1_config(str(settings_path))
        except Exception as error:  # a settings file that no longer loads: log what the folder holds
            result.notes.append(f"settings {settings_path} not loaded ({error}); parameters from the manifest only")
    else:
        result.notes.append(f"settings file {manifest.get('settings')} not found; parameters from the manifest only")
        settings_path = None
    name = experiment or (a1.settings.mlflow_experiment if a1 is not None else None)
    if not name:
        raise ValueError(f"{run_dir}: the settings file cannot be read here; pass the experiment (--experiment).")
    if config is None and a1 is not None:
        config = a1.settings.load_config()
    configure_a1_tracking(config, name)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(name).experiment_id

    folder = repository_path(run_dir)
    e5 = is_e5(manifest)
    parent_id = _find_parent(client, experiment_id, folder, manifest)
    tags = {
        FOLDER_TAG: folder,
        PLANS_TAG: plans_hash(manifest),
        SYNC_HOST_TAG: socket.gethostname(),
        "a1_run_name": run_dir.name,
        "a1_git_commit": manifest.get("git_commit") or "unknown",
    }
    if parent_id is None:
        stamp = str(manifest.get("created", "")).replace("-", "").replace(":", "").replace("T", "_")
        parent = mlflow.start_run(run_name=f"{'e5' if e5 else 'A1'}_{run_dir.name}_{stamp}".rstrip("_"))
        # A run that only ever existed as a folder: the machine that ran it, from the manifest.
        tags.update({SYNCED_TAG: "true", "host_name": manifest.get("host", "unknown")})
        result.created_parent = True
    else:
        parent = mlflow.start_run(run_id=parent_id)
    with parent:
        result.run_id = parent.info.run_id
        mlflow.set_tags(tags)
        rows = _plan_rows(run_dir)
        _log_new_params(client, result.run_id, _parent_params(a1, manifest, run_dir, rows))

        existing = _children(client, experiment_id, result.run_id)
        devices = manifest.get("devices") or {}
        seeds = manifest.get("seeds") or []
        first_seed = seeds[0] if seeds else None
        child_tags = {"experiment": "E5"} if e5 else {}
        for plan_id in sorted(manifest.get("plans") or {}):
            row = rows.get(plan_id)
            if row is None:
                result.notes.append(f"{plan_id}: not in the plan index; no child run")
                continue
            for seed in sorted((manifest.get("seeds_done") or {}).get(plan_id, [])):
                if e5:
                    created, kept = _log_variant_children(run_dir, row, int(seed), existing, child_tags)
                    result.children_created += created
                    result.children_kept += kept
                elif f"{plan_id}__seed_{seed}" in existing:
                    result.children_kept += 1
                else:
                    _log_plan(row, int(seed), run_dir, devices, first=seed == first_seed, tags=child_tags or None)
                    result.children_created += 1

        files = [
            settings_path,
            run_dir / METRICS_FILE,
            run_dir / MANIFEST_FILE,
            run_dir / DISSIMILARITY_SUMMARY_FILE,
            run_dir / LOG_FILE,
        ]
        result.artifacts_logged += _log_artifacts(files)
        evaluation_dir = run_dir / "evaluation"
        result.artifacts_logged += _log_artifacts([evaluation_dir / name for name in EVALUATION_FILES], "evaluation")
        result.metrics_logged += _log_new_metrics(client, result.run_id, headline_from_files(run_dir, first_seed))
        regions_path = evaluation_dir / "region_names.json"
        if regions_path.exists():
            regions = json.loads(regions_path.read_text())
            mlflow.set_tags({f"region__{key}": value for key, value in regions.items()})
        for path, artifact_path in figure_files(run_dir / "figures"):
            result.artifacts_logged += _log_artifacts([path], artifact_path)
        if e5:
            result.children_created += _sync_e5_summary(experiment_id, result, run_dir, existing)

    if write_manifest:
        manifest[MANIFEST_KEY] = {
            "run_id": result.run_id,
            "experiment_id": experiment_id,
            "tracking_uri": stored_tracking_uri(mlflow.get_tracking_uri()),
        }
        _write_json(manifest_path, manifest)
    return result


def _sync_e5_summary(experiment_id: str, result: SyncResult, run_dir: Path, existing: set) -> int:
    """E5's ``e5_summary`` child with the output folder's tables and figures, when there and missing."""
    import mlflow

    out = run_dir.parent.parent
    tables = [out / name for name in E5_TABLES if (out / name).exists()]
    if not tables or "e5_summary" in existing:
        return 0
    tags = {"experiment": "E5", "a1_step": "e5_summary", "mlflow.parentRunId": result.run_id}
    with mlflow.start_run(run_name="e5_summary", experiment_id=experiment_id, nested=True, tags=tags):
        result.artifacts_logged += _log_artifacts(tables)
        if (out / "figures").is_dir():
            mlflow.log_artifacts(str(out / "figures"), "figures")
    return 1
