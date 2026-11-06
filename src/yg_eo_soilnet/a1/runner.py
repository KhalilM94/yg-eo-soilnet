"""Run the A1 experiments over the plan index, resumably, writing one table per plan and model.

For each plan of ``index.csv`` (written by ``experiments/a1/make_plans.py``):

1. stage 1 predicts CaCO3 and CEC for every point (:func:`~yg_eo_soilnet.a1.auxiliary.predict_auxiliary`);
   its audit table is saved as ``stage1/<plan_id>.parquet``;
2. each stage-2 model (:mod:`yg_eo_soilnet.a1.stage2`) is fitted and its test predictions (and
   validation predictions, when the plans have a validation set; by default they do not) saved as ``predictions/<plan_id>/<model>__<covariate group>.parquet``;
3. with ``groups``, ``global_aux`` is also fitted on each covariate group alone, and for ``logo``
   plans a within-region ``local`` model is scored inside the held-out region.

Everything lands in one run folder, ``<output_dir>/runs/<run name>/``. A file that already exists is
not computed again, so an interrupted run carries on where it stopped. Files are written under a
temporary name and renamed when complete, so a run killed mid-write never leaves a half-file that
looks finished. The folder's ``manifest.json`` records what the outputs were made from - a hash of
the settings and of every plan - and a run whose settings or plans differ refuses to add to it,
since its files would silently mix two experiments; ``restart`` clears it instead. The same goes
for a folder written in an older file format (:data:`PREDICTION_FORMAT`).

Every row of a predictions file carries what the evaluation needs without the data or the settings
(:mod:`yg_eo_soilnet.a1.evaluation`): the plan's held-out region, share and repeat, and the point's
position, from which the spatial blocks of the bootstrap are drawn.

Each ``logo`` plan also gets ``dissimilarity/<plan_id>.parquet``: how far each point lies from the
plan's training points, in the inputs of ``global_aux`` weighted by its feature importances
(:mod:`yg_eo_soilnet.a1.dissimilarity`), and the run ends with ``dissimilarity_summary.csv``, one
row per held-out region.

Each plan that computed something gets an MLflow sub-run with its scores; the run's parent records
the git commit and the settings, and at the end a ``metrics.csv`` of every plan, model, group and
split - recomputed from all the prediction files, resumed ones included.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1.auxiliary import AuxiliaryPredictions, predict_auxiliary
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP, COMBINED_GROUP, A1Config
from yg_eo_soilnet.a1.dissimilarity import importance_weights, plan_dissimilarity, region_summary
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.plans import CONTROL, INDEX_COLUMNS, INDEX_FILE, LOGO, MIXED
from yg_eo_soilnet.a1.settings import dev_mode_settings
from yg_eo_soilnet.a1.stage2 import (
    DIAGNOSTIC_MODELS,
    ROLE_DIAGNOSTIC,
    ROLE_MODEL,
    GLOBAL_AUX,
    LOCAL,
    MODELS,
    ModelResult,
    build_context,
    run_group,
    local_folds,
    run_local,
    run_model,
)
from yg_eo_soilnet.datamodules.splitting import TEST, SplitPlan
from yg_eo_soilnet.metrics import regression_metrics

#: The columns of every predictions file, in order.
PREDICTION_COLUMNS = (
    "plan_id",
    "kind",
    "held_out",
    "fraction",
    "repeat",
    "block_size_m",
    "point_id",
    "split",
    "region",
    "lat",
    "lon",
    "carbonate_class_measured",
    "carbonate_class_predicted",
    "model",
    "covariate_group",
    "observed",
    "predicted",
    "fallback",
    "local_fold",
    "local_block",
    "model_role",
    "tuned_params",
)
#: The version of that layout, recorded in the manifest. A run folder written in another version is
#: refused like one made from other settings: its old files would sit beside new ones.
PREDICTION_FORMAT = 4
MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.csv"
DISSIMILARITY_DIR = "dissimilarity"
DISSIMILARITY_SUMMARY_FILE = "dissimilarity_summary.csv"


class RunMismatch(RuntimeError):
    """The run folder holds outputs made from other settings or other plans."""


@dataclass
class RunResult:
    """What a run did.

    Attributes
    ----------
    run_dir : pathlib.Path
    computed : list of str
        The output files written by this run.
    skipped : list of str
        The output files that already existed.
    metrics : pandas.DataFrame
        Every plan x model x group x split, from every prediction file in the folder.
    dissimilarity : pandas.DataFrame
        One row per ``logo`` plan: how far its held-out region lies from its training points
        (:func:`~yg_eo_soilnet.a1.dissimilarity.region_summary`); empty without ``logo`` plans.
    notes : list of str
        Things not done and why - a ``local`` score skipped for a small region, say.
    """

    run_dir: Path
    computed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    dissimilarity: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)


def run_plans(
    a1: A1Config,
    *,
    plans_dir: Optional[str] = None,
    run_name: str = "default",
    kinds: Optional[Iterable[str]] = None,
    groups: bool = False,
    restart: bool = False,
    tracking: bool = True,
    dev_mode: bool = False,
    logger: Optional[logging.Logger] = None,
) -> RunResult:
    """Run every plan of the index (or of the kinds asked for), skipping what is already done.

    Parameters
    ----------
    a1 : A1Config
        From :func:`~yg_eo_soilnet.a1.config.load_a1_config`.
    plans_dir : str, optional
        The folder ``make_plans.py`` wrote; ``<output_dir>/plans`` unless given.
    run_name : str, default "default"
        The run folder's name, under ``<output_dir>/runs/``.
    kinds : iterable of str, optional
        Only plans of these kinds; every kind unless given. ``mixed`` brings its ``control`` plans
        with it, as in ``make_plans.py``: a mixed score means little without them.
    groups : bool, default False
        Also fit ``global_aux`` per covariate group, and the ``local`` models of ``logo`` plans.
    restart : bool, default False
        Clear a run folder made from other settings or plans, instead of refusing to add to it.
    tracking : bool, default True
        Record the run in MLflow.
    dev_mode : bool, default False
        Cut the models down to the quickest run that still goes through every step
        (:func:`~yg_eo_soilnet.a1.settings.dev_mode_settings`): it shows the pipeline works, and its
        scores mean nothing. Recorded in the run folder's settings hash, so a dev-mode run never
        resumes into a full one.
    logger : logging.Logger, optional

    Returns
    -------
    RunResult

    Raises
    ------
    FileNotFoundError
        If there is no plan index.
    RunMismatch
        If the run folder was made from other settings or plans and ``restart`` is not set.
    """
    logger = logger or logging.getLogger(__name__)
    overrides = None
    if dev_mode:
        a1 = replace(a1, settings=dev_mode_settings(a1.settings))
        overrides = {"dev_mode": True}
        logger.warning(
            "Dev mode: XGBoost in both stages, 50 trees, no tuning, 2 inner folds. This run shows the pipeline "
            "works; its scores mean nothing."
        )
    settings = a1.settings
    config = settings.load_config()
    plans_path = Path(plans_dir or Path(settings.output_dir) / "plans")
    index_path = plans_path / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"No plan index at {index_path}. Write the plans first: python experiments/a1/make_plans.py "
            f"--config {settings.source_path or '<settings>'}"
        )
    index = pd.read_csv(index_path, keep_default_na=False, na_values=[""])
    missing = [column for column in INDEX_COLUMNS if column not in index.columns]
    if missing:
        raise ValueError(
            f"{index_path} has no {missing}: the plans were written by an older make_plans.py. Write them again "
            f"(python experiments/a1/make_plans.py --config {settings.source_path or '<settings>'}), then run with "
            "--restart."
        )
    if kinds is not None:
        kinds = list(kinds)
        if MIXED in kinds and CONTROL not in kinds:
            kinds.append(CONTROL)
        index = index[index["kind"].isin(kinds)]
    run_dir = Path(settings.output_dir) / "runs" / run_name
    manifest = _check_manifest(
        a1, config, plans_path, index, run_dir, restart=restart, overrides=overrides, logger=logger
    )

    table = build_table(settings, config, logger)
    resolved = a1.resolve_groups(table, logger=logger)
    result = RunResult(run_dir=run_dir)

    parent = (
        _start_parent(a1, config, run_name, kinds, groups, dev_mode, manifest=manifest, run_dir=run_dir)
        if tracking
        else None
    )
    try:
        for row in index.itertuples(index=False):
            computed_here = _run_one_plan(
                row, plans_path, run_dir, table, resolved, a1, config, groups=groups, result=result, logger=logger
            )
            if tracking and computed_here:
                _log_plan(row, run_dir)
        manifest["plans"].update(
            {row.plan_id: _file_hash(plans_path / f"{row.plan_id}.parquet") for row in index.itertuples()}
        )
        _write_json(run_dir / MANIFEST_FILE, manifest)
        result.metrics = collect_metrics(run_dir)
        _atomic_write_csv(result.metrics, run_dir / METRICS_FILE)
        result.dissimilarity = collect_dissimilarity(run_dir, settings)
        if not result.dissimilarity.empty:
            _atomic_write_csv(result.dissimilarity, run_dir / DISSIMILARITY_SUMMARY_FILE)
        if tracking:
            import mlflow

            mlflow.log_artifact(str(run_dir / METRICS_FILE))
            mlflow.log_artifact(str(run_dir / MANIFEST_FILE))
            if not result.dissimilarity.empty:
                mlflow.log_artifact(str(run_dir / DISSIMILARITY_SUMMARY_FILE))
    finally:
        if parent is not None:
            import mlflow

            mlflow.end_run()
    logger.info(
        f"A1 run {run_name}: {len(result.computed)} file(s) written, {len(result.skipped)} already there, in {run_dir}"
    )
    return result


# --- one plan --------------------------------------------------------------------------------


def _run_one_plan(row, plans_path, run_dir, table, resolved, a1, config, *, groups, result, logger) -> bool:
    """Stage 1 and every stage-2 output of one plan; True when anything was computed."""
    settings = a1.settings
    plan = SplitPlan.from_frame(pd.read_parquet(plans_path / f"{row.plan_id}.parquet"), strategy=row.kind)
    out = run_dir / "predictions" / row.plan_id

    jobs = [(model, COMBINED_GROUP) for model in MODELS]
    if groups:
        jobs += [(GLOBAL_AUX, name) for name in resolved.groups]
        if row.kind == LOGO:
            jobs += [(LOCAL, name) for name in [*resolved.groups, COMBINED_GROUP]]
    todo = [(model, group) for model, group in jobs if not (out / _output_name(model, group)).exists()]
    result.skipped += [str(out / _output_name(model, group)) for model, group in jobs if (model, group) not in todo]
    dissimilarity_path = run_dir / DISSIMILARITY_DIR / f"{row.plan_id}.parquet"
    needs_dissimilarity = row.kind == LOGO and not dissimilarity_path.exists()
    if row.kind == LOGO and not needs_dissimilarity:
        result.skipped.append(str(dissimilarity_path))
    if not todo and not needs_dissimilarity:
        return False

    auxiliary = _stage1(row, plan, table, settings, config, run_dir, logger)
    context = build_context(table, plan, auxiliary, settings, config)
    auxiliary_columns = list(resolved.groups.get(AUXILIARY_GROUP, ()))
    global_columns = [column for column in resolved.combined if column not in auxiliary_columns]
    region = table.frame[table.region_column] if table.region_column else None

    for model, group in todo:
        if model == LOCAL:
            columns = list(resolved.combined) if group == COMBINED_GROUP else list(resolved.groups[group])
            local = local_folds(context, plan.point_ids_for(TEST), table, logger=logger)
            outcome = run_local(context, columns, local)
            if outcome is None:
                note = f"{row.plan_id}: no local score - {local.reason}"
                if note not in result.notes:
                    result.notes.append(note)
                continue
        elif group == COMBINED_GROUP:
            outcome = run_model(
                context, model, global_columns=global_columns, auxiliary_columns=auxiliary_columns, logger=logger
            )
        else:
            outcome = run_group(context, list(resolved.groups[group]))
        frame = _rows(row, context, outcome, model, group, region, table)
        path = out / _output_name(model, group)
        _atomic_write_parquet(frame, path)
        result.computed.append(str(path))

    if needs_dissimilarity:
        frame = _dissimilarity(row, context, table, global_columns, auxiliary_columns, result, logger)
        _atomic_write_parquet(frame, dissimilarity_path)
        result.computed.append(str(dissimilarity_path))
    return True


def _dissimilarity(row, context, table, global_columns, auxiliary_columns, result, logger) -> pd.DataFrame:
    """How far each point of a ``logo`` plan lies from its training points, in ``global_aux``'s inputs.

    The inputs are weighted by ``global_aux``'s feature importances; that model is fitted here if
    its predictions were already on disk.
    """
    settings = context.settings
    outcome = run_model(
        context, GLOBAL_AUX, global_columns=global_columns, auxiliary_columns=auxiliary_columns, logger=logger
    )
    columns = list(global_columns) + [column for column in auxiliary_columns if column not in global_columns]
    weights = importance_weights(outcome.fitted, columns) if outcome.fitted is not None else None
    if weights is None:
        result.notes.append(
            f"{row.plan_id}: {GLOBAL_AUX} has no feature importances; the DI weighs every input the same"
        )

    carbonate = settings.carbonate.column
    lab = {"som": settings.target, "caco3": carbonate}
    cec = [target for target in settings.auxiliary.targets if target != carbonate]
    if cec:
        lab["cec"] = cec[0]
    labels = pd.DataFrame({name: table.frame[column] for name, column in lab.items() if column in table.frame})

    frame = plan_dissimilarity(
        context.X[columns],
        context.split,
        train_ids=context.train_ids,
        weights=weights,
        folds=context.folds,
        categorical=context.categorical,
        labels=labels,
        carbonate_class=context.class_measured,
    )
    left_out = json.loads(frame["left_out"].iloc[0])["categorical"]
    note = f"The dissimilarity index leaves out the categorical input(s) {left_out}"
    if left_out and note not in result.notes:
        result.notes.append(note)
    frame.insert(0, "held_out", None if pd.isna(row.held_out) else str(row.held_out))
    frame.insert(0, "plan_id", row.plan_id)
    return frame


def _stage1(row, plan, table, settings, config, run_dir, logger) -> AuxiliaryPredictions:
    """The plan's stage-1 predictions: read back when saved, otherwise computed and saved."""
    path = run_dir / "stage1" / f"{row.plan_id}.parquet"
    if path.exists():
        return _auxiliary_from_frame(pd.read_parquet(path), settings)
    auxiliary = predict_auxiliary(table, plan, settings, config, kind=row.kind, logger=logger)
    # The point ids are the frame's index; written as a column, so they survive the file.
    _atomic_write_parquet(auxiliary.to_frame().reset_index(), path)
    return auxiliary


def _auxiliary_from_frame(frame: pd.DataFrame, settings) -> AuxiliaryPredictions:
    """Rebuild stage 1's result from its saved audit table."""
    from yg_eo_soilnet.a1.config import stage1_prediction_column

    if "point_id" not in frame.columns:
        raise ValueError("A saved stage-1 table has no point_id column; delete it so it is computed again.")
    frame = frame.set_index("point_id")
    columns = [stage1_prediction_column(target) for target in settings.auxiliary.targets]
    audit = {target: frame[f"predicted_by_{target}"] for target in settings.auxiliary.targets}
    folds = frame["inner_fold"].astype(int)
    return AuxiliaryPredictions(
        predictions=frame[columns],
        folds=folds,
        predicted_by=pd.DataFrame(audit, index=frame.index),
        split=frame["split"],
        cv_kind="loaded",
        n_folds=int(folds[folds >= 0].nunique()),
    )


def _rows(row, context, outcome: ModelResult, model, group, region, table) -> pd.DataFrame:
    """The predictions table of one model on one plan."""
    ids = outcome.predicted.index
    coordinates = table.coordinates.reindex(ids)
    return pd.DataFrame(
        {
            "plan_id": row.plan_id,
            "kind": row.kind,
            "held_out": None if pd.isna(row.held_out) else str(row.held_out),
            # Whole float columns even when empty: a column of bare NaN would be read back as object.
            "fraction": np.full(len(ids), float(row.fraction) if pd.notna(row.fraction) else np.nan),
            "repeat": np.full(len(ids), float(row.repeat) if pd.notna(row.repeat) else np.nan),
            "block_size_m": np.full(len(ids), float(row.block_size_m) if pd.notna(row.block_size_m) else np.nan),
            "point_id": ids.to_numpy(),
            "split": context.split.reindex(ids).to_numpy(),
            "region": (region.reindex(ids).to_numpy() if region is not None else None),
            "lat": coordinates["lat"].to_numpy(dtype=float),
            "lon": coordinates["lon"].to_numpy(dtype=float),
            "carbonate_class_measured": context.class_measured.reindex(ids).astype(object).to_numpy(),
            "carbonate_class_predicted": context.class_predicted.reindex(ids).astype(object).to_numpy(),
            "model": model,
            "covariate_group": group,
            "observed": context.y.reindex(ids).to_numpy(dtype=float),
            "predicted": outcome.predicted.to_numpy(dtype=float),
            "fallback": outcome.fallback.reindex(ids).fillna(False).to_numpy(dtype=bool),
            "local_fold": _local_column(outcome.local_fold, ids),
            "local_block": _local_column(outcome.local_block, ids),
            "model_role": outcome.role,
            "tuned_params": outcome.tuned_params_json,
        },
        columns=list(PREDICTION_COLUMNS),
    )


def _local_column(values: Optional[pd.Series], ids: pd.Index) -> np.ndarray:
    """A ``local`` model's per-point fold or block, as floats; NaN for every other model."""
    if values is None:
        return np.full(len(ids), np.nan)
    return values.reindex(ids).to_numpy(dtype=float)


# --- scores ------------------------------------------------------------------------------------


def collect_metrics(run_dir: Path) -> pd.DataFrame:
    """Score every prediction file of a run: one row per plan, model, covariate group and split.

    Returns
    -------
    pandas.DataFrame
        ``plan_id, kind, model, covariate_group, model_role, split, n, rmse, mae, bias, r2, rpd,
        rpiq, n_fallback``.
    """
    rows = []
    for path in sorted((Path(run_dir) / "predictions").glob("*/*.parquet")):
        frame = pd.read_parquet(path)
        for (split,), part in frame.groupby(["split"]):
            scores = regression_metrics(part["observed"], part["predicted"], split="x")
            first = part.iloc[0]
            rows.append(
                {
                    "plan_id": first["plan_id"],
                    "kind": first["kind"],
                    "model": first["model"],
                    "covariate_group": first["covariate_group"],
                    "model_role": first["model_role"],
                    "split": split,
                    **{name[: -len("_x")]: value for name, value in scores.items()},
                    "n_fallback": int(part["fallback"].sum()),
                }
            )
    return pd.DataFrame(rows)


def collect_dissimilarity(run_dir: Path, settings) -> pd.DataFrame:
    """One row per ``logo`` plan of the run: :func:`~yg_eo_soilnet.a1.dissimilarity.region_summary`."""
    paths = sorted((Path(run_dir) / DISSIMILARITY_DIR).glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return region_summary(frame, labels=list(settings.carbonate.labels))


# --- the run folder ------------------------------------------------------------------------------


def _check_manifest(a1, config, plans_path, index, run_dir, *, restart, logger, overrides=None) -> dict:
    """Make sure the run folder's outputs came from these settings and plans; clear it on restart."""
    fingerprint = _config_hash(a1, config, overrides)
    manifest_path = run_dir / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        problems = []
        if manifest.get("format") != PREDICTION_FORMAT:
            problems.append(
                f"its files are in format {manifest.get('format', 1)}, this code writes format {PREDICTION_FORMAT}"
            )
        if manifest.get("config_hash") != fingerprint:
            problems.append("the settings differ")
        changed = [
            row.plan_id
            for row in index.itertuples()
            if row.plan_id in manifest.get("plans", {})
            and manifest["plans"][row.plan_id] != _file_hash(plans_path / f"{row.plan_id}.parquet")
        ]
        if changed:
            problems.append(f"{len(changed)} plan(s) changed, e.g. {changed[:3]}")
        if problems and not restart:
            raise RunMismatch(
                f"The run folder {run_dir} was made from other inputs ({'; '.join(problems)}). Its files would "
                "be mixed with this run's. Use another run name, or restart to clear it."
            )
        if not problems:
            if manifest.get("git_commit") != a1.git_commit:
                logger.warning(
                    f"Resuming {run_dir}, started at commit {manifest.get('git_commit')}; the code is now at "
                    f"{a1.git_commit}. Files already written are kept as they are."
                )
            return manifest
        logger.warning(f"Clearing {run_dir}: {'; '.join(problems)}.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": PREDICTION_FORMAT,
        "config_hash": fingerprint,
        "settings": a1.path,
        "overrides": overrides or {},
        "git_commit": a1.git_commit,
        "git_dirty": a1.git_dirty,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "plans": {},
    }
    _write_json(manifest_path, manifest)
    return manifest


def _config_hash(a1: A1Config, config, overrides: Optional[dict] = None) -> str:
    """One hash of every file the outputs depend on: the A1 settings, the groups, the main config, the spec.

    Settings changed after reading (``overrides``, e.g. dev mode) are hashed too - but only when
    there are some, so a plain run's hash, and every run folder made before overrides existed,
    stays as it was.
    """
    paths = [a1.path, a1.groups.source_path, a1.settings.main_config_path, getattr(config, "data_spec_path", None)]
    digest = hashlib.sha256()
    for path in paths:
        if path and os.path.exists(path):
            digest.update(Path(path).read_bytes())
    if overrides:
        digest.update(json.dumps(overrides, sort_keys=True).encode())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    """The SHA-256 of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _output_name(model: str, group: str) -> str:
    """``<model>__<group>.parquet``."""
    return f"{model}__{group}.parquet"


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write under a temporary name, then rename: a half-written file never has the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """:func:`_atomic_write_parquet`, as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


# --- MLflow ---------------------------------------------------------------------------------------


def _start_parent(
    a1: A1Config,
    config,
    run_name: str,
    kinds,
    groups: bool,
    dev_mode: bool = False,
    *,
    manifest: Optional[dict] = None,
    run_dir: Optional[Path] = None,
):
    """Open the run folder's parent MLflow run, with its provenance - the same one on every resume.

    The parent's ids are kept in the folder's manifest (:mod:`yg_eo_soilnet.a1.tracking`): a resumed
    run reopens that parent, so a folder's child runs, evaluation and figures all sit under one run.
    A folder without one (new, cleared by ``restart``, or run without MLflow until now) gets a new
    parent. A recorded parent MLflow no longer has stops the run: its outputs would be split.
    """
    import mlflow

    from yg_eo_soilnet.a1.tracking import A1TrackingError, check_run, record_parent, recorded_run_id
    from yg_eo_soilnet.tracking import close_stale_runs, configure_tracking, log_params_once, run_owner_tags

    experiment = configure_tracking(config, a1.settings.mlflow_experiment)
    close_stale_runs(experiment)
    existing = recorded_run_id(manifest or {})
    if existing:
        try:
            check_run(existing, run_dir)
        except A1TrackingError as error:
            raise RunMismatch(str(error)) from error
        run = mlflow.start_run(run_id=existing)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run = mlflow.start_run(run_name=f"A1_{run_name}_{stamp}")
    if manifest is not None and run_dir is not None:
        record_parent(run_dir, manifest, run)
    roles = {model: (ROLE_DIAGNOSTIC if model in DIAGNOSTIC_MODELS else ROLE_MODEL) for model in (*MODELS, LOCAL)}
    mlflow.set_tags(
        {
            **run_owner_tags(),
            "a1_run_name": run_name,
            "model_role": json.dumps(roles),
            "a1_diagnostic_models": ",".join(DIAGNOSTIC_MODELS),
        }
    )
    log_params_once(
        {
            **a1.describe(),
            "a1_kinds": ",".join(kinds) if kinds else "all",
            "a1_groups": groups,
            "a1_dev_mode": dev_mode,
            "a1_logo_min_points": a1.settings.plans.logo.min_points,
            "a1_logo_held_out": ";".join(a1.settings.plans.logo.held_out or ()) or "all eligible",
            "a1_kfold_method": a1.settings.plans.spatial_kfold.method,
            "a1_kfold_block_sizes_m": ",".join(
                f"{size:g}" for size in a1.settings.plans.spatial_kfold.block_sizes_m() if size is not None
            )
            or "none",
            "a1_min_cell_size": a1.settings.min_cell_size,
        }
    )
    mlflow.log_artifact(a1.path)
    return run


def _log_plan(row, run_dir: Path) -> None:
    """One sub-run per plan, with every model's scores on the test points (and validation, if any)."""
    import mlflow

    from yg_eo_soilnet.tracking import start_child_run

    held_out = "" if pd.isna(row.held_out) else str(row.held_out)
    with start_child_run(row.plan_id, tags={"plan_id": row.plan_id, "kind": row.kind, "held_out": held_out}):
        metrics = {}
        for path in sorted((run_dir / "predictions" / row.plan_id).glob("*.parquet")):
            frame = pd.read_parquet(path)
            model, group = frame["model"].iloc[0], frame["covariate_group"].iloc[0]
            for (split,), part in frame.groupby(["split"]):
                scores = regression_metrics(part["observed"], part["predicted"], split=split)
                metrics.update({f"{name}__{model}__{group}": value for name, value in scores.items()})
        dissimilarity_path = run_dir / DISSIMILARITY_DIR / f"{row.plan_id}.parquet"
        if dissimilarity_path.exists():
            summary = region_summary(pd.read_parquet(dissimilarity_path)).iloc[0]
            for name in ("median_di", "share_outside", "carbonate_tvd"):
                if name in summary and pd.notna(summary[name]):
                    metrics[name] = float(summary[name])
        if metrics:
            mlflow.log_metrics(metrics)
