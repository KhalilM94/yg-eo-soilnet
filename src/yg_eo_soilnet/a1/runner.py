"""Run the A1 experiments over the plan index, resumably, writing one table per plan and model.

For each plan of ``index.csv`` (written by ``experiments/a1/make_plans.py``), and for each seed of
the settings (``seeds``; the plans themselves come from the first alone, so every seed repeats the
model fits on the same plans):

1. stage 1 predicts CaCO3 and CEC for every point (:func:`~yg_eo_soilnet.a1.auxiliary.predict_auxiliary`);
   its audit table is saved as ``stage1/<plan_id>/seed_<seed>.parquet``. A run whose SOM models
   (``stage2.models``) never read those predictions - ``global`` alone - skips it, and only cuts
   the folds stage 2 tunes through;
2. each stage-2 model (:mod:`yg_eo_soilnet.a1.stage2`) is fitted and its test predictions (and
   validation predictions, when the plans have a validation set; by default they do not) saved as
   ``predictions/<plan_id>/seed_<seed>/<model>__<covariate group>.parquet`` (:mod:`yg_eo_soilnet.a1.layout`);
3. for ``logo`` plans, a within-region ``local`` model on every covariate is scored inside the
   held-out region - the reference of the transfer loss; with ``groups``, ``global_aux`` and
   ``local`` are also fitted on each covariate group alone. Both need ``global_aux`` among the models.

Plans (and seeds) can run side by side in separate processes (``workers``); each process then
lets XGBoost use ``physical cores // workers`` threads (:func:`threads_per_worker`). Every file is still written whole or not at all,
and the predictions are the same as one process at a time gives.

Everything lands in one run folder, ``<output_dir>/runs/<run name>/``. A file that already exists is
not computed again, so an interrupted run carries on where it stopped. Files are written under a
temporary name and renamed when complete, so a run killed mid-write never leaves a half-file that
looks finished. The folder's ``manifest.json`` records what the outputs were made from - a hash of
the settings (but for the list of seeds, so a seed can be added to a run) and of every plan, the
seeds done per plan and the processor of each stage - and a run whose settings or plans differ
refuses to add to it, since its files would silently mix two experiments; ``restart`` clears it
instead. The same goes for a folder written in an older file format (:data:`PREDICTION_FORMAT`).

Every row of a predictions file carries what the evaluation needs without the data or the settings
(:mod:`yg_eo_soilnet.a1.evaluation`): the plan's held-out region, share and repeat, and the point's
position, from which the spatial blocks of the bootstrap are drawn.

Each ``logo`` plan also gets, from its first seed, ``dissimilarity/<plan_id>.parquet``: how far each point lies from the
plan's training points, in the inputs of ``global_aux`` weighted by its feature importances
(:mod:`yg_eo_soilnet.a1.dissimilarity`), and the run ends with ``dissimilarity_summary.csv``, one
row per held-out region.

Each plan and seed that computed something gets an MLflow sub-run with its scores and devices; the run's parent records
the git commit and the settings, and at the end a ``metrics.csv`` of every plan, model, group and
split - recomputed from all the prediction files, resumed ones included.
"""

from __future__ import annotations

import datetime
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

import yaml

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1.arms import INDICATOR_COLUMN, adds_indicator, indicator, weights_for
from yg_eo_soilnet.a1.auxiliary import AuxiliaryPredictions, folds_only, predict_auxiliary
from yg_eo_soilnet.a1.config import AUXILIARY_GROUP, COMBINED_GROUP, A1Config
from yg_eo_soilnet.a1.dissimilarity import importance_weights, plan_dissimilarity, region_summary
from yg_eo_soilnet.a1.logs import format_duration
from yg_eo_soilnet.a1.features import build_table
from yg_eo_soilnet.a1.layout import prediction_dir, prediction_paths, stage1_path
from yg_eo_soilnet.a1.plans import CONTROL, INDEX_COLUMNS, INDEX_FILE, LOGO, MIXED, RANDOM, SPATIAL_KFOLD
from yg_eo_soilnet.a1.settings import dev_mode_settings
from yg_eo_soilnet.a1.stage2 import (
    DIAGNOSTIC_MODELS,
    ROLE_DIAGNOSTIC,
    ROLE_MODEL,
    GLOBAL_AUX,
    GLOBAL_Q,
    LOCAL,
    GLOBAL,
    MODELS,
    ModelResult,
    build_context,
    needs_stage1,
    run_group,
    local_folds,
    run_local,
    run_model,
)
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, SplitPlan
from yg_eo_soilnet.metrics import regression_metrics

#: The columns of every predictions file, in order.
PREDICTION_COLUMNS = (
    "plan_id",
    "kind",
    "held_out",
    "fraction",
    "repeat",
    "block_size_m",
    "seed",
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
    "device_stage1",
    "device_stage2",
    "local_weight",
)
#: The columns format 5 added; a format-4 file (one seed, devices not recorded) is read without them.
SEED_DEVICE_COLUMNS = ("seed", "device_stage1", "device_stage2")
#: The version of that layout, recorded in the manifest. A run folder written in another version is
#: refused like one made from other settings: its old files would sit beside new ones.
PREDICTION_FORMAT = 5
MANIFEST_FILE = "manifest.json"
#: The run's log, appended to by every run_a1.py invocation on the folder; kept by ``restart``.
LOG_FILE = "run_a1.log"
METRICS_FILE = "metrics.csv"
DISSIMILARITY_DIR = "dissimilarity"
#: The dissimilarity files of the random plan and the 10 km folds, written for figure U3 when
#: ``stage2.models`` lists ``global_q``; never part of ``dissimilarity_summary.csv``.
DESIGN_DISSIMILARITY_DIR = "dissimilarity_designs"
#: ``global_q``'s conformal corrections, one file per plan and seed.
CALIBRATION_DIR = "conformal"
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
    seeds: Optional[Iterable[int]] = None,
    only: Optional[Iterable[str]] = None,
    workers: int = 1,
    device: Optional[str] = None,
    log_file: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> RunResult:
    """Run every plan of the index (or of the kinds asked for) for every seed, skipping what is already done.

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
        Also fit ``global_aux`` and (for ``logo`` plans) ``local`` on each covariate group alone.
        The ``local`` model on every covariate is fitted for ``logo`` plans either way (when
        ``global_aux`` is among ``stage2.models``).
    restart : bool, default False
        Clear a run folder made from other settings or plans, instead of refusing to add to it.
    tracking : bool, default True
        Record the run in MLflow.
    dev_mode : bool, default False
        Cut the models down to the quickest run that still goes through every step
        (:func:`~yg_eo_soilnet.a1.settings.dev_mode_settings`): it shows the pipeline works, and its
        scores mean nothing. Recorded in the run folder's settings hash, so a dev-mode run never
        resumes into a full one.
    seeds : iterable of int, optional
        Only these of the settings' seeds; every one unless given.
    only : iterable of str, optional
        Only the plans whose ``plan_id`` matches one of these shell patterns (``kfold_10km_*``),
        on top of ``kinds``.
    workers : int, default 1
        Plans (and seeds) fitted side by side, each in its own process, with
        :func:`threads_per_worker` threads for each estimator.
    device : str, optional
        ``cpu`` or ``cuda``: the processor of both stages, replacing the settings'. Recorded in the
        settings hash.
    log_file : pathlib.Path, optional
        The run's log (``run_a1.py`` writes ``run_a1.log`` in the run folder); added to the MLflow
        parent run at the end, failed or not.
    logger : logging.Logger, optional
        Where the progress lines go: one when each plan starts and ends (with an estimate of the
        time left), one for stage 1 and one for each model and covariate group. With several
        workers, one line per finished plan and seed.

    Returns
    -------
    RunResult

    Raises
    ------
    FileNotFoundError
        If there is no plan index.
    ValueError
        If ``seeds`` names a seed the settings do not list, or ``only`` matches no plan.
    RunMismatch
        If the run folder was made from other settings or plans and ``restart`` is not set.
    DeviceUnavailable
        If a stage asks for the graphics card and XGBoost cannot use it
        (:func:`~yg_eo_soilnet.a1.devices.check_cuda`).
    """
    from yg_eo_soilnet.a1.devices import check_cuda, check_cuda_quantiles, is_cuda
    from yg_eo_soilnet.a1.pipelines import stage_device

    logger = logger or logging.getLogger(__name__)
    overrides: dict = {}
    if dev_mode:
        a1 = replace(a1, settings=dev_mode_settings(a1.settings))
        overrides["dev_mode"] = True
        logger.warning(
            "Dev mode: XGBoost in both stages, 50 trees, no tuning, 2 inner folds. This run shows the pipeline "
            "works; its scores mean nothing."
        )
    if device is not None:
        overrides["device"] = device
    workers = max(1, int(workers))
    a1 = replace(a1, settings=replace(a1.settings, device=device, n_jobs=threads_per_worker(workers)))
    settings = a1.settings
    config = settings.load_config()
    run_seeds = _seeds_to_run(settings, seeds)
    devices = {
        "stage1": stage_device(settings, config, settings.stage1) if needs_stage1(settings.som_models) else "none",
        "stage2": stage_device(settings, config, settings.stage2),
    }
    for stage, name in devices.items():
        if is_cuda(name):
            check_cuda(name)
            logger.info(f"{stage} fits on {name}: XGBoost fitted a test model there.")
    if GLOBAL_Q in settings.som_models:
        # The quantile objective is checked on its own; if it cannot run on the card, global_q
        # alone moves to the processor, and every record says so.
        devices[GLOBAL_Q] = devices["stage2"]
        if is_cuda(devices["stage2"]):
            problem = check_cuda_quantiles(devices["stage2"], settings.uncertainty.quantiles)
            if problem:
                devices[GLOBAL_Q] = "cpu"
                logger.warning(
                    f"{GLOBAL_Q} cannot fit its quantile objective on {devices['stage2']} ({problem}); it fits on "
                    "the processor, and its predictions and the manifest record cpu."
                )
            else:
                logger.info(f"{GLOBAL_Q} fits on {devices['stage2']}: XGBoost fitted a quantile test model there.")

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
    if only is not None:
        index = select_plans(index, only)
    run_dir = Path(settings.output_dir) / "runs" / run_name
    manifest = _check_manifest(
        a1, config, plans_path, index, run_dir, restart=restart, overrides=overrides or None, logger=logger
    )
    manifest["seeds"] = list(settings.seeds)
    manifest["devices"] = devices
    manifest["models"] = list(settings.som_models)
    manifest.setdefault("seeds_done", {})
    _write_json(run_dir / MANIFEST_FILE, manifest)

    table = build_table(settings, config, logger)
    resolved = a1.resolve_groups(table, logger=logger)
    result = RunResult(run_dir=run_dir)
    if groups and GLOBAL_AUX not in settings.som_models:
        result.notes.append(f"--groups fits {GLOBAL_AUX} and local per group; {GLOBAL_AUX} is not in stage2.models.")

    parent = (
        _start_parent(
            a1, config, run_name, kinds, groups, dev_mode, manifest=manifest, run_dir=run_dir, devices=devices
        )
        if tracking
        else None
    )
    jobs = [(row, seed) for row in index.itertuples(index=False) for seed in run_seeds]
    total = len(jobs)
    logger.info(
        f"A1 run {run_name}: {len(index)} plan(s) ({', '.join(f'{n} {k}' for k, n in index['kind'].value_counts().items())}) "
        f"x {len(run_seeds)} seed(s) {run_seeds}, models {', '.join(settings.som_models)}, stage 1 on "
        f"{devices['stage1']}, stage 2 on {devices['stage2']}, {workers} worker(s) of {settings.n_jobs} thread(s), "
        f"in {run_dir}"
    )
    shared = _Shared(
        a1=a1, plans_path=plans_path, run_dir=run_dir, groups=groups, devices=devices, table=table, resolved=resolved
    )
    started_run = time.monotonic()
    durations: list[float] = []
    failed = False

    def finished(position: int, outcome: JobOutcome, row) -> None:
        """Book one finished plan and seed: the result, the manifest, the log line, MLflow."""
        result.computed += outcome.computed
        result.skipped += outcome.skipped
        result.notes += [note for note in outcome.notes if note not in result.notes]
        tag = f"[{position}/{total}] {outcome.plan_id} seed {outcome.seed}"
        if outcome.did_work:
            durations.append(outcome.elapsed)
            logger.info(
                f"{tag} done in {format_duration(outcome.elapsed)}; {outcome.summary}; "
                f"{_time_left(durations, total - position, workers)}"
            )
        else:
            logger.info(f"{tag} already done")
        done = set(manifest["seeds_done"].get(outcome.plan_id, []))
        manifest["seeds_done"][outcome.plan_id] = sorted(done | {outcome.seed})
        manifest["plans"][outcome.plan_id] = _file_hash(plans_path / f"{outcome.plan_id}.parquet")
        _write_json(run_dir / MANIFEST_FILE, manifest)
        if tracking and outcome.did_work:
            _log_plan(row, outcome.seed, run_dir, devices, first=outcome.seed == settings.seed)

    try:
        if workers == 1:
            for position, (row, seed) in enumerate(jobs, 1):
                logger.info(f"[{position}/{total}] {row.plan_id} seed {seed} ({row.kind}): {_counts(row)}")
                finished(position, _run_one_plan(row, seed, shared, logger=logger), row)
        else:
            _run_in_workers(jobs, shared, workers, log_file, finished)
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
    except BaseException:
        failed = True
        logger.exception(f"A1 run {run_name} failed after {format_duration(time.monotonic() - started_run)}.")
        raise
    finally:
        if parent is not None:
            import mlflow

            if log_file is not None and Path(log_file).exists():
                try:
                    mlflow.log_artifact(str(log_file))
                except Exception as error:  # the run's own error, if any, must not be hidden by this one
                    logger.warning(f"Could not add {log_file} to the MLflow run: {error}")
            mlflow.end_run(status="FAILED" if failed else "FINISHED")
    logger.info(
        f"A1 run {run_name}: {len(result.computed)} file(s) written, {len(result.skipped)} already there, "
        f"in {format_duration(time.monotonic() - started_run)}; {run_dir}"
    )
    return result


def physical_cores() -> int:
    """The processor's physical cores; the logical ones (``os.cpu_count``) when they cannot be told apart."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
    except ImportError:
        cores = None
    return int(cores or os.cpu_count() or 1)


def threads_per_worker(workers: int) -> int:
    """The threads each estimator gets with ``workers`` processes: ``max(1, physical cores // workers)``.

    Physical cores, not logical ones: XGBoost's tree building gains nothing from a core's second
    hardware thread, and slows down when it runs on them (one fit on 4,644 points of the real data:
    7.8 s with 12 threads on a 6-core processor, 4.0 s with 6). The thread count changes no prediction.
    """
    return max(1, physical_cores() // max(1, int(workers)))


def _seeds_to_run(settings, seeds: Optional[Iterable[int]]) -> list[int]:
    """The seeds asked for, in the settings' order; every seed of the settings by default."""
    if seeds is None:
        return list(settings.seeds)
    asked = [int(seed) for seed in seeds]
    unknown = [seed for seed in asked if seed not in settings.seeds]
    if unknown:
        raise ValueError(f"--seeds {unknown} are not among the settings' seeds {list(settings.seeds)}.")
    return [seed for seed in settings.seeds if seed in asked]


def select_plans(index: pd.DataFrame, patterns: Iterable[str]) -> pd.DataFrame:
    """The rows of a plan index whose ``plan_id`` matches one of these shell patterns.

    Examples
    --------
    >>> index = pd.DataFrame({"plan_id": ["random", "kfold_10km_f0", "kfold_25km_f0", "logo_r01_sebou"]})
    >>> select_plans(index, ["random", "kfold_10km_*", "logo_*"])["plan_id"].tolist()
    ['random', 'kfold_10km_f0', 'logo_r01_sebou']

    Raises
    ------
    ValueError
        If no plan matches.
    """
    patterns = [pattern.strip() for pattern in patterns if pattern and pattern.strip()]
    keep = index["plan_id"].map(lambda plan: any(fnmatch.fnmatchcase(str(plan), pattern) for pattern in patterns))
    if not keep.any():
        raise ValueError(
            f"--only {','.join(patterns)} matches none of the {len(index)} plan(s), e.g. {index['plan_id'].head(3).tolist()}."
        )
    return index[keep.to_numpy()]


# --- one plan --------------------------------------------------------------------------------


@dataclass
class _Shared:
    """What every plan of a run reads; built once, and once per worker process."""

    a1: A1Config
    plans_path: Path
    run_dir: Path
    groups: bool
    devices: dict
    table: object
    resolved: object
    config: object = None

    def load(self):
        """The main configuration, read once per process (it does not travel between processes)."""
        if self.config is None:
            self.config = self.a1.settings.load_config()
        return self.config

    def __getstate__(self):
        state = dict(self.__dict__)
        state["config"] = None
        return state


@dataclass
class JobOutcome:
    """What one plan and seed did; returned by a worker process to the one running the run."""

    plan_id: str
    seed: int
    computed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    elapsed: float = 0.0
    did_work: bool = False
    summary: str = ""


#: A worker process's :class:`_Shared`, set by :func:`_start_worker`.
_WORKER: dict = {}


def _start_worker(shared: _Shared, log_file: Optional[str]) -> None:
    """Set up a worker process: its share of the run, and its log lines (warnings on the console)."""
    import sys

    from yg_eo_soilnet.a1.logs import LOG_FORMAT, NOISY_LOGGERS

    handlers: list = []
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    handlers.append(console)
    if log_file:
        to_file = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        to_file.setLevel(logging.INFO)
        handlers.append(to_file)
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter(LOG_FORMAT.replace("%(message)s", f"[worker {os.getpid()}] %(message)s"))
        )
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    _WORKER["shared"] = shared
    # The numerical libraries' own thread pools (BLAS, OpenMP), held to this worker's share of the
    # processor: otherwise every worker starts one thread per core and they slow each other down.
    from threadpoolctl import threadpool_limits

    _WORKER["limits"] = threadpool_limits(limits=shared.a1.settings.n_jobs)


def _worker_job(row: dict, seed: int) -> JobOutcome:
    """One plan and seed, in a worker process."""
    from types import SimpleNamespace

    return _run_one_plan(SimpleNamespace(**row), seed, _WORKER["shared"], logger=logging.getLogger(__name__))


def _run_in_workers(jobs, shared: _Shared, workers: int, log_file, finished) -> None:
    """Run the plan-and-seed jobs in ``workers`` processes; ``finished`` books each as it ends."""
    import multiprocessing

    # spawn, not fork: XGBoost's and the BLAS libraries' threads do not survive a fork. The thread
    # counts are set in the environment the workers start with, before they load those libraries.
    context = multiprocessing.get_context("spawn")
    threads = str(shared.a1.settings.n_jobs)
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update({name: threads for name in THREAD_VARIABLES})
    try:
        _run_pool(jobs, shared, workers, log_file, finished, context)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


#: The environment variables that size the numerical libraries' thread pools.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run_pool(jobs, shared: _Shared, workers: int, log_file, finished, context) -> None:
    """The process pool of :func:`_run_in_workers`."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_start_worker,
        initargs=(shared, str(log_file) if log_file else None),
    ) as pool:
        futures = {pool.submit(_worker_job, row._asdict(), seed): row for row, seed in jobs}
        try:
            for position, future in enumerate(as_completed(futures), 1):
                finished(position, future.result(), futures[future])
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _run_one_plan(row, seed: int, shared: _Shared, *, logger) -> JobOutcome:
    """Stage 1 and every stage-2 output of one plan and seed."""
    started_plan = time.monotonic()
    settings = shared.a1.settings.for_seed(seed)
    config = shared.load()
    table, resolved, run_dir = shared.table, shared.resolved, shared.run_dir
    outcome = JobOutcome(plan_id=row.plan_id, seed=int(seed))
    first = int(seed) == settings.seed
    plan = SplitPlan.from_frame(pd.read_parquet(shared.plans_path / f"{row.plan_id}.parquet"), strategy=row.kind)
    out = prediction_dir(run_dir, row.plan_id, seed)
    models = settings.som_models
    with_aux = GLOBAL_AUX in models

    jobs = [(model, COMBINED_GROUP) for model in models]
    if row.kind == LOGO and with_aux:
        # The reference of the transfer loss, on every covariate: always, not only with --groups.
        jobs.append((LOCAL, COMBINED_GROUP))
    if shared.groups and with_aux:
        jobs += [(GLOBAL_AUX, name) for name in resolved.groups]
        if row.kind == LOGO:
            jobs += [(LOCAL, name) for name in resolved.groups]
    todo = [(model, group) for model, group in jobs if not (out / _output_name(model, group)).exists()]
    outcome.skipped += [str(out / _output_name(model, group)) for model, group in jobs if (model, group) not in todo]
    for model, group in jobs:
        if (model, group) not in todo:
            logger.info(f"  {model} / {group}: already done, {_rmse_note(out / _output_name(model, group))}")
    dissimilarity_path = dissimilarity_file(run_dir, row, settings)
    # From the first seed only: how far a region lies from its training points does not depend on the seed.
    needs_dissimilarity = first and dissimilarity_path is not None and not dissimilarity_path.exists()
    if first and dissimilarity_path is not None and not needs_dissimilarity:
        outcome.skipped.append(str(dissimilarity_path))
    if not todo and not needs_dissimilarity:
        return outcome

    # The weighted arm: its moved points weigh w in every fit, stage 1's included.
    fitted_ids = plan.point_ids_for(TRAIN)
    fitted_ids = fitted_ids[table.frame[settings.target].reindex(fitted_ids).notna().to_numpy()]
    weights, w = weights_for(row.kind, plan, settings, fitted_ids)
    if weights is not None:
        logger.info(f"  weighted arm: each of {int((weights != 1.0).sum())} moved point(s) weighs {w:.3g}")
    if needs_stage1(models):
        auxiliary = _stage1(row, seed, plan, table, settings, config, run_dir, logger, sample_weight=weights)
    else:
        auxiliary = folds_only(plan, settings, kind=row.kind, logger=logger)
        logger.info(f"  stage 1: skipped, no model of {list(models)} reads it; {auxiliary.n_folds} fold(s) cut")
    context = build_context(table, plan, auxiliary, settings, config)
    context.sample_weight = weights
    context.quantile_device = shared.devices.get(GLOBAL_Q)
    if (GLOBAL_Q, COMBINED_GROUP) in todo and (GLOBAL, COMBINED_GROUP) not in todo:
        # A resumed run: Global is on disk, and global_q reuses the settings its search chose.
        saved = pd.read_parquet(out / _output_name(GLOBAL, COMBINED_GROUP), columns=["tuned_params"])
        context.global_settings = json.loads(saved["tuned_params"].iloc[0])
    auxiliary_columns = list(resolved.groups.get(AUXILIARY_GROUP, ())) if needs_stage1(models) else []
    global_columns = [column for column in resolved.combined if column not in resolved.groups.get(AUXILIARY_GROUP, ())]
    if adds_indicator(row.kind):
        # The indicator arm: one more input, read only by the models on every covariate.
        context.X[INDICATOR_COLUMN] = indicator(plan).reindex(context.X.index)
        global_columns.append(INDICATOR_COLUMN)
    region = table.frame[table.region_column] if table.region_column else None
    scores: dict = {}

    for model, group in todo:
        started = time.monotonic()
        if model == LOCAL:
            columns = list(resolved.combined) if group == COMBINED_GROUP else list(resolved.groups[group])
            local = local_folds(context, plan.point_ids_for(TEST), table, logger=logger)
            result = run_local(context, columns, local)
            if result is None:
                note = f"{row.plan_id}: no local score - {local.reason}"
                if note not in outcome.notes:
                    outcome.notes.append(note)
                logger.info(f"  {model} / {group}: skipped - {local.reason}")
                continue
        elif group == COMBINED_GROUP:
            result = run_model(
                context, model, global_columns=global_columns, auxiliary_columns=auxiliary_columns, logger=logger
            )
        else:
            result = run_group(context, list(resolved.groups[group]))
        devices = shared.devices
        if model == GLOBAL_Q:
            devices = {**devices, "stage2": devices.get(GLOBAL_Q, devices["stage2"])}
        frame = _rows(row, seed, context, result, model, group, region, table, devices, w)
        path = out / _output_name(model, group)
        if result.calibration is not None:
            _write_calibration(run_dir, row, seed, result.calibration, logger)
        _atomic_write_parquet(frame, path)
        outcome.computed.append(str(path))
        note = _rmse_note(frame)
        if group == COMBINED_GROUP:
            scores[model] = note
        logger.info(f"  {model} / {group}: {format_duration(time.monotonic() - started)}, {note}")

    if needs_dissimilarity:
        frame = _dissimilarity(row, context, table, global_columns, auxiliary_columns, outcome, logger)
        _atomic_write_parquet(frame, dissimilarity_path)
        outcome.computed.append(str(dissimilarity_path))
    outcome.did_work = True
    outcome.elapsed = time.monotonic() - started_plan
    shown = scores.get(GLOBAL) or next(iter(scores.values()), "")
    outcome.summary = f"{GLOBAL if GLOBAL in scores else next(iter(scores), '')} {shown}".strip() or "dissimilarity"
    return outcome


def dissimilarity_file(run_dir: Path, row, settings) -> Optional[Path]:
    """Where a plan's dissimilarity file goes, or None for a plan that gets none.

    Every ``logo`` plan has one (``dissimilarity/``), read by the transfer figures and
    ``dissimilarity_summary.csv``. With ``global_q`` in ``stage2.models``, the random plan and the
    spatial folds of ``plans.spatial_kfold.cell_size_m`` (10 km) get one too, in
    ``dissimilarity_designs/``, so figure U3 can relate each test point's interval width to how far
    it lies from its training points; kept apart, they never enter the regional summary.
    """
    if row.kind == LOGO:
        return Path(run_dir) / DISSIMILARITY_DIR / f"{row.plan_id}.parquet"
    if GLOBAL_Q not in settings.som_models:
        return None
    cell = settings.plans.spatial_kfold.cell_size_m
    paper_folds = row.kind == SPATIAL_KFOLD and pd.notna(row.block_size_m) and float(row.block_size_m) == float(cell)
    if row.kind == RANDOM or paper_folds:
        return Path(run_dir) / DESIGN_DISSIMILARITY_DIR / f"{row.plan_id}.parquet"
    return None


def calibration_path(run_dir: Path, plan_id: str, seed: int) -> Path:
    """``conformal/<plan>/seed_<s>.json``: the conformal correction of one plan and seed (``global_q``)."""
    return Path(run_dir) / CALIBRATION_DIR / plan_id / f"seed_{int(seed)}.json"


def _write_calibration(run_dir: Path, row, seed: int, calibration: dict, logger) -> None:
    """Save ``global_q``'s conformal correction (Q per level) of one plan and seed, and log it."""
    path = calibration_path(run_dir, row.plan_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"plan_id": row.plan_id, "kind": row.kind, "seed": int(seed), **calibration}
    _write_json(path, record)
    levels = record.get("levels_q") or {}
    shown = ", ".join(f"{tag} % Q {entry['q']:+.2f}" for tag, entry in levels.items()) or record.get("note", "")
    logger.info(
        f"  {GLOBAL_Q}: {shown}; quantiles crossed at {100 * record['crossing_share_held_out']:.1f} % of held-out points"
    )


def _dissimilarity(row, context, table, global_columns, auxiliary_columns, result, logger) -> pd.DataFrame:
    """How far each point of a plan lies from its training points, in ``global_aux``'s inputs.

    The inputs are weighted by ``global_aux``'s feature importances; that model is fitted here if
    its predictions were already on disk. A run without stage 1 (``global`` alone) uses ``global``
    and its inputs instead.
    """
    settings = context.settings
    weighting = GLOBAL_AUX if auxiliary_columns else GLOBAL
    outcome = run_model(
        context, weighting, global_columns=global_columns, auxiliary_columns=auxiliary_columns, logger=logger
    )
    columns = list(global_columns) + [column for column in auxiliary_columns if column not in global_columns]
    weights = importance_weights(outcome.fitted, columns) if outcome.fitted is not None else None
    if weights is None:
        result.notes.append(
            f"{row.plan_id}: {weighting} has no feature importances; the DI weighs every input the same"
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


def _stage1(row, seed, plan, table, settings, config, run_dir, logger, *, sample_weight=None) -> AuxiliaryPredictions:
    """The plan's stage-1 predictions for one seed: read back when saved, otherwise computed and saved."""
    path = stage1_path(run_dir, row.plan_id, seed)
    if path.exists():
        auxiliary = _auxiliary_from_frame(pd.read_parquet(path), settings)
        logger.info(
            f"  stage 1: read from stage1/{row.plan_id}/{path.name}; {_out_of_fold_r2(auxiliary, table, settings)}"
        )
        return auxiliary
    started = time.monotonic()
    auxiliary = predict_auxiliary(
        table, plan, settings, config, kind=row.kind, sample_weight=sample_weight, logger=logger
    )
    # The point ids are the frame's index; written as a column, so they survive the file.
    _atomic_write_parquet(auxiliary.to_frame().reset_index().assign(seed=int(seed)), path)
    logger.info(
        f"  stage 1: {format_duration(time.monotonic() - started)}; {_out_of_fold_r2(auxiliary, table, settings)}"
    )
    return auxiliary


def _counts(row) -> str:
    """``4,644 train / 1,162 test`` (and validation, when the plan has any) from an index row."""
    text = f"{int(row.n_train):,} train / {int(row.n_test):,} test"
    return text + (f" / {int(row.n_val):,} val" if int(row.n_val) else "")


def _time_left(durations: list[float], plans_left: int, workers: int = 1) -> str:
    """An estimate of the time left: the mean time of the plans computed so far, times those left, over the workers."""
    mean = sum(durations) / len(durations)
    if not plans_left:
        return "that was the last plan"
    return (
        f"about {format_duration(mean * plans_left / max(1, workers))} left ({format_duration(mean)} per plan over "
        f"{len(durations)} computed)"
    )


def _rmse_note(predictions) -> str:
    """``test RMSE 8.10`` of a predictions table or file; what is missing when it has no test score."""
    frame = pd.read_parquet(predictions) if isinstance(predictions, (str, Path)) else predictions
    test = frame[frame["split"] == TEST]
    scores = regression_metrics(test["observed"], test["predicted"], split="test")
    return f"test RMSE {scores['rmse_test']:.2f}" if "rmse_test" in scores else "no test score"


def _out_of_fold_r2(auxiliary: AuxiliaryPredictions, table, settings) -> str:
    """``out-of-fold R2 caco3_pct_total 0.58, c_e_c_meq_100g 0.62`` on the plan's training points."""
    from yg_eo_soilnet.a1.config import stage1_prediction_column

    training = auxiliary.folds.index[(auxiliary.folds >= 0).to_numpy()]
    parts = []
    for target in settings.auxiliary.targets:
        scores = regression_metrics(
            table.frame[target].reindex(training),
            auxiliary.predictions[stage1_prediction_column(target)].reindex(training),
        )
        parts.append(f"{target} {scores['r2_test']:.2f}" if "r2_test" in scores else f"{target} -")
    return "out-of-fold R2 " + ", ".join(parts)


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


def _rows(
    row, seed, context, outcome: ModelResult, model, group, region, table, devices, local_weight=np.nan
) -> pd.DataFrame:
    """The predictions table of one model on one plan."""
    ids = outcome.predicted.index
    coordinates = table.coordinates.reindex(ids)
    frame = pd.DataFrame(
        {
            "plan_id": row.plan_id,
            "kind": row.kind,
            "held_out": None if pd.isna(row.held_out) else str(row.held_out),
            # Whole float columns even when empty: a column of bare NaN would be read back as object.
            "fraction": np.full(len(ids), float(row.fraction) if pd.notna(row.fraction) else np.nan),
            "repeat": np.full(len(ids), float(row.repeat) if pd.notna(row.repeat) else np.nan),
            "block_size_m": np.full(len(ids), float(row.block_size_m) if pd.notna(row.block_size_m) else np.nan),
            "seed": np.full(len(ids), int(seed), dtype="int64"),
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
            "device_stage1": devices["stage1"],
            "device_stage2": devices["stage2"],
            # The weight of each moved training point (mixed_weighted); empty for every other kind.
            "local_weight": np.full(len(ids), float(local_weight)),
        },
        columns=list(PREDICTION_COLUMNS),
    )
    if outcome.extra is not None:
        # global_q's quantiles and intervals, after the usual columns; no other model has them.
        frame = pd.concat([frame, outcome.extra.reindex(ids).reset_index(drop=True)], axis=1)
    return frame


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
        ``plan_id, seed, kind, model, covariate_group, model_role, split, n, rmse, mae, bias, r2, rpd,
        rpiq, n_fallback``.
    """
    rows = []
    for path in prediction_paths(run_dir):
        frame = pd.read_parquet(path)
        for (split,), part in frame.groupby(["split"]):
            scores = regression_metrics(part["observed"], part["predicted"], split="x")
            first = part.iloc[0]
            rows.append(
                {
                    "plan_id": first["plan_id"],
                    "seed": int(first["seed"]) if "seed" in part else np.nan,
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
        _clear_run_folder(run_dir)
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


def _clear_run_folder(run_dir: Path) -> None:
    """Delete everything in the run folder but its log, which the running process is writing to."""
    for entry in run_dir.iterdir():
        if entry.name == LOG_FILE:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _config_hash(a1: A1Config, config, overrides: Optional[dict] = None) -> str:
    """One hash of every file the outputs depend on: the A1 settings, the groups, the main config, the spec.

    The A1 settings are hashed as read, without their ``seeds``: a seed added to the list adds
    outputs beside the others' and changes none of them, so the run folder takes it. (Comments
    and layout of that file do not count either.) Settings changed after reading (``overrides``,
    e.g. dev mode or ``--device``) are hashed too, when there are some.
    """
    paths = [a1.groups.source_path, a1.settings.main_config_path, getattr(config, "data_spec_path", None)]
    digest = hashlib.sha256()
    if a1.path and os.path.exists(a1.path):
        settings = dict(yaml.safe_load(Path(a1.path).read_text()) or {})
        settings.pop("seeds", None)
        digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
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
    devices: Optional[dict] = None,
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
            "a1_models": ",".join(a1.settings.som_models),
            **{f"a1_device_{stage}": name for stage, name in (devices or {}).items()},
        }
    )
    mlflow.log_artifact(a1.path)
    return run


def _log_plan(row, seed: int, run_dir: Path, devices: dict, *, first: bool = True) -> None:
    """One sub-run per plan and seed, with every model's scores on the test points (and validation, if any)."""
    import mlflow

    from yg_eo_soilnet.tracking import start_child_run

    held_out = "" if pd.isna(row.held_out) else str(row.held_out)
    tags = {"plan_id": row.plan_id, "kind": row.kind, "held_out": held_out, "seed": str(seed)}
    with start_child_run(f"{row.plan_id}__seed_{seed}", tags=tags):
        mlflow.log_params({"seed": int(seed), **{f"device_{stage}": name for stage, name in devices.items()}})
        metrics = {}
        for path in sorted(prediction_dir(run_dir, row.plan_id, seed).glob("*.parquet")):
            frame = pd.read_parquet(path)
            model, group = frame["model"].iloc[0], frame["covariate_group"].iloc[0]
            for (split,), part in frame.groupby(["split"]):
                scores = regression_metrics(part["observed"], part["predicted"], split=split)
                metrics.update({f"{name}__{model}__{group}": value for name, value in scores.items()})
        dissimilarity_path = run_dir / DISSIMILARITY_DIR / f"{row.plan_id}.parquet"
        if dissimilarity_path.exists() and first:
            summary = region_summary(pd.read_parquet(dissimilarity_path)).iloc[0]
            for name in ("median_di", "share_outside", "carbonate_tvd"):
                if name in summary and pd.notna(summary[name]):
                    metrics[name] = float(summary[name])
        calibration = calibration_path(run_dir, row.plan_id, seed)
        if calibration.exists():
            # global_q: the conformal correction Q of each level, and how often its quantiles crossed.
            record = json.loads(calibration.read_text())
            for tag, entry in (record.get("levels_q") or {}).items():
                metrics[f"cqr_q{tag}__{GLOBAL_Q}"] = float(entry["q"])
            metrics[f"crossing_share__{GLOBAL_Q}"] = float(record["crossing_share_held_out"])
            # Its settings are Global's, chosen by Global's search in this plan, seed and arm.
            mlflow.log_params(
                {
                    f"{GLOBAL_Q}_settings_from": record.get("settings_from", ""),
                    **{f"{GLOBAL_Q}__{name}": value for name, value in (record.get("settings") or {}).items()},
                }
            )
        if metrics:
            mlflow.log_metrics(metrics)
