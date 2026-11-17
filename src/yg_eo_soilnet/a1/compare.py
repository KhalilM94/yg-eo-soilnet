"""Compare A1 runs across run folders: the mixed arms, matched leave-one-basin-out, processor vs card.

Each experiment of round 3 writes its own run folder, so its comparisons read several. Plans are
only ever paired on the same test points, which are read from the plan files themselves (each
checked against the hash the run's manifest recorded), and runs fitted on different processors
are only paired when asked (``--allow-mixed-devices``): XGBoost on the graphics card grows slightly
different trees, so such a difference could be the processor's alone.

The pairing keys:

- the mixed kinds (``mixed``, ``mixed_weighted``, ``mixed_indicator``, ``control``): held-out
  basin, share and repeat - every plan of one key must test exactly the same points, and the
  basin's ``logo`` plan (the model with no local points) is scored on those points too;
- plain ``logo``: held-out basin;
- matched ``logo`` (``plans.logo.training_size``): held-out basin and repeat; every draw must test
  the same points as the basin's plain ``logo`` plan.

What is written (:func:`compare_runs`):

``mixed_arms.csv``
    Per run, basin, share and repeat: the RMSE of logo, control, mixed and each arm present on the
    shared test points, the local model's when a run has it, the moved and test point counts.
``mixed_arms_paired.csv``
    Per run, basin and share: each arm minus control and each arm minus mixed - the mean of the
    per-repeat RMSE differences, a 95 % interval from resampling 10 km blocks of the test points
    of every repeat together, its p-value, Holm-corrected across the shares of one basin and
    comparison, and the number of repeats in which the arm does better.
``logo_matched.csv`` and ``logo_matched_draws.csv``
    Per basin and training size: the unmatched plan's RMSE, R² and bias and its training size, the
    matched draws' mean and min-max, and the differences; every draw on its own beside it.
``transfer_status.csv``
    From the main run (``results/a1_xgb``, when it is among the runs): per basin, the transferred
    minus the local RMSE on every input with its block-bootstrap interval and Holm-corrected
    p-value across the basins; the basin's transfer *fails* when that difference is above 0 with an
    interval above 0 and p_holm < 0.05.
``cross_run.csv`` and ``cross_run_summary.csv``
    A plan run in two folders (the processor-against-card check): its RMSE in each and the
    difference, the points' mean absolute prediction difference and their correlation; summarised
    per kind beside the mean mixed - control difference, so the processor noise can be read against
    the effect.
``matched_paired.csv`` and ``matched_intervals.csv``
    The matched sizes, repeats stacked (one row per point and repeat) and paired on the same points:
    per basin, the largest size against the full leave-one-basin-out model (the main run's first
    seed), each size against the next larger, each size against the local model (``local_global``;
    ``local``, named in ``local_model``, when the main run lacks it) - RMSE, bias and R²
    differences with 10 km block-bootstrap intervals, Holm within each basin and score; and each
    size's (and the full and local models') scores with their intervals.
``arms_vs_transferred.csv``
    "Local data against transferred": each mixed kind and control, per basin and share, against the
    basin's leave-one-basin-out model on the same test points, repeats stacked, paired block
    bootstrap, Holm across the kinds and shares of one basin and score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.a1 import evaluation
from yg_eo_soilnet.a1.config import COMBINED_GROUP
from yg_eo_soilnet.a1.devices import UNKNOWN_DEVICE, check_same_devices
from yg_eo_soilnet.a1.plans import (
    CONTROL,
    DIVIDED_KINDS,
    INDEX_FILE,
    LOGO,
    MIXED,
    MIXED_INDICATOR,
    MIXED_WEIGHTED,
    is_matched,
    matched_training_size,
)
from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL, LOCAL_GLOBAL, LOCAL_OF
from yg_eo_soilnet.datamodules.splitting import TEST, TRAIN, SplitPlan

logger = logging.getLogger(__name__)

#: The output folder name of the main run, whose transfer losses classify the basins.
TRANSFER_RUN = "a1_xgb"
#: The mixed kinds in the order the tables and figures show them.
ARM_ORDER = (CONTROL, MIXED, MIXED_WEIGHTED, MIXED_INDICATOR)
#: The paired comparisons ``(a, b)`` of ``mixed_arms_paired.csv``: score(a) - score(b).
COMPARISONS = (
    (MIXED, CONTROL),
    (MIXED_WEIGHTED, CONTROL),
    (MIXED_INDICATOR, CONTROL),
    (MIXED_WEIGHTED, MIXED),
    (MIXED_INDICATOR, MIXED),
)
#: The significance level of the transfer classification and the intervals.
ALPHA = 0.05
DEVICE_COLUMNS = ("device_stage1", "device_stage2")


class CompareError(ValueError):
    """Runs or plans that must not be paired: other test points, or a plan file changed since the run."""


# --- reading the run folders ----------------------------------------------------------------------


@dataclass
class RunFolder:
    """One run folder, read for comparison.

    Attributes
    ----------
    name : str
        The output folder's name (``a1_mixed_arms``), with ``/<run name>`` unless ``default``.
    run_dir : pathlib.Path
        ``<output_dir>/runs/<run name>``.
    plans_dir : pathlib.Path
        ``<output_dir>/plans``, the plans the run read.
    manifest : dict
    devices : dict
        ``stage1`` / ``stage2``: where each stage was fitted (``unknown`` before devices were recorded).
    index : pandas.DataFrame
        The plans' ``index.csv``, indexed by plan id.
    predictions : pandas.DataFrame
        The test predictions of the first seed, every input (``combined``), of the model compared
        and the local model.
    """

    name: str
    run_dir: Path
    plans_dir: Path
    manifest: dict
    devices: dict
    index: pd.DataFrame
    predictions: pd.DataFrame
    _test_points: dict = field(default_factory=dict, repr=False)
    _moved: dict = field(default_factory=dict, repr=False)

    @property
    def output_name(self) -> str:
        """The output folder's name alone (``a1_xgb``)."""
        return self.name.split("/")[0]

    def plan(self, plan_id: str) -> SplitPlan:
        """The plan file, checked against the hash the manifest recorded when the run read it.

        Raises
        ------
        CompareError
            If the file is missing, or changed since the run.
        """
        path = self.plans_dir / f"{plan_id}.parquet"
        if not path.exists():
            raise CompareError(f"{self.name}: the plan file {path} is missing; it is needed to check the test points.")
        recorded = (self.manifest.get("plans") or {}).get(plan_id)
        if isinstance(recorded, str) and hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
            raise CompareError(
                f"{self.name}: {path} is not the plan the run read (its hash changed since); the predictions "
                "cannot be checked against it."
            )
        return SplitPlan.from_frame(pd.read_parquet(path))

    def test_points(self, plan_id: str) -> frozenset:
        """The plan's test points, from its file."""
        if plan_id not in self._test_points:
            plan = self.plan(plan_id)
            self._test_points[plan_id] = frozenset(plan.point_ids_for(TEST))
            region = (
                plan.clusters.reindex(plan.point_ids_for(TEST)).dropna().unique() if plan.clusters is not None else []
            )
            if len(region) == 1:
                in_region = plan.clusters.reindex(plan.point_ids_for(TRAIN)) == region[0]
                self._moved[plan_id] = int(in_region.sum())
            else:
                self._moved[plan_id] = 0
        return self._test_points[plan_id]

    def moved(self, plan_id: str) -> int:
        """The held-out basin's points in the plan's training set (the moved points of a mixed plan)."""
        self.test_points(plan_id)
        return self._moved[plan_id]

    def n_train(self, plan_id: str) -> float:
        """The plan's training size, from ``index.csv``."""
        return float(self.index.loc[plan_id, "n_train"]) if plan_id in self.index.index else np.nan


def read_run(
    run_dir, *, model: str = GLOBAL, kinds: Optional[Sequence[str]] = None, local: Optional[str] = None
) -> RunFolder:
    """Read one run folder: its manifest, plan index and the test predictions compared.

    The predictions of ``model``, of its local model (:func:`local_of`), of ``local`` (the fallback
    of a run without ``local_global``), and of ``local`` the argument when given.

    Raises
    ------
    CompareError
        If the folder has no manifest or no plan index.
    """
    from yg_eo_soilnet.a1.runner import MANIFEST_FILE

    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise CompareError(f"No {MANIFEST_FILE} in {run_dir}: is it a run folder (<output_dir>/runs/<run name>)?")
    manifest = json.loads(manifest_path.read_text())
    plans_dir = run_dir.parent.parent / "plans"
    if not (plans_dir / INDEX_FILE).exists():
        raise CompareError(f"No {plans_dir / INDEX_FILE}: the plans of {run_dir} are needed to check the test points.")
    index = pd.read_csv(plans_dir / INDEX_FILE).set_index("plan_id")
    recorded = manifest.get("devices") or {}
    devices = {stage: str(recorded.get(stage, UNKNOWN_DEVICE)) for stage in ("stage1", "stage2")}

    frame = evaluation.load_predictions(run_dir, first_seed=_first_seed(manifest))
    first = evaluation.seed_order(frame, _first_seed(manifest))[0] if not frame.empty else None
    keep = (
        (frame["seed"] == first)
        & (frame["split"] == TEST)
        & (frame["covariate_group"] == COMBINED_GROUP)
        & frame["model"].isin([model, local_of(model), LOCAL, *([local] if local else [])])
    )
    if kinds:
        keep &= frame["kind"].isin(list(kinds))
    columns = ["plan_id", "kind", "held_out", "fraction", "repeat", "point_id", "split", "lat", "lon", "model"]
    predictions = frame.loc[keep, [*columns, "observed", "predicted"]].reset_index(drop=True)
    predictions["fraction"] = predictions["fraction"].astype(float).round(6)
    name = run_dir.parent.parent.name + ("" if run_dir.name == "default" else f"/{run_dir.name}")
    logger.info(f"{name}: {predictions['plan_id'].nunique()} plan(s), devices {devices}")
    return RunFolder(name, run_dir, plans_dir, manifest, devices, index, predictions)


def _first_seed(manifest: dict) -> Optional[int]:
    """The run's first seed, from the settings file the manifest names; None if it cannot be read."""
    from yg_eo_soilnet.a1.settings import load_settings

    try:
        return load_settings(manifest["settings"]).seed
    except Exception:  # an older or moved settings file: the first seed in the predictions is used
        return None


def _rmse(part: pd.DataFrame) -> float:
    """RMSE of the rows' predictions."""
    if part.empty:
        return np.nan
    return float(np.sqrt(np.mean((part["predicted"].to_numpy(float) - part["observed"].to_numpy(float)) ** 2)))


def _scores(part: pd.DataFrame) -> dict:
    """RMSE, R² and bias of the rows' predictions."""
    if part.empty:
        return {"rmse": np.nan, "r2": np.nan, "bias": np.nan}
    found = evaluation._point_scores(part["observed"].to_numpy(float), part["predicted"].to_numpy(float))
    return {"rmse": found["rmse"], "r2": found["r2"], "bias": found["bias"]}


def _devices(run: RunFolder, prefix: str = "") -> dict:
    return {f"{prefix}{column}": run.devices[column.split("_")[1]] for column in DEVICE_COLUMNS}


# --- the mixed experiment and its arms ----------------------------------------------------------


def _mixed_plans(run: RunFolder) -> pd.DataFrame:
    """One row per mixed-kind plan of the run: ``plan_id, kind, basin, fraction, repeat``."""
    frame = run.predictions[run.predictions["kind"].isin(DIVIDED_KINDS)]
    plans = frame.drop_duplicates("plan_id")[["plan_id", "kind", "held_out", "fraction", "repeat"]]
    return plans.rename(columns={"held_out": "basin"}).reset_index(drop=True)


def _plain_logo(run: RunFolder) -> dict:
    """The run's plain ``logo`` plan id per held-out basin."""
    frame = run.predictions[(run.predictions["kind"] == LOGO) & ~is_matched(run.predictions["plan_id"])]
    return dict(frame.drop_duplicates("held_out")[["held_out", "plan_id"]].itertuples(index=False, name=None))


def check_mixed_pairing(runs: Sequence[RunFolder]) -> None:
    """Refuse mixed-kind plans of one key that test different points, or a logo plan that misses some.

    Raises
    ------
    CompareError
        Naming the two plans (and runs) that differ.
    """
    seen: dict = {}
    for run in runs:
        for row in _mixed_plans(run).itertuples():
            key = (row.basin, row.fraction, row.repeat)
            points = run.test_points(row.plan_id)
            if key in seen and seen[key][2] != points:
                other_run, other_plan, _ = seen[key]
                raise CompareError(
                    f"{row.plan_id} ({run.name}) and {other_plan} ({other_run}) share the key basin, share, repeat "
                    f"{key} but test different points; they cannot be paired."
                )
            seen.setdefault(key, (run.name, row.plan_id, points))
    for run in runs:
        for basin, plan_id in _plain_logo(run).items():
            logo_points = run.test_points(plan_id)
            for key, (other_run, other_plan, points) in seen.items():
                if key[0] == basin and not points <= logo_points:
                    raise CompareError(
                        f"{plan_id} ({run.name}) does not test every point of {other_plan} ({other_run}): the model "
                        "with no local points cannot be scored on the shared test points."
                    )


def local_of(model: str) -> str:
    """The local model with ``model``'s inputs: ``local_global`` for Global, ``local`` otherwise."""
    return LOCAL_OF.get(model, LOCAL)


def _reference(runs: Sequence[RunFolder], basin: str, model: str, prefer: RunFolder):
    """The run (``prefer`` first) whose plain logo plan of ``basin`` has predictions of ``model``."""
    for run in [prefer, *[other for other in runs if other is not prefer]]:
        plan_id = _plain_logo(run).get(basin)
        if plan_id is None:
            continue
        rows = run.predictions[(run.predictions["plan_id"] == plan_id) & (run.predictions["model"] == model)]
        if not rows.empty:
            return run, rows
    return None, None


def mixed_arms(runs: Sequence[RunFolder], *, model: str = GLOBAL, with_devices: bool = False) -> pd.DataFrame:
    """``mixed_arms.csv``: per run, basin, share and repeat, every mixed kind's RMSE on the shared test points."""
    rows = []
    for run in runs:
        plans = _mixed_plans(run)
        if plans.empty:
            continue
        predictions = run.predictions[run.predictions["model"] == model]
        for (basin, fraction, repeat), group in plans.groupby(["basin", "fraction", "repeat"], sort=True):
            first = group.iloc[0]["plan_id"]
            points = run.test_points(first)
            mixed_id = group.loc[group["kind"] == MIXED, "plan_id"]
            row = {
                "run": run.name,
                "basin": basin,
                "fraction": float(fraction),
                "repeat": int(repeat),
                "n_moved": run.moved(mixed_id.iloc[0] if len(mixed_id) else first),
                "n_test": len(points),
            }
            for kind in ARM_ORDER:
                plan = group.loc[group["kind"] == kind, "plan_id"]
                row[f"rmse_{kind}"] = (
                    _rmse(predictions[predictions["plan_id"] == plan.iloc[0]]) if len(plan) else np.nan
                )
            for name, wanted in (("logo", model), ("local", local_of(model))):
                source, reference = _reference(runs, basin, wanted, run)
                row[f"rmse_{name}"] = (
                    _rmse(reference[reference["point_id"].isin(points)]) if reference is not None else np.nan
                )
                row[f"{name}_run"] = source.name if source is not None else ""
            if with_devices:
                row.update(_devices(run))
            rows.append(row)
    columns = [
        "run",
        "basin",
        "fraction",
        "repeat",
        "n_moved",
        "n_test",
        "rmse_logo",
        *[f"rmse_{kind}" for kind in ARM_ORDER],
        "rmse_local",
        "logo_run",
        "local_run",
        *(DEVICE_COLUMNS if with_devices else ()),
    ]
    return pd.DataFrame(rows, columns=columns)


def mixed_arms_paired(
    runs: Sequence[RunFolder],
    arms: pd.DataFrame,
    *,
    model: str = GLOBAL,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
    with_devices: bool = False,
) -> pd.DataFrame:
    """``mixed_arms_paired.csv``: each arm minus control and minus mixed, per run, basin and share.

    ``mean_difference`` is the mean over repeats of the per-repeat RMSE differences. The interval
    and p-value come from :func:`~yg_eo_soilnet.a1.evaluation.block_bootstrap`, which resamples the
    10 km blocks of the test points of every repeat of the share together (a point counts once per
    repeat) and scores both kinds on each resample; ``pooled_difference`` is its estimate, on those
    pooled points. ``p_holm`` corrects across the shares of one run, basin and comparison.
    ``n_repeats_better`` counts the repeats in which ``a`` has the lower RMSE.
    """
    columns = [
        "run",
        "basin",
        "fraction",
        "n_moved",
        "a",
        "b",
        "comparison",
        "mean_difference",
        "pooled_difference",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n_repeats",
        "n_repeats_better",
        "n",
        "n_blocks",
        *(DEVICE_COLUMNS if with_devices else ()),
    ]
    parts = []
    for run in runs:
        frame = run.predictions[run.predictions["kind"].isin(DIVIDED_KINDS) & (run.predictions["model"] == model)]
        if frame.empty:
            continue
        present = set(frame["kind"])
        pairs = [(a, b) for a, b in COMPARISONS if a in present and b in present]
        if not pairs:
            continue
        keyed = frame.assign(
            plan_id=frame["held_out"].astype(str)
            + "|"
            + frame["fraction"].astype(str)
            + "|"
            + frame["repeat"].astype(str)
        )
        result = evaluation.block_bootstrap(
            keyed, ["kind", "held_out", "fraction"], block_km=block_km, n=n, seed=seed, compare=pairs, compare_on="kind"
        )
        found = result.differences[result.differences["metric"] == "rmse"].rename(
            columns={"held_out": "basin", "estimate": "pooled_difference"}
        )
        found = found.assign(run=run.name, fraction=found["fraction"].astype(float))
        mine = arms[arms["run"] == run.name]
        extra = []
        for row in found.itertuples():
            share = mine[(mine["basin"] == row.basin) & np.isclose(mine["fraction"], row.fraction)]
            gaps = (share[f"rmse_{row.a}"] - share[f"rmse_{row.b}"]).dropna()
            extra.append(
                {
                    "mean_difference": float(gaps.mean()) if len(gaps) else np.nan,
                    "n_repeats": len(gaps),
                    "n_repeats_better": int((gaps < 0).sum()),
                    "n_moved": float(share["n_moved"].mean()) if len(share) else np.nan,
                }
            )
        found = pd.concat([found.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        found["comparison"] = found["a"] + " - " + found["b"]
        found["p_holm"] = found.groupby(["basin", "a", "b"], sort=False)["p_value"].transform(evaluation.holm)
        if with_devices:
            for column, value in _devices(run).items():
                found[column] = value
        parts.append(found)
    if not parts:
        return pd.DataFrame(columns=columns)
    table = pd.concat(parts, ignore_index=True)[columns]
    order = {pair: position for position, pair in enumerate(COMPARISONS)}
    table["_order"] = [order[(a, b)] for a, b in zip(table["a"], table["b"])]
    return table.sort_values(["run", "basin", "_order", "fraction"]).drop(columns="_order").reset_index(drop=True)


# --- leave one basin out, training size matched ---------------------------------------------------


def logo_draws(runs: Sequence[RunFolder], *, model: str = GLOBAL, with_devices: bool = False) -> tuple:
    """Every plain and matched ``logo`` plan's scores, after checking they test the same points per basin.

    Returns
    -------
    unmatched : pandas.DataFrame
        One row per basin (the first run that has its plain plan): ``run, basin, plan_id,
        training_size`` (its ``n_train``), ``rmse, r2, bias, n_test``.
    draws : pandas.DataFrame
        One row per matched draw: ``run, basin, training_size`` (asked), ``repeat, plan_id,
        n_train`` (drawn), ``rmse, r2, bias, n_test``.

    Raises
    ------
    CompareError
        If a basin's plans do not all test the same points.
    """
    unmatched, draws, points_of = [], [], {}

    def check(basin, run, plan_id):
        points = run.test_points(plan_id)
        if basin in points_of and points_of[basin][2] != points:
            other_run, other_plan, _ = points_of[basin]
            raise CompareError(
                f"{plan_id} ({run.name}) and {other_plan} ({other_run}) both hold out {basin} but test different "
                "points; they cannot be paired."
            )
        points_of.setdefault(basin, (run.name, plan_id, points))

    for run in runs:
        frame = run.predictions[(run.predictions["kind"] == LOGO) & (run.predictions["model"] == model)]
        for plan_id, part in frame.groupby("plan_id", sort=True):
            basin = str(part["held_out"].iloc[0])
            check(basin, run, plan_id)
            size = matched_training_size(plan_id)
            common = {"run": run.name, "basin": basin, "plan_id": plan_id, **_scores(part), "n_test": len(part)}
            if with_devices:
                common.update(_devices(run))
            if size is None:
                if basin not in {row["basin"] for row in unmatched}:
                    unmatched.append({**common, "training_size": run.n_train(plan_id)})
            else:
                repeat = part["repeat"].dropna()
                draws.append(
                    {
                        **common,
                        "training_size": size,
                        "repeat": int(repeat.iloc[0]) if len(repeat) else 0,
                        "n_train": run.n_train(plan_id),
                    }
                )
    device = list(DEVICE_COLUMNS) if with_devices else []
    unmatched_columns = ["run", "basin", "plan_id", "training_size", "rmse", "r2", "bias", "n_test", *device]
    draw_columns = ["run", "basin", "training_size", "repeat", "plan_id", "n_train", "rmse", "r2", "bias", "n_test"]
    return (
        pd.DataFrame(unmatched, columns=unmatched_columns),
        pd.DataFrame(draws, columns=[*draw_columns, *device]),
    )


def logo_matched(unmatched: pd.DataFrame, draws: pd.DataFrame, *, with_devices: bool = False) -> pd.DataFrame:
    """``logo_matched.csv``: per basin and training size, unmatched against the matched draws."""
    columns = [
        "basin",
        "unmatched_run",
        "unmatched_training_size",
        "unmatched_rmse",
        "unmatched_r2",
        "unmatched_bias",
        "matched_run",
        "training_size",
        "n_train_mean",
        "n_draws",
        *[f"matched_{score}_{end}" for score in ("rmse", "r2", "bias") for end in ("mean", "min", "max")],
        "rmse_difference",
        "r2_difference",
        "bias_difference",
        *(("unmatched_device_stage2", "matched_device_stage2") if with_devices else ()),
    ]
    if draws.empty:
        return pd.DataFrame(columns=columns)
    plain = unmatched.set_index("basin")
    rows = []
    for (basin, run, size), part in draws.groupby(["basin", "run", "training_size"], sort=True):
        reference = plain.loc[basin] if basin in plain.index else None
        row = {
            "basin": basin,
            "unmatched_run": reference["run"] if reference is not None else "",
            "unmatched_training_size": reference["training_size"] if reference is not None else np.nan,
            "matched_run": run,
            "training_size": int(size),
            "n_train_mean": float(part["n_train"].mean()),
            "n_draws": len(part),
        }
        for score in ("rmse", "r2", "bias"):
            row[f"unmatched_{score}"] = float(reference[score]) if reference is not None else np.nan
            row[f"matched_{score}_mean"] = float(part[score].mean())
            row[f"matched_{score}_min"] = float(part[score].min())
            row[f"matched_{score}_max"] = float(part[score].max())
            row[f"{score}_difference"] = row[f"matched_{score}_mean"] - row[f"unmatched_{score}"]
        if with_devices:
            row["unmatched_device_stage2"] = reference["device_stage2"] if reference is not None else ""
            row["matched_device_stage2"] = part["device_stage2"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# --- matched sizes, paired on stacked repeats ------------------------------------------------------

#: The label of the full leave-one-basin-out model in the matched tables.
FULL = "full"
#: The label of the local model in the matched tables.
LOCAL_LABEL = "local"


def matched_stack(runs: Sequence[RunFolder], reference: Optional[RunFolder], *, model: str = GLOBAL) -> tuple:
    """The matched draws and their references as one table, one row per point, model and repeat.

    ``model`` is a size label (``"3600"``, ``"2000"``) for each matched draw, :data:`FULL` for the
    reference run's plain ``logo`` plan and :data:`LOCAL_LABEL` for its local model on Global's inputs
    (``local``, with predicted CaCO3/CEC, when the run has no ``local_global``). ``plan_id`` is
    ``r<repeat>``: the key a paired comparison matches rows on. The full and local rows appear once,
    under ``r0``; :func:`matched_paired` repeats them per repeat.

    Returns
    -------
    stack : pandas.DataFrame
    local_model : str
        The local model used (empty without a reference).
    n_train : dict
        ``(basin, label)`` to the mean training size.
    """
    columns = ["held_out", "plan_id", "point_id", "split", "lat", "lon", "observed", "predicted", "model", "repeat"]
    parts, n_train = [], {}
    for run in runs:
        frame = run.predictions[
            (run.predictions["kind"] == LOGO)
            & is_matched(run.predictions["plan_id"])
            & (run.predictions["model"] == model)
        ]
        for plan_id, part in frame.groupby("plan_id", sort=True):
            label = str(matched_training_size(plan_id))
            repeat = int(part["repeat"].dropna().iloc[0]) if part["repeat"].notna().any() else 0
            parts.append(part.assign(model=label, repeat=repeat, plan_id=f"r{repeat}"))
            key = (str(part["held_out"].iloc[0]), label)
            n_train.setdefault(key, []).append(run.n_train(plan_id))
    local_model = ""
    if reference is not None:
        plain = reference.predictions[
            (reference.predictions["kind"] == LOGO) & ~is_matched(reference.predictions["plan_id"])
        ]
        present = set(plain["model"])
        local_model = LOCAL_GLOBAL if LOCAL_GLOBAL in present else (LOCAL if LOCAL in present else "")
        for wanted, label in ((model, FULL), (local_model, LOCAL_LABEL)):
            if not wanted:
                continue
            part = plain[plain["model"] == wanted]
            parts.append(part.assign(model=label, repeat=0, plan_id="r0"))
            if label == FULL:
                for plan_id, rows in part.groupby("plan_id"):
                    n_train[(str(rows["held_out"].iloc[0]), FULL)] = [reference.n_train(plan_id)]
    if not parts:
        return pd.DataFrame(columns=columns), local_model, {}
    stack = pd.concat(parts, ignore_index=True)
    stack["held_out"] = stack["held_out"].astype(str)
    means = {key: float(np.nanmean(values)) for key, values in n_train.items()}
    return stack[columns], local_model, means


def _sizes(stack: pd.DataFrame) -> list:
    """The matched size labels, largest first."""
    return sorted({label for label in stack["model"] if label not in (FULL, LOCAL_LABEL)}, key=float, reverse=True)


def matched_paired(
    stack: pd.DataFrame, *, local_model: str = "", block_km: float = 10.0, n: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """``matched_paired.csv``: per basin, the largest size against the full model, each size against the next larger,
    and each size against the local model - RMSE, bias and R² differences on the same points.

    Repeats are stacked (one row per point and repeat); the full and local predictions, one fit
    each, are repeated for every repeat so each draw is paired with them on its points; size
    against size pairs repeat r with repeat r. Intervals and p-values from resampling the 10 km
    blocks of the stacked points; ``p_holm`` corrects across the comparisons of one basin and score.
    """
    columns = [
        "basin",
        "a",
        "b",
        "metric",
        "estimate",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n",
        "n_blocks",
        "n_unpaired",
        "local_model",
    ]
    sizes = _sizes(stack)
    if not sizes:
        return pd.DataFrame(columns=columns)
    repeats = sorted({key for key in stack.loc[stack["model"].isin(sizes), "plan_id"]})
    single = stack[stack["model"].isin([FULL, LOCAL_LABEL])]
    paired = pd.concat(
        [stack[stack["model"].isin(sizes)], *[single.assign(plan_id=key) for key in repeats]], ignore_index=True
    )
    present = set(paired["model"])
    comparisons = [(sizes[0], FULL)] if FULL in present else []
    comparisons += [(smaller, larger) for larger, smaller in zip(sizes, sizes[1:])]
    comparisons += [(size, LOCAL_LABEL) for size in sizes if LOCAL_LABEL in present]
    result = evaluation.block_bootstrap(
        paired, ["held_out"], block_km=block_km, n=n, seed=seed, compare=comparisons, compare_on="model"
    )
    found = result.differences[result.differences["metric"].isin(evaluation.STACKED_SCORES)].copy()
    if found.empty:
        return pd.DataFrame(columns=columns)
    found["p_holm"] = found.groupby(["held_out", "metric"], sort=False)["p_value"].transform(evaluation.holm)
    found = found.rename(columns={"held_out": "basin"}).assign(local_model=local_model)
    return found[columns].reset_index(drop=True)


def matched_intervals(
    stack: pd.DataFrame,
    n_train: dict,
    *,
    local_model: str = "",
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """``matched_intervals.csv``: RMSE, bias and R² per basin and size (and full, local), repeats stacked.

    Each with a 95 % interval from resampling the 10 km blocks of the stacked points; the full and
    local models count once. ``n_train`` is the mean training size of the size's draws.
    """
    columns = ["basin", "model", "n_train", "metric", "estimate", "low", "high", "n", "n_blocks", "local_model"]
    if stack.empty or not _sizes(stack):  # no matched draws: nothing to set the references against
        return pd.DataFrame(columns=columns)
    intervals = evaluation.block_bootstrap(stack, ["held_out", "model"], block_km=block_km, n=n, seed=seed).intervals
    intervals = intervals[intervals["metric"].isin(evaluation.STACKED_SCORES)].rename(columns={"held_out": "basin"})
    intervals["n_train"] = [
        n_train.get((basin, label), np.nan) for basin, label in zip(intervals["basin"], intervals["model"])
    ]
    return intervals.assign(local_model=local_model)[columns].reset_index(drop=True)


def matched_repeats(stack: pd.DataFrame) -> pd.DataFrame:
    """RMSE, bias and R² of every matched draw on its own (the dots of figure 9's E4 panel)."""
    sizes = _sizes(stack)
    part = stack[stack["model"].isin(sizes)]
    if part.empty:
        return pd.DataFrame(columns=["basin", "model", "repeat", "n", *evaluation.STACKED_SCORES])
    table = evaluation.grouped_metrics(part, ["held_out", "model", "repeat"], min_n=0)
    return table.rename(columns={"held_out": "basin"})[["basin", "model", "repeat", "n", *evaluation.STACKED_SCORES]]


# --- the mixed kinds against the model with no local points -----------------------------------------


def arms_vs_transferred(
    runs: Sequence[RunFolder],
    arms: pd.DataFrame,
    *,
    model: str = GLOBAL,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """``arms_vs_transferred.csv``: each kind (control, mixed, arms) against the logo model on the same test points.

    Per run, basin and share, repeats stacked (:func:`~yg_eo_soilnet.a1.evaluation.mixed_stack`),
    paired block bootstrap; Holm across the kinds and shares of one basin and score
    (:func:`~yg_eo_soilnet.a1.evaluation.arms_vs_transferred`). ``estimate`` < 0: the local data do
    better than the model trained on the other basins alone.
    """
    columns = [
        "run",
        "basin",
        "fraction",
        "n_moved",
        "a",
        "b",
        "metric",
        "estimate",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n",
        "n_blocks",
        "n_unpaired",
    ]
    parts = []
    for run in runs:
        stacked = evaluation.mixed_stack(run.predictions, model=model)
        found = evaluation.arms_vs_transferred(stacked, block_km=block_km, n=n, seed=seed)
        if found.empty:
            continue
        found = found.rename(columns={"held_out": "basin"}).assign(run=run.name)
        mine = arms[arms["run"] == run.name] if not arms.empty else arms
        moved = mine.groupby(["basin", "fraction"])["n_moved"].mean() if not mine.empty else pd.Series(dtype=float)
        found["n_moved"] = [
            float(moved.get((basin, fraction), np.nan)) if len(moved) else np.nan
            for basin, fraction in zip(found["basin"], found["fraction"])
        ]
        parts.append(found)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


# --- whether transfer fails, from the main run -------------------------------------------------------


def transfer_status(
    run: Optional[RunFolder],
    *,
    model: str = GLOBAL,
    block_km: float = 10.0,
    n: int = 1000,
    seed: int = 42,
    local: Optional[str] = None,
) -> pd.DataFrame:
    """``transfer_status.csv``: per basin, transferred minus local RMSE on every input, and whether transfer fails.

    The plain ``logo`` plans of ``run``: ``model`` (transferred, trained on the other basins)
    against the local model on the same inputs (:func:`local_of`, or ``local`` when given; the basin's own
    points, out of fold) - ``local`` (with predicted CaCO3/CEC) when the run lacks ``local_global``,
    named in ``local_model`` - paired on the basin's points,
    with a block-bootstrap interval and p-value; Holm across the basins. A basin's transfer fails
    when the difference is above 0, its interval lies above 0 and ``p_holm`` < 0.05.
    """
    columns = [
        "run",
        "basin",
        "rmse_transferred",
        "rmse_local",
        "difference",
        "low",
        "high",
        "p_value",
        "p_holm",
        "n",
        "transfer_fails",
        "local_model",
    ]
    if run is None:
        return pd.DataFrame(columns=columns)
    frame = run.predictions[(run.predictions["kind"] == LOGO) & ~is_matched(run.predictions["plan_id"])]
    local = local or local_of(model)
    if local == LOCAL_GLOBAL and local not in set(frame["model"]) and LOCAL in set(frame["model"]):
        logger.warning(f"{run.name} has no {LOCAL_GLOBAL}; transfer status against {LOCAL} (with predicted CaCO3/CEC).")
        local = LOCAL
    if frame.empty or local not in set(frame["model"]) or model not in set(frame["model"]):
        logger.warning(f"{run.name} has no plain logo plans with both the {model} and the {local} model.")
        return pd.DataFrame(columns=columns)
    result = evaluation.block_bootstrap(
        frame, ["held_out", "model"], block_km=block_km, n=n, seed=seed, compare=[(model, local)], compare_on="model"
    )
    found = result.differences[result.differences["metric"] == "rmse"].copy()
    found["p_holm"] = evaluation.holm(found["p_value"].to_numpy())
    rows = []
    for row in found.itertuples():
        part = frame[frame["held_out"] == row.held_out]
        rows.append(
            {
                "run": run.name,
                "basin": row.held_out,
                "rmse_transferred": _rmse(part[part["model"] == model]),
                "rmse_local": _rmse(part[part["model"] == local]),
                "difference": row.estimate,
                "low": row.low,
                "high": row.high,
                "p_value": row.p_value,
                "p_holm": row.p_holm,
                "n": row.n,
                "transfer_fails": bool(row.estimate > 0 and row.low > 0 and row.p_holm < ALPHA),
                "local_model": local,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# --- the same plans in two runs -------------------------------------------------------------------


def cross_run(runs: Sequence[RunFolder], *, model: str = GLOBAL, with_devices: bool = False) -> pd.DataFrame:
    """``cross_run.csv``: every plan predicted in two runs, scored in each on the same points.

    Raises
    ------
    CompareError
        If the two runs' files of one plan test different points.
    """
    rows = []
    for i, run_a in enumerate(runs):
        for run_b in runs[i + 1 :]:
            a = run_a.predictions[run_a.predictions["model"] == model]
            b = run_b.predictions[run_b.predictions["model"] == model]
            for plan_id in sorted(set(a["plan_id"]) & set(b["plan_id"])):
                if run_a.test_points(plan_id) != run_b.test_points(plan_id):
                    raise CompareError(
                        f"{plan_id} tests different points in {run_a.name} and {run_b.name}; they cannot be paired."
                    )
                part_a = a[a["plan_id"] == plan_id]
                merged = part_a[["point_id", "observed", "predicted"]].merge(
                    b.loc[b["plan_id"] == plan_id, ["point_id", "predicted"]], on="point_id", suffixes=("_a", "_b")
                )
                first = part_a.iloc[0]
                gap = merged["predicted_a"] - merged["predicted_b"]
                rmse_a = _rmse(merged.rename(columns={"predicted_a": "predicted"}))
                rmse_b = _rmse(merged.rename(columns={"predicted_b": "predicted"}))
                row = {
                    "kind": first["kind"],
                    "plan_id": plan_id,
                    "basin": first["held_out"],
                    "fraction": first["fraction"],
                    "repeat": first["repeat"],
                    "run_a": run_a.name,
                    "run_b": run_b.name,
                    "rmse_a": rmse_a,
                    "rmse_b": rmse_b,
                    "rmse_difference": rmse_a - rmse_b,
                    "mean_abs_prediction_difference": float(gap.abs().mean()),
                    "pearson_r": _pearson(merged["predicted_a"], merged["predicted_b"]),
                    "n": len(merged),
                }
                if with_devices:
                    row.update(_devices(run_a, "a_"))
                    row.update(_devices(run_b, "b_"))
                rows.append(row)
    columns = [
        "kind",
        "plan_id",
        "basin",
        "fraction",
        "repeat",
        "run_a",
        "run_b",
        "rmse_a",
        "rmse_b",
        "rmse_difference",
        "mean_abs_prediction_difference",
        "pearson_r",
        "n",
        *([f"{side}_{column}" for side in ("a", "b") for column in DEVICE_COLUMNS] if with_devices else []),
    ]
    return pd.DataFrame(rows, columns=columns)


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cross_run_summary(runs: Sequence[RunFolder], crossed: pd.DataFrame, arms: pd.DataFrame, *, model: str = GLOBAL):
    """``cross_run_summary.csv``: per kind and pair of runs, how far the two runs' scores and predictions lie apart.

    ``rmse_difference_mean`` / ``_sd`` and ``max_abs_rmse_difference`` over the plans;
    ``mean_abs_prediction_difference`` and ``pearson_r`` over every point of those plans pooled;
    and, to read them against, the mean mixed - control RMSE difference (and its mean absolute
    value) over the keys of the first of the two runs that has both kinds (``effect_run``).
    """
    columns = [
        "kind",
        "run_a",
        "run_b",
        "n_plans",
        "rmse_difference_mean",
        "rmse_difference_sd",
        "max_abs_rmse_difference",
        "mean_abs_prediction_difference",
        "pearson_r",
        "n_points",
        "effect_run",
        "mean_mixed_minus_control",
        "mean_abs_mixed_minus_control",
    ]
    if crossed.empty:
        return pd.DataFrame(columns=columns)
    by_name = {run.name: run for run in runs}
    rows = []
    for (kind, name_a, name_b), part in crossed.groupby(["kind", "run_a", "run_b"], sort=True):
        a = by_name[name_a].predictions
        b = by_name[name_b].predictions
        a = a[(a["model"] == model) & a["plan_id"].isin(part["plan_id"])]
        b = b[(b["model"] == model) & b["plan_id"].isin(part["plan_id"])]
        points = a[["plan_id", "point_id", "predicted"]].merge(
            b[["plan_id", "point_id", "predicted"]], on=["plan_id", "point_id"], suffixes=("_a", "_b")
        )
        effect_run, effect, effect_abs = "", np.nan, np.nan
        for name in (name_a, name_b):
            mine = arms[arms["run"] == name] if not arms.empty else arms
            gaps = (mine["rmse_mixed"] - mine["rmse_control"]).dropna() if not mine.empty else pd.Series(dtype=float)
            if len(gaps):
                effect_run, effect, effect_abs = name, float(gaps.mean()), float(gaps.abs().mean())
                break
        rows.append(
            {
                "kind": kind,
                "run_a": name_a,
                "run_b": name_b,
                "n_plans": len(part),
                "rmse_difference_mean": float(part["rmse_difference"].mean()),
                "rmse_difference_sd": float(part["rmse_difference"].std(ddof=1)) if len(part) > 1 else np.nan,
                "max_abs_rmse_difference": float(part["rmse_difference"].abs().max()),
                "mean_abs_prediction_difference": float((points["predicted_a"] - points["predicted_b"]).abs().mean()),
                "pearson_r": _pearson(points["predicted_a"], points["predicted_b"]),
                "n_points": len(points),
                "effect_run": effect_run,
                "mean_mixed_minus_control": effect,
                "mean_abs_mixed_minus_control": effect_abs,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# --- everything at once ----------------------------------------------------------------------------


@dataclass
class CompareResult:
    """What :func:`compare_runs` computed and wrote."""

    out_dir: Path
    runs: list
    tables: dict
    figures: dict
    captions: dict
    mlflow_run_id: Optional[str] = None


#: The tables written, each as CSV, and as LaTeX unless listed in :data:`CSV_ONLY`.
TABLE_NAMES = (
    "mixed_arms",
    "mixed_arms_paired",
    "logo_matched",
    "logo_matched_draws",
    "transfer_status",
    "cross_run",
    "cross_run_summary",
)
#: Tables too long for a LaTeX page: one row per plan or draw.
CSV_ONLY = (
    "mixed_arms",
    "logo_matched_draws",
    "cross_run",
    "matched_paired",
    "matched_intervals",
    "arms_vs_transferred",
)
CAPTIONS_FILE = "captions.md"


def check_devices(runs: Sequence[RunFolder], *, allow_mixed: bool = False) -> None:
    """Refuse runs fitted on different processors (:func:`~yg_eo_soilnet.a1.devices.check_same_devices`).

    A stage a run did not fit (``none``: stage 1 of a run of the global model alone) is left out of
    that stage's check: it cannot make two runs differ.
    """
    for stage in ("stage1", "stage2"):
        fitted = {run.name: {stage: run.devices[stage]} for run in runs if run.devices[stage] != "none"}
        check_same_devices(fitted, allow_mixed=allow_mixed)


def pick_transfer_run(runs: Sequence[RunFolder], transfer_run: Optional[str] = None) -> Optional[RunFolder]:
    """The run whose transfer losses classify the basins: ``transfer_run`` if given, else ``a1_xgb`` among the runs."""
    if transfer_run is not None:
        wanted = Path(transfer_run).resolve()
        for run in runs:
            if run.run_dir.resolve() == wanted:
                return run
        return None
    for run in runs:
        if run.output_name == TRANSFER_RUN:
            return run
    return None


def compare_runs(
    run_dirs: Sequence,
    out_dir,
    *,
    kinds: Optional[Sequence[str]] = None,
    model: str = GLOBAL,
    allow_mixed_devices: bool = False,
    transfer_run: Optional[str] = None,
    block_km: float = 10.0,
    n_boot: int = 1000,
    seed: int = 42,
    tracking: bool = True,
) -> CompareResult:
    """Read the run folders, check they can be paired, and write every table, figure and caption.

    Raises
    ------
    CompareError
        If plans of one key test different points, or a plan file changed since its run.
    DeviceMismatch
        If the runs were fitted on different processors and ``allow_mixed_devices`` is not set.
    """
    from yg_eo_soilnet.a1.figures import compare as compare_figures
    from yg_eo_soilnet.a1.figures.core import save_figure, write_table

    runs = [read_run(path, model=model, kinds=kinds) for path in run_dirs]
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise CompareError(f"Two run folders have the same name: {names}")
    check_devices(runs, allow_mixed=allow_mixed_devices)
    with_devices = allow_mixed_devices
    check_mixed_pairing(runs)

    arms = mixed_arms(runs, model=model, with_devices=with_devices)
    paired = mixed_arms_paired(
        runs, arms, model=model, block_km=block_km, n=n_boot, seed=seed, with_devices=with_devices
    )
    unmatched, draws = logo_draws(runs, model=model, with_devices=with_devices)
    matched = logo_matched(unmatched, draws, with_devices=with_devices)
    reference = pick_transfer_run(runs, transfer_run)
    if reference is None and transfer_run is not None:
        reference = read_run(transfer_run, model=model)
    status = transfer_status(reference, model=model, block_km=block_km, n=n_boot, seed=seed)
    crossed = cross_run(runs, model=model, with_devices=with_devices)
    stack, local_model, n_train = matched_stack(runs, reference, model=model)
    tables = {
        "mixed_arms": arms,
        "mixed_arms_paired": paired,
        "logo_matched": matched,
        "logo_matched_draws": draws,
        "transfer_status": status,
        "cross_run": crossed,
        "cross_run_summary": cross_run_summary(runs, crossed, arms, model=model),
        "matched_paired": matched_paired(stack, local_model=local_model, block_km=block_km, n=n_boot, seed=seed),
        "matched_intervals": matched_intervals(
            stack, n_train, local_model=local_model, block_km=block_km, n=n_boot, seed=seed
        ),
        "arms_vs_transferred": arms_vs_transferred(runs, arms, model=model, block_km=block_km, n=n_boot, seed=seed),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name in CSV_ONLY:
            table.to_csv(out / f"{name}.csv", index=False)
        else:
            write_table(table, out / name, float_format="%.3f")
    figures, captions = {}, {}
    for name, (figure, caption) in compare_figures.draw_all(tables, unmatched, model=model).items():
        save_figure(figure, name, out)
        figures[name], captions[name] = figure, caption
    write_compare_captions(captions, out / CAPTIONS_FILE)
    run_id = _log_to_mlflow(runs[0], out, runs) if tracking else None
    return CompareResult(out, runs, tables, figures, captions, run_id)


def write_compare_captions(captions: dict, path) -> Path:
    """``captions.md``: one section per figure, its draft caption with the numbers under it."""
    lines = ["# Draft figure captions", "", "Written by compare.py from the compared runs; edit before use.", ""]
    for name, caption in captions.items():
        lines += [f"## {name}", "", caption.strip(), ""]
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _log_to_mlflow(first: RunFolder, out: Path, runs: Sequence[RunFolder]) -> Optional[str]:
    """An MLflow parent run ``compare`` in the first run's experiment, holding everything written."""
    import mlflow

    entry = first.manifest.get("mlflow") or {}
    if not entry.get("experiment_id"):
        logger.warning(f"{first.name} records no MLflow experiment; the comparison is not added to MLflow.")
        return None
    if entry.get("tracking_uri"):
        mlflow.set_tracking_uri(entry["tracking_uri"])
    tags = {"a1_compare_runs": ";".join(run.name for run in runs), "a1_step": "compare"}
    with mlflow.start_run(experiment_id=entry["experiment_id"], run_name="compare", tags=tags) as run:
        mlflow.log_artifacts(str(out), "compare")
        return run.info.run_id


def finite(value) -> bool:
    """Whether a value is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
