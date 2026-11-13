"""Where a run folder keeps each plan's files, one set per seed.

A run folder (``<output_dir>/runs/<run name>/``) written by ``run_a1.py`` holds, per plan and seed:

``stage1/<plan_id>/seed_<seed>.parquet``
    Stage 1's audit table (absent when no requested SOM model reads stage 1);
``predictions/<plan_id>/seed_<seed>/<model>__<covariate group>.parquet``
    One predictions table per SOM model and covariate group.

Folders written before seeds were repeated (prediction format 4) kept one file per plan,
``stage1/<plan_id>.parquet`` and ``predictions/<plan_id>/<model>__<group>.parquet``; they are read
as the run's first seed, and never written to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PREDICTIONS_DIR = "predictions"
STAGE1_DIR = "stage1"


def seed_name(seed: int) -> str:
    """``seed_<seed>``: the folder (predictions) or file stem (stage 1) of one seed."""
    return f"seed_{int(seed)}"


def prediction_dir(run_dir, plan_id: str, seed: int) -> Path:
    """``predictions/<plan_id>/seed_<seed>/``."""
    return Path(run_dir) / PREDICTIONS_DIR / plan_id / seed_name(seed)


def stage1_path(run_dir, plan_id: str, seed: int) -> Path:
    """``stage1/<plan_id>/seed_<seed>.parquet``: where a run writes stage 1's audit of one plan and seed."""
    return Path(run_dir) / STAGE1_DIR / plan_id / f"{seed_name(seed)}.parquet"


def find_stage1(run_dir, plan_id: str, seed: int) -> Optional[Path]:
    """Stage 1's audit of one plan and seed, if the run has it.

    A folder of the older layout (one file per plan, ``stage1/<plan_id>.parquet``) answers for any
    seed: it was run with one.
    """
    path = stage1_path(run_dir, plan_id, seed)
    if path.exists():
        return path
    legacy = Path(run_dir) / STAGE1_DIR / f"{plan_id}.parquet"
    return legacy if legacy.exists() else None


def has_stage1(run_dir, plan_id: str) -> bool:
    """Whether any seed of this plan has a stage-1 audit (the plan ran, with stage 1)."""
    root = Path(run_dir) / STAGE1_DIR
    return (root / f"{plan_id}.parquet").exists() or any((root / plan_id).glob("seed_*.parquet"))


def ran_plan(run_dir, plan_id: str) -> bool:
    """Whether the run has outputs for this plan: a stage-1 audit, or prediction files."""
    root = Path(run_dir) / PREDICTIONS_DIR / plan_id
    return has_stage1(run_dir, plan_id) or any(root.glob("*.parquet")) or any(root.glob("seed_*/*.parquet"))


def prediction_paths(run_dir, plan_id: Optional[str] = None) -> list[Path]:
    """Every predictions file of a run (or of one plan), in either layout, sorted."""
    root = Path(run_dir) / PREDICTIONS_DIR
    plans = plan_id or "*"
    return sorted([*root.glob(f"{plans}/*.parquet"), *root.glob(f"{plans}/seed_*/*.parquet")])
