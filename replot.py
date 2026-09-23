"""Redraw the figures of runs that have already finished, in place.

Use this after the plotting style changes, to bring older runs' figures up to date without
retraining (retraining would give a different model, so the figures would no longer match the
scores recorded next to them).

    python replot.py --parent-run-id <run id>                     # a whole training run
    python replot.py --run-id <run id> --only pred_obs            # one model's run, one figure kind
    python replot.py --experiment Soil_Model_Training_v2 --dry-run

Only the run's own saved tables are read: eval_results/eval_results.csv (test measurements,
predictions and, with uncertainty on, the standard deviation and interval) and, for scikit-learn
models, cv/cv_results.csv. No data, checkpoint or model is loaded, so it takes seconds.

Each figure replaces the file at the same place in the run (for example plots/pred_obs.png).
Figures under names no longer used are left in place; delete them by hand if needed. Runs with no
eval_results.csv - usually ensemble copies and runs that failed early - are skipped and counted.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Any

import mlflow

from config import Config
from yg_eo_soilnet.logger import TrainingLogger
from yg_eo_soilnet.replotting import (
    ALL_KINDS,
    regenerate_child_figures,
    regenerate_parent_figures,
    regenerate_tree,
)
from yg_eo_soilnet.tracking import configure_tracking_uri


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options; ``argv`` defaults to the real command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run-id",
        help="Redraw one run's figures. For a main (parent) run, that means its two leaderboard figures.",
    )
    target.add_argument(
        "--parent-run-id",
        help="Redraw a whole training run: every model's figures, then the leaderboard figures.",
    )
    target.add_argument(
        "--experiment",
        help="Redraw every training run in this MLflow experiment (name or id).",
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main configuration file (default: configs/main_config.yml), read only for the MLflow location.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=("Only redraw these kinds of figure, comma-separated (default: all): " + ", ".join(ALL_KINDS)),
    )
    parser.add_argument(
        "--since",
        default=None,
        help="With --experiment, skip runs that started before this date (YYYY-MM-DD), read in local time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which runs would be redrawn and which files would be written, without writing.",
    )
    return parser.parse_args(argv)


def _kinds(only: str | None) -> list[str] | None:
    """Turn ``--only`` into a list of figure kinds; ``None`` means all. Unknown kinds stop the run."""
    if not only:
        return None
    kinds = [kind.strip() for kind in only.split(",") if kind.strip()]
    unknown = [kind for kind in kinds if kind not in ALL_KINDS]
    if unknown:
        raise SystemExit(f"Unknown figure kind(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_KINDS)}")
    return kinds


def _since_filter(since: str | None) -> str:
    """The MLflow filter for ``--since``, or an empty string when no date was given.

    A run's start time is stored as milliseconds since the epoch, and MLflow refuses a quoted
    string there, so the date is converted rather than passed through. Read in local time, which is
    the clock a run's name is written in.

    Raises
    ------
    SystemExit
        If the date is not ``YYYY-MM-DD``, said here rather than as an MLflow parse error.
    """
    if not since:
        return ""
    try:
        moment = datetime.datetime.strptime(since, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"--since expects a date as YYYY-MM-DD, got {since!r}") from None
    return f"attributes.start_time >= {int(moment.timestamp() * 1000)}"


def _parent_runs(experiment: str, since: str | None) -> list:
    """Return every main (top-level) run of an experiment, newest first.

    A main run is one with no ``mlflow.parentRunId`` tag.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        found = client.get_experiment_by_name(experiment) or client.get_experiment(experiment)
    except Exception:
        # get_experiment takes an id, so a name that is not an experiment raises rather than
        # returning None. Either way the answer is the same: there is no such experiment.
        found = None
    if found is None:
        raise SystemExit(f"No experiment named or numbered {experiment!r}")

    filter_string = _since_filter(since)
    runs = client.search_runs(
        experiment_ids=[found.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=50_000,
    )
    return [run for run in runs if "mlflow.parentRunId" not in run.data.tags]


def _report(outcomes: list[dict[str, Any]], logger, dry_run: bool) -> int:
    """Log one line per run and return how many runs got at least one figure."""
    written_runs = 0
    for outcome in outcomes:
        name = outcome.get("run_name") or outcome["run_id"][:8]
        if outcome.get("skipped"):
            logger.info(f"  skip {name}: {outcome['skipped']}")
            continue
        paths = outcome.get("would_write" if dry_run else "written", [])
        if not paths:
            logger.info(f"  skip {name}: nothing to redraw")
            continue
        written_runs += 1
        verb = "would write" if dry_run else "wrote"
        logger.info(f"  {verb} {len(paths)} figure(s) for {name}: {', '.join(paths)}")
    return written_runs


def main(argv: list[str] | None = None) -> int:
    """Redraw the requested runs' figures and report what was done; returns 0."""
    args = parse_args(argv)
    only = _kinds(args.only)

    config = Config(config_path=args.config_path)
    # Only the MLflow location is taken from the config: runs are opened by id, in their own
    # experiment.
    configure_tracking_uri(config)

    logger = TrainingLogger(name="replot", enable_file_logging=False).get_logger()
    client = mlflow.tracking.MlflowClient()

    outcomes: list[dict[str, Any]] = []
    if args.run_id:
        run = client.get_run(args.run_id)
        # Reopening a run requires its own experiment to be the active one.
        mlflow.set_experiment(experiment_id=run.info.experiment_id)
        if "mlflow.parentRunId" in run.data.tags:
            outcomes.append(regenerate_child_figures(run, only=only, dry_run=args.dry_run))
        else:
            # A main run has no predictions of its own; its figures are the leaderboard pair.
            outcomes.append(regenerate_parent_figures(args.run_id, only=only, dry_run=args.dry_run))
    elif args.parent_run_id:
        parent = client.get_run(args.parent_run_id)
        mlflow.set_experiment(experiment_id=parent.info.experiment_id)
        logger.info(f"Replotting {parent.info.run_name or args.parent_run_id}")
        outcomes.extend(regenerate_tree(args.parent_run_id, only=only, dry_run=args.dry_run))
    else:
        parents = _parent_runs(args.experiment, args.since)
        logger.info(f"Found {len(parents)} parent run(s) in {args.experiment}")
        for parent in parents:
            mlflow.set_experiment(experiment_id=parent.info.experiment_id)
            logger.info(f"Replotting {parent.info.run_name or parent.info.run_id}")
            outcomes.extend(regenerate_tree(parent.info.run_id, only=only, dry_run=args.dry_run))

    written = _report(outcomes, logger, args.dry_run)
    verb = "would be refreshed" if args.dry_run else "refreshed"
    logger.info(f"{written} of {len(outcomes)} run(s) {verb}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
