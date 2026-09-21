"""Redraw the figures of a run that has already finished, in place.

The plotters are restyled from time to time - most recently onto the shared style that
``notebooks/paper_figures.ipynb`` defines. Every run trained before such a change keeps the old
picture, and retraining to refresh it is the wrong trade: it would produce a different fit, so the
figure and the metrics logged beside it would stop describing the same model.

    python replot.py --parent-run-id 73410639b8a4417395d5bec247b6540a
    python replot.py --run-id f2e7d1a5530545aba7afbf39277d8a75 --only pred_obs
    python replot.py --experiment Soil_Model_Training_v2 --since 2026-09-01 --dry-run

Nothing is loaded but the run's own artifacts. ``eval_results/eval_results.csv`` holds the
observations, the predictions and - on an uncertainty run - sigma and the interval bounds;
``cv/cv_results.csv`` holds the sklearn sweep; the parent figures come from the children's CSVs.
No source data, no checkpoint, no inference, so this runs in seconds and cannot be affected by the
dataset having moved since.

Figures are re-logged onto the artifact path they already occupy, so ``plots/pred_obs.png`` is
replaced rather than duplicated. Figures written under names this project has since retired - the
run-root ``obs_pred_and_residual_plot.png`` that sklearn runs carried before the MLflow evaluator
hook was dropped - are LEFT ALONE: deleting artifacts from a finished run is not something this
command does. Delete them by hand if they bother you.

Runs that never wrote an eval CSV, which in a large experiment means most ensemble members and
every run that failed early, are skipped and counted rather than raised on.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run-id",
        help="Replot ONE run's own figures. A parent run given here gets only its leaderboard pair.",
    )
    target.add_argument(
        "--parent-run-id",
        help="Replot a whole run tree: every scoring child, then the parent's leaderboard pair",
    )
    target.add_argument(
        "--experiment",
        help="Replot every parent run in this experiment, by name or id. Use --since to bound it.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main config, read only for the tracking URI",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated figure kinds to redraw instead of all of them: "
            + ", ".join(ALL_KINDS)
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        help="With --experiment, skip runs that started before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which runs would be replotted and what each would write, writing nothing",
    )
    return parser.parse_args(argv)


def _kinds(only: str | None) -> list[str] | None:
    if not only:
        return None
    kinds = [kind.strip() for kind in only.split(",") if kind.strip()]
    unknown = [kind for kind in kinds if kind not in ALL_KINDS]
    if unknown:
        raise SystemExit(
            f"Unknown figure kind(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_KINDS)}"
        )
    return kinds


def _parent_runs(experiment: str, since: str | None) -> list:
    """Every top-level run in an experiment, newest first.

    A parent is identified by the ABSENCE of an ``mlflow.parentRunId`` tag, filtered here rather
    than in the search string because MLflow's filter syntax cannot express "this tag is not set".
    """
    client = mlflow.tracking.MlflowClient()
    found = client.get_experiment_by_name(experiment) or client.get_experiment(experiment)
    if found is None:
        raise SystemExit(f"No experiment named or numbered {experiment!r}")

    filter_string = f"attributes.start_time >= '{since}'" if since else ""
    runs = client.search_runs(
        experiment_ids=[found.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=50_000,
    )
    return [run for run in runs if "mlflow.parentRunId" not in run.data.tags]


def _report(outcomes: list[dict[str, Any]], logger, dry_run: bool) -> int:
    """Print one line per run and a closing tally. Returns the number of runs actually written to."""
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
    args = parse_args(argv)
    only = _kinds(args.only)

    config = Config(config_path=args.config_path)
    # The URI only, exactly as relog.py and export_predictions.py do: switching to the config's
    # experiment would make start_run(run_id=...) fail whenever the run belongs to a different one.
    # A run is reached by id, not by experiment.
    configure_tracking_uri(config)

    logger = TrainingLogger(name="replot", enable_file_logging=False).get_logger()
    client = mlflow.tracking.MlflowClient()

    outcomes: list[dict[str, Any]] = []
    if args.run_id:
        run = client.get_run(args.run_id)
        # start_run(run_id=...) requires the active experiment to be the run's own.
        mlflow.set_experiment(experiment_id=run.info.experiment_id)
        if "mlflow.parentRunId" in run.data.tags:
            outcomes.append(regenerate_child_figures(run, only=only, dry_run=args.dry_run))
        else:
            # A top-level run holds no eval CSV of its own; its figures are the leaderboard pair.
            outcomes.append(
                regenerate_parent_figures(args.run_id, only=only, dry_run=args.dry_run)
            )
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
            outcomes.extend(
                regenerate_tree(parent.info.run_id, only=only, dry_run=args.dry_run)
            )

    written = _report(outcomes, logger, args.dry_run)
    verb = "would be refreshed" if args.dry_run else "refreshed"
    logger.info(f"{written} of {len(outcomes)} run(s) {verb}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
