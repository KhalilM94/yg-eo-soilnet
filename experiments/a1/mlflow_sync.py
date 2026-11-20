"""Log finished A1 run folders to MLflow afterwards (``results/`` is the record; MLflow an index).

For runs made elsewhere - an HPC cluster with ``--no-mlflow``, or a per-job store - and for runs
whose logging failed. Each folder (``<output_dir>/runs/<run name>``) becomes a parent run with one
child per plan and seed (E5: per plan, seed and variant), its evaluation tables and headline
metrics, its figures, and for E5 the ``e5_summary`` child, as the scripts would have logged them.
Syncing again updates the same parent (matched by the run id the manifest records in this store,
or by the folder and plan hashes); nothing is duplicated. The parent's id is written into the
manifest. See :mod:`yg_eo_soilnet.a1.sync`.

The store is the current A1 tracking URI: ``MLFLOW_TRACKING_URI``, else ``sqlite:///mlflow.db`` at
the repository root.

Run from the repository root::

    python experiments/a1/mlflow_sync.py results/a1_xgb/runs/default results/a1_e5/runs/default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yg_eo_soilnet.a1.logs import summary  # noqa: E402
from yg_eo_soilnet.a1.sync import sync_run_folder  # noqa: E402


def main(argv: list[str] | None = None) -> list:
    """Sync run folders from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    list of yg_eo_soilnet.a1.sync.SyncResult
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run folders, <output_dir>/runs/<run name>.")
    parser.add_argument("--experiment", default=None, help="The experiment (default: the settings' mlflow_experiment).")
    parser.add_argument(
        "--no-write-manifest", action="store_true", help="Leave the manifests as they are (a scratch store)."
    )
    args = parser.parse_args(argv)

    import mlflow

    results = []
    for run_dir in args.run_dirs:
        result = sync_run_folder(run_dir, experiment=args.experiment, write_manifest=not args.no_write_manifest)
        results.append(result)
        verb = "created" if result.created_parent else "updated"
        summary(
            f"{run_dir}: parent {result.run_id} {verb}; {result.children_created} child run(s) added, "
            f"{result.children_kept} already there; {result.metrics_logged} metric(s) and "
            f"{result.artifacts_logged} artifact(s) logged."
        )
        for note in result.notes:
            summary(f"  note: {note}")
    summary(f"Store: {mlflow.get_tracking_uri()}")
    return results


if __name__ == "__main__":
    main()
