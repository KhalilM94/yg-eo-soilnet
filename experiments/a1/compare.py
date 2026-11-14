"""Compare Paper A1 runs across run folders: the mixed arms, matched leave-one-basin-out, processor vs card.

Reads the manifests, plan files and prediction files of several run folders and writes, into
``--out``: ``mixed_arms.csv``, ``mixed_arms_paired.csv``, ``logo_matched.csv``,
``logo_matched_draws.csv``, ``transfer_status.csv``, ``cross_run.csv`` and
``cross_run_summary.csv`` (the summary tables also as LaTeX), figures ``figC1_mixed_arms`` and
``figC2_logo_matched`` (PNG and PDF), and ``captions.md``; then records them in an MLflow run named
``compare`` in the experiment of the first run. See :mod:`yg_eo_soilnet.a1.compare` for what each
holds.

Plans are paired only on the same test points, read from the plan files; runs fitted on different
processors are refused unless ``--allow-mixed-devices``, which adds the devices to every table.

Run from the repository root::

    python experiments/a1/compare.py --runs results/a1_mixed_arms/runs/default \\
        results/a1_mixed_contrast/runs/default --out results/a1_compare_mixed
    python experiments/a1/compare.py --runs results/a1_xgb/runs/default \\
        results/a1_logo_matched/runs/default results/a1_logo_matched_2000/runs/default \\
        --out results/a1_compare_logo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yg_eo_soilnet.a1.compare import compare_runs  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.stage2 import GLOBAL  # noqa: E402


def main(argv: list[str] | None = None):
    """Compare run folders from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    CompareResult
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Run folders, <output_dir>/runs/<run name>.")
    parser.add_argument("--out", required=True, help="The folder to write the tables, figures and captions into.")
    parser.add_argument("--kinds", default=None, help="Comma-separated plan kinds to read (default all).")
    parser.add_argument("--model", default=GLOBAL, help="The model compared (default global).")
    parser.add_argument(
        "--allow-mixed-devices",
        action="store_true",
        help="Pair runs fitted on different processors; the devices become a column of every table.",
    )
    parser.add_argument(
        "--transfer-run",
        default=None,
        help="The run whose transfer losses classify the basins in figure C1 (default: a1_xgb, if among --runs).",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000).")
    parser.add_argument("--block-km", type=float, default=10.0, help="Side of a spatial block, km (default 10).")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the resampling (default 42).")
    parser.add_argument("--no-mlflow", action="store_true", help="Write the files only.")
    args = parser.parse_args(argv)

    out = Path(args.out)
    setup_logging(out / "compare.log")
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()] if args.kinds else None
    result = compare_runs(
        args.runs,
        out,
        kinds=kinds,
        model=args.model,
        allow_mixed_devices=args.allow_mixed_devices,
        transfer_run=args.transfer_run,
        block_km=args.block_km,
        n_boot=args.n_boot,
        seed=args.seed,
        tracking=not args.no_mlflow,
    )
    for name, table in result.tables.items():
        summary(f"  {name}: {len(table)} row(s)")
    summary(f"Tables, figures and captions in {out}/")
    if result.mlflow_run_id:
        summary(f"Added to MLflow run {result.mlflow_run_id} (compare).")
    return result


if __name__ == "__main__":
    main()
