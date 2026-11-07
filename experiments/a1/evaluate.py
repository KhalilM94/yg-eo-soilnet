"""Score a Paper A1 run: tables of scores, block-bootstrap intervals and transfer losses.

Reads only the per-point prediction files ``run_a1.py`` wrote (see
:mod:`yg_eo_soilnet.a1.evaluation`) and writes, into ``<output_dir>/runs/<run name>/evaluation/``:

``scores.csv``
    RMSE, MAE, R², bias, RPD and RPIQ per plan kind, model, covariate group and split. Every table
    keeps ``spatial_kfold`` scores of different block sizes apart (``block_size_m``): each point is
    tested once per size, so pooling them would count it several times. Comparing the sizes is the
    block-size sensitivity check.
``scores_by_class.csv``
    The same on the test points, per kind, model and measured carbonate class.
``intervals.csv``
    95 % intervals of each test score per kind and model (every covariate), from resampling whole
    spatial blocks.
``differences.csv``
    The same for ``per_class_measured``, ``per_class_predicted`` and ``global_aux`` each minus
    ``global``, scored on the same points, with p-values and their Holm correction.
``intervals_by_region.csv``
    The same intervals for each held-out region of the ``logo`` plans.
``transfer.csv``
    For ``logo`` and ``mixed`` plans, the local against the transferred score per region and
    covariate group, and the drop between them.

Unless ``--no-mlflow`` is given, the tables, ``region_names.json`` and the headline metrics
(:func:`yg_eo_soilnet.a1.evaluation.headline_metrics`) are also added to the run folder's MLflow
run - the parent run ``run_a1.py`` opened, whose id the folder's ``manifest.json`` records.

Run from the repository root, after ``run_a1.py``::

    python experiments/a1/evaluate.py --config configs/a1/a1_demo.yml
    python experiments/a1/evaluate.py --config configs/a1/a1.yml --run-name default --n-boot 1000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    # The project's Config lives at the repository root, which is not on the path when this file
    # is run as a script.
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from yg_eo_soilnet.a1 import evaluation  # noqa: E402
from yg_eo_soilnet.a1.config import COMBINED_GROUP  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402
from yg_eo_soilnet.a1.plans import LOGO  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.stage2 import GLOBAL, GLOBAL_AUX, PER_CLASS_MEASURED, PER_CLASS_PREDICTED  # noqa: E402
from yg_eo_soilnet.a1.tracking import reopen_parent  # noqa: E402
from yg_eo_soilnet.datamodules.splitting import TEST  # noqa: E402

#: The comparisons written to ``differences.csv``: each model against the plain global one.
COMPARISONS = [(PER_CLASS_MEASURED, GLOBAL), (PER_CLASS_PREDICTED, GLOBAL), (GLOBAL_AUX, GLOBAL)]
#: The tables written, by file name.
TABLES = (
    "scores.csv",
    "scores_by_class.csv",
    "intervals.csv",
    "intervals_by_region.csv",
    "differences.csv",
    "transfer.csv",
)
#: The held-out regions' names as they appear in metric names, beside the tables.
REGION_NAMES_FILE = "region_names.json"


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Score a run from the command line and write its tables.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict of str to pandas.DataFrame
        Each table written, by file name.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/a1/a1.yml", help="The A1 settings file (default configs/a1/a1.yml)."
    )
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000).")
    parser.add_argument(
        "--block-km", type=float, default=None, help="Side of a spatial block, km (default: bootstrap.block_size_km)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed of the resampling (default: seeds[0]).")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Write the tables only; do not add them to the run's MLflow run."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Show only warnings and the end summary (the log file keeps everything)."
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    run_dir = Path(settings.output_dir) / "runs" / args.run_name
    setup_logging(run_dir / "evaluation" / "evaluate.log", quiet=args.quiet)
    block_km = args.block_km if args.block_km is not None else settings.bootstrap.block_size_km
    seed = args.seed if args.seed is not None else settings.seed

    with contextlib.ExitStack() as stack:
        # Opened first, so a run folder whose MLflow run is gone fails before any work is done.
        tracking = None if args.no_mlflow else stack.enter_context(reopen_parent(run_dir, settings.load_config()))
        tables = _evaluate(run_dir, block_km, seed, args.n_boot, tracking)
    return tables


def _evaluate(run_dir: Path, block_km: float, seed: int, n_boot: int, tracking) -> dict[str, pd.DataFrame]:
    """Write the tables, print the headline, and add both to the open MLflow run (``tracking``)."""
    frame = evaluation.load_predictions(run_dir)
    test = frame[frame["split"] == TEST]
    combined = test[test["covariate_group"] == COMBINED_GROUP]
    boot = evaluation.block_bootstrap(
        combined, ["kind", "block_size_m", "model"], block_km=block_km, n=n_boot, seed=seed, compare=COMPARISONS
    )
    # Each held-out region on its own too, for the per-region intervals of the headline metrics.
    logo = combined[combined["kind"] == LOGO]
    by_region = (
        evaluation.block_bootstrap(
            logo, ["kind", "held_out", "model"], block_km=block_km, n=n_boot, seed=seed
        ).intervals
        if not logo.empty
        else pd.DataFrame(columns=["kind", "held_out", "model", "metric", "estimate", "low", "high"])
    )
    transfer = evaluation.transfer_table(frame)
    tables = {
        "scores.csv": evaluation.grouped_metrics(frame, ["kind", "block_size_m", "model", "covariate_group", "split"]),
        "scores_by_class.csv": evaluation.grouped_metrics(
            combined, ["kind", "block_size_m", "model", "carbonate_class_measured"]
        ),
        "intervals.csv": boot.intervals,
        "intervals_by_region.csv": by_region,
        "differences.csv": boot.differences,
        "transfer.csv": transfer,
    }
    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    regions = evaluation.region_names(frame)
    (out / REGION_NAMES_FILE).write_text(json.dumps(regions, indent=2, ensure_ascii=False) + "\n")
    headline = evaluation.headline_metrics(
        frame, intervals=boot.intervals, region_intervals=by_region, transfer=transfer
    )
    if tracking is not None:
        import mlflow

        for name in (*TABLES, REGION_NAMES_FILE):
            mlflow.log_artifact(str(out / name), "evaluation")
        mlflow.log_metrics(headline)
        mlflow.set_tags({f"region__{key}": name for key, name in regions.items()})

    rmse = boot.intervals[boot.intervals["metric"] == "rmse"]
    summary(
        f"Test RMSE, every covariate, with 95 % block-bootstrap intervals ({block_km:g} km blocks, {n_boot} draws):"
    )
    summary(
        rmse[["kind", "block_size_m", "model", "estimate", "low", "high", "n", "n_blocks"]]
        .round(3)
        .to_string(index=False)
    )
    differences = boot.differences[boot.differences["metric"] == "rmse"]
    if not differences.empty:
        summary("RMSE differences (a - b; negative means a is better):")
        summary(
            differences[["kind", "block_size_m", "a", "b", "estimate", "low", "high", "p_holm"]]
            .round(4)
            .to_string(index=False)
        )
    summary(f"Tables in {out}/: {', '.join(TABLES)}")
    if tracking is not None:
        summary(
            f"Added to MLflow run {tracking.info.run_id}: the tables (evaluation/) and {len(headline)} headline metrics."
        )
    return tables


if __name__ == "__main__":
    main()
