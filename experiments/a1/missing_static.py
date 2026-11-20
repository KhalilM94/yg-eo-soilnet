"""The points absent from the static-covariate export: errors, transfer loss and within-class relations without them.

No model is fitted: everything is read from the run's saved predictions (first seed) and points.
See :mod:`yg_eo_soilnet.a1.missing_static`. Writes into ``<output_dir>/robustness/`` unless
``--out`` is given:

- ``missing_static.csv``: per design (random, 10 km folds, each held-out basin), model (Global,
  local_global) and basin, RMSE, bias and n of the absent points and the others, and the difference
  with a 10 km block-bootstrap interval;
- ``missing_static_transfer.csv``: the transfer loss per held-out basin with every point and without
  the absent ones;
- ``missing_static_relations.csv`` and ``missing_static_relations_counts.csv``: the basin-vs-rest
  relations of ``--relations-basins`` recomputed without them, and the within-class counts before
  and after (before: ``basin_relations/basin_relations.csv``).

In MLflow (unless ``--no-mlflow``) the tables are a child run ``missing_static`` of the newest
``compare`` run of the run's experiment, like the other robustness checks.

Run from the repository root::

    python experiments/a1/missing_static.py --config configs/a1/a1_xgb.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from yg_eo_soilnet.a1 import evaluation  # noqa: E402
from yg_eo_soilnet.a1 import missing_static as ms  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    """Run the check from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict of str to pandas.DataFrame
        Each table written, by name.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="The A1 settings file.")
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument(
        "--static-file",
        default=None,
        help=f"The static-covariate export (default {ms.STATIC_EXPORT} in the data folder).",
    )
    parser.add_argument("--id-column", default="uuid", help="Its point-id column (default uuid).")
    parser.add_argument(
        "--relations-basins", default="Sebou", help='Comma-separated basins whose relations are recomputed ("Sebou").'
    )
    parser.add_argument(
        "--relations", default=None, help="basin_relations.csv (default <output_dir>/basin_relations/)."
    )
    parser.add_argument("--out", default=None, help="Where to write (default <output_dir>/robustness).")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000).")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not add the tables to MLflow.")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    config = settings.load_config()
    output_dir = Path(settings.output_dir)
    run_dir = output_dir / "runs" / args.run_name
    out = Path(args.out) if args.out else output_dir / "robustness"
    setup_logging(out / "missing_static.log")
    relations_path = Path(args.relations) if args.relations else output_dir / "basin_relations" / "basin_relations.csv"
    if not relations_path.exists():
        raise SystemExit(f"No {relations_path}: run basin_relations.py first.")
    export = ms.static_export_path(settings, config, args.static_file)
    if not export.exists():
        raise SystemExit(f"No {export}: pass --static-file.")

    from yg_eo_soilnet.a1.basin_relations import load_basin_data
    from yg_eo_soilnet.a1.compare import read_run
    from yg_eo_soilnet.a1.plans import LOGO
    from yg_eo_soilnet.a1.stage2 import GLOBAL, LOCAL_GLOBAL

    data = load_basin_data(settings, config)
    absent = ms.absent_points(data.frame.index, ms.export_ids(export, args.id_column))
    block_km = settings.bootstrap.block_size_km
    frame = evaluation.load_predictions(run_dir, first_seed=settings.seed)
    errors = ms.missing_static_table(
        frame, absent, first_seed=settings.seed, block_km=block_km, n=args.n_boot, seed=settings.seed
    )
    run = read_run(run_dir, model=GLOBAL, kinds=[LOGO], local=LOCAL_GLOBAL)
    transfer = ms.transfer_without(run, absent, block_km=block_km, n=args.n_boot, seed=settings.seed)
    basins = [b.strip() for b in args.relations_basins.split(",") if b.strip()]
    unknown = [b for b in basins if b not in data.basins]
    if unknown:
        raise SystemExit(f"--relations-basins: {unknown} are not held-out basins: {sorted(data.basins)}")
    before = pd.read_csv(relations_path)
    after = ms.relations_without(
        data, absent, basins=basins, min_cell_size=settings.min_cell_size, n=args.n_boot, seed=settings.seed
    )
    counts = ms.within_class_counts(before, after)

    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "missing_static": errors,
        "missing_static_transfer": transfer,
        "missing_static_relations": after,
        "missing_static_relations_counts": counts,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    if not args.no_mlflow:
        from yg_eo_soilnet.a1 import robustness

        robustness.log_robustness(run_dir, out, step="missing_static")
    summary(f"{len(absent)} of {len(data.frame)} points absent from {export.name}.")
    shown = ["design", "model", "basin", "n_absent", "n_other", "rmse_difference", "rmse_low", "rmse_high"]
    summary("RMSE absent - others (first seed, 95 % 10 km block bootstrap):")
    summary(errors[shown + ["bias_difference", "bias_low", "bias_high"]].round(2).to_string(index=False))
    summary("Transfer loss (Global - local_global RMSE, same points):")
    summary(
        transfer[["points", "basin", "difference", "low", "high", "p_holm", "n", "transfer_fails"]]
        .round(3)
        .to_string(index=False)
    )
    summary("Within-class Holm-significant differences:")
    summary(counts.to_string(index=False))
    summary(f"Tables in {out}/")
    return tables


if __name__ == "__main__":
    main()
