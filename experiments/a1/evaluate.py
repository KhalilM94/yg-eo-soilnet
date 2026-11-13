"""Score a Paper A1 run: tables of scores, block-bootstrap intervals and transfer losses.

Reads only the per-point prediction files ``run_a1.py`` wrote (see
:mod:`yg_eo_soilnet.a1.evaluation`) and writes, into ``<output_dir>/runs/<run name>/evaluation/``:

``scores.csv``
    RMSE, MAE, R², bias, RPD and RPIQ per plan kind, model, covariate group and split, for the
    kinds that can be pooled - ``mixed`` and ``control`` plans are scored per region and share in
    ``scores_mixed.csv`` instead, since pooled they would count one point many times. Every table
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
    ``global``, scored on the same points, with p-values and their Holm correction; and, in rows of
    kind ``mixed_vs_control``, each ``mixed`` plan minus its ``control`` plan per region, share and
    model, paired on the test points they share, Holm-corrected across the shares.
``scores_mixed.csv``
    The ``mixed`` and ``control`` scores per region, share, model and covariate group, each repeat
    scored on its own test points and the repeats averaged.
``intervals_by_region.csv``
    The same intervals for each held-out region of the ``logo`` plans.
``transfer.csv``
    For ``logo`` and ``mixed`` plans, the local against the transferred score per region and
    covariate group, and the drop between them.
``seed_summary.csv``
    For a run of several seeds: each headline metric's value on the first seed and its mean, SD,
    min and max over the seeds.

Every table has a ``seed`` column: one block of rows per seed the run has. The primary result is
the first seed's, with its block-bootstrap intervals; the other seeds repeat the model fits on the
same plans, and ``seed_summary.csv`` says how far the scores move between them.
``differences.csv`` also gives, for each comparison, the number of seeds whose difference has the
sign of the first seed's (``n_seeds_same_sign`` of ``n_seeds``).

Unless ``--no-mlflow`` is given, the tables, ``region_names.json`` and the headline metrics
(:func:`yg_eo_soilnet.a1.evaluation.headline_metrics`, of the first seed; with several seeds also
``<metric>__..._sd``, their SD over the seeds) are also added to the run folder's MLflow
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
    "scores_mixed.csv",
    "seed_summary.csv",
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
        tables = _evaluate(run_dir, block_km, seed, args.n_boot, tracking, first_seed=settings.seed)
    return tables


def _evaluate(
    run_dir: Path, block_km: float, seed: int, n_boot: int, tracking, *, first_seed: int | None = None
) -> dict[str, pd.DataFrame]:
    """Write the tables, print the headline, and add both to the open MLflow run (``tracking``).

    ``seed`` seeds the resampling; ``first_seed`` is the run's first seed, the primary result.
    """
    everything = evaluation.load_predictions(run_dir, first_seed=first_seed)
    seeds = evaluation.seed_order(everything, first_seed)
    per_seed = {
        fit_seed: _seed_tables(everything[everything["seed"] == fit_seed], block_km, seed, n_boot) for fit_seed in seeds
    }
    tables = {
        name: _stack([per_seed[fit_seed][name].assign(seed=fit_seed) for fit_seed in seeds])
        for name in TABLES
        if name != "seed_summary.csv"
    }
    tables["differences.csv"] = evaluation.same_sign_counts(tables["differences.csv"])
    headlines = pd.concat(
        [per_seed[fit_seed]["headline"].assign(seed=fit_seed) for fit_seed in seeds], ignore_index=True
    )
    tables["seed_summary.csv"] = evaluation.seed_summary(headlines)

    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    regions = evaluation.region_names(everything)
    (out / REGION_NAMES_FILE).write_text(json.dumps(regions, indent=2, ensure_ascii=False) + "\n")
    primary = per_seed[seeds[0]]
    headline = dict(zip(primary["headline"]["name"], primary["headline"]["value"]))
    if len(seeds) > 1:
        spread = tables["seed_summary.csv"].dropna(subset=["sd"])
        headline.update({f"{name}_sd": float(sd) for name, sd in zip(spread["name"], spread["sd"])})
    if tracking is not None:
        import mlflow

        for name in (*TABLES, REGION_NAMES_FILE):
            mlflow.log_artifact(str(out / name), "evaluation")
        mlflow.log_metrics(headline)
        mlflow.set_tags({f"region__{key}": name for key, name in regions.items()})
    _print_summary(primary, tables, seeds, block_km, n_boot, out, tracking, len(headline))
    return tables


def _stack(parts: list) -> pd.DataFrame:
    """One table from every seed's, ``seed`` first; an empty table keeps its columns."""
    kept = [part for part in parts if not part.empty]
    stacked = pd.concat(kept, ignore_index=True) if kept else parts[0]
    return stacked[["seed", *[column for column in stacked.columns if column != "seed"]]]


def _seed_tables(everything: pd.DataFrame, block_km: float, seed: int, n_boot: int) -> dict[str, pd.DataFrame]:
    """Every table of one seed's predictions, and its headline metrics (``headline``, a table)."""
    # Mixed and control plans are scored per region and share (below); pooled they mean nothing.
    frame = evaluation.pooled_rows(everything)
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
    transfer = evaluation.transfer_table(everything)
    mixed = evaluation.mixed_scores(everything)
    paired = evaluation.mixed_differences(everything, block_km=block_km, n=n_boot, seed=seed)
    # A run of one model (stage2.models: [global]) has neither: the table keeps its columns, empty.
    kept = [table for table in (boot.differences, paired) if not table.empty]
    differences = pd.concat(kept, ignore_index=True) if kept else boot.differences
    return {
        "scores.csv": evaluation.grouped_metrics(frame, ["kind", "block_size_m", "model", "covariate_group", "split"]),
        "scores_by_class.csv": evaluation.grouped_metrics(
            combined, ["kind", "block_size_m", "model", "carbonate_class_measured"]
        ),
        "intervals.csv": boot.intervals,
        "intervals_by_region.csv": by_region,
        "differences.csv": differences,
        "transfer.csv": transfer,
        "scores_mixed.csv": mixed,
        "headline": evaluation.headline_table(
            frame, intervals=boot.intervals, region_intervals=by_region, transfer=transfer, mixed=mixed
        ),
        "_model_differences": boot.differences,
        "_paired": paired,
    }


def _print_summary(primary, tables, seeds, block_km, n_boot, out, tracking, n_headline) -> None:
    """The end summary: the first seed's scores and differences, and the spread over the seeds."""
    intervals = primary["intervals.csv"]
    rmse = intervals[intervals["metric"] == "rmse"]
    summary(
        f"Test RMSE, every covariate, with 95 % block-bootstrap intervals ({block_km:g} km blocks, {n_boot} draws)"
        + (f", seed {seeds[0]} of {len(seeds)}:" if len(seeds) > 1 else ":")
    )
    summary(
        rmse[["kind", "block_size_m", "model", "estimate", "low", "high", "n", "n_blocks"]]
        .round(3)
        .to_string(index=False)
    )
    differences = primary["_model_differences"]
    differences = differences[differences["metric"] == "rmse"]
    if not differences.empty:
        summary("RMSE differences (a - b; negative means a is better):")
        summary(
            differences[["kind", "block_size_m", "a", "b", "estimate", "low", "high", "p_holm"]]
            .round(4)
            .to_string(index=False)
        )
    paired = primary["_paired"]
    paired_rmse = paired[(paired["metric"] == "rmse") & (paired["model"] == GLOBAL)] if not paired.empty else paired
    if not paired_rmse.empty:
        summary("Mixed minus control, global model, RMSE (negative: the other basins help):")
        summary(
            paired_rmse[["region", "fraction", "estimate", "low", "high", "p_holm", "n"]]
            .round(4)
            .to_string(index=False)
        )
    if len(seeds) > 1:
        spread = tables["seed_summary.csv"]
        spread = spread[spread["name"].str.startswith("rmse__") & (spread["covariate_group"] == "")]
        summary(f"Test RMSE over the {len(seeds)} seeds {seeds} (mean, SD, min, max):")
        summary(spread[["scope", "model", "first", "mean", "sd", "min", "max"]].round(3).to_string(index=False))
    summary(f"Tables in {out}/: {', '.join(TABLES)}")
    if tracking is not None:
        summary(
            f"Added to MLflow run {tracking.info.run_id}: the tables (evaluation/) and {n_headline} headline metrics."
        )


if __name__ == "__main__":
    main()
