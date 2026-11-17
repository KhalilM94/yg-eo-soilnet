"""E5: fit Global on each input variant, on E1's plans and seeds.

One part at a time (``--part``):

``A``
    Covariate ablation: ``all`` (Global's inputs), ``all_minus_<group>`` and
    ``all_minus_<satellite sub-block>``, ``only_satellite``, ``satellite_plus_terrain``,
    ``all_plus_xy`` (projected coordinates - an interpolation check, not a transfer model).
``B``
    Time-series representations: every percentile column replaced by R1, R2, R3, R4, R6 (and R5
    when switched on), on ``all``, on ``all_minus_legacy_soil`` and (R2, R6) on ``only_satellite``.
    The point-level features are computed once from the monthly composites and cached in
    ``<output_dir>/features/``, with a report on them (``features/report_*.csv``).
``C``
    External covariates (``e5.extra_covariates``): ``all_plus_<group>`` and ``only_<group>``.

Each variant is Global with exactly its column list - same plans, inner folds, grid and seeds - so
variants are compared point by point (``evaluate_e5.py``). Its predictions go to
``<output_dir>/runs/<run name>/predictions/<plan>/seed_<s>/global__<variant>.parquet``, its column list
to ``variants/<variant>.txt``; the manifest keeps each list's hash and refuses a rerun where it
changed. A run resumes: what is written is skipped, so Part B reuses Part A's ``all``. MLflow: a
parent run ``e5_<run name>_<time>`` in the settings' experiment, one sub-run per variant, plan and
seed (tagged ``experiment=E5``, ``variant``). See :mod:`yg_eo_soilnet.a1.e5`.

Run from the repository root (after ``make_plans.py`` with the same settings)::

    python experiments/a1/run_e5.py --config configs/a1/a1_e5.yml --part A --workers 2
    python experiments/a1/run_e5.py --config configs/a1/a1_e5.yml --part B --workers 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from yg_eo_soilnet.a1 import e5  # noqa: E402
from yg_eo_soilnet.a1.config import load_a1_config  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.plans import LOGO, RANDOM, SPATIAL_KFOLD  # noqa: E402
from yg_eo_soilnet.a1.runner import LOG_FILE, RunResult, run_plans  # noqa: E402

#: E1's designs: the random split, the 10 km block folds, the held-out basins.
DEFAULT_ONLY = ("random", "kfold_10km_*", "logo_*")
#: The cache of the point-level time-series features, under the output folder.
FEATURES_DIR = "features"


def build_hook_and_variants(a1, part: str, *, wanted=None, log=None):
    """The table hook and the variant builder of one part (for :func:`~yg_eo_soilnet.a1.runner.run_plans`)."""
    import logging

    log = log or logging.getLogger(__name__)
    settings = a1.settings
    options = e5.e5_settings(settings)
    output_dir = Path(settings.output_dir)
    state: dict = {}

    def hook(table):
        new = []
        if part == "A":
            new.append(e5.projected_xy(table))
        if part == "B":
            config = settings.load_config()
            root = Path(getattr(config, "DATA_FOLDER", None) or settings.data.get("root", "."))
            if not options.timeseries_file:
                raise ValueError("Part B needs e5.timeseries_file (the monthly composites).")
            ts = e5.load_timeseries(
                root / options.timeseries_file,
                id_column=options.timeseries_id_column,
                time_column=options.timeseries_time_column,
            )
            features = e5.build_representations(table, ts, options, output_dir / FEATURES_DIR, log=log)
            regions = table.frame[table.region_column] if table.region_column else None
            for name, report in e5.feature_report(features, table, ts, regions).items():
                report.to_csv(output_dir / FEATURES_DIR / f"report_{name}.csv")
            new += list(features.values())
            state["r5"] = e5.PlanPCA(tuple(features["R5"].columns), options.r5_components) if "R5" in features else None
        if part == "C":
            extras = {}
            for spec in options.extra_covariates:
                if not spec.path:
                    log.warning(f"e5.extra_covariates {spec.group}: no path set; skipped.")
                    continue
                columns = e5.extra_covariates(spec, table, lab_columns=table.label_columns)
                extras[spec.group] = list(columns.columns)
                new.append(columns)
            state["extras"] = extras
        frames = [frame for frame in new if frame is not None and len(frame.columns)]
        if not frames:
            return table
        return e5.extend_table(table, pd.concat([frame.set_axis(frame.index.astype(str)) for frame in frames], axis=1))

    def variants(table, resolved):
        if part == "A":
            found = e5.part_a_variants(resolved)
        elif part == "B":
            reps = list(options.representations) + (["R5"] if options.r5_enabled else [])
            found = e5.part_b_variants(resolved, table.feature_columns, reps, r5=state.get("r5"))
        else:
            found = e5.part_c_variants(resolved, state.get("extras", {}))
        chosen = wanted or (options.variants if part == "A" else None)
        if chosen:
            unknown = [name for name in chosen if name not in found]
            if unknown:
                raise ValueError(f"--variants: {unknown} are not variants of part {part}: {sorted(found)}")
            found = {name: variant for name, variant in found.items() if name in chosen}
        return found

    return hook, variants


def main(argv: list[str] | None = None) -> RunResult:
    """Fit one E5 part from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    RunResult
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/a1/a1_e5.yml", help="The E5 settings file.")
    parser.add_argument("--part", choices=("A", "B", "C"), required=True, help="Which part to fit.")
    parser.add_argument("--variants", default=None, help="Comma-separated variant ids of the part (default all).")
    parser.add_argument("--plans", default=None, help="The plan folder (default <output_dir>/plans).")
    parser.add_argument(
        "--only",
        default=None,
        help='Comma-separated plan_id patterns (default "random,kfold_10km_*,logo_*": the E1 designs).',
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, among the settings' (default all).")
    parser.add_argument("--workers", type=int, default=1, help="Plans and seeds fitted side by side.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Replaces the settings' processor.")
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument("--restart", action="store_true", help="Clear a run folder made from other settings.")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not record the run in MLflow.")
    parser.add_argument("--quiet", action="store_true", help="Only warnings and the end summary on the console.")
    args = parser.parse_args(argv)

    a1 = load_a1_config(args.config)
    run_dir = Path(a1.settings.output_dir) / "runs" / args.run_name
    log_file = setup_logging(run_dir / LOG_FILE, quiet=args.quiet)
    wanted = [name.strip() for name in args.variants.split(",") if name.strip()] if args.variants else None
    only = [p.strip() for p in args.only.split(",") if p.strip()] if args.only else list(DEFAULT_ONLY)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    hook, variants = build_hook_and_variants(a1, args.part, wanted=wanted)
    result = run_plans(
        a1,
        plans_dir=args.plans,
        run_name=args.run_name,
        kinds=[RANDOM, SPATIAL_KFOLD, LOGO],
        restart=args.restart,
        tracking=not args.no_mlflow,
        seeds=seeds,
        only=only,
        workers=args.workers,
        device=args.device,
        log_file=log_file,
        table_hook=hook,
        variants=variants,
        parent_prefix="e5",
        tags={"experiment": "E5"},
    )
    lines = [
        f"E5 part {args.part}: {len(result.computed)} file(s) written, {len(result.skipped)} already done, in {result.run_dir}/"
    ]
    test = result.metrics[result.metrics["split"] == "test"] if not result.metrics.empty else result.metrics
    if not test.empty:
        lines += [
            "Mean test RMSE over plans, per variant:",
            test.groupby("covariate_group")["rmse"].mean().round(3).to_string(),
        ]
    lines.append(f"Log: {log_file}. Next: python experiments/a1/evaluate_e5.py --config {args.config}")
    summary("\n".join(lines))
    return result


if __name__ == "__main__":
    main()
