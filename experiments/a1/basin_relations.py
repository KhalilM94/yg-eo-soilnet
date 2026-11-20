"""How SOM relates to CaCO3, CEC and the covariates in each held-out basin and in the rest of the data.

No model is fitted. For each held-out basin and its complement (every other point: the training
data when that basin is held out), the Spearman correlation of SOM with CaCO3, CEC and the 15
covariates of figure 3, overall and within each measured carbonate class with at least
``min_cell_size`` points; 95 % block-bootstrap intervals; the basin - complement difference with its
interval and a Holm-corrected p-value within each basin, in two families (``family``): primary
(CaCO3 and CEC, overall) and exploratory (everything else). Writes, into
``<output_dir>/basin_relations/`` unless ``--out`` is given: ``basin_relations.csv``,
``basin_som_by_class.csv`` (and ``.tex``), ``figC3_basin_relations`` (PNG and PDF) and ``captions.md``.
See :mod:`yg_eo_soilnet.a1.basin_relations`.

With ``--subsample``, the power check instead (:func:`yg_eo_soilnet.a1.robustness.subsample_relations`):
each basin's classes cut to ``--n-sub`` points by whole blocks, ``--n-draws`` times, the
within-class tests repeated (``--n-boot`` defaults to 300 there), Holm within the basin over the
same 83-test family. ``--p-method`` (both modes) picks how each p-value is computed: ``normal``
(default; Fisher's z over the bootstrap standard error) or ``draws`` (the share of draws on either
side of 0, which cannot go below 2/n). It reads the ``basin_relations.csv`` already written, leaves it as it is, and
writes ``basin_relations_subsample.csv`` (the summary) and ``basin_relations_subsample_draws.csv``
beside it, and figure C4 (with its data and caption) into ``<output_dir>/robustness/``.

Run from the repository root::

    python experiments/a1/basin_relations.py --config configs/a1/a1_xgb.yml
    python experiments/a1/basin_relations.py --config configs/a1/a1_xgb.yml --subsample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yg_eo_soilnet.a1.basin_relations import run_basin_relations  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    """Compute the basin relations from the command line.

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
    parser.add_argument("--out", default=None, help="Where to write (default <output_dir>/basin_relations).")
    parser.add_argument(
        "--n-boot", type=int, default=None, help="Bootstrap resamples (default 1000; 300 with --subsample)."
    )
    parser.add_argument("--subsample", action="store_true", help="Run the power check (figure C4) instead.")
    parser.add_argument(
        "--p-method",
        choices=("normal", "draws"),
        default="normal",
        help="How each difference's p-value is computed: normal (Fisher's z / bootstrap SE, the default) or "
        "draws (share of draws on either side of 0; never below 2/n).",
    )
    parser.add_argument("--n-sub", default="240,55", help="Subsample sizes per class (default 240,55).")
    parser.add_argument("--n-draws", type=int, default=100, help="Subsamples per size (default 100).")
    parser.add_argument("--run-name", default="default", help="The run folder whose failing basins C4 labels.")
    parser.add_argument(
        "--local-model",
        choices=("local_global", "local"),
        default="local_global",
        help="The local model that decides which basins fail (default local_global; local when it is missing).",
    )
    parser.add_argument(
        "--robustness-out", default=None, help="Where figure C4 goes (default <output_dir>/robustness)."
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Do not add figure C4 to MLflow.")
    parser.add_argument(
        "--compare",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Compare basin_relations.csv tables already written (repeat; the first is the reference): writes "
        "p_method_comparison.csv and p_method_counts.csv into --out, computing nothing.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    out = Path(args.out) if args.out else Path(settings.output_dir) / "basin_relations"
    if args.compare:
        return _compare(out, args.compare)
    if args.subsample:
        return _subsample(settings, out, args)
    setup_logging(out / "basin_relations.log")
    tables, _ = run_basin_relations(settings, out, n=args.n_boot or 1000, p_method=args.p_method)
    relations = tables["basin_relations"]
    primary = relations[relations["family"] == "primary"]
    if not primary.empty:
        summary("SOM with CaCO3 and CEC, basin against the rest (primary family, Holm within each basin):")
        summary(
            primary[["basin", "variable", "rho_basin", "rho_complement", "difference", "p_holm"]]
            .round(3)
            .to_string(index=False)
        )
    exploratory = relations[(relations["class"] != "all") & (relations["p_holm"] < 0.05)]
    summary(
        f"p-values: {args.p_method}. Within-class differences with p_Holm < 0.05: "
        + (", ".join(f"{b} {n}" for b, n in exploratory.groupby("basin").size().items()) or "none")
    )
    summary(f"Tables, figure and caption in {out}/")
    return tables


def _compare(out: Path, specs: list) -> dict:
    """Line up the Holm decisions of several basin_relations.csv tables."""
    import pandas as pd

    from yg_eo_soilnet.a1.basin_relations import method_comparison, significant_counts

    tables = {}
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not Path(path).exists():
            raise SystemExit(f"--compare {spec!r}: expected LABEL=PATH to an existing basin_relations.csv")
        tables[label.strip()] = pd.read_csv(path)
    setup_logging(out / "p_method_comparison.log")
    comparison = method_comparison(tables)
    counts = significant_counts(comparison)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "p_method_comparison.csv", index=False)
    counts.to_csv(out / "p_method_counts.csv", index=False)
    summary("Within-class Holm-significant differences per basin:")
    summary(counts.to_string(index=False))
    differ = comparison[comparison["differs"]]
    summary(f"{len(differ)} test(s) whose Holm decision differs:")
    shown = ["basin", "class", "variable", "difference", *[c for c in comparison.columns if c.startswith("p_holm [")]]
    summary(differ[shown].round(4).to_string(index=False) if not differ.empty else "none")
    return {"p_method_comparison": comparison, "p_method_counts": counts}


def _subsample(settings, out: Path, args) -> dict:
    """The power check: subsample tables beside basin_relations.csv, figure C4 in the robustness folder."""
    import pandas as pd

    from yg_eo_soilnet.a1 import robustness
    from yg_eo_soilnet.a1.basin_relations import load_basin_data
    from yg_eo_soilnet.a1.figures.robustness import figure_within_class

    path = out / "basin_relations.csv"
    if not path.exists():
        raise SystemExit(f"No {path}: run basin_relations.py without --subsample first.")
    robust = Path(args.robustness_out) if args.robustness_out else Path(settings.output_dir) / "robustness"
    setup_logging(robust / "basin_relations_subsample.log")
    relations = pd.read_csv(path)
    data = load_basin_data(settings)
    sizes = [int(size) for size in args.n_sub.split(",") if size.strip()]
    n_boot = args.n_boot or 300
    found = [
        robustness.subsample_relations(
            data,
            relations,
            n_sub=size,
            n_draws=args.n_draws,
            n_boot=n_boot,
            seed=settings.seed,
            min_cell_size=settings.min_cell_size,
            p_method=args.p_method,
        )
        for size in sizes
    ]
    # A size no class reaches gives no rows (its basins are "not testable").
    found = [frame for frame in found if not frame.empty]
    draws = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=list(robustness.SUBSAMPLE_COLUMNS))
    table = robustness.subsample_summary(draws, list(data.basins), sizes)
    table.to_csv(out / "basin_relations_subsample.csv", index=False)
    draws.to_csv(out / "basin_relations_subsample_draws.csv", index=False)

    run_dir = Path(settings.output_dir) / "runs" / args.run_name
    failing = []
    if run_dir.exists():
        local_model = robustness.choose_local_model(robustness.models_in_run(run_dir), args.local_model)
        failing = robustness.failing_basins(
            run_dir, local_model, block_km=settings.bootstrap.block_size_km, seed=settings.seed
        )
    figure, caption = figure_within_class(
        relations,
        table,
        primary=data.primary,
        labels=data.labels,
        failing=failing,
        sizes=sizes,
        n_draws=args.n_draws,
        block_km=settings.bootstrap.block_size_km,
    )
    robustness.save_figures({"figC4_within_class": (figure, caption, {"figC4_within_class": table})}, robust)
    if not args.no_mlflow and run_dir.exists():
        robustness.log_robustness(run_dir, robust, step="power_check")
    basins = table[table["level"] == "basin"]
    summary("Power check: share of draws with at least one significant within-class difference")
    summary(basins[["basin", "n_sub", "testable", "mean_significant", "share_any"]].round(3).to_string(index=False))
    summary(f"Tables in {out}/, figure C4 in {robust}/")
    return {"basin_relations_subsample": table, "basin_relations_subsample_draws": draws}


if __name__ == "__main__":
    main()
