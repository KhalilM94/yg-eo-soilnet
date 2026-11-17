"""Do E1's transfer errors follow the basins' differences, cell by cell? (No model is fitted.)

One cell per held-out basin and measured carbonate class (20 cells). From the E1 leave-one-basin-out
predictions of Global and of the local model, both scored on the same points, per seed and as the
mean over the seeds: the transferred bias, RMSE and debiased RMSE sqrt(RMSE² - bias²), the local
RMSE and the loss (transferred - local). Beside them: the level shift (the cell's median SOM minus
that of the same class in every other point), the relationship change (from ``basin_relations.csv``:
mean |rho difference| over the 15 covariates, and the number significant, without and with CaCO3 and
CEC) and the input dissimilarity (median dissimilarity index and share outside the area of
applicability, from the run's ``dissimilarity/``). Then the Spearman correlation across the cells
of bias ~ level shift, loss ~ relationship change, loss ~ input dissimilarity and (debiased RMSE -
local RMSE) ~ relationship change, with an interval from resampling the 5 basins - exploratory.

Writes into ``<output_dir>/robustness/``: ``basin_cells.csv``, ``basin_cells_correlations.csv``,
figures C5 (SOM by class, basin against the rest) and C7 (the cells), their data and captions. See
:mod:`yg_eo_soilnet.a1.robustness`.

``--local-model`` (default ``local_global``, the local model on Global's inputs) names the local
model; a run without it falls back to ``local`` (with predicted CaCO3/CEC), with a warning, and
every table and caption says which was used.

Run from the repository root, after ``run_a1.py`` and ``basin_relations.py``::

    python experiments/a1/basin_cells.py --config configs/a1/a1_xgb.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from yg_eo_soilnet.a1 import evaluation, robustness  # noqa: E402
from yg_eo_soilnet.a1.basin_relations import load_basin_data  # noqa: E402
from yg_eo_soilnet.a1.figures.robustness import figure_cells, figure_som_profiles  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402
from yg_eo_soilnet.a1.stage2 import GLOBAL  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    """Build the cell table and its correlations from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``basin_cells`` and ``basin_cells_correlations``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="The A1 settings file.")
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument(
        "--local-model",
        choices=robustness.LOCAL_MODEL_CHOICES,
        default=robustness.LOCAL_MODEL_CHOICES[0],
        help="The local model compared with (default local_global; local, with a warning, when it is missing).",
    )
    parser.add_argument(
        "--relations", default=None, help="basin_relations.csv (default <output_dir>/basin_relations/)."
    )
    parser.add_argument("--out", default=None, help="Where to write (default <output_dir>/robustness).")
    parser.add_argument("--n-boot", type=int, default=1000, help="Resamples of the basins (default 1000).")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not add the outputs to MLflow.")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    output_dir = Path(settings.output_dir)
    run_dir = output_dir / "runs" / args.run_name
    out = Path(args.out) if args.out else output_dir / "robustness"
    setup_logging(out / "basin_cells.log")
    relations_path = Path(args.relations) if args.relations else output_dir / "basin_relations" / "basin_relations.csv"
    if not relations_path.exists():
        raise SystemExit(f"No {relations_path}: run basin_relations.py first.")

    local_model = robustness.choose_local_model(robustness.models_in_run(run_dir), args.local_model)
    frame = evaluation.load_predictions(run_dir, first_seed=settings.seed)
    pairs = robustness.paired(robustness.logo_test_rows(frame, [GLOBAL, local_model]), local_model)
    data = load_basin_data(settings)
    relations = pd.read_csv(relations_path)
    paths = sorted((run_dir / "dissimilarity").glob("*.parquet"))
    dissimilarity = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True) if paths else None
    dissimilar = (
        robustness.cell_dissimilarity(dissimilarity)
        if dissimilarity is not None
        else pd.DataFrame(columns=["basin", "class", "median_di", "share_outside"])
    )
    cells = robustness.cell_table(
        pairs,
        robustness.level_shifts(data)[["basin", "class", "level_shift", "som_median_basin", "som_median_complement"]],
        robustness.relationship_changes(relations, data.primary),
        dissimilar,
        local_model=local_model,
        first_seed=settings.seed,
    )
    correlations = robustness.cell_correlations(cells, n_draws=args.n_boot, seed=settings.seed)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "basin_cells.csv", index=False)
    correlations.to_csv(out / "basin_cells_correlations.csv", index=False)

    failing = robustness.failing_basins(
        run_dir, local_model, block_km=settings.bootstrap.block_size_km, seed=settings.seed
    )
    profiles = robustness.som_profiles(data)
    first = str(settings.seed) if str(settings.seed) in set(cells["seed"]) else str(cells["seed"].iloc[0])
    c5, c5_caption = figure_som_profiles(profiles, labels=data.labels, failing=failing)
    c7, c7_caption = figure_cells(cells, correlations, labels=data.labels, local_model=local_model, scope=first)
    robustness.save_figures(
        {
            "figC5_som_profiles": (c5, c5_caption, {"figC5_som_profiles": profiles}),
            "figC7_cells": (c7, c7_caption, {"figC7_cells": cells[cells["seed"] == first]}),
        },
        out,
    )
    if not args.no_mlflow:
        robustness.log_robustness(run_dir, out, step="basin_cells")
    summary(f"Local model: {local_model} ({robustness.local_text(local_model)}); failing basins: {failing or 'none'}")
    summary(
        "Spearman rho across the cells (95 % interval from resampling the basins; exploratory, 20 cells, 5 basins):"
    )
    summary(correlations[["seed", "y", "x", "rho", "low", "high", "share_skipped"]].round(3).to_string(index=False))
    summary(f"Tables and figures C5, C7 in {out}/")
    return {"basin_cells": cells, "basin_cells_correlations": correlations}


if __name__ == "__main__":
    main()
