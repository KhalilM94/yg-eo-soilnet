"""Run the Paper A1 experiments on the saved split plans.

For each plan of the plan index (``make_plans.py`` writes it), stage 1 predicts CaCO3 and CEC
without ever seeing a point's own value, and the stage-2 soil organic matter models are fitted on
the training points and scored on the validation and test points:

``global``, ``global_aux``
    Every covariate, without and with the predicted CaCO3 and CEC.
``per_class_measured``
    One model per FAO carbonate class, by MEASURED CaCO3 - a diagnostic upper bound, since a new
    point has no measured CaCO3.
``per_class_predicted``
    One model per class, by PREDICTED CaCO3.

``--groups`` also fits ``global_aux`` on each covariate group alone, and for leave-one-region-out
plans a ``local`` model inside the held-out region. Per-point predictions go to
``<output_dir>/runs/<run name>/predictions/``, scores to ``metrics.csv`` there and to MLflow. For
leave-one-region-out plans, how far each held-out region lies from its training data (dissimilarity
index, area of applicability, MESS, carbonate mix, SOM correlations) goes to ``dissimilarity/`` and
``dissimilarity_summary.csv``. A run resumes: what is already written is skipped. See :mod:`yg_eo_soilnet.a1.runner`.

Run from the repository root::

    python experiments/a1/make_plans.py --config configs/a1/a1_demo.yml
    python experiments/a1/run_a1.py --config configs/a1/a1_demo.yml --kinds random,logo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    # The project's Config lives at the repository root, which is not on the path when this file
    # is run as a script.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yg_eo_soilnet.a1.config import load_a1_config  # noqa: E402
from yg_eo_soilnet.a1.runner import RunResult, run_plans  # noqa: E402


def main(argv: list[str] | None = None) -> RunResult:
    """Run the A1 experiments from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    RunResult
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/a1/a1.yml", help="The A1 settings file (default configs/a1/a1.yml)."
    )
    parser.add_argument("--plans", default=None, help="The plan folder (default <output_dir>/plans).")
    parser.add_argument("--kinds", default=None, help="Comma-separated plan kinds to run (default all).")
    parser.add_argument("--groups", action="store_true", help="Also fit one model per covariate group.")
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Clear a run folder made from other settings or plans, instead of refusing to add to it.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Do not record the run in MLflow.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()] if args.kinds else None
    result = run_plans(
        load_a1_config(args.config),
        plans_dir=args.plans,
        run_name=args.run_name,
        kinds=kinds,
        groups=args.groups,
        restart=args.restart,
        tracking=not args.no_mlflow,
    )
    print(f"{len(result.computed)} file(s) written, {len(result.skipped)} already done, in {result.run_dir}/")
    for note in result.notes:
        print(f"  {note}")
    test = result.metrics[result.metrics["split"] == "test"] if not result.metrics.empty else result.metrics
    if not test.empty:
        summary = test.groupby(["model", "covariate_group"])["rmse"].mean().round(3)
        print("Mean test RMSE over plans:")
        print(summary.to_string())
    if not result.dissimilarity.empty:
        columns = [c for c in ("held_out", "median_di", "share_outside", "carbonate_tvd") if c in result.dissimilarity]
        print(f"Held-out regions against their training data (dissimilarity_summary.csv in {result.run_dir}/):")
        print(result.dissimilarity[columns].round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
