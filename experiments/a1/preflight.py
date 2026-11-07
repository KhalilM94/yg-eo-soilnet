"""Check that the data holds up the Paper A1 experiments, before running any of them.

Reads the data named by an A1 settings file and writes, to one folder, a markdown report
(``preflight.md``) and the tables behind it as CSVs:

1. how many points have SOM, CaCO3 and CEC all measured, and how empty every column is;
2. points per region and FAO carbonate class, with thin cells flagged;
3. points per region and sampling year;
4. with a split plan, how earlier predictions of CaCO3 and CEC score on the training, validation
   and test points - flagged when far better on training than on test, the mark of a leak.

See :mod:`yg_eo_soilnet.a1.preflight` for what each table means. Run from the repository root::

    python experiments/a1/preflight.py --config configs/a1/a1_demo.yml \\
        --split-plan experiments/a1/demo_data/split_assignments.parquet
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

from yg_eo_soilnet.a1.preflight import PreflightResult, run_preflight  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> PreflightResult:
    """Run the pre-flight diagnostics from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    PreflightResult
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/a1/a1.yml", help="The A1 settings file (default configs/a1/a1.yml)."
    )
    parser.add_argument(
        "--split-plan",
        default=None,
        help="A saved split (data_splits/split_assignments.parquet), to score the earlier predictions on.",
    )
    parser.add_argument("--out", default=None, help="Output folder (default <output_dir>/preflight).")
    parser.add_argument(
        "--min-cell-points", type=int, default=None, help="Flag region x class cells under this many points."
    )
    parser.add_argument(
        "--overfit-r2-gap", type=float, default=None, help="Flag a training R2 this far above the test R2."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_preflight(
        load_settings(args.config),
        split_plan_path=args.split_plan,
        out_dir=args.out,
        min_cell_points=args.min_cell_points,
        overfit_r2_gap=args.overfit_r2_gap,
    )
    print(f"Wrote {len(result.files)} file(s) to {result.out_dir}/ - read {result.files[-1].name} first.")
    for section, reason in result.skipped.items():
        print(f"  skipped {section}: {reason}")
    if not result.small_cells.empty:
        print(f"  {len(result.small_cells)} region x carbonate cell(s) are under the threshold.")
    if result.unassigned:
        print(f"  {len(result.unassigned)} input(s) in no covariate group (see {result.out_dir}/inputs.csv).")
    if result.overfit:
        print(f"  earlier predictions flagged as leaky: {', '.join(result.overfit)}")
    return result


if __name__ == "__main__":
    main()
