"""Write every split the Paper A1 experiments are trained and scored on.

Each plan is saved as ``<plan_id>.parquet`` in the form of a run's
``data_splits/split_assignments.parquet`` (columns ``point_id``, ``split``, ``cluster``,
``eligible_sklearn``), and ``index.csv`` lists them all: ``plan_id, kind, held_out, fraction,
repeat, n_train, n_val, n_test``. The kinds - ``random``, ``spatial_kfold``, ``logo``, ``mixed`` and
its ``control`` plans - are described in :mod:`yg_eo_soilnet.a1.plans`, and their settings are the
``plans:`` section of the A1 settings file. Every plan is checked before anything is written.

Run from the repository root::

    python experiments/a1/make_plans.py --config configs/a1/a1_demo.yml
    python experiments/a1/make_plans.py --config configs/a1/a1.yml --kinds random,logo
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

from yg_eo_soilnet.a1.plans import KINDS, PlanSet, write_plans  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402


def main(argv: list[str] | None = None) -> PlanSet:
    """Write the split plans from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    PlanSet
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/a1/a1.yml", help="The A1 settings file (default configs/a1/a1.yml)."
    )
    parser.add_argument("--out", default=None, help="Output folder (default <output_dir>/plans).")
    parser.add_argument(
        "--kinds",
        default=",".join(KINDS),
        help=f"Comma-separated kinds to write (default all: {','.join(KINDS)}); mixed brings its controls.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    result = write_plans(load_settings(args.config), args.out, kinds=kinds)

    print(f"Wrote {len(result.index)} plan(s) and {result.out_dir / 'index.csv'}:")
    for kind, count in result.index["kind"].value_counts().items():
        print(f"  {kind}: {count}")
    for kind, reason in result.skipped.items():
        print(f"  skipped {kind}: {reason}")
    if result.too_small:
        listed = ", ".join(f"{name} ({n})" for name, n in sorted(result.too_small.items(), key=lambda item: item[1]))
        print(f"  never held out (too few points): {listed}")
    return result


if __name__ == "__main__":
    main()
