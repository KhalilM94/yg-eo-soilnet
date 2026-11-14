"""Check that two plan folders hold the same plans: the same files, the same test points.

For every plan id written in both folders, the two files are compared byte for byte (SHA-256) and
their test points as sets. A table is printed, one row per plan, and the script exits with code 1
if any shared plan differs - so a configuration meant to draw the same plans as another (the mixed
arms beside the plain mixed run) can be checked before its scores are compared plan by plan. Plans
in one folder only (the arms, say) are counted, not compared.

Run from the repository root::

    python experiments/a1/check_plans.py results/a1_mixed/plans results/a1_mixed_arms/plans
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yg_eo_soilnet.a1.plans import compare_plan_folders  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Compare two plan folders from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    int
        0 when every shared plan is the same in both folders; 1 otherwise, or when they share none.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans_a", help="A plan folder (<output_dir>/plans).")
    parser.add_argument("plans_b", help="The plan folder to compare it with.")
    parser.add_argument("--all", action="store_true", help="List every shared plan, not only those that differ.")
    args = parser.parse_args(argv)

    table, only_a, only_b = compare_plan_folders(args.plans_a, args.plans_b)
    differ = table[~(table["same_file"] & table["same_test_points"])]
    shown = table if args.all else differ
    if not shown.empty:
        print(shown.to_string(index=False))
    print(
        f"{len(table)} plan(s) in both folders: {len(table) - len(differ)} the same, {len(differ)} different; "
        f"{len(only_a)} only in {args.plans_a}, {len(only_b)} only in {args.plans_b}."
    )
    if table.empty:
        print("The two folders share no plan id: nothing was compared.")
        return 1
    if not differ.empty:
        print(f"DIFFERENT: {', '.join(differ['plan_id'].head(10))}{' ...' if len(differ) > 10 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
