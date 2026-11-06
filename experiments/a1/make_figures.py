"""Draw the eight Paper A1 figures from a run, as PNG and PDF.

The figures are described in :mod:`yg_eo_soilnet.a1.figures`. They are saved in the run folder's
``figures/`` unless ``--out`` says otherwise; ``notebooks/a1_figures.ipynb`` draws the same ones
interactively. A figure whose plans the run does not hold - no mixed plans, say - is saved as a
short message naming what to run, and listed as such here.

Run from the repository root, after ``run_a1.py``::

    python experiments/a1/make_figures.py --config configs/a1/a1_demo.yml \\
        --regions experiments/a1/demo_data/regions.geojson --regions-name-field name
    python experiments/a1/make_figures.py --config configs/a1/a1.yml
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # files only, no window

from yg_eo_soilnet.a1 import figures  # noqa: E402
from yg_eo_soilnet.a1.config import load_a1_config  # noqa: E402
from yg_eo_soilnet.a1.plans import RANDOM  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    """Draw and save the figures from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict
        File name to figure (closed once saved).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/a1/a1.yml", help="The A1 settings file (default configs/a1/a1.yml)."
    )
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument("--out", default=None, help="Where to save (default <run folder>/figures).")
    parser.add_argument(
        "--regions",
        default=None,
        help="A polygon file of the regions, for the map's outlines (default: the settings' own polygons).",
    )
    parser.add_argument("--regions-name-field", default=None, help="The polygon file's region-name column.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for figures 2 and 4.")
    parser.add_argument("--shap-plan", default=RANDOM, help="The plan figures 3 and 8 read (default random).")
    parser.add_argument("--no-relief", action="store_true", help="Flat ground under the map, no relief tiles.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a1 = load_a1_config(args.config)
    inputs = figures.FigureInputs.from_run(
        a1, args.run_name, regions_path=args.regions, regions_name_field=args.regions_name_field
    )
    folder = Path(args.out) if args.out else inputs.run_dir / "figures"
    drawn = figures.render_all(
        inputs, folder, relief=not args.no_relief, n_boot=args.n_boot, shap_plan=args.shap_plan, close=True
    )
    print(f"Figures in {folder}/ (PNG and PDF):")
    for name, figure in drawn.items():
        note = f"  - not drawn: {figure._a1_message}" if figures.is_message_figure(figure) else ""
        print(f"  {name}{note}")
    return drawn


if __name__ == "__main__":
    main()
