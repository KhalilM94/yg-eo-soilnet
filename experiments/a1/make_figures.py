"""Draw the Paper A1 figures from a run, as PNG and PDF, with their data and draft captions.

The figures are described in :mod:`yg_eo_soilnet.a1.figures`. They are saved in the run folder's
``figures/`` unless ``--out`` says otherwise, with the tables each was drawn from in
``figures/data/`` and a draft caption for each in ``figures/captions.md``;
``notebooks/a1_figures.ipynb`` draws the same ones interactively. ``--only`` and ``--skip`` choose
figures by file name. A figure whose plans the run does not hold - no mixed plans, say - is saved as a
short message naming what to run, and listed as such here. Unless ``--no-mlflow`` is given, each is
also added to the run folder's MLflow run under ``figures/`` - the PDF as a file, the PNG as a figure
the MLflow browser previews - with the data and the captions.

Run from the repository root, after ``run_a1.py``::

    python experiments/a1/make_figures.py --config configs/a1/a1_demo.yml \\
        --regions experiments/a1/demo_data/regions.geojson --regions-name-field name
    python experiments/a1/make_figures.py --config configs/a1/a1.yml
    python experiments/a1/make_figures.py --only fig05_spatial,fig06_obs_pred --no-mlflow
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    # The project's Config lives at the repository root, which is not on the path when this file
    # is run as a script.
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402

from yg_eo_soilnet.a1 import figures  # noqa: E402
from yg_eo_soilnet.a1.config import load_a1_config  # noqa: E402
from yg_eo_soilnet.a1.plans import RANDOM  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.tracking import log_figures, reopen_parent  # noqa: E402


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
    parser.add_argument(
        "--n-boot", type=int, default=1000, help="Bootstrap resamples for the intervals the figures compute."
    )
    parser.add_argument(
        "--shap-plan", default=RANDOM, help="The plan whose global model figures 11 and S4 explain (default random)."
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Draw only these figures, comma-separated file names (e.g. fig05_spatial,fig06_obs_pred).",
    )
    parser.add_argument("--skip", default=None, help="Do not draw these figures, comma-separated file names.")
    parser.add_argument("--no-relief", action="store_true", help="Flat ground under the map, no relief tiles.")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Save the files only; do not add them to the run's MLflow run."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Show only warnings and the end summary (the log file keeps everything)."
    )
    args = parser.parse_args(argv)
    only, skip = _names(args.only), _names(args.skip)
    try:
        figures.select_figures(only, skip)
    except ValueError as error:
        parser.error(str(error))

    a1 = load_a1_config(args.config)
    inputs = figures.FigureInputs.from_run(
        a1, args.run_name, regions_path=args.regions, regions_name_field=args.regions_name_field, n_boot=args.n_boot
    )
    folder = Path(args.out) if args.out else inputs.run_dir / "figures"
    setup_logging(folder / "make_figures.log", quiet=args.quiet)
    with contextlib.ExitStack() as stack:
        # Opened first, so a run folder whose MLflow run is gone fails before anything is drawn.
        tracking = None if args.no_mlflow else stack.enter_context(reopen_parent(inputs.run_dir, inputs.config))
        drawn = figures.render_all(
            inputs,
            folder,
            relief=not args.no_relief,
            n_boot=args.n_boot,
            shap_plan=args.shap_plan,
            close=False,
            only=only,
            skip=skip,
        )
        if tracking is not None:
            log_figures(drawn, folder)
    for figure in drawn.values():
        plt.close(figure)
    summary(f"Figures in {folder}/ (PNG and PDF; data in data/, draft captions in captions.md):")
    for name, figure in drawn.items():
        note = f"  - not drawn: {figure._a1_message}" if figures.is_message_figure(figure) else ""
        summary(f"  {name}{note}")
    if tracking is not None:
        summary(f"Added to MLflow run {tracking.info.run_id} under figures/.")
    return drawn


def _names(text):
    """A comma-separated list of figure names, or None."""
    return [name.strip() for name in text.split(",") if name.strip()] if text else None


if __name__ == "__main__":
    main()
