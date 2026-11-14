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

Run from the repository root::

    python experiments/a1/basin_relations.py --config configs/a1/a1_xgb.yml
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
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000).")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    out = Path(args.out) if args.out else Path(settings.output_dir) / "basin_relations"
    setup_logging(out / "basin_relations.log")
    tables, _ = run_basin_relations(settings, out, n=args.n_boot)
    relations = tables["basin_relations"]
    primary = relations[relations["family"] == "primary"]
    if not primary.empty:
        summary("SOM with CaCO3 and CEC, basin against the rest (primary family, Holm within each basin):")
        summary(
            primary[["basin", "variable", "rho_basin", "rho_complement", "difference", "p_holm"]]
            .round(3)
            .to_string(index=False)
        )
    summary(f"Tables, figure and caption in {out}/")
    return tables


if __name__ == "__main__":
    main()
