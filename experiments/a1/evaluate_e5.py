"""E5: score every variant and compare each with its reference, on every design.

Reads the E5 run folder (``run_e5.py``) and writes into ``<output_dir>`` (``--out``):

``scores.csv``
    RMSE, MAE, R², bias, RPD, RPIQ per seed, design and ``variant`` (E1's schema plus ``variant``).
``ablation.csv``
    Part A (and C): each variant against ``all`` - ΔRMSE, Δbias, ΔR² (variant - reference) on the
    random split, the 10 km folds pooled, the held-out basins pooled and each basin; the first
    seed's estimate with a 95 % paired 10 km block-bootstrap interval and p-value, Holm within each
    part and design (``p_holm``); every seed's mean (``mean_over_seeds``) and how many seeds share
    the first seed's sign.
``ts_representation.csv``
    Part B, the same: each representation against R0 on every input (``all``), without the
    soil-map layers (``all_minus_legacy_soil``) and on satellite alone.
``figures/``
    figE5a_ablation, figE5b_ablation_basins, figE5c_ts_representation (PNG, PDF), their data and
    ``captions.md``.

Unless ``--no-mlflow``, a sub-run ``e5_summary`` of the run folder's parent MLflow run holds all of it.

Run from the repository root, after ``run_e5.py``::

    python experiments/a1/evaluate_e5.py --config configs/a1/a1_e5.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from yg_eo_soilnet.a1 import e5, evaluation  # noqa: E402
from yg_eo_soilnet.a1.config import load_a1_config  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.runner import VARIANTS_DIR  # noqa: E402


def variant_sizes(run_dir: Path) -> dict:
    """Each variant's number of inputs, from ``variants/<id>.txt``."""
    return {
        path.stem: sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        for path in sorted((run_dir / VARIANTS_DIR).glob("*.txt"))
    }


def evaluate(run_dir: Path, out: Path, *, first_seed: int, block_km: float, n_boot: int, seed: int) -> dict:
    """Every table and figure of E5; returns ``{name: table}`` and writes them."""
    from yg_eo_soilnet.a1.figures import e5 as figures
    from yg_eo_soilnet.a1.figures.core import save_figure
    from yg_eo_soilnet.a1.robustness import write_captions

    frame = evaluation.load_predictions(run_dir, first_seed=first_seed)
    names = sorted(frame["covariate_group"].unique()) if not frame.empty else []
    sizes = variant_sizes(run_dir)
    kwargs = dict(first_seed=first_seed, n_features=sizes, block_km=block_km, n=n_boot, seed=seed)
    parts = [e5.variant_differences(frame, e5.comparisons(part, names), part=part, **kwargs) for part in ("A", "C")]
    found = [table for table in parts if not table.empty]
    ablation = pd.concat(found, ignore_index=True) if found else parts[0]
    representation = e5.variant_differences(frame, e5.comparisons("B", names), part="B", **kwargs)
    scores = e5.variant_scores(frame)
    out.mkdir(parents=True, exist_ok=True)
    evaluation_tables = {"scores": scores, "ablation": ablation, "ts_representation": representation}
    for name, table in evaluation_tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    folder = out / "figures"
    drawn = {
        "figE5a_ablation": figures.figure_ablation(
            ablation[ablation["part"] == "A"] if not ablation.empty else ablation
        ),
        "figE5b_ablation_basins": figures.figure_ablation_basins(
            ablation[ablation["part"] == "A"] if not ablation.empty else ablation
        ),
        "figE5c_ts_representation": figures.figure_representations(representation),
    }
    captions = {}
    for name, (figure, caption) in drawn.items():
        save_figure(figure, name, folder)
        captions[name] = caption
    (folder / "data").mkdir(parents=True, exist_ok=True)
    ablation.to_csv(folder / "data" / "figE5a_ablation.csv", index=False)
    representation.to_csv(folder / "data" / "figE5c_ts_representation.csv", index=False)
    write_captions(captions, folder / "captions.md")
    return evaluation_tables


def main(argv: list[str] | None = None) -> dict:
    """Evaluate E5 from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``scores``, ``ablation``, ``ts_representation``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/a1/a1_e5.yml", help="The E5 settings file.")
    parser.add_argument("--run-name", default="default", help="The run folder under <output_dir>/runs/.")
    parser.add_argument("--out", default=None, help="Where to write (default <output_dir>).")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000).")
    parser.add_argument("--no-mlflow", action="store_true", help="Write the files only.")
    args = parser.parse_args(argv)

    a1 = load_a1_config(args.config)
    settings = a1.settings
    run_dir = Path(settings.output_dir) / "runs" / args.run_name
    out = Path(args.out) if args.out else Path(settings.output_dir)
    setup_logging(out / "evaluate_e5.log")
    tables = evaluate(
        run_dir,
        out,
        first_seed=settings.seed,
        block_km=settings.bootstrap.block_size_km,
        n_boot=args.n_boot,
        seed=settings.seed,
    )
    if not args.no_mlflow:
        import mlflow

        from yg_eo_soilnet.a1.tracking import reopen_parent

        with reopen_parent(run_dir, settings.load_config()) as parent:
            tags = {"experiment": "E5", "a1_step": "e5_summary", "mlflow.parentRunId": parent.info.run_id}
            with mlflow.start_run(
                run_name="e5_summary", experiment_id=parent.info.experiment_id, nested=True, tags=tags
            ):
                for name in ("scores", "ablation", "ts_representation"):
                    mlflow.log_artifact(str(out / f"{name}.csv"))
                mlflow.log_artifacts(str(out / "figures"), "figures")
    shown = ["part", "design", "variant", "estimate", "low", "high", "p_holm", "n_seeds_same_sign"]
    for name in ("ablation", "ts_representation"):
        table = tables[name]
        if not table.empty:
            summary(f"{name}.csv, ΔRMSE:")
            summary(table[table["metric"] == "rmse"][shown].round(3).to_string(index=False))
    summary(f"Tables and figures in {out}/")
    return tables


if __name__ == "__main__":
    main()
