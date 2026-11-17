"""How much of E1's transfer loss would k local samples correct, without refitting? (No model is fitted.)

For each held-out basin, from the leave-one-basin-out Global predictions of the first seed: k of the
basin's points (k = 10, 25, 50, 100, 200, 400, where at least 100 points remain) are drawn by whole
10 km blocks as local samples, 100 draws per k; the rest of the last, partly used block is set aside
(no test point shares a block with a sample); every other basin point is a test point. On the same
test points: the uncorrected predictions; minus the samples' mean residual (offset); recalibrated by
measured ~ a + b x predicted fitted on the samples (linear, k >= 25); one offset per measured
carbonate class (a diagnostic - it needs the measured class), the overall offset for a class with
fewer than 5 samples; and the local model. Writes into ``<output_dir>/robustness/``:
``bias_correction.csv`` (per basin and k: each RMSE's mean and 5-95 % over the draws, the
transferred RMSE on every point), ``bias_correction_draws.csv`` and figure C6 with its data and
caption. See :mod:`yg_eo_soilnet.a1.robustness`.

With ``--mixed-run`` (default ``results/a1_mixed_arms/runs/default``, when it exists) the E2
retraining curves of Oum Er Rbia and Sebou - control and mixed, mean over repeats against the number
of moved points - are drawn in the same panels; their test points differ from the draws'. The script
runs without them.

``--local-model`` (default ``local_global``, the local model on Global's inputs) names the local
model; a run without it falls back to ``local`` (with predicted CaCO3/CEC), with a warning, and
every table and caption says which was used.

Run from the repository root, after ``run_a1.py``::

    python experiments/a1/bias_correction.py --config configs/a1/a1_xgb.yml
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
from yg_eo_soilnet.a1.figures.robustness import figure_bias_correction  # noqa: E402
from yg_eo_soilnet.a1.logs import setup_logging, summary  # noqa: E402
from yg_eo_soilnet.a1.settings import load_settings  # noqa: E402
from yg_eo_soilnet.a1.stage2 import GLOBAL  # noqa: E402

#: The E2 run folder read by default, when it exists.
DEFAULT_MIXED_RUN = REPOSITORY_ROOT / "results" / "a1_mixed_arms" / "runs" / "default"
#: The basins E2 holds out.
E2_BASINS = ("Oum Er Rbia", "Sebou")


def main(argv: list[str] | None = None) -> dict:
    """Run the correction draws from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``None`` reads them from the command line.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``bias_correction``, ``bias_correction_draws`` and ``bias_correction_e2`` (empty without E2).
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
    parser.add_argument("--k", default=",".join(str(k) for k in robustness.K_VALUES), help="The values of k.")
    parser.add_argument("--n-draws", type=int, default=100, help="Draws per basin and k (default 100).")
    parser.add_argument(
        "--mixed-run", default=None, help="The E2 run folder (default results/a1_mixed_arms/runs/default, if there)."
    )
    parser.add_argument("--out", default=None, help="Where to write (default <output_dir>/robustness).")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not add the outputs to MLflow.")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    output_dir = Path(settings.output_dir)
    run_dir = output_dir / "runs" / args.run_name
    out = Path(args.out) if args.out else output_dir / "robustness"
    setup_logging(out / "bias_correction.log")
    block_km = settings.bootstrap.block_size_km

    local_model = robustness.choose_local_model(robustness.models_in_run(run_dir), args.local_model)
    frame = evaluation.load_predictions(run_dir, first_seed=settings.seed)
    first = evaluation.seed_order(frame, settings.seed)[0]
    rows = robustness.logo_test_rows(frame[frame["seed"] == first], [GLOBAL, local_model])
    pairs = robustness.paired(rows, local_model, classed_only=False)  # every point; no class: overall offset
    positions = pairs.drop_duplicates("point_id")[["point_id", "lat", "lon"]]
    blocks = pd.Series(
        evaluation.spatial_blocks(positions, block_km).to_numpy(), index=positions["point_id"].to_numpy()
    )
    ks = [int(k) for k in args.k.split(",") if k.strip()]
    draws = robustness.correction_draws(pairs, blocks, ks=ks, n_draws=args.n_draws, seed=settings.seed)
    table = robustness.correction_summary(draws, pairs, local_model=local_model)

    mixed_run = Path(args.mixed_run) if args.mixed_run else DEFAULT_MIXED_RUN
    e2 = robustness.read_e2_curves(mixed_run if mixed_run.exists() else None, E2_BASINS)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "bias_correction.csv", index=False)
    draws.assign(local_model=local_model, seed=first).to_csv(out / "bias_correction_draws.csv", index=False)
    if not e2.empty:
        e2.to_csv(out / "bias_correction_e2.csv", index=False)

    failing = robustness.failing_basins(run_dir, local_model, block_km=block_km, seed=settings.seed)
    figure, caption = figure_bias_correction(
        table, local_model=local_model, failing=failing, e2=e2, n_draws=args.n_draws, block_km=block_km
    )
    tables = {"figC6_bias_correction": table}
    if not e2.empty:
        tables["figC6_bias_correction_e2"] = e2
    robustness.save_figures({"figC6_bias_correction": (figure, caption, tables)}, out)
    if not args.no_mlflow:
        robustness.log_robustness(run_dir, out, step="bias_correction")
    summary(f"Local model: {local_model} ({robustness.local_text(local_model)}); seed {first}")
    shown = ["basin", "k", "rmse_uncorrected", "rmse_offset", "rmse_linear", "rmse_per_class_offset", "rmse_local"]
    summary(table[shown].round(2).to_string(index=False))
    summary(
        f"E2 curves: {'drawn' if not e2.empty else 'not found (' + str(mixed_run) + ')'}; tables and figure C6 in {out}/"
    )
    return {"bias_correction": table, "bias_correction_draws": draws, "bias_correction_e2": e2}


if __name__ == "__main__":
    main()
