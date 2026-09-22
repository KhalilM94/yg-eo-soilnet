"""Save and register a deep-learning model from its checkpoint file, without retraining.

Use this when a training run finished but its model was not saved to MLflow (for example because
saving failed at the end). The checkpoint file holds everything needed to rebuild the model: its
weights, its hyperparameters, and the scaling statistics and category lists it uses to prepare new
data - so no dataset is needed.

    python relog.py --checkpoint lightning_logs/version_12/checkpoints/<file>.ckpt --run-id <run id>

The model is saved into the ORIGINAL run (the one whose id you give), next to the scores it earned
there, and registered as a new version of ``<target>_<model>``; it becomes the champion if its
rmse_test beats the current champion's. A summary is written to meta/relog_summary.json in the run.

The model class is found from the run's model_name tag and the deep-learning model list; pass
--model-class to give it yourself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import mlflow

from config import Config
from yg_eo_soilnet.artifacts import ArtifactLayout, log_json
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.tracking import CHAMPION_METRIC, configure_tracking_uri, promote_if_better

#: File name of the summary written into the run's ``meta/`` folder.
RELOG_SUMMARY_FILE = "relog_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options; ``argv`` defaults to the real command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="The checkpoint (.ckpt) file to rebuild the model from.")
    parser.add_argument("--run-id", required=True, help="Id of the MLflow run the model was trained in.")
    parser.add_argument(
        "--model-class",
        default=None,
        help="Import path of the model class, e.g. "
        "yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module.SoilCNNLightningModule. "
        "Default: found from the run's model_name tag.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main configuration file (default: configs/main_config.yml), read for the MLflow "
        "location and the model list.",
    )
    parser.add_argument(
        "--no-register", action="store_true", help="Save the model in the run but do not register it."
    )
    parser.add_argument(
        "--allow-ensemble-member",
        action="store_true",
        help=(
            "Accept a run that trained an ensemble (several copies of the model). Refused by "
            "default: one copy is not the ensemble, and the run's scores describe the ensemble."
        ),
    )
    parser.add_argument(
        "--rows", type=int, default=3, help="Rows in the example input saved with the model (default 3)."
    )
    return parser.parse_args(argv)


def resolve_model_class(class_path: str):
    """Import and return the class named by a dotted path such as ``package.module.ClassName``."""
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


#: Older model names, now options of ``soil_cnn``, and the class that still loads their checkpoints.
#: Used for runs whose ``model_name`` is no longer in the model list.
LEGACY_MODEL_CLASSES = {
    "soil_residual_cnn": (
        "yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module."
        "SoilResidualCNNLightningModule"
    ),
    "soil_residual_attention_cnn": (
        "yg_eo_soilnet.models.lightningmodules.soil_residual_attention_cnn_lightning_module."
        "SoilResidualAttentionCNNLightningModule"
    ),
}


def infer_model_class_path(config, model_name: str) -> str:
    """Return the import path of a model's class, from the model list (or the older names above).

    Raises
    ------
    SystemExit
        If the model name is in neither; pass ``--model-class`` in that case.
    """
    entry = (getattr(config, "LIGHTNING_MODEL_REGISTRY", None) or {}).get(model_name)
    import_path = (entry or {}).get("import_path") or LEGACY_MODEL_CLASSES.get(model_name)
    if not import_path:
        raise SystemExit(
            f"Cannot infer the model class: the run's model_name is {model_name!r}, which is not in "
            f"the Lightning registry. Pass --model-class explicitly."
        )
    return str(import_path)


def load_static_frame(run_id: str, rows: int):
    """Return the first ``rows`` rows of the run's test predictions table, or ``None`` if missing.

    Their static covariates make the saved example input realistic; without them the example uses
    the training averages stored in the checkpoint.
    """
    import pandas as pd

    from yg_eo_soilnet.logger.mlflow_loggers import _eval_results_artifact_paths

    for artifact_path in _eval_results_artifact_paths("", ""):
        try:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            return pd.read_csv(local, nrows=max(1, rows))
        except Exception:
            continue
    return None


def relog(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the model from its checkpoint and save (and register) it into the original run.

    Parameters
    ----------
    args : argparse.Namespace
        The options from :func:`parse_args`.

    Returns
    -------
    dict
        The summary also written to ``meta/relog_summary.json``: the run, the model name, the new
        registered version (or ``None``) and whether it became the champion.

    Raises
    ------
    SystemExit
        If the run trained an ensemble (without ``--allow-ensemble-member``), if the model class
        cannot be found, or if the checkpoint lacks the scaling statistics needed to prepare data.
    """
    config = Config(config_path=args.config_path)
    # Only the MLflow location is taken from the config: the run is opened by its id, in its own
    # experiment.
    configure_tracking_uri(config)

    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    # Reopening a run requires its own experiment to be the active one.
    mlflow.set_experiment(experiment_id=run.info.experiment_id)
    tags = run.data.tags
    model_name = tags.get("model_name") or "model"
    target = tags.get("target") or "target"

    # An ensemble's prediction is the average of several checkpoints, and its uncertainty their
    # spread. One checkpoint reproduces neither, so registering it would misrepresent the run.
    n_members = run.data.params.get("uncertainty_n_members")
    if n_members and not args.allow_ensemble_member:
        raise SystemExit(
            f"Run {args.run_id} trained an ensemble of {n_members} members, and a single checkpoint "
            "cannot reproduce it: its predictions are the members' mean and its interval is their "
            "spread. Re-running the training is the only way to recover the ensemble. Pass "
            "--allow-ensemble-member to register this checkpoint anyway, as a single model whose "
            "metrics on this run will overstate it."
        )

    class_path = args.model_class or infer_model_class_path(config, model_name)
    module_class = resolve_model_class(class_path)

    # Loading also restores the scaling statistics and category lists saved in the checkpoint.
    model = module_class.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    state = model.get_preprocessing_state()
    if not state:
        raise SystemExit(
            f"{args.checkpoint} carries no preprocessing state, so the model cannot standardize raw "
            "input and is not servable. It predates attach_preprocessing_state; retrain instead."
        )

    from yg_eo_soilnet.serving.lightning_pyfunc import example_from_state, required_label_columns

    input_example = example_from_state(
        state,
        static_frame=load_static_frame(args.run_id, args.rows),
        n_rows=args.rows,
        auxiliary_columns=required_label_columns(model),
    )

    logger = ChildRunLogger()
    register = not args.no_register

    with mlflow.start_run(run_id=args.run_id):
        logged = logger._log_lightning_serialized_model(
            model=model,
            model_name=model_name,
            target=target,
            bundle=None,
            best_model_path=args.checkpoint,
            config=type("_C", (), {"MLFLOW_REGISTER_MODELS": register})(),
            input_example=input_example,
        )
        version = getattr(logger, "_registered_version", None)
        logger._tag_model_logging(target, model_name, logged, None if logged else "relog failed")

        champion = {"promoted": False, "reason": "registration was disabled"}
        if register:
            champion = promote_if_better(
                ArtifactLayout.logged_model_name(target, model_name),
                version,
                run.data.metrics.get(CHAMPION_METRIC),
            )

        summary = {
            "recovered_from_checkpoint": args.checkpoint,
            "model_class": class_path,
            "run_id": args.run_id,
            "target": target,
            "model_name": model_name,
            "logged_model_name": ArtifactLayout.logged_model_name(target, model_name),
            "registered_model_version": version,
            CHAMPION_METRIC: run.data.metrics.get(CHAMPION_METRIC),
            "champion": champion,
        }
        # A separate file, so the run's original summary is kept as it was.
        log_json(summary, RELOG_SUMMARY_FILE, ArtifactLayout.META)

    return summary


def main(argv: list[str] | None = None) -> int:
    """Run :func:`relog`, print its summary, and return 1 if registration was asked for but failed."""
    summary = relog(parse_args(argv))
    print(json.dumps(summary, indent=2, default=str))
    if summary["registered_model_version"] is None and summary["champion"].get("reason") != (
        "registration was disabled"
    ):
        print("\nThe model was logged but NOT registered.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
