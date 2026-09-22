"""Add a table of every point's predictions to a training run that has already finished.

During training this table is written only when export_point_predictions.enabled is on. This tool
writes it afterwards, from the models the run saved, without retraining:

    python export_predictions.py --parent-run-id <main run id>

For every model of the run, it rebuilds the features from the data named in the configuration,
checks they match what the run trained on (same columns, same points), reloads the saved model and
predicts every point - training points included. Each model's run gets
predictions/point_predictions.csv; the main run gets point_predictions_wide.csv (one column per
target and model) and point_predictions_long.csv (one row per point, target and model).

scikit-learn models are reloaded from MLflow. Deep-learning models are rebuilt from their
checkpoint file. For a deep-learning ensemble, every copy's checkpoint is needed: copies not saved in
MLflow are looked for in lightning_logs/ and matched to their run by validation loss. If any copy is
missing, that model is skipped rather than averaging an incomplete ensemble.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd

from config import Config
from yg_eo_soilnet.artifacts import ArtifactLayout, log_json
from yg_eo_soilnet.logger import TrainingLogger
from yg_eo_soilnet.logger.mlflow_loggers import (
    MEMBER_RUN_KIND,
    ChildRunLogger,
    ParentRunLogger,
    _scoring_runs,
)
from yg_eo_soilnet.predictions_export import point_id_column
from yg_eo_soilnet.targets import split_target_names
from yg_eo_soilnet.tracking import configure_tracking_uri

#: File name of the summary written into the main run's ``predictions/`` folder.
BACKFILL_SUMMARY_FILE = "backfill_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options; ``argv`` defaults to the real command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--parent-run-id", required=True, help="Id of the finished main (parent) run to add the predictions to."
    )
    parser.add_argument(
        "--config-path",
        default="configs/main_config.yml",
        help="Main configuration file (default: configs/main_config.yml). Its data must be the "
        "data the run trained on: the features are rebuilt from it.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Only export these models, comma-separated (e.g. Ridge,soil_cnn). Default: every model "
        "the run trained.",
    )
    parser.add_argument(
        "--skip-models",
        default=None,
        help="Models to leave out, comma-separated. Default: export_point_predictions.skip_models "
        "from the config.",
    )
    parser.add_argument(
        "--allow-population-drift",
        action="store_true",
        help=(
            "Go on even if the rebuilt data has different points from the run (for example points "
            "added since), instead of stopping. Different feature columns still stop the export: "
            "the models could not predict on them."
        ),
    )
    parser.add_argument(
        "--member-checkpoint-dir",
        default="lightning_logs",
        help=(
            "Folder to search for the checkpoints of a deep-learning ensemble's copies that were "
            "not saved in MLflow (default: lightning_logs). Each is matched to its run by "
            "validation loss."
        ),
    )
    parser.add_argument(
        "--no-member-recovery",
        action="store_true",
        help="Do not search for ensemble checkpoints; deep-learning ensembles are then skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rebuild the data, run the checks and list what would be exported, without writing.",
    )
    return parser.parse_args(argv)


# --- data ------------------------------------------------------------------


def rebuild_features(config, logger) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the model inputs of every point from the configured data, as training did.

    No split is made (that would overwrite the finished run's own split record); predicting every
    point needs none.

    Returns
    -------
    features : pandas.DataFrame
        One row per point, with the same columns and column types the scikit-learn models were
        trained on.
    point_ids : pandas.Series
        Each row's point id, on the same index.
    """
    from yg_eo_soilnet.data_manager import DataManager
    from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor

    data_manager = DataManager(config, logger)
    frame = data_manager.load_dataset().tabular
    processed = TabularPreprocessor(config, logger, data_manager).preprocess_data(frame)

    features = data_manager.filter_schema(processed["X"], list(config.TARGET_COLUMNS))
    # Integer columns become floats, as they did before training.
    features = features.astype(
        {column: "float64" for column in features.select_dtypes(include=["int64", "int32"]).columns}
    )
    point_ids = pd.Series(
        np.asarray(processed["point_ids"]), index=features.index, name="point_id"
    )
    logger.info(f"Rebuilt {len(features)} points x {features.shape[1]} features from source.")
    return features, point_ids


def _download(run_id: str, artifact_path: str):
    """Download one file from a run and return its local path, or ``None`` if it is not there."""
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    except Exception:
        return None


def check_for_drift(
    parent_run_id: str,
    features: pd.DataFrame,
    point_ids: pd.Series,
    *,
    allow_population_drift: bool,
    logger,
) -> dict[str, Any]:
    """Check the rebuilt data against what the run recorded, before any model is loaded.

    Two checks, against the run's ``data_splits/`` files: the feature columns must be exactly the
    same, and the set of points should be the same.

    Parameters
    ----------
    parent_run_id : str
        The main run.
    features, point_ids
        The output of :func:`rebuild_features`.
    allow_population_drift : bool
        Go on, with a warning, when the points differ (or the run recorded no split to compare
        with) instead of stopping.
    logger : logging.Logger
        Where warnings go.

    Returns
    -------
    dict
        What was compared: point counts, points added or removed, and any column differences.

    Raises
    ------
    SystemExit
        If the columns differ, or if the points differ without ``allow_population_drift``.
    """
    report: dict[str, Any] = {"checked": False}

    split_path = _download(parent_run_id, f"{ArtifactLayout.DATA_SPLITS}/split_assignments.parquet")
    test_path = _download(parent_run_id, f"{ArtifactLayout.DATA_SPLITS}/X_test.parquet")
    if split_path is None or test_path is None:
        # Older runs recorded no usable split, so the data cannot be checked against them.
        message = (
            "This run logged no usable data_splits/ artifacts (no split_assignments.parquet or no "
            "X_test.parquet), so the rebuilt data cannot be checked against what it trained on."
        )
        if not allow_population_drift:
            raise SystemExit(
                message + "\nPass --allow-population-drift to export anyway, unverified."
            )
        logger.warning(message + " Continuing unverified because --allow-population-drift was passed.")
        return report

    recorded_columns = [c for c in pd.read_parquet(test_path).columns if c != "point_id"]
    rebuilt_columns = list(features.columns)
    missing = [c for c in recorded_columns if c not in rebuilt_columns]
    extra = [c for c in rebuilt_columns if c not in recorded_columns]

    recorded_points = set(pd.read_parquet(split_path)["point_id"])
    rebuilt_points = set(point_ids)
    added = rebuilt_points - recorded_points
    removed = recorded_points - rebuilt_points

    report = {
        "checked": True,
        "n_recorded_points": len(recorded_points),
        "n_rebuilt_points": len(rebuilt_points),
        "n_points_added": len(added),
        "n_points_removed": len(removed),
        "missing_columns": missing,
        "extra_columns": extra,
    }

    if missing or extra:
        raise SystemExit(
            "The rebuilt features do not match the columns this run was trained on, so its models "
            "cannot predict on them.\n"
            f"  missing: {', '.join(missing[:10]) or '(none)'}\n"
            f"  extra:   {', '.join(extra[:10]) or '(none)'}\n"
            "The config or the source data has changed since the run."
        )

    if added or removed:
        message = (
            f"The rebuilt population differs from the run's: {len(added)} point(s) added, "
            f"{len(removed)} removed."
        )
        if added and not removed:
            # Extra points with none missing usually just mean the run's population_policy was
            # `intersect`, which recorded only the points every model family could use.
            message += (
                " Nothing was removed, so this is most likely the split's population_policy rather "
                "than changed data: a run whose families disagreed about usable rows records only "
                "the intersection."
            )
        if not allow_population_drift:
            raise SystemExit(
                message + "\nThe exported predictions would describe a different set of points "
                "than the run did. Pass --allow-population-drift to export anyway."
            )
        logger.warning(message + " Continuing because --allow-population-drift was passed.")

    return report


# --- per child -------------------------------------------------------------


def _export_config(config, args: argparse.Namespace):
    """Return a copy of the config with the export switched on and the command's model filters applied.

    Errors are also switched to raise, so this command can report which model failed.
    """
    proxy = copy.copy(config)
    proxy.EXPORT_POINT_PREDICTIONS = True
    proxy.EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR = True
    if args.models is not None:
        proxy.EXPORT_POINT_PREDICTIONS_MODELS = [n.strip() for n in args.models.split(",") if n.strip()]
    else:
        proxy.EXPORT_POINT_PREDICTIONS_MODELS = []
    if args.skip_models is not None:
        proxy.EXPORT_POINT_PREDICTIONS_SKIP_MODELS = [
            n.strip() for n in args.skip_models.split(",") if n.strip()
        ]
    return proxy


def _run_summary(run_id: str) -> dict:
    """Return a run's ``meta/run_summary.json`` as a dict, or ``{}`` if it has none."""
    path = _download(run_id, f"{ArtifactLayout.META}/{ArtifactLayout.RUN_SUMMARY_FILE}")
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --- recovering a Lightning ensemble's members -----------------------------


def _version_candidates(checkpoint_dir: str, target_names: list[str]) -> list[dict]:
    """List the ``lightning_logs/version_*`` folders holding a checkpoint for exactly these targets.

    Each record gives the folder, its checkpoint file and its lowest recorded validation loss.
    The targets are read from the small ``hparams.yaml`` rather than the large checkpoint.
    """
    import glob

    import yaml

    wanted = [str(name) for name in target_names]
    candidates = []
    for hparams_path in sorted(glob.glob(os.path.join(checkpoint_dir, "version_*", "hparams.yaml"))):
        version_dir = os.path.dirname(hparams_path)
        checkpoints = glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt"))
        if not checkpoints:
            continue
        try:
            with open(hparams_path, encoding="utf-8") as handle:
                hparams = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if [str(name) for name in (hparams.get("target_names") or [])] != wanted:
            continue
        candidates.append(
            {
                "version_dir": version_dir,
                "checkpoint": sorted(checkpoints)[0],
                "val_loss": _min_val_loss_from_csv(os.path.join(version_dir, "metrics.csv")),
            }
        )
    return candidates


def _min_val_loss_from_csv(metrics_path: str) -> Optional[float]:
    """Return the lowest ``val_loss`` in a Lightning ``metrics.csv``, or ``None``."""
    if not os.path.isfile(metrics_path):
        return None
    try:
        frame = pd.read_csv(metrics_path)
    except Exception:
        return None
    if "val_loss" not in frame.columns:
        return None
    values = frame["val_loss"].dropna()
    return float(values.min()) if len(values) else None


def _min_val_loss_from_run(client, run_id: str) -> Optional[float]:
    """Return the lowest ``val_loss`` an MLflow run recorded, or ``None``."""
    try:
        history = client.get_metric_history(run_id, "val_loss")
    except Exception:
        return None
    return min((point.value for point in history), default=None)


def match_member_checkpoints(client, child_run, checkpoint_dir: str, target_names: list[str]) -> list[dict]:
    """Find the checkpoint file of each copy of a deep-learning ensemble.

    A copy that saved its checkpoint in MLflow is taken from there. Otherwise its checkpoint is
    looked for in ``checkpoint_dir`` and matched by the lowest validation loss, which both the run
    and the checkpoint folder record and which differs between copies (file dates and folder
    numbers are not reliable).

    Parameters
    ----------
    client : mlflow.MlflowClient
        Used to find the copies' runs.
    child_run : mlflow.entities.Run
        The ensemble's model run.
    checkpoint_dir : str
        The folder to search, usually ``lightning_logs``.
    target_names : list of str
        The targets the model predicts.

    Returns
    -------
    list of dict
        One record per copy, in order, with its ``checkpoint`` path (``None`` if not found) and
        where it came from (``"mlflow"`` or ``"lightning_logs"``).

    Raises
    ------
    SystemExit
        If a copy matches several checkpoints equally well.
    """
    members = [
        run
        for run in client.search_runs(
            experiment_ids=[child_run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{child_run.info.run_id}'",
        )
        if run.data.tags.get("run_kind") == MEMBER_RUN_KIND
    ]
    members.sort(key=lambda run: int(run.data.params.get("ensemble_member", 0)))

    candidates = _version_candidates(checkpoint_dir, target_names)
    matched: list[dict] = []
    used: set[str] = set()

    for member in members:
        # Recent runs save each copy's checkpoint in MLflow.
        logged = _download(
            member.info.run_id, f"{ArtifactLayout.CHECKPOINTS}/{ArtifactLayout.CHECKPOINT_FILE}"
        )
        index = int(member.data.params.get("ensemble_member", len(matched)))
        record = {
            "ensemble_member": index,
            "ensemble_seed": member.data.params.get("ensemble_seed"),
            "run_id": member.info.run_id,
        }
        if logged is not None:
            record.update({"checkpoint": logged, "source": "mlflow"})
            matched.append(record)
            continue

        target_loss = _min_val_loss_from_run(client, member.info.run_id)
        hits = [
            candidate
            for candidate in candidates
            if candidate["version_dir"] not in used
            and candidate["val_loss"] is not None
            and target_loss is not None
            and np.isclose(candidate["val_loss"], target_loss, rtol=0.0, atol=1e-9)
        ]
        if len(hits) > 1:
            raise SystemExit(
                f"Member {index} of {child_run.info.run_id} matches {len(hits)} checkpoints on "
                f"val_loss={target_loss}. Refusing to guess which is which."
            )
        if hits:
            used.add(hits[0]["version_dir"])
            record.update(
                {
                    "checkpoint": hits[0]["checkpoint"],
                    "version_dir": hits[0]["version_dir"],
                    "val_loss": hits[0]["val_loss"],
                    "source": "lightning_logs",
                }
            )
        else:
            record.update({"checkpoint": None, "val_loss": target_loss})
        matched.append(record)

    return matched


def sklearn_predictor(run, features: pd.DataFrame):
    """Reload a scikit-learn model from its run and return ``(predict, number of rows expected)``.

    The model is loaded as ``runs:/<run id>/<model name>``, or else from the model address recorded
    in the run's summary.

    Raises
    ------
    SystemExit
        If neither address can be loaded.
    """
    import mlflow.sklearn

    target = run.data.tags.get("target") or ""
    model_name = run.data.tags.get("model_name") or ""
    candidates = [
        f"runs:/{run.info.run_id}/{ArtifactLayout.logged_model_name(target, model_name)}",
    ]
    recorded = _run_summary(run.info.run_id).get("logged_model_uri")
    if recorded:
        candidates.append(str(recorded))

    errors = []
    for uri in candidates:
        try:
            model = mlflow.sklearn.load_model(uri)
            return (lambda: model.predict(features)), len(features)
        except Exception as exc:
            errors.append(f"{uri}: {type(exc).__name__}: {exc}")

    raise SystemExit(
        f"Could not reload the fitted model for run {run.info.run_id}. Tried:\n  "
        + "\n  ".join(errors)
    )


# Sequence bundles already built in this invocation, by model name.
_BUNDLE_CACHE: dict[str, Any] = {}


def sequence_bundle_for(model_name: str, config, logger):
    """Return the deep-learning model's input data for every point, building it only once.

    The data depends only on the configuration, so the copies of an ensemble and the runs of the
    same model all reuse it.
    """
    if model_name not in _BUNDLE_CACHE:
        from yg_eo_soilnet.data_manager import DataManager
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        entry = (getattr(config, "LIGHTNING_MODEL_REGISTRY", None) or {}).get(model_name, {})
        logger.info(f"Building the sequence bundle for {model_name} (once, then reused).")
        _BUNDLE_CACHE[model_name] = SoilSequenceBuilder(
            config, logger, DataManager(config, logger)
        ).build(sequence_data_args=dict(entry.get("sequence_data_args", {}) or {}))
    return _BUNDLE_CACHE[model_name]


def _restore_lightning_model(checkpoint: str, model_name: str, config):
    """Rebuild a deep-learning model from its checkpoint file, ready to predict."""
    from relog import infer_model_class_path, resolve_model_class

    module_class = resolve_model_class(infer_model_class_path(config, model_name))
    model = module_class.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


def lightning_predictor(run, config, logger):
    """Rebuild a single deep-learning model from its run's checkpoint.

    Returns
    -------
    tuple
        ``(predict, point_ids, target_names)``: a function returning the predictions for every
        point, the points in that order, and the targets in column order.

    Raises
    ------
    SystemExit
        If the run saved no checkpoint.
    """
    from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

    model_name = run.data.tags.get("model_name") or ""
    checkpoint = _download(run.info.run_id, f"{ArtifactLayout.CHECKPOINTS}/{ArtifactLayout.CHECKPOINT_FILE}")
    if checkpoint is None:
        raise SystemExit(
            f"Run {run.info.run_id} logged no {ArtifactLayout.CHECKPOINT_FILE}, so its weights "
            "cannot be restored."
        )

    predictor = SoilSequencePredictor(_restore_lightning_model(checkpoint, model_name, config))
    bundle = sequence_bundle_for(model_name, config, logger)
    # The target names stored in the model, in its own output order.
    target_names = list(predictor.preprocessing_state.get("target_names") or [])
    point_ids = list(bundle.point_ids)
    return (lambda: predictor.predict(bundle)), point_ids, target_names


def lightning_ensemble_predictor(run, config, logger, matched: list[dict]):
    """Like :func:`lightning_predictor`, but predicting the average of every copy of an ensemble.

    Only the average is exported; the spread between copies is in the run's own uncertainty files.
    """
    from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor
    from yg_eo_soilnet.uncertainty import aggregate

    model_name = run.data.tags.get("model_name") or ""
    bundle = sequence_bundle_for(model_name, config, logger)
    predictors = [
        SoilSequencePredictor(_restore_lightning_model(record["checkpoint"], model_name, config))
        for record in matched
    ]
    target_names = list(predictors[0].preprocessing_state.get("target_names") or [])

    def predict():
        return aggregate([predictor.predict(bundle) for predictor in predictors]).mean

    return predict, list(bundle.point_ids), target_names


def backfill_child(
    run, *, client, config, export_config, features, point_ids, logger, checkpoint_dir
) -> dict[str, Any]:
    """Predict every point with one model and write ``predictions/point_predictions.csv`` to its run.

    Returns
    -------
    dict
        What happened: the run, model and target, and the number of points written, or why it was
        skipped.
    """
    model_name = run.data.tags.get("model_name")
    framework = run.data.tags.get("framework", "sklearn")
    target = run.data.tags.get("target") or ""
    outcome: dict[str, Any] = {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "framework": framework,
        "target": target,
    }

    n_members = run.data.params.get("uncertainty_n_members")
    if framework == "lightning" and n_members:
        # Every copy is needed: the average of some copies is not the ensemble's prediction.
        if not checkpoint_dir:
            outcome["skipped"] = (
                f"trained as an ensemble of {n_members} members and member recovery is disabled"
            )
            return outcome

        matched = match_member_checkpoints(
            client, run, checkpoint_dir, split_target_names(target) or [target]
        )
        found = [record for record in matched if record.get("checkpoint")]
        outcome["members"] = matched
        if len(found) != int(n_members):
            outcome["skipped"] = (
                f"trained as an ensemble of {n_members} members but only {len(found)} checkpoint(s) "
                f"could be recovered from {checkpoint_dir}; averaging a subset would not be the "
                "ensemble the run reported"
            )
            return outcome

        logger.info(
            f"  recovered {len(found)}/{n_members} members for {target} "
            f"({sum(r.get('source') == 'mlflow' for r in found)} from MLflow, "
            f"{sum(r.get('source') == 'lightning_logs' for r in found)} from {checkpoint_dir})"
        )
        predict, child_point_ids, target_names = lightning_ensemble_predictor(
            run, config, logger, found
        )
        n_expected = len(child_point_ids)
    elif framework == "lightning":
        predict, child_point_ids, target_names = lightning_predictor(run, config, logger)
        n_expected = len(child_point_ids)
    else:
        predict, n_expected = sklearn_predictor(run, features)
        child_point_ids = point_ids.to_numpy()
        target_names = split_target_names(target) or [target]

    with mlflow.start_run(run_id=run.info.run_id):
        written = ChildRunLogger()._log_point_predictions(
            config=export_config,
            model_name=model_name,
            target_names=target_names,
            predict=predict,
            point_ids=child_point_ids,
            n_expected=n_expected,
        )
    if not written:
        outcome["skipped"] = "excluded by --models / --skip-models"
    else:
        outcome.update(written)
    return outcome


# --- orchestration ---------------------------------------------------------


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    """Export every point's predictions for each model of a finished run.

    Parameters
    ----------
    args : argparse.Namespace
        The options from :func:`parse_args`.

    Returns
    -------
    dict
        The summary also saved as ``predictions/backfill_summary.json`` on the main run: the data
        check and one outcome per model.

    Raises
    ------
    SystemExit
        If the rebuilt data does not match the run (see :func:`check_for_drift`).
    """
    config = Config(config_path=args.config_path)
    # Only the MLflow location is taken from the config: runs are opened by id, in their own
    # experiment.
    configure_tracking_uri(config)

    logger = TrainingLogger(name="prediction-backfill", enable_file_logging=False).get_logger()

    client = mlflow.MlflowClient()
    parent = client.get_run(args.parent_run_id)
    mlflow.set_experiment(experiment_id=parent.info.experiment_id)

    features, point_ids = rebuild_features(config, logger)
    drift = check_for_drift(
        args.parent_run_id,
        features,
        point_ids,
        allow_population_drift=args.allow_population_drift,
        logger=logger,
    )

    children = _scoring_runs(
        client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{args.parent_run_id}'",
        )
    )
    children = [run for run in children if run.data.tags.get("model_name")]
    logger.info(f"{len(children)} child model run(s) to back-fill.")

    # Where to look for ensemble checkpoints not saved in MLflow; None switches the search off.
    checkpoint_dir = None if args.no_member_recovery else args.member_checkpoint_dir

    if args.dry_run:
        for run in children:
            logger.info(
                f"  would export {run.data.tags.get('target')} / {run.data.tags.get('model_name')} "
                f"({run.data.tags.get('framework', 'sklearn')})"
            )
        return {"dry_run": True, "drift": drift, "n_children": len(children)}

    export_config = _export_config(config, args)
    outcomes = []
    for run in children:
        label = f"{run.data.tags.get('target')} / {run.data.tags.get('model_name')}"
        try:
            outcome = backfill_child(
                run,
                client=client,
                config=config,
                export_config=export_config,
                features=features,
                point_ids=point_ids,
                logger=logger,
                checkpoint_dir=checkpoint_dir,
            )
        except SystemExit:
            raise
        except Exception as exc:
            outcome = {
                "run_id": run.info.run_id,
                "model_name": run.data.tags.get("model_name"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        # One model failing does not stop the others; failures are listed in the summary.
        if outcome.get("error"):
            logger.warning(f"  {label}: {outcome['error']}")
        elif outcome.get("skipped"):
            logger.info(f"  {label}: skipped - {outcome['skipped']}")
        else:
            logger.info(f"  {label}: exported {outcome.get('n_points')} points")
        outcomes.append(outcome)

    exported = [o for o in outcomes if not o.get("skipped") and not o.get("error")]
    summary = {
        "backfilled": True,
        "parent_run_id": args.parent_run_id,
        "config_path": args.config_path,
        "data_file": getattr(config, "DATA_FILE", None),
        "targets_file": getattr(config, "TARGETS_FILE", None),
        "n_points": int(len(features)),
        "id_column": point_id_column(config),
        "drift": drift,
        "member_checkpoint_dir": checkpoint_dir,
        "children": outcomes,
    }

    with mlflow.start_run(run_id=args.parent_run_id):
        if exported:
            ParentRunLogger()._log_point_prediction_export(args.parent_run_id, export_config)
        # Always written, even if nothing was exported, so the run shows it was processed here.
        log_json(summary, BACKFILL_SUMMARY_FILE, ArtifactLayout.PREDICTIONS)
        mlflow.set_tags({"point_predictions_backfilled": "true"})

    return summary


def main(argv: list[str] | None = None) -> int:
    """Run :func:`backfill` and print its summary; returns 0."""
    args = parse_args(argv)
    summary = backfill(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
