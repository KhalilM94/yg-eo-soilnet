"""Train the scikit-learn models: search their settings, fit them, record the results."""

from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import clone
import mlflow
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import traceback

from yg_eo_soilnet.logger import ChildRunLogger, TrainingLogger
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import (CVSplitter, PipelineBuilder, TargetNanFilter)
from yg_eo_soilnet.targets import split_target_names
from yg_eo_soilnet.tracking import start_child_run
from yg_eo_soilnet.uncertainty import (
    aggregate,
    bootstrap_indices,
    fit_calibrators,
    member_seeds,
    should_bootstrap,
    uncertainty_enabled_for,
)
from yg_eo_soilnet.uncertainty.intervals import (
    build_interval_estimators,
    needs_calibration_set,
    normalize_method,
)
from yg_eo_soilnet.uncertainty.predictors import EnsembleRegressor
from yg_eo_soilnet.utils import LogTransformer

#: Tag marking a sub-run as one :term:`ensemble` member rather than one target's results. The
#: leaderboard reads it, or it would list five members in place of the model.
MEMBER_RUN_KIND = "ensemble_member"

class ModelTrainer:
    """Train every scikit-learn model switched on, one :term:`target group` at a time.

    For each model it searches the hyperparameter grid from the model list by cross-validation
    inside the :term:`fit pool`, fits the winner, and hands the result to the logger, which scores
    it on the test points and records everything under a sub-run of its own. With uncertainty on it
    fits an :term:`ensemble` instead of a single model and calibrates the intervals.

    Parameters
    ----------
    config : Config
        The run configuration.
    columns_to_transform : list of str, optional
        Targets trained on 10·ln(1 + *y*) rather than *y*.
    enable_clustering : bool, default False
        Keep spatial groups whole in the cross-validation folds.
    split_strategy : {"kfold", "groupkfold"}, default "kfold"
        How those folds are made.
    seed : int, default 42
        The run's random seed.
    n_splits : int, default 5
        How many folds.
    logger : logging.Logger, optional
        Where progress messages go.
    tuning_verbose : int, default 0
        How much the search itself prints.

    Examples
    --------
    >>> trainer = ModelTrainer(config, logger=logger)               # doctest: +SKIP
    >>> trainer.train("clay_pct", data, model_pipelines)            # doctest: +SKIP
    """

    def __init__(
        self,
        config,
        columns_to_transform: Optional[List[str]] = None,
        enable_clustering: bool = False,
        split_strategy: str = 'kfold',
        seed: int = 42,
        n_splits: int = 5,
        logger= None,
        tuning_verbose: int = 0
    ):
        self.config = config
        self.columns_to_transform = columns_to_transform or []
        self.enable_clustering = enable_clustering
        self.split_strategy = split_strategy
        self.seed = seed
        self.n_splits = n_splits
        if logger is not None:
            self.logger = logger
        else:
            self.logger = TrainingLogger(
                enable_file_logging=getattr(config, 'SKLEARN_FILE_LOGGING_ENABLED', True),
            ).get_logger()
        self.tuning_verbose = tuning_verbose
        self.pipeline_builder = PipelineBuilder(
            tree_categorical_encoding=getattr(config, "TREE_CATEGORICAL_ENCODING", "ordinal"),
            tree_onehot_max_categories=getattr(config, "TREE_ONEHOT_MAX_CATEGORIES", None),
        )
        
        self.log_transformer = LogTransformer()

    def train(
        self,
        target: str,
        data: Dict,
        model_pipelines: Dict[str, Dict],
        targets: Optional[list] = None,
        ):
        """Train every model in ``model_pipelines`` for one :term:`target group`.

        Each model is trained inside a sub-run named ``<target group>_<model>``. A model that fails
        is reported and the others carry on, unless ``FAIL_ON_MODEL_ERROR`` is set.

        Parameters
        ----------
        target : str
            The group's name.
        data : dict
            What
            :meth:`ScikitDataModule.prepare
            <yg_eo_soilnet.datamodules.scikit.scikit_datamodule.ScikitDataModule.prepare>`
            returned.
        model_pipelines : dict
            The built models, from
            :meth:`ModelConfigFactory.build_model_configs
            <yg_eo_soilnet.models.config_fatories.model_config_factory.ModelConfigFactory.build_model_configs>`.
        targets : list of str, optional
            The targets in the group; read from its name unless given.

        Raises
        ------
        ValueError
            If a model entry is incomplete, or the group's targets disagree about the log transform.
        RuntimeError
            If every model failed and ``FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET`` is set.
        """
        self._validate_model_pipelines(model_pipelines)
        target_names = [str(name) for name in (targets or split_target_names(target))] or [str(target)]
        self._guard_uniform_log_transform(target_names)

        # Decided once for every model here, not per model: two models on one leaderboard must
        # have been fitted on the same rows for their scores to be comparable.
        fit_pool, calibration_data = self._resolve_fit_pool(data)
        X_train = fit_pool['X']
        X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include=['int64', 'int32']).columns})
        # One target stays a single column; several give the table a multi-target model needs.
        y_train = self._select_targets(fit_pool['y'], target_names)
        X_test = data['X_test']
        # Read from the test table itself: the training one has already been converted.
        X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include=['int64', 'int32']).columns})
        y_test = self._select_targets(data['y_test'], target_names)
        groups_train = data['groups_train'] if self.enable_clustering else None

        X_train, y_train, groups_train = TargetNanFilter().transform(X_train, y_train, groups_train)

        if X_test is not None and y_test is not None:
            X_test, y_test, _ = TargetNanFilter().transform(X_test, y_test)
        else:
            X_test, y_test = None, None

        # The same filter as the other splits, so the number of calibration rows reported is the
        # number actually used.
        if calibration_data is not None:
            X_calib = calibration_data['X'].astype(
                {col: 'float64' for col in calibration_data['X'].select_dtypes(include=['int64', 'int32']).columns}
            )
            y_calib = self._select_targets(calibration_data['y'], target_names)
            X_calib, y_calib, _ = TargetNanFilter().transform(X_calib, y_calib)
            calibration_data = {'X': X_calib, 'y': y_calib}

        min_feature_count = int(getattr(self.config, "MIN_FEATURE_COUNT", 10))
        min_valid_rows = max(5, min_feature_count, int(X_train.shape[1]))
        if y_train is not None and not y_train.empty and len(y_train) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_train)} valid training rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )
        if y_test is not None and hasattr(y_test, 'empty') and not y_test.empty and len(y_test) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_test)} valid test rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )

        # Every point, for the per-point export. Points whose target was never measured are kept:
        # they can still be predicted, and the export is meant to be complete.
        export_data = None
        if data.get('X_all') is not None and data.get('point_ids') is not None:
            X_all = data['X_all']
            export_data = {
                'X': X_all.astype(
                    {col: 'float64' for col in X_all.select_dtypes(include=['int64', 'int32']).columns}
                ),
                'point_ids': data['point_ids'],
            }

        if not self._should_skip_target(y_train, y_test, target):

            # The same for every target in the group; a mixed group was refused above.
            is_log_target = target_names[0] in self.columns_to_transform
            mlflow_logger = ChildRunLogger()

            trained_models = 0
            for model_name, config in model_pipelines.items():
                try:
                    self.logger.info(f"Training {model_name} for {target}")

                    # The sub-run is opened here, before the search, so everything the search
                    # reports is attached to it.
                    with start_child_run(f"{target}_{model_name}"):
                        self._train_one(
                            config=config,
                            model_name=model_name,
                            target=target,
                            target_names=target_names,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            groups_train=groups_train,
                            is_log_target=is_log_target,
                            mlflow_logger=mlflow_logger,
                            calibration_data=calibration_data,
                            export_data=export_data,
                        )
                    trained_models += 1
                except Exception as e:
                    self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                    self.logger.debug(traceback.format_exc())
                    if getattr(self.config, "FAIL_ON_MODEL_ERROR", False):
                        raise

            # Without this, a target whose models all failed would end in success with nothing in
            # its run.
            if trained_models == 0 and getattr(self.config, "FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET", True):
                raise RuntimeError(
                    f"All {len(model_pipelines)} model(s) failed to train for target {target!r}; "
                    "see the warnings above. Set FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET: false to continue anyway."
                )

        else:
            self.logger.warning(f"Skipping training for target {target} due to insufficient data.")

    def _train_one(
        self,
        *,
        config,
        model_name,
        target,
        target_names,
        X_train,
        y_train,
        X_test,
        y_test,
        groups_train,
        is_log_target,
        mlflow_logger,
        calibration_data=None,
        export_data=None,
    ):
        """Search, fit and record one model for one target group, inside its open sub-run.

        Raises
        ------
        ValueError
            If the entry's ``modeltype`` is not ``"ml"``.
        """
        model_seed = int(config.get("random_seed", self.seed))

        cv_splitter = CVSplitter(
            cv_strategy=self.split_strategy,
            n_splits=self.n_splits,
            random_state=model_seed,
        )
        splits = cv_splitter.create_splits(X_train, y_train, groups_train)

        model = config["model"]
        params = config.get("params", {})
        modeltype = config.get("modeltype", "ml")
        if modeltype == "ml":
            is_tree_model = self.pipeline_builder._is_tree_based_model(model)
            categorical_encoding = "ordinal" if is_tree_model else "onehot"
            categorical_cols = [
                col for col in self.config.CATEGORICAL_FEATURES
                if col in X_train.columns
            ]
            numeric_cols = [
                col for col in X_train.columns
                if col not in categorical_cols
            ]
            # Fill gaps, scale, encode categories, then the model - all as one object.
            pipeline = self.pipeline_builder.build(
                model,
                is_log_target,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
            )

            # A logged target wraps the model one layer deeper, so the grid's names follow.
            if is_log_target and bool(params):
                params = {
                    k.replace("model__", "model__regressor__") : v
                    for k, v in params.items()
                }
            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid= params if params is not None else {},
                cv=splits, refit=False,
                scoring= "neg_root_mean_squared_error",
                # Every processor by default; a model that loads a large file per process
                # limits this in its entry.
                n_jobs=int(config.get("search_n_jobs", -1)),
                return_train_score=True,
                verbose=self.tuning_verbose
            )

            search.fit(X_train, y_train)
            best_params = search.best_params_ if params is not None else {}
            cv_results = pd.DataFrame(search.cv_results_)

            # One search whatever the ensemble size: the members share the winning settings, so
            # searching once per member would repeat the most expensive part of the run.
            if uncertainty_enabled_for(self.config, model_name):
                best_model = self._fit_ensemble(
                    pipeline=pipeline,
                    best_params=best_params,
                    model=model,
                    model_name=model_name,
                    target=target,
                    target_names=target_names,
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    model_seed=model_seed,
                    splits=splits,
                    calibration_data=calibration_data,
                )
            else:
                best_model = clone(pipeline)
                if params is not None:
                    best_model.set_params(**best_params)
                best_model.fit(X_train, y_train)

            if not any(cv_results.get("params", [])):
                cv_results["params"] = [best_model.get_params()]
            #Evaluate model
            param_names = list(params.keys()) if params else []
            plot_func = {}
            if len(param_names) > 1:
                cv_plot = "yg_eo_soilnet.plot_utils.cv_parallel_coordinates"
            elif len(param_names) == 1:
                cv_plot = "yg_eo_soilnet.plot_utils.cv_val_curve"
            else:
                cv_plot = None  # No hyperparameters to plot
                param_names = list(best_model.get_params().keys())

            if cv_plot is not None:
                plot_func.update({cv_plot: {"args": [cv_results]}})
            mlflow_logger.log_child_run(
                config=self.config,
                search=search,
                cv_results=cv_results,
                best_model=best_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                target=target,
                targets=target_names,
                param_names=param_names,
                model_name=model_name,
                plot_functions=plot_func,
                extra_params={
                    "categorical_encoding": categorical_encoding,
                },
                export_data=export_data,
                )
        else:
            raise ValueError(f"Unknown modeltype: {modeltype}")

    # --- uncertainty --------------------------------------------------------

    def _resolve_fit_pool(self, data: Dict) -> tuple[Dict, Optional[Dict]]:
        """Decide which rows are fitted on, and which are kept back to calibrate the intervals.

        Normally everything: the :term:`fit pool` is the training and validation points together,
        because the hyperparameters are chosen by cross-validation inside it.

        :term:`Conformal <conformal>` intervals need points the model has not seen, so with
        ``uncertainty.calibration.source: val`` the models fit on the training points only and the
        validation points are reserved. That costs about 15% of the training rows - calibrating on
        rows the model was fitted on would make the intervals too narrow and the promised coverage
        meaningless. ``uncertainty_fit_pool`` is recorded with the run, so the slightly worse
        ``rmse_test`` is not mistaken for something going wrong.

        Returns
        -------
        fit_pool : dict
            The covariates and targets to fit on.
        calibration_data : dict or None
            The rows reserved for calibration, if any.
        """
        default_pool = {'X': data['X_train'], 'y': data['y_train']}

        if not bool(getattr(self.config, "UNCERTAINTY_ENABLED", False)):
            return default_pool, None
        # Only the interval methods that need held-out rows pay for them: the others turn the
        # spread into an interval by arithmetic.
        if not needs_calibration_set(
            getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal")
        ):
            return default_pool, None
        if str(getattr(self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val")).lower() != "val":
            # cv_oof calibrates on the folds the search already ran, so it keeps every row.
            return default_pool, None

        train_only = data.get('X_train_only')
        val_features = data.get('X_val')
        if train_only is None or val_features is None or len(val_features) == 0:
            self.logger.warning(
                "uncertainty.calibration.source is 'val' but the split carries no separate val "
                "rows; falling back to the full fit pool and calibrating out-of-fold instead."
            )
            return default_pool, None

        self.logger.info(
            f"Uncertainty calibration reserves the val split: fitting on {len(train_only)} rows "
            f"and calibrating on {len(val_features)}."
        )
        return (
            {'X': train_only, 'y': data['y_train_only']},
            {'X': val_features, 'y': data['y_val']},
        )

    def _fit_ensemble(
        self,
        *,
        pipeline,
        best_params,
        model,
        model_name,
        target,
        target_names,
        X_train,
        y_train,
        groups_train,
        model_seed,
        splits,
        calibration_data,
    ) -> EnsembleRegressor:
        """Fit the :term:`ensemble` members, calibrate the intervals, and return them as one model.

        Each member is trained from a different seed, and from resampled rows when the estimator
        would otherwise fit identically every time. Every member gets a sub-run of its own, tagged
        :data:`MEMBER_RUN_KIND` so the leaderboard can tell members from results.

        Returns
        -------
        EnsembleRegressor
            Predicts the members' average, with their spread as the uncertainty.
        """
        n_members = int(getattr(self.config, "UNCERTAINTY_N_MEMBERS", 5))
        stride = int(getattr(self.config, "UNCERTAINTY_SEED_STRIDE", 1000))
        seeds = member_seeds(model_seed, n_members, stride)
        bootstrap = should_bootstrap(
            model, str(getattr(self.config, "UNCERTAINTY_BOOTSTRAP", "auto"))
        )

        members = []
        for index, seed in enumerate(seeds):
            with start_child_run(
                f"{target}_{model_name}_member{index}",
                tags={
                    "run_kind": MEMBER_RUN_KIND,
                    "target": target,
                    "model_name": model_name,
                    "ensemble_member": str(index),
                    "ensemble_seed": str(seed),
                },
            ):
                member = clone(pipeline)
                if best_params:
                    member.set_params(**best_params)
                self._seed_member(member, seed)

                member_X, member_y = X_train, y_train
                if bootstrap:
                    positions = bootstrap_indices(len(X_train), seed)
                    member_X = X_train.iloc[positions]
                    member_y = y_train.iloc[positions]

                member.fit(member_X, member_y)
                mlflow.log_params(
                    {
                        "ensemble_member": index,
                        "ensemble_seed": seed,
                        "ensemble_bootstrapped": bootstrap,
                        "ensemble_n_rows": len(member_X),
                    }
                )
            members.append(member)

        # Which rows these members fitted on. Recorded because it changes what rmse_test means:
        # a calibrated ensemble trained on about 15% fewer rows than an ordinary run.
        mlflow.log_params(
            {
                "uncertainty_fit_pool": "train_only" if calibration_data is not None else "train_val",
                "uncertainty_calibration_source": getattr(
                    self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val"
                ),
                "uncertainty_n_train_rows": len(X_train),
            }
        )

        ensemble = EnsembleRegressor(
            members=members,
            target_names=target_names,
            member_seeds=seeds,
            bootstrapped=bootstrap,
        )
        ensemble.calibrators = self._calibrate(
            ensemble=ensemble,
            pipeline=pipeline,
            best_params=best_params,
            target_names=target_names,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            splits=splits,
            calibration_data=calibration_data,
        )
        self._warn_if_ensemble_collapsed(ensemble, X_train, model_name, bootstrap)
        return ensemble

    def _calibrate(
        self,
        *,
        ensemble,
        pipeline,
        best_params,
        target_names,
        X_train,
        y_train,
        groups_train,
        splits,
        calibration_data,
    ) -> dict:
        """Build one :term:`prediction interval` estimator per target.

        ``sigma`` and ``gaussian`` turn the spread into an interval by arithmetic and need no data.
        ``conformal`` measures how far the predictions actually fall from the measurements, on the
        reserved rows or, failing those, on the cross-validation folds.

        Returns
        -------
        dict of str to object
            One estimator per target.
        """
        method = normalize_method(getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal"))
        alpha = float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05))

        if not needs_calibration_set(method):
            return build_interval_estimators(
                method,
                target_names,
                alpha=alpha,
                k=float(getattr(self.config, "UNCERTAINTY_INTERVAL_K", 1.0)),
            )

        if calibration_data is not None:
            prediction = ensemble.predict_uncertainty(calibration_data['X'])
            return fit_calibrators(
                prediction,
                calibration_data['y'],
                target_names,
                alpha=alpha,
                logger=self.logger,
            )

        return self._calibration_from_out_of_fold(
            ensemble=ensemble,
            pipeline=pipeline,
            best_params=best_params,
            target_names=target_names,
            X_train=X_train,
            y_train=y_train,
            splits=splits,
            alpha=alpha,
        )

    def _calibration_from_out_of_fold(
        self,
        *,
        ensemble,
        pipeline,
        best_params,
        target_names,
        X_train,
        y_train,
        splits,
        alpha,
    ) -> dict:
        """Calibrate on the cross-validation folds instead, keeping every row for fitting.

        An approximation, and the reason reserving the validation points is the default: the errors
        come from a model fitted on part of the pool, while the spread comes from the full ensemble,
        so the two do not describe the same model. It errs wide rather than narrow, but it does not
        carry the guarantee the reserved rows give.

        Returns
        -------
        dict of str to object
            One interval estimator per target.
        """
        estimator = clone(pipeline)
        if best_params:
            estimator.set_params(**best_params)

        out_of_fold = np.asarray(
            cross_val_predict(estimator, X_train, y_train, cv=splits, n_jobs=1)
        )
        if out_of_fold.ndim == 1:
            out_of_fold = out_of_fold.reshape(-1, 1)

        sigma = ensemble.predict_uncertainty(X_train).total_std
        prediction = aggregate([out_of_fold], [np.zeros_like(out_of_fold)])
        # One set of predictions has no spread of its own; the spread scaled here is the
        # ensemble's, measured on the same rows.
        prediction = type(prediction)(
            mean=prediction.mean,
            epistemic_std=sigma,
            aleatoric_std=np.zeros_like(sigma),
        )

        return fit_calibrators(
            prediction, y_train, target_names, alpha=alpha, logger=self.logger
        )

    @staticmethod
    def _seed_member(member, seed: int) -> None:
        """Give one ensemble member its own seed, wherever the estimator sits in the pipeline.

        An estimator that takes no seed is varied by resampling its rows instead.
        """
        available = member.get_params(deep=True)
        for key in ("model__random_state", "model__regressor__random_state"):
            if key in available:
                member.set_params(**{key: int(seed)})
                return

    def _warn_if_ensemble_collapsed(self, ensemble, X_train, model_name: str, bootstrap: bool) -> None:
        """Warn when every member predicts the same thing, instead of reporting perfect confidence.

        A spread of exactly zero looks like an extremely confident model everywhere downstream: the
        error bars vanish and the coverage reads 0 or 1, with nothing to say the ensemble never
        really formed.
        """
        sample = X_train.iloc[: min(len(X_train), 256)]
        spread = float(np.max(ensemble.predict_uncertainty(sample).epistemic_std))
        if spread > 0.0:
            return
        self.logger.warning(
            f"Every ensemble member of {model_name} predicts identically (epistemic std is exactly "
            f"0), so the ensemble carries no model uncertainty"
            + (
                ". The members were bootstrapped, so this points at an estimator whose fit does not "
                "depend on the sampled rows."
                if bootstrap
                else " because bootstrapping is disabled and this estimator ignores its seed. Set "
                "uncertainty.bootstrap: auto."
            )
        )

    @staticmethod
    def _select_targets(frame: pd.DataFrame, target_names: list):
        """The group's target columns: one column on its own, or a table for several."""
        if len(target_names) == 1:
            return frame[target_names[0]]
        return frame[list(target_names)]

    def _guard_uniform_log_transform(self, target_names: list) -> None:
        """Refuse a group whose targets disagree about the log transform.

        One model applies one transform to everything it predicts, so a group where some targets
        are listed in ``COLUMNS_TO_TRANSFORM`` and others are not cannot be trained honestly.

        Raises
        ------
        ValueError
            Naming the targets that disagree, and the two ways out.
        """
        if len(target_names) < 2:
            return
        logged = [name for name in target_names if name in self.columns_to_transform]
        if logged and len(logged) != len(target_names):
            raise ValueError(
                f"Targets {sorted(target_names)} are fitted jointly but disagree about the log "
                f"transform: {sorted(logged)} are in COLUMNS_TO_TRANSFORM and the rest are not. "
                "One fit applies one transform. Either align COLUMNS_TO_TRANSFORM or set "
                "MULTI_TARGET_MODE: per_target."
            )

    def _should_skip_target(self, y_train: pd.Series, y_test: Optional[pd.Series], target: str) -> bool:
        """Whether a target has too little measured data to train on; reports the row counts."""
        n_train = len(y_train) if y_train is not None else 0
        n_test = len(y_test) if y_test is not None else 0
        self.logger.info(f"Training for target: {target} — {n_train} train samples, {n_test} test samples.")
        if y_train is None or y_train.empty or (y_test is not None and hasattr(y_test, 'empty') and y_test.empty):
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return True
        return False

    def _validate_model_pipelines(self, model_pipelines: Dict[str, Dict]):
        """Check every model entry carries an estimator and a grid of settings to search."""
        for model_name, config in model_pipelines.items():
            if "model" not in config or "params" not in config:
                raise ValueError(f"Model pipeline '{model_name}' must have 'model' and 'params' keys.")

