"""The training machinery every deep-learning model here shares.

Choosing the loss, running the training, validation and test steps, reporting the scores each epoch,
converting predictions back to the target's own units, and building the optimizer. Only how a model
reads its inputs differs, so that is all a model class defines.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from lightning.pytorch import LightningModule

from yg_eo_soilnet.models.lightningmodules.losses import (
    STRUCTURAL_LOSSES,
    build_loss_fn,
    inverse_transform_targets,
)


def batch_get(batch: Any, key: str, default=None):
    """Read one entry from a batch, whether it is a dict or an object with named fields."""
    if isinstance(batch, Mapping):
        return batch.get(key, default)
    if hasattr(batch, "get"):
        return batch.get(key, default)
    if hasattr(batch, "__getitem__"):
        try:
            return batch[key]
        except Exception:
            return default
    return getattr(batch, key, default)


def as_float_list(value: Any) -> Optional[list[float]]:
    """Return any array of numbers as a flat list of plain floats.

    A model's settings are saved in its :term:`checkpoint`, which can only hold plain values.

    Examples
    --------
    >>> as_float_list([1, 2.5])
    [1.0, 2.5]
    >>> as_float_list(None) is None
    True
    """
    if value is None:
        return None
    return [float(item) for item in torch.as_tensor(value, dtype=torch.float32).flatten().tolist()]


def as_float_matrix(value: Any) -> Optional[list[list[float]]]:
    """Like :func:`as_float_list` for a table of numbers, keeping its rows.

    Raises
    ------
    ValueError
        If the value is not a table.

    Examples
    --------
    >>> as_float_matrix([[1, 0], [0, 1]])
    [[1.0, 0.0], [0.0, 1.0]]
    """
    if value is None:
        return None
    rows = torch.as_tensor(value, dtype=torch.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix; got shape {tuple(rows.shape)}.")
    return [[float(item) for item in row] for row in rows.tolist()]


class SoilRegressionLightningBase(LightningModule):
    """What every deep-learning model here inherits: loss, steps, scores and optimizer.

    A model class adds how it reads its inputs - its ``forward`` - and calls
    ``_init_regression_targets`` from its constructor with the settings below.

    The model is trained in standardized units (and log-transformed ones, if that is switched on);
    :meth:`predict_step` converts its predictions back to the target's own units, so everything the
    run reports is comparable with the lab measurements.
    """

    def _init_regression_targets(
        self,
        *,
        target_dim: int,
        target_mean: Optional[Any],
        target_scale: Optional[Any],
        target_transform: Optional[str],
        loss_name: str,
        huber_delta: float,
        # Settings for the losses that read across targets; ignored by the others.
        loss_base: str = "mse",
        loss_lambda: float = 0.1,
        loss_shrinkage: float = 0.05,
        loss_min_batch: int = 16,
        cosine_space: str = "original",
        target_covariance: Optional[Any] = None,
        learning_rate: float,
        optimizer_name: str,
        weight_decay: float,
        scheduler_type: str,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        scheduler_monitor: str,
        target_names: Optional[Any] = None,
        predict_variance: bool = False,
        beta_nll: float = 0.5,
    ) -> None:
        """Set up the loss, the target statistics, the optimizer settings and the score tracking.

        Called from a model's constructor.

        Parameters
        ----------
        target_dim : int
            How many targets the model predicts.
        target_mean, target_scale : array-like, optional
            The target standardization statistics, from the datamodule.
        target_transform : {None, "log1p"}, optional
            Whether the targets were log-transformed.
        loss_name : str
            What to minimize; see :func:`~yg_eo_soilnet.models.lightningmodules.losses.build_loss_fn`.
        huber_delta : float
            Where ``huber`` and ``smooth_l1`` switch from squared to absolute error.
        loss_base : str, default "mse"
            The per-target loss a structural loss adds its penalty to.
        loss_lambda : float, default 0.1
            How heavily that penalty counts.
        loss_shrinkage : float, default 0.05
            For ``mahalanobis``.
        loss_min_batch : int, default 16
            For ``correlation_penalty``.
        cosine_space : str, default "original"
            For ``cosine``.
        target_covariance : array-like, optional
            How the training targets vary together, needed by the structural losses.
        learning_rate : float
            How large a step training takes.
        optimizer_name : {"adam", "adamw"}
            Which optimizer.
        weight_decay : float
            How strongly large weights are penalized.
        scheduler_type : str
            ``"plateau"`` lowers the learning rate when the watched score stops improving; anything
            else leaves it fixed.
        scheduler_factor : float
            What the learning rate is multiplied by then.
        scheduler_patience : int
            How many epochs without improvement to wait first.
        scheduler_min_lr : float
            The lowest the learning rate may go.
        scheduler_monitor : str
            Which score to watch, normally ``"val_loss"``.
        target_names : sequence of str, optional
            The target names, used to name the per-target scores.
        predict_variance : bool, default False
            Predict a spread alongside each value; see :term:`variance head`.
        beta_nll : float, default 0.5
            How the variance head balances fitting the values against fitting their spread; 0 is the
            plain likelihood, 1 weights every point equally.

        Raises
        ------
        ValueError
            If ``target_transform`` is unknown, or a structural loss is combined with a variance
            head, which would silently ignore it.
        """
        self.target_dim = int(target_dim)
        # With a variance head the model predicts two numbers per target - the value and how
        # uncertain it is - and is trained accordingly. That is what makes an interval narrow where
        # the data is clean and wide where it is noisy, which an ensemble alone cannot tell.
        self.predict_variance = bool(predict_variance)
        self.head_output_dim = self.target_dim * (2 if self.predict_variance else 1)
        self.beta_nll = float(beta_nll)
        # Used to name the per-target scores, so a model predicting several targets reports an R²
        # for each rather than only an average.
        if not getattr(self, "target_names", None):
            self.target_names = [str(name) for name in (target_names or [])]
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.scheduler_type = str(scheduler_type).lower()
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_patience = max(0, int(scheduler_patience))
        self.scheduler_min_lr = float(scheduler_min_lr)
        self.scheduler_monitor = str(scheduler_monitor)

        # The target statistics from the datamodule. Training happens in standardized units;
        # predict_step converts back, so the scores are in the target's own units.
        self.register_buffer(
            "target_mean",
            torch.zeros(self.target_dim) if target_mean is None else torch.as_tensor(target_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_scale",
            torch.ones(self.target_dim) if target_scale is None else torch.as_tensor(target_scale, dtype=torch.float32),
            persistent=True,
        )
        # Saved with the weights: a checkpoint that lost these would quietly report predictions in
        # standardized units instead of the target's own.
        self.register_buffer(
            "targets_are_standardized",
            torch.tensor(target_mean is not None and target_scale is not None),
            persistent=True,
        )
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        self.register_buffer(
            "targets_are_log1p",
            torch.tensor(self.target_transform == "log1p"),
            persistent=True,
        )
        # Likewise: without it a restored variance head would report its spreads as if they were
        # predictions of targets that do not exist.
        self.register_buffer(
            "head_predicts_variance",
            torch.tensor(self.predict_variance),
            persistent=True,
        )

        # val_loss decides when training stops and which checkpoint is kept. It is only comparable
        # between runs using the same loss.
        self.loss_name = str(loss_name).lower()
        self.huber_delta = float(huber_delta)
        # A variance head is trained on its own loss, so a structural one would be ignored without
        # a word. Refuse instead of reporting a loss the run never minimized.
        if self.predict_variance and self.loss_name in STRUCTURAL_LOSSES:
            raise ValueError(
                f"loss_name '{self.loss_name}' cannot be combined with a heteroscedastic head: "
                f"predict_variance replaces the point loss with beta-NLL. Set "
                f"uncertainty.heteroscedastic to false, or use a point loss."
            )
        # Held as part of the model, so whatever the loss carries moves to the GPU with it.
        self.loss_fn = self._build_loss_fn(
            self.loss_name,
            self.huber_delta,
            loss_base=loss_base,
            loss_lambda=loss_lambda,
            loss_shrinkage=loss_shrinkage,
            loss_min_batch=loss_min_batch,
            cosine_space=cosine_space,
            target_dim=self.target_dim,
            target_covariance=target_covariance,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=self.target_transform,
        )
        # Running totals for the per-epoch scores, one set per stage.
        self._metric_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _build_loss_fn(loss_name: str, huber_delta: float, **kwargs):
        """Build the loss; see :func:`~yg_eo_soilnet.models.lightningmodules.losses.build_loss_fn`."""
        return build_loss_fn(loss_name, huber_delta=huber_delta, **kwargs)

    # --- steps -------------------------------------------------------------

    #: Limits on the spread a :term:`variance head` may predict. Without them the model can claim
    #: near-certainty about a point it happens to fit early, and the training then blows up. The
    #: range covers every plausible noise level in standardized units.
    LOG_VARIANCE_MIN = -10.0
    LOG_VARIANCE_MAX = 10.0

    def _split_head_output(self, raw: torch.Tensor):
        """Split the model's output into the predicted values and their spread, if it predicts one."""
        if not self.predict_variance:
            return raw, None
        mean, log_variance = raw[..., : self.target_dim], raw[..., self.target_dim :]
        return mean, log_variance.clamp(self.LOG_VARIANCE_MIN, self.LOG_VARIANCE_MAX)

    def _beta_nll_loss(self, mean, log_variance, targets):
        """The loss a :term:`variance head` is trained on.

        It rewards predicting a value close to the measurement *and* being honest about how far off
        it might be. The plain version of this loss has a known failing: a point the model is
        unsure about barely counts, so it never learns to fit it, which then justifies the
        uncertainty. ``beta_nll`` corrects that; 0.5 is the usual recommendation.
        """
        variance = torch.exp(log_variance)
        negative_log_likelihood = 0.5 * (log_variance + (targets - mean) ** 2 / variance)
        if self.beta_nll > 0.0:
            negative_log_likelihood = negative_log_likelihood * variance.detach() ** self.beta_nll
        return negative_log_likelihood.mean()

    def _shared_step(self, batch: Any, stage: str):
        """One training, validation or test step: predict, score, record.

        Raises
        ------
        KeyError
            If the batch carries no measured targets.
        ValueError
            If a target, a prediction or the loss is missing or infinite.
        """
        raw_output = self.forward(batch)
        predictions, log_variance = self._split_head_output(raw_output)
        targets = batch_get(batch, "y")
        if targets is None:
            raise KeyError("Batch is missing 'y'")
        targets = targets.to(device=predictions.device, dtype=predictions.dtype)

        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite target values encountered during {stage} step.")
        if not torch.isfinite(predictions).all():
            raise ValueError(f"Non-finite prediction values encountered during {stage} step.")
        if log_variance is None:
            loss = self.loss_fn(predictions, targets)
        else:
            if not torch.isfinite(log_variance).all():
                raise ValueError(f"Non-finite predicted variance encountered during {stage} step.")
            loss = self._beta_nll_loss(predictions, log_variance, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered during {stage} step.")

        # The real number of points: the epoch average is weighted by it, so a short last batch
        # must not count as much as a full one.
        batch_size = int(targets.shape[0]) if targets.ndim else 1
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=batch_size,
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )
        # A loss with a penalty reports both halves, so a penalty weight that does nothing can be
        # told from one that has swamped the accuracy term.
        for name, value in getattr(self.loss_fn, "last_components", {}).items():
            self.log(
                f"{stage}_loss_{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        self._accumulate_metrics(stage, predictions.detach(), targets.detach())
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """Score one training batch; its loss is what the weights are updated from."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        """Score one validation batch; ``val_loss`` decides when training stops."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        """Score one test batch, once training has finished."""
        return self._shared_step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Predict, in the target's own units.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The predictions; or ``(predictions, sigma)`` when the model has a :term:`variance head`,
            where sigma is the predicted spread, also in the target's own units.
        """
        mean, log_variance = self._split_head_output(self.forward(batch))
        if log_variance is None:
            return self.inverse_transform_targets(mean)
        sigma = torch.exp(0.5 * log_variance)
        # The spread is converted using the standardized prediction, so it is worked out before
        # that prediction is converted.
        return (
            self.inverse_transform_targets(mean),
            self.inverse_transform_sigma(sigma, mean),
        )

    # --- scores reported each epoch ----------------------------------------
    # R², and how much the predictions vary compared with the measurements. Together they say
    # whether a model is learning or has settled for predicting the average of everything.

    def _metric_target_names(self) -> list[str]:
        """One name per predicted target, for naming the scores."""
        names = [str(name) for name in (getattr(self, "target_names", None) or [])]
        if len(names) == self.target_dim:
            return names
        return [f"target_{index}" for index in range(self.target_dim)]

    def _accumulate_metrics(self, stage: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch to the running totals the epoch scores are computed from."""
        # Totalled per target, not pooled across them: pooling would hide a target the model has
        # given up on behind one it predicts well.
        predictions = predictions.double().reshape(-1, self.target_dim)
        targets = targets.double().reshape(-1, self.target_dim)
        state = self._metric_state.setdefault(stage, {})
        if not state:
            # Kept on the processor, so the totals do not hold GPU memory for a whole epoch.
            zeros = torch.zeros(self.target_dim, dtype=torch.float64)
            state.update(
                {
                    "n": 0.0,
                    "sum_y": zeros.clone(),
                    "sum_y2": zeros.clone(),
                    "sum_p": zeros.clone(),
                    "sum_p2": zeros.clone(),
                    "sse": zeros.clone(),
                }
            )
        state["n"] += float(targets.shape[0])
        state["sum_y"] += targets.sum(dim=0).cpu()
        state["sum_y2"] += (targets**2).sum(dim=0).cpu()
        state["sum_p"] += predictions.sum(dim=0).cpu()
        state["sum_p2"] += (predictions**2).sum(dim=0).cpu()
        state["sse"] += ((predictions - targets) ** 2).sum(dim=0).cpu()

    def _log_epoch_metrics(self, stage: str) -> None:
        """Report R² and the prediction spread for the epoch, per target and averaged."""
        state = self._metric_state.pop(stage, None)
        if not state or state["n"] < 2:
            return

        count = state["n"]
        target_variance = state["sum_y2"] / count - (state["sum_y"] / count) ** 2
        prediction_variance = state["sum_p2"] / count - (state["sum_p"] / count) ** 2

        names = self._metric_target_names()
        r2_scores: list[float] = []
        std_ratios: list[float] = []
        for index, name in enumerate(names):
            variance = float(target_variance[index])
            # A target that never varies has nothing to explain, so R² is undefined rather than
            # zero. Skipped for that target alone.
            if variance <= 1e-12:
                continue
            # In training units this is exactly 1 minus the mean squared error, which is how a
            # run's health can be read straight off its loss.
            r2 = 1.0 - (float(state["sse"][index]) / count) / variance
            std_ratio = (max(float(prediction_variance[index]), 0.0) ** 0.5) / (variance**0.5)
            r2_scores.append(r2)
            std_ratios.append(std_ratio)
            if len(names) > 1:
                self.log(f"{stage}_r2_{name}", r2, on_step=False, on_epoch=True)
                self.log(f"{stage}_pred_std_ratio_{name}", std_ratio, on_step=False, on_epoch=True)

        if not r2_scores:
            return
        # The unnamed pair is the average across targets, comparable with a single-target run.
        self.log(
            f"{stage}_r2",
            sum(r2_scores) / len(r2_scores),
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )
        self.log(
            f"{stage}_pred_std_ratio",
            sum(std_ratios) / len(std_ratios),
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_start(self) -> None:
        """Clear the training totals at the start of an epoch."""
        self._metric_state.pop("train", None)

    def on_validation_epoch_start(self) -> None:
        """Clear the validation totals at the start of an epoch."""
        self._metric_state.pop("val", None)

    def on_test_epoch_start(self) -> None:
        """Clear the test totals before scoring."""
        self._metric_state.pop("test", None)

    def on_train_epoch_end(self) -> None:
        """Report the training scores for the epoch."""
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        """Report the validation scores for the epoch."""
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        """Report the test scores."""
        self._log_epoch_metrics("test")

    # --- preprocessing state ------------------------------------------------

    def attach_preprocessing_state(self, state: Mapping[str, Any] | None) -> None:
        """Keep the datamodule's fitted input statistics on the model.

        They go into the checkpoint alongside the weights, which is what lets a saved model prepare
        raw data by itself. They also carry the column names the SHAP figures label their rows with.

        Parameters
        ----------
        state : mapping or None
            What :meth:`SoilSequenceDataModule.preprocessing_state
            <yg_eo_soilnet.datamodules.sequence.sequence_datamodule.SoilSequenceDataModule.preprocessing_state>`
            returned.
        """
        if not state:
            return

        payload = dict(state)
        self.preprocessing_state = payload

        # Kept as attributes too: the explanations and the serving wrapper read them constantly.
        self.static_feature_names = list(payload.get("static_feature_names") or [])
        self.modality_column_names = {
            str(name): list(columns) for name, columns in (payload.get("modality_column_names") or {}).items()
        }
        # Names only: they label the figures and change nothing about the model's shape.
        self.context_feature_names = list(payload.get("context_feature_names") or [])
        self.coord_names = list(payload.get("coord_names") or [])

    def get_preprocessing_state(self) -> dict:
        """The fitted input statistics, or an empty dict when the model has none."""
        return dict(getattr(self, "preprocessing_state", None) or {})

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Write the fitted input statistics into the checkpoint, beside the weights."""
        state = self.get_preprocessing_state()
        if state:
            checkpoint["preprocessing_state"] = state

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Read those statistics back when a checkpoint is loaded."""
        self.attach_preprocessing_state(checkpoint.get("preprocessing_state"))

    # --- target inversion and optimizer ------------------------------------

    def inverse_transform_targets(self, predictions):
        """Convert predictions from training units back to the target's own units."""
        return inverse_transform_targets(
            predictions,
            mean=self.target_mean,
            scale=self.target_scale,
            standardized=bool(self.targets_are_standardized),
            log1p=bool(self.targets_are_log1p),
        )

    def inverse_transform_sigma(self, sigma, predictions):
        """Convert a predicted spread from training units back to the target's own units.

        Undoing the standardization only rescales it. Undoing the log transform does not: that
        transform stretches the scale by a different amount at every value, so the spread is
        converted using the slope at this point's own prediction. On a log-transformed target the
        spread therefore grows with the prediction - a wide prediction is uncertain by more g/kg
        than a small one, even at the same relative uncertainty.

        Parameters
        ----------
        sigma : torch.Tensor
            The predicted spread, in training units.
        predictions : torch.Tensor
            The predictions **in training units**, not the converted ones.

        Returns
        -------
        torch.Tensor
            The spread in the target's own units.
        """
        if bool(self.targets_are_standardized):
            scale = self.target_scale.to(sigma.device)
            sigma = sigma * scale
            transformed = predictions * scale + self.target_mean.to(predictions.device)
        else:
            transformed = predictions

        if bool(self.targets_are_log1p):
            sigma = sigma * torch.exp(transformed / 10.0) / 10.0
        return sigma

    def configure_optimizers(self):
        """Build the optimizer, and the learning-rate schedule when one is configured.

        Returns
        -------
        torch.optim.Optimizer or dict
            A dict when ``scheduler_type`` asks for the learning rate to be lowered once the watched
            score stops improving.
        """
        if self.optimizer_name in {"adamw", "adam_w"}:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type in {"plateau", "reducelronplateau", "reduce_on_plateau"}:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
