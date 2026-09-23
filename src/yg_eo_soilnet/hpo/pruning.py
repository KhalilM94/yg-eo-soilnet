"""Abandon a :term:`trial` that is clearly going nowhere, before it finishes.

The score is reported after every epoch, and a trial well behind the ones already finished is
stopped there - which is what lets a study try many more combinations in the same time.
"""
from __future__ import annotations

from typing import Any

import optuna

try:  # pragma: no cover - optional dependency, mirrors lightning_trainer.py
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover
    LightningCallback = object  # type: ignore[assignment]


def metric_to_float(value: Any) -> float | None:
    """A score Lightning reported as a plain number, or None when it is not one."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.item() if getattr(value, "ndim", 0) == 0 else value.numpy()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # A pruner fed NaN would rank it against real values; treat it as "no reading".
    return None if result != result else result


class OptunaPruningCallback(LightningCallback):
    """Reports the score each epoch and stops a trial the study judges hopeless.

    Written here rather than taken from a library so it follows this project's own settings for which
    score to watch and which direction is better.

    Parameters
    ----------
    trial : optuna.Trial
        The trial being run.
    monitor : str
        Which score to report, usually ``val_loss``.
    mode : {"min", "max"}
        Whether lower or higher is better.
        """

    def __init__(self, trial: optuna.Trial, monitor: str, mode: str = "min", *, report: bool = True):
        """Hold the trial, the score to watch and which direction is better."""
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}.")
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        # When a trial averages several seeds, only the first repeat reports: Optuna keeps one
        # intermediate value per step, so later repeats would overwrite the first one's curve.
        self.report = report
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self._reported_epoch: int | None = None

    def _is_better(self, value: float) -> bool:
        """Whether one score beats another, in the configured direction."""
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def on_validation_end(self, trainer, pl_module) -> None:
        # on_validation_end, NOT on_validation_epoch_end. Lightning runs callback
        # on_validation_epoch_end hooks *before* the LightningModule's, and `val_r2` is logged in
        # the module's hook (_regression_base._log_epoch_metrics). Reading it there yields the
        # previous epoch's value - None on epoch 0 - so every report, the best value and the best
        # epoch would be off by one. `val_loss` is logged in validation_step and so is current
        # either way, which is what made this easy to miss.
        """Report this epoch's score, and stop the trial if it is hopeless."""
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        if self._reported_epoch == epoch:
            return

        value = metric_to_float(trainer.callback_metrics.get(self.monitor))
        if value is None:
            # `val_r2` is skipped by _log_epoch_metrics when the batch is degenerate (n < 2, or a
            # constant target). A gap in the series is not grounds to prune.
            return

        self._reported_epoch = epoch
        if self._is_better(value):
            self.best_value, self.best_epoch = value, epoch

        if not self.report:
            return
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at epoch {epoch} ({self.monitor}={value:.5f})")
