"""Train one :term:`trial` and return its score.

Deliberately not the ordinary trainer, which validates, tests, predicts, draws figures, saves a
model and opens a run - worth doing once per model, not a few hundred times. This trains, watches
the score, and stops early when a trial is clearly going nowhere.
"""

from __future__ import annotations

import gc
import importlib
import logging
import sys
from dataclasses import dataclass

import optuna

from yg_eo_soilnet.hpo.pruning import OptunaPruningCallback
from yg_eo_soilnet.hpo.search_space import Objective

# Everything a trial must not do, whatever the registry says.
TRIAL_TRAINER_OVERRIDES = {
    "enable_checkpointing": False,  # hundreds of trials must not each write a .ckpt
    "logger": False,  # no lightning_logs/version_N per trial
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


class UnrecoverableAcceleratorError(RuntimeError):
    """Raised when the GPU itself has died and no later trial can succeed.

    Kept apart from an ordinary trial failure on purpose: a bad corner of the search space should cost
    one trial, while a dead GPU should stop the study rather than let it record hundreds of failures.
    """


@dataclass
class TrialResult:
    """What one trial produced.

    Attributes
    ----------
    value : float or None
        The score the study is optimizing, or None when the trial failed.
    epochs : int
        How many epochs it ran.
    pruned : bool
        Whether it was abandoned early as hopeless.
    """

    value: float
    best_epoch: int | None
    epochs_run: int


def cuda_context_is_dead() -> bool:
    """Whether the GPU is still usable, tested by asking it for a little memory.

    Asked rather than guessed from the error message, which names whatever call came next rather than
    the one that failed.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return False
    try:
        torch.zeros(1, device="cuda").add_(1).cpu()
        return False
    except Exception:
        return True


def raise_if_accelerator_is_dead(exc: BaseException, trial_number: int) -> None:
    """Turn a dead GPU into :class:`UnrecoverableAcceleratorError`; let anything else pass.

    Running out of memory needs nothing special: it leaves the GPU usable, so the trial fails and the
    study carries on.
    """
    if not cuda_context_is_dead():
        return
    raise UnrecoverableAcceleratorError(
        f"The CUDA context died during trial {trial_number}, so no later trial in this process can "
        f"succeed. Every completed trial is safe in the study storage. Restore the GPU and re-run - "
        f"on WSL2 that means `wsl --shutdown` from Windows, then check `nvidia-smi` before starting."
    ) from exc


def release_dataloader_workers() -> None:
    """Free the previous trial's data-loading processes before the next one starts.

    They are not released on their own, and a few hundred trials would otherwise leave hundreds of
    processes behind.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def silence_lightning() -> None:
    """Quieten the per-trial messages that would otherwise scroll the study off the screen."""
    for name in (
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators",
        # seed_everything logs "Seed set to N" per trial, from the fabric namespace.
        "lightning.fabric",
        "lightning.fabric.utilities.seed",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrialRunner:
    """Trains one trial and reports its score.

    Parameters
    ----------
    objective : Objective
        Which score to watch, and whether lower or higher is better.
    extra_callbacks : list, optional
        Anything extra to attach to the training, such as the progress display.
    """

    def __init__(self, objective: Objective, *, logger=None, fail_fast: bool = False, extra_callbacks=None):
        """Hold what to watch and anything extra to attach to each trial's training."""
        self.objective = objective
        self.logger = logger
        self.fail_fast = fail_fast
        # Lightning callbacks that observe a trial without steering it - the progress display's
        # epoch bar. Kept separate from the pruning and early-stopping callbacks the runner owns.
        self.extra_callbacks = list(extra_callbacks or [])

    def _lightning(self):
        """Import PyTorch Lightning, with a clear message when it is missing."""
        return importlib.import_module("lightning.pytorch")

    def build_trainer(self, bundle, pruning_callback):
        """Build the trainer for one trial: quiet, with early stopping and pruning attached."""
        lightning = self._lightning()
        trainer_kwargs = {**bundle.trainer_kwargs, **TRIAL_TRAINER_OVERRIDES}

        callbacks = [pruning_callback, *self.extra_callbacks]
        early_stopping = dict(bundle.callback_specs.get("early_stopping") or {})
        if early_stopping:
            # Rewritten to the study objective so early stopping and Optuna never disagree about
            # which direction is better. strict=False for the same reason the pruning callback
            # tolerates a gap: a metric _log_epoch_metrics skipped is not a misconfiguration.
            early_stopping.update(monitor=self.objective.metric, mode=self.objective.mode, strict=False)
            callbacks.append(lightning.callbacks.EarlyStopping(**early_stopping))

        return lightning.Trainer(**trainer_kwargs, callbacks=callbacks)

    def run(self, bundle, trial: optuna.Trial, *, report: bool = True) -> TrialResult:
        """Train one trial and return its score.

        Returns
        -------
        TrialResult

        Raises
        ------
        UnrecoverableAcceleratorError
            If the GPU has died, so the study can stop rather than fail every remaining trial.
        """
        pruning_callback = OptunaPruningCallback(
            trial, monitor=self.objective.metric, mode=self.objective.mode, report=report
        )
        trainer = self.build_trainer(bundle, pruning_callback)

        try:
            try:
                trainer.fit(bundle.model, datamodule=bundle.datamodule)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                if self.fail_fast:
                    raise
                # Before pruning: was it this configuration that failed, or the device? Pruning a
                # dead GPU would spend the rest of the study reloading data to fail again.
                raise_if_accelerator_is_dead(exc, trial.number)
                # A single bad corner of the space - an OOM, a non-finite loss from _shared_step, an
                # architecture combination the model rejects - must not end a long study.
                if self.logger is not None:
                    self.logger.warning(f"Trial {trial.number} failed and was pruned: {exc!r}", exc_info=True)
                raise optuna.TrialPruned(f"Trial {trial.number} raised {type(exc).__name__}: {exc}") from exc

            if pruning_callback.best_value is None:
                raise optuna.TrialPruned(
                    f"Trial {trial.number} never logged {self.objective.metric!r}; nothing to optimize."
                )

            return TrialResult(
                value=pruning_callback.best_value,
                best_epoch=pruning_callback.best_epoch,
                epochs_run=int(getattr(trainer, "current_epoch", 0)),
            )
        finally:
            # Drop this frame's reference so the Trainer becomes collectable once the caller drops
            # the bundle - Lightning leaves model._trainer pointing back here, so the bundle is the
            # other half of the cycle. TrialObjective owns the actual collection.
            trainer = None  # noqa: F841
