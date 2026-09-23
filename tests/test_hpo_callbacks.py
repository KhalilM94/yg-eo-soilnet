"""The Lightning callbacks an HPO trial runs with: study progress and Optuna pruning."""

import logging
from io import StringIO
from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.progress import (
    StudyProgress,
    TqdmLoggingHandler,
    resolve_mode,
    tqdm_safe_logging,
)
from yg_eo_soilnet.hpo.pruning import OptunaPruningCallback, metric_to_float
from yg_eo_soilnet.hpo.search_space import Objective
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker

MAXIMIZE = Objective(metric="val_r2", direction="maximize")


class TtyStream(StringIO):
    """A StringIO that claims to be a terminal, so `auto` resolves to bars."""

    def isatty(self):
        return True


class FakeTrainer:
    """The attributes the pruning and progress callbacks read off a lightning.pytorch.Trainer."""

    def __init__(self, callback_metrics=None, current_epoch=0, sanity_checking=False, max_epochs=10):
        self.callback_metrics = callback_metrics or {}
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking
        self.max_epochs = max_epochs


def _frozen_trial(study, value=0.5, prune=False):
    """Run one real trial through `study` so we get a genuine FrozenTrial back."""

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.set_user_attr("epochs_run", 22)
        if prune:
            raise optuna.TrialPruned()
        return value

    study.optimize(objective, n_trials=1)
    return study.trials[-1]


def _logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream


# --- mode resolution ---------------------------------------------------------


def test_auto_becomes_bar_on_a_terminal():
    assert resolve_mode("auto", TtyStream()) == "bar"


def test_auto_becomes_plain_when_piped():
    assert resolve_mode("auto", StringIO()) == "plain"


def test_an_explicit_mode_is_respected():
    assert resolve_mode("plain", TtyStream()) == "plain"
    assert resolve_mode("none", TtyStream()) == "none"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="progress mode must be one of"):
        resolve_mode("fancy")


# --- the logging bridge ------------------------------------------------------


def test_the_bridge_swaps_the_console_handler_and_restores_it():
    logger, _ = _logger("bridge_swap")
    original = logger.handlers[0]

    with tqdm_safe_logging(logger):
        assert isinstance(logger.handlers[0], TqdmLoggingHandler)
        assert logger.handlers[0].formatter is original.formatter

    assert logger.handlers == [original]


def test_the_bridge_leaves_a_file_handler_alone(tmp_path):
    """logging.FileHandler subclasses StreamHandler; hijacking it would corrupt the .log file."""
    logger, _ = _logger("bridge_file")
    file_handler = logging.FileHandler(tmp_path / "run.log")
    logger.addHandler(file_handler)

    with tqdm_safe_logging(logger):
        assert file_handler in logger.handlers
        assert sum(isinstance(h, TqdmLoggingHandler) for h in logger.handlers) == 1

    file_handler.close()


def test_the_bridge_tolerates_no_logger():
    with tqdm_safe_logging(None):
        pass


def test_bridged_records_still_reach_the_stream():
    logger, stream = _logger("bridge_output")
    with tqdm_safe_logging(logger):
        logger.info("hello from a trial")

    assert "hello from a trial" in stream.getvalue()


# --- none mode ---------------------------------------------------------------


def test_none_mode_writes_nothing_and_raises_nothing():
    stream = StringIO()
    logger, log_stream = _logger("none_mode")
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, mode="none", logger=logger, stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        progress.advance_epoch(0, 0.5)
        progress.on_trial_end(study, _frozen_trial(study))
        progress.end_trial()

    assert stream.getvalue() == ""
    assert log_stream.getvalue() == ""


# --- bar mode ----------------------------------------------------------------


def test_bar_mode_renders_both_bars():
    stream = TtyStream()
    tracker = ObjectiveTracker(MAXIMIZE)
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, tracker=tracker, mode="bar", stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        progress.advance_epoch(0, 0.42)
        trial = _frozen_trial(study, value=0.42)
        tracker.record(study, trial)
        progress.on_trial_end(study, trial)

    output = stream.getvalue()
    assert "Trials" in output
    assert "Trial " in output
    assert "best=0.4200" in output


def test_the_outer_bar_closes_a_leaked_inner_bar():
    """A pruned trial raises out of the Lightning callback, so on_fit_end never runs."""
    stream = TtyStream()
    progress = StudyProgress(n_trials=2, objective=MAXIMIZE, mode="bar", stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        assert progress._inner is not None
        progress.on_trial_end(study, _frozen_trial(study, prune=True))
        assert progress._inner is None


# --- plain mode --------------------------------------------------------------


def test_plain_mode_logs_one_line_per_trial():
    logger, log_stream = _logger("plain_trial")
    tracker = ObjectiveTracker(MAXIMIZE)
    progress = StudyProgress(
        n_trials=3, objective=MAXIMIZE, tracker=tracker, mode="plain", logger=logger, stream=StringIO()
    )
    study = optuna.create_study(direction="maximize")

    with progress:
        trial = _frozen_trial(study, value=0.42)
        tracker.record(study, trial)
        progress.on_trial_end(study, trial)

    line = log_stream.getvalue()
    assert "trial   0" in line
    assert "val_r2=0.4200" in line


def test_plain_mode_marks_a_pruned_trial():
    logger, log_stream = _logger("plain_pruned")
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO())
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.on_trial_end(study, _frozen_trial(study, prune=True))

    output = log_stream.getvalue()
    assert "PRUNED" in output
    assert "@ep22" in output


def test_a_pruned_trial_that_reported_a_value_still_reads_as_pruned():
    """Optuna keeps the last intermediate value on a PRUNED trial, so `value is None` is not the
    test for completion - such a trial used to print as if it had finished."""
    logger, log_stream = _logger("plain_pruned_valued")
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO())
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.2583, step=7)
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=1)
    trial = study.trials[-1]
    assert trial.value is not None  # the trap

    with progress:
        progress.on_trial_end(study, trial)

    output = log_stream.getvalue()
    assert "PRUNED" in output
    assert "@ep8" in output  # last reported epoch, since a pruned trial has no epochs_run
    assert "0.2583" in output  # the value it reached is still shown, in parentheses


def test_plain_mode_emits_an_epoch_heartbeat():
    logger, log_stream = _logger("plain_epochs")
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO())

    with progress:
        progress.start_trial(20)  # heartbeat every 20 // 10 == 2 epochs
        for epoch in range(4):
            progress.advance_epoch(epoch, 0.1 * epoch)

    lines = [line for line in log_stream.getvalue().splitlines() if "ep " in line]
    assert len(lines) == 2  # epochs 2 and 4, not all four


# --- the epoch callback ------------------------------------------------------


def test_the_epoch_callback_advances_on_validation_end():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_r2": 0.3}, current_epoch=0), None)

        assert progress._inner.n == 1
        assert progress._best_this_trial == pytest.approx(0.3)


def test_the_epoch_callback_ignores_the_sanity_check():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_r2": 0.9}, sanity_checking=True), None)

        assert progress._inner.n == 0
        assert progress._best_this_trial is None


def test_the_epoch_callback_tolerates_a_missing_metric():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_loss": 0.4}), None)

        assert progress._inner.n == 1  # still advanced
        assert progress._best_this_trial is None


def test_the_epoch_callback_tracks_the_best_in_the_objective_direction():
    progress = StudyProgress(
        n_trials=1, objective=Objective(metric="val_loss", direction="minimize"), mode="bar", stream=TtyStream()
    )
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        for epoch, value in enumerate([0.9, 0.2, 0.7]):
            callback.on_validation_end(FakeTrainer({"val_loss": value}, current_epoch=epoch), None)

        assert progress._best_this_trial == pytest.approx(0.2)


def test_the_epoch_callback_closes_the_bar_on_fit_end():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_fit_end(FakeTrainer(), None)

        assert progress._inner is None


def test_the_epoch_callback_is_accepted_by_the_trial_runner():
    """The wiring contract: TrialRunner puts extra callbacks on the Trainer it builds."""
    from yg_eo_soilnet.hpo.trial_runner import TrialRunner

    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="none", stream=StringIO())
    runner = TrialRunner(MAXIMIZE, extra_callbacks=[progress.epoch_callback()])

    assert len(runner.extra_callbacks) == 1


def test_a_trial_objective_forwards_progress_to_the_runner():
    from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
    from yg_eo_soilnet.hpo.search_space import SearchSpace

    registry = {
        "e": {
            "enabled": True,
            "modeltype": "dl",
            "input_kind": "sequence",
            "import_path": "x.Y",
            "datamodule_import_path": "x.Z",
        }
    }
    context = ObjectiveContext.from_config(
        "e", SimpleNamespace(LIGHTNING_MODEL_REGISTRY=registry, TARGET_COLUMNS=["t"], RANDOM_SEED=1), data={}
    )
    space = SearchSpace.from_mapping("e", {"params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5}}})
    progress = StudyProgress(n_trials=1, objective=space.objective, mode="none", stream=StringIO())

    objective = TrialObjective(context, space, progress=progress)
    assert len(objective.runner.extra_callbacks) == 1

    assert TrialObjective(context, space).runner.extra_callbacks == []


# --- the pruning callback ---------------------------------------------------------------------


class FakeTrial:
    def __init__(self, should_prune=False, number=0):
        self.number = number
        self.reports: list[tuple[float, int]] = []
        self._should_prune = should_prune

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._should_prune


def _run_epoch(callback, trainer):
    callback.on_validation_end(trainer, SimpleNamespace())


def test_metric_to_float_unwraps_a_zero_dim_tensor():
    torch = pytest.importorskip("torch")
    assert metric_to_float(torch.tensor(0.75)) == pytest.approx(0.75)


def test_metric_to_float_rejects_nan():
    """NaN must not reach the pruner, which would rank it against real values."""
    assert metric_to_float(float("nan")) is None


def test_metric_to_float_rejects_a_non_number():
    assert metric_to_float("not a metric") is None
    assert metric_to_float(None) is None


def test_callback_reports_the_monitored_metric():
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_r2": 0.42, "val_loss": 1.0}, current_epoch=3))

    assert trial.reports == [(0.42, 3)]


def test_callback_prunes_when_the_trial_says_so():
    trial = FakeTrial(should_prune=True, number=7)
    callback = OptunaPruningCallback(trial, monitor="val_r2")

    with pytest.raises(optuna.TrialPruned, match="Trial 7 pruned at epoch 2"):
        _run_epoch(callback, FakeTrainer({"val_r2": 0.1}, current_epoch=2))


def test_callback_ignores_the_sanity_check():
    """Sanity-check metrics predate any training, so they say nothing about the trial."""
    trial = FakeTrial(should_prune=True)
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_r2": 0.0}, sanity_checking=True))

    assert trial.reports == []


def test_a_missing_metric_is_a_gap_not_a_pruning_signal():
    """_log_epoch_metrics skips val_r2 on a degenerate batch; that must not end the trial."""
    trial = FakeTrial(should_prune=True)
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_loss": 0.5}, current_epoch=1))

    assert trial.reports == []


def test_the_same_epoch_is_reported_only_once():
    """Optuna raises if a step is reported twice, and Lightning can revisit an epoch hook."""
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial, monitor="val_loss")
    trainer = FakeTrainer({"val_loss": 0.5}, current_epoch=0)
    _run_epoch(callback, trainer)
    _run_epoch(callback, trainer)

    assert trial.reports == [(0.5, 0)]


def test_the_callback_hooks_on_validation_end_not_on_validation_epoch_end():
    """Lightning runs callback on_validation_epoch_end BEFORE the LightningModule's.

    `val_r2` is logged in the module's hook (_regression_base._log_epoch_metrics), so reading it
    from on_validation_epoch_end yields the previous epoch's value - None on epoch 0. Every report,
    the best value and the best epoch would silently be off by one. `val_loss` is logged in
    validation_step and is current under either hook, which is what made this easy to miss.
    """
    # Checked on the class's own __dict__: the Lightning Callback base defines both hooks as
    # no-ops, so hasattr is true either way. What matters is which one we override.
    assert "on_validation_end" in OptunaPruningCallback.__dict__
    assert "on_validation_epoch_end" not in OptunaPruningCallback.__dict__


def test_callback_drives_a_real_optuna_trial():
    """End to end against a real study: a hopeless trial is pruned by the median pruner."""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0),
    )

    def objective(trial):
        callback = OptunaPruningCallback(trial, monitor="val_loss")
        offset = trial.suggest_float("offset", 0.0, 10.0)
        for epoch in range(5):
            _run_epoch(callback, FakeTrainer({"val_loss": offset + epoch * 0.0}, current_epoch=epoch))
        return offset

    study.optimize(lambda t: objective(t), n_trials=2)
    # The good trial completed; the study holds a usable best value either way.
    assert study.best_value is not None
