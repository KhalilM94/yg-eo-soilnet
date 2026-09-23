"""One HPO trial: the runner that fits and scores it, and the objective that builds it."""

import sys
from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.objective import OVERRIDES_ATTR, ObjectiveContext, TrialObjective
from yg_eo_soilnet.hpo.search_space import Objective, SearchSpace
from yg_eo_soilnet.hpo.trial_runner import (
    TRIAL_TRAINER_OVERRIDES,
    TrialRunner,
    UnrecoverableAcceleratorError,
    cuda_context_is_dead,
    release_dataloader_workers,
)


class FakeBundle:
    def __init__(self, trainer_kwargs=None, callback_specs=None):
        self.name = "fake_model"
        self.model = SimpleNamespace()
        self.datamodule = SimpleNamespace()
        self.trainer_kwargs = trainer_kwargs or {"max_epochs": 5, "enable_checkpointing": True, "deterministic": True}
        self.callback_specs = (
            callback_specs
            if callback_specs is not None
            else {
                "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 30},
                "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
            }
        )


class RecordingTrainer:
    """Stands in for lightning.pytorch.Trainer, replaying a metric series through the callbacks."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callbacks = kwargs.get("callbacks", [])
        self.callback_metrics = {}
        self.current_epoch = 0
        self.sanity_checking = False
        self.series: list[dict] = []
        RecordingTrainer.last = self

    def fit(self, model, datamodule=None):
        for epoch, metrics in enumerate(self.series):
            self.current_epoch = epoch
            self.callback_metrics = metrics
            for callback in self.callbacks:
                hook = getattr(callback, "on_validation_end", None)
                if hook is not None:
                    hook(self, model)


class FakeEarlyStopping:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _runner(monkeypatch, series, *, objective=None, bundle=None, **runner_kwargs):
    fake_lightning = SimpleNamespace(
        Trainer=RecordingTrainer,
        callbacks=SimpleNamespace(EarlyStopping=FakeEarlyStopping),
    )
    runner = TrialRunner(objective or Objective(metric="val_r2", direction="maximize"), **runner_kwargs)
    monkeypatch.setattr(runner, "_lightning", lambda: fake_lightning)

    original_init = RecordingTrainer.__init__

    def init_with_series(self, **kwargs):
        original_init(self, **kwargs)
        self.series = series

    monkeypatch.setattr(RecordingTrainer, "__init__", init_with_series)
    return runner, bundle or FakeBundle()


def _trial():
    return optuna.create_study(direction="maximize").ask()


# --- forced trainer settings -------------------------------------------------


def test_a_trial_never_checkpoints_or_writes_lightning_logs(monkeypatch):
    """Hundreds of trials must leave no .ckpt files and no lightning_logs/version_N directories."""
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    kwargs = RecordingTrainer.last.kwargs
    for key, expected in TRIAL_TRAINER_OVERRIDES.items():
        assert kwargs[key] is expected, key
    # Registry settings that are not about noise or disk survive untouched.
    assert kwargs["max_epochs"] == 5
    assert kwargs["deterministic"] is True


def test_early_stopping_is_rewired_to_the_study_objective(monkeypatch):
    """Otherwise early stopping could halt on val_loss while Optuna scores val_r2."""
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    early_stopping = [cb for cb in RecordingTrainer.last.callbacks if isinstance(cb, FakeEarlyStopping)][0]
    assert early_stopping.kwargs["monitor"] == "val_r2"
    assert early_stopping.kwargs["mode"] == "max"
    assert early_stopping.kwargs["strict"] is False
    assert early_stopping.kwargs["patience"] == 30  # the registry's value is kept


def test_the_checkpoint_spec_never_becomes_a_callback(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    assert len(RecordingTrainer.last.callbacks) == 2  # pruning + early stopping only


def test_a_bundle_without_an_early_stopping_spec_still_runs(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], bundle=FakeBundle(callback_specs={}))
    result = runner.run(bundle, _trial())

    assert result.value == 0.5
    assert not any(isinstance(cb, FakeEarlyStopping) for cb in RecordingTrainer.last.callbacks)


# --- scoring -----------------------------------------------------------------


def test_the_objective_is_the_best_epoch_not_the_last(monkeypatch):
    """Under early stopping the last epoch is `patience` epochs past the best one."""
    series = [{"val_r2": 0.1}, {"val_r2": 0.9}, {"val_r2": 0.4}, {"val_r2": 0.2}]
    runner, bundle = _runner(monkeypatch, series)
    result = runner.run(bundle, _trial())

    assert result.value == 0.9
    assert result.best_epoch == 1


def test_best_means_lowest_when_minimizing(monkeypatch):
    series = [{"val_loss": 1.0}, {"val_loss": 0.3}, {"val_loss": 0.8}]
    runner, bundle = _runner(monkeypatch, series, objective=Objective(metric="val_loss", direction="minimize"))
    result = runner.run(bundle, _trial())

    assert result.value == 0.3


def test_a_metric_that_is_never_logged_prunes_the_trial(monkeypatch):
    """val_r2 is skipped by _log_epoch_metrics on a degenerate validation split."""
    runner, bundle = _runner(monkeypatch, [{"val_loss": 0.5}, {"val_loss": 0.4}])

    with pytest.raises(optuna.TrialPruned, match="never logged 'val_r2'"):
        runner.run(bundle, _trial())


# --- failure policy ----------------------------------------------------------


def test_a_raising_trial_is_pruned_rather_than_killing_the_study(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(optuna.TrialPruned, match="raised ZeroDivisionError"):
        runner.run(bundle, _trial())


def test_fail_fast_surfaces_the_original_exception(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], fail_fast=True)
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        runner.run(bundle, _trial())


def test_a_pruned_trial_propagates_as_pruned(monkeypatch):
    """TrialPruned from the callback must not be swallowed by the generic failure handler."""
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)
    )
    # A strong completed baseline, with the intermediates the median pruner compares against.
    baseline = study.ask()
    baseline.report(10.0, step=0)
    baseline.report(10.0, step=1)
    study.tell(baseline, 10.0)

    runner, bundle = _runner(monkeypatch, [{"val_r2": -5.0}, {"val_r2": -4.0}])
    with pytest.raises(optuna.TrialPruned, match="pruned at epoch"):
        runner.run(bundle, study.ask())


def test_the_trainer_is_collectable_once_the_bundle_is_dropped(monkeypatch):
    """run() must not leave a reference behind, or the Trainer's DataLoader iterators outlive it.

    Modelled on the real cycle: Lightning points model._trainer back at the Trainer, so the bundle
    is the other half. Uses its own trainer class because RecordingTrainer pins the last instance
    on a class attribute.
    """
    import gc
    import weakref

    created: list[weakref.ref] = []

    class UnpinnedTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = kwargs.get("callbacks", [])
            self.callback_metrics = {"val_r2": 0.5}
            self.current_epoch = 0
            self.sanity_checking = False
            created.append(weakref.ref(self))

        def fit(self, model, datamodule=None):
            model.trainer = self  # the cycle Lightning creates
            for callback in self.callbacks:
                hook = getattr(callback, "on_validation_end", None)
                if hook is not None:
                    hook(self, model)

    fake_lightning = SimpleNamespace(
        Trainer=UnpinnedTrainer, callbacks=SimpleNamespace(EarlyStopping=FakeEarlyStopping)
    )
    runner = TrialRunner(Objective(metric="val_r2", direction="maximize"))
    monkeypatch.setattr(runner, "_lightning", lambda: fake_lightning)

    bundle = FakeBundle()
    runner.run(bundle, _trial())
    assert created[0]() is not None  # still reachable through bundle.model.trainer

    del bundle
    gc.collect()
    assert created[0]() is None


def test_report_false_still_tracks_the_best_without_reporting(monkeypatch):
    """Seed repeats past the first must not overwrite the first repeat's intermediate curve."""
    trial = _trial()
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.2}, {"val_r2": 0.7}])
    result = runner.run(bundle, trial, report=False)

    assert result.value == 0.7
    assert trial.storage.get_trial(trial._trial_id).intermediate_values == {}


# --- a dead accelerator is not a bad hyperparameter ---------------------------


class FakeCuda:
    """Stands in for torch.cuda: `initialized` gates the probe, `alive` decides its verdict."""

    def __init__(self, initialized=True, alive=True):
        self.initialized = initialized
        self.alive = alive
        self.empty_cache_calls = 0

    def is_initialized(self):
        return self.initialized

    def empty_cache(self):
        self.empty_cache_calls += 1


def _fake_torch(monkeypatch, initialized=True, alive=True):
    """Install a stand-in `torch` in sys.modules, which is where the probe looks it up."""
    cuda = FakeCuda(initialized=initialized, alive=alive)

    def zeros(*args, **kwargs):
        if not cuda.alive:
            raise RuntimeError("CUDA error: unknown error")
        return SimpleNamespace(add_=lambda *a: SimpleNamespace(cpu=lambda: None))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, zeros=zeros))
    return cuda


def test_a_dead_context_aborts_the_study_instead_of_pruning(monkeypatch):
    """Pruning a dead GPU would spend the rest of the study reloading data to fail again."""
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(
        RecordingTrainer, "fit", lambda self, model, datamodule=None: (_ for _ in ()).throw(RuntimeError("CUDA error"))
    )

    with pytest.raises(UnrecoverableAcceleratorError, match="died during trial"):
        runner.run(bundle, _trial())


def test_the_abort_message_says_how_to_recover(monkeypatch):
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(UnrecoverableAcceleratorError, match="wsl --shutdown"):
        runner.run(bundle, _trial())


def test_a_live_context_still_prunes_the_trial(monkeypatch):
    """An OOM leaves the context usable: a too-large corner of the space stays one pruned trial."""
    _fake_torch(monkeypatch, alive=True)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(
        RecordingTrainer,
        "fit",
        lambda self, model, datamodule=None: (_ for _ in ()).throw(MemoryError("CUDA out of memory")),
    )

    with pytest.raises(optuna.TrialPruned, match="raised MemoryError"):
        runner.run(bundle, _trial())


def test_a_process_with_no_cuda_context_never_probes(monkeypatch):
    """A CPU run must behave exactly as it did before the probe existed."""
    _fake_torch(monkeypatch, initialized=False, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(optuna.TrialPruned, match="raised ZeroDivisionError"):
        runner.run(bundle, _trial())


def test_fail_fast_still_wins_over_the_probe(monkeypatch):
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], fail_fast=True)
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        runner.run(bundle, _trial())


def test_cuda_context_is_dead_is_false_without_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert cuda_context_is_dead() is False


# --- releasing a trial's resources --------------------------------------------


def test_the_allocator_is_emptied_between_trials(monkeypatch):
    """Trials range from one small block to five wide ones; freed blocks are the wrong shapes."""
    cuda = _fake_torch(monkeypatch)

    release_dataloader_workers()

    assert cuda.empty_cache_calls == 1


def test_nothing_is_emptied_when_no_context_was_built(monkeypatch):
    cuda = _fake_torch(monkeypatch, initialized=False)

    release_dataloader_workers()

    assert cuda.empty_cache_calls == 0


def test_releasing_works_without_torch_imported(monkeypatch):
    """The module lazy-imports on purpose; the collect must still run on a torch-free path."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    release_dataloader_workers()


# --- the trial objective ----------------------------------------------------------------------


class FakeDataModule:
    """Mirrors the shape contract SoilSequenceDataModule exposes to the factory."""

    instances = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.static_dim = 3
        self.target_dim = 1
        self.temporal_enabled = False
        self.target_mean_ = [1.0]
        self.target_scale_ = [2.0]
        self.target_transform = "log1p"
        FakeDataModule.instances += 1

    def setup(self, stage=None):
        return None


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


REGISTRY = {
    "fake_entry": {
        "enabled": False,  # the study must not care what the file has switched on
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.FakeModel",
        "datamodule_import_path": f"{__name__}.FakeDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto", "learning_rate": 0.001, "dropout": 0.1},
        "datamodule_init_args": {"batch_size": 32, "val_size": 0.3, "test_size": 0.15, "seed": 42},
        "trainer_args": {"max_epochs": 500, "deterministic": True},
        "callbacks": {"early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 30}},
    }
}


def _config(**overrides):
    base = dict(LIGHTNING_MODEL_REGISTRY=REGISTRY, TARGET_COLUMNS=["organic_matter_pct"], RANDOM_SEED=42)
    base.update(overrides)
    return SimpleNamespace(**base)


def _space(**overrides):
    mapping = {
        "objective": {"metric": "val_r2", "direction": "maximize"},
        "fixed": {"trainer.max_epochs": 150},
        "params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05}},
    }
    mapping.update(overrides)
    return SearchSpace.from_mapping("fake_entry", mapping)


def _objective(context=None, space=None, **kwargs):
    context = context or ObjectiveContext.from_config("fake_entry", _config(), data={"sequence_bundle": object()})
    return TrialObjective(context, space or _space(), **kwargs)


# --- context -----------------------------------------------------------------


def test_context_combines_multiple_targets_like_main_does():
    context = ObjectiveContext.from_config("fake_entry", _config(TARGET_COLUMNS=["a", "b"]), data={})

    assert context.target == "a__b"


def test_context_uses_the_single_target_verbatim():
    assert ObjectiveContext.from_config("fake_entry", _config(), data={}).target == "organic_matter_pct"


def test_a_missing_registry_entry_names_what_is_available():
    with pytest.raises(KeyError, match="fake_entry"):
        ObjectiveContext.from_config("no_such_entry", _config(), data={})


def test_context_snapshots_the_registry_entry():
    """Trials mutate copies; the pristine entry must survive the study."""
    context = ObjectiveContext.from_config("fake_entry", _config(), data={})
    context.registry_entry["init_args"]["dropout"] = 0.99

    assert REGISTRY["fake_entry"]["init_args"]["dropout"] == 0.1


# --- bundle construction -----------------------------------------------------


def test_overrides_reach_the_model_and_the_disabled_entry_is_still_built():
    bundle = _objective().build_bundle({"model.dropout": 0.35, "trainer.max_epochs": 150})

    assert bundle.model.kwargs["dropout"] == 0.35
    assert bundle.trainer_kwargs["max_epochs"] == 150
    # Untouched registry values survive, and `auto` is still resolved from the datamodule.
    assert bundle.model.kwargs["learning_rate"] == 0.001
    assert bundle.model.kwargs["static_dim"] == 3
    assert bundle.model.kwargs["target_dim"] == 1


def test_datamodule_overrides_reach_the_datamodule():
    bundle = _objective().build_bundle({"datamodule.batch_size": 64})

    assert bundle.datamodule.kwargs["batch_size"] == 64


def test_the_supplied_sequence_bundle_is_reused_rather_than_rebuilt():
    """Rebuilding from the raw CSVs per trial would dominate the cost of a study."""
    payload = object()
    context = ObjectiveContext.from_config("fake_entry", _config(), data={"sequence_bundle": payload})
    bundle = TrialObjective(context, _space()).build_bundle({})

    assert bundle.datamodule.kwargs["sequence_bundle"] is payload


def test_the_datamodule_cache_reuses_one_instance_across_trials():
    context = ObjectiveContext.from_config(
        "fake_entry", _config(), data={"sequence_bundle": object()}, datamodule_cache={}
    )
    objective = TrialObjective(context, _space())

    FakeDataModule.instances = 0
    first = objective.build_bundle({"model.dropout": 0.1})
    second = objective.build_bundle({"model.dropout": 0.4})
    third = objective.build_bundle({"datamodule.batch_size": 64})

    assert first.datamodule is second.datamodule  # only model args changed
    assert third.datamodule is not first.datamodule  # a datamodule arg changed
    assert FakeDataModule.instances == 2


def test_without_a_cache_every_trial_gets_a_fresh_datamodule():
    objective = _objective()

    FakeDataModule.instances = 0
    first = objective.build_bundle({})
    second = objective.build_bundle({})

    assert first.datamodule is not second.datamodule
    assert FakeDataModule.instances == 2


# --- the objective call ------------------------------------------------------


def _run_one_trial(objective, values):
    """Drive one trial with a stubbed runner that returns `values` in order."""
    calls = []

    def fake_run(bundle, trial, *, report=True):
        calls.append((bundle, report))
        return SimpleNamespace(value=values[len(calls) - 1], best_epoch=3, epochs_run=9)

    objective.runner.run = fake_run
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    return study.trials[0], calls


def test_the_resolved_overrides_are_stored_on_the_trial():
    """Export reads this back; a conditional draw cannot be replayed outside a live trial."""
    trial, _ = _run_one_trial(_objective(), [0.6])

    overrides = trial.user_attrs[OVERRIDES_ATTR]
    assert overrides["trainer.max_epochs"] == 150
    assert "model.dropout" in overrides
    assert trial.value == 0.6
    assert trial.user_attrs["best_epoch"] == 3


def test_seed_repeats_average_and_report_only_once():
    trial, calls = _run_one_trial(_objective(seed_repeats=3), [0.2, 0.4, 0.6])

    assert trial.value == pytest.approx(0.4)
    assert trial.user_attrs["seed_values"] == [0.2, 0.4, 0.6]
    assert [report for _, report in calls] == [True, False, False]


def test_workers_are_released_once_per_seed_repeat(monkeypatch):
    """Without this the Trainer cycle survives and the next trial's fork inherits its iterators."""
    releases = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: releases.append(1))

    _run_one_trial(_objective(seed_repeats=3), [0.1, 0.2, 0.3])

    assert len(releases) == 3


def test_workers_are_released_even_when_the_trial_is_pruned(monkeypatch):
    """A pruned trial raises out of run(); the traceback keeps the frame, so the finally matters."""
    releases = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: releases.append(1))

    objective = _objective()

    def pruning_run(bundle, trial, *, report=True):
        raise optuna.TrialPruned("pruned in the test")

    objective.runner.run = pruning_run
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert study.trials[0].state.name == "PRUNED"
    assert len(releases) == 1


def test_release_dataloader_workers_finalizes_a_reference_cycle():
    """The mechanism itself: Lightning's Trainer graph is cyclic, so only a collection frees it.

    Until it is freed its DataLoader iterators stay alive to be inherited by the next fork.
    """
    import weakref

    from yg_eo_soilnet.hpo.trial_runner import release_dataloader_workers

    finalized = []

    class Cyclic:
        def __init__(self):
            self.self_reference = self  # the cycle refcounting cannot break

        def __del__(self):
            finalized.append(1)

    reference = weakref.ref(Cyclic())
    assert reference() is not None  # refcounting alone never frees this

    release_dataloader_workers()

    assert reference() is None
    assert finalized == [1]


def test_each_trial_is_seeded_before_the_model_is_built(monkeypatch):
    """LightningTrainer seeds after construction, so weight init is unseeded on that path."""
    seeded: list[int] = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: seeded.append(seed))

    objective = _objective(seed=7, seed_repeats=2)
    _run_one_trial(objective, [0.1, 0.2])

    assert seeded == [7, 8]


# --- a lost accelerator must stop the study, not kill the process -------------


def test_a_dead_accelerator_during_seeding_aborts_the_study(monkeypatch):
    """The gap that ended a 266-trial study: seed_everything sat outside every guard.

    It reaches torch.cuda.manual_seed_all, so once the device is gone it raises before the runner
    is ever entered - and the raw error escaped study.optimize and killed tune.py.
    """
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything",
        lambda seed: (_ for _ in ()).throw(RuntimeError("CUDA error: unknown error")),
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: True)

    with pytest.raises(UnrecoverableAcceleratorError, match="died during trial"):
        objective(optuna.create_study(direction="maximize").ask())


def test_a_live_accelerator_lets_a_seeding_failure_propagate_as_itself(monkeypatch):
    """Only a dead device is special. Anything else keeps its own type and traceback."""
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: (_ for _ in ()).throw(ValueError("bad seed"))
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: False)

    with pytest.raises(ValueError, match="bad seed"):
        objective(optuna.create_study(direction="maximize").ask())


def test_workers_are_still_released_when_seeding_fails(monkeypatch):
    """The finally must survive the new except, or a dead trial leaks its DataLoader workers."""
    released = []
    objective = _objective()
    monkeypatch.setattr(
        "yg_eo_soilnet.hpo.objective.seed_everything", lambda seed: (_ for _ in ()).throw(ValueError("bad seed"))
    )
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: False)
    monkeypatch.setattr("yg_eo_soilnet.hpo.objective.release_dataloader_workers", lambda: released.append(True))

    with pytest.raises(ValueError):
        objective(optuna.create_study(direction="maximize").ask())
    assert released == [True]


def test_a_pruned_trial_is_not_mistaken_for_a_device_failure(monkeypatch):
    """TrialPruned must reach Optuna untouched, whatever the probe would have said."""
    objective = _objective()
    probed = []
    monkeypatch.setattr("yg_eo_soilnet.hpo.trial_runner.cuda_context_is_dead", lambda: probed.append(True) or True)
    monkeypatch.setattr(
        objective.runner,
        "run",
        lambda bundle, trial, report=True: (_ for _ in ()).throw(optuna.TrialPruned("pruned")),
    )

    with pytest.raises(optuna.TrialPruned):
        objective(optuna.create_study(direction="maximize").ask())
    assert probed == []
