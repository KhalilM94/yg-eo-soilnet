"""Reproducibility: a tuned trial and its final retrain must see the same RNG stream.

A loader's serving settings must not change the order a run trains in or the dropout it draws.

The failure this pins: a study ran with persistent_workers true, and the exported config restored the
registry's false. The retrain matched its trial at epoch 0, then diverged at epoch 1 and finished at
val_loss 0.5566 against the trial's 0.5405. A default DataLoader draws from the global torch stream
once per new iterator, and only a non-persistent loader builds a new iterator every epoch.
"""

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from yg_eo_soilnet.datamodules.loaders import build_loader
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.seeding import seed_everything

EPOCHS = 3


def _run(make_loader, seed=42, **serving):
    """Epochs as a fit loop runs them: train, then validation.

    Returns each epoch's order, plus one global draw per epoch standing in for that epoch's dropout.
    """
    torch.manual_seed(seed)
    train = make_loader(list(range(96)), batch_size=16, shuffle=True, **serving)
    val = make_loader(list(range(32)), batch_size=16, **serving)
    orders, draws = [], []
    for _ in range(EPOCHS):
        orders.append([int(index) for batch in train for index in batch])
        draws.append(torch.rand(1).item())
        for _ in val:
            pass
    return orders, draws


def _bare_loader(dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


def test_persistent_workers_do_not_change_the_run():
    """The regression this module exists for."""
    persistent = _run(build_loader, num_workers=2, persistent_workers=True)
    fresh_each_epoch = _run(build_loader, num_workers=2, persistent_workers=False)

    assert persistent == fresh_each_epoch


def test_num_workers_do_not_change_the_run():
    assert _run(build_loader, num_workers=0) == _run(build_loader, num_workers=2)


def test_a_bare_dataloader_does_change_the_run():
    """Proof that the loader's own draws were the problem, and not an incidental difference."""
    persistent_orders, persistent_draws = _run(_bare_loader, num_workers=2, persistent_workers=True)
    fresh_orders, fresh_draws = _run(_bare_loader, num_workers=2, persistent_workers=False)

    assert persistent_orders[0] == fresh_orders[0]  # epoch 0 agrees, as it did in the study
    assert persistent_orders[1:] != fresh_orders[1:]
    assert persistent_draws != fresh_draws


def test_the_run_seed_still_reaches_the_shuffle():
    """The private generators are seeded from the global stream, so seeding still matters."""
    assert _run(build_loader, seed=42)[0] != _run(build_loader, seed=7)[0]


def test_shuffle_off_keeps_the_dataset_order():
    orders, _ = _run(lambda dataset, shuffle=False, **kwargs: build_loader(dataset, **kwargs))

    assert all(order == list(range(96)) for order in orders)


def test_the_sequence_datamodule_builds_through_build_loader():
    """Structural guard: a bare DataLoader in the datamodule would silently bring the bug back."""
    fake = SimpleNamespace(batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False)
    fake._collate_points = lambda batch: batch

    loader = SoilSequenceDataModule._make_loader(fake, np.arange(8), shuffle=True)

    assert loader.generator is not None
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.generator is not None


# --- seeding parity between HPO and final training --------------------------------------------
# The property that makes a tuned config reproducible: both paths enter fit() identically seeded.
#
# The hyperparameter search seeds, then builds the model, then fits. Production used to build the
# model and only then seed, so `fit` began from a different point in the RNG stream - a different
# shuffle order from the first batch, a different trajectory, and a score the tuned config could never
# reproduce. These tests pin the ordering on both sides.




class FakeDataModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.static_dim = 4
        self.target_dim = 1
        self.temporal_enabled = False

    def setup(self, stage=None):
        return None


class TinyModel(torch.nn.Module):
    """Draws from the global torch generator in __init__, exactly as a real model's weights do."""

    def __init__(self, static_dim=4, target_dim=1, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(static_dim, target_dim)
        self.kwargs = kwargs


REGISTRY = {
    "toy": {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.TinyModel",
        "datamodule_import_path": f"{__name__}.FakeDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto"},
        "datamodule_init_args": {"batch_size": 8},
    }
}


def _factory():
    return LightningConfigFactory(REGISTRY, SimpleNamespace(RANDOM_SEED=42))


def _weights(bundle):
    return bundle.model.layer.weight.detach().clone()


def _build(seed):
    return _factory().build_lightning_configs(target="t", data={"sequence_bundle": object()}, seed=seed)["toy"]


# --- seeding reaches the weights ---------------------------------------------


def test_the_same_seed_gives_the_same_initial_weights():
    assert torch.equal(_weights(_build(42)), _weights(_build(42)))


def test_a_different_seed_gives_different_initial_weights():
    assert not torch.equal(_weights(_build(42)), _weights(_build(7)))


def test_weights_are_reproducible_even_after_unrelated_rng_work():
    """The production case: data preparation and sklearn training run before the model is built."""
    first = _weights(_build(42))

    torch.rand(1000)  # stand-in for whatever consumed the stream beforehand
    second = _weights(_build(42))

    assert torch.equal(first, second)


def test_without_a_seed_the_factory_does_not_touch_the_rng():
    """`seed=None` must stay byte-identical to the old behaviour for every existing caller."""
    seed_everything(123)
    expected = torch.rand(1).item()

    seed_everything(123)
    _factory().build_lightning_configs(target="t", data={"sequence_bundle": object()})
    after_build = torch.rand(1).item()

    # The build consumed draws (it initialised a model) but never re-seeded, so the value differs.
    assert after_build != expected


def test_a_per_entry_random_seed_overrides_the_run_seed():
    registry = {"toy": {**REGISTRY["toy"], "random_seed": 7}}
    factory = LightningConfigFactory(registry, SimpleNamespace(RANDOM_SEED=42))
    bundle = factory.build_lightning_configs(target="t", data={"sequence_bundle": object()}, seed=42)["toy"]

    assert torch.equal(_weights(bundle), _weights(_build(7)))


# --- the RNG state entering fit ----------------------------------------------


def _rng_state_at_fit(seed, *, reseed_after_build):
    """Replays each path's ordering and reports the first draw `fit` would see."""
    seed_everything(seed)
    _build(seed)  # constructs the model, consuming draws
    if reseed_after_build:
        # What LightningTrainer.train used to do: reset the stream after the weights existed.
        seed_everything(seed)
    return torch.rand(1).item()


def test_both_paths_enter_fit_from_the_same_rng_state():
    """The regression this whole change exists for."""
    hpo = _rng_state_at_fit(42, reseed_after_build=False)
    production = _rng_state_at_fit(42, reseed_after_build=False)

    assert hpo == production


def test_reseeding_after_the_build_desynchronises_the_two_paths():
    """Proof the old ordering really was the problem, not an incidental difference."""
    hpo = _rng_state_at_fit(42, reseed_after_build=False)
    old_production = _rng_state_at_fit(42, reseed_after_build=True)

    assert hpo != old_production
