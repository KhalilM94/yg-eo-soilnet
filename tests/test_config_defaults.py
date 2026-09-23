"""The guard that keeps ``config.py``'s fallbacks and the shipped configuration in step.

Every setting in ``config.py`` has a hard-coded fallback that applies when your files do not
mention it. Where that fallback disagrees with what the project ships, deleting one line from a
YAML file changes the run without saying so - a different number of training passes, a different
seed, a step quietly switched on.

This file builds two ``Config`` objects - one from the shipped files, one from a minimal file that
declares only what is required - and compares them setting by setting. Anything that differs must
be listed in :data:`INTENTIONAL_DIVERGENCES` with the reason it is allowed to. A new disagreement
therefore fails here with the setting's name, and the choice is explicit: align the fallback, or
write down why the two values should stay apart.
"""

from pathlib import Path

import pytest
import yaml

from config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = PROJECT_ROOT / "configs" / "main_config.yml"
LIGHTNING_DEFAULTS = PROJECT_ROOT / "configs" / "lightning" / "models" / "defaults.yml"

# Settings that hold the loader's own state - file paths it was handed, and the raw YAML blocks it
# read them from. They differ for the uninteresting reason that the two configs are different
# files. The settings derived from those blocks are compared individually instead.
LOADER_STATE = {
    "COMMON_CONFIG",
    "DATA_CONFIG",
    "DATA_SPEC_CONFIG",
    "LIGHTNING_CONFIG",
    "LIGHTNING_MODEL_REGISTRY",
    "MODEL_REGISTRY",
    "SKLEARN_CONFIG",
}

# Settings that name your data: file names, folder names, column names. There is no meaningful
# "shipped value" for these - they describe one dataset - and the four that must never be guessed
# are required outright (tests/test_config.py covers that). Comparing them here would report a
# difference on every run without ever finding a defect.
NAMES_YOUR_DATA = {
    "DATA_FILE",
    "DATA_FOLDER",
    "DATA_ROOT",
    "POINT_ID_COLUMN",
    "ELIMINATED_FEATURES",
    "IGNORED_COLUMNS",
    "LABEL_COLUMNS",
    "MODALITY_PREFIX_MAP",
    "STATIC_CSV_PATH",
    "STATIC_FEATURES_FILE",
    "STATIC_SOURCE",
    "TARGETS_CSV_PATH",
    "TARGETS_FILE",
    "TARGETS_SOURCE",
    "TARGET_COLUMNS",
    "TIMESERIES_CSV_PATH",
    "TIMESERIES_SOURCE",
    "TIME_COLUMN",
}

# The one decision this file exists to make explicit. Each entry says why the code's fallback and
# the shipped value are allowed to disagree; anything not listed here has to be aligned.
INTENTIONAL_DIVERGENCES = {
    "CARRY_LABEL_COLUMNS": (
        "A per-dataset experiment switch, not a project-wide constant: the real configuration "
        "carries the lab columns and the demo does not. Off is the choice that needs no data."
    ),
    "EXPORT_POINT_PREDICTIONS_SKIP_MODELS": (
        "Protective. The fallback skips TabICL because predicting for every point with it takes "
        "hours; the shipped file re-includes it deliberately, having accepted that cost."
    ),
    "SPLIT_TEST_SIZE": (
        "The fallback is only reached when the whole split: block is absent, which the shipped "
        "file never does. 0.2 is the conventional value; 0.15 is this dataset's choice."
    ),
    "SPLIT_VAL_SIZE": (
        "As SPLIT_TEST_SIZE: only reached with no split: block at all."
    ),
    "TEMPORAL_FEATURES_ENABLED": (
        "Off unless a time series is configured. A dataset of static covariates alone is a "
        "legitimate run, and it has no dates to build sequences from."
    ),
    "UNCERTAINTY_ENABLED": (
        "Opt-in: training an ensemble multiplies the cost of every model. The shipped file asks "
        "for it explicitly."
    ),
    "UNCERTAINTY_N_MEMBERS": (
        "Cost scales linearly with it. The fallback of 5 is the YAML's own 'for a report' value; "
        "the shipped 10 is its 'for a final result' one."
    ),
    "UNCERTAINTY_CALIBRATION_METHOD": (
        "sigma and conformal are both supported. conformal is the fallback because it makes no "
        "assumption about the shape of the errors; the shipped sigma is a deliberate choice."
    ),
    "UNCERTAINTY_INTERVAL_METHOD": (
        "As UNCERTAINTY_CALIBRATION_METHOD - the same choice, read by the interval builder."
    ),
    "USE_CONTEXT_FEATURES": (
        "The fallback keeps every declared feature, which is what an unconfigured run should do. "
        "The shipped file switches the context group off to measure what it is worth."
    ),
    "USE_HARMONIC_COORDS": (
        "A per-dataset switch like CARRY_LABEL_COLUMNS, and one that needs coordinates: off is "
        "the only fallback that works for a dataset without them."
    ),
}

# Settings whose fallback lives in config.py but whose shipped value lives in the deep-learning
# model list, which is NOT part of the settings lookup chain - defaults.yml is merged into each
# model entry later, so a code fallback applies whenever a model's entry omits the key. The
# attribute comparison above cannot see these, so they are checked separately.
LIGHTNING_DEFAULT_SETTINGS = {
    ("trainer_args", "max_epochs"): "LIGHTNING_MAX_EPOCHS",
    ("trainer_args", "accelerator"): "LIGHTNING_ACCELERATOR",
    ("trainer_args", "devices"): "LIGHTNING_DEVICES",
    ("trainer_args", "precision"): "LIGHTNING_PRECISION",
    ("trainer_args", "accumulate_grad_batches"): "LIGHTNING_ACCUMULATE_GRAD_BATCHES",
    ("trainer_args", "gradient_clip_val"): "LIGHTNING_GRADIENT_CLIP_VAL",
    ("trainer_args", "log_every_n_steps"): "LIGHTNING_LOG_EVERY_N_STEPS",
    ("callbacks", "early_stopping", "monitor"): "LIGHTNING_EARLY_STOPPING_MONITOR",
    ("callbacks", "early_stopping", "mode"): "LIGHTNING_EARLY_STOPPING_MODE",
    ("callbacks", "early_stopping", "patience"): "LIGHTNING_EARLY_STOPPING_PATIENCE",
    ("callbacks", "checkpoint", "monitor"): "LIGHTNING_CHECKPOINT_MONITOR",
    ("callbacks", "checkpoint", "mode"): "LIGHTNING_CHECKPOINT_MODE",
    ("callbacks", "checkpoint", "save_top_k"): "LIGHTNING_SAVE_TOP_K",
    ("datamodule_init_args", "batch_size"): "LIGHTNING_BATCH_SIZE",
    ("datamodule_init_args", "num_workers"): "LIGHTNING_NUM_WORKERS",
    ("datamodule_init_args", "pin_memory"): "LIGHTNING_PIN_MEMORY",
    ("datamodule_init_args", "persistent_workers"): "LIGHTNING_PERSISTENT_WORKERS",
}

LIGHTNING_INTENTIONAL_DIVERGENCES = {
    "LIGHTNING_ACCELERATOR": (
        "The code is right and the file is not: auto picks the GPU when there is one and still "
        "runs on a machine without a card. Fixed in the model list separately."
    ),
    "LIGHTNING_DEVICES": (
        "As LIGHTNING_ACCELERATOR: asking for one device of a kind the machine does not have "
        "fails, and auto asks for whatever is there."
    ),
    "LIGHTNING_NUM_WORKERS": (
        "11 is the core count of the machine the shipped file was written on. 0 loads data in the "
        "training process, which is the value that works everywhere."
    ),
}


@pytest.fixture(scope="module")
def shipped_config() -> Config:
    """The configuration the project ships, loaded once for the whole file."""
    return Config(config_path=str(SHIPPED_CONFIG))


@pytest.fixture(scope="module")
def minimal_config(tmp_path_factory) -> Config:
    """A configuration declaring only what is required, so every other setting falls back.

    The required settings get placeholder values: nothing here reads a data file, and the point of
    the fixture is what the loader fills in around them.
    """
    folder = tmp_path_factory.mktemp("minimal_config")
    (folder / "data_spec.yml").write_text("TARGET_COLUMNS: [target]\n")
    (folder / "sklearn.yml").write_text("{}\n")
    (folder / "lightning.yml").write_text("{}\n")
    (folder / "main_config.yml").write_text(
        "common:\n"
        "    DATA_FOLDER: data\n"
        "    STATIC_FEATURES_FILE: static.csv\n"
        "    POINT_ID_COLUMN: point_id\n"
        "    DATA_SPEC_PATH: data_spec.yml\n"
        "    SKLEARN_CONFIG_PATH: sklearn.yml\n"
        "    LIGHTNING_CONFIG_PATH: lightning.yml\n"
        f"    SKLEARN_REGISTRY_PATH: {PROJECT_ROOT / 'configs' / 'sklearn' / 'model_registry.yml'}\n"
        f"    LIGHTNING_REGISTRY_PATH: {LIGHTNING_DEFAULTS}\n"
    )
    return Config(config_path=str(folder / "main_config.yml"))


def _comparable_settings(config: Config) -> dict:
    """The settings worth comparing: upper-case, not loader state, not a name from your data."""
    return {
        name: value
        for name, value in vars(config).items()
        if name.isupper()
        and name not in LOADER_STATE
        and name not in NAMES_YOUR_DATA
        and not isinstance(value, dict)  # raw YAML blocks; their settings are compared one by one
    }


def test_every_disagreement_with_the_shipped_configuration_is_written_down(
    shipped_config, minimal_config
):
    """The guard itself: a fallback that disagrees with the shipped file must be listed as such."""
    shipped = _comparable_settings(shipped_config)
    fallbacks = _comparable_settings(minimal_config)

    unexplained = {
        name: (value, fallbacks.get(name))
        for name, value in shipped.items()
        if name in fallbacks
        and value != fallbacks[name]
        and name not in INTENTIONAL_DIVERGENCES
    }
    assert not unexplained, (
        "These settings fall back to a value the shipped configuration disagrees with, so "
        "removing the line from the YAML file changes the run silently:\n"
        + "\n".join(
            f"  {name}: shipped {shipped_value!r}, falls back to {fallback!r}"
            for name, (shipped_value, fallback) in sorted(unexplained.items())
        )
        + "\nEither change the fallback in config.py to match, or add the setting to "
        "INTENTIONAL_DIVERGENCES in this file with the reason it should stay as it is."
    )


def test_the_divergence_table_has_no_stale_entries(shipped_config, minimal_config):
    """An entry left behind after a setting was aligned makes the table lie about the code."""
    shipped = _comparable_settings(shipped_config)
    fallbacks = _comparable_settings(minimal_config)

    stale = [
        name
        for name in INTENTIONAL_DIVERGENCES
        if name in shipped and name in fallbacks and shipped[name] == fallbacks[name]
    ]
    assert not stale, (
        f"These settings now agree with their fallback: {sorted(stale)}. Remove them from "
        "INTENTIONAL_DIVERGENCES."
    )


def test_every_divergence_carries_a_reason():
    """A table of names with empty reasons would pass the guard while explaining nothing."""
    for table in (INTENTIONAL_DIVERGENCES, LIGHTNING_INTENTIONAL_DIVERGENCES):
        for name, reason in table.items():
            assert len(reason.split()) >= 8, f"{name} needs a reason, not a note"


# --- the deep-learning model list, which the comparison above cannot see -------------------


def _lightning_defaults() -> dict:
    """The ``defaults:`` block every deep-learning model inherits from."""
    return yaml.safe_load(LIGHTNING_DEFAULTS.read_text())["defaults"]


def _at(block: dict, path: tuple):
    """Follow a dotted path into the defaults block, or None if it is not there."""
    for key in path:
        if not isinstance(block, dict) or key not in block:
            return None
        block = block[key]
    return block


def test_every_disagreement_with_the_model_list_is_written_down(minimal_config):
    """The model list is merged into each model entry, not looked up - so a missing key falls
    through to config.py, and the two values need to agree for that to be harmless."""
    defaults = _lightning_defaults()

    unexplained = {}
    for path, setting in LIGHTNING_DEFAULT_SETTINGS.items():
        listed = _at(defaults, path)
        fallback = getattr(minimal_config, setting)
        if listed is not None and listed != fallback and setting not in LIGHTNING_INTENTIONAL_DIVERGENCES:
            unexplained[setting] = (".".join(path), listed, fallback)

    assert not unexplained, (
        "These settings are listed in configs/lightning/models/defaults.yml with one value and "
        "fall back to another in config.py. A model entry that omits the key gets the fallback:\n"
        + "\n".join(
            f"  {setting}: {path} is {listed!r}, falls back to {fallback!r}"
            for setting, (path, listed, fallback) in sorted(unexplained.items())
        )
        + "\nEither align the two, or add the setting to LIGHTNING_INTENTIONAL_DIVERGENCES."
    )


def test_the_model_list_still_names_every_setting_this_file_checks():
    """If a key is renamed in the YAML file, the check above would silently stop covering it."""
    defaults = _lightning_defaults()
    missing = [".".join(path) for path in LIGHTNING_DEFAULT_SETTINGS if _at(defaults, path) is None]
    assert not missing, (
        f"These paths are no longer in {LIGHTNING_DEFAULTS.name}: {missing}. Update "
        "LIGHTNING_DEFAULT_SETTINGS, or the settings they cover stop being checked."
    )
