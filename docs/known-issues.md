# Known issues

Things that do not work the way the configuration or the help text suggests. They were found while
writing this documentation and are recorded here rather than quietly fixed, so that nothing about
how the code behaves changed underneath you. None of them affects a model's scores.

## Command-line tools

## Configuration

**Some settings default differently in code and in YAML.** The code's own default is used when a
setting is absent from your configuration file, and two of them disagree with the shipped file:
explanations default to on in code and are `false` in `main_config.yml`; early-stopping patience
defaults to 5 in code and is `100` in `configs/lightning/models/defaults.yml`. Both shipped values
win as long as you keep them in the file.

**`existing_hs_features.prefix` only works as a single string.** Given a list, the prefix match
never fires and no column is ignored. Use `band_names` to name the columns instead.

**Turning on `CLUSTERING_STRATEGY` with a random split fails.** `configs/sklearn/config.yml` has a
clustering block, but the groups it makes are only produced by `split.strategy: spatial_group` in
`main_config.yml`. Switching the block on without that leaves the training looking for groups that
were never made. Use `split.strategy: spatial_group` with `split.group`, which is the supported way
to hold out whole areas.

**There is a `config.yml` and a `model_registry.yml` at the repository root.** Nothing reads them;
the files in use are `configs/main_config.yml` and the two model lists it names.

## Runs and results

**The GPU is assumed.** `configs/lightning/models/defaults.yml` has `accelerator: cuda`, and TabICL
in the scikit-learn model list has `device: "cuda"`. On a machine without an NVIDIA card both have
to be set to `cpu`. The demo configuration already does.

**A run's log file is written to a temporary folder.** It is uploaded to the run when training
finishes, so it survives there; but a run that is killed partway leaves its log in the system
temporary folder, which is cleaned up eventually.

## Naming

**`yg_eo_soilnet/models/config_fatories/`** is spelled that way in the source - a typo for
"factories". Renaming it would break every saved model that records where its class came from, so
it has been left alone.
