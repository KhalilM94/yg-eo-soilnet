# Known issues

Things that do not work the way the configuration or the help text suggests. They were found while
writing this documentation and are recorded here rather than quietly fixed, so that nothing about
how the code behaves changed underneath you. None of them affects a model's scores.

## Command-line tools

## Configuration

**There is a `config.yml` and a `model_registry.yml` at the repository root.** Nothing reads them;
the files in use are `configs/main_config.yml` and the two model lists it names.

## Runs and results

**A run's log file is written to a temporary folder.** It is uploaded to the run when training
finishes, so it survives there; but a run that is killed partway leaves its log in the system
temporary folder, which is cleaned up eventually.

## Naming

**`yg_eo_soilnet/models/config_fatories/`** is spelled that way in the source - a typo for
"factories". Renaming it would break every saved model that records where its class came from, so
it has been left alone.
