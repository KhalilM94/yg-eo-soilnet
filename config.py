"""Read the project's YAML configuration files into one :class:`Config` object.

A run is configured by a main file (``configs/main_config.yml`` by default) which names the other
files it needs, relative to its own folder:

- ``DATA_SPEC_PATH`` - which columns are targets, lab columns, categorical, ignored
  (``configs/data_spec.yml``);
- ``SKLEARN_CONFIG_PATH`` and ``LIGHTNING_CONFIG_PATH`` - options for the two model families;
- ``SKLEARN_REGISTRY_PATH`` and ``LIGHTNING_REGISTRY_PATH`` - the two :term:`model lists
  <model registry>`, with ``enabled: true/false`` per model.

Every setting ends up as an upper-case attribute of :class:`Config` (``config.TARGET_COLUMNS``,
``config.SPLIT_TEST_SIZE``, ...), which every other part of the code reads.

Where a setting's value comes from, first match wins:

1. an environment variable with the same name (``RANDOM_SEED=7 python main.py``);
2. the ``common:`` block of the main file;
3. ``data_spec.yml``, then the scikit-learn config, then the Lightning config;
4. the top level of the main file;
5. the default written in this module.

Nested blocks (``split:``, ``uncertainty:``, ``data_quality:``, ``export_point_predictions:``)
have their own flat environment-variable names, listed in the configuration guide.
"""

import os
import warnings
import yaml
import json
from copy import deepcopy
from typing import Any, Mapping, Optional

#: The top-level key of a Lightning model-list file that holds settings shared by every model.
LIGHTNING_REGISTRY_DEFAULTS_KEY = 'defaults'


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Merge ``override`` on top of ``base``, recursing into nested dictionaries.

    Where both sides hold a dictionary, they are merged key by key; anything else in ``override`` (a
    number, a string, a list) replaces the value in ``base``. So a model entry that sets only
    ``trainer_args.max_epochs`` keeps the other shared trainer settings, while a list such as
    ``head_hidden_dims: [64]`` replaces the default list outright.

    Parameters
    ----------
    base : Mapping
        The shared defaults.
    override : Mapping
        The values that win.

    Returns
    -------
    dict
        A new dictionary; neither input is modified.

    Examples
    --------
    >>> deep_merge(
    ...     {"trainer": {"max_epochs": 500, "devices": 1}, "dims": [64, 32]},
    ...     {"trainer": {"max_epochs": 100}, "dims": [16]},
    ... )
    {'trainer': {'max_epochs': 100, 'devices': 1}, 'dims': [16]}
    """
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_lightning_registry(registry_path: str) -> dict:
    """Load the deep-learning model list, with the shared ``defaults:`` merged into every model.

    Two layouts are understood:

    - A file holding **only** a ``defaults:`` block (``configs/lightning/models/defaults.yml``) is
      the shared part of a folder: every other ``.yml``/``.yaml`` file beside it is read as one or
      more model entries.
    - A file declaring a model entry itself (such as a :term:`tuned config`) is read on its own,
      so the other tuned files in its folder are not picked up.

    Parameters
    ----------
    registry_path : str
        The ``defaults.yml`` of a model folder, or a single model-list file.

    Returns
    -------
    dict
        Model name to its complete settings, defaults included.

    Raises
    ------
    ValueError
        If the same model name is declared in two files of the folder.

    Examples
    --------
    >>> registry = load_lightning_registry("examples/demo_config/lightning_models/defaults.yml")
    >>> sorted(registry)
    ['soil_cnn']
    >>> registry["soil_cnn"]["trainer_args"]["max_epochs"]   # from defaults.yml
    40
    """
    with open(registry_path, 'r') as f:
        document = yaml.safe_load(f) or {}

    # Merged here, at load time, so everything downstream sees each model's complete settings.
    # A tuned file has no `defaults:` block, so nothing is merged into it.
    defaults = document.pop(LIGHTNING_REGISTRY_DEFAULTS_KEY, None) or {}

    # Which file each entry came from, so a duplicate name reports both files.
    sources = {name: registry_path for name in document}

    if not document:
        models_dir = os.path.dirname(registry_path) or '.'
        own_filename = os.path.basename(registry_path)
        for filename in sorted(os.listdir(models_dir)):
            if filename == own_filename or not filename.endswith(('.yml', '.yaml')):
                continue
            model_file_path = os.path.join(models_dir, filename)
            with open(model_file_path, 'r') as f:
                entries = yaml.safe_load(f) or {}
            for name, spec in entries.items():
                if name in document:
                    raise ValueError(
                        f"Lightning registry entry '{name}' is declared both in {sources[name]} "
                        f"and in {model_file_path} - remove it from one of the two."
                    )
                sources[name] = model_file_path
                document[name] = spec

    return {name: deep_merge(defaults, spec or {}) for name, spec in document.items()}


class Config:
    """All settings of a run, read from the YAML configuration files and the environment.

    Every setting is an upper-case attribute. The ones most code reads:

    - **data**: ``DATA_FOLDER``, ``STATIC_SOURCE``, ``TARGETS_SOURCE``, ``TIMESERIES_SOURCE``
      (full paths to the three data sources), ``POINT_ID_COLUMN``, ``LAT_COLUMN``, ``LON_COLUMN``,
      ``TIME_COLUMN``, ``MODALITY_PREFIX_MAP``;
    - **columns**: ``TARGET_COLUMNS``, ``LABEL_COLUMNS``, ``CATEGORICAL_FEATURES``,
      ``IGNORED_COLUMNS``, ``MULTI_TARGET_MODE``;
    - **split**: ``SPLIT_HOLDOUT_STRATEGY``, ``SPLIT_TEST_SIZE``, ``SPLIT_VAL_SIZE``,
      ``SPLIT_SEED``, ``SPLIT_POPULATION_POLICY``, ``SPLIT_PLAN_PATH``;
    - **models**: ``MODEL_REGISTRY`` (scikit-learn) and ``LIGHTNING_MODEL_REGISTRY``
      (deep learning), each a dict of model name to settings;
    - **optional extras**: ``UNCERTAINTY_*``, ``EXPLAIN_*``, ``EXPORT_POINT_PREDICTIONS*``;
    - **MLflow**: ``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT_NAME``, ``MLFLOW_REGISTER_MODELS``.

    See the module description for the order in which sources are searched.

    Parameters
    ----------
    config_path : str, optional
        The main configuration file. Defaults to the ``CONFIG_PATH`` environment variable, then
        ``configs/main_config.yml``.
    registry_path : str, optional
        The scikit-learn model list. Defaults to the ``MODEL_REGISTRY_PATH`` environment variable,
        then ``SKLEARN_REGISTRY_PATH`` in the main file.
    lightning_registry_path : str, optional
        The deep-learning model list. Defaults to the ``LIGHTNING_MODEL_REGISTRY_PATH`` environment
        variable, then ``LIGHTNING_REGISTRY_PATH`` in the main file.

    Raises
    ------
    FileNotFoundError
        If a configuration file or a model list named by the main file does not exist.

    Examples
    --------
    >>> config = Config(config_path="examples/demo_config/main_config.yml")
    >>> config.TARGET_COLUMNS
    ['organic_matter_g_kg', 'clay_pct', 'ph_water']
    >>> config.SPLIT_HOLDOUT_STRATEGY, config.SPLIT_TEST_SIZE
    ('random', 0.15)
    >>> sorted(config.MODEL_REGISTRY), sorted(config.LIGHTNING_MODEL_REGISTRY)
    (['Ridge'], ['soil_cnn'])
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        registry_path: Optional[str] = None,
        lightning_registry_path: Optional[str] = None,
    ):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'configs/main_config.yml')
        self._config_dir = os.path.dirname(os.path.abspath(self.config_path))

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.DATA_SPEC_CONFIG = {}
        self.SKLEARN_CONFIG = {}
        self.LIGHTNING_CONFIG = {}

        self.COMMON_CONFIG = self._normalize_mapping(self.config.get('common', {}))
        self.DATA_CONFIG = self._normalize_mapping(self.config.get('data', {}))
        common_data = self._normalize_mapping(self.COMMON_CONFIG.get('data', {}))
        if common_data:
            self.DATA_CONFIG = {**self.DATA_CONFIG, **common_data}
        self.data_spec_path = self._resolve_config_path(self._get_config('DATA_SPEC_PATH', self._get_config('data_spec_path', None)))
        self.MAIN_TEMPORAL_FEATURES = self._normalize_mapping(self.config.get('temporal', {}))
        common_temporal = self._normalize_mapping(self.COMMON_CONFIG.get('temporal', {}))
        if common_temporal:
            self.MAIN_TEMPORAL_FEATURES = {**self.MAIN_TEMPORAL_FEATURES, **common_temporal}
        self.sklearn_config_path = self._resolve_config_path(
            self._get_config('SKLEARN_CONFIG_PATH', self._get_config('sklearn_config_path', None))
        )
        self.lightning_config_path = self._resolve_config_path(
            self._get_config('LIGHTNING_CONFIG_PATH', self._get_config('lightning_config_path', None))
        )

        self.DATA_SPEC_CONFIG = self._load_yaml_mapping(self.data_spec_path)
        self.SKLEARN_CONFIG = self._load_yaml_mapping(self.sklearn_config_path)
        self.SKLEARN_CATEGORICAL_CONFIG = self._normalize_mapping(self.SKLEARN_CONFIG.get('categorical', {}))
        self.LIGHTNING_CONFIG = self._load_yaml_mapping(self.lightning_config_path)
        self.EXISTING_HS_FEATURES = self._normalize_mapping(self._get_config('existing_hs_features', {}))

        registry_path_value = (
            registry_path
            or os.getenv('MODEL_REGISTRY_PATH')
            or self._get_config('SKLEARN_REGISTRY_PATH', 'configs/sklearn/model_registry.yml')
        )
        lightning_registry_path_value = (
            lightning_registry_path
            or os.getenv('LIGHTNING_MODEL_REGISTRY_PATH')
            or self._get_config('LIGHTNING_REGISTRY_PATH', 'configs/lightning/models/defaults.yml')
        )

        self.registry_path = self._resolve_config_path(registry_path_value)
        self.lightning_registry_path = self._resolve_config_path(lightning_registry_path_value)

        self.TEMPORAL_FEATURES = {
            **self._normalize_mapping(self.LIGHTNING_CONFIG.get('temporal', {})),
            **self.MAIN_TEMPORAL_FEATURES,
        }
        if not self.TEMPORAL_FEATURES:
            self.TEMPORAL_FEATURES = self._normalize_mapping(self._get_config('TEMPORAL_FEATURES', {}))

        # --- data sources ----------------------------------------------------------------------
        # These name YOUR data, so there is nothing sensible to invent when one is missing: the
        # placeholders that used to stand here were a retired dataset's, and a run that fell back on
        # them looked for a folder that has not existed for a long time. See _require.
        self.DATA_FOLDER = self._require(
            self._get_data_config('root', 'DATA_FOLDER', None),
            'common.data.root',
            'the folder your data files live in',
        )
        self.DATA_ROOT = self.DATA_FOLDER
        self.DATA_INDEX_MANIFEST = self._get_data_config('manifest', 'DATA_INDEX_MANIFEST', None)
        self.DATA_INDEX_MANIFEST_PATH = self._resolve_data_path(self.DATA_INDEX_MANIFEST)
        self.DATA_MANIFEST_PATH = self.DATA_INDEX_MANIFEST_PATH
        self.DATA_FILE = self._get_data_config('static', 'DATA_FILE', None)
        self.STATIC_FEATURES_FILE = self._get_config('STATIC_FEATURES_FILE', self.DATA_FILE)
        self.TARGETS_FILE = self._get_data_config('targets', 'TARGETS_FILE', self.DATA_FILE)
        self.STATIC_FEATURES_FOLDER = self._resolve_data_path(self._get_config('STATIC_FEATURES_FOLDER', None))
        self.TARGETS_FOLDER = self._resolve_data_path(self._get_config('TARGETS_FOLDER', None))
        self.STATIC_CSV_PATH = self._resolve_data_path(self.STATIC_FEATURES_FILE)
        self.TARGETS_CSV_PATH = self._resolve_data_path(self.TARGETS_FILE)
        self.TIMESERIES_FOLDER = self._resolve_data_path(
            self._get_temporal_config('timeseries_folder', 'TIMESERIES_FOLDER', None)
        )
        self.TIMESERIES_CSV_PATH = self._resolve_data_path(
            self._get_data_config('timeseries', 'TIMESERIES_CSV_PATH', None)
            or self._get_temporal_config('timeseries_file', 'TIMESERIES_CSV_PATH', None)
            or self._get_temporal_config('timeseries_csv_path', 'TIMESERIES_CSV_PATH', None)
        )

        # One full path per source; each may be a file or a folder of files.
        self.STATIC_SOURCE = self.STATIC_FEATURES_FOLDER or self.STATIC_CSV_PATH
        self.TARGETS_SOURCE = self.TARGETS_FOLDER or self._explicit_targets_path()
        self.TIMESERIES_SOURCE = self.TIMESERIES_FOLDER or self.TIMESERIES_CSV_PATH
        self.POINT_ID_COLUMN = self._require(
            self._get_config('POINT_ID_COLUMN', None),
            'POINT_ID_COLUMN',
            'the column identifying each point, which the shared split is keyed on',
        )
        self.LAT_COLUMN = self._get_config('LAT_COLUMN', 'lat')
        self.LON_COLUMN = self._get_config('LON_COLUMN', 'lon')
        # Only meaningful with a time series: a covariates-only run has no date column to name.
        self.TIME_COLUMN = self._get_temporal_config('time_column', 'TIME_COLUMN', None)
        if self.TIMESERIES_SOURCE:
            self.TIME_COLUMN = self._require(
                self.TIME_COLUMN,
                'temporal.time_column',
                'the column holding each reading\'s date',
            )
        self.TEMPORAL_FEATURES_ENABLED = self._get_temporal_config('enabled', 'TEMPORAL_FEATURES_ENABLED', False)
        self.MODALITY_PREFIX_MAP = self._normalize_mapping(
            self._get_temporal_config('modality_prefix_map', 'MODALITY_PREFIX_MAP', {})
        )
        self.S1_COLUMNS = self._get_temporal_config('s1_columns', 'S1_COLUMNS', [])
        self.S2_COLUMNS = self._get_temporal_config('s2_columns', 'S2_COLUMNS', [])
        self.MODIS_COLUMNS = self._get_temporal_config('modis_columns', 'MODIS_COLUMNS', [])
        self.LIGHTNING_BATCH_SIZE = self._get_config('LIGHTNING_BATCH_SIZE', 32)
        self.LIGHTNING_VAL_SIZE = self._get_config('LIGHTNING_VAL_SIZE', 0.2)
        self.LIGHTNING_NUM_WORKERS = self._get_config('LIGHTNING_NUM_WORKERS', 0)
        self.LIGHTNING_PIN_MEMORY = self._get_config('LIGHTNING_PIN_MEMORY', False)
        self.LIGHTNING_PERSISTENT_WORKERS = self._get_config('LIGHTNING_PERSISTENT_WORKERS', False)
        self.LIGHTNING_MAX_EPOCHS = self._get_config('LIGHTNING_MAX_EPOCHS', 500)
        self.LIGHTNING_ACCELERATOR = self._get_config('LIGHTNING_ACCELERATOR', 'auto')
        self.LIGHTNING_DEVICES = self._get_config('LIGHTNING_DEVICES', 'auto')
        self.LIGHTNING_PRECISION = self._get_config('LIGHTNING_PRECISION', '32-true')
        self.LIGHTNING_ENABLE_DEFAULT_LOGGER = self._get_config('LIGHTNING_ENABLE_DEFAULT_LOGGER', True)
        self.LIGHTNING_ACCUMULATE_GRAD_BATCHES = self._get_config('LIGHTNING_ACCUMULATE_GRAD_BATCHES', 1)
        self.LIGHTNING_GRADIENT_CLIP_VAL = self._get_config('LIGHTNING_GRADIENT_CLIP_VAL', 1.0)
        self.LIGHTNING_LOG_EVERY_N_STEPS = self._get_config('LIGHTNING_LOG_EVERY_N_STEPS', 5)
        self.MAIN_FILE_LOGGING_ENABLED = self._get_config('MAIN_FILE_LOGGING_ENABLED', True)
        # What happens when a scikit-learn model fails to train.
        self.FAIL_ON_MODEL_ERROR = self._get_config('FAIL_ON_MODEL_ERROR', False)
        self.FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET = self._get_config('FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET', True)
        self.SKLEARN_FILE_LOGGING_ENABLED = self._get_config('SKLEARN_FILE_LOGGING_ENABLED', True)
        self.MLFLOW_EXPERIMENT_EXPORT_ENABLED = self._get_config('MLFLOW_EXPERIMENT_EXPORT_ENABLED', False)
        self.MLFLOW_EXPERIMENT_EXPORT_PATH = self._get_config('MLFLOW_EXPERIMENT_EXPORT_PATH', 'mlflow_exports')
        # Where MLflow records runs ('' means the repository's mlruns/ folder), and under which
        # experiment name.
        self.MLFLOW_TRACKING_URI = self._get_config('MLFLOW_TRACKING_URI', '')
        self.MLFLOW_EXPERIMENT_NAME = self._get_config('MLFLOW_EXPERIMENT_NAME', 'Soil_Model_Training_v2')
        # Register each trained model in the MLflow model registry as a new version of
        # <target>_<model>, so it can be loaded as models:/<name>@champion. Set false for throwaway
        # experiments.
        self.MLFLOW_REGISTER_MODELS = self._get_config('MLFLOW_REGISTER_MODELS', True)
        # --- SHAP explanations ------------------------------------------------------------------
        # When EXPLAIN_ENABLED is false, the shap library is not even imported.
        self.EXPLAIN_ENABLED = self._get_config('EXPLAIN_ENABLED', False)
        self.EXPLAIN_MAX_SAMPLES = self._get_config('EXPLAIN_MAX_SAMPLES', 500)
        self.EXPLAIN_BACKGROUND_SAMPLES = self._get_config('EXPLAIN_BACKGROUND_SAMPLES', 100)
        self.EXPLAIN_MAX_DISPLAY = self._get_config('EXPLAIN_MAX_DISPLAY', 25)
        # Maximum number of model predictions the generic (slow) SHAP explainer may make, used for
        # models that are neither tree-based nor linear. A model that would need more is skipped.
        self.EXPLAIN_MAX_EVALS = self._get_config('EXPLAIN_MAX_EVALS', 200000)
        # Models to explain; empty means every model.
        self.EXPLAIN_MODELS = self._get_config('EXPLAIN_MODELS', [])
        # Models never explained unless EXPLAIN_MODELS names them. TabICL is skipped by default:
        # each of its predictions re-reads the training set, which makes SHAP very slow.
        self.EXPLAIN_SKIP_MODELS = self._get_config('EXPLAIN_SKIP_MODELS', ['TabICL'])
        self.EXPLAIN_FAIL_ON_ERROR = self._get_config('EXPLAIN_FAIL_ON_ERROR', False)

        # --- uncertainty ------------------------------------------------------------------------
        # When enabled, each model is trained UNCERTAINTY_N_MEMBERS times (an ensemble) and every
        # prediction gets a standard deviation and an interval. With calibration source `val`, the
        # scikit-learn models are fitted on the training points only, so the validation points stay
        # unseen for calibration.
        self.UNCERTAINTY_CONFIG = {
            **self._normalize_mapping(self.config.get('uncertainty', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('uncertainty', {})),
        }
        self.UNCERTAINTY_ENABLED = self._get_uncertainty_config('enabled', 'UNCERTAINTY_ENABLED', False)
        self.UNCERTAINTY_N_MEMBERS = int(
            self._get_uncertainty_config('n_members', 'UNCERTAINTY_N_MEMBERS', 5)
        )
        # Ensemble member k uses seed RANDOM_SEED + k * stride, well away from any other seed.
        self.UNCERTAINTY_SEED_STRIDE = int(
            self._get_uncertainty_config('member_seed_stride', 'UNCERTAINTY_SEED_STRIDE', 1000)
        )
        # auto | always | never: resample the training rows for each member. `auto` does it only for
        # models with no randomness of their own (such as Ridge), whose members would otherwise be
        # identical.
        self.UNCERTAINTY_BOOTSTRAP = str(
            self._get_uncertainty_config('bootstrap', 'UNCERTAINTY_BOOTSTRAP', 'auto')
        ).lower()
        # Deep learning only: also predict a per-point noise level (a variance head).
        self.UNCERTAINTY_HETEROSCEDASTIC = self._get_uncertainty_config(
            'heteroscedastic', 'UNCERTAINTY_HETEROSCEDASTIC', True
        )
        # Weight of the beta-NLL loss used with a variance head (Seitzer et al. 2022): 0.0 is plain
        # Gaussian negative log-likelihood, 1.0 fully variance-weighted; 0.5 is the recommended value.
        self.UNCERTAINTY_BETA_NLL = float(
            self._get_uncertainty_config('beta_nll', 'UNCERTAINTY_BETA_NLL', 0.5)
        )
        self.UNCERTAINTY_CALIBRATION = {
            **self._normalize_mapping(self.UNCERTAINTY_CONFIG.get('calibration', {})),
        }
        self.UNCERTAINTY_INTERVAL = {
            **self._normalize_mapping(self.UNCERTAINTY_CONFIG.get('interval', {})),
        }
        # How intervals are built: conformal | gaussian | sigma | none (see
        # yg_eo_soilnet.uncertainty.intervals). Older configs set this as `calibration.method`,
        # which is still read when `interval.method` is absent; `split_conformal` means `conformal`.
        self.UNCERTAINTY_INTERVAL_METHOD = str(
            self._get_interval_config(
                'method',
                'UNCERTAINTY_INTERVAL_METHOD',
                self._get_calibration_config(
                    'method', 'UNCERTAINTY_CALIBRATION_METHOD', 'conformal'
                ),
            )
        ).lower()
        # The same setting under its older name, which some code still reads.
        self.UNCERTAINTY_CALIBRATION_METHOD = self.UNCERTAINTY_INTERVAL_METHOD
        # The share of true values allowed to fall outside the interval: 0.05 means a 95% interval.
        # Not used by the `sigma` method.
        self.UNCERTAINTY_ALPHA = float(
            self._get_interval_config(
                'alpha',
                'UNCERTAINTY_ALPHA',
                self._get_calibration_config('alpha', 'UNCERTAINTY_ALPHA', 0.05),
            )
        )
        # Interval half-width in standard deviations, for `method: sigma` only.
        self.UNCERTAINTY_INTERVAL_K = float(
            self._get_interval_config('k', 'UNCERTAINTY_INTERVAL_K', 1.0)
        )
        # Which errors calibrate the intervals: `val` (the validation points, the same ones for both
        # model families) or `cv_oof` (scikit-learn's cross-validation errors, which keeps the
        # validation points in its fit pool).
        self.UNCERTAINTY_CALIBRATION_SOURCE = str(
            self._get_calibration_config('source', 'UNCERTAINTY_CALIBRATION_SOURCE', 'val')
        ).lower()
        self.UNCERTAINTY_MODELS = self._get_uncertainty_config('models', 'UNCERTAINTY_MODELS', [])
        # TabICL is skipped by default: training it several times needs too much memory.
        self.UNCERTAINTY_SKIP_MODELS = self._get_uncertainty_config(
            'skip_models', 'UNCERTAINTY_SKIP_MODELS', ['TabICL']
        )
        self.UNCERTAINTY_FAIL_ON_ERROR = self._get_uncertainty_config(
            'fail_on_error', 'UNCERTAINTY_FAIL_ON_ERROR', False
        )

        # --- per-point prediction export --------------------------------------------------------
        # One table per run with every model's prediction for every point (not only the test
        # points). Off by default: it costs one prediction pass over all points per model.
        self.EXPORT_PREDICTIONS_CONFIG = {
            **self._normalize_mapping(self.config.get('export_point_predictions', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('export_point_predictions', {})),
        }
        self.EXPORT_POINT_PREDICTIONS = self._get_export_config(
            'enabled', 'EXPORT_POINT_PREDICTIONS', False
        )
        self.EXPORT_POINT_PREDICTIONS_MODELS = self._get_export_config(
            'models', 'EXPORT_POINT_PREDICTIONS_MODELS', []
        )
        # TabICL is skipped by default: predicting every point with it takes hours.
        self.EXPORT_POINT_PREDICTIONS_SKIP_MODELS = self._get_export_config(
            'skip_models', 'EXPORT_POINT_PREDICTIONS_SKIP_MODELS', ['TabICL']
        )
        self.EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR = self._get_export_config(
            'fail_on_error', 'EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR', False
        )

        self.LIGHTNING_EARLY_STOPPING_MONITOR = self._get_config('LIGHTNING_EARLY_STOPPING_MONITOR', 'val_loss')
        self.LIGHTNING_EARLY_STOPPING_MODE = self._get_config('LIGHTNING_EARLY_STOPPING_MODE', 'min')
        self.LIGHTNING_EARLY_STOPPING_PATIENCE = self._get_config('LIGHTNING_EARLY_STOPPING_PATIENCE', 100)
        self.LIGHTNING_CHECKPOINT_MONITOR = self._get_config('LIGHTNING_CHECKPOINT_MONITOR', 'val_loss')
        self.LIGHTNING_CHECKPOINT_MODE = self._get_config('LIGHTNING_CHECKPOINT_MODE', 'min')
        self.LIGHTNING_SAVE_TOP_K = self._get_config('LIGHTNING_SAVE_TOP_K', 1)
        self.RANDOM_SEED = self._get_config('RANDOM_SEED', 42)
        self.TEST_SIZE = self._get_config('TEST_SIZE', 0.2)
        self.CLUSTERING_STRATEGY = self._get_config('CLUSTERING_STRATEGY', None)
        self.CLUSTERING_STRATEGY = self._normalize_mapping(self.CLUSTERING_STRATEGY)
        self.ENABLE_CLUSTERING = self.CLUSTERING_STRATEGY.get('enabled', False)
        # How scikit-learn's cross-validation cuts its fit pool into folds ('kfold' or 'groupkfold').
        # Not the train/validation/test split, which is SPLIT_HOLDOUT_STRATEGY below.
        self.SPLIT_STRATEGY = self._get_config('SPLIT_STRATEGY', 'kfold')

        # --- the train/validation/test split, shared by every model --------------------------
        # Decided once, by point id, before any model is trained. See datamodules/splitting.py.
        self.SPLIT_CONFIG = {
            **self._normalize_mapping(self.config.get('split', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('split', {})),
        }
        # An older config that only sets CLUSTERING_STRATEGY.enabled gets the spatial split.
        legacy_grouped = 'spatial_group' if self.ENABLE_CLUSTERING else 'random'
        self.SPLIT_HOLDOUT_STRATEGY = self._get_split_config(
            'strategy', 'SPLIT_HOLDOUT_STRATEGY', legacy_grouped
        )
        # An older TEST_SIZE key is used when split.test_size is absent.
        self.SPLIT_TEST_SIZE = self._get_split_config('test_size', 'SPLIT_TEST_SIZE', self.TEST_SIZE)
        self.SPLIT_VAL_SIZE = self._get_split_config(
            'val_size', 'SPLIT_VAL_SIZE', self._get_config('LIGHTNING_VAL_SIZE', 0.2)
        )
        self.SPLIT_SEED = self._get_split_config('seed', 'SPLIT_SEED', self.RANDOM_SEED)
        self.SPLIT_POPULATION_POLICY = self._get_split_config(
            'population_policy', 'SPLIT_POPULATION_POLICY', 'intersect'
        )
        self.SPLIT_PLAN_PATH = self._get_split_config('plan_path', 'SPLIT_PLAN_PATH', None)
        # With population_policy `intersect`, stop the run if fewer than this share of the points
        # are usable by every model family: that points to a data problem worth fixing.
        self.SPLIT_MIN_POPULATION_RATIO = self._get_split_config(
            'min_population_ratio', 'SPLIT_MIN_POPULATION_RATIO', 0.5
        )
        self.SPLIT_GROUP_STRATEGY = self._normalize_mapping(
            self._get_split_config('group', 'SPLIT_GROUP_STRATEGY', None)
        ) or dict(self.CLUSTERING_STRATEGY)
        if 'test_size' in self.SPLIT_CONFIG and 'TEST_SIZE' in getattr(self, 'COMMON_CONFIG', {}):
            print(
                "[Warning] Both TEST_SIZE and split.test_size are set; split.test_size wins for the "
                "unified holdout. Remove TEST_SIZE to avoid the ambiguity."
            )

        # --- missing data: one rule for every model ------------------------------------------------
        # A covariate blank in more than MAX_MISSING_COLUMN_RATIO of rows stops the run; below that,
        # gaps are median-filled and flagged. See datamodules/frame_cleaning.py.
        self.DATA_QUALITY_CONFIG = {
            **self._normalize_mapping(self.config.get('data_quality', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('data_quality', {})),
        }
        self.MAX_MISSING_COLUMN_RATIO = self._get_data_quality_config(
            'max_missing_column_ratio', 'MAX_MISSING_COLUMN_RATIO', 0.2
        )
        self.ALLOW_SPARSE_COLUMNS = self._get_data_quality_config(
            'allow_sparse_columns', 'ALLOW_SPARSE_COLUMNS', []
        )
        self.FAIL_ON_SPARSE_COLUMNS = self._get_data_quality_config(
            'fail_on_sparse_columns', 'FAIL_ON_SPARSE_COLUMNS', True
        )

        # --- targets and features ---------------------------------------------------------------
        self.IGNORE_BANDS = self._get_ignore_bands([])
        self.COLUMNS_TO_TRANSFORM = self._get_config('COLUMNS_TO_TRANSFORM', [])
        self.TARGET_COLUMNS = self._get_config('TARGET_COLUMNS', self._get_config('target_columns', []))
        # 'joint': one model predicts every target; 'per_target': one model per target. A model-list
        # entry can override it with its own `multi_target:` key. See yg_eo_soilnet.targets.
        self.MULTI_TARGET_MODE = self._get_config('MULTI_TARGET_MODE', 'joint')
        # Every lab-measured column, whether or not it is a target. Never used as ordinary features.
        self.LABEL_COLUMNS = self._get_config('LABEL_COLUMNS', self._get_config('label_columns', []))
        # Keep the lab columns in the loaded data so soil_cnn can use some of them as auxiliary
        # inputs (its auxiliary_label_columns). They still never become ordinary features.
        self.CARRY_LABEL_COLUMNS = self._get_config('CARRY_LABEL_COLUMNS', False)
        # Feed lat/lon to soil_cnn's location branch, encoded as sine/cosine waves. They are never
        # ordinary features, and the scikit-learn models do not see them.
        self.USE_HARMONIC_COORDS = self._get_config('USE_HARMONIC_COORDS', False)
        # Columns describing a point's surroundings (neighbourhood statistics), grouped so they can
        # be switched off or explained together.
        self.CONTEXT_FEATURES = self._get_config('CONTEXT_FEATURES', [])
        # Switch for that group. Setting it false removes the CONTEXT_FEATURES columns from the
        # inputs; true (the default) keeps them as ordinary features.
        self.USE_CONTEXT_FEATURES = self._get_config('USE_CONTEXT_FEATURES', True)
        self.PREDICTOR_COLUMNS = self._get_config('PREDICTOR_COLUMNS', self._get_config('predictor_columns', []))
        self.IGNORED_COLUMNS = self._get_config('IGNORED_COLUMNS', self._get_config('ignored_columns', []))
        self.TREE_CATEGORICAL_ENCODING = self._get_sklearn_categorical_config('TREE_CATEGORICAL_ENCODING', 'onehot')
        self.TREE_ONEHOT_MAX_CATEGORIES = self._get_sklearn_categorical_config('TREE_ONEHOT_MAX_CATEGORIES', 30)
        self.MIN_FEATURE_COUNT = self._get_sklearn_categorical_config('MIN_FEATURE_COUNT', 10)
        # Also score scikit-learn models on their own training points (r2_train_fit): a large gap
        # to r2_test points to overfitting.
        self.LOG_TRAIN_FIT_METRIC = self._get_config('LOG_TRAIN_FIT_METRIC', True)
        # Models that skip that extra score.
        self.LOG_TRAIN_FIT_METRIC_SKIP_MODELS = self._get_config(
            'LOG_TRAIN_FIT_METRIC_SKIP_MODELS', ['TabICL']
        )
        self.MAX_FEATURE_DROP_RATIO_WARNING = self._get_sklearn_categorical_config(
            'MAX_FEATURE_DROP_RATIO_WARNING', 0.9
        )
        self.CATEGORICAL_FEATURES = self._get_config(
            'CATEGORICAL_FEATURES', self._get_sklearn_categorical_config('CATEGORICAL_FEATURES', [])
        )
        self.EXCLUDE_CATEGORICAL = self._get_sklearn_categorical_config('EXCLUDE_CATEGORICAL', [])
        self.ELIMINATED_FEATURES = self._get_config('ELIMINATED_FEATURES', self.IGNORED_COLUMNS)

        # --- the two model lists ---------------------------------------------------------------
        self.MODEL_REGISTRY = self._load_model_registry()
        self.LIGHTNING_MODEL_REGISTRY = self._load_lightning_model_registry()

        self._validate()

    # --- checks -------------------------------------------------------------

    def _require(self, value: Any, setting: str, describes: str) -> Any:
        """Return a setting that names your own data, or say which one is missing.

        These have no sensible fallback: a guess would send the run at a file, folder or column
        that does not exist, and the failure would surface far from the setting that caused it.

        Parameters
        ----------
        value : object
            What was read, or None when nothing was.
        setting : str
            The setting's name, as it is written in the file.
        describes : str
            What it names, so the message says what to write rather than only what is missing.

        Raises
        ------
        ValueError
            If the setting is absent or empty.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"{setting} is not set in {self.config_path}. It names {describes}, so there is "
                "nothing to fall back on - set it rather than let the run guess."
            )
        return value

    def _validate(self) -> None:
        """Refuse a configuration whose settings contradict each other.

        Checked here, before any data is read, so the message names the setting rather than
        surfacing as a missing column or a missing key somewhere inside training.

        Raises
        ------
        ValueError
            If a column is declared both as a lab column and as a category, or if clustering is
            switched on without the split that builds the clusters.
        """
        self._validate_static_source()
        self._validate_categorical_not_label()
        self._validate_clustering_has_a_spatial_split()

    def _validate_static_source(self) -> None:
        """Refuse a configuration that never says where the static covariates are.

        Checked on the resolved source rather than on one key, because three spellings name it -
        ``common.data.static``, ``STATIC_FEATURES_FILE`` and ``STATIC_FEATURES_FOLDER`` - and a
        manifest can list the files instead.
        """
        if self.STATIC_SOURCE or self.DATA_INDEX_MANIFEST_PATH:
            return
        raise ValueError(
            f"No static data source is set in {self.config_path}. Set common.data.static to the "
            "file (or folder) holding one row per sample point, beside common.data.root."
        )

    def _validate_categorical_not_label(self) -> None:
        """Refuse a column declared as both a lab column and a category.

        A lab column is removed from the inputs before the category handling sees it, so a column in
        both lists reaches no model at all - and did so silently until this check existed.
        """
        both = [
            column
            for column in dict.fromkeys(str(name) for name in (self.CATEGORICAL_FEATURES or []))
            if column in {str(name) for name in (self.LABEL_COLUMNS or [])}
        ]
        if not both:
            return
        raise ValueError(
            f"Column(s) {sorted(both)} are declared in both LABEL_COLUMNS and CATEGORICAL_FEATURES "
            f"in {self.data_spec_path}. A lab column is removed from the feature set before the "
            "category handling sees it, so no model would receive them. Remove each one from "
            "CATEGORICAL_FEATURES to keep it out of the inputs, or from LABEL_COLUMNS to make it a "
            "real categorical input."
        )

    def _validate_clustering_has_a_spatial_split(self) -> None:
        """Refuse clustering without the split that builds the clusters.

        The groups the scikit-learn folds need are written only by ``split.strategy:
        spatial_group``; with any other strategy the training used to stop on a missing key, after
        the data had been loaded.

        A second, milder case only warns: the groups are built, but ``SPLIT_STRATEGY: kfold`` folds
        without them, so the run is fine and the clustering does nothing.
        """
        if not self.ENABLE_CLUSTERING:
            return
        if self.SPLIT_HOLDOUT_STRATEGY != 'spatial_group':
            raise ValueError(
                f"CLUSTERING_STRATEGY is enabled in {self.sklearn_config_path}, but split.strategy "
                f"is {self.SPLIT_HOLDOUT_STRATEGY!r} in {self.config_path}. The clusters the folds "
                "group by are only built by a spatial split, so training would stop on a missing "
                "key part way through. Set split.strategy: spatial_group with a split.group block "
                "to hold out whole areas, or switch CLUSTERING_STRATEGY off."
            )
        if self.SPLIT_STRATEGY != 'groupkfold':
            warnings.warn(
                f"CLUSTERING_STRATEGY is enabled in {self.sklearn_config_path}, but SPLIT_STRATEGY "
                f"is {self.SPLIT_STRATEGY!r} there, which folds without the groups. The clusters "
                "are built and then ignored, so nearby points can still be split across a fold's "
                "training and validation halves. Set SPLIT_STRATEGY: groupkfold to use them.",
                UserWarning,
                stacklevel=2,
            )

    def _get_config(self, key: str, default: Any) -> Any:
        """Return a setting: environment variable, then each config file, then ``default``.

        An environment variable is converted to the type of ``default`` (true/1/yes for booleans,
        comma-separated for lists, JSON for dictionaries).
        """
        val = os.environ.get(key)
        if val is not None:
            if isinstance(default, bool):
                return val.lower() in ('true', '1', 't', 'y', 'yes')
            elif isinstance(default, int):
                return int(val)
            elif isinstance(default, float):
                return float(val)
            elif isinstance(default, list):
                return [x.strip() for x in val.split(',')]
            elif isinstance(default, dict):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format for environment variable {key}: {val}")
            return val
        for section in (
            getattr(self, 'COMMON_CONFIG', {}),
            getattr(self, 'DATA_SPEC_CONFIG', {}),
            getattr(self, 'SKLEARN_CONFIG', {}),
            getattr(self, 'LIGHTNING_CONFIG', {}),
        ):
            if key in section:
                config_val = section[key]
                if isinstance(default, list) and config_val is None:
                    return []
                return config_val
        config_val = self.config.get(key, default)
        if isinstance(default, list) and config_val is None:
            return []
        return config_val

    def _load_yaml_mapping(self, path_value: Optional[str]) -> dict:
        """Read one YAML file into a dictionary; raise ``FileNotFoundError`` if it does not exist."""
        if not path_value:
            raise FileNotFoundError("Expected a config file path, but none was provided.")
        resolved_path = self._resolve_config_path(path_value)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Config YAML not found at {resolved_path}.")
        with open(resolved_path, 'r') as f:
            return self._normalize_mapping(yaml.safe_load(f) or {})

    def _load_model_registry(self):
        """Read the scikit-learn model list."""
        try:
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry YAML not found at {self.registry_path}. Stopping execution.")

    def _load_lightning_model_registry(self):
        """Read the deep-learning model list; see :func:`load_lightning_registry`."""
        try:
            return load_lightning_registry(self.lightning_registry_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Lightning model registry YAML not found at {self.lightning_registry_path}. Stopping execution."
            )

    def _get_data_config(self, data_key: str, flat_key: str, default: Any) -> Any:
        """Return ``data.<data_key>``, or else the older flat key ``flat_key``."""
        if data_key in self.DATA_CONFIG and self.DATA_CONFIG[data_key] is not None:
            return self.DATA_CONFIG[data_key]
        return self._get_config(flat_key, default)

    def _explicit_targets_path(self) -> Optional[str]:
        """Return the targets file's path only if one is configured separately, else ``None``.

        ``None`` means the targets are in the static file.
        """
        if self.DATA_CONFIG.get('targets'):
            return self._resolve_data_path(self.DATA_CONFIG['targets'])
        for key in ('TARGETS_FILE', 'TARGETS_CSV_PATH'):
            value = self._get_config(key, None)
            if value:
                return self._resolve_data_path(value)
        return None

    def _get_temporal_config(self, temporal_key: str, flat_key: str, default: Any) -> Any:
        """Return ``temporal.<temporal_key>``, or else the flat key, or else ``default``."""
        if temporal_key in self.TEMPORAL_FEATURES and self.TEMPORAL_FEATURES[temporal_key] is not None:
            return self.TEMPORAL_FEATURES[temporal_key]
        return self._get_config(flat_key, default)

    def _get_split_config(self, split_key: str, flat_key: str, default: Any) -> Any:
        """Return ``split.<split_key>``, or else the flat key, or else ``default``.

        An environment variable named ``flat_key`` wins over the YAML, e.g.
        ``SPLIT_POPULATION_POLICY=assign_all python main.py``.
        """
        env_value = os.environ.get(flat_key)
        if env_value is None and split_key in self.SPLIT_CONFIG and self.SPLIT_CONFIG[split_key] is not None:
            return self.SPLIT_CONFIG[split_key]
        return self._get_config(flat_key, default)

    def _get_uncertainty_config(self, uncertainty_key: str, flat_key: str, default: Any) -> Any:
        """Return ``uncertainty.<uncertainty_key>``, or else the flat key, or else ``default``.

        An environment variable named ``flat_key`` wins, e.g. ``UNCERTAINTY_N_MEMBERS=2``.
        """
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and uncertainty_key in self.UNCERTAINTY_CONFIG
            and self.UNCERTAINTY_CONFIG[uncertainty_key] is not None
        ):
            return self.UNCERTAINTY_CONFIG[uncertainty_key]
        return self._get_config(flat_key, default)

    def _get_export_config(self, export_key: str, flat_key: str, default: Any) -> Any:
        """Return ``export_point_predictions.<export_key>``, or else the flat key, or else ``default``.

        An environment variable named ``flat_key`` wins, e.g. ``EXPORT_POINT_PREDICTIONS=true``.
        """
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and export_key in self.EXPORT_PREDICTIONS_CONFIG
            and self.EXPORT_PREDICTIONS_CONFIG[export_key] is not None
        ):
            return self.EXPORT_PREDICTIONS_CONFIG[export_key]
        return self._get_config(flat_key, default)

    def _get_interval_config(self, interval_key: str, flat_key: str, default: Any) -> Any:
        """Return ``uncertainty.interval.<interval_key>``, or else the flat key, or else ``default``."""
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and interval_key in self.UNCERTAINTY_INTERVAL
            and self.UNCERTAINTY_INTERVAL[interval_key] is not None
        ):
            return self.UNCERTAINTY_INTERVAL[interval_key]
        return self._get_config(flat_key, default)

    def _get_calibration_config(self, calibration_key: str, flat_key: str, default: Any) -> Any:
        """Return ``uncertainty.calibration.<calibration_key>``, or else the flat key, or else ``default``."""
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and calibration_key in self.UNCERTAINTY_CALIBRATION
            and self.UNCERTAINTY_CALIBRATION[calibration_key] is not None
        ):
            return self.UNCERTAINTY_CALIBRATION[calibration_key]
        return self._get_config(flat_key, default)

    def _get_data_quality_config(self, quality_key: str, flat_key: str, default: Any) -> Any:
        """Return ``data_quality.<quality_key>``, or else the flat key, or else ``default``.

        An environment variable named ``flat_key`` wins, e.g. ``MAX_MISSING_COLUMN_RATIO=0.9``.
        """
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and quality_key in self.DATA_QUALITY_CONFIG
            and self.DATA_QUALITY_CONFIG[quality_key] is not None
        ):
            return self.DATA_QUALITY_CONFIG[quality_key]
        return self._get_config(flat_key, default)

    def _get_sklearn_categorical_config(self, key: str, default: Any) -> Any:
        """Return a key of the scikit-learn config's ``categorical:`` block, or else the flat key."""
        if key in self.SKLEARN_CATEGORICAL_CONFIG and self.SKLEARN_CATEGORICAL_CONFIG[key] is not None:
            return self.SKLEARN_CATEGORICAL_CONFIG[key]
        return self._get_config(key, default)
    
    def _get_ignore_bands(self, default: Any) -> list:
        """Return the band columns to ignore, from ``existing_hs_features`` or ``IGNORE_BANDS``."""
        hs_config = self._normalize_mapping(getattr(self, 'EXISTING_HS_FEATURES', {}))
        if hs_config.get('enabled', False) and hs_config.get('ignore', False):
            band_names = hs_config.get('band_names', [])
            if isinstance(band_names, list) and band_names:
                return [str(name) for name in band_names if name]

            band_count = hs_config.get('band_count', None)
            # One prefix or a list of them, as DataManager.hyperspectral_drop_columns also reads it.
            # A list used to be formatted into the name, giving "['S2_', ...]1".
            prefix = hs_config.get('prefix', '')
            prefixes = [str(one) for one in (prefix if isinstance(prefix, (list, tuple, set)) else [prefix]) if one]
            if band_count and prefixes:
                try:
                    return [f"{one}{i}" for one in prefixes for i in range(1, int(band_count) + 1)]
                except (TypeError, ValueError):
                    return default if isinstance(default, list) else []

        if self._get_config('IGNORE_BANDS', False):
            return [f"Band_{i}" for i in range(1, self._get_config('N_BANDS', 234) + 1)]
        else:
            return default if isinstance(default, list) else []

    def _resolve_data_path(self, path_value: Optional[str]) -> Optional[str]:
        """Return the full path of a data file or folder named relative to ``DATA_FOLDER``."""
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value
        # A full path, so the data loader can never join the data folder on a second time. A
        # relative DATA_FOLDER is read from the directory the command is run in.
        return os.path.abspath(os.path.join(self.DATA_FOLDER, path_value))

    def _resolve_config_path(self, path_value: Optional[str]) -> Optional[str]:
        """Return the path of a config file named in the main file.

        A relative path is looked for next to the main file first, then in the current directory.
        """
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value

        candidate_from_config_dir = os.path.normpath(os.path.join(self._config_dir, path_value))
        if os.path.exists(candidate_from_config_dir):
            return candidate_from_config_dir

        candidate_from_cwd = os.path.normpath(os.path.abspath(path_value))
        if os.path.exists(candidate_from_cwd):
            return candidate_from_cwd

        return candidate_from_config_dir

    @staticmethod
    def _normalize_mapping(value: Any) -> dict:
        """Return ``value`` as a dictionary: a dict as-is, a JSON string parsed, anything else ``{}``."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}