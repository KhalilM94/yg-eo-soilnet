"""Load the Paper A1 configuration: the settings, the covariate groups, and where the code was.

:func:`load_a1_config` is the one way in for every A1 script. It reads ``configs/a1/a1.yml`` (see
:mod:`yg_eo_soilnet.a1.settings`), the covariate-groups file it names, and checks them against the
main configuration: the data block must be the same, and both model stages must be entries of the
scikit-learn model list. With ``check_data=True`` it also reads the data and checks that every
column the groups name is one of the inputs.

The git commit the code was at is recorded on the result (:attr:`A1Config.git_commit`), so every
table and run made from it can say which code produced it.

Covariate groups
----------------
Each group of ``covariate_groups.yml`` lists ``columns`` (exact names) and/or ``prefixes``. A prefix
matches every input starting with it, and also every time-series percentile of a variable starting
with it, so ``"S2_"`` matches ``ts_S2_B4_p10``. ``combined`` is their union. The ``auxiliary`` group
names soil properties rather than columns: its inputs are their stage-1 predictions,
``a1_pred_<property>``, which do not exist until stage 1 has run.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from yg_eo_soilnet.a1.features import PERCENTILE_PREFIX, A1Table
from yg_eo_soilnet.a1.settings import DATA_KEYS, A1Settings, load_settings

#: The group whose inputs are the stage-1 predictions.
AUXILIARY_GROUP = "auxiliary"
#: The union of every group; reserved, never written in the file.
COMBINED_GROUP = "combined"
#: The prefix of a stage-1 prediction column: ``a1_pred_caco3_pct_total``.
STAGE1_PREDICTION_PREFIX = "a1_pred_"

_GROUP_KEYS = ("columns", "prefixes")


def stage1_prediction_column(target: str) -> str:
    """The column holding the stage-1 prediction of one auxiliary property.

    Examples
    --------
    >>> stage1_prediction_column("caco3_pct_total")
    'a1_pred_caco3_pct_total'
    """
    return f"{STAGE1_PREDICTION_PREFIX}{target}"


@dataclass(frozen=True)
class CovariateGroup:
    """One named group of covariates.

    Attributes
    ----------
    name : str
        The group's name.
    columns : tuple of str
        Exact column names. For the ``auxiliary`` group, soil properties instead.
    prefixes : tuple of str
        Column-name prefixes, matched against the inputs and the time-series variables.
    """

    name: str
    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()

    def matches(self, column: str) -> bool:
        """Whether a prefix of this group claims ``column``.

        Examples
        --------
        >>> group = CovariateGroup("satellite", prefixes=("S2_",))
        >>> group.matches("S2_B4"), group.matches("ts_S2_B4_p10"), group.matches("ts_CLIM_ETa_p10")
        (True, True, False)
        """
        return any(_prefix_matches(prefix, column) for prefix in self.prefixes)


@dataclass(frozen=True)
class ResolvedGroups:
    """The covariate groups, resolved against one table's inputs.

    Attributes
    ----------
    groups : dict of str to tuple of str
        Each group's input columns, in the table's order; the auxiliary group's are the stage-1
        prediction columns.
    unassigned : tuple of str
        Inputs no group claims. They are in no model, ``combined`` included.
    """

    groups: Mapping[str, tuple[str, ...]]
    unassigned: tuple[str, ...] = ()

    @property
    def combined(self) -> tuple[str, ...]:
        """Every group's columns together, in group order."""
        return tuple(column for columns in self.groups.values() for column in columns)

    def columns(self, name: str) -> tuple[str, ...]:
        """One group's columns; ``combined`` gives the union."""
        if name == COMBINED_GROUP:
            return self.combined
        if name not in self.groups:
            raise KeyError(f"No covariate group {name!r}; the groups are {[*self.groups, COMBINED_GROUP]}")
        return self.groups[name]


@dataclass(frozen=True)
class CovariateGroups:
    """Every covariate group, in the order of the file.

    Attributes
    ----------
    groups : tuple of CovariateGroup
    source_path : str or None
        The file they were read from.
    """

    groups: tuple[CovariateGroup, ...]
    source_path: Optional[str] = None

    @property
    def names(self) -> tuple[str, ...]:
        """The group names, in order."""
        return tuple(group.name for group in self.groups)

    def resolve(
        self,
        feature_columns: Iterable[str],
        auxiliary_targets: Iterable[str],
        *,
        not_inputs: Iterable[str] = (),
        logger: Optional[logging.Logger] = None,
    ) -> ResolvedGroups:
        """Match every group to the inputs, refusing anything that does not fit.

        Parameters
        ----------
        feature_columns : iterable of str
            The inputs: the static covariates and the time-series percentiles.
        auxiliary_targets : iterable of str
            The properties stage 1 predicts; the auxiliary group may name only these.
        not_inputs : iterable of str
            Columns of the data that are not inputs - the lab columns, ids, coordinates - so that
            naming one says so rather than just "unknown".
        logger : logging.Logger, optional
            Where an input no group claims is reported.

        Returns
        -------
        ResolvedGroups

        Raises
        ------
        ValueError
            If a column or prefix matches no input, a column is claimed by two groups, a group is
            left with nothing, or the auxiliary group names a property stage 1 does not predict.

        Examples
        --------
        >>> groups = covariate_groups_from_mapping(
        ...     {"terrain": {"columns": ["slope"]}, "satellite": {"prefixes": ["S2_"]}})
        >>> resolved = groups.resolve(["slope", "ts_S2_B4_p10", "ts_S2_B4_p90"], [])
        >>> dict(resolved.groups)
        {'terrain': ('slope',), 'satellite': ('ts_S2_B4_p10', 'ts_S2_B4_p90')}
        """
        logger = logger or logging.getLogger(__name__)
        features = list(dict.fromkeys(feature_columns))
        known_features = set(features)
        auxiliary_targets = list(auxiliary_targets)
        not_inputs = set(not_inputs)

        claimed_by: dict[str, str] = {}
        resolved: dict[str, tuple[str, ...]] = {}
        for group in self.groups:
            if group.name == AUXILIARY_GROUP:
                resolved[group.name] = self._resolve_auxiliary(group, auxiliary_targets)
                continue

            for column in group.columns:
                if column not in known_features:
                    reason = (
                        "is a lab column, id or coordinate, not an input"
                        if column in not_inputs
                        else "is not one of the inputs"
                    )
                    raise ValueError(f"Covariate group {group.name!r}: column {column!r} {reason}.")
            for prefix in group.prefixes:
                if not any(_prefix_matches(prefix, column) for column in features):
                    raise ValueError(f"Covariate group {group.name!r}: prefix {prefix!r} matches no input.")

            exact = set(group.columns)
            members = tuple(column for column in features if column in exact or group.matches(column))
            if not members:
                raise ValueError(f"Covariate group {group.name!r} is empty: it matches no input.")
            for column in members:
                if column in claimed_by:
                    raise ValueError(
                        f"Column {column!r} is in two covariate groups, {claimed_by[column]!r} and "
                        f"{group.name!r}. Each input belongs to one group; remove it from one of them "
                        "(it may be matched by a prefix)."
                    )
                claimed_by[column] = group.name
            resolved[group.name] = members

        unassigned = tuple(column for column in features if column not in claimed_by)
        if unassigned:
            shown = ", ".join(unassigned[:10]) + (" ..." if len(unassigned) > 10 else "")
            logger.warning(f"{len(unassigned)} input(s) are in no covariate group, so no model will read them: {shown}")
        return ResolvedGroups(groups=resolved, unassigned=unassigned)

    @staticmethod
    def _resolve_auxiliary(group: CovariateGroup, auxiliary_targets: list[str]) -> tuple[str, ...]:
        """The auxiliary group's stage-1 prediction columns."""
        if group.prefixes:
            raise ValueError(f"Covariate group {AUXILIARY_GROUP!r} names soil properties, so it takes no prefixes.")
        stray = [column for column in group.columns if column not in auxiliary_targets]
        if stray:
            raise ValueError(
                f"Covariate group {AUXILIARY_GROUP!r} names {stray}, which stage 1 does not predict; it may "
                f"only name auxiliary.targets ({auxiliary_targets})."
            )
        return tuple(stage1_prediction_column(target) for target in group.columns)


def load_covariate_groups(path: str) -> CovariateGroups:
    """Read and check a covariate-groups file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a group is empty, holds an unknown key, repeats a column, or is named ``combined``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Covariate-groups file not found: {path}")
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return covariate_groups_from_mapping(raw, source_path=path)


def covariate_groups_from_mapping(raw: Mapping[str, Any], *, source_path: Optional[str] = None) -> CovariateGroups:
    """Build :class:`CovariateGroups` from an already-read mapping, checking its shape.

    The checks here need no data; the ones that do are made by :meth:`CovariateGroups.resolve`.

    Raises
    ------
    ValueError
        If there are no groups, a group is empty, holds an unknown key, repeats a column, or is
        named ``combined``.
    """
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("The covariate-groups file must hold at least one group.")
    groups = []
    for name, values in raw.items():
        name = str(name)
        if name == COMBINED_GROUP:
            raise ValueError(f"{COMBINED_GROUP!r} is the union of the other groups; it cannot be defined.")
        if values is None:
            raise ValueError(f"Covariate group {name!r} is empty: give it columns or prefixes.")
        if not isinstance(values, Mapping):
            raise ValueError(f"Covariate group {name!r} must hold columns: and/or prefixes:; got {values!r}")
        unknown = sorted(set(values) - set(_GROUP_KEYS))
        if unknown:
            raise ValueError(f"Covariate group {name!r} has unknown key(s) {unknown}; expected {list(_GROUP_KEYS)}")
        lists = {}
        for key in _GROUP_KEYS:
            entries = values.get(key) or []
            if isinstance(entries, str):
                raise ValueError(f"Covariate group {name!r}: {key} must be a list; got {entries!r}")
            entries = [str(entry) for entry in entries]
            repeated = sorted({entry for entry in entries if entries.count(entry) > 1})
            if repeated:
                raise ValueError(f"Covariate group {name!r} lists {repeated} twice under {key}.")
            lists[key] = tuple(entries)
        if not lists["columns"] and not lists["prefixes"]:
            raise ValueError(f"Covariate group {name!r} is empty: give it columns or prefixes.")
        groups.append(CovariateGroup(name=name, **lists))
    return CovariateGroups(groups=tuple(groups), source_path=source_path)


@dataclass(frozen=True)
class A1Config:
    """The loaded A1 configuration.

    Attributes
    ----------
    settings : A1Settings
        Everything from ``a1.yml``.
    groups : CovariateGroups
        The covariate groups.
    path : str
        The ``a1.yml`` it was read from.
    git_commit : str or None
        The commit the code was at when loaded; None outside a git checkout.
    git_dirty : bool or None
        Whether the checkout had uncommitted changes, which the commit alone would not describe.
    resolved : ResolvedGroups or None
        The groups resolved against the data, when loaded with ``check_data=True``.
    """

    settings: A1Settings
    groups: CovariateGroups
    path: str
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    resolved: Optional[ResolvedGroups] = field(default=None, compare=False)

    def resolve_groups(self, table: A1Table, logger: Optional[logging.Logger] = None) -> ResolvedGroups:
        """Resolve the covariate groups against one table's inputs.

        Raises
        ------
        ValueError
            See :meth:`CovariateGroups.resolve`.
        """
        not_inputs = set(table.frame.columns) - set(table.feature_columns)
        return self.groups.resolve(
            table.feature_columns, self.settings.auxiliary.targets, not_inputs=not_inputs, logger=logger
        )

    def describe(self) -> dict[str, Any]:
        """The configuration's provenance as flat name/value pairs, ready to be recorded with a run."""
        described: dict[str, Any] = {
            "a1_config_path": self.path,
            "a1_git_commit": self.git_commit or "unknown",
            "a1_git_dirty": self.git_dirty,
            "a1_target": self.settings.target,
            "a1_stage1_estimator": self.settings.stage1.estimator,
            "a1_stage2_estimator": self.settings.stage2.estimator,
            "a1_inner_cv_folds": self.settings.inner_cv_folds,
            "a1_seeds": ",".join(str(seed) for seed in self.settings.seeds),
            "a1_covariate_groups": ",".join(self.groups.names),
        }
        if self.resolved is not None:
            for name, columns in self.resolved.groups.items():
                described[f"a1_n_{name}"] = len(columns)
            described["a1_n_combined"] = len(self.resolved.combined)
            described["a1_n_unassigned"] = len(self.resolved.unassigned)
        return described


def load_a1_config(
    path: str = "configs/a1/a1.yml",
    *,
    check_data: bool = False,
    logger: Optional[logging.Logger] = None,
) -> A1Config:
    """Read and check the whole A1 configuration.

    Parameters
    ----------
    path : str, default "configs/a1/a1.yml"
        The A1 settings file.
    check_data : bool, default False
        Also read the data and check every covariate group against its inputs. Reading the time
        series takes a while on the real data, which is why it is not the default.
    logger : logging.Logger, optional
        Where warnings go.

    Returns
    -------
    A1Config

    Raises
    ------
    FileNotFoundError
        If a file it names does not exist.
    ValueError
        If a setting is wrong, the data block disagrees with the main configuration, a model stage
        names no model-list entry, or (with ``check_data``) a covariate group does not fit the data.

    Examples
    --------
    >>> a1 = load_a1_config("configs/a1/a1_demo.yml")
    >>> a1.groups.names
    ('satellite', 'terrain', 'climate', 'legacy_soil', 'auxiliary')
    >>> a1.git_commit is not None
    True
    """
    logger = logger or logging.getLogger(__name__)
    settings = load_settings(path)
    if not settings.covariate_groups_path:
        raise ValueError(f"{path} must name its covariate_groups file.")
    groups = load_covariate_groups(settings.covariate_groups_path)
    group_names = {group.name for group in groups.groups}
    if AUXILIARY_GROUP not in group_names:
        logger.warning(f"No {AUXILIARY_GROUP!r} covariate group: stage 2 will not read the stage-1 predictions.")

    config = settings.load_config()
    _check_data_block(settings, config)
    _check_estimators(settings, config)

    commit, dirty = git_state(logger=logger)
    resolved = None
    if check_data:
        from yg_eo_soilnet.a1.features import build_table

        table = build_table(settings, config, logger)
        not_inputs = set(table.frame.columns) - set(table.feature_columns)
        resolved = groups.resolve(
            table.feature_columns, settings.auxiliary.targets, not_inputs=not_inputs, logger=logger
        )
    return A1Config(
        settings=settings,
        groups=groups,
        path=os.path.abspath(path),
        git_commit=commit,
        git_dirty=dirty,
        resolved=resolved,
    )


def git_state(repository: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """The commit a git checkout is at, and whether it has uncommitted changes.

    Parameters
    ----------
    repository : str, optional
        A folder inside the checkout; the one holding this package unless given.
    logger : logging.Logger, optional
        Where a missing git is reported.

    Returns
    -------
    tuple of (str or None, bool or None)
        The full commit hash and the dirty flag; ``(None, None)`` outside a checkout or without git.
    """
    logger = logger or logging.getLogger(__name__)
    folder = repository or str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=folder, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning(f"Could not read the git commit ({error}); results will not record which code made them.")
        return None, None
    return commit, bool(status.strip())


def _check_data_block(settings: A1Settings, config) -> None:
    """Refuse a ``data:`` block that names other data than the main configuration does."""
    if not settings.data:
        return
    main_data = dict(getattr(config, "DATA_CONFIG", {}) or {})
    for key in DATA_KEYS:
        if key not in settings.data:
            continue
        mine, theirs = settings.data[key], main_data.get(key)
        if theirs is None or os.path.normpath(str(mine)) != os.path.normpath(str(theirs)):
            raise ValueError(
                f"data.{key} in {settings.source_path} is {mine!r} but the main configuration "
                f"({settings.main_config_path}) has {theirs!r}. The two must name the same data; change both."
            )


def _check_estimators(settings: A1Settings, config) -> None:
    """Refuse a model stage whose estimator is not in the model list, or whose class does not import."""
    registry = getattr(config, "MODEL_REGISTRY", {}) or {}
    for stage in ("stage1", "stage2"):
        stage_settings = getattr(settings, stage)
        name = stage_settings.estimator
        if stage_settings.import_path:
            from yg_eo_soilnet.models import ModelConfigFactory  # local: avoids a circular import

            try:
                ModelConfigFactory._dynamic_import(stage_settings.import_path)
            except (ImportError, AttributeError, ValueError) as error:
                raise ValueError(f"{stage}.import_path {stage_settings.import_path!r} does not import: {error}")
        if name not in registry:
            raise ValueError(
                f"{stage}.estimator {name!r} is not an entry of the scikit-learn model list "
                f"({getattr(config, 'SKLEARN_REGISTRY_PATH', 'configs/sklearn/model_registry.yml')}); "
                f"its entries are {sorted(registry)}."
            )


def _prefix_matches(prefix: str, column: str) -> bool:
    """Whether ``prefix`` starts ``column``, or the time-series variable a percentile column summarises."""
    if column.startswith(prefix):
        return True
    return column.startswith(PERCENTILE_PREFIX) and column[len(PERCENTILE_PREFIX) :].startswith(prefix)
