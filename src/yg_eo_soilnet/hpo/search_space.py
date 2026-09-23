"""The YAML :term:`search space`: which settings to try, and between what limits.

A search space is data rather than code, and it lives beside the model list it tunes, so the models
themselves know nothing about tuning. Each entry names a setting by its path in the model-list entry
(``model.learning_rate``), how to draw it (a range, a choice, a log scale), and optionally a
condition under which it applies at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import optuna
import yaml

from yg_eo_soilnet.hpo.constraints import ConstraintHook, resolve_constraints, split_derive_entry, split_when
from yg_eo_soilnet.hpo.overrides import to_builtin, validate_override_keys

DIRECTIONS = {"minimize": "min", "maximize": "max"}

DEFAULT_OBJECTIVE_METRIC = "val_r2"
DEFAULT_OBJECTIVE_DIRECTION = "maximize"

# Exactly what SoilRegressionLightningBase logs: `{stage}_loss` in _shared_step, `{stage}_r2` and
# `{stage}_pred_std_ratio` in _log_epoch_metrics. Every Lightning model in the registry inherits it.
#
# Checked at load time on purpose. An objective naming a metric nothing logs is not detectable at
# runtime until a trial finishes and finds nothing to score - at which point every trial is pruned
# and a study of any length yields nothing. Failing here costs a second instead of a night.
KNOWN_METRICS = frozenset(
    f"{stage}_{name}"
    for stage in ("train", "val", "test")
    # loss_base/loss_penalty are the two halves a composite loss reports separately - they are
    # logged only when loss_name is correlation_penalty or cosine, so an objective naming one is
    # valid but will find nothing under any other loss.
    for name in ("loss", "loss_base", "loss_penalty", "r2", "pred_std_ratio")
)

_SPACE_KEYS = {"objective", "sampler", "pruner", "fixed", "params", "derive"}

# How many hex characters of the search-space digest go into a study name. Six is short enough to
# type and read in a log line, and collisions do not silently corrupt anything: the full digest is
# stored on the study and compared on resume.
FINGERPRINT_CHARS = 6


def describe_derive(entry: Any) -> str:
    """A one-line rendering of a ``derive:`` entry, for recording with the study."""
    name, options = split_derive_entry(entry)
    if not options:
        return name
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(options.items()))
    return f"{name}({rendered})"


@dataclass(frozen=True)
class Objective:
    """What a study is trying to make better, and which score it watches.

    Attributes
    ----------
    metric : str
        The score, usually ``val_loss``.
    direction : str
        ``"minimize"`` or ``"maximize"``.
    """

    metric: str = DEFAULT_OBJECTIVE_METRIC
    direction: str = DEFAULT_OBJECTIVE_DIRECTION

    def __post_init__(self) -> None:
        """Check the direction is one of the two, and the metric is named."""
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"objective.direction must be one of {', '.join(sorted(DIRECTIONS))}; got {self.direction!r}."
            )
        if self.metric not in KNOWN_METRICS:
            hint = ""
            if "mse" in self.metric.lower() or "loss" in self.metric.lower():
                hint = (
                    " For mean squared error use 'val_loss' with direction 'minimize' - the models "
                    "log the loss under that name, and loss_name: mse makes it the MSE."
                )
            raise ValueError(
                f"objective.metric {self.metric!r} is not logged by any model in this repo. "
                f"Expected one of: {', '.join(sorted(KNOWN_METRICS))}.{hint} "
                f"(If you added a new logged metric, extend KNOWN_METRICS in hpo/search_space.py.)"
            )

    @property
    def mode(self) -> str:
        """The direction as early stopping spells it: ``"min"`` or ``"max"``."""
        return DIRECTIONS[self.direction]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "Objective":
        """Read an objective from its YAML block."""
        mapping = mapping or {}
        return cls(
            metric=str(mapping.get("metric", DEFAULT_OBJECTIVE_METRIC)),
            direction=str(mapping.get("direction", DEFAULT_OBJECTIVE_DIRECTION)),
        )


@dataclass(frozen=True)
class Distribution:
    """One setting to search, and how to draw it.

    Attributes
    ----------
    path : str
        Where it goes in the model-list entry, such as ``model.dropout``.
    kind : str
        How to draw it: a number between limits, a choice from a list, and so on.
    low, high : float, optional
        The limits.
    choices : list, optional
        The values to choose between.
    log : bool
        Draw across orders of magnitude rather than evenly - right for a learning rate.
    when : dict, optional
        Draw it only in trials where the named settings took these values.
    """

    dotted: str
    kind: str
    spec: dict[str, Any] = field(default_factory=dict)
    when: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, dotted: str, mapping: Mapping[str, Any]) -> "Distribution":
        """Read one searched setting from its YAML block."""
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Search space entry {dotted!r} must be a mapping, e.g. {{type: float, low: 0, high: 1}}.")
        spec = dict(mapping)
        when = dict(spec.pop("when", {}) or {})
        kind = str(spec.pop("type", "")).lower()

        if kind == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"{dotted!r}: a categorical needs a non-empty 'choices' list.")
            # Optuna stores categorical choices in the study database, so they must be scalars.
            # A list-valued hyperparameter (head_hidden_dims) belongs in a constraint hook.
            for choice in choices:
                if not isinstance(choice, (bool, int, float, str)) and choice is not None:
                    raise ValueError(
                        f"{dotted!r}: categorical choices must be null, bool, int, float or str; "
                        f"got {type(choice).__name__}. Use a 'derive' hook for structured values."
                    )
        elif kind in {"float", "int"}:
            missing = [bound for bound in ("low", "high") if bound not in spec]
            if missing:
                raise ValueError(f"{dotted!r}: a {kind} distribution needs {' and '.join(missing)}.")
            if spec.get("log") and spec.get("step") is not None:
                raise ValueError(f"{dotted!r}: Optuna rejects 'log' and 'step' together.")
            if float(spec["low"]) > float(spec["high"]):
                raise ValueError(f"{dotted!r}: low ({spec['low']}) is above high ({spec['high']}).")
        else:
            raise ValueError(f"{dotted!r}: unknown type {kind!r}; expected float, int or categorical.")

        return cls(dotted=dotted, kind=kind, spec=spec, when=when)

    def applies(self, chosen: Mapping[str, Any]) -> bool:
        """Whether this setting is drawn in this trial, given what has been drawn already.

        A condition on a setting that was itself not drawn counts as not applying, rather than as an error.
        """
        return all(dotted in chosen and chosen[dotted] == expected for dotted, expected in self.when.items())

    def suggest(self, trial: optuna.Trial) -> Any:
        """Draw a value for this trial."""
        if self.kind == "categorical":
            return trial.suggest_categorical(self.dotted, self.spec["choices"])
        low, high, step = self.spec["low"], self.spec["high"], self.spec.get("step")
        log = bool(self.spec.get("log", False))
        if self.kind == "int":
            return trial.suggest_int(self.dotted, int(low), int(high), step=int(step or 1), log=log)
        if step is None:
            return trial.suggest_float(self.dotted, float(low), float(high), log=log)
        return trial.suggest_float(self.dotted, float(low), float(high), step=float(step))

    def describe(self) -> str:
        """A one-line rendering of this setting, for recording with the study."""
        if self.kind == "categorical":
            body = "|".join(str(choice) for choice in self.spec["choices"])
        else:
            body = f"{self.spec['low']}..{self.spec['high']}"
            if self.spec.get("log"):
                body += " log"
            if self.spec.get("step") is not None:
                body += f" step={self.spec['step']}"
        guard = f" when {self.when}" if self.when else ""
        return f"{self.kind}({body}){guard}"


def load_search_spaces_document(path: str) -> dict:
    """Read the search spaces, one per model.

    Parameters
    ----------
    path : str
        A file, a folder of files, or a folder containing the file - so where the spaces live is a
        matter of what you point at.

    Returns
    -------
    dict
        Model name to its search space, as read from the YAML.
    """
    document: dict = {}
    if os.path.isdir(path):
        split_dir = path
    else:
        split_dir, _ext = os.path.splitext(path)
        if os.path.exists(path):
            with open(path, "r") as handle:
                document = yaml.safe_load(handle) or {}

    # Tracks which file each entry came from, purely so a collision names both real files instead
    # of always blaming `path` - a second split file colliding with a FIRST one would otherwise be
    # misreported as colliding with `path` itself.
    sources = {name: path for name in document}

    if os.path.isdir(split_dir):
        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith((".yml", ".yaml")):
                continue
            file_path = os.path.join(split_dir, filename)
            with open(file_path, "r") as handle:
                entries = yaml.safe_load(handle) or {}
            for name, spec in entries.items():
                if name in document:
                    raise ValueError(
                        f"Search space '{name}' is declared both in {sources[name]} and in "
                        f"{file_path} - remove it from one of the two."
                    )
                document[name] = spec
                sources[name] = file_path

    return document


@dataclass
class SearchSpace:
    """A whole study's declaration: what to improve, what to fix, and what to search.

    Attributes
    ----------
    entry : str
        Which model in the model list it tunes.
    objective : Objective
        What to make better.
    fixed : dict
        Settings pinned for every trial.
    params : list of Distribution
        Settings to search.
    constraints : list
        Named hooks for settings that depend on one another; see
        :mod:`yg_eo_soilnet.hpo.constraints`.
    """

    entry: str
    objective: Objective
    fixed: dict[str, Any] = field(default_factory=dict)
    distributions: list[Distribution] = field(default_factory=list)
    # Each entry is a hook name, or a single-key mapping of a hook name to its options.
    derive: list[Any] = field(default_factory=list)
    sampler_spec: dict[str, Any] = field(default_factory=dict)
    pruner_spec: dict[str, Any] = field(default_factory=dict)
    _hooks: list[ConstraintHook] = field(default_factory=list, repr=False)

    @classmethod
    def from_mapping(cls, entry: str, mapping: Mapping[str, Any]) -> "SearchSpace":
        """Read a search space from its YAML block."""
        unknown = sorted(set(mapping) - _SPACE_KEYS)
        if unknown:
            raise ValueError(
                f"Search space {entry!r} has unknown key(s): {', '.join(unknown)}. "
                f"Expected any of: {', '.join(sorted(_SPACE_KEYS))}."
            )

        fixed = {key: to_builtin(value) for key, value in (mapping.get("fixed") or {}).items()}
        validate_override_keys(fixed, searched=False)

        params = mapping.get("params") or {}
        if not params and not mapping.get("derive"):
            raise ValueError(f"Search space {entry!r} declares no 'params' and no 'derive' hooks.")
        validate_override_keys(params, searched=True)

        distributions = [Distribution.from_mapping(dotted, spec) for dotted, spec in params.items()]

        # A `when:` guard may only reference a parameter drawn earlier - dict order is the draw
        # order, so a forward reference would silently never match - or one pinned under `fixed:`,
        # which suggest() puts in place before anything is drawn.
        drawn: set[str] = set(fixed)
        for distribution in distributions:
            for guarded in distribution.when:
                if guarded not in drawn:
                    raise ValueError(
                        f"{distribution.dotted!r}: 'when' references {guarded!r}, which is not "
                        f"declared before it or pinned under 'fixed'. Move {guarded!r} earlier in "
                        "the 'params' block."
                    )
            drawn.add(distribution.dotted)

        derive = list(mapping.get("derive") or [])
        # Derive hooks run after every draw, so their guards may name any fixed or searched key.
        for entry in derive:
            name, options = split_derive_entry(entry)
            unknown = [key for key in split_when(options)[1] if key not in drawn]
            if unknown:
                raise ValueError(
                    f"derive entry {name!r}: 'when' references {unknown}, which is neither pinned "
                    "under 'fixed' nor declared under 'params', so it could never match."
                )
        return cls(
            entry=entry,
            objective=Objective.from_mapping(mapping.get("objective")),
            fixed=fixed,
            distributions=distributions,
            derive=derive,
            sampler_spec=dict(mapping.get("sampler") or {}),
            pruner_spec=dict(mapping.get("pruner") or {}),
            _hooks=resolve_constraints(derive),
        )

    @classmethod
    def from_yaml(cls, path: str, entry: str) -> "SearchSpace":
        """Read one model's search space from a file or folder."""
        document = load_search_spaces_document(path)
        if entry not in document:
            available = ", ".join(sorted(document)) or "(none)"
            raise KeyError(f"No search space for registry entry {entry!r} in {path}. Available: {available}.")
        return cls.from_mapping(entry, document[entry] or {})

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one full set of settings for a trial: the pinned ones, then the searched ones."""
        chosen: dict[str, Any] = dict(self.fixed)
        for distribution in self.distributions:
            if distribution.applies(chosen):
                chosen[distribution.dotted] = to_builtin(distribution.suggest(trial))
        for hook in self._hooks:
            hook(trial, chosen)
        return {key: to_builtin(value) for key, value in chosen.items()}

    def make_sampler(self) -> optuna.samplers.BaseSampler:
        """Build the strategy that chooses what to try next."""
        spec = dict(self.sampler_spec)
        name = str(spec.pop("name", "tpe")).lower()
        if name == "tpe":
            return optuna.samplers.TPESampler(**spec)
        if name == "random":
            return optuna.samplers.RandomSampler(**spec)
        raise ValueError(f"Unknown sampler {name!r}; expected 'tpe' or 'random'.")

    def make_pruner(self) -> optuna.pruners.BasePruner:
        """Build the rule that abandons a trial going nowhere."""
        spec = dict(self.pruner_spec)
        name = str(spec.pop("name", "median")).lower()
        if name == "median":
            return optuna.pruners.MedianPruner(**spec)
        if name == "hyperband":
            return optuna.pruners.HyperbandPruner(**spec)
        if name == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unknown pruner {name!r}; expected 'median', 'hyperband' or 'none'.")

    def describe(self) -> dict[str, str]:
        """A flat summary of the space, recorded with the study's run."""
        described = {f"space.{distribution.dotted}": distribution.describe() for distribution in self.distributions}
        described.update({f"fixed.{key}": str(value) for key, value in self.fixed.items()})
        if self.derive:
            described["space.derive"] = ", ".join(describe_derive(entry) for entry in self.derive)
        return described

    def fingerprint(self) -> str:
        """A short digest of everything that makes two trials comparable - the :term:`fingerprint`.

        The study's name ends with it, so editing a limit, the objective or a pinned value starts a new
        study rather than mixing trials that were not run under the same rules.
        """
        payload = {
            "metric": self.objective.metric,
            "direction": self.objective.direction,
            **self.describe(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return digest[:FINGERPRINT_CHARS]
