"""Re-run the best few :term:`trials <trial>`, because the winner is partly luck.

A study reports the best of hundreds of trials, each itself the best epoch of a noisy run: the
headline number is the luckiest draw as much as the best settings, and a retrain usually lands
somewhat worse. :term:`Reranking <rerank>` re-runs the shortlist over several seeds and picks the
best *average*, which is a far better guide to what the settings will actually give.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import optuna
import pandas as pd

from yg_eo_soilnet.hpo.export import OVERRIDES_ATTR
from yg_eo_soilnet.hpo.tracker import COMPLETE

# How close a re-run at the trial's own seed must land to count as reproducing it. Generous, because
# cuDNN kernel selection and atomics still move the last digits even under deterministic=True.
REPRODUCTION_TOLERANCE = 0.02


@dataclass
class RerankResult:
    """One shortlisted trial, re-run several times.

    Attributes
    ----------
    trial_number : int
        Which trial it was.
    values : list of float
        What each re-run scored.
    overrides : dict
        The settings it ran with.
    headline : float
        What the study originally reported for it.
    """

    trial_number: int
    original_value: float
    overrides: dict[str, Any]
    values: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float | None:
        """The average of the re-runs - what this configuration is really worth."""
        return statistics.fmean(self.values) if self.values else None

    @property
    def std(self) -> float:
        """How much the re-runs varied; 0 for a single seed rather than an error."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def reproduced(self) -> bool | None:
        """Whether re-running at the trial's own seed landed back on its original score.

        The first seed is deliberately the trial's own, so this doubles as a check that seeding really
        reaches the weights. False means a run is not reproducible from its seed.
        """
        if not self.values:
            return None
        return abs(self.values[0] - self.original_value) <= REPRODUCTION_TOLERANCE

    @property
    def drift(self) -> float | None:
        """How far the average sits from the headline score - the size of the luck."""
        return None if self.mean is None else self.mean - self.original_value


def top_trials(study: optuna.Study, k: int) -> list[optuna.trial.FrozenTrial]:
    """The best few finished trials, in the study's own order."""
    completed = [t for t in study.trials if t.state.name == COMPLETE and t.value is not None]
    reverse = study.direction.name.lower() == "maximize"
    completed.sort(key=lambda trial: trial.value, reverse=reverse)
    return completed[: max(0, int(k))]


def rerank(
    objective: Any,
    study: optuna.Study,
    *,
    top_k: int = 10,
    seeds: int = 3,
    logger: Any = None,
) -> list[RerankResult]:
    """Re-run each shortlisted trial several times and collect what they score.

    Trained by exactly the same code that ran the trials, so a re-run is comparable with the original.

    Returns
    -------
    list of RerankResult
    """
    candidates = top_trials(study, top_k)
    if not candidates:
        return []

    # A throwaway in-memory study supplies real Trial objects for the runner's messages without
    # writing anything to the study under test. `report=False` keeps them out of the pruner, so a
    # re-run is never cut short by the original study's history.
    scratch = optuna.create_study(direction=study.direction.name.lower())
    results: list[RerankResult] = []

    for position, trial in enumerate(candidates, start=1):
        overrides = trial.user_attrs.get(OVERRIDES_ATTR)
        if overrides is None:
            if logger is not None:
                logger.warning(f"Trial {trial.number} has no {OVERRIDES_ATTR!r} attribute; skipping.")
            continue

        result = RerankResult(trial_number=trial.number, original_value=float(trial.value), overrides=dict(overrides))
        for index in range(max(1, int(seeds))):
            # Seed 0 is the trial's own, so it re-runs the original; the rest are fresh draws.
            seed = objective.seed + index
            try:
                value = _run_once(objective, scratch, result.overrides, seed)
            except Exception as exc:  # noqa: BLE001 - one bad re-run must not end the pass
                if logger is not None:
                    logger.warning(f"Trial {trial.number} re-run at seed {seed} failed: {exc!r}")
                continue
            result.values.append(value)
            result.seeds.append(seed)

        results.append(result)
        if logger is not None:
            prefix = f"[{position}/{len(candidates)}] trial {result.trial_number}"
            if result.values:
                logger.info(
                    f"{prefix}: {result.original_value:.4f} -> {result.mean:.4f} "
                    f"+- {result.std:.4f} over {len(result.values)} seed(s)"
                )
            else:
                logger.warning(f"{prefix}: every re-run failed; excluded from the ranking")

    return results


def _run_once(objective: Any, scratch: optuna.Study, overrides: dict[str, Any], seed: int) -> float:
    """Train one shortlisted trial at one seed."""
    from yg_eo_soilnet.hpo.objective import seed_everything
    from yg_eo_soilnet.hpo.trial_runner import release_dataloader_workers

    bundle = None
    try:
        seed_everything(seed)
        bundle = objective.build_bundle(overrides)
        return objective.runner.run(bundle, scratch.ask(), report=False).value
    finally:
        bundle = None  # noqa: F841
        release_dataloader_workers()


def choose_winner(results: list[RerankResult], direction: str) -> RerankResult | None:
    """The best average, which is often a different trial from the best headline score."""
    scored = [result for result in results if result.mean is not None]
    if not scored:
        return None
    return (max if direction == "maximize" else min)(scored, key=lambda result: result.mean)


def rerank_frame(results: list[RerankResult], metric: str = "value") -> pd.DataFrame:
    """The re-ranking as a table: each trial, its re-runs, its average and its spread."""
    if not results:
        return pd.DataFrame(columns=["trial", f"original_{metric}", f"mean_{metric}", "std", "drift", "seeds"])
    return pd.DataFrame(
        [
            {
                "trial": result.trial_number,
                f"original_{metric}": round(result.original_value, 6),
                f"mean_{metric}": None if result.mean is None else round(result.mean, 6),
                "std": round(result.std, 6),
                "drift": None if result.drift is None else round(result.drift, 6),
                "reproduced": result.reproduced,
                "seeds": len(result.values),
                "values": [round(value, 6) for value in result.values],
            }
            for result in results
        ]
    )
