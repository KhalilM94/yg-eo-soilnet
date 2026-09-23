"""Follow how a study is going: the score of each trial, and the best so far.

The question a running study cannot otherwise answer is not "is it alive" but "is the search
working". This keeps the scores as they arrive, so the command line can show a trend, and turns the
finished study into a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from yg_eo_soilnet.hpo.search_space import Objective

# Eight levels, low to high. A pruned or failed trial has no value, and showing a gap where it
# happened is more informative than dropping it - a run of gaps is the pruner doing its job.
BLOCKS = "▁▂▃▄▅▆▇█"
GAP = "·"

COMPLETE = "COMPLETE"


def best_value_or_none(study: optuna.Study) -> float | None:
    """The best score so far, or None when no trial has finished yet."""
    try:
        return float(study.best_value)
    except (ValueError, RuntimeError):
        return None


def best_trial_number_or_none(study: optuna.Study) -> int | None:
    """Which trial holds the best score, or None when none has finished."""
    try:
        return int(study.best_trial.number)
    except (ValueError, RuntimeError):
        return None


@dataclass
class TrialRecord:
    """One trial's outcome: its number, its score, its state and its settings."""

    number: int
    value: float | None
    state: str
    duration_s: float | None = None
    best_epoch: int | None = None
    epochs_run: int | None = None

    @property
    def is_complete(self) -> bool:
        """Whether the trial finished, rather than being abandoned partway.

        Not simply "has a score": an abandoned trial carries its last score too.
        """
        return self.state == COMPLETE

    @classmethod
    def from_trial(cls, trial: optuna.trial.FrozenTrial) -> "TrialRecord":
        """Read one record from a finished trial."""
        duration = getattr(trial, "duration", None)
        return cls(
            number=int(trial.number),
            value=None if trial.value is None else float(trial.value),
            state=trial.state.name,
            duration_s=None if duration is None else duration.total_seconds(),
            best_epoch=trial.user_attrs.get("best_epoch"),
            epochs_run=trial.user_attrs.get("epochs_run"),
        )


class ObjectiveTracker:
    """Keeps every trial's score, and the renderings built from them."""

    def __init__(self, objective: Objective):
        """Start with no trials recorded."""
        self.objective = objective
        self.records: list[TrialRecord] = []
        self.best_value: float | None = None
        self.best_trial: int | None = None

    # --- accumulation ------------------------------------------------------

    def prime(self, study: optuna.Study) -> None:
        """Load the trials a resumed study already holds, so its history is shown too."""
        self.records = [TrialRecord.from_trial(trial) for trial in study.trials]
        self._refresh_best(study)

    def record(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Record one finished trial; passed to the study as a callback."""
        self.records.append(TrialRecord.from_trial(trial))
        self._refresh_best(study)

    def _refresh_best(self, study: optuna.Study) -> None:
        """Update the best score and which trial holds it."""
        self.best_value = best_value_or_none(study)
        self.best_trial = best_trial_number_or_none(study)

    @property
    def counts(self) -> dict[str, int]:
        """How many trials finished, were abandoned, or failed."""
        counts = {"complete": 0, "pruned": 0, "failed": 0}
        for record in self.records:
            if record.state == COMPLETE:
                counts["complete"] += 1
            elif record.state == "PRUNED":
                counts["pruned"] += 1
            elif record.state == "FAIL":
                counts["failed"] += 1
        return counts

    # --- renderings --------------------------------------------------------

    def sparkline(self, width: int = 24) -> str:
        """The last few trials as a row of block characters, tall for good scores."""
        window = self.records[-width:] if width > 0 else []
        if not window:
            return ""

        values = [record.value for record in window if record.is_complete and record.value is not None]
        if not values:
            return GAP * len(window)

        low, high = min(values), max(values)
        span = high - low
        # A flat history divides by zero under naive min-max scaling. Every value being equal is a
        # real state early in a study (or with a degenerate space), not an error, so pin it midway.
        if span <= 0:
            middle = BLOCKS[len(BLOCKS) // 2]
            return "".join(middle if record.is_complete else GAP for record in window)

        return "".join(
            BLOCKS[min(len(BLOCKS) - 1, int((record.value - low) / span * len(BLOCKS)))]
            if record.is_complete and record.value is not None
            else GAP
            for record in window
        )

    def summary_line(self) -> str:
        """One line: the best score, which trial, and how the rest went."""
        counts = self.counts
        if self.best_value is None:
            best = "best=n/a"
        else:
            best = f"best={self.best_value:.4f} (t{self.best_trial})"
        parts = [best, f"ok{counts['complete']}"]
        if counts["pruned"]:
            parts.append(f"pruned{counts['pruned']}")
        if counts["failed"]:
            parts.append(f"failed{counts['failed']}")
        return " ".join(parts)

    def trials_frame(self, study: optuna.Study) -> pd.DataFrame:
        """Every trial with every setting - what goes into ``trials.csv``."""
        frame = study.trials_dataframe()
        return frame if frame is not None else pd.DataFrame()

    def top_frame(self, study: optuna.Study, n: int = 10) -> pd.DataFrame:
        """The best few finished trials, in a few narrow columns.

        The settings are left out: a dozen of them would make the table unreadable in a terminal, and the
        winning set is printed separately.
        """
        completed = [record for record in self.records if record.is_complete and record.value is not None]
        if not completed:
            return pd.DataFrame(columns=["trial", self.objective.metric, "best_epoch", "epochs_run", "duration_s"])

        completed.sort(key=lambda record: record.value, reverse=self.objective.direction == "maximize")
        return pd.DataFrame(
            [
                {
                    "trial": record.number,
                    self.objective.metric: round(record.value, 6),
                    "best_epoch": record.best_epoch,
                    "epochs_run": record.epochs_run,
                    "duration_s": None if record.duration_s is None else round(record.duration_s, 1),
                }
                for record in completed[:n]
            ]
        )

    def best_params(self, study: optuna.Study) -> dict[str, Any]:
        """The settings of the best trial so far."""
        try:
            return dict(study.best_trial.params)
        except (ValueError, RuntimeError):
            return {}
