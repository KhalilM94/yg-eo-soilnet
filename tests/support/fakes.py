"""Stand-ins for MLflow shared by the logger tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace


class RecordingRuns:
    """Stands in for mlflow.start_run, remembering the run names it was asked to open."""

    def __init__(self):
        self.names: list[str] = []

    def __call__(self, run_name=None, nested=False, **kwargs):
        self.names.append(run_name)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="run")))
