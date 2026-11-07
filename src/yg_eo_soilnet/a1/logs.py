"""What the A1 scripts tell you while they run: the console, and a log file that keeps the history.

Every A1 script calls :func:`setup_logging` first. Progress goes to the console and is appended to a
log file in the script's output folder (``run_a1.log`` in the run folder, for instance), so a run
stopped half-way keeps its history and a long one can be followed with ``tail -f``. Progress and
warnings go to standard error; the end summary (:func:`summary`) to standard output. ``--quiet``
keeps only the warnings and the end summary on the console; the log file still gets everything.

``force=True`` matters: importing MLflow's PyTorch support calls ``logging.basicConfig`` at the
ERROR level, which would otherwise make every later attempt to show progress a silent no-op.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

#: How every line starts: when, and how serious.
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
#: Libraries that would bury the progress lines in their own; only their warnings are shown.
NOISY_LOGGERS = (
    "mlflow",
    "urllib3",
    "matplotlib",
    "PIL",
    "fiona",
    "pyogrio",
    "rasterio",
    "numba",
    "xgboost",
    "git",
    "alembic",
    "sqlalchemy",
    "httpx",
    "filelock",
    "fsspec",
    "asyncio",
)
#: The logger :func:`summary` writes through, so the summary reaches the log file too.
SUMMARY_LOGGER = "yg_eo_soilnet.a1.summary"


class _Summary(logging.Filter):
    """Pass only the end summary (``keep=True``), or everything but it (``keep=False``)."""

    def __init__(self, keep: bool):
        super().__init__()
        self.keep = keep

    def filter(self, record: logging.LogRecord) -> bool:
        return (record.name == SUMMARY_LOGGER) == self.keep


def setup_logging(log_file, *, quiet: bool = False) -> Path:
    """Show progress on the console and append it to ``log_file``.

    Parameters
    ----------
    log_file : str or pathlib.Path
        The log file; its folder is created. Appended to, so every invocation's lines are kept.
    quiet : bool, default False
        Show only warnings, errors and the end summary on the console. The file still gets every
        line.

    Returns
    -------
    pathlib.Path
        The log file.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Progress and warnings on stderr; the end summary on stdout, plain, as the scripts printed it.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING if quiet else logging.INFO)
    console.addFilter(_Summary(keep=False))
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    result = logging.StreamHandler(sys.stdout)
    result.addFilter(_Summary(keep=True))
    result.setFormatter(logging.Formatter("%(message)s"))
    to_file = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    to_file.setLevel(logging.INFO)
    to_file.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[console, result, to_file], force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(__name__).info(f"$ {shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])} -> log {log_file}")
    return log_file


def summary(message: str) -> None:
    """Write the end summary: on standard output, even with ``--quiet``, and in the log file."""
    logging.getLogger(SUMMARY_LOGGER).info(message)


def format_duration(seconds: float) -> str:
    """A duration a person reads at a glance.

    Examples
    --------
    >>> format_duration(4.23), format_duration(134), format_duration(3790)
    ('4.2 s', '2m14s', '1h03m')
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
