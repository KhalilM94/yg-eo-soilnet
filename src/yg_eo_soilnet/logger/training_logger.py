"""Where a run's messages go: the terminal, and a log file saved with the run."""

import logging
import os
import sys
import tempfile
from typing import Optional


class TrainingLogger:
    """Set up where a run's messages go: the terminal, and a log file.

    The log file is uploaded with the run at the end, so the messages stay with the results.

    Parameters
    ----------
    name : str, default "ML"
        The logger's name, shown in every line.
    log_dir : str, optional
        Where the log file is written. By default a new temporary directory, which is not cleaned
        up while the run is going: the file is uploaded once training has finished.
    log_filename : str, default "logger"
        The file is named ``<log_filename>_training.log``.
    enable_file_logging : bool, default True
        Write a file at all; false leaves only the terminal.

    Examples
    --------
    >>> logger = TrainingLogger(name="demo").get_logger()
    >>> logger.name
    'demo'
    """

    def __init__(
        self,
        name: str = "ML",
        log_dir: Optional[str] = None,
        log_filename: str = "logger",
        enable_file_logging: bool = True,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # A directory nothing will clean up while the run is going: the log file has to outlive
        # this object, because it is uploaded with the run after training.
        self._owns_log_dir = log_dir is None
        if log_dir is None:
            log_dir = os.path.join(tempfile.mkdtemp(prefix="yg_eo_soilnet_logs_"), "logs")
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.enable_file_logging = enable_file_logging
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Send the messages to the terminal, and to the log file when one is wanted."""
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.log_file: Optional[str] = None
        if self.enable_file_logging:
            self.log_file = os.path.join(self.log_dir, f"{self.log_filename}_training.log")
            os.makedirs(self.log_dir, exist_ok=True)

        formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        if self.log_file is not None:
            file_handler: logging.FileHandler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """The configured logger, ready to use.

        Returns
        -------
        logging.Logger
        """
        return self.logger
