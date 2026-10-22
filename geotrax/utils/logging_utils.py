# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Logging setup with colored terminal output and a custom NOTICE level."""

import logging
import os
from datetime import datetime
from pathlib import Path

from geotrax.utils.constants import MACOS, WINDOWS
from geotrax.utils.version_check import check_for_updates_once


class BColors:
    """Color palette for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


NOTICE_LEVEL = 25
"""Custom level between INFO and WARNING, shown on the console without `--verbose`; logged with `logger.notice(...)`."""
logging.addLevelName(NOTICE_LEVEL, "NOTICE")

LOG_FILE_SUFFIXES = ('.log', '.txt')


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    def format(self, record):
        message = super().format(record)
        if record.levelno == NOTICE_LEVEL:
            message = f"{BColors.OKCYAN}{message}{BColors.ENDC}"
        elif record.levelno == logging.WARNING:
            message = f"{BColors.WARNING}{message}{BColors.ENDC}"
        elif record.levelno == logging.ERROR:
            message = f"{BColors.FAIL}{message}{BColors.ENDC}"
        elif record.levelno == logging.CRITICAL:
            message = f"{BColors.FAIL}{BColors.BOLD}{message}{BColors.ENDC}"
        return message


class FileFormatter(logging.Formatter):
    """Custom formatter for log output to file."""
    def format(self, record):
        message = super().format(record)
        for color in vars(BColors).values():
            if isinstance(color, str):
                message = message.replace(color, '')
        return message


def notice(self, message, *args, **kwargs):
    """Log a message at the custom NOTICE level (between INFO and WARNING)."""
    if self.isEnabledFor(NOTICE_LEVEL):
        self._log(NOTICE_LEVEL, message, args, **kwargs)
logging.Logger.notice = notice


def default_log_dir() -> Path:
    """Return the platform-native directory for geo-trax log files.

    Returns:
        `~/Library/Logs/geo-trax` on macOS, `%LOCALAPPDATA%\\geo-trax\\Logs` on Windows, and
            `$XDG_STATE_HOME/geo-trax/logs` (default `~/.local/state/geo-trax/logs`) elsewhere.
    """
    if WINDOWS:  # %LOCALAPPDATA%\geo-trax\Logs
        base = Path(os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local'))
        return base / 'geo-trax' / 'Logs'
    if MACOS:  # ~/Library/Logs/geo-trax
        return Path.home() / 'Library' / 'Logs' / 'geo-trax'
    # Linux and other Unix (XDG base directory spec): ~/.local/state/geo-trax/logs
    base = Path(os.environ.get('XDG_STATE_HOME') or (Path.home() / '.local' / 'state'))
    return base / 'geo-trax' / 'logs'


def setup_logger(name: str, verbose: bool = False, log_path: str | Path | None = None, dry_run: bool = False) -> logging.Logger:
    """Create the logger that every geo-trax command uses: colored console output plus a log file.

    The console shows NOTICE and above (INFO and above when `verbose`); the file always records INFO
    and above. The first call also starts a non-blocking, cached check for a newer geo-trax on PyPI
    (disable it with `GEOTRAX_DISABLE_UPDATE_CHECK=1`). Each call adds handlers, so create a
    logger once per name.

    Args:
        name: Logger name; its last dotted component also names the auto-named log file.
        verbose: Show INFO messages on the console too.
        log_path: Where to write the log. A directory, existing or not, gets an auto-named
            `<name>_<timestamp>_<pid>.log` file, whose timestamp and PID keep it from ever appending
            to an unrelated earlier run. A full file path is used verbatim; a path that does not
            exist yet counts as a file only if it ends in `.log` or `.txt`, so a dotted directory
            name such as `logs.v2` still works. `None` uses
            [`default_log_dir`][geotrax.utils.logging_utils.default_log_dir].
        dry_run: Log to the console only, without a file.

    Returns:
        The configured logger, which also has a `notice()` method for the
            [`NOTICE_LEVEL`][geotrax.utils.logging_utils.NOTICE_LEVEL].

    Examples:
        ```python
        logger = setup_logger("my_analysis", verbose=True, dry_run=True)
        logger.notice("Shown without --verbose.")
        ```
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_format = '%(asctime)s - %(levelname)s - %(name)s:%(module)s:%(funcName)s - %(message)s'
    colored_formatter = ColoredFormatter(log_format)
    file_formatter = FileFormatter(log_format)

    console_level = NOTICE_LEVEL if not verbose else logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    if not dry_run:
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        stage_filename = f"{name.split('.')[-1]}_{run_id}.log"
        if log_path is None:
            log_filepath = default_log_dir() / stage_filename
        else:
            log_path = Path(log_path)
            treat_as_dir = log_path.is_dir() or (
                not log_path.exists() and log_path.suffix.lower() not in LOG_FILE_SUFFIXES
            )
            log_filepath = log_path / stage_filename if treat_as_dir else log_path
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        print(f"Saving logs to: {log_filepath}")  # console-only notice; not written to the log file itself

    logger._original_formatters = {h: h.formatter for h in logger.handlers}  # used by suppress/restore_logging_format in check_dataset.py

    # Non-blocking, cached for 24 h, silent when offline; GEOTRAX_DISABLE_UPDATE_CHECK=1 opts out.
    # Placed last so the notice reaches both the console and (when enabled) the file handler.
    # Belt-and-braces guard: a convenience notice must never prevent a run from getting a logger
    # (starting the daemon thread can fail under resource pressure).
    try:
        check_for_updates_once(logger)
    except Exception:  # noqa: BLE001 - best-effort by design
        pass

    return logger
