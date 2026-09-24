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
logging.addLevelName(NOTICE_LEVEL, "NOTICE")


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
    """Return the platform-native directory for geo-trax log files."""
    if WINDOWS:  # %LOCALAPPDATA%\geo-trax\Logs
        base = Path(os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local'))
        return base / 'geo-trax' / 'Logs'
    if MACOS:  # ~/Library/Logs/geo-trax
        return Path.home() / 'Library' / 'Logs' / 'geo-trax'
    # Linux and other Unix (XDG base directory spec): ~/.local/state/geo-trax/logs
    base = Path(os.environ.get('XDG_STATE_HOME') or (Path.home() / '.local' / 'state'))
    return base / 'geo-trax' / 'logs'


def setup_logger(name: str, verbose: bool = False, log_path: str | Path | None = None, dry_run: bool = False) -> logging.Logger:
    """Set up a logger with a given name, verbosity level, and optional log path.

    ``log_path`` may be a directory (an auto-named ``<stage>_<timestamp>_<pid>.log`` file is
    created inside it) or a full file path (used verbatim). A path that does not exist yet counts
    as a directory unless it carries a suffix, so '--log-path ./logs' creates the directory the
    user asked for; testing ``is_dir()`` alone took that down the file branch and left a regular
    file named 'logs' that every later run appended to, since the test then kept returning False.
    Give a full file path a suffix to pin it. When omitted, logs go to a
    platform-specific directory (see default_log_dir). The timestamp keeps the auto-derived name
    from ever landing on a stale file left by an earlier, unrelated run (a plain PID would
    eventually be reused, e.g. after a reboot, and FileHandler's append mode would silently merge
    into it); the PID breaks the rare tie of two processes started in the same second. Pass a full
    file path explicitly to opt out of this and pin a stable, reusable name.
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
            treat_as_dir = log_path.is_dir() or (not log_path.exists() and not log_path.suffix)
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
