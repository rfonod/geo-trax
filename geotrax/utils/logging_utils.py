# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Logging setup with colored terminal output and a custom NOTICE level."""

import logging
import os
from pathlib import Path
from typing import Union

from geotrax.utils.constants import MACOS, WINDOWS


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


def setup_logger(name: str, verbose: bool = False, log_path: Union[str, Path, None] = None, dry_run: bool = False) -> logging.Logger:
    """Set up a logger with a given name, verbosity level, and optional log path.

    ``log_path`` may be a directory (the default ``<stage>.log`` name is used inside it) or a
    full file path. When omitted, logs go to a platform-specific directory (see default_log_dir).
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
        stage_filename = f"{name.split('.')[-1]}.log"
        if log_path is None:
            log_filepath = default_log_dir() / stage_filename
        else:
            log_path = Path(log_path)
            log_filepath = log_path / stage_filename if log_path.is_dir() else log_path
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        print(f"Saving logs to: {log_filepath}")  # console-only notice; not written to the log file itself

    logger._original_formatters = {h: h.formatter for h in logger.handlers}  # used by suppress/restore_logging_format in check_dataset.py

    return logger
