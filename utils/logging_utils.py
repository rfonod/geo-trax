# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Logging setup with colored terminal output and a custom NOTICE level."""

import logging
from pathlib import Path


class bcolors:
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
            message = f"{bcolors.OKCYAN}{message}{bcolors.ENDC}"
        elif record.levelno == logging.WARNING:
            message = f"{bcolors.WARNING}{message}{bcolors.ENDC}"
        elif record.levelno == logging.ERROR:
            message = f"{bcolors.FAIL}{message}{bcolors.ENDC}"
        elif record.levelno == logging.CRITICAL:
            message = f"{bcolors.FAIL}{bcolors.BOLD}{message}{bcolors.ENDC}"
        return message


class FileFormatter(logging.Formatter):
    """Custom formatter for log output to file."""
    def format(self, record):
        message = super().format(record)
        for color in vars(bcolors).values():
            if isinstance(color, str):
                message = message.replace(color, '')
        return message


def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LEVEL):
        self._log(NOTICE_LEVEL, message, args, **kwargs)
logging.Logger.notice = notice


def setup_logger(name: str, verbose: bool = False, filename: str = '', dry_run: bool = False, log_dir: str = 'logs') -> logging.Logger:
    """Set up a logger with a given name, verbosity level, and optional log file name."""
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
        log_filepath = Path(log_dir) / (filename or f"{name.split('.')[0]}.log")
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_filepath}")

    logger._original_formatters = {h: h.formatter for h in logger.handlers}

    return logger
