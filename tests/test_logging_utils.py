# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the logging utilities: platform log directory and colored formatter."""

import logging
import sys
from pathlib import Path

from geotrax.utils.logging_utils import ColoredFormatter, default_log_dir


# --- default_log_dir ---------------------------------------------------------

def test_default_log_dir_returns_path():
    result = default_log_dir()
    assert isinstance(result, Path)


def test_default_log_dir_contains_geo_trax():
    assert 'geo-trax' in str(default_log_dir())


def test_default_log_dir_platform_specific():
    result = default_log_dir()
    if sys.platform == 'darwin':
        assert str(result).startswith(str(Path.home() / 'Library' / 'Logs'))
    elif sys.platform.startswith('linux'):
        assert 'state' in str(result) or '.local' in str(result)
    elif sys.platform == 'win32':
        assert 'Logs' in str(result)


# --- ColoredFormatter --------------------------------------------------------

def _make_record(level, msg='test message'):
    return logging.LogRecord(
        name='test', level=level, pathname='', lineno=0, msg=msg, args=(), exc_info=None
    )


def test_colored_formatter_warning_includes_ansi():
    formatter = ColoredFormatter('%(message)s')
    output = formatter.format(_make_record(logging.WARNING))
    assert 'test message' in output
    assert '\033[' in output


def test_colored_formatter_error_includes_ansi():
    formatter = ColoredFormatter('%(message)s')
    output = formatter.format(_make_record(logging.ERROR))
    assert 'test message' in output
    assert '\033[' in output


def test_colored_formatter_info_no_ansi():
    formatter = ColoredFormatter('%(message)s')
    output = formatter.format(_make_record(logging.INFO))
    assert 'test message' in output
    assert '\033[' not in output


def test_colored_formatter_critical_includes_bold():
    formatter = ColoredFormatter('%(message)s')
    output = formatter.format(_make_record(logging.CRITICAL))
    assert '\033[1m' in output  # BColors.BOLD
