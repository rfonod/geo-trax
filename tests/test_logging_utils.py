# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the logging utilities: platform log directory, colored formatter, and setup_logger."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from geotrax.utils import logging_utils
from geotrax.utils.logging_utils import ColoredFormatter, default_log_dir, setup_logger

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


# --- setup_logger --------------------------------------------------------------

class _FixedDatetime(datetime):
    """Stand-in for datetime.now() so filename tests don't depend on wall-clock time."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 7, 9, 15, 55, 34)


def _cleanup(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_setup_logger_default_log_path_includes_timestamp_and_pid(tmp_path, monkeypatch):
    # No --log-path given: the auto-named file must carry a timestamp (so it never lands on a
    # stale file from an earlier run whose PID got reused) and this process's PID (so two
    # processes started in the very same second still don't collide).
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)
    monkeypatch.setattr(logging_utils, 'datetime', _FixedDatetime)
    logger = setup_logger('geotrax.test_stage_default', verbose=False, log_path=None)
    try:
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        expected = f'test_stage_default_20260709_155534_{os.getpid()}.log'
        assert Path(file_handler.baseFilename).name == expected
    finally:
        _cleanup(logger)


def test_setup_logger_directory_log_path_includes_timestamp_and_pid(tmp_path, monkeypatch):
    # --log-path pointing at an existing directory: same auto-naming applies inside it.
    monkeypatch.setattr(logging_utils, 'datetime', _FixedDatetime)
    logger = setup_logger('geotrax.test_stage_dir', verbose=False, log_path=tmp_path)
    try:
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        expected = f'test_stage_dir_20260709_155534_{os.getpid()}.log'
        assert Path(file_handler.baseFilename).name == expected
        assert Path(file_handler.baseFilename).parent == tmp_path
    finally:
        _cleanup(logger)


def test_setup_logger_explicit_file_path_used_verbatim(tmp_path):
    # --log-path pointing at a specific (non-directory) file: honored exactly, no auto-naming,
    # since this is the user's deliberate choice of a stable, shared filename.
    target = tmp_path / 'custom_name.log'
    logger = setup_logger('geotrax.test_stage_explicit', verbose=False, log_path=target)
    try:
        file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        assert Path(file_handler.baseFilename) == target
    finally:
        _cleanup(logger)


def test_setup_logger_two_processes_in_the_same_second_do_not_collide(tmp_path, monkeypatch):
    # Same timestamp (simulating two processes started in the same second) but different PIDs:
    # the PID tiebreaker must still keep the filenames distinct.
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)
    monkeypatch.setattr(logging_utils, 'datetime', _FixedDatetime)
    monkeypatch.setattr(logging_utils.os, 'getpid', lambda: 111)
    setup_logger('geotrax.test_stage_concurrent', verbose=False, log_path=None)
    monkeypatch.setattr(logging_utils.os, 'getpid', lambda: 222)
    logger = setup_logger('geotrax.test_stage_concurrent', verbose=False, log_path=None)
    try:
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        filenames = {Path(h.baseFilename).name for h in file_handlers}
        assert filenames == {
            'test_stage_concurrent_20260709_155534_111.log',
            'test_stage_concurrent_20260709_155534_222.log',
        }
    finally:
        _cleanup(logger)  # both setup_logger calls returned the same underlying Logger object (same name)


def test_setup_logger_reused_pid_on_a_later_run_does_not_reuse_the_old_file(tmp_path, monkeypatch):
    # Same PID (simulating PID reuse after a reboot) but a later timestamp: the two runs must
    # still land on different files instead of the new run silently appending into the old one.
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)
    monkeypatch.setattr(logging_utils.os, 'getpid', lambda: 111)

    class _Earlier(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 0, 0, 0)

    monkeypatch.setattr(logging_utils, 'datetime', _Earlier)
    setup_logger('geotrax.test_stage_reused_pid', verbose=False, log_path=None)
    monkeypatch.setattr(logging_utils, 'datetime', _FixedDatetime)
    logger = setup_logger('geotrax.test_stage_reused_pid', verbose=False, log_path=None)
    try:
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        filenames = {Path(h.baseFilename).name for h in file_handlers}
        assert filenames == {
            'test_stage_reused_pid_20260101_000000_111.log',
            'test_stage_reused_pid_20260709_155534_111.log',
        }
    finally:
        _cleanup(logger)


def test_setup_logger_dry_run_skips_file_handler(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)
    logger = setup_logger('geotrax.test_stage_dry_run', verbose=False, dry_run=True)
    try:
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    finally:
        _cleanup(logger)


def test_setup_logger_triggers_update_check(tmp_path, monkeypatch):
    """setup_logger is the single hook point for the PyPI update check."""
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)
    calls = []
    monkeypatch.setattr(logging_utils, 'check_for_updates_once', lambda logger=None: calls.append(logger))
    logger = setup_logger('geotrax.test_stage_update_check', verbose=False)
    try:
        assert calls == [logger]
    finally:
        _cleanup(logger)


def test_setup_logger_survives_failing_update_check(tmp_path, monkeypatch):
    """The check is best-effort: a failure in it must never break logger setup."""
    monkeypatch.setattr(logging_utils, 'default_log_dir', lambda: tmp_path)

    def _boom(logger=None):
        raise RuntimeError('network exploded')

    monkeypatch.setattr(logging_utils, 'check_for_updates_once', _boom)
    logger = setup_logger('geotrax.test_stage_update_check_raises', verbose=False)
    try:
        assert isinstance(logger, logging.Logger)
        logger.info('still usable')
    finally:
        _cleanup(logger)


def test_log_path_creates_a_missing_directory(tmp_path):
    """
    A directory that does not exist yet is still a directory.

    Testing is_dir() alone took '--log-path ./logs' down the file branch and created a regular
    file named 'logs' that every later run then appended to, since the test kept returning False.
    """
    target = tmp_path / 'runlogs'
    setup_logger('geotrax.extract', log_path=target)
    assert target.is_dir()
    assert [f.suffix for f in target.iterdir()] == ['.log']


def test_log_path_with_a_suffix_is_used_verbatim(tmp_path):
    target = tmp_path / 'pinned.log'
    setup_logger('geotrax.georeference', log_path=target)
    assert target.is_file()


@pytest.mark.parametrize('name', ['logs.v2', '2026.09'])
def test_log_path_with_a_dotted_directory_name_creates_a_directory(tmp_path, name):
    """Only a log-file suffix pins a new path as a file; a dotted directory name has a suffix too."""
    target = tmp_path / name
    logger = setup_logger(f'geotrax.dotted_{name.replace(".", "_")}', log_path=target)
    try:
        assert target.is_dir()
    finally:
        _cleanup(logger)


def test_log_path_to_an_existing_file_is_used_verbatim_whatever_its_suffix(tmp_path):
    target = tmp_path / 'run.out'
    target.touch()
    logger = setup_logger('geotrax.existing_file', log_path=target)
    try:
        assert target.is_file()
    finally:
        _cleanup(logger)
