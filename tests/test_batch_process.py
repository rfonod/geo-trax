# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure filtering and decision helpers in batch_process.py."""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from geotrax.batch_process import filter_files_to_process, handle_existing_results, process_file

logger = logging.getLogger(__name__)


def _args(overwrite=False, yes=False, folders_exclude=None, exclude_patterns=None):
    return argparse.Namespace(
        overwrite=overwrite,
        yes=yes,
        folders_exclude=folders_exclude or [],
        exclude_patterns=exclude_patterns,
    )


# --- filter_files_to_process -------------------------------------------------

def test_filter_files_excludes_by_folder_name():
    files = [Path('PROCESSED/v.mp4'), Path('VIDEOS/v.mp4')]
    result = filter_files_to_process(files, _args(folders_exclude=['PROCESSED']), logger)
    assert result == [Path('VIDEOS/v.mp4')]


def test_filter_files_excludes_by_pattern():
    files = [Path('VIDEOS/drone_test.mp4'), Path('VIDEOS/v.mp4')]
    result = filter_files_to_process(files, _args(exclude_patterns=['drone']), logger)
    assert result == [Path('VIDEOS/v.mp4')]


def test_filter_files_passes_all_when_no_exclusions():
    files = [Path('VIDEOS/v1.mp4'), Path('VIDEOS/v2.mp4')]
    result = filter_files_to_process(files, _args(), logger)
    assert result == files


def test_filter_files_both_criteria_applied():
    files = [
        Path('PROCESSED/v1.mp4'),   # excluded folder
        Path('VIDEOS/drone.mp4'),   # excluded pattern
        Path('VIDEOS/v2.mp4'),      # passes both
    ]
    result = filter_files_to_process(
        files, _args(folders_exclude=['PROCESSED'], exclude_patterns=['drone']), logger
    )
    assert result == [Path('VIDEOS/v2.mp4')]


# --- handle_existing_results -------------------------------------------------

def test_handle_existing_results_not_exists_returns_true():
    assert handle_existing_results(Path('v.mp4'), _args(), logger, exists=False, action='extract') is True


def test_handle_existing_results_exists_no_overwrite_returns_false():
    assert handle_existing_results(Path('v.mp4'), _args(overwrite=False), logger, exists=True, action='extract') is False


def test_handle_existing_results_exists_overwrite_yes_returns_true():
    assert handle_existing_results(Path('v.mp4'), _args(overwrite=True, yes=True), logger, exists=True, action='extract') is True


@pytest.mark.parametrize('user_input, expected', [('y', True), ('n', False)])
def test_handle_existing_results_exists_overwrite_prompts(user_input, expected):
    with patch('builtins.input', return_value=user_input):
        result = handle_existing_results(
            Path('v.mp4'), _args(overwrite=True, yes=False), logger, exists=True, action='extract'
        )
    assert result is expected


def test_filter_files_excludes_nested_folders():
    files = [Path('archive/2024/v1.mp4'), Path('archive/v2.mp4'), Path('keep/v3.mp4')]
    result = filter_files_to_process(files, _args(folders_exclude=['archive']), logger)
    assert result == [Path('keep/v3.mp4')]


def test_filter_files_ignores_folder_names_above_the_scan_root():
    files = [Path('/data/P/keep/v.mp4')]
    result = filter_files_to_process(files, _args(folders_exclude=['data']), logger, Path('/data/P'))
    assert result == files


def test_filter_files_tolerates_null_folders_exclude():
    files = [Path('VIDEOS/v.mp4')]
    args = _args()
    args.folders_exclude = None
    assert filter_files_to_process(files, args, logger) == files


# --- process_file isolation --------------------------------------------------

def _process_file_args(**overrides):
    args = argparse.Namespace(
        viz_only=False, geo_only=False, plot_only=False, no_geo=True,
        save=False, show=False, plot_save=False, plot_show=False,
        input=Path('/videos'), dry_run=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_process_file_contains_a_stage_sys_exit():
    """A stage that calls sys.exit must not terminate a directory run (SystemExit is a BaseException)."""
    with patch('geotrax.batch_process.process_step', side_effect=SystemExit(1)):
        assert process_file(Path('bad.mp4'), _process_file_args(), logger) is False


def test_process_file_contains_a_stage_exception():
    with patch('geotrax.batch_process.process_step', side_effect=RuntimeError('boom')):
        assert process_file(Path('bad.mp4'), _process_file_args(), logger) is False


def test_process_file_reports_success():
    with patch('geotrax.batch_process.process_step'):
        assert process_file(Path('good.mp4'), _process_file_args(), logger) is True


def test_geo_only_skips_visualization():
    """--geo-only must not reach the visualization stage, which cfg -> visualization -> save enables."""
    with patch('geotrax.batch_process.process_step') as step:
        process_file(Path('v.mp4'), _process_file_args(geo_only=True, no_geo=False, save=True), logger)
    actions = [call.args[3] for call in step.call_args_list]
    assert 'Visualizing' not in actions and 'Georeferencing' in actions
