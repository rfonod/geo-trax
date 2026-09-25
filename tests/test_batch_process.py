# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure filtering and decision helpers in batch_process.py."""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from geotrax.batch_process import filter_files_to_process, handle_existing_results, process_file, process_input

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


def test_filter_files_excludes_videos_directly_inside_an_excluded_scan_root():
    """Pointing batch straight at an output folder must not ingest the annotated videos in it."""
    files = [Path('/p/D1/results/v_mode_0.mp4'), Path('/p/D1/results/sub/w.mp4')]
    result = filter_files_to_process(files, _args(folders_exclude=['results']), logger, Path('/p/D1/results'))
    assert result == []


def test_filter_files_resolves_a_dot_scan_root(tmp_path, monkeypatch):
    root = tmp_path / 'results'
    root.mkdir()
    monkeypatch.chdir(root)
    files = [Path('v_mode_0.mp4')]
    assert filter_files_to_process(files, _args(folders_exclude=['results']), logger, Path('.')) == []


# --- process_input interruption ----------------------------------------------

def _run_process_input(tmp_path, process_file_effect):
    (tmp_path / 'a.mp4').touch()
    (tmp_path / 'b.mp4').touch()
    args = argparse.Namespace(
        input=tmp_path, cfg=None, cut_frame_right=None, folders_exclude=[], exclude_patterns=None,
        plot_save=True, plot_show=False, viz_only=False, geo_only=False,
    )
    with patch('geotrax.batch_process.load_config', return_value={}), \
            patch('geotrax.batch_process.sync_args_with_config'), \
            patch('geotrax.batch_process.process_file', side_effect=process_file_effect), \
            patch('geotrax.batch_process.run_plotting') as plotting, \
            patch.object(logger, 'error') as error:
        with pytest.raises(SystemExit) as exc:
            process_input(args, logger)
    return exc.value.code, plotting, error


def test_interrupted_run_reports_failures_and_exits_non_zero(tmp_path):
    code, plotting, error = _run_process_input(tmp_path, [False, KeyboardInterrupt])
    assert code == 130
    plotting.assert_not_called()
    assert any('a.mp4' in call.args[0] for call in error.call_args_list)


def test_interrupted_run_without_failures_still_exits_non_zero(tmp_path):
    code, _, _ = _run_process_input(tmp_path, [True, KeyboardInterrupt])
    assert code == 130


def test_completed_run_with_failures_exits_one(tmp_path):
    code, plotting, _ = _run_process_input(tmp_path, [False, True])
    assert code == 1
    plotting.assert_called_once()



def test_filter_files_excludes_an_absolute_output_folder_by_path_only(tmp_path):
    """An absolute -of /x/D1 must skip that folder, not every input folder named 'D1'."""
    out = tmp_path / 'out' / 'D1'
    files = [tmp_path / 'data' / 'D1' / 'v.mp4', out / 'v_mode_0.mp4']
    result = filter_files_to_process(files, _args(), logger, tmp_path, out.resolve())
    assert result == [tmp_path / 'data' / 'D1' / 'v.mp4']


def test_absolute_output_folder_name_is_not_added_to_folders_exclude(tmp_path):
    (tmp_path / 'data' / 'D1').mkdir(parents=True)
    (tmp_path / 'data' / 'D1' / 'v.mp4').touch()
    args = argparse.Namespace(
        input=tmp_path / 'data' / 'D1', cfg=None, cut_frame_right=None, folders_exclude=['results'],
        exclude_patterns=None, plot_save=False, plot_show=False, viz_only=False, geo_only=False,
    )
    config = {'output': {'folder': str(tmp_path / 'out' / 'D1')}}
    with patch('geotrax.batch_process.load_config', return_value=config), \
            patch('geotrax.batch_process.sync_args_with_config'), \
            patch('geotrax.batch_process.process_file', return_value=True) as processed:
        process_input(args, logger)
    assert args.folders_exclude == ['results']
    assert [call.args[0].name for call in processed.call_args_list] == ['v.mp4']
