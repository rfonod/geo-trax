# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for configuration loading and path resolution."""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geotrax import CFG_DIR
from geotrax.utils.config_utils import (
    backfill_args_from_config,
    load_class_names_from_model,
    load_config,
    load_config_all,
    resolve_asset_path,
    resolve_class_names,
    resolve_config_path,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'given',
    [
        'geotrax/cfg/default.yaml',  # documented default
        'cfg/default.yaml',          # legacy pre-package path
        'default.yaml',              # bare filename
        'default',                   # bare name without suffix
    ],
)
def test_resolve_default_config(given):
    assert resolve_config_path(given).resolve() == CFG_DIR / 'default.yaml'


def test_resolve_prefers_existing_local_file(tmp_path, monkeypatch):
    local = tmp_path / 'default.yaml'
    local.write_text('stabilize: false\n')
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path('default.yaml') == Path('default.yaml')


def test_resolve_missing_returns_input_unchanged():
    assert resolve_config_path('no/such/config.yaml') == Path('no/such/config.yaml')


def test_load_config_missing_exits():
    with pytest.raises(SystemExit):
        load_config('no/such/config.yaml', logger)


@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_unified_configs_load(preset, tmp_path):
    mock_model = MagicMock()
    mock_model.names = {0: 'Car', 1: 'Bus', 2: 'Truck', 3: 'Motorcycle'}
    # The presets reference the model via hf://; patch the downloader so no network is hit.
    fake_weights = tmp_path / 'model.pt'
    fake_weights.touch()
    with patch('geotrax.utils.config_utils.YOLO', return_value=mock_model), \
         patch('geotrax.utils.config_utils.hf_hub_download', return_value=str(fake_weights)):
        args = argparse.Namespace(cfg=preset, classes=None, conf=None, show=None, model=None, class_names=None)
        config = load_config_all(args, logger)
    assert set(config) == {'main', 'stabilo', 'ultralytics', 'georef'}
    assert config['main']['visualization']['viz_mode'] in (0, 1, 2, 3)
    assert isinstance(config['main']['class_names'], dict)
    # The active tracker block is materialized to a temp YAML file for Ultralytics.
    assert Path(config['ultralytics']['tracker']).is_file()


@pytest.mark.parametrize(
    'tracker', ['botsort', 'bytetrack', 'ocsort', 'deepocsort', 'fasttrack', 'tracktrack']
)
def test_unified_config_contains_all_tracker_blocks(tracker):
    full = load_config('default', logger)
    assert tracker in full['tracker']
    assert full['tracker'][tracker]['tracker_type'] == tracker


def test_active_tracker_is_valid():
    full = load_config('default', logger)
    active = full['tracker']['active']
    assert active in full['tracker']
    assert full['tracker'][active]['tracker_type'] == active


def test_resolve_asset_path_missing_returns_unchanged():
    assert resolve_asset_path('no/such/model.pt') == Path('no/such/model.pt')


def test_resolve_asset_path_absolute_returns_unchanged(tmp_path):
    absolute = tmp_path / 'model.pt'
    assert resolve_asset_path(absolute) == absolute


def test_load_class_names_from_model_success():
    mock_model = MagicMock()
    mock_model.names = {0: 'Car', 1: 'Bus', 2: 'Truck', 3: 'Motorcycle'}
    with patch('geotrax.utils.config_utils.YOLO', return_value=mock_model):
        result = load_class_names_from_model(Path('dummy.pt'), logger)
    assert result == {0: 'Car', 1: 'Bus', 2: 'Truck', 3: 'Motorcycle'}


def test_load_class_names_from_model_missing_returns_none():
    with patch('geotrax.utils.config_utils.YOLO', side_effect=FileNotFoundError('not found')):
        result = load_class_names_from_model(Path('no/such/model.pt'), logger)
    assert result is None


# --- resolve_model_path -------------------------------------------------------

def test_resolve_model_path_local_passes_through():
    # A non-hf:// reference keeps the historical local-path behaviour (no download).
    assert resolve_model_path('no/such/model.pt', logger) == Path('no/such/model.pt')


def test_resolve_model_path_hf_downloads_and_parses(tmp_path):
    cached = tmp_path / 'cached.pt'
    cached.touch()
    with patch('geotrax.utils.config_utils.hf_hub_download', return_value=str(cached)) as mock_dl:
        result = resolve_model_path('hf://rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt', logger)
    mock_dl.assert_called_once_with(repo_id='rfonod/geo-trax', filename='geotrax_hbb_yolov8s_1920_v1.pt')
    assert result == cached


def test_resolve_model_path_hf_malformed_exits():
    with patch('geotrax.utils.config_utils.hf_hub_download', return_value='x'):
        with pytest.raises(SystemExit):
            resolve_model_path('hf://rfonod/onlytwoparts.pt', logger)


def test_resolve_model_path_hf_missing_dependency_exits():
    with patch('geotrax.utils.config_utils.hf_hub_download', None):
        with pytest.raises(SystemExit):
            resolve_model_path('hf://rfonod/geo-trax/model.pt', logger)


# --- resolve_class_names ------------------------------------------------------

def test_resolve_class_names_cli_inline_pairs_win():
    mapping, source = resolve_class_names(Path('m.pt'), ['0=auto', '1=van'], {0: 'car'}, [0, 1], logger)
    assert mapping == {0: 'auto', 1: 'van'}
    assert source == 'cli'


def test_resolve_class_names_config_when_no_cli():
    mapping, source = resolve_class_names(Path('m.pt'), None, {0: 'car', 1: 'bus'}, [0, 1], logger)
    assert mapping == {0: 'car', 1: 'bus'}
    assert source == 'config'


def test_resolve_class_names_from_file(tmp_path):
    f = tmp_path / 'names.yaml'
    f.write_text('0: car\n1: bus\n')
    mapping, source = resolve_class_names(Path('m.pt'), [str(f)], None, [0, 1], logger)
    assert mapping == {0: 'car', 1: 'bus'}
    assert source == 'cli'


def test_resolve_class_names_falls_back_to_model():
    mock_model = MagicMock()
    mock_model.names = {0: 'Car', 1: 'Bus'}
    with patch('geotrax.utils.config_utils.YOLO', return_value=mock_model):
        mapping, source = resolve_class_names(Path('m.pt'), None, None, [0, 1], logger)
    assert mapping == {0: 'Car', 1: 'Bus'}
    assert source == 'model'


def test_resolve_class_names_integer_fallback_warns(caplog):
    with patch('geotrax.utils.config_utils.YOLO', side_effect=RuntimeError('boom')):
        with caplog.at_level('WARNING'):
            mapping, source = resolve_class_names(Path('m.pt'), None, None, [0, 1, 2, 3], logger)
    assert mapping == {0: '0', 1: '1', 2: '2', 3: '3'}
    assert source == 'fallback'
    assert any('integer class IDs' in r.message for r in caplog.records)


# --- backfill_args_from_config -----------------------------------------------

def test_backfill_args_fills_none_values():
    args = argparse.Namespace(conf=None, classes=[0, 1])
    backfill_args_from_config(args, {'conf': 0.25, 'classes': [0, 1, 2]})
    assert args.conf == 0.25           # was None → filled
    assert args.classes == [0, 1]      # already set → unchanged


def test_backfill_args_noop_when_already_set():
    args = argparse.Namespace(verbose=True)
    backfill_args_from_config(args, {'verbose': False})
    assert args.verbose is True        # pre-existing value must not be overwritten


def test_backfill_args_missing_key_raises():
    # The function only operates on existing Namespace attributes; a missing key raises.
    args = argparse.Namespace()
    with pytest.raises(AttributeError):
        backfill_args_from_config(args, {'no_such_attr': 42})
