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
    load_class_names_from_model,
    load_config,
    load_config_all,
    resolve_asset_path,
    resolve_config_path,
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
def test_unified_configs_load(preset):
    mock_model = MagicMock()
    mock_model.names = {0: 'Car', 1: 'Bus', 2: 'Truck', 3: 'Motorcycle'}
    with patch('geotrax.utils.config_utils.YOLO', return_value=mock_model):
        args = argparse.Namespace(cfg=preset, classes=None, conf=None, show=None)
        config = load_config_all(args, logger)
    assert set(config) == {'main', 'stabilo', 'ultralytics', 'georef'}
    assert config['main']['visualization']['viz_mode'] in (0, 1, 2)
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


def test_load_class_names_from_model_missing_returns_defaults():
    with patch('geotrax.utils.config_utils.YOLO', side_effect=FileNotFoundError('not found')):
        result = load_class_names_from_model(Path('no/such/model.pt'), logger)
    assert result[0] == 'class_0'
    assert len(result) == 100
