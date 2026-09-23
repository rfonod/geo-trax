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
    apply_cli_overrides,
    backfill_args_from_config,
    merge_bundled_defaults,
    load_class_names_from_model,
    load_config,
    load_config_all,
    resolve_asset_path,
    resolve_class_names,
    resolve_config_path,
    resolve_model_path,
    sync_args_with_config,
)
from geotrax.utils.cli_utils import CfgArg

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
    assert config['main']['visualization']['viz_mode'] in (0, 1, 2, 3, 4)
    assert isinstance(config['main']['class_names'], dict)
    # The active tracker block is materialized to a temp YAML file for Ultralytics.
    assert Path(config['ultralytics']['tracker']).is_file()


@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_presets_expose_gpu_keys(preset):
    """Every preset carries the CUDA toggles for both the stabilization and georef paths."""
    full = load_config(preset, logger)
    assert full['stabilo']['gpu'] is False
    assert full['stabilo']['gpu_device_id'] == 0
    assert full['georef']['matching']['gpu'] is False
    assert full['georef']['matching']['gpu_device_id'] == 0


@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_presets_expose_dl_detector_keys(preset):
    """Every preset carries the stabilo 1.4.0 learning-based detector settings on both paths."""
    full = load_config(preset, logger)
    for block in (full['stabilo'], full['georef']['matching']):
        assert block['device'] == 'auto'
        assert block['loftr_weights'] == 'outdoor'
        assert block['loftr_confidence'] == 0.0
        assert block['disk_weights'] == 'depth'
        assert block['dedode_detector_weights'] == 'L-C4-v2'
        assert block['dedode_descriptor_weights'] == 'B-upright'
    # Previously hardcoded in registration.py; now tunable so learned detectors can shrink the ortho.
    assert full['georef']['matching']['downsample_ratio'] == 1.0


def _config_key_paths(node, prefix=''):
    """Recursively collect dotted key paths from a nested config mapping."""
    paths = set()
    for key, value in (node or {}).items():
        path = f'{prefix}{key}'
        paths.add(path)
        if isinstance(value, dict):
            paths |= _config_key_paths(value, f'{path}.')
    return paths


@pytest.mark.parametrize('preset', ['confident', 'lenient', 'stable'])
def test_presets_share_default_key_structure(preset):
    """
    Presets differ from default.yaml only in tuned *values*, never in which keys exist.

    Mechanically enforces the preset-mirroring rule: adding a key to default.yaml without
    mirroring it into all three presets fails here.
    """
    expected = _config_key_paths(load_config('default', logger))
    actual = _config_key_paths(load_config(preset, logger))
    assert actual == expected, f'missing={sorted(expected - actual)} extra={sorted(actual - expected)}'


@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_presets_expose_sahi_block(preset):
    """Every preset carries the full SAHI sub-block in extraction, disabled by default."""
    sahi = load_config(preset, logger)['extraction']['sahi']
    assert sahi['enable'] is False
    assert sahi['slice_height'] == 1080
    assert sahi['slice_width'] == 1920
    assert set(sahi) == {
        'enable', 'slice_height', 'slice_width', 'overlap_height_ratio', 'overlap_width_ratio',
        'perform_standard_pred', 'postprocess_type', 'postprocess_match_metric',
        'postprocess_match_threshold', 'class_agnostic',
    }


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
    mock_dl.assert_called_once_with(repo_id='rfonod/geo-trax', filename='geotrax_hbb_yolov8s_1920_v1.pt', revision=None)
    assert result == cached


def test_resolve_model_path_hf_pins_a_revision(tmp_path):
    cached = tmp_path / 'cached.pt'
    cached.touch()
    with patch('geotrax.utils.config_utils.hf_hub_download', return_value=str(cached)) as mock_dl:
        resolve_model_path('hf://rfonod/geo-trax@0123abc/sub/model.pt', logger)
    mock_dl.assert_called_once_with(repo_id='rfonod/geo-trax', filename='sub/model.pt', revision='0123abc')


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


# --- integer-keyed config blocks ---------------------------------------------

def test_set_override_reaches_an_integer_keyed_leaf():
    """
    tau_c is keyed by integer class id; dotted paths are strings.

    The path helpers must agree on the key type, or --set writes a string-keyed duplicate that
    nothing reads while reporting the override as applied.
    """
    cfg = load_config('default', logger)
    apply_cli_overrides(cfg, ['tau_c.0=2.5'], None, logger)
    tau_c = cfg['extraction']['dimension_estimation']['tau_c']
    assert tau_c[0] == 2.5
    assert '0' not in tau_c


def test_set_override_type_checks_an_integer_keyed_leaf():
    cfg = load_config('default', logger)
    with pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['tau_c.0=not-a-number'], None, logger)


# --- malformed configs fail fast ---------------------------------------------

@pytest.mark.parametrize('content', ['', 'a: [1,\nb: }{\n', '- just\n- a list\n'])
def test_load_config_unusable_file_exits(tmp_path, content):
    cfg_file = tmp_path / 'broken.yaml'
    cfg_file.write_text(content)
    with pytest.raises(SystemExit):
        load_config(cfg_file, logger)


@pytest.mark.parametrize('given', ['', '.', '..'])
def test_resolve_config_path_tolerates_an_empty_name(given):
    """A suffix-less path with no final component must not raise; load_config reports it."""
    assert resolve_config_path(given) == Path(given)


# --- bundled-default merge for config-only blocks -----------------------------

def test_merge_bundled_defaults_fills_a_missing_block():
    """A config predating extraction.sahi must still reach the SAHI stage fully populated."""
    merged = merge_bundled_defaults({'enable': True}, 'extraction.sahi')
    assert merged['enable'] is True
    assert merged['slice_height'] == 1080
    assert 'postprocess_match_threshold' in merged


def test_merge_bundled_defaults_keeps_user_values():
    merged = merge_bundled_defaults({'slice_height': 640}, 'extraction.sahi')
    assert merged['slice_height'] == 640


# --- pinned dests --------------------------------------------------------------

def _pin_args(pinned):
    args = argparse.Namespace(
        cut_frame_right=None,
        _cli_provided=set(),
        _cfg_paths={'cut_frame_right': CfgArg(path='processing.cut_frame_right', flag='--cut-frame-right')},
    )
    if pinned:
        args._cfg_pinned = {'cut_frame_right'}
    return args


def test_unpinned_dest_is_backfilled_from_the_config():
    cfg = {'processing': {'cut_frame_right': 900}}
    args = _pin_args(pinned=False)
    sync_args_with_config(args, cfg, logger)
    assert args.cut_frame_right == 900


def test_pinned_dest_survives_the_config_resync():
    """
    'batch' clears cut_frame_right in directory mode; every stage re-loads the config and
    re-syncs, which would otherwise pull the value straight back in and truncate every video.
    """
    cfg = {'processing': {'cut_frame_right': 900}}
    args = _pin_args(pinned=True)
    sync_args_with_config(args, cfg, logger)
    assert args.cut_frame_right is None
    assert cfg['processing']['cut_frame_right'] is None


# --- ultralytics key drift -----------------------------------------------------

@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_presets_carry_no_dead_half_key(preset):
    """
    ultralytics drops 'half' whenever 'quantize' is also present, so shipping both made 'half'
    an inert knob that silently never took effect. 'quantize' is the live one.
    """
    ultralytics = load_config(preset, logger)['ultralytics']
    assert 'half' not in ultralytics
    assert 'quantize' in ultralytics
