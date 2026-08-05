# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the generic --set config overrides and the CLI <-> config reconciliation."""

import argparse
import logging
from pathlib import Path

import pytest

from geotrax.utils.cli_utils import CfgArg, add_cfg_arg, add_common_args, finalize_cli_args
from geotrax.utils.config_utils import (
    _MISSING,
    _get_by_path,
    _iter_leaf_paths,
    _set_by_path,
    apply_cli_overrides,
    load_config,
    sync_args_with_config,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def cfg():
    """A miniature config with the shapes that matter: nested sections, twins, and a null."""
    return {
        'output': {'folder': 'results', 'tracks_postfix': ''},
        'processing': {'cut_frame_left': 0, 'cut_frame_right': None},
        'extraction': {'interpolate': False, 'sahi': {'enable': False, 'slice_height': 1080}},
        'stabilo': {'gpu': False, 'detector_name': 'orb', 'max_features': 2000},
        'georef': {'matching': {'gpu': False, 'detector_name': 'rsift', 'downsample_ratio': 1.0}},
        'ultralytics': {'conf': 0.25, 'classes': [0, 1, 2, 3]},
        'tracker': {'active': 'botsort', 'botsort': {'track_buffer': 30}, 'bytetrack': {'track_buffer': 30}},
    }


# --- path helpers ------------------------------------------------------------------------------

def test_leaf_paths_treats_null_as_a_leaf_and_sections_as_branches(cfg):
    paths = set(_iter_leaf_paths(cfg))
    assert 'processing.cut_frame_right' in paths  # a null value is still a settable leaf
    assert 'extraction.sahi.enable' in paths      # nesting is followed all the way down
    assert 'extraction.sahi' not in paths         # a section is not itself settable
    assert 'stabilo' not in paths


def test_set_by_path_creates_missing_intermediate_sections():
    target = {}
    _set_by_path(target, 'extraction.sahi.enable', True)
    assert target == {'extraction': {'sahi': {'enable': True}}}


def test_get_by_path_missing_key_is_distinguishable_from_null(cfg):
    assert _get_by_path(cfg, 'processing.cut_frame_right') is None
    assert _get_by_path(cfg, 'processing.nope', 'sentinel') == 'sentinel'


# --- key resolution ----------------------------------------------------------------------------

def test_exact_dotted_path(cfg):
    apply_cli_overrides(cfg, ['ultralytics.conf=0.4'], None, logger)
    assert cfg['ultralytics']['conf'] == 0.4


def test_unique_leaf_shorthand(cfg):
    apply_cli_overrides(cfg, ['conf=0.4'], None, logger)
    assert cfg['ultralytics']['conf'] == 0.4


def test_unique_multi_segment_suffix_disambiguates_the_twins(cfg):
    """'gpu' alone is ambiguous, but 'matching.gpu' names exactly one of the two."""
    apply_cli_overrides(cfg, ['matching.gpu=true'], None, logger)
    assert cfg['georef']['matching']['gpu'] is True
    assert cfg['stabilo']['gpu'] is False


def test_ambiguous_key_aborts_and_lists_every_candidate(cfg, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['gpu=true'], None, logger)
    assert 'stabilo.gpu' in caplog.text and 'georef.matching.gpu' in caplog.text


def test_ambiguous_tracker_key_points_at_the_active_block(cfg, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['track_buffer=45'], None, logger)
    assert 'tracker.botsort.track_buffer' in caplog.text
    assert 'the active tracker' in caplog.text


def test_unknown_key_aborts_with_a_suggestion(cfg, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['cnf=0.3'], None, logger)
    assert 'ultralytics.conf' in caplog.text


def test_section_key_is_rejected(cfg, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['stabilo=1'], None, logger)
    assert 'not a single value' in caplog.text


def test_missing_equals_sign_is_rejected(cfg, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['conf'], None, logger)
    assert 'KEY=VALUE' in caplog.text


# --- value parsing -----------------------------------------------------------------------------

@pytest.mark.parametrize('token, path, expected', [
    ('interpolate=true', 'extraction.interpolate', True),
    ('interpolate=false', 'extraction.interpolate', False),
    ('interpolate=1', 'extraction.interpolate', True),           # 0/1 is a common spelling of a flag
    ('max_features=4000', 'stabilo.max_features', 4000),
    ('conf=0.4', 'ultralytics.conf', 0.4),
    ('conf=1', 'ultralytics.conf', 1.0),                         # int widens into a float slot
    ('botsort.track_buffer=45.0', 'tracker.botsort.track_buffer', 45),  # whole float into an int slot
    ('cut_frame_right=null', 'processing.cut_frame_right', None),
    ('cut_frame_right=120', 'processing.cut_frame_right', 120),  # a null slot accepts any type
    ('classes=[0, 2]', 'ultralytics.classes', [0, 2]),
    ('stabilo.detector_name=sift', 'stabilo.detector_name', 'sift'),
    ('tracks_postfix=_raw', 'output.tracks_postfix', '_raw'),
    ('tracks_postfix=', 'output.tracks_postfix', ''),            # cleared, not set to null
    ('folder=/abs/path=weird', 'output.folder', '/abs/path=weird'),  # split on the first '=' only
])
def test_value_parsing(cfg, token, path, expected):
    apply_cli_overrides(cfg, [token], None, logger)
    assert _get_by_path(cfg, path) == expected


def test_bool_slot_keeps_its_type(cfg):
    apply_cli_overrides(cfg, ['interpolate=1'], None, logger)
    assert cfg['extraction']['interpolate'] is True


def test_int_slot_accepts_a_fractional_value(cfg):
    """
    Whether a key reads as int or float in the config is often incidental: plotting's
    stationary_speed_cutoff ships as '1' but 1.5 is a valid cutoff. Rejecting it would block
    real input, and editing the config file by hand has no such guard either.
    """
    cfg['plotting'] = {'stationary_speed_cutoff': 1}
    apply_cli_overrides(cfg, ['stationary_speed_cutoff=1.5'], None, logger)
    assert cfg['plotting']['stationary_speed_cutoff'] == 1.5


@pytest.mark.parametrize('token', [
    'conf=notanumber',                 # str into a float
    'max_features=true',               # bool into an int
    'stabilo.detector_name=3',         # int into a str
    'interpolate=maybe',               # str into a bool
    'classes=0',                       # scalar into a list
])
def test_type_mismatch_is_rejected(cfg, token, caplog):
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, [token], None, logger)
    assert 'wrong type' in caplog.text


def test_null_is_rejected_for_a_non_nullable_slot(cfg, caplog):
    """output.folder is a str in cfg/default.yaml, never null, so an explicit null must not land."""
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, ['folder=null'], None, logger)
    assert 'wrong type' in caplog.text
    assert cfg['output']['folder'] == 'results'


def test_null_is_accepted_for_a_documented_nullable_slot(cfg):
    """georef.processing.geo_source ships as null in cfg/default.yaml, so null is a valid value."""
    cfg['georef'] = {'processing': {'geo_source': 'metadata-tif'}}
    apply_cli_overrides(cfg, ['geo_source=null'], None, logger)
    assert cfg['georef']['processing']['geo_source'] is None


# --- precedence and conflicts ------------------------------------------------------------------

def _extract_args(argv):
    """Parse *argv* with the real extract flags, so the tests exercise the shipped registry."""
    from geotrax.extract import add_processing_args
    parser = argparse.ArgumentParser()
    paths = add_common_args(parser)
    paths |= add_processing_args(parser)
    return finalize_cli_args(parser, paths, argv)


def test_set_beats_the_config_file(cfg):
    args = _extract_args(['--set', 'conf=0.4'])
    apply_cli_overrides(cfg, args.set, args, logger)
    sync_args_with_config(args, cfg, logger)
    assert cfg['ultralytics']['conf'] == 0.4
    assert args.conf == 0.4  # and the flag is backfilled from the overridden value


def test_dedicated_flag_beats_the_config_file(cfg):
    args = _extract_args(['--conf', '0.5'])
    sync_args_with_config(args, cfg, logger)
    assert cfg['ultralytics']['conf'] == 0.5


def test_set_and_its_dedicated_flag_together_abort(cfg, caplog):
    args = _extract_args(['--set', 'conf=0.1', '--conf', '0.5'])
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, args.set, args, logger)
    assert '--set conf=0.1' in caplog.text and '--conf 0.5' in caplog.text


def test_conflict_message_names_the_real_flag_when_dest_and_flag_differ(cfg, caplog):
    """
    geotrax plot registers dest_prefix='' (unlike batch's 'plot_'), so its dest 'points' does not
    spell the real flag '--plot-points'; the reported flag must come from the registry, not dest.
    """
    from geotrax.plot import add_plotting_args

    parser = argparse.ArgumentParser()
    paths = add_common_args(parser)
    paths |= add_plotting_args(parser)
    args = finalize_cli_args(parser, paths, ['--set', 'plot_points=true', '--plot-points'])

    cfg['plotting'] = {'plot_points': False}
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        apply_cli_overrides(cfg, args.set, args, logger)
    assert '--plot-points' in caplog.text
    assert '--points' not in caplog.text


def test_conflict_leaves_the_config_untouched(cfg):
    """Validation completes before anything is written, so a rejected command line is inert."""
    args = _extract_args(['--set', 'iou=0.9', '--set', 'conf=0.1', '--conf', '0.5'])
    cfg['ultralytics']['iou'] = 0.7
    with pytest.raises(SystemExit):
        apply_cli_overrides(cfg, args.set, args, logger)
    assert cfg['ultralytics']['iou'] == 0.7  # the valid override in the same call did not land
    assert cfg['ultralytics']['conf'] == 0.25


def test_set_and_an_unrelated_flag_coexist(cfg):
    args = _extract_args(['--set', 'conf=0.4', '--cut-frame-left', '10'])
    apply_cli_overrides(cfg, args.set, args, logger)
    sync_args_with_config(args, cfg, logger)
    assert cfg['ultralytics']['conf'] == 0.4
    assert cfg['processing']['cut_frame_left'] == 10


def test_set_survives_a_reloaded_config(cfg, caplog):
    """
    Regression: 'geotrax batch' re-loads the config once per stage per video, and every load after
    the first sees a namespace that sync_args_with_config has already filled in. Conflict detection
    keys off what was typed, not off what the namespace now holds, so the second pass must not
    mistake a pulled-back value for a dedicated flag.
    """
    args = _extract_args(['--set', 'output.folder=results_new', '--set', 'processing.cut_frame_left=50'])
    apply_cli_overrides(cfg, args.set, args, logger)
    sync_args_with_config(args, cfg, logger)
    assert args.output_folder == 'results_new' and args.cut_frame_left == 50

    reloaded = dict(cfg, output={'folder': 'results', 'tracks_postfix': ''}, processing={'cut_frame_left': 0})
    with caplog.at_level(logging.CRITICAL):
        apply_cli_overrides(reloaded, args.set, args, logger)  # must not raise SystemExit
    sync_args_with_config(args, reloaded, logger)
    assert reloaded['output']['folder'] == 'results_new'
    assert reloaded['processing']['cut_frame_left'] == 50
    assert 'Conflicting' not in caplog.text


def test_synthesized_plot_args_carry_the_registry(cfg):
    """
    'geotrax batch' builds the plot stage's namespace itself instead of parsing one. It still needs
    the dest -> CfgArg registry, or neither --output-folder nor --set reaches the plotting config
    and the stage looks for its inputs in the default folder.
    """
    from geotrax.plot import default_plot_args

    args = default_plot_args(input=Path('video.mp4'), output_folder='results_new', set=['conf=0.4'])
    assert args._cfg_paths, 'the synthesized namespace carries no config-backed flags'
    apply_cli_overrides(cfg, args.set, args, logger)  # must not read as a conflict
    sync_args_with_config(args, cfg, logger)
    assert cfg['output']['folder'] == 'results_new'
    assert cfg['ultralytics']['conf'] == 0.4


def test_repeated_set_flags_accumulate(cfg):
    args = _extract_args(['--set', 'conf=0.4', '--set', 'max_features=4000'])
    apply_cli_overrides(cfg, args.set, args, logger)
    assert cfg['ultralytics']['conf'] == 0.4
    assert cfg['stabilo']['max_features'] == 4000


def test_set_does_not_swallow_a_following_positional():
    """--set takes exactly one value per occurrence, so it cannot eat the positional source arg."""
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    paths = add_common_args(parser)

    args = finalize_cli_args(parser, paths, ['--set', 'conf=0.35', '--set', 'iou=0.6', 'video.mp4'])
    assert args.source == Path('video.mp4')
    assert args.set == ['conf=0.35', 'iou=0.6']


# --- sync_args_with_config ---------------------------------------------------------------------

def _synced(paths, cfg, **arg_values):
    args = argparse.Namespace(_cfg_paths=paths, **arg_values)
    sync_args_with_config(args, cfg, logger)
    return args


def test_sync_pulls_when_the_flag_is_unset(cfg):
    args = _synced({'conf': CfgArg('ultralytics.conf')}, cfg, conf=None)
    assert args.conf == 0.25


def test_sync_pushes_when_the_flag_is_set(cfg):
    _synced({'conf': CfgArg('ultralytics.conf')}, cfg, conf=0.5)
    assert cfg['ultralytics']['conf'] == 0.5


def test_sync_inverts_both_ways():
    cfg = {'georef': {'processing': {'use_master': True}}}
    spec = {'no_master': CfgArg('georef.processing.use_master', invert=True)}
    assert _synced(spec, cfg, no_master=None).no_master is False
    _synced(spec, cfg, no_master=True)
    assert cfg['georef']['processing']['use_master'] is False


def test_sync_skips_no_sync_in_both_directions(cfg):
    """
    --class-names must not be touched either way: pushing would put ID=NAME tokens where a mapping
    belongs, and pulling would make a config-set class_rename look like it came from the CLI,
    mislabelling class_names_source in the run metadata.
    """
    spec = {'class_names': CfgArg('extraction.class_rename', no_sync=True)}
    cfg['extraction']['class_rename'] = {0: 'car'}
    args = _synced(spec, cfg, class_names=['0=van'])
    assert cfg['extraction']['class_rename'] == {0: 'car'}  # not pushed
    assert args.class_names == ['0=van']
    args = _synced(spec, cfg, class_names=None)
    assert args.class_names is None                          # not pulled


def test_sync_coerces_on_pull():
    cfg = {'input': {'ortho_folder': '/data/ORTHOPHOTOS'}}
    args = _synced({'ortho_folder': CfgArg('input.ortho_folder', coerce=Path)}, cfg, ortho_folder=None)
    assert args.ortho_folder == Path('/data/ORTHOPHOTOS')


def test_sync_does_not_coerce_a_null(cfg):
    cfg['input'] = {'ortho_folder': None}
    args = _synced({'ortho_folder': CfgArg('input.ortho_folder', coerce=Path)}, cfg, ortho_folder=None)
    assert args.ortho_folder is None


def test_sync_falls_back_to_the_bundled_default_when_the_key_is_absent(cfg):
    """A custom config predating a new key must still load, falling back to the shipped default."""
    args = _synced({'stab_device': CfgArg('stabilo.device')}, cfg, stab_device=None)
    assert args.stab_device == 'auto'  # cfg/default.yaml -> stabilo -> device
    assert 'device' not in cfg['stabilo']  # the loaded config itself is left untouched


# --- anti-drift --------------------------------------------------------------------------------

def _all_parsers():
    from geotrax import batch_process, georeference, plot, visualize
    from geotrax import extract as extract_mod
    return {
        'extract': extract_mod.parse_cli_args,
        'georeference': georeference.parse_cli_args,
        'visualize': visualize.parse_cli_args,
        'plot': plot.parse_cli_args,
        'batch': batch_process.parse_cli_args,
    }


@pytest.mark.parametrize('command', ['extract', 'georeference', 'visualize', 'plot', 'batch'])
def test_every_registered_path_exists_in_the_default_config(command, monkeypatch):
    """
    The dest -> config path map is hand-written next to each flag; this is what stops it drifting
    from the config it points at. A renamed config key or a typo'd cfg= fails here rather than
    silently turning a flag into a no-op.
    """
    monkeypatch.setattr('sys.argv', [command, 'dummy_source'])
    args = _all_parsers()[command]()
    full = load_config('default', logger)
    assert args._cfg_paths, f'{command} registered no config-backed flags'
    for dest, spec in args._cfg_paths.items():
        assert _get_by_path(full, spec.path) is not _MISSING, \
            f"{command}: --{dest.replace('_', '-')} points at '{spec.path}', which is not in default.yaml"


def test_visualize_and_plot_disagree_about_bare_save(monkeypatch):
    """
    Both register dest 'save', for different config keys. This is why the map is per-parser and
    not a module-level dict; merging them globally would silently mis-route one of the two.
    """
    from geotrax import plot, visualize
    monkeypatch.setattr('sys.argv', ['visualize', 'dummy'])
    viz_paths = visualize.parse_cli_args()._cfg_paths
    monkeypatch.setattr('sys.argv', ['plot', 'dummy'])
    plot_paths = plot.parse_cli_args()._cfg_paths
    assert viz_paths['save'].path == 'visualization.save'
    assert plot_paths['save'].path == 'plotting.save'


def test_batch_keeps_the_visualization_and_plotting_flags_apart(monkeypatch):
    from geotrax import batch_process
    monkeypatch.setattr('sys.argv', ['batch', 'dummy'])
    paths = batch_process.parse_cli_args()._cfg_paths
    assert paths['save'].path == 'visualization.save'
    assert paths['plot_save'].path == 'plotting.save'


# --- help text ---------------------------------------------------------------------------------

def _help(parser):
    """The rendered help with argparse's line wrapping collapsed, so assertions can span it."""
    return ' '.join(parser.format_help().split())


def test_default_sentence_is_generated_from_the_registered_path():
    parser = argparse.ArgumentParser()
    paths = {}
    add_cfg_arg(parser, '--thing', cfg='extraction.sahi.enable', paths=paths, help='Do the thing.')
    assert 'Defaults to cfg -> extraction -> sahi -> enable.' in _help(parser)
    assert paths['thing'].path == 'extraction.sahi.enable'


def test_default_note_is_appended():
    parser = argparse.ArgumentParser()
    add_cfg_arg(parser, '--thing', cfg='input.ortho_folder', paths={}, help='Do it.',
                default_note='then the sibling folder')
    assert 'Defaults to cfg -> input -> ortho_folder, then the sibling folder.' in _help(parser)


def test_default_note_none_suppresses_generation():
    parser = argparse.ArgumentParser()
    add_cfg_arg(parser, '--thing', cfg='input.ortho_folder', paths={}, help='Phrased by hand.',
                default_note=None)
    assert 'Defaults to cfg' not in _help(parser)
