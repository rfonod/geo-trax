# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure track post-processing helpers in extract.py."""

import argparse
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from ultralytics.engine.results import Results
from ultralytics.trackers.bot_sort import BOTSORT

from geotrax.extract import (
    ClassConf,
    add_processing_args,
    aggregate_results,
    calculate_unique_classes,
    class_conf_mask,
    create_manual_tracker,
    detect_frame_sahi,
    estimate_vehicle_dimensions,
    interpolate_tracks,
    load_sahi_detector,
    make_class_conf_callback,
    postprocess_tracks,
    remove_short_tracks,
    resolve_class_conf,
    sahi_predictions_to_boxes,
    validate_sahi_tracker,
)
from geotrax.utils.config_utils import load_config
from geotrax.utils.constants import DEFAULT_TRACK_BUFFER

logger = logging.getLogger(__name__)


def _parse_processing(argv):
    parser = argparse.ArgumentParser()
    add_processing_args(parser)
    return parser.parse_args(argv)


def test_stab_gpu_flags_default_to_none():
    args = _parse_processing([])
    assert args.stab_gpu is None
    assert args.stab_gpu_device_id is None


def test_stab_gpu_flags_parse():
    args = _parse_processing(['--stab-gpu', '--stab-gpu-device-id', '2'])
    assert args.stab_gpu is True
    assert args.stab_gpu_device_id == 2
    assert _parse_processing(['--no-stab-gpu']).stab_gpu is False


def test_stab_detector_and_device_default_to_none():
    args = _parse_processing([])
    assert args.stab_detector is None
    assert args.stab_device is None


def test_stab_detector_and_device_parse():
    args = _parse_processing(['--stab-detector', 'xfeat', '--stab-device', 'cpu'])
    assert args.stab_detector == 'xfeat'
    assert args.stab_device == 'cpu'
    assert _parse_processing(['-sdet', 'orb', '-sdev', 'auto']).stab_detector == 'orb'


def test_stab_detector_rejects_unknown_value():
    with pytest.raises(SystemExit):
        _parse_processing(['--stab-detector', 'not-a-detector'])
    with pytest.raises(SystemExit):
        _parse_processing(['--stab-device', 'tpu'])


def test_sahi_flag_defaults_to_none():
    assert _parse_processing([]).sahi is None


def test_sahi_flag_parses():
    assert _parse_processing(['--sahi']).sahi is True
    assert _parse_processing(['--no-sahi']).sahi is False


# --- SAHI helpers -------------------------------------------------------------

def _make_sahi_pred(x1, y1, x2, y2, score, cls):
    # Duck-typed stand-in for a sahi ObjectPrediction; no sahi install needed.
    return SimpleNamespace(
        bbox=SimpleNamespace(to_xyxy=lambda: [x1, y1, x2, y2]),
        score=SimpleNamespace(value=score),
        category=SimpleNamespace(id=cls),
    )


def test_sahi_predictions_to_boxes_converts_and_filters_classes():
    preds = [_make_sahi_pred(10, 20, 30, 60, 0.9, 2), _make_sahi_pred(0, 0, 10, 10, 0.5, 7)]
    boxes = sahi_predictions_to_boxes(preds, (1080, 1920), classes=[0, 1, 2, 3])
    assert len(boxes) == 1  # class 7 filtered out
    np.testing.assert_allclose(boxes.xywh.numpy(force=True), [[20, 40, 20, 40]])  # xyxy -> center xywh
    assert boxes.cls.item() == 2
    assert boxes.conf.item() == pytest.approx(0.9)
    assert boxes.id is None  # detections carry no track IDs yet


def test_sahi_predictions_to_boxes_no_class_filter_keeps_all():
    preds = [_make_sahi_pred(10, 20, 30, 60, 0.9, 2), _make_sahi_pred(0, 0, 10, 10, 0.5, 7)]
    assert len(sahi_predictions_to_boxes(preds, (1080, 1920), classes=None)) == 2


def test_sahi_predictions_to_boxes_empty():
    boxes = sahi_predictions_to_boxes([], (1080, 1920), classes=[0, 1, 2, 3])
    assert len(boxes) == 0


def _fake_sahi_modules(prediction_result):
    sahi_mod = types.ModuleType('sahi')
    sahi_mod.AutoDetectionModel = SimpleNamespace(from_pretrained=lambda **kwargs: None)
    predict_mod = types.ModuleType('sahi.predict')
    predict_mod.get_sliced_prediction = lambda *a, **k: prediction_result
    sahi_mod.predict = predict_mod
    return {'sahi': sahi_mod, 'sahi.predict': predict_mod}


def _default_sahi_cfg(enable=True):
    return {
        'enable': enable, 'slice_height': 1080, 'slice_width': 1920,
        'overlap_height_ratio': 0.2, 'overlap_width_ratio': 0.2,
        'perform_standard_pred': True, 'postprocess_type': 'GREEDYNMM',
        'postprocess_match_metric': 'IOS', 'postprocess_match_threshold': 0.5,
        'class_agnostic': True,
    }


def test_detect_frame_sahi_maps_tracker_output_to_boxes():
    prediction = SimpleNamespace(
        object_prediction_list=[_make_sahi_pred(10, 20, 30, 60, 0.9, 2)],
        durations_in_seconds={'slice': 0.001, 'prediction': 0.002},
    )
    tracker = SimpleNamespace(update=lambda det, img: np.array([[10, 20, 30, 60, 5, 0.9, 2, 0]]))
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    with patch.dict(sys.modules, _fake_sahi_modules(prediction)):
        boxes, speed = detect_frame_sahi(None, tracker, frame, _default_sahi_cfg(), classes=[0, 1, 2, 3])
    assert boxes.id.item() == 5
    np.testing.assert_allclose(boxes.xywh.numpy(force=True), [[20, 40, 20, 40]])
    assert boxes.cls.item() == 2
    assert boxes.conf.item() == pytest.approx(0.9)
    assert set(speed) == {'preprocess', 'inference', 'postprocess'}  # keys used by update_progress_bar


def test_detect_frame_sahi_keeps_detections_when_tracker_returns_nothing():
    # Mirrors ultralytics: with no confirmed tracks, the raw detections survive with id None
    # (written as -1 downstream) instead of dropping the frame.
    prediction = SimpleNamespace(
        object_prediction_list=[_make_sahi_pred(10, 20, 30, 60, 0.9, 2)],
        durations_in_seconds={},
    )
    tracker = SimpleNamespace(update=lambda det, img: np.empty((0, 8)))
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    with patch.dict(sys.modules, _fake_sahi_modules(prediction)):
        boxes, _ = detect_frame_sahi(None, tracker, frame, _default_sahi_cfg(), classes=None)
    assert len(boxes) == 1
    assert boxes.id is None


def test_create_manual_tracker_botsort_from_default_config():
    full = load_config('default', logger)
    tracker = create_manual_tracker({'tracker_active': 'botsort', 'tracker_params': full['tracker']['botsort']})
    assert isinstance(tracker, BOTSORT)


def test_validate_sahi_tracker_rejects_tracktrack():
    with pytest.raises(ValueError, match='tracktrack'):
        validate_sahi_tracker({'tracker_active': 'tracktrack', 'tracker_params': {}})


def test_validate_sahi_tracker_rejects_auto_reid():
    with pytest.raises(ValueError, match='ReID'):
        validate_sahi_tracker({'tracker_active': 'botsort', 'tracker_params': {'with_reid': True, 'model': 'auto'}})


def test_validate_sahi_tracker_accepts_botsort_defaults():
    validate_sahi_tracker({'tracker_active': 'botsort', 'tracker_params': {'with_reid': False, 'model': 'auto'}})


def test_load_sahi_detector_missing_dependency_exits(caplog):
    config = {'model': 'no/such/model.pt', 'conf': 0.25, 'device': None, 'imgsz': 1920}
    with patch.dict(sys.modules, {'sahi': None}), caplog.at_level(logging.CRITICAL):
        with pytest.raises(SystemExit):
            load_sahi_detector(config, logger)
    assert any("geo-trax[sahi]" in r.message for r in caplog.records)


def test_load_sahi_detector_rejects_multi_gpu_device_list():
    config = {'model': 'no/such/model.pt', 'conf': 0.25, 'device': [0, 1], 'imgsz': 1920}
    with patch.dict(sys.modules, _fake_sahi_modules(None)):
        with pytest.raises(ValueError, match='Multi-GPU'):
            load_sahi_detector(config, logger)


def test_remove_short_tracks_drops_below_min_length():
    # columns: [frame_id, vehicle_id]; id 1 appears 3x (kept), id 2 appears 2x (dropped)
    tracks = np.array(
        [[0, 1], [1, 1], [2, 1], [0, 2], [1, 2]], dtype=np.float32
    )
    result = remove_short_tracks(tracks, logger, min_length=3)
    assert result.shape == (3, 2)
    assert set(np.unique(result[:, 1])) == {1}


def test_remove_short_tracks_empty():
    empty = np.empty((0, 2), dtype=np.float32)
    assert remove_short_tracks(empty, logger).size == 0


def test_calculate_unique_classes_uses_confidence_weighted_vote():
    # columns: [frame_id, vehicle_id, class_id, conf]
    # id 1: class 0 (conf 0.9) vs class 1 (conf 0.95) -> class 1 wins for all its rows
    tracks = np.array(
        [[0, 1, 0, 0.90], [1, 1, 1, 0.95], [0, 2, 2, 0.80]], dtype=np.float32
    )
    result = calculate_unique_classes(tracks)
    id1_classes = result[result[:, 1] == 1][:, -2]
    np.testing.assert_array_equal(id1_classes, [1, 1])
    # single-detection track keeps its only class
    assert result[result[:, 1] == 2][0, -2] == 2


def test_calculate_unique_classes_handles_arbitrary_class_ids():
    # A high class id (e.g. from a many-class model) must not raise; the vote is dict-based,
    # not sized off a configured class list.
    tracks = np.array([[0, 1, 7, 0.9], [1, 1, 7, 0.8]], dtype=np.float32)
    result = calculate_unique_classes(tracks)
    np.testing.assert_array_equal(result[:, -2], [7, 7])


def test_aggregate_results_concatenates_and_drops_unmatched():
    frame_arr = [np.array([[0], [0]])]
    track_id = [np.array([[1], [-1]])]  # -1 == unmatched detection, must be dropped
    bbox = [np.array([[10, 10, 4, 2], [20, 20, 4, 2]])]
    bbox_stab = [np.array([[11, 11, 4, 2], [21, 21, 4, 2]])]
    class_id = [np.array([[0], [0]])]
    conf = [np.array([[0.9], [0.8]])]
    transforms = [np.zeros((1, 10))]

    tracks, transf = aggregate_results(
        frame_arr, track_id, bbox, bbox_stab, class_id, conf, transforms, logger
    )
    # 12 columns: frame + id + 4 bbox + 4 bbox_stab + class + conf
    assert tracks.shape == (1, 12)
    assert tracks[0, 1] == 1
    assert transf.shape == (1, 10)


# --- estimate_vehicle_dimensions -------------------------------------------

def _make_dim_config():
    return {
        'args': argparse.Namespace(source=Path('dummy.mp4')),
        'extraction': {
            'dimension_estimation': {
                'eps': 5,
                'r0': 3.0,
                'gsd': 0.02725,
                'theta_bar': 15,
                'tau_c': {-1: 1.0},  # l/w >= 1.0 → all bboxes pass ratio check
            }
        },
    }


def test_estimate_vehicle_dimensions_empty_input():
    empty = np.empty((0, 12), dtype=np.float32)
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)):
        result = estimate_vehicle_dimensions(empty, _make_dim_config())
    assert result.shape == (0, 14)


def test_estimate_vehicle_dimensions_well_inside_frame():
    # 3 detections for vehicle 1: stationary at frame centre (azimuth=None → ratio fallback)
    # w=100 > h=30 → length=100, width=30; 100/30 > tau_c[-1]=1.0 → kept
    tracks = np.array(
        [[0, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [1, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [2, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9]],
        dtype=np.float32,
    )
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)):
        result = estimate_vehicle_dimensions(tracks, _make_dim_config())
    assert result.shape == (3, 14)
    np.testing.assert_allclose(result[:, -2], 100.0)  # length_px
    np.testing.assert_allclose(result[:, -1], 30.0)   # width_px


def test_estimate_vehicle_dimensions_boundary_vehicle_gets_nan():
    # Vehicle right at frame edge: x=5, w=20 → x-w/2=-5, which fails the eps=5 check
    tracks = np.array(
        [[0, 1, 5, 540, 20, 10, 5, 540, 20, 10, 0, 0.9]],
        dtype=np.float32,
    )
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)):
        result = estimate_vehicle_dimensions(tracks, _make_dim_config())
    assert result.shape == (1, 14)
    assert np.isnan(result[0, -2])
    assert np.isnan(result[0, -1])


# --- interpolate_tracks -------------------------------------------------------

def test_interpolate_tracks_empty_input():
    empty = np.empty((0, 14), dtype=np.float32)
    result = interpolate_tracks(empty, logger, max_gap=30)
    assert result.size == 0


def test_interpolate_tracks_no_gaps_adds_flag_column():
    # Consecutive frames (0, 1, 2) — no gaps, only is_interpolated column added.
    tracks = np.zeros((3, 14), dtype=np.float32)
    tracks[:, 0] = [0, 1, 2]   # frame_id
    tracks[:, 1] = [1, 1, 1]   # vehicle_id
    result = interpolate_tracks(tracks, logger, max_gap=30)
    assert result.shape == (3, 15)
    np.testing.assert_array_equal(result[:, 14], [0, 0, 0])   # all detected


def test_interpolate_tracks_fills_gap_with_linear_interpolation():
    # Vehicle 1: frames 0 and 3 — gap of 2 frames (1 and 2 must be inserted).
    tracks = np.zeros((2, 14), dtype=np.float32)
    tracks[0, 0] = 0;  tracks[0, 1] = 1;  tracks[0, 6] = 0.0   # x_stab at frame 0
    tracks[1, 0] = 3;  tracks[1, 1] = 1;  tracks[1, 6] = 3.0   # x_stab at frame 3
    result = interpolate_tracks(tracks, logger, max_gap=30)
    assert result.shape == (4, 15)
    # Check frame ids are complete
    frames = result[result[:, 1] == 1, 0].astype(int)
    np.testing.assert_array_equal(sorted(frames), [0, 1, 2, 3])
    # Check x_stab (col 6) is linearly interpolated
    sorted_idx = np.argsort(result[:, 0])
    x_vals = result[sorted_idx, 6]
    np.testing.assert_allclose(x_vals, [0.0, 1.0, 2.0, 3.0], atol=1e-5)
    # Detected rows have flag 0, interpolated have flag 1
    flags = result[sorted_idx, 14]
    np.testing.assert_array_equal(flags, [0, 1, 1, 0])


def test_interpolate_tracks_dimensions_unchanged_by_interpolation():
    # Dimension columns (12, 13) are per-track constants; after interpolation they must
    # be identical in the interpolated rows (linear interpolation of equal values).
    tracks = np.zeros((2, 14), dtype=np.float32)
    tracks[0, 0] = 0;  tracks[0, 1] = 1;  tracks[0, 12] = 5.0;  tracks[0, 13] = 2.0
    tracks[1, 0] = 2;  tracks[1, 1] = 1;  tracks[1, 12] = 5.0;  tracks[1, 13] = 2.0
    result = interpolate_tracks(tracks, logger, max_gap=30)
    assert result.shape == (3, 15)
    np.testing.assert_allclose(result[:, 12], 5.0)
    np.testing.assert_allclose(result[:, 13], 2.0)


def test_interpolate_tracks_multiple_tracks_independent():
    # Two tracks with independent gaps — no cross-contamination.
    tracks = np.zeros((4, 14), dtype=np.float32)
    tracks[0, 0] = 0;  tracks[0, 1] = 1   # track 1, frame 0
    tracks[1, 0] = 2;  tracks[1, 1] = 1   # track 1, frame 2  (gap at frame 1)
    tracks[2, 0] = 0;  tracks[2, 1] = 2   # track 2, frame 0
    tracks[3, 0] = 1;  tracks[3, 1] = 2   # track 2, frame 1  (no gap)
    result = interpolate_tracks(tracks, logger, max_gap=30)
    # track 1 gets 1 synthetic row; track 2 gets none → 5 rows total, 15 columns
    assert result.shape == (5, 15)
    t1_rows = result[result[:, 1] == 1]
    t2_rows = result[result[:, 1] == 2]
    assert len(t1_rows) == 3
    assert len(t2_rows) == 2
    assert int(t1_rows[t1_rows[:, 14] == 1, 0][0]) == 1   # synthetic row is frame 1


def test_interpolate_tracks_skips_gap_exceeding_max_gap():
    # Vehicle 1: frames 0 and 5 — gap of 4 frames, exceeds max_gap=2, so left unfilled.
    tracks = np.zeros((2, 14), dtype=np.float32)
    tracks[0, 0] = 0;  tracks[0, 1] = 1
    tracks[1, 0] = 5;  tracks[1, 1] = 1
    result = interpolate_tracks(tracks, logger, max_gap=2)
    assert result.shape == (2, 15)   # no synthetic rows added
    np.testing.assert_array_equal(result[:, 14], [0, 0])


# --- postprocess_tracks (smoke test) ----------------------------------------

def _make_postprocess_config(interpolate=False, tracker_params=None):
    # Mirrors the real shape returned by load_config_all() (config_utils.py): 'tracker' is
    # never a key of 'main' — it's exposed as 'tracker_active' + 'tracker_params' instead.
    if tracker_params is None:
        tracker_params = {'track_buffer': 30}
    return {
        'main': {
            'extraction': {
                'min_track_length': 3,
                'interpolate': interpolate,
                'dimension_estimation': {
                    'eps': 5, 'r0': 3.0, 'gsd': 0.02725, 'theta_bar': 15, 'tau_c': {-1: 1.0},
                },
            },
            'tracker_active': 'botsort',
            'tracker_params': tracker_params,
            'args': argparse.Namespace(source=Path('dummy.mp4'), interpolate=interpolate),
        }
    }


def test_postprocess_tracks_removes_short_and_adds_dimension_columns():
    # vehicle 1: 3 consecutive rows (kept), vehicle 2: 2 rows (removed)
    tracks = np.array(
        [[0, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [1, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [2, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [0, 2, 480, 270, 100, 30, 480, 270, 100, 30, 0, 0.9],
         [1, 2, 480, 270, 100, 30, 480, 270, 100, 30, 0, 0.9]],
        dtype=np.float32,
    )
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)):
        result = postprocess_tracks(tracks, _make_postprocess_config(interpolate=False), logger)
    assert result.shape == (3, 14)   # 14 columns — no is_interpolated when disabled
    assert set(result[:, 1].astype(int)) == {1}


def test_postprocess_tracks_with_interpolation_adds_15th_column():
    # vehicle 1: frames 0 and 2 (gap at frame 1); interpolation inserts the missing row.
    tracks = np.array(
        [[0, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [1, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [2, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9]],
        dtype=np.float32,
    )
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)):
        result = postprocess_tracks(tracks, _make_postprocess_config(interpolate=True), logger)
    assert result.shape[1] == 15   # is_interpolated column present
    np.testing.assert_array_equal(result[:, 14], [0, 0, 0])   # no gaps → all detected


def test_postprocess_tracks_falls_back_to_default_track_buffer_when_missing(caplog):
    # tracker_params empty (e.g. active tracker config omits 'track_buffer') -> DEFAULT_TRACK_BUFFER
    # is used instead of raising, and a warning is logged.
    tracks = np.array(
        [[0, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [1, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9],
         [2, 1, 960, 540, 100, 30, 960, 540, 100, 30, 0, 0.9]],
        dtype=np.float32,
    )
    config = _make_postprocess_config(interpolate=True, tracker_params={})
    with patch('geotrax.extract.get_video_dimensions', return_value=(1920, 1080)), \
         patch('geotrax.extract.interpolate_tracks', wraps=interpolate_tracks) as mock_interp, \
         caplog.at_level(logging.WARNING):
        result = postprocess_tracks(tracks, config, logger)
    assert mock_interp.call_args.args[2] == DEFAULT_TRACK_BUFFER
    assert any("no 'track_buffer' parameter" in r.message for r in caplog.records)
    assert result.shape[1] == 15


# --- run metadata ------------------------------------------------------------------------------

def _metadata_config(source, output_cfg, **overrides):
    """A minimal load_config_all()-shaped config, enough to drive save_results()."""
    main = {
        'args': SimpleNamespace(source=source, cfg='default'),
        'output': output_cfg,
        'processing': {'cut_frame_left': 0, 'cut_frame_right': None},
        'extraction': {'save_stab': False, 'interpolate': False},
        'class_names': {0: 'Car'},
        'class_names_source': 'model',
        'model_configured': 'hf://rfonod/geo-trax/model.pt',
        'tracker_active': 'botsort',
        'tracker_params': {'track_buffer': 30},
        'visualization': {'show_lanes': False},
        'plotting': {}, 'batch': {}, 'input': {},
    }
    main.update(overrides)
    return {'main': main, 'ultralytics': {'model': '/tmp/model.pt', 'conf': 0.25},
            'stabilo': {'gpu': False}, 'georef': {}}


def _run_save_results(tmp_path, output_cfg, **overrides):
    from geotrax.extract import save_results
    source = tmp_path / 'A1.mp4'
    source.touch()
    config = _metadata_config(source, output_cfg, **overrides)
    save_results(np.zeros((0, 14)), np.zeros((0, 10)), config, logger, output_cfg)
    return source


def test_run_metadata_is_written_inside_the_output_folder(tmp_path):
    source = _run_save_results(tmp_path, {'folder': 'results'})
    assert (tmp_path / 'results' / 'A1.yaml').is_file()
    assert not source.with_suffix('.yaml').exists()  # no longer written next to the video


def test_run_metadata_honours_the_configured_postfix(tmp_path):
    _run_save_results(tmp_path, {'folder': 'out', 'metadata_postfix': '_run'})
    assert (tmp_path / 'out' / 'A1_run.yaml').is_file()


def test_run_metadata_records_the_effective_configuration(tmp_path):
    """
    The point of the file: it must show what the run used, not what the config file said.
    save_results() reads the config dict, which sync_args_with_config has already reconciled
    with the CLI flags, so a --cut-frame-left 10 run records 10.
    """
    _run_save_results(tmp_path, {'folder': 'results'},
                      processing={'cut_frame_left': 10, 'cut_frame_right': 90},
                      extraction={'save_stab': False, 'interpolate': True})
    written = yaml.safe_load((tmp_path / 'results' / 'A1.yaml').read_text())
    assert written['processing'] == {'cut_frame_left': 10, 'cut_frame_right': 90}
    assert written['extraction']['interpolate'] is True
    assert written['output']['folder'] == 'results'


def test_save_results_removes_stale_outputs_of_an_earlier_run(tmp_path):
    """With no new tracks or transforms, older files must not be left for later stages to read."""
    (tmp_path / 'results').mkdir()
    stale_tracks = tmp_path / 'results' / 'A1.txt'
    stale_transforms = tmp_path / 'results' / 'A1_vid_transf.txt'
    stale_tracks.write_text('0,1,2\n')
    stale_transforms.write_text('1,1,0,0,0,1,0,0,0,1\n')
    _run_save_results(tmp_path, {'folder': 'results'})
    assert not stale_tracks.exists()
    assert not stale_transforms.exists()
    assert (tmp_path / 'results' / 'A1.yaml').is_file()


def test_save_results_writes_the_tracks_file_without_leftovers(tmp_path):
    from geotrax.extract import save_results
    source = tmp_path / 'A1.mp4'
    source.touch()
    output_cfg = {'folder': 'results'}
    tracks = np.array([[0, 1, 10, 20, 4, 2, 10, 20, 4, 2, 0, 0.9, 4.5, 1.8]], dtype=np.float32)
    save_results(tracks, np.zeros((0, 10)), _metadata_config(source, output_cfg), logger, output_cfg)
    assert np.loadtxt(tmp_path / 'results' / 'A1.txt', delimiter=',').shape == (14,)
    assert sorted(p.name for p in (tmp_path / 'results').iterdir()) == ['A1.txt', 'A1.yaml']


def test_aggregate_results_raises_instead_of_returning_empty_tracks():
    from geotrax.extract import ExtractionError
    with pytest.raises(ExtractionError):
        aggregate_results([np.zeros((2, 1))], [np.zeros((3, 1))], [np.zeros((2, 4))], [], [np.zeros((2, 1))],
                          [np.zeros((2, 1))], [], logger)


# --- Per-class confidence thresholds ------------------------------------------

def test_resolve_class_conf_null_keeps_global_threshold():
    cc = resolve_class_conf({'conf': 0.25, 'classes': [0, 1, 2, 3]}, {'class_conf': None}, logger)
    assert cc == ClassConf(0.25, 0.25, None)


def test_resolve_class_conf_missing_key_keeps_global_threshold():
    assert resolve_class_conf({'conf': 0.3}, {}, logger).thresholds is None


def test_resolve_class_conf_lowers_predict_conf_to_the_minimum():
    cc = resolve_class_conf({'conf': 0.25, 'classes': None}, {'class_conf': {1: 0.1, 2: 0.6}}, logger)
    assert cc.predict_conf == pytest.approx(0.1)
    assert cc.default == pytest.approx(0.25)
    assert cc.thresholds == {1: 0.1, 2: 0.6}


def test_resolve_class_conf_higher_thresholds_keep_global_predict_conf():
    cc = resolve_class_conf({'conf': 0.25, 'classes': None}, {'class_conf': {2: 0.6}}, logger)
    assert cc.predict_conf == pytest.approx(0.25)


def test_resolve_class_conf_null_global_conf_uses_track_fallback():
    cc = resolve_class_conf({'conf': None, 'classes': None}, {'class_conf': {2: 0.6}}, logger)
    assert cc.default == pytest.approx(0.1)


def test_resolve_class_conf_accepts_string_keys():
    cc = resolve_class_conf({'conf': 0.25, 'classes': None}, {'class_conf': {'2': 0.5}}, logger)
    assert cc.thresholds == {2: 0.5}


@pytest.mark.parametrize('class_conf', [
    0.5,
    [0.5],
    {'car': 0.5},
    {True: 0.5},
    {2: 1.5},
    {2: -0.1},
    {2: 'high'},
    {2: True},
])
def test_resolve_class_conf_rejects_invalid_values(class_conf):
    with pytest.raises(ValueError, match='class_conf'):
        resolve_class_conf({'conf': 0.25, 'classes': None}, {'class_conf': class_conf}, logger)


def test_resolve_class_conf_warns_about_excluded_classes(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_class_conf({'conf': 0.25, 'classes': [0, 1]}, {'class_conf': {3: 0.5}}, logger)
    assert any('[3]' in r.message for r in caplog.records)


def test_class_conf_mask_applies_per_class_and_default_thresholds():
    cc = ClassConf(0.1, 0.25, {1: 0.1, 2: 0.6})
    cls = np.array([0, 0, 1, 2, 2])
    conf = np.array([0.2, 0.3, 0.15, 0.5, 0.7])
    np.testing.assert_array_equal(class_conf_mask(cls, conf, cc), [False, True, True, False, True])


def test_class_conf_mask_is_strict_like_ultralytics_nms():
    cc = ClassConf(0.25, 0.25, {2: 0.6})
    np.testing.assert_array_equal(class_conf_mask(np.array([2, 0]), np.array([0.6, 0.25]), cc), [False, False])


def _fake_predictor(rows, feats=None):
    result = Results(
        orig_img=np.zeros((108, 192, 3), dtype=np.uint8),
        path='frame.jpg',
        names={0: 'car', 1: 'bus', 2: 'truck'},
        boxes=torch.tensor(rows, dtype=torch.float32),
    )
    if feats is not None:
        result.feats = feats
    return SimpleNamespace(results=[result])


def test_class_conf_callback_filters_boxes_before_the_tracker():
    predictor = _fake_predictor([
        [0, 0, 10, 10, 0.20, 0],
        [0, 0, 10, 10, 0.30, 0],
        [0, 0, 10, 10, 0.50, 2],
        [0, 0, 10, 10, 0.70, 2],
    ])
    make_class_conf_callback(ClassConf(0.1, 0.25, {2: 0.6}))(predictor)
    boxes = predictor.results[0].boxes
    np.testing.assert_allclose(boxes.conf.numpy(force=True), [0.3, 0.7])
    np.testing.assert_array_equal(boxes.cls.numpy(force=True), [0, 2])
    assert predictor.results[0].path == 'frame.jpg'


def test_class_conf_callback_slices_reid_features():
    feats = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    predictor = _fake_predictor([
        [0, 0, 10, 10, 0.9, 0],
        [0, 0, 10, 10, 0.5, 2],
        [0, 0, 10, 10, 0.8, 2],
    ], feats=feats)
    make_class_conf_callback(ClassConf(0.25, 0.25, {2: 0.6}))(predictor)
    np.testing.assert_array_equal(predictor.results[0].feats.numpy(), [[0.0], [2.0]])


def test_class_conf_callback_leaves_unaffected_results_untouched():
    predictor = _fake_predictor([[0, 0, 10, 10, 0.9, 0]])
    original = predictor.results[0]
    make_class_conf_callback(ClassConf(0.25, 0.25, {2: 0.6}))(predictor)
    assert predictor.results[0] is original


def test_class_conf_callback_can_drop_every_box():
    predictor = _fake_predictor([[0, 0, 10, 10, 0.5, 2]])
    make_class_conf_callback(ClassConf(0.25, 0.25, {2: 0.6}))(predictor)
    assert len(predictor.results[0].boxes) == 0


def test_class_conf_callback_filters_the_tracktrack_loose_nms_pass():
    predictor = _fake_predictor([[0, 0, 10, 10, 0.9, 0]])
    loose = _fake_predictor([[0, 0, 10, 10, 0.9, 0], [50, 50, 60, 60, 0.5, 2]]).results
    calls = []

    def orig_postprocess(*args, **kwargs):
        calls.append(kwargs)
        return loose

    predictor._orig_postprocess = orig_postprocess
    callback = make_class_conf_callback(ClassConf(0.25, 0.25, {2: 0.6}))
    callback(predictor)
    callback(predictor)
    results = predictor._orig_postprocess(None, None, None, iou=0.95)
    assert calls == [{'iou': 0.95}]
    np.testing.assert_array_equal(results[0].boxes.cls.numpy(force=True), [0])


def test_sahi_predictions_to_boxes_applies_class_conf():
    preds = [
        _make_sahi_pred(0, 0, 10, 10, 0.3, 0),
        _make_sahi_pred(0, 0, 10, 10, 0.5, 2),
        _make_sahi_pred(0, 0, 10, 10, 0.7, 2),
    ]
    boxes = sahi_predictions_to_boxes(preds, (1080, 1920), classes=None, class_conf=ClassConf(0.25, 0.25, {2: 0.6}))
    np.testing.assert_allclose(boxes.conf.numpy(force=True), [0.3, 0.7])


def test_sahi_predictions_to_boxes_without_class_conf_is_unchanged():
    preds = [_make_sahi_pred(0, 0, 10, 10, 0.5, 2)]
    boxes = sahi_predictions_to_boxes(preds, (1080, 1920), classes=None, class_conf=ClassConf(0.25, 0.25, None))
    assert len(boxes) == 1


def test_detect_frame_sahi_filters_before_tracker_update():
    prediction = SimpleNamespace(
        object_prediction_list=[_make_sahi_pred(10, 20, 30, 60, 0.5, 2), _make_sahi_pred(10, 20, 30, 60, 0.9, 0)],
        durations_in_seconds={},
    )
    seen = []

    def update(det, img):
        seen.append(len(det))
        return np.empty((0, 8))

    tracker = SimpleNamespace(update=update)
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    with patch.dict(sys.modules, _fake_sahi_modules(prediction)):
        boxes, _ = detect_frame_sahi(None, tracker, frame, _default_sahi_cfg(), classes=None,
                                     class_conf=ClassConf(0.25, 0.25, {2: 0.6}))
    assert seen == [1]
    assert boxes.cls.item() == 0


def test_load_sahi_detector_uses_lowered_conf():
    captured = {}
    modules = _fake_sahi_modules(None)
    modules['sahi'].AutoDetectionModel = SimpleNamespace(from_pretrained=lambda **kwargs: captured.update(kwargs))
    config = {'model': 'no/such/model.pt', 'conf': 0.25, 'device': 'cpu', 'imgsz': 1920}
    with patch.dict(sys.modules, modules), patch('geotrax.extract.check_yolo'):
        load_sahi_detector(config, logger, conf=0.1)
    assert captured['confidence_threshold'] == pytest.approx(0.1)


@pytest.mark.parametrize('preset', ['default', 'confident', 'lenient', 'stable'])
def test_presets_ship_class_conf_disabled(preset):
    assert load_config(preset, logger)['extraction']['class_conf'] is None
