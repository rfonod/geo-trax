# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure track post-processing helpers in extract.py."""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geotrax.extract import (
    aggregate_results,
    calculate_unique_classes,
    estimate_vehicle_dimensions,
    interpolate_tracks,
    postprocess_tracks,
    remove_short_tracks,
)

logger = logging.getLogger(__name__)


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
    result = interpolate_tracks(empty, logger)
    assert result.size == 0


def test_interpolate_tracks_no_gaps_adds_flag_column():
    # Consecutive frames (0, 1, 2) — no gaps, only is_interpolated column added.
    tracks = np.zeros((3, 14), dtype=np.float32)
    tracks[:, 0] = [0, 1, 2]   # frame_id
    tracks[:, 1] = [1, 1, 1]   # vehicle_id
    result = interpolate_tracks(tracks, logger)
    assert result.shape == (3, 15)
    np.testing.assert_array_equal(result[:, 14], [0, 0, 0])   # all detected


def test_interpolate_tracks_fills_gap_with_linear_interpolation():
    # Vehicle 1: frames 0 and 3 — gap of 2 frames (1 and 2 must be inserted).
    tracks = np.zeros((2, 14), dtype=np.float32)
    tracks[0, 0] = 0;  tracks[0, 1] = 1;  tracks[0, 6] = 0.0   # x_stab at frame 0
    tracks[1, 0] = 3;  tracks[1, 1] = 1;  tracks[1, 6] = 3.0   # x_stab at frame 3
    result = interpolate_tracks(tracks, logger)
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
    result = interpolate_tracks(tracks, logger)
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
    result = interpolate_tracks(tracks, logger)
    # track 1 gets 1 synthetic row; track 2 gets none → 5 rows total, 15 columns
    assert result.shape == (5, 15)
    t1_rows = result[result[:, 1] == 1]
    t2_rows = result[result[:, 1] == 2]
    assert len(t1_rows) == 3
    assert len(t2_rows) == 2
    assert int(t1_rows[t1_rows[:, 14] == 1, 0][0]) == 1   # synthetic row is frame 1


# --- postprocess_tracks (smoke test) ----------------------------------------

def _make_postprocess_config(interpolate=False):
    return {
        'main': {
            'extraction': {
                'min_track_length': 3,
                'interpolate': interpolate,
                'dimension_estimation': {
                    'eps': 5, 'r0': 3.0, 'gsd': 0.02725, 'theta_bar': 15, 'tau_c': {-1: 1.0},
                },
            },
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
