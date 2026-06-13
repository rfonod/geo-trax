# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure track post-processing helpers in extract.py."""

import logging

import numpy as np

from geotrax.extract import aggregate_results, calculate_unique_classes, remove_short_tracks

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
