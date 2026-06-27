# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for visualization helpers."""

import logging

import numpy as np
import pandas as pd
import pytest

from geotrax.visualize import compute_headings, normalize_viz_modes, read_transforms

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'given, expected',
    [
        (0, [0]),                              # scalar from config
        ([1], [1]),                            # single-element list from CLI
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),    # all modes (incl. both oriented)
        ([0, 4], [0, 4]),                      # original + oriented (stabilized frame)
        ((2, 0), [2, 0]),                      # tuple, order preserved
        ([0, 0, 1], [0, 1]),                   # duplicates removed
    ],
)
def test_normalize_viz_modes(given, expected):
    assert normalize_viz_modes(given, logger) == expected


@pytest.mark.parametrize('given', [5, -1, [0, 5], 'all'])
def test_normalize_viz_modes_invalid_exits(given):
    with pytest.raises(SystemExit):
        normalize_viz_modes(given, logger)


# --- compute_headings --------------------------------------------------------

def _tracks_with_centers(track_id, xs, ys, w=10.0, h=10.0):
    """Build a minimal tracks frame with raw bbox w/h in cols 4/5 and stabilized centers in 6/7."""
    n = len(xs)
    df = pd.DataFrame(np.zeros((n, 8)))
    df[0] = np.arange(n)        # frame ids
    df[1] = track_id            # vehicle id
    df[4] = w                   # raw bbox width
    df[5] = h                   # raw bbox height
    df[6] = xs                  # stabilized center x
    df[7] = ys                  # stabilized center y
    return df


def test_compute_headings_straight_line_constant():
    # Vehicle moving along +x (image coords) -> heading ~ 0 rad everywhere.
    tracks = _tracks_with_centers(1, xs=np.arange(30) * 5.0, ys=np.full(30, 100.0))
    headings = compute_headings(tracks, smoothing=5, min_speed=0.5, logger=logger)
    assert not headings.isna().any()
    np.testing.assert_allclose(headings.to_numpy(), 0.0, atol=1e-6)


def test_compute_headings_diagonal():
    # Vehicle moving along +x/+y (down-right in image coords) -> heading ~ +pi/4.
    tracks = _tracks_with_centers(1, xs=np.arange(30) * 5.0, ys=np.arange(30) * 5.0)
    headings = compute_headings(tracks, smoothing=5, min_speed=0.5, logger=logger)
    np.testing.assert_allclose(headings.to_numpy(), np.pi / 4, atol=1e-6)


def test_compute_headings_stationary_vertical_box():
    # Stationary vehicle whose detection is taller than wide -> oriented vertically (heading pi/2).
    tracks = _tracks_with_centers(1, xs=np.full(30, 50.0), ys=np.full(30, 50.0), w=10.0, h=25.0)
    headings = compute_headings(tracks, smoothing=5, min_speed=0.5, logger=logger)
    assert not headings.isna().any()
    np.testing.assert_allclose(headings.to_numpy(), np.pi / 2)


def test_compute_headings_stationary_horizontal_box():
    # Stationary vehicle whose detection is wider than tall -> oriented horizontally (heading 0).
    tracks = _tracks_with_centers(1, xs=np.full(30, 50.0), ys=np.full(30, 50.0), w=25.0, h=10.0)
    headings = compute_headings(tracks, smoothing=5, min_speed=0.5, logger=logger)
    assert not headings.isna().any()
    np.testing.assert_allclose(headings.to_numpy(), 0.0)


# --- read_transforms ---------------------------------------------------------

def test_read_transforms_none_returns_none():
    assert read_transforms(None, logger) is None


def test_read_transforms_valid_file(tmp_path):
    # Two frames with identity matrices (det=1>0), space-delimited, 10 columns each.
    content = '0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n1 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n'
    filepath = tmp_path / 'transforms.txt'
    filepath.write_text(content)
    result = read_transforms(filepath, logger)
    assert set(result.keys()) == {0, 1}
    np.testing.assert_allclose(result[0], np.eye(3))
    np.testing.assert_allclose(result[1], np.eye(3))


def test_read_transforms_wrong_column_count_exits(tmp_path):
    content = '0 1.0 0.0 0.0\n'
    filepath = tmp_path / 'bad.txt'
    filepath.write_text(content)
    with pytest.raises(SystemExit):
        read_transforms(filepath, logger)


def test_read_transforms_singular_matrix_exits(tmp_path):
    # All-zero matrix → det=0 → invalid → SystemExit.
    content = '0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
    filepath = tmp_path / 'singular.txt'
    filepath.write_text(content)
    with pytest.raises(SystemExit):
        read_transforms(filepath, logger)


def test_read_transforms_non_consecutive_frames_warns(tmp_path, caplog):
    # Frames 0 and 2 (frame 1 missing) → warning logged.
    content = '0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n'
    filepath = tmp_path / 'gap.txt'
    filepath.write_text(content)
    with caplog.at_level(logging.WARNING):
        result = read_transforms(filepath, logger)
    assert set(result.keys()) == {0, 2}
    assert any('Missing frame' in r.message for r in caplog.records)
