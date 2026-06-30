# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for visualization helpers."""

import logging

import numpy as np
import pandas as pd
import pytest

from geotrax.visualize import (
    _clip_poly_to_rect,
    _clip_segment_to_rect,
    _smooth_clip_dims,
    compute_headings,
    normalize_viz_modes,
    read_tracks,
    read_tracks_oriented,
    read_transforms,
)

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


# --- read_tracks_oriented ----------------------------------------------------

def _make_oriented_tracks(n=10, length_val=20.0, width_val=8.0, nan_dims=False):
    """Build a minimal 14-column tracks DataFrame for read_tracks_oriented tests."""
    df = pd.DataFrame(np.zeros((n, 14)))
    df[0] = np.arange(n)           # frame ids
    df[1] = 1                       # vehicle id
    df[2] = 100.0                   # unstab center x
    df[3] = 100.0                   # unstab center y
    df[4] = 18.0                    # unstab bbox width  (raw fallback length = max(w,h))
    df[5] = 10.0                    # unstab bbox height (raw fallback width  = min(w,h))
    df[6] = np.arange(n) * 5.0     # stab center x (moving right so headings are reliable)
    df[7] = 100.0                   # stab center y
    df[8] = 18.0                    # stab bbox width
    df[9] = 10.0                    # stab bbox height
    df[10] = 0                      # class id (car)
    df[11] = 0.9                    # confidence
    df[12] = np.nan if nan_dims else length_val  # estimated length
    df[13] = np.nan if nan_dims else width_val   # estimated width
    return df


def _make_oriented_args():
    import argparse
    args = argparse.Namespace(heading_smoothing=3, heading_min_speed=0.5)
    return args


def test_read_tracks_oriented_estimated_dims():
    tracks = _make_oriented_tracks(nan_dims=False, length_val=20.0, width_val=8.0)
    args = _make_oriented_args()
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    oriented, _ = read_tracks_oriented(tracks, None, class_names, args, logger)
    assert oriented.shape[1] == 13, "oriented layout should have 13 columns"
    assert not oriented[9].any(), "col 9 (is_fallback) should be all False when dims are estimated"
    np.testing.assert_allclose(oriented[4].to_numpy(), 20.0, err_msg="col 4 should use estimated length")
    np.testing.assert_allclose(oriented[5].to_numpy(), 8.0, err_msg="col 5 should use estimated width")
    np.testing.assert_allclose(oriented[10].to_numpy(), 18.0, err_msg="col 10 should carry stab bbox width")
    np.testing.assert_allclose(oriented[11].to_numpy(), 10.0, err_msg="col 11 should carry stab bbox height")
    # Detections sit at x=100 well inside the frame, and args has no source -> on_border all False.
    assert not oriented[12].any(), "col 12 (on_border) should be all False for interior detections"


def test_read_tracks_oriented_fallback_dims():
    tracks = _make_oriented_tracks(nan_dims=True)
    args = _make_oriented_args()
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    oriented, _ = read_tracks_oriented(tracks, None, class_names, args, logger)
    assert oriented[9].all(), "col 9 (is_fallback) should be all True when dims are NaN"
    # Constant raw bbox -> Q25 of constant = that constant: max(18, 10) = 18, min = 10.
    np.testing.assert_allclose(oriented[4].to_numpy(), 18.0, err_msg="col 4 should be Q25 of track-level lengths")
    np.testing.assert_allclose(oriented[5].to_numpy(), 10.0, err_msg="col 5 should be Q25 of track-level widths")


def test_read_tracks_oriented_flags_border_detections(monkeypatch):
    # Frame is 100x100; a detection whose raw bbox (center 100,100, 18x10) spills past the right/bottom
    # edge must be flagged on_border, while an interior one must not.
    monkeypatch.setattr('geotrax.visualize.get_video_dimensions', lambda _: (100, 100))
    tracks = _make_oriented_tracks(n=2, nan_dims=False)
    tracks[2] = [50.0, 100.0]   # raw center x: interior vs. right edge
    tracks[3] = [50.0, 100.0]   # raw center y: interior vs. bottom edge
    args = _make_oriented_args()
    args.source = 'dummy.mp4'   # value irrelevant; get_video_dimensions is patched
    args.edge_clip_margin = 4
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    oriented, _ = read_tracks_oriented(tracks, None, class_names, args, logger)
    np.testing.assert_array_equal(oriented[12].to_numpy(), [False, True])


# --- clip helpers ------------------------------------------------------------

def test_clip_poly_to_rect_noop_when_inside():
    # A box fully inside the rectangle is returned unchanged (same vertex set).
    box = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=float)
    clipped = _clip_poly_to_rect(box, 0, 0, 10, 10)
    assert {tuple(p) for p in np.round(clipped)} == {tuple(p) for p in box}


def test_clip_poly_to_rect_trims_overhang():
    # A box straddling the right edge is trimmed to x <= 10; no vertex exceeds the bound.
    box = np.array([[5, 2], [15, 2], [15, 8], [5, 8]], dtype=float)
    clipped = _clip_poly_to_rect(box, 0, 0, 10, 10)
    assert len(clipped) >= 3
    assert clipped[:, 0].max() <= 10 + 1e-6
    assert clipped[:, 0].min() >= 5 - 1e-6


def test_clip_poly_to_rect_empty_when_outside():
    box = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=float)
    assert len(_clip_poly_to_rect(box, 0, 0, 10, 10)) == 0


def test_clip_segment_to_rect_trims_to_bound():
    q = _clip_segment_to_rect(np.array([5.0, 5.0]), np.array([15.0, 5.0]), 0, 0, 10, 10)
    assert q is not None
    q0, q1 = q
    np.testing.assert_allclose(q0, [5.0, 5.0])
    np.testing.assert_allclose(q1, [10.0, 5.0])


def test_clip_segment_to_rect_none_when_outside():
    assert _clip_segment_to_rect(np.array([20.0, 20.0]), np.array([30.0, 20.0]), 0, 0, 10, 10) is None


# --- clip-dimension smoothing ------------------------------------------------

def _oriented_with_clip_dims(clip_w, clip_h):
    n = len(clip_w)
    df = pd.DataFrame(0, index=range(n), columns=range(13), dtype=float)
    df[0] = np.arange(n)   # frame ids
    df[1] = 1               # single vehicle id
    df[10] = clip_w
    df[11] = clip_h
    return df


def test_smooth_clip_dims_constant_unchanged():
    df = _oriented_with_clip_dims(np.full(20, 18.0), np.full(20, 10.0))
    out = _smooth_clip_dims(df, smoothing=5)
    np.testing.assert_allclose(out[10].to_numpy(), 18.0)
    np.testing.assert_allclose(out[11].to_numpy(), 10.0)


def test_smooth_clip_dims_reduces_jitter():
    # A steady shrink with alternating +/- noise: smoothing must cut the frame-to-frame variation.
    base = np.linspace(40.0, 10.0, 30)
    noisy = base + np.where(np.arange(30) % 2 == 0, 4.0, -4.0)
    df = _oriented_with_clip_dims(noisy, np.full(30, 10.0))
    out = _smooth_clip_dims(df, smoothing=4)[10].to_numpy()
    assert np.abs(np.diff(out)).mean() < np.abs(np.diff(noisy)).mean()
    # The underlying downward trend is preserved (still shrinks overall).
    assert out[0] > out[-1]


def test_read_tracks_oriented_fallback_dims_q25():
    # 4 frames, varying raw bbox widths (h=5 < all w values, so raw_l=w, raw_w=5).
    # Verifies that the fallback uses per-vehicle Q25 aggregation, not per-row bbox values.
    tracks = _make_oriented_tracks(n=4, nan_dims=True)
    tracks[4] = [10.0, 20.0, 30.0, 40.0]   # unstab w varies (simulates bbox inflation in turns)
    tracks[5] = 5.0                          # unstab h constant, always < w

    args = _make_oriented_args()
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    oriented, _ = read_tracks_oriented(tracks, None, class_names, args, logger)

    expected_length = np.percentile([10.0, 20.0, 30.0, 40.0], 25)  # 17.5
    expected_width = np.percentile([5.0, 5.0, 5.0, 5.0], 25)        # 5.0
    # All rows of the same vehicle should get the same Q25 value (not the per-row bbox value).
    np.testing.assert_allclose(oriented[4].to_numpy(), expected_length,
                                err_msg="col 4 should be per-vehicle Q25 length, not per-row bbox")
    np.testing.assert_allclose(oriented[5].to_numpy(), expected_width,
                                err_msg="col 5 should be per-vehicle Q25 width, not per-row bbox")


def test_read_tracks_oriented_interpolated_rows_are_dashed():
    # 15-col input: rows 1 and 3 are interpolated (col 14 = 1), rows 0 and 2 are detected.
    # Dims are non-NaN (is_fallback = False), so col 9 being True proves the OR with is_interpolated.
    tracks = _make_oriented_tracks(n=4, nan_dims=False)
    tracks[14] = [0, 1, 0, 1]   # add is_interpolated column

    args = _make_oriented_args()
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    oriented, _ = read_tracks_oriented(tracks, None, class_names, args, logger)

    np.testing.assert_array_equal(
        oriented[9].to_numpy(), [False, True, False, True],
        err_msg="col 9 should be True for interpolated rows even when dims are estimated (not NaN)"
    )


def test_read_tracks_standard_carries_interp_flag(tmp_path):
    import argparse
    # 15-col space-separated file (stab enabled + interp): vehicle 1, frames 0-2; frame 1 is interpolated.
    lines = [
        "0 1 100 100 20 10 100 100 20 10 0 0.9 15.0 8.0 0",
        "1 1 101 100 20 10 101 100 20 10 0 0.9 15.0 8.0 1",
        "2 1 102 100 20 10 102 100 20 10 0 0.9 15.0 8.0 0",
    ]
    filepath = tmp_path / "tracks.txt"
    filepath.write_text("\n".join(lines) + "\n")

    args = argparse.Namespace(viz_mode=0, plot_trajectories=False, source=None)
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    tracks, _ = read_tracks(filepath, class_names, args, logger)

    assert tracks.shape[1] == 9, "15-col stab+interp input should produce 9-col output (8 standard + is_interpolated)"
    np.testing.assert_array_equal(tracks[8].tolist(), [0, 1, 0],
                                   err_msg="col 8 should carry the is_interpolated flag")


def test_read_tracks_no_stab_interp_flag_at_col_10(tmp_path):
    import argparse
    # 11-col space-separated file (no-stab + interp): frame, id, x, y, w, h, class, conf, len, wid, is_interp.
    # Frame 1 is interpolated. is_interpolated must land at col 10, NOT corrupt col 6 (class).
    lines = [
        "0 1 100 100 20 10 0 0.9 15.0 8.0 0",
        "1 1 101 100 20 10 0 0.9 15.0 8.0 1",
        "2 1 102 100 20 10 0 0.9 15.0 8.0 0",
    ]
    filepath = tmp_path / "tracks_nostab.txt"
    filepath.write_text("\n".join(lines) + "\n")

    args = argparse.Namespace(viz_mode=0, plot_trajectories=False, source=None)
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}
    tracks, _ = read_tracks(filepath, class_names, args, logger)

    assert tracks.shape[1] == 11, "11-col no-stab+interp input should produce 11-col output (10 no-stab + is_interpolated)"
    # col 6 must be class_id (0), not is_interpolated
    np.testing.assert_array_equal(tracks[6].tolist(), [0, 0, 0],
                                   err_msg="col 6 should still be class_id (0), not is_interpolated")
    np.testing.assert_array_equal(tracks[10].tolist(), [0, 1, 0],
                                   err_msg="col 10 should carry the is_interpolated flag")
