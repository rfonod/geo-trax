# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure coordinate-transform and kinematics helpers in georeference.py."""

import numpy as np
import pytest

from geotrax.georeference import (
    apply_filter,
    apply_homography,
    calculate_visibility,
    compute_acceleration,
    compute_speed,
    interpolate_missing_points,
    ortho2geo,
    ortho2local,
)


def test_apply_homography_identity():
    x = np.array([0.0, 1.0, 5.0])
    y = np.array([0.0, 2.0, 5.0])
    out_x, out_y = apply_homography(x, y, np.eye(3))
    np.testing.assert_allclose(out_x, x)
    np.testing.assert_allclose(out_y, y)


def test_apply_homography_translation():
    homography = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=float)
    out_x, out_y = apply_homography(np.array([0.0, 1.0]), np.array([0.0, 1.0]), homography)
    np.testing.assert_allclose(out_x, [5.0, 6.0])
    np.testing.assert_allclose(out_y, [3.0, 4.0])


def test_ortho2geo_affine():
    # params: (lng0, lat0, dlng, dlat, skew_x, skew_y)
    params = (100.0, 50.0, 2.0, 3.0, 0.0, 0.0)
    lat, lng = ortho2geo(np.array([1.0, 2.0]), np.array([4.0, 5.0]), params)
    np.testing.assert_allclose(lng, [102.0, 104.0])
    np.testing.assert_allclose(lat, [62.0, 65.0])


def test_ortho2local_identity_crs_returns_lng_lat():
    # With identical source/target CRS, local coords equal (longitude, latitude).
    params = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    x_local, y_local = ortho2local(
        np.array([10.0]), np.array([20.0]), params, 'EPSG:4326', 'EPSG:4326'
    )
    np.testing.assert_allclose(x_local, [10.0])
    np.testing.assert_allclose(y_local, [20.0])


def test_compute_speed():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(compute_speed(x, y, fps=1.0), [1.0, 1.0])
    np.testing.assert_allclose(compute_speed(x, y, fps=2.0), [2.0, 2.0])


def test_compute_acceleration():
    speed = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(compute_acceleration(speed, fps=1.0), [1.0, 2.0])


def test_apply_filter_gaussian_preserves_constant():
    data = np.full(10, 3.0)
    np.testing.assert_allclose(apply_filter(data, kernel_size=2, filter_type='gaussian'), 3.0)


def test_apply_filter_savgol_preserves_linear_interior():
    # A quadratic-order Savitzky-Golay filter reproduces linear data exactly in the
    # interior; only the edges differ due to 'nearest' boundary handling.
    data = np.arange(10.0)
    filtered = apply_filter(data, kernel_size=5, filter_type='savgol')
    np.testing.assert_allclose(filtered[2:-2], data[2:-2], atol=1e-9)


def test_apply_filter_savgol_forces_odd_window():
    # Even kernel sizes are bumped to the next odd window without error.
    out = apply_filter(np.arange(8.0), kernel_size=4, filter_type='savgol')
    assert out.shape == (8,)


def test_apply_filter_invalid_type_raises():
    with pytest.raises(ValueError):
        apply_filter(np.zeros(5), kernel_size=3, filter_type='nope')


def test_interpolate_missing_points_no_gaps():
    frames = np.array([0, 1, 2])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    xi, yi, present = interpolate_missing_points(frames, x, y)
    np.testing.assert_allclose(xi, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(present, [0, 1, 2])


def test_interpolate_missing_points_fills_gap():
    frames = np.array([0, 2])
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 4.0])
    xi, yi, present = interpolate_missing_points(frames, x, y)
    # One synthetic midpoint inserted at frame 1.
    np.testing.assert_allclose(xi, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(yi, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(present, [0, 2])  # original (present) samples


def test_calculate_visibility_margins():
    track_ids = np.array([1, 2])
    # frame_size is (height, width)
    bbox = np.array([[50.0, 50.0, 10.0, 10.0],   # well inside -> visible
                     [2.0, 2.0, 10.0, 10.0]])     # straddles the edge -> not visible
    visibility = calculate_visibility(track_ids, bbox, frame_size=(100, 100), visibility_margin=4)
    np.testing.assert_array_equal(visibility, [True, False])
