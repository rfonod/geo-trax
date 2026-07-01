# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure coordinate-transform and kinematics helpers in georeference.py."""

import logging

import numpy as np
import pandas as pd
import pytest

from geotrax.georeference import (
    apply_filter,
    apply_homography,
    calculate_visibility,
    compute_acceleration,
    compute_hash,
    compute_kinematics,
    compute_speed,
    create_and_format_georeferenced_df,
    create_polygon,
    interpolate_missing_points,
    ortho2geo,
    ortho2local,
    read_ortho_config_file,
)

logger = logging.getLogger(__name__)


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


def test_ortho2local_reprojects_to_utm():
    # Exercises the real geographic->projected reprojection (not an identity CRS).
    # With identity params, ortho_x->longitude and ortho_y->latitude; reprojecting
    # (lon=6.6, lat=46.5) into UTM 31N (EPSG:32631) yields a known easting/northing.
    params = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    x_local, y_local = ortho2local(
        np.array([6.6]), np.array([46.5]), params, 'EPSG:4326', 'EPSG:32631'
    )
    np.testing.assert_allclose(x_local, [776225.4478], atol=1e-2)
    np.testing.assert_allclose(y_local, [5155902.1301], atol=1e-2)


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


# --- read_ortho_config_file --------------------------------------------------

def test_read_ortho_config_file_parses_values(tmp_path):
    content = '# origin params\n10.5 20.3\n15.0 25.1\n'
    filepath = tmp_path / 'A.txt'
    filepath.write_text(content)
    result = read_ortho_config_file(filepath)
    np.testing.assert_allclose(result, [[10.5, 20.3], [15.0, 25.1]])


def test_read_ortho_config_file_skips_blank_lines(tmp_path):
    content = '1.0 2.0\n\n3.0 4.0\n'
    filepath = tmp_path / 'B.txt'
    filepath.write_text(content)
    result = read_ortho_config_file(filepath)
    assert result.shape == (2, 2)


# --- create_polygon ----------------------------------------------------------

def test_create_polygon_area():
    row = pd.Series({
        'tlx': 0, 'tly': 0,
        'blx': 0, 'bly': 100,
        'brx': 100, 'bry': 100,
        'trx': 100, 'try': 0,
    })
    poly = create_polygon(row)
    assert poly.is_valid
    assert poly.area == pytest.approx(10000.0)


# --- compute_hash ------------------------------------------------------------

def test_compute_hash_is_deterministic():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert compute_hash(img) == compute_hash(img)


def test_compute_hash_differs_for_different_images():
    img_a = np.zeros((10, 10, 3), dtype=np.uint8)
    img_b = np.ones((10, 10, 3), dtype=np.uint8)
    assert compute_hash(img_a) != compute_hash(img_b)


def test_compute_hash_is_md5_hex():
    img = np.zeros((4, 4), dtype=np.uint8)
    h = compute_hash(img)
    assert isinstance(h, str) and len(h) == 32


# --- compute_kinematics ------------------------------------------------------

def test_compute_kinematics_single_track_constant_speed():
    # Vehicle moves 1 m/frame on the x-axis for 5 consecutive frames.
    track_ids = np.array([1, 1, 1, 1, 1])
    frame_num = np.array([0, 1, 2, 3, 4])
    x_local = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_local = np.zeros(5)
    visibility = np.ones(5, dtype=bool)
    speed, acc = compute_kinematics(
        track_ids, frame_num, x_local, y_local, visibility,
        fps=1.0, filter_type='gaussian', kernel_size=1,
    )
    assert np.isnan(speed[0])
    np.testing.assert_allclose(speed[1:], 3.6, rtol=1e-4)  # 1 m/s * 3.6 = 3.6 km/h
    assert np.isnan(acc[0]) and np.isnan(acc[1])
    np.testing.assert_allclose(acc[2:], 0.0, atol=1e-6)


def test_compute_kinematics_insufficient_visible_points_returns_nan():
    # Only 2 visible detections → threshold of 3 not met → all NaN.
    track_ids = np.array([2, 2])
    frame_num = np.array([0, 1])
    x_local = np.array([0.0, 1.0])
    y_local = np.zeros(2)
    visibility = np.ones(2, dtype=bool)
    speed, acc = compute_kinematics(
        track_ids, frame_num, x_local, y_local, visibility,
        fps=1.0, filter_type='gaussian', kernel_size=1,
    )
    assert np.all(np.isnan(speed))
    assert np.all(np.isnan(acc))


def test_compute_kinematics_interpolated_rows_excluded():
    # Real frames at indices 0, 2, 4, 6, 8 (every other); odd indices are synthetic.
    # Kinematics must be NaN for interpolated rows and correct for real ones.
    track_ids = np.ones(9, dtype=int)
    frame_num = np.arange(9)
    x_local = np.arange(9, dtype=float)
    y_local = np.zeros(9)
    visibility = np.ones(9, dtype=bool)
    is_interp = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

    speed, acc = compute_kinematics(
        track_ids, frame_num, x_local, y_local, visibility,
        fps=1.0, filter_type='gaussian', kernel_size=1,
        is_interpolated=is_interp,
    )
    # Interpolated rows must remain NaN
    assert np.all(np.isnan(speed[[1, 3, 5, 7]]))
    assert np.all(np.isnan(acc[[1, 3, 5, 7]]))
    # First real frame always NaN; remaining real frames have computed speed
    assert np.isnan(speed[0])
    np.testing.assert_allclose(speed[[2, 4, 6, 8]], 3.6, rtol=1e-4)


# --- create_and_format_georeferenced_df --------------------------------------

def _make_georef_inputs(n_veh=3, n_frames=4):
    total = n_veh * n_frames
    return dict(
        track_id=np.repeat(np.arange(1, n_veh + 1), n_frames),
        timestamps=np.array([]),
        frame_num=np.tile(np.arange(n_frames), n_veh),
        x_stab_ortho=np.full(total, 123.456),
        y_stab_ortho=np.full(total, 234.567),
        x_local=np.full(total, 1000.123),
        y_local=np.full(total, 2000.456),
        latitude=np.full(total, 37.123456789),
        longitude=np.full(total, 127.987654321),
        veh_dim_real=(np.full(total, 4.567), np.full(total, 1.876)),
        class_id=np.zeros(total, dtype=int),
        v_speed=np.full(total, 50.0),
        v_acceleration=np.zeros(total),
        road_section=None,
        lane_number=None,
        visibility=np.ones(total, dtype=bool),
    )


def test_create_and_format_georeferenced_df_columns_and_length():
    inputs = _make_georef_inputs()
    df = create_and_format_georeferenced_df(**inputs, min_traj_length=0, logger=logger)
    assert len(df) == 12
    assert 'Timestamp' not in df.columns
    assert 'Road_Section' not in df.columns
    assert 'Lane_Number' not in df.columns
    assert 'Vehicle_ID' in df.columns and 'Visibility' in df.columns


def test_create_and_format_georeferenced_df_rounding():
    inputs = _make_georef_inputs(n_veh=1, n_frames=1)
    df = create_and_format_georeferenced_df(**inputs, min_traj_length=0, logger=logger)
    assert df['Ortho_X'].iloc[0] == pytest.approx(round(123.456, 1))
    assert df['Local_X'].iloc[0] == pytest.approx(round(1000.123, 2))
    assert df['Longitude'].iloc[0] == pytest.approx(round(127.987654321, 7), abs=1e-6)
    assert df['Latitude'].iloc[0] == pytest.approx(round(37.123456789, 7), abs=1e-6)


def test_create_and_format_georeferenced_df_visibility_is_int():
    inputs = _make_georef_inputs(n_veh=1, n_frames=1)
    df = create_and_format_georeferenced_df(**inputs, min_traj_length=0, logger=logger)
    assert df['Visibility'].dtype.kind == 'i'


def test_create_and_format_georeferenced_df_min_traj_length_filters():
    # 4 rows per vehicle but min_traj_length=5 → all three vehicles removed.
    inputs = _make_georef_inputs(n_veh=3, n_frames=4)
    df = create_and_format_georeferenced_df(**inputs, min_traj_length=5, logger=logger)
    assert len(df) == 0
