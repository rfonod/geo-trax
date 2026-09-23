# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the aggregation helpers shared with tools/find_source_id.py and tools/check_dataset.py."""

import logging

import pandas as pd

from geotrax.aggregate import group_result_files, load_source_rows, trace_dataset_vehicle

logger = logging.getLogger(__name__)


def _result_rows(vehicle_ids, timestamp='2022-10-07 08:00:00.000', with_timestamp=True):
    """Build a minimal georeferenced CSV frame with one row per vehicle ID."""
    n = len(vehicle_ids)
    df = pd.DataFrame({
        'Vehicle_ID': vehicle_ids,
        'Ortho_X': range(n), 'Ortho_Y': range(n), 'Local_X': [float(v) for v in vehicle_ids], 'Local_Y': 0.0,
        'Latitude': 37.0, 'Longitude': 126.0, 'Vehicle_Length': 4.5, 'Vehicle_Width': 1.8,
        'Vehicle_Class': 0, 'Vehicle_Speed': 30.0, 'Vehicle_Acceleration': 0.0,
        'Road_Section': 1, 'Lane_Number': 2, 'Visibility': 1,
    })
    if with_timestamp:
        df.insert(1, 'Timestamp', timestamp)
    return df


def _write(tmp_path, drone, name, df):
    path = tmp_path / '2022-10-07' / drone / 'AM1' / 'results' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_load_source_rows_skips_header_only_and_timestampless(tmp_path):
    empty = _write(tmp_path, 'D1', 'A1.csv', _result_rows([]))
    no_ts = _write(tmp_path, 'D2', 'A1.csv', _result_rows([1, 2], with_timestamp=False))
    assert load_source_rows(empty, 'D1', logger) is None
    assert load_source_rows(no_ts, 'D2', logger) is None


def test_load_source_rows_drops_only_undefined_timestamps(tmp_path):
    df = _result_rows([1, 2, 3])
    df.loc[1, 'Timestamp'] = '0000-00-00 00:00:00.000'
    rows = load_source_rows(_write(tmp_path, 'D1', 'A1.csv', df), 'D1', logger)
    assert rows['Vehicle_ID'].tolist() == [1, 3]
    assert (rows['Drone_ID'] == 1).all()
    assert rows['Local_Time'].tolist() == ['08:00:00.000'] * 2


def test_group_result_files_orders_by_drone_number_and_skips_bad_drone_folders(tmp_path):
    files = [
        _write(tmp_path, 'D10', 'A1.csv', _result_rows([1])),
        _write(tmp_path, 'D2', 'A1.csv', _result_rows([1])),
        _write(tmp_path, 'drone3', 'A1.csv', _result_rows([1])),
    ]
    groups = group_result_files(files, 'results', logger)
    assert list(groups) == [('2022-10-07', 'A', 'AM1')]
    assert [drone for _, drone in groups[('2022-10-07', 'A', 'AM1')]] == ['D2', 'D10']


def test_trace_dataset_vehicle_replays_skipped_files(tmp_path):
    """A header-only and a timestamp-less file before the source must not shift the offsets."""
    _write(tmp_path, 'D1', 'A0.csv', _result_rows([]))
    _write(tmp_path, 'D1', 'A1.csv', _result_rows([1, 2, 5]))
    _write(tmp_path, 'D2', 'A0.csv', _result_rows([1, 2, 3, 4, 5, 6, 7], with_timestamp=False))
    source = _write(tmp_path, 'D2', 'A1.csv', _result_rows([1, 3]))

    assert trace_dataset_vehicle(tmp_path, '2022-10-07_A_AM1', 8, 'results', logger) == (source, 3)
    assert trace_dataset_vehicle(tmp_path, '2022-10-07_A_AM1', 5, 'results', logger)[1] == 5
    assert trace_dataset_vehicle(tmp_path, '2022-10-07_A_AM1', 7, 'results', logger) == (None, None)
