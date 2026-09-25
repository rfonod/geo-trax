# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the GIS export subcommand (export.py)."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geotrax.export import (
    build_lines,
    export_trajectories,
    find_input_csvs,
    is_trajectory_csv,
    output_path_for,
    parse_cli_args,
)

logger = logging.getLogger(__name__)


def _per_video_df():
    """Two vehicles with shuffled rows, a single-point vehicle, a placeholder timestamp, and blank speeds."""
    rows = [
        # Vehicle_ID, Timestamp, Frame_Number, Lat, Lon, Local_X, Local_Y, Class, Speed, Length, Width
        (1, '0000-00-00 00:00:00.000', 0, 37.0000, 126.0000, 100.0, 200.0, 0, None, 4.5, 1.9),
        (1, '2022-10-07 17:52:13.533', 1, 37.0001, 126.0001, 101.0, 201.0, 0, 30.0, 4.6, 1.9),
        (2, '2022-10-07 17:52:13.500', 0, 37.1000, 126.1000, 150.0, 250.0, 2, None, 9.0, 2.5),
        (1, '2022-10-07 17:52:13.567', 2, 37.0002, 126.0002, 102.0, 202.0, 0, 40.0, 4.7, 2.0),
        (2, '2022-10-07 17:52:13.533', 1, 37.1001, 126.1001, 151.0, 251.0, 2, 20.0, 9.2, 2.5),
        (3, '2022-10-07 17:52:13.500', 0, 37.2000, 126.2000, 170.0, 270.0, 1, None, 11.0, 2.6),
    ]
    cols = ['Vehicle_ID', 'Timestamp', 'Frame_Number', 'Latitude', 'Longitude', 'Local_X', 'Local_Y',
            'Vehicle_Class', 'Vehicle_Speed', 'Vehicle_Length', 'Vehicle_Width']
    return pd.DataFrame(rows, columns=cols)


def _aggregated_df():
    rows = [
        (10, '17:52:13.567', 'D1', 37.0002, 126.0002, 0, 40.0),
        (10, '17:52:13.500', 'D1', 37.0000, 126.0000, 0, 30.0),
        (11, '17:52:13.500', 'D2', 37.3000, 126.3000, 3, 25.0),
        (11, '17:52:13.533', 'D2', 37.3001, 126.3001, 3, 26.0),
    ]
    cols = ['Vehicle_ID', 'Local_Time', 'Drone_ID', 'Latitude', 'Longitude', 'Vehicle_Class', 'Vehicle_Speed']
    return pd.DataFrame(rows, columns=cols)


def _write_flight_log(path):
    pd.DataFrame({'frame': [0, 1], 'timestamp': ['a', 'b'], 'latitude': [37.0, 37.0], 'longitude': [126.0, 126.0]}).to_csv(path, index=False)


def _run(argv):
    args = parse_cli_args([str(a) for a in argv])
    export_trajectories(args, logger)
    return args


def test_build_lines_orders_vertices_and_summarizes(tmp_path):
    gdf = build_lines(_per_video_df(), 'wgs84', 'EPSG:4326', tmp_path / 'x.csv', logger)
    assert list(gdf['Vehicle_ID']) == [1, 2]  # vehicle 3 has a single point
    line = gdf.set_index('Vehicle_ID').loc[1, 'geometry']
    np.testing.assert_allclose(np.asarray(line.coords), [[126.0, 37.0], [126.0001, 37.0001], [126.0002, 37.0002]])
    row = gdf.set_index('Vehicle_ID').loc[1]
    assert row['Num_Points'] == 3
    assert row['Start_Time'] == '2022-10-07 17:52:13.533'  # the placeholder counts as unknown
    assert row['End_Time'] == '2022-10-07 17:52:13.567'
    assert (row['Start_Frame'], row['End_Frame']) == (0, 2)
    assert row['Mean_Speed'] == pytest.approx(35.0)
    assert row['Max_Speed'] == pytest.approx(40.0)
    assert row['Vehicle_Length'] == pytest.approx(4.6)
    assert gdf.set_index('Vehicle_ID').loc[2, 'Vehicle_Class'] == 2


def test_build_lines_aggregated_dataset_orders_by_local_time(tmp_path):
    gdf = build_lines(_aggregated_df(), 'wgs84', 'EPSG:4326', tmp_path / 'x.csv', logger)
    row = gdf.set_index('Vehicle_ID').loc[10]
    np.testing.assert_allclose(np.asarray(row['geometry'].coords), [[126.0, 37.0], [126.0002, 37.0002]])
    assert row['Drone_ID'] == 'D1'
    assert row['Start_Time'] == '17:52:13.500'
    assert 'Start_Frame' not in gdf.columns


@pytest.mark.parametrize('fmt, ext', [('gpkg', '.gpkg'), ('geojson', '.geojson')])
def test_export_lines_roundtrip(tmp_path, fmt, ext):
    csv = tmp_path / 'video.csv'
    _per_video_df().to_csv(csv, index=False)
    _run([csv, '--format', fmt])
    out = tmp_path / f'video_lines{ext}'
    gdf = gpd.read_file(out)
    assert len(gdf) == 2
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf.geom_type) == {'LineString'}
    assert not list(tmp_path.glob('.*'))
    layers = gpd.list_layers(out)
    assert list(layers['name']) == ['video_lines']  # never the temporary file's name


def test_export_points_keeps_every_row_and_column(tmp_path):
    csv = tmp_path / 'video.csv'
    df = _per_video_df()
    df.to_csv(csv, index=False)
    _run([csv, '--geometry', 'points'])
    gdf = gpd.read_file(tmp_path / 'video_points.gpkg')
    assert len(gdf) == len(df)
    assert set(df.columns).issubset(gdf.columns)
    assert set(gdf.geom_type) == {'Point'}


def test_export_local_crs_uses_target_crs_and_suffix(tmp_path):
    csv = tmp_path / 'video.csv'
    _per_video_df().to_csv(csv, index=False)
    _run([csv, '--crs', 'local', '--set', 'target_crs=epsg:5186'])
    gdf = gpd.read_file(tmp_path / 'video_lines_local.gpkg')
    assert gdf.crs.to_epsg() == 5186
    line = gdf.set_index('Vehicle_ID').loc[1, 'geometry']
    np.testing.assert_allclose(np.asarray(line.coords)[0], [100.0, 200.0])


def test_rows_without_coordinates_are_dropped(tmp_path, caplog):
    csv = tmp_path / 'video.csv'
    df = _per_video_df()
    df.loc[1, 'Latitude'] = np.nan
    df.to_csv(csv, index=False)
    with caplog.at_level(logging.WARNING):
        _run([csv, '--geometry', 'points'])
    assert len(gpd.read_file(tmp_path / 'video_points.gpkg')) == len(df) - 1
    assert any('Dropped 1 row' in r.message for r in caplog.records)


def test_folder_input_skips_flight_logs_and_mirrors_layout(tmp_path):
    src = tmp_path / 'PROCESSED'
    (src / 'D1' / 'results').mkdir(parents=True)
    (src / 'D2' / 'results').mkdir(parents=True)
    _per_video_df().to_csv(src / 'D1' / 'results' / 'video.csv', index=False)
    _per_video_df().to_csv(src / 'D2' / 'results' / 'video.csv', index=False)
    _write_flight_log(src / 'D1' / 'video.csv')
    out = tmp_path / 'GIS'
    _run([src, '-of', out])
    assert (out / 'D1' / 'results' / 'video_lines.gpkg').exists()
    assert (out / 'D2' / 'results' / 'video_lines.gpkg').exists()
    assert not (out / 'D1' / 'video_lines.gpkg').exists()


def test_video_input_resolves_the_georeferenced_csv_not_the_flight_log(tmp_path):
    video = tmp_path / 'video.mp4'
    video.touch()
    _write_flight_log(tmp_path / 'video.csv')
    (tmp_path / 'results').mkdir()
    _per_video_df().to_csv(tmp_path / 'results' / 'video.csv', index=False)
    args = parse_cli_args([str(video)])
    assert find_input_csvs(video, 'wgs84', {}, logger) == [tmp_path / 'results' / 'video.csv']
    assert output_path_for(tmp_path / 'results' / 'video.csv', args) == tmp_path / 'results' / 'video_lines.gpkg'


def test_is_trajectory_csv(tmp_path):
    good, log = tmp_path / 'good.csv', tmp_path / 'log.csv'
    _aggregated_df().to_csv(good, index=False)
    _write_flight_log(log)
    assert is_trajectory_csv(good, 'wgs84')
    assert not is_trajectory_csv(good, 'local')  # aggregated sample has no Local_X/Local_Y
    assert not is_trajectory_csv(log, 'wgs84')


def test_missing_input_exits(tmp_path):
    with pytest.raises(SystemExit):
        _run([tmp_path / 'nope.csv'])


def test_non_trajectory_csv_exits(tmp_path):
    log = tmp_path / 'log.csv'
    _write_flight_log(log)
    with pytest.raises(SystemExit):
        _run([log])


def test_single_point_vehicles_only_is_skipped_not_failed(tmp_path):
    csv = tmp_path / 'video.csv'
    _per_video_df().query('Vehicle_ID == 3').to_csv(csv, index=False)
    _run([csv])
    assert not (tmp_path / 'video_lines.gpkg').exists()
