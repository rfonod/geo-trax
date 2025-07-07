#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_cut_video_issues.py - Flight Log Analysis and Anomaly Detection Tool

This script analyzes flight log data from drone videos to detect anomalies and quality issues.
It extracts flight log data from CSV files, calculates positional and altitude deviations from
reference frame hover locations, and checks frame numbers, timestamps, and camera settings for inconsistencies.

The script performs comprehensive anomaly detection including spatial deviations, temporal irregularities,
and camera parameter variations. It provides visualization capabilities and generates detailed reports
for quality assessment of drone flight data.

Usage:
  python tools/find_cut_video_issues.py <input_folder> [options]

Arguments:
  input_folder : str
                 Path to the folder containing videos and flight logs (i.e., the PROCESSED folder).

Options:
  -h, --help            : Show this help message and exit.
  -o, --output-folder <path> : str, optional
                        Path to the folder where extracted flight log data will be saved
                        (default: same as input folder).
  -s, --save            : bool, optional
                        Save extracted flight log data stats to a CSV file (default: False).
  -f, --force           : bool, optional
                        Force extraction even if output files already exist (default: False).
  -rf, --ref-frame <int> : int, optional
                        Reference frame used for stabilization/georeferencing (default: 0).
  -viz, --visualize     : bool, optional
                        Visualize flight logs and anomalies (default: False).
  -sv, --save-viz       : bool, optional
                        Save visualization to PDF file (default: False).
  -m, --match-pattern <str> : str, optional
                        Pattern to match flight logs (default: '??.CSV').
  -fe, --folders-exclude <str> [<str> ...] : list of str, optional
                        Folders to exclude from search
                        (e.g., 'results' for non-flight log CSV files) (default: ['results']).
  -tcrs, --target-crs <str> : str, optional
                        Target CRS for local coordinates (default: 'epsg:5186').
  -tc, --track-check    : bool, optional
                        Check if all frames are present in tracking results and vice versa (default: False).
  -v, --verbose         : bool, optional
                        Print extracted flight log data (default: False).

Anomaly Detection Thresholds:
  -rdt, --radius-diff-threshold <float> : float, optional
                        Max positional deviation from initial hover in meters (default: 15.0).
  -adt, --altitude-diff-threshold <float> : float, optional
                        Max altitude deviation from initial hover in meters (default: 5.0).
  -fdt, --frame-diff-threshold <int> : int, optional
                        Max frame difference threshold (default: 2).
  -tdt, --timestamp-diff-threshold <float> : float, optional
                        Max timestamp difference in seconds (default: 0.5).
  -idt, --iso-diff-threshold <int> : int, optional
                        Max ISO deviation threshold (default: 300).
  -sdt, --shutter-diff-threshold <float> : float, optional
                        Max shutter speed deviation threshold (default: 0.02).
  -fndt, --fnum-diff-threshold <float> : float, optional
                        Max f-number deviation threshold (default: 0.1).
  -cdt, --ct-diff-threshold <int> : int, optional
                        Max color temperature deviation threshold (default: 2000).
  -fldt, --focal-len-diff-threshold <float> : float, optional
                        Max focal length deviation threshold (default: 0.5).

Examples:
1. Basic analysis with saved results and visualization:
   python tools/find_cut_video_issues.py ../datasets/PROJECT/PROCESSED -s -sv -f

2. Analysis with custom thresholds and tracking check:
   python tools/find_cut_video_issues.py /path/to/PROCESSED --radius-diff-threshold 10 --track-check

3. Visualization only with custom output folder:
   python tools/find_cut_video_issues.py /path/to/PROCESSED -viz -o /path/to/output/

4. Comprehensive analysis with all options:
   python tools/find_cut_video_issues.py /path/to/PROCESSED -s -f -viz -sv -tc -v

Input:
- PROCESSED folder containing drone flight logs in CSV format
- Flight logs should follow the pattern specified by --match-pattern
- Video files and associated flight log CSV files in structured directories

Output:
- flight_log_stats.csv: Comprehensive statistics for all flight logs
- flight_log_anomalies.csv: Detected anomalies based on thresholds
- PDF visualizations: Spatial deviations and camera parameter plots (if --save-viz)
- Console output: Anomaly detection results and statistics

Notes:
- Uses reference frame hover location as baseline for deviation calculations
- Converts geographic coordinates to local coordinate system for spatial analysis
- Validates timestamp consistency and expected time windows for flight sessions
- Checks frame sequence integrity between flight logs and tracking results
- Supports visualization of positional deviations and camera parameter variations
- Color-coded visualizations show altitude deviations and parameter distributions
- Anomaly detection covers spatial, temporal, and camera parameter deviations
- Session time windows are predefined (AM1-AM5, PM1-PM5) with configurable tolerance
"""

import argparse
import fnmatch
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.patches import Circle

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.utils import detect_delimiter, determine_location_id

VIDEO_SUFFIX = '.MP4' # Video file format to report in the output file
SESSION2TIME_WINDOW = {
    'AM1' : ('07:00:00', '07:30:00'),
    'AM2' : ('07:40:00', '08:10:00'),
    'AM3' : ('08:20:00', '08:50:00'),
    'AM4' : ('09:00:00', '09:30:00'),
    'AM5' : ('09:40:00', '10:10:00'),
    'PM1' : ('15:00:00', '15:30:00'),
    'PM2' : ('15:40:00', '16:10:00'),
    'PM3' : ('16:20:00', '16:50:00'),
    'PM4' : ('17:00:00', '17:30:00'),
    'PM5' : ('17:40:00', '18:10:00')
} # Expected time window for each session
WINDOW_TOLERANCE = 30  # +/- seconds

def find_cut_video_issues(args: argparse.Namespace) -> None:
    """
    Find the issues with the cut videos.
    """
    flight_logs_stats_filepath = args.output_folder / 'flight_log_stats.csv'
    if flight_logs_stats_filepath.exists() and not args.force:
        print(f"Flight logs stats already exist in {flight_logs_stats_filepath}. Use --force to overwrite.")
        print(f"Loading the existing flight logs stats from {flight_logs_stats_filepath}.")
        df_flight_logs_stats = pd.read_csv(flight_logs_stats_filepath)
    else:
        flight_logs_filepaths = find_all_flight_logs(args.input_folder, args.match_pattern, args.folders_exclude)
        df_flight_logs_stats = extract_flight_logs_stats(flight_logs_filepaths, args.input_folder, args.ref_frame, args.target_crs, args.timestamp_diff_threshold, args.track_check)
        if args.save:
            save_flight_logs(df_flight_logs_stats, flight_logs_stats_filepath)

    df_flight_logs_anomalies = find_anomalies(df_flight_logs_stats, args)
    if args.save:
        save_flight_logs(df_flight_logs_anomalies, args.output_folder / 'flight_log_anomalies.csv')

    if args.visualize or args.save_viz:
        visualize_flight_logs_data(df_flight_logs_stats, args.output_folder, args.visualize, args.save_viz)


def find_all_flight_logs(input_folder: Path, match_pattern: str, folders_exclude: list) -> list:
    """
    Find all the flight logs in the input folder.
    """
    flight_logs = []
    for item in input_folder.iterdir():
        if item.is_dir() and item.name not in folders_exclude:
            flight_logs.extend(find_all_flight_logs(item, match_pattern, folders_exclude))
        elif item.is_file() and fnmatch.fnmatch(item.name, match_pattern):
            flight_logs.append(item)

    if not flight_logs:
        print(f"Warning: No flight logs found in the input folder {input_folder}.")
    return flight_logs


def extract_flight_logs_stats(flight_logs: list, input_folder: Path, ref_frame: int, target_crs: str, timestamp_diff_threshold: float, track_check: bool) -> pd.DataFrame:
    """
    Extract the flight logs stats.
    """
    data = []
    start_time_deviations = []
    end_time_deviations = []
    for flight_log in tqdm.tqdm(flight_logs, desc='Extracting flight logs'):
        if not flight_log.exists():
            print(f'Flight log {flight_log} could not be opened. Skipping...')
            continue

        delimiter = detect_delimiter(flight_log)
        df = pd.read_csv(flight_log, delimiter=delimiter)
        location_id = determine_location_id(flight_log)
        video_path = flight_log.relative_to(input_folder).with_suffix(VIDEO_SUFFIX)

        try:
            longitude_ref, latitude_ref, rel_altitude_ref = df.loc[df['frame'] == ref_frame, ['longitude', 'latitude', 'rel_alt']].values[0]
            iso_ref, shutter_ref, fnum_ref, ct_ref, focal_len_ref = df.loc[df['frame'] == ref_frame, ['iso', 'shutter', 'fnum', 'ct', 'focal_len']].values[0]
        except IndexError:
            print(f'Warning: Reference frame {ref_frame} not found in {flight_log}. Skipping...')
            continue

        try:
            frame, timestamp = df.loc[:, ['frame', 'timestamp']].values.T
            longitude, latitude, rel_altitude = df.loc[:, ['longitude', 'latitude', 'rel_alt']].values.T
            iso, shutter, fnum, ct, focal_len = df.loc[:, ['iso', 'shutter', 'fnum', 'ct', 'focal_len']].values.T
        except KeyError as e:
            print(f'Warning: {e} not found in {flight_log}. Skipping...')
            continue
        except IndexError:
            print(f'Warning: No data found in {flight_log}. Skipping...')
            continue

        # check if the all frame numbers are present in the tracking results (if tracking has been performed)
        if track_check:
            tracking_results_filepath = flight_log.parent / 'results' / (flight_log.stem + '.txt')
            if tracking_results_filepath.exists():
                tracking_frames = np.loadtxt(tracking_results_filepath, delimiter=detect_delimiter(tracking_results_filepath), usecols=0, dtype=int)
                missing_in_tracking = set(frame) - set(tracking_frames)
                missing_in_flight_log = set(tracking_frames) - set(frame)
                if missing_in_tracking:
                    print(f"\033[91mWarning: Missing frames {sorted(missing_in_tracking)} in the tracking results file {tracking_results_filepath} for flight log {flight_log}.\033[0m")
                if missing_in_flight_log:
                    print(f"\033[91mWarning: Missing frames {sorted(missing_in_flight_log)} in the flight log {flight_log} that are present in the tracking results file {tracking_results_filepath}.\033[0m")

        # time-related data/checks
        frame_diff = np.diff(frame)
        frame_max_abs_diff = np.max(np.abs(frame_diff))

        # check if the timestamps are monotonically increasing and calculate the maximal timestamp difference and location
        timestamp_diff = np.diff(pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S.%f'))
        timestamp_max_abs_diff = np.max(np.abs(timestamp_diff)) / np.timedelta64(1, 's')
        idx = np.argmax(np.abs(timestamp_diff))
        timestamp_anomaly_time = timestamp[idx]
        timestamp_anomaly_frame = int(frame[idx])

        # check if the timestamps in the filename match the timestamps in the flight log
        date_in_filename = str(video_path).split(os.sep)[-4]
        if any(date_in_filename != ts.split(' ')[0] for ts in timestamp):
            print(f"\033[91mWarning: Date mismatch found in {video_path}. The date differs from the video path.\033[0m")

        # check if the timestamps are within the expected time window
        session = str(video_path).split(os.sep)[-2]
        timestamp_start, timestamp_end = SESSION2TIME_WINDOW.get(session, (None, None))
        if timestamp_start and timestamp_end:
            timestamp_start = pd.to_datetime(timestamp_start, format='%H:%M:%S')
            timestamp_end = pd.to_datetime(timestamp_end, format='%H:%M:%S')
            timestamp_start_margin = (timestamp_start - pd.Timedelta(seconds=WINDOW_TOLERANCE)).strftime('%H:%M:%S')
            timestamp_end_margin = (timestamp_end + pd.Timedelta(seconds=WINDOW_TOLERANCE)).strftime('%H:%M:%S')

            if any(not (timestamp_start_margin <= ts.split(' ')[1] <= timestamp_end_margin) for ts in timestamp):
                print(f"\033[91mWarning: Timestamp mismatch found in {video_path}. The timestamps are not within the expected time window.\033[0m")
                print(f"Expected time window: {timestamp_start_margin} - {timestamp_end_margin}. First timestamp: {timestamp[0].split(' ')[1]}. Last timestamp: {timestamp[-1].split(' ')[1]}")

            # collect the max time deviations for each hovering, if exists
            timestamp_start_actual = pd.to_datetime(timestamp[0].split(' ')[1], format='%H:%M:%S.%f')
            if timestamp_start_actual < timestamp_start:
                start_time_deviations.append((timestamp_start - timestamp_start_actual).total_seconds())

            timestamp_end_actual = pd.to_datetime(timestamp[-1].split(' ')[1], format='%H:%M:%S.%f')
            if timestamp_end_actual > timestamp_end:
                end_time_deviations.append((timestamp_end_actual - timestamp_end).total_seconds())

        else:
            print(f"\033[91mWarning: Unknown session {session} found in {video_path}. The timestamps will not be checked.\033")

        # spatial data
        longitude[longitude == 0] = np.nan
        latitude[latitude == 0] = np.nan
        if np.isnan(longitude).any() or np.isnan(latitude).any():
            print(f'Warning: Missing GPS data in {flight_log}. Missing values will be ignored.')

        x_local_ref = geo2local(np.array([latitude_ref]), np.array([longitude_ref]), target_crs=target_crs)[0][0]
        y_local_ref = geo2local(np.array([latitude_ref]), np.array([longitude_ref]), target_crs=target_crs)[1][0]
        x_local_all, y_local_all = geo2local(latitude, longitude, target_crs=target_crs)

        x_deviation = x_local_all - x_local_ref
        y_deviation = y_local_all - y_local_ref
        rel_altitude_deviation = rel_altitude - rel_altitude_ref

        x_max_idx = np.argmax(np.abs(x_deviation))
        y_max_idx = np.argmax(np.abs(y_deviation))
        rel_altitude_max_idx = np.argmax(np.abs(rel_altitude_deviation))

        x_max_deviation = x_deviation[x_max_idx]
        y_max_deviation = y_deviation[y_max_idx]
        r_max_deviation = np.sqrt(x_max_deviation**2 + y_max_deviation**2).round(2)
        x_max_deviation = x_max_deviation.round(2)
        y_max_deviation = y_max_deviation.round(2)
        rel_altitude_max_deviation = rel_altitude_deviation[rel_altitude_max_idx].round(2)

        # camera parameters
        iso_max_deviation = np.max(np.abs(iso - iso_ref))
        shutter_max_deviation = np.max(np.abs(np.array([eval(shutter) for shutter in shutter]) - eval(shutter_ref)))
        fnum_max_deviation = np.max(np.abs(fnum - fnum_ref))
        ct_max_deviation = np.max(np.abs(ct - ct_ref))
        focal_len_max_deviation = np.max(np.abs(focal_len - focal_len_ref))

        data.append([location_id, video_path, r_max_deviation, x_max_deviation, y_max_deviation, rel_altitude_max_deviation,
                     frame_max_abs_diff, timestamp_max_abs_diff, timestamp_anomaly_time, timestamp_anomaly_frame,
                     iso_max_deviation, shutter_max_deviation, fnum_max_deviation, ct_max_deviation, focal_len_max_deviation,
                     longitude_ref, latitude_ref, rel_altitude_ref, x_local_ref, y_local_ref])

    # print the start/end time deviation statistics
    if start_time_deviations:
        print(f"There were {len(start_time_deviations)} hoverings that started before the expected time window.")
        print(f"The mean ± std. dev. of these cases is: {np.mean(start_time_deviations).round(2)} ± {np.std(start_time_deviations).round(2)} seconds.")
    if end_time_deviations:
        print(f"There were {len(end_time_deviations)} hoverings that ended after the expected time window.")
        print(f"The mean ± std. dev. of these cases is: {np.mean(end_time_deviations).round(2)} ± {np.std(end_time_deviations).round(2)} seconds.")

    df = pd.DataFrame(data, columns=['location_id', 'video_path', 'radius_max_deviation', 'x_max_deviation', 'y_max_deviation',
                                     'rel_altitude_max_deviation', 'frame_max_abs_diff', 'timestamp_max_abs_diff', 'timestamp_anomaly_location', 'timestamp_anomaly_frame',
                                     'iso_max_deviation', 'shutter_max_deviation', 'fnum_max_deviation', 'ct_max_deviation', 'focal_len_max_deviation',
                                     'longitude_ref', 'latitude_ref', 'rel_altitude_ref', 'x_local_ref', 'y_local_ref'])
    df.sort_values(by=['location_id', 'video_path'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def find_anomalies(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Find location anomalies in the flight logs.
    """
    anomaly_conditions = {
        'radius': ('radius_max_deviation', args.radius_diff_threshold),
        'altitude': ('rel_altitude_max_deviation', args.altitude_diff_threshold),
        'frame': ('frame_max_abs_diff', args.frame_diff_threshold),
        'timestamp': ('timestamp_max_abs_diff', args.timestamp_diff_threshold),
        'iso': ('iso_max_deviation', args.iso_diff_threshold),
        'shutter': ('shutter_max_deviation', args.shutter_diff_threshold),
        'fnum': ('fnum_max_deviation', args.fnum_diff_threshold),
        'ct': ('ct_max_deviation', args.ct_diff_threshold),
        'focal_len': ('focal_len_max_deviation', args.focal_len_diff_threshold)
    }

    anomalies = []
    for name, (column, threshold) in anomaly_conditions.items():
        condition_anomalies = df.loc[df[column] >= threshold]
        anomalies.append(condition_anomalies)
        print(f"Found {len(condition_anomalies)} {name} anomalies - {column} >= {threshold}.")
        if not condition_anomalies.empty:
            if name == 'timestamp':
                print(condition_anomalies[['location_id', 'video_path', column, 'timestamp_anomaly_location', 'timestamp_anomaly_frame']].to_string(index=False))
            else:
                print(condition_anomalies[['location_id', 'video_path', column]].to_string(index=False))

    return pd.concat(anomalies, ignore_index=True)


def geo2local(latitude: np.ndarray, longitude: np.ndarray, source_crs: str = 'epsg:4326', target_crs: str = 'epsg:5186') -> tuple:
    """
    Convert geographic coordinates to local coordinates.
    """
    geo_coordinates_gdf = gpd.GeoDataFrame({'Latitude': latitude, 'Longitude': longitude},
        geometry=gpd.points_from_xy(longitude, latitude), crs=source_crs) # type: ignore

    local_coordinates_gdf = geo_coordinates_gdf.to_crs(target_crs)
    x_local = local_coordinates_gdf.geometry.x.to_numpy() # type: ignore
    y_local = local_coordinates_gdf.geometry.y.to_numpy() # type: ignore
    return x_local, y_local


def save_flight_logs(df: pd.DataFrame, flight_logs_filepath: Path) -> None:
    """
    Save all the extracted flight log stats to a CSV file.
    """
    flight_logs_filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flight_logs_filepath, index=False)
    print(f"Flight logs stats saved to {flight_logs_filepath}")


def visualize_flight_logs_data(df_all: pd.DataFrame, output_folder: Path, visualize: bool, save: bool) -> None:
    """
    Visualize the flight logs and anomalies.
    """

    location_ids = df_all['location_id'].unique()
    n_location_ids = len(location_ids)
    n_rows = n_location_ids // 5 + 1 if n_location_ids % 5 else n_location_ids // 5
    n_cols = 5 if n_location_ids > 5 else n_location_ids

    # Plot the positional and altitude deviations
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    rel_altitude_max_deviation_all = df_all['rel_altitude_max_deviation']
    global_min = rel_altitude_max_deviation_all.min()
    global_max = rel_altitude_max_deviation_all.max()
    handles, labels = [], []
    for i, location_id in enumerate(location_ids):
        ax = axs[i]
        df_location = df_all.loc[df_all['location_id'] == location_id]
        x_max_deviation = df_location['x_max_deviation']
        y_max_deviation = df_location['y_max_deviation']
        relative_altitude_max_deviation = df_location['rel_altitude_max_deviation']
        radius = df_location['radius_max_deviation'].max()

        h1 = ax.scatter(x_max_deviation, y_max_deviation, c=relative_altitude_max_deviation, marker='x', s=100, alpha=0.8, vmin=global_min, vmax=global_max, cmap='winter')
        circle = Circle((0, 0), radius, color='gray', fill=False, linestyle='--', linewidth=1, label=f'r={radius:.2f} m')
        h2 = ax.add_artist(circle)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'Intersection {location_id} - {len(df_location)} hoverings')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(handles=[h2], labels=[f'radius={radius:.2f} m'], loc='upper right')

    handles.extend([h1])
    labels.extend(['maximal positional deviation from the initial hover'])

    unique_handles_labels = dict(zip(labels, handles))
    fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper center', ncol=4, fontsize='large', bbox_to_anchor=(0.5, 0.92))

    cbar = fig.colorbar(h1, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    cbar.set_label('Maximal altitude deviation from the reference frame (m)')

    for i in range(n_location_ids, n_rows * n_cols):
        fig.delaxes(axs[i])

    if visualize:
        plt.show()
    if save:
        filepath = output_folder / 'flight_logs_hovering_stats_viz.pdf'
        plt.savefig(filepath, transparent=True, bbox_inches='tight')
        print(f"Hovering stats visualization saved to {filepath}")
    plt.close()

    # Plot selected camera settings deviations
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    shutter_max_deviation_all = df_all['shutter_max_deviation']
    global_min = shutter_max_deviation_all.min()
    global_max = shutter_max_deviation_all.max()
    handles, labels = [], []
    for i, location_id in enumerate(location_ids):
        ax = axs[i]
        df_location = df_all.loc[df_all['location_id'] == location_id]
        iso_max_deviation = df_location['iso_max_deviation']
        ct_max_deviation = df_location['ct_max_deviation']
        shutter_max_deviation = df_location['shutter_max_deviation']

        h1 = ax.scatter(ct_max_deviation, iso_max_deviation, c=shutter_max_deviation, s=100, alpha=0.6, vmin=global_min, vmax=global_max, cmap='summer', marker='o', edgecolors='k')
        ax.set_title(f'Intersection {location_id} - {len(df_location)} hoverings')
        ax.set_xlabel('Color temperature deviation')
        ax.set_ylabel('ISO deviation')

    cbar = fig.colorbar(h1, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    cbar.set_label('Maximal shutter speed deviation')

    for i in range(n_location_ids, n_rows * n_cols):
        fig.delaxes(axs[i])

    if visualize:
        plt.show()
    if save:
        filepath = output_folder / 'flight_logs_camera_stats_viz.pdf'
        plt.savefig(filepath, transparent=True, bbox_inches='tight')
        print(f"Visualization saved to {filepath}")
    plt.close()


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Flight Log Analysis and Anomaly Detection Tool')

    # Main arguments
    parser.add_argument('input_folder', type=Path, help="Path to the folder containing the videos and flight logs, i.e., the PROCESSED folder.")
    parser.add_argument("--output-folder", "-o", type=Path, default=None, help="Path to the folder where the extracted flight log data will be saved. (default: same as input folder)")
    parser.add_argument("--save", "-s", action="store_true", help="Save the extracted flight log data stats to a CSV file.")
    parser.add_argument("--force", "-f", action="store_true", help="Force the extraction of flight log data even if the output files already exist.")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="Reference frame used for stabilization/georeferencing.")
    parser.add_argument("--visualize", "-viz", action="store_true", help="Visualize the flight logs and anomalies.")
    parser.add_argument("--save-viz", "-sv", action="store_true", help="Save the visualization to a PDF file.")
    parser.add_argument("--match-pattern", "-m", type=str, default='??.CSV', help="Pattern to match the flight logs.")
    parser.add_argument("--folders-exclude", "-fe", type=str, nargs='+', default=['results'], help="Folders to exclude from the search (e.g, 'results' as these may contain non-flight log .CSV files).")
    parser.add_argument("--target-crs", "-tcrs", default='epsg:5186', help="Target CRS for local coordinates")
    parser.add_argument("--track-check", "-tc", action="store_true", help="Check if all frames are present in the tracking results and vice versa.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print the extracted flight log data.")

    # Anomalies detection related arguments
    parser.add_argument("--radius-diff-threshold", "-rdt", type=float, default=15, help="Threshold for the maximal positional deviation (in meters) from the initial hover.")
    parser.add_argument("--altitude-diff-threshold", "-adt", type=float, default=5, help="Threshold for the maximal altitude deviation (in meters) from the initial hover.")
    parser.add_argument("--frame-diff-threshold", "-fdt", type=int, default=2, help="Threshold for the maximal frame difference.")
    parser.add_argument("--timestamp-diff-threshold", "-tdt", type=float, default=0.5, help="Threshold for the maximal timestamp difference (in seconds).")
    parser.add_argument("--iso-diff-threshold", "-idt", type=int, default=300, help="Threshold for the maximal ISO deviation.")
    parser.add_argument("--shutter-diff-threshold", "-sdt", type=float, default=0.02, help="Threshold for the maximal shutter speed deviation.")
    parser.add_argument("--fnum-diff-threshold", "-fndt", type=float, default=0.1, help="Threshold for the maximal f-number deviation.")
    parser.add_argument("--ct-diff-threshold", "-cdt", type=int, default=2000, help="Threshold for the maximal color temperature deviation.")
    parser.add_argument("--focal-len-diff-threshold", "-fldt", type=float, default=0.5, help="Threshold for the maximal focal length deviation.")

    cli_args = parser.parse_args()

    if not cli_args.output_folder:
        cli_args.output_folder = cli_args.input_folder

    return cli_args


if __name__ == "__main__":
    cli_args = get_cli_arguments()
    find_cut_video_issues(cli_args)
