#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
av_comparison_and_filter_tuning.py - AV Trajectory Comparison and Filter Tuning

Compares autonomous vehicle trajectories from Stanford dataset against extracted trajectories.
Supports smoothing parameter tuning and comprehensive error analysis with visualization.

Usage:
  python tools/av_comparison_and_filter_tuning.py --data <path> [options]

Arguments:
  --data <path> : Path to folder containing AV trajectories and results.

Options:
  --help, -h            : Show this help message and exit.
  --save                : Save plots as PDF files (default: False).
  --show                : Display plots interactively (default: False).
  --coords <str>        : Plot coordinates: local or global (default: local).
  -t, --tune            : Enable smoothing parameter tuning (default: False).
  -f, --filter <str>    : Filter type: gaussian or savitzky_golay (default: gaussian).
  -d, --debug           : Enable debug mode for additional output (default: False).

Examples:
1. Basic comparison with visualization:
   python tools/av_comparison_and_filter_tuning.py --data data/ --show

2. Parameter tuning with plots saved:
   python tools/av_comparison_and_filter_tuning.py --data data/ --tune --save

3. Global coordinates with Savitzky-Golay filter:
   python tools/av_comparison_and_filter_tuning.py --data data/ --coords global --filter savitzky_golay

Input:
- AV dataset: av_trajectories/ subfolder with RTK-GNSS data
- Extracted data: results/ subfolder with CSV trajectory files
- Video data for image coordinate visualization

Output:
- Trajectory comparison plots in local/global coordinates
- Positional and speed error statistics
- Kinematics analysis (speed, acceleration)
- Filter tuning results and optimal parameters
- Optional PDF plots saved to results/plots/

Notes:
- Data format specific to geo-trax paper experiments (DOI: 10.1016/j.trc.2025.105205)
- Does not require synchronized timestamps between AV and extracted data
- Supports Gaussian and Savitzky-Golay smoothing filters for parameter optimization
"""

import argparse
import datetime
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from stabilo import Stabilizer

try:
    from utils.utils import detect_delimiter
except ImportError as e:
    print(
        "\033[91mCould not import 'detect_delimiter' from 'utils.utils'.\n"
        "Make sure you have installed the geo-trax package in editable mode:\n"
        "    pip install -e .\n"
        "from the project root directory.\033[0m"
    )
    raise e

# AV's vehicle ID in the trajectory files
video2id = {
    'E1_AV': 17,
    'E2_AV': 14,
    'G1_AV': 16,
    'J1_AV': 66,
    'J2_AV': 50,
    'K1_AV': 27,
    'K2_AV': 78,
    'L1_AV': 62,
    'L2_AV': 82,
    'M1_AV': 27,
    'M2_AV': 41,
    'O1_AV': 21,
    'P1_AV': 42,
    'Q1_AV': 27,
    'Q2_AV': 25,
}

# AV start/stop timestamps (rounded to seconds) the AV was seen in the corresponding video
video2timestamp = {
    'E1_AV': ('2022-10-07 09:14:22', '2022-10-07 09:14:40'),  # D2, AM4, E3 (0:24~0:38)*
    'E2_AV': ('2022-10-07 09:55:21', '2022-10-07 09:55:35'),  # D2, AM5, E3 (1:22~1:35)*
    'G1_AV': ('2022-10-07 09:11:49', '2022-10-07 09:12:06'),  # D4, AM4, G1 (0:13~0:29)*
    'J1_AV': ('2022-10-07 09:11:10', '2022-10-07 09:11:30'),  # D6, AM4, J1 (0:00~0:19)*
    'J2_AV': ('2022-10-07 09:53:59', '2022-10-07 09:54:02'),  # D6, AM5, J1 (2:34~2:37)*
    'K1_AV': ('2022-10-07 09:10:18', '2022-10-07 09:10:32'),  # D6, AM4, K2 (2:54~3:07)*
    'K2_AV': ('2022-10-07 09:47:34', '2022-10-07 09:47:44'),  # D6, AM5, K2 (0:00~0:09)*
    'L1_AV': ('2022-10-07 09:08:23', '2022-10-07 09:08:43'),  # D7, AM4, L2 (0:44~1:02)
    'L2_AV': ('2022-10-07 09:47:59', '2022-10-07 09:48:18'),  # D7, AM5, L2 (0:19~0:37)
    'M1_AV': ('2022-10-07 09:02:08', '2022-10-07 09:02:44'),  # D3, AM4, M1 (1:30~2:05)*
    'M2_AV': ('2022-10-07 09:40:55', '2022-10-07 09:41:12'),  # D3, AM5, M1 (0:10~0:24)
    'O1_AV': ('2022-10-07 09:05:41', '2022-10-07 09:05:56'),  # D5, AM4, O1 (0:49~1:02)*
    'P1_AV': ('2022-10-07 09:06:09', '2022-10-07 09:06:49'),  # D7, AM4, P1 (3:03~3:41)
    'Q1_AV': ('2022-10-07 09:01:20', '2022-10-07 09:01:36'),  # D8, AM4, Q1 (0:40~0:58)
    'Q2_AV': ('2022-10-07 09:26:23', '2022-10-07 09:26:42'),  # D8, AM4, O5 (1:16~1:33)*
}

AV_SPEED_THRESHOLD = 1  # AV speed below this threshold in [km/h] is considered as stopped
FPS = 29.97002997002997  # frames per second of the videos

COLORS = [
    '#3274d9',
    '#ff9d00',
    '#9954bb',
    '#ffc000',
    '#ff61b4',
    '#76b041',
    '#00a8b5',
    '#f9c80e',
    '#f86624',
    '#a45060',
    '#f86624',
    '#f9c80e',
    '#00a8b5',
    '#76b041',
    '#ff61b4',
]
SAVE_FONT_SIZE = 14


def evaluate_av_data(args):
    # get the AV trajectory and speed from the Stanford dataset
    df_stanford_geo = get_on_board_av_data(args.data / 'av_trajectories')

    # get the AV trajectory and speed from our dataset
    df_extracted_geo, df_extracted_img = get_extracted_av_data(args.data / 'results')

    if args.tune:
        tune_smoothing_parameters(df_stanford_geo, df_extracted_geo, args)
    else:
        # plot the AV trajectories in the image coordinates
        plot_img_trajectories_video(df_extracted_img, args)

        # plot the global AV trajectory (on-board sensor only, hue=speed) in the global/local coordinates
        plot_geo_trajectories_all(df_stanford_geo, df_extracted_geo, True, args)

        # plot the global AV trajectory (on-board vs. extracted) in the global/local coordinates
        plot_geo_trajectories_all(df_stanford_geo, df_extracted_geo, False, args)

        # plot per video AV trajectory (on-board vs. extracted) in the global/local coordinates
        plot_geo_trajectories_video(df_stanford_geo, df_extracted_geo, args)

        # compute the positional and speed errors
        df_stanford_geo = compute_positional_and_speed_errors(df_stanford_geo, df_extracted_geo, args)[0]

        # plot the positional and speed errors
        plot_positional_and_speed_errors(df_stanford_geo, args)

        # plot the AV velocities
        plot_kinematics(df_stanford_geo, df_extracted_geo, args, 'speed')

        # plot the AV acceleration
        plot_kinematics(df_stanford_geo, df_extracted_geo, args, 'acceleration')

        # plot the sampling differences
        plot_sampling_diff(df_stanford_geo, df_extracted_geo, args)


def get_on_board_av_data(av_trajectories_folder) -> pd.DataFrame:
    if not av_trajectories_folder.is_dir():
        raise ValueError(f"Data {av_trajectories_folder} is not a directory.")

    av_trajectories = np.loadtxt(av_trajectories_folder / 'sec_nsec_lon_lat_2022_10_07_08_53_38.txt')
    av_velocities = np.loadtxt(av_trajectories_folder / 'sec_nsec_horSpd_2022_10_07_08_53_38.txt')

    df_av = pd.DataFrame(
        {
            'Timestamp': av_trajectories[:, 0] + [round(float_s * 1e-9, 3) for float_s in av_trajectories[:, 1]],
            'Longitude': av_trajectories[:, 2],
            'Latitude': av_trajectories[:, 3],
            'Vehicle_Speed': av_velocities[:, 2] * 3.6,  # convert m/s to km/h
        }
    )

    # convert the time to a datetime object and use GMT+9
    df_av['Timestamp'] = pd.to_datetime(df_av['Timestamp'], unit='s') + datetime.timedelta(hours=9)

    # compute the vehicle acceleration
    time_diffs = (df_av['Timestamp'] - df_av['Timestamp'].shift()).dt.total_seconds()
    Acceleration = (np.array(av_velocities)[1:, 2] - np.array(av_velocities)[:-1, 2]) / time_diffs[1:]
    df_av['Vehicle_Acceleration'] = np.insert(Acceleration, 0, np.nan)

    # convert the latitude and longitude to local coordinates
    df_av = global_to_local_coords(df_av)

    return df_av


def get_extracted_av_data(results_folder) -> Union[pd.DataFrame, pd.DataFrame]:
    # check if data is a directory
    if not results_folder.is_dir():
        print(f"Error: Data {results_folder} is not a directory.")
        exit(1)

    # get the filepaths of the extracted vehicle trajectories
    available_filenames = [file.name for file in results_folder.glob('*.csv')]

    # extract the AV vehicle trajectories
    dfs_geo, dfs_img = [], []
    for filename in available_filenames:
        # read the csv file
        results_filepath = results_folder / filename
        df_geo = pd.read_csv(results_filepath, delimiter=',', header=0)

        # filter out av data
        video_name = filename.split('.')[0]
        id_av = video2id[video_name]
        df_geo = df_geo[(df_geo['Vehicle_ID'] == id_av) & (df_geo['Visibility'] == 1)]

        # drop unnecessary columns and convert the time to a datetime object
        df_geo = df_geo[['Timestamp', 'Longitude', 'Latitude', 'Vehicle_Speed', 'Vehicle_Acceleration']]
        df_geo['Timestamp'] = pd.to_datetime(df_geo['Timestamp'])

        # add the video name and elapsed time to the dataframe
        df_geo.insert(0, 'Video', video_name)
        df_geo.insert(1, 'Elapsed_Time', (df_geo['Timestamp'] - df_geo['Timestamp'].iloc[0]).dt.total_seconds())

        # append the dataframe to the list
        dfs_geo.append(df_geo)

        # read the txt file if it exists
        tracks_filepath = results_folder / filename.replace('.csv', '.txt')
        if tracks_filepath.is_file():
            delimiter = detect_delimiter(tracks_filepath)
            df_img = pd.read_csv(
                tracks_filepath, delimiter=delimiter, header=None, usecols=[1, 6, 7], names=['ID', 'X_img', 'Y_img']
            )
            df_img = df_img[df_img['ID'] == id_av].drop(columns=['ID'])
            df_img.insert(0, 'Video', video_name)
            dfs_img.append(df_img)

    # concatenate the dataframes
    df_geo = pd.concat(dfs_geo, ignore_index=True)
    df_img = pd.concat(dfs_img, ignore_index=True) if dfs_img else None

    # convert the latitude and longitude to local coordinates
    df_geo = global_to_local_coords(df_geo)

    return df_geo, df_img


def global_to_local_coords(df_av) -> pd.DataFrame:
    # define transformer from epsg:4326 to epsg:5186
    transformer = Transformer.from_crs('epsg:4326', 'epsg:5186', always_xy=True)

    # apply transformation to each row
    local_coordinates = df_av.apply(lambda row: transformer.transform(row['Longitude'], row['Latitude']), axis=1)

    # extract transformed coordinates
    idx = df_av.columns.get_loc('Latitude')
    df_av.insert(idx + 1, 'Local_X', [xy[0] for xy in local_coordinates])
    df_av.insert(idx + 2, 'Local_Y', [xy[1] for xy in local_coordinates])

    return df_av


def compute_kinematics(df_extracted, video, sigma, filter_name):
    def calculate_speed(Vehicle_Local_X_Interpolated, Vehicle_Local_Y_Interpolated, fps):
        Del_X = np.array(Vehicle_Local_X_Interpolated)[1:] - np.array(Vehicle_Local_X_Interpolated)[:-1]
        Del_Y = np.array(Vehicle_Local_Y_Interpolated)[1:] - np.array(Vehicle_Local_Y_Interpolated)[:-1]
        Speed = np.array([np.sqrt(Del_X[i] ** 2 + Del_Y[i] ** 2) * fps for i in range(len(Del_X))])
        return Speed

    def calculate_acceleration(Speed, fps):
        Del_Speed = np.array(Speed)[1:] - np.array(Speed)[:-1]
        Acceleration = np.array([Del_Speed[i] * fps for i in range(len(Del_Speed))])
        return Acceleration

    def apply_smoothing(array, sigma, filter_name='gaussian'):
        if filter_name == 'gaussian':
            return gaussian_filter1d(array, sigma, mode='reflect', truncate=3.0)
        elif filter_name == 'savitzky_golay':
            return savgol_filter(array, int(sigma), 3, mode='interp')
        else:
            raise ValueError(f"Unknown filter '{filter_name}'.")

    # get the vehicle local coordinates
    Vehicle_Local_X = df_extracted[df_extracted['Video'] == video]['Local_X'].values
    Vehicle_Local_Y = df_extracted[df_extracted['Video'] == video]['Local_Y'].values

    # compute speed and acceleration
    Speed = calculate_speed(Vehicle_Local_X, Vehicle_Local_Y, FPS)
    Speed = apply_smoothing(Speed, sigma, filter_name)
    Acceleration = calculate_acceleration(Speed, FPS)
    # Acceleration = apply_smoothing(Acceleration, sigma)

    # insert nan values to the beginning of the arrays
    Speed = np.insert(Speed * 3.6, 0, np.nan)
    Acceleration = np.insert(Acceleration, 0, [np.nan] * 2)

    return Speed, Acceleration


def tune_smoothing_parameters(df_stanford, df_extracted, args):
    # define the sigma range and step
    if args.filter == 'gaussian':
        sigma_min, sigma_max, sigma_step = 1, 25, 0.5  # for Gaussian filter
    elif args.filter == 'savitzky_golay':
        sigma_min, sigma_max, sigma_step = 30, 80, 3  # for Savitzky-Golay filter

    # loop over the sigma values
    error_stats = {}
    df_av_extracted_smoothed = df_extracted.copy()
    for i, sigma in enumerate(np.linspace(sigma_min, sigma_max, int((sigma_max - sigma_min) / sigma_step) + 1)):
        for video in sorted(df_extracted['Video'].unique()):
            Speed, Acceleration = compute_kinematics(df_extracted, video, sigma, args.filter)

            df_av_extracted_smoothed.loc[df_av_extracted_smoothed['Video'] == video, 'Vehicle_Speed'] = Speed
            df_av_extracted_smoothed.loc[df_av_extracted_smoothed['Video'] == video, 'Vehicle_Acceleration'] = (
                Acceleration
            )

        if i == 0:
            df_av_stanford_with_errors, _, intersection_error_stats_sigma, intersection_meta = (
                compute_positional_and_speed_errors(df_stanford, df_av_extracted_smoothed, args)
            )
        else:
            _, _, intersection_error_stats_sigma, _ = compute_positional_and_speed_errors(
                df_stanford, df_av_extracted_smoothed, args
            )

        for intersection in intersection_error_stats_sigma.keys():
            if intersection in error_stats:
                error_stats[intersection][sigma] = intersection_error_stats_sigma[intersection]
            else:
                error_stats[intersection] = {sigma: intersection_error_stats_sigma[intersection]}

    # find the best sigma for each intersection based on the lowest mean absolute speed error
    print('Based on lowest MEAN absolute speed error:')
    sigma_best_all = 0
    sigma_best_weighted = 0
    for intersection, stats in error_stats.items():
        # choose the sigma with the lowest mean absolute speed error
        sigma_best = min(stats, key=lambda x: stats[x][2])
        sigma_best_all += sigma_best
        sigma_best_weighted += sigma_best * intersection_meta[intersection]['length']
        for video in df_extracted['Video'].unique():
            if video[0] == intersection:
                Speed, Acceleration = compute_kinematics(df_extracted, video, sigma_best, args.filter)
                df_av_extracted_smoothed.loc[df_av_extracted_smoothed['Video'] == video, 'Vehicle_Speed'] = Speed
                df_av_extracted_smoothed.loc[df_av_extracted_smoothed['Video'] == video, 'Vehicle_Acceleration'] = (
                    Acceleration
                )
        print(f'The best sigma for intersection {intersection} is {sigma_best}, resulting in speed error of {stats[sigma_best][2]:.3f} +/- {stats[sigma_best][3]:.3f} km/h')
    print(f'The average best sigma for all intersections: {sigma_best_all / len(error_stats):.2f}')
    print(f'The weighted average best sigma for all intersections: {sigma_best_weighted / sum([intersection_meta[intersection]["length"] for intersection in intersection_meta]):.2f}')

    # plot the error statistics for the speed errors for different tuning parameters
    plot_tuned_speed_errors(error_stats, args)
    plot_kinematics(df_av_stanford_with_errors, df_av_extracted_smoothed, args, 'speed')
    plot_kinematics(df_av_stanford_with_errors, df_av_extracted_smoothed, args, 'acceleration')

    # find the best sigma for each intersection based on the lowest speed error standard deviation
    print('\nBased on lowest speed error STD. DEV.:')
    sigma_best_all = 0
    for intersection, stats in error_stats.items():
        # choose the sigma with the lowest speed error standard deviation
        sigma_best = min(stats, key=lambda x: stats[x][3])
        sigma_best_all += sigma_best
        for video in df_extracted['Video'].unique():
            if video[0] == intersection:
                Speed, Acceleration = compute_kinematics(
                    df_extracted, video, sigma_best, args.filter
                )
                df_av_extracted_smoothed.loc[
                    df_av_extracted_smoothed['Video'] == video, 'Vehicle_Speed'
                ] = Speed
                df_av_extracted_smoothed.loc[
                    df_av_extracted_smoothed['Video'] == video, 'Vehicle_Acceleration'
                ] = Acceleration
        print(
            f'The best sigma for intersection {intersection} is {sigma_best}, resulting in speed error of '
            f'{stats[sigma_best][2]:.3f} +/- {stats[sigma_best][3]:.3f} km/h'
        )
    print(f'The average best sigma for all intersections is {sigma_best_all / len(error_stats):.2f}')

    # plot the speed errors for different tuning parameters
    plot_kinematics(df_av_stanford_with_errors, df_av_extracted_smoothed, args, 'speed')
    plot_kinematics(df_av_stanford_with_errors, df_av_extracted_smoothed, args, 'acceleration')


def compute_positional_and_speed_errors(df_stanford, df_extracted, args) -> Union[pd.DataFrame, dict]:
    # define helper function to compute the length of the trajectory
    def compute_trajectory_length(df):
        trajectory_length = 0
        for i in range(1, len(df)):
            trajectory_length += np.sqrt(
                (df['Local_X'].iloc[i] - df['Local_X'].iloc[i - 1]) ** 2
                + (df['Local_Y'].iloc[i] - df['Local_Y'].iloc[i - 1]) ** 2
            )
        return trajectory_length

    # initialize the lists and dictionaries
    dfs_with_errors = []
    intersection_errors, intersection_meta = {}, {}
    video_error_stats, intersection_error_stats = {}, {}

    # loop over the videos
    for video in sorted(df_extracted['Video'].unique()):
        # get the intersection name
        intersection = video[0]

        # get the filtered AV trajectories per video from the Stanford and extracted datasets
        df_stanford_f, df_extracted_f = get_filtered_av_trajectories(df_stanford, df_extracted, video)

        # find the exact start/stop timestamps for the Stanford dataset
        t_start_exact, t_stop_exact = find_start_end_times(df_stanford_f, df_extracted_f)

        # further filter the AV trajectory based on found start/stop times
        df_stanford_f = df_stanford_f[
            (df_stanford_f['Timestamp'] >= t_start_exact) & (df_stanford_f['Timestamp'] <= t_stop_exact)
        ]

        # add some metadata to the dataframe
        df_stanford_f.insert(0, 'Video', video)
        df_stanford_f.insert(1, 'Elapsed_Time', (df_stanford_f['Timestamp'] - df_stanford_f['Timestamp'].iloc[0]).dt.total_seconds())

        # compute the positional and speed errors per video
        positional_errors, speed_errors = compute_errors_per_video(df_stanford_f, df_extracted_f)

        # compute and print the positional and speed errors per video
        positional_error_mean, positional_error_std = np.nanmean(positional_errors), np.nanstd(positional_errors)
        speed_error_mean, speed_error_std = np.nanmean(speed_errors), np.nanstd(speed_errors)
        trajectory_length = compute_trajectory_length(df_stanford_f)
        trajectory_duration = (t_stop_exact - t_start_exact).total_seconds()

        video_error_stats[video] = (
            positional_error_mean,
            positional_error_std,
            speed_error_mean,
            speed_error_std,
            trajectory_duration,
            trajectory_length,
        )
        if not args.tune:
            print(f'Video {video:<6}: Positional error: {positional_error_mean:.3f} +/- {positional_error_std:.3f} m')
            print(f'{"":<13} Speed error:   {speed_error_mean:.3f} +/- {speed_error_std:.3f} km/h')
            print(f'{"":<13} Length:           {round(trajectory_length, 2)} m')
            print(f'{"":<13} Duration:         {round(trajectory_duration, 2)} s\n')

        # merge the errors and duration for the intersection
        if intersection in intersection_errors:
            intersection_errors[intersection][0].extend(positional_errors)
            intersection_errors[intersection][1].extend(speed_errors)
            intersection_meta[intersection]['length'] += trajectory_length
            intersection_meta[intersection]['duration'] += trajectory_duration
        else:
            intersection_errors[intersection] = [positional_errors, speed_errors]
            intersection_meta[intersection] = {}
            intersection_meta[intersection]['length'] = trajectory_length
            intersection_meta[intersection]['duration'] = trajectory_duration

        # add the positional and speed errors to the Stanford dataframe
        df_stanford_f['Positional_Error'] = positional_errors
        df_stanford_f['Speed_Error'] = speed_errors

        # append the dataframes to the list
        dfs_with_errors.append(df_stanford_f)

        if args.debug:
            print(df_extracted_f.head().to_string(index=False))
            print(df_stanford_f.head().to_string(index=False))
            print(df_extracted_f.tail().to_string(index=False))
            print(df_stanford_f.tail().to_string(index=False))

    # compute and print the positional and speed errors per intersection
    for intersection, (positional_errors, speed_errors) in intersection_errors.items():
        speed_errors_abs = np.abs(speed_errors)
        positional_error_mean, positional_error_std = np.nanmean(positional_errors), np.nanstd(positional_errors)
        speed_error_mean, speed_error_std = np.nanmean(speed_errors_abs), np.nanstd(speed_errors_abs)
        intersection_error_stats[intersection] = (
            positional_error_mean,
            positional_error_std,
            speed_error_mean,
            speed_error_std,
        )
        if not args.tune:
            print(f'Intersec. {intersection:<2}: Positional error: {positional_error_mean:.3f} +/- {positional_error_std:.3f} m')
            print(f'{"":<13} Speed error:   {speed_error_mean:.3f} +/- {speed_error_std:.3f} km/h')
            print(f'{"":<13} Length:           {round(intersection_meta[intersection]["length"], 2)} m')
            print(f'{"":<13} Duration:         {round(intersection_meta[intersection]["duration"], 2)} s\n')

    # concatenate the dataframes
    df_stanford_with_errors = pd.concat(dfs_with_errors, ignore_index=True)

    # save the errors to a file
    if args.save and not args.tune:
        with open(args.data / 'results' / 'plots' / 'AV_positional_and_speed_errors_per_video.tex', 'w') as f:
            for video in sorted(video_error_stats.keys()):
                (
                    positional_error_mean,
                    positional_error_std,
                    speed_error_mean,
                    speed_error_std,
                    trajectory_duration,
                    trajectory_length,
                ) = video_error_stats[video]
                f.write(f'    {video[-2:]} & ${positional_error_mean:.3f} \pm {positional_error_std:.3f}$ & ${speed_error_mean:.3f} \pm {speed_error_std:.3f}$ & {round(trajectory_length, 2):.2f} & {round(trajectory_duration, 2):.2f}\\\\ \n')

        with open(args.data / 'results' / 'plots' / 'AV_positional_and_speed_errors_per_intersection.tex', 'w') as f:
            for intersection in sorted(intersection_error_stats.keys()):
                positional_error_mean, positional_error_std, speed_error_mean, speed_error_std = (
                    intersection_error_stats[intersection]
                )
                trajectory_duration = intersection_meta[intersection]['duration']
                trajectory_length = intersection_meta[intersection]['length']
                f.write(f'    {intersection} & ${positional_error_mean:.3f} \pm {positional_error_std:.3f}$ & ${speed_error_mean:.3f} \pm {speed_error_std:.3f}$ & {round(trajectory_length, 2):.2f} & {round(trajectory_duration, 2):.2f}\\\\ \n')

    return df_stanford_with_errors, video_error_stats, intersection_error_stats, intersection_meta


def compute_errors_per_video(df_stanford, df_extracted):
    # define a helper function to find the 2 closest indices
    def find_min_2_indices(arr):
        i_min1 = np.argmin(arr)
        if i_min1 == 0:
            i_min2 = 1
        elif i_min1 == len(arr) - 1:
            i_min2 = len(arr) - 2
        elif arr[i_min1 - 1] < arr[i_min1 + 1]:
            i_min2 = i_min1 - 1
        else:
            i_min2 = i_min1 + 1

        return i_min1, i_min2

    # loop over the AV trajectory points from the Stanford dataset and find the closest (interpolated) point from our dataset
    positional_errors, speed_errors = [], []
    for point_stanford, speed_stanford in zip(
        zip(df_stanford['Local_X'], df_stanford['Local_Y']), df_stanford['Vehicle_Speed']
    ):
        # skip if the speed is below AV_SPEED_THRESHOLD [km/h]
        if speed_stanford < AV_SPEED_THRESHOLD:
            positional_errors.append(np.nan)
            speed_errors.append(np.nan)
            continue

        # find the index of the closest point and the second closest point to it
        distances = np.sqrt(
            (df_extracted['Local_X'] - point_stanford[0]) ** 2 + (df_extracted['Local_Y'] - point_stanford[1]) ** 2
        ).values
        i_min1, i_min2 = find_min_2_indices(distances)

        # get the coordinates of these 2 points
        point_1 = (df_extracted['Local_X'].iloc[i_min1], df_extracted['Local_Y'].iloc[i_min1])
        point_2 = (df_extracted['Local_X'].iloc[i_min2], df_extracted['Local_Y'].iloc[i_min2])

        # get the velocities of these 2 points
        speed_1 = df_extracted['Vehicle_Speed'].iloc[i_min1]
        speed_2 = df_extracted['Vehicle_Speed'].iloc[i_min2]

        # find the closest distance between the Stanford AV trajectory point and the line connecting the 2 closest points from our dataset
        denumerator = np.linalg.norm(np.array(point_2) - np.array(point_1))
        closest_distance = (
            np.abs(np.cross(np.array(point_2) - np.array(point_1), np.array(point_1) - np.array(point_stanford)))
            / denumerator
        )
        positional_errors.append(closest_distance)

        # compute the weighted speed error
        if speed_1 is np.nan or speed_2 is np.nan:
            if speed_1 is np.nan:
                weight_1, weight_2 = 0, 1
            else:
                weight_1, weight_2 = 1, 0
        else:
            point_1_distance = np.sqrt((point_1[0] - point_stanford[0]) ** 2 + (point_1[1] - point_stanford[1]) ** 2)
            point_2_distance = np.sqrt((point_2[0] - point_stanford[0]) ** 2 + (point_2[1] - point_stanford[1]) ** 2)
            weight_1 = 1 - point_1_distance / (point_1_distance + point_2_distance)
            weight_2 = 1 - point_2_distance / (point_1_distance + point_2_distance)
        speed_error = speed_stanford - (weight_1 * speed_1 + weight_2 * speed_2)
        # speed_error = np.abs(speed_stanford -  (weight_1 * speed_1 + weight_2 * speed_2))
        speed_errors.append(speed_error)

    return positional_errors, speed_errors


def get_filtered_av_trajectories(df_stanford, df_extracted, video):
    # get the timestamps corresponding to the video
    time_start, time_ends = video2timestamp[video]

    time_start = datetime.datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S')
    time_ends = datetime.datetime.strptime(time_ends, '%Y-%m-%d %H:%M:%S')

    # filter the AV trajectories
    df_stanford_f = df_stanford[(df_stanford['Timestamp'] >= time_start) & (df_stanford['Timestamp'] <= time_ends)]
    df_extracted_f = df_extracted[df_extracted['Video'] == video]

    return df_stanford_f, df_extracted_f


def find_start_end_times(df_stanford, df_extracted):
    # define a helper function to find the closest point
    def find_closest_point(df, x, y):
        distances = np.sqrt((df['Local_X'] - x) ** 2 + (df['Local_Y'] - y) ** 2)
        i_min = np.argmin(distances)

        return df['Timestamp'].iloc[i_min]

    # find the closest distance between the first extracted AV position and the AV positions from the Stanford dataset
    x_start_extracted, y_start_extracted = df_extracted['Local_X'].iloc[0], df_extracted['Local_Y'].iloc[0]
    t_start = find_closest_point(df_stanford, x_start_extracted, y_start_extracted)

    # find the closest distance between the last extracted AV position and the AV positions from the Stanford dataset
    x_end_extracted, y_end_extracted = df_extracted['Local_X'].iloc[-1], df_extracted['Local_Y'].iloc[-1]
    t_end = find_closest_point(df_stanford, x_end_extracted, y_end_extracted)

    return t_start, t_end


def plot_tuned_speed_errors(error_stats, args):
    if args.show or args.save:
        num_cols = 3
        intersection_num = len(error_stats)
        _, axs = plt.subplots(
            1 + (intersection_num - 1) // num_cols, num_cols, figsize=(26, 2 + 4 * (intersection_num - 1) // num_cols)
        )
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)

        sns.set_palette("tab10")
        for i, intersection in enumerate(sorted(error_stats.keys())):
            row = i // num_cols
            col = i % num_cols
            error_stats_video = error_stats[intersection]
            mean_errors = [error_stats_video[sigma][2] for sigma in sorted(error_stats_video.keys())]
            std_errors = [error_stats_video[sigma][3] for sigma in sorted(error_stats_video.keys())]

            best_sigma = min(
                error_stats_video, key=lambda x: error_stats_video[x][2]
            )  # based on the lowest mean speed error
            axs[row, col].scatter(
                best_sigma,
                error_stats_video[best_sigma][2],
                color='red',
                marker='o',
                label=f'Best sigma: {best_sigma}',
                s=80,
            )
            sns.lineplot(
                x=sorted(error_stats_video.keys()),
                y=mean_errors,
                ax=axs[row, col],
                marker='o',
                color='green',
                label='Mean error' if not i else '',
            )
            axs[row, col].fill_between(
                sorted(error_stats_video.keys()),
                [mean - std for mean, std in zip(mean_errors, std_errors)],
                [mean + std for mean, std in zip(mean_errors, std_errors)],
                alpha=0.1,
                color='green',
                label='Error std. dev. (+/-)' if not i else '',
            )
            axs[row, col].tick_params(axis='x', labelsize=SAVE_FONT_SIZE - 1 if args.save else None)
            axs[row, col].tick_params(axis='y', labelsize=SAVE_FONT_SIZE - 1 if args.save else None)
            axs[row, col].set_xlabel(
                f'Sigma (Intersection: {intersection})', fontsize=SAVE_FONT_SIZE if args.save else 10
            )
            if not (i % num_cols) or not args.save:
                axs[row, col].set_ylabel(
                    'Speed error distribution (km/h)', fontsize=SAVE_FONT_SIZE if args.save else 10
                )
            if not args.save:
                axs[row, col].set_title(f'AV speed error for {intersection}')
            ymin, ymax = axs[row, col].get_ylim()
            axs[row, col].set_yticks(range(int(ymin), round(ymax) + 1))
            axs[row, col].legend(loc='upper right', fontsize=SAVE_FONT_SIZE if args.save else 10)
        save_show_plot(plt, 'speed_error_vs_sigma', args)


def plot_img_trajectories_video(df_extracted, args):
    if args.show or args.save:
        scenes, transforms = {}, {}
        for video in sorted(df_extracted['Video'].unique()):
            video_filepath = args.data / (video + '.mp4')
            if not video_filepath.is_file():
                print(f"WARNING: File {video_filepath} does not exist.")
                continue
            cap = cv2.VideoCapture(str(video_filepath))
            ret, scene = cap.read()
            cap.release()
            if not ret:
                print(f'WARNING: Could not read the first frame of {video_filepath}.')
                continue

            intersection = video[0]
            if intersection not in scenes:
                scenes[intersection] = scene
                transforms[video] = None
            else:
                # compute the transformation matrix between the 2 scenes
                stabilizer = Stabilizer(downsample_ratio=1.0, mask_use=False, max_features=4000)
                stabilizer.set_ref_frame(scenes[intersection], None)
                stabilizer.stabilize(scene, None)
                transf_matrix = stabilizer.get_cur_trans_matrix()
                transforms[video] = transf_matrix

        # plot the AV trajectories on the image scenes
        i = 0
        for intersection, scene in scenes.items():
            df_extracted_f = df_extracted[df_extracted['Video'].str.contains(intersection)]
            if args.save:
                plt.figure(figsize=(28, 15.75))  # 16:9 aspect ratio
            plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
            legend_elements = []
            for video in sorted(df_extracted_f['Video'].unique()):
                df_extracted_f_video = df_extracted_f[df_extracted_f['Video'] == video]
                if transforms[video] is not None:
                    X_img, Y_img = df_extracted_f_video['X_img'], df_extracted_f_video['Y_img']
                    trajectories = cv2.perspectiveTransform(
                        np.array([[[x, y]] for x, y in zip(X_img, Y_img)]).astype(np.float32), transforms[video]
                    )
                    df_extracted_f_video.loc[:, 'X_img'], df_extracted_f_video.loc[:, 'Y_img'] = (
                        trajectories[:, 0, 0],
                        trajectories[:, 0, 1],
                    )
                legend_elements.append(
                    plt.Line2D([0], [0], lw=2 + 4 * int(args.save), color=COLORS[i], label=f'Intersection {video[0]}')
                )
                num_of_passes = int(video[1])
                if num_of_passes == 2:
                    legend_elements[-2].set_label(f'Intersection {video[0]} (1$^{{st}}$ AV pass)')
                    legend_elements[-1].set_label(f'Intersection {video[0]} (2$^{{nd}}$ AV pass)')
                elif num_of_passes == 3:
                    legend_elements[-1].set_label(f'Intersection {video[0]} (3$^{{rd}}$ AV pass)')
                elif num_of_passes > 3:
                    legend_elements[-1].set_label(f'Intersection {video[0]} ({num_of_passes}$^{{th}}$ AV pass)')

                eps_start, eps_end = 0, 2
                while True:
                    if (
                        df_extracted_f_video['X_img'].iloc[eps_start] < 0
                        or df_extracted_f_video['X_img'].iloc[eps_start] > scene.shape[1] - 1
                    ) or (
                        df_extracted_f_video['Y_img'].iloc[eps_start] < 0
                        or df_extracted_f_video['Y_img'].iloc[eps_start] > scene.shape[0] - 1
                    ):
                        eps_start += 1
                    else:
                        break
                while True:
                    if (
                        df_extracted_f_video['X_img'].iloc[-eps_end] <= 1
                        or df_extracted_f_video['X_img'].iloc[-eps_end] >= scene.shape[1] - 5
                    ) or (
                        df_extracted_f_video['Y_img'].iloc[-eps_end] <= 1
                        or df_extracted_f_video['Y_img'].iloc[-eps_end] >= scene.shape[0] - 5
                    ):
                        eps_end += 1
                    else:
                        break

                plt.plot(
                    df_extracted_f_video['X_img'][eps_start:-eps_end],
                    df_extracted_f_video['Y_img'][eps_start:-eps_end],
                    linewidth=2 + 5 * args.save,
                    color=COLORS[i],
                )
                plt.arrow(
                    df_extracted_f_video['X_img'].iloc[-2 * eps_end],
                    df_extracted_f_video['Y_img'].iloc[-2 * eps_end],
                    df_extracted_f_video['X_img'].iloc[-(eps_end - 1)]
                    - df_extracted_f_video['X_img'].iloc[-2 * eps_end],
                    df_extracted_f_video['Y_img'].iloc[-(eps_end - 1)]
                    - df_extracted_f_video['Y_img'].iloc[-2 * eps_end],
                    head_width=80,
                    head_length=80,
                    fc=COLORS[i],
                    ec=COLORS[i],
                    length_includes_head=True,
                )
                i += 1
            plt.axis('off')
            plt.legend(
                handles=legend_elements, fontsize=2 * SAVE_FONT_SIZE if args.save else 10, bbox_to_anchor=(0.99, 1)
            )
            save_show_plot(plt, f'trajectories_found_in_{intersection}', args)


def plot_geo_trajectories_all(df_stanford, df_extracted, speed_on, args):
    if args.show or args.save:
        if args.coords == 'local':
            x_label, y_label = 'Local_X', 'Local_Y'
        elif args.coords == 'global':
            x_label, y_label = 'Longitude', 'Latitude'

        # plot the AV trajectory from the Stanford dataset and our dataset
        plt.figure()
        if speed_on:
            plt.scatter(df_stanford[x_label], df_stanford[y_label], c=df_stanford['Vehicle_Speed'], cmap='jet', s=0.5)
            plt.colorbar(label='AV speed (km/h)')
        else:
            plt.plot(df_stanford[x_label], df_stanford[y_label], label='Stanford on-board RTK', color='black')
            for i, video in enumerate(sorted(df_extracted['Video'].unique())):
                df_extracted_f = df_extracted[df_extracted['Video'] == video]
                plt.plot(
                    df_extracted_f[x_label], df_extracted_f[y_label], label=f'Extracted from {video}', color=COLORS[i]
                )
                # plot the direction of the AV trajectory. Use an appropriate arrow size
                dx = df_extracted_f[x_label].iloc[-1] - df_extracted_f[x_label].iloc[-2]
                dy = df_extracted_f[y_label].iloc[-1] - df_extracted_f[y_label].iloc[-2]
                plt.arrow(
                    df_extracted_f[x_label].iloc[-1],
                    df_extracted_f[y_label].iloc[-1],
                    dx,
                    dy,
                    head_width=10,
                    head_length=10,
                    fc=COLORS[i],
                    ec=COLORS[i],
                )
            plt.legend()
        plt.xlabel(x_label.replace('_', ' '))
        plt.ylabel(y_label.replace('_', ' '))
        plt.title('Stanford AV Trajectory (on-board RTK-GNSS)' if speed_on else 'AV Trajectory Comparison')
        save_show_plot(plt, f'trajectory_{args.coords}{"" if speed_on else "_with_sep"}', args)


def plot_positional_and_speed_errors(df_errors, args):
    # plot the positional and speed error per video and against elapsed time
    if args.show or args.save:
        axs = plt.subplots(1, 2, figsize=(15, 7.5))[1]
        sns.scatterplot(x='Elapsed_Time', y='Positional_Error', hue='Video', data=df_errors, ax=axs[0])
        axs[0].set_xlabel('Elapsed Time (s)')
        axs[0].set_ylabel('Positional Error (m)')
        axs[0].set_title('AV Positional error')
        sns.scatterplot(x='Elapsed_Time', y='Speed_Error', hue='Video', data=df_errors, ax=axs[1])
        axs[1].set_xlabel('Elapsed Time (s)')
        axs[1].set_ylabel('Speed Error (km/h)')
        axs[1].set_title('AV speed error')
        save_show_plot(plt, 'positional_and_speed_errors', args)


def plot_geo_trajectories_video(df_stanford, df_extracted, args, shared_axs=True):
    if args.show or args.save:
        if args.coords == 'local':
            x_label, y_label = 'Local_X', 'Local_Y'
        elif args.coords == 'global':
            x_label, y_label = 'Longitude', 'Latitude'

        num_cols = 5
        num_videos = len(df_extracted['Video'].unique())
        fig, axs = plt.subplots(
            1 + (num_videos - 1) // num_cols, num_cols, figsize=(25, 4.5 + 4.5 * (num_videos - 1) // num_cols)
        )
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)

        legend_elements = []
        half_range = 100  # meters
        for i, video in enumerate(sorted(df_extracted['Video'].unique())):
            row = i // num_cols
            col = i % num_cols
            df_av_stanford_f, df_extracted_f = get_filtered_av_trajectories(df_stanford, df_extracted, video)
            axs[row, col].plot(df_av_stanford_f[x_label], df_av_stanford_f[y_label], color='black', linewidth=1)
            axs[row, col].scatter(df_av_stanford_f[x_label], df_av_stanford_f[y_label], color='black', s=2)
            axs[row, col].plot(
                df_extracted_f[x_label], df_extracted_f[y_label], color=COLORS[i], linewidth=1, alpha=0.8
            )
            axs[row, col].scatter(df_extracted_f[x_label], df_extracted_f[y_label], color=COLORS[i], s=2, alpha=0.8)

            # axs[row, col].plot([], [], '-o', color=COLORS[i], lw=1, alpha=0.8, label = f'Extracted at {video[0]}')
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=COLORS[i],
                    lw=1,
                    marker='o',
                    markersize=4,
                    alpha=0.8,
                    label=f'Extracted at {video[0]}',
                )
            )
            number_of_passes = int(video[1])
            if int(number_of_passes) == 2:
                legend_elements[-2].set_label(f'Extracted at {video[0]} (1st AV pass)')
                legend_elements[-1].set_label(f'Extracted at {video[0]} (2nd AV pass)')
            elif int(number_of_passes) == 3:
                legend_elements[-1].set_label(f'Extracted at {video[0]} (3rd AV pass)')
            elif int(number_of_passes) > 3:
                legend_elements[-1].set_label(f'Extracted at {video[0]} ({number_of_passes}th AV pass)')

            t_start, t_end = find_start_end_times(df_av_stanford_f, df_extracted_f)
            df_av_stanford_f_start = df_av_stanford_f[df_av_stanford_f['Timestamp'] == t_start]
            df_av_stanford_f_end = df_av_stanford_f[df_av_stanford_f['Timestamp'] == t_end]

            axs[row, col].scatter(
                df_av_stanford_f_start[x_label],
                df_av_stanford_f_start[y_label],
                color='green',
                facecolors='none',
                linewidth=2,
                s=60,
            )
            axs[row, col].scatter(
                df_av_stanford_f_end[x_label],
                df_av_stanford_f_end[y_label],
                color='red',
                facecolors='none',
                linewidth=2,
                s=60,
            )

            axs[row, col].tick_params(axis='x', labelsize=SAVE_FONT_SIZE - 2 if args.save else None)
            axs[row, col].tick_params(axis='y', labelsize=SAVE_FONT_SIZE - 2 if args.save else None)
            axs[row, col].set_xlabel(x_label.replace('_', ' '), fontsize=SAVE_FONT_SIZE if args.save else None)
            if not (i % num_cols) or not args.save:
                axs[row, col].set_ylabel(y_label.replace('_', ' '), fontsize=SAVE_FONT_SIZE if args.save else None)
            axs[row, col].set_title(None if args.save else f'AV Trajectory Comparison for {video}')

            if shared_axs:
                axs[row, col].set_aspect('equal', adjustable='box', anchor='C')
                # Calculate center of the plot
                x_center = (df_extracted_f[x_label].min() + df_extracted_f[x_label].max()) / 2
                y_center = (df_extracted_f[y_label].min() + df_extracted_f[y_label].max()) / 2

                axs[row, col].set_xlim(x_center - half_range, x_center + half_range)
                axs[row, col].set_ylim(y_center - half_range, y_center + half_range)
            else:
                axs[row, col].set_aspect('equal', adjustable='datalim', anchor='C')

        for i, ax in enumerate(axs.flat):
            if i >= num_videos:
                ax.set_visible(False)
                continue
            ax.legend(
                handles=[legend_elements[i]], loc='upper right', fontsize=SAVE_FONT_SIZE - 1 if args.save else None
            )

        top_legend = [
            plt.Line2D([0], [0], color='black', lw=1, marker='o', markersize=4, label='Stanford on-board RTK-GPS')
        ]
        top_legend.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="None",
                marker='o',
                color='w',
                markeredgecolor='green',
                markeredgewidth=2,
                markersize=7,
                label='Start point',
            )
        )
        top_legend.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="None",
                marker='o',
                color='w',
                markeredgecolor='red',
                markeredgewidth=2,
                markersize=7,
                label='End point',
            )
        )
        fig.legend(
            handles=top_legend,
            loc='upper center',
            ncol=3,
            bbox_to_anchor=(0.5, 1.03 if args.save else 1.01),
            fontsize=SAVE_FONT_SIZE - 1 if args.save else None,
            fancybox=True,
        )

        save_show_plot(plt, f'trajectory_{args.coords}_sep', args)


def plot_kinematics(df_stanford, df_extracted, args, plot_variable_type='speed'):
    if args.show or args.save:
        if plot_variable_type == 'speed':
            y_name = 'Vehicle_Speed'
            y_label = 'Speed (km/h)'
        elif plot_variable_type == 'acceleration':
            y_name = 'Vehicle_Acceleration'
            y_label = 'Acceleration (m/s$^2$)'

        num_cols = 5
        num_videos = len(df_extracted['Video'].unique())
        fig, axs = plt.subplots(
            1 + (num_videos - 1) // num_cols, num_cols, figsize=(25, 4.5 + 4.5 * (num_videos - 1) // num_cols)
        )
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)

        legend_elements = []
        for i, video in enumerate(sorted(df_extracted['Video'].unique())):
            row = i // num_cols
            col = i % num_cols
            df_stanford_f = df_stanford[df_stanford['Video'] == video]
            df_extracted_f = df_extracted[df_extracted['Video'] == video]
            stanford_time_stop = df_stanford_f['Elapsed_Time'].iloc[-1]

            axs[row, col].plot(df_stanford_f['Elapsed_Time'], df_stanford_f[y_name], color='black', linewidth=1)
            axs[row, col].scatter(df_stanford_f['Elapsed_Time'], df_stanford_f[y_name], color='black', s=2)
            axs[row, col].plot(
                df_extracted_f['Elapsed_Time'][df_extracted_f['Elapsed_Time'] <= stanford_time_stop],
                df_extracted_f[y_name][df_extracted_f['Elapsed_Time'] <= stanford_time_stop],
                color=COLORS[i],
                linewidth=1,
                alpha=0.8,
            )
            axs[row, col].scatter(
                df_extracted_f['Elapsed_Time'][df_extracted_f['Elapsed_Time'] <= stanford_time_stop],
                df_extracted_f[y_name][df_extracted_f['Elapsed_Time'] <= stanford_time_stop],
                color=COLORS[i],
                s=2,
                alpha=0.8,
            )

            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=COLORS[i],
                    lw=1,
                    marker='o',
                    markersize=4,
                    alpha=0.8,
                    label=f'Extracted at {video[0]}',
                )
            )
            number_of_passes = int(video[1])
            if int(number_of_passes) == 2:
                legend_elements[-2].set_label(f'Extracted at {video[0]} (1st pass)')
                legend_elements[-1].set_label(f'Extracted at {video[0]} (2nd pass)')
            elif int(number_of_passes) == 3:
                legend_elements[-1].set_label(f'Extracted at {video[0]} (3rd pass)')
            elif int(number_of_passes) > 3:
                legend_elements[-1].set_label(f'Extracted at {video[0]} ({number_of_passes}th pass)')

            axs[row, col].set_xlabel('Elapsed time (s)', fontsize=SAVE_FONT_SIZE if args.save else None)
            if not (i % num_cols) or not args.save:
                axs[row, col].set_ylabel(y_label, fontsize=SAVE_FONT_SIZE if args.save else None)
            axs[row, col].tick_params(axis='x', labelsize=SAVE_FONT_SIZE - 1 if args.save else None)
            axs[row, col].tick_params(axis='y', labelsize=SAVE_FONT_SIZE - 1 if args.save else None)

        for i, ax in enumerate(axs.flat):
            if i >= num_videos:
                ax.set_visible(False)
                continue
            ax.legend(
                handles=[legend_elements[i]], loc='upper right', fontsize=SAVE_FONT_SIZE - 1 if args.save else None
            )

        top_legend = [
            plt.Line2D([0], [0], color='black', lw=1, marker='o', markersize=4, label='Stanford on-board RTK-GPS')
        ]
        fig.legend(
            handles=top_legend,
            loc='upper center',
            ncol=1,
            bbox_to_anchor=(0.5, 1.03 if args.save else 1.01),
            fontsize=SAVE_FONT_SIZE - 1 if args.save else None,
            fancybox=True,
        )

        max_value = np.ceil(max(df_stanford[y_name].max(), df_extracted[y_name].max()))
        min_value = np.floor(min(df_stanford[y_name].min(), df_extracted[y_name].min()))
        round_to = 5 if plot_variable_type == 'speed' else 1
        max_value = round_to * np.ceil(max_value / round_to)
        min_value = round_to * np.floor(min_value / round_to)
        for ax in axs.flatten():
            ax.set_ylim(min_value, max_value)

        save_show_plot(plt, f'{plot_variable_type}_comparison' + '_tuned_sigma' * args.tune, args)


def plot_sampling_diff(df_stanford, df_extracted, args):
    if (args.show or args.save) and args.debug:
        axs = plt.subplots(1, 2, figsize=(15, 7.5))[1]
        sampling_diff_stanford = df_stanford['Timestamp'].diff().dt.total_seconds().dropna()
        diff_mean = sampling_diff_stanford.mean()
        diff_std = sampling_diff_stanford.std()
        axs[0].plot(sampling_diff_stanford, color='black')
        axs[0].axhline(y=diff_mean, color='r', linestyle='--', label=f'Mean: {diff_mean:.3f}')
        axs[0].axhline(y=diff_mean + diff_std, color='g', linestyle='--', label=f'+/-Std: {diff_std:.3f}')
        axs[0].axhline(y=diff_mean - diff_std, color='g', linestyle='--')
        axs[0].legend()
        axs[0].set_xlabel('Sample')
        axs[0].set_ylabel('Time Difference ($s$)')
        axs[0].set_title('Sampling Differences for Stanford')

        sampling_diff_extracted_all = []
        for video in sorted(df_extracted['Video'].unique()):
            df_extracted_f = df_extracted[df_extracted['Video'] == video]
            sampling_diff_extracted = df_extracted_f['Timestamp'].diff().dt.total_seconds().dropna()
            sampling_diff_extracted_all.append(sampling_diff_extracted)
        sampling_diff_extracted_all = pd.concat(sampling_diff_extracted_all)
        diff_mean = sampling_diff_extracted_all.mean()
        axs[1].plot(sampling_diff_extracted_all, color='black')
        axs[1].axhline(y=diff_mean, color='r', linestyle='--', label=f'Mean: {diff_mean:.3f}')
        axs[1].set_xlabel('Sample')
        axs[1].set_ylabel('Time Difference ($s$)')
        axs[1].set_title('Sampling Differences for Extracted Data')
        save_show_plot(plt, 'sampling_diff', args)


def save_show_plot(plt, file_name, args):
    plt.tight_layout()
    if args.save:
        img_filepath = args.data / 'results' / 'plots' / f'AV_{file_name}.pdf'
        if not img_filepath.parent.is_dir():
            img_filepath.parent.mkdir(parents=True)
        plt.savefig(str(img_filepath), bbox_inches='tight', pad_inches=0)
    if args.show:
        plt.show()


def get_cli_arguments():
    parser = argparse.ArgumentParser(description="Compare AV trajectories and tune smoothing filters")
    parser.add_argument("--data", type=Path, required=True, help="Path to AV trajectory data folder")
    parser.add_argument("--save", action="store_true", help="Save plots as PDF files")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--coords", type=str, default="local", choices=["local", "global"], help="Plot coordinates")
    parser.add_argument("--tune", "-t", action="store_true", help="Enable smoothing parameter tuning")
    parser.add_argument("--filter", "-f", default="gaussian",
                        choices=["gaussian", "savitzky_golay"], help="Filter type")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    evaluate_av_data(args)
