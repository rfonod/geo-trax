#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
plot.py - Trajectory plotting tool

This script generates plots for the input video file or directory with video or .txt/.csv files.
The script reads the tracking results in image and/or geo coordinates, and plots the trajectory data,
speed and acceleration distributions, class distribution, and vehicle dimensions distribution.

Usage:
  python plot.py input [options]

Arguments:
  input: Path to an individual video file, .txt/.csv file, or folder with video or .txt/.csv files.

Options:
  --save, -s: Save the plots as .pdf files [default: False]
  --no-show, -ns: Do not show the plots [default: False]
  --cfg, -c: Path to the main geo-trax configuration file [default: cfg/default.yaml]
  --log-file, -lf: Filename to save detailed logs. Saved in the 'logs' folder. [default: plot.log]
  --verbose, -v: Set print verbosity level to INFO (default: WARNING)
  --aggregate, -a: Aggregate the input data (folder input) per location ID (intersection) [default: False]
  --ortho-folder, -of: Custom path to the folder with orthophotos (.png). Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.
  --segmentations, -seg: Use the segmented orthophotos [default: False]
  --id, -i: Vehicle ID to print/plot in detail (only for non-folder input) [default: None]
  --points, -p: Plot the trajectory points [default: False]
  --class-filter, -cf: Exclude specified vehicle classes [default: None]

Examples:
1. Plot the trajectory data for a video file:
   python plot.py /path/to/video.mp4

2. Plot the trajectory data using the georeferenced tracking results:
   python plot.py /path/to/results/video.csv

3. Plot the trajectory data for a folder with video files and aggregate the data:
   python plot.py /path/to/videos -a

4. Plot the trajectory data for a folder with video files, use the segmented orthophotos, and filter out vehicle class 1:
   python plot.py /path/to/videos -seg -cf 1 -of /path/to/orthophotos
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.utils import (
    PlotColors,
    detect_delimiter,
    determine_location_id,
    get_ortho_folder,
    load_config_all,
    setup_logger,
)

VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}
RESULTS_FORMATS = {'.txt', '.csv'}

ACC_THRESHOLD_ALERT = 5 # [m/s^2]
SPEED_THRESHOLD_ALERT = 90 # [km/h]
colors = PlotColors()

def generate_plots(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Generate plots for the input file or directory.
    """
    config = load_config_all(args, logger)['main']
    colors.set_colors(config['plotting']['colors'])
    files = determine_files_to_process(args.input, config['plotting']['skip_filenames_with'], logger)
    ortho_folder = get_ortho_folder(args.input, args.ortho_folder, logger, critical=False)

    data_at_location_id = {}
    try:
        for file in files:
            process_file(file, ortho_folder, data_at_location_id, config, logger)
    except KeyboardInterrupt:
        logger.notice("Keyboard interrupt detected. Exiting...")
        return

    if args.aggregate:
        handle_aggregation(data_at_location_id, config, logger)


def process_file(file: Path, ortho_folder: Union[Path, None], data_at_location_id: dict, config: dict, logger: logging.Logger) -> None:
    """
    Process an individual file and generate plots or aggregate data.
    """
    filepath_img, filepath_geo, filepath_ortho, location_id = get_filepaths(file, ortho_folder, config, logger)
    if filepath_img is None and filepath_geo is None:
        logger.warning(f"No tracking results found for {file.stem}. Skipping...")
        return

    df_img, df_geo, coordinates = read_trajectory_data(filepath_img, filepath_geo, config, logger)
    if not config['args'].aggregate or (df_geo is not None and 'Drone_ID' in df_geo.columns):
        plot_data((df_img, df_geo), (filepath_img, filepath_geo, filepath_ortho), coordinates, config, logger)
    else:
        aggregate_data(file, df_img, df_geo, location_id, data_at_location_id, filepath_img, filepath_geo, coordinates, filepath_ortho)


def aggregate_data(file: Path, df_img: pd.DataFrame, df_geo: pd.DataFrame, location_id: str, data_at_location_id: dict, filepath_img: Path, filepath_geo: Path, coordinates: tuple, filepath_ortho: Path) -> None:
    """
    Aggregate data for a specific location ID.
    """
    if df_img is not None:
        df_img['Vehicle_ID'] = file.stem + '_' + df_img['Vehicle_ID'].astype(str)
    if df_geo is not None:
        df_geo['Vehicle_ID'] = file.stem + '_' + df_geo['Vehicle_ID'].astype(str)

    if location_id not in data_at_location_id:
        data_at_location_id[location_id] = {
            'df_img_list': [],
            'df_geo_list': [],
            'filepath_img_base': filepath_img.parent if filepath_img else '',
            'filepath_geo_base': filepath_geo.parent if filepath_geo else '',
            'filepath_img_file': 'agg',
            'filepath_geo_file': 'agg',
            'coordinates': coordinates,
            'filepath_ortho': filepath_ortho,
        }
    data_at_location_id[location_id]['df_img_list'].append(df_img)
    data_at_location_id[location_id]['df_geo_list'].append(df_geo)
    if filepath_img:
        data_at_location_id[location_id]['filepath_img_file'] += '_' + filepath_img.stem
    if filepath_geo:
        data_at_location_id[location_id]['filepath_geo_file'] += '_' + filepath_geo.stem


def handle_aggregation(data_at_location_id: dict, config: dict, logger: logging.Logger) -> None:
    """
    Handle the aggregation of data per location ID and generate plots.
    """
    if config['args'].id > 0:
        logger.warning("The vehicle ID argument is not supported when aggregating data per location ID. Ignoring the vehicle ID argument.")
        config['args'].id = 0
    for location_id, data in data_at_location_id.items():
        logger.notice(f"Aggregating data for location ID {location_id}")
        df_img = pd.concat(data['df_img_list'], ignore_index=True) if data['df_img_list'][0] is not None else None
        df_geo = pd.concat(data['df_geo_list'], ignore_index=True) if data['df_geo_list'][0] is not None else None
        filepath_img = data['filepath_img_base'] / f"{data['filepath_img_file']}.txt" if df_img is not None else None
        filepath_geo = data['filepath_geo_base'] / f"{data['filepath_geo_file']}.csv" if df_geo is not None else None
        plot_data((df_img, df_geo), (filepath_img, filepath_geo, data['filepath_ortho']), data['coordinates'], config, logger)


def plot_data(dfs: tuple, filepaths: tuple, coordinates: tuple, config: dict, logger: logging.Logger) -> None:
    """
    Plot the trajectory data, speed and acceleration distributions, class distribution, and vehicle dimensions distribution.
    """
    df_img, df_geo = dfs
    filepath_img, filepath_geo = filepaths[:2]

    plot_trajectories((df_img, df_geo), coordinates, filepaths, config, logger)

    if config['args'].id > 0:
        plot_kinematics_for_vehicle_id(df_geo, filepath_geo, config, logger)
    elif df_geo is not None:
        plot_kinematic_distribution(df_geo, filepath_geo, config, logger, 'speed')
        plot_kinematic_distribution(df_geo, filepath_geo, config, logger, 'acceleration')
        plot_kinematic_distribution_jointly(df_geo, filepath_geo, config, logger)
        plot_class_distribution(df_geo, filepath_geo, config, logger)
        plot_vehicle_dimensions_distribution(df_geo, filepath_geo, config, 'GEO', logger)
    elif df_img is not None:
        plot_vehicle_dimensions_distribution(df_img, filepath_img, config, 'IMG', logger)


def determine_files_to_process(input_path: Path, skip_filenames_with: list, logger: logging.Logger) -> list:
    """
    Determine the files to process based on the input path and the filenames to skip.
    """
    if not input_path.exists():
        logger.critical(f"File or directory '{input_path}' not found.")
        sys.exit(1)

    files = [input_path]
    if input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in VIDEO_FORMATS or f.suffix in RESULTS_FORMATS]
        files = [f for f in files if not any(word in f.stem for word in skip_filenames_with)]
        files = sorted(files)
        if len(files) == 0:
            logger.error(f"No valid video files ({VIDEO_FORMATS}) or result files ({RESULTS_FORMATS}) found in the input folder {input_path}")
            sys.exit(1)
    return files


def get_filepaths(file: Path, ortho_folder: Union[Path, None], config: dict, logger: logging.Logger) -> tuple:
    """
    Get the file paths for the image, geo, and orthophoto files.
    """
    filepath_img, filepath_geo, filepath_ortho = None, None, None
    if file.suffix.lower() in VIDEO_FORMATS:
        filepath_img = file.parent / 'results' / file.name.replace(file.suffix, '.txt')
        filepath_geo = file.parent / 'results' / file.name.replace(file.suffix, '.csv')
        if not filepath_img.is_file():
            filepath_img = None
        if not filepath_geo.is_file():
            filepath_geo = None
    elif file.suffix == '.txt' and file.exists():
        filepath_img = file
    elif file.suffix == '.csv' and file.exists():
        filepath_geo = file

    location_id = determine_location_id(file, logger)
    if filepath_geo and ortho_folder:
        subfolder = 'segmentations' if config['args'].segmentations else ''
        filepath_ortho = ortho_folder / subfolder / f"{location_id}.png"

    return filepath_img, filepath_geo, filepath_ortho, location_id


def read_trajectory_data(filepath_img: Path, filepath_geo: Path, config: dict, logger: logging.Logger) -> tuple:
    """
    Read the tracking results in image and geo coordinates.
    """
    df_img, df_geo = None, None
    coordinates_img, coordinates_geo = None, None
    if filepath_img:
        try:
            delimiter = detect_delimiter(filepath_img)
            df_img = pd.read_csv(filepath_img, sep=delimiter, header=None, skiprows=0)
            coordinates_img = {'Unstabilized image coordinates': ['X_unstabilized', 'Y_unstabilized']}
            all_columns = [
                'Frame_ID',
                'Vehicle_ID',
                'X_unstabilized',
                'Y_unstabilized',
                'W_unstabilized',
                'H_unstabilized',
                'X_stabilized',
                'Y_stabilized',
                'W_stabilized',
                'H_stabilized',
                'Vehicle_Class',
                'Confidence',
                'Vehicle_Length',
                'Vehicle_Width',
            ]
            if df_img.shape[1] == 14:
                df_img.columns = all_columns
                coordinates_img['Stabilized image coordinates'] = ['X_stabilized', 'Y_stabilized']
                coordinates_img['Stabilized image coordinates'] = ['X_stabilized', 'Y_stabilized']
            elif df_img.shape[1] == 10:
                df_img.columns = all_columns[:6] + all_columns[10:]
            else:
                logger.error(f"Invalid number of columns in the tracking results file {filepath_img}")
                raise ValueError("Invalid number of columns")

            df_img['Vehicle_Length'] = df_img['Vehicle_Length'].astype(float)
            df_img['Vehicle_Width'] = df_img['Vehicle_Width'].astype(float)
            df_img['Vehicle_Class'] = df_img['Vehicle_Class'].astype(int)
        except Exception as e:
            logger.error(f"Error reading the tracking results in image coordinates: {str(e)}")
            df_img, coordinates_img = None, None
        else:
            df_img = filter_classes(df_img, config['args'].class_filter)
            df_img['Vehicle_Class'] = df_img['Vehicle_Class'].map(config['class_names'])

    if filepath_geo:
        try:
            df_geo = pd.read_csv(filepath_geo, low_memory=False)
            coordinates_geo = {
                'Orthophoto image coordinates': ['Ortho_X', 'Ortho_Y'],
                'Local planar coordinates': ['Local_X', 'Local_Y'],
                'Geographic coordinates': ['Longitude', 'Latitude'],
            }
        except Exception as e:
            logger.error(f"Error reading the tracking results in geo coordinates: {str(e)}")
            df_geo, coordinates_geo = None, None
        else:
            df_geo = filter_classes(df_geo, config['args'].class_filter)
            df_geo['Vehicle_Class'] = df_geo['Vehicle_Class'].astype(int).map(config['class_names'])

    return df_img, df_geo, (coordinates_img, coordinates_geo)


def filter_classes(df: pd.DataFrame, class_filter: list) -> pd.DataFrame:
    """
    Filter the data frame by excluding the specified vehicle classes.
    """
    if class_filter:
        df = df[~df['Vehicle_Class'].isin([int(c) for c in class_filter])]
    return df


def plot_trajectories(dfs: tuple, coordinates: tuple, filepaths: tuple, config: dict, logger: logging.Logger) -> None:
    """
    Plot the trajectory data in image and geo coordinates.
    """
    for i, df in enumerate(dfs):
        if df is not None:
            filepath_ortho = filepaths[2]
            for coordinate, (x_key, y_key) in coordinates[i].items():
                plot_trajectories_in_given_coordinates(df, coordinate, x_key, y_key, filepaths[i], None, config, logger)
                if "Orthophoto" in coordinate and filepath_ortho:
                    plot_trajectories_in_given_coordinates(df, coordinate, x_key, y_key, filepaths[i], filepath_ortho, config, logger)


def plot_trajectories_in_given_coordinates(df: pd.DataFrame, coordinates: str, x_key: str, y_key: str, filepath: Path, filepath_ortho: Union[Path, None], config: dict, logger: logging.Logger) -> None:
    """
    Plot the trajectories in the given coordinates.
    """
    args = config['args']
    lw = 0.6 if args.save else 1
    alpha_max = 0.45 if args.save else 0.35
    alpha_min = 0.225 if args.save else 0.125
    alpha_step = 0.075
    if df is not None:
        if filepath_ortho:
            try:
                ortho = plt.imread(filepath_ortho)
            except Exception as e:
                logger.warning(f"Error reading the orthophoto {filepath_ortho}: {str(e)}")
                ortho = None
        else:
            ortho = None

        df = df.copy()
        df['Vehicle_ID_orig'] = df['Vehicle_ID']
        if 'Drone_ID' in df.columns:
            df['Vehicle_ID'] = 'D' + df['Drone_ID'].astype(str) + '_' + df['Vehicle_ID'].astype(str)
        x = df.groupby('Vehicle_ID')[x_key].apply(list)
        y = df.groupby('Vehicle_ID')[y_key].apply(list)

        plt.figure()
        if ortho is not None:
            plt.imshow(ortho)
            plt.axis('off')

        source_label_mapping = dict()
        for vehicle_id, (x_i, y_i) in zip(x.index, zip(x, y)):
            if args.id == 0 or vehicle_id != args.id:
                if not isinstance(vehicle_id, str):
                    plt.plot(x_i, y_i, color='black', linewidth=0.5)
                    if args.points:
                        plt.scatter(x_i, y_i, color='black', s=0.5)
                else:
                    label = vehicle_id.split('_')[0]
                    label_legend = label if label not in source_label_mapping else None
                    source_label_mapping.setdefault(label, len(source_label_mapping))
                    i = source_label_mapping[label]
                    alpha = max(alpha_max - alpha_step*i, alpha_min)
                    color = colors.get_color(i)
                    plt.plot(x_i, y_i, color=color, lw=lw, alpha=alpha, label=label_legend)
                    if args.points:
                        plt.scatter(x_i, y_i, color=color, s=0.4, alpha=alpha)
        if args.id > 0:
            vehicle = df[df['Vehicle_ID_orig'] == args.id]
            plt.plot(vehicle[x_key], vehicle[y_key], color='red', linewidth=2*lw)
            if args.points:
                plt.scatter(vehicle[x_key], vehicle[y_key], color='red', s=4)

        if ortho is None:
            if "image" in coordinates:
                plt.gca().invert_yaxis()
            plt.title('' if args.save else f'{coordinates} for: {filepath.stem}')
            plt.xlabel(get_xlabel(x_key), fontsize=config['plotting']['savefig_font_size'] if args.save else None)
            plt.ylabel(get_ylabel(y_key), fontsize=config['plotting']['savefig_font_size'] if args.save else None)
            plt.tick_params(axis='x', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
            plt.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
            if len(source_label_mapping) > 1:
                plt.legend(loc='best', fontsize=config['plotting']['savefig_font_size'] - 1 if args.save else None)
            save_or_show_plot(coordinates, filepath, args, logger)
        else:
            if len(source_label_mapping) > 1:
                plt.legend(loc='best', fontsize=config['plotting']['savefig_font_size'] - 5 if args.save else None)
            save_or_show_plot(coordinates + ' on orthophoto', filepath, args, logger, contains_raster=True)


def plot_kinematic_distribution(df: pd.DataFrame, filepath: Path, config: dict, logger: logging.Logger, kinematic_type: str) -> None:
    """
    Plot the speed or acceleration distribution.
    """
    args = config['args']
    df = df.copy()

    if kinematic_type == 'speed':
        df = df[df['Vehicle_Speed'] > config['plotting']['stationary_speed_cutoff']]
        if df.empty:
            logger.warning(f"No data available for speed distribution plot for {filepath.stem}")
            return
        y_column = 'Vehicle_Speed'
        y_label = 'Speed [km/h]'
        cut = 2
    else:  # acceleration
        y_column = 'Vehicle_Acceleration'
        y_label = 'Acceleration [m/s$^2$]'
        cut = 0

    plt.figure()
    sns.violinplot(
        data=df,
        x='Vehicle_Class',
        y=y_column,
        inner='quartile',
        split=True,
        hue='Vehicle_Class',
        saturation=0.75,
        order=[cls for cls in config['class_names'].values() if cls in df['Vehicle_Class'].unique()],
        cut=cut
    )
    plt.title('' if args.save else f'{kinematic_type.capitalize()} distribution for {filepath.stem.replace("_", " & ")}')
    plt.xlabel('' if args.save else 'Vehicle class')
    plt.ylabel(y_label, fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    save_or_show_plot(f'{kinematic_type.capitalize()}_distribution', filepath, args, logger)
    report_high_value_instances(df, kinematic_type[:5], logger)


def plot_kinematic_distribution_jointly(df: pd.DataFrame, filepath: Path, config: dict, logger: logging.Logger) -> None:
    """
    Plot the speed and acceleration distribution in one plot.
    """
    args = config['args']
    df = df.copy()
    df = df[df['Vehicle_Speed'] > config['plotting']['stationary_speed_cutoff']]
    if df.empty:
        logger.warning(f"No data available for speed distribution plot for {filepath.stem}")
        return
    df1, df2 = df.copy(), df.copy()
    df1['Kinematics_Type'] = 'Speed'
    df1['Vehicle_Acceleration'] = np.nan
    df2['Kinematics_Type'] = 'Acceleration'
    df2['Vehicle_Speed'] = np.nan
    df_new = pd.concat([df1, df2], ignore_index=True)
    df_new.loc[df_new['Kinematics_Type'] == 'Speed', 'Kinematics_Value'] = df['Vehicle_Speed'].values
    df_new.loc[df_new['Kinematics_Type'] == 'Acceleration', 'Kinematics_Value'] = df['Vehicle_Acceleration'].values

    plt.figure()
    palette = sns.color_palette(["olivedrab", "brown"]) # "Accent"
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    sns.violinplot(
        ax=ax1,
        data=df_new,
        x="Vehicle_Class",
        y="Vehicle_Speed",
        hue='Kinematics_Type',
        split=True,
        gap=0.05,
        gridsize=300,
        order=[cls for cls in config['class_names'].values() if cls in df_new['Vehicle_Class'].unique()],
        saturation=0.75,
        inner='quartile',
        width=1,
        legend=True,
        palette=palette,
    )  # cut=0
    ax1.set_ylabel('Speed [km/h]', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    sns.violinplot(
        ax=ax2,
        data=df_new,
        x="Vehicle_Class",
        y="Vehicle_Acceleration",
        hue='Kinematics_Type',
        split=True,
        gap=0.05,
        gridsize=300,
        order=[cls for cls in config['class_names'].values() if cls in df_new['Vehicle_Class'].unique()],
        saturation=0.75,
        inner='quartile',
        width=1,
        legend=False,
        palette=palette,
    )  # cut=0)
    ax2.set_ylabel('Acceleration [m/s$^2$]', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    ax1.set_xlabel('' if args.save else 'Vehicle class')
    ax1.tick_params(axis='x', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    ax1.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    ax2.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.title('' if args.save else f'Speed and acceleration distribution for {filepath.stem.replace("_", " & ")}')
    plt.xlabel('' if args.save else 'Vehicle class')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=labels, title=None, loc='best', ncol=2, fontsize=config['plotting']['savefig_font_size'] - 1 if args.save else None)
    save_or_show_plot('Speed_and_accleration_distribution', filepath, args, logger)


def report_high_value_instances(df: pd.DataFrame, flag: str, logger: logging.Logger) -> None:
    """
    Report the instances where the speed or acceleration exceeds the threshold.
    """
    df = df.copy()
    columns = ['Vehicle_ID', 'Vehicle_Class', 'Vehicle_Acceleration', 'Vehicle_Speed', 'Traj_Count']
    if 'Timestamp' in df.columns:
        columns.insert(1, 'Timestamp')
    elif 'Local_Time' in df.columns:
        columns.insert(1, 'Local_Time')
    elif 'Frame_Number' in df.columns:
        columns.insert(1, 'Frame_Number')
    if 'Drone_ID' in df.columns:
        columns.insert(2, 'Drone_ID')
    threshold = SPEED_THRESHOLD_ALERT if flag == 'speed' else ACC_THRESHOLD_ALERT
    column = 'Vehicle_Speed' if flag == 'speed' else 'Vehicle_Acceleration'
    unit = 'km/h' if flag == 'speed' else 'm/s^2'
    df['Traj_Count'] = df.groupby('Vehicle_ID')['Vehicle_ID'].transform('count')
    df_high = df[df[column].abs() > threshold][columns]
    df_high['abs' + column] = df_high[column].abs()
    df_high = df_high.loc[df_high.groupby('Vehicle_ID')['abs' + column].idxmax()]
    df_high = df_high.sort_values(by='abs' + column, ascending=False).drop(columns=['abs' + column])

    if not df_high.empty:
        logger.warning(f"Threshold {column.lower()} of {threshold} ({unit}) violated for the following instances:" + "\n%s", repr(df_high))


def plot_class_distribution(df: pd.DataFrame, filepath: Path, config: dict, logger: logging.Logger) -> None:
    """
    Plot the class distribution.
    """
    args = config['args']
    df = df.copy()
    df = df.groupby('Vehicle_ID').first().reset_index()

    plt.figure()
    sns.countplot(data=df, x='Vehicle_Class', order=[cls for cls in config['class_names'].values() if cls in df['Vehicle_Class'].unique()], edgecolor='black', hue='Vehicle_Class')
    value_counts = df['Vehicle_Class'].value_counts()
    for i, category in config['class_names'].items():
        if category in value_counts:
            count = value_counts.get(category, 0)
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=config['plotting']['savefig_font_size'] - 3 if args.save else 10)
    plt.title('' if args.save else f'Vehicle class distribution for {filepath.stem.replace("_", " & ")}')
    plt.xlabel('' if args.save else 'Vehicle class')
    plt.ylabel('Count', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='x', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    save_or_show_plot('Class_distribution', filepath, args, logger)


def plot_vehicle_dimensions_distribution(df: pd.DataFrame, filepath: Path, config: dict, coordinates: str, logger: logging.Logger) -> None:
    """
    Plot the vehicle dimensions distribution.
    """
    args = config['args']
    df = df.copy()
    df = df.groupby('Vehicle_ID').first().reset_index()

    plt.figure()
    sns.boxplot(data=df, x='Vehicle_Class', y='Vehicle_Length', hue='Vehicle_Class',
        order=[cls for cls in config['class_names'].values() if cls in df['Vehicle_Class'].unique()], saturation=0.75, fliersize=2)
    plt.title('' if args.save else f'Vehicle length distribution for {filepath.stem.replace("_", " & ")}')
    plt.xlabel('' if args.save else 'Vehicle class')
    plt.ylabel(f'Vehicle length {"[m]" if coordinates == "GEO" else "[px]"}', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='x', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    save_or_show_plot('Vehicle_length_distribution', filepath, args, logger)

    plt.figure()
    sns.boxplot(data=df, x='Vehicle_Class', y='Vehicle_Width', hue='Vehicle_Class',
                order=[cls for cls in config['class_names'].values() if cls in df['Vehicle_Class'].unique()], saturation=0.75, fliersize=2)
    plt.title('' if args.save else f'Vehicle width distribution for {filepath.stem.replace("_", " & ")}')
    plt.xlabel('' if args.save else 'Vehicle class')
    plt.ylabel(f'Vehicle width {"[m]" if coordinates == "GEO" else "[px]"}', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='x', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.tick_params(axis='y', labelsize=config['plotting']['savefig_font_size'] if args.save else None)
    save_or_show_plot('Vehicle_width_distribution', filepath, args, logger)

    logger.info(f'Mean vehicle dimensions ({("in meters" if coordinates == "GEO" else "in pixels")}) for {filepath.stem.replace("_", " & ")}:')
    logger.info("\n%s", df.groupby('Vehicle_Class')[['Vehicle_Length', 'Vehicle_Width']].mean().to_string())


def plot_kinematics_for_vehicle_id(df: pd.DataFrame, filepath: Path, config: dict, logger: logging.Logger) -> None:
    """
    Plot the speed and acceleration for the specified vehicle ID.
    """
    args = config['args']
    df = df.copy()
    vehicle = df[df['Vehicle_ID'] == args.id]
    logger.info(f"Vehicle ID={args.id} in {filepath.stem}:\n{vehicle.describe()}")

    x_label = 'Elapsed time [s]'
    if 'Timestamp' in vehicle.columns:
        elapsed_time = pd.to_datetime(vehicle['Timestamp'])
        elapsed_time = (elapsed_time - elapsed_time.iloc[0]).dt.total_seconds()
    elif 'Local_Time' in vehicle.columns:
        elapsed_time = pd.to_datetime(vehicle['Local_Time'], format='%H:%M:%S.%f')
        elapsed_time = (elapsed_time - elapsed_time.iloc[0]).dt.total_seconds()
    elif 'Frame_Number' in vehicle.columns:
        elapsed_time = vehicle['Frame_Number']
        x_label = 'Frame #'
    else:
        logger.error(f"Neither 'Timestamp' nor 'Frame_Number' column found in {filepath.stem}")
        return

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(elapsed_time, vehicle['Vehicle_Speed'], color='black', linewidth=1, label='Speed')
    plt.grid()
    plt.xlim(elapsed_time.iloc[0], elapsed_time.iloc[-1])
    plt.title('' if args.save else f'Speed of vehicle ID={args.id} in {filepath.stem}')
    plt.xlabel(x_label, fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.ylabel('Speed [km/h]', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.subplot(1, 2, 2)
    plt.plot(elapsed_time, vehicle['Vehicle_Acceleration'], color='black', linewidth=1, label='Acceleration')
    plt.grid()
    plt.xlim(elapsed_time.iloc[0], elapsed_time.iloc[-1])
    plt.title('' if args.save else f'Acceleration of vehicle ID={args.id} in {filepath.stem}')
    plt.xlabel(x_label, fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    plt.ylabel('Acceleration [m/s$^2$]', fontsize=config['plotting']['savefig_font_size'] if args.save else None)
    save_or_show_plot(f'Speed_and_acceleration_of_id_{args.id}', filepath, args, logger)


def get_xlabel(key: str) -> str:
    if key in ['X_stabilized', 'X_unstabilized', 'Ortho_X']:
        return key.replace('_', ' ') + ' [px]'
    elif key == 'Longitude':
        return key.replace('_', ' ') + ' [deg]'
    else:
        return key.replace('_', ' ') + ' [m]'


def get_ylabel(key: str) -> str:
    if key in ['Y_stabilized', 'Y_unstabilized', 'Ortho_Y']:
        return key.replace('_', ' ') + ' [px]'
    elif key == 'Latitude':
        return key.replace('_', ' ') + ' [deg]'
    else:
        return key.replace('_', ' ') + ' [m]'


def save_or_show_plot(name: str, filepath: Path, args: argparse.Namespace, logger: logging.Logger, contains_raster: bool = False) -> None:
    """
    Save or show the plot.
    """
    if not args.no_show:
        plt.show()
    if args.save:
        img_dir = filepath.parent / 'plots'
        img_dir.mkdir(parents=True, exist_ok=True)
        img_filepath = img_dir / f"{filepath.stem}_{name.replace('(', '').replace(')', '').replace(' ', '_')}.pdf"
        if contains_raster:
            plt.savefig(img_filepath, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
        else:
            plt.savefig(img_filepath, bbox_inches='tight', pad_inches=0, transparent=False)
        logger.info(f"Plot saved as {img_filepath}")
    plt.close()


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Trajectory visualization tool.")

    # Required argument
    parser.add_argument("input", type=Path, help="Path to the video source or .txt/.csv file or folder with video or .csv/.txt files")

    # Main arguments
    parser.add_argument("--save", "-s", action="store_true", help="Save the plots as .pdf files [default: False]")
    parser.add_argument("--no-show", "-ns", action="store_true", help="Do not show the plots [default: False]")

    # Optional arguments
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    parser.add_argument("--log-file", "-lf", type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument("--verbose", "-v", action='store_true', help='Set print verbosity level to INFO (default: WARNING)')
    parser.add_argument("--aggregate", "-a", action="store_true", help="aggregate the data for the same intersection [default: False]")
    parser.add_argument("--ortho-folder", "-of", type=Path, default=None, help="Custom path to the folder with orthophotos (.png). Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.")
    parser.add_argument("--segmentations", "-seg", action="store_true", help="Use the segmented orthophotos [default: False]")
    parser.add_argument("--id", "-i", type=int, default=0, help="vehicle ID to print/plot in detail (only for non-folder input) [default: 0]")
    parser.add_argument("--points", "-p", action="store_true", help="plot the trajectory points [default: False]")
    parser.add_argument("--class-filter", "-cf", nargs="+", default=[], help="exclude specified vehicle classes [default: None]")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    generate_plots(args, logger)
