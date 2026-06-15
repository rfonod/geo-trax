#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
plot.py - Trajectory and Distribution Plotting Tool

Generates trajectory plots and kinematic, class, and dimension distribution charts from geo-trax
detection and tracking output. Reads pixel-space tracking results (.txt), georeferenced results
(.csv), or paired results from video files. For georeferenced trajectories, plots can be overlaid
on plain orthophotos or pre-rendered lane segmentation overlay images.

Usage:
  geotrax plot <input> [options]

Arguments:
  input : Path to a video file, .txt (pixel-space results), .csv (georeferenced results),
          or a folder containing any of the above.

Options:
  --help, -h       : Show this help message and exit.
  --cfg, -c        : Path to a custom pipeline config file. Defaults to the bundled config;
                     run 'geotrax config show' to view it or 'geotrax config copy' to customize.
  --log-path, -lp  : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  --verbose, -v    : Set verbosity to INFO (default: WARNING).

Plot Background Options:
  --ortho-folder, -of <path>         : Path to the folder with orthophoto images (.png) used as
                                       plot backgrounds for georeferenced trajectories. Defaults to
                                       cfg -> folders -> ortho_folder, then 'ORTHOPHOTOS' at the same
                                       level as 'PROCESSED' in 'input'.
  --segmentation-folder, -osf <path> : Path to the folder containing lane segmentation CSV files
                                       (used during georeferencing for lane assignment) and,
                                       when --plot-segmentations is enabled, the corresponding
                                       overlay PNG files used as plot backgrounds. Defaults to
                                       cfg -> folders -> segmentation_folder, then '<ortho-folder>/segmentations'.

Plotting Options (the --plot-* names are shared with 'geotrax batch'):
  --plot-save / --no-plot-save, -ps  : Save plots as PDF files. Defaults to cfg -> plotting -> save.
  --plot-show / --no-plot-show, -psh : Show plots interactively. Defaults to cfg -> plotting -> show.
  --plot-aggregate, -pa              : When the input is a folder, merge trajectories sharing the
                                       same location ID into a single plot per location.
                                       Defaults to cfg -> plotting -> aggregate.
  --plot-points, -pp                 : Plot discrete trajectory points instead of continuous lines.
                                       Defaults to cfg -> plotting -> plot_points.
  --plot-segmentations, -pseg        : Produce an additional trajectory plot overlaid on the lane
                                       segmentation overlay PNG (from --segmentation-folder),
                                       alongside the standard plain-orthophoto plot. Overlay PNGs
                                       must be pre-generated with tools/viz_segmentations.py.
                                       Defaults to cfg -> plotting -> use_segmentations.
  --id, -i                           : Vehicle ID to print/plot in detail (single-file input only)
                                       [default: 0].
  --plot-class-filter, -pcf <int> [<int> ...] : Vehicle class IDs to exclude from plots.
                                       Defaults to cfg -> plotting -> class_filter.

Examples:
1. Plot results for a single video file (pixel-space only):
   geotrax plot /path/to/video.mp4

2. Plot georeferenced trajectories from a CSV file, overlaid on the orthophoto:
   geotrax plot /path/to/results/video.csv -of /path/to/orthophotos/

3. Aggregate and plot all results from a folder:
   geotrax plot /path/to/videos/ -pa

4. Plot with lane segmentation overlays as backgrounds, excluding buses:
   geotrax plot /path/to/videos/ -of /path/to/orthophotos/ -osf /path/to/segmentations/ -pseg -pcf 1
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from geotrax.utils.cli_utils import DEFAULT_CFG, add_common_args
from geotrax.utils.config_utils import backfill_args_from_config, load_config_all
from geotrax.utils.constants import (
    ACC_THRESHOLD_ALERT,
    RESULTS_FORMATS,
    SPEED_THRESHOLD_ALERT,
    VIDEO_FORMATS,
)
from geotrax.utils.data_utils import PlotColors
from geotrax.utils.file_utils import detect_delimiter, determine_location_id, get_ortho_folder
from geotrax.utils.logging_utils import setup_logger

colors = PlotColors()

def generate_plots(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Generate plots for the input file or directory.
    """
    config = load_config_all(args, logger)['main']
    plot_cfg = config['plotting']
    folders = config['folders']
    backfill_args_from_config(args, {
        'save': plot_cfg['save'],
        'show': plot_cfg['show'],
        'aggregate': plot_cfg['aggregate'],
        'points': plot_cfg['plot_points'],
        'segmentations': plot_cfg['use_segmentations'],
        'class_filter': plot_cfg['class_filter'],
        'ortho_folder': Path(folders['ortho_folder']) if folders['ortho_folder'] else None,
        'segmentation_folder': Path(folders['segmentation_folder']) if folders['segmentation_folder'] else None,
    })
    colors.set_colors(plot_cfg['colors'])
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
    filepath_img, filepath_geo, filepath_ortho, filepath_seg, location_id = get_filepaths(file, ortho_folder, config, logger)
    if filepath_img is None and filepath_geo is None:
        logger.warning(f"No tracking results found for {file.stem}. Skipping...")
        return

    df_img, df_geo, coordinates = read_trajectory_data(filepath_img, filepath_geo, config, logger)
    if not config['args'].aggregate or (df_geo is not None and 'Drone_ID' in df_geo.columns):
        plot_data((df_img, df_geo), (filepath_img, filepath_geo, filepath_ortho, filepath_seg), coordinates, config, logger)
    else:
        aggregate_data(file, df_img, df_geo, location_id, data_at_location_id, filepath_img, filepath_geo, coordinates, filepath_ortho, filepath_seg)


def aggregate_data(file: Path, df_img: pd.DataFrame, df_geo: pd.DataFrame, location_id: str, data_at_location_id: dict, filepath_img: Path, filepath_geo: Path, coordinates: tuple, filepath_ortho: Path, filepath_seg: Path) -> None:
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
            'filepath_seg': filepath_seg,
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
        valid_img = [df for df in data['df_img_list'] if df is not None]
        df_img = pd.concat(valid_img, ignore_index=True) if valid_img else None
        valid_geo = [df for df in data['df_geo_list'] if df is not None]
        df_geo = pd.concat(valid_geo, ignore_index=True) if valid_geo else None
        filepath_img = data['filepath_img_base'] / f"{data['filepath_img_file']}.txt" if df_img is not None else None
        filepath_geo = data['filepath_geo_base'] / f"{data['filepath_geo_file']}.csv" if df_geo is not None else None
        plot_data((df_img, df_geo), (filepath_img, filepath_geo, data['filepath_ortho'], data['filepath_seg']), data['coordinates'], config, logger)


def plot_data(dfs: tuple, filepaths: tuple, coordinates: tuple, config: dict, logger: logging.Logger) -> None:
    """
    Plot the trajectory data, speed and acceleration distributions, class distribution, and vehicle dimensions distribution.
    """
    df_img, df_geo = dfs
    filepath_img, filepath_geo = filepaths[:2]
    args = config['args']

    n_steps = 1  # plot_trajectories
    if args.id > 0 and df_geo is not None:
        n_steps += 1  # plot_kinematics_for_vehicle_id
    elif df_geo is not None:
        n_steps += 5  # speed, acceleration, joint, class, dimensions
    elif df_img is not None:
        n_steps += 1  # dimensions

    name = filepath_geo.name if filepath_geo else (filepath_img.name if filepath_img else 'unknown')
    _bar_w = max(10, shutil.get_terminal_size().columns - 88)
    pbar = tqdm(total=n_steps, unit='plot', colour='magenta', leave=True,
                desc=f'{name} - plotting            ',
                bar_format=f'{{l_bar}}{{bar:{_bar_w}}}{{r_bar}}')

    pbar.set_postfix_str('trajectories')
    plot_trajectories((df_img, df_geo), coordinates, filepaths, config, logger)
    pbar.update()

    if args.id > 0:
        pbar.set_postfix_str('kinematics for vehicle')
        plot_kinematics_for_vehicle_id(df_geo, filepath_geo, config, logger)
        pbar.update()
    elif df_geo is not None:
        pbar.set_postfix_str('speed distribution')
        plot_kinematic_distribution(df_geo, filepath_geo, config, logger, 'speed')
        pbar.update()
        pbar.set_postfix_str('acceleration distribution')
        plot_kinematic_distribution(df_geo, filepath_geo, config, logger, 'acceleration')
        pbar.update()
        pbar.set_postfix_str('joint kinematic distribution')
        plot_kinematic_distribution_jointly(df_geo, filepath_geo, config, logger)
        pbar.update()
        pbar.set_postfix_str('class distribution')
        plot_class_distribution(df_geo, filepath_geo, config, logger)
        pbar.update()
        pbar.set_postfix_str('vehicle dimensions')
        plot_vehicle_dimensions_distribution(df_geo, filepath_geo, config, 'GEO', logger)
        pbar.update()
    elif df_img is not None:
        pbar.set_postfix_str('vehicle dimensions')
        plot_vehicle_dimensions_distribution(df_img, filepath_img, config, 'IMG', logger)
        pbar.update()

    pbar.set_postfix_str('done')
    pbar.close()


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
            logger.critical(f"No valid video files ({VIDEO_FORMATS}) or result files ({RESULTS_FORMATS}) found in the input folder {input_path}")
            sys.exit(1)
    return files


def get_filepaths(file: Path, ortho_folder: Union[Path, None], config: dict, logger: logging.Logger) -> tuple:
    """
    Get the file paths for the image, geo, orthophoto, and segmentation overlay files.
    """
    filepath_img, filepath_geo, filepath_ortho, filepath_seg = None, None, None, None
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
        filepath_ortho = ortho_folder / f"{location_id}.png"
        if config['args'].segmentations:
            seg_folder = config['args'].segmentation_folder or ortho_folder / 'segmentations'
            filepath_seg = seg_folder / f"{location_id}.png"

    return filepath_img, filepath_geo, filepath_ortho, filepath_seg, location_id


def read_trajectory_data(filepath_img: Path, filepath_geo: Path, config: dict, logger: logging.Logger) -> tuple:
    """
    Read the tracking results in image and geo coordinates.
    """
    df_img, df_geo = None, None
    coordinates_img, coordinates_geo = None, None
    if filepath_img:
        try:
            delimiter = detect_delimiter(filepath_img)
            df_img = pd.read_csv(filepath_img, sep=delimiter, header=None)
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
            filepath_seg = filepaths[3] if len(filepaths) > 3 else None
            for coordinate, (x_key, y_key) in coordinates[i].items():
                plot_trajectories_in_given_coordinates(df, coordinate, x_key, y_key, filepaths[i], None, config, logger)
                if "Orthophoto" in coordinate and filepath_ortho:
                    plot_trajectories_in_given_coordinates(df, coordinate, x_key, y_key, filepaths[i], filepath_ortho, config, logger)
                if "Orthophoto" in coordinate and filepath_seg:
                    if not filepath_seg.exists():
                        logger.warning(
                            f"Segmentation overlay PNG not found: {filepath_seg}. "
                            f"Generate it with: python tools/viz_segmentations.py "
                            f"{config['args'].ortho_folder} -sf {filepath_seg.parent}"
                        )
                    else:
                        plot_trajectories_in_given_coordinates(df, coordinate, x_key, y_key, filepaths[i], filepath_seg, config, logger, is_seg=True)


def plot_trajectories_in_given_coordinates(df: pd.DataFrame, coordinates: str, x_key: str, y_key: str, filepath: Path, filepath_ortho: Union[Path, None], config: dict, logger: logging.Logger, is_seg: bool = False) -> None:
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
                logger.warning(f"Could not read orthophoto '{filepath_ortho}': {str(e)}")
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
            background_label = 'on segmentation overlay' if is_seg else 'on orthophoto'
            save_or_show_plot(coordinates + f' {background_label}', filepath, args, logger, contains_raster=True)


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
    save_or_show_plot('Speed_and_acceleration_distribution', filepath, args, logger)


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

    order = [cls for cls in config['class_names'].values() if cls in df['Vehicle_Class'].unique()]
    plt.figure()
    sns.countplot(data=df, x='Vehicle_Class', order=order, edgecolor='black', hue='Vehicle_Class')
    position = {category: idx for idx, category in enumerate(order)}
    value_counts = df['Vehicle_Class'].value_counts()
    for category, count in value_counts.items():
        if category in position:
            plt.text(position[category], count, str(count), ha='center', va='bottom', fontsize=config['plotting']['savefig_font_size'] - 3 if args.save else 10)
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
    if args.show:
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


def default_plot_args(**overrides) -> argparse.Namespace:
    """
    Build an args Namespace carrying plot.py's own defaults, applying any overrides.

    This keeps plot.py the single source of truth for the fields ``generate_plots`` reads
    (including non-CLI defaults like ``id=0``), so callers such as ``geotrax batch`` don't
    hand-maintain a parallel argument list that can silently drift.
    """
    defaults = {
        'input': None,
        'save': None,
        'show': None,
        'cfg': DEFAULT_CFG,
        'log_path': None,
        'verbose': False,
        'aggregate': None,
        'ortho_folder': None,
        'segmentation_folder': None,
        'segmentations': None,
        'id': 0,
        'points': None,
        'class_filter': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def add_plotting_args(group, dest_prefix: str = '') -> None:
    """
    Register the shared plotting CLI flags on the given argparse group.

    The flag names (``--plot-save``, ``--plot-show``, ...) and short options are identical for
    ``geotrax plot`` and ``geotrax batch``; the ``--plot-`` prefix distinguishes them from the
    visualization flags (``--save``, ``--show``, ...) in batch's combined parser and maps each
    to the cfg -> plotting section. ``dest_prefix='plot_'`` is used by batch so the resulting
    attribute names don't collide with the visualization ones. Every flag defaults to ``None``
    and is backfilled from the config.
    """
    group.add_argument("--plot-save", "-ps", dest=f"{dest_prefix}save", action=argparse.BooleanOptionalAction, default=None,
                       help="Save the plots as .pdf files. Defaults to cfg -> plotting -> save.")
    group.add_argument("--plot-show", "-psh", dest=f"{dest_prefix}show", action=argparse.BooleanOptionalAction, default=None,
                       help="Show plots in an interactive window. Defaults to cfg -> plotting -> show.")
    group.add_argument("--plot-aggregate", "-pa", dest=f"{dest_prefix}aggregate", action=argparse.BooleanOptionalAction, default=None,
                       help="When the input is a folder, merge trajectories from all videos sharing the same location ID into a single plot per location. Defaults to cfg -> plotting -> aggregate.")
    group.add_argument("--plot-points", "-pp", dest=f"{dest_prefix}points", action=argparse.BooleanOptionalAction, default=None,
                       help="Plot discrete trajectory points instead of connected lines. Defaults to cfg -> plotting -> plot_points.")
    group.add_argument("--plot-segmentations", "-pseg", dest=f"{dest_prefix}segmentations", action=argparse.BooleanOptionalAction, default=None,
                       help="Produce an additional trajectory plot overlaid on the lane segmentation overlay PNG (from --segmentation-folder), alongside the standard plain-orthophoto plot. Requires pre-generated overlays (run: python tools/viz_segmentations.py <ortho_folder>/). Defaults to cfg -> plotting -> use_segmentations.")
    group.add_argument("--plot-class-filter", "-pcf", dest=f"{dest_prefix}class_filter", type=int, nargs="+", default=None,
                       help="Vehicle class IDs to exclude from plots (e.g. 1 2). Defaults to cfg -> plotting -> class_filter.")


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Trajectory and distribution plotting tool.")

    parser.add_argument("input", type=Path, help="Path to a video file, a .txt/.csv results file, or a folder containing any of these.")

    optional = parser.add_argument_group('Optional arguments')
    add_common_args(optional)

    georef = parser.add_argument_group('Plot background arguments')
    georef.add_argument("--ortho-folder", "-of", type=Path, default=None, help="Path to the folder with orthophoto images (.png) used as plot backgrounds for georeferenced trajectories. Defaults to cfg -> folders -> ortho_folder, then 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.")
    georef.add_argument("--segmentation-folder", "-osf", type=Path, default=None, help="Path to the folder containing lane segmentation CSV files (used during georeferencing for lane assignment) and, when --plot-segmentations is enabled, the corresponding overlay PNG files used as plot backgrounds. Defaults to cfg -> folders -> segmentation_folder, then '<ortho-folder>/segmentations'.")

    plotting = parser.add_argument_group('Plotting arguments')
    add_plotting_args(plotting)
    plotting.add_argument("--id", "-i", type=int, default=0, help="Vehicle ID to print/plot in detail (only for non-folder input) [default: 0]")

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(__name__, args.verbose, args.log_path)

    generate_plots(args, logger)


if __name__ == '__main__':
    main()
