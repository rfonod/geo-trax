#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_master_frames.py - Master Frame Selection Tool for Georeferencing

This script identifies the optimal reference frames to serve as master frames for georeferencing
drone video footage. It analyzes flight log data, spatial coordinates, and object detection results
to select frames with minimal positional deviation and optimal object coverage characteristics.

The tool processes flight logs to extract reference frame coordinates, calculates distances to
mean hover locations, and applies selection criteria based on spatial proximity and object
detection coverage to identify the best master frames for each intersection location.

Usage:
  python tools/find_master_frames.py <input_folder> <output_folder> [options]

Arguments:
  input_folder : str
                 Path to the folder containing videos, flight logs, and optional detection/tracking results.
  output_folder : str
                 Path to the output folder to save results (e.g., ../ORTHOPHOTOS/master_frames).

Options:
  -h, --help            : Show this help message and exit.
  -s, --save            : bool, optional
                        Save extracted reference frame stats and best master frames list (default: False).
  -smf, --save-master-frames : bool, optional
                        Save best master frames extracted from videos (existing files overwritten) (default: False).
  -f, --force           : bool, optional
                        Force extraction of flight log data (do not use existing data) (default: False).
  -rf, --ref-frame <int> : int, optional
                        Custom reference frame used for stabilization/georeferencing (default: 0).
  -viz, --visualize     : bool, optional
                        Visualize best master frame selection (default: False).
  -sv, --save-viz       : bool, optional
                        Save best master frames visualization to PDF (default: False).
  -n, --best_n <int>    : int, optional
                        Number of reference frames to consider for best master frame selection (default: 20).
  -m, --match-pattern <str> : str, optional
                        Pattern to match files in folder (case-sensitive) (default: '??.CSV').
  -fe, --folders-exclude <str> [<str> ...] : list of str, optional
                        Folders to exclude from search
                        (e.g., 'results' for already georeferenced data) (default: ['results']).
  -b, --bounding-box-cols <int> [<int> ...] : list of int, optional
                        Columns of bounding box in detection/tracking results (default: [2, 3, 4, 5]).
  -tcrs, --target-crs <str> : str, optional
                        Target CRS for local coordinates (default: 'epsg:5186').
  -fw, --frame-width <int> : int, optional
                        Video frame width in pixels (default: 3840).
  -fh, --frame-height <int> : int, optional
                        Video frame height in pixels (default: 2160).

Examples:
1. Basic master frame selection with saved results and visualization:
   python tools/find_master_frames.py /path/to/PROCESSED /path/to/ORTHOPHOTOS/master_frames -s -smf -sv -f

2. Selection with custom parameters and tracking data:
   python tools/find_master_frames.py /path/to/PROCESSED /path/to/output --best_n 10 --ref-frame 5

3. Analysis with custom frame dimensions and CRS:
   python tools/find_master_frames.py /path/to/PROCESSED /path/to/output -fw 1920 -fh 1080 -tcrs epsg:32633

Input:
- PROCESSED folder containing drone flight logs in CSV format
- Video files (.MP4) corresponding to flight logs
- Optional detection/tracking results in 'results' subdirectories
- Flight logs should follow the pattern specified by --match-pattern

Output:
- reference_frame_stats.csv: Comprehensive statistics for all reference frames
- best_master_frames.csv: Selected optimal master frames for each location
- Master frame images: PNG files extracted from videos (if --save-master-frames)
- PDF visualization: Spatial distribution plots (if --save-viz)

Notes:
- Selection algorithm prioritizes frames closest to mean hover location
- Secondary criterion considers minimal object detection coverage area
- Converts geographic coordinates to local coordinate system for spatial analysis
- Supports visualization of spatial distribution and selection results
- Master frames are identified per intersection location ID
- Handles missing detection data gracefully with fallback criteria
"""

import argparse
import fnmatch
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.utils import detect_delimiter, determine_location_id

VIDEO_SUFFIX = '.MP4' # video file format to report in the output file


def find_master_frames(args: argparse.Namespace) -> None:
    """
    Find the best master frame for georeferencing.
    """
    ref_frames_filepath = args.output_folder / 'reference_frame_stats.csv'
    if ref_frames_filepath.exists() and not args.force:
        print(f"Reference frame data already exists in {ref_frames_filepath}. Use --force to overwrite.")
        print(f"Reading existing reference frame data from {ref_frames_filepath}")
        df_ref_frames_all = pd.read_csv(ref_frames_filepath)
    else:
        flight_logs = find_all_flight_logs(args.input_folder, args.match_pattern, args.folders_exclude)
        df_ref_frames_all = extract_ref_frame_data(flight_logs, args)
        if args.save:
            save_df(df_ref_frames_all, ref_frames_filepath)
            print(f"Reference frame data saved to {ref_frames_filepath}")

    df_best_master_frames = find_best_master_frames(df_ref_frames_all, args.best_n)
    print(f"Best master frames found for {len(df_best_master_frames)} unique location IDs:")
    print(df_best_master_frames.iloc[:,:6].to_string(index=False))

    if args.save:
        best_master_frames_filepath = args.output_folder / 'best_master_frames.csv'
        save_df(df_best_master_frames, best_master_frames_filepath)
        print(f"Best master frames saved to {best_master_frames_filepath}")

    if args.save_master_frames:
        extract_and_save_master_frames(df_best_master_frames, args)

    if args.visualize or args.save_viz:
        visualize_best_master_frames(df_best_master_frames, df_ref_frames_all, args.output_folder, args.visualize, args.save_viz)


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


def extract_ref_frame_data(flight_logs: list, args: argparse.Namespace) -> pd.DataFrame:
    """
    Extract the longitude, latitude, and relative altitude for the reference frame.
    Extract the number of detections and the area covered by the detections for the reference frame (if available).
    """
    data = []
    for flight_log in tqdm.tqdm(flight_logs, desc='Extracting flight logs and detection results'):
        if not flight_log.exists():
            print(f"Warning: {flight_log} does not exist. Skipping...")
            continue

        delimiter = detect_delimiter(flight_log)
        df = pd.read_csv(flight_log, delimiter=delimiter)
        location_id = determine_location_id(flight_log)
        video_path = flight_log.relative_to(args.input_folder).with_suffix(VIDEO_SUFFIX)

        try:
            longitude = df.loc[df['frame'] == args.ref_frame, 'longitude'].values[0]
            latitude = df.loc[df['frame'] == args.ref_frame, 'latitude'].values[0]
            relative_altitude = df.loc[df['frame'] == args.ref_frame, 'rel_alt'].values[0]
        except IndexError:
            print(f'Reference frame {args.ref_frame} not found in {flight_log}. Skipping...')
            continue

        geo_gdf = gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs='epsg:4326')
        local_coords = geo_gdf.to_crs(args.target_crs).geometry[0]
        x_local, y_local = round(local_coords.x, 2), round(local_coords.y, 2)

        number_of_objects, objects_covered_area = get_objects_and_area_covered(flight_log, args)

        data.append([location_id, video_path, longitude, latitude, x_local, y_local,
                     relative_altitude, number_of_objects, objects_covered_area, args.ref_frame])

    df = pd.DataFrame(data, columns=['location_id', 'video_path', 'longitude', 'latitude', 'x_local', 'y_local',
                                     'relative_altitude', 'number_of_objects', 'covered_area_by_objects', 'reference_frame'])

    for location_id in df['location_id'].unique():
        df_location = df.loc[df['location_id'] == location_id]
        mean_x_local = df_location['x_local'].mean()
        mean_y_local = df_location['y_local'].mean()
        mean_relative_altitude = df_location['relative_altitude'].mean()
        df.loc[df['location_id'] == location_id, 'distance_to_mean_location'] = np.sqrt(
            (df_location['x_local'] - mean_x_local)**2 +
            (df_location['y_local'] - mean_y_local)**2
        )
        df.loc[df['location_id'] == location_id, 'distance_to_mean_altitude'] = np.sqrt(
            (df_location['relative_altitude'] - mean_relative_altitude)**2
        )

    df['distance_to_mean_location'] = df['distance_to_mean_location'].round(3)
    df['distance_to_mean_altitude'] = df['distance_to_mean_altitude'].round(3)
    df.sort_values(by=['location_id', 'video_path'], inplace=True)

    columns_order = ['location_id', 'video_path', 'distance_to_mean_location', 'distance_to_mean_altitude',
                     'number_of_objects', 'covered_area_by_objects', 'longitude', 'latitude',
                     'x_local', 'y_local', 'relative_altitude', 'reference_frame']
    df = df[columns_order]

    return df


def extract_and_save_master_frames(df_best: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Extract and save the best master frames from the videos using OpenCV.
    """
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    for _, row in df_best.iterrows():
        video_path = str(args.input_folder / row['video_path'])
        output_filepath = output_folder / f"{row['location_id']}.png"

        cap = cv2.VideoCapture(video_path)
        frame_number = row['reference_frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(str(output_filepath), frame)
            print(f"Master frame {output_filepath} saved.")
        else:
            print(f"Failed to extract frame {frame_number} from {video_path}.")
        cap.release()


def get_objects_and_area_covered(flight_log: Path, args: argparse.Namespace) -> tuple:
    """
    Get the number of objects and the area covered by the objects in the reference frame.
    """
    detection_file = flight_log.parent / 'results' / (flight_log.stem + '.txt')
    if not detection_file.exists():
        return 'N/A', 'N/A'

    delimiter = detect_delimiter(detection_file)
    detections = np.loadtxt(detection_file, delimiter=delimiter, usecols=(0, *args.bbox_cols))

    ref_frame_detections = detections[detections[:, 0] == args.ref_frame]
    if ref_frame_detections.size == 0:
        number_of_objects = 0
        objects_covered_area = 0
    else:
        number_of_objects = ref_frame_detections.shape[0]
        bounding_boxes = ref_frame_detections[:, 1:]
        objects_covered_area = compute_area_covered(bounding_boxes, args.frame_width, args.frame_height)

    return number_of_objects, objects_covered_area


def compute_area_covered(bounding_boxes: np.ndarray, img_width: int, img_height: int) -> float:
    """
    Compute the area covered by the bounding boxes in percentage.
    """
    if bounding_boxes.size == 0:
        return 0
    areas = np.prod(bounding_boxes[:, 2:], axis=1)  # multiply w*h
    total_area = areas.sum() / (img_width * img_height)
    return round(100 * total_area, 2)


def find_best_master_frames(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Find the best master frame for georeferencing.
    """
    best_master_frames = []
    for location_id in tqdm.tqdm(df['location_id'].unique(), desc='Finding best master frames'):
        df_location = df.loc[df['location_id'] == location_id].copy()
        top_n_master_frames = df_location.nsmallest(N, 'distance_to_mean_location')

        if top_n_master_frames['covered_area_by_objects'].isna().all():
            best_master_frame = top_n_master_frames.nsmallest(1, 'distance_to_mean_location')
        else:
            best_master_frame = top_n_master_frames.loc[top_n_master_frames['covered_area_by_objects'] != 'N/A']
            best_master_frame = best_master_frame.nsmallest(1, 'covered_area_by_objects')

        best_master_frames.append(best_master_frame.to_dict(orient='records')[0])

    df_best_master_frames = pd.DataFrame(best_master_frames).sort_values(by=['location_id'])
    df_best_master_frames = df_best_master_frames.reset_index(drop=True)

    return df_best_master_frames


def save_df(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the data frame to a CSV file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def visualize_best_master_frames(df_best: pd.DataFrame, df_all: pd.DataFrame, output_folder: Path, visualize: bool, save_viz: bool) -> None:
    """
    Visualize the best master frames among all the locations.
    """

    location_ids = df_best['location_id'].unique()
    n_location_ids = len(location_ids)
    n_rows = n_location_ids // 5 + 1 if n_location_ids % 5 else n_location_ids // 5
    n_cols = 5 if n_location_ids > 5 else n_location_ids

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    global_min = np.inf
    global_max = -np.inf
    for location_id in location_ids:
        mean_relative_altitude = df_all.loc[df_all['location_id'] == location_id, 'relative_altitude'].mean()
        relative_altitude_diff = df_all.loc[df_all['location_id'] == location_id, 'relative_altitude'] - mean_relative_altitude
        if relative_altitude_diff.empty:
            continue

        global_min = min(global_min, relative_altitude_diff.min())
        global_max = max(global_max, relative_altitude_diff.max())

    handles, labels = [], []
    for i, location_id in enumerate(location_ids):
        ax = axs[i]
        df_location = df_all.loc[df_all['location_id'] == location_id]
        x_local_mean = df_location['x_local'].mean()
        y_local_mean = df_location['y_local'].mean()
        mean_relative_altitude = df_location['relative_altitude'].mean()
        x_local_all = df_location['x_local'].values
        y_local_all = df_location['y_local'].values
        relative_altitude_all = df_location['relative_altitude'].values

        distance_to_mean = np.sqrt((x_local_all - x_local_mean)**2 + (y_local_all - y_local_mean)**2)
        x_closest_to_mean = x_local_all[np.argmin(distance_to_mean)]
        y_closest_to_mean = y_local_all[np.argmin(distance_to_mean)]

        sc = ax.scatter(x_local_all - x_local_mean, y_local_all - y_local_mean, c=relative_altitude_all - mean_relative_altitude, marker='x', s=50, alpha=0.5, vmin=global_min, vmax=global_max, cmap='winter')
        h1 = ax.scatter(0, 0, color='red', label='Mean location', marker='+', s=250, linewidth=2)
        h2 = ax.scatter(x_closest_to_mean - x_local_mean, y_closest_to_mean - y_local_mean, color='black', label='Closest hovering to mean', marker='+', s=250, linewidth=2)
        h3 = ax.scatter(df_best.loc[df_best['location_id'] == location_id, 'x_local'] - x_local_mean,
                df_best.loc[df_best['location_id'] == location_id, 'y_local'] - y_local_mean, color='green', label='Best master frame location', marker='+', s=250, linewidth=2)
        radius = np.max(np.sqrt((df_location['x_local'] - x_local_mean)**2 + (df_location['y_local'] - y_local_mean)**2))
        circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', linewidth=1, label=f'r={radius:.2f} m')
        h4 = ax.add_artist(circle)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'Intersection {location_id} - {len(df_location)} hoverings')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        ax.legend(handles=[h4], labels=[f'radius={radius:.2f} m'], loc='upper right')
        handles.extend([sc, h1, h2, h3])
        labels.extend(['Hover location', 'Mean location', 'Closest hovering to mean', 'Best master frame location'])

    unique_handles_labels = dict(zip(labels, handles))
    fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper center', ncol=4, fontsize='large', bbox_to_anchor=(0.5, 0.92))

    cbar = fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    cbar.set_label('Relative altitude to mean (m)')

    for i in range(n_location_ids, n_rows * n_cols):
        fig.delaxes(axs[i])

    if visualize:
        plt.show()
    if save_viz:
        filepath = output_folder / 'best_master_frames.pdf'
        plt.savefig(filepath, transparent=False, bbox_inches='tight')
        print(f"Best master frames visualization saved to {filepath}")
    plt.close()


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Find the best master frame for georeferencing.')

    # Input and output paths
    parser.add_argument("input_folder", type=Path, help="Path to the folder containing the videos and flight logs and optional detection/tracking results.")
    parser.add_argument("output_folder", type=Path, help="Path to the output folder to save the results (e.g., ../ORTHOPHOTOS/master_frames).")

    # Optional arguments
    parser.add_argument("--save", "-s", action="store_true", help="Save the extracted reference frame stats and the list of best master frames.")
    parser.add_argument("--save-master-frames", "-smf", action="store_true", help="Save the best master frames extracted from the videos (existing files will be overwritten).")
    parser.add_argument("--force", "-f", action="store_true", help="Force the extraction of the flight log data (do not use the existing data).")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="Use custom reference frame that is used for stabilization/georeferencing.")
    parser.add_argument("--visualize", "-viz", action="store_true", help="Visualize the best master frame selection.")
    parser.add_argument("--save-viz", "-sv", action="store_true", help="Save the best master frames visualization.")
    parser.add_argument("--best_n", "-n", type=int, default=20, help="Number of reference frames to consider for the best master frame selection.")
    parser.add_argument("--match-pattern", "-m", type=str, default='??.CSV', help="Pattern to match (case-sensitive) the files in the folder.")
    parser.add_argument("--folders-exclude", "-fe", type=str, nargs='+', default=['results'], help="Folders to exclude from the search (e.g, 'results' as these may already contain georeferenced data).")
    parser.add_argument("--bounding-box-cols", "-b", type=int, nargs='+', default=[2, 3, 4, 5], dest="bbox_cols", help="Columns of the bounding box in the detection/tracking results.")
    parser.add_argument("--target-crs", "-tcrs", default='epsg:5186', help="Target CRS for local coordinates")
    parser.add_argument("--frame-width", "-fw", type=int, default=3840, help="Default width of the video frames.")
    parser.add_argument("--frame-height", "-fh", type=int, default=2160, help="Default height of the video frames.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()
    find_master_frames(args)
