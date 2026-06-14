#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_source_id.py - Vehicle ID Traceability Tool

This script traces back a vehicle ID from an aggregated dataset to its original ID and source
video file. It reverse-engineers the aggregation process to identify the original detection
data for specific vehicles, supporting debugging, validation, and detailed analysis workflows.

The tool parses aggregated dataset files, extracts metadata (date, location, session), searches
corresponding PROCESSED directories, and applies the same ID offset logic used during aggregation
to locate the original vehicle ID and source video file.

Usage:
  python tools/find_source_id.py <dataset_filepath> <vehicle_id> [options]

Arguments:
  dataset_filepath : Path to aggregated dataset CSV file (e.g., 2022-10-04_A/2022-10-04_A_AM1.csv).
  vehicle_id       : Vehicle ID from the aggregated dataset to trace back.

Options:
  -h, --help                    : Show this help message and exit.
  -p, --processed-folder <path> : Custom path to the PROCESSED directory. If not provided, auto-detected from the dataset file location.
  -lp, --log-path <str>         : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                   : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Find source information for vehicle ID 5 from aggregated dataset:
   python tools/find_source_id.py 2022-10-04_A/2022-10-04_A_AM1.csv 5

2. Use custom PROCESSED folder location:
   python tools/find_source_id.py dataset.csv 12 --processed-folder /path/to/PROCESSED/

3. Trace vehicle from different aggregated dataset:
   python tools/find_source_id.py 2022-10-05_B/2022-10-05_B_PM3.csv 27

Input:
- Aggregated dataset CSV file with Vehicle_ID and Drone_ID columns
- PROCESSED directory structure: DATE/DRONE_ID/SESSION/results/LOCATION_ID*.csv
- Source video files (.MP4) and georeferenced results (.csv)

Output:
- Console output with detailed traceability information:
  * Date, Drone ID, Session, Video ID
  * Vehicle ID mapping (dataset → original)
  * Source video file path (.MP4)
  * Source CSV results file path

Notes:
- Automatically detects PROCESSED folder from dataset location or uses custom path
- Applies same ID offset logic as aggregation process to reverse-map vehicle IDs
- Searches within specific date/location/session scope for efficiency
- Handles multiple drone data with proper sorting and offset calculation
- Useful for validation, debugging, and detailed vehicle trajectory analysis
- Requires consistent folder structure: DATASET and PROCESSED at same level
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd

from geotrax.utils.logging_utils import setup_logger


def find_source_id(dataset_filepath: Path, vehicle_id: int, logger: logging.Logger,
                   processed_folder: Union[Path, None] = None) -> tuple:
    """
    Find the original vehicle ID extracted from the source video from the dataset ID.
    """
    if not dataset_filepath.exists():
        logger.error(f"Input folder '{dataset_filepath}' does not exist.")
        return None, None

    # Get the PROCESSED folder
    processed_folder = get_processed_folder(dataset_filepath, processed_folder, logger)

    # Load the dataset and find the vehicle ID
    df = pd.read_csv(dataset_filepath, dtype={'Column14': str}, low_memory=False)
    vehicle_df = df[df['Vehicle_ID'] == vehicle_id]
    if vehicle_df.empty:
        logger.warning(f"Vehicle ID {vehicle_id} not found in the dataset.")
        return None, None

    # Get the drone ID
    drone_id = vehicle_df['Drone_ID'].iloc[0]

    # Narrow down the search to the specific date, location, and flight session
    date, location_id, flight_session = dataset_filepath.stem.split('_')[0:3]
    search_space =  f"{date}/D*/{flight_session}/results/{location_id}*.csv"

    # Find all relevant georeferenced results (.csv files) in the PROCESSED directory
    csv_files = list(processed_folder.rglob(search_space))
    if not csv_files:
        logger.warning(f"No CSV files found in '{processed_folder}'.")
        return None, None

    # Group files by date, flight session, and location id
    files = []
    for source_results in csv_files:
        try:
            drone_id = source_results.parents[2].name
            files.append((source_results, drone_id))
        except Exception as e:
            logger.warning(f"Skipping invalid file path: {source_results} ({str(e)})")

    # Sort values for each key based on drone ID and file name
    files = sorted(files, key=lambda x: (int(x[1][1:]), x[0]))

    # Find the source ID and source video
    source_id, source_video = None, None
    vehicle_id_offset = 0
    for source_results, drone_id in files:
        try:
            df = pd.read_csv(source_results)
            df['Vehicle_ID'] = df['Vehicle_ID'] + vehicle_id_offset
            if vehicle_id in df['Vehicle_ID'].values:
                source_id = vehicle_id - vehicle_id_offset
                source_video = source_results.parents[1] / (source_results.stem + '.MP4')
                logger.notice(
                    f"Date     : {date}\n"
                    f"Drone ID : {drone_id}\n"
                    f"Session  : {flight_session}\n"
                    f"Video ID : {source_results.stem}\n"
                    f"Vehicle ID (dataset) : {vehicle_id}\n"
                    f"Vehicle ID (video)   : {source_id}\n"
                    f"{source_video}\n"
                    f"{source_results}"
                )
                break
            vehicle_id_offset = df['Vehicle_ID'].max()
        except Exception as e:
            logger.error(f"Error processing file {source_results}: {str(e)}")

    return source_id, source_video



def get_processed_folder(source: Path, processed_folder: Union[Path, None], logger: logging.Logger) -> Path:
    """
    Get the processed folder from the provided path or use the default folder structure.
    """
    if processed_folder is None:
        processed_folder = source.parent

        while processed_folder != processed_folder.parent:
            if processed_folder.name == 'DATASET':
                break
            processed_folder = processed_folder.parent

        if processed_folder.name != 'DATASET':
            logger.critical(f"Failed to find the processed folder for source {source}. "
                            f"Use the --processed-folder argument to provide a custom path or "
                            f"ensure the default folder structure.")
            sys.exit(1)

        processed_folder = processed_folder.parent / 'PROCESSED'

    return processed_folder


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Find the original vehicle ID extracted from the source video from the dataset ID.')
    parser.add_argument('dataset_filepath', type=Path, help='Filepath to the source dataset (.csv) file (e.g. 2022-10-04_A/2022-10-04_A_AM1.csv)')
    parser.add_argument('vehicle_id', type=int, help='Vehicle ID of interest from the dataset (e.g. 1)')
    parser.add_argument('--processed-folder', '-p', type=Path, help='Custom path to the PROCESSED directory containing the georeferenced results')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    find_source_id(args.dataset_filepath, args.vehicle_id, logger, processed_folder=args.processed_folder)


if __name__ == '__main__':
    main()
