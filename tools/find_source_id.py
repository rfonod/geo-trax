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
  -c, --cfg <path>              : Pipeline config used to resolve the output folder name where georeferenced CSVs are located.
                                  Defaults to the bundled config (geotrax/cfg/default.yaml).
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
- Source video files (.mp4/.mov/.avi/.mkv, any case) and georeferenced results (.csv)

Output:
- Console output with detailed traceability information:
  * Date, Drone ID, Session, Video ID
  * Vehicle ID mapping (dataset → original)
  * Source video file path
  * Source CSV results file path

Notes:
- Automatically detects PROCESSED folder from dataset location or uses custom path
- Replays the aggregation's ID offsets with the same code (geotrax.aggregate), so files and rows
  skipped during aggregation shift the offsets exactly as they did then
- Searches within specific date/location/session scope for efficiency
- Handles multiple drone data with proper sorting and offset calculation
- Useful for validation, debugging, and detailed vehicle trajectory analysis
- Requires consistent folder structure: DATASET and PROCESSED at same level
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from geotrax.aggregate import trace_dataset_vehicle
from geotrax.utils.cli_utils import DEFAULT_CFG
from geotrax.utils.config_utils import load_config
from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.file_utils import DEFAULT_OUTPUT
from geotrax.utils.logging_utils import setup_logger


def find_source_id(dataset_filepath: Path, vehicle_id: int, logger: logging.Logger,
                   processed_folder: Path | None = None, folder_name: str = None) -> tuple:
    """
    Find the original vehicle ID extracted from the source video from the dataset ID.

    The per-drone ID offsets are replayed by geotrax.aggregate.trace_dataset_vehicle, with the same
    file grouping, skipped files and dropped rows as the aggregation itself.
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

    date, _, flight_session = dataset_filepath.stem.split('_')[0:3]
    folder = folder_name or DEFAULT_OUTPUT['folder']
    source_results, source_id = trace_dataset_vehicle(processed_folder, dataset_filepath.stem, vehicle_id, folder, logger)
    if source_results is None:
        logger.warning(f"No georeferenced results in '{processed_folder}' contain dataset vehicle ID {vehicle_id}.")
        return None, None

    clip_dir = source_results.parents[1]
    source_video = next(
        (p for p in clip_dir.glob(source_results.stem + '.*') if p.suffix.lower() in VIDEO_FORMATS),
        clip_dir / (source_results.stem + '.MP4'),
    )
    logger.notice(
        f"Date     : {date}\n"
        f"Drone ID : {source_results.parents[2].name}\n"
        f"Session  : {flight_session}\n"
        f"Video ID : {source_results.stem}\n"
        f"Vehicle ID (dataset) : {vehicle_id}\n"
        f"Vehicle ID (video)   : {source_id}\n"
        f"{source_video}\n"
        f"{source_results}"
    )
    return source_id, source_video



def get_processed_folder(source: Path, processed_folder: Path | None, logger: logging.Logger) -> Path:
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
    parser.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG, help='Pipeline config used to resolve the output folder name where georeferenced CSVs are located. Defaults to the bundled config.')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    out_cfg = load_config(args.cfg, logger).get('output', DEFAULT_OUTPUT)
    folder_name = out_cfg.get('folder', DEFAULT_OUTPUT['folder'])
    find_source_id(args.dataset_filepath, args.vehicle_id, logger, processed_folder=args.processed_folder, folder_name=folder_name)


if __name__ == '__main__':
    main()
