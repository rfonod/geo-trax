#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
check_dataset.py - Dataset Verification Tool

Validates Geo-Trax datasets by identifying vehicles with excessive speed and acceleration values.
Reports violations with source video information for further investigation.

Usage:
  python tools/check_dataset.py <input> [options]

Arguments:
  input : Path to CSV file, directory with CSV files, or dataset root directory.

Options:
  -h, --help                             : Show this help message and exit.
  -at, --acceleration-threshold <float>  : Acceleration threshold in m/s² (default: 12).
  -st, --speed-threshold <float>         : Speed threshold in km/h (default: 130).
  -lp, --log-path <str>                  : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                            : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Check single CSV file:
   python tools/check_dataset.py data.csv

2. Check dataset with custom thresholds:
   python tools/check_dataset.py dataset/ --speed-threshold 100 --acceleration-threshold 10

3. Quiet output, logging to a custom file:
   python tools/check_dataset.py dataset/ --quiet --log-path validation.log

Input:
- CSV files with vehicle tracking data including Vehicle_Speed and Vehicle_Acceleration columns
- Individual files or directories containing multiple CSV files

Output:
- Console report of speed and acceleration violations
- Detailed violation tables with source video information
- Optional log file with processing details

Notes:
- Processes files recursively in subdirectories if no CSV files found in root
- Links violations to source videos using Vehicle_ID mapping
- Reports maximum violation per vehicle to avoid duplicate entries
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import pandas as pd
import tqdm

from geotrax.utils.logging_utils import setup_logger


def find_source_id(dataset_filepath: Path, vehicle_id: int, processed_folder: Union[Path, None] = None) -> tuple:
    """
    Trace an aggregated-dataset vehicle ID back to its original ID and source video, by
    reversing the per-drone ID offset applied during aggregation. (Adapted from
    tools/find_source_id.py.)
    """
    if not dataset_filepath.exists():
        print(f"Input folder '{dataset_filepath}' does not exist.")
        return None, None

    processed_folder = get_processed_folder(dataset_filepath, processed_folder)

    df = pd.read_csv(dataset_filepath, dtype={'Column14': str}, low_memory=False)
    if df[df['Vehicle_ID'] == vehicle_id].empty:
        print(f"Vehicle ID {vehicle_id} not found in the dataset.")
        return None, None

    date, location_id, flight_session = dataset_filepath.stem.split('_')[0:3]
    search_space = f"{date}/D*/{flight_session}/results/{location_id}*.csv"
    csv_files = list(processed_folder.rglob(search_space))
    if not csv_files:
        print(f"No CSV files found in '{processed_folder}'.")
        return None, None

    files = []
    for source_results in csv_files:
        try:
            drone_id = source_results.parents[2].name
            files.append((source_results, drone_id))
        except Exception as e:
            print(f"Skipping invalid file path: {source_results} ({str(e)})")
    files = sorted(files, key=lambda x: (int(x[1][1:]), x[0]))

    source_id, source_video = None, None
    vehicle_id_offset = 0
    for source_results, _drone_id in files:
        try:
            df = pd.read_csv(source_results)
            df['Vehicle_ID'] = df['Vehicle_ID'] + vehicle_id_offset
            if vehicle_id in df['Vehicle_ID'].values:
                source_id = vehicle_id - vehicle_id_offset
                source_video = source_results.parents[1] / (source_results.stem + '.MP4')
                break
            vehicle_id_offset = df['Vehicle_ID'].max()
        except Exception as e:
            print(f"Error processing file {source_results}: {str(e)}")

    return source_id, source_video


def get_processed_folder(source: Path, processed_folder: Union[Path, None]) -> Path:
    """
    Resolve the PROCESSED folder from the provided path or the default folder structure.
    """
    if processed_folder is None:
        processed_folder = source.parent
        while processed_folder != processed_folder.parent:
            if processed_folder.name == 'DATASET':
                break
            processed_folder = processed_folder.parent

        if processed_folder.name != 'DATASET':
            print(f"Failed to find the processed folder for source {source}. "
                  f"Use the --processed-folder argument to provide a custom path or "
                  f"ensure the default folder structure.")
            sys.exit(1)

        processed_folder = processed_folder.parent / 'PROCESSED'

    return processed_folder


def validate_speed_acceleration(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Check the dataset for high speed and acceleration values.
    """
    csv_files = determine_files_to_process(args.input, logger)
    check_for_excessive_values(csv_files, args, logger)


def check_for_excessive_values(csv_files: list, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Check for excessive speed and acceleration values in the dataset files.
    """
    speed_violations_df = pd.DataFrame()
    acceleration_violations_df = pd.DataFrame()
    columns = ['Dataset', 'Vehicle_ID', 'Drone_ID', 'Vehicle_Class', 'Vehicle_Acceleration', 'Vehicle_Speed', 'Source_ID', 'Source_Video']

    pbar = tqdm.tqdm(csv_files, desc="Processing files", unit="file")
    for csv_file in pbar:
        df = pd.read_csv(csv_file, dtype={'Column14': str}, low_memory=False)
        df['Dataset'] = csv_file
        df['Source_ID'] = None
        df['Source_Video'] = None

        speed_violations = df[df['Vehicle_Speed'] > args.speed_threshold][columns]
        speed_violations = speed_violations.loc[speed_violations.groupby('Vehicle_ID')['Vehicle_Speed'].idxmax()]
        speed_violations_df = pd.concat([speed_violations_df, speed_violations])

        acc_violations = df[df['Vehicle_Acceleration'].abs() > args.acceleration_threshold][columns]
        acc_violations = acc_violations.loc[acc_violations.groupby('Vehicle_ID')['Vehicle_Acceleration'].idxmax()]
        acceleration_violations_df = pd.concat([acceleration_violations_df, acc_violations])

    logger.notice(f"Checking for excessive speed values above {args.speed_threshold} km/h in the dataset...")
    report_violations(speed_violations_df, 'speed', logger)

    logger.notice(f"Checking for excessive absolute acceleration values above {args.acceleration_threshold} m/s^2 in the dataset...")
    report_violations(acceleration_violations_df, 'acceleration', logger)


def report_violations(violations_df: pd.DataFrame, violation_type: str, logger: logging.Logger) -> None:
    """
    Report violations found in the dataset.
    """
    if violations_df.empty:
        return

    if violation_type == 'acceleration':
        violations_df['abs_acceleration'] = violations_df['Vehicle_Acceleration'].abs()
        violations_df = violations_df.sort_values(by='abs_acceleration', ascending=False)
        violations_df = violations_df.drop(columns=['abs_acceleration'])
    else:
        violations_df = violations_df.sort_values(by='Vehicle_Speed', ascending=False)

    for index, row in violations_df.iterrows():
        source_id, source_video = find_source_id(Path(row['Dataset']), row['Vehicle_ID'])
        violations_df.at[index, 'Dataset'] = row['Dataset'].name
        if source_id is None:
            continue
        violations_df.at[index, 'Source_ID'] = int(source_id)
        violations_df.at[index, 'Source_Video'] = source_video

    suppress_logging_format(logger)
    with pd.option_context('display.max_colwidth', None):
        logger.warning("\n%s", repr(violations_df))
    restore_logging_format(logger)


def determine_files_to_process(input_path: Path, logger: logging.Logger) -> list:
    """
    Collect the files to process based from the input path.
    """
    if not input_path.exists():
        logger.critical(f"File or directory '{input_path}' not found.")
        sys.exit(1)

    csv_files = [input_path]
    if input_path.is_dir():
        csv_files = [f for f in input_path.iterdir() if f.suffix.lower() == '.csv']
        if len(csv_files) == 0:
            logger.info(f"No .csv files found in the directory '{input_path}'. Searching for subfolders...")
            sub_folders = [f for f in input_path.iterdir() if f.is_dir()]
            for folder in sub_folders:
                csv_files.extend([f for f in folder.iterdir() if f.suffix.lower() == '.csv'])
        if len(csv_files) == 0:
            logger.error(f"No .csv files found in the directory '{input_path}'. Skipping...")
            sys.exit(1)
        csv_files = sorted(csv_files)
    return csv_files


class PlainFormatter(logging.Formatter):
    """Formatter that only outputs the log message."""
    def format(self, record):
        return record.getMessage()


def suppress_logging_format(logger: logging.Logger):
    """Temporarily switch all handlers to a plain formatter."""
    for handler in logger.handlers:
        handler.setFormatter(PlainFormatter())


def restore_logging_format(logger: logging.Logger):
    """Restore original formatters."""
    for handler, formatter in logger._original_formatters.items():
        handler.setFormatter(formatter)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dataset verification tool for Geo-Trax generated dataset")

    parser.add_argument("input", type=Path, help="Path to CSV file or directory containing CSV files")
    parser.add_argument("--acceleration-threshold", "-at", type=float, default=12, help="Acceleration threshold in m/s² (default: 12)")
    parser.add_argument("--speed-threshold", "-st", type=float, default=130, help="Speed threshold in km/h (default: 130)")
    parser.add_argument("--log-path", "-lp", type=Path, default=None, help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console verbosity to important messages only (default: show INFO-level detail).")

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    validate_speed_acceleration(args, logger)


if __name__ == '__main__':
    main()
