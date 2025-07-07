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
  -h, --help                     : Show this help message and exit.
  -at, --acceleration-threshold <float> : Acceleration threshold in m/s² (default: 12).
  -st, --speed-threshold <float>         : Speed threshold in km/h (default: 130).
  -lf, --log-file <str>                  : Log filename saved in logs/ folder (default: None).
  -v, --verbose                          : Set verbosity to INFO level (default: WARNING).

Examples:
1. Check single CSV file:
   python tools/check_dataset.py data.csv

2. Check dataset with custom thresholds:
   python tools/check_dataset.py dataset/ --speed-threshold 100 --acceleration-threshold 10

3. Verbose output with logging:
   python tools/check_dataset.py dataset/ --verbose --log-file validation.log

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

import pandas as pd
import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.find_source_id import find_source_id
from utils.utils import setup_logger


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
        logger.warning(f"\n%s", repr(violations_df))
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
    parser.add_argument("--log-file", "-lf", type=str, default=None, help="Log filename saved in logs/ folder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Set verbosity to INFO level")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    validate_speed_acceleration(args, logger)
