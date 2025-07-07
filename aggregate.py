#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
aggregate.py - Aggregate Georeferenced Vehicle Tracking Results

This script aggregates the vehicle tracking results from multiple drone flights
into a unified dataset, organized by date, location, and flight session.

The script:
1. Scans through the PROCESSED directory to find all CSV result files
2. Groups files by date, location ID, and flight session
3. Combines data from each group, ensuring vehicle IDs are unique
4. Processes timestamps and adds drone identification
5. Standardizes column formats and orders
6. Creates aggregated CSV files for each group
7. Generates zip archives for convenient distribution

Usage:
  python aggregate.py <input> [options]

Arguments:
  input : Path to the PROCESSED folder containing georeferenced tracking results.

Options:
  -h, --help          : Show this help message and exit.
  -o, --output <path> : Path to the output folder. If not provided, the output folder will be
                        created in the same directory as the PROCESSED folder and will be
                        named 'DATASET' (default: None).
  -lf, --log-file <str> : Filename to save detailed logs. Saved in the 'logs' folder (default: None).
  -v, --verbose       : Set print verbosity level to INFO (default: WARNING).

Examples:
1. Basic aggregation with default output location:
   python aggregate.py /path/to/PROCESSED/

2. Aggregate with custom output folder:
   python aggregate.py /path/to/PROCESSED/ --output /path/to/custom/output/

3. Enable verbose logging and save to custom log file:
   python aggregate.py /path/to/PROCESSED/ --verbose --log-file custom_aggregate.log

Input:
- Path to PROCESSED folder containing georeferenced tracking results in CSV format
- CSV files should be organized in subdirectories: date/drone_id/flight_session/results/

Output:
- CSV files with aggregated tracking data, named by date_location_session
- ZIP archives containing all CSV files for each date_location combination
- Detailed logging information

Notes:
- The script expects CSV files to be located in a specific directory structure: date/drone_id/flight_session/results/
- Vehicle IDs are automatically offset to ensure uniqueness across different drone data
- Timestamps are converted to local time format (HH:MM:SS.fff)
- Lane numbers are standardized as strings
- Output files are organized by date and location for easy access
"""

import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.utils import determine_location_id, setup_logger


def aggregate_results(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Aggregate the georeferenced results by day, location, and flight session.
    """
    input_path = args.input
    output_path = args.output or input_path.parent / 'DATASET'
    logger.info(f"Starting aggregation process. Input folder: {input_path}, Output folder: {output_path}")

    if not input_path.exists():
        logger.critical(f"Input folder '{input_path}' does not exist.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_path.rglob('**/results/*.csv'))
    if not csv_files:
        logger.critical(f"No CSV files found in '{input_path}'")
        return

    file_groups = {}
    for file_path in csv_files:
        try:
            date = file_path.parents[3].name
            drone_id = file_path.parents[2].name
            flight_session = file_path.parents[1].name
            location_id = determine_location_id(file_path, logger)

            key = (date, location_id, flight_session)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append((file_path, drone_id))
        except Exception as e:
            logger.warning(f"Skipping invalid file path: {file_path} ({str(e)})")

    for key, files in file_groups.items():
        file_groups[key] = sorted(files, key=lambda x: (int(x[1][1:]), x[0]))

    total_unique_vehicles = 0

    pbar = tqdm(file_groups.items(), desc="Aggregating results", unit="aggregated file")
    for (date, location_id, flight_session), files in pbar:
        try:
            subfolder = output_path / f"{date}_{location_id}"
            subfolder.mkdir(exist_ok=True)
            output_file = subfolder / f"{date}_{location_id}_{flight_session}.csv"

            dfs = []
            vehicle_id_offset = 0

            for file_path, drone_id in files:
                try:
                    df = pd.read_csv(file_path)
                    df['Local_Time'] = pd.to_datetime(df['Timestamp']).dt.strftime('%H:%M:%S.%f').str[:-3]

                    df['Drone_ID'] = int(drone_id[1:])
                    df['Vehicle_ID'] = df['Vehicle_ID'] + vehicle_id_offset
                    vehicle_id_offset = df['Vehicle_ID'].max()
                    df['Lane_Number'] = df['Lane_Number'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

                    columns = [
                        'Vehicle_ID',
                        'Local_Time',
                        'Drone_ID',
                        'Ortho_X',
                        'Ortho_Y',
                        'Local_X',
                        'Local_Y',
                        'Latitude',
                        'Longitude',
                        'Vehicle_Length',
                        'Vehicle_Width',
                        'Vehicle_Class',
                        'Vehicle_Speed',
                        'Vehicle_Acceleration',
                        'Road_Section',
                        'Lane_Number',
                        'Visibility',
                    ]
                    df = df[columns]
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {str(e)}")

            if dfs:
                result_df = pd.concat(dfs, ignore_index=True)
                result_df.sort_values(['Vehicle_ID', 'Local_Time'], inplace=True)

                unique_vehicles = len(result_df['Vehicle_ID'].unique())
                logger.info(f"Group {date}_{location_id}_{flight_session}: {unique_vehicles} unique vehicles and {len(result_df)} trajectory points.")
                total_unique_vehicles += unique_vehicles

                result_df.to_csv(output_file, index=False)
                logger.info(f"Saved aggregated results to {output_file}")

                zip_path = output_path / f"{date}_{location_id}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in subfolder.glob('*.csv'):
                        zipf.write(file, file.name)
                logger.info(f"Created zip archive: {zip_path}")

        except Exception as e:
            logger.error(f"Error processing group {date}_{location_id}_{flight_session}: {str(e)}")

    logger.info(f"Total number of unique vehicles detected: {total_unique_vehicles}")
    logger.info("Aggregation process completed")


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Aggregate georeferenced tracking results')
    parser.add_argument('input', type=Path, help='Path to the PROCESSED folder')
    parser.add_argument('--output', '-o', type=Path, default=None, help="Path to the output folder. If not provided, the output folder will be created in the same directory as the PROCESSED folder and will be named 'DATASET'")
    parser.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Set print verbosity level to INFO (default: WARNING)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    aggregate_results(args, logger)
