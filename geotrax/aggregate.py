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
  geotrax aggregate <input> [options]

Arguments:
  input : Path to the PROCESSED folder of georeferenced results.

Options:
  -h, --help                 : Show this help message and exit.
  -of, --output-folder <path>: Path to the output folder for aggregated results. If not provided,
                               a 'DATASET' folder is created next to the PROCESSED folder (default: None).
  -c, --cfg <path>           : Pipeline config used to resolve the output folder name where
                               georeferenced CSVs are located. Defaults to the bundled config.
  -st, --set <KEY=VALUE>     : Override a pipeline config value for this run; repeat for more
                               than one, e.g. --set folder=out. KEY is a dotted path or any
                               unambiguous tail of one; VALUE uses YAML rules.
  -lp, --log-path <str>      : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -v, --verbose              : Set print verbosity level to INFO (default: WARNING).

Examples:
1. Basic aggregation with default output location:
   geotrax aggregate /path/to/PROCESSED/

2. Aggregate with custom output folder:
   geotrax aggregate /path/to/PROCESSED/ --output-folder /path/to/custom/output/

3. Enable verbose logging and save to custom log file:
   geotrax aggregate /path/to/PROCESSED/ --verbose --log-path custom_aggregate.log

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
- Rows whose timestamp is undefined (frames missing from the drone flight log, written as
  '0000-00-00 00:00:00.000' by georeferencing) are dropped with a warning; the rest of the file is kept
- Drone folders must be named 'D<number>'; results in other folders are skipped with a warning
- An absolute output folder (cfg -> output -> folder) is not supported, since grouping relies on the
  per-video folder layout
- Results without a 'Timestamp' column (georeferenced without the drone flight log) are skipped,
  since their rows cannot be time-aligned with the other drones of the group
- Empty (header-only) result files are skipped so they cannot disturb the vehicle ID offset
- 'Is_Interpolated' is preserved when present; rows from runs without it are marked 0, since a run
  with interpolation disabled has no synthetic points
- Road sections and lane numbers left empty by georeferencing without a segmentation file are kept empty
- Lane numbers are standardized as strings
- Output files are organized by date and location for easy access
"""

import argparse
import logging
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from geotrax.utils.cli_utils import add_common_args, finalize_cli_args
from geotrax.utils.config_utils import load_config
from geotrax.utils.file_utils import DEFAULT_OUTPUT, determine_location_id
from geotrax.utils.logging_utils import setup_logger

DATASET_COLUMNS = (
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
)


def aggregate_results(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Aggregate the georeferenced results by day, location, and flight session."""
    input_path = args.input
    output_path = args.output_folder or input_path.parent / 'DATASET'
    logger.info(f"Starting aggregation process. Input folder: {input_path}, Output folder: {output_path}")

    if not input_path.exists():
        logger.critical(f"Input folder '{input_path}' does not exist.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    output_cfg = load_config(args.cfg, logger, args).get('output', DEFAULT_OUTPUT)
    folder_name = output_cfg.get('folder', DEFAULT_OUTPUT['folder'])
    if Path(folder_name).is_absolute():
        logger.critical(
            f"cfg -> output -> folder is an absolute path ('{folder_name}'). Aggregation needs the per-video "
            f"'<date>/<drone>/<session>/<output folder>/' layout to group results, so it only supports a relative "
            f"output folder name."
        )
        sys.exit(1)
    csv_files = list(input_path.rglob(f'**/{folder_name}/*.csv'))
    if not csv_files:
        logger.critical(f"No CSV files found in '{input_path}'")
        sys.exit(1)

    file_groups = group_result_files(csv_files, folder_name, logger)
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
                df = load_source_rows(file_path, drone_id, logger)
                if df is None:
                    continue
                df['Vehicle_ID'] = df['Vehicle_ID'] + vehicle_id_offset
                vehicle_id_offset = df['Vehicle_ID'].max()
                dfs.append(df)

            if not dfs:
                logger.warning(
                    f"Group {date}_{location_id}_{flight_session}: no usable georeferenced results among "
                    f"{len(files)} file(s); nothing was aggregated for this group."
                )

            if dfs:
                result_df = pd.concat(dfs, ignore_index=True)
                if 'Is_Interpolated' in result_df.columns:
                    result_df['Is_Interpolated'] = result_df['Is_Interpolated'].fillna(0).astype(int)
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


def group_result_files(csv_files: list, folder_name: str, logger: logging.Logger) -> dict:
    """Group georeferenced CSVs by (date, location, session), each group sorted in aggregation order.

    Files are expected at <date>/D<n>/<session>/<folder_name>/<file>.csv; any other path is skipped with
    a warning. Within a group, files are ordered by drone number, then path, which fixes the order in
    which the Vehicle_ID offsets are applied (see trace_dataset_vehicle).
    """
    file_groups = {}
    for file_path in csv_files:
        try:
            date = file_path.parents[3].name
            drone_id = file_path.parents[2].name
            if not re.fullmatch(r'D\d+', drone_id):
                raise ValueError(
                    f"drone folder '{drone_id}' is not named 'D<number>'; expected <date>/<drone>/<session>/"
                    f"{folder_name}/<file>.csv"
                )
            flight_session = file_path.parents[1].name
            location_id = determine_location_id(file_path, logger)
            file_groups.setdefault((date, location_id, flight_session), []).append((file_path, drone_id))
        except Exception as e:
            logger.warning(f"Skipping invalid file path: {file_path} ({str(e)})")

    return {key: sorted(files, key=lambda x: (int(x[1][1:]), x[0])) for key, files in file_groups.items()}


def load_source_rows(file_path: Path, drone_id: str, logger: logging.Logger) -> pd.DataFrame | None:
    """Return the rows of one georeferenced CSV as they enter the aggregated dataset, or None if skipped.

    Vehicle_ID is left as in the source file; the caller applies the group's running offset. Every
    skipped file and dropped row therefore also changes the offsets of the files after it, which is
    why trace_dataset_vehicle replays the offsets through this same function.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(
                f"Skipping '{file_path}': no trajectory rows (every vehicle was filtered out by "
                f"cfg -> georef -> filtering -> min_traj_length, or the video yielded no tracks)."
            )
            return None
        if 'Timestamp' not in df.columns:
            logger.warning(
                f"Skipping '{file_path}': no 'Timestamp' column, so its rows cannot be time-aligned "
                f"with the other drones of this group. Georeferencing only writes timestamps when the "
                f"drone flight log ('{file_path.stem}.csv' next to the video) is present; restore it and "
                f"re-run 'geotrax georeference'."
            )
            return None
        timestamps = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
        n_undefined = int(timestamps.isna().sum())
        if n_undefined == len(df):
            logger.warning(
                f"Skipping '{file_path}': none of its timestamps are valid, so its rows cannot be "
                f"time-aligned with the other drones of this group."
            )
            return None
        if n_undefined:
            logger.warning(
                f"Dropping {n_undefined} of {len(df)} rows from '{file_path}' with an undefined "
                f"timestamp (frames missing from the drone flight log)."
            )
            df = df[timestamps.notna()].copy()
            timestamps = timestamps[timestamps.notna()]
        df['Local_Time'] = timestamps.dt.strftime('%H:%M:%S.%f').str[:-3]

        df['Drone_ID'] = int(drone_id[1:])
        for optional_column in ('Road_Section', 'Lane_Number'):
            if optional_column not in df.columns:
                logger.warning(
                    f"'{optional_column}' column missing from '{file_path}' (georeferenced without a "
                    f"segmentation file); it will be empty in the aggregated dataset."
                )
                df[optional_column] = pd.NA
        df['Lane_Number'] = df['Lane_Number'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

        columns = list(DATASET_COLUMNS)
        if 'Is_Interpolated' in df.columns:
            columns.append('Is_Interpolated')
        return df[columns]
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {str(e)}")
        return None


def trace_dataset_vehicle(
    processed_folder: Path, dataset_stem: str, vehicle_id: int, folder_name: str, logger: logging.Logger
) -> tuple:
    """Trace an aggregated-dataset Vehicle_ID back to its source CSV and per-video vehicle ID.

    dataset_stem is the aggregated file's '<date>_<location>_<session>' stem. The group is rebuilt with
    group_result_files and its offsets are replayed with load_source_rows, the same functions
    aggregate_results uses, so files skipped and rows dropped during aggregation shift the offsets
    exactly as they did then. Returns (source_csv, source_id), or (None, None) if the ID is not found.
    """
    date, location_id, flight_session = dataset_stem.split('_')[0:3]
    csv_files = list(processed_folder.rglob(f"{date}/*/{flight_session}/{folder_name}/*.csv"))
    files = group_result_files(csv_files, folder_name, logger).get((date, location_id, flight_session), [])

    vehicle_id_offset = 0
    for file_path, drone_id in files:
        df = load_source_rows(file_path, drone_id, logger)
        if df is None:
            continue
        if vehicle_id - vehicle_id_offset in df['Vehicle_ID'].values:
            return file_path, vehicle_id - vehicle_id_offset
        vehicle_id_offset += df['Vehicle_ID'].max()
    return None, None


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Aggregate georeferenced tracking results')
    parser.add_argument('input', type=Path, help='Path to the PROCESSED folder of georeferenced results.')

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--output-folder', '-of', type=Path, default=None, help="Path to the output folder for aggregated results. If not provided, a 'DATASET' folder is created next to the PROCESSED folder.")
    cfg_paths = add_common_args(optional, output_folder=False)

    return finalize_cli_args(parser, cfg_paths)


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(__name__, args.verbose, args.log_path)

    aggregate_results(args, logger)


if __name__ == '__main__':
    main()
