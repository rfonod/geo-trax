#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_source_id.py - Find the original vehicle ID and source video from aggregated dataset ID.

This script traces back a vehicle ID from the aggregated dataset to its original ID and source
video file. It's useful for debugging, validation, or detailed analysis of specific vehicles
by locating their original detection data.

Usage:
    python tools/find_source_id.py <dataset_filepath> <vehicle_id> [options]

Arguments:
    dataset_filepath              : Path to the aggregated dataset CSV file (e.g., 2022-10-04_A/2022-10-04_A_AM1.csv).
    vehicle_id                    : Vehicle ID from the aggregated dataset to trace back.

Options:
    -h, --help                    : Show this help message and exit.
    -p, --processed-folder <path> : Custom path to the PROCESSED directory. If not provided,
                                    automatically detected from dataset file location.

Examples:
  1. Find source information for vehicle ID 5 from an aggregated dataset:
     python tools/find_source_id.py 2022-10-04_A/2022-10-04_A_AM1.csv 5

  2. Use custom PROCESSED folder location:
     python tools/find_source_id.py dataset.csv 12 --processed-folder /path/to/PROCESSED/

Output:
  - Original vehicle ID in the source video
  - Source video file path
  - Source CSV results file path
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import pandas as pd


def find_source_id(dataset_filepath: Path, vehicle_id: int,
                   processed_folder: Union[Path, None] = None, verbose: bool = False) -> tuple:
    """
    Find the original vehicle ID extracted from the source video from the dataset ID.
    """
    if not dataset_filepath.exists():
        print(f"Input folder '{dataset_filepath}' does not exist.")
        return None, None

    # Get the PROCESSED folder
    processed_folder = get_processed_folder(dataset_filepath, processed_folder)

    # Load the dataset and find the vehicle ID
    df = pd.read_csv(dataset_filepath, dtype={'Column14': str}, low_memory=False)
    vehicle_df = df[df['Vehicle_ID'] == vehicle_id]
    if vehicle_df.empty:
        print(f"Vehicle ID {vehicle_id} not found in the dataset.")
        return None, None

    # Get the drone ID
    drone_id = vehicle_df['Drone_ID'].iloc[0]

    # Narrow down the search to the specific date, location, and flight session
    date, location_id, flight_session = dataset_filepath.stem.split('_')[0:3]
    search_space =  f"{date}/D*/{flight_session}/results/{location_id}*.csv"

    # Find all relevant georeferenced results (.csv files) in the PROCESSED directory
    csv_files = list(processed_folder.rglob(search_space))
    if not csv_files:
        print(f"No CSV files found in '{processed_folder}'.")
        return None, None

    # Group files by date, flight session, and location id
    files = []
    for source_results in csv_files:
        try:
            drone_id = source_results.parents[2].name
            files.append((source_results, drone_id))
        except Exception as e:
            print(f"Skipping invalid file path: {source_results} ({str(e)})")

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
                if verbose:
                    print(f"Date     : {date}")
                    print(f"Drone ID : {drone_id}")
                    print(f"Session  : {flight_session}")
                    print(f"Video ID : {source_results.stem}")
                    print(f"Vehicle ID (dataset) : {vehicle_id}")
                    print(f"Vehicle ID (video)   : {source_id}")
                    print(source_video)
                    print(source_results)
                break
            vehicle_id_offset = df['Vehicle_ID'].max()
        except Exception as e:
            print(f"Error processing file {source_results}: {str(e)}")

    return source_id, source_video



def get_processed_folder(source: Path, processed_folder: Union[Path, None]) -> Path:
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
            print(f"Failed to find the processed folder for source {source}. "
                  f"Use the --processed-folder argument to provide a custom path or "
                  f"ensure the default folder structure.")
            sys.exit(1)

        processed_folder = processed_folder.parent / 'PROCESSED'

    return processed_folder


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Find the original vehicle ID extracted from the source video from the dataset ID.')
    parser.add_argument('dataset_filepath', type=Path,
                        help='Filepath to the source dataset (.csv) file (e.g. 2022-10-04_A/2022-10-04_A_AM1.csv)')
    parser.add_argument('vehicle_id', type=int, help='Vehicle ID of interest from the dataset (e.g. 1)')
    parser.add_argument('--processed-folder', '-p', type=Path,
                        help='Custom path to the PROCESSED directory containing the georeferenced results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli_args()
    find_source_id(args.dataset_filepath, args.vehicle_id, processed_folder = args.processed_folder, verbose=True)
