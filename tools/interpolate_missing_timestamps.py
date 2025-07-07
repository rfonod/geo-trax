#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
interpolate_missing_timestamps.py - CSV Timestamp Interpolation Tool

This script interpolates missing timestamps in CSV files using a precise 29.97 fps video frame timing pattern.
It fills in missing timestamp values by calculating appropriate time intervals based on the alternating
millisecond increments used in 29.97 fps video recording (33ms, 33ms, 34ms cycle).

The script analyzes the existing timestamps to determine the appropriate starting point in the timing cycle
and then interpolates missing values either forward or backward through the dataset.

Usage:
  python interpolate_missing_timestamps.py <input_csv> [options]

Arguments:
  input_csv : Path to the input CSV file containing timestamps to interpolate.

Options:
  --backward    : Perform backward interpolation instead of forward interpolation (default: False).

Input:
- CSV file with a 'timestamp' column that may contain missing (NaN) values
- Timestamps should be in datetime format compatible with pandas

Output:
- New CSV file with interpolated timestamps, saved as '<original_name>_interpolated.CSV'
- Console output showing before and after data samples

Notes:
- The script uses a 29.97 fps timing pattern with alternating increments of 33ms, 33ms, 34ms
- The interpolation direction (forward/backward) affects which existing timestamps are used as reference points
- Missing timestamps are filled using precise video frame timing calculations
- The output file preserves all original data while filling in the missing timestamp values
"""

import argparse
from pathlib import Path
import re

import pandas as pd


def interpolate_timestamps(args):
    df = pd.read_csv(args.input_csv)
    print(df.head())
    print(df.tail())

    # Define the cycle increments for 29.97 fps
    delta_times = [33, 33, 34]  # Alternating increments in milliseconds

    # Calculate the total time to interpolate
    first_valid_idx = df['timestamp'].last_valid_index()
    last_valid_idx = df['timestamp'].first_valid_index()
    total_time_to_interpolate = (pd.to_datetime(df.loc[first_valid_idx, 'timestamp']) -
                                pd.to_datetime(df.loc[last_valid_idx, 'timestamp'])).total_seconds()

    # Determine starting point in the cycle
    cycle_time = sum(delta_times) / 1000.0  # Convert cycle time to seconds
    starting_offset = (total_time_to_interpolate % cycle_time) / cycle_time
    if starting_offset < 0.33:
        starting_index = 0  # Start with 33 ms
    elif starting_offset < 0.66:
        starting_index = 1  # Start with second 33 ms
    else:
        starting_index = 2  # Start with 34 ms

    # Perform backward interpolation
    if args.backward:
        counter = starting_index
        for i in range(len(df)-1, -1, -1):
            if pd.isna(df.loc[i, 'timestamp']):
                prev_timestamp = df.loc[i+1, 'timestamp']
                prev_timestamp = pd.to_datetime(prev_timestamp)
                new_timestamp = prev_timestamp - pd.Timedelta(milliseconds=delta_times[counter % 3])
                df.loc[i, 'timestamp'] = new_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                counter += 1
    else:
        counter = starting_index
        for i in range(len(df)):
            if pd.isna(df.loc[i, 'timestamp']):
                prev_timestamp = df.loc[i-1, 'timestamp']
                prev_timestamp = pd.to_datetime(prev_timestamp)
                new_timestamp = prev_timestamp + pd.Timedelta(milliseconds=delta_times[counter % 3])
                df.loc[i, 'timestamp'] = new_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                counter += 1

    print(df.head())
    print(df.tail())

    # Save the interpolated CSV
    output_csv = args.input_csv.parent / (args.input_csv.stem + '_interpolated' + '.CSV')
    df.to_csv(output_csv, index=False)


def get_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Interpolate missing timestamps in a CSV file')
    parser.add_argument('input_csv', type=Path, help='Input CSV file')
    parser.add_argument('--backward', action='store_true', help='Perform backward interpolation')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    interpolate_timestamps(args)
