#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
fix_timestamp_annomalies.py - Timestamp Anomaly Correction Tool

This script automatically fixes timestamp anomalies detected in drone flight logs by cutting videos
and flight logs at anomaly frames. It implements a cutting strategy that preserves video integrity 
while removing problematic timestamp regions.

The tool analyzes anomaly locations and creates one or two cuts depending on the anomaly position,
ensuring minimum video duration requirements are met. It renames original files for backup,
generates cut specification files, and automatically reprocesses the resulting video segments.

Usage:
  python tools/fix_timestamp_annomalies.py <input> [options]

Arguments:
  input : str
          Path to file containing flight log anomalies (created by tools/find_cut_video_issues.py).

Options:
  -h, --help            : Show this help message and exit.
  -o, --processed-folder <path> : str, optional
                        Path to root of processed folder containing cut videos and flight logs
                        (default: same as input directory).
  -d, --debug           : bool, optional
                        Run in debug mode - no files will be modified (default: False).

Examples:
1. Fix anomalies using default processed folder location:
   python tools/fix_timestamp_annomalies.py flight_log_anomalies.csv

2. Production run with custom processed folder:
   python tools/fix_timestamp_annomalies.py flight_log_anomalies.csv --processed-folder /data/PROCESSED/

Input:
- CSV file with anomaly data containing columns:
  * location_id, video_path, timestamp_max_abs_diff
  * timestamp_anomaly_location, timestamp_anomaly_frame
- PROCESSED folder with video files (.MP4) and flight logs (.CSV)
- Expected file structure: videos and CSV files with matching names

Output:
- Original files renamed with "_original" suffix for backup
- Cut specification files (.TXT) with frame ranges
- Recut video files with corrected timestamps
- Reprocessed flight logs and results
- Console output with detailed operation status

Cutting Strategy:
- Single cut: If anomaly near start/end (< 15s), cut from opposite end
- Double cut: If anomaly in middle, create two segments around anomaly
- Frame margins: ±30 frames (1 second at 30 FPS) around anomaly point
- Minimum duration: 15 seconds per resulting video segment

Notes:
- Processes anomalies from tools/find_cut_video_issues.py output
- Maintains file naming conventions and sequence numbers
- Prevents conflicts with existing higher sequence numbers
- Uses recut_video_and_csv.py and batch_process.py for processing
- Debug mode shows operations without modifying files
- Automatically handles backup file creation and restoration
"""

import argparse
import os
from pathlib import Path

import pandas as pd

MIN_VIDEO_DURATION = 15  # do not cut videos shorter than 15 seconds
FPS = 30 # used to calculate the minimum video duration and add margins to the cuts


def fix_timetamp_annomalies(args: argparse.Namespace) -> None:
    """
    Fix timestamp anomalies found in in the flight logs.
    """

    df_anomalies = pd.read_csv(args.input)
    df_anomalies = df_anomalies[['location_id', 'video_path', 'timestamp_max_abs_diff', 'timestamp_anomaly_location', 'timestamp_anomaly_frame']]

    # keep only the rows with valid timestamp anomalies
    df_anomalies = df_anomalies.dropna(subset=['timestamp_anomaly_frame'])
    print(f"Found {len(df_anomalies)} anomalies in total.")
    if len(df_anomalies) == 0:
        return
    print(df_anomalies.to_string(index=False))

    # iterate over the anomalies and fix them
    processed_folder = args.input.parent if args.processed_folder is None else args.processed_folder
    for _, row in df_anomalies.iterrows():
        video_path = Path(row['video_path'])
        video_filepath = processed_folder / video_path
        csv_filepath = video_filepath.with_suffix('.CSV')
        location_id = row['location_id']
        location_sequence = int(video_path.stem.replace(location_id, '').replace('.MP4', ''))
        timestamp_anomaly_frame = int(row['timestamp_anomaly_frame'])

        if not video_filepath.exists() or not csv_filepath.exists():
            print(f"\033[91mWARNING: Skipping: {video_filepath} (not found)\033[0m")
            continue

        video_filepath_next = video_filepath.with_name(f"{location_id}{location_sequence+1}.CSV")
        if video_filepath_next.exists():
            print(f"\033[93mWARNING: Skipping: {video_filepath} (higher sequence number exists)\033[0m")
            print("Rename the subsequent files manually and run the script again.")
            continue

        print(f"\033[92mFixing: {video_filepath}\033[0m")

        # determine the cuts
        df_csv = pd.read_csv(csv_filepath)
        last_frame = df_csv['frame'].max()
        cuts = []
        if timestamp_anomaly_frame / FPS < MIN_VIDEO_DURATION:
            cut_filepath = video_filepath.with_name(f"0_{location_id}{location_sequence}_recut.TXT")
            cut_start = round(timestamp_anomaly_frame + FPS)
            cut_end = -1
            cuts.append((cut_filepath, cut_start, cut_end))
        elif (last_frame - timestamp_anomaly_frame) / FPS < MIN_VIDEO_DURATION:
            cut_filepath = video_filepath.with_name(f"0_{location_id}{location_sequence}_recut.TXT")
            cut_start = 0
            cut_end = round(timestamp_anomaly_frame - FPS)
            cuts.append((cut_filepath, cut_start, cut_end))
        else:
            cut_filepath_1 = video_filepath.with_name(f"0_{location_id}{location_sequence}_{location_id}{location_sequence}_recut.TXT")
            cut_filepath_2 = video_filepath.with_name(f"0_{location_id}{location_sequence}_{location_id}{location_sequence+1}_recut.TXT")
            cut_start_1 = 0
            cut_end_1 = round(timestamp_anomaly_frame - FPS)
            cut_start_2 = round(timestamp_anomaly_frame + FPS)
            cut_end_2 = -1
            cuts.append((cut_filepath_1, cut_start_1, cut_end_1))
            cuts.append((cut_filepath_2, cut_start_2, cut_end_2))

        # create cut files
        for cut in cuts:
            cut_filepath, cut_start, cut_end = cut
            print(f"Creating cut: {cut_filepath} with start: {cut_start} and end: {cut_end}")
            if not args.debug:
                with open(cut_filepath, 'w') as f:
                    f.write(f"{cut_start}, {cut_end}")

        # rename the original files
        video_filepath_original = video_filepath.with_name(video_filepath.stem + "_original" + video_filepath.suffix)
        csv_filepath_original = csv_filepath.with_name(csv_filepath.stem + "_original" + csv_filepath.suffix)
        print(f"Renaming: {video_filepath} to {video_filepath_original}")
        if not args.debug:
            os.rename(video_filepath, video_filepath_original)
        print(f"Renaming: {csv_filepath} to {csv_filepath_original}")
        if not args.debug:
            os.rename(csv_filepath, csv_filepath_original)

        # recut the videos and process the cut files
        for cut in cuts:
            cut_filepath = cut[0]
            output_filepath = cut_filepath.with_name(cut_filepath.stem.split('_')[-2] + video_filepath.suffix)
            cmd1 = f"python tools/recut_video_and_csv.py {video_filepath_original} {cut_filepath} -o {output_filepath}"
            print(f"\033[92mRunning: {cmd1}\033[0m")
            if not args.debug:
                os.system(cmd1)

            cmd2 = f"python batch_process.py {output_filepath} -y -o"
            print(f"\033[92mRunning: {cmd2}\033[0m")
            if not args.debug:
                os.system(cmd2)


def get_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fix timestamp anomalies in flight logs by cutting videos and logs at anomaly frames.')
    parser.add_argument('input', type=Path, help="Path to file containing the flight log anomalies.")
    parser.add_argument("--processed-folder", "-o", type=Path, default= None, help="Path to the root of the processed folder containing the cut videos and flight logs (default: same as input).")
    parser.add_argument("--debug", "-d", action='store_true', help="Run in debug mode (no files will be modified).")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()
    fix_timetamp_annomalies(args)
