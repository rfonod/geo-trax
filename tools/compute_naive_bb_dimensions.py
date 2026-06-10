#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
compute_naive_bb_dimensions.py - Naive Bounding-Box Vehicle Dimension Estimator

Estimates vehicle pixel-space dimensions (length and width) from bounding box
tracking results using a naive approach: for each tracked vehicle, the longer
and shorter bounding box sides are averaged over all fully visible frames.

This serves as a baseline for comparison against the main geo-trax pipeline's
sophisticated azimuth-based dimension estimator (see detect_track_stabilize.py).

Usage:
  python tools/compute_naive_bb_dimensions.py <directory> [options]

Arguments:
  directory : Path to directory containing tracking result files and vehicle ID files.

Options:
  -h, --help                     : Show this help message and exit.
  --av                           : Use built-in video-to-ID mapping for AV vehicles (Songdo experiment).
  --frame-size <height> <width>  : Frame dimensions in pixels (default: 2160 3840).
  --visibility-margin <px>       : Pixel margin from each frame edge to consider a bounding box
                                   fully visible (default: 4).

Examples:
1. Estimate AV vehicle dimensions from Songdo experiment results:
   python tools/compute_naive_bb_dimensions.py data/results/ --av

2. Estimate bus dimensions using bus ID files:
   python tools/compute_naive_bb_dimensions.py data/results/

3. Custom frame size and visibility margin:
   python tools/compute_naive_bb_dimensions.py data/results/ --frame-size 1080 1920 --visibility-margin 8

Input:
- Tracking result files (*.txt) with MOT-style columns: frame, id, x, y, width, height, ...
- Vehicle ID files (*_AV.txt or *_bus_ids.txt) listing vehicle IDs per video
- In --av mode, vehicle IDs are resolved from the built-in video2id mapping

Output:
- Per-vehicle average pixel length (longer bounding box side) and width (shorter side)
- Summary statistics (mean ± std) across all processed vehicles

Notes:
- Dimensions are in pixels; no conversion to physical units is performed
- Only fully visible bounding boxes (within the frame with a configurable margin) are used
- Length is defined as the longer bounding box side; width as the shorter side
- Bus ID files shall list only vehicles of the same bus type: meaningful average dimensions
  require a homogeneous fleet. Not all tracked buses qualify — the user must curate the
  ID list to include only vehicles whose type is known and consistent across videos.
- The AV mode (--av) was introduced because the AV's exact physical dimensions are known,
  making it a convenient ground-truth vehicle for validating the naive estimator against
  the pipeline's azimuth-based method (see detect_track_stabilize.py).
- AV vehicle IDs are specific to the Songdo experiment (DOI: 10.1016/j.trc.2025.105205)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.file_utils import detect_delimiter
except ImportError as e:
    print(
        "\033[91mCould not import 'detect_delimiter' from 'utils.file_utils'.\n"
        "Make sure you have installed the geo-trax package in editable mode:\n"
        "    pip install -e .\n"
        "from the project root directory.\033[0m"
    )
    raise e

# AV vehicle ID per video (specific to the Songdo experiment)
video2id = {
    'E1_AV': 17,
    'E2_AV': 14,
    'G1_AV': 16,
    'J1_AV': 66,
    'J2_AV': 50,
    'K1_AV': 27,
    'K2_AV': 78,
    'L1_AV': 62,
    'L2_AV': 82,
    'M1_AV': 27,
    'M2_AV': 41,
    'O1_AV': 21,
    'P1_AV': 42,
    'Q1_AV': 27,
    'Q2_AV': 25,
}

# Tracking result file column names (MOT-style format with stabilised coordinates)
TRACKING_COLUMNS = [
    'frame', 'id', 'x', 'y', 'width', 'height',
    'x_stab', 'y_stab', 'width_stab', 'height_stab',
    'confidence', 'class', 'our_length', 'our_width',
]


def filter_by_visibility(df: pd.DataFrame, frame_size: tuple = (2160, 3840), eps: int = 4) -> pd.Series:
    """
    Return a boolean mask of bounding boxes fully visible within the frame.

    Args:
        df: DataFrame with centre-format bounding box columns x, y, width, height.
        frame_size: (height, width) of the frame in pixels.
        eps: Minimum pixel margin from each frame edge.

    Returns:
        Boolean Series; True where the bounding box is fully inside the frame.
    """
    frame_height, frame_width = frame_size

    left   = df['x'] - df['width']  / 2
    right  = df['x'] + df['width']  / 2
    top    = df['y'] - df['height'] / 2
    bottom = df['y'] + df['height'] / 2

    visible_x = (left > eps) & (right  < frame_width  - eps - 1)
    visible_y = (top  > eps) & (bottom < frame_height - eps - 1)

    return visible_x & visible_y


def estimate_dimensions(args: argparse.Namespace) -> None:
    directory = args.directory
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    files = sorted(directory.glob("*_AV.txt" if args.av else "*_bus_ids.txt"))
    if not files:
        pattern = "*_AV.txt" if args.av else "*_bus_ids.txt"
        print(f"No files matching '{pattern}' found in '{directory}'.")
        sys.exit(1)

    all_lengths, all_widths = [], []

    print(f"\n{'Video':<8} {'ID':<5} {'Avg Length (px)':>15} {'Avg Width (px)':>14}")
    print("-" * 45)

    for file in files:
        file_stem = file.stem
        results_filename = file_stem if args.av else file_stem.replace("_bus_ids", "")
        results_file = file.parent / f"{results_filename}.txt"

        if not results_file.exists():
            print(f"\033[93mWarning: Results file '{results_file}' does not exist, skipping.\033[0m")
            continue

        # Resolve vehicle IDs
        if args.av:
            video_name = results_file.stem
            if video_name not in video2id:
                print(f"\033[93mWarning: '{video_name}' not found in video2id mapping, skipping.\033[0m")
                continue
            vehicle_ids = [video2id[video_name]]
        else:
            with open(file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                print(f"\033[93mWarning: Vehicle ID file '{file}' is empty, skipping.\033[0m")
                continue
            vehicle_ids = [int(v) for v in lines]

        # Load tracking results
        delimiter = detect_delimiter(results_file)
        tracking_df = pd.read_csv(
            results_file,
            delimiter=delimiter,
            header=None,
            names=TRACKING_COLUMNS,
        )

        for vehicle_id in vehicle_ids:
            vehicle_df = tracking_df[tracking_df['id'] == vehicle_id].copy()

            if vehicle_df.empty:
                print(f"\033[93mWarning: No tracking data for vehicle ID {vehicle_id} in '{results_filename}'.\033[0m")
                continue

            visibility_mask = filter_by_visibility(vehicle_df, frame_size=args.frame_size, eps=args.visibility_margin)
            vehicle_df = vehicle_df[visibility_mask]

            if vehicle_df.empty:
                print(f"\033[93mWarning: No fully visible detections for vehicle ID {vehicle_id} in '{results_filename}'.\033[0m")
                continue

            vehicle_df['longer_side']  = vehicle_df[['width', 'height']].max(axis=1)
            vehicle_df['shorter_side'] = vehicle_df[['width', 'height']].min(axis=1)

            avg_length = vehicle_df['longer_side'].mean()
            avg_width  = vehicle_df['shorter_side'].mean()

            all_lengths.append(avg_length)
            all_widths.append(avg_width)

            print(f"{results_filename:<8} {vehicle_id:<5} {avg_length:>15.4f} {avg_width:>14.4f}")

    # Summary statistics
    if all_lengths:
        print("-" * 45)
        print(f"\nSummary ({len(all_lengths)} vehicle(s) processed)")
        print(f"  Avg length (px) : {np.mean(all_lengths):.4f} ± {np.std(all_lengths):.4f}")
        print(f"  Avg width  (px) : {np.mean(all_widths):.4f} ± {np.std(all_widths):.4f}")
    else:
        print("\nNo valid results found.")


def get_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate vehicle pixel-space dimensions from bounding box tracking results."
    )
    parser.add_argument(
        "directory", type=Path,
        help="Directory containing tracking result files and vehicle ID files."
    )
    parser.add_argument(
        "--av", action="store_true",
        help="Use built-in video-to-ID mapping for AV vehicles (geo-trax experiments)."
    )
    parser.add_argument(
        "--frame-size", nargs=2, type=int, default=[2160, 3840], metavar=("HEIGHT", "WIDTH"),
        help="Frame size in pixels as HEIGHT WIDTH (default: 2160 3840)."
    )
    parser.add_argument(
        "--visibility-margin", type=int, default=4,
        help="Pixel margin from each frame edge to consider a bounding box fully visible (default: 4)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()
    args.frame_size = tuple(args.frame_size)
    estimate_dimensions(args)
