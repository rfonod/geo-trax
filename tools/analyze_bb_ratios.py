#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
analyze_bb_ratios.py - Vehicle Bounding Box Ratio Analysis

This script analyzes the length-to-width ratios of vehicle bounding boxes based on detection and tracking data.
It can process a single data file or recursively scan a directory of files. For each processed file, it estimates
vehicle dimensions and calculates the length-to-width ratio.

The script aggregates these ratios by vehicle class and computes descriptive statistics (mean, standard deviation,
median, min/max, and percentiles). It can also generate and display histograms for the ratio distribution of each
vehicle class.

Usage:
  python tools/analyze_bb_ratios.py <source> [options]

Arguments:
  source              : Path to a video/yaml file or a directory containing tracking data.

Options:
  -h, --help            : Show this help message and exit.
  -hs, --hist           : Generate and display a histogram of length-to-width ratios for each vehicle class.
  -p, --plot            : Plot length and height histogram per video and vehicle ID.
  -i, --id <int>        : Specify a vehicle ID for detailed analysis.
  -c, --cfg <path>      : Pipeline config used to resolve the output folder and filename postfixes.
                          Defaults to the bundled config (geotrax/cfg/default.yaml).
  -lp, --log-path <str> : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet           : Reduce console verbosity to important messages only (default: show INFO-level per-video detail).

Examples:
1. Analyze a single video file:
   python tools/analyze_bb_ratios.py video.yaml

2. Perform a batch analysis on a directory and show histograms:
   python tools/analyze_bb_ratios.py data/ --hist

3. Analyze a specific vehicle ID from a video:
   python tools/analyze_bb_ratios.py video.yaml --id 42

Input:
- A path to a video file (e.g., .mp4, .mov), a YAML configuration file, or a directory.
- Corresponding tracking data must be available in a '.txt' file located in a 'results/' subdirectory
  (e.g., for 'data/video.mp4', the script expects 'data/results/video.txt').
- The tracking file should contain columns for frame number, object ID, class ID, and vehicle dimensions (length and width).

Output:
- Console output summarizing the statistical analysis of length-to-width ratios for each vehicle class,
  including count, mean, standard deviation, median, min/max, and various percentiles.
- If the '--hist' option is used, matplotlib plots showing the distribution of the ratios for each class.

Notes:
- The script uses restrictive thresholds for vehicle speed (tau_c) and orientation change (theta_bar_deg)
  to filter out stationary or erratically moving vehicles from the analysis.
- It ignores any files located directly within a 'results/' directory to avoid processing its own output.
- Vehicle classes are predefined as: Car, Bus, Truck, Motorcycle, Pedestrian, Bicycle.
- The data format is specific to the experiments in the geo-trax paper (DOI: 10.1016/j.trc.2025.105205).
"""

import argparse
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compare_dimension_estimators import estimate_vehicle_dimensions_new

from geotrax.utils.cli_utils import DEFAULT_CFG
from geotrax.utils.config_utils import load_config
from geotrax.utils.file_utils import DEFAULT_OUTPUT, detect_delimiter, get_output_dir
from geotrax.utils.logging_utils import setup_logger

DEFAULT_CLASS_NAMES = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Pedestrian', 'Bicycle']
VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}

# Set tau_c such that stationary vehicles are ignored in this analysis
tau_c_restrictive = {
    0: 100,  # car (vans, SUVs, etc.)
    1: 100,  # bus
    2: 100,  # truck
    3: 100,  # motorcycle
    -1: 100,
}  # unknown

theta_bar_deg_restrictive = 5


def analyze_bb_ratios(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze vehicle bounding-box length-to-width ratios for the input file or directory."""
    # Initialize results
    results = {}

    # Check if input is a file or a directory
    if args.source.is_file():
        results = process_file(args.source, args, logger)
    elif args.source.is_dir():
        results = process_dir(args.source, args, logger)
    else:
        raise FileNotFoundError(f"File or directory {args.source} not found.")

    # Analyze the results
    analyze_results(results, args, logger)


def process_dir(directory, args, logger):
    all_class_ratios = {}
    for file in sorted(directory.iterdir()):
        if file.is_file():
            class_ratios = process_file(file, args, logger)
            all_class_ratios = append_results(all_class_ratios, class_ratios)
        elif file.is_dir():
            sub_dir_ratios = process_dir(file, args, logger)
            all_class_ratios = append_results(all_class_ratios, sub_dir_ratios)
        else:
            raise FileNotFoundError(f"File or directory {file} not found.")

    return all_class_ratios


def process_file(file, args, logger):
    # Check if the input is a valid video file or a YAML file
    if file.suffix.lower() not in {'.yaml'} | VIDEO_FORMATS:
        return None
    # Skip files that live inside the output folder itself
    output_cfg = load_config(args.cfg, logger).get('output', DEFAULT_OUTPUT)
    folder_name = output_cfg.get('folder', DEFAULT_OUTPUT['folder'])
    if file.parent.name == folder_name:
        return None

    # Load tracks
    tracks_postfix = output_cfg.get('tracks_postfix', DEFAULT_OUTPUT['tracks_postfix'])
    out_dir = get_output_dir(file, output_cfg)
    tracks_txt_file = out_dir / f"{file.stem}{tracks_postfix}.txt"
    if not tracks_txt_file.exists():
        return None

    # Detect delimiter
    delimiter = detect_delimiter(tracks_txt_file)

    # Load tracks
    tracks = np.loadtxt(tracks_txt_file, delimiter=delimiter)

    # Modify args to process the file
    args = copy.deepcopy(args)
    args.source = file

    # Estimate vehicle dimensions
    logger.info(f"Processing: {tracks_txt_file}")
    tracks = estimate_vehicle_dimensions_new(tracks, args, logger, tau_c=tau_c_restrictive, theta_bar_deg=theta_bar_deg_restrictive)

    # Extract the width and length ratios per vehicle class
    class2ratios = extract_ratios(tracks)
    if not args.quiet:
        for class_id, ratios in class2ratios.items():
            logger.info(f"  Class: {DEFAULT_CLASS_NAMES[class_id]} - N: {len(ratios)}")

        # Analyze the results per video
        analyze_results(class2ratios, args, logger)

    return class2ratios


def extract_ratios(tracks):
    has_stab = tracks.shape[1] >= 12  # stab: 14/15-col; no-stab: 10/11-col
    idx_c = 10 if has_stab else 6
    dim_start = 12 if has_stab else 8
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    unique_cls = np.unique(tracks[:, idx_c]).astype(int)
    class2ratios = {c: [] for c in unique_cls}
    for class_id in unique_cls:
        for vehicle_id in unique_ids:
            mask = (tracks[:, 1] == vehicle_id) & (tracks[:, idx_c] == class_id)
            if np.sum(mask) > 0:
                L = tracks[mask, dim_start][0]
                W = tracks[mask, dim_start + 1][0]
                ratio = L / W if W is not None and W > 0 else None
                if ratio is not None:
                    class2ratios[class_id].append(ratio)

    return class2ratios


def append_results(results, new_results):
    if new_results is not None:
        for class_id, ratios in new_results.items():
            if class_id not in results:
                results[class_id] = []
            results[class_id].extend(ratios)

    return results


def analyze_results(class2ratios, args, logger):
    for class_id, ratios in class2ratios.items():
        if len(ratios) == 0:
            continue
        ratios_N = len(ratios)
        ratios_mean = np.mean(ratios)
        ratios_std = np.std(ratios)
        ratios_median = np.median(ratios)
        ratios_min = np.min(ratios)
        ratios_max = np.max(ratios)
        ratios_q10 = np.percentile(ratios, 10)
        ratios_q5 = np.percentile(ratios, 5)
        ratios_q1 = np.percentile(ratios, 1)
        logger.notice(
            f"Class: {DEFAULT_CLASS_NAMES[class_id]}\n"
            f"  N: {ratios_N}\n"
            f"  Mean: {ratios_mean:.2f}\n"
            f"  Std: {ratios_std:.2f}\n"
            f"  Median: {ratios_median:.2f}\n"
            f"  Min: {ratios_min:.2f}\n"
            f"  Max: {ratios_max:.2f}\n"
            f"  Q10: {ratios_q10:.2f}\n"
            f"  Q5: {ratios_q5:.2f}\n"
            f"  Q1: {ratios_q1:.2f}"
        )

        if args.hist:
            plt.figure()
            plt.hist(ratios, bins=50, color='c', edgecolor='k', alpha=0.7)
            plt.axvline(ratios_mean, color='k', linestyle='dashed', linewidth=1.5, label='Mean')
            plt.axvline(ratios_mean - ratios_std, color='r', linestyle='dashed', linewidth=1.5, label='Mean +/- 1*Std')
            plt.axvline(ratios_mean + ratios_std, color='r', linestyle='dashed', linewidth=1.5)
            plt.axvline(
                ratios_mean - 2 * ratios_std, color='gray', linestyle='dashdot', linewidth=1.5, label='Mean +/- 2*Std'
            )
            plt.axvline(ratios_mean + 2 * ratios_std, color='gray', linestyle='dashdot', linewidth=1.5)
            plt.axvline(
                ratios_mean - 3 * ratios_std, color='orange', linestyle='dashdot', linewidth=1.5, label='Mean +/- 3*Std'
            )
            plt.axvline(ratios_mean + 3 * ratios_std, color='orange', linestyle='dashdot', linewidth=1.5)
            plt.axvline(ratios_q10, color='g', linestyle='solid', linewidth=1.5, label='Q10')
            plt.axvline(ratios_q5, color='m', linestyle='solid', linewidth=1.5, label='Q5')
            plt.axvline(ratios_q1, color='b', linestyle='solid', linewidth=1.5, label='Q1')
            plt.xlabel("L/W ratio")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title(f"{DEFAULT_CLASS_NAMES[class_id]}")
            plt.show(block=False)

    plt.show()


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze vehicle bounding box ratios from tracking data")
    parser.add_argument("source", type=Path, help="Path to a directory containing detection and tracking results or to a specific video/yaml file")
    parser.add_argument("--hist", "-hs", action="store_true", help="Plot ratio histograms per vehicle class")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot length and height histogram per video and vehicle ID")
    parser.add_argument("--id", "-i", type=int, default=0, help="Vehicle ID to analyze in detail (default: User prompt)")
    parser.add_argument("--cfg", "-c", type=Path, default=DEFAULT_CFG, help="Pipeline config used to resolve the output folder and filename postfixes. Defaults to the bundled config.")
    parser.add_argument("--log-path", "-lp", type=Path, default=None, help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console verbosity to important messages only (default: show INFO-level per-video detail).")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    analyze_bb_ratios(args, logger)


if __name__ == "__main__":
    main()
