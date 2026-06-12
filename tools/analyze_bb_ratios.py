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
  -h, --help          : Show this help message and exit.
  -v, --verbose       : Print detailed statistical results for each video file processed.
  -hs, --hist         : Generate and display a histogram of length-to-width ratios for each vehicle class.
  -p, --plot          : Plot length and height histogram per video and vehicle ID.
  -i, --id <int>      : Specify a vehicle ID for detailed analysis.

Examples:
1. Analyze a single video file with verbose output:
   python tools/analyze_bb_ratios.py video.yaml --verbose

2. Perform a batch analysis on a directory and show histograms:
   python tools/analyze_bb_ratios.py data/ --hist

3. Analyze a specific vehicle ID from a video:
   python tools/analyze_bb_ratios.py video.yaml --id 42 --verbose

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compare_dimension_estimators import estimate_vehicle_dimensions_new

from geotrax.utils.file_utils import detect_delimiter

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


def main(args):
    # Initialize results
    results = {}

    # Check if input is a file or a directory
    if args.source.is_file():
        results = process_file(args.source, args)
    elif args.source.is_dir():
        results = process_dir(args.source, args)
    else:
        raise FileNotFoundError(f"File or directory {args.source} not found.")

    # Analyze the results
    analyze_results(results, args)


def process_dir(directory, args):
    all_class_ratios = {}
    for file in sorted(directory.iterdir()):
        if file.is_file():
            class_ratios = process_file(file, args)
            all_class_ratios = append_results(all_class_ratios, class_ratios)
        elif file.is_dir():
            sub_dir_ratios = process_dir(file, args)
            all_class_ratios = append_results(all_class_ratios, sub_dir_ratios)
        else:
            raise FileNotFoundError(f"File or directory {file} not found.")

    return all_class_ratios


def process_file(file, args):
    # Check if the input is a valid video file or a YAML file
    if file.suffix.lower() not in {'.yaml'} | VIDEO_FORMATS:
        return None
    # Check if the file is in the results directory
    if file.parent.name == 'results':
        return None

    # Load tracks
    tracks_txt_file = Path(f"{str(file.parent / 'results' / file.stem)}.txt")
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
    print(f"Processing: {tracks_txt_file}")
    tracks = estimate_vehicle_dimensions_new(tracks, args, tau_c=tau_c_restrictive, theta_bar_deg=theta_bar_deg_restrictive)

    # Extract the width and length ratios per vehicle class
    class2ratios = extract_ratios(tracks)
    if args.verbose:
        for class_id, ratios in class2ratios.items():
            print(f"  Class: {DEFAULT_CLASS_NAMES[class_id]} - N: {len(ratios)}")

        # Analyze the results per video
        analyze_results(class2ratios, args)

    return class2ratios


def extract_ratios(tracks):
    idx_c = 10 if tracks.shape[1] > 10 else 6
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    unique_cls = np.unique(tracks[:, idx_c]).astype(int)
    class2ratios = {c: [] for c in unique_cls}
    for class_id in unique_cls:
        for vehicle_id in unique_ids:
            mask = (tracks[:, 1] == vehicle_id) & (tracks[:, idx_c] == class_id)
            if np.sum(mask) > 0:
                L = tracks[mask, -2][0]
                W = tracks[mask, -1][0]
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


def analyze_results(class2ratios, args):
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
        print(f"\nClass: {DEFAULT_CLASS_NAMES[class_id]}")
        print(f"  N: {ratios_N}")
        print(f"  Mean: {ratios_mean:.2f}")
        print(f"  Std: {ratios_std:.2f}")
        print(f"  Median: {ratios_median:.2f}")
        print(f"  Min: {ratios_min:.2f}")
        print(f"  Max: {ratios_max:.2f}")
        print(f"  Q10: {ratios_q10:.2f}")
        print(f"  Q5: {ratios_q5:.2f}")
        print(f"  Q1: {ratios_q1:.2f}")

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


def get_cli_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze vehicle bounding box ratios from tracking data")
    parser.add_argument("source", type=Path, help="Path to a directory containing detection and tracking results or to a specific video/yaml file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results per video")
    parser.add_argument("--hist", "-hs", action="store_true", help="Plot ratio histograms per vehicle class")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot length and height histogram per video and vehicle ID")
    parser.add_argument("--id", "-i", type=int, default=0, help="Vehicle ID to analyze in detail (default: User prompt)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_arguments()
    main(args)
