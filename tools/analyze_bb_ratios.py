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
  -c, --cfg <path>      : Pipeline config used to resolve the output folder, filename postfixes, and the
                          dimension-estimation parameters.
                          Defaults to the bundled config (geotrax/cfg/default.yaml).
  -lp, --log-path <str> : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet           : Reduce console verbosity to important messages only (default: show INFO-level per-video detail).

Examples:
1. Analyze a single video file:
   python tools/analyze_bb_ratios.py video.mp4

2. Perform a batch analysis on a directory and show histograms:
   python tools/analyze_bb_ratios.py data/ --hist

3. Name a video through its run-metadata YAML:
   python tools/analyze_bb_ratios.py results/video.yaml

Input:
- A path to a video file (e.g., .mp4, .mov), a run-metadata YAML file, or a directory.
- Corresponding tracking data must be available in a '.txt' file located in a 'results/' subdirectory
  (e.g., for 'data/video.mp4', the script expects 'data/results/video.txt').
- A run-metadata YAML only names its video: the video file itself must sit next to the output folder,
  since the estimator needs the frame size to apply the visibility filter.
- The tracking file should contain columns for frame number, object ID, class ID, and vehicle dimensions (length and width).

Output:
- Console output summarizing the statistical analysis of length-to-width ratios for each vehicle class,
  including count, mean, standard deviation, median, min/max, and various percentiles.
- If the '--hist' option is used, matplotlib plots showing the distribution of the ratios for each class.

Notes:
- The script uses restrictive thresholds for vehicle speed (tau_c) and orientation change (theta_bar_deg)
  to filter out stationary or erratically moving vehicles from the analysis.
- A directory scan ignores files inside the output folder and does not descend into it: the run-metadata
  YAML there only names the video one level up, which the scan has already counted. Naming the output
  folder itself on the command line still works.
- Dimensions are recomputed with geotrax's own estimator (geotrax.extract.estimate_vehicle_dimensions),
  overriding only tau_c and theta_bar with the restrictive values defined below; every other parameter
  comes from the pipeline config. For a step-by-step visualization of that estimator on a single
  vehicle ID, use tools/viz_dimension_estimation.py.
- Vehicle classes are predefined as: Car, Bus, Truck, Motorcycle, Pedestrian, Bicycle.
- The data format is specific to the experiments in the geo-trax paper (DOI: 10.1016/j.trc.2025.105205).
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geotrax.extract import estimate_vehicle_dimensions
from geotrax.utils.cli_utils import DEFAULT_CFG
from geotrax.utils.config_utils import load_config
from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.file_utils import DEFAULT_OUTPUT, detect_delimiter, get_output_dir
from geotrax.utils.logging_utils import setup_logger

DEFAULT_CLASS_NAMES = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Pedestrian', 'Bicycle']

# Set tau_c such that stationary vehicles are ignored in this analysis
TAU_C_RESTRICTIVE = {
    0: 100,  # car (vans, SUVs, etc.)
    1: 100,  # bus
    2: 100,  # truck
    3: 100,  # motorcycle
    -1: 100,  # unknown
}

THETA_BAR_RESTRICTIVE = 5  # [deg]


def analyze_bb_ratios(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze vehicle bounding-box length-to-width ratios for the input file or directory."""
    # Initialize results
    results = {}

    # Parse the config once: a recursive scan would otherwise re-read the same static YAML for
    # every directory and every candidate file.
    cfg = build_context(args, logger)

    # Check if input is a file or a directory
    if args.source.is_file():
        results = process_file(args.source, args, logger, cfg)
    elif args.source.is_dir():
        results = process_dir(args.source, args, logger, cfg)
    else:
        raise FileNotFoundError(f"File or directory {args.source} not found.")

    # Analyze the results (process_file returns None for anything it skipped)
    analyze_results(results or {}, args, logger)


def build_context(args, logger):
    """Resolve everything the scan needs from the pipeline config, once."""
    cfg = load_config(args.cfg, logger)
    output_cfg = cfg.get('output', DEFAULT_OUTPUT)
    dim_cfg = dict(cfg.get('extraction', {}).get('dimension_estimation', {}))
    if not dim_cfg:
        logger.critical(f"Config '{args.cfg}' has no 'extraction -> dimension_estimation' section.")
        sys.exit(1)
    dim_cfg['tau_c'] = TAU_C_RESTRICTIVE
    dim_cfg['theta_bar'] = THETA_BAR_RESTRICTIVE
    # output.folder may be nested ('out/results') or absolute (a shared root), so compare its leaf
    return {
        'output': output_cfg,
        'folder_name': Path(output_cfg.get('folder', DEFAULT_OUTPUT['folder'])).name,
        'dim': dim_cfg,
    }


def process_dir(directory, args, logger, cfg):
    folder_name = cfg['folder_name']
    all_class_ratios = {}
    for file in sorted(directory.iterdir()):
        if file.is_file():
            class_ratios = process_file(file, args, logger, cfg)
            all_class_ratios = append_results(all_class_ratios, class_ratios)
        elif file.is_dir():
            # Do not descend into an output folder: its run-metadata YAML only names the video
            # sitting one level up, which this scan has already processed. Counting both would
            # double every ratio. A folder named directly on the command line is still scanned.
            if file.name == folder_name:
                continue
            sub_dir_ratios = process_dir(file, args, logger, cfg)
            all_class_ratios = append_results(all_class_ratios, sub_dir_ratios)
        else:
            raise FileNotFoundError(f"File or directory {file} not found.")

    return all_class_ratios


def process_file(file, args, logger, cfg):
    # Check if the input is a valid video file or a YAML file
    if file.suffix.lower() not in {'.yaml'} | VIDEO_FORMATS:
        return None
    output_cfg, folder_name = cfg['output'], cfg['folder_name']
    # The run-metadata YAML is written inside the output folder, so a '<results>/<stem>.yaml'
    # argument names a video one level up. Rebase it before the output folder is derived from
    # the path; a YAML still sitting next to its video (pre-v1.4.0 layout) needs no rebasing.
    if file.suffix.lower() == '.yaml' and file.parent.name == folder_name:
        file = file.parent.parent / file.name
    # Skip files that live inside the output folder itself
    elif file.parent.name == folder_name:
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
    if tracks.size == 0:  # an empty file reshapes to a fabricated (1, 0) row, so bail out first
        logger.warning(f"No tracks in '{tracks_txt_file}'; skipping.")
        return None
    tracks = np.atleast_2d(tracks)
    has_stab = tracks.shape[1] >= 12  # stab: 14/15-col; no-stab: 10/11-col

    # estimate_vehicle_dimensions() re-derives the column layout from the row width, and its
    # `shape[1] > 8` test only holds for the pre-append layout it sees inside the pipeline. A saved
    # file already carries the two dimension columns (plus is_interpolated when extraction.interpolate
    # is on), so trim back to the 12/8 core: without this a no-stab file raises IndexError, and a
    # no-stab interpolated one silently reads dimensions as centres and returns all-NaN ratios.
    tracks = tracks[:, :12] if has_stab else tracks[:, :8]

    # The estimator needs the frame size, so a YAML source must be traded for its video
    video = resolve_video(file, logger)
    if video is None:
        return None

    # Modify args to process the file
    args = copy.deepcopy(args)
    args.source = video

    # Estimate vehicle dimensions, overriding only tau_c/theta_bar with the restrictive values
    logger.info(f"Processing: {tracks_txt_file}")
    tracks = estimate_vehicle_dimensions(tracks, {'args': args, 'extraction': {'dimension_estimation': cfg['dim']}})

    # Extract the width and length ratios per vehicle class
    class2ratios = extract_ratios(tracks, has_stab)
    if not args.quiet:
        for class_id, ratios in class2ratios.items():
            logger.info(f"  Class: {DEFAULT_CLASS_NAMES[class_id]} - N: {len(ratios)}")

        # Analyze the results per video
        analyze_results(class2ratios, args, logger)

    return class2ratios


def resolve_video(file, logger):
    """Return the video *file* names: itself, or the sibling video sharing its stem for a YAML source."""
    if file.suffix.lower() in VIDEO_FORMATS:
        return file
    # Match on the lowercased suffix rather than probing lowercase names: DJI writes '.MP4', and on a
    # case-sensitive filesystem with_suffix('.mp4') would never find it (cf. tools/find_source_id.py).
    video = next((p for p in sorted(file.parent.glob(file.stem + '.*')) if p.suffix.lower() in VIDEO_FORMATS), None)
    if video is None:
        logger.warning(f"No video found next to '{file}'; skipping (the estimator needs the frame size).")
    return video


def extract_ratios(tracks, has_stab):
    idx_c = 10 if has_stab else 6
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    unique_cls = np.unique(tracks[:, idx_c]).astype(int)
    class2ratios = {c: [] for c in unique_cls}
    for class_id in unique_cls:
        for vehicle_id in unique_ids:
            mask = (tracks[:, 1] == vehicle_id) & (tracks[:, idx_c] == class_id)
            if np.sum(mask) > 0:
                L = tracks[mask, -2][0]  # the estimator appends length/width as the last two columns
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
    parser.add_argument("--cfg", "-c", type=Path, default=DEFAULT_CFG, help="Pipeline config used to resolve the output folder, filename postfixes, and dimension-estimation parameters. Defaults to the bundled config.")
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
