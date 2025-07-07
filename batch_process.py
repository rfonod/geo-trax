#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
batch_process.py - Process all videos in a folder (including sub-folders) or a single video file.

This script allows batch processing of videos for detection, tracking, stabilization, georeferencing,
and visualization. It provides extensive customization options for each processing step.

Usage:
  python batch_process.py <input_path> [options]

Arguments:
  input_path : Path to the input directory or specific video file.

Batch Processing Options:
    -h, --help        : Show this help message and exit.
    -y, --yes         : Automatically confirm prompts without waiting for user input (default: False).
    -o, --overwrite   : Overwrite existing files that have already been processed (default: False).
    -dr, --dry-run    : Simulate the command execution without actually running any processes (default: False).
    -vo, --viz-only   : Only visualize the results; skip any processing operations (default: False).
    -go, --geo-only   : Only run georeferencing; skip detection, tracking, and stabilization (default: False).
    -ng, --no-geo     : Do not georeference the tracking data (default: False).
    -fe, --folders-exclude <str> [<str> ...] : Folders to exclude from the batch processing (default: ['results']).
    -ep, --exclude-patterns <str> [<str> ...] : File name patterns to exclude
                        (e.g., --exclude-patterns car_test drone_2023) (default: None).

Shared Processing, Georeferencing, and Visualization Options:
    -c, --cfg <path>    : Path to the main geo-trax configuration file (default: cfg/default.yaml).
    -lf, --log-file <str> : Filename to save detailed logs. Saved in the 'logs' folder (default: None).
    -v, --verbose       : Set print verbosity level to INFO (default: WARNING).

Detection and Tracking Options:
    -cls, --classes <int> [<int> ...] : Class IDs to extract (e.g., --classes 0 1 2).
                        Defaults to cfg -> cfg_ultralytics -> classes (default: None).
    -cfl, --cut-frame-left <int> : Skip the first N frames (default: 0).
    -cfr, --cut-frame-right <int> : Stop processing after this frame. Only considered if input
                        is a single video file (default: None).

Georeferencing Options:
    -of, --ortho-folder <path> : Custom path to the folder with orthophotos (.png, .tif, .txt).
                        Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input' (default: None).
    -gs, --geo-source <choice> : Source of georeferencing parameters. Choices: metadata-tif,
                        text-file, center-text-file. If not provided, the system will auto-detect (default: None).
    -rf, --ref-frame <int> : Use custom reference frame number (should be the same as the one
                        used for stabilization) (default: 0).
    -nm, --no-master    : Disable the master frame approach (default: False).
    -mf, --master-folder <path> : Custom path to the folder containing master frame files (.png).
                        If not provided, '--ortho-folder / master_frames' will be used (default: None).
    -r, --recompute     : Force recompute master-> ortho homography even if it exists (default: False).
    -osf, --segmentation-folder <path> : Custom path to the folder containing orthophoto
                        segmentation files (.csv). If not provided, '--ortho-folder / segmentations'
                        will be used (default: None).

Visualization Options:
    -s, --save          : Save the processing results to a video file (default: False).
    -sh, --show         : Visualize results during processing (default: False).
    -vm, --viz-mode <int> : Set visualization mode for the output video: 0 - original,
                        1 - stabilized, 2 - reference frame (default: 0).
    -pt, --plot-trajectories : Plot trajectories on the reference frame at the beginning
                        of the video (default: False).
    -pd, --plot-delay <int> : Delay in frames for plotting trajectories (default: 30).
    -sc, --show-conf    : Show confidence values (default: False).
    -sl, --show-lanes   : Show lane numbers (default: False).
    -scn, --show-class-names : Show class names (default: False).
    -hl, --hide-labels  : Hide labels entirely (default: False).
    -ht, --hide-tracks  : Hide trailing tracking lines (default: False).
    -hs, --hide-speed   : Hide speed values (default: False).
    -cf, --class-filter <int> [<int> ...] : Exclude specified classes (e.g., -cf 1 2) (default: None).

Examples:
  1. Process a directory without saving/showing video visualization:
     python batch_process.py path/to/videos/

  2. Process a directory without georeferencing and save video visualization:
     python batch_process.py path/to/videos/ --no-geo --save

  3. Process a single video file with a custom configuration file:
     python batch_process.py path/to/video.mp4 -c cfg/custom_config.yaml

  4. Save video visualization without re-running detection, tracking, stabilization, and georeferencing:
     python batch_process.py video.mp4 --save --viz-only

  5. Overwrite existing files with no confirmation needed:
     python batch_process.py path/to/videos/ -o -y

Notes:
  - Ensure that all paths provided are accessible and that necessary permissions are set.
  - Check that all required dependencies and modules are properly installed.
  - Additional configurations can be set in the main configuration file (default: cfg/default.yaml)
    and linked config files therein.
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from detect_track_stabilize import detect_track_stabilize
from georeference import georeference
from utils.utils import bcolors, check_if_results_exist, determine_suffix_and_fourcc, setup_logger
from visualize import visualize_results

VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}


def process_input(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the input file or directory.
    """
    input_path = args.input
    if not input_path.exists():
        logger.critical(f"File or directory '{input_path}' not found.")
        return

    try:
        if input_path.is_file() and input_path.suffix.lower() in VIDEO_FORMATS:
            process_file(input_path, args, logger)
        elif input_path.is_dir():
            logger.notice(f"{bcolors.OKGREEN}Batch processing all videos in: '{input_path}'{bcolors.ENDC}")
            args.cut_frame_right = None
            potential_files_to_process = [file for file in input_path.rglob('*') if file.is_file() and file.suffix.lower() in VIDEO_FORMATS]
            files_to_process = filter_files_to_process(potential_files_to_process, args, logger)
            files_to_process = sorted(files_to_process)

            pbar = tqdm(files_to_process, unit="video")
            for file in files_to_process:
                pbar.set_description(f"Processing: '{file}'")
                process_file(file, args, logger)
                pbar.update(1)
    except KeyboardInterrupt:
        logger.error("Batch processing interrupted by user.")


def process_file(file: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the file if it is a video file and not in the results directory.
    """
    try:
        logger.info(f"Processing: '{file}'")
        if not args.viz_only and not args.geo_only:
            process_step(file, args, logger, "Detecting, tracking, and stabilizing", detect_track_stabilize)

        if not args.viz_only and not args.no_geo:
            process_step(file, args, logger, "Georeferencing", georeference)

        if args.save or args.show:
            process_step(file, args, logger, "Visualizing", visualize_results)

    except Exception as e:
        logger.error(f"Error with {file}: {e}")


def process_step(file: Path, args: argparse.Namespace, logger: logging.Logger, action: str, func) -> None:
    """
    Process a specific step (processing, georeferencing, visualizing) for the file.
    """
    if should_process_file(file, args, logger, action):
        logger.info(f"{action}: '{file}'")
        if not args.dry_run:
            args.source = file
            func(args, logger)


def filter_files_to_process(files: list, args: argparse.Namespace, logger: logging.Logger) -> list:
    """
    Filter files based on exclusion criteria (folders and patterns).
    """
    filtered_files = []
    for file in files:
        if file.parent.name in args.folders_exclude:
            logger.info(f"Skipping '{file}' as it's in an excluded folder.")
            continue

        if args.exclude_patterns and any(pattern in file.name for pattern in args.exclude_patterns):
            logger.info(f"Skipping '{file}' due to matching exclusion pattern.")
            continue

        filtered_files.append(file)

    return filtered_files


def should_process_file(file: Path, args: argparse.Namespace, logger: logging.Logger, action: str) -> bool:
    """
    Determine if the video should be processed, georeferenced, or visualized based on the existing results and user input.
    """
    suffix = determine_suffix_and_fourcc()[0]
    txt_exists = check_if_results_exist(file, "processed")[0]
    csv_exists = check_if_results_exist(file, "georeferenced")[0]
    vid_exists = check_if_results_exist(file, "visualized", args.viz_mode, suffix)[0]

    processing_steps = "detection, tracking, and stabilization"
    if action == "Detecting, tracking, and stabilizing":
        return handle_existing_results(file, args, logger, txt_exists, processing_steps)
    elif action == "Georeferencing":
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping georeferencing.")
            return False
        return handle_existing_results(file, args, logger, csv_exists, action)
    elif action == "Visualizing":
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping visualization.")
            return False
        return handle_existing_results(file, args, logger, vid_exists, action)
    return False


def handle_existing_results(file: Path, args: argparse.Namespace, logger: logging.Logger, exists: bool, action: str) -> bool:
    """
    Handle existing results based on user input and overwrite options.
    """
    if exists and not args.overwrite:
        logger.warning(f"'{file}' - {action} results already exist and overwrite not allowed.")
        return False
    elif exists and args.overwrite and not args.yes:
        user_input = input(f"{bcolors.BOLD}Overwrite {action} results for: '{file}'? [y/n]: {bcolors.ENDC}").lower()
        return user_input == 'y'
    return True


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process all videos in a folder (including sub-folders) or a single video file.')

    # Required arguments
    parser.add_argument('input', type=Path, help='Path to the input directory or video file (e.g., path/to/video_dir/)')

    # Batch processing options
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirm prompts without waiting for user input')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing files that have already been processed')
    parser.add_argument('--dry-run', '-dr', action='store_true', help='Simulate the command execution without actually running any processes')
    parser.add_argument('--viz-only', '-vo', action='store_true', help='Only visualize the results; skip any processing operations')
    parser.add_argument('--geo-only', '-go', action='store_true', help='Only run georeferencing; skip detection, tracking, and stabilization')
    parser.add_argument('--no-geo', '-ng', action='store_true', help='Do not georeference the tracking data')
    parser.add_argument("--folders-exclude", "-fe", type=str, nargs='+', default=['results'], help="Folders to exclude from the batch processing")
    parser.add_argument("--exclude-patterns", "-ep", type=str, nargs='+', default=None, help="File name patterns to exclude (e.g., --exclude-patterns car_test drone_2023)")

    # Shared processing, georeferencing, and visualization options
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING)')

    # Detection and tracking options
    parser.add_argument('--classes', '-cls', nargs='+', type=int, help='Class IDs to extract (e.g., --classes 0 1 2). Defaults to cfg -> cfg_ultralytics -> classes.')
    parser.add_argument('--cut-frame-left', '-cfl', type=int, default=0, help='Skip the first N frames. Default: 0.')
    parser.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Stop processing after this frame. Only considered if input is a single video file.')

    # Georeferencing options
    parser.add_argument("--ortho-folder", "-of", type=Path, default=None, help="Custom path to the folder with orthophotos (.png, .tif, .txt). Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.")
    parser.add_argument("--geo-source", "-gs", choices=['metadata-tif', 'text-file', 'center-text-file'], default=None, help="Source of georeferencing parameters. If not provided, the system will auto-detect")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="Use custom reference frame number (should be the same as the one used for stabilization).")
    parser.add_argument("--no-master", "-nm", action="store_true", help="Disable the master frame approach.")
    parser.add_argument("--master-folder", "-mf", type=Path, default=None, help="Custom path to the folder containing master frame files (.png). If not provided, '--ortho-folder / master_frames' will be used.")
    parser.add_argument("--recompute", "-r", action="store_true", help="Force recompute master-> ortho homography even if it exists.")
    parser.add_argument("--segmentation-folder", "-osf", type=Path, default=None, help="Custom path to the folder containing orthophoto segmentation files (.csv). If not provided, '--ortho-folder / segmentations' will be used.")

    # Visualization options
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--save', '-s', action='store_true', help='Save the processing results to a video file')
    group.add_argument('--show', '-sh', action='store_true', help='Visualize results during processing')
    parser.add_argument('--viz-mode', '-vm', type=int, default=0, choices=[0, 1, 2], help='Set visualization mode for the output video: 0 - original, 1 - stabilized, 2 - reference frame')
    parser.add_argument("--plot-trajectories", "-pt", action="store_true", help='Plot trajectories on the reference frame')
    parser.add_argument("--plot-delay", "-pd", type=int, default=30, help='Delay in frames for plotting trajectories')
    parser.add_argument("--show-conf", "-sc", action="store_true", help='Show confidence values')
    parser.add_argument("--show-lanes", "-sl", action="store_true", help='Show lane numbers')
    parser.add_argument("--show-class-names", "-scn", action="store_true", help='Show class names')
    parser.add_argument("--hide-labels", "-hl", action="store_true", help='Hide labels entirely')
    parser.add_argument("--hide-tracks", "-ht", action="store_true", help='Hide trailing tracking lines')
    parser.add_argument("--hide-speed", "-hs", action="store_true", help='Hide speed values')
    parser.add_argument('--class-filter', '-cf', type=int, nargs='+', help='Exclude specified classes (e.g., -cf 1 2)')

    return parser.parse_args()


def main() -> None:
    """
    Main function to process the input file or directory.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file, args.dry_run)

    process_input(args, logger)


if __name__ == "__main__":
    main()
