#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
batch_process.py - Batch Process Videos for Detection, Tracking, Stabilization, and Visualization

This script manages the batch processing of video files, applying vehicle detection, tracking, and 
trajectory stabilization techniques to generate and visualize results. It supports handling either single video files 
or entire directories of videos, including subdirectories.

Usage:
  python batch_process.py <input_path> [options]

Arguments:
  input_path : Path to the input directory or specific video file.

Batch Processing Options:
  --overwrite, -o        : Overwrite existing processed files (text or video) [default: False].
  --yes, -y              : Confirm all prompts automatically without user interaction [default: False].
  --dry-run, -dr         : Simulate the command execution without actually running any processes [default: False].
  --viz-only, -vo        : Run only the visualization step without processing [default: False].

Shared Processing and Visualization Options:
  --cfg, -c CFG          : Path to the main configuration file [default: 'cfg/default.yaml'].
  --log-file, -lf LOG_FILE : Specify log file name for saving detailed execution logs [default: None].
  --verbose, -v          : Enable verbose output [default: False].
  --cut-frame-left, -cfl CUT_FRAME_LEFT : Start processing from this frame number [default: 0].
  --cut-frame-right, -cfr CUT_FRAME_RIGHT : Stop processing at this frame number [default: None].    

Processing Options:
  --classes C            : Specify classes to detect and track (e.g., 0 1 2) [default: as per cfg file].
  --interpolate, -i      : Interpolate missing frames in tracks (not implemented yet) [default: False].

Visualization Options:
  --save, -s             : Save the visualization of extracted results to a video file [default: False].
  --show, -sh            : Show the visualization of extracted results after processing [default: False].
  --viz-mode, -vm MODE   : Choose visualization mode (0 - original, 1 - stabilized, 2 - reference frame) [default: 0].
  --hide-labels          : Do not display labels in the visualization [default: False].
  --hide-tracks          : Do not display tracking lines in the visualization [default: False].
  --hide-conf            : Do not display confidence scores in the visualization [default: False].
  --line-width, -lw LINE_WIDTH : Line width for bounding boxes in the visualization [default: 2].
  --class-filter, -cf CLASS_FILTER : Exclude specified classes (e.g., -cf 1 2) from the visualization [default: None].

Examples:
  1. Process a directory without saving/showing output videos:
     python batch_process.py path/to/videos/

  2. Process a directory and save output videos:
     python batch_process.py path/to/videos/ --save 

  3. Customize arguments for a specific video:
     python batch_process.py video.mp4 -c cfg/custom_config.yaml

  4. Save visualization in video without re-running the processing:
     python batch_process.py video.mp4 --viz-only --save

  5. Overwrite existing files with no confirmation needed:
     python batch_process.py path/to/videos/ -o -y

Notes:
  - Ensure that all paths provided are accessible and that necessary permissions are set.
  - Check that all required dependencies and modules are properly installed.
"""

import argparse
import logging
from pathlib import Path

from utils import bcolors, setup_logger
from process_video import process_video
from visualize import visualize_results

VIDEO_FORMATS = {'.mp4', '.mov'}  # Video file formats to consider (case-insensitive)
LOGGER_PREFIX = f'[{Path(__file__).name}]'

def process_input(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the input file or directory.
    """
    input_path = args.input
    if not input_path.exists():
        logger.critical(f"{LOGGER_PREFIX} File or directory {input_path} not found.")
        return

    try:
        if input_path.is_file():
            process_file(input_path, args, logger)
        elif input_path.is_dir():
            for file in sorted(input_path.rglob('*')):
                if file.is_file() and file.suffix.lower() in VIDEO_FORMATS:
                    process_file(file, args, logger)
    except KeyboardInterrupt:
        logger.warning(f"{LOGGER_PREFIX} Batch processing interrupted by user.")
    else:
        logger.info(f"{LOGGER_PREFIX} {bcolors.OKGREEN}Batch processing completed successfully.{bcolors.ENDC}")
    
def process_file(file: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the file if it is a video file and not in the results directory.
    """
    if file.parent.name == 'results':
        return

    try:
        print(f"{bcolors.OKBLUE}File: {file}...{bcolors.ENDC}")
        if not args.viz_only:
            should_process = determine_if_should_process(file, args, logger, "Processing")
            if should_process:
                logger.info(f"{LOGGER_PREFIX} Processing {file}")                
                if not args.dry_run:
                    args.source = file
                    process_video(args, logger)

        if args.save or args.show:
            should_visualize = determine_if_should_process(file, args, logger, "Visualizing")
            if should_visualize:
                #logger.info(f"{LOGGER_PREFIX} Visualizing {file}")
                if not args.dry_run:
                    args.source = file
                    #visualize_results(args, logger)
                    pass

    except Exception as e:
        logger.error(f"{LOGGER_PREFIX} Error with {file}: {e}")      

def determine_if_should_process(file: Path, args: argparse.Namespace, logger: logging.Logger, action: str) -> bool:
    """
    Determine if the video should be processed or visualized based on user arguments and existing files.
    """
    txt_exists = check_if_video_processed(file)
    vid_exists = check_if_results_visualized(file, args.viz_mode) if action == "Visualizing" else False

    if action == "Visualizing":
        if not txt_exists:
            logger.error(f"{LOGGER_PREFIX} {file} - No processing results found. Cannot visualize.")
            return False
        if vid_exists and not args.overwrite:
            logger.warning(f"{LOGGER_PREFIX} {file} - Visualization already exists and overwrite not allowed.")
            return False
        elif vid_exists and args.overwrite and not args.yes:
            user_input = input(f"{bcolors.BOLD}Visualize (overwrite?) {file} [y/n]?: {bcolors.ENDC}").lower()
            return user_input == 'y'
        return True  # Proceed to visualize if no previous video exists or overwrite allowed

    else:  # Processing logic
        if txt_exists and not args.overwrite:
            logger.warning(f"{LOGGER_PREFIX} {file} - Processing results already exist and overwrite not allowed.")
            return False
        elif txt_exists and args.overwrite and not args.yes:
            user_input = input(f"{bcolors.BOLD}Process (overwrite?) {file} [y/n]?: {bcolors.ENDC}").lower()
            return user_input == 'y'
        return True  # Proceed to process if no previous text exists or overwrite allowed

def check_if_video_processed(file: Path) -> bool:
    """
    Checks if the video processing results exist.
    """
    results_dir = file.parent / 'results'
    txt_file_path = results_dir / f"{file.stem}.txt"
    return txt_file_path.exists()

def check_if_results_visualized(file: Path, viz_mode: int) -> bool:
    """
    Checks if the visualization results exist.
    """
    results_dir = file.parent / 'results'
    vid_file_paths = [results_dir / f"{file.stem}_mode{viz_mode}.{ext}" for ext in ('mp4', 'avi')]
    return any(vid_file_path.exists() for vid_file_path in vid_file_paths)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process unprocessed videos in a batch.')

    # Required arguments
    parser.add_argument('input', type=Path, help='Path to the input directory or video file (e.g., path/to/video_dir/)')

    # Batch processing options
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically confirm prompts without waiting for user input [default: False]')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing files that have already been processed [default: False]')
    parser.add_argument('--dry-run', '-dr', action='store_true', help='Simulate the command execution without actually running any processes [default: False]')
    parser.add_argument('--viz-only', '-vo', action='store_true', help='Only visualize the results; skip any processing operations [default: False]')

    # Shared arguments for processing and visualization
    parser.add_argument('--cfg', '-c', type=str, default='cfg/default.yaml', help='Path to the main configuration file [default: cfg/default.yaml]')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help='Filename for detailed logs (e.g., info.log). Saved in the script directory [default: None]')
    parser.add_argument('--verbose', '-v', action='store_true', help='Set verbosity level [default: False]')
    parser.add_argument('--cut-frame-left', '-cfl', type=int, default=0, help='Cut video from the start at this frame number [default: 0]')
    parser.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Cut video from the end at this frame number [default: None]')

    # Processing options    
    parser.add_argument('--classes', nargs='+', type=int, help='Overwrite default classes to extract (e.g., --classes 0 1 2) [default: see cfg -> cfg_ultralytics -> classes]')
    parser.add_argument('--interpolate', '-i', action='store_true', help='Interpolate tracks between missing frames (not implemented yet) [default: False]')
    
    # Visualization options
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--save', '-s', action='store_true', help='Save the processing results to a video file [default: False]')
    group.add_argument('--show', '-sh', action='store_true', help='visualize results after processing [default: False]')
    parser.add_argument('--viz-mode', '-vm', type=int, default=0, choices=[0, 1, 2], help='Set visualization mode for the output video: 0 - original, 1 - stabilized, 2 - reference frame [default: 0]')
    parser.add_argument('--hide-labels', '-hl', action='store_true', help='Hide labels on the visualization output video [default: False]')
    parser.add_argument('--hide-tracks', '-ht', action='store_true', help='Hide tracking lines in the visualization output video [default: False]')
    parser.add_argument('--hide-conf', '-hc', action='store_true', help='Hide confidence scores on the visualization output video [default: False]')
    parser.add_argument('--line-width', '-lw', type=int, default=2, help='Set the line width for bounding boxes [default: 2].')
    parser.add_argument('--class-filter', '-cf', type=int, nargs='+', help='exclude specified classes (e.g., -cf 1 2) [default: None]')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)
    if args.save or args.show:
        logger.warning(f"{LOGGER_PREFIX} --save or --show options are not implemented yet.")
    process_input(args, logger)