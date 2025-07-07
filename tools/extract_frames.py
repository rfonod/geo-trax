#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
extract_frames.py - Video Frame Extraction Tool

This script extracts frames from video files and saves them as JPEG images. It supports both
fixed-step and random frame extraction methods, with configurable frame count and sampling parameters.
The script can process individual video files or entire directories of videos.

The script automatically skips frames at the beginning and end of videos (configurable) and excludes
certain directories from processing. Output files are named with a structured convention that includes
the source path and frame number.

Usage:
  python extract_frames.py <input> <output> [options]

Arguments:
  input  : Path to the input directory or specific video file.
  output : Path to the output directory for the extracted frames.

Options:
  -h, --help             : Show this help message and exit.
  -n, --num-frames <int> : Total number of frames to extract from each video (default: 10).
  -s, --step <int>       : Step size in frames for fixed-step frame extraction (default: 1000).
  -r, --random           : Randomly extract frames instead of fixed-step extraction (default: False).
  -d, --dry-run          : Dry run mode, do not save frames (default: False).

Extraction Methods (mutually exclusive):
  --step    : Extract frames at regular intervals (every N frames)
  --random  : Extract frames randomly from the valid frame range

Examples:
1. Extract 10 frames at regular intervals from a single video:
   python extract_frames.py video.mp4 output_dir/

2. Extract 20 frames randomly from all videos in a directory:
   python extract_frames.py video_dir/ output_dir/ --num-frames 20 --random

3. Extract frames every 500 frames without saving (dry run):
   python extract_frames.py video.mp4 output_dir/ --step 500 --dry-run

4. Extract 50 random frames from videos in a directory:
   python extract_frames.py /path/to/videos/ /path/to/output/ -n 50 -r

Input:
- Video files in supported formats: .mp4, .mov, .avi
- Single video file or directory containing video files
- Automatically excludes 'track' and 'results' directories

Output:
- JPEG images saved with structured naming: path_components_framenum.jpg
- Progress tracking with tqdm progress bars
- Console output showing extraction progress

Notes:
- Ignores first 100 and last 100 frames by default (configurable via constants)
- Skips hidden files and excluded directories during directory processing
- Frame numbering is zero-padded to 6 digits for consistent sorting
- Output directory is created automatically if it doesn't exist
- Mutually exclusive options: --step and --random cannot be used together
"""

import argparse
import random
import re
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_FORMATS = {'.mp4', '.mov', '.avi'}
IGNORE_START_FRAMES = 100 # 100 for HBB, 0 for OBB
IGNORE_END_FRAMES = 100 # 100 for HBB, 0 for OBB
DELIMITERS = r'[\s\[\],]'
DIR_SKIP = 3
EXCLUDED_DIRS = {'track', 'results'}

def process_input(args: argparse.Namespace) -> None:
    video_files = get_video_files(args.input)
    if not video_files:
        print(f"No valid video files found at '{args.input}'.")
        return

    for video_file in video_files:
        extract_frames(video_file, args.output, args.num_frames, args.step, args.random, args.dry_run)

def get_video_files(input_path: Path) -> list:
    """Get a list of video files in the specified directory or the file itself if it is a video."""
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_FORMATS:
        return [input_path]
    elif input_path.is_dir():
        return [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_FORMATS and not f.name.startswith('.') and f.parts[-2] not in EXCLUDED_DIRS]
    else:
        return []

def extract_frames(video_file: Path, output_dir: Path, num_frames: int, step: int, randomize: bool, dry_run: bool):
    """Extract frames from the specified video file."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_frames = range(IGNORE_START_FRAMES, total_frames - IGNORE_END_FRAMES)

        if randomize:
            frames_to_extract = random.sample(valid_frames, min(num_frames, len(valid_frames)))
        else:
            frames_to_extract = valid_frames[::step][:num_frames]

        print(f"Processing video '{video_file.stem}' with {total_frames} frames.")

        for frame_num in tqdm(sorted(frames_to_extract), desc=f"Extracting frames from '{video_file.stem}'"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            file_name = '_'.join(str(video_file).split('/')[DIR_SKIP:])
            file_name = f"{'_'.join(re.split(DELIMITERS, file_name))}_{frame_num:06d}.jpg"
            frame_path = output_dir / file_name
            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame_path), frame)
            print(f"Saved frame to '{frame_path}'.")

    finally:
        cap.release()


def get_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the frame extraction tool."""
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument("input", type=Path, help="Path to the input directory or specific video file.")
    parser.add_argument("output", type=Path, help="Path to the output directory for the extracted frames.")
    parser.add_argument("--num-frames", "-n", type=int, default=10, help="Total number of frames to extract.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", "-s", type=int, default=1000, help="Step size in frames for fixed-step frame extraction.")
    group.add_argument("--random", "-r", action="store_true", help="Randomly extract frames.")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run, do not save frames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    process_input(args)


