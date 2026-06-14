#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
random_frame_extraction.py - Video Frame Extraction Tool

Extracts random frames from video files with optional SRT metadata filtering.
Optimized for DJI drone footage with timestamp-based SRT alignment but works with any video format.

Key Features:
- Multi-field SRT filtering with complex criteria (altitude, ISO, GPS, etc.)
- Precise timestamp mapping between video frames and SRT metadata
- Supports DJI format: [rel_alt: 155.0] [abs_alt: 500.0] [iso: 100]
- Universal video support (.mp4, .avi, .mov, .mkv)
- Reproducible results with configurable random seeds

Usage:
  python tools/random_frame_extraction.py <input_dir> <output_dir> [options]

Arguments:
  input_dir  : Input directory containing video files.
  output_dir : Output directory for the extracted frames.

Options:
  -h, --help                       : Show this help message and exit.
  --total-frames <int>             : Number of frames to extract (default: 100).
  --srt-filter <str>               : Multi-field filter "field:min:max" (repeatable).
  --srt-field <str>                : Legacy single-field mode (default: rel_alt).
  --srt-min-value, --srt-max-value : Legacy min/max values (default: 130, 160).
  --output-format {png,jpg,jpeg}   : Image format (default: png).
  --prefix <str>                   : Filename prefix (default: frame).
  -r, --recursive                  : Process subdirectories.
  -s, --seed <int>                 : Random seed (default: 0).
  -lp, --log-path <str>            : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                      : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Basic extraction (50 random frames):
   python tools/random_frame_extraction.py videos/ output/ --total-frames 50

2. Multi-field filtering (altitude 130-160m AND ISO ≤400):
   python tools/random_frame_extraction.py videos/ output/ --srt-filter "rel_alt:130:160" --srt-filter "iso::400"

3. Recursive processing with custom seed:
   python tools/random_frame_extraction.py videos/ output/ --recursive --seed 42

Output: {prefix}_{parent}_{video}_{frame_idx}.{format}
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pysrt

from geotrax.utils.logging_utils import setup_logger

# Constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
DEFAULT_TOTAL_FRAMES = 100
DEFAULT_OUTPUT_FORMAT = 'png'
DEFAULT_PREFIX = 'frame'
DEFAULT_SRT_FIELD = 'rel_alt'
DEFAULT_SRT_MIN_VALUE = 130
DEFAULT_SRT_MAX_VALUE = 160


def collect_all_frames(
    input_dir: Path,
    logger: logging.Logger,
    srt_filters: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    recursive: bool = False,
) -> List[Tuple[Path, int]]:
    """Collect all valid frames from videos, optionally filtering by multiple SRT criteria."""
    valid_frames = []

    def process_directory(directory: Path):
        video_files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_FORMATS]

        for video_path in video_files:
            frame_count = get_video_frame_count(video_path)
            if frame_count == 0:
                continue

            # Get video FPS for proper timestamp mapping
            fps = get_video_fps(video_path)

            # Load SRT file if filtering is enabled
            srt_data = None
            srt_field_missing_warned = set()  # Track which fields have been warned about
            if srt_filters:
                srt_path = video_path.with_suffix('.SRT')
                if not srt_path.exists():
                    srt_path = video_path.with_suffix('.srt')
                if srt_path.exists():
                    try:
                        srt_data = pysrt.open(str(srt_path))
                    except Exception:
                        pass

            # Add frames to the list (with optional SRT filtering)
            for frame_idx in range(frame_count):
                should_keep_frame = True

                # Apply SRT filters using proper timestamp mapping
                if srt_data and srt_filters:
                    subtitle_entry = get_srt_entry_for_frame(srt_data, frame_idx, fps)

                    if subtitle_entry:
                        # Check all filter criteria - frame must pass ALL filters
                        for field_name, (min_val, max_val) in srt_filters.items():
                            field_value = parse_srt_field(subtitle_entry.text, field_name)

                            if field_value is not None:
                                # Check min/max constraints for this field
                                if min_val is not None and field_value < min_val:
                                    should_keep_frame = False
                                    break
                                elif max_val is not None and field_value > max_val:
                                    should_keep_frame = False
                                    break
                            else:
                                # Field not found in this SRT entry - warn once per field per video but keep frame
                                if field_name not in srt_field_missing_warned:
                                    logger.warning(f"Field '{field_name}' not found in SRT file for {video_path.name}")
                                    srt_field_missing_warned.add(field_name)

                if should_keep_frame:
                    valid_frames.append((video_path, frame_idx))

    if recursive:
        # Process root directory first
        process_directory(input_dir)
        # Then process all subdirectories recursively
        for subdir in input_dir.rglob('*'):
            if subdir.is_dir():
                process_directory(subdir)
    else:
        process_directory(input_dir)

    return valid_frames


def extract_frame(video_path: Path, frame_number: int) -> Optional[np.ndarray]:
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_random_frames(
    input_dir: Path,
    output_dir: Path,
    total_frames: int,
    logger: logging.Logger,
    srt_filters: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    output_format: str = 'png',
    prefix: str = 'frame',
    recursive: bool = False,
    seed: int = 0,
) -> None:
    """Extract random frames from all videos in the directory."""
    # Set random seed for reproducible results
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

    logger.info(f"Collecting frames from: {input_dir}")

    # Collect all valid frames
    all_frames = collect_all_frames(input_dir, logger, srt_filters, recursive)

    if not all_frames:
        logger.warning("No valid frames found")
        return

    logger.info(f"Found {len(all_frames)} valid frames")

    # Randomly select frames
    num_to_extract = min(total_frames, len(all_frames))
    selected_indices = np.random.choice(len(all_frames), num_to_extract, replace=False)

    extracted_count = 0

    for idx in selected_indices:
        video_path, frame_idx = all_frames[idx]

        # Extract the frame
        frame = extract_frame(video_path, frame_idx)
        if frame is not None:
            # Create output filename
            video_name = video_path.stem
            parent_name = video_path.parent.name
            output_filename = f"{prefix}_{parent_name}_{video_name}_{frame_idx}.{output_format}"
            output_path = output_dir / output_filename

            # Save the frame
            if cv2.imwrite(str(output_path), frame):
                extracted_count += 1

    logger.notice(f"Successfully extracted {extracted_count} frames")


def get_video_frame_count(video_path: Path) -> int:
    """Get the total number of frames in a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_video_fps(video_path: Path) -> float:
    """Get the frame rate (FPS) of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0  # Default fallback FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def get_srt_entry_for_frame(srt_data, frame_idx: int, fps: float):
    """Get the SRT entry that corresponds to a specific video frame using timestamp mapping."""
    if not srt_data:
        return None

    # Calculate frame timestamp in seconds
    frame_time_seconds = frame_idx / fps

    # Convert to pysrt time format (milliseconds)
    frame_time_ms = int(frame_time_seconds * 1000)

    # Use pysrt's built-in method to find subtitle at specific time
    try:
        # Find the subtitle that contains this timestamp
        for subtitle in srt_data:
            if subtitle.start.ordinal <= frame_time_ms <= subtitle.end.ordinal:
                return subtitle
        return None
    except (AttributeError, TypeError):
        # Fallback: use closest subtitle by index if timestamp method fails
        max_idx = len(srt_data) - 1
        subtitle_idx = min(frame_idx, max_idx)
        return srt_data[subtitle_idx] if subtitle_idx >= 0 else None


def parse_filter_args(filter_strings: List[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Parse filter arguments in format 'field:min:max' into a dictionary.

    Args:
        filter_strings: List of filter strings in format 'field:min:max' or 'field:min:' or 'field::max'

    Returns:
        Dictionary mapping field names to (min_value, max_value) tuples

    Examples:
        ['rel_alt:130:160', 'iso::800'] -> {'rel_alt': (130.0, 160.0), 'iso': (None, 800.0)}
    """
    filters = {}

    for filter_str in filter_strings:
        parts = filter_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid filter format '{filter_str}'. Expected 'field:min:max'")

        field, min_str, max_str = parts

        min_val = float(min_str) if min_str.strip() else None
        max_val = float(max_str) if max_str.strip() else None

        if min_val is None and max_val is None:
            raise ValueError(f"At least one of min or max value must be specified for field '{field}'")

        filters[field] = (min_val, max_val)

    return filters


def parse_srt_field(srt_text: str, field_name: str) -> Optional[float]:
    """Parse a specific field from DJI SRT text format."""
    try:
        # DJI SRT format uses square brackets: [field_name: value]
        # Example: [rel_alt: 155.0] [abs_alt: 500.0]
        pattern = rf'\[{field_name}\s*:\s*([-+]?[0-9]*\.?[0-9]+)\]'
        match = re.search(pattern, srt_text, re.IGNORECASE)

        if match:
            return float(match.group(1))

        # Fallback to legacy format without brackets for backward compatibility
        pattern_legacy = rf'{field_name}\s*:\s*([-+]?[0-9]*\.?[0-9]+)'
        match_legacy = re.search(pattern_legacy, srt_text, re.IGNORECASE)

        if match_legacy:
            return float(match_legacy.group(1))

        return None

    except (ValueError, AttributeError):
        return None


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract random frames from videos")
    parser.add_argument('input_dir', type=Path, help='Input directory containing video files')
    parser.add_argument('output_dir', type=Path, help='Output directory for extracted frames')
    parser.add_argument('--total-frames', type=int, default=DEFAULT_TOTAL_FRAMES, help='Total number of frames to extract')

    # Multi-field filtering support
    parser.add_argument('--srt-filter', action='append', dest='srt_filters',
                       help='SRT field filter in format "field:min:max" (e.g., "rel_alt:130:160"). Can be specified multiple times.')

    # Legacy single-field support for backward compatibility
    parser.add_argument('--srt-field', type=str, default=DEFAULT_SRT_FIELD, help='SRT field to filter frames (legacy - use --srt-filter instead)')
    parser.add_argument('--srt-min-value', type=float, default=DEFAULT_SRT_MIN_VALUE, help='Minimum value for SRT field filter (legacy)')
    parser.add_argument('--srt-max-value', type=float, default=DEFAULT_SRT_MAX_VALUE, help='Maximum value for SRT field filter (legacy)')

    parser.add_argument('--output-format', choices=['png', 'jpg', 'jpeg'], default=DEFAULT_OUTPUT_FORMAT, help='Output image format')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help='Prefix for output filenames')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed for reproducible results')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Process SRT filters
    srt_filters = None

    # Handle new multi-field format
    if args.srt_filters:
        try:
            srt_filters = parse_filter_args(args.srt_filters)
            logger.info(f"Using multi-field SRT filters: {srt_filters}")
        except ValueError as e:
            logger.error(f"Error parsing SRT filters: {e}")
            sys.exit(1)

    # Handle legacy single-field format for backward compatibility
    elif args.srt_field and (args.srt_min_value is not None or args.srt_max_value is not None):
        srt_filters = {args.srt_field: (args.srt_min_value, args.srt_max_value)}
        logger.info(f"Using legacy single-field SRT filter: {srt_filters}")

    # Extract frames
    extract_random_frames(
        args.input_dir,
        args.output_dir,
        args.total_frames,
        logger,
        srt_filters,
        args.output_format,
        args.prefix,
        args.recursive,
        args.seed,
    )

    logger.notice("Frame extraction completed!")


if __name__ == "__main__":
    main()
