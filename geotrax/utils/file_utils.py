# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""File I/O helpers, path utilities, and video metadata functions."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import cv2

from geotrax.utils.constants import MACOS, WINDOWS


def detect_delimiter(filepath: Path, lines_to_check: int = 5) -> str:
    """Detect the delimiter of a CSV file by reading a few lines."""
    delimiters = {',': 0, ' ': 0, '\t': 0}
    with open(filepath, 'r') as file:
        for _ in range(lines_to_check):
            line = file.readline()
            if not line:
                break
            delimiters[','] += line.count(',')
            delimiters[' '] += line.count(' ')
            delimiters['\t'] += line.count('\t')
    return max(delimiters, key=lambda k: delimiters[k])


def convert_to_serializable(obj):
    """Convert an object to a serializable format."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, argparse.Namespace):
        return {k: convert_to_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def determine_location_id(source: Path, logger: logging.Logger = None) -> str:
    """
    Extract the location ID from the source filename. Location ID is the first sequence
    of alphabetic characters in the filename. Symbols '_' and '-' can act as separators.
    Examples:
    'A1.mp4' -> 'A'
    '2025-01-01_A_PM1.mp4' -> 'A'
    'A1_AV.csv' -> 'A'
    """
    location_id = []
    for char in source.stem:
        if char.isalpha():
            location_id.append(char)
        elif len(location_id) and (char in '_-' or char.isdigit()):
            break
    location_id = ''.join(location_id)

    if not location_id:
        message = f"Error: Failed to extract location ID from the source filename {source}."
        if logger:
            logger.error(message)
        else:
            print(message)
        sys.exit(1)

    if logger:
        logger.info(f"Detected location ID: '{location_id}' from the source filename {source.name}.")

    return location_id


def get_ortho_folder(source: Path, ortho_folder: Union[Path, None], logger: logging.Logger, critical: bool = True) -> Path:
    """Get the orthophoto folder from the provided path or use the default folder structure."""
    if ortho_folder is None:
        ortho_folder = source.parent

        while ortho_folder != ortho_folder.parent:
            if ortho_folder.name in ['PROCESSED', 'DATASET']:
                break
            ortho_folder = ortho_folder.parent

        if ortho_folder.name not in ['PROCESSED', 'DATASET']:
            if critical:
                logger.critical(
                    f"Failed to find the orthophoto folder for source '{source}'. "
                    f"Please either provide a custom path using the --ortho-folder argument, "
                    f"skip georeferencing with the --no-geo argument, "
                    f"or ensure that the default folder structure is in place."
                )
                sys.exit(1)
            else:
                logger.info(
                    f"Failed to find the orthophoto folder for source '{source}'. "
                    f"Please either provide a custom path using the --ortho-folder argument, "
                    f"skip georeferencing with the --no-geo argument, "
                    f"or ensure that the default folder structure is in place."
                )
                return None

        ortho_folder = ortho_folder.parent / 'ORTHOPHOTOS'

    if not ortho_folder.exists():
        if critical:
            logger.critical(f"Orthophoto folder '{ortho_folder}' not found. Use the '--ortho-folder' argument to provide a custom path or ensure the default folder structure.")
            sys.exit(1)
        else:
            logger.info(f"Orthophoto folder '{ortho_folder}' not found. Use the '--ortho-folder' argument to provide a custom path or ensure the default folder structure.")
            return None
    else:
        logger.info(f"Using orthophoto folder: '{ortho_folder}'.")

    return ortho_folder


def determine_suffix_and_fourcc() -> Tuple[str, str]:
    """Determine the suffix and fourcc for the output video format."""
    suffix = 'mp4' if MACOS else 'avi' if WINDOWS else 'mp4'
    fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'
    return suffix, fourcc


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Get the width and height of the video."""
    reader = cv2.VideoCapture(str(video_path))
    w_I = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_I = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.release()
    return w_I, h_I


def check_if_results_exist(file: Path, result_type: str, viz_mode: int = None, ext: str = None) -> Tuple[bool, Path]:
    """Check if the results already exist for the video file."""
    results_dir = file.parent / 'results'
    result_path = {
        "video": file,
        "processed": results_dir / f"{file.stem}.txt",
        'video_transformations': results_dir / f"{file.stem}_vid_transf.txt",
        'geo_transformations': results_dir / f"{file.stem}_geo_transf.txt",
        "georeferenced": results_dir / f"{file.stem}.csv",
        "visualized": results_dir / f"{file.stem}_mode_{viz_mode}.{ext}"
    }.get(result_type)
    return result_path.exists() if result_path else False, result_path
