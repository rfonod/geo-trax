# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""File I/O helpers, path utilities, and video metadata functions."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

from geotrax.utils.constants import MACOS, WINDOWS

# Fallback output naming conventions — reproduces the historical 'results/' layout.
# Used ONLY when no config has been loaded (output_cfg=None in the path helpers, or
# .get('key', DEFAULT_OUTPUT['key']) guards against a custom config missing a key).
# In normal pipeline operation every stage loads the YAML and threads the full
# cfg['output'] dict through, so DEFAULT_OUTPUT never takes precedence over config.
DEFAULT_OUTPUT = {
    'folder': 'results',
    'tracks_postfix': '',
    'georeferenced_postfix': '',
    'stab_transform_postfix': '_vid_transf',
    'geo_transform_postfix': '_geo_transf',
    'visualization_postfix': '',
}


def get_output_dir(source: Path, output_cfg: Optional[dict] = None) -> Path:
    """Return the output directory for *source*.

    If ``output_cfg['folder']`` is an absolute path it is used as-is (shared
    across all inputs in a batch). A relative name is resolved next to the
    input video's parent directory.
    """
    cfg = output_cfg or DEFAULT_OUTPUT
    folder = Path(cfg.get('folder', DEFAULT_OUTPUT['folder']))
    return folder if folder.is_absolute() else source.parent / folder


def build_result_path(
    source: Path,
    result_type: str,
    output_cfg: Optional[dict] = None,
    viz_mode: Optional[int] = None,
    ext: Optional[str] = None,
) -> Optional[Path]:
    """Return the expected output path for *result_type* given *source*.

    result_type choices: 'video', 'processed', 'video_transformations',
    'geo_transformations', 'georeferenced', 'visualized'.
    Returns ``None`` for unknown types.
    """
    if result_type == 'video':
        return source
    cfg = output_cfg or DEFAULT_OUTPUT
    out_dir = get_output_dir(source, cfg)
    stem = source.stem
    if result_type == 'processed':
        return out_dir / f"{stem}{cfg.get('tracks_postfix', DEFAULT_OUTPUT['tracks_postfix'])}.txt"
    if result_type == 'video_transformations':
        return out_dir / f"{stem}{cfg.get('stab_transform_postfix', DEFAULT_OUTPUT['stab_transform_postfix'])}.txt"
    if result_type == 'geo_transformations':
        return out_dir / f"{stem}{cfg.get('geo_transform_postfix', DEFAULT_OUTPUT['geo_transform_postfix'])}.txt"
    if result_type == 'georeferenced':
        return out_dir / f"{stem}{cfg.get('georeferenced_postfix', DEFAULT_OUTPUT['georeferenced_postfix'])}.csv"
    if result_type == 'visualized':
        return out_dir / f"{stem}{cfg.get('visualization_postfix', DEFAULT_OUTPUT['visualization_postfix'])}_mode_{viz_mode}.{ext}"
    return None


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
    frame_w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.release()
    return frame_w, frame_h


def check_if_results_exist(
    file: Path,
    result_type: str,
    viz_mode: Optional[int] = None,
    ext: Optional[str] = None,
    output_cfg: Optional[dict] = None,
) -> Tuple[bool, Optional[Path]]:
    """Check if the results already exist for *file*.

    *output_cfg* is the ``cfg -> output`` dict (or ``None`` to use the
    historical defaults). Existing callers that pass ``viz_mode`` / ``ext``
    positionally are unaffected because ``output_cfg`` is keyword-only.
    """
    result_path = build_result_path(file, result_type, output_cfg, viz_mode, ext)
    return (result_path.exists() if result_path else False), result_path
