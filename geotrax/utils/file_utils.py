# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""File I/O helpers, path utilities, and video metadata functions."""

import argparse
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

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
    'metadata_postfix': '',
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
    'geo_transformations', 'georeferenced', 'visualized', 'metadata'.
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
    if result_type == 'metadata':
        return out_dir / f"{stem}{cfg.get('metadata_postfix', DEFAULT_OUTPUT['metadata_postfix'])}.yaml"
    return None


def read_recorded_stab_anchor(source: Path, output_cfg: Optional[dict] = None) -> Optional[int]:
    """Return the stabilization anchor frame recorded in the run-metadata YAML of *source*, or None.

    The stabilized track coordinates are expressed relative to the ``cut_frame_left`` that ``extract``
    ran with, which it records under ``processing`` in the metadata YAML. Stages that run separately
    (``georeference``, ``visualize``) must use that frame rather than their own ``cut_frame_left``.
    Returns None when the file is missing or unreadable (e.g. results from before v1.4.0) or holds no
    integer anchor, leaving the fallback to the caller.
    """
    metadata_path = build_result_path(source, 'metadata', output_cfg)
    try:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    processing = metadata.get('processing') if isinstance(metadata, dict) else None
    anchor = processing.get('cut_frame_left') if isinstance(processing, dict) else None
    if not isinstance(anchor, int) or isinstance(anchor, bool):
        return None
    return anchor


@contextmanager
def atomic_output(path: Path, keep_suffix: bool = False) -> Iterator[Path]:
    """Yield a temporary path next to *path* that replaces *path* only if the block completes.

    Result files are written in place otherwise, and ``batch`` treats any existing result as complete,
    so an interrupted or failed write (Ctrl+C, a SLURM kill, a full disk) would leave a truncated file
    that later runs skip and downstream stages read as valid. With this, *path* holds either its
    previous content or the complete new one, and the temporary file is always removed.

    The temporary name ends in '.tmp' by default, so that no '*.csv'/'*.txt' result scan can pick it
    up. ``keep_suffix=True`` keeps the real suffix last ('.name.tmp.gpkg') for writers such as GDAL
    that check the extension of the file they write; use it only for formats no result scan globs.
    """
    tmp_path = path.with_name(f'.{path.stem}.tmp{path.suffix}' if keep_suffix else f'.{path.name}.tmp')
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


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


def get_keyframe_times(video_path: Path) -> np.ndarray:
    """Return the presentation times (seconds) of the video keyframes, in stream order.

    Reads ffprobe's per-packet ``pts_time,flags`` output and keeps the packets flagged ``K``. ffprobe
    runs with an argument list, without a shell, so any path works on every platform (a shell pipe
    through awk would need POSIX quoting, which cmd.exe does not honor). Raises
    ``subprocess.CalledProcessError`` if ffprobe fails.
    """
    output = subprocess.check_output(
        ['ffprobe', '-loglevel', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time,flags',
         '-of', 'csv=print_section=0', str(video_path)],
        text=True,
    )
    times = []
    for line in output.splitlines():
        fields = line.strip().split(',')
        if len(fields) >= 2 and 'K' in fields[1] and fields[0] not in ('', 'N/A'):
            times.append(float(fields[0]))
    return np.array(times)
