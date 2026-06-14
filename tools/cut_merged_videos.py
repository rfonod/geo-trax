#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
cut_merged_videos.py - Merged Video and Flight Log Cutting Tool

Recursively searches a directory for video files whose stem contains a
configurable keyword (default: 'merged') and cuts them according to a cut
specification file with the same stem (e.g. '0_merged.mp4' → '0_merged.txt').
A DJI SRT flight log with the same stem ('0_merged.srt') is optional: when
present it is parsed and saved as a CSV alongside the cut video; when absent
only the video is cut. All companion files must reside in the same directory as
the merged video. Cut start frames are snapped to the nearest I-frame to avoid
re-encoding. Output filenames are derived from an optional location label
(nearest point in a user-supplied JSON map) and a per-label sequential counter
(e.g. A1.mp4/A1.csv, A2.mp4/A2.csv, ...).

This tool was developed for the Songdo UAV dataset accompanying:
  Fonod et al., "Advanced computer vision for extracting georeferenced vehicle
  trajectories from drone imagery," Transportation Research Part C, 2025.
  https://doi.org/10.1016/j.trc.2025.105205

Usage:
  python tools/cut_merged_videos.py <data_dir> [options]

Arguments:
  data_dir : Root directory to search recursively for merged video files.

Options:
  -h, --help                 : Show this help message and exit.
  -lm, --location-map <path> : Path to a JSON file mapping location labels to [lat, lon].
                               If omitted, all cuts are labeled 'unknown'.
  -nf, --name-filter <str>   : Keyword in the video filename used to identify merged sessions
                               (default: 'merged'). Case-insensitive substring match.
  --cleanup                  : Delete each merged video after processing (default: off).
  -d, --debug                : Verbose ffmpeg output and OpenCV frame-level verification (default: off).
  -lp, --log-path <str>      : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Cut all merged videos found under a PROCESSED directory:
   python tools/cut_merged_videos.py /path/to/PROCESSED

2. Cut with a location map and delete merged files after processing:
   python tools/cut_merged_videos.py /path/to/PROCESSED --location-map /path/to/locations.json --cleanup

3. Songdo dataset (see paper above), single session with intersection labels:
   python tools/cut_merged_videos.py /path/to/PROCESSED/2022-10-04/D1/AM1 \\
     --location-map /path/to/songdo_intersections.json

4. Custom name filter (e.g. files named 'combined_flight.mp4'):
   python tools/cut_merged_videos.py /path/to/PROCESSED --name-filter combined

Input:
- Video files (any format: .mp4, .mov, .avi, .mkv) whose stem contains the
  name filter keyword (case-insensitive, default: 'merged'), e.g. '0_merged.mp4'.
- For each matched video, the script looks for companion files with the exact
  same stem in the same directory:
    <stem>.txt : Cut specification file (required), one cut per line:
                    cut_start_frame, cut_end_frame[, rotation]
                  where:
                    cut_start_frame : 1-indexed frame where the cut starts.
                    cut_end_frame   : 1-indexed frame where the cut ends (inclusive).
                    rotation        : Optional CCW rotation in degrees (0, ±90, ±180, ±270).
                  Note: cut_start_frame may be adjusted forward to the nearest
                  I-frame to avoid re-encoding.
    <stem>.srt : DJI SRT flight log (optional). If absent, the video is still
                 cut but no CSV flight log is generated for that session.
                 Both bracket-delimited ([k: v] [k: v] ...) and comma-separated
                 ([k: v, k: v, ...]) DJI SRT variants are supported. Field-name
                 aliases across drone families (Mavic, Air, Inspire, Matrice, ...)
                 are resolved automatically; see _FIELD_ALIASES in the source.
                 Missing fields default to 0 / empty string rather than crashing.

Output (all files written to the same directory as the merged video):
- Per-cut video clip     : <label><N>.mp4
- Per-cut CSV flight log : <label><N>.csv  (frame numbers start at 0; only when SRT is present)
- Adjusted cuts file     : <stem>_adjusted.txt

  <label> is the nearest location label from --location-map, or 'unknown' if omitted.
  <N>     is a sequential counter per label within the session (e.g. two cuts at
          the same location yield A1.mp4/A1.csv and A2.mp4/A2.csv).

Directory structure:
  The script imposes no required folder layout — it works on any directory tree.
  For multi-drone/multi-day campaigns, the following structure is recommended
  (used for the Songdo dataset accompanying the paper cited above):

  PROCESSED/
  └── <ISO8601_date>/          # e.g. 2022-10-04, 2022-10-05, ...
      └── D<drone_id>/         # e.g. D1, D2, ..., D10
          └── <session>/       # e.g. AM1, AM2, PM1, PM2, ...
              ├── 0_merged.mp4
              ├── 0_merged.srt
              └── 0_merged.txt

Notes:
- The --location-map JSON must follow this schema:
    {"<label>": [<latitude>, <longitude>], ...}
  Example (Songdo intersections used in the paper):
    {
      "A": [37.39611, 126.63283],
      "B": [37.39781, 126.63473],
      "C": [37.40039, 126.63749],
      "E": [37.39365, 126.63434],
      "F": [37.38938, 126.63724],
      "G": [37.39042, 126.63825],
      "H": [37.39537, 126.64476],
      "I": [37.38597, 126.63951],
      "J": [37.38815, 126.64189],
      "K": [37.39121, 126.64513],
      "L": [37.39335, 126.64749],
      "M": [37.38229, 126.64225],
      "N": [37.38577, 126.64577],
      "O": [37.38890, 126.64892],
      "P": [37.39108, 126.65109],
      "Q": [37.37802, 126.64548],
      "R": [37.37366, 126.64867],
      "S": [37.38018, 126.65492],
      "T": [37.38421, 126.65882],
      "U": [37.38862, 126.66307]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.logging_utils import setup_logger

# Canonical CSV field name → known DJI SRT key aliases (case-insensitive).
# Covers bracket-delimited (Mavic 3, Air 3, ...) and comma-separated (older
# firmware / Inspire / Matrice) variants. First matching alias wins.
#
# Non-DJI or non-standard SRT users: inspect one subtitle block from your .srt
# file to find the actual key names your drone uses, then add them as extra
# aliases here.  No other code change is needed — _parse_srt_line and parse_srt
# will pick them up automatically.  If your drone writes a structurally
# different format (e.g. GPS as a single "GPS(lat,lon,alt)" field, or bare
# key=value pairs without brackets), you will need to extend _parse_srt_line.
_FIELD_ALIASES: dict[str, list[str]] = {
    'iso':       ['iso'],
    'shutter':   ['shutter'],
    'fnum':      ['fnum', 'f_num'],
    'ev':        ['ev'],
    'ct':        ['ct'],
    'color_md':  ['color_md', 'color_mode', 'color'],
    'focal_len': ['focal_len', 'focal_length', 'focal_mm'],
    'latitude':  ['latitude', 'lat'],
    'longitude': ['longitude', 'lon'],
    'rel_alt':   ['rel_alt', 'altitude_rel', 'height'],
    'abs_alt':   ['abs_alt', 'altitude', 'altitude_msl'],
}


def find_merged_videos(data_dir: Path, name_filter: str, logger: logging.Logger) -> list[Path]:
    videos = sorted(
        p for p in data_dir.rglob('*')
        if p.suffix.lower() in VIDEO_FORMATS and name_filter.lower() in p.stem.lower()
    )
    logger.info(f"Found {len(videos)} merged video file(s) under '{data_dir}'.")
    return videos


def find_session_files(video_path: Path, logger: logging.Logger) -> dict[str, Path] | None:
    cuts_path = video_path.with_suffix('.txt')
    srt_path = video_path.with_suffix('.srt')

    if not cuts_path.exists():
        logger.warning(f"No cuts file '{cuts_path.name}' found next to '{video_path.name}', skipping.")
        return None

    srt: Path | None = srt_path if srt_path.exists() else None
    if srt is None:
        logger.info(f"No SRT flight log '{srt_path.name}' found; video will be cut without CSV output.")

    return {
        'merged_video': video_path,
        'merged_srt': srt,
        'cuts_txt': cuts_path,
    }


def load_location_map(path: Path, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    try:
        with open(path) as f:
            raw = json.load(f)
        location_map = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
        logger.info(f"Loaded {len(location_map)} location(s) from '{path}'.")
        return location_map
    except Exception as e:
        logger.error(f"Failed to load location map from '{path}': {e}")
        sys.exit(1)


def process_session(
    filepaths: dict[str, Path],
    location_map: dict[str, tuple[float, float]],
    cleanup: bool,
    debug: bool,
    logger: logging.Logger,
) -> None:
    intersections: dict[str, int] = {}

    all_cuts = get_cuts(filepaths['cuts_txt'], logger)
    if not all_cuts:
        return

    try:
        perform_sanity_checks(all_cuts, filepaths, logger)
    except AssertionError as e:
        logger.error(str(e))
        return

    all_cuts_adjusted = get_and_save_adjusted_cuts(all_cuts, filepaths, logger, debug)

    for cut_num in all_cuts_adjusted:
        cut_video_path = cut_and_save_srt(
            filepaths, all_cuts_adjusted[cut_num], location_map, intersections, logger
        )
        cut_and_save_video(filepaths, all_cuts_adjusted[cut_num], cut_video_path, debug, logger)

    if cleanup:
        filepaths['merged_video'].unlink(missing_ok=True)
        logger.info(f"Deleted merged video '{filepaths['merged_video']}'.")


def cut_and_save_srt(
    filepaths: dict[str, Path],
    cut: tuple[int, int, int],
    location_map: dict[str, tuple[float, float]],
    intersections: dict[str, int],
    logger: logging.Logger,
) -> Path:
    cut_start, cut_end, _ = cut
    session_dir = filepaths['merged_video'].parent

    if filepaths['merged_srt'] is None:
        label = determine_intersection(0.0, 0.0, location_map, logger)
        return get_cut_filepath(session_dir, label, intersections, '.mp4')

    try:
        with open(filepaths['merged_srt'], 'r') as f:
            merged_srt = [line.rstrip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Problem reading '{filepaths['merged_srt']}': {e}")
        label = determine_intersection(0.0, 0.0, location_map, logger)
        return get_cut_filepath(session_dir, label, intersections, '.mp4')

    cut_flight_log: dict[str, list] = {
        'frame': [], 'timestamp': [], 'iso': [], 'shutter': [], 'fnum': [],
        'ev': [], 'ct': [], 'color_md': [], 'focal_len': [],
        'latitude': [], 'longitude': [], 'rel_alt': [], 'abs_alt': [],
    }
    frame_num_local = -1

    for line_num, line in enumerate(merged_srt):
        if line_num % 5 != 0:
            continue
        frame_num_global = int(line)
        if not (cut_start <= frame_num_global < cut_end):
            continue

        frame_num_local += 1
        attr = parse_srt(merged_srt[line_num + 3], merged_srt[line_num + 4])

        cut_flight_log['frame'].append(frame_num_local)
        cut_flight_log['timestamp'].append(attr['time'])
        cut_flight_log['iso'].append(attr['iso'])
        cut_flight_log['shutter'].append(attr['shutter'])
        cut_flight_log['fnum'].append(attr['fnum'])
        cut_flight_log['ev'].append(attr['ev'])
        cut_flight_log['ct'].append(attr['ct'])
        cut_flight_log['color_md'].append(attr['color_md'])
        cut_flight_log['focal_len'].append(attr['focal_len'])
        cut_flight_log['latitude'].append(attr['latitude'])
        cut_flight_log['longitude'].append(attr['longitude'])
        cut_flight_log['rel_alt'].append(attr['rel_alt'])
        cut_flight_log['abs_alt'].append(attr['abs_alt'])

    hover_lats = [x for x, m in zip(cut_flight_log['latitude'], cut_flight_log['color_md']) if m != 'dummy']
    hover_lons = [x for x, m in zip(cut_flight_log['longitude'], cut_flight_log['color_md']) if m != 'dummy']
    avg_lat = sum(hover_lats) / len(hover_lats) if hover_lats else 0.0
    avg_lon = sum(hover_lons) / len(hover_lons) if hover_lons else 0.0

    label = determine_intersection(avg_lat, avg_lon, location_map, logger)
    video_path = get_cut_filepath(session_dir, label, intersections, '.mp4')
    csv_path = video_path.with_suffix('.csv')

    try:
        pd.DataFrame(cut_flight_log).to_csv(csv_path, index=False)
        logger.info(f"Cut flight log saved to '{csv_path}'.")
    except Exception as e:
        logger.error(f"Problem saving '{csv_path}': {e}")

    return video_path


def cut_and_save_video(
    filepaths: dict[str, Path],
    cut: tuple[int, int, int],
    cut_video_path: Path,
    debug: bool,
    logger: logging.Logger,
) -> None:
    cut_start_adjusted, cut_end, rotation = cut
    # SRT frame numbering is 1-indexed; video is 0-indexed
    cut_start_adjusted -= 1
    cut_end -= 1

    cap = cv2.VideoCapture(str(filepaths['merged_video']))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_q = shlex.quote(str(filepaths['merged_video']))
    out_q = shlex.quote(str(cut_video_path))
    cmd = f'ffmpeg -y -i {video_q} -ss {cut_start_adjusted / fps} -to {cut_end / fps} -c copy {out_q}'
    if not debug:
        cmd += ' -v quiet'

    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        logger.notice(f"Cut video saved to '{cut_video_path}'.")
    else:
        logger.error(f"ffmpeg exited with code {result.returncode} for '{filepaths['merged_video']}'.")

    if debug:
        verify_cut(filepaths, cut_video_path, cut_start_adjusted, cut_end, logger)

    if int(rotation) != 0 and not debug:
        temp_path = cut_video_path.with_name(cut_video_path.stem + '_temp' + cut_video_path.suffix)
        temp_q = shlex.quote(str(temp_path))
        cut_video_path.rename(temp_path)
        subprocess.run(
            f'ffmpeg -i {temp_q} -c copy -map_metadata 0 -metadata:s:v rotate="{rotation}" {out_q} -v quiet',
            shell=True,
        )
        temp_path.unlink(missing_ok=True)


def verify_cut(
    filepaths: dict[str, Path],
    cut_video_path: Path,
    cut_start: int,
    cut_end: int,
    logger: logging.Logger,
    verify_n_frames: int = 10,
) -> None:
    total = cut_end - cut_start
    step = max(total // verify_n_frames, 1)
    merged_frames, cut_frames = [], []

    try:
        cap = cv2.VideoCapture(str(cut_video_path))
        logger.info(f"Cut video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {cap.get(cv2.CAP_PROP_FPS)} fps")
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            cut_frames.append(cap.read()[1])
    except Exception as e:
        logger.error(f"Problem reading '{cut_video_path}': {e}")
    finally:
        cap.release()

    try:
        cap = cv2.VideoCapture(str(filepaths['merged_video']))
        logger.info(f"Merged video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {cap.get(cv2.CAP_PROP_FPS)} fps")
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cut_start + i)
            merged_frames.append(cap.read()[1])
    except Exception as e:
        logger.error(f"Problem reading '{filepaths['merged_video']}': {e}")
    finally:
        cap.release()

    for i, (mf, cf) in enumerate(zip(merged_frames, cut_frames)):
        if mf is None or cf is None:
            break
        rmse = np.sqrt(np.mean((mf.astype(float) - cf.astype(float)) ** 2))
        logger.info(f"RMSE local frame {i * step} vs global frame {cut_start + i * step}: {rmse:.4f}")
        cv2.imshow(f"diff {i}", cv2.absdiff(mf, cf))
        cv2.waitKey(200)
    cv2.destroyAllWindows()


def get_cut_filepath(
    session_dir: Path,
    label: str,
    intersections: dict[str, int],
    suffix: str,
) -> Path:
    intersections[label] = intersections.get(label, 0) + 1
    return session_dir / f"{label}{intersections[label]}{suffix}"


def determine_intersection(
    avg_lat: float,
    avg_lon: float,
    location_map: dict[str, tuple[float, float]],
    logger: logging.Logger,
) -> str:
    if not location_map:
        return 'unknown'
    if avg_lat == 0.0 or avg_lon == 0.0:
        logger.warning("Could not determine location label: SRT contained only 'dummy' values.")
        return 'unknown'
    return min(
        location_map,
        key=lambda lbl: math.sqrt(
            (location_map[lbl][0] - avg_lat) ** 2 + (location_map[lbl][1] - avg_lon) ** 2
        ),
    )


def _parse_srt_line(log_line: str) -> dict[str, str]:
    """Extract all key-value pairs from a DJI SRT log line.

    Handles bracket-delimited fields ([k: v] [k: v] ...) and comma-separated
    fields within a single bracket ([k: v, k: v, ...]), as well as mixed forms
    produced by different DJI drone families and firmware versions.
    """
    fields: dict[str, str] = {}
    for content in re.findall(r'\[([^\[\]]+)\]', log_line):
        for part in re.split(r',', content):
            m = re.match(r'(\w+)\s*:\s*(.+)', part.strip())
            if m:
                fields[m.group(1).lower()] = m.group(2).strip()
    return fields


def _to_float(value: str) -> float:
    """Extract the leading numeric value from a string, ignoring unit suffixes (e.g. ' m', ' ft')."""
    m = re.search(r'[-+]?\d*\.?\d+', value)
    if not m:
        raise ValueError(f"no numeric value in '{value}'")
    return float(m.group())


def parse_srt(timestamp_line: str, log_line: str) -> dict:
    """Parse one DJI SRT subtitle block into a flat attribute dict.

    Field names are normalised via _FIELD_ALIASES so that variant spellings
    across DJI drone families all map to the same canonical CSV column names.
    Missing fields are filled with safe defaults (0 / '' ) rather than raising.
    """
    raw = _parse_srt_line(log_line)

    # Resolve each canonical field from the first matching alias
    resolved: dict[str, str] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in raw:
                resolved[canonical] = raw[alias]
                break

    def _s(key: str, default: str = '') -> str:
        return resolved.get(key, default)

    def _i(key: str, default: int = 0) -> int:
        val = resolved.get(key)
        if val is None:
            return default
        try:
            return int(_to_float(val))
        except ValueError:
            return default

    def _f(key: str, default: float = 0.0) -> float:
        val = resolved.get(key)
        if val is None:
            return default
        try:
            return _to_float(val)
        except ValueError:
            return default

    return {
        'time':      timestamp_line,
        'iso':       _i('iso'),
        'shutter':   _s('shutter', 'unknown'),
        'fnum':      _f('fnum'),
        'ev':        _f('ev'),
        'ct':        _i('ct'),
        'color_md':  _s('color_md', ''),
        'focal_len': _f('focal_len'),
        'latitude':  _f('latitude'),
        'longitude': _f('longitude'),
        'rel_alt':   _f('rel_alt'),
        'abs_alt':   _f('abs_alt'),
    }


def get_cuts(cuts_txt_path: Path, logger: logging.Logger) -> dict[int, tuple[int, int, int]]:
    try:
        with open(cuts_txt_path, 'r') as f:
            lines = [line.rstrip().split(',') for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Problem reading '{cuts_txt_path}': {e}")
        return {}

    if not lines:
        logger.error(f"The file '{cuts_txt_path}' is empty!")
        return {}

    cuts = {}
    for cut_num, line in enumerate(lines, start=1):
        cut_start = int(line[0].strip())
        cut_end = int(line[1].strip())
        try:
            rotation = int(line[2].strip())
        except (IndexError, ValueError):
            rotation = 0
        cuts[cut_num] = (cut_start, cut_end, rotation)
    return cuts


def perform_sanity_checks(
    all_cuts: dict[int, tuple[int, int, int]],
    filepaths: dict[str, Path],
    logger: logging.Logger,
) -> None:
    cap = cv2.VideoCapture(str(filepaths['merged_video']))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for cut_num, (cut_start, cut_end, rotation) in all_cuts.items():
        assert cut_start > 0 and cut_end > 0, (
            f"Cut {cut_num}: 'cut_start' and 'cut_end' must be positive in '{filepaths['cuts_txt']}'"
        )
        assert cut_start < cut_end, (
            f"Cut {cut_num}: 'cut_start' >= 'cut_end' in '{filepaths['cuts_txt']}'"
        )
        assert cut_end - 1 <= frame_count, (
            f"Cut {cut_num}: 'cut_end' exceeds total frame count ({frame_count}) in '{filepaths['cuts_txt']}'"
        )
        assert rotation in {0, 90, 180, 270, -90, -180, -270}, (
            f"Cut {cut_num}: invalid rotation {rotation} in '{filepaths['cuts_txt']}'"
        )


def get_and_save_adjusted_cuts(
    all_cuts: dict[int, tuple[int, int, int]],
    filepaths: dict[str, Path],
    logger: logging.Logger,
    debug: bool = False,
) -> dict[int, tuple[int, int, int]]:
    video_q = shlex.quote(str(filepaths['merged_video']))

    cap = cv2.VideoCapture(str(filepaths['merged_video']))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if platform.system() in ('Windows', 'Darwin'):
        key_frames_cmd = (
            f"ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags "
            f"-of csv=print_section=0 {video_q} | awk -F',' '/K/ {{print $1}}'"
        )
    else:
        key_frames_cmd = (
            f"ffmpeg -i {video_q} -vf select='eq(pict_type\\,PICT_TYPE_I)',showinfo "
            f"-vsync vfr -f null - -loglevel debug 2>&1 | "
            "awk '/pts_time/ {gsub(/.*pts_time:/, \"\"); gsub(/ .*/, \"\"); print;}'"
        )

    try:
        key_frames = subprocess.check_output(key_frames_cmd, shell=True, text=True).split()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to retrieve keyframes from '{filepaths['merged_video']}': {e}")
        sys.exit(1)
    key_frames_arr = np.array([float(kf) for kf in key_frames])

    all_cuts_adjusted: dict[int, tuple[int, int, int]] = {}
    for cut_num, (cut_start, cut_end, rotation) in all_cuts.items():
        # SRT is 1-indexed; convert to 0-indexed for keyframe comparison
        cut_start_0 = cut_start - 1
        diffs = key_frames_arr - cut_start_0 / fps
        i_closest = int(np.where(diffs >= 0, diffs, np.inf).argmin())
        closest_kf = float(key_frames[i_closest])
        cut_start_adjusted = round(closest_kf * fps) + 1  # back to 1-indexed

        all_cuts_adjusted[cut_num] = (cut_start_adjusted, cut_end, rotation)
        if debug:
            logger.info(
                f"Cut {cut_num}: start adjusted from {cut_start} to {cut_start_adjusted} "
                f"(keyframe at {closest_kf:.4f}s)."
            )

    adjusted_txt = filepaths['cuts_txt'].with_stem(filepaths['cuts_txt'].stem + '_adjusted')
    with open(adjusted_txt, 'w') as f:
        for cut in all_cuts_adjusted.values():
            f.write("{},{},{}\n".format(*cut))
    logger.info(f"Adjusted cuts saved to '{adjusted_txt}'.")

    return all_cuts_adjusted


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the cut_merged_videos tool."""
    parser = argparse.ArgumentParser(
        description="Recursively cut merged drone videos and DJI SRT flight logs into per-location clips."
    )
    parser.add_argument('data_dir', type=Path,
                        help="Root directory to search recursively for merged video files.")
    parser.add_argument('--location-map', '-lm', type=Path, default=None,
                        help="Path to a JSON file mapping location labels to [lat, lon]; if omitted, cuts are labeled 'unknown'.")
    parser.add_argument('--name-filter', '-nf', type=str, default='merged',
                        help="Keyword in the video filename used to identify merged sessions (default: 'merged').")
    parser.add_argument('--cleanup', action='store_true',
                        help="Delete each merged .mp4 after processing (default: off).")
    parser.add_argument('--debug', '-d', action='store_true',
                        help="Enable verbose ffmpeg output and OpenCV frame-level verification (default: off).")
    parser.add_argument('--log-path', '-lp', type=Path, default=None,
                        help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Reduce console verbosity to important messages only (default: show INFO-level detail).")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    location_map: dict[str, tuple[float, float]] = {}
    if args.location_map:
        location_map = load_location_map(args.location_map, logger)

    merged_videos = find_merged_videos(args.data_dir, args.name_filter, logger)
    if not merged_videos:
        logger.warning(f"No *{args.name_filter}* video files found under '{args.data_dir}'.")
        return

    for video_path in tqdm(merged_videos, desc="Processing sessions", unit="video"):
        filepaths = find_session_files(video_path, logger)
        if filepaths is None:
            continue
        logger.info(f"Processing '{video_path}'.")
        process_session(filepaths, location_map, args.cleanup, args.debug, logger)


if __name__ == "__main__":
    main()
