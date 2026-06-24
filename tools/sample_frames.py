#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
sample_frames.py - Random Frame Sampler for Drone Videos

Randomly samples frames from drone video files for use in object detection annotation
workflows. While the tool works on any video file, its primary use case is sampling
from merged videos produced by tools/merge_videos_and_logs.py (default --name-filter
'merged'). The same approach can be applied to cut (per-location) videos produced by
tools/cut_merged_videos_and_logs.py, but in that case it is assumed that the merged
videos have already been removed using the --cleanup flag of that tool; otherwise the
name filter would inadvertently match both merged and cut video files simultaneously.

Sampled frames can feed into annotation workflows ranging from fully manual labelling,
to semi-automatic pipelines (e.g. tools/annotate_frames.py, which runs a YOLO detector
to produce pre-label files that a human then verifies and corrects), to fully automatic
evaluation pipelines.

Two sampling modes are supported:
- Global (default): selects frames uniformly at random across the combined pool of all
  discovered video files, giving proportionally more frames from longer videos.
- Balanced (--balanced): distributes the total quota as evenly as possible across all
  discovered video files, so shorter videos are not under-represented.

Metadata pre-filtering is supported via --srt-filter (for merged videos with companion
.srt/.SRT files) and --csv-filter (for cut videos with companion .csv files produced
by tools/cut_merged_videos_and_logs.py). Both use the same 'field:min:max' syntax and
field names (rel_alt, abs_alt, iso, latitude, longitude, ...).

This tool was developed for the Songdo Traffic dataset accompanying:
  Fonod et al., "Advanced computer vision for extracting georeferenced vehicle
  trajectories from drone imagery," Transportation Research Part C, 2025.
  https://doi.org/10.1016/j.trc.2025.105205
In the Songdo experiment, each drone hovered over multiple intersections per session.
After merging, each session directory contains one 0_merged.mp4 and a companion
0_merged.srt. The balanced mode ensures all video files contribute equally to the
annotation pool regardless of their duration.

Usage:
  python tools/sample_frames.py <data_dir> <output_dir> [options]

Arguments:
  data_dir   : Root directory to search recursively for video files, or a single
               video file path.
  output_dir : Directory where extracted frame images are saved.

Options:
  -h, --help                   : Show this help message and exit.
  -n, --num-frames <int>       : Total number of frames to extract (default: 100).
  -nf, --name-filter <str>     : Keyword that must appear in the video filename stem
                                 (default: 'merged'). Case-insensitive. Pass an empty
                                 string to discover all video files. Ignored when
                                 <data_dir> is a single file.
  -b, --balanced               : Distribute the frame quota evenly across all discovered
                                 video files; default is global random sampling across
                                 the entire frame pool.
  -of, --output-format <str>   : Image format for extracted frames: 'png', 'jpg', or
                                 'jpeg' (default: 'png').
  -s, --seed <int>             : Random seed for reproducibility (default: 42).
  -ss, --skip-start <int>      : Skip the first N frames of each video, e.g. to exclude
                                 takeoff (default: 0).
  -se, --skip-end <int>        : Skip the last N frames of each video, e.g. to exclude
                                 landing (default: 0).
  --srt-filter <str>           : Pre-filter frames using companion .srt/.SRT metadata
                                 (format: 'field:min:max'; repeatable; empty bound means
                                 unbounded). Mutually exclusive with --csv-filter.
  --csv-filter <str>           : Pre-filter frames using companion .csv metadata
                                 (format: 'field:min:max'; repeatable; same syntax as
                                 --srt-filter). Mutually exclusive with --srt-filter.
  -dr, --dry-run               : Log all planned extractions without writing any files
                                 (default: off).
  -lp, --log-path <str>        : Where to write logs: a directory or a full file path;
                                 defaults to a platform-specific log directory.
  -q, --quiet                  : Reduce console verbosity to important messages only
                                 (default: show INFO-level detail).

Examples:
1. Extract 100 frames globally at random from all merged videos:
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames

2. Extract 200 frames balanced across merged video files:
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames -n 200 --balanced

3. Sample from a single video file:
   python tools/sample_frames.py /path/to/PROCESSED/2022-10-04/D1/AM1/0_merged.mp4 \\
     /path/to/frames -n 50

4. Skip takeoff/landing frames, balanced across all videos:
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames \\
     --skip-start 300 --skip-end 300 --balanced

5. Filter merged videos to hover altitude only (rel_alt 130–160 m):
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames \\
     --srt-filter rel_alt:130:160 -n 200

6. Multi-field SRT filter (altitude AND ISO) with balanced sampling:
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames \\
     --balanced --srt-filter rel_alt:130:160 --srt-filter iso::400 -n 200

7. Sample from cut videos (after --cleanup removed merged files) using CSV filter:
   python tools/sample_frames.py /path/to/PROCESSED /path/to/frames \\
     --name-filter '' --csv-filter rel_alt:130:160 --balanced -n 200

Input:
- Video files (any of: .mp4, .mov, .avi, .mkv) whose stem contains the name filter
  keyword (default: 'merged') anywhere in the filename, located recursively under
  <data_dir>, or a single video file given directly as <data_dir>.
- For metadata filtering, companion files are expected in the same directory:
    --srt-filter: <video_stem>.srt or <video_stem>.SRT (DJI flight log)
    --csv-filter: <video_stem>.csv (produced by tools/cut_merged_videos_and_logs.py)

Output:
- One image file per extracted frame, written to <output_dir> (created if absent).
- Filename encoding: frame_<relative_path>_<frame_idx:06d>.<format>
  where <relative_path> is the path from <data_dir> to the video file (without
  extension) with path separators and dots replaced by underscores.
  Example: frame_2022-10-04_D1_AM1_0_merged_001234.png

Notes:
- Frame counts are read from the video container headers via OpenCV
  (cv2.CAP_PROP_FRAME_COUNT). For variable-frame-rate or damaged files the count
  may be inaccurate; frames that cannot be decoded are skipped with a warning.
- Global sampling uses numpy's default_rng for reproducible pseudorandom draws
  without replacement across the entire multi-video frame pool.
- Balanced mode distributes the quota across video files iteratively: videos are sorted
  ascending by available frame count so under-capacity videos resolve first and their
  deficit naturally redistributes to larger videos.
- --srt-filter reads the companion .srt/.SRT file using pysrt and maps passing subtitle
  timestamps to frame index ranges. Each filter criterion must be satisfied simultaneously.
  If a field is absent from a subtitle entry it is treated as passing (not rejected).
  Videos with no SRT companion are excluded from the pool when --srt-filter is active.
- --csv-filter reads the companion .csv file (columns: frame, rel_alt, abs_alt, iso,
  latitude, longitude, shutter, fnum, ...) and returns the frame indices of rows that
  satisfy all criteria. The 'frame' column is 0-indexed. Videos with no CSV companion
  are excluded from the pool when --csv-filter is active.
- Sampling from merged videos (default) includes transitional frames captured while
  the drone moves between locations. Alternatively, if merged files were deleted via
  --cleanup in tools/cut_merged_videos_and_logs.py, the same approach can be applied
  to the cut (per-location) videos instead using --name-filter ''. Merged-video
  sampling is generally preferred for object detection: transitional frames add scene
  diversity (varied altitudes, angles, backgrounds) that improves detector robustness.
- Extracted frames are saved with default cv2.imwrite quality settings (lossless
  for PNG; JPEG quality 95 for jpg/jpeg).

Directory structure (Songdo dataset — non-prescriptive example):
  PROCESSED/
  └── <ISO8601_date>/         # e.g. 2022-10-04
      └── D<drone_id>/        # e.g. D1
          └── <session>/      # e.g. AM1
              ├── 0_merged.mp4       ← merged video (primary input)
              ├── 0_merged.srt       ← companion for --srt-filter
              └── 0_merged.txt

  frames/                            ← output_dir
  └── frame_2022-10-04_D1_AM1_0_merged_001234.png
"""

from __future__ import annotations

import argparse
import bisect
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pysrt
from tqdm import tqdm

from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.logging_utils import setup_logger

_OUTPUT_FORMATS = {'png', 'jpg', 'jpeg'}


def find_videos(data_dir: Path, name_filter: str, logger: logging.Logger) -> list[Path]:
    """Recursively find all video files under data_dir matching name_filter."""
    videos = sorted(
        p for p in data_dir.rglob('*')
        if p.suffix.lower() in VIDEO_FORMATS and name_filter.lower() in p.stem.lower()
    )
    label = f"'*{name_filter}*' " if name_filter else ''
    logger.info(f"Found {len(videos)} {label}video file(s) under '{data_dir}'.")
    return videos


def get_frame_count(video_path: Path) -> int:
    """Return the frame count from the video container header, or 0 on failure."""
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def get_fps(video_path: Path) -> float:
    """Return the frame rate from the video container header; fallback 30.0."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def make_output_name(video_path: Path, data_dir: Path, frame_idx: int, ext: str) -> str:
    """Build a collision-free output filename encoding the video's relative path."""
    rel = str(video_path.relative_to(data_dir).with_suffix(''))
    safe = rel.replace('/', '_').replace('\\', '_').replace('.', '_')
    return f"frame_{safe}_{frame_idx:06d}.{ext}"


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video by 0-based index; returns None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def save_frame(frame: np.ndarray, output_path: Path, logger: logging.Logger) -> bool:
    """Write a frame to disk; returns True on success."""
    try:
        cv2.imwrite(str(output_path), frame)
        return True
    except Exception as e:
        logger.error(f"Failed to save '{output_path}': {e}")
        return False


def _build_cumulative(videos: list[Path], frame_counts: dict[Path, int]) -> list[int]:
    """Return cumulative frame count list aligned with videos."""
    cumulative: list[int] = []
    total = 0
    for v in videos:
        total += frame_counts[v]
        cumulative.append(total)
    return cumulative


def _resolve_index(idx: int, videos: list[Path], cumulative: list[int]) -> tuple[Path, int]:
    """Map a flat pool index to (video_path, local_frame_idx)."""
    lo = bisect.bisect_right(cumulative, idx)
    video = videos[lo]
    local_frame = idx - (cumulative[lo - 1] if lo > 0 else 0)
    return video, local_frame


def _parse_meta_filters(
    filter_strings: list[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Parse 'field:min:max' filter strings into a dict; raises ValueError on bad input."""
    filters: dict[str, tuple[float | None, float | None]] = {}
    for s in filter_strings:
        parts = s.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid filter format '{s}'. Expected 'field:min:max'.")
        field, min_str, max_str = parts
        field = field.strip()
        min_val = float(min_str) if min_str.strip() else None
        max_val = float(max_str) if max_str.strip() else None
        if min_val is None and max_val is None:
            raise ValueError(f"At least one of min or max must be specified for field '{field}'.")
        filters[field] = (min_val, max_val)
    return filters


def _parse_srt_field(text: str, field: str) -> float | None:
    """Extract the first numeric value for field from DJI SRT text; returns None if absent."""
    m = re.search(rf'(?<!\w){re.escape(field)}\s*:\s*([-+]?\d*\.?\d+)', text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _load_srt_companion(video_path: Path) -> pysrt.SubRipFile | None:
    """Return the pysrt subtitle file for a video's companion SRT, or None if not found."""
    for ext in ('.SRT', '.srt'):
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            try:
                return pysrt.open(str(candidate), encoding='utf-8', error_handling=pysrt.ERROR_PASS)
            except Exception:
                return None
    return None


def _entry_passes_filters(
    text: str,
    filters: dict[str, tuple[float | None, float | None]],
) -> bool:
    """Return True if all filter criteria are satisfied by the SRT entry text."""
    for field, (min_val, max_val) in filters.items():
        value = _parse_srt_field(text, field)
        if value is None:
            continue  # field absent in this entry; do not reject
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
    return True


def _get_filtered_frames_srt(
    video_path: Path,
    filters: dict[str, tuple[float | None, float | None]],
    skip_start: int,
    skip_end: int,
    frame_count: int,
    logger: logging.Logger,
) -> list[int]:
    """Return sorted frame indices that pass SRT filters for this video."""
    srt = _load_srt_companion(video_path)
    if srt is None:
        logger.warning(
            f"No SRT companion found for '{video_path.name}'; "
            f"excluding it from the pool (--srt-filter requires a companion .srt file)."
        )
        return []

    fps = get_fps(video_path)
    valid_start = skip_start
    valid_end = frame_count - skip_end

    pool: set[int] = set()
    for entry in srt:
        if not _entry_passes_filters(entry.text, filters):
            continue
        start_f = max(valid_start, int(entry.start.ordinal / 1000 * fps))
        end_f = min(valid_end, int(entry.end.ordinal / 1000 * fps) + 1)
        pool.update(range(start_f, end_f))

    return sorted(pool)


def _get_filtered_frames_csv(
    video_path: Path,
    filters: dict[str, tuple[float | None, float | None]],
    skip_start: int,
    skip_end: int,
    frame_count: int,
    logger: logging.Logger,
) -> list[int]:
    """Return sorted frame indices that pass CSV column filters for this video."""
    csv_path = video_path.with_suffix('.csv')
    if not csv_path.exists():
        logger.warning(
            f"No CSV companion found for '{video_path.name}'; "
            f"excluding it from the pool (--csv-filter requires a companion .csv file)."
        )
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read '{csv_path.name}': {e}")
        return []

    if 'frame' not in df.columns:
        logger.warning(f"'frame' column missing in '{csv_path.name}'; skipping.")
        return []

    mask = pd.Series(True, index=df.index)
    for field, (min_val, max_val) in filters.items():
        if field not in df.columns:
            logger.warning(f"Field '{field}' not found in '{csv_path.name}'; filter for this field ignored.")
            continue
        if min_val is not None:
            mask &= df[field] >= min_val
        if max_val is not None:
            mask &= df[field] <= max_val

    valid_start = skip_start
    valid_end = frame_count - skip_end
    frames = sorted(
        int(f) for f in df.loc[mask, 'frame']
        if valid_start <= int(f) < valid_end
    )
    return frames


def sample_global(
    videos: list[Path],
    frame_counts: dict[Path, int],
    num_frames: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> list[tuple[Path, int]]:
    """Sample num_frames (video, local_idx) pairs uniformly at random across all videos."""
    cumulative = _build_cumulative(videos, frame_counts)
    total = cumulative[-1]

    n = min(num_frames, total)
    if n < num_frames:
        logger.warning(f"Requested {num_frames} frames but only {total} are available; extracting {total}.")

    chosen = np.sort(rng.choice(total, n, replace=False))
    result = [_resolve_index(int(idx), videos, cumulative) for idx in chosen]

    logger.info(f"Global sampling: selected {len(result)} frame(s) from {len(videos)} video(s).")
    return result


def sample_balanced(
    videos: list[Path],
    frame_counts: dict[Path, int],
    num_frames: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> list[tuple[Path, int]]:
    """Sample num_frames distributed evenly across all discovered video files."""
    total_available = sum(frame_counts.values())
    if num_frames > total_available:
        logger.warning(
            f"Requested {num_frames} frames but only {total_available} are available; extracting {total_available}."
        )

    # Sort ascending by frame count: under-capacity videos resolve first so
    # their deficit redistributes naturally to larger videos.
    videos_sorted = sorted(videos, key=lambda v: frame_counts[v])

    result: list[tuple[Path, int]] = []
    remaining = num_frames
    for i, video in enumerate(videos_sorted):
        videos_left = len(videos_sorted) - i
        quota = remaining // videos_left
        actual = min(quota, frame_counts[video])
        remaining -= actual

        if actual == 0:
            continue

        chosen = np.sort(rng.choice(frame_counts[video], actual, replace=False))
        for idx in chosen:
            result.append((video, int(idx)))

        logger.info(f"'{video.name}': selected {actual} frame(s) from {frame_counts[video]} available.")

    logger.info(f"Balanced sampling: selected {len(result)} frame(s) across {len(videos_sorted)} video(s).")
    return result


def resolve_input(
    input_path: Path,
    name_filter: str,
    logger: logging.Logger,
) -> tuple[list[Path], Path] | None:
    """Return (videos, data_dir) from a file or directory path, or None on error."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_FORMATS:
            logger.error(f"'{input_path}' is not a supported video file.")
            return None
        return [input_path], input_path.parent
    if input_path.is_dir():
        videos = find_videos(input_path, name_filter, logger)
        if not videos:
            filter_hint = f"'*{name_filter}*' " if name_filter else ''
            logger.warning(f"No {filter_hint}video files found under '{input_path}'.")
            return None
        return videos, input_path
    logger.error(f"'{input_path}' is not a file or directory.")
    return None


def collect_frame_counts(videos: list[Path], logger: logging.Logger) -> dict[Path, int]:
    """Return frame counts for all readable videos; warn and skip unreadable ones."""
    frame_counts: dict[Path, int] = {}
    for v in videos:
        fc = get_frame_count(v)
        if fc == 0:
            logger.warning(f"Could not determine frame count for '{v.name}'; skipping.")
        else:
            frame_counts[v] = fc
    return frame_counts


def build_sampling_pools(
    videos: list[Path],
    frame_counts: dict[Path, int],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[Path], dict[Path, int], dict[Path, list[int]] | None] | None:
    """Build per-video effective frame pools applying filters and skip bounds.

    Returns (videos, effective_counts, frame_pools) or None on fatal error.
    frame_pools is None when no filter is active (pool is an implicit offset range).
    """
    active_filter_strings = args.srt_filters or args.csv_filters
    use_csv = bool(args.csv_filters)

    if active_filter_strings:
        try:
            filters = _parse_meta_filters(active_filter_strings)
        except ValueError as e:
            logger.error(str(e))
            return None

        get_pool = _get_filtered_frames_csv if use_csv else _get_filtered_frames_srt
        frame_pools: dict[Path, list[int]] = {}
        for v in videos:
            pool = get_pool(v, filters, args.skip_start, args.skip_end, frame_counts[v], logger)
            if pool:
                frame_pools[v] = pool
            else:
                logger.warning(f"No frames passed the filter for '{v.name}'.")

        videos = [v for v in videos if v in frame_pools]
        if not videos:
            logger.error("No frames passed the filter criteria across all videos.")
            return None
        return videos, {v: len(frame_pools[v]) for v in videos}, frame_pools

    effective_counts = {
        v: max(0, fc - args.skip_start - args.skip_end)
        for v, fc in frame_counts.items()
    }
    videos = [v for v in videos if effective_counts[v] > 0]
    if not videos:
        logger.error("No frames remain after applying --skip-start / --skip-end.")
        return None
    return videos, effective_counts, None


def extract_and_save(
    selections: list[tuple[Path, int]],
    data_dir: Path,
    output_dir: Path,
    output_format: str,
    dry_run: bool,
    logger: logging.Logger,
) -> int:
    """Extract and save selected frames; return the count of successfully saved frames."""
    saved = 0
    for video_path, frame_idx in tqdm(selections, desc="Extracting frames", unit="frame"):
        output_path = output_dir / make_output_name(video_path, data_dir, frame_idx, output_format)

        if dry_run:
            logger.info(f"[dry-run] Would save '{output_path.name}' (frame {frame_idx} from '{video_path.name}').")
            saved += 1
            continue

        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            logger.warning(f"Failed to decode frame {frame_idx} from '{video_path.name}'; skipping.")
            continue

        if save_frame(frame, output_path, logger):
            saved += 1

    return saved


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Randomly sample frames from drone videos for annotation workflows.'
    )
    parser.add_argument('data_dir', type=Path,
                        help='Root directory (or single video file) to search for video files.')
    parser.add_argument('output_dir', type=Path,
                        help='Directory where extracted frame images are saved.')
    parser.add_argument('--num-frames', '-n', type=int, default=100,
                        help='Total number of frames to extract (default: 100).')
    parser.add_argument('--name-filter', '-nf', type=str, default='merged',
                        help="Keyword that must appear in the video filename stem (default: 'merged'). "
                             "Pass an empty string to discover all video files. Ignored for single-file input.")
    parser.add_argument('--balanced', '-b', action='store_true',
                        help='Distribute the frame quota evenly across all discovered video files; '
                             'default is global random sampling.')
    parser.add_argument('--output-format', '-of', type=str, default='png',
                        choices=sorted(_OUTPUT_FORMATS),
                        help="Image format for extracted frames (default: 'png').")
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--skip-start', '-ss', type=int, default=0,
                        help='Skip the first N frames of each video, e.g. to exclude takeoff (default: 0).')
    parser.add_argument('--skip-end', '-se', type=int, default=0,
                        help='Skip the last N frames of each video, e.g. to exclude landing (default: 0).')

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--srt-filter', action='append', dest='srt_filters',
                              metavar='FIELD:MIN:MAX',
                              help="Pre-filter frames using companion .srt/.SRT metadata "
                                   "(format: 'field:min:max'; repeatable; empty bound = unbounded). "
                                   "Mutually exclusive with --csv-filter.")
    filter_group.add_argument('--csv-filter', action='append', dest='csv_filters',
                              metavar='FIELD:MIN:MAX',
                              help="Pre-filter frames using companion .csv metadata "
                                   "(format: 'field:min:max'; repeatable; same syntax as --srt-filter). "
                                   "Mutually exclusive with --srt-filter.")

    parser.add_argument('--dry-run', '-dr', action='store_true',
                        help='Log all planned extractions without writing any files (default: off).')
    parser.add_argument('--log-path', '-lp', type=Path, default=None,
                        help='Where to write logs: a directory or a full file path; '
                             'defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    result = resolve_input(args.data_dir.resolve(), args.name_filter, logger)
    if result is None:
        return
    videos, data_dir = result

    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_counts = collect_frame_counts(videos, logger)
    videos = list(frame_counts)
    if not videos:
        logger.error("No readable video files found.")
        return

    pool_result = build_sampling_pools(videos, frame_counts, args, logger)
    if pool_result is None:
        return
    videos, effective_counts, frame_pools = pool_result

    rng = np.random.default_rng(args.seed)
    sample_fn = sample_balanced if args.balanced else sample_global
    selections_local = sample_fn(videos, effective_counts, args.num_frames, rng, logger)

    if frame_pools is not None:
        selections = [(v, frame_pools[v][li]) for v, li in selections_local]
    else:
        selections = [(v, li + args.skip_start) for v, li in selections_local]

    saved = extract_and_save(selections, data_dir, output_dir, args.output_format, args.dry_run, logger)
    logger.notice(
        f"Done: {'(dry-run) ' if args.dry_run else ''}{saved}/{len(selections)} frame(s) saved to '{output_dir}'."
    )


if __name__ == '__main__':
    main()
