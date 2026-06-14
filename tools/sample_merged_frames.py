#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
sample_merged_frames.py - Random Frame Sampler for Drone Videos

Randomly samples frames from drone video files for use in object detection annotation
workflows. While the tool works on any video file, its primary use case is sampling
from merged videos produced by tools/merge_videos_and_logs.py (default --name-filter
'merged'). Sampled frames can be fed into annotation workflows ranging from fully
manual labelling to semi-automatic pipelines (e.g. tools/annotate_frames.py, which
runs a YOLO detector to produce pre-label files that a human then verifies and corrects)
or fully automatic evaluation pipelines.

Two sampling modes are supported:
- Global (default): selects frames uniformly at random across the combined pool of all
  discovered video files, giving proportionally more frames from longer videos.
- Balanced (--balanced): distributes the total quota as evenly as possible across all
  discovered video files, so shorter videos are not under-represented in the
  annotation set.

This tool was developed for the Songdo Traffic dataset accompanying:
  Fonod et al., "Advanced computer vision for extracting georeferenced vehicle
  trajectories from drone imagery," Transportation Research Part C, 2025.
  https://doi.org/10.1016/j.trc.2025.105205
In the Songdo experiment, each drone hovered over multiple intersections per session.
After merging, each session directory contains one 0_merged.mp4. The balanced mode
ensures all merged video files contribute equally to the annotation pool regardless
of their duration.

Usage:
  python tools/sample_merged_frames.py <data_dir> <output_dir> [options]

Arguments:
  data_dir   : Root directory to search recursively for video files.
  output_dir : Directory where extracted frame images are saved.

Options:
  -h, --help                   : Show this help message and exit.
  -n, --num-frames <int>       : Total number of frames to extract (default: 100).
  -nf, --name-filter <str>     : Keyword that must appear in the video filename stem
                                 (default: 'merged'). Case-insensitive. Pass an empty
                                 string to discover all video files.
  -b, --balanced               : Distribute the frame quota evenly across all discovered
                                 video files; default is global random sampling across
                                 the entire frame pool.
  -of, --output-format <str>   : Image format for extracted frames: 'png', 'jpg', or
                                 'jpeg' (default: 'png').
  -s, --seed <int>             : Random seed for reproducibility (default: 42).
  -dr, --dry-run               : Log all planned extractions without writing any files
                                 (default: off).
  -lp, --log-path <str>        : Where to write logs: a directory or a full file path;
                                 defaults to a platform-specific log directory.
  -q, --quiet                  : Reduce console verbosity to important messages only
                                 (default: show INFO-level detail).

Examples:
1. Extract 100 frames globally at random from all merged videos:
   python tools/sample_merged_frames.py /path/to/PROCESSED /path/to/frames

2. Extract 200 frames balanced across merged video files, reproducible:
   python tools/sample_merged_frames.py /path/to/PROCESSED /path/to/frames -n 200 --balanced

3. Dry run — preview balanced selection without writing files:
   python tools/sample_merged_frames.py /path/to/PROCESSED /path/to/frames -n 50 --balanced --dry-run

4. Songdo dataset: balanced sampling from merged videos, JPEG output:
   python tools/sample_merged_frames.py /path/to/PROCESSED/2022-10-04 /path/to/frames \\
     --balanced --output-format jpg --seed 0

5. Sample from cut (per-intersection) videos instead (after --cleanup removed merged files):
   python tools/sample_merged_frames.py /path/to/PROCESSED /path/to/frames \\
     --name-filter '' --balanced

Input:
- Video files (any of: .mp4, .mov, .avi, .mkv) whose stem contains the name filter
  keyword (default: 'merged') anywhere in the filename, located recursively under
  <data_dir>. Pass an empty string for --name-filter to discover all video files.

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
              ├── 0_merged.srt
              └── 0_merged.txt

  frames/                            ← output_dir
  └── frame_2022-10-04_D1_AM1_0_merged_001234.png
"""

from __future__ import annotations

import argparse
import bisect
import logging
from pathlib import Path

import cv2
import numpy as np
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


def sample_global(
    videos: list[Path],
    frame_counts: dict[Path, int],
    num_frames: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> list[tuple[Path, int]]:
    """Sample num_frames (video, frame_idx) pairs uniformly at random across all videos."""
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


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Randomly sample frames from drone videos for annotation workflows.'
    )
    parser.add_argument('data_dir', type=Path,
                        help='Root directory to search recursively for video files.')
    parser.add_argument('output_dir', type=Path,
                        help='Directory where extracted frame images are saved.')
    parser.add_argument('--num-frames', '-n', type=int, default=100,
                        help='Total number of frames to extract (default: 100).')
    parser.add_argument('--name-filter', '-nf', type=str, default='merged',
                        help="Keyword that must appear in the video filename stem (default: 'merged'). Pass an empty string to discover all video files.")
    parser.add_argument('--balanced', '-b', action='store_true',
                        help='Distribute the frame quota evenly across all discovered video files; default is global random sampling.')
    parser.add_argument('--output-format', '-of', type=str, default='png',
                        choices=sorted(_OUTPUT_FORMATS),
                        help="Image format for extracted frames (default: 'png').")
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--dry-run', '-dr', action='store_true',
                        help='Log all planned extractions without writing any files (default: off).')
    parser.add_argument('--log-path', '-lp', type=Path, default=None,
                        help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        logger.error(f"'{data_dir}' is not a directory.")
        return

    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(data_dir, args.name_filter, logger)
    if not videos:
        filter_hint = f"'*{args.name_filter}*' " if args.name_filter else ''
        logger.warning(f"No {filter_hint}video files found under '{data_dir}'.")
        return

    frame_counts: dict[Path, int] = {}
    for v in videos:
        fc = get_frame_count(v)
        if fc == 0:
            logger.warning(f"Could not determine frame count for '{v.name}'; skipping.")
        else:
            frame_counts[v] = fc

    videos = [v for v in videos if v in frame_counts]
    if not videos:
        logger.error("No readable video files found.")
        return

    rng = np.random.default_rng(args.seed)

    if args.balanced:
        selections = sample_balanced(videos, frame_counts, args.num_frames, rng, logger)
    else:
        selections = sample_global(videos, frame_counts, args.num_frames, rng, logger)

    saved = 0
    for video_path, frame_idx in tqdm(selections, desc="Extracting frames", unit="frame"):
        output_name = make_output_name(video_path, data_dir, frame_idx, args.output_format)
        output_path = output_dir / output_name

        if args.dry_run:
            logger.info(f"[dry-run] Would save '{output_path.name}' (frame {frame_idx} from '{video_path.name}').")
            saved += 1
            continue

        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            logger.warning(f"Failed to decode frame {frame_idx} from '{video_path.name}'; skipping.")
            continue

        if save_frame(frame, output_path, logger):
            saved += 1

    logger.notice(
        f"Done: {'(dry-run) ' if args.dry_run else ''}{saved}/{len(selections)} frame(s) saved to '{output_dir}'."
    )


if __name__ == '__main__':
    main()
