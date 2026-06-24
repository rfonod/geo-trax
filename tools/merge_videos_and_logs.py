#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Sohyeong Kim (sohyeong.kim@epfl.ch) and Robert Fonod (robert.fonod@ieee.org)

"""
merge_videos_and_logs.py - Video and Flight Log Merging Tool

Merges multiple per-flight video files and their DJI SRT flight logs from one or
more session directories into merged video and SRT files, one pair per session.

DJI drones automatically split continuous recordings into multiple smaller files
(typically capped at ~4 GB to remain within FAT32 file-system limits), so a single
hover session produces several .mp4/.srt file pairs. This tool concatenates them in
sorted filename order into a single .mp4 and a single .srt that span the full session.
The merged .srt preserves correct wall-clock timestamps and frame counters across files.

The merged files serve as input to tools/cut_merged_videos_and_logs.py, which cuts
them into labelled clips according to user-defined cut specification files. Cutting
points must be defined by the user — either manually by inspecting the merged video
or with an automated scene-change detector. An optional location map can then be
supplied to auto-label each clip with the nearest named location (e.g. an intersection
ID); this is what determines the output filenames, not the cutting itself.

This tool was developed for the Songdo Traffic dataset accompanying:
  Fonod et al., "Advanced computer vision for extracting georeferenced vehicle
  trajectories from drone imagery," Transportation Research Part C, 2025.
  https://doi.org/10.1016/j.trc.2025.105205
In the Songdo experiment, each drone monitored multiple intersections per session,
moving between them to maximise coverage. Merging the split files into one merged
video per session allows the downstream cutting tool to produce one clip per
intersection from a single merged source.

Usage:
  python tools/merge_videos_and_logs.py <source_dir> [options]

Arguments:
  source_dir : Root directory to search recursively for per-flight video files.
               Can be a single session directory or the root of an entire experiment
               (all session sub-directories are discovered automatically).

Options:
  -h, --help                 : Show this help message and exit.
  -od, --output-dir <path>   : Root directory for merged output files. The
                               subdirectory structure under <source_dir> is mirrored
                               here (default: <source_dir>, i.e. merged files are
                               written into each session directory alongside the raws).
  -os, --output-stem <str>   : Stem for merged output filenames (default: '0_merged').
                               Produces <output_stem>.mp4 and <output_stem>.srt per session.
  -ve, --video-ext <str>     : Video file extension to search for, including the leading
                               dot (default: '.MP4'). Case-insensitive.
  -ow, --overwrite           : Overwrite existing merged output files (default: skip).
  -dr, --dry-run             : Simulate merging without writing any files (default: off).
  -lp, --log-path <str>      : Where to write logs: a directory or a full file path;
                               defaults to a platform-specific log directory.
  -q, --quiet                : Reduce console verbosity to important messages only
                               (default: show INFO-level detail).

Examples:
1. Merge a single session:
   python tools/merge_videos_and_logs.py /path/to/RAW/2022-10-04/D1/AM1 \\
     --output-dir /path/to/PROCESSED/2022-10-04/D1/AM1

2. Merge all sessions across an entire experiment in one pass:
   python tools/merge_videos_and_logs.py /path/to/RAW --output-dir /path/to/PROCESSED

3. Dry run to preview what would be merged across a full experiment:
   python tools/merge_videos_and_logs.py /path/to/RAW --output-dir /path/to/PROCESSED --dry-run

4. Re-merge, overwriting previously merged files:
   python tools/merge_videos_and_logs.py /path/to/RAW --output-dir /path/to/PROCESSED --overwrite

Input:
- One or more session directories, each containing per-flight video files
  (e.g. DJI_0001.MP4, DJI_0002.MP4) and their companion DJI SRT flight logs
  (e.g. DJI_0001.SRT, DJI_0002.SRT). Files within each session are sorted by
  filename — DJI names files with monotonically increasing counters (DJI_0001,
  DJI_0002, ...) and files are typically copied directly from the drone without
  renaming, so filename order equals recording order. No SRT timestamp inspection
  is needed.
  Some DJI drones append '_trimmed' to the last video in a series
  (e.g. DJI_0212_trimmed.mp4) while the companion SRT keeps the base stem
  (DJI_0212.SRT); both exact and base-stem lookups are performed automatically.
  Video files are validated with ffprobe before merging; corrupted files are skipped.

Output (per discovered session, written to the mirrored path under <output_dir>):
- <output_stem>.mp4 : Merged video (stream-copy concatenation, no re-encoding).
- <output_stem>.srt : Merged DJI SRT flight log with adjusted timestamps and frame
                      counters. Written only when at least one SRT companion is found.

Notes:
- Video merging uses ffmpeg stream-copy (no quality loss, very fast).
- SRT merging is implemented in pure Python; no additional dependencies are required
  beyond the core geo-trax install. Both 'FrameCnt : N' and 'SrtCnt : N' counter
  formats used across DJI drone families are handled automatically.
- GPS coordinates and other telemetry fields in the SRT content are preserved
  verbatim from the original per-flight files.
- If a video file has no companion SRT, it is still included in the video merge but
  its corresponding time window will be absent from the merged SRT (with a warning).
- Existing output files are skipped unless --overwrite is specified.
- It is recommended to keep <source_dir> and <output_dir> as separate root directories
  to prevent the recursive scan from picking up previously merged files on a second run.

Directory structure (Songdo dataset — non-prescriptive example):
  RAW/
  └── <ISO8601_date>/       # e.g. 2022-10-04
      └── D<drone_id>/      # e.g. D1, D2
          └── <session>/    # e.g. AM1, AM2, PM1, PM2
              ├── DJI_0001.MP4
              ├── DJI_0001.SRT
              ├── DJI_0002.MP4
              ├── DJI_0002.SRT
              └── ...

  PROCESSED/
  └── <ISO8601_date>/
      └── D<drone_id>/
          └── <session>/
              ├── 0_merged.mp4
              ├── 0_merged.srt
              └── 0_merged.txt  ← cut specification (created manually by the user)
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from geotrax.utils.logging_utils import setup_logger


def find_session_dirs(source_dir: Path, video_ext: str, logger: logging.Logger) -> list[Path]:
    """Recursively find all directories that directly contain at least one video file."""
    session_dirs = sorted({
        p.parent for p in source_dir.rglob('*')
        if p.is_file() and p.suffix.lower() == video_ext.lower()
    })
    logger.info(f"Found {len(session_dirs)} session director{'y' if len(session_dirs) == 1 else 'ies'} under '{source_dir}'.")
    return session_dirs


def find_video_srt_pairs(
    session_dir: Path,
    video_ext: str,
    logger: logging.Logger,
) -> list[tuple[Path, Path | None]]:
    """Scan session_dir for video files (sorted by name), validate them, and find companion SRTs."""
    video_files = sorted(
        p for p in session_dir.iterdir()
        if p.is_file() and p.suffix.lower() == video_ext.lower()
    )

    if not video_files:
        logger.error(f"No '{video_ext}' files found in '{session_dir}'.")
        return []

    pairs = []
    for video in video_files:
        if not _is_valid_video(video, logger):
            logger.warning(f"Skipping corrupted or unreadable video: '{video.name}'.")
            continue
        srt = _find_companion_srt(video, logger)
        pairs.append((video, srt))

    return pairs


def _is_valid_video(video_path: Path, logger: logging.Logger) -> bool:
    """Return True if ffprobe reports no errors for the video file."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        logger.error(
            f"ffprobe error for '{video_path.name}': "
            f"{result.stdout.decode(errors='replace').strip()}"
        )
        return False
    return True


def _find_companion_srt(video: Path, logger: logging.Logger) -> Path | None:
    """Return the SRT companion for a video (tries lower- and upper-case extensions).

    Some DJI drones append '_trimmed' to the last video file in a series
    (e.g. DJI_0212_trimmed.mp4) while the companion SRT retains the base stem
    (DJI_0212.SRT). Both the exact stem and the de-trimmed base stem are tried.
    """
    stems_to_try = [video.stem]
    if video.stem.lower().endswith('_trimmed'):
        stems_to_try.append(video.stem[:-len('_trimmed')])

    for stem in stems_to_try:
        for ext in ('.srt', '.SRT'):
            candidate = video.with_name(stem + ext)
            if candidate.exists():
                if candidate.stat().st_size == 0:
                    logger.warning(f"'{candidate.name}' is empty; skipping SRT for this flight.")
                    return None
                if stem != video.stem:
                    logger.info(f"Using '{candidate.name}' as SRT companion for '{video.name}' (base stem match).")
                return candidate

    logger.warning(f"No SRT companion found for '{video.name}'; this flight will be excluded from the SRT merge.")
    return None


def merge_videos(
    video_files: list[Path],
    output_path: Path,
    overwrite: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> bool:
    """Concatenate video_files into output_path using ffmpeg stream-copy."""
    if output_path.exists() and not overwrite:
        logger.info(f"Merged video already exists at '{output_path}'; skipping (use --overwrite to force).")
        return True

    logger.info(f"Merging {len(video_files)} video file(s) into '{output_path.name}':")
    for video in video_files:
        logger.info(f"  + {video.name}")

    if dry_run:
        logger.info(f"[dry-run] Would write merged video to '{output_path}'.")
        return True

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        manifest_path = Path(f.name)
        for video in video_files:
            f.write(f"file '{video}'\n")

    result = subprocess.run([
        'ffmpeg', '-loglevel', 'error', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', str(manifest_path),
        '-codec', 'copy',
        str(output_path),
    ])
    manifest_path.unlink(missing_ok=True)

    if result.returncode == 0:
        logger.notice(f"Merged video saved to '{output_path}'.")
        return True
    else:
        logger.error(f"ffmpeg failed (exit code {result.returncode}) while merging video.")
        return False


def merge_srt_files(
    srt_files: list[Path],
    output_path: Path,
    overwrite: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> bool:
    """Merge SRT flight logs into a single file with adjusted timestamps and frame counters."""
    if output_path.exists() and not overwrite:
        logger.info(f"Merged SRT already exists at '{output_path}'; skipping (use --overwrite to force).")
        return True

    logger.info(f"Merging {len(srt_files)} SRT file(s) into '{output_path.name}':")
    for srt in srt_files:
        logger.info(f"  + {srt.name}")

    merged_blocks: list[dict] = []
    time_offset_ms = 0
    frame_offset = 0

    for srt_file in srt_files:
        text = srt_file.read_text(encoding='utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')
        blocks = _parse_srt_blocks(text)
        if not blocks:
            logger.warning(f"No SRT blocks parsed from '{srt_file.name}'; skipping.")
            continue

        for block in blocks:
            merged_blocks.append({
                'start_ms': block['start_ms'] + time_offset_ms,
                'end_ms': block['end_ms'] + time_offset_ms,
                'content': _update_srt_frame_count(block['content'], frame_offset),
            })

        time_offset_ms += blocks[-1]['end_ms']
        last_fc = _get_last_frame_count(blocks)
        if last_fc is not None:
            frame_offset += last_fc

    if not merged_blocks:
        logger.error("No SRT blocks could be merged.")
        return False

    srt_text = _blocks_to_srt(merged_blocks)

    if dry_run:
        logger.info(f"[dry-run] Would write merged SRT ({len(merged_blocks)} blocks) to '{output_path}'.")
        return True

    output_path.write_text(srt_text, encoding='utf-8')
    logger.notice(f"Merged SRT saved to '{output_path}'.")
    return True


def _srt_ts_to_ms(ts: str) -> int:
    """Convert SRT timestamp 'HH:MM:SS,mmm' to milliseconds."""
    h, m, rest = ts.split(':')
    s, ms = rest.split(',')
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def _ms_to_srt_ts(ms: int) -> str:
    """Convert milliseconds to SRT timestamp 'HH:MM:SS,mmm'."""
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1_000)
    return f'{h:02d}:{m:02d}:{s:02d},{msec:03d}'


def _parse_srt_blocks(text: str) -> list[dict]:
    """Parse SRT text into a list of block dicts (start_ms, end_ms, content)."""
    blocks = []
    for raw in re.split(r'\n\s*\n', text.strip()):
        lines = raw.strip().splitlines()
        if len(lines) < 3:
            continue
        timing_match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            lines[1].strip(),
        )
        if not timing_match:
            continue
        blocks.append({
            'start_ms': _srt_ts_to_ms(timing_match.group(1)),
            'end_ms': _srt_ts_to_ms(timing_match.group(2)),
            'content': '\n'.join(lines[2:]),
        })
    return blocks


def _get_last_frame_count(blocks: list[dict]) -> int | None:
    """Extract the frame counter value from the last block's content."""
    match = re.search(r'(?:FrameCnt|SrtCnt)\s*:\s*(\d+)', blocks[-1]['content'])
    return int(match.group(1)) if match else None


def _update_srt_frame_count(content: str, offset: int) -> str:
    """Add offset to the FrameCnt / SrtCnt value in an SRT block's content."""
    if offset == 0:
        return content
    return re.sub(
        r'((?:FrameCnt|SrtCnt)\s*:\s*)(\d+)',
        lambda m: m.group(1) + str(int(m.group(2)) + offset),
        content,
    )


def _blocks_to_srt(blocks: list[dict]) -> str:
    """Serialise merged blocks to SRT text with sequential 1-based indices."""
    parts = []
    for i, block in enumerate(blocks, start=1):
        parts.append(
            f'{i}\n'
            f'{_ms_to_srt_ts(block["start_ms"])} --> {_ms_to_srt_ts(block["end_ms"])}\n'
            f'{block["content"]}\n'
        )
    return '\n'.join(parts)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge per-flight DJI videos and SRT flight logs into one merged file per session.'
    )
    parser.add_argument('source_dir', type=Path, help='Root directory to search recursively for per-flight video files.')
    parser.add_argument('--output-dir', '-od', type=Path, default=None, help='Root directory for merged output files; subdirectory structure under <source_dir> is mirrored (default: <source_dir>).')
    parser.add_argument('--output-stem', '-os', type=str, default='0_merged', help="Stem for merged output filenames (default: '0_merged').")
    parser.add_argument('--video-ext', '-ve', type=str, default='.MP4', help="Video file extension to search for, including the leading dot (default: '.MP4'). Case-insensitive.")
    parser.add_argument('--overwrite', '-ow', action='store_true', help='Overwrite existing merged output files (default: skip).')
    parser.add_argument('--dry-run', '-dr', action='store_true', help='Simulate merging without writing any files (default: off).')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        logger.error(f"'{source_dir}' is not a directory.")
        return

    output_root = (args.output_dir or source_dir).resolve()

    session_dirs = find_session_dirs(source_dir, args.video_ext, logger)
    if not session_dirs:
        logger.error(f"No '{args.video_ext}' files found under '{source_dir}'.")
        return

    ext_lower = args.video_ext.lstrip('.').lower()

    for session_dir in session_dirs:
        rel = session_dir.relative_to(source_dir)
        out_dir = output_root / rel
        logger.info(f"--- Session: '{session_dir}' ---")

        pairs = find_video_srt_pairs(session_dir, args.video_ext, logger)
        if not pairs:
            logger.warning(f"No valid video files in '{session_dir}'; skipping.")
            continue

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        output_video = out_dir / f'{args.output_stem}.{ext_lower}'
        output_srt = out_dir / f'{args.output_stem}.srt'

        video_files = [v for v, _ in pairs]
        srt_files = [s for _, s in pairs if s is not None]

        merge_videos(video_files, output_video, args.overwrite, args.dry_run, logger)

        if srt_files:
            missing = len(pairs) - len(srt_files)
            if missing:
                logger.warning(f"{missing} flight(s) have no SRT; their metadata will be absent from the merged log.")
            merge_srt_files(srt_files, output_srt, args.overwrite, args.dry_run, logger)
        else:
            logger.warning("No SRT flight logs found in this session; only the video will be merged.")


if __name__ == '__main__':
    main()
