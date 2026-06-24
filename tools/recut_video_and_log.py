#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
recut_video_and_log.py - Video and Flight Log Re-cutting Tool

This script re-cuts video files and their corresponding flight logs according to specified frame ranges.
It performs keyframe-aligned cutting to avoid re-encoding and maintains synchronization between video
and CSV flight data.

The tool reads cut specifications from a text file and applies them to both the video and its
associated flight log, adjusting frame numbers and timestamps accordingly.

Usage:
  python tools/recut_video_and_log.py <input_video> [<cuts>] [options]
  python tools/recut_video_and_log.py <input_video> --start <frame> --end <frame> [options]

Arguments:
  input_video : Path to the input video file to be cut.
  cuts        : Path to the cuts specification file (optional; mutually exclusive with --start/--end).

Options:
  -h, --help              : Show this help message and exit.
  -i, --input-csv <path>  : Full path to the input CSV flight log (default: same directory and stem as
                            the input video, tried with both .csv and .CSV extensions).
  -s, --start <frame>     : Cut start frame number (mutually exclusive with <cuts>; requires --end).
  -e, --end <frame>       : Cut end frame number, -1 for end of video (mutually exclusive with <cuts>;
                            requires --start).
  -r, --rotate <deg>      : Optional counter-clockwise rotation in degrees (0, ±90, ±180, ±270).
  -o, --output <path>     : Full file path for the output video, including filename and extension
                            (e.g., /path/to/cut.MP4); the companion CSV log is saved to the same path
                            with the extension replaced by .csv or .CSV to match the video suffix.
                            Default: <input_video_dir>/<input_video_stem>_cut<ext> with a matching CSV
                            saved alongside it.
  -ec, --exact-cut        : Perform exact frame cutting with re-encoding (slower but precise; default: off).
  -b, --bitrate <rate>    : Video bitrate, used only with --exact-cut (e.g., '5M', '10000k'; default: auto).
  -d, --debug             : Run in debug mode with frame verification and detailed output (default: off).
  -lp, --log-path <str>   : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet             : Reduce console verbosity to important messages only (default: show INFO-level detail).

Cut File Format:
  cut_start_frame_number, cut_end_frame_number, [rotation]

  Where:
  - cut_start_frame_number : Frame where the cut video should start
  - cut_end_frame_number   : Frame where the cut video should end (-1 for end of video)
  - rotation               : Optional counter-clockwise rotation in degrees (0, ±90, ±180, ±270)

Examples:
1. Basic video cutting:
   python tools/recut_video_and_log.py video.MP4 cuts.txt

2. Cut with custom output path:
   python tools/recut_video_and_log.py video.MP4 cuts.txt --output cut_video.MP4

3. Debug mode with verification:
   python tools/recut_video_and_log.py video.MP4 cuts.txt --debug

4. Cut specified directly via CLI (no cuts file):
    python tools/recut_video_and_log.py video.MP4 --start 120 --end 540 --rotate 90 --output cut_video.MP4

Input:
- Video file (any format supported by ffmpeg)
- CSV flight log expected alongside the video with the same stem and .CSV extension
  (optional; skipped if absent or missing a 'frame' column)
- Cuts file with a single line specifying the frame range and optional rotation
  (omit if using --start/--end instead)

Output:
- Cut video saved to the --output path, or to the input video's directory as <stem>_cut<ext>
- Cut CSV log saved alongside the output video with the same stem and .csv/.CSV extension
- Debug verification output (if --debug is enabled)

Notes:
- By default, cut start frame is adjusted to the nearest keyframe to avoid re-encoding (fast)
- Use --exact-cut to cut at exact frames with re-encoding (slower but precise)
- Flight log is optional; if not found or missing 'frame' column, only video will be cut
- Frame numbers in the CSV are adjusted relative to the new cut start
- Rotation is applied at metadata level without re-encoding
- Debug mode verifies cut accuracy by comparing frame differences
"""

import argparse
import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from geotrax.utils.logging_utils import setup_logger


def process_cutting(filepaths: Dict[str, Path], logger: logging.Logger, cuts: Tuple[int, int, int] = None, debug: bool = False, exact_cut: bool = False, bitrate: str = None) -> None:
    if cuts is None:
        cuts = get_cuts(filepaths, logger)
    perform_sanity_checks(cuts, filepaths)
    cuts_adjusted = get_adjusted_cuts(cuts, filepaths, logger, debug, exact_cut)
    cut_and_save_video(filepaths, cuts_adjusted, logger, debug, exact_cut, bitrate)
    cut_and_save_csv(filepaths, cuts_adjusted, logger, debug)


def cut_and_save_video(filepaths: Dict[str, Path], cuts: Tuple[int, int, int], logger: logging.Logger, debug: bool = False, exact_cut: bool = False, bitrate: str = None) -> None:
    cut_start, cut_end, rotation = cuts
    input_video = str(filepaths["input_video"])
    output_video = str(filepaths["output_video"])

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Quote paths for safe shell execution
    input_video_q = shlex.quote(input_video)
    output_video_q = shlex.quote(output_video)

    # Choose cutting mode
    if exact_cut:
        # Exact frame cutting with re-encoding
        cmd = f'ffmpeg -y -i {input_video_q} -ss {cut_start/fps} -to {cut_end/fps}'
        if bitrate:
            cmd += f' -b:v {bitrate}'
        cmd += f' -async 1 -strict -2 {output_video_q}'
    else:
        # Fast keyframe-aligned cutting without re-encoding
        cmd = f'ffmpeg -y -i {input_video_q}'
        if cut_start > 0:
            cmd += f' -ss {cut_start / fps}'
        if cut_end != -1:
            cmd += f' -to {cut_end / fps}'
        cmd += f' -c copy {output_video_q}'

    if not debug:
        cmd += ' -v quiet'
    logger.info(f"Running the following command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        logger.notice(f"Cut video saved to '{output_video}'")
    else:
        logger.error(f"ffmpeg exited with code {result.returncode} for '{input_video}'")
    if debug:
        verify_cut(filepaths, cut_start, cut_end, logger, debug)

    # if needed, perform counter-clockwise rotation (metadata level rotation, no re-encoding needed)
    if int(rotation) != 0 and not debug:
        output_path = Path(output_video)
        temp_filepath = str(output_path.with_name(output_path.stem + "_temp" + output_path.suffix))
        temp_filepath_q = shlex.quote(temp_filepath)
        os.rename(output_video, temp_filepath)
        subprocess.run(
            f'ffmpeg -i {temp_filepath_q} -c copy -map_metadata 0 -metadata:s:v rotate="{rotation}" {output_video_q} -v quiet',
            shell=True
        )
        os.remove(temp_filepath)


def cut_and_save_csv(filepaths: Dict[str, Path], cuts: Tuple[int, int, int], logger: logging.Logger, debug: bool = False) -> None:
    input_csv = filepaths["input_csv"]
    output_csv = filepaths["output_csv"]

    # Check if CSV file exists
    if not input_csv.exists():
        logger.warning(f"No flight log found at '{input_csv}', skipping CSV cutting.")
        return

    cut_start = cuts[0]
    cut_end = cuts[1]
    if cut_end == -1:
        cap = cv2.VideoCapture(str(filepaths["input_video"]))
        cut_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()

    try:
        df = pd.read_csv(input_csv)
        # Check if 'frame' column exists
        if 'frame' not in df.columns:
            logger.warning(f"'frame' column not found in '{input_csv}', skipping CSV cutting.")
            return
        df = df[(df['frame'] >= cut_start) & (df['frame'] <= cut_end)]
        df['frame'] = df['frame'] - cut_start
        df.to_csv(output_csv, index=False)
        logger.notice(f"Saved the cut flight log to '{output_csv}'")
    except Exception as e:
        logger.error(f"Problem with cutting the flight log '{input_csv}': {e}")


def verify_cut(
    filepaths: Dict[str, Path], cut_start: int, cut_end: int, logger: logging.Logger, debug=False, verify_N_frames: int = 30
) -> None:
    input_video = str(filepaths["input_video"])
    output_video = str(filepaths["output_video"])

    if cut_end == -1:
        cap = cv2.VideoCapture(input_video)
        cut_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
    total_frame_count = cut_end - cut_start
    verify_every_nth_frame = total_frame_count // verify_N_frames
    input_frames_selected, cut_frames_selected = [], []

    try:
        cap = cv2.VideoCapture(input_video)
    except Exception as e:
        logger.error(f"Problem with reading '{input_video}': {e}")
    else:
        logger.info(f"Total number of frames in the input video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        logger.info(f"FPS of the input video: {cap.get(cv2.CAP_PROP_FPS)}")
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i >= cut_start and (i - cut_start) % verify_every_nth_frame == 0:
                input_frames_selected.append(frame)
            i += 1
    finally:
        cap.release()

    try:
        cap = cv2.VideoCapture(output_video)
    except Exception as e:
        logger.error(f"Problem with reading '{output_video}': {e}")
    else:
        logger.info(f"Total number of frames in the cut video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        logger.info(f"FPS of the cut video: {cap.get(cv2.CAP_PROP_FPS)}")
        i = 0
        while cap.isOpened() and i < cut_end:
            ret, frame = cap.read()
            if not ret:
                break
            if i % verify_every_nth_frame == 0:
                cut_frames_selected.append(frame)
            i += 1
    finally:
        cap.release()

    for i, imgs_i in enumerate(zip(input_frames_selected, cut_frames_selected)):
        img_input_i, img_cut_i = imgs_i
        if img_input_i is None or img_cut_i is None:
            logger.error(f"Problem with reading frame {i} from the input/cut video")
            break
        img_diff = cv2.absdiff(img_input_i, img_cut_i)
        img_diff_rmse = np.sqrt(np.mean(img_diff**2))
        logger.info(
            f"({i}) Mean RMSE of the cut frame #: {i * verify_every_nth_frame} wrt input frame #: {cut_start + i * verify_every_nth_frame} is: {img_diff_rmse}"
        )
        if debug:
            cv2.imshow(f"image {i}", img_diff)
            cv2.waitKey(200)
            # k = cv2.waitKey(0) & 0xFF
            # if k == ord('q'):
            #     break
    if debug:
        cv2.destroyAllWindows()


def get_adjusted_cuts(
    cuts: Tuple[int, int, int], filepaths: Dict[str, Path], logger: logging.Logger, debug: bool = False, exact_cut: bool = False
) -> Tuple[int, int, int]:
    cut_start = cuts[0]
    cut_end = cuts[1]

    # Skip keyframe adjustment if exact cutting is requested
    if exact_cut:
        logger.info(f"Exact cutting enabled: cutting from frame {cut_start} to frame {cut_end} (with re-encoding).")
        return cuts

    input_video = str(filepaths['input_video'])
    input_video_q = shlex.quote(input_video)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Handle -1 for cut_end (end of video)
    if cut_end == -1:
        cut_end = frame_count - 1

    # Retrieve key-frames using ffprobe/ffmpeg and awk based on OS
    if platform.system() == "Windows" or platform.system() == "Darwin":
        key_frames_retrieval_cmd = (
            "ffprobe -loglevel error -select_streams v:0 "
            "-show_entries packet=pts_time,flags -of "
            f"csv=print_section=0 {input_video_q}"
            " | awk -F',' '/K/ {print $1}'"
        )
    elif platform.system() == "Linux":
        key_frames_retrieval_cmd = (
            f"ffmpeg -i {input_video_q} -vf select='eq(pict_type\\,PICT_TYPE_I)',showinfo "
            "-vsync vfr -f null - -loglevel debug 2>&1 | "
            "awk '/pts_time/ {gsub(/.*pts_time:/, \"\"); gsub(/ .*/, \"\"); print;}'"
        )
    else:
        raise RuntimeError("Unsupported operating system for key-frame retrieval.")

    key_frames = subprocess.check_output(key_frames_retrieval_cmd, shell=True, text=True).split()
    key_frames_arr = np.array([float(key_frame) for key_frame in key_frames])

    # Find the closest key-frame (from right) for the provided cut_start
    if cut_start == 0:
        key_frames_diffs_start = key_frames_arr
        i_key_frame_closest_start = 0
    else:
        key_frames_diffs_start = key_frames_arr - cut_start / fps
        i_key_frame_closest_start = np.where(key_frames_diffs_start >= 0, key_frames_diffs_start, np.inf).argmin()

    closest_key_frame_start = float(key_frames[i_key_frame_closest_start])
    cut_start_adjusted = round(closest_key_frame_start * fps)

    # Find the closest key-frame (from left) for the provided cut_end
    key_frames_diffs_end = key_frames_arr - cut_end / fps
    i_key_frame_closest_end = np.where(key_frames_diffs_end <= 0, -key_frames_diffs_end, np.inf).argmin()
    closest_key_frame_end = float(key_frames[i_key_frame_closest_end])
    cut_end_adjusted = round(closest_key_frame_end * fps)

    # Print informative messages
    start_adjusted = cut_start_adjusted != cut_start
    end_adjusted = cut_end_adjusted != cut_end

    if not start_adjusted and not end_adjusted:
        logger.info(f"Requested cut frames ({cut_start} to {cut_end}) are already at keyframes. No adjustment needed.")
    else:
        if start_adjusted:
            frame_diff_start = cut_start_adjusted - cut_start
            logger.info(f"Adjusted cut start from frame {cut_start} to frame {cut_start_adjusted} (+{frame_diff_start} frames to nearest keyframe).")
        else:
            logger.info(f"Requested cut start frame {cut_start} is already at a keyframe.")

        if end_adjusted:
            frame_diff_end = cut_end_adjusted - cut_end
            logger.info(f"Adjusted cut end from frame {cut_end} to frame {cut_end_adjusted} ({frame_diff_end:+d} frames to nearest keyframe).")
        else:
            logger.info(f"Requested cut end frame {cut_end} is already at a keyframe.")

    if debug:
        logger.info("Key frames (in sec): %s", key_frames)
        logger.info("Key frame diffs wrt cut_start: %s", key_frames_diffs_start)
        logger.info("Key frame diffs wrt cut_end: %s", key_frames_diffs_end)
        logger.info("Closest (from right) key frame for start (in sec): %s", closest_key_frame_start)
        logger.info("Closest (from left) key frame for end (in sec): %s", closest_key_frame_end)
        logger.info(f"Original cut start ({cut_start}) adjusted to {cut_start_adjusted}")
        logger.info(f"Original cut end ({cut_end}) adjusted to {cut_end_adjusted}")

    return (cut_start_adjusted, cut_end_adjusted, cuts[2])


def get_cuts(filepaths: Dict[str, Path], logger: logging.Logger) -> Tuple[int, int, int]:
    try:
        with open(filepaths['cuts_txt'], 'r') as f:
            cuts = [line.rstrip().split(',') for line in f if line.strip()]
    except FileNotFoundError:
        logger.critical(f"Problem with reading '{filepaths['cuts_txt']}'")
        sys.exit(1)

    if len(cuts) == 0:
        logger.critical(f"The file '{filepaths['cuts_txt']}' is empty!")
        sys.exit(1)
    elif len(cuts) > 1:
        logger.critical(f"The file '{filepaths['cuts_txt']}' contains more than one line!")
        sys.exit(1)
    else:
        cuts = cuts[0]
        cut_start = int(cuts[0].strip())
        cut_end = int(cuts[1].strip())
        try:
            cut_video_rotate = int(cuts[2].strip())
        except (IndexError, ValueError):
            cut_video_rotate = 0

    logger.info(f"Requested to cut the input video from frame {cut_start} to frame {cut_end} with rotation {cut_video_rotate}.")
    cuts = (cut_start, cut_end, cut_video_rotate)
    return cuts


def perform_sanity_checks(cuts: Tuple[int, int, int], filepaths: Dict[str, Path]) -> None:
    cap = cv2.VideoCapture(str(filepaths["input_video"]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cut_start, cut_end, rotation = cuts
    assert cut_start >= 0, f"Error, 'cut_start' must be non-negative numbers in {filepaths['cuts_txt']}"
    if cut_end == -1:
        cut_end = frame_count - 1
    assert cut_start < cut_end, f"Error, 'cut_start' >= 'cut_end' in {filepaths['cuts_txt']}"
    assert cut_end <= frame_count - 1, (
        f"Error, 'cut_end' in {filepaths['cuts_txt']} is greater than the total number of frames in the input video"
    )
    assert rotation in {0, 90, 180, 270, -90, -180, -270}, (
        f"Error, invalid rotation specified in {filepaths['cuts_txt']}"
    )


def _default_csv_path(video_path: Path) -> Path:
    """Return the companion CSV path for a video, trying both .csv and .CSV.

    Prefers the case that matches the video suffix; falls back to the other
    case so that e.g. a .mp4 video paired with a .CSV log is still found on
    case-sensitive file systems.
    """
    primary = video_path.with_suffix('.csv' if video_path.suffix.islower() else '.CSV')
    fallback = video_path.with_suffix('.CSV' if video_path.suffix.islower() else '.csv')
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary  # neither exists; return primary so the later warning names the expected path


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cut video and flight log according to specified frame ranges.")
    parser.add_argument('input_video', type=Path, help="File path to the input video")
    parser.add_argument('cuts', type=Path, nargs='?', help="Path to the cuts specification file (mutually exclusive with --start/--end)")
    parser.add_argument('--input-csv', '-i', type=Path, default=None, help="Full path to the input CSV flight log (default: same directory and stem as the input video, trying .csv/.CSV in both cases)")
    parser.add_argument('--start', '-s', type=int, help="Cut start frame number (mutually exclusive with <cuts>; requires --end)")
    parser.add_argument('--end', '-e', type=int, help="Cut end frame number, -1 for end of video (mutually exclusive with <cuts>; requires --start)")
    parser.add_argument('--rotate', '-r', type=int, default=None, help="Optional counter-clockwise rotation in degrees (0, ±90, ±180, ±270)")
    parser.add_argument('--output', '-o', type=Path, default=None, help="Full file path for the output video (e.g., /path/to/cut.MP4); companion CSV saved alongside with matching .csv/.CSV extension. Default: <video_dir>/<stem>_cut<ext> with a matching CSV.")
    parser.add_argument('--exact-cut', '-ec', action='store_true', help="Perform exact frame cutting with re-encoding (slower but precise)")
    parser.add_argument('--bitrate', '-b', type=str, default=None, help="Video bitrate, used only with --exact-cut (e.g., '5M', '10000k'; default: auto)")
    parser.add_argument('--debug', '-d', action='store_true', help="Run in debug mode with frame verification and detailed output")
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument('--quiet', '-q', action='store_true', help="Reduce console verbosity to important messages only (default: show INFO-level detail).")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    # CSV suffix mirrors the video extension case for output file naming
    csv_suffix = '.csv' if args.input_video.suffix.islower() else '.CSV'

    # Setup input file paths
    filepaths = {
        'input_video': args.input_video,
        'input_csv': args.input_csv if args.input_csv else _default_csv_path(args.input_video),
    }

    # Determine cuts source: CLI args or cuts file
    cuts: Tuple[int, int, int] = None
    if args.start is not None or args.end is not None:
        # Require both start and end when using CLI-based cuts
        if args.start is None or args.end is None:
            raise SystemExit("When using --start/--end, both must be provided.")
        rotation = args.rotate if args.rotate is not None else 0
        cuts = (int(args.start), int(args.end), int(rotation))
        filepaths['cuts_txt'] = Path('<cli-args>')  # Placeholder for error messages
    else:
        if not args.cuts:
            raise SystemExit("You must provide either a cuts file or --start and --end via CLI.")
        filepaths['cuts_txt'] = args.cuts

    # Setup output file paths
    if args.output:
        filepaths['output_video'] = args.output
        output_csv_suffix = '.csv' if args.output.suffix.islower() else '.CSV'
        filepaths['output_csv'] = args.output.with_suffix(output_csv_suffix)
    else:
        filepaths['output_video'] = args.input_video.with_name(f"{args.input_video.stem}_cut{args.input_video.suffix}")
        filepaths['output_csv'] = args.input_video.with_name(f"{args.input_video.stem}_cut{csv_suffix}")

    # Process cutting - both paths converge here
    process_cutting(filepaths, logger, cuts=cuts, debug=args.debug, exact_cut=args.exact_cut, bitrate=args.bitrate)


if __name__ == '__main__':
    main()
