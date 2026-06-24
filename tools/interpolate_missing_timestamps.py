#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
interpolate_missing_timestamps.py - CSV Timestamp Interpolation Tool

Fills missing (NaN) timestamps in a CSV file by reconstructing them from the video frame
rate and the surrounding known timestamps. The frame rate is either supplied via --fps or
inferred from the spacing of the existing timestamps, so the tool works for any constant
frame-rate video (e.g. 25, 29.97, 30, 60 fps).

Each missing timestamp is computed as an exact multiple of the frame period away from the
nearest known (original) timestamp, rather than accumulated step-by-step, which avoids
cumulative rounding drift across long gaps. For 29.97 fps this reproduces the familiar
33/33/34 ms millisecond cadence.

Usage:
  python tools/interpolate_missing_timestamps.py <input_csv> [options]

Arguments:
  input_csv : Path to the input CSV file containing a 'timestamp' column to interpolate.

Options:
  -h, --help            : Show this help message and exit.
  -f, --fps <float>     : Video frame rate (frames per second) to use for interpolation. If
                          omitted, it is inferred from the spacing of the existing timestamps.
  --backward            : Anchor each gap to the next valid timestamp and fill backward,
                          instead of forward from the previous valid timestamp (default: forward).
  -lp, --log-path <str> : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet           : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Forward-interpolate, inferring the frame rate from the data:
   python tools/interpolate_missing_timestamps.py flight_log.CSV

2. Backward-interpolate with an explicit frame rate:
   python tools/interpolate_missing_timestamps.py flight_log.CSV --fps 30 --backward

Input:
- CSV file with a 'timestamp' column that may contain missing (NaN) values.
- Timestamps in a datetime format parseable by pandas.

Output:
- New CSV file with interpolated timestamps, saved as '<original_name>_interpolated.CSV'.
- Console output showing before/after data samples.

Notes:
- Frame-rate inference uses the first and last valid timestamps and the number of frames
  (rows) between them, giving a robust average fps; pass --fps to override.
- Missing values are anchored to original known timestamps, so rounding does not accumulate
  across long gaps.
- Forward mode cannot fill leading NaNs (no earlier anchor) and backward mode cannot fill
  trailing NaNs; such rows are left as NaN with a warning.
- The output preserves all original columns and only fills the missing 'timestamp' values.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from geotrax.utils.logging_utils import setup_logger

_TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f'


def infer_fps(timestamps: pd.Series, logger: logging.Logger) -> float:
    """Infer the video frame rate from the spacing of the known timestamps.

    Uses the first and last valid timestamps and the frame-index (row) distance between
    them, yielding a robust average frames-per-second estimate that is insensitive to
    individual gaps.
    """
    valid = pd.to_datetime(timestamps, errors='coerce').dropna()
    if len(valid) < 2:
        raise ValueError("Need at least two valid timestamps to infer the frame rate; pass --fps explicitly.")

    span_s = (valid.iloc[-1] - valid.iloc[0]).total_seconds()
    frame_span = int(valid.index[-1] - valid.index[0])
    if span_s <= 0 or frame_span <= 0:
        raise ValueError("Could not infer a positive frame rate from the timestamps; pass --fps explicitly.")

    fps = frame_span / span_s
    logger.info(f"Inferred frame rate from timestamps: {fps:.4f} fps.")
    return fps


def interpolate_timestamps(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Interpolate missing timestamps from the (given or inferred) frame rate."""
    df = pd.read_csv(args.input_csv)
    if 'timestamp' not in df.columns:
        logger.error(f"No 'timestamp' column found in '{args.input_csv.name}'.")
        return
    logger.info("Before interpolation:\n%s\n...\n%s", df.head().to_string(), df.tail().to_string())

    if args.fps is not None:
        if args.fps <= 0:
            logger.error("--fps must be a positive number.")
            return
        fps = args.fps
        logger.info(f"Using specified frame rate: {fps:.4f} fps.")
    else:
        try:
            fps = infer_fps(df['timestamp'], logger)
        except ValueError as e:
            logger.error(str(e))
            return

    period_ms = 1000.0 / fps

    # Anchor to original known timestamps (never to previously filled values) so that
    # millisecond rounding does not accumulate across a gap.
    original = pd.to_datetime(df['timestamp'], errors='coerce')
    is_valid = original.notna().to_numpy()
    n = len(df)
    n_filled = 0

    indices = range(n - 1, -1, -1) if args.backward else range(n)
    anchor_idx: int | None = None
    anchor_time: pd.Timestamp | None = None
    direction = 'backward' if args.backward else 'forward'

    for i in indices:
        if is_valid[i]:
            anchor_idx, anchor_time = i, original.iloc[i]
        elif anchor_time is not None:
            frame_distance = anchor_idx - i if args.backward else i - anchor_idx
            offset = pd.Timedelta(milliseconds=round(frame_distance * period_ms))
            new_ts = anchor_time - offset if args.backward else anchor_time + offset
            df.loc[i, 'timestamp'] = new_ts.strftime(_TIMESTAMP_FMT)[:-3]
            n_filled += 1
        else:
            logger.warning(f"Row {i}: no valid timestamp to anchor {direction} interpolation; left as NaN.")

    logger.info("After interpolation:\n%s\n...\n%s", df.head().to_string(), df.tail().to_string())

    output_csv = args.input_csv.parent / (args.input_csv.stem + '_interpolated' + '.CSV')
    df.to_csv(output_csv, index=False)
    logger.notice(f"Filled {n_filled} timestamp(s); saved to {output_csv}")


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Interpolate missing timestamps in a CSV file')
    parser.add_argument('input_csv', type=Path, help='Input CSV file')
    parser.add_argument('--fps', '-f', type=float, default=None,
                        help='Video frame rate to use for interpolation; inferred from the data if omitted.')
    parser.add_argument('--backward', action='store_true',
                        help='Fill each gap backward from the next valid timestamp (default: forward).')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    interpolate_timestamps(args, logger)


if __name__ == '__main__':
    main()
