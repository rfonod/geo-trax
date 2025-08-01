#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
visualize.py - Video Visualization Tool

This script visualizes tracking results by overlaying them on the original video frames.
It supports various visualization options, including showing bounding boxes, class labels, tracking lines, confidence values,
speed estimates, and lane numbers. The script can also save the annotated video to a file or display it in real-time.
It is designed to work with the results generated by the 'detect_track_stabilize.py' script, which processes video files
and generates tracking data in a specific format. If georeferenced results are available, the script can also visualize speed estimates and lane numbers.

Usage:
  python visualize.py <source> [options]

Arguments:
  source : Path to video source.

Options:
  --help, -h          : Show this help message and exit.
  --cfg, -c <path>    : Path to the main geo-trax configuration file (default: cfg/default.yaml).
  --log-file, -lf <str> : Filename to save detailed logs. Saved in the 'logs' folder (default: None).
  --verbose, -v       : Set print verbosity level to INFO (default: WARNING).

Visualization Options (one required):
  --save, -s          : Save the processing results to a video file.
  --show, -sh         : Visualize results during processing.

Additional Visualization Options:
  --viz-mode, -vm <int> : Set visualization mode for the output video: 0 - original, 
                        1 - stabilized, 2 - reference frame (default: 0).
  --plot-trajectories, -pt : Plot trajectories on the reference frame (default: False).
  --plot-delay, -pd <int> : Delay in frames for plotting trajectories (default: 30).
  --show-conf, -sc    : Show confidence values (default: False).
  --show-lanes, -sl   : Show lane numbers (default: False).
  --show-class-names, -scn : Show class names (default: False).
  --hide-labels, -hl  : Hide labels entirely (default: False).
  --hide-tracks, -ht  : Hide trailing tracking lines (default: False).
  --hide-speed, -hs   : Hide speed values (if available) (default: False).
  --class-filter, -cf <int> [<int> ...] : Exclude specified classes (e.g., -cf 1 2) (default: None).
  --cut-frame-left, -cfl <int> : Skip the first N frames (default: 0).
  --cut-frame-right, -cfr <int> : Stop processing after this frame (default: None).

Examples:
1. Visualize the tracking results on a video using the default settings:
   python visualize.py path/to/video.mp4 --show

2. Save with confidence values and exclude class 0 (default: car):
   python visualize.py path/to/video.mp4 --save --class-filter 0 --show-conf

3. Hide labels, trajectory trails, and exclude multiple classes:
   python visualize.py path/to/video.mp4 --show --hide-labels --hide-tracks --class-filter 1 2

4. Save with class names and cut off the first 100 frames:
   python visualize.py path/to/video.mp4 --save --show-class-names --cut-frame-left 100

5. Visualize tracking results on a stabilized video without speed estimates:
   python visualize.py path/to/video.mp4 --show --viz-mode 1 --hide-speed

Notes:
- The script reads tracking results from a 'video.txt' file located in the 'results' subdirectory generated by the 'detect_track_stabilize.py' script.
- For viz-mode = [1, 2], assumes that transformation matrices are saved in 'video_vid_transf.txt' file also located in the 'results' subdirectory.
- Additional configurations can be set in the main configuration file (default: cfg/default.yaml) and linked files therein.
- Press 'q' during visualization to stop (with --show).
"""


import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import (
    VizColors,
    check_if_results_exist,
    detect_delimiter,
    determine_suffix_and_fourcc,
    load_config_all,
    setup_logger,
)


def visualize_results(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Visualize the tracking results on a video.
    """
    config = load_config_all(args, logger)['main']
    class_names = config['class_names']
    viz_config = config['visualization']
    tracks_txt_filepath, transforms_filepath, tracks_csv_filepath = get_and_verify_filepaths(args, logger)
    tracks, tracks_plotting = read_tracks(tracks_txt_filepath, class_names, args, logger)
    transforms = read_transforms(transforms_filepath, logger)
    speed_lane_data = read_georeferenced_results(tracks_csv_filepath, logger)
    vid_reader, vid_writer, pbar = initialize_streams(args, logger)

    try:
        for frame_num, annotated_frame in process_frames(tracks, tracks_plotting, transforms, speed_lane_data, vid_reader, pbar, class_names, viz_config, args, logger):
            if args.show:
                display_frame(annotated_frame, frame_num, logger)
            if args.save:
                save_frame(vid_writer, annotated_frame, logger)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        finalize_video(vid_reader, vid_writer, pbar, frame_num, args.show, logger)


def process_frames(tracks: pd.DataFrame, tracks_plotting: pd.DataFrame, transforms: dict, speed_lane_data: pd.DataFrame, cap: cv2.VideoCapture, pbar: tqdm, class_names: dict, viz_config: dict, args: argparse.Namespace, logger: logging.Logger):
    """
    Process the video frames and annotate them with tracking results.
    """
    track_history = defaultdict(list)
    frame_num = 0
    viz_phase = args.plot_trajectories  # 0: normal processing phase, 1: trajectory plotting phase
    trajectory_frame = None

    if viz_phase:
        trajectory_frame = plot_trajectories(cap, tracks_plotting, args.cut_frame_left, args.cut_frame_right, viz_config)

    while True:
        if viz_phase:
            # Trajectory plotting phase
            if frame_num < args.plot_delay:
                yield 0, trajectory_frame
                frame_num += 1
                continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                viz_phase = 0
                continue

        # Normal video processing phase
        success, frame = cap.read()
        if not success:
            break

        if frame_num < args.cut_frame_left:
            frame_num += 1
            pbar.update()
            continue
        elif frame_num == args.cut_frame_left:
            ref_frame = frame.copy()
        elif args.cut_frame_right is not None and frame_num >= args.cut_frame_right:
            break

        tracks_frame = tracks[tracks[0] == frame_num]
        speed_lane_frame = speed_lane_data[speed_lane_data['Frame_ID'] == frame_num].drop(columns=['Frame_ID']) if speed_lane_data is not None else None

        if args.viz_mode == 1 and frame_num in transforms:
            h, w = frame.shape[:2]
            frame = cv2.warpPerspective(frame, transforms[frame_num], (w, h))
        elif args.viz_mode == 2:
            frame = ref_frame.copy()

        annotated_frame = annotate_frame(frame, frame_num, tracks_frame, track_history, class_names, speed_lane_frame, viz_config, args, logger)
        yield frame_num, annotated_frame

        if args.cut_frame_right is not None and frame_num >= args.cut_frame_right:
            break

        frame_num += 1
        pbar.update()


def get_and_verify_filepaths(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """
    Get and verify the filepaths for the provided video and tracking and georeferenced results.
    """

    video_exists, video_filepath = check_if_results_exist(args.source, 'video')
    if not video_exists:
        logger.critical(f"Video file '{video_filepath}' not found.")
        sys.exit(1)

    tracks_txt_exist, tracks_txt_filepath = check_if_results_exist(args.source, 'processed')
    if not tracks_txt_exist:
        logger.critical(f"Tracking results file '{tracks_txt_filepath}' not found. Make sure you have run the 'detect_track_stabilze.py' script.")
        sys.exit(1)

    if args.viz_mode == 1:
        transforms_exist, transforms_filepath = check_if_results_exist(args.source, 'video_transformations')
        if not transforms_exist:
            logger.critical(f"Transformation file '{transforms_filepath}' not found. Make sure you have enabled stabilization and run the 'detect_track_stabilze.py' script.")
            sys.exit(1)
    else:
        transforms_filepath = None

    tracks_csv_exist, tracks_csv_filepath = check_if_results_exist(args.source, 'georeferenced')
    if not tracks_csv_exist:
        logger.warning(f"Georeferenced file '{tracks_csv_filepath}' not found. Speed estimates will not be visualized.")
        tracks_csv_filepath = None

    return tracks_txt_filepath, transforms_filepath, tracks_csv_filepath


def read_tracks(tracks_txt_filepath: Path, class_names: dict, args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """
    Read the tracking results from the text file.
    """
    delimiter = detect_delimiter(tracks_txt_filepath)
    tracks = pd.read_csv(tracks_txt_filepath, header=None, delimiter=delimiter)

    if tracks.shape[1] == 10 or tracks.shape[1] == 14:
        # drop the last two columns (vehicle length and width)
        tracks = tracks.drop(tracks.columns[-2:], axis=1)
    if args.plot_trajectories and tracks.shape[1] < 11:
        logger.error(f"No stabilized bounding boxes found in: '{tracks_txt_filepath}'. Disable the trajectory plotting option or re-run the 'detect_track_stabilize.py' script.")
        sys.exit(1)
    else:
        tracks_plotting = tracks[[0, 6, 7, 10]].copy()
        tracks_plotting.columns = list(range(tracks_plotting.shape[1]))
    if args.viz_mode > 0:
        if tracks.shape[1] < 11:
            logger.error(f"No stabilized bounding boxes found in: '{tracks_txt_filepath}'. Choose a different visualization mode or re-run the 'detect_track_stabilize.py' script.")
            sys.exit(1)
        tracks = tracks.drop(tracks.columns[2:6], axis=1)
    elif tracks.shape[1] > 10:
        tracks = tracks.drop(tracks.columns[6:10], axis=1)
    elif tracks.shape[1] < 7:
        logger.error(f"No valid tracking results found in: '{tracks_txt_filepath}'.")
        sys.exit(1)
    tracks.columns = list(range(tracks.shape[1]))

    if len(class_names) < tracks[6].max() + 1:
        logger.error(f"At least {tracks[6].max() + 1} class names must be provided. Current class names defined for the used model are {class_names.values()}.")
        sys.exit(1)

    return tracks, tracks_plotting


def read_transforms(transforms_filepath: Path, logger: logging.Logger) -> dict:
    """
    Read the transformation matrices from the text file.
    """
    if transforms_filepath is None:
        return None

    delimiter = detect_delimiter(transforms_filepath)
    transforms = np.loadtxt(transforms_filepath, delimiter=delimiter)

    if transforms.shape[1] != 10:
        logger.error(f"Not valid transforms in: '{transforms_filepath}'.")
        sys.exit(1)

    frame_nums = transforms[:, 0].astype(int)
    matrices = transforms[:, 1:].reshape((-1, 3, 3))

    if not np.all(np.diff(frame_nums) == 1):
        logger.warning(f"Missing frame ids found in: '{transforms_filepath}'.")

    if not np.all(np.linalg.det(matrices) > 0):
        logger.error(f"Not valid transforms found in: '{transforms_filepath}'.")
        sys.exit(1)

    transforms = dict(zip(frame_nums, matrices))
    return transforms


def read_georeferenced_results(tracks_csv_filepath: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read the georeferenced tracking results from the CSV file.
    """
    if tracks_csv_filepath is None:
        return None

    georeferenced_data = pd.read_csv(tracks_csv_filepath)
    if 'Frame_Number' in georeferenced_data.columns:
        georeferenced_data.rename(columns={'Frame_Number': 'Frame_ID'}, inplace=True)
    elif 'Timestamp' in georeferenced_data.columns:
        georeferenced_data['Timestamp'] = georeferenced_data['Timestamp'].astype('category').cat.codes
        georeferenced_data.rename(columns={'Timestamp': 'Frame_ID'}, inplace=True)
    else:
        logger.error(f"Neither 'Frame_Number' nor 'Timestamp' column found in: '{tracks_csv_filepath}'.")
        sys.exit(1)
    georeferenced_data = georeferenced_data[['Frame_ID', 'Vehicle_ID', 'Vehicle_Speed', 'Lane_Number']]
    return georeferenced_data


def initialize_streams(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """
    Initialize video reader, writer, and progress bar.
    """
    vid_reader = cv2.VideoCapture(str(args.source))
    if not vid_reader.isOpened():
        logger.error(f"Failed to open: '{args.source}'.")
        sys.exit(1)

    frame_count = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.save:
        frame_width = int(vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid_reader.get(cv2.CAP_PROP_FPS) # int() might be required, floats might produce error in MP4 codec

        suffix, fourcc = determine_suffix_and_fourcc()
        vid_file = f"{str(args.source.parent / 'results' / args.source.stem)}_mode_{args.viz_mode}.{suffix}"

        vid_writer = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc(*fourcc), fps, (frame_width, frame_height))
    else:
        vid_writer = None

    pbar = tqdm(total=frame_count, unit='f', leave=True, colour='green', desc=f'{args.source.name} - visualizing @ mode {args.viz_mode}')
    return vid_reader, vid_writer, pbar


def plot_trajectories(cap: cv2.VideoCapture, tracks: pd.DataFrame, cut_frame_left: int, cut_frame_right: int, viz_config: dict) -> np.ndarray:
    """
    Plot the trajectories on the reference frame with an alpha channel.
    """
    success, ref_frame = cap.read()
    if not success:
        logger.error("Failed to read the reference frame.")
        sys.exit(1)
    tracks_plot = tracks[tracks[0] >= cut_frame_left]
    if cut_frame_right is not None:
        tracks_plot = tracks_plot[tracks_plot[0] <= cut_frame_right]

    colors = VizColors()
    line_width = viz_config['line_width']
    overlay = ref_frame.copy()

    for _, row in tracks.iterrows():
        xc_stab, yc_stab, c = row[1:4]
        color = colors(c, True)
        cv2.circle(overlay, (int(xc_stab), int(yc_stab)), 1, color, line_width)

    cv2.addWeighted(overlay, 0.75, ref_frame, 0.25, 0, ref_frame)

    return ref_frame


def annotate_frame(frame: np.ndarray, frame_num: int, tracks_frame: pd.DataFrame, track_history: dict, class_names: dict, speed_lane_frame: pd.DataFrame, viz_config: dict, args: argparse.Namespace, logger: logging.Logger) -> np.ndarray:
    """
    Annotate the frame with the tracking results.
    """
    tail_length = viz_config['tail_length']
    line_width = viz_config['line_width']
    colors = VizColors()
    annotated_frame = frame.copy()
    if tracks_frame.empty:
        logger.warning(f"No detection results for frame {frame_num:05d}")
        return annotated_frame

    ids = tracks_frame.iloc[:, 1].values
    Xcn, Ycn, Wn, Hn = tracks_frame.iloc[:, 2:6].values.T
    classes = tracks_frame.iloc[:, 6].values
    scores = tracks_frame.iloc[:, 7].values if tracks_frame.shape[1] == 8 else [''] * len(ids)

    for track_id, xcn, ycn, wn, hn, c, s in zip(ids, Xcn, Ycn, Wn, Hn, classes, scores):
        if args.class_filter and c in args.class_filter:
            continue

        speed, lane = None, None
        if speed_lane_frame is not None:
            vehicle_data = speed_lane_frame[speed_lane_frame['Vehicle_ID'] == track_id]
            if not vehicle_data.empty:
                speed = vehicle_data['Vehicle_Speed'].values[0]
                lane = vehicle_data['Lane_Number'].values[0]
                speed = int(speed) if not np.isnan(speed) else None
                lane = int(lane) if not np.isnan(lane) else None

        color = colors(c, True)
        x1n, y1n = int(xcn - wn / 2), int(ycn - hn / 2)
        x2n, y2n = int(xcn + wn / 2), int(ycn + hn / 2)
        cv2.rectangle(annotated_frame, (x1n, y1n), (x2n, y2n), color, line_width, lineType=cv2.LINE_AA)

        if not args.hide_labels:
            label_parts = []
            if track_id not in {None, -1}:
                label_parts.append(f'id:{track_id}')
            if args.show_class_names:
                label_parts.append(class_names[c])
            if not args.hide_speed and speed is not None:
                label_parts.append(f'{speed} km/h')
            if args.show_lanes and lane is not None:
                label_parts.append(f'L{lane}')
            if args.show_conf and s != '':
                label_parts.append(f'{s:.2f}')
            label = ' '.join(label_parts)

            tf = max(line_width - 1, 1)
            twn, thn = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=tf)[0]
            outside = y1n - thn >= 3
            xt2n = x1n + twn
            yt2n = y1n - thn - 3 if outside else y1n + thn + 3
            cv2.rectangle(annotated_frame, (x1n, y1n), (xt2n, yt2n), color, -1, cv2.LINE_AA)
            cv2.putText(annotated_frame, label, (x1n, y1n - 2 if outside else y1n + thn + 2),
                        0, line_width / 3, colors.txt_color, thickness=tf, lineType=cv2.LINE_AA)

        if not args.hide_tracks:
            track = track_history[track_id]
            track.append((float(xcn), float(ycn)))
            if len(track) > tail_length:
                track.pop(0)
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            track_len = len(points)
            for i, point in enumerate(points):
                cv2.circle(annotated_frame, tuple(point[0]), int(1 + 8 * (i + 1) / track_len), color, line_width)
    return annotated_frame


def display_frame(annotated_frame: np.ndarray, frame_num: int, logger: logging.Logger) -> None:
    """
    Display the annotated frame.
    """
    cv2.putText(annotated_frame, f'Frame {frame_num:05d}', org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=2, color=(0, 255, 100))
    cv2.putText(annotated_frame, '(Press <q> to stop)', org=(375, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=2, color=(0, 125, 255))
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        logger.warning('Visualization interrupted by user.')
        raise KeyboardInterrupt


def save_frame(vid_writer: cv2.VideoWriter, annotated_frame: np.ndarray, logger: logging.Logger) -> None:
    """
    Save the annotated frame to the video file.
    """
    try:
        vid_writer.write(annotated_frame)
    except cv2.error as e:
        logger.error(f'Failed to write frame to video: {e}')
        sys.exit(1)


def finalize_video(vid_reader: cv2.VideoCapture, vid_writer: cv2.VideoWriter, pbar: tqdm, frame_num: int, show: bool, logger: logging.Logger) -> None:
    """
    Finalize the video processing.
    """
    vid_reader.release()
    if vid_writer is not None:
        vid_writer.release()
        logger.info('Visualization video saved successfully')
    if show:
        cv2.destroyAllWindows()
    pbar.total = frame_num + 1
    pbar.n = frame_num + 1
    pbar.refresh()
    pbar.close()


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Tracking Results Visualization')

    # Required arguments
    parser.add_argument('source', type=Path, help='Path to video source')

    # Optional arguments
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING)')

    # Visualization arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--save', '-s', action='store_true', help='Save the processing results to a video file')
    group.add_argument('--show', '-sh', action='store_true', help='Visualize results during processing')
    parser.add_argument('--viz-mode', '-vm', type=int, default=0, choices=[0, 1, 2], help='Set visualization mode for the output video: 0 - original, 1 - stabilized, 2 - reference frame')
    parser.add_argument('--plot-trajectories', '-pt', action='store_true', help='Plot trajectories on the reference frame')
    parser.add_argument('--plot-delay', '-pd', type=int, default=30, help='Delay in frames for plotting trajectories')
    parser.add_argument('--show-conf', '-sc', action='store_true', help='Show confidence values')
    parser.add_argument('--show-lanes', '-sl', action='store_true', help='Show lane numbers')
    parser.add_argument('--show-class-names', '-scn', action='store_true', help='Show class names')
    parser.add_argument('--hide-labels', '-hl', action='store_true', help='Hide labels entirely')
    parser.add_argument('--hide-tracks', '-ht', action='store_true', help='Hide trailing tracking lines')
    parser.add_argument('--hide-speed', '-hs', action='store_true', help='Hide speed values (if available)')
    parser.add_argument('--class-filter', '-cf', type=int, nargs='+', help='Exclude specified classes (e.g., -cf 1 2)')
    parser.add_argument('--cut-frame-left', '-cfl', type=int, default=0, help='Skip the first N frames. Default: 0.')
    parser.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Stop processing after this frame. Default: None.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    visualize_results(args, logger)
