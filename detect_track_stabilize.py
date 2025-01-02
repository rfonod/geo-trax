#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
detect_track_stabilize.py - Performs video processing for vehicle trajectory extraction in image coordinates.

This script is integral to the Geo-trax pipeline, focusing on the extraction of vehicle trajectories in
image coordinates from drone-derived video footage. Designed for quasi-stationary drone operations providing
a bird's-eye view, it is ideal for tasks such as intersection monitoring and similar applications. It leverages
a pre-trained YOLOv8 model to detect four vehicle classes, applies the selected tracking algorithm to maintain
consistent vehicle identification across frames, and employs a custom video stabilization routine to correct
for drone movement and ensure accurate vehicle trajectories. The script also estimates vehicle dimensions based
on bounding boxes and azimuth data. The extracted trajectories are saved to a text file, along with additional
metadata.

Usage:
    python detect_track_stabilize.py <source> [options]

Arguments:
    source                    : Path to the video file (e.g., path/to/video/video.mp4).

Options:
    -h, --help                : Show this help message and exit.
    -c, --cfg <path>          : Path to the main geo-trax configuration file (default: cfg/default.yaml).
    -lf, --log-file <str>     : Custom filename to save detailed logs. Saved in the 'logs' folder.
    -v, --verbose             : Set print verbosity level to INFO (default: WARNING).

    --classes <int>           : Class IDs to extract (e.g., --classes 0 1 2). Defaults to cfg -> cfg_ultralytics -> classes.
    --cut-frame-left <int>    : Skip the first N frames. Default: 0.
    --cut-frame-right <int>   : Stop processing after this frame. Default: None.

Examples:
  1. Process a video with default settings:
        python detect_track_stabilize.py path/to/video.mp4

  2. Use a custom config and consider only the first two vehicle classes:
        python detect_track_stabilize.py path/to/video.mp4 --cfg cfg/custom.yaml --classes 0 1

  3. Skip the first 100 frames and stop processing after frame 500:
        python detect_track_stabilize.py path/to/video.mp4 --cut-frame-left 100 --cut-frame-right 500

Notes:
  - Additional configurations can be set in the main configuration file (default: cfg/default.yaml) and linked config files therein.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import yaml
from stabilo import Stabilizer
from tqdm import tqdm
from ultralytics import RTDETR, YOLO
from ultralytics.utils.checks import check_yolo
from ultralytics.utils.files import increment_path

from utils.utils import (
    check_if_results_exist,
    convert_to_serializable,
    get_video_dimensions,
    load_config_all,
    setup_logger,
)


def detect_track_stabilize(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process video based on provided arguments.
    """
    config = load_config_all(args, logger)
    model = load_detector(config['ultralytics'], logger)
    tracks, transforms = track_with_model(model, config, logger)
    tracks = postprocess_tracks(tracks, config, logger)
    save_results(tracks, transforms, config, logger)


def track_with_model(model: Union[YOLO, RTDETR], config: Dict, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track vehicles in the video using the provided model.
    """
    reader, pbar = initialize_streams(config['main'], config['ultralytics']['imgsz'], logger)
    stabilizer = Stabilizer(**config['stabilo'])

    frame_num, yolo_time, stab_time = 0, [], []
    frame_arr, track_id, bbox, bbox_stab, class_id, conf, transforms = [], [], [], [], [], [], []

    try:
        while reader.isOpened():
            success, frame = reader.read()
            if frame_num < config['main']['args'].cut_frame_left:
                frame_num += 1
                pbar.update()
                continue

            if success:
                results = model.track(frame, **config['ultralytics'], persist=True)
                boxes = results[0].boxes
                speed = results[0].speed
                yolo_time.append(sum(speed.values()))

                class_freq = {c: 0 for c in config['ultralytics']['classes']}
                if len(boxes) > 0:
                    frame_arr.append(np.full((len(boxes), 1), frame_num, dtype=np.uint32))
                    if boxes.id is not None:
                        track_ids = boxes.id.detach().numpy(force=True).astype(np.uint16).reshape(-1, 1)
                    else:
                        track_ids = np.full((len(boxes), 1), -1)
                    track_id.append(track_ids)
                    bbox.append(boxes.xywh.detach().numpy(force=True).astype(np.float32))
                    class_id.append(boxes.cls.detach().numpy(force=True).astype(np.uint8).reshape(-1, 1))
                    conf.append(boxes.conf.detach().numpy(force=True).astype(np.float32).reshape(-1, 1))

                    if config['main']['args'].verbose:
                        unique, counts = np.unique(class_id[-1], return_counts=True)
                        class_freq.update(dict(zip(unique, counts)))

                if config['main']['stabilize']:
                    start_time = time.time()
                    if frame_num == config['main']['args'].cut_frame_left:
                        stabilizer.set_ref_frame(frame, bbox[-1] if len(boxes) > 0 else None)
                        if len(boxes) > 0:
                            bbox_stab.append(bbox[-1])
                    else:
                        stabilizer.stabilize(frame, bbox[-1] if len(boxes) > 0 else None)
                        if len(boxes) > 0:
                            bbox_stab.append(stabilizer.transform_cur_boxes())
                        transf_matrix = stabilizer.get_cur_trans_matrix()
                        if transf_matrix is not None:
                            transf_matrix = transf_matrix.flatten().reshape(1, -1)
                            transforms.append(np.hstack((np.array([[frame_num]]), transf_matrix)))
                    stab_time.append(1000 * (time.time() - start_time))
            else:
                break

            update_progress_bar(pbar, class_freq, speed, stab_time, config['main'])
            if config['main']['args'].cut_frame_right is not None and frame_num >= config['main']['args'].cut_frame_right:
                break

            frame_num += 1
            pbar.update()
    except Exception as e:
        logger.error(f"Error processing: '{config['main']['args'].source}' due to: {e}")
        return np.array([[]]), np.array([[]])
    else:
        pbar.total = frame_num
        pbar.refresh()
        logger.info(f"Average YOLOv8 (preprocess + inference + postprocess) time: {sum(yolo_time) / len(yolo_time):5.1f}ms.")
        logger.info(f"Average stabilization time: {sum(stab_time) / len(stab_time):5.1f}ms") if stab_time else None
        logger.info(f"Average pipeline time: {1000 / ((sum(yolo_time) + sum(stab_time)) / (1 + frame_num)):4.1f}fps.")
    finally:
        reader.release()
        pbar.close()

    tracks, transforms = aggregate_results(frame_arr, track_id, bbox, bbox_stab, class_id, conf, transforms, logger)
    return tracks, transforms


def load_detector(config: Dict, logger: logging.Logger) -> Union[YOLO, RTDETR]:
    """
    Load the detection model based on configuration.
    """
    try:
        model = YOLO(model=config['model'], task=config['task'])
        yaml_file = getattr(model.model, 'yaml_file', '') or getattr(model.model, 'yaml', {}).get('yaml_file', '')
        if 'rtdetr' in yaml_file:
            model = RTDETR(config['model'])
    except KeyError as e:
        logger.critical(f"Configuration key error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading the YOLOv8 model: {e}")
        sys.exit(1)
    else:
        logger.info(f"Detection model '{config['model']}' loaded successfully.")

    check_yolo(device=config['device'])
    return model


def initialize_streams(config: Dict, imgsz: int, logger: logging.Logger) -> Tuple[cv2.VideoCapture, tqdm]:
    """
    Initialize video reader and progress bar.
    """
    video_exists, video_filepath = check_if_results_exist(config['args'].source, 'video')
    if not video_exists:
        logger.critical(f"Video file '{video_filepath}' not found.")
        sys.exit(1)

    reader = cv2.VideoCapture(str(video_filepath))
    if not reader.isOpened():
        logger.error(f"Failed to open: '{video_filepath}'.")
        sys.exit(1)

    pbar = tqdm(total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), unit='f', leave=True, colour='yellow',
                desc=f'{video_filepath.name} - {"" if config["args"].verbose else "processing"} @ {imgsz}px ')
    return reader, pbar


def update_progress_bar(pbar: tqdm, class_freq: Dict, speed: Dict, stab_time: list, config: Dict) -> None:
    """
    Update the progress bar with additional information.
    """
    if config['args'].verbose:
        postfix_txt = {config['class_names'][c][:5]: class_freq[c] for c in class_freq}
        postfix_txt['pre-proc'] = f'{speed["preprocess"]:.1f}ms'
        postfix_txt['infer'] = f'{speed["inference"]:.1f}ms'
        postfix_txt['post-proc'] = f'{speed["postprocess"]:.1f}ms'
        postfix_txt['stab'] = f'{stab_time[-1]:.1f}ms' if stab_time else 'N/A'
        pbar.set_postfix(postfix_txt)


def aggregate_results(frame_arr: list, track_id: list, bbox: list, bbox_stab: list, class_id: list, conf: list, transforms: list, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate the results from all frames.
    """
    try:
        frame_arr = np.concatenate(frame_arr, axis=0) if frame_arr else np.array([[]])
        track_id = np.concatenate(track_id, axis=0) if track_id else np.array([[]])
        bbox = np.concatenate(bbox, axis=0) if bbox else np.array([[]])
        bbox_stab = np.concatenate(bbox_stab, axis=0) if bbox_stab else np.array([[]]).reshape(len(track_id), 0)
        class_id = np.concatenate(class_id, axis=0) if class_id else np.array([[]])
        conf = np.concatenate(conf, axis=0) if conf else np.array([[]])

        tracks = np.concatenate([frame_arr, track_id, bbox, bbox_stab, class_id, conf], axis=1, dtype=np.float32)
        tracks = tracks[tracks[:, 1] != -1]
        transforms = np.concatenate(transforms, axis=0) if transforms else np.array([[]])
    except Exception as e:
        logger.error(f'Error aggregating results: {e}')
        tracks = np.array([[]])
        transforms = np.array([[]])
    return tracks, transforms


def postprocess_tracks(tracks: np.ndarray, config: Dict, logger: logging.Logger) -> np.ndarray:
    """
    Postprocess the extracted tracks.
    """
    tracks = remove_short_tracks(tracks, logger)
    tracks = calculate_unique_classes(tracks, config['ultralytics'])
    tracks = estimate_vehicle_dimensions(tracks, config['main'])
    return tracks


def remove_short_tracks(tracks: np.ndarray, logger: logging.Logger, min_length: int = 3) -> np.ndarray:
    """
    Remove tracks with trajectory length shorter than specified.
    """
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    count = 0
    for track_id in unique_ids:
        mask = tracks[:, 1] == track_id
        if sum(mask) < min_length:
            tracks = tracks[~mask]
            count += 1
    if count > 0:
        logger.info(f'{count} short tracks removed.')
    return tracks


def calculate_unique_classes(tracks: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calculate the unique class labels for each track.
    """
    id2weighted_class_freq = {}
    if tracks.size != 0:
        for track in tracks:
            track_id, c, conf_score = int(track[1]), int(track[-2]), track[-1]
            if track_id not in id2weighted_class_freq:
                id2weighted_class_freq[track_id] = [0] * (max(config['classes']) + 1)
            id2weighted_class_freq[track_id][c] += conf_score

        id2class_max = {}
        for track_id in id2weighted_class_freq:
            id2class_max[track_id] = id2weighted_class_freq[track_id].index(max(id2weighted_class_freq[track_id]))

        for i, track in enumerate(tracks):
            track_id = int(track[1])
            tracks[i, -2] = id2class_max[track_id]

    return tracks


def estimate_vehicle_dimensions(tracks: np.ndarray, config: Dict) -> np.ndarray:
    """
    Estimate vehicle dimensions based on bounding boxes and azimuths.
    """

    w_I, h_I = get_video_dimensions(config['args'].source)

    # Step 1: visibility filtering
    eps = config['dimension_estimation']['eps']
    mask = (tracks[:, 2] - tracks[:, 4]/2 > eps) & (tracks[:, 3] - tracks[:, 5]/2 > eps)
    mask &= (tracks[:, 2] + tracks[:, 4]/2 < w_I - 1 - eps) & (tracks[:, 3] + tracks[:, 5]/2 < h_I - 1 - eps)
    valid_tracks = tracks[mask]

    # Step 2: initial dimensions computation
    unique_ids = np.unique(valid_tracks[:, 1]).astype(int)
    id2lengths, id2widths = {track_id: [] for track_id in unique_ids}, {track_id: [] for track_id in unique_ids}
    id2x_centers, id2y_centers = {track_id: [] for track_id in unique_ids}, {track_id: [] for track_id in unique_ids}
    id2class = {}

    if valid_tracks.shape[1] > 8:
        idx_x, idx_y, idx_c = 6, 7, 10 # stabilized tracks available
    else:
        idx_x, idx_y, idx_c = 2, 3, 6  # only unstabilzed tracks available

    for track in valid_tracks:
        track_id = int(track[1])
        w, h = track[4], track[5]
        x_center, y_center = track[idx_x], track[idx_y]
        v_class = int(track[idx_c])
        id2lengths[track_id].append(max(w, h))
        id2widths[track_id].append(min(w, h))
        id2x_centers[track_id].append(x_center)
        id2y_centers[track_id].append(y_center)
        if track_id not in id2class:
            id2class[track_id] = v_class

    # Step 3: azimuth-based filtering
    r0 = config['dimension_estimation']['r0']
    gsd = config['dimension_estimation']['gsd']
    theta_bar = config['dimension_estimation']['theta_bar']
    theta_bar_rad = np.deg2rad(theta_bar)
    tau_c = config['dimension_estimation']['tau_c']
    radius_threshold = r0 / gsd

    for track_id in unique_ids:
        lengths, widths = id2lengths[track_id], id2widths[track_id]
        x_centers, y_centers = id2x_centers[track_id], id2y_centers[track_id]
        azimuth = None
        idx_prev = 0
        x_c_prev, y_c_prev = x_centers[idx_prev], y_centers[idx_prev]
        mask = np.zeros(len(lengths), dtype=bool)
        for idx, point in enumerate(zip(x_centers[1:], y_centers[1:]), start=1):
            x_c, y_c = point
            distance = np.sqrt((x_c - x_c_prev) ** 2 + (y_c - y_c_prev) ** 2)
            if distance >= radius_threshold:
                azimuth = np.arctan2(-(y_c - y_c_prev), x_c - x_c_prev)
                x_c_prev, y_c_prev = x_c, y_c
                if np.any(np.abs(azimuth - np.array([0, np.pi / 2, np.pi, -np.pi / 2, -np.pi])) <= theta_bar_rad):
                    mask[idx_prev:idx] = True
                idx_prev = idx

        lengths, widths = np.array(lengths), np.array(widths)
        if azimuth is None:
            mask = lengths >= widths * tau_c.get(id2class[track_id], tau_c[-1])  # ratio l/w > threshold
        id2lengths[track_id] = list(lengths[mask])
        id2widths[track_id] = list(widths[mask])

    # Step 4: final dimension computation
    id2length, id2width = {}, {}
    for track_id in unique_ids:
        id2length[track_id] = np.percentile(id2lengths[track_id], 25) if len(id2lengths[track_id]) > 0 else np.nan
        id2width[track_id] = np.percentile(id2widths[track_id], 25) if len(id2widths[track_id]) > 0 else np.nan

    # Finally: append v_length and v_width to each track, per id, as two last columns
    tracks = np.append(tracks, np.zeros((len(tracks), 2)), axis=1)
    for i, track in enumerate(tracks):
        track_id = int(track[1])
        tracks[i, -2] = id2length.get(track_id, np.nan)
        tracks[i, -1] = id2width.get(track_id, np.nan)

    return tracks


def save_results(tracks: np.ndarray, transforms: np.ndarray, config: Dict, logger: logging.Logger) -> None:
    """
    Save the detection, tracking, and stabilization results to files.
    """
    save_dir = increment_path(config['main']['args'].source.parent.resolve() / 'results', exist_ok=True, mkdir=True)
    tracks_txt_file = save_dir / f'{config["main"]["args"].source.stem}.txt'
    transf_txt_file = save_dir / f'{config["main"]["args"].source.stem}_vid_transf.txt'
    info_yaml_file = config['main']['args'].source.with_suffix('.yaml')

    try:
        if tracks.size != 0:
            np.savetxt(tracks_txt_file, tracks, fmt='%g', delimiter=',')
            logger.info(f"Tracking results saved to: '{tracks_txt_file.resolve()}'")
    except Exception as e:
        logger.error(f"Failed to save the tracking results to: '{tracks_txt_file.resolve()}' due to: {e}")

    try:
        if transforms.size != 0 and config['main']['save_stab']:
            frame_nums = transforms[:, 0].astype(int)
            matrices = transforms[:, 1:].reshape((-1, 3, 3))
            if not np.all(np.diff(frame_nums) == 1):
                logger.warning(f"Missing frame ids found in: '{transf_txt_file}'.")
            if not np.all(np.linalg.det(matrices) > 0):
                logger.warning(f"Invalid transforms found in: '{transf_txt_file}'.")
            np.savetxt(transf_txt_file, transforms, fmt='%.16g', delimiter=',')
    except Exception as e:
        logger.error(f"Failed to save the video stabilization results to: '{transf_txt_file.resolve()}' due to: {e}")
    else:
        logger.info(f"Video stabilization results saved to: '{transf_txt_file.resolve()}'")

    serializable_config = convert_to_serializable(config)
    with open(info_yaml_file, 'w') as f:
        yaml.dump(serializable_config, f, default_flow_style=False)
    logger.info(f"Video info and configs saved to: '{info_yaml_file.resolve()}'")


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Vehicle Detection, Tracking, and Stabilization Pipeline')

    # Required arguments
    parser.add_argument('source', type=Path, help='Path to the video file (e.g., path/to/video/video.mp4)')

    # Optional arguments
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING)')

    # Additional arguments
    parser.add_argument('--classes', '-cls', nargs='+', type=int, help='Class IDs to extract (e.g., --classes 0 1 2). Defaults to cfg -> cfg_ultralytics -> classes.')
    parser.add_argument('--cut-frame-left', '-cfl', type=int, default=0, help='Skip the first N frames. Default: 0.')
    parser.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Stop processing after this frame. Default: None.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    detect_track_stabilize(args, logger)
