#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
process_video.py - Performs video processing for vehicle trajectory extraction in image coordinates.

This script is integral to the Geo-trax pipeline, focusing on the extraction of vehicle trajectories in 
image coordinates from drone-derived video footage. Designed for quasi-stationary drone operations providing 
a bird's-eye view, it is ideal for tasks such as intersection monitoring and similar applications. It leverages 
a pre-trained YOLOv8 model to detect four vehicle classes, applies the selected tracking algorithm to maintain 
consistent vehicle identification across frames, and employs a custom video stabilization routine to correct 
for drone movement and ensure accurate vehicle trajectories. The script also estimates vehicle dimensions based 
on bounding boxes and azimuth data. The extracted trajectories are saved to a text file, along with additional 
metadata.

Usage:
  python process_video.py <video_source> [options]

Arguments:
  video_source : Path to the video file.

Options:
  --cfg, -c CFG            : Path to the main configuration file [default: 'cfg/default.yaml'].
  --classes CLASSES        : Specify classes to track (e.g., 0 1 2) [default: as per cfg file].
  --log-file, -lf LOG_FILE : Specify log file name for detailed logs [default: None].
  --verbose, -v            : Enable detailed logging to console [default: False].
  --interpolate, -i        : Interpolate missing frames in tracks (not yet implemented).
  --cut-frame-left, -cfl CUT_FRAME_LEFT : Start processing from this frame number [default: 0].
  --cut-frame-right, -cfr CUT_FRAME_RIGHT : Stop processing at this frame number [default: None].  

Examples:
  1. Process a video with default settings:
     python process_video.py path/to/video.mp4

  2. Use a custom config and enable verbose logging into a log file:
     python process_video.py path/to/video.mp4 -c cfg/custom.yaml -v -lf video.log

  3. Specifying vehicle classes and video segment:
     python process_video.py path/to/video.mp4 --classes 0 2 --cut-frame-right 800

Notes:
  - This script is in a pre-release state and intended for experimental use only.
  - Features like 'interpolate' are planned but not yet implemented.
  - Refer to 'cfg/default.yaml' for default configuration parameters.
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Union

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_yolo

from stabilo import Stabilizer
from utils import setup_logger, load_config_all, convert_to_serializable

LOGGER_PREFIX = f'[{Path(__file__).name}]'

def process_video(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process video based on provided arguments.
    """
    config = load_config_all(args, logger, LOGGER_PREFIX)
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
    frame_arr, id, bbox, bbox_stab, cls, conf, transforms = [], [], [], [], [], [], []

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
                    id.append(boxes.id.detach().numpy(force=True).astype(np.uint16).reshape(-1, 1) if boxes.id is not None else np.full((len(boxes), 1), -1))
                    bbox.append(boxes.xywh.detach().numpy(force=True).astype(np.float32))
                    cls.append(boxes.cls.detach().numpy(force=True).astype(np.uint8).reshape(-1, 1))
                    conf.append(boxes.conf.detach().numpy(force=True).astype(np.float32).reshape(-1, 1))

                    if config['main']['args'].verbose:
                        unique, counts = np.unique(cls[-1], return_counts=True)
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
        logger.error(f'{LOGGER_PREFIX} Error processing {config["main"]["args"].source}: {e}')
        return np.array([[]]), np.array([[]])
    else:
        pbar.total = frame_num
        pbar.refresh()
        logger.info(f"{LOGGER_PREFIX} {sum(yolo_time) / len(yolo_time):5.1f}ms - average YOLOv8 (preprocess + inference + postprocess) time.") 
        logger.info(f"{LOGGER_PREFIX} {sum(stab_time) / len(stab_time):5.1f}ms - average stabilization time.") if stab_time else None
        logger.info(f"{LOGGER_PREFIX} {1000 / ((sum(yolo_time) + sum(stab_time)) / (1 + frame_num)):4.1f}fps - pipeline's average frames per second (fps).") 
    finally:
        reader.release()
        pbar.close()

    tracks, transforms = aggregate_results(frame_arr, id, bbox, bbox_stab, cls, conf, transforms, logger)
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
        logger.critical(f"{LOGGER_PREFIX} Configuration key error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"{LOGGER_PREFIX} Error loading the YOLOv8 model: {e}")
        sys.exit(1)
    else:
        logger.info(f"{LOGGER_PREFIX} Detection model '{config['model']}' loaded successfully.")
    
    check_yolo(device=config['device'])
    return model

def initialize_streams(config: Dict, imgsz: int, logger: logging.Logger) -> Tuple[cv2.VideoCapture, tqdm]:
    """
    Initialize video reader and progress bar.
    """
    reader = cv2.VideoCapture(str(config['args'].source))
    if not reader.isOpened():
        logger.error(f"{LOGGER_PREFIX} Failed to open {config['args'].source}.")
        sys.exit(1)

    config['video'] = {
        'frame_count': int(reader.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': reader.get(cv2.CAP_PROP_FPS),
        'w_I': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'h_I': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    pbar = tqdm(total=config['video']['frame_count'], unit='f', leave=True, colour='yellow', 
                desc=f'{config["args"].source.name} - {"" if config["args"].verbose else "processing"} @ {imgsz}px ')
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

def aggregate_results(frame_arr: list, id: list, bbox: list, bbox_stab: list, cls: list, conf: list, transforms: list, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate the results from all frames.
    """
    try:
        frame_arr = np.concatenate(frame_arr, axis=0) if frame_arr else np.array([[]])
        id = np.concatenate(id, axis=0) if id else np.array([[]])
        bbox = np.concatenate(bbox, axis=0) if bbox else np.array([[]])
        bbox_stab = np.concatenate(bbox_stab, axis=0) if bbox_stab else np.array([[]]).reshape(len(id), 0)
        cls = np.concatenate(cls, axis=0) if cls else np.array([[]])
        conf = np.concatenate(conf, axis=0) if conf else np.array([[]])
        
        tracks = np.concatenate([frame_arr, id, bbox, bbox_stab, cls, conf], axis=1, dtype=np.float32)
        tracks = tracks[tracks[:, 1] != -1]
        transforms = np.concatenate(transforms, axis=0) if transforms else np.array([[]])
    except Exception as e:
        logger.error(f'{LOGGER_PREFIX} Error aggregating results: {e}')
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
    if config['main']['args'].interpolate:
        logger.warning(f'{LOGGER_PREFIX} Track interpolation is not yet implemented.')
    return tracks    

def remove_short_tracks(tracks: np.ndarray, logger: logging.Logger, min_length: int = 3) -> np.ndarray:
    """
    Remove tracks with trajectory length shorter than specified.
    """
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    count = 0
    for id in unique_ids:
        mask = tracks[:, 1] == id
        if sum(mask) < min_length:
            tracks = tracks[~mask]
            count += 1
    if count > 0:
        logger.info(f'{LOGGER_PREFIX} {count} short tracks removed.')
    return tracks

def calculate_unique_classes(tracks: np.ndarray, config: Dict) -> np.ndarray:
    """
    Calculate the unique class labels for each track.
    """
    id2weighted_class_freq = {}
    if tracks.size != 0:
        for track in tracks:
            id, c, conf_score = int(track[1]), int(track[-2]), track[-1]
            if id not in id2weighted_class_freq:
                id2weighted_class_freq[id] = [0] * (max(config['classes']) + 1)
            id2weighted_class_freq[id][c] += conf_score

        id2class_max = {}
        for id in id2weighted_class_freq:
            id2class_max[id] = id2weighted_class_freq[id].index(max(id2weighted_class_freq[id]))

        for i, track in enumerate(tracks):
            id = int(track[1])
            tracks[i, -2] = id2class_max[id]

    return tracks

def estimate_vehicle_dimensions(tracks: np.ndarray, config: Dict) -> np.ndarray:
    """
    Estimate vehicle dimensions based on bounding boxes and azimuths.
    """
    w_I, h_I = config['video']['w_I'], config['video']['h_I']

    # Step 1: visibility filtering
    eps = config['eps']
    mask = (tracks[:, 2] - tracks[:, 4]/2 > eps) & (tracks[:, 3] - tracks[:, 5]/2 > eps)
    mask &= (tracks[:, 2] + tracks[:, 4]/2 < w_I - 1 - eps) & (tracks[:, 3] + tracks[:, 5]/2 < h_I - 1 - eps)
    valid_tracks = tracks[mask]
    
    # Step 2: initial dimensions computation
    unique_ids = np.unique(valid_tracks[:, 1]).astype(int)
    id2lengths, id2widths = {id: [] for id in unique_ids}, {id: [] for id in unique_ids}
    id2x_centers, id2y_centers = {id: [] for id in unique_ids}, {id: [] for id in unique_ids}
    id2class = {}

    if valid_tracks.shape[1] > 8:
        idx_x, idx_y, idx_c = 6, 7, 10 # stabilized tracks available
    else:
        idx_x, idx_y, idx_c = 2, 3, 6  # only unstabilzed tracks available

    for track in valid_tracks:
        id = int(track[1])
        w, h = track[4], track[5]
        x_center, y_center = track[idx_x], track[idx_y]
        v_class = int(track[idx_c])
        id2lengths[id].append(max(w, h))
        id2widths[id].append(min(w, h))
        id2x_centers[id].append(x_center)
        id2y_centers[id].append(y_center)
        if id not in id2class:
            id2class[id] = v_class

    # Step 3: azimuth-based filtering
    radius, theta_bar_rad, tau_c = config['r0'] / config['gsd'], np.deg2rad(config['theta_bar']), config['tau_c']
    for id in unique_ids:
        lengths, widths = id2lengths[id], id2widths[id]
        x_centers, y_centers = id2x_centers[id], id2y_centers[id]
        azimuth = None
        idx_prev = 0
        x_c_prev, y_c_prev = x_centers[idx_prev], y_centers[idx_prev]
        mask = np.zeros(len(lengths), dtype=bool)
        for idx, point in enumerate(zip(x_centers[1:], y_centers[1:]), start=1):
            x_c, y_c = point
            distance = np.sqrt((x_c - x_c_prev) ** 2 + (y_c - y_c_prev) ** 2)
            if distance >= radius:
                azimuth = np.arctan2(-(y_c - y_c_prev), x_c - x_c_prev)
                x_c_prev, y_c_prev = x_c, y_c
                if np.any(np.abs(azimuth - np.array([0, np.pi / 2, np.pi, -np.pi / 2, -np.pi])) <= theta_bar_rad):
                    mask[idx_prev:idx] = True
                idx_prev = idx

        lengths, widths = np.array(lengths), np.array(widths)
        if azimuth is None:
            mask = lengths >= widths * tau_c.get(id2class[id], tau_c[-1])  # ratio l/w > threshold 
        id2lengths[id] = list(lengths[mask])
        id2widths[id] = list(widths[mask])
            
    # Step 4: final dimension computation
    id2length, id2width = {}, {}
    for id in unique_ids:
        id2length[id] = np.percentile(id2lengths[id], 25) if len(id2lengths[id]) > 0 else np.nan
        id2width[id] = np.percentile(id2widths[id], 25) if len(id2widths[id]) > 0 else np.nan

    # Finally: append v_length and v_width to each track, per id, as two last columns
    tracks = np.append(tracks, np.zeros((len(tracks), 2)), axis=1)
    for i, track in enumerate(tracks):
        id = int(track[1])
        tracks[i, -2] = id2length.get(id, np.nan)
        tracks[i, -1] = id2width.get(id, np.nan)

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
            logger.info(f'{LOGGER_PREFIX} Tracking results saved to {tracks_txt_file.resolve()}')
    except Exception as e:
        logger.error(f'{LOGGER_PREFIX} Failed to save the tracking results to {tracks_txt_file.resolve()}: {e}')
        
    try:
        if transforms.size != 0 and config['main']['save_stab']:
            frame_nums = transforms[:, 0].astype(int)
            matrices = transforms[:, 1:].reshape((-1, 3, 3))
            if not np.all(np.diff(frame_nums) == 1):
                logger.warning(f'{LOGGER_PREFIX} Missing frame ids found in {transf_txt_file}.')
            if not np.all(np.linalg.det(matrices) > 0):
                logger.warning(f'{LOGGER_PREFIX} Invalid transforms found in {transf_txt_file}.')
            np.savetxt(transf_txt_file, transforms, fmt='%g', delimiter=',')
    except Exception as e:
        logger.error(f'{LOGGER_PREFIX} Failed to save the video stabilization results to {transf_txt_file.resolve()}: {e}')
    else:
        logger.info(f'{LOGGER_PREFIX} Video stabilization results saved to: {transf_txt_file.resolve()}')
        
    serializable_config = convert_to_serializable(config)
    with open(info_yaml_file, 'w') as f:
        yaml.dump(serializable_config, f, default_flow_style=False)
    logger.info(f'{LOGGER_PREFIX} Video info saved to {info_yaml_file.resolve()}')

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Vehicle Detection, Tracking, and Stabilization Pipeline')
    
    parser.add_argument('source', type=Path, help='Path to the video file (e.g., path/to/video/video.mp4)')
    parser.add_argument('--cfg', '-c', type=str, default='cfg/default.yaml', help='Path to the main configuration file [default: cfg/default.yaml]')
    parser.add_argument('--classes', nargs='+', type=int, help='Overwrite classes to extract (e.g., --classes 0 1 2) [default: see cfg -> cfg_ultralytics -> classes]')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help='Filename for detailed logs (e.g., info.log). Saved in the script directory [default: None]')
    parser.add_argument('--verbose', '-v', action='store_true', help='Set verbosity level [default: False]')
    parser.add_argument('--interpolate', '-i', action='store_true', help='Interpolate tracks between missing frames (not implemented yet) [default: False]')
    parser.add_argument('--cut-frame-left', '-cfl', type=int, default=0, help='Cut video from the start at this frame number [default: 0]')
    parser.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Cut video from the end at this frame number [default: None]')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)
    process_video(args, logger)
