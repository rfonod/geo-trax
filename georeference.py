#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Robert Fonod (robert.fonod@ieee.org) and Haechan Cho (gkqkemwh@kaist.ac.kr)

"""
georeference.py - Georeferences the extracted vehicle tracking data using orthophotos.

This script is a crucial component of the Geo-trax pipeline that georeferences vehicle tracking data from stabilized drone footage
using orthophotos. It converts vehicle trajectories from video frame coordinates to geospatial coordinates (WGS84 or local CRS).
Key features include:
- Optional master frame approach for robust homography estimation
- Road section and lane assignment using segmentation data
- Speed and acceleration estimation
- Vehicle dimension conversion to real-world units
- Bounding box visibility tracking
The script is designed for quasi-stationary drone operations like intersection monitoring.

Usage:
    python georeference.py <source> [options]

Arguments:
    source                         : Path to the video file (e.g., path/to/video/video.mp4).

Options:
    --help, -h                     : Show this help message and exit.
    --cfg, -c <path>               : Path to the main geo-trax configuration file (default: cfg/default.yaml).
    --log-file, -lf <str>          : Filename to save detailed logs. Saved in the 'logs' folder.
    --verbose, -v                  : Set print verbosity level to INFO (default: WARNING).

    --ortho-folder, -of <path>     : Custom path to the folder with orthophotos (.png, .tif, .txt).
                                     Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.
    --geo-source, -gs <str>        : Source of georeferencing parameters (metadata-tif, text-file, center-text-file).
                                     If not provided, the system will auto-detect.
    --ref-frame, -rf <int>         : Use custom reference frame number (default: 0).
                                     Should be the same as the one used for stabilization.
    --no-master, -nm               : Disable the master frame approach.
    --master-folder, -mf <path>    : Custom path to the folder containing master frame files (.png).
                                     If not provided, '--ortho-folder / master_frames' will be used.
    --recompute, -r                : Force recompute master-> ortho homography even if it exists.
    --segmentation-folder, -osf <path> : Custom path to the folder containing orthophoto segmentation files (.csv).
                                         If not provided, '--ortho-folder / segmentations' will be used.

Examples:

  1. Georeference tracking data using default settings and custom orthophoto folder:
     python georeference.py path/to/video.mp4 -of path/to/orthophotos

  2. Use a custom master frames folder and always recompute master->ortho homography:
     python georeference.py path/to/video.mp4 -mf path/to/master -r

  3. Provide a custom segmentation folder and use a text file as the georeferencing source:
     python georeference.py path/to/video.mp4 -osf path/to/segmentations -gs text-file

  4. Specify georeferencing source to be a .tif metadata file and disable the master frame approach:
     python georeference.py path/to/video.mp4 -gs metadata-tif -nm

Notes:
  - Ensure that the orthophotos and segmentation data are correctly formatted and located in the specified folders.
  - The script assumes that the orthophotos are georeferenced to a known coordinate system.
  - The master frame approach can improve the robustness of the homography estimation but may require extra processing.
  - Additional configurations can be set in the main configuration file (default: cfg/default.yaml) and its associated config files.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from shapely.geometry import Polygon

from utils.utils import (
    check_if_results_exist,
    detect_delimiter,
    determine_location_id,
    get_ortho_folder,
    load_config_all,
    setup_logger,
)


def georeference(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Georeference the tracking data using orthophotos.
    """
    config = load_config_all(args, logger)['georef']
    location_id = determine_location_id(args.source, logger)
    track_id, frame_num, bbox_unstab, x_stab_frame, y_stab_frame, class_id, veh_dim_px = get_tracking_data(args.source, logger)
    timestamps = get_timestamps(args.source, frame_num, logger)
    reference_frame, frame_size, fps = get_video_data(args.source, args.ref_frame, logger)
    ortho_folder = get_ortho_folder(args.source, args.ortho_folder, logger)
    geo_source = get_geo_params_source(args.geo_source, ortho_folder, location_id, logger)
    ortho = get_orthophoto(ortho_folder, location_id, logger)
    ortho_params = get_ortho_parameters(ortho_folder, location_id, geo_source, ortho, config['cutout_width_px'], logger)
    ortho_segmentation = get_road_section_lane_geometry(ortho_folder, args.segmentation_folder, location_id, logger)

    if args.no_master:
        homography_reference_to_ortho = get_reference_to_ortho_homography(reference_frame, ortho, config['matching'], logger)
    else:
        master_frame = get_master_frame(ortho_folder, args.master_folder, location_id, logger)
        homography_reference_to_master = get_reference_to_master_homography(reference_frame, master_frame, config['matching'], logger)
        homography_master_to_ortho = get_master_to_ortho_homography(master_frame, ortho, ortho_folder, args.master_folder, location_id, args.recompute, config['matching'], logger)
        homography_reference_to_ortho = np.dot(homography_master_to_ortho, homography_reference_to_master)

    x_stab_ortho, y_stab_ortho = apply_homography(x_stab_frame, y_stab_frame, homography_reference_to_ortho)
    latitude, longitude = ortho2geo(x_stab_ortho, y_stab_ortho, ortho_params)
    x_local, y_local = geo2local(latitude, longitude, **config['transformation'])
    veh_dim_real = convert_dimensions(track_id, veh_dim_px, frame_size, homography_reference_to_ortho, ortho_params, **config['transformation'])
    visibility = calculate_visibility(track_id, bbox_unstab, frame_size, config['visibility_margin'])
    veh_speed, veh_acceleration = compute_kinematics(track_id, frame_num, x_local, y_local, visibility, fps, **config['filtering'])
    road_section, lane_number = assign_road_section_lane(x_stab_ortho, y_stab_ortho, ortho_segmentation)

    georeferenced_df = create_and_format_georeferenced_df(track_id, timestamps, frame_num, x_stab_ortho, y_stab_ortho, x_local, y_local,
                                                      latitude, longitude, veh_dim_real, class_id, veh_speed, veh_acceleration,
                                                      road_section, lane_number, visibility, config['min_traj_length'], logger)

    save_georeferenced_data(args.source, georeferenced_df, logger)
    save_homography(args.source, homography_reference_to_ortho, logger)


def get_tracking_data(source: Path, logger: logging.Logger) -> tuple:
    """
    Load tracking data from the provided file.
    """
    tracking_data_exists, tracking_data_filepath = check_if_results_exist(source, 'processed')
    if not tracking_data_exists:
        logger.critical(f"No tracking data found for: '{source}'. Run the 'detect_track_stabilize.py' script first.")
        sys.exit(1)
    delimiter = detect_delimiter(tracking_data_filepath)
    try:
        tracks = np.loadtxt(tracking_data_filepath, delimiter=delimiter, dtype=np.float64)
    except Exception as e:
        logger.critical(f"Failed to load tracking data from: '{tracking_data_filepath}' due to: {e}")
        sys.exit(1)

    if tracks.size == 0 or tracks.ndim != 2:
        logger.critical(f"No valid tracking data found in: '{tracking_data_filepath}'.")
    elif tracks.shape[1] < 14:
        logger.critical(f"Invalid tracking data format in: '{tracking_data_filepath}'. Expected at least 14 columns: \
                [track_id, frame_num, x_c_unstab, y_c_unstab, w_unstab, h_unstab, x_c_stab, y_c_stab, \
                    w_stab, h_stab, class_id, confidence, vehicle_length, vehicle_width]. Make sure you run \
                    geo-trax with stabilization enabled.")
        sys.exit(1)

    return (tracks[:, 1].astype('int'),      # track_id
            tracks[:, 0].astype('int'),      # frame_num
            tracks[:, 2:6],                  # bbox_unstab
            tracks[:, 6],                    # x_stab
            tracks[:, 7],                    # y_stab
            tracks[:, 10].astype('int'),     # class_id
            tracks[:, 12:14])                # dimensions


def get_timestamps(source: Path, frame_num: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Get timestamps from the provided file (e.g. extracted from drone flight log).
    """
    timestamp_filepath = source.with_suffix('.CSV' if source.suffix.isupper() else '.csv')
    if timestamp_filepath.exists():
        timestamps = pd.read_csv(timestamp_filepath, index_col='frame')
        results = []
        undefined_timestamp = "0000-00-00 00:00:00.000"
        if timestamps.index[0] != 0:
            logger.warning("The first frame number in the timestamps file is not 0. Adjusting the timestamps.")
            timestamps.index = timestamps.index - timestamps.index[0]
        for frame in frame_num:
            if frame in timestamps.index:
                results.append(timestamps.loc[frame, 'timestamp'])
            else:
                results.append(undefined_timestamp)
        logger.info(f"Loaded timestamps from: '{timestamp_filepath}'.")
        return np.array(results)
    else:
        logger.warning(f"No timestamp file found for: '{timestamp_filepath}'. Timestamps will be replaced by frame numbers.")
        return np.array([])


def get_video_data(video_filepath: Path, ref_frame_num: int, logger: logging.Logger) -> tuple:
    """
    Get video data (reference frame, frame dimensions, and FPS) from the video file.
    """
    cap = cv2.VideoCapture(str(video_filepath))
    if not cap.isOpened():
        logger.critical(f"Failed to open video file: '{video_filepath}'.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_num)
    ret, ref_frame = cap.read()
    if not ret:
        logger.critical(f"Failed to read frame {ref_frame_num} from video file: '{video_filepath}'.")
        cap.release()
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.critical(f"Failed to retrieve FPS from video file: '{video_filepath}'.")
        sys.exit(1)

    frame_dimensions = ref_frame.shape[:2]
    cap.release()

    logger.info(f"Loaded reference frame {ref_frame_num} from: '{video_filepath}' with dimensions {frame_dimensions} and FPS {fps}.")
    return ref_frame, frame_dimensions, fps


def get_orthophoto(ortho_folder: Path, location_id: str, logger: logging.Logger) -> np.ndarray:
    """
    Get orthophoto data (orthophoto image and dimensions) from the orthophoto
    """
    ortho_filepath = ortho_folder / (location_id + '.png')
    if not ortho_filepath.exists():
        logger.critical(f"Orthophoto file '{ortho_filepath}' not found.")
        sys.exit(1)
    try:
        orthophoto = cv2.imread(str(ortho_filepath))
    except Exception as e:
        logger.critical(f"Failed to load orthophoto '{ortho_filepath}' due to: {e}")
        sys.exit(1)

    logger.info(f"Loaded orthophoto from '{ortho_filepath}' with dimensions: {orthophoto.shape}.")
    return orthophoto


def get_ortho_parameters(ortho_folder: Path, location_id: str, geo_source: str, ortho: np.ndarray, cutout_width_px: Union[int, None], logger: logging.Logger) -> tuple:
    """
    Get orthophoto parameters from .tif metadata or .txt files.
    """
    ortho_filepath = ortho_folder / (location_id + '.png')
    if geo_source == "metadata-tif":
        img_tif = Image.open(ortho_filepath.with_suffix('.tif'))
        if isinstance(img_tif, TiffImagePlugin.TiffImageFile):
            lng0, lat0 =  img_tif.tag_v2[33922][3], img_tif.tag_v2[33922][4]
            dlng, dlat =  img_tif.tag_v2[33550][0], -img_tif.tag_v2[33550][1]
            skew_x, skew_y = 0.0, 0.0
            if 34264 in img_tif.tag_v2:
                skew_x, skew_y = img_tif.tag_v2[34264][1], img_tif.tag_v2[34264][2]
        else:
            logger.error(f"Failed to read georeferencing parameters from .tif metadata for orthophoto: '{ortho_filepath}'.")
            sys.exit(1)
    elif geo_source == "text-file":
        ortho_params = read_ortho_config_file(ortho_filepath.with_suffix('.txt'))
        lng0, lat0, dlng, dlat = ortho_params[:4]
        skew_x, skew_y = ortho_params[4:6] if len(ortho_params) == 6 else (0.0, 0.0)
    elif geo_source == "center-text-file":
        center_offset_x, center_offset_y = read_ortho_config_file(ortho_filepath.with_name(f"{ortho_filepath.stem}_center.txt"))[:2]
        ortho_width_px = ortho.shape[1]
        if cutout_width_px is None:
            width_half = ortho_width_px // 2
        else:
            width_half = cutout_width_px // 2

        ortho_params = read_ortho_config_file(ortho_filepath.with_name("ortho_parameters.txt"))
        lngs, lats, dlng, dlat = ortho_params[:4]
        skew_x, skew_y = ortho_params[4:6] if len(ortho_params) == 6 else (0.0, 0.0)

        lng0 = lngs + (center_offset_x - width_half) * dlng + (center_offset_y - width_half) * skew_x
        lat0 = lats + (center_offset_y - width_half) * dlat + (center_offset_x - width_half) * skew_y

        if cutout_width_px is not None and cutout_width_px != ortho_width_px:
            cutout_scale = cutout_width_px/ortho_width_px
            dlng, dlat, skew_x, skew_y = (param * cutout_scale for param in (dlng, dlat, skew_x, skew_y))
    else:
        logger.error(f"Invalid geo_source: '{geo_source}'.")
        sys.exit(1)
    logger.info(f"Loaded orthophoto parameters from a '{geo_source}' for orthophoto: '{ortho_filepath.name}'.")

    return lng0, lat0, dlng, dlat, skew_x, skew_y


def get_geo_params_source(geo_source: Union[str, None], ortho_folder: Path, location_id: str, logger: logging.Logger) -> str:
    """
    Detects the source of georeferencing parameters (either in .tif metadata or .txt files).
    Determines if an orthophoto cutout approach is used based on the presence of *_center.txt and ortho_parameters.txt.
    Creates a .png file from the .tif file if it does not exist.
    """
    if geo_source is not None:
        if geo_source not in ['metadata-tif', 'text-file', 'center-text-file']:
            logger.error(f"Invalid --geo-source argument: '{geo_source}'. Use 'metadata-tif', 'text-file', or 'center-text-file'.")
        return geo_source

    ortho_filepath = ortho_folder / (location_id + '.png')
    png_file = ortho_filepath.with_suffix('.png')
    tif_file = ortho_filepath.with_suffix('.tif')
    txt_file = ortho_filepath.with_suffix('.txt')
    txt_center_file = ortho_filepath.with_name(f"{ortho_filepath.stem}_center.txt")
    txt_params_file = ortho_filepath.with_name("ortho_parameters.txt")

    if tif_file.exists() and (txt_file.exists() or (txt_center_file.exists() and txt_params_file.exists())):
        logger.error(f"Both .tif and .txt files are present for orthophoto '{ortho_filepath}'. Specify the source using the '--geo-source' argument.")
        sys.exit(1)

    if tif_file.exists():
        if not png_file.exists():
            logger.warning(f"No '.png' file found for orthophoto '{ortho_filepath}'. Converting the .tif file to '.png'...")
            try:
                ortho = cv2.imread(str(tif_file))
                cv2.imwrite(str(png_file), ortho)
            except Exception as e:
                logger.error(f"Failed to convert '.tif' to '.png' due to: {e}")
                sys.exit(1)
            else:
                logger.info(f"Converted '{tif_file}' to '{png_file}'.")
        return "metadata-tif"

    if txt_file.exists() and txt_center_file.exists() and txt_params_file.exists():
        logger.error(f"Both '.txt' and '_center.txt' files are present for orthophoto: '{ortho_filepath}'. Specify the source using the '--geo-source' argument.")
        sys.exit(1)
    if txt_file.exists():
        return "text-file"
    if txt_center_file.exists() and txt_params_file.exists():
        return "center-text-file"

    logger.error(f"No georeferencing parameters found for orthophoto: '{ortho_filepath}'. Specify the source using the '--geo-source' argument.")
    sys.exit(1)


def read_ortho_config_file(filepath: Path) -> np.ndarray:
    """
    Load orthophoto parameters from a .txt file.
    """
    processed_lines = []
    with open(filepath, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                processed_lines.append(stripped_line)

    ortho_params = np.genfromtxt(processed_lines, delimiter=' ')
    return ortho_params


def get_road_section_lane_geometry(ortho_folder: Path, segmentation_folder: Union[Path, None], location_id: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load road section and lane number geometry from the orthophoto segmentation file.
    """
    if segmentation_folder is None:
        segmentation_filepath = ortho_folder / 'segmentations' / (location_id + '.csv')
    else:
        segmentation_filepath = segmentation_folder / (location_id + '.csv')
    if segmentation_filepath.exists():
        logger.info(f"Loaded road section and lane number geometry from: '{segmentation_filepath}'.")
        return pd.read_csv(segmentation_filepath).iloc[:,:10]
    else:
        logger.warning(f"No segmentation file found for: '{segmentation_filepath}'. Road section and lane number will not be assigned.")
        return pd.DataFrame()


def create_polygon(row: pd.Series) -> Polygon:
    """
    Create a polygon from the row of a DataFrame.
    """
    return Polygon([(row['tlx'], row['tly']), (row['blx'], row['bly']), (row['brx'], row['bry']), (row['trx'], row['try'])])


def assign_road_section_lane(ortho_x: np.ndarray, ortho_y: np.ndarray, ortho_segmentation: pd.DataFrame) -> tuple:
    """
    Assign road section and lane number to the vehicle trajectory points based on orthophoto segmentation.
    """
    if ortho_segmentation.empty:
        return None, None

    ortho_segmentation['geometry'] = ortho_segmentation.apply(create_polygon, axis=1)

    ortho_segmentation_gdf = gpd.GeoDataFrame(ortho_segmentation, geometry='geometry')
    ortho_segmentation_gdf.columns = ['section', 'lane', 'tlx', 'tly', 'blx', 'bly', 'brx', 'bry', 'trx', 'try', 'geometry']

    geometry = gpd.points_from_xy(ortho_x, ortho_y)
    ortho_points_gdf = gpd.GeoDataFrame({'geometry': geometry})

    ortho_points_gdf = gpd.sjoin(ortho_points_gdf, ortho_segmentation_gdf[['section', 'lane', 'geometry']], how='left', predicate='within')
    ortho_points_gdf = ortho_points_gdf.groupby(ortho_points_gdf.index).first()

    road_section = ortho_points_gdf['section'].to_numpy()
    lane_number = ortho_points_gdf['lane'].to_numpy()

    return road_section, lane_number


def get_master_frame(ortho_folder: Path, master_folder: Union[Path, None], location_id: str, logger: logging.Logger) -> np.ndarray:
    """
    Get the master frame from the master frames folder.
    """
    if master_folder is None:
        master_filepath = ortho_folder / 'master_frames' / (location_id + '.png')
    else:
        master_filepath = master_folder / (location_id + '.png')
    if not master_filepath.exists():
        logger.error(f"Master frame file '{master_filepath}' not found. If you do not want to use a master frame, use the '--no-master' option.")
        sys.exit(1)
    try:
        master_frame = cv2.imread(str(master_filepath))
    except Exception as e:
        logger.error(f"Failed to load master frame: '{master_filepath}' due to: {e}")
        sys.exit(1)
    logger.info(f"Loaded master frame from: '{master_filepath}' to act as an intermediate frame.")

    return master_frame


def get_reference_to_ortho_homography(reference_frame: np.ndarray, ortho: np.ndarray, config: dict, logger: logging.Logger) -> np.ndarray:
    """
    Compute the homography matrix directly between the reference frame and the orthophoto.
    """
    homography_reference_to_ortho = compute_homography(reference_frame, ortho, ('reference', 'ortho'), logger, **config)[0]
    return homography_reference_to_ortho


def get_reference_to_master_homography(reference_frame: np.ndarray, master_frame: np.ndarray, config: dict, logger: logging.Logger) -> np.ndarray:
    """
    Compute the homography matrix between the reference frame and the master frame.
    """
    homography_reference_to_master = compute_homography(reference_frame, master_frame, ('reference', 'master'), logger, **config)[0]
    return homography_reference_to_master


def get_master_to_ortho_homography(master_frame: np.ndarray, ortho: np.ndarray, ortho_folder: Path, master_folder: Union[Path, None], location_id: str, recompute: bool, config:dict, logger: logging.Logger) -> np.ndarray:
    """
    Get the homography matrix between the master frame and the orthophoto.
    """
    if master_folder is None:
        homography_filepath = ortho_folder / 'master_frames' / (location_id + '.txt')
    else:
        homography_filepath = master_folder / (location_id + '.txt')

    current_master_hash = compute_hash(master_frame)

    if homography_filepath.exists() and not recompute:
        try:
            with open(homography_filepath, 'r') as file:
                lines = file.readlines()
                homography_master_to_ortho = np.fromstring(lines[0], sep=',').reshape(3, 3)
                saved_master_hash = lines[3].strip().split(': ')[1]

            if saved_master_hash == current_master_hash:
                logger.info(f"Loaded 'master -> orthophoto' homography from: '{homography_filepath}'.")
                return homography_master_to_ortho
            else:
                logger.warning("Master frame has changed. Recomputing 'master -> orthophoto' homography.")
        except Exception as e:
            logger.error(f"Failed to load 'master -> orthophoto' homography from '{homography_filepath}' due to: {e}")
            sys.exit(1)

    homography_master_to_ortho, stats_txt = compute_homography(master_frame, ortho, ('master', 'ortho'), logger, **config)
    try:
        with open(homography_filepath, 'w') as file:
            np.savetxt(file, homography_master_to_ortho.reshape(1, -1), fmt='%.20g', delimiter=',')
            file.write('\n# Hash of the master frame\n')
            file.write(f'Hash: {current_master_hash}\n')
            file.write('\n# Image matching stats\n')
            file.write(f'Stats: {stats_txt}\n')
    except Exception as e:
        logger.error(f"Failed to save 'master -> orthophoto' homography to '{homography_filepath}' due to: {e}")
        sys.exit(1)
    logger.info(f"Computed and saved 'master -> orthophoto' homography to: '{homography_filepath}'.")

    return homography_master_to_ortho


def compute_hash(image: np.ndarray) -> str:
    """
    Compute a hash for the given image.
    """
    return hashlib.md5(image.tobytes()).hexdigest()


def compute_homography(img_src: np.ndarray, img_dst: np.ndarray, src_dst: tuple, logger: logging.Logger,
                       max_features: int = 250000, filter_ratio: float = 0.55, ransac_method: int = cv2.USAC_MAGSAC,
                       ransac_epipolar_threshold: float = 3.0, ransac_max_iter: int = 10000,
                       ransac_confidence: float = 0.999999, rsift_eps: float = 1e-8) -> tuple:
    """
    Compute homography between a source image and a destination image.
    """
    def convert_to_rootsift(descriptors, eps):
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        return descriptors

    def try_compute_homography(max_features):
        img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT.create(nfeatures=max_features, enable_precise_upscale=True)
        kpt_src, desc_src = sift.detectAndCompute(img_src_gray, None)  # type: ignore
        kpt_dst, desc_dst = sift.detectAndCompute(img_dst_gray, None)  # type: ignore

        if kpt_src is None or kpt_dst is None:
            return None, None

        desc_src = convert_to_rootsift(desc_src, rsift_eps)
        desc_dst = convert_to_rootsift(desc_dst, rsift_eps)

        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(desc_src, desc_dst, k=2)
        except cv2.error:
            return None, None

        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < filter_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return None, None

        pts_src = np.array([kpt_src[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 2)
        pts_dst = np.array([kpt_dst[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 2)

        homography, inliers = cv2.findHomography(pts_src, pts_dst, method = ransac_method, confidence = ransac_confidence,
                                                 ransacReprojThreshold = ransac_epipolar_threshold, maxIters = ransac_max_iter)

        stats_txt = f"Keypoints in {src_dst[0]} frame: {len(kpt_src)}, in {src_dst[1]}: {len(kpt_dst)}. Inliers: {inliers.sum()} out of {len(inliers)} matches"

        return homography, stats_txt, inliers

    max_features_to_try = max_features
    while max_features_to_try > 10000:
        homography, stats_txt, inliers = try_compute_homography(max_features_to_try)
        if homography is not None:
            if inliers.sum() < 50:
                logger.warning(stats_txt)
            else:
                logger.info(stats_txt)
            return homography, stats_txt
        max_features_to_try //= 2
        logger.warning(f"SIFT detection or matching failed with {max_features_to_try*2} max_features. Trying with {max_features_to_try} max_features.")

    logger.error("SIFT detection failed with all attempted feature counts.")
    sys.exit(1)


def apply_homography(input_x: np.ndarray, input_y: np.ndarray, homography: np.ndarray) -> tuple:
    """
    Convert input coordinates to output coordinates using the homography matrix.
    """
    input_points = np.column_stack((input_x, input_y)).reshape(-1, 1, 2)
    output_points = cv2.perspectiveTransform(input_points, homography).reshape(-1, 2)
    return output_points[:, 0], output_points[:, 1]


def ortho2geo(ortho_x: np.ndarray, ortho_y: np.ndarray, ortho_params: tuple) -> tuple:
    """
    Convert orthophoto coordinates to geographic coordinates.
    """
    lng0, lat0, dlng, dlat, skew_x, skew_y = ortho_params
    longitude = lng0 + dlng * ortho_x + skew_x * ortho_y
    latitude = lat0 + dlat * ortho_y + skew_y * ortho_x
    return latitude, longitude


def geo2local(latitude: np.ndarray, longitude: np.ndarray, source_crs: str, target_crs: str) -> tuple:
    """
    Convert geographic coordinates to local coordinates.
    """
    geo_coordinates_gdf = gpd.GeoDataFrame({'Latitude': latitude, 'Longitude': longitude},
        geometry=gpd.points_from_xy(longitude, latitude), crs=source_crs)

    local_coordinates_gdf = geo_coordinates_gdf.to_crs(target_crs)
    x_local = local_coordinates_gdf.geometry.x.to_numpy()
    y_local = local_coordinates_gdf.geometry.y.to_numpy()
    return x_local, y_local


def ortho2local(ortho_x: np.ndarray, ortho_y: np.ndarray, ortho_params: tuple, source_crs: str, target_crs: str) -> tuple:
    """
    Convert orthophoto coordinates to local coordinates.
    """
    latitude, longitude = ortho2geo(ortho_x, ortho_y, ortho_params)
    x_local, y_local = geo2local(latitude, longitude, source_crs, target_crs)
    return x_local, y_local


def frame2local(points_px: np.ndarray, homography: np.ndarray, ortho_params: tuple, source_crs: str, target_crs: str) -> np.ndarray:
    """
    Convert frame coordinates to local coordinates.
    """
    x_px, y_px = points_px[:, 0], points_px[:, 1]
    x_ortho, y_ortho = apply_homography(x_px, y_px, homography)
    x_local, y_local = ortho2local(x_ortho, y_ortho, ortho_params, source_crs, target_crs)

    return np.array([x_local, y_local]).T


def convert_dimensions(track_ids: np.ndarray, veh_dim_px: np.ndarray, frame_size: tuple, homography: np.ndarray,
                       ortho_params: tuple, source_crs: str, target_crs: str) -> tuple:
    """
    Convert dimensions from pixels to meters.
    """
    veh_length_px, veh_width_px = veh_dim_px.T
    veh_length_real = np.full(len(veh_length_px), np.nan)
    veh_width_real = np.full(len(veh_width_px), np.nan)
    p1_px = np.array([frame_size[1] / 2, frame_size[0] / 2])

    unique_track_ids = np.unique(track_ids)
    for track_id in unique_track_ids:
        indices = track_ids == track_id
        length_px = veh_length_px[indices][0]
        width_px = veh_width_px[indices][0]

        if not np.isnan(length_px) and not np.isnan(width_px):
            p2_px = p1_px + [0, width_px / 2]
            p3_px = p1_px + [length_px / 2, 0]
            points_px = np.array([p1_px, p2_px, p3_px])
            points_real = frame2local(points_px, homography, ortho_params, source_crs, target_crs)
            p1, p2, p3 = points_real

            length_real = 2 * np.linalg.norm(p1 - p3)
            width_real = 2 * np.linalg.norm(p1 - p2)

            veh_length_real[indices] = length_real
            veh_width_real[indices] = width_real

    return veh_length_real, veh_width_real


def calculate_visibility(track_ids: np.ndarray, bbox_unstab: np.ndarray, frame_size: tuple, visibility_margin: int = 4) -> np.ndarray:
    """
    Determine the visibility of the vehicle in the frame.
    A vehicle is considered visible if its bounding box is inside the frame with a margin.
    """
    x_unstab, y_unstab, w_unstab, h_unstab = bbox_unstab.T
    frame_width, frame_height = frame_size[1], frame_size[0]
    visibility = np.zeros(len(track_ids), dtype=bool)

    unique_track_ids = np.unique(track_ids)
    for track_id in unique_track_ids:
        indices = track_ids == track_id
        x, y, w, h = x_unstab[indices], y_unstab[indices], w_unstab[indices], h_unstab[indices]

        visible_x = (x - w / 2 > visibility_margin) & (x + w / 2 < frame_width - visibility_margin - 1)
        visible_y = (y - h / 2 > visibility_margin) & (y + h / 2 < frame_height - visibility_margin - 1)

        visibility[indices] = visible_x & visible_y

    return visibility


def compute_kinematics(track_ids: np.ndarray, frame_num: np.ndarray, x_local: np.ndarray, y_local: np.ndarray, visibility: np.ndarray,
                       fps: float, filter_type: str, kernel_size: int, conversion_factor: float = 3.6) -> tuple:
    """
    Compute vehicle speed and acceleration.
    """
    speed = np.full(len(track_ids), np.nan)
    acceleration = np.full(len(track_ids), np.nan)

    unique_track_ids = np.unique(track_ids)
    for track_id in unique_track_ids:
        indices = np.where(track_ids == track_id)[0]
        visible_indices = visibility[indices]

        if sum(visible_indices) >= 3:
            frames = frame_num[indices][visible_indices]
            x_coords = x_local[indices][visible_indices]
            y_coords = y_local[indices][visible_indices]

            x_interpolated, y_interpolated, present_indices = interpolate_missing_points(frames, x_coords, y_coords)
            speed_values = compute_speed(x_interpolated, y_interpolated, fps)
            speed_values = apply_filter(speed_values, kernel_size, filter_type)
            acceleration_values = compute_acceleration(speed_values, fps)

            speed_values *= conversion_factor # convert from m/s to a chosen unit (e.g. km/h)
            speed_values = np.insert(speed_values, 0, np.nan)
            acceleration_values = np.insert(acceleration_values, 0, [np.nan] * 2)

            speed[indices[visible_indices]] = speed_values[present_indices]
            acceleration[indices[visible_indices]] = acceleration_values[present_indices]

    return speed, acceleration


def interpolate_missing_points(frames: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Interpolate missing points in the trajectory.
    """
    x_interpolated = [x[0]]
    y_interpolated = [y[0]]
    point_presence = [1]
    previous_frame = frames[0]

    for i in range(1, len(frames)):
        current_frame = frames[i]
        frame_diff = current_frame - previous_frame

        if frame_diff > 1:
            x_diff = (x[i] - x[i - 1]) / frame_diff
            y_diff = (y[i] - y[i - 1]) / frame_diff
            for step in range(1, frame_diff):
                x_interpolated.append(x[i - 1] + step * x_diff)
                y_interpolated.append(y[i - 1] + step * y_diff)
                point_presence.append(0)

        x_interpolated.append(x[i])
        y_interpolated.append(y[i])
        point_presence.append(1)
        previous_frame = current_frame

    present_point_indices = np.nonzero(point_presence)[0]

    return x_interpolated, y_interpolated, present_point_indices


def compute_speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute the speed using a simple numerical differentiation method.
    """
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    speed = np.sqrt(delta_x**2 + delta_y**2) * fps
    return speed


def compute_acceleration(speed: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute the acceleration using a simple numerical differentiation method.
    """
    delta_speed = np.diff(speed)
    acceleration = delta_speed * fps
    return acceleration


def apply_filter(data: np.ndarray, kernel_size: int, filter_type: str = 'gaussian') -> np.ndarray:
    """
    Apply a filter to the data.
    """
    if filter_type == 'gaussian':
        return gaussian_filter1d(data, kernel_size, mode='reflect', truncate=3.0)
    elif filter_type == 'savgol':
        window_length = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        polyorder = 2
        return savgol_filter(data, window_length=window_length, polyorder=polyorder, mode='nearest')
    else:
        sys.exit(f"Error: Invalid filter type {filter_type}.")


def create_and_format_georeferenced_df(track_id, timestamps, frame_num, x_stab_ortho, y_stab_ortho, x_local, y_local,
                                       latitude, longitude, veh_dim_real, class_id, v_speed, v_acceleration,
                                       road_section, lane_number, visibility, min_traj_length, logger) -> pd.DataFrame:
    """
    Create and format the georeferenced data as a DataFrame.
    """
    try:
        data = {
            'Vehicle_ID': track_id,
            'Timestamp': timestamps if timestamps.size > 0 else None,
            'Frame_Number': frame_num if timestamps.size == 0 else None,
            'Ortho_X': x_stab_ortho,
            'Ortho_Y': y_stab_ortho,
            'Local_X': x_local,
            'Local_Y': y_local,
            'Latitude': latitude,
            'Longitude': longitude,
            'Vehicle_Length': veh_dim_real[0],
            'Vehicle_Width': veh_dim_real[1],
            'Vehicle_Class': class_id,
            'Vehicle_Speed': v_speed,
            'Vehicle_Acceleration': v_acceleration,
            'Road_Section': road_section,
            'Lane_Number': lane_number,
            'Visibility': visibility
        }

        georeferenced_df = pd.DataFrame({k: v for k, v in data.items() if v is not None})

        georeferenced_df['Ortho_X'] = np.round(georeferenced_df['Ortho_X'], 1)
        georeferenced_df['Ortho_Y'] = np.round(georeferenced_df['Ortho_Y'], 1)
        georeferenced_df['Local_X'] = np.round(georeferenced_df['Local_X'], 2)
        georeferenced_df['Local_Y'] = np.round(georeferenced_df['Local_Y'], 2)
        georeferenced_df['Longitude'] = np.round(georeferenced_df['Longitude'], 7)
        georeferenced_df['Latitude'] = np.round(georeferenced_df['Latitude'], 7)
        georeferenced_df['Vehicle_Length'] = np.round(georeferenced_df['Vehicle_Length'], 2)
        georeferenced_df['Vehicle_Width'] = np.round(georeferenced_df['Vehicle_Width'], 2)
        georeferenced_df['Vehicle_Speed'] = np.round(georeferenced_df['Vehicle_Speed'], 1)
        georeferenced_df['Vehicle_Acceleration'] = np.round(georeferenced_df['Vehicle_Acceleration'], 2)
        georeferenced_df['Visibility'] = georeferenced_df['Visibility'].astype('int')

        if 'Lane_Number' in georeferenced_df.columns:
            georeferenced_df['Lane_Number'] = georeferenced_df['Lane_Number'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

        if min_traj_length > 0:
            vehicle_count_before = len(georeferenced_df['Vehicle_ID'].unique())
            georeferenced_df = georeferenced_df.groupby('Vehicle_ID').filter(lambda x: len(x) > min_traj_length)
            vehicle_count_after = len(georeferenced_df['Vehicle_ID'].unique())
            vehicle_count_removed = vehicle_count_before - vehicle_count_after
            if vehicle_count_removed > 0:
                logger.info(f"Removed {vehicle_count_removed} vehicles with trajectory length less than {min_traj_length} points.")

        logger.info("Georeferenced DataFrame successfully created and formatted.")
        return georeferenced_df

    except Exception as e:
        logger.error(f"Error creating and formatting georeferenced DataFrame due to: {e}")
        sys.exit(1)


def save_georeferenced_data(source: Path, georeferenced_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Save the georeferenced data to a CSV file.
    """
    georeferenced_results_filepath = source.parent / 'results' / source.with_suffix('.csv').name
    georeferenced_df.to_csv(georeferenced_results_filepath, index=False)
    logger.info(f"Georeferenced data saved to: '{georeferenced_results_filepath}'.")


def save_homography(source: Path, homography: np.ndarray, logger: logging.Logger) -> None:
    """
    Save the computed homography to a .txt file.
    """
    geo_transf_filepath = source.parent / 'results' / (source.stem + '_geo_transf.txt')
    try:
        np.savetxt(geo_transf_filepath, homography.reshape(1, -1), fmt='%.20g', delimiter=',')
    except Exception as e:
        logger.error(f"Failed to save 'reference -> orthophoto' homography '{geo_transf_filepath}' due to: {e}")
        sys.exit(1)
    logger.info(f"Homography 'reference -> orthophoto' saved to: '{geo_transf_filepath}'.")


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Georeferencing the tracking data using orthophotos.")

    # Required arguments
    parser.add_argument("source", type=Path, help="Path to the input video file")

    # Optional arguments
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    parser.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    parser.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING)')

    # Georeferencing arguments
    parser.add_argument("--ortho-folder", "-of", type=Path, default=None, help="Custom path to the folder with orthophotos (.png, .tif, .txt). Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.")
    parser.add_argument("--geo-source", "-gs", choices=['metadata-tif', 'text-file', 'center-text-file'], default=None, help="Source of georeferencing parameters. If not provided, the system will auto-detect")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="Use custom reference frame number (should be the same as the one used for stabilization).")
    parser.add_argument("--no-master", "-nm", action="store_true", help="Disable the master frame approach.")
    parser.add_argument("--master-folder", "-mf", type=Path, default=None, help="Custom path to the folder containing master frame files (.png). If not provided, '--ortho-folder / master_frames' will be used.")
    parser.add_argument("--recompute", "-r", action="store_true", help="Force recompute master-> ortho homography even if it exists.")
    parser.add_argument("--segmentation-folder", "-osf", type=Path, default=None, help="Custom path to the folder containing orthophoto segmentation files (.csv). If not provided, '--ortho-folder / segmentations' will be used.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file)

    georeference(args, logger)
