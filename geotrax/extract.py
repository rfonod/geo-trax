#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
extract.py - Performs video processing for vehicle trajectory extraction in image coordinates.

This script is integral to the Geo-trax pipeline, focusing on the extraction of vehicle trajectories in
image coordinates from drone-derived video footage. Designed for quasi-stationary drone operations providing
a bird's-eye view, it is ideal for tasks such as intersection monitoring and similar applications. It leverages
a pre-trained YOLOv8 model to detect four vehicle classes, applies the selected tracking algorithm to maintain
consistent vehicle identification across frames, and employs a custom video stabilization routine to correct
for drone movement and ensure accurate vehicle trajectories. The script also estimates vehicle dimensions based
on bounding boxes and azimuth data. The extracted trajectories are saved to a text file, along with additional
metadata.

Usage:
    geotrax extract <source> [options]

Arguments:
    source                    : Path to the input video file.

Options:
    --help, -h                : Show this help message and exit.
    --cfg, -c <path>          : Path to a custom pipeline config file. Defaults to the bundled config;
                                run 'geotrax config show' to view it or 'geotrax config copy' to customize.
    --set, -st <KEY=VALUE>    : Override a pipeline config value for this run; repeat for more than
                                one, e.g. --set conf=0.35 --set iou=0.6. KEY is a dotted path or any
                                unambiguous tail of one; VALUE uses YAML rules. Prefer the dedicated
                                flag where one exists; passing both for the same key is an error.
    --output-folder, -of <str> : Root folder for outputs (bare name or absolute path).
                                Defaults to cfg -> output -> folder (historical default: 'results').
    --log-path, -lp <str>     : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
    --verbose, -v             : Set print verbosity level to INFO (default: WARNING).

Processing Options:
    --model, -m <str>         : Detection model to use — a local file path OR an
                              'hf://<org>/<repo>/<path/to/file>.pt' Hugging Face reference (auto-downloaded
                              and cached). Defaults to cfg -> extraction -> model.
    --class-names, -cn <ID=NAME|FILE> [...] : Rename class-id -> name labels with a .yaml/.json mapping
                              file or inline ID=NAME pairs (e.g. -cn 0=car 1=bus). Defaults to
                              cfg -> extraction -> class_rename, then the model's own names, then integer IDs.
    --conf, -co <float>       : Detection confidence threshold. Defaults to cfg -> ultralytics -> conf.
    --classes, -cls <int> [<int> ...] : Class IDs to extract (e.g., --classes 0 1 2).
                              Defaults to cfg -> ultralytics -> classes.
    --cut-frame-left, -cfl <int> : Skip the first N frames. Defaults to cfg -> processing -> cut_frame_left.
    --cut-frame-right, -cfr <int> : Stop processing after this frame. Defaults to cfg -> processing -> cut_frame_right.
    --interpolate / --no-interpolate : Fill per-track frame gaps with linear interpolation; adds a 15th
                              is_interpolated column to the .txt output (0 = real detection, 1 = synthetic).
                              Defaults to cfg -> extraction -> interpolate (default: false).
    --sahi / --no-sahi        : Detect via SAHI sliced inference (improves small-object recall; requires
                              'pip install geo-trax[sahi]'; roughly 5x slower per frame). Slicing parameters
                              live in cfg -> extraction -> sahi. Defaults to cfg -> extraction -> sahi -> enable.
    --stab-gpu / --no-stab-gpu, -sg : CUDA-accelerate stabilization image matching (requires a CUDA-enabled
                              OpenCV build; no CPU fallback). Defaults to cfg -> stabilo -> gpu.
    --stab-gpu-device-id, -sgid <int> : CUDA device index used when stabilization GPU is enabled.
                              Defaults to cfg -> stabilo -> gpu_device_id.
    --stab-detector, -sdet <str> : Stabilization feature detector: classical (orb, sift, rsift, brisk, kaze,
                              akaze) or learning-based (xfeat, disk, dedode, keynet, loftr).
                              Defaults to cfg -> stabilo -> detector_name.
    --stab-device, -sdev <str> : Torch device for the learning-based detectors/matchers (auto, cpu, cuda, mps);
                              ignored by the classical detectors. Defaults to cfg -> stabilo -> device.
    For full detection, tracking, and stabilization control, edit cfg -> ultralytics, cfg -> tracker,
    and cfg -> stabilo. Run 'geotrax config copy' to get an editable local copy of the pipeline config.
    Object-detection GPU use is set via cfg -> ultralytics -> device (default: auto, uses CUDA when available).

Examples:
  1. Process a video with default settings:
        geotrax extract path/to/video.mp4

  2. Use a custom (locally copied) pipeline config and consider only the first two vehicle classes:
        geotrax extract path/to/video.mp4 --cfg default_copy.yaml --classes 0 1

  3. Skip the first 100 frames and stop processing after frame 500:
        geotrax extract path/to/video.mp4 --cut-frame-left 100 --cut-frame-right 500

  4. Rename class labels for visualization (overrides the model's built-in names):
        geotrax extract path/to/video.mp4 -cn 0=vehicle 1=bus 2=truck 3=bike

  5. Fill per-track detection gaps with linear interpolation:
        geotrax extract path/to/video.mp4 --interpolate

Notes:
  - Detection, tracking, and stabilization parameters all live in a single pipeline config file
    (cfg -> ultralytics, cfg -> tracker, cfg -> stabilo). Run 'geotrax config copy' to copy the
    defaults locally, then edit and pass via -c. For a one-off change, --set KEY=VALUE reaches
    any of these without a config file.
  - The active tracker is chosen by cfg -> tracker -> active; the full parameter block for every
    supported tracker is kept in the config so you can switch by changing that one line.
  - Extraction-stage settings (stabilize/save_stab toggles, min_track_length, and the
    dimension_estimation block) are in cfg -> extraction; min_track_length has no CLI flag.
  - Per-class confidence thresholds live in cfg -> extraction -> class_conf, e.g. {1: 0.15, 2: 0.4}
    (set it for one run with --set 'class_conf={1: 0.15, 2: 0.4}'); classes it does not list keep
    cfg -> ultralytics -> conf. The detector runs at the lowest threshold and every detection is held
    to its own class threshold before tracking, so rejected boxes never reach the tracker.
  - SAHI mode honors cfg -> ultralytics conf, device, imgsz (applied per slice), and classes, plus
    cfg -> extraction -> class_conf; iou, max_det, augment, half, vid_stride, and agnostic_nms are
    not used (detection merging is controlled by cfg -> extraction -> sahi instead). SAHI mode supports YOLO models only (no RTDETR) and cannot be
    combined with the tracktrack tracker or with ReID model 'auto' (both need a live Ultralytics predictor).
  - Output filename postfixes (e.g. _vid_transf suffix) are set in cfg -> output; use
    --output-folder / cfg -> output -> folder to redirect where outputs are written.
  - A '<stem>.yaml' run-metadata file is written into the output folder next to the results,
    recording the effective configuration (config file + every CLI flag and --set override).
    Before v1.4.0 this file was saved next to the input video.
"""

import argparse
import datetime
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from stabilo import Stabilizer
from tqdm import tqdm
from ultralytics import RTDETR, YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.checks import check_yolo
from ultralytics.utils.files import increment_path

from geotrax import __version__
from geotrax.utils.cli_utils import add_cfg_arg, add_common_args, finalize_cli_args
from geotrax.utils.config_utils import load_config_all, merge_bundled_defaults
from geotrax.utils.constants import DEFAULT_TRACK_BUFFER
from geotrax.utils.file_utils import (
    atomic_output,
    build_result_path,
    check_if_results_exist,
    convert_to_serializable,
    get_output_dir,
    get_video_dimensions,
)
from geotrax.utils.logging_utils import setup_logger
from geotrax.utils.registration import DETECTOR_CHOICES, DEVICE_CHOICES

_INFERENCE_KEYS = {
    'conf', 'iou', 'imgsz', 'max_det', 'classes',
    'augment', 'agnostic_nms', 'half', 'device', 'vid_stride',
    'mode', 'task', 'stream_buffer',
}

TRACK_FALLBACK_CONF = 0.1


def to_homography(matrix: np.ndarray) -> np.ndarray:
    """Return *matrix* as a 3x3 homography.

    Stabilo returns a 2x3 matrix when ``stabilo -> transformation_type`` is 'affine' and a 3x3 one
    for 'projective'. The stabilization transform file and every consumer of it (``visualize`` and
    ``georeference``) are defined in terms of 3x3 matrices, so promote the affine form by appending
    its implicit ``[0, 0, 1]`` row. Without this, an affine run wrote 7-column rows that the 3x3
    reshape in :func:`save_results` either rejected, losing the file entirely, or silently
    reinterpreted as unrelated matrices.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        return np.vstack((matrix, np.array([0.0, 0.0, 1.0])))
    return matrix


class ExtractionError(RuntimeError):
    """Detection, tracking or saving failed for a video.

    Raised instead of returning empty results, so that standalone ``extract`` exits 1 and ``batch``
    counts the video as failed and skips its later stages, rather than reporting success.
    """


def detect_track_stabilize(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process video based on provided arguments.
    """
    # load_config_all() has already reconciled the CLI flags with the config in both directions
    # (see sync_args_with_config), so args and config agree and either can be read from here on.
    config = load_config_all(args, logger)
    out_cfg = config['main'].get('output', {})
    class_conf = resolve_class_conf(config['ultralytics'], config['main']['extraction'], logger)
    if args.sahi:
        validate_sahi_tracker(config['main'])
        model = load_sahi_detector(config['ultralytics'], logger, conf=class_conf.predict_conf)
    else:
        model = load_detector(config['ultralytics'], logger)
    tracks, transforms = track_with_model(model, config, logger, class_conf)
    tracks = postprocess_tracks(tracks, config, logger)
    save_results(tracks, transforms, config, logger, out_cfg)


def track_with_model(
    model: Any, config: Dict, logger: logging.Logger, class_conf: Union['ClassConf', None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track vehicles in the video using the provided model.

    The model is either an Ultralytics YOLO/RTDETR model or, in SAHI mode, a SAHI AutoDetectionModel
    whose detections are fed to a manually created tracker.

    With per-class confidence thresholds (``class_conf.thresholds`` set), the detector runs at the
    lowered ``class_conf.predict_conf`` and each detection is then held to its own class threshold
    before it reaches the tracker: through an ``on_predict_postprocess_end`` callback registered here,
    ahead of the tracker callback Ultralytics adds on the first ``track()`` call, or directly in
    :func:`detect_frame_sahi`. The run-metadata ``detection`` block keeps the configured ``conf``,
    since the lowered value is an implementation detail of the per-class filter.
    """
    reader, pbar = initialize_streams(config['main'], config['ultralytics']['imgsz'], logger)
    stabilizer = Stabilizer(**config['stabilo'])
    per_class = class_conf if class_conf is not None and class_conf.thresholds else None

    sahi_cfg = merge_bundled_defaults(config['main']['extraction'].get('sahi'), 'extraction.sahi')
    if sahi_cfg.get('enable', False):
        tracker = create_manual_tracker(config['main'])

        def detect_frame(frame: np.ndarray) -> Tuple[Boxes, Dict]:
            return detect_frame_sahi(
                model, tracker, frame, sahi_cfg, config['ultralytics'].get('classes'), class_conf=per_class
            )
    else:
        track_cfg = config['ultralytics']
        if per_class is not None:
            track_cfg = {**track_cfg, 'conf': per_class.predict_conf}
            model.add_callback('on_predict_postprocess_end', make_class_conf_callback(per_class))

        def detect_frame(frame: np.ndarray) -> Tuple[Boxes, Dict]:
            return detect_frame_ultralytics(model, frame, track_cfg)

    frame_num, yolo_time, stab_time = 0, [], []
    frame_arr, track_id, bbox, bbox_stab, class_id, conf, transforms = [], [], [], [], [], [], []

    try:
        while reader.isOpened():
            success, frame = reader.read()
            if success and frame_num < config['main']['args'].cut_frame_left:
                frame_num += 1
                pbar.update()
                continue

            if success:
                boxes, speed = detect_frame(frame)
                yolo_time.append(sum(speed.values()))

                class_freq = {c: 0 for c in config['ultralytics'].get('classes') or []}
                if len(boxes) > 0:
                    frame_arr.append(np.full((len(boxes), 1), frame_num, dtype=np.uint32))
                    if boxes.id is not None:
                        track_ids = boxes.id.detach().numpy(force=True).astype(np.uint32).reshape(-1, 1)
                    else:
                        track_ids = np.full((len(boxes), 1), -1)
                    track_id.append(track_ids)
                    bbox.append(boxes.xywh.detach().numpy(force=True).astype(np.float32))
                    class_id.append(boxes.cls.detach().numpy(force=True).astype(np.uint8).reshape(-1, 1))
                    conf.append(boxes.conf.detach().numpy(force=True).astype(np.float32).reshape(-1, 1))

                    if config['main']['args'].verbose:
                        unique, counts = np.unique(class_id[-1], return_counts=True)
                        class_freq.update(dict(zip(unique, counts)))

                if config['main']['extraction']['stabilize']:
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
                            transf_matrix = to_homography(transf_matrix).flatten().reshape(1, -1)
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
        raise ExtractionError(f"Error processing: '{config['main']['args'].source}' due to: {e}") from e
    else:
        pbar.total = frame_num
        pbar.refresh()
        if yolo_time:
            logger.info(f"Average detection (preprocess + inference + postprocess) time: {sum(yolo_time) / len(yolo_time):5.1f}ms.")
            logger.info(f"Average stabilization time: {sum(stab_time) / len(stab_time):5.1f}ms") if stab_time else None
            logger.info(f"Average pipeline time: {1000 * len(yolo_time) / (sum(yolo_time) + sum(stab_time)):4.1f}fps.")
    finally:
        reader.release()
        pbar.set_postfix_str('done')
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


def load_sahi_detector(config: Dict, logger: logging.Logger, conf: Union[float, None] = None) -> Any:
    """
    Load the detection model wrapped in a SAHI AutoDetectionModel for sliced inference.

    ``conf`` overrides ``config['conf']`` as SAHI's confidence threshold; it carries the lowered
    threshold of :func:`resolve_class_conf` when per-class thresholds are configured.
    """
    try:
        from sahi import AutoDetectionModel
    except ImportError:
        logger.critical(
            "SAHI mode is enabled but the 'sahi' package is not installed. Install the optional extra:\n"
            "  python -m pip install 'geo-trax[sahi]'   # if geo-trax was installed from PyPI\n"
            "  python -m pip install -e '.[sahi]'       # if working from a source checkout\n"
            "Alternatively, disable SAHI with --no-sahi or cfg -> extraction -> sahi -> enable: false."
        )
        sys.exit(1)

    device = config['device']
    if isinstance(device, (list, tuple)):
        raise ValueError(
            "Multi-GPU is not supported in SAHI mode; set cfg -> ultralytics -> device to a single device."
        )
    if isinstance(device, int):
        device = f'cuda:{device}'

    try:
        model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=config['model'],
            confidence_threshold=config['conf'] if conf is None else conf,
            device=device,
            image_size=config['imgsz'],
        )
    except Exception as e:
        logger.error(f"Error loading the detection model for SAHI: {e}")
        sys.exit(1)
    else:
        logger.info(f"Detection model '{config['model']}' loaded successfully (SAHI sliced inference mode).")

    check_yolo(device=config['device'])
    return model


def validate_sahi_tracker(main_cfg: Dict) -> None:
    """
    Check that the active tracker can be fed detections manually (required in SAHI mode).
    """
    tracker_params = main_cfg.get('tracker_params', {})
    if main_cfg['tracker_active'] == 'tracktrack':
        raise ValueError(
            "SAHI mode cannot use the 'tracktrack' tracker (it requires a live Ultralytics predictor); "
            "set cfg -> tracker -> active to botsort, bytetrack, ocsort, deepocsort, or fasttrack."
        )
    if tracker_params.get('with_reid') and tracker_params.get('model') == 'auto':
        raise ValueError(
            "SAHI mode cannot use ReID model 'auto' (it derives appearance features from the detector's "
            "forward pass, unavailable with sliced inference); set a concrete ReID model or "
            "with_reid: false in cfg -> tracker."
        )


class ClassConf(NamedTuple):
    """Per-class confidence thresholds resolved from cfg -> extraction -> class_conf.

    ``default`` is the global cfg -> ultralytics -> conf and applies to every class absent from
    ``thresholds``. ``predict_conf`` is the threshold handed to the detector: the lowest of
    ``default`` and every per-class value, so that no class is cut before its own threshold is
    applied. ``thresholds`` is None when no per-class threshold is configured; no filter is then
    installed and extraction behaves exactly as with the global threshold alone.
    """

    predict_conf: float
    default: float
    thresholds: Union[Dict[int, float], None]


def resolve_class_conf(ultra_cfg: Dict, extraction_cfg: Dict, logger: logging.Logger) -> ClassConf:
    """Resolve cfg -> extraction -> class_conf against the global cfg -> ultralytics -> conf.

    A null global conf falls back to 0.1, the value Ultralytics' ``Model.track`` substitutes. Keys
    must be class IDs (ints, or strings of ints as a JSON mapping would give) and values numbers in
    [0, 1]; anything else raises ValueError, since a silently ignored threshold is worse than a stop.
    A class ID excluded by cfg -> ultralytics -> classes is accepted with a warning, because its
    threshold can never apply.
    """
    default = float(ultra_cfg.get('conf') or TRACK_FALLBACK_CONF)
    raw = extraction_cfg.get('class_conf')
    if not raw:
        return ClassConf(default, default, None)
    if not isinstance(raw, dict):
        raise ValueError(
            f"cfg -> extraction -> class_conf must map class IDs to thresholds, e.g. {{2: 0.4}}; got {raw!r}."
        )

    thresholds = {}
    for raw_key, value in raw.items():
        key = int(raw_key) if isinstance(raw_key, str) and raw_key.strip().lstrip('-').isdigit() else raw_key
        if not isinstance(key, int) or isinstance(key, bool):
            raise ValueError(f"cfg -> extraction -> class_conf key {raw_key!r} is not an integer class ID.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"cfg -> extraction -> class_conf value for class {key} must be a number in [0, 1]; got {value!r}."
            )
        thresholds[key] = float(value)

    classes = ultra_cfg.get('classes')
    unused = sorted(k for k in thresholds if classes is not None and k not in classes)
    if unused:
        logger.warning(
            f"cfg -> extraction -> class_conf sets thresholds for class(es) {unused}, "
            "which cfg -> ultralytics -> classes excludes."
        )

    predict_conf = min(default, *thresholds.values())
    logger.info(
        f"Per-class confidence thresholds: {thresholds} (other classes: {default}; detector runs at {predict_conf})."
    )
    return ClassConf(predict_conf, default, thresholds)


def class_conf_mask(cls: np.ndarray, conf: np.ndarray, class_conf: ClassConf) -> np.ndarray:
    """Return a boolean mask of the detections scoring above the threshold of their class.

    The comparison is strict and in float32, matching the confidence filter Ultralytics applies in
    NMS, so a class held at the global threshold keeps exactly the detections it kept before.
    """
    cls = np.asarray(cls).reshape(-1)
    limits = np.array([class_conf.thresholds.get(int(c), class_conf.default) for c in cls], dtype=np.float32)
    return np.asarray(conf, dtype=np.float32).reshape(-1) > limits


def make_class_conf_callback(class_conf: ClassConf) -> Callable[[Any], None]:
    """Build an Ultralytics ``on_predict_postprocess_end`` callback applying per-class thresholds.

    Registered on the model before its first ``track()`` call, it runs ahead of the tracker callback
    Ultralytics appends then, so rejected detections never reach the tracker. ``Results`` indexing
    does not carry ``feats`` (the ReID embeddings of tracker ``model: auto``), so they are sliced
    alongside the boxes; otherwise the tracker would receive the kept boxes without their features.
    """

    def filter_results(predictor: Any) -> None:
        for i, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            cls = boxes.cls.detach().numpy(force=True)
            keep = class_conf_mask(cls, boxes.conf.detach().numpy(force=True), class_conf)
            if keep.all():
                continue
            idx = np.flatnonzero(keep)
            filtered = result[idx]
            feats = getattr(result, 'feats', None)
            if feats is not None:
                filtered.feats = feats[idx]
            predictor.results[i] = filtered

    return filter_results


def create_manual_tracker(main_cfg: Dict) -> Any:
    """
    Instantiate the active tracker directly; SAHI mode feeds it detections manually.
    """
    return TRACKER_MAP[main_cfg['tracker_active']](args=IterableSimpleNamespace(**main_cfg['tracker_params']))


def sahi_predictions_to_boxes(
    object_predictions: list,
    orig_shape: Tuple[int, int],
    classes: Union[list, None],
    class_conf: Union[ClassConf, None] = None,
) -> Boxes:
    """
    Convert SAHI object predictions to an Ultralytics Boxes object, applying the class-ID filter
    (cfg -> ultralytics -> classes), which SAHI itself does not support, and the per-class
    confidence thresholds of ``class_conf`` (cfg -> extraction -> class_conf) when given.
    """
    rows = []
    for pred in object_predictions:
        cls = int(pred.category.id)
        if classes is not None and cls not in classes:
            continue
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        rows.append([x1, y1, x2, y2, pred.score.value, cls])
    data = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 6), dtype=torch.float32)
    if class_conf is not None and class_conf.thresholds and len(data):
        data = data[torch.from_numpy(class_conf_mask(data[:, 5].numpy(), data[:, 4].numpy(), class_conf))]
    return Boxes(data, orig_shape)


def detect_frame_ultralytics(model: Union[YOLO, RTDETR], frame: np.ndarray, config: Dict) -> Tuple[Boxes, Dict]:
    """
    Detect and track objects in a single frame via the Ultralytics pipeline.
    """
    results = model.track(frame, **config, persist=True)
    return results[0].boxes, results[0].speed


def detect_frame_sahi(
    model: Any,
    tracker: Any,
    frame: np.ndarray,
    sahi_cfg: Dict,
    classes: Union[list, None],
    class_conf: Union[ClassConf, None] = None,
) -> Tuple[Boxes, Dict]:
    """
    Detect objects in a single frame via SAHI sliced inference and update the tracker manually.

    Per-class confidence thresholds (``class_conf``) are applied before the tracker update.

    Mirrors ultralytics.trackers.track.on_predict_postprocess_end: when the tracker returns no tracks,
    the raw detections are kept (their IDs stay None, written as -1 downstream).
    """
    from sahi.predict import get_sliced_prediction

    start_time = time.time()
    result = get_sliced_prediction(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),  # SAHI expects RGB input
        model,
        slice_height=sahi_cfg['slice_height'],
        slice_width=sahi_cfg['slice_width'],
        overlap_height_ratio=sahi_cfg['overlap_height_ratio'],
        overlap_width_ratio=sahi_cfg['overlap_width_ratio'],
        perform_standard_pred=sahi_cfg['perform_standard_pred'],
        postprocess_type=sahi_cfg['postprocess_type'],
        postprocess_match_metric=sahi_cfg['postprocess_match_metric'],
        postprocess_match_threshold=sahi_cfg['postprocess_match_threshold'],
        postprocess_class_agnostic=sahi_cfg['class_agnostic'],
        verbose=0,
    )
    boxes = sahi_predictions_to_boxes(result.object_prediction_list, frame.shape[:2], classes, class_conf)
    tracks = tracker.update(boxes.cpu().numpy(), frame)
    if len(tracks):
        boxes = Boxes(torch.as_tensor(tracks[:, :-1]), frame.shape[:2])  # drop the detection-index column

    total_time = 1000 * (time.time() - start_time)
    durations = getattr(result, 'durations_in_seconds', None) or {}
    speed = {
        'preprocess': 1000 * durations.get('slice', 0.0),
        'inference': 1000 * durations.get('prediction', 0.0),
    }
    speed['postprocess'] = max(0.0, total_time - speed['preprocess'] - speed['inference'])  # merge + tracking
    return boxes, speed


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

    _bar_w = max(10, shutil.get_terminal_size().columns - 88)
    pbar = tqdm(total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), unit='f', leave=True, colour='yellow',
                desc=f'{video_filepath.name} - {"" if config["args"].verbose else "processing"} @ {imgsz}px ',
                bar_format=f'{{l_bar}}{{bar:{_bar_w}}}{{r_bar}}')
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
        if not frame_arr:
            logger.warning('No detections in the processed frame range; no tracks will be written.')
            return np.empty((0, 12)), (np.concatenate(transforms, axis=0) if transforms else np.empty((0, 10)))

        frame_arr = np.concatenate(frame_arr, axis=0) if frame_arr else np.array([[]])
        track_id = np.concatenate(track_id, axis=0) if track_id else np.array([[]])
        bbox = np.concatenate(bbox, axis=0) if bbox else np.array([[]])
        bbox_stab = np.concatenate(bbox_stab, axis=0) if bbox_stab else np.array([[]]).reshape(len(track_id), 0)
        class_id = np.concatenate(class_id, axis=0) if class_id else np.array([[]])
        conf = np.concatenate(conf, axis=0) if conf else np.array([[]])

        tracks = np.concatenate([frame_arr, track_id, bbox, bbox_stab, class_id, conf], axis=1, dtype=np.float32)
        if tracks.size > 0:
            tracks = tracks[tracks[:, 1] != -1]
        transforms = np.concatenate(transforms, axis=0) if transforms else np.empty((0, 10))
    except Exception as e:
        raise ExtractionError(f'Error aggregating results: {e}') from e
    return tracks, transforms


def postprocess_tracks(tracks: np.ndarray, config: Dict, logger: logging.Logger) -> np.ndarray:
    """
    Postprocess the extracted tracks.
    """
    tracks = remove_short_tracks(tracks, logger, config['main']['extraction']['min_track_length'])
    tracks = calculate_unique_classes(tracks)
    tracks = estimate_vehicle_dimensions(tracks, config['main'])
    if config['main']['args'].interpolate:
        max_gap = config['main']['tracker_params'].get('track_buffer')
        if max_gap is None:
            max_gap = DEFAULT_TRACK_BUFFER
            logger.warning(
                f"Active tracker '{config['main'].get('tracker_active')}' has no 'track_buffer' "
                f"parameter; falling back to a max interpolation gap of {max_gap} frames."
            )
        tracks = interpolate_tracks(tracks, logger, max_gap)
    return tracks


def interpolate_tracks(tracks: np.ndarray, logger: logging.Logger, max_gap: int) -> np.ndarray:
    """Fill per-track frame-id gaps via linear interpolation; appends is_interpolated flag column.

    Gaps larger than max_gap (the active tracker's track_buffer) are left unfilled, since the
    tracker itself would not persist a lost track's ID across a longer occlusion — a wider gap
    signals ID reuse for an unrelated detection rather than a genuine, bridgeable occlusion.
    """
    if tracks.size == 0:
        return tracks

    interpolated_rows = []
    interpolated_track_ids = set()
    skipped_gaps = 0

    for track_id in np.unique(tracks[:, 1]):
        mask = tracks[:, 1] == track_id
        t = tracks[mask]
        t = t[np.argsort(t[:, 0])]
        frames = t[:, 0].astype(int)

        for i in range(1, len(frames)):
            gap = frames[i] - frames[i - 1]
            if gap <= 1:
                continue
            if gap > max_gap:
                skipped_gaps += 1
                continue
            for step in range(1, gap):
                alpha = step / gap
                row = t[i - 1] * (1.0 - alpha) + t[i] * alpha
                row[0] = float(frames[i - 1] + step)
                interpolated_rows.append(row)
            interpolated_track_ids.add(track_id)

    is_interp_col = np.zeros((len(tracks), 1), dtype=tracks.dtype)
    tracks = np.concatenate([tracks, is_interp_col], axis=1)

    if skipped_gaps > 0:
        logger.warning(f"Skipped {skipped_gaps} frame gap(s) exceeding the tracker's track_buffer ({max_gap} frames); left unfilled.")

    if interpolated_rows:
        interp_arr = np.array(interpolated_rows, dtype=tracks.dtype)
        is_interp_flag = np.ones((len(interp_arr), 1), dtype=tracks.dtype)
        interp_arr = np.concatenate([interp_arr, is_interp_flag], axis=1)
        tracks = np.concatenate([tracks, interp_arr], axis=0)
        sort_idx = np.lexsort((tracks[:, 0], tracks[:, 1]))
        tracks = tracks[sort_idx]
        n_added = len(interp_arr)
        logger.info(f"Interpolated {n_added} missing frame(s) across {len(interpolated_track_ids)} track(s).")

    return tracks


def remove_short_tracks(tracks: np.ndarray, logger: logging.Logger, min_length: int = 3) -> np.ndarray:
    """
    Remove tracks with trajectory length shorter than specified.
    """
    if tracks.size == 0:
        return tracks
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


def calculate_unique_classes(tracks: np.ndarray) -> np.ndarray:
    """
    Assign each track a single class: the one with the highest confidence-weighted vote.
    """
    id2weighted_class_freq = {}
    if tracks.size != 0:
        for track in tracks:
            track_id, c, conf_score = int(track[1]), int(track[-2]), track[-1]
            class_freq = id2weighted_class_freq.setdefault(track_id, {})
            class_freq[c] = class_freq.get(c, 0.0) + conf_score

        # highest-weighted class per track; ties resolve to the lowest class id
        id2class_max = {
            track_id: max(sorted(class_freq), key=class_freq.get)
            for track_id, class_freq in id2weighted_class_freq.items()
        }

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
    eps = config['extraction']['dimension_estimation']['eps']
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
        idx_x, idx_y, idx_c = 2, 3, 6  # only unstabilized tracks available

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
    r0 = config['extraction']['dimension_estimation']['r0']
    gsd = config['extraction']['dimension_estimation']['gsd']
    theta_bar = config['extraction']['dimension_estimation']['theta_bar']
    theta_bar_rad = np.deg2rad(theta_bar)
    tau_c = config['extraction']['dimension_estimation']['tau_c']
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

    # Step 5: append estimated dimensions to each track row
    tracks = np.append(tracks, np.zeros((len(tracks), 2)), axis=1)
    for i, track in enumerate(tracks):
        track_id = int(track[1])
        tracks[i, -2] = id2length.get(track_id, np.nan)
        tracks[i, -1] = id2width.get(track_id, np.nan)

    return tracks


def save_results(tracks: np.ndarray, transforms: np.ndarray, config: Dict, logger: logging.Logger, out_cfg: Dict) -> None:
    """
    Save the detection, tracking, and stabilization results to files.

    batch treats an existing tracks file as a finished extraction, so the tracks file of an earlier
    run is removed first and the new one is written last, after the transforms and the metadata. An
    interrupted or failed save therefore never leaves a tracks file that later stages would pair
    with the new metadata. Every file is written atomically (atomic_output). A transforms file this
    run does not write (stabilization or save_stab off) is removed, since visualize would otherwise
    warp the new tracks with the old matrices. With no tracks, no tracks file is left at all.
    """
    source = config['main']['args'].source
    save_dir = increment_path(get_output_dir(source, out_cfg), exist_ok=True, mkdir=True)
    tracks_postfix = out_cfg.get('tracks_postfix', '')
    stab_postfix = out_cfg.get('stab_transform_postfix', '_vid_transf')
    tracks_txt_file = save_dir / f'{source.stem}{tracks_postfix}.txt'
    transf_txt_file = save_dir / f'{source.stem}{stab_postfix}.txt'
    info_yaml_file = build_result_path(source, 'metadata', out_cfg)

    tracks_txt_file.unlink(missing_ok=True)

    if transforms.size != 0 and config['main']['extraction']['save_stab']:
        frame_nums = transforms[:, 0].astype(int)
        matrices = transforms[:, 1:].reshape((-1, 3, 3))
        if not np.all(np.diff(frame_nums) == 1):
            logger.warning(f"Missing frame ids found in: '{transf_txt_file}'.")
        if not np.all(np.linalg.det(matrices) > 0):
            logger.warning(f"Invalid transforms found in: '{transf_txt_file}'.")
        try:
            with atomic_output(transf_txt_file) as tmp_file:
                np.savetxt(tmp_file, transforms, fmt='%.16g', delimiter=',')
        except Exception as e:
            logger.error(f"Failed to save the video stabilization results to: '{transf_txt_file.resolve()}' due to: {e}")
        else:
            logger.info(f"Video stabilization results saved to: '{transf_txt_file.resolve()}'")
    elif transf_txt_file.exists():
        transf_txt_file.unlink()
        logger.warning(
            f"Removed '{transf_txt_file.name}' left by an earlier run: this run saved no stabilization transforms "
            f"(stabilization or save_stab disabled, or none computed), so it no longer matches the tracks."
        )

    metadata = convert_to_serializable(_build_run_metadata(config, save_dir))
    with atomic_output(info_yaml_file) as tmp_file, open(tmp_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Video info and configs saved to: '{info_yaml_file.resolve()}'")

    if tracks.size == 0:
        logger.warning(f"No tracks to save for '{source}'; no tracking results file was written.")
        return
    try:
        with atomic_output(tracks_txt_file) as tmp_file:
            np.savetxt(tmp_file, tracks, fmt='%g', delimiter=',')
    except Exception as e:
        raise ExtractionError(f"Failed to save the tracking results to: '{tracks_txt_file.resolve()}' due to: {e}") from e
    logger.info(f"Tracking results saved to: '{tracks_txt_file.resolve()}'")


def _build_run_metadata(config: Dict, save_dir: Path) -> Dict:
    """Build a structured, human-readable record of the configuration this run actually used.

    Reads the config dict, which ``sync_args_with_config`` has already reconciled with the CLI
    flags and ``--set`` overrides, so the saved file reflects the run rather than the config file
    it started from.
    """
    main = config['main']
    ul = config['ultralytics']
    args = main['args']
    active_classes = ul.get('classes') or list(main.get('class_names', {}))
    class_mapping = main.get('class_names', {})

    return {
        'run': {
            'geotrax_version': __version__,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'source': str(args.source),
            'config': str(args.cfg),
            'output_folder': str(save_dir),
        },
        'model': {
            'configured': main.get('model_configured'),
            'resolved': ul.get('model'),
        },
        'class_names': {
            'source': main.get('class_names_source', 'unknown'),
            'mapping': {k: class_mapping[k] for k in sorted(active_classes) if k in class_mapping},
        },
        'extraction': {k: v for k, v in main.get('extraction', {}).items() if k != 'model'},
        'processing': main.get('processing', {}),
        'output': main.get('output', {}),
        'detection': {k: v for k, v in ul.items() if k in _INFERENCE_KEYS},
        'tracker': {
            'active': main.get('tracker_active'),
            'params': main.get('tracker_params', {}),
        },
        'stabilo': config['stabilo'],
        'georef': config['georef'],
        'paths': {
            'ortho_folder': getattr(args, 'ortho_folder', None),
            'master_folder': getattr(args, 'master_folder', None),
            'segmentation_folder': getattr(args, 'segmentation_folder', None),
        },
        'visualization': main.get('visualization', {}),
        'plotting': main.get('plotting', {}),
        'batch': main.get('batch', {}),
    }


def add_processing_args(group) -> dict:
    """
    Register the shared detection/frame-range CLI flags on the given argparse group.

    Used by both ``geotrax extract`` and ``geotrax batch`` so the two expose an identical set
    of processing options. Every flag defaults to ``None``; the returned ``dest -> CfgArg`` map
    tells ``sync_args_with_config`` which config key each one stands for.
    """
    paths = {}
    add_cfg_arg(group, '--model', '-m', nargs='+', metavar='MODEL', cfg='extraction.model', paths=paths, no_sync=True,
                help="Detection model to use: a local file path OR an 'hf://<org>/<repo>/<path/to/file>.pt' Hugging Face reference (auto-downloaded & cached).")
    add_cfg_arg(group, '--class-names', '-cn', nargs='+', metavar='ID=NAME|FILE', cfg='extraction.class_rename', paths=paths, no_sync=True,
                help="Rename class-id -> name labels: a .yaml/.json mapping file or inline ID=NAME pairs (e.g. -cn 0=car 1=bus).",
                default_note="then the model's own names")
    add_cfg_arg(group, '--conf', '-co', type=float, cfg='ultralytics.conf', paths=paths,
                help='Detection confidence threshold.')
    add_cfg_arg(group, '--classes', '-cls', nargs='+', type=int, cfg='ultralytics.classes', paths=paths,
                help='Class IDs to extract (e.g., --classes 0 1 2).')
    add_cfg_arg(group, '--cut-frame-left', '-cfl', type=int, cfg='processing.cut_frame_left', paths=paths,
                help='Skip the first N frames.')
    add_cfg_arg(group, '--cut-frame-right', '-cfr', type=int, cfg='processing.cut_frame_right', paths=paths,
                help='Stop processing after this frame.')
    add_cfg_arg(group, '--interpolate', action=argparse.BooleanOptionalAction, cfg='extraction.interpolate', paths=paths,
                help='Fill per-track frame gaps with linear interpolation; adds is_interpolated column to output.')
    add_cfg_arg(group, '--sahi', action=argparse.BooleanOptionalAction, cfg='extraction.sahi.enable', paths=paths,
                help="Detect via SAHI sliced inference for improved small-object recall (requires: pip install 'geo-trax[sahi]'). Slicing parameters live in cfg -> extraction -> sahi.")
    add_cfg_arg(group, '--stab-gpu', '-sg', action=argparse.BooleanOptionalAction, cfg='stabilo.gpu', paths=paths,
                help='CUDA-accelerate stabilization (requires a CUDA-enabled OpenCV build; no CPU fallback).')
    add_cfg_arg(group, '--stab-gpu-device-id', '-sgid', type=int, cfg='stabilo.gpu_device_id', paths=paths,
                help='CUDA device index used when stabilization GPU is enabled.')
    add_cfg_arg(group, '--stab-detector', '-sdet', choices=DETECTOR_CHOICES, cfg='stabilo.detector_name', paths=paths,
                help="Stabilization feature detector. Classical (OpenCV): orb, sift, rsift, brisk, kaze, akaze. Learning-based (kornia, use --stab-device): xfeat, disk, dedode, keynet, loftr. All learned ones except keynet are upright models (matching collapses past ~30 deg rotation) and are memory hungry at high resolution - lower cfg -> stabilo -> downsample_ratio for them.")
    add_cfg_arg(group, '--stab-device', '-sdev', choices=DEVICE_CHOICES, cfg='stabilo.device', paths=paths,
                help="Torch device for the learning-based stabilization detectors/matchers ('auto' picks cuda > mps > cpu); ignored by the classical detectors and independent of --stab-gpu (OpenCV CUDA).")
    return paths


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Vehicle Detection, Tracking, and Stabilization Pipeline')

    parser.add_argument('source', type=Path, help='Path to the input video file.')

    optional = parser.add_argument_group('Optional arguments')
    cfg_paths = add_common_args(optional)

    processing = parser.add_argument_group('Processing arguments',
        'For full detection and tracking control (model, IoU, image size, tracker settings, etc.), '
        "use --set (e.g. --set iou=0.6) or edit cfg -> ultralytics and cfg -> tracker in the "
        "pipeline config (run 'geotrax config copy').")
    cfg_paths |= add_processing_args(processing)

    return finalize_cli_args(parser, cfg_paths)

def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(__name__, args.verbose, args.log_path)

    try:
        detect_track_stabilize(args, logger)
    except ExtractionError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
