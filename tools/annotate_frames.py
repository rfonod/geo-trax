#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
annotate_frames.py - Generate YOLO-format annotations for vehicle detection in images.

This script runs vehicle detection on images using YOLOv8 models and generates YOLO-format
annotation files with normalized bounding box coordinates. Optionally saves images with
blacked-out vehicle regions for visualization purposes.

Usage:
    python tools/annotate_frames.py <source> [options]

Arguments:
    source                        : Path to the directory containing images to be annotated.

Options:
    -h, --help                    : Show this help message and exit.
    -a, --annotations <path>      : Custom path to save the annotation files. If not provided, defaults to
                                    '<source>/../pre-labels'.
    -c, --cfg <path>              : Path to the ultralytics configuration file
                                    (default: cfg/ultralytics/annotator/default.yaml).
    -s, --save                    : Save images with blacked-out vehicle regions for visualization.
    -m, --margin <float>          : Margin factor to enlarge bounding boxes for visualization only
                                    (default: 0.0).

Examples:
  1. Generate annotations for images in a directory:
     python tools/annotate_frames.py path/to/images/

  2. Generate annotations and save visualization images with enlarged bounding boxes:
     python tools/annotate_frames.py path/to/images/ --save --margin 0.2

Notes:
  - Generates YOLO-format .txt files with: class_id x_center y_center width height (normalized to [0,1])
  - The margin parameter only affects visualization, not the annotation coordinates
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yolo

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.utils import load_config

LOGGER_PREFIX = f'[{Path(__file__).name}]'


def run_annotator(args, logger=None) -> None:
    """Run the annotator."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if args.cfg:
        config = load_config(args.cfg, logger)
    else:
        logger.error(f"{LOGGER_PREFIX} Error loading the configuration.")
        return

    model = load_detector(config, logger)

    output_dir = args.annotations if args.annotations else args.source.parent / "pre-labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # det_results = det_model(data, stream=True, device=device)
    det_results = model(args.source, **config, stream=True)

    logger.info(f"{LOGGER_PREFIX} Annotating images in '{args.source}'...")
    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xywhn

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for box, class_id in zip(boxes, class_ids):
                    f.write(f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}\n")

            if args.save:
                boxes = result.boxes.xywh
                img = cv2.imread(result.path)
                for box in boxes:
                    xc, yc, w, h = box.int().tolist()
                    w = int(w * (1 + args.margin))
                    h = int(h * (1 + args.margin))
                    x = int(xc - w / 2)
                    y = int(yc - h / 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                cv2.imwrite(f"{Path(output_dir) / Path(result.path).name}", img)

    print(f"Annotations saved to '{output_dir}'.")


def load_detector(config: Dict, logger: logging.Logger) -> YOLO:
    """Load the detection model."""
    try:
        model = YOLO(model=config['model'], task=config['task'])
    except Exception as e:
        logger.error(f"{LOGGER_PREFIX} Error loading detection model: {e}")
        return
    else:
        logger.info(f"{LOGGER_PREFIX} Detection model '{config['model']}' loaded successfully.")

    check_yolo(device=config['device'])

    return model


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Annotate images with vehicle bounding boxes.')
    parser.add_argument('source', type=Path, help='Path to the images to be annotated')
    parser.add_argument('--annotations', '-a', type=Path, help='Path to save the annotations (default: <source>/../pre-labels)')
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/ultralytics/annotator/default.yaml', help='Path to the ultralytics configuration file')
    parser.add_argument('--save', '-s', action='store_true', help='Save the annotated images')
    parser.add_argument('--margin', '-m', type=float, default=0.0, help='Margin factor to enlarge the bounding boxes (for plotting only)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    run_annotator(args)
