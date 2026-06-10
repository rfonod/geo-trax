#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
annotate_frames.py - Generate YOLO-format annotations for vehicle detection in images.

This script runs vehicle detection on images using a YOLOv8 model and produces YOLO-format
annotation files with normalized bounding box coordinates. Optionally saves visualization
images with overlaid bounding boxes or masked images with blacked-out vehicle regions.

Usage:
    python tools/annotate_frames.py <source> [options]

Arguments:
    source                              : Path to the directory containing images to annotate.

Options:
    -h, --help                          : Show this help message and exit.
    -a, --annotations <path>            : Directory to save the annotation .txt files. Defaults to
                                          '<source>/../pre-labels'.
    -c, --cfg <path>                    : Path to the ultralytics configuration YAML file
                                          (default: cfg/ultralytics/annotator/default.yaml).
    -v, --save-viz                      : Save images with colored bounding boxes overlaid.
    -z, --viz-dir <path>                : Directory to save visualization images. Defaults to
                                          '<annotations>/visualizations' when --save-viz is set.
    -m, --save-masked                   : Save images with blacked-out vehicle regions.
    -g, --margin <float>                : Margin factor to enlarge bounding boxes for masked images
                                          only (default: 0.0).
    -f, --conf <float>                  : Override the confidence threshold from the config file.
    -i, --iou <float>                   : Override the IoU threshold for NMS from the config file.
    -k, --classes <ID> [<ID> ...]       : Restrict inference to the specified class IDs.
    -t, --class-conf <ID=THRESH> [...]  : Per-class confidence thresholds as CLASS_ID=THRESHOLD
                                          pairs (e.g. -t 0=0.3 1=0.5). Classes not listed fall
                                          back to the base --conf threshold. Applied as a
                                          post-inference filter affecting both annotations and
                                          visualizations.
    -s, --save-conf                     : Append the detection confidence score to each annotation
                                          line: class_id x_center y_center width height confidence.
    -hc, --hide-conf                    : Hide confidence scores on visualizations.
    -hl, --hide-labels                  : Hide class labels on visualizations.
    -w, --line-width <int>              : Line thickness for bounding boxes in visualizations
                                          (default: auto-scaled).

Examples:
  1. Generate annotations for images in a directory:
     python tools/annotate_frames.py path/to/images/

  2. Generate annotations and save visualization images:
     python tools/annotate_frames.py path/to/images/ --save-viz

  3. Save visualizations to a custom directory, without confidence scores and with thick lines:
     python tools/annotate_frames.py path/to/images/ -v -z path/to/viz/ -hc -w 3

  4. Override confidence and IoU thresholds:
     python tools/annotate_frames.py path/to/images/ --conf 0.4 --iou 0.5

  5. Restrict inference to class IDs 0 and 2 with per-class confidence thresholds:
     python tools/annotate_frames.py path/to/images/ -k 0 2 -t 0=0.3 2=0.6

  6. Generate masked images with enlarged bounding boxes:
     python tools/annotate_frames.py path/to/images/ --save-masked --margin 0.2

Notes:
  - Generates YOLO-format .txt files: class_id x_center y_center width height (normalized to [0,1]).
  - The --margin parameter only affects masked images, not annotation coordinates or visualizations.
  - Per-class thresholds (--class-conf) are applied as a post-inference filter on top of the base
    --conf threshold. Both annotations and visualizations reflect only the surviving detections.
  - When --save-conf is set, annotation lines use the extended format:
    class_id x_center y_center width height confidence
  - Visualizations use the ultralytics color scheme with distinct colors per object class.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yolo

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.config_utils import load_config

LOGGER_PREFIX = f'[{Path(__file__).name}]'


def parse_class_conf(pairs: List[str]) -> Dict[int, float]:
    """Parse CLASS_ID=THRESHOLD pairs (e.g. ['0=0.3', '1=0.5']) into a {class_id: threshold} dict."""
    class_conf: Dict[int, float] = {}
    for pair in pairs:
        try:
            class_id_str, threshold_str = pair.split('=')
            class_conf[int(class_id_str)] = float(threshold_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid --class-conf entry '{pair}'. Expected format: CLASS_ID=THRESHOLD (e.g. 0=0.3)."
            )
    return class_conf


def run_annotator(args, logger=None) -> None:
    """Run the annotator."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if args.cfg:
        config = load_config(args.cfg, logger)
    else:
        logger.error(f"{LOGGER_PREFIX} Error loading the configuration.")
        return

    # Apply CLI overrides to config
    if args.conf is not None:
        config['conf'] = args.conf
    if args.iou is not None:
        config['iou'] = args.iou
    if args.classes is not None:
        config['classes'] = args.classes

    # Parse per-class confidence thresholds
    class_conf = parse_class_conf(args.class_conf) if args.class_conf else {}
    base_conf = config.get('conf', 0.25)

    model = load_detector(config, logger)

    output_dir = Path(args.annotations) if args.annotations else args.source.parent / "pre-labels"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Resolve visualization and masked output directories
    if args.save_viz:
        viz_dir = Path(args.viz_dir) if args.viz_dir else output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
    if args.save_masked:
        masked_dir = output_dir / "masked"
        masked_dir.mkdir(exist_ok=True, parents=True)

    det_results = model(args.source, **config, stream=True)

    logger.info(f"{LOGGER_PREFIX} Annotating images in '{args.source}'...")
    for result in det_results:
        all_class_ids = result.boxes.cls.int().tolist()  # noqa
        if not all_class_ids:
            continue

        # Apply per-class confidence filtering (post-inference).
        # Filters result.boxes in place so that visualizations and masked images are consistent.
        if class_conf:
            confs = result.boxes.conf.tolist()
            keep_indices = [
                i for i, (cls_id, conf_val) in enumerate(zip(all_class_ids, confs))
                if conf_val >= class_conf.get(cls_id, base_conf)
            ]
            result.boxes = result.boxes[keep_indices]

        class_ids = result.boxes.cls.int().tolist()
        if not class_ids:
            continue

        boxes_xywhn = result.boxes.xywhn
        boxes_xywh = result.boxes.xywh
        box_confs = result.boxes.conf.tolist()

        # Save YOLO-format annotations
        with open(output_dir / f"{Path(result.path).stem}.txt", "w") as f:
            for box, class_id, conf_val in zip(boxes_xywhn, class_ids, box_confs):
                line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                if args.save_conf:
                    line += f" {conf_val:.6f}"
                f.write(line + "\n")

        # Save visualization with colored bounding boxes
        if args.save_viz:
            plot_args = {
                'conf': args.show_conf,
                'labels': args.show_labels,
            }
            if args.line_width is not None:
                plot_args['line_width'] = args.line_width
            annotated_img = result.plot(**plot_args)
            cv2.imwrite(str(viz_dir / Path(result.path).name), annotated_img)

        # Save masked images with blacked-out regions
        if args.save_masked:
            img = cv2.imread(result.path)
            for box in boxes_xywh:
                xc, yc, w, h = box.int().tolist()
                w = int(w * (1 + args.margin))
                h = int(h * (1 + args.margin))
                x = int(xc - w / 2)
                y = int(yc - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.imwrite(str(masked_dir / Path(result.path).name), img)

    logger.info(f"{LOGGER_PREFIX} Annotations saved to '{output_dir}'.")
    if args.save_viz:
        logger.info(f"{LOGGER_PREFIX} Visualizations saved to '{viz_dir}'.")
    if args.save_masked:
        logger.info(f"{LOGGER_PREFIX} Masked images saved to '{masked_dir}'.")


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
    parser = argparse.ArgumentParser(description='Annotate images with vehicle bounding boxes using a YOLOv8 model.')

    # Required
    parser.add_argument('source', type=Path,
                        help='Path to the directory containing images to annotate')

    # Output paths
    parser.add_argument('--annotations', '-a', type=Path,
                        help='Directory to save annotation .txt files (default: <source>/../pre-labels)')
    parser.add_argument('--cfg', '-c', type=Path, default='cfg/ultralytics/annotator/default.yaml',
                        help='Path to the ultralytics configuration YAML file')

    # Output modes
    parser.add_argument('--save-viz', '-v', action='store_true',
                        help='Save images with colored bounding boxes overlaid')
    parser.add_argument('--viz-dir', '-z', type=Path,
                        help='Directory to save visualization images (default: <annotations>/visualizations)')
    parser.add_argument('--save-masked', '-m', action='store_true',
                        help='Save images with blacked-out vehicle regions')
    parser.add_argument('--margin', '-g', type=float, default=0.0,
                        help='Margin factor to enlarge bounding boxes for masked images only (default: 0.0)')
    parser.add_argument('--save-conf', '-s', action='store_true',
                        help='Append detection confidence to each annotation line')

    # Inference overrides
    parser.add_argument('--conf', '-f', type=float, default=None,
                        help='Override the confidence threshold from the config file')
    parser.add_argument('--iou', '-i', type=float, default=None,
                        help='Override the IoU threshold for NMS from the config file')
    parser.add_argument('--classes', '-k', type=int, nargs='+', metavar='CLASS_ID',
                        help='Restrict inference to the given class IDs (e.g. -k 0 1 2)')
    parser.add_argument('--class-conf', '-t', nargs='+', metavar='CLASS_ID=THRESHOLD',
                        help='Per-class confidence thresholds, e.g. -t 0=0.3 1=0.5. '
                             'Classes not listed fall back to the base --conf threshold.')

    # Visualization style
    parser.add_argument('--hide-conf', '-hc', dest='show_conf', action='store_false', default=True,
                        help='Hide confidence scores on visualizations')
    parser.add_argument('--hide-labels', '-hl', dest='show_labels', action='store_false', default=True,
                        help='Hide class labels on visualizations')
    parser.add_argument('--line-width', '-w', type=int, default=None,
                        help='Line thickness for bounding boxes in visualizations (default: auto-scaled)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    run_annotator(args)
