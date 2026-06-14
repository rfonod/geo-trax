#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
yolo_to_coco.py - YOLO to COCO Annotation Converter

Converts YOLO-format annotations to COCO-style JSON for easier visualization and compatibility.
It transforms normalized YOLO coordinates into absolute pixel coordinates, preserving bounding
box information and class labels so the annotations work with COCO-compatible visualization tools.

Usage:
  python tools/yolo_to_coco.py <input_labels> [options]

Arguments:
  input_labels : Directory containing YOLO format annotation files (.txt).

Options:
  -h, --help                  : Show this help message and exit.
  -ii, --input_images <path>  : Relative path to images w.r.t. the output labels directory (default: '../images').
  -ol, --output_labels <path> : Path to save COCO annotations (default: same as input_labels).
  -cm, --class_map <path>     : YAML file mapping class IDs to labels (default: 'models/yolov8s_merger8_exp1.yaml').
  -dp, --decimal_places <int> : Decimal places for rounding bounding-box coordinates (default: 2).
  -lp, --log-path <str>       : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                 : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Convert YOLO labels (images in ../images relative to labels):
   python tools/yolo_to_coco.py path/to/labels/

2. Convert with a custom images path and class map:
   python tools/yolo_to_coco.py path/to/labels/ -ii ../imgs -cm models/custom.yaml

Input Format:
  YOLO format: class_id center_x center_y width height (coordinates normalized to [0, 1]).

Output Format:
  COCO JSON files with absolute pixel coordinates and class labels.

Notes:
  - Input images must exist for the script to calculate absolute pixel coordinates.
  - Class map file should contain a dictionary mapping numeric IDs to string labels.
  - The script skips labels without corresponding images.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml

from geotrax.utils.logging_utils import setup_logger

# Image file formats to consider (case-insensitive)
IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_annotations(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Convert YOLO annotations to COCO format.
    """

    labels_dir = args.input_labels
    output_dir = args.output_labels if args.output_labels else labels_dir
    decimal_places = args.decimal_places
    class_map = args.class_map

    # Check if the input images path is a directory
    images_dir = output_dir / args.input_images
    if not images_dir.is_dir():
        logger.error(f"Input images path '{images_dir}' is not a directory.")
        return

    # Load all image paths
    image_paths = [f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_FORMATS]

    # Check if there are any image files in the input directory
    if not image_paths:
        logger.error(f"No image files found in input directory '{images_dir}'.")
        return

    # Load all label paths
    label_paths = [f for f in labels_dir.rglob("*") if f.is_file() and f.suffix.lower() == ".txt"]

    # Check if there are any label files in the input directory
    if not label_paths:
        logger.error(f"No label files found in input directory '{labels_dir}'.")
        return

    # Check if the number of images and labels match
    if len(image_paths) != len(label_paths):
        logger.warning(f"Number of images ({len(image_paths)}) and labels ({len(label_paths)}) do not match.")

    # Load class ID to label mapping
    try:
        with open(class_map, 'r') as f:
            class_id_to_label = yaml.safe_load(f)
        if not isinstance(class_id_to_label, dict):
            raise ValueError("Class map YAML file must contain a dictionary mapping class IDs to labels.")
    except Exception as e:
        logger.error(f"Error loading class map file '{class_map}': {e}")
        logger.warning("Using default class mapping.")
        class_id_to_label = {}

    logger.notice(f"Found {len(image_paths)} images and {len(label_paths)} label files.")
    logger.info(f"Converting annotations with {decimal_places} decimal places precision...")

    # Track processing statistics
    processed_count = 0
    skipped_count = 0

    # Loop through all the images and labels
    for image_path in image_paths:
        # Load the corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"Label file '{label_path}' not found. Skipping image '{image_path.name}'.")
            skipped_count += 1
            continue

        # Get image size
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Unable to read image '{image_path}'. Skipping.")
            skipped_count += 1
            continue

        height, width, _ = image.shape

        coco_annotations = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": str(args.input_images / image_path.name),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        # Read the label file
        with open(label_path, "r") as label_file:
            for annotation_line in label_file:
                parts = annotation_line.strip().split(" ")
                if len(parts) < 5:
                    logger.warning(f"Invalid line in label file '{label_path}': {annotation_line.strip()}")
                    continue

                class_id, x, y, w, h = parts[:5]
                class_id = int(class_id)

                x, y, w, h = float(x), float(y), float(w), float(h)

                # Convert normalized YOLO coordinates to absolute pixel coordinates
                x1 = round((x - w / 2) * width, decimal_places)
                y1 = round((y - h / 2) * height, decimal_places)
                x2 = round((x + w / 2) * width, decimal_places)
                y2 = round((y + h / 2) * height, decimal_places)

                coco_annotations["shapes"].append(
                    {
                        "label": class_id_to_label.get(class_id, str(class_id)),
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None,
                    }
                )

        # Save the COCO annotations
        output_dir.mkdir(parents=True, exist_ok=True)
        coco_path = output_dir / f"{image_path.stem}.json"
        with open(coco_path, "w") as json_file:
            json.dump(coco_annotations, json_file, indent=2)

        processed_count += 1
        if processed_count % 10 == 0:
            logger.info(f"Processed {processed_count} images...")

    logger.notice(f"Conversion complete: {processed_count} files processed, {skipped_count} files skipped.")


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the YOLO to COCO conversion script.
    """
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format.")

    parser.add_argument("input_labels", type=Path, help="Directory containing YOLO format annotation files (.txt)")
    parser.add_argument("--input_images", "-ii", type=Path, default='../images', help="Relative path to images with respect to the input labels directory. Default: '../images'")
    parser.add_argument("--output_labels", "-ol", type=Path, help="Path to save COCO annotations. If not provided, annotations will be saved in the input labels directory")
    parser.add_argument("--class_map", "-cm", type=Path, default="models/yolov8s_merger8_exp1.yaml", help="Path to YAML file mapping class IDs to labels")
    parser.add_argument("--decimal_places", "-dp", type=int, default=2, help="Number of decimal places for rounding bounding box coordinates. Default: 2")
    parser.add_argument("--log-path", "-lp", type=Path, default=None, help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console verbosity to important messages only (default: show INFO-level detail).")

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    convert_annotations(args, logger)


if __name__ == "__main__":
    main()
