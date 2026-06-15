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
  -cm, --class-map <ID=Name> [...] : Class ID-to-label pairs entered directly on the CLI
                                (e.g. -cm 0=Car 1=Bus 2=Truck). Takes priority over all other sources.
  -mf, --map-file <path>      : YAML or JSON file mapping class IDs to labels
                                (e.g. {0: Car, 1: Bus}). Takes priority over --cfg.
  -c, --cfg <path>            : Pipeline config used to resolve the model when neither --class-map
                                nor --map-file is provided (default: bundled geotrax/cfg/default.yaml).
                                A bundled preset name (default, confident, lenient, stable) also works.
  -dp, --decimal_places <int> : Decimal places for rounding bounding-box coordinates (default: 2).
  -lp, --log-path <str>       : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                 : Reduce console verbosity to important messages only (default: show INFO-level detail).

Class map resolution order (first match wins):
  1. --class-map  : inline ID=Name pairs on the command line
  2. --map-file   : a YAML or JSON file with the mapping
  3. --cfg        : class names are extracted from the YOLO model referenced in the pipeline config

Examples:
1. Convert using class names from the bundled default model (via config):
   python tools/yolo_to_coco.py path/to/labels/

2. Provide the class map directly on the CLI:
   python tools/yolo_to_coco.py path/to/labels/ -cm 0=Car 1=Bus 2=Truck 3=Motorcycle

3. Load the class map from a YAML or JSON file:
   python tools/yolo_to_coco.py path/to/labels/ -mf models/class_map.yaml

4. Convert with a custom images path and a different model via a custom config:
   python tools/yolo_to_coco.py path/to/labels/ -ii ../imgs -c path/to/custom_config.yaml

Input Format:
  YOLO format: class_id center_x center_y width height (coordinates normalized to [0, 1]).

Output Format:
  COCO JSON files with absolute pixel coordinates and class labels.

Notes:
  - Input images must exist for the script to calculate absolute pixel coordinates.
  - The script skips labels without corresponding images.
  - When none of the class map options are provided, class names are extracted from the YOLO
    model embedded in the pipeline config. If loading fails, class IDs are used as labels.
  - Map files (--map-file) may be YAML or JSON; keys may be integers or strings.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import yaml
from ultralytics import YOLO

from geotrax.utils.cli_utils import DEFAULT_CFG
from geotrax.utils.config_utils import load_config, resolve_asset_path
from geotrax.utils.logging_utils import setup_logger

# Image file formats to consider (case-insensitive)
IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_class_map(pairs: List[str], logger: logging.Logger) -> Dict[int, str]:
    """Parse ID=Name pairs (e.g. ['0=Car', '1=Bus']) into a {class_id: label} dict."""
    class_map = {}
    for pair in pairs:
        try:
            id_str, name = pair.split('=', 1)
            class_map[int(id_str)] = name
        except ValueError:
            logger.warning(f"Skipping invalid --class-map entry '{pair}'. Expected format: ID=Name (e.g. 0=Car).")
    return class_map


def load_class_map_from_file(filepath: Path, logger: logging.Logger) -> Dict[int, str]:
    """Load a class ID to label mapping from a YAML or JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f) if filepath.suffix.lower() == '.json' else yaml.safe_load(f)
        class_map = {int(k): v for k, v in data.items()}
        logger.info(f"Class map loaded from: '{filepath}'.")
        return class_map
    except Exception as e:
        logger.error(f"Error loading class map file '{filepath}': {e}. Using default class mapping.")
        return {}


def resolve_class_map(args: argparse.Namespace, logger: logging.Logger) -> Dict[int, str]:
    """Resolve class ID-to-label mapping from the highest-priority available source."""
    if args.class_map:
        return parse_class_map(args.class_map, logger)
    if args.map_file:
        return load_class_map_from_file(args.map_file, logger)
    config = load_config(args.cfg, logger)
    model_path = resolve_asset_path(config.get('ultralytics', config)['model'])
    try:
        names = YOLO(str(model_path)).names
        logger.info(f"Class names loaded from model: '{model_path}'.")
        return names
    except Exception as e:
        logger.error(f"Error loading class names from model '{model_path}': {e}. Class IDs will be used as labels.")
        return {}


def convert_annotations(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Convert YOLO annotations to COCO format."""
    labels_dir = args.input_labels
    output_dir = args.output_labels if args.output_labels else labels_dir
    decimal_places = args.decimal_places

    images_dir = output_dir / args.input_images
    if not images_dir.is_dir():
        logger.error(f"Input images path '{images_dir}' is not a directory.")
        return

    image_paths = [f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_FORMATS]
    if not image_paths:
        logger.error(f"No image files found in input directory '{images_dir}'.")
        return

    label_paths = [f for f in labels_dir.rglob("*") if f.is_file() and f.suffix.lower() == ".txt"]
    if not label_paths:
        logger.error(f"No label files found in input directory '{labels_dir}'.")
        return

    if len(image_paths) != len(label_paths):
        logger.warning(f"Number of images ({len(image_paths)}) and labels ({len(label_paths)}) do not match.")

    class_id_to_label = resolve_class_map(args, logger)

    logger.notice(f"Found {len(image_paths)} images and {len(label_paths)} label files.")
    logger.info(f"Converting annotations with {decimal_places} decimal places precision...")

    processed_count = 0
    skipped_count = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"Label file '{label_path}' not found. Skipping image '{image_path.name}'.")
            skipped_count += 1
            continue

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

        with open(label_path, "r") as label_file:
            for annotation_line in label_file:
                parts = annotation_line.strip().split(" ")
                if len(parts) < 5:
                    logger.warning(f"Invalid line in label file '{label_path}': {annotation_line.strip()}")
                    continue

                class_id, x, y, w, h = parts[:5]
                class_id = int(class_id)
                x, y, w, h = float(x), float(y), float(w), float(h)

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

        output_dir.mkdir(parents=True, exist_ok=True)
        coco_path = output_dir / f"{image_path.stem}.json"
        with open(coco_path, "w") as json_file:
            json.dump(coco_annotations, json_file, indent=2)

        processed_count += 1
        if processed_count % 10 == 0:
            logger.info(f"Processed {processed_count} images...")

    logger.notice(f"Conversion complete: {processed_count} files processed, {skipped_count} files skipped.")


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the YOLO to COCO conversion script."""
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format.")

    parser.add_argument("input_labels", type=Path,
                        help="Directory containing YOLO format annotation files (.txt)")
    parser.add_argument("--input_images", "-ii", type=Path, default='../images',
                        help="Relative path to images with respect to the input labels directory (default: '../images')")
    parser.add_argument("--output_labels", "-ol", type=Path,
                        help="Path to save COCO annotations (default: same as input_labels)")

    map_group = parser.add_argument_group("class map (first match wins)")
    map_group.add_argument("--class-map", "-cm", nargs='+', metavar='ID=Name',
                           help="Inline class ID-to-label pairs, e.g. -cm 0=Car 1=Bus 2=Truck")
    map_group.add_argument("--map-file", "-mf", type=Path,
                           help="YAML or JSON file mapping class IDs to labels")
    map_group.add_argument("--cfg", "-c", type=Path, default=DEFAULT_CFG,
                           help="Pipeline config whose 'ultralytics.model' is used as the class name source "
                                "(default: bundled geotrax/cfg/default.yaml; a preset name also works)")

    parser.add_argument("--decimal_places", "-dp", type=int, default=2,
                        help="Decimal places for rounding bounding-box coordinates (default: 2)")
    parser.add_argument("--log-path", "-lp", type=Path, default=None,
                        help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce console verbosity to important messages only (default: show INFO-level detail).")

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)
    convert_annotations(args, logger)


if __name__ == '__main__':
    main()
