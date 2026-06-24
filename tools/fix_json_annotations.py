#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
fix_json_annotations.py - Clean and fix JSON annotation files for COCO-like datasets.

This script processes JSON annotation files (typically from LabelMe or similar tools) to:
- Optionally remove embedded base64 image data to reduce file size
- Normalize image paths for cross-platform compatibility (Windows ↔ Unix)
- Modify image paths (remove or replace substrings)
- Convert between bounding box formats (HBB ↔ OBB)

Usage:
    python tools/fix_json_annotations.py <labels_dir> [options]

Arguments:
    labels_dir                      : Path to directory containing JSON annotation files.

Options:
    -h, --help                      : Show this help message and exit.

    Image Data:
    --remove-image-data, -ri        : Remove embedded base64 image data to reduce file size.

    Bounding Box Conversion:
    --to-obb                        : Convert HBB (axis-aligned) rectangles to OBB (oriented) 4-point polygons.
    --to-hbb                        : Convert OBB polygons to HBB axis-aligned rectangles.

    Path Normalization:
    --normalize-to-unix, -nu        : Normalize paths to Unix format (backslash → forward slash).
    --normalize-to-windows, -nw     : Normalize paths to Windows format (forward slash → backslash).

    Path Modification:
    --remove-from-path, -r <str>    : Remove specified substring from all image paths.
    --replace-path, -p <old> <new>  : Replace substring <old> with <new> in all image paths.

    Debugging:
    --debug, -d                     : Preview changes without modifying files.

    Logging:
    --log-path, -lp <str>           : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
    --quiet, -q                     : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
  1. Remove embedded image data to reduce file size:
     python tools/fix_json_annotations.py path/to/annotations/ --remove-image-data

  2. Convert HBB rectangles to OBB 4-point polygons:
     python tools/fix_json_annotations.py path/to/annotations/ --to-obb

  3. Convert OBB polygons to HBB rectangles:
     python tools/fix_json_annotations.py path/to/annotations/ --to-hbb

  4. Normalize paths for Unix/Linux/macOS systems:
     python tools/fix_json_annotations.py path/to/annotations/ --normalize-to-unix

  5. Normalize paths for Windows systems:
     python tools/fix_json_annotations.py path/to/annotations/ --normalize-to-windows

  6. Remove unwanted path prefix:
     python tools/fix_json_annotations.py path/to/annotations/ -r "old_folder/"

  7. Update directory structure in paths:
     python tools/fix_json_annotations.py path/to/annotations/ -p "/old/path/" "/new/path/"

  8. Combine operations (remove image data, normalize paths, convert to OBB):
     python tools/fix_json_annotations.py path/to/annotations/ --remove-image-data --normalize-to-unix --to-obb

  9. Preview changes without modifying files:
     python tools/fix_json_annotations.py path/to/annotations/ --remove-image-data --debug

Notes:
  - Recursively processes all .json files in the specified directory
  - Image data removal sets 'imageData' field to null, significantly reducing file size
  - Path normalization converts between Windows (\\) and Unix (/) separators
  - Path operations applied in order: normalize → remove → replace
  - HBB to OBB: Creates 4-point polygons [top-left, bottom-left, bottom-right, top-right]
  - OBB to HBB: Creates axis-aligned rectangles using bounding box of polygon points
  - Conversion options (--to-obb, --to-hbb) are mutually exclusive
  - Normalization options (--normalize-to-unix, --normalize-to-windows) are mutually exclusive
  - All operations preserve original annotation data unless explicitly modified
  - Use --debug to safely preview all changes before applying them
"""

import argparse
import json
import logging
from pathlib import Path

from geotrax.utils.logging_utils import setup_logger


def process_input(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Remove image data from COCO annotations and correct the image path.
    """

    # Check if the labels path is provided
    if not args.labels_dir:
        logger.error("Labels path not provided.")
        return

    # Load all label paths
    label_paths = [f for f in args.labels_dir.rglob("*") if f.is_file() and f.suffix.lower() == ".json"]

    # Check if there are any label files in the input directory
    if not label_paths:
        logger.error(f"No label files found in input directory '{args.labels_dir}'.")
        return

    logger.notice(f"Found {len(label_paths)} JSON annotation files in '{args.labels_dir}'.")
    if args.debug:
        logger.info("Running in DEBUG mode - no files will be modified.")
    if args.remove_image_data:
        logger.info("Image data removal enabled - will remove embedded base64 image data.")
    if args.to_obb:
        logger.info("HBB to OBB conversion - rectangles will be converted to 4-point polygons.")
    if args.to_hbb:
        logger.info("OBB to HBB conversion - polygons will be converted to axis-aligned rectangles.")
    if args.normalize_to_unix:
        logger.info("Path normalization to Unix format - converting backslashes to forward slashes.")
    if args.normalize_to_windows:
        logger.info("Path normalization to Windows format - converting forward slashes to backslashes.")
    if args.remove_from_path:
        logger.info(f"Path substring removal - will remove '{args.remove_from_path}' from image paths.")
    if args.replace_path:
        logger.info(f"Path substring replacement - will replace '{args.replace_path[0]}' with '{args.replace_path[1]}' in image paths.")

    processed_count = 0
    image_data_removed_count = 0
    obb_converted_count = 0
    hbb_converted_count = 0
    path_modified_count = 0
    path_normalized_count = 0

    # Loop through all the label files
    for label_path in label_paths:
        # Load the COCO annotations
        with open(label_path, "r") as file:
            coco_annotations = json.load(file)

        had_image_data = coco_annotations.get("imageData") is not None

        # Remove the image data if requested
        if args.remove_image_data and had_image_data:
            coco_annotations["imageData"] = None
            image_data_removed_count += 1

        # Normalize paths for cross-platform compatibility
        if args.normalize_to_unix and "\\" in coco_annotations["imagePath"]:
            old_path = coco_annotations["imagePath"]
            coco_annotations["imagePath"] = coco_annotations["imagePath"].replace("\\", "/")
            path_normalized_count += 1
            logger.info(f"  Path normalized to Unix: '{old_path}' -> '{coco_annotations['imagePath']}'")

        if args.normalize_to_windows and "/" in coco_annotations["imagePath"]:
            old_path = coco_annotations["imagePath"]
            coco_annotations["imagePath"] = coco_annotations["imagePath"].replace("/", "\\")
            path_normalized_count += 1
            logger.info(f"  Path normalized to Windows: '{old_path}' -> '{coco_annotations['imagePath']}'")

        # Remove substring from the image path
        if args.remove_from_path and args.remove_from_path in coco_annotations["imagePath"]:
            old_path = coco_annotations["imagePath"]
            coco_annotations["imagePath"] = coco_annotations["imagePath"].replace(args.remove_from_path, "")
            path_modified_count += 1
            logger.info(f"  Path modified: '{old_path}' -> '{coco_annotations['imagePath']}'")

        # Replace part of the image path
        if args.replace_path and args.replace_path[0] in coco_annotations["imagePath"]:
            old_path = coco_annotations["imagePath"]
            coco_annotations["imagePath"] = coco_annotations["imagePath"].replace(args.replace_path[0], args.replace_path[1])
            path_modified_count += 1
            logger.info(f"  Path replaced: '{old_path}' -> '{coco_annotations['imagePath']}'")

        # Convert between HBB and OBB formats
        rectangles_converted = 0
        polygons_converted = 0

        if args.to_obb:
            for annotation in coco_annotations["shapes"]:
                if annotation["shape_type"] == "rectangle":
                    x_TL, y_TL = annotation["points"][0]
                    x_BR, y_BR = annotation["points"][1]
                    x_BL, y_BL = x_TL, y_BR
                    x_TR, y_TR = x_BR, y_TL
                    annotation["shape_type"] = "polygon"
                    annotation["points"] = [[x_TL, y_TL], [x_BL, y_BL], [x_BR, y_BR], [x_TR, y_TR]]
                    rectangles_converted += 1

            if rectangles_converted > 0:
                obb_converted_count += 1
                logger.info(f"  Converted {rectangles_converted} HBB rectangle(s) to OBB polygon(s) in '{label_path.name}'")

            # Check if all "points" contain exactly 4 points
            for annotation in coco_annotations["shapes"]:
                if annotation["shape_type"] == "polygon" and len(annotation["points"]) != 4:
                    logger.error(f"Polygon in '{label_path}' does not contain exactly 4 points.")

        if args.to_hbb:
            for annotation in coco_annotations["shapes"]:
                if annotation["shape_type"] == "polygon":
                    # Get bounding box from polygon points
                    xs = [point[0] for point in annotation["points"]]
                    ys = [point[1] for point in annotation["points"]]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    annotation["shape_type"] = "rectangle"
                    annotation["points"] = [[x_min, y_min], [x_max, y_max]]
                    polygons_converted += 1

            if polygons_converted > 0:
                hbb_converted_count += 1
                logger.info(f"  Converted {polygons_converted} OBB polygon(s) to HBB rectangle(s) in '{label_path.name}'")

        # Save the COCO annotations
        if not args.debug:
            with open(label_path, "w") as file:
                json.dump(coco_annotations, file, indent=2)

        logger.info(f"Processed '{label_path.name}'")

        processed_count += 1

    summary = [f"Total files processed: {processed_count}"]
    if args.remove_image_data:
        summary.append(f"Files with image data removed: {image_data_removed_count}")
    if args.normalize_to_unix or args.normalize_to_windows:
        summary.append(f"Files with normalized paths: {path_normalized_count}")
    if args.remove_from_path or args.replace_path:
        summary.append(f"Files with modified paths: {path_modified_count}")
    if args.to_obb:
        summary.append(f"Files with HBB to OBB conversions: {obb_converted_count}")
    if args.to_hbb:
        summary.append(f"Files with OBB to HBB conversions: {hbb_converted_count}")
    summary.append("Note: No files were modified (debug mode)" if args.debug else "All changes saved successfully.")
    logger.notice("Summary:\n  %s", "\n  ".join(summary))


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Clean and fix JSON annotation files for COCO-like datasets")
    parser.add_argument("labels_dir", type=Path, help="Path to directory containing JSON annotation files")

    # Image data options
    parser.add_argument("--remove-image-data", "-ri", action="store_true", help="Remove embedded base64 image data to reduce file size")

    # Bounding box conversion options (mutually exclusive)
    conversion_group = parser.add_mutually_exclusive_group()
    conversion_group.add_argument("--to-obb", "-to", action="store_true", help="Convert HBB rectangles to OBB polygons (4-point)")
    conversion_group.add_argument("--to-hbb", "-th", action="store_true", help="Convert OBB polygons to HBB axis-aligned rectangles")

    # Path normalization options (mutually exclusive)
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize-to-unix", "-nu", action="store_true", help="Normalize paths to Unix format (backslash to forward slash)")
    normalize_group.add_argument("--normalize-to-windows", "-nw", action="store_true", help="Normalize paths to Windows format (forward slash to backslash)")

    parser.add_argument("--remove-from-path", "-r", type=str, help="Remove specified substring from image paths")
    parser.add_argument("--replace-path", "-p", nargs=2, metavar=("OLD", "NEW"), help="Replace substring in image paths")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode - show changes without modifying files")
    parser.add_argument("--log-path", "-lp", type=Path, default=None, help="Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console verbosity to important messages only (default: show INFO-level detail).")

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    process_input(args, logger)


if __name__ == "__main__":
    main()
