#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Authors: Haechan Cho (gkqkemwh@kaist.ac.kr)

"""
viz_segmentations.py - Lane Segmentation Visualization Tool

This script overlays lane segmentation data onto orthophoto images by drawing polygonal lane boundaries
and labeling them with lane and section identifiers. It processes orthophotos and their corresponding
CSV segmentation files to create annotated visualization images.

The tool draws red contours for individual lanes with lane IDs and blue labels for section identifiers,
providing a clear visual representation of the road network structure.

Usage:
  python tools/viz_segmentations.py --ortho-folder <path> --segmentation-path <path> [options]

Arguments:
  --ortho-folder <path>      : Path to folder containing orthophoto files (.png, .tif, .tiff).
  --segmentation-path <path> : Path to folder containing lane segmentation CSV files.

Options:
  -h, --help                 : Show this help message and exit.
  --ortho-suffix <str>       : File extension of orthophoto files (default: png).

Input:
- Orthophoto images in specified format (.png, .tif, .tiff)
- CSV segmentation files with matching filenames containing lane polygon coordinates
- CSV format: columns for Section, lane ID, and 4 polygon corner coordinates (x1,y1,x2,y2,x3,y3,x4,y4)

Output:
- Annotated orthophoto images saved to ortho_folder/segmentations/
- Red polygonal lane boundaries with lane ID labels
- Blue section labels positioned at section centers
- PNG format output images

Notes:
- CSV files must have the same base filename as corresponding orthophoto files
- Lane polygons are drawn as 4-point contours with 15-pixel red borders
- Section labels are positioned at the geometric center of middle lanes
- Output images are saved in a 'segmentations' subfolder within the orthophoto folder
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point


def process_ortho_images(ortho_folder, segmentation_path, ortho_suffix):
    """
    Draw the lane segmentation for each orthophoto in the ortho_folder.
    """
    ortho_files = list(ortho_folder.glob(f"*.{ortho_suffix}"))

    for ortho_file in ortho_files:
        segmentation_file = segmentation_path / f"{ortho_file.stem}.csv"

        if not segmentation_file.exists():
            print(f"No segmentation file found for {ortho_file.name}. Skipping...")
            continue

        ortho = cv2.imread(str(ortho_file))
        lanes = pd.read_csv(segmentation_file)

        for i in range(len(lanes)):
            Poly = np.array([[
                [lanes.iloc[i, 2], lanes.iloc[i, 3]],
                [lanes.iloc[i, 4], lanes.iloc[i, 5]],
                [lanes.iloc[i, 6], lanes.iloc[i, 7]],
                [lanes.iloc[i, 8], lanes.iloc[i, 9]]
            ]])

            center = Point(int((lanes.iloc[i, 2] + lanes.iloc[i, 4] + lanes.iloc[i, 6] + lanes.iloc[i, 8]) / 4),
                           int((lanes.iloc[i, 3] + lanes.iloc[i, 5] + lanes.iloc[i, 7] + lanes.iloc[i, 9]) / 4))

            ortho = cv2.drawContours(ortho, Poly, -1, (0, 0, 255), 15)
            text = str(lanes.iloc[i, 1])
            ortho = cv2.putText(ortho, text, (int(center.x)-30, int(center.y)+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_8)

        sections = np.unique(lanes['Section'])
        for section in sections:
            section_lanes = lanes[lanes['Section'] == section].reset_index(drop=True)
            mid = int(len(section_lanes) / 2)
            center = Point(int((section_lanes.iloc[mid, 2] + section_lanes.iloc[mid, 4] + section_lanes.iloc[mid, 6] + section_lanes.iloc[mid, 8]) / 4),
                           int((section_lanes.iloc[mid, 3] + section_lanes.iloc[mid, 5] + section_lanes.iloc[mid, 7] + section_lanes.iloc[mid, 9]) / 4))
            text = str(section)
            ortho = cv2.putText(ortho, text, (int(center.x) - 160, int(center.y)+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8, cv2.LINE_AA)

        output_path = ortho_folder / "segmentations" / f"{ortho_file.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), ortho)
        print(f"Processed and saved orthophoto for {ortho_file.stem}")


def parse_opt():
    parser = argparse.ArgumentParser(description="Visualize lane segmentations on orthophoto images.")
    parser.add_argument('--ortho-folder', type=Path, required=True, help='Path to the folder containing orthophoto files (.png, .tif, .tiff).')
    parser.add_argument('--segmentation-path', type=Path, required=True, help='Path to the folder containing lane and road segmentation csv files.')
    parser.add_argument('--ortho-suffix', type=str, default='png', help='Suffix of the orthophoto files (e.g., "png", "tif").')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    process_ortho_images(**vars(opt))
