#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Haechan Cho (gkqkemwh@kaist.ac.kr)

"""
viz_segmentations.py - Road Segmentation Visualization Tool

Overlays lane and road-section segmentation data onto orthophoto images. For each
orthophoto that has a matching CSV file, the script draws red polygonal lane contours
with lane-number labels and blue section identifiers, then saves the annotated image.

Usage:
  python tools/viz_segmentations.py <ortho_folder> [options]

Arguments:
  ortho_folder               : Path to folder containing orthophoto images.

Options:
  -h, --help                 : Show this help message and exit.
  -sf, --seg-folder <path>   : Path to folder containing lane segmentation CSV files
                               (default: <ortho_folder>/segmentations).
  -o, --output <path>        : Output folder for annotated images
                               (default: same as --seg-folder).
  -e, --ext <str>            : File extension of orthophoto images (default: png).
  -lp, --log-path <str>      : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                : Reduce console verbosity to important messages only (default: show INFO-level detail).

Input:
  - Orthophoto images (PNG/TIF/TIFF) in <ortho_folder>.
  - CSV segmentation files with columns:
      Section, Lane, tlx, tly, blx, bly, brx, bry, trx, try
    where tl/bl/br/tr are top-left, bottom-left, bottom-right, top-right corner coordinates.

Output:
  - Annotated orthophotos saved as PNG files to <output>.
  - Red polygonal lane boundaries labeled with lane numbers.
  - Blue section identifiers placed at the center of each section's middle lane.

Notes:
  - CSV files must share the same filename stem as their orthophoto (e.g., A.csv <-> A.png).
  - Orthophotos without a matching CSV are silently skipped.

Examples:
  1. Default run — CSVs and output both resolve to <ortho_folder>/segmentations/:
       python tools/viz_segmentations.py data/orthophotos/

  2. Explicit segmentation folder and output:
       python tools/viz_segmentations.py data/orthophotos/ -sf data/segmentations/ -o data/output/

  3. Process TIFF orthophotos:
       python tools/viz_segmentations.py data/orthophotos/ -e tif
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geotrax.utils.logging_utils import setup_logger

LANE_COLOR = (0, 0, 255)
SECTION_COLOR = (255, 0, 0)
LANE_BORDER = 15
LANE_LABEL_SCALE = 3.0
LANE_LABEL_THICKNESS = 3
SECTION_LABEL_SCALE = 4.0
SECTION_LABEL_THICKNESS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def visualize_segmentations(ortho_folder, seg_folder, output, ext, logger: logging.Logger):
    """Overlay lane segmentations onto each orthophoto and write annotated images."""
    ext = ext.lstrip('.')
    ortho_files = sorted(ortho_folder.glob(f'*.{ext}'))
    if not ortho_files:
        logger.warning(f'No *.{ext} files found in "{ortho_folder}".')
        return

    output.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for ortho_file in ortho_files:
        seg_file = seg_folder / f'{ortho_file.stem}.csv'
        if not seg_file.exists():
            logger.warning(f'No segmentation CSV for "{ortho_file.name}" — skipping.')
            continue

        img = cv2.imread(str(ortho_file))
        if img is None:
            logger.warning(f'Could not read "{ortho_file}" — skipping.')
            continue

        lanes = pd.read_csv(seg_file)
        _draw_lanes(img, lanes)
        _draw_sections(img, lanes)

        out_path = output / f'{ortho_file.stem}.png'
        cv2.imwrite(str(out_path), img)
        logger.info(f'Saved "{out_path.name}".')
        n_saved += 1

    if n_saved:
        logger.notice(f'Done — {n_saved} image(s) saved to "{output}".')
    else:
        logger.warning('No images were processed.')


def _draw_lanes(img, lanes):
    """Draw a filled red contour and lane-number label for every lane row."""
    for _, row in lanes.iterrows():
        poly = np.array([[
            [row['tlx'], row['tly']],
            [row['blx'], row['bly']],
            [row['brx'], row['bry']],
            [row['trx'], row['try']],
        ]], dtype=np.int32)
        cx, cy = _poly_center(row)
        cv2.drawContours(img, poly, -1, LANE_COLOR, LANE_BORDER)
        cv2.putText(img, str(int(row['Lane'])), (cx - 30, cy + 20),
                    FONT, LANE_LABEL_SCALE, LANE_COLOR, LANE_LABEL_THICKNESS, cv2.LINE_8)


def _draw_sections(img, lanes):
    """Draw a blue section label at the centre of each section's middle lane."""
    for section in lanes['Section'].unique():
        sec_lanes = lanes[lanes['Section'] == section].reset_index(drop=True)
        mid_row = sec_lanes.iloc[len(sec_lanes) // 2]
        cx, cy = _poly_center(mid_row)
        cv2.putText(img, str(section), (cx - 160, cy + 20),
                    FONT, SECTION_LABEL_SCALE, SECTION_COLOR, SECTION_LABEL_THICKNESS, cv2.LINE_AA)


def _poly_center(row):
    """Return the integer centroid (cx, cy) of a four-corner lane polygon row."""
    cx = int((row['tlx'] + row['blx'] + row['brx'] + row['trx']) / 4)
    cy = int((row['tly'] + row['bly'] + row['bry'] + row['try']) / 4)
    return cx, cy


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Overlay road segmentation data onto orthophoto images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('ortho_folder', type=Path,
                        help='Path to folder containing orthophoto images.')
    parser.add_argument('-sf', '--seg-folder', type=Path, default=None,
                        help='Folder with lane segmentation CSV files '
                             '(default: <ortho_folder>/segmentations).')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output folder for annotated images (default: same as --seg-folder).')
    parser.add_argument('-e', '--ext', type=str, default='png',
                        help='File extension of orthophoto images (default: png).')
    parser.add_argument('-lp', '--log-path', type=Path, default=None,
                        help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    seg_folder = args.seg_folder or args.ortho_folder / 'segmentations'
    output = args.output or seg_folder

    visualize_segmentations(
        ortho_folder=args.ortho_folder,
        seg_folder=seg_folder,
        output=output,
        ext=args.ext,
        logger=logger,
    )


if __name__ == '__main__':
    main()
