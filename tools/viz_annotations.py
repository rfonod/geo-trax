#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
viz_annotations.py - YOLO Annotation Visualizer

Visualizes YOLO-format annotations by drawing bounding boxes and optional class name labels
on images. Supports single images or batch processing of directories, filtering by class ID,
and saving results to a configurable output directory.

Usage:
  python tools/viz_annotations.py <source> [options]

Arguments:
  source : Path to an image file or a directory containing images.

Options:
  -h, --help                          : Show this help message and exit.
  -a, --annotations <path>            : Annotations directory or file (default: <source>/../labels).
  -e, --ext <str>                     : Image extension to match in directory mode (default: any common image format).
  -n, --top-n <int>                   : Top-N most-annotated frames to process in directory mode (default: 10).
  -s, --save                          : Save visualizations to the output directory (default: False).
  --show / --no-show                  : Display visualizations interactively (default: True unless --save is given).
  -o, --output-dir <path>             : Output directory for saved visualizations (default: <source>/../visualizations).
  -ow, --overwrite                    : Overwrite existing output files when saving (default: False).
  -lw, --line-width <int>             : Bounding box line width in pixels (default: 3).
  --show-labels / --no-show-labels    : Overlay class name on each bounding box (default: True).
  -cn, --class-names <...>            : Class ID-to-name mapping. Accepts one of:
                                          • path to a YAML/JSON file: names.yaml or names.json
                                            (file must contain a list [name0, name1, ...] or a
                                             dict {0: name0, 1: name1, ...})
                                          • key:value pairs: 0:car 1:bus 2:truck
                                          • positional names: car bus truck (mapped to IDs 0, 1, 2, ...)
                                        (default: show raw class ID as label)
  -t, --type <int> [<int> ...]        : Class IDs to visualize (default: all).
  -lp, --log-path <str>               : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                         : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Visualize a single image, show raw class IDs:
   python tools/viz_annotations.py image.jpg

2. Save visualizations for a directory (top 20 most-annotated frames):
   python tools/viz_annotations.py images/ --save -n 20

3. Display and save simultaneously, filtered to class 0 and 1 only:
   python tools/viz_annotations.py images/ --show --save --type 0 1

4. Load class names from a YAML file and save to an explicit output directory:
   python tools/viz_annotations.py images/ --save -o /tmp/viz -cn names.yaml

5. Provide class names as key:value pairs:
   python tools/viz_annotations.py images/ -cn 0:car 1:bus 2:truck 3:motorcycle

6. Provide class names positionally (mapped to IDs 0, 1, 2, ...):
   python tools/viz_annotations.py images/ -cn car bus truck motorcycle

7. Match PNG images only, hide labels, use thicker boxes:
   python tools/viz_annotations.py images/ -e png --no-show-labels -lw 5

Input:
- Image files (jpg/png/bmp/tiff, or filtered by --ext)
- YOLO annotation files (.txt) with one annotation per line: class_id cx cy w h (normalized)

Output:
- Interactive display window (press any key to advance, 'q' or Esc to quit)
- Saved visualizations in the configured output directory

Notes:
- In directory mode, frames are ranked by annotation count and the top-N are processed.
  Ranking respects --type: only matching class annotations count toward the rank.
- When --save is given without --show, the display is suppressed by default.
  Pass --show explicitly to display and save simultaneously.
- When --class-names is omitted, each box is labelled with its raw integer class ID.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml

from find_max_annotations import find_max_annotations
from geotrax.utils.data_utils import VizColors
from geotrax.utils.logging_utils import setup_logger

IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_image(image_path: Path):
    """Load an image; raise FileNotFoundError if missing or unreadable."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Image not found or unreadable: {image_path}')
    return img


def load_annotations(annotation_path: Path) -> list:
    """Load non-empty annotation lines from a YOLO .txt file."""
    with open(annotation_path) as f:
        return [line for line in f if line.strip()]


def parse_annotation(line: str, img_width: int, img_height: int):
    """Parse a normalized YOLO annotation line into (class_id, x1, y1, x2, y2) pixel coords."""
    class_id, cx, cy, w, h = map(float, line.split()[:5])
    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)
    return int(class_id), x1, y1, x2, y2


def resolve_class_names(raw: list | None) -> dict[int, str] | None:
    """
    Parse --class-names into a dict mapping int class_id → str name.
    Accepts: None (returns None), a single YAML/JSON path, key:value pairs, or a positional list.
    """
    if raw is None:
        return None
    if len(raw) == 1:
        path = Path(raw[0])
        if path.suffix.lower() in ('.yaml', '.yml', '.json'):
            with open(path) as f:
                data = yaml.safe_load(f) if path.suffix.lower() in ('.yaml', '.yml') else json.load(f)
            if isinstance(data, list):
                return {i: str(name) for i, name in enumerate(data)}
            return {int(k): str(v) for k, v in data.items()}
    if all(':' in item for item in raw):
        return {int(k): v for item in raw for k, v in [item.split(':', 1)]}
    return {i: name for i, name in enumerate(raw)}


def draw_annotation(img, class_id: int, x1: int, y1: int, x2: int, y2: int,
                    colors, line_width: int, class_names: dict[int, str] | None,
                    show_labels: bool) -> None:
    """Draw a bounding box and optional class label on img in place."""
    color = colors(class_id, True)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_AA)
    if show_labels:
        label = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        font_scale = max(0.35, line_width * 0.18)
        thickness = max(1, line_width // 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y1 - baseline, th + baseline)
        cv2.rectangle(img, (x1, ty - th - baseline), (x1 + tw + 2, ty + baseline), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def visualize_one(image_path: Path, annotation_path: Path, args: argparse.Namespace,
                  colors, logger: logging.Logger) -> bool:
    """
    Render annotations on one image, then show and/or save it.
    Returns False if the user pressed 'q' or Esc (signals batch to stop), True otherwise.
    """
    if not annotation_path.exists():
        logger.warning(f'Annotation file not found, skipping: {annotation_path}')
        return True

    try:
        img = load_image(image_path)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return True

    img_h, img_w = img.shape[:2]
    lines = load_annotations(annotation_path)

    drawn = 0
    for line in lines:
        class_id, x1, y1, x2, y2 = parse_annotation(line, img_w, img_h)
        if args.type is not None and class_id not in args.type:
            continue
        draw_annotation(img, class_id, x1, y1, x2, y2,
                        colors, args.line_width, args.class_names, args.show_labels)
        drawn += 1

    logger.info(f'{image_path.name}: {drawn} annotation(s) drawn')

    if args.save:
        out_path = args.output_dir / image_path.name
        if out_path.exists() and not args.overwrite:
            logger.warning(f'Skipping existing file (use --overwrite): {out_path}')
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
            logger.info(f'Saved: {out_path}')

    if args.show:
        cv2.imshow(f'Annotations — {image_path.name}', img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (ord('q'), 27):  # 'q' or Esc
            return False

    return True


def run_visualizer(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Dispatch to single-image or directory mode."""
    try:
        args.class_names = resolve_class_names(args.class_names)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error(f'Failed to parse --class-names: {exc}')
        return
    colors = VizColors()

    if args.source.is_dir():
        ann_dir = (args.annotations if args.annotations and args.annotations.is_dir()
                   else args.source.parent / 'labels')
        if not ann_dir.is_dir():
            logger.error(f'Annotations directory not found: {ann_dir}')
            return

        if args.output_dir is None:
            args.output_dir = args.source.parent / 'visualizations'

        top_files = find_max_annotations(ann_dir, args.top_n, args.type)
        if not top_files:
            logger.warning(f'No annotation files found in: {ann_dir}')
            return

        logger.notice(f'Processing {len(top_files)} most-annotated frame(s) from: {ann_dir}')

        exts = ([f'.{args.ext.lstrip(".")}', f'.{args.ext.lstrip(".").upper()}']
                if args.ext else sorted(IMAGE_FORMATS))

        processed = 0
        for ann_file, _ in top_files:
            image_file = next(
                (args.source / f'{ann_file.stem}{e}' for e in exts
                 if (args.source / f'{ann_file.stem}{e}').exists()),
                None,
            )
            if image_file is None:
                logger.warning(f'No matching image for annotation: {ann_file.name}')
                continue
            if not visualize_one(image_file, ann_file, args, colors, logger):
                break
            processed += 1

        logger.notice(f'Done. {processed} image(s) processed.')

    else:
        if not args.source.exists():
            logger.error(f'Source image not found: {args.source}')
            return

        if args.annotations is not None:
            ann_file = (args.annotations if args.annotations.is_file()
                        else args.annotations / f'{args.source.stem}.txt')
        else:
            ann_file = args.source.parent.parent / 'labels' / f'{args.source.stem}.txt'

        if args.output_dir is None:
            args.output_dir = args.source.parent.parent / 'visualizations'

        visualize_one(args.source, ann_file, args, colors, logger)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize YOLO-format vehicle annotations on images.',
    )
    parser.add_argument('source', type=Path,
                        help='Path to an image file or directory containing images.')
    parser.add_argument('--annotations', '-a', type=Path, default=None,
                        help='Annotations directory or file (default: <source>/../labels).')
    parser.add_argument('--ext', '-e', type=str, default=None,
                        help='Image extension to match in directory mode (default: any common image format).')
    parser.add_argument('--top-n', '-n', type=int, default=10,
                        help='Top-N most-annotated frames to process in directory mode (default: 10).')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save visualizations to the output directory (default: False).')
    parser.add_argument('--show', action=argparse.BooleanOptionalAction, default=None,
                        help='Display visualizations interactively (default: True unless --save is given).')
    parser.add_argument('--output-dir', '-o', type=Path, default=None,
                        help='Output directory for saved visualizations (default: <source>/../visualizations).')
    parser.add_argument('--overwrite', '-ow', action='store_true',
                        help='Overwrite existing output files when saving (default: False).')
    parser.add_argument('--line-width', '-lw', type=int, default=3,
                        help='Bounding box line width in pixels (default: 3).')
    parser.add_argument('--show-labels', action=argparse.BooleanOptionalAction, default=True,
                        help='Overlay class name on each bounding box (default: True).')
    parser.add_argument('--class-names', '-cn', nargs='+', default=None, metavar='NAME_OR_PATH',
                        help='Class ID-to-name mapping: a YAML/JSON file path, key:value pairs (0:car 1:bus), '
                             'or positional names (car bus truck). Default: show raw class ID.')
    parser.add_argument('--type', '-t', nargs='+', type=int, default=None,
                        help='Class IDs to visualize (default: all).')
    parser.add_argument('--log-path', '-lp', type=Path, default=None,
                        help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce console verbosity to important messages only (default: show INFO-level detail).')

    args = parser.parse_args()
    if args.show is None:
        args.show = not args.save
    return args


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)
    run_visualizer(args, logger)


if __name__ == '__main__':
    main()
