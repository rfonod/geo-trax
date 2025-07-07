#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
viz_annotations.py - Vehicle Annotation Visualizer

Visualizes YOLO-format vehicle annotations by drawing bounding boxes on images.
Supports single images or batch processing of directories with annotation filtering.

Usage:
  python tools/viz_annotations.py <image_path> [options]

Arguments:
  image_path : Path to image file or directory containing images.

Options:
  -a, --annotations <path>  : Annotations directory (default: auto-detect).
  -s, --save                : Save visualizations instead of displaying (default: False).
  -lw, --line_width <int>   : Bounding box line width (default: 3).
  -n, --N <int>             : Number of top annotated frames to process (default: 10).
  -t, --type <int> [<int>]  : Vehicle class IDs to visualize (default: all).

Examples:
1. Visualize single image:
   python tools/viz_annotations.py image.jpg

2. Save visualizations for directory:
   python tools/viz_annotations.py images/ --save

3. Filter specific vehicle types:
   python tools/viz_annotations.py images/ --type 0 1 2

Input:
- Images in JPG format
- YOLO annotation files (.txt) with normalized coordinates

Output:
- Interactive display or saved images in visualizations/ directory
- Colored bounding boxes with vehicle class-specific colors
"""

import argparse
import sys
from pathlib import Path

import cv2
from find_max_annotations import find_max_annotations

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.utils import VizColors


def run_visualizer(args):
    """Run the visualizer."""
    if args.image_path.is_dir():
        if args.annotations is not None and args.annotations.is_dir():
            annotations_dir = args.annotations
        else:
            annotations_dir = args.image_path.parent / 'labels'
        top_N_annotations = find_max_annotations(annotations_dir, args.N, args.type)
        for annotation_file, _ in top_N_annotations:
            image_file = args.image_path / f'{annotation_file.stem}.jpg'
            visualize_annotations(image_file, annotation_file, args.line_width, args.save, args.type)
    else:
        if args.annotations is not None:
            annotation_file = args.annotations
        else:
            annotation_file = args.image_path.parent.parent / 'labels' / f'{args.image_path.stem}.txt'
        visualize_annotations(args.image_path, annotation_file, args.line_width, args.save, args.type)


def load_image(image_path):
    """Load an image from the given path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    return img


def load_annotations(annotation_path):
    """Load annotations from the given path."""
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    return lines


def parse_annotation(line, img_width, img_height):
    """Parse a single annotation line."""
    class_id, center_x, center_y, width, height = map(float, line.split())
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    return int(class_id), x1, y1, x2, y2


def draw_bounding_box(img, annotation, colors, line_width):
    """Draw a bounding box on the image."""
    class_id, x1, y1, x2, y2 = annotation
    color = colors(class_id, True)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_AA)


def count_annotations(annotations, type):
    """Count the number of vehicle annotations in the given list."""
    if type is None:
        return len(annotations)
    else:
        return sum(1 for line in annotations if int(line.split()[0]) in type)


def visualize_annotations(image_path, annotation_file, line_width, save=False, veh_type=None):
    """Visualize the annotations on the image."""
    img = load_image(image_path)
    img_height, img_width = img.shape[:2]
    annotations = load_annotations(annotation_file)
    colors = VizColors()

    for line in annotations:
        annotation = parse_annotation(line, img_width, img_height)
        draw_bounding_box(img, annotation, colors, line_width)

    num_annotations = count_annotations(annotations, veh_type)
    print(f'Number of annotations: {num_annotations}')

    if save:
        save_dir = image_path.parent.parent / 'visualizations'
        save_dir.mkdir(exist_ok=True)
        destination = save_dir / image_path.name
        cv2.imwrite(str(destination), img)
    else:
        # check if 'q' is pressed
        cv2.imshow('Annotations', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            exit(1)
        else:
            cv2.destroyAllWindows()
            return


def get_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Visualize the frame annotations of the vehicles.')
    parser.add_argument('image_path', type=Path, help='Path to the image to be visualized or to the directory containing the images')
    parser.add_argument('--annotations', '-a', type=Path, help='Path to the annotations directory (default: <image_path>/../labels)')
    parser.add_argument('--save', '-s', action='store_true', help='Save the visualization')
    parser.add_argument('--line_width', '-lw', type=int, default=3, help='Width of the bounding box line')
    parser.add_argument('--N', '-n', type=int, default=10, help='If folder input, number of top image frames to find (default: 10)')
    parser.add_argument('--type', '-t', nargs="+", type=int, help='Type of vehicle to find (default: all)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_args()
    run_visualizer(args)
