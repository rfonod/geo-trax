#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_max_annotations.py - Top Annotation Files Finder

This script analyzes YOLO format annotation files to identify the top N files containing
the most vehicle annotations. It processes all .txt annotation files in a specified
directory, counts annotations per file, and ranks them by annotation density.

The tool supports filtering by specific vehicle types and provides a ranked list of
annotation files with their corresponding annotation counts for dataset analysis
and quality assessment purposes.

Usage:
  python tools/find_max_annotations.py <source> [options]

Arguments:
  source : str
           Path to directory containing annotations in YOLO format.

Options:
  -h, --help            : Show this help message and exit.
  -N, -n <int>          : int, optional
                        Number of top annotation files to find (default: 10).
  -t, --type <int> [<int> ...] : list of int, optional
                        Specific vehicle type IDs to count (default: all types).

Examples:
1. Find top 10 annotation files with most annotations:
   python tools/find_max_annotations.py /path/to/annotations/

2. Find top 5 annotation files:
   python tools/find_max_annotations.py /path/to/annotations/ -N 5

3. Find top files with specific vehicle types (e.g., cars and trucks):
   python tools/find_max_annotations.py /path/to/annotations/ --type 0 1

Input:
- Directory containing YOLO format annotation files (.txt)
- Each annotation file contains one annotation per line
- Annotation format: class_id x_center y_center width height

Output:
- Console output: Ranked list of annotation files with counts
- Format: "Annotation N :: filepath (X annotations)"

Notes:
- Processes all .txt files in the specified directory
- Supports filtering by specific vehicle class IDs
- Annotations are counted based on YOLO format class IDs
- Files are ranked in descending order by annotation count
- Useful for dataset analysis and sample selection
"""

import argparse
from pathlib import Path


def show_top_annotations(args):
    top_N_annotations = find_max_annotations(args.source, args.N, args.type)
    for i, (annotation_file, count) in enumerate(top_N_annotations):
        print(f'Annotation {i+1:<2} :: {annotation_file} ({count} annotations)')


def load_annotations(annotation_path):
    """Load annotations from the given path."""
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    return lines


def count_annotations(annotations, veh_type):
    """Count the number of vehicle annotations in the given list."""
    if veh_type is None:
        return len(annotations)
    else:
        return sum(1 for line in annotations if int(line.split()[0]) in veh_type)


def find_max_annotations(source, N, veh_type):
    """Find the top N annotation files with the most vehicle annotations."""
    annotation_counts = []
    for annotation_file in source.glob('*.txt'):
        annotations = load_annotations(annotation_file)
        count = count_annotations(annotations, veh_type)
        annotation_counts.append((annotation_file, count))

    # Sort the annotations by the number of vehicle annotations in descending order
    annotation_counts.sort(key=lambda x: x[1], reverse=True)

    return annotation_counts[:N]


def get_cli_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Find the first N image annotations that contain the most vehicle annotations.')
    parser.add_argument('source', type=Path, help='Path to a directory containing the annotations in YOLO format')
    parser.add_argument('--N', '-n', type=int, default=10, help='Number of top image frames to find (default: 10)')
    parser.add_argument('--type', '-t', nargs="+", type=int, help='Type of vehicle to find (default: all)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    show_top_annotations(args)
