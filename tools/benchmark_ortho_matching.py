#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
benchmark_ortho_matching.py - Orthophoto Matching Accuracy Benchmark Tool

This script evaluates orthophoto matching accuracy by computing reprojection errors between ground truth
labels and homography-transformed coordinates. It benchmarks the impact of orthophoto resolution on
matching performance using RSIFT feature detection and RANSAC-based homography estimation.

The tool processes multiple resolution levels, measures computation times, analyzes reprojection errors,
and generates comprehensive statistics for performance evaluation. It also provides visualization
capabilities for ground truth label verification.

Usage:
  python tools/benchmark_ortho_matching.py <data> [options]

Arguments:
  data : Path to benchmark data folder containing images, orthos, and ground truth labels.

Options:
  -h, --help                   : Show this help message and exit.
  -sb, --skip-benchmark        : Skip benchmark execution and only visualize ground truths (default: False).
  -o, --overwrite              : Overwrite existing results.txt file and visualizations (default: False).
  -v, --visualize              : Visualize ground truths for input data (default: False).
  -mr, --min-resolution <int>  : Minimum orthophoto resolution to consider (default: 2000).
  -xr, --max-resolution <int>  : Maximum orthophoto resolution to consider (default: 15000).
  -rs, --resolution-step <int> : Step size for orthophoto resolution (default: 1000).
  -lp, --log-path <str>        : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
  -q, --quiet                  : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
1. Run benchmark and visualize ground truths:
   python tools/benchmark_ortho_matching.py path/to/data -v

2. Skip benchmark and only visualize ground truths:
   python tools/benchmark_ortho_matching.py path/to/data -sb -v

3. Custom resolution range with overwrite:
   python tools/benchmark_ortho_matching.py path/to/data -mr 1000 -xr 10000 -rs 500 -o

Input:
- Data folder structure:
  ├── images/          (drone images in PNG format)
  ├── orthos/          (orthophoto references in PNG format)
  └── labels/          (ground truth labels in CSV format)
- Label CSV format: columns 'pnum', 'px', 'py' (point number, x-coordinate, y-coordinate)
- Matching filenames between images, orthos, and corresponding labels

Output:
- Console output: Per-image reprojection errors, computation times, inlier counts
- results.txt: LaTeX-formatted table with aggregated statistics
- visualizations/ directory: Ground truth overlays on images and orthophotos
- visualizations/paper/ directory: Publication-ready visualizations

Methodology:
- RSIFT feature detection with adaptive feature count
- Brute-force matching with ratio test filtering
- RANSAC homography estimation with USAC_MAGSAC
- Perspective transformation for coordinate mapping
- Euclidean distance calculation for reprojection errors
- Statistical analysis across multiple resolution levels

Performance Metrics:
- Mean and standard deviation of reprojection errors
- Computation time per homography estimation
- Inlier count and ratio for RANSAC consensus
- Aggregated statistics across locations and resolutions

Notes:
- Resolution testing performed by downsampling orthophotos
- Skips resolutions larger than original orthophoto dimensions
- Adaptive feature detection with fallback for failed matches
- Supports both individual and aggregated performance analysis
- Visualizations include point number annotations and styling
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from geotrax.utils.logging_utils import setup_logger
from geotrax.utils.registration import estimate_homography


def run_benchmark(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the orthophoto matching benchmark and/or ground-truth visualizations.
    """

    images_dir = args.data / 'images'
    orthos_dir = args.data / 'orthos'
    labels_dir = args.data / 'labels'
    visual_dir = args.data / 'visualizations'

    if not args.skip_benchmark:
        execute_ortho_benchmark(images_dir, orthos_dir, labels_dir, args, logger)
    if args.visualize:
        generate_and_save_visualizations(images_dir, orthos_dir, labels_dir, visual_dir, args, logger)


def execute_ortho_benchmark(images_dir, orthos_dir, labels_dir, args, logger):
    """
    Run the orthophoto matching benchmark.
    """

    images_filepaths = sorted(images_dir.glob('*.png'))
    orthos_filepaths = sorted(orthos_dir.glob('*.png'))

    results_all = {}
    for ortho_filepath in orthos_filepaths:
        location_id = ortho_filepath.stem
        logger.info(f"Processing location_ID: {location_id}")

        ortho_labels = pd.read_csv(labels_dir / f"{location_id}.csv")
        ortho = cv2.imread(str(ortho_filepath))
        ortho_h_original, ortho_w_original = ortho.shape[:2]

        results_location_id = {}
        ortho_w_resolutions = range(args.min_resolution, args.max_resolution + 1, args.resolution_step)
        for ortho_w_new in ortho_w_resolutions:
            if ortho_w_new > ortho_w_original:
                logger.warning(f"Orthophoto width {ortho_w_new} is larger than the original width {ortho_w_original}. Skipping.")
                continue

            ortho_labels_resized = ortho_labels.copy()
            ortho_h_new = ortho_h_original * ortho_w_new // ortho_w_original
            if ortho_w_new == ortho_w_original and ortho_h_new == ortho_h_original:
                ortho_resized = ortho.copy()
            else:
                ortho_resized = cv2.resize(ortho, (ortho_w_new, ortho_h_new))
                ortho_labels_resized['px'] = ortho_labels_resized['px'] * ortho_w_new / ortho_w_original
                ortho_labels_resized['py'] = ortho_labels_resized['py'] * ortho_h_new / ortho_h_original

            comp_times_list, pixel_error_list, inliers_list = [], [], []
            for image_filepath in images_filepaths:
                if image_filepath.stem.split('_')[-2][0] != location_id:
                    continue

                image = cv2.imread(str(image_filepath))
                image_labels = pd.read_csv(labels_dir / (image_filepath.stem + '.csv'))

                start_time = time.time()
                H, inliers_count, num_matches = compute_homography(image, ortho_resized, logger)
                comp_times_list.append(time.time() - start_time)
                inliers_list.append(inliers_count)

                pixel_error_image_list = []
                for pnum, ortho_x_label, ortho_y_label in ortho_labels_resized[['pnum', 'px', 'py']].values:
                    image_x, image_y = cv2.perspectiveTransform(np.array([[[ortho_x_label, ortho_y_label]]], dtype=np.float64), np.linalg.inv(H))[0][0]
                    image_x_label, image_y_label = image_labels.loc[image_labels['pnum'] == pnum, ['px', 'py']].values[0]
                    pixel_error = np.sqrt((image_x - image_x_label) ** 2 + (image_y - image_y_label) ** 2)
                    pixel_error_image_list.append(pixel_error)

                logger.info(
                    f"{ortho_filepath.stem}({ortho_w_new})/{image_filepath.stem}: "
                    f"{np.mean(pixel_error_image_list):.3f}±{np.std(pixel_error_image_list):.3f}, "
                    f"Inliers/total: {inliers_count:3}/{num_matches:<4} | "
                    + ' '.join(f'{i+1})={pixel_error:.2f}' for i, pixel_error in enumerate(pixel_error_image_list))
                )
                pixel_error_list.extend(pixel_error_image_list)

            results_location_id[ortho_w_new] = {
                'Comp_times': comp_times_list,
                'Errors': pixel_error_list,
                'Inliers': inliers_list,
            }

        results_all[location_id] = results_location_id

    to_latex = ['Intersection & Resolution & Comp. time & Error & Avg. inliers & Min. inliers \\\\']
    for location_id, results_location_id in results_all.items():
        for ortho_w_new, results in results_location_id.items():
            formatted_resolution = format_with_apostrophe(ortho_w_new)
            to_latex.append(f"{location_id} & {formatted_resolution:<6} & {np.mean(results['Comp_times']):>6.3f} & {np.mean(results['Errors']):>6.3f} $\pm$ {np.std(results['Errors']):.3f}  & {np.mean(results['Inliers'])} & {np.min(results['Inliers'])} \\\\")

    to_latex.append("\nAggregated results for all intersections:")
    for ortho_w_new in ortho_w_resolutions:
        if ortho_w_new not in results_all[location_id]:
            continue
        errors, comp_times, inliers = [], [], []
        for _, results_location_id in results_all.items():
            errors.extend(results_location_id[ortho_w_new]['Errors'])
            comp_times.extend(results_location_id[ortho_w_new]['Comp_times'])
            inliers.extend(results_location_id[ortho_w_new]['Inliers'])
        formatted_resolution = format_with_apostrophe(ortho_w_new)
        to_latex.append(f"{formatted_resolution:<6} & {np.mean(comp_times):>6.3f} & {np.mean(errors):>6.3f} $\pm$ {np.std(errors):.3f} & {np.mean(inliers)} & {np.min(inliers)} \\\\")
    logger.notice("\n%s", '\n'.join(to_latex))

    results_filepath = args.data / 'results.txt'
    if args.overwrite or not results_filepath.exists():
        with open(results_filepath, 'w') as f:
            f.write('\n'.join(to_latex))


def format_with_apostrophe(number):
    return f"{number:,}".replace(",", "'")


def compute_homography(img_src: np.ndarray, img_dst: np.ndarray, logger: logging.Logger, max_features: int = 250000,
                       filter_ratio: float = 0.55, ransac_epipolar_threshold: float = 3.0,
                       ransac_confidence: float = 0.999999, ransac_max_iters: int = 10000) -> tuple:
    """
    Compute homography between two images using RSIFT keypoints and descriptors.
    """
    homography, inliers_count, num_matches, _ = estimate_homography(
        img_src, img_dst, logger, max_features=max_features, filter_ratio=filter_ratio,
        ransac_epipolar_threshold=ransac_epipolar_threshold, ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iters,
    )
    return homography, inliers_count, num_matches


def generate_and_save_visualizations(images_dir: Path, orthos_dir: Path, labels_dir: Path, visual_dir: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Generate and save visualizations for the ground truths.
    """

    def save_visualization(filepaths: list, labels_dir: Path, visual_dir: Path, overwrite: bool) -> None:
        """
        Save the visualizations for the ground truths.
        """
        for filepath in filepaths:
            save_filename = filepath.stem
            if not (visual_dir / f'{save_filename}.png').exists() or overwrite:
                logger.info(f'Saving visualization for {filepath}')
                labels = pd.read_csv(labels_dir / f'{save_filename}.csv')
                image = cv2.imread(str(filepath))
                image_paper = cv2.addWeighted(image, 0.4, 255 * np.ones_like(image), 0.6, 0)

                image = render_image_labels(image, labels)
                image_paper = render_image_labels(image_paper, labels)

                if image.shape[1] > 3840:
                    image = cv2.resize(image, (3840, int(3840 * image.shape[0] / image.shape[1])))
                image_paper = cv2.resize(image_paper, (1920, int(1920 * image.shape[0] / image.shape[1])))

                cv2.imwrite(str(visual_dir / f'{save_filename}.png'), image)
                cv2.imwrite(str(visual_dir / 'paper' / f'{save_filename}.png'), image_paper)

    images_filepaths = sorted(images_dir.glob('*.png'))
    orthos_filepaths = sorted(orthos_dir.glob('*.png'))

    visual_dir.mkdir(parents=True, exist_ok=True)
    (visual_dir / 'paper').mkdir(parents=True, exist_ok=True)

    save_visualization(images_filepaths, labels_dir, visual_dir, args.overwrite)
    save_visualization(orthos_filepaths, labels_dir, visual_dir, args.overwrite)


def render_image_labels(image: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    """
    Draw the labels on the image.
    """
    factor = 2 if image.shape[1] > 3840 else 1
    radius = factor * 17 * image.shape[1] // 3840
    font_scale = factor * 2.7 * image.shape[1] // 3840
    thickness = round(factor * 3.5 * image.shape[1] // 3840)
    font_distance = factor * 11 * image.shape[1] // 3840

    for pnum, x, y in labels[['pnum', 'px', 'py']].values:
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), radius, (0, 0, 255), thickness)
        cv2.circle(image, (x, y), 0, (0, 0, 255), 3 * thickness)
        cv2.putText(image, str(pnum), (x + font_distance, y - font_distance), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    return image


def parse_cli_args() -> argparse.Namespace:
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of orthophoto matching and visualize the ground truths.")
    parser.add_argument('data', type=Path, help='Path to the benchmark data folder containing images, orthos, and labels')
    parser.add_argument('--skip-benchmark', '-sb', default=False, action='store_true', help='Skip the benchmark and only visualize the ground truths')
    parser.add_argument('--overwrite', '-o', default=False, action='store_true', help='Overwrite the existing results.txt file and visualizations')
    parser.add_argument('--visualize', '-v', default=False, action='store_true', help='Visualize the ground truths for the input data')
    parser.add_argument('--min-resolution', '-mr', type=int, default=2000, help='Minimum orthophoto resolution to consider')
    parser.add_argument('--max-resolution', '-xr', type=int, default=15000, help='Maximum orthophoto resolution to consider')
    parser.add_argument('--resolution-step', '-rs', type=int, default=1000, help='Step size for the orthophoto resolution')
    parser.add_argument('--log-path', '-lp', type=Path, default=None, help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce console verbosity to important messages only (default: show INFO-level detail).')
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    run_benchmark(args, logger)


if __name__ == '__main__':
    main()


