#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
ortho_matching_benchmark.py - Evaluate the accuracy of orthophoto matching and visualize the ground truths.

This script benchmarks the orthophoto matching process by comparing the reprojection errors of ground truth labels.
Different orthophoto resolutions are considered to evaluate the impact on the accuracy of the matching process.
The script uses RSIFT for feature detection and homography computation, but it can be easily extended to use other methods.

Usage:
  python ortho_matching_benchmark.py <data> [options]

Arguments:
  data : Path to the benchmark data folder containing images, orthos, and ground truth labels.

Options:
  --skip-benchmark, -sb       : Skip the benchmark and only visualize the ground truths.
  --overwrite, -o             : Overwrite the existing results.txt file and visualizations.
  --visualize, -v             : Visualize the ground truths for the input data.
  --min-resolution, -mr       : Minimum orthophoto resolution to consider [default: 2000].
  --max-resolution, -xr       : Maximum orthophoto resolution to consider [default: 15000].
  --resolution-step, -rs      : Step size for the orthophoto resolution [default: 1000].

Examples:

  1. Run the benchmark and visualize the ground truths:
     python ortho_matching_benchmark.py path/to/data -v

  2. Skip the benchmark and only visualize the ground truths:
     python ortho_matching_benchmark.py path/to/data -sb -v

  3. Overwrite existing results and visualizations:
     python ortho_matching_benchmark.py path/to/data -o

Notes:
  - Ensure that the data folder contains subfolders named 'images', 'orthos', and 'labels' with the respective files.
  - Images and orthophotos should be in PNG format, and the labels should be in CSV format.
  - The labels should contain the columns 'pnum', 'px', and 'py' for point number, x-coordinate, and y-coordinate, respectively.
  - The filenames of the images and orthophotos should match with those of the labels.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the orthophoto matching benchmark.
    """

    images_dir = args.data / 'images'
    orthos_dir = args.data / 'orthos'
    labels_dir = args.data / 'labels'
    visual_dir = args.data / 'visualizations'

    if not args.skip_benchmark:
        execute_ortho_benchmark(images_dir, orthos_dir, labels_dir, args)
    if args.visualize:
        generate_and_save_visualizations(images_dir, orthos_dir, labels_dir, visual_dir, args)


def execute_ortho_benchmark(images_dir, orthos_dir, labels_dir, args):
    """
    Run the orthophoto matching benchmark.
    """

    images_filepaths = sorted(images_dir.glob('*.png'))
    orthos_filepaths = sorted(orthos_dir.glob('*.png'))

    results_all = {}
    for ortho_filepath in orthos_filepaths:
        location_id = ortho_filepath.stem
        print(f"Processing location_ID: {location_id}")

        ortho_labels = pd.read_csv(labels_dir / f"{location_id}.csv")
        ortho = cv2.imread(str(ortho_filepath))
        ortho_h_original, ortho_w_original = ortho.shape[:2]

        results_location_id = {}
        ortho_w_resolutions = range(args.min_resolution, args.max_resolution + 1, args.resolution_step)
        for ortho_w_new in ortho_w_resolutions:
            if ortho_w_new > ortho_w_original:
                print(f"\033[93mOrthophoto width {ortho_w_new} is larger than the original width {ortho_w_original}. Skipping.\033[0m")
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
                H, inliers = compute_homography(image, ortho_resized)
                comp_times_list.append(time.time() - start_time)
                inliers_list.append(inliers.sum())

                pixel_error_image_list = []
                for pnum, ortho_x_label, ortho_y_label in ortho_labels_resized[['pnum', 'px', 'py']].values:
                    image_x, image_y = cv2.perspectiveTransform(np.array([[[ortho_x_label, ortho_y_label]]], dtype=np.float64), np.linalg.inv(H))[0][0]
                    image_x_label, image_y_label = image_labels.loc[image_labels['pnum'] == pnum, ['px', 'py']].values[0]
                    pixel_error = np.sqrt((image_x - image_x_label) ** 2 + (image_y - image_y_label) ** 2)
                    pixel_error_image_list.append(pixel_error)

                print(f"{ortho_filepath.stem}({ortho_w_new})/{image_filepath.stem}:", f"{np.mean(pixel_error_image_list):.3f}±{np.std(pixel_error_image_list):.3f},",
                      f"Inliers/total: {inliers.sum():3}/{len(inliers):<4} |", ' '.join(f'{i+1})={pixel_error:.2f}' for i, pixel_error in enumerate(pixel_error_image_list)))
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
    print('\n'.join(to_latex))

    results_filepath = args.data / 'results.txt'
    if args.overwrite or not results_filepath.exists():
        with open(results_filepath, 'w') as f:
            f.write('\n'.join(to_latex))


def format_with_apostrophe(number):
    return f"{number:,}".replace(",", "'")


def compute_homography(img_src: np.ndarray, img_dst: np.ndarray, max_features: int = 250000,
                       filter_ratio: float = 0.55, ransac_epipolar_threshold: float = 3.0,
                       ransac_confidence: float = 0.999999, ransac_max_iters: int = 10000) -> tuple:
    """
    Compute homography between two images using RSIFT keypoints and descriptors.
    """

    def convert_to_rootsift(descriptors: np.ndarray, eps = 1e-8) -> np.ndarray:
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        return descriptors

    def try_compute_homography(max_features: int) -> tuple:
        img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT.create(nfeatures=max_features, enable_precise_upscale=True)
        kpt_src, desc_src = sift.detectAndCompute(img_src_gray, None)  # type: ignore
        kpt_dst, desc_dst = sift.detectAndCompute(img_dst_gray, None)  # type: ignore

        if kpt_src is None or kpt_dst is None:
            return None, None

        desc_src = convert_to_rootsift(desc_src)
        desc_dst = convert_to_rootsift(desc_dst)

        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(desc_src, desc_dst, k=2)
        except cv2.error:
            return None, None

        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < filter_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return None, None

        pts_src = np.array([kpt_src[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 2)
        pts_dst = np.array([kpt_dst[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 2)

        homography, inliers = cv2.findHomography(pts_src, pts_dst, method=cv2.USAC_MAGSAC, confidence=ransac_confidence,
                                                 ransacReprojThreshold=ransac_epipolar_threshold, maxIters=ransac_max_iters)

        return homography, inliers

    max_features_to_try = max_features
    while max_features_to_try > 10000:
        homography, inliers = try_compute_homography(max_features_to_try)
        if homography is not None:
            return homography, inliers
        max_features_to_try //= 2
        print(f"\033[93mSIFT detection or matching failed with {max_features_to_try*2} max_features. Trying with {max_features_to_try} max_features.\033[0m")

    print("SIFT detection failed with all attempted feature counts.")
    return None, None


def generate_and_save_visualizations(images_dir: Path, orthos_dir: Path, labels_dir: Path, visual_dir: Path, args: argparse.Namespace) -> None:
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
                print(f'Saving visualization for {filepath}')
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


def get_cli_arguments() -> argparse.Namespace:
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
    return parser.parse_args()


if __name__ == "__main__":
    main(get_cli_arguments())


