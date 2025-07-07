#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
compute_bb_center_error.py - Bounding Box Center Error Analysis Tool

This script computes statistical metrics for bounding box center prediction accuracy by comparing
human annotations (ground truth) with predicted annotations from object detection models.
It calculates pixel-wise center coordinate errors and provides comprehensive statistical analysis
with visualization capabilities.

The tool matches predicted detections to ground truth annotations by checking if predicted
centers fall within ground truth bounding boxes, then computes Euclidean distances between
centers. Analysis can be performed class-agnostic or with per-class breakdown.

Usage:
  python tools/compute_bb_center_error.py <source> [options]

Arguments:
  source : str
           Path to directory containing images to be analyzed.

Options:
  -h, --help            : Show this help message and exit.
  -ha, --human-annotations <path> : str, optional
                        Relative path to human annotations directory (default: '../labels').
  -pa, --predicted-annotations <path> : str, optional
                        Relative path to predicted annotations directory (default: '../pre-labels').
  -s, --save            : bool, optional
                        Save error distribution plots as figures (default: False).
  -ca, --class-agnostic : bool, optional
                        Compute class-agnostic statistics instead of per-class breakdown (default: False).

Examples:
1. Basic analysis with default annotation paths:
   python tools/compute_bb_center_error.py /path/to/images/

2. Class-agnostic analysis with saved plots:
   python tools/compute_bb_center_error.py /path/to/images/ --class-agnostic --save

3. Custom annotation paths with per-class analysis:
   python tools/compute_bb_center_error.py /path/to/images/ -ha labels -pa predictions

Input:
- Image directory containing .jpg files
- Human annotations directory with .txt files (YOLO format)
- Predicted annotations directory with .txt files (YOLO format)
- Expected directory structure:
  ├── images/          (source argument)
  │   ├── 000001.jpg
  │   └── 000002.jpg
  ├── labels/          (human annotations)
  │   ├── 000001.txt
  │   └── 000002.txt
  └── pre-labels/      (predicted annotations)
      ├── 000001.txt
      └── 000002.txt

Output:
- Console output: Statistical metrics (mean, median, std dev, valid/NaN counts)
- Class-agnostic mode: Single set of statistics for all classes combined
- Per-class mode: Statistics breakdown by class ID plus overall summary
- Visualization: Error distribution plots with statistical overlays
- Optional: Saved plots (PDF/PNG) in parent directory of source

Annotation Format:
- YOLO format: <class_id> <x_center> <y_center> <width> <height>
- Coordinates normalized to [0,1] relative to image dimensions
- Center coordinates (x, y) represent bounding box center
- Predicted annotations can be generated using annotate_frames.py

Error Computation:
- Matches predictions to ground truth by spatial overlap (center-in-box)
- Calculates Euclidean distance between matched centers in pixels
- Reports minimum error for each ground truth annotation
- Handles unmatched annotations as NaN values
- Provides robust statistical analysis with outlier-aware visualizations

Notes:
- Requires corresponding .jpg and .txt files with matching names
- Supports both class-specific and class-agnostic analysis modes
- Generates professional publication-quality plots with statistical overlays
- Handles missing annotations gracefully with warning messages
- Uses spatial containment for prediction-to-ground-truth matching
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def compute_bb_center_error(args):
    human_annotation_path = args.source / args.human_annotations
    predicted_annotation_path = args.source / args.predicted_annotations

    if not human_annotation_path.is_dir():
        print(f"Error: {human_annotation_path} is not a valid directory.")
        return
    if not predicted_annotation_path.is_dir():
        print(f"Error: {predicted_annotation_path} is not a valid directory.")
        return

    images = list(args.source.glob("*.jpg"))

    if args.class_agnostic:
        # Class agnostic mode - single list of errors
        errors = []
        for image in tqdm.tqdm(images, desc="Processing images"):
            image_id = image.stem
            image = cv2.imread(str(image))
            image_height, image_width = image.shape[:2]

            human_annotations = load_annotations(image_id, human_annotation_path)
            predicted_annotations = load_annotations(image_id, predicted_annotation_path)
            image_errors = compute_error(human_annotations, predicted_annotations, image_width, image_height)
            errors.extend(image_errors)

        errors = np.array(errors)
        report_results(errors, args)
        plot_error_distribution(errors, args.source, args.save)
    else:
        # Per-class mode - dictionary of errors by class ID
        errors_by_class = defaultdict(list)
        class_stats = {}
        for image in tqdm.tqdm(images, desc="Processing images"):
            image_id = image.stem
            image = cv2.imread(str(image))
            image_height, image_width = image.shape[:2]

            human_annotations = load_annotations(image_id, human_annotation_path)
            predicted_annotations = load_annotations(image_id, predicted_annotation_path)
            image_errors_by_class = compute_error_by_class(human_annotations, predicted_annotations, image_width, image_height)

            # Aggregate errors by class ID
            for class_id, class_errors in image_errors_by_class.items():
                errors_by_class[class_id].extend(class_errors)

        # Convert lists to numpy arrays for each class
        for class_id in errors_by_class:
            errors_by_class[class_id] = np.array(errors_by_class[class_id])

        # Report and plot per-class results
        report_results_by_class(errors_by_class, args)
        plot_error_distribution_by_class(errors_by_class, args.source, args.save)


def load_annotations(image_id, annotation_path):
    annotation_file = annotation_path / f"{image_id}.txt"
    if not annotation_file.exists():
        print(f"\033[93mError: {annotation_file} does not exist.\033[0m")
        return None
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        class_id, x, y, width, height = line.strip().split()
        class_id = int(class_id)
        x, y, width, height = map(float, [x, y, width, height])
        annotations.append((class_id, x, y, width, height))
    return annotations


def compute_error(human_annotations, predicted_annotations, image_width, image_height):
    errors = []
    for human_annotation in human_annotations:
        human_class_id, human_x, human_y, human_width, human_height = human_annotation
        human_x = human_x * image_width
        human_y = human_y * image_height
        human_width = human_width * image_width
        human_height = human_height * image_height
        human_center = np.array([human_x, human_y])

        min_error = np.inf
        for predicted_annotation in predicted_annotations:
            predicted_class_id, predicted_x, predicted_y, predicted_width, predicted_height = predicted_annotation
            predicted_x = predicted_x * image_width
            predicted_y = predicted_y * image_height
            # Check if the predicted annotation center is within the human annotation bounding box
            if human_x - human_width / 2 < predicted_x < human_x + human_width / 2 and \
                    human_y - human_height / 2 < predicted_y < human_y + human_height / 2:
                predicted_center = np.array([predicted_x, predicted_y])
                error = np.linalg.norm(human_center - predicted_center)
                if error < min_error:
                    min_error = error
        if min_error != np.inf:
            errors.append(min_error)
        else:
            errors.append(np.nan)
    return errors


def compute_error_by_class(human_annotations, predicted_annotations, image_width, image_height):
    """Compute error for each bounding box, organized by class ID."""
    errors_by_class = defaultdict(list)

    if human_annotations is None or predicted_annotations is None:
        return errors_by_class

    for human_annotation in human_annotations:
        human_class_id, human_x, human_y, human_width, human_height = human_annotation
        human_x = human_x * image_width
        human_y = human_y * image_height
        human_width = human_width * image_width
        human_height = human_height * image_height
        human_center = np.array([human_x, human_y])

        min_error = np.inf
        for predicted_annotation in predicted_annotations:
            predicted_class_id, predicted_x, predicted_y, predicted_width, predicted_height = predicted_annotation
            predicted_x = predicted_x * image_width
            predicted_y = predicted_y * image_height
            # Check if the predicted annotation center is within the human annotation bounding box
            if human_x - human_width / 2 < predicted_x < human_x + human_width / 2 and \
                    human_y - human_height / 2 < predicted_y < human_y + human_height / 2:
                predicted_center = np.array([predicted_x, predicted_y])
                error = np.linalg.norm(human_center - predicted_center)
                if error < min_error:
                    min_error = error
        if min_error != np.inf:
            errors_by_class[human_class_id].append(min_error)
        else:
            errors_by_class[human_class_id].append(np.nan)

    return errors_by_class


def report_results(errors, args):
    mean_error = np.nanmean(errors)
    median_error = np.nanmedian(errors)
    std_error = np.nanstd(errors)
    nan_count = np.sum(np.isnan(errors))
    print("\nClass-agnostic error statistics:")
    print(f"Mean error: {mean_error:.2f}")
    print(f"Median error: {median_error:.2f}")
    print(f"Standard deviation: {std_error:.2f}")
    print(f"Number of valid errors: {len(errors) - nan_count}")
    print(f"Number of NaN errors: {nan_count}")


def report_results_by_class(errors_by_class, args):
    """Report error statistics broken down by class ID."""
    print("\nClass-specific error statistics:")
    print("-" * 80)
    print(f"{'Class ID':^10} | {'Mean':^10} | {'Median':^10} | {'Std Dev':^10} | {'Valid Errors':^15} | {'NaN Errors':^10}")
    print("-" * 80)

    # Sort class IDs for consistent reporting
    for class_id in sorted(errors_by_class.keys()):
        errors = errors_by_class[class_id]
        mean_error = np.nanmean(errors)
        median_error = np.nanmedian(errors)
        std_error = np.nanstd(errors)
        nan_count = np.sum(np.isnan(errors))

        print(f"{class_id:^10} | {mean_error:^10.2f} | {median_error:^10.2f} | {std_error:^10.2f} | {len(errors) - nan_count:^15} | {nan_count:^10}")

    print("-" * 80)

    # Calculate and report overall statistics
    all_errors = np.concatenate(list(errors_by_class.values()))
    mean_error = np.nanmean(all_errors)
    median_error = np.nanmedian(all_errors)
    std_error = np.nanstd(all_errors)
    nan_count = np.sum(np.isnan(all_errors))

    print(f"{'All':^10} | {mean_error:^10.2f} | {median_error:^10.2f} | {std_error:^10.2f} | {len(all_errors) - nan_count:^15} | {nan_count:^10}")
    print("-" * 80)


def plot_error_distribution(errors, source, save=False):
    errors = errors[~np.isnan(errors)]
    total_instances = len(errors)

    mean_error = np.nanmean(errors)
    median_error = np.nanmedian(errors)
    std_error = np.nanstd(errors)
    percentile_90 = np.percentile(errors, 90)

    # Try to use seaborn for better aesthetics if available
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette("colorblind")
    except ImportError:
        pass

    # Create a more professional figure
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot all data points but with minimal visual footprint
    plt.plot(np.arange(len(errors)), errors, 'o', markersize=0.8, alpha=0.2, 
             color='#3274A1', rasterized=True)

    # Add horizontal lines for statistical measures
    plt.axhline(mean_error, color='#C44E52', linestyle='-', linewidth=2, 
                label=f"Mean error: {mean_error:.2f} px")
    plt.axhline(median_error, color='#55A868', linestyle='--', linewidth=2, 
                label=f"Median error: {median_error:.2f} px")

    # Add standard deviation band
    plt.axhspan(mean_error - std_error, mean_error + std_error, color='#C44E52', 
                alpha=0.15, label=f"Standard deviation: {std_error:.2f} px")

    # Add 90th percentile line
    plt.axhline(percentile_90, color='#8172B3', linestyle=':', linewidth=1.5,
                label=f"90th percentile: {percentile_90:.2f} px")

    # Adjust y-axis to focus on meaningful range but still show all data
    # Calculate reasonable y limits based on percentiles
    y_max_view = max(percentile_90 * 1.5, mean_error + 2*std_error)  # Show up to 90th percentile or 2-sigma

    # Set linear scale with custom tick locations
    plt.ylim(0, y_max_view)

    # Create custom y ticks focusing on the important ranges
    custom_yticks = [0]

    # Add median if not too close to 0
    if median_error > 0.1:
        custom_yticks.append(median_error)

    # Add mean
    custom_yticks.append(mean_error)

    # Add 1-sigma bounds
    custom_yticks.extend([mean_error - std_error, mean_error + std_error])

    # Add 90th percentile
    custom_yticks.append(percentile_90)

    # Sort and remove duplicates
    custom_yticks = sorted(list(set([round(y, 2) for y in custom_yticks])))

    plt.yticks(custom_yticks)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Improve labels and title
    plt.xlabel("Bounding box index", fontweight='bold', fontsize=12)
    plt.ylabel("Error (pixels)", fontweight='bold', fontsize=12)
    plt.title(f"Bounding Box Center Error Distribution (n={total_instances:,})", fontweight='bold', fontsize=14)

    # Place legend in the best position to avoid data obstruction
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)

    # Add tight layout
    plt.tight_layout()

    # Save with higher quality if requested
    if save:
        # save it to the parent folder of source
        plt.savefig(source.parent / "error_distribution.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(source.parent / "error_distribution.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_error_distribution_by_class(errors_by_class, source, save=False):
    """Plot error distribution broken down by class ID."""
    # Create a figure with subplots: one for each class plus one for all classes combined
    n_classes = len(errors_by_class)
    fig_height = 4 * (n_classes + 1)  # +1 for the combined plot

    fig, axs = plt.subplots(n_classes + 1, 1, figsize=(10, fig_height), dpi=300)

    # Try to use seaborn for better aesthetics if available
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette("colorblind")
    except ImportError:
        pass

    # Plot for all classes combined (first subplot)
    all_errors = np.concatenate(list(errors_by_class.values()))
    ax_all = axs[0]

    plot_single_distribution(ax_all, all_errors, "All Classes Combined")

    # Plot individual class distributions
    for i, (class_id, errors) in enumerate(sorted(errors_by_class.items())):
        ax = axs[i+1]
        plot_single_distribution(ax, errors, f"Class ID: {class_id}")

    plt.tight_layout()

    if save:
        plt.savefig(source.parent / "error_distribution_by_class.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(source.parent / "error_distribution_by_class.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_single_distribution(ax, errors, title_prefix):
    """Plot a single error distribution on the given axis."""
    total_boxes = len(errors)
    errors_no_nan = errors[~np.isnan(errors)]

    mean_error = np.nanmean(errors)
    median_error = np.nanmedian(errors)
    std_error = np.nanstd(errors)
    percentile_90 = np.percentile(errors_no_nan, 90) if len(errors_no_nan) > 0 else 0

    # Plot all data points but with minimal visual footprint
    ax.plot(np.arange(len(errors_no_nan)), errors_no_nan, 'o', markersize=0.8, alpha=0.2,
            color='#3274A1', rasterized=True)

    # Add horizontal lines for statistical measures
    ax.axhline(mean_error, color='#C44E52', linestyle='-', linewidth=2,
               label=f"Mean error: {mean_error:.2f} px")
    ax.axhline(median_error, color='#55A868', linestyle='--', linewidth=2,
               label=f"Median error: {median_error:.2f} px")

    # Add standard deviation band
    ax.axhspan(mean_error - std_error, mean_error + std_error, color='#C44E52',
               alpha=0.15, label=f"Standard deviation: {std_error:.2f} px")

    # Add 90th percentile line
    ax.axhline(percentile_90, color='#8172B3', linestyle=':', linewidth=1.5,
               label=f"90th percentile: {percentile_90:.2f} px")

    # Adjust y-axis to focus on meaningful range
    y_max_view = max(percentile_90 * 1.5, mean_error + 2*std_error)

    # Set linear scale with custom tick locations
    ax.set_ylim(0, y_max_view)

    # Create custom y ticks focusing on the important ranges
    custom_yticks = [0]

    # Add median if not too close to 0
    if median_error > 0.1:
        custom_yticks.append(median_error)

    # Add mean
    custom_yticks.append(mean_error)

    # Add 1-sigma bounds
    custom_yticks.extend([mean_error - std_error, mean_error + std_error])

    # Add 90th percentile
    custom_yticks.append(percentile_90)

    # Sort and remove duplicates
    custom_yticks = sorted(list(set([round(y, 2) for y in custom_yticks])))

    ax.set_yticks(custom_yticks)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Improve labels and title
    ax.set_xlabel("Bounding box index", fontweight='bold', fontsize=12)
    ax.set_ylabel("Error (pixels)", fontweight='bold', fontsize=12)
    ax.set_title(f"{title_prefix} (n={total_boxes:,})", fontweight='bold', fontsize=14)

    # Place legend in the best position to avoid data obstruction
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=10)


def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Compute bounding box center error statistics.')
    parser.add_argument('source', type=Path, help='Path to the images to be analyzed')
    parser.add_argument('--human-annotations', '-ha', type=Path, default="../labels", help='Relative path to the human annotations with respect to the source')
    parser.add_argument('--predicted-annotations', '-pa', type=Path, default="../pre-labels", help='Relative path to the predicted annotations with respect to the source')
    parser.add_argument('--save', '-s', action='store_true', help='Save the error distribution as a figure')
    parser.add_argument('--class-agnostic', '-ca', action='store_true', help='Compute class-agnostic statistics instead of per-class breakdown')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    compute_bb_center_error(args)
